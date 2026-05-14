//! Classify a bubble into one of the mechanism types.
//!
//! The classifier follows the panarg Python reference (`classify_bubbles.py`)
//! and uses type-specific walks from the source's immediate successors. Each
//! walk requires single-out interior nodes (no internal branch points); a
//! bubble whose BFS sink is reachable but whose interior contains nested
//! structure falls through to `Complex`.
//!
//! Typology:
//!   Snp                  : 2 single-base branches, each with exactly one
//!                          outgoing edge leading to the same downstream node,
//!                          distinct nucleotides
//!   MultiAllelicSnp      : same as Snp but with 3-4 branches
//!   SmallIndel           : 2 branches; one is the "skip" (one step lands on
//!                          the other branch's start), the other is a chain
//!                          of 1-50 nodes with no internal branch points
//!   Microsatellite       : ≥3 branches whose simple-chain sequences (walked
//!                          up to the BFS-found sink) all decompose as `M^k`
//!                          for some shared motif `M` of length 1-6 bp. Picks
//!                          the shortest motif length that works. Detects
//!                          mono-, di-, tri-, tetra-, penta- and hexa-
//!                          nucleotide STRs/VNTRs. (The panarg Python
//!                          reference only detected mononucleotide tracts.)
//!   Complex              : anything else

use std::collections::HashSet;

use crate::bubble::Bubble;
use crate::gfa::{Graph, NodeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BubbleType {
    Snp,
    MultiAllelicSnp,
    SmallIndel,
    Microsatellite,
    Complex,
}

impl BubbleType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BubbleType::Snp => "snp",
            BubbleType::MultiAllelicSnp => "msnp",
            BubbleType::SmallIndel => "indel",
            BubbleType::Microsatellite => "microsat",
            BubbleType::Complex => "complex",
        }
    }

    /// Order-of-magnitude per-event mutation rate to assign as metadata.
    pub fn mu_event(&self) -> f64 {
        match self {
            BubbleType::Snp | BubbleType::MultiAllelicSnp => 1.2e-8,
            BubbleType::SmallIndel => 1.0e-9,
            BubbleType::Microsatellite => 1.0e-3,
            BubbleType::Complex => 1.0e-5,
        }
    }
}

const MAX_INDEL_LEN: usize = 50;
const MAX_MICROSAT_STEPS: usize = 200;
const MAX_MOTIF_LEN: usize = 6;

/// Single-base sequence of a node, or empty if missing.
fn node_base(graph: &Graph, n: NodeId) -> &[u8] {
    graph.seq.get(&n).map(|v| v.as_slice()).unwrap_or(&[])
}

fn is_single_base(graph: &Graph, n: NodeId) -> bool {
    node_base(graph, n).len() == 1
}

/// Classify the bubble rooted at `bubble.source` using its successors.
pub fn classify(bubble: &Bubble, graph: &Graph) -> BubbleType {
    let succs = graph.successors(bubble.source);
    if succs.len() < 2 {
        return BubbleType::Complex;
    }
    // bubble.sink == bubble.source signals "BFS didn't converge"; pass None
    // so microsat/indel checks know there's no usable bubble boundary.
    let sink = if bubble.sink != bubble.source {
        Some(bubble.sink)
    } else {
        None
    };
    if let Some(t) = try_snp(graph, succs) {
        return t;
    }
    if let Some(t) = try_microsatellite(graph, succs, sink) {
        return t;
    }
    if let Some(t) = try_small_indel(graph, succs) {
        return t;
    }
    BubbleType::Complex
}

/// SNP / multi-allelic SNP: each immediate successor is a single-base node with
/// exactly one outgoing edge; all those outgoing edges land on the same node;
/// nucleotides are pairwise distinct.
fn try_snp(graph: &Graph, succs: &[NodeId]) -> Option<BubbleType> {
    let k = succs.len();
    if !(2..=4).contains(&k) {
        return None;
    }
    if !succs.iter().all(|&s| is_single_base(graph, s)) {
        return None;
    }
    let mut shared_next: Option<NodeId> = None;
    for &s in succs {
        let nxt = graph.successors(s);
        if nxt.len() != 1 {
            return None;
        }
        match shared_next {
            None => shared_next = Some(nxt[0]),
            Some(n) if n == nxt[0] => {}
            _ => return None,
        }
    }
    let mut letters: Vec<u8> = succs.iter().map(|&s| node_base(graph, s)[0]).collect();
    letters.sort_unstable();
    letters.dedup();
    if letters.len() != k {
        return None;
    }
    if k == 2 {
        Some(BubbleType::Snp)
    } else {
        Some(BubbleType::MultiAllelicSnp)
    }
}

/// Microsatellite: ≥3 branches, all immediate successors are 1-bp same letter,
/// each walks along a same-letter chain (with single-out interior nodes) to a
/// common exit node.
fn try_microsatellite(
    graph: &Graph,
    succs: &[NodeId],
    sink: Option<NodeId>,
) -> Option<BubbleType> {
    if succs.len() < 3 {
        return None;
    }
    // We need a BFS-found sink to bound the walk; otherwise the chain may
    // run on past the bubble.
    let sink = sink?;

    // Walk each branch as a simple single-out chain up to the sink.
    let mut sequences: Vec<Vec<u8>> = Vec::with_capacity(succs.len());
    for &s in succs {
        let seq = walk_branch_to_sink(graph, s, sink, MAX_MICROSAT_STEPS)?;
        sequences.push(seq);
    }

    // Find the shortest motif M (1..=6 bp) such that every non-empty branch
    // sequence is exactly M^k for some k ≥ 1. Empty branches always pass.
    for motif_len in 1..=MAX_MOTIF_LEN {
        if is_consistent_motif(&sequences, motif_len) {
            return Some(BubbleType::Microsatellite);
        }
    }
    None
}

/// Walk from `start` toward `sink` through single-out interior nodes. The
/// walk stops cleanly when the next step lands on `sink` (sink's bases are
/// excluded from the returned sequence). Returns None if the walk encounters
/// a multi-out node before reaching the sink (the chain isn't simple), or
/// if `max_steps` runs out.
///
/// If `start == sink` the branch is empty and the function returns
/// `Some((vec![], sink))` semantically — here just the empty Vec.
fn walk_branch_to_sink(
    graph: &Graph,
    start: NodeId,
    sink: NodeId,
    max_steps: usize,
) -> Option<Vec<u8>> {
    if start == sink {
        return Some(Vec::new());
    }
    let mut seq: Vec<u8> = Vec::new();
    let mut cur = start;
    for _ in 0..max_steps {
        seq.extend_from_slice(node_base(graph, cur));
        let nxt = graph.successors(cur);
        if nxt.len() != 1 {
            return None; // chain branches before reaching sink
        }
        let nn = nxt[0];
        if nn == sink {
            return Some(seq);
        }
        cur = nn;
    }
    None
}

/// True iff every non-empty sequence is exactly `M^k` for some `k ≥ 1`, where
/// `M` is the prefix of length `motif_len` from the shortest non-empty
/// sequence. Empty sequences trivially pass.
fn is_consistent_motif(seqs: &[Vec<u8>], motif_len: usize) -> bool {
    let shortest = seqs.iter().filter(|s| !s.is_empty()).min_by_key(|s| s.len());
    let Some(shortest) = shortest else {
        return false;
    };
    if shortest.len() < motif_len {
        return false;
    }
    let motif = &shortest[..motif_len];
    for s in seqs {
        if s.is_empty() {
            continue;
        }
        if s.len() % motif_len != 0 {
            return false;
        }
        let k = s.len() / motif_len;
        for i in 0..k {
            if &s[i * motif_len..(i + 1) * motif_len] != motif {
                return false;
            }
        }
    }
    true
}

/// Small indel: 2 branches; one is the "skip" (one step lands on the other
/// branch's start), the other walks 1-50 nodes (single-out interior) before
/// converging.
///
/// Mirrors the two checks in classify_bubbles.py:
///   (a) walk one branch (`b`) up to 50 steps; if a successor equals the other
///       branch's start `a`, accept.
///   (b) walk both branches as simple chains; if they share a node within 51
///       steps and one of them reaches it within ≤ 1 step (so one branch is
///       effectively the "skip"), accept.
fn try_small_indel(graph: &Graph, succs: &[NodeId]) -> Option<BubbleType> {
    if succs.len() != 2 {
        return None;
    }

    // Check (a): walk b looking for a.
    for (i, j) in [(0usize, 1usize), (1, 0)] {
        let a = succs[i];
        let b = succs[j];
        let mut cur = b;
        for _ in 0..(MAX_INDEL_LEN + 1) {
            let nxt = graph.successors(cur);
            if nxt.len() != 1 {
                break;
            }
            if nxt[0] == a {
                return Some(BubbleType::SmallIndel);
            }
            cur = nxt[0];
        }
    }

    // Check (b): walk both chains; meet within 51 steps with one reaching in ≤ 1.
    let chain_a = walk_simple_chain(graph, succs[0], MAX_INDEL_LEN + 1);
    let chain_b = walk_simple_chain(graph, succs[1], MAX_INDEL_LEN + 1);
    let set_a: HashSet<NodeId> = chain_a.iter().copied().collect();
    if let Some(&meet) = chain_b.iter().find(|n| set_a.contains(n)) {
        let ia = chain_a.iter().position(|&x| x == meet).unwrap();
        let ib = chain_b.iter().position(|&x| x == meet).unwrap();
        let diff = ia.abs_diff(ib);
        let shorter = ia.min(ib);
        if (1..=MAX_INDEL_LEN).contains(&diff) && shorter <= 1 {
            return Some(BubbleType::SmallIndel);
        }
    }
    None
}

/// Walk forward as long as each node has exactly one outgoing edge, up to
/// `max_steps`. Returns the sequence of visited nodes starting from `start`.
fn walk_simple_chain(graph: &Graph, start: NodeId, max_steps: usize) -> Vec<NodeId> {
    let mut out = vec![start];
    let mut cur = start;
    for _ in 0..max_steps {
        let nxt = graph.successors(cur);
        if nxt.len() != 1 {
            break;
        }
        cur = nxt[0];
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bubble::find_bubble;
    use crate::gfa::Graph;
    use std::collections::HashMap;

    fn mk_graph(nodes: &[(NodeId, &[u8])], links: &[(NodeId, NodeId)]) -> Graph {
        let mut seq = HashMap::new();
        for (id, s) in nodes {
            seq.insert(*id, s.to_vec());
        }
        let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut backward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (a, b) in links {
            forward.entry(*a).or_default().push(*b);
            backward.entry(*b).or_default().push(*a);
        }
        Graph { seq, forward, backward, paths: Vec::new() }
    }

    fn classify_from_source(graph: &Graph, source: NodeId) -> BubbleType {
        let bubble = find_bubble(graph, source, 200).expect("bubble must close");
        classify(&bubble, graph)
    }

    #[test]
    fn snp_two_branches_distinct_nucleotides() {
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"R")],
            &[(1, 2), (1, 3), (2, 4), (3, 4)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Snp);
    }

    #[test]
    fn snp_with_branching_interior_is_complex() {
        // Like a "SNP" but one branch has an internal fork: 1 → 2 → 4, 1 → 3 → 4 OR 3 → 5 → 4.
        // Node 3 has 2 outgoing edges; Python (and now Rust) reject this as a simple SNP.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"R"), (5, b"X")],
            &[(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (5, 4)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn small_indel_two_branches_one_skip() {
        // 1 → 4 (skip) OR 1 → 2 → 3 → 4 (2-bp insertion)
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"C"), (4, b"R")],
            &[(1, 4), (1, 2), (2, 3), (3, 4)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::SmallIndel);
    }

    #[test]
    fn indel_with_branching_chain_is_complex() {
        // 1 → 4 (skip), 1 → 2 → 3 → 4, but 3 also branches to 5 → 4.
        // Python (and Rust now) reject: the chain is not simple.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"C"), (4, b"R"), (5, b"X")],
            &[(1, 4), (1, 2), (2, 3), (3, 4), (3, 5), (5, 4)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn microsatellite_chain_same_letter() {
        // 1 → {2, 3, 4, 5} where 2/3/4/5 are all 'A' forming a chain to 6.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"A"), (4, b"A"), (5, b"A"), (6, b"R")],
            &[(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Microsatellite);
    }

    #[test]
    fn microsatellite_ta_dinucleotide() {
        // A TA-repeat STR with 4 length classes: 0 (skip), TA, TATA, TATATA.
        // Source 1 connects to four entry points along the chain (offsets 0, 2, 4, 6).
        // Layout (each node is 1 bp):
        //   1 → 8 (sink, skip = 0 TA)
        //   1 → 6 → 7 → 8 (= TA)
        //   1 → 4 → 5 → 6 → 7 → 8 (= TATA)
        //   1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 (= TATATA)
        let g = mk_graph(
            &[(1, b"L"),
              (2, b"T"), (3, b"A"),
              (4, b"T"), (5, b"A"),
              (6, b"T"), (7, b"A"),
              (8, b"R")],
            &[(1, 2), (1, 4), (1, 6), (1, 8),
              (2, 3), (3, 4),
              (4, 5), (5, 6),
              (6, 7), (7, 8)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Microsatellite);
    }

    #[test]
    fn microsatellite_cag_trinucleotide() {
        // CAG-repeat STR with 3 length classes: 0, CAG, CAGCAG.
        let g = mk_graph(
            &[(1, b"L"),
              (2, b"C"), (3, b"A"), (4, b"G"),
              (5, b"C"), (6, b"A"), (7, b"G"),
              (8, b"R")],
            &[(1, 2), (1, 5), (1, 8),
              (2, 3), (3, 4), (4, 5),
              (5, 6), (6, 7), (7, 8)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Microsatellite);
    }

    #[test]
    fn near_microsatellite_inconsistent_motif_is_complex() {
        // 3 branches with sequences "AT", "ATAT" but the third is "AC" (broken motif).
        // No M^k works across all three → not a microsat.
        let g = mk_graph(
            &[(1, b"L"),
              (2, b"A"), (3, b"T"),    // "AT"
              (4, b"A"), (5, b"T"),    // "AT" prefix of "ATAT"
              (6, b"A"), (7, b"C"),    // "AC" (different)
              (8, b"R")],
            &[(1, 2), (1, 4), (1, 6),
              (2, 3), (3, 8),
              (4, 5), (5, 8),
              (6, 7), (7, 8)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    // --- boundary cases ----------------------------------------------------

    #[test]
    fn microsat_motif_len_6_is_detected() {
        // ATGCAT-repeat (6 bp motif, at the upper bound). 3 branches with
        // 0, 1 and 2 motif copies, sharing the trailing motif.
        //   1 → 99                              (0 copies)
        //   1 → 16 → 17 → 18 → 19 → 20 → 21 → 99 (1 copy: ATGCAT)
        //   1 → 10 → 11 → ... → 15 → 16 → ... → 21 → 99 (2 copies)
        let g = mk_graph(
            &[(1, b"L"),
              (10, b"A"), (11, b"T"), (12, b"G"), (13, b"C"), (14, b"A"), (15, b"T"),
              (16, b"A"), (17, b"T"), (18, b"G"), (19, b"C"), (20, b"A"), (21, b"T"),
              (99, b"R")],
            &[(1, 99), (1, 16), (1, 10),
              (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
              (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 99)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Microsatellite);
    }

    #[test]
    fn microsat_motif_len_7_is_not_detected() {
        // 7-bp repeat ATGCAAT — outside our MAX_MOTIF_LEN = 6 cap → Complex.
        let g = mk_graph(
            &[(1, b"L"),
              (10, b"A"), (11, b"T"), (12, b"G"), (13, b"C"), (14, b"A"), (15, b"A"), (16, b"T"),
              (17, b"A"), (18, b"T"), (19, b"G"), (20, b"C"), (21, b"A"), (22, b"A"), (23, b"T"),
              (99, b"R")],
            &[(1, 99), (1, 17), (1, 10),
              (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
              (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 99)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn small_indel_at_50_bp() {
        // 50-node insertion — well below the 51-bp internal walk limit.
        let mut nodes: Vec<(NodeId, Vec<u8>)> =
            vec![(1, b"L".to_vec()), (99, b"R".to_vec())];
        for i in 0..50 {
            nodes.push((100 + i as NodeId, vec![b"A"[0]]));
        }
        let node_refs: Vec<(NodeId, &[u8])> =
            nodes.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        let mut links: Vec<(NodeId, NodeId)> = vec![(1, 99), (1, 100)];
        for i in 0..49 {
            links.push((100 + i, 101 + i));
        }
        links.push((100 + 49, 99));
        let g = mk_graph(&node_refs, &links);
        assert_eq!(classify_from_source(&g, 1), BubbleType::SmallIndel);
    }

    #[test]
    fn indel_at_52_bp_is_complex() {
        // The classifier's check (a) walks up to MAX_INDEL_LEN+1 = 51 steps.
        // A 52-node chain exhausts the loop without finding the other branch,
        // and check (b)'s `shorter <= 1` keeps the empty branch as the
        // "skip" — but the chain length 52 exceeds the diff cap of 50, so
        // the structural check fails. Falls to Complex.
        let mut nodes: Vec<(NodeId, Vec<u8>)> =
            vec![(1, b"L".to_vec()), (99, b"R".to_vec())];
        for i in 0..52 {
            nodes.push((100 + i as NodeId, vec![b"A"[0]]));
        }
        let node_refs: Vec<(NodeId, &[u8])> =
            nodes.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        let mut links: Vec<(NodeId, NodeId)> = vec![(1, 99), (1, 100)];
        for i in 0..51 {
            links.push((100 + i, 101 + i));
        }
        links.push((100 + 51, 99));
        let g = mk_graph(&node_refs, &links);
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn msnp_3_alleles_detected() {
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"G"), (5, b"R")],
            &[(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::MultiAllelicSnp);
    }

    #[test]
    fn msnp_4_alleles_detected() {
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"G"), (5, b"C"), (6, b"R")],
            &[(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::MultiAllelicSnp);
    }

    #[test]
    fn snp_two_same_nucleotides_is_not_snp() {
        // Two branches with same base A,A → not a SNP (requires distinct).
        // Falls through to Microsatellite check, then SmallIndel. With 2
        // single-base same-letter branches that both go to sink, the
        // microsat check needs ≥3 branches → fails. Indel needs one empty
        // branch → fails (both are 1-bp). So Complex.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"A"), (4, b"R")],
            &[(1, 2), (1, 3), (2, 4), (3, 4)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn snp_with_multibase_alt_is_classified_as_indel() {
        // 2 branches: one 1-bp (A), one 2-bp (TG). The shorter chain
        // reaches the sink in 1 step and the longer one within +1 step, so
        // check (b) of the small-indel logic catches it. This mirrors how
        // the panarg Python reference behaves on the same shape — biology-
        // wise it's a 1-bp insertion with simultaneous base-change, but
        // the classifier prefers SmallIndel over Complex on length-1 diffs.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"G"), (5, b"R")],
            &[(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::SmallIndel);
    }

    #[test]
    fn microsat_inconsistent_motif_via_different_shortest_paths_is_complex() {
        // Two branches "AT" and "AC" — same length 2, different content → no
        // shared motif M of length 1 or 2 works. With a third empty branch
        // we'd still fail (M=1: "AT" needs A or T uniformly; "AC" breaks).
        let g = mk_graph(
            &[(1, b"L"),
              (10, b"A"), (11, b"T"),  // "AT"
              (20, b"A"), (21, b"C"),  // "AC"
              (99, b"R")],
            &[(1, 99), (1, 10), (1, 20),
              (10, 11), (11, 99),
              (20, 21), (21, 99)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn microsat_motif_must_divide_all_branches() {
        // Branches "TA" (len 2) and "TAT" (len 3). Motif "TA" divides "TA"
        // but not "TAT" (3 isn't a multiple of 2). Motif "TAT" divides
        // "TAT" but not "TA". No common motif → Complex.
        let g = mk_graph(
            &[(1, b"L"),
              (10, b"T"), (11, b"A"),       // "TA"
              (20, b"T"), (21, b"A"), (22, b"T"),  // "TAT"
              (99, b"R")],
            &[(1, 99), (1, 10), (1, 20),
              (10, 11), (11, 99),
              (20, 21), (21, 22), (22, 99)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }

    #[test]
    fn microsat_when_one_branch_is_empty_skip() {
        // 4 branches: 0, A, AA, AAA — same pattern as the chr12:60 Mb gold std.
        let g = mk_graph(
            &[(1, b"L"),
              (10, b"A"),
              (20, b"A"), (21, b"A"),
              (30, b"A"), (31, b"A"), (32, b"A"),
              (99, b"R")],
            &[(1, 99), (1, 10), (1, 20), (1, 30),
              (10, 99),
              (20, 21), (21, 99),
              (30, 31), (31, 32), (32, 99)],
        );
        assert_eq!(classify_from_source(&g, 1), BubbleType::Microsatellite);
    }

    #[test]
    fn complex_branches_longer_than_50_bp() {
        let mut nodes: Vec<(NodeId, Vec<u8>)> = Vec::new();
        nodes.push((1, b"L".to_vec()));
        for i in 0..60 {
            nodes.push((100 + i as NodeId, vec![b"ACGT"[(i % 4) as usize]]));
        }
        nodes.push((2, b"R".to_vec()));
        let node_refs: Vec<(NodeId, &[u8])> =
            nodes.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        let mut links: Vec<(NodeId, NodeId)> = vec![(1, 2), (1, 100)];
        for i in 0..59 {
            links.push((100 + i, 101 + i));
        }
        links.push((100 + 59, 2));
        let g = mk_graph(&node_refs, &links);
        assert_eq!(classify_from_source(&g, 1), BubbleType::Complex);
    }
}
