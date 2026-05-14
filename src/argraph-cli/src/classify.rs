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
//!   Microsatellite       : ≥3 branches, all 1-bp same letter, each walks
//!                          along a same-letter simple chain to a common exit
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
    if let Some(t) = try_snp(graph, succs) {
        return t;
    }
    if let Some(t) = try_microsatellite(graph, succs) {
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
fn try_microsatellite(graph: &Graph, succs: &[NodeId]) -> Option<BubbleType> {
    if succs.len() < 3 {
        return None;
    }
    if !succs.iter().all(|&s| is_single_base(graph, s)) {
        return None;
    }
    let target = node_base(graph, succs[0])[0];
    if !succs.iter().all(|&s| node_base(graph, s)[0] == target) {
        return None;
    }
    let mut exits: HashSet<NodeId> = HashSet::new();
    for &s in succs {
        let mut cur = s;
        for _ in 0..MAX_MICROSAT_STEPS {
            let nxt = graph.successors(cur);
            if nxt.len() != 1 {
                return None;
            }
            let nn = nxt[0];
            let nn_seq = node_base(graph, nn);
            let nn_single_out = graph.successors(nn).len() == 1;
            if nn_seq.len() == 1 && nn_seq[0] == target && nn_single_out {
                cur = nn;
                continue;
            }
            exits.insert(nn);
            break;
        }
    }
    if exits.len() == 1 {
        Some(BubbleType::Microsatellite)
    } else {
        None
    }
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
