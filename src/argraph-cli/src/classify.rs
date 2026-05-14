//! Classify a bubble into one of the mechanism types.
//!
//! Typology (see panarg/notes/bubble_typology.md):
//!   Snp                  : 2 branches, both single-base, distinct nucleotides
//!   MultiAllelicSnp      : 3-4 branches, all single-base, distinct nucleotides
//!   SmallIndel           : 2 branches; one empty, the other 1-50 single-base
//!                          nodes (sequence of any one nucleotide or mixed —
//!                          mixed insertion); or two non-empty branches whose
//!                          symmetric structure looks like indel polymorphism
//!   Microsatellite       : >=3 branches forming a chain where all internal
//!                          nodes carry the same single base
//!   Complex              : anything else (branches >50 bp, structural)

use crate::bubble::Bubble;
use crate::gfa::{Graph, NodeId};

/// Bubble mechanism type. The associated rate (per generation, locus-level
/// where applicable) is what tsdate would need to date the corresponding
/// branch correctly. Order of magnitude only.
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
    /// Used by downstream (tsdate, post-hoc reweighting) only; not by the
    /// matcher itself.
    pub fn mu_event(&self) -> f64 {
        match self {
            BubbleType::Snp | BubbleType::MultiAllelicSnp => 1.2e-8,
            BubbleType::SmallIndel => 1.0e-9,
            BubbleType::Microsatellite => 1.0e-3,
            BubbleType::Complex => 1.0e-5,
        }
    }
}

fn branch_seq(graph: &Graph, branch: &[NodeId]) -> Vec<u8> {
    let mut out = Vec::new();
    for n in branch {
        if let Some(s) = graph.seq.get(n) {
            out.extend_from_slice(s);
        }
    }
    out
}

/// Classify a bubble.
pub fn classify(bubble: &Bubble, graph: &Graph) -> BubbleType {
    let k = bubble.n_branches();
    if k < 2 {
        return BubbleType::Complex;
    }

    let branch_seqs: Vec<Vec<u8>> = bubble.branches.iter().map(|b| branch_seq(graph, b)).collect();

    // All branches single-base (length 1)?
    let all_single = branch_seqs.iter().all(|s| s.len() == 1);
    if all_single {
        // distinct nucleotides among branches → SNP / multi-allelic SNP
        let mut letters: Vec<u8> = branch_seqs.iter().map(|s| s[0]).collect();
        letters.sort_unstable();
        letters.dedup();
        if letters.len() >= 2 {
            return if k == 2 { BubbleType::Snp } else { BubbleType::MultiAllelicSnp };
        }
        // All branches same single base — falls through to microsat check
    }

    // Microsatellite: all internal nodes (across all non-empty branches) carry
    // the same single base, and there is at least one empty branch (the
    // "skip" path). The chain structure is implicit in how branches differ in
    // length.
    let has_empty = branch_seqs.iter().any(|s| s.is_empty());
    let non_empty: Vec<&Vec<u8>> = branch_seqs.iter().filter(|s| !s.is_empty()).collect();
    if has_empty && !non_empty.is_empty() {
        // All non-empty branches must be made of the same single base repeated.
        let mut base: Option<u8> = None;
        let mut uniform = true;
        for seq in &non_empty {
            for &b in seq.iter() {
                match base {
                    None => base = Some(b),
                    Some(b0) if b0 == b => {}
                    _ => {
                        uniform = false;
                        break;
                    }
                }
            }
            if !uniform {
                break;
            }
        }
        if uniform && base.is_some() && k >= 3 {
            return BubbleType::Microsatellite;
        }
    }

    // Small indel: exactly 2 branches, one empty, the other 1-50 bp.
    if k == 2 {
        let lens: Vec<usize> = branch_seqs.iter().map(|s| s.len()).collect();
        let min_len = *lens.iter().min().unwrap();
        let max_len = *lens.iter().max().unwrap();
        if min_len == 0 && max_len > 0 && max_len <= 50 {
            return BubbleType::SmallIndel;
        }
    }

    BubbleType::Complex
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn snp_two_branches_distinct_nucleotides() {
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"R")],
            &[(1, 2), (1, 3), (2, 4), (3, 4)],
        );
        let bubble = crate::bubble::find_bubble(&g, 1, 10).unwrap();
        assert_eq!(classify(&bubble, &g), BubbleType::Snp);
    }

    #[test]
    fn small_indel_two_branches_one_empty() {
        // 1 → 2 (skip) → 4; or 1 → 3 (A) → 2 → 4 — but that's a microsat shape.
        // Cleaner: 1 → 4 directly OR 1 → 2 → 3 → 4 with 2 and 3 carrying ACG.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"C"), (4, b"R")],
            &[(1, 4), (1, 2), (2, 3), (3, 4)],
        );
        let bubble = crate::bubble::find_bubble(&g, 1, 10).unwrap();
        assert_eq!(classify(&bubble, &g), BubbleType::SmallIndel);
    }

    #[test]
    fn microsatellite_chain_same_letter() {
        // 1 → {2,3,4,5} → 5 → R, where 2,3,4,5 are all 'A' and form a chain.
        // Paths take 1,2,3 or 4 A's.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"A"), (4, b"A"), (5, b"A"), (6, b"R")],
            &[(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6)],
        );
        let bubble = crate::bubble::find_bubble(&g, 1, 10).unwrap();
        assert_eq!(classify(&bubble, &g), BubbleType::Microsatellite);
    }

    #[test]
    fn complex_branches_longer_than_50_bp() {
        // 2-branch bubble where one branch is 60 bp long (not a microsat).
        let mut nodes: Vec<(NodeId, Vec<u8>)> = Vec::new();
        nodes.push((1, b"L".to_vec()));
        for i in 0..60 {
            nodes.push((100 + i as NodeId, vec![b"ACGT"[(i % 4) as usize]]));
        }
        nodes.push((2, b"R".to_vec()));
        let node_refs: Vec<(NodeId, &[u8])> = nodes.iter().map(|(n, s)| (*n, s.as_slice())).collect();
        let mut links: Vec<(NodeId, NodeId)> = vec![(1, 2), (1, 100)];
        for i in 0..59 {
            links.push((100 + i, 101 + i));
        }
        links.push((100 + 59, 2));
        let g = mk_graph(&node_refs, &links);
        let bubble = crate::bubble::find_bubble(&g, 1, 100).unwrap();
        assert_eq!(classify(&bubble, &g), BubbleType::Complex);
    }
}
