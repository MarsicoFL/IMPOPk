//! Property-based tests for Jacquard classify_locus.
//!
//! Tests systemic properties:
//! - All 9 delta states are reachable
//! - Monotonicity: adding IBD pairs can only merge blocks (state ≤ original)
//! - A↔B symmetry: swapping individuals maps states 3↔5, 4↔6
//! - Self-loops are no-ops
//! - Transitivity via Union-Find

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════
// Reimplementation of private types from jacquard main.rs
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Pair {
    left: String,
    right: String,
}

impl Pair {
    fn new(a: String, b: String) -> Self {
        Self { left: a, right: b }
    }
}

#[derive(Default, Clone)]
struct LocusData {
    pairs: BTreeSet<Pair>,
}

struct HaplotypeSet {
    a: [String; 2],
    b: [String; 2],
    #[allow(dead_code)]
    all: HashSet<String>,
}

impl HaplotypeSet {
    fn new(a1: String, a2: String, b1: String, b2: String) -> Self {
        let mut all = HashSet::new();
        all.insert(a1.clone());
        all.insert(a2.clone());
        all.insert(b1.clone());
        all.insert(b2.clone());
        Self {
            a: [a1, a2],
            b: [b1, b2],
            all,
        }
    }

    fn is_a(&self, hap: &str) -> bool {
        self.a.iter().any(|h| h == hap)
    }

    fn is_b(&self, hap: &str) -> bool {
        self.b.iter().any(|h| h == hap)
    }

    fn nodes(&self) -> [&str; 4] {
        [
            self.a[0].as_str(),
            self.a[1].as_str(),
            self.b[0].as_str(),
            self.b[1].as_str(),
        ]
    }
}

#[derive(Clone, Copy)]
struct BlockStat {
    size: usize,
    count_a: usize,
    count_b: usize,
}

struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new(nodes: &[&str]) -> Self {
        let mut parent = HashMap::new();
        for n in nodes {
            parent.insert((*n).to_string(), (*n).to_string());
        }
        Self { parent }
    }

    fn find(&mut self, node: &str) -> String {
        let parent = self
            .parent
            .get(node)
            .cloned()
            .unwrap_or_else(|| node.to_string());
        if parent == node {
            return parent;
        }
        let root = self.find(&parent);
        self.parent.insert(node.to_string(), root.clone());
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a != root_b {
            self.parent.insert(root_b, root_a);
        }
    }
}

fn classify_locus(data: &LocusData, haps: &HaplotypeSet) -> Option<u8> {
    let nodes = haps.nodes();
    let mut uf = UnionFind::new(&nodes);
    for pair in &data.pairs {
        uf.union(&pair.left, &pair.right);
    }

    let mut block_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for node in nodes {
        let root = uf.find(node);
        block_map.entry(root).or_default().push(node.to_string());
    }

    let mut stats: Vec<BlockStat> = Vec::new();
    for members in block_map.values() {
        let mut count_a = 0;
        let mut count_b = 0;
        for m in members {
            if haps.is_a(m) {
                count_a += 1;
            } else if haps.is_b(m) {
                count_b += 1;
            }
        }
        stats.push(BlockStat {
            size: members.len(),
            count_a,
            count_b,
        });
    }

    classify_state(&stats)
}

fn classify_state(blocks: &[BlockStat]) -> Option<u8> {
    let nb = blocks.len();
    if nb == 0 {
        return None;
    }
    if nb == 1 {
        if blocks[0].size == 4 {
            return Some(1);
        }
        return None;
    }
    if nb == 4 {
        if blocks.iter().all(|b| b.size == 1) {
            return Some(9);
        }
        return None;
    }
    if nb == 2 {
        let b1 = &blocks[0];
        let b2 = &blocks[1];
        if b1.size == 2 && b2.size == 2 {
            let cond_a =
                b1.count_a == 2 && b1.count_b == 0 && b2.count_a == 0 && b2.count_b == 2;
            let cond_b =
                b2.count_a == 2 && b2.count_b == 0 && b1.count_a == 0 && b1.count_b == 2;
            if cond_a || cond_b {
                return Some(2);
            }
            if b1.count_a == 1 && b1.count_b == 1 && b2.count_a == 1 && b2.count_b == 1 {
                return Some(7);
            }
            return None;
        }
        if (b1.size == 3 && b2.size == 1) || (b1.size == 1 && b2.size == 3) {
            let trip = if b1.size == 3 { b1 } else { b2 };
            if trip.count_a == 2 && trip.count_b == 1 {
                return Some(3);
            }
            if trip.count_a == 1 && trip.count_b == 2 {
                return Some(5);
            }
            return None;
        }
        return None;
    }
    if nb == 3 {
        let mut pair_idx: Option<&BlockStat> = None;
        for block in blocks {
            match block.size {
                2 => pair_idx = Some(block),
                1 => continue,
                _ => return None,
            }
        }
        let pair = pair_idx?;
        if pair.count_a == 2 && pair.count_b == 0 {
            return Some(4);
        }
        if pair.count_a == 0 && pair.count_b == 2 {
            return Some(6);
        }
        if pair.count_a == 1 && pair.count_b == 1 {
            return Some(8);
        }
        return None;
    }
    None
}

// ── Helpers ──

fn haps() -> HaplotypeSet {
    HaplotypeSet::new(
        "A#1".to_string(),
        "A#2".to_string(),
        "B#1".to_string(),
        "B#2".to_string(),
    )
}

/// Swapped HaplotypeSet: what was A is now B and vice versa.
fn haps_swapped() -> HaplotypeSet {
    HaplotypeSet::new(
        "B#1".to_string(),
        "B#2".to_string(),
        "A#1".to_string(),
        "A#2".to_string(),
    )
}

fn locus_with_pairs(pairs: &[(&str, &str)]) -> LocusData {
    let mut data = LocusData::default();
    for &(l, r) in pairs {
        data.pairs.insert(Pair::new(l.to_string(), r.to_string()));
    }
    data
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: All 9 delta states are reachable
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_nine_states_reachable() {
    let h = haps();

    // Δ₁: all 4 in one block (a1-a2, a1-b1, a1-b2 → chain)
    let d1 = classify_locus(
        &locus_with_pairs(&[("A#1", "A#2"), ("A#1", "B#1"), ("A#1", "B#2")]),
        &h,
    );
    assert_eq!(d1, Some(1));

    // Δ₂: {a1,a2} and {b1,b2}
    let d2 = classify_locus(&locus_with_pairs(&[("A#1", "A#2"), ("B#1", "B#2")]), &h);
    assert_eq!(d2, Some(2));

    // Δ₃: triplet with 2A+1B, singleton
    let d3 = classify_locus(&locus_with_pairs(&[("A#1", "A#2"), ("A#1", "B#1")]), &h);
    assert_eq!(d3, Some(3));

    // Δ₄: pair {a1,a2}, singletons {b1},{b2}
    let d4 = classify_locus(&locus_with_pairs(&[("A#1", "A#2")]), &h);
    assert_eq!(d4, Some(4));

    // Δ₅: triplet with 1A+2B, singleton
    let d5 = classify_locus(&locus_with_pairs(&[("B#1", "B#2"), ("A#1", "B#1")]), &h);
    assert_eq!(d5, Some(5));

    // Δ₆: pair {b1,b2}, singletons {a1},{a2}
    let d6 = classify_locus(&locus_with_pairs(&[("B#1", "B#2")]), &h);
    assert_eq!(d6, Some(6));

    // Δ₇: {a1,b1} and {a2,b2}
    let d7 = classify_locus(&locus_with_pairs(&[("A#1", "B#1"), ("A#2", "B#2")]), &h);
    assert_eq!(d7, Some(7));

    // Δ₈: pair {a1,b1}, singletons {a2},{b2}
    let d8 = classify_locus(&locus_with_pairs(&[("A#1", "B#1")]), &h);
    assert_eq!(d8, Some(8));

    // Δ₉: no pairs = all singletons
    let d9 = classify_locus(&locus_with_pairs(&[]), &h);
    assert_eq!(d9, Some(9));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: A↔B symmetry
// Swapping A and B should map: 3↔5, 4↔6, others unchanged (1,2,7,8,9)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symmetry_delta3_maps_to_delta5() {
    let h = haps();
    let h_swap = haps_swapped();

    // Δ₃ under original assignment
    let pairs = [("A#1", "A#2"), ("A#1", "B#1")];
    let data = locus_with_pairs(&pairs);
    assert_eq!(classify_locus(&data, &h), Some(3));
    // Same pairs under swapped assignment → should be Δ₅
    assert_eq!(classify_locus(&data, &h_swap), Some(5));
}

#[test]
fn symmetry_delta5_maps_to_delta3() {
    let h = haps();
    let h_swap = haps_swapped();

    let pairs = [("B#1", "B#2"), ("A#1", "B#1")];
    let data = locus_with_pairs(&pairs);
    assert_eq!(classify_locus(&data, &h), Some(5));
    assert_eq!(classify_locus(&data, &h_swap), Some(3));
}

#[test]
fn symmetry_delta4_maps_to_delta6() {
    let h = haps();
    let h_swap = haps_swapped();

    let pairs = [("A#1", "A#2")];
    let data = locus_with_pairs(&pairs);
    assert_eq!(classify_locus(&data, &h), Some(4));
    assert_eq!(classify_locus(&data, &h_swap), Some(6));
}

#[test]
fn symmetry_delta6_maps_to_delta4() {
    let h = haps();
    let h_swap = haps_swapped();

    let pairs = [("B#1", "B#2")];
    let data = locus_with_pairs(&pairs);
    assert_eq!(classify_locus(&data, &h), Some(6));
    assert_eq!(classify_locus(&data, &h_swap), Some(4));
}

#[test]
fn symmetry_preserves_symmetric_states() {
    let h = haps();
    let h_swap = haps_swapped();

    // Δ₁: all connected
    let d1 = locus_with_pairs(&[("A#1", "A#2"), ("A#1", "B#1"), ("A#1", "B#2")]);
    assert_eq!(classify_locus(&d1, &h), classify_locus(&d1, &h_swap));

    // Δ₂: {a1,a2} and {b1,b2}
    let d2 = locus_with_pairs(&[("A#1", "A#2"), ("B#1", "B#2")]);
    assert_eq!(classify_locus(&d2, &h), classify_locus(&d2, &h_swap));

    // Δ₇: cross-pairs
    let d7 = locus_with_pairs(&[("A#1", "B#1"), ("A#2", "B#2")]);
    assert_eq!(classify_locus(&d7, &h), classify_locus(&d7, &h_swap));

    // Δ₈: single cross-pair
    let d8 = locus_with_pairs(&[("A#1", "B#1")]);
    assert_eq!(classify_locus(&d8, &h), classify_locus(&d8, &h_swap));

    // Δ₉: no pairs
    let d9 = locus_with_pairs(&[]);
    assert_eq!(classify_locus(&d9, &h), classify_locus(&d9, &h_swap));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: Monotonicity — adding pairs can only merge blocks
// If state S is produced by pairs P, then P ∪ {new_pair} produces S' ≤ S.
// (Lower delta = more IBD sharing.)
// ═══════════════════════════════════════════════════════════════════════════

/// All possible pairs among the 4 haplotypes.
fn all_possible_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("A#1", "A#2"),
        ("A#1", "B#1"),
        ("A#1", "B#2"),
        ("A#2", "B#1"),
        ("A#2", "B#2"),
        ("B#1", "B#2"),
    ]
}

#[test]
fn monotonicity_adding_pair_never_increases_state() {
    let h = haps();
    let all_pairs = all_possible_pairs();

    // Test all single-pair starting configurations
    for i in 0..all_pairs.len() {
        let base_data = locus_with_pairs(&[all_pairs[i]]);
        let base_state = classify_locus(&base_data, &h);

        // Add each additional pair
        for j in 0..all_pairs.len() {
            if i == j {
                continue;
            }
            let extended_data = locus_with_pairs(&[all_pairs[i], all_pairs[j]]);
            let ext_state = classify_locus(&extended_data, &h);

            if let (Some(base_s), Some(ext_s)) = (base_state, ext_state) {
                assert!(
                    ext_s <= base_s,
                    "Adding pair {} to pair {}: state went from {} to {} (should not increase)",
                    j, i, base_s, ext_s
                );
            }
        }
    }
}

#[test]
fn monotonicity_from_delta9_to_delta1() {
    let h = haps();

    // Start with Δ₉ (no pairs)
    let d9 = classify_locus(&locus_with_pairs(&[]), &h);
    assert_eq!(d9, Some(9));

    // Add one pair → some state < 9
    let d_one = classify_locus(&locus_with_pairs(&[("A#1", "B#1")]), &h);
    assert!(d_one.unwrap() < 9);

    // Add two pairs → state ≤ previous
    let d_two = classify_locus(
        &locus_with_pairs(&[("A#1", "B#1"), ("A#2", "B#2")]),
        &h,
    );
    assert!(d_two.unwrap() <= d_one.unwrap());

    // Add three pairs → state ≤ previous (all connected → Δ₁)
    let d_three = classify_locus(
        &locus_with_pairs(&[("A#1", "B#1"), ("A#2", "B#2"), ("A#1", "A#2")]),
        &h,
    );
    assert_eq!(d_three, Some(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: Self-loops are no-ops
// A pair (X, X) should not affect classification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn self_loop_pairs_are_noop() {
    let h = haps();

    let base = locus_with_pairs(&[("A#1", "B#1")]);
    let base_state = classify_locus(&base, &h);

    // Adding self-loops should not change the state
    let with_selfloops = locus_with_pairs(&[
        ("A#1", "B#1"),
        ("A#1", "A#1"),
        ("B#2", "B#2"),
    ]);
    let sl_state = classify_locus(&with_selfloops, &h);
    assert_eq!(base_state, sl_state, "self-loops should not change classification");
}

#[test]
fn self_loop_only_gives_delta9() {
    let h = haps();
    let data = locus_with_pairs(&[("A#1", "A#1"), ("B#1", "B#1")]);
    let state = classify_locus(&data, &h);
    assert_eq!(state, Some(9), "only self-loops → Δ₉ (no real merges)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: Transitivity — if a1~a2 and a2~b1, then a1~b1 (Union-Find)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn transitivity_chain_merges_all_connected() {
    let h = haps();

    // Chain: a1-a2, a2-b1, b1-b2 → all in one block → Δ₁
    let data = locus_with_pairs(&[("A#1", "A#2"), ("A#2", "B#1"), ("B#1", "B#2")]);
    assert_eq!(classify_locus(&data, &h), Some(1));
}

#[test]
fn transitivity_long_chain_same_as_star() {
    let h = haps();

    // Chain topology: a1-b2, b2-a2, a2-b1
    let chain = locus_with_pairs(&[("A#1", "B#2"), ("B#2", "A#2"), ("A#2", "B#1")]);

    // Star topology: a1-b1, a1-b2, a1-a2
    let star = locus_with_pairs(&[("A#1", "B#1"), ("A#1", "B#2"), ("A#1", "A#2")]);

    // Both should produce Δ₁
    assert_eq!(classify_locus(&chain, &h), Some(1));
    assert_eq!(classify_locus(&star, &h), Some(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: External pairs (not involving known haplotypes) are ignored
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn external_pairs_ignored() {
    let h = haps();

    // Only known-haplotype pairs matter
    let data = locus_with_pairs(&[
        ("A#1", "B#1"),
        ("EXTERNAL#1", "EXTERNAL#2"), // not in haplotype set
        ("A#1", "UNKNOWN#1"),          // one known, one unknown
    ]);
    // Only (A#1, B#1) should affect classification → Δ₈
    assert_eq!(classify_locus(&data, &h), Some(8));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property: Redundant pairs don't change state
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn duplicate_pairs_idempotent() {
    let h = haps();

    let single = locus_with_pairs(&[("A#1", "B#1")]);
    // BTreeSet deduplicates, so duplicate insertion is a no-op
    let mut dup_data = LocusData::default();
    dup_data.pairs.insert(Pair::new("A#1".to_string(), "B#1".to_string()));
    dup_data.pairs.insert(Pair::new("A#1".to_string(), "B#1".to_string()));

    assert_eq!(
        classify_locus(&single, &h),
        classify_locus(&dup_data, &h),
        "duplicate pairs should not change classification"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Exhaustive: all 2^6 subsets of 6 possible pairs produce valid states
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_pair_subsets_produce_valid_or_none_state() {
    let h = haps();
    let all = all_possible_pairs();
    let valid_states: HashSet<u8> = [1, 2, 3, 4, 5, 6, 7, 8, 9].iter().cloned().collect();

    // 2^6 = 64 subsets
    for mask in 0u32..64 {
        let pairs: Vec<(&str, &str)> = (0..6)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| all[i])
            .collect();
        let data = locus_with_pairs(&pairs);
        let state = classify_locus(&data, &h);

        if let Some(s) = state {
            assert!(
                valid_states.contains(&s),
                "mask={}: state {} not in valid set",
                mask, s
            );
        }
        // None is also acceptable for some configurations
    }
}

#[test]
fn exhaustive_all_states_covered_by_subsets() {
    let h = haps();
    let all = all_possible_pairs();
    let mut seen_states: HashSet<u8> = HashSet::new();

    for mask in 0u32..64 {
        let pairs: Vec<(&str, &str)> = (0..6)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| all[i])
            .collect();
        let data = locus_with_pairs(&pairs);
        if let Some(s) = classify_locus(&data, &h) {
            seen_states.insert(s);
        }
    }

    for s in 1..=9u8 {
        assert!(
            seen_states.contains(&s),
            "State {} never produced by any pair subset",
            s
        );
    }
}
