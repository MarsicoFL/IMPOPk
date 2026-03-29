//! Direct unit tests for jacquard-cli's classify_locus function (Delta 3-8),
//! Pair deduplication in BTreeSet, hap_key edge cases, and classify_state
//! additional branches.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════
// Reimplementations of private types and functions from jacquard main.rs
// These mirror the actual implementations exactly.
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

#[derive(Default)]
struct LocusData {
    pairs: BTreeSet<Pair>,
}

struct HaplotypeSet {
    a: [String; 2],
    b: [String; 2],
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

    fn contains(&self, hap: &str) -> bool {
        self.all.contains(hap)
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
            let cond_a = b1.count_a == 2 && b1.count_b == 0 && b2.count_a == 0 && b2.count_b == 2;
            let cond_b = b2.count_a == 2 && b2.count_b == 0 && b1.count_a == 0 && b1.count_b == 2;
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

fn hap_key(raw: &str) -> String {
    let mut parts = raw.split('#');
    match (parts.next(), parts.next()) {
        (Some(sample), Some(hap)) => format!("{}#{}", sample, hap),
        _ => raw.to_string(),
    }
}

fn bs(size: usize, count_a: usize, count_b: usize) -> BlockStat {
    BlockStat { size, count_a, count_b }
}

fn make_haps() -> HaplotypeSet {
    HaplotypeSet::new("A#1".into(), "A#2".into(), "B#1".into(), "B#2".into())
}

// ═══════════════════════════════════════════════════════════════════════════
// classify_locus — direct unit tests for Delta 3 through 8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_locus_delta3_a1a2_a1b1() {
    let haps = make_haps();
    // A1-A2 + A1-B1 → triplet(A1,A2,B1) + singleton(B2) → Delta 3 (2A+1B)
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    assert_eq!(classify_locus(&data, &haps), Some(3));
}

#[test]
fn classify_locus_delta3_a2b2_a1a2() {
    let haps = make_haps();
    // A1-A2 + A2-B2 → triplet(A1,A2,B2) + singleton(B1) → Delta 3
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(3));
}

#[test]
fn classify_locus_delta4_only_a_pair() {
    let haps = make_haps();
    // A1-A2 only → A-pair + 2 B-singletons → Delta 4
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(4));
}

#[test]
fn classify_locus_delta5_b1b2_a1b1() {
    let haps = make_haps();
    // B1-B2 + A1-B1 → triplet(A1,B1,B2) + singleton(A2) → Delta 5 (1A+2B)
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    assert_eq!(classify_locus(&data, &haps), Some(5));
}

#[test]
fn classify_locus_delta5_b1b2_a2b2() {
    let haps = make_haps();
    // B1-B2 + A2-B2 → triplet(A2,B1,B2) + singleton(A1) → Delta 5
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(5));
}

#[test]
fn classify_locus_delta6_only_b_pair() {
    let haps = make_haps();
    // B1-B2 only → B-pair + 2 A-singletons → Delta 6
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(6));
}

#[test]
fn classify_locus_delta7_cross_pairs() {
    let haps = make_haps();
    // A1-B1 + A2-B2 → two cross-group blocks → Delta 7
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(7));
}

#[test]
fn classify_locus_delta7_alternate_cross_pairs() {
    let haps = make_haps();
    // A1-B2 + A2-B1 → two cross-group blocks → Delta 7
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "B#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#1".into()));
    assert_eq!(classify_locus(&data, &haps), Some(7));
}

#[test]
fn classify_locus_delta8_single_mixed_pair() {
    let haps = make_haps();
    // A1-B1 only → mixed pair + A2 singleton + B2 singleton → Delta 8
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    assert_eq!(classify_locus(&data, &haps), Some(8));
}

#[test]
fn classify_locus_delta8_a2b2_only() {
    let haps = make_haps();
    // A2-B2 only → mixed pair + A1 singleton + B1 singleton → Delta 8
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(8));
}

#[test]
fn classify_locus_delta8_a1b2_only() {
    let haps = make_haps();
    // A1-B2 only → Delta 8
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(8));
}

// ═══════════════════════════════════════════════════════════════════════════
// Pair deduplication in BTreeSet
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pair_btreeset_deduplication() {
    let mut data = LocusData::default();
    // Insert same pair twice — BTreeSet should deduplicate
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    assert_eq!(data.pairs.len(), 1);
}

#[test]
fn pair_btreeset_order_matters() {
    let mut data = LocusData::default();
    // Pair(A,B) != Pair(B,A) — they are different entries
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    data.pairs.insert(Pair::new("B#1".into(), "A#1".into()));
    assert_eq!(data.pairs.len(), 2);
}

#[test]
fn pair_dedup_does_not_affect_classification() {
    let haps = make_haps();
    // Insert A1-A2 twice — should still be Delta 4, not double-counted
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    assert_eq!(data.pairs.len(), 1);
    assert_eq!(classify_locus(&data, &haps), Some(4));
}

// ═══════════════════════════════════════════════════════════════════════════
// hap_key — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hap_key_hash_only() {
    // "#" → parts = ["", ""] → format!("{}#{}", "", "") = "#"
    assert_eq!(hap_key("#"), "#");
}

#[test]
fn hap_key_multiple_hash_empty_sample() {
    // "#1#suffix" → parts = ["", "1", "suffix"] → format!("{}#{}", "", "1") = "#1"
    assert_eq!(hap_key("#1#suffix"), "#1");
}

#[test]
fn hap_key_double_hash() {
    // "##" → parts = ["", "", ""] → format!("{}#{}", "", "") = "#"
    assert_eq!(hap_key("##"), "#");
}

#[test]
fn hap_key_hash_only_hap_number() {
    // "#2" → format!("{}#{}", "", "2") = "#2"
    assert_eq!(hap_key("#2"), "#2");
}

// ═══════════════════════════════════════════════════════════════════════════
// classify_state — additional branch coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_state_single_block_size_2_returns_none() {
    // nb==1, size==2 (not 4) → None
    assert_eq!(classify_state(&[bs(2, 1, 1)]), None);
}

#[test]
fn classify_state_single_block_size_0_returns_none() {
    // nb==1, size==0 → None
    assert_eq!(classify_state(&[bs(0, 0, 0)]), None);
}

#[test]
fn classify_state_nb3_pair_count_a0_b0_returns_none() {
    // nb==3 with a pair (size 2) but count_a==0 and count_b==0
    assert_eq!(classify_state(&[bs(2, 0, 0), bs(1, 1, 0), bs(1, 0, 1)]), None);
}

#[test]
fn classify_state_nb3_pair_count_a1_b0_returns_none() {
    // nb==3 with pair having 1A+0B — doesn't match Delta4(2A+0B) or Delta8(1A+1B)
    assert_eq!(classify_state(&[bs(2, 1, 0), bs(1, 1, 0), bs(1, 0, 1)]), None);
}

#[test]
fn classify_state_nb3_pair_count_a0_b1_returns_none() {
    // nb==3 with pair having 0A+1B — doesn't match Delta6(0A+2B) or Delta8(1A+1B)
    assert_eq!(classify_state(&[bs(2, 0, 1), bs(1, 1, 0), bs(1, 0, 1)]), None);
}

#[test]
fn classify_state_nb2_sizes_1_3_trip_0a_0b_returns_none() {
    // nb==2, triplet with 0A+0B (impossible in real data, but tests the None path)
    assert_eq!(classify_state(&[bs(3, 0, 0), bs(1, 0, 0)]), None);
}

#[test]
fn classify_state_nb2_sizes_4_0_returns_none() {
    // nb==2 with size 4+0 — neither (2,2) nor (3,1) pattern
    assert_eq!(classify_state(&[bs(4, 2, 2), bs(0, 0, 0)]), None);
}

#[test]
fn classify_state_nb2_both_size_1_returns_none() {
    // nb==2 with size 1+1 — neither (2,2) nor (3,1)
    assert_eq!(classify_state(&[bs(1, 1, 0), bs(1, 0, 1)]), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// UnionFind — path compression verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn union_find_path_compression_three_levels() {
    let mut uf = UnionFind::new(&["a", "b", "c"]);
    uf.union("a", "b"); // b's root = a
    uf.union("b", "c"); // c's root → b → a; after find, c → a directly

    let root = uf.find("c");
    assert_eq!(root, uf.find("a"));

    // After path compression, c's parent should be directly the root
    // Verify by checking that a second find is the same
    let root2 = uf.find("c");
    assert_eq!(root, root2);
}

#[test]
fn union_find_self_union_is_noop() {
    let mut uf = UnionFind::new(&["a", "b"]);
    uf.union("a", "a");
    // a is still its own root
    assert_eq!(uf.find("a"), "a");
    // b is unaffected
    assert_ne!(uf.find("a"), uf.find("b"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Pair — Ord trait correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pair_ordering_left_field_primary() {
    let p1 = Pair::new("A".into(), "Z".into());
    let p2 = Pair::new("B".into(), "A".into());
    assert!(p1 < p2); // "A" < "B" in left field
}

#[test]
fn pair_ordering_right_field_secondary() {
    let p1 = Pair::new("A".into(), "B".into());
    let p2 = Pair::new("A".into(), "C".into());
    assert!(p1 < p2); // Same left, "B" < "C" in right
}

#[test]
fn pair_equality() {
    let p1 = Pair::new("X".into(), "Y".into());
    let p2 = Pair::new("X".into(), "Y".into());
    assert_eq!(p1, p2);
}

// ═══════════════════════════════════════════════════════════════════════════
// classify_locus — transitive union-find scenarios
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn classify_locus_transitive_reaches_delta1() {
    let haps = make_haps();
    // A1-A2, A2-B1, B1-B2 → all connected transitively → Delta 1
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#1".into()));
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(1));
}

#[test]
fn classify_locus_chain_3_gives_delta3() {
    let haps = make_haps();
    // A1-A2, A2-B1 → chain of 3 → triplet(2A+1B) → Delta 3
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#1".into()));
    assert_eq!(classify_locus(&data, &haps), Some(3));
}

#[test]
fn classify_locus_redundant_pairs_still_delta1() {
    let haps = make_haps();
    // All 6 possible pairs plus duplicates — still Delta 1
    let mut data = LocusData::default();
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
    data.pairs.insert(Pair::new("A#1".into(), "B#2".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#1".into()));
    data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    // Add duplicates
    data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
    data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
    assert_eq!(classify_locus(&data, &haps), Some(1));
}
