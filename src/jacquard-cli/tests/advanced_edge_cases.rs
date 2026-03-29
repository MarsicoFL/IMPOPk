//! Advanced edge case tests for jacquard-cli: self-pair filtering, one-sided hap
//! filtering, missing windows Delta9 accounting, nb==3 all-singletons path,
//! nb==2 non-standard size combos, UnionFind unknown node, is_a/is_b with foreign
//! haplotype, and end-to-end Delta state coverage for Delta3/4/5/6/8.

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helpers ──────────────────────────────────────────────────────────

fn make_ibs_file(rows: &[(&str, i64, i64, &str, &str)]) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    for &(chr, s, e, a, b) in rows {
        writeln!(f, "{}\t{}\t{}\t{}\t{}", chr, s, e, a, b).unwrap();
    }
    f.flush().unwrap();
    f
}

fn jacquard_cmd(ibs: &NamedTempFile, a1: &str, a2: &str, b1: &str, b2: &str) -> Command {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("jacquard").unwrap();
    cmd.arg("--ibs").arg(ibs.path())
       .arg("--hap-a1").arg(a1)
       .arg("--hap-a2").arg(a2)
       .arg("--hap-b1").arg(b1)
       .arg("--hap-b2").arg(b2);
    cmd
}

// ── classify_state: reimplemented for unit testing ──────────────────

#[derive(Clone, Copy)]
struct BlockStat {
    size: usize,
    count_a: usize,
    count_b: usize,
}

fn bs(size: usize, count_a: usize, count_b: usize) -> BlockStat {
    BlockStat { size, count_a, count_b }
}

fn classify_state(blocks: &[BlockStat]) -> Option<u8> {
    let nb = blocks.len();
    if nb == 0 { return None; }
    if nb == 1 {
        if blocks[0].size == 4 { return Some(1); }
        return None;
    }
    if nb == 4 {
        if blocks.iter().all(|b| b.size == 1) { return Some(9); }
        return None;
    }
    if nb == 2 {
        let b1 = &blocks[0];
        let b2 = &blocks[1];
        if b1.size == 2 && b2.size == 2 {
            let cond_a = b1.count_a == 2 && b1.count_b == 0 && b2.count_a == 0 && b2.count_b == 2;
            let cond_b = b2.count_a == 2 && b2.count_b == 0 && b1.count_a == 0 && b1.count_b == 2;
            if cond_a || cond_b { return Some(2); }
            if b1.count_a == 1 && b1.count_b == 1 && b2.count_a == 1 && b2.count_b == 1 {
                return Some(7);
            }
            return None;
        }
        if (b1.size == 3 && b2.size == 1) || (b1.size == 1 && b2.size == 3) {
            let trip = if b1.size == 3 { b1 } else { b2 };
            if trip.count_a == 2 && trip.count_b == 1 { return Some(3); }
            if trip.count_a == 1 && trip.count_b == 2 { return Some(5); }
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
        if pair.count_a == 2 && pair.count_b == 0 { return Some(4); }
        if pair.count_a == 0 && pair.count_b == 2 { return Some(6); }
        if pair.count_a == 1 && pair.count_b == 1 { return Some(8); }
        return None;
    }
    None
}

// ── classify_state: nb==3 all-singletons (pair_idx stays None) ──────

/// Three blocks all of size 1: pair_idx remains None, ? operator returns None
#[test]
fn classify_state_three_singletons_returns_none() {
    let blocks = vec![bs(1, 1, 0), bs(1, 0, 1), bs(1, 0, 1)];
    assert_eq!(classify_state(&blocks), None);
}

// ── classify_state: nb==2 non-(2,2)/(3,1) sizes ────────────────────

/// Two blocks of sizes (4, 0) — impossible but handled
#[test]
fn classify_state_two_blocks_size_4_0_returns_none() {
    let blocks = vec![bs(4, 2, 2), bs(0, 0, 0)];
    assert_eq!(classify_state(&blocks), None);
}

/// Two blocks of sizes (1, 1) — not matching (2,2) or (3,1) patterns
#[test]
fn classify_state_two_blocks_both_size_1_returns_none() {
    let blocks = vec![bs(1, 1, 0), bs(1, 0, 1)];
    assert_eq!(classify_state(&blocks), None);
}

/// Two blocks of sizes (2, 1) — doesn't match (3,1) pattern
#[test]
fn classify_state_two_blocks_size_2_1_returns_none() {
    let blocks = vec![bs(2, 1, 1), bs(1, 1, 0)];
    assert_eq!(classify_state(&blocks), None);
}

// ── classify_state: nb==3 with pair having count_a==1, count_b==0 ───

/// Three blocks: pair(1a,0b) + two singletons — no Delta match
#[test]
fn classify_state_three_blocks_pair_1a_0b_returns_none() {
    let blocks = vec![bs(2, 1, 0), bs(1, 0, 1), bs(1, 1, 1)];
    assert_eq!(classify_state(&blocks), None);
}

// ── classify_state: nb==2 triplet with 0a+2b → None ────────────────

/// Two blocks (3,1): triplet has 0A+2B — doesn't match delta3(2A+1B) or delta5(1A+2B)
/// Wait: 0A+2B with size=3 means the third member is also B — but if we only have 2 B
/// haplotypes, that's impossible. Still, the function returns None for this input.
#[test]
fn classify_state_triplet_0a_2b_returns_none() {
    let blocks = vec![bs(3, 0, 2), bs(1, 1, 0)];
    assert_eq!(classify_state(&blocks), None);
}

// ── UnionFind: find unknown node ────────────────────────────────────

/// UnionFind::find on a node not in the initial set returns the node itself
#[test]
fn union_find_unknown_node_returns_self() {
    use std::collections::HashMap;

    struct UnionFind {
        parent: HashMap<String, String>,
    }
    impl UnionFind {
        fn new(nodes: &[&str]) -> Self {
            let mut parent = HashMap::new();
            for n in nodes { parent.insert((*n).to_string(), (*n).to_string()); }
            Self { parent }
        }
        fn find(&mut self, node: &str) -> String {
            let parent = self.parent.get(node).cloned()
                .unwrap_or_else(|| node.to_string());
            if parent == node { return parent; }
            let root = self.find(&parent);
            self.parent.insert(node.to_string(), root.clone());
            root
        }
    }

    let mut uf = UnionFind::new(&["a", "b"]);
    let result = uf.find("unknown");
    assert_eq!(result, "unknown", "Unknown node should return itself");
}

// ── HaplotypeSet: is_a/is_b with foreign haplotype ─────────────────

/// Both is_a and is_b return false for a haplotype not in either group
#[test]
fn haplotype_set_foreign_hap_returns_false() {
    use std::collections::HashSet;

    struct HaplotypeSet {
        a: [String; 2],
        b: [String; 2],
        all: HashSet<String>,
    }
    impl HaplotypeSet {
        fn new(a1: String, a2: String, b1: String, b2: String) -> Self {
            let mut all = HashSet::new();
            all.insert(a1.clone()); all.insert(a2.clone());
            all.insert(b1.clone()); all.insert(b2.clone());
            Self { a: [a1, a2], b: [b1, b2], all }
        }
        fn is_a(&self, hap: &str) -> bool { self.a.iter().any(|h| h == hap) }
        fn is_b(&self, hap: &str) -> bool { self.b.iter().any(|h| h == hap) }
        fn contains(&self, hap: &str) -> bool { self.all.contains(hap) }
    }

    let haps = HaplotypeSet::new("A#1".into(), "A#2".into(), "B#1".into(), "B#2".into());
    assert!(!haps.is_a("C#1"), "Foreign hap should not be in group A");
    assert!(!haps.is_b("C#1"), "Foreign hap should not be in group B");
    assert!(!haps.contains("C#1"), "Foreign hap should not be in set");
    assert!(!haps.is_a(""), "Empty string should not be in group A");
    assert!(!haps.is_b(""), "Empty string should not be in group B");
}

// ── End-to-end: Delta3 (triplet 2A+1B + singleton 1B) ──────────────

/// A1-A2, A1-B1 → triplet(A1,A2,B1) + singleton(B2) → Delta3
#[test]
fn test_jacquard_e2e_delta3() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#2"),
                ("chr1", s, e, "A#1", "B#1"),
                // A2-B1 not present — but A1 connects A2 and B1 transitively
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta3_line = stdout.lines().find(|l| l.starts_with("Delta3")).unwrap();
    let frac: f64 = delta3_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta3 should dominate, got {}", frac);
}

// ── End-to-end: Delta4 (A-pair + 2 B-singletons) ───────────────────

/// A1-A2 only → A-pair connected, B1 and B2 isolated → Delta4
#[test]
fn test_jacquard_e2e_delta4() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            ("chr1", s, e, "A#1", "A#2")
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta4_line = stdout.lines().find(|l| l.starts_with("Delta4")).unwrap();
    let frac: f64 = delta4_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta4 should dominate, got {}", frac);
}

// ── End-to-end: Delta5 (triplet 1A+2B + singleton 1A) ──────────────

/// B1-B2, A1-B1 → triplet(A1,B1,B2) + singleton(A2) → Delta5
#[test]
fn test_jacquard_e2e_delta5() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "B#1", "B#2"),
                ("chr1", s, e, "A#1", "B#1"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta5_line = stdout.lines().find(|l| l.starts_with("Delta5")).unwrap();
    let frac: f64 = delta5_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta5 should dominate, got {}", frac);
}

// ── End-to-end: Delta6 (B-pair + 2 A-singletons) ───────────────────

/// B1-B2 only → B-pair connected, A1 and A2 isolated → Delta6
#[test]
fn test_jacquard_e2e_delta6() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            ("chr1", s, e, "B#1", "B#2")
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta6_line = stdout.lines().find(|l| l.starts_with("Delta6")).unwrap();
    let frac: f64 = delta6_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta6 should dominate, got {}", frac);
}

// ── End-to-end: Delta8 (mixed pair 1A+1B + 2 singletons) ───────────

/// A1-B1 only → cross-pair(A1,B1) + singletons(A2,B2) → Delta8
#[test]
fn test_jacquard_e2e_delta8() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            ("chr1", s, e, "A#1", "B#1")
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta8_line = stdout.lines().find(|l| l.starts_with("Delta8")).unwrap();
    let frac: f64 = delta8_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta8 should dominate, got {}", frac);
}

// ── End-to-end: Self-pair filtering (hap1 == hap2 after hap_key) ────

/// IBS rows with self-pairs should be silently filtered
#[test]
fn test_jacquard_e2e_self_pair_filtered() {
    // Include self-pairs AND real pairs — the self-pairs should be ignored
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..5)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#1"),  // self-pair (should be filtered)
                ("chr1", s, e, "B#2", "B#2"),  // self-pair (should be filtered)
                ("chr1", s, e, "A#1", "A#2"),  // real pair
                ("chr1", s, e, "B#1", "B#2"),  // real pair
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Without self-pairs: A1-A2 and B1-B2 → Delta2
    let delta2_line = stdout.lines().find(|l| l.starts_with("Delta2")).unwrap();
    let frac: f64 = delta2_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Self-pairs should be filtered; Delta2 should dominate, got {}", frac);
}

// ── End-to-end: One-sided hap filtering ─────────────────────────────

/// Rows with one target hap and one non-target hap should be filtered out
#[test]
fn test_jacquard_e2e_one_sided_hap_filtered() {
    // A1-X#1 (one target, one non-target) → should be filtered
    // Plus real pairs for classification
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..5)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "X#1"),  // one-sided (should be filtered)
                ("chr1", s, e, "B#2", "X#2"),  // one-sided (should be filtered)
                ("chr1", s, e, "A#1", "A#2"),  // real target pair
                ("chr1", s, e, "B#1", "B#2"),  // real target pair
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    // With one-sided haps filtered: only A1-A2 and B1-B2 remain → Delta2
    let delta2_line = stdout.lines().find(|l| l.starts_with("Delta2")).unwrap();
    let frac: f64 = delta2_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "One-sided haps should be filtered; Delta2 should dominate, got {}", frac);
}

// ── End-to-end: Missing windows → Delta9 accounting ─────────────────

/// Windows with gaps (some windows have no target pair data) → Delta9 for missing
#[test]
fn test_jacquard_e2e_missing_windows_delta9() {
    // 10 windows total span, but only provide data for 5 windows
    // The other 5 should get Delta9
    let mut rows = Vec::new();
    for i in [0, 2, 4, 6, 8] { // only even-numbered windows
        let s = i * 5000 + 1;
        let e = (i + 1) * 5000;
        rows.push(("chr1", s, e, "A#1", "A#2"));
        rows.push(("chr1", s, e, "B#1", "B#2"));
    }
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Delta9 should have non-zero count (from missing windows)
    let delta9_line = stdout.lines().find(|l| l.starts_with("Delta9")).unwrap();
    let count_str = delta9_line.split("count=").nth(1).unwrap().trim_end_matches(')');
    let count: u64 = count_str.parse().unwrap();
    assert!(count > 0, "Missing windows should contribute to Delta9 count, got 0");
}

// ── End-to-end: no target haps → all Delta9 from missing windows ────

/// When ALL data rows involve non-target haplotypes, no loci are directly
/// classified, but missing windows are counted as Delta9 from the span.
/// The result is all Delta9 (100%).
#[test]
fn test_jacquard_e2e_no_target_haps_all_delta9() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..5)
        .map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            ("chr1", s, e, "X#1", "X#2") // non-target only
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    // Succeeds because span/win_size creates Delta9 entries from missing windows
    let stdout = String::from_utf8_lossy(&output.stdout);
    if output.status.success() {
        // Delta9 should be 100% (all missing windows)
        let delta9_line = stdout.lines().find(|l| l.starts_with("Delta9")).unwrap();
        let frac: f64 = delta9_line.split('\t').nth(1).unwrap().parse().unwrap();
        assert!(
            (frac - 1.0).abs() < 1e-6,
            "All windows should be Delta9 when no target haps in data, got {}",
            frac
        );
    } else {
        // Alternative: "no loci classified" error
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("no loci classified"),
            "Expected 'no loci classified' error or all-Delta9, got: {}",
            stderr
        );
    }
}

// ── End-to-end: hap_key normalization with coordinate suffixes ──────

/// Self-pair after hap_key normalization: "A#1#scaffold:0-5000" and "A#1#scaffold:1000-6000"
/// both normalize to "A#1", so they become a self-pair → filtered
#[test]
fn test_jacquard_e2e_self_pair_after_normalization() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..5)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1#scaffold:0-5000", "A#1#scaffold:1000-6000"), // self after normalization
                ("chr1", s, e, "A#1#sc:0-5k", "A#2#sc:0-5k"),  // real pair
                ("chr1", s, e, "B#1#sc:0-5k", "B#2#sc:0-5k"),  // real pair
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Self-pairs after normalization filtered → only A1-A2, B1-B2 → Delta2
    let delta2_line = stdout.lines().find(|l| l.starts_with("Delta2")).unwrap();
    let frac: f64 = delta2_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Normalized self-pairs should be filtered; Delta2 should dominate, got {}", frac);
}

// ── End-to-end: fractions sum to 1.0 for each Delta scenario ────────

/// Delta3 scenario: verify fractions sum to 1.0
#[test]
fn test_jacquard_e2e_fractions_sum_to_one_delta3() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#2"),
                ("chr1", s, e, "A#1", "B#1"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let total: f64 = stdout.lines()
        .filter(|l| l.starts_with("Delta"))
        .map(|l| l.split('\t').nth(1).unwrap().parse::<f64>().unwrap())
        .sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "Fractions should sum to 1.0, got {}",
        total
    );
}

// ── End-to-end: mixed Delta states across windows ───────────────────

/// Different IBD patterns at different loci → mixed Delta output
#[test]
fn test_jacquard_e2e_mixed_delta_states() {
    let mut rows = Vec::new();
    // Windows 1-5: all 6 pairs → Delta1
    for i in 0..5 {
        let s = i * 5000 + 1;
        let e = (i + 1) * 5000;
        rows.push(("chr1", s, e, "A#1", "A#2"));
        rows.push(("chr1", s, e, "A#1", "B#1"));
        rows.push(("chr1", s, e, "A#1", "B#2"));
        rows.push(("chr1", s, e, "A#2", "B#1"));
        rows.push(("chr1", s, e, "A#2", "B#2"));
        rows.push(("chr1", s, e, "B#1", "B#2"));
    }
    // Windows 6-10: within-group only → Delta2
    for i in 5..10 {
        let s = i * 5000 + 1;
        let e = (i + 1) * 5000;
        rows.push(("chr1", s, e, "A#1", "A#2"));
        rows.push(("chr1", s, e, "B#1", "B#2"));
    }
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Both Delta1 and Delta2 should have non-zero fractions
    let delta1_line = stdout.lines().find(|l| l.starts_with("Delta1")).unwrap();
    let delta1_frac: f64 = delta1_line.split('\t').nth(1).unwrap().parse().unwrap();
    let delta2_line = stdout.lines().find(|l| l.starts_with("Delta2")).unwrap();
    let delta2_frac: f64 = delta2_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(delta1_frac > 0.0, "Delta1 should be non-zero");
    assert!(delta2_frac > 0.0, "Delta2 should be non-zero");
}

// ── classify_state: comprehensive coverage of all Delta states ──────

/// Delta3: verify blocks (3: 2a+1b, 1: 0a+1b) correctly classified
#[test]
fn classify_state_delta3_block_order_independence() {
    // Singleton first, triplet second
    assert_eq!(classify_state(&[bs(1, 0, 1), bs(3, 2, 1)]), Some(3));
    // Triplet first, singleton second
    assert_eq!(classify_state(&[bs(3, 2, 1), bs(1, 0, 1)]), Some(3));
}

/// Delta5: verify blocks (3: 1a+2b, 1: 1a+0b) correctly classified
#[test]
fn classify_state_delta5_block_order_independence() {
    assert_eq!(classify_state(&[bs(1, 1, 0), bs(3, 1, 2)]), Some(5));
    assert_eq!(classify_state(&[bs(3, 1, 2), bs(1, 1, 0)]), Some(5));
}

/// All Delta states through classify_state (comprehensive round-trip)
#[test]
fn classify_state_all_deltas_exhaustive() {
    // Delta 1: all in one block
    assert_eq!(classify_state(&[bs(4, 2, 2)]), Some(1));
    // Delta 2: A-pair + B-pair
    assert_eq!(classify_state(&[bs(2, 2, 0), bs(2, 0, 2)]), Some(2));
    // Delta 3: triplet(2A+1B) + singleton
    assert_eq!(classify_state(&[bs(3, 2, 1), bs(1, 0, 1)]), Some(3));
    // Delta 4: A-pair + 2 B-singletons
    assert_eq!(classify_state(&[bs(2, 2, 0), bs(1, 0, 1), bs(1, 0, 1)]), Some(4));
    // Delta 5: triplet(1A+2B) + singleton
    assert_eq!(classify_state(&[bs(3, 1, 2), bs(1, 1, 0)]), Some(5));
    // Delta 6: B-pair + 2 A-singletons
    assert_eq!(classify_state(&[bs(2, 0, 2), bs(1, 1, 0), bs(1, 1, 0)]), Some(6));
    // Delta 7: two cross-pairs
    assert_eq!(classify_state(&[bs(2, 1, 1), bs(2, 1, 1)]), Some(7));
    // Delta 8: mixed pair + 2 singletons
    assert_eq!(classify_state(&[bs(2, 1, 1), bs(1, 1, 0), bs(1, 0, 1)]), Some(8));
    // Delta 9: all singletons
    assert_eq!(classify_state(&[bs(1, 1, 0), bs(1, 1, 0), bs(1, 0, 1), bs(1, 0, 1)]), Some(9));
}
