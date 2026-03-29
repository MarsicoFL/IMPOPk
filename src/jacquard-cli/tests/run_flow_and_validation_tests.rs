//! Additional tests for jacquard-cli run() flow that aren't covered by cli_edge_cases.rs:
//! - Span/win_size divisibility warning path
//! - Multiple chromosomes handling
//! - Large gap between windows → many Delta9
//! - Duplicate pair deduplication within a locus
//! - Header with "chrom" keyword detected and skipped

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helpers ────────────────────────────────────────────────────────────

fn make_ibs_file(rows: &[(&str, i64, i64, &str, &str)]) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    for &(chr, s, e, a, b) in rows {
        writeln!(f, "{}\t{}\t{}\t{}\t{}", chr, s, e, a, b).unwrap();
    }
    f.flush().unwrap();
    f
}

fn jacquard_cmd(ibs: &NamedTempFile) -> Command {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("jacquard").unwrap();
    cmd.arg("--ibs").arg(ibs.path())
       .arg("--hap-a1").arg("A#1")
       .arg("--hap-a2").arg("A#2")
       .arg("--hap-b1").arg("B#1")
       .arg("--hap-b2").arg("B#2");
    cmd
}

// ── Span divisibility warning ──────────────────────────────────────────

#[test]
fn span_not_divisible_by_win_size_warns() {
    // Window 1: 1-5000 (size=5000), Window 2: 5001-7000 (size=2000)
    // span=7000, win_size=5000 → 7000 % 5000 = 2000 → WARNING
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "A#2"),
        ("chr1", 1, 5000, "B#1", "B#2"),
        ("chr1", 5001, 7000, "A#1", "A#2"),
        ("chr1", 5001, 7000, "B#1", "B#2"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("WARNING") && stderr.contains("not divisible"),
        "Expected span divisibility warning, got: {}", stderr
    );
}

// ── Deduplication of pairs within a locus ──────────────────────────────

#[test]
fn duplicate_pair_at_same_locus_deduplicated() {
    // Same pair appears twice at the same locus → should be deduplicated
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "A#2"),
        ("chr1", 1, 5000, "A#1", "A#2"), // duplicate
        ("chr1", 1, 5000, "B#1", "B#2"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success(), "Should handle duplicate pairs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Still Delta2 despite duplicate — dedup handled by BTreeSet
    assert!(stdout.contains("Delta2"), "Expected Delta2, got: {}", stdout);
}

// ── Header detection with case-insensitive "chrom" ─────────────────────

#[test]
fn header_with_uppercase_chrom_skipped() {
    // "CHROM" (uppercase) should be detected as header and skipped
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "CHROM\tstart\tend\tgroup.a\tgroup.b").unwrap();
    writeln!(f, "chr1\t1\t5000\tA#1\tA#2").unwrap();
    writeln!(f, "chr1\t1\t5000\tB#1\tB#2").unwrap();
    f.flush().unwrap();

    let output = jacquard_cmd(&f).output().unwrap();
    assert!(
        output.status.success(),
        "Should skip case-insensitive 'CHROM' header, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn no_header_first_row_treated_as_data() {
    // First field is NOT "chrom" (case insensitive) → treated as data
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chr1\t1\t5000\tA#1\tA#2").unwrap();
    writeln!(f, "chr1\t1\t5000\tB#1\tB#2").unwrap();
    f.flush().unwrap();

    let output = jacquard_cmd(&f).output().unwrap();
    assert!(
        output.status.success(),
        "Should treat first row as data when no header keyword, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ── Many missing windows → many Delta9 ─────────────────────────────────

#[test]
fn large_gap_produces_many_delta9() {
    // Window at 1-5000 and 50001-55000. Gap: 5001-50000 = 9 missing windows
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "A#2"),
        ("chr1", 1, 5000, "B#1", "B#2"),
        ("chr1", 50001, 55000, "A#1", "A#2"),
        ("chr1", 50001, 55000, "B#1", "B#2"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Delta9 should dominate (9 missing + 0 unclassified vs 2 loci with Delta2)
    let delta9_line = stdout.lines().find(|l| l.starts_with("Delta9")).unwrap();
    let frac: f64 = delta9_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(frac > 0.5, "Delta9 should dominate with large gap, got {}", frac);
}

// ── Pair reversed relative to canonical but should still match ──────

#[test]
fn reversed_pair_in_ibs_still_matches_target() {
    // B#2 before A#1 lexicographically but stored as pair (A#1, B#2) after canonical ordering
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "B#2", "A#1"), // reversed but should still match
        ("chr1", 1, 5000, "A#2", "B#1"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(
        output.status.success(),
        "Should handle reversed pairs, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A1-B2 and A2-B1 connected → Delta7
    assert!(stdout.contains("Delta7"), "Expected Delta7, got: {}", stdout);
}

// ── Single locus with triplet patterns (Delta3/Delta5) ──────────────

#[test]
fn delta3_from_a_pair_plus_cross_pair() {
    // A1-A2 (within A) + A1-B1 (cross) → block {A1,A2,B1} size=3 (2A+1B) + {B2} → Delta3
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "A#2"),
        ("chr1", 1, 5000, "A#1", "B#1"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta3"), "Expected Delta3 from triplet(2A+1B), got: {}", stdout);
}

#[test]
fn delta5_from_b_pair_plus_cross_pair() {
    // B1-B2 (within B) + A1-B1 (cross) → block {A1,B1,B2} size=3 (1A+2B) + {A2} → Delta5
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "B#1", "B#2"),
        ("chr1", 1, 5000, "A#1", "B#1"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta5"), "Expected Delta5 from triplet(1A+2B), got: {}", stdout);
}

// ── Delta4 and Delta6 from isolated pair ────────────────────────────

#[test]
fn delta4_from_a_pair_only() {
    // Only A1-A2 connected → block {A1,A2} (2A+0B) + {B1} + {B2} → Delta4
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "A#2"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta4"), "Expected Delta4 from A-pair only, got: {}", stdout);
}

#[test]
fn delta6_from_b_pair_only() {
    // Only B1-B2 connected → block {B1,B2} (0A+2B) + {A1} + {A2} → Delta6
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "B#1", "B#2"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta6"), "Expected Delta6 from B-pair only, got: {}", stdout);
}

// ── Delta8 from single mixed pair ───────────────────────────────────

#[test]
fn delta8_from_single_cross_pair() {
    // Only A1-B1 connected → block {A1,B1} (1A+1B) + {A2} + {B2} → Delta8
    let rows = vec![
        ("chr1", 1_i64, 5000_i64, "A#1", "B#1"),
    ];
    let ibs = make_ibs_file(&rows);
    let output = jacquard_cmd(&ibs).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta8"), "Expected Delta8 from single cross pair, got: {}", stdout);
}
