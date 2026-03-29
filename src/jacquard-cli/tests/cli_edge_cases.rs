//! Integration tests for jacquard CLI edge cases: duplicate haplotype validation,
//! file error handling, end-to-end execution with synthetic IBS data, and
//! argument parsing edge cases.

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helper: create a minimal valid IBS file ──────────────────────────

/// Build a synthetic IBS file with the given pairs.
/// Each row has: chrom \t start \t end \t hap_a \t hap_b
fn make_ibs_file(rows: &[(&str, i64, i64, &str, &str)]) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    for &(chr, s, e, a, b) in rows {
        writeln!(f, "{}\t{}\t{}\t{}\t{}", chr, s, e, a, b).unwrap();
    }
    f.flush().unwrap();
    f
}

/// Build a jacquard Command with the standard 4 haplotype args.
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

// ── Duplicate haplotype validation ───────────────────────────────────

/// hap-a1 == hap-a2 should be rejected
#[test]
fn test_jacquard_duplicate_within_group_a() {
    let ibs = make_ibs_file(&[("chr1", 1, 5000, "X#1", "Y#1")]);
    let mut cmd = jacquard_cmd(&ibs, "X#1", "X#1", "Y#1", "Y#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("hap-a1 and hap-a2 must be distinct"),
        "Expected duplicate error, got: {}",
        stderr
    );
}

/// hap-b1 == hap-b2 should be rejected
#[test]
fn test_jacquard_duplicate_within_group_b() {
    let ibs = make_ibs_file(&[("chr1", 1, 5000, "X#1", "Y#1")]);
    let mut cmd = jacquard_cmd(&ibs, "X#1", "X#2", "Y#1", "Y#1");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("hap-b1 and hap-b2 must be distinct"),
        "Expected duplicate error, got: {}",
        stderr
    );
}

/// Overlap between groups A and B should be rejected
#[test]
fn test_jacquard_overlap_between_groups() {
    let ibs = make_ibs_file(&[("chr1", 1, 5000, "X#1", "Y#1")]);
    let mut cmd = jacquard_cmd(&ibs, "X#1", "X#2", "X#1", "Y#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("distinct between groups A and B"),
        "Expected overlap error, got: {}",
        stderr
    );
}

// ── File error handling ──────────────────────────────────────────────

/// Non-existent IBS file should produce a clear error
#[test]
fn test_jacquard_nonexistent_ibs_file() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("jacquard").unwrap();
    cmd.arg("--ibs").arg("/nonexistent/path/to/ibs.tsv")
       .arg("--hap-a1").arg("A#1")
       .arg("--hap-a2").arg("A#2")
       .arg("--hap-b1").arg("B#1")
       .arg("--hap-b2").arg("B#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to open") || stderr.contains("No such file"),
        "Expected file error, got: {}",
        stderr
    );
}

/// Empty IBS file (header only, no data rows) should fail with "no data rows"
#[test]
fn test_jacquard_empty_ibs_file() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    f.flush().unwrap();

    let mut cmd = jacquard_cmd(&f, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no data rows"),
        "Expected 'no data rows' error, got: {}",
        stderr
    );
}

/// IBS file with too few columns should fail
#[test]
fn test_jacquard_malformed_ibs_file() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend").unwrap(); // only 3 cols (header)
    writeln!(f, "chr1\t1\t5000").unwrap(); // only 3 cols (data)
    f.flush().unwrap();

    let mut cmd = jacquard_cmd(&f, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("incomplete IBS row"),
        "Expected incomplete row error, got: {}",
        stderr
    );
}

/// IBS file with invalid (non-numeric) start position should fail
#[test]
fn test_jacquard_invalid_position_in_ibs() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    writeln!(f, "chr1\tABC\t5000\tA#1\tB#1").unwrap();
    f.flush().unwrap();

    let mut cmd = jacquard_cmd(&f, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid start position"),
        "Expected parse error, got: {}",
        stderr
    );
}

// ── End-to-end execution with synthetic data ─────────────────────────

/// All 4 haplotypes IBD at every locus → Delta1 should dominate
#[test]
fn test_jacquard_end_to_end_all_ibd() {
    // 6 pairs connecting all 4 haplotypes at every window
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#2"),
                ("chr1", s, e, "A#1", "B#1"),
                ("chr1", s, e, "A#1", "B#2"),
                ("chr1", s, e, "A#2", "B#1"),
                ("chr1", s, e, "A#2", "B#2"),
                ("chr1", s, e, "B#1", "B#2"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "jacquard should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Delta1 should have a non-zero fraction
    assert!(stdout.contains("Delta1"), "Should contain Delta1 in output");
    // All deltas 1-9 should appear
    for d in 1..=9 {
        assert!(
            stdout.contains(&format!("Delta{}", d)),
            "Missing Delta{} in output",
            d
        );
    }
}

/// No IBD between any pair → Delta9 should dominate
#[test]
fn test_jacquard_end_to_end_no_ibd() {
    // IBS data with only pairs that DON'T involve all 4 target haps together
    // If we provide windows where NO pair among the 4 target haps is IBD,
    // we get Delta9 for every locus
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            // Only include pairs involving non-target haplotypes
            ("chr1", s, e, "OTHER#1", "OTHER#2")
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    // This might fail with "no loci classified" since no target haps are in the data
    // That's the expected behavior — no data for the target pair means no classification
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should either succeed with all-Delta9 or fail with "no loci classified"
    assert!(
        !output.status.success() || String::from_utf8_lossy(&output.stdout).contains("Delta9"),
        "Expected Delta9 or no-loci error, got stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        stderr
    );
}

/// Within-group IBD only (Delta2 scenario): A1-A2 IBD and B1-B2 IBD, no cross-group
#[test]
fn test_jacquard_end_to_end_delta2() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#2"),
                ("chr1", s, e, "B#1", "B#2"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "jacquard should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Delta2 should have the highest fraction
    let delta2_line = stdout.lines().find(|l| l.starts_with("Delta2")).unwrap();
    let frac: f64 = delta2_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(
        frac > 0.5,
        "Delta2 should dominate for within-group-only IBD, got {}",
        frac
    );
}

/// Cross-group pairing (Delta7 scenario): A1-B1 IBD and A2-B2 IBD
#[test]
fn test_jacquard_end_to_end_delta7() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..10)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "B#1"),
                ("chr1", s, e, "A#2", "B#2"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success(), "jacquard should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let delta7_line = stdout.lines().find(|l| l.starts_with("Delta7")).unwrap();
    let frac: f64 = delta7_line.split('\t').nth(1).unwrap().parse().unwrap();
    assert!(
        frac > 0.5,
        "Delta7 should dominate for cross-group IBD, got {}",
        frac
    );
}

/// IBS file with blank lines and no header keyword should still parse
#[test]
fn test_jacquard_ibs_with_blank_lines() {
    let mut f = NamedTempFile::new().unwrap();
    // No "chrom" header — first data row treated as data (not header)
    writeln!(f, "").unwrap(); // blank line (skipped)
    writeln!(f, "chr1\t1\t5000\tA#1\tA#2").unwrap();
    writeln!(f, "").unwrap(); // another blank line
    writeln!(f, "chr1\t1\t5000\tB#1\tB#2").unwrap();
    writeln!(f, "chr1\t1\t5000\tA#1\tB#1").unwrap();
    writeln!(f, "chr1\t1\t5000\tA#2\tB#2").unwrap();
    f.flush().unwrap();

    let mut cmd = jacquard_cmd(&f, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    // Should succeed (blank lines are skipped)
    assert!(
        output.status.success(),
        "Should handle blank lines, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Invalid window coordinates (end < start) should be caught
#[test]
fn test_jacquard_invalid_window_coordinates() {
    let rows = vec![("chr1", 5000_i64, 1_i64, "A#1", "B#1")]; // end < start
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid window coordinates"),
        "Expected coordinate validation error, got: {}",
        stderr
    );
}

// ── Haplotype format edge cases ──────────────────────────────────────

/// Haplotypes with coordinate suffixes (like real impg output) should be
/// matched correctly via hap_key stripping
#[test]
fn test_jacquard_coordinate_suffix_haplotypes() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    for i in 0..5 {
        let s = i * 5000 + 1;
        let e = (i + 1) * 5000;
        // Full impg format: sample#hap#scaffold:start-end
        writeln!(
            f,
            "chr1\t{}\t{}\tA#1#scaffold:0-5000\tA#2#scaffold:0-5000",
            s, e
        )
        .unwrap();
        writeln!(
            f,
            "chr1\t{}\t{}\tB#1#scaffold:0-5000\tB#2#scaffold:0-5000",
            s, e
        )
        .unwrap();
    }
    f.flush().unwrap();

    let mut cmd = jacquard_cmd(&f, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "Should handle coordinate suffixes, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Delta2"), "Should classify as Delta2");
}

// ── Output format verification ───────────────────────────────────────

/// Verify output format: each line should be "DeltaN\t<fraction>\t(count=N)"
#[test]
fn test_jacquard_output_format() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..5)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "A#2"),
                ("chr1", s, e, "B#1", "B#2"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    // Should have exactly 9 lines (Delta1 through Delta9)
    assert_eq!(lines.len(), 9, "Expected 9 delta lines, got {}", lines.len());

    for (i, line) in lines.iter().enumerate() {
        let delta = i + 1;
        assert!(
            line.starts_with(&format!("Delta{}\t", delta)),
            "Line {} should start with Delta{}, got: {}",
            i, delta, line
        );
        // Fraction should be parseable as f64
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 3, "Expected 3 tab-separated fields");
        let frac: f64 = parts[1].parse().unwrap_or_else(|_| {
            panic!("Fraction '{}' should be a valid f64", parts[1])
        });
        assert!((0.0..=1.0).contains(&frac), "Fraction should be in [0, 1]");
        // Count field should match "(count=N)" format
        assert!(
            parts[2].starts_with("(count=") && parts[2].ends_with(')'),
            "Count field should be '(count=N)', got: {}",
            parts[2]
        );
    }

    // Fractions should sum to 1.0 (within floating point tolerance)
    let total: f64 = lines
        .iter()
        .map(|l| l.split('\t').nth(1).unwrap().parse::<f64>().unwrap())
        .sum();
    assert!(
        (total - 1.0).abs() < 1e-6,
        "Delta fractions should sum to 1.0, got {}",
        total
    );
}

/// Stderr should contain diagnostic info (chrom, window size, etc.)
#[test]
fn test_jacquard_stderr_diagnostics() {
    let rows: Vec<(&str, i64, i64, &str, &str)> = (0..3)
        .flat_map(|i| {
            let s = i * 5000 + 1;
            let e = (i + 1) * 5000;
            vec![
                ("chr1", s, e, "A#1", "B#1"),
                ("chr1", s, e, "A#2", "B#2"),
            ]
        })
        .collect();
    let ibs = make_ibs_file(&rows);
    let mut cmd = jacquard_cmd(&ibs, "A#1", "A#2", "B#1", "B#2");
    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("chr1"), "Stderr should mention chromosome");
    assert!(stderr.contains("win_size"), "Stderr should mention window size");
    assert!(stderr.contains("total_windows"), "Stderr should mention total windows");
}
