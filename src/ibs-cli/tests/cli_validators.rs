//! CLI argument validation tests for ibs: cutoff boundaries, window size=0,
//! region/bed conflicts, and missing file paths.

#![allow(deprecated)]

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helper: minimal valid args (minus the one we're testing) ─────────

fn ibs_base_cmd() -> Command {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    // Provide all required args except the one under test
    cmd.arg("--sequence-files").arg("/dummy/seq")
       .arg("-a").arg("/dummy/align")
       .arg("-r").arg("REF");
    cmd
}

// ── validate_cutoff boundary tests ───────────────────────────────────

/// cutoff = 0.0 (lower boundary, should be accepted)
#[test]
fn test_ibs_cutoff_zero_accepted() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("0.0");
    // Should fail at runtime (files don't exist), not at arg parsing
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Arg parsing succeeds — error should be about impg/files, not about cutoff
    assert!(
        !stderr.contains("cutoff must be in"),
        "cutoff=0.0 should be accepted, got: {}",
        stderr
    );
}

/// cutoff = 1.0 (upper boundary, should be accepted)
#[test]
fn test_ibs_cutoff_one_accepted() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("1.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("cutoff must be in"),
        "cutoff=1.0 should be accepted, got: {}",
        stderr
    );
}

/// cutoff = -0.1 (below range, should be rejected)
#[test]
fn test_ibs_cutoff_negative_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("-0.1");
    cmd.assert().failure();
}

/// cutoff = 1.1 (above range, should be rejected)
#[test]
fn test_ibs_cutoff_above_one_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("1.1");
    cmd.assert().failure();
}

/// cutoff = non-numeric (should be rejected)
#[test]
fn test_ibs_cutoff_non_numeric_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("abc");
    cmd.assert().failure();
}

// ── validate_positive_u64 (window size) ──────────────────────────────

/// window_size = 0 should be rejected (prevents infinite loop)
#[test]
fn test_ibs_window_size_zero_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("0")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    let out = cmd.output().unwrap();
    assert!(
        !out.status.success(),
        "window_size=0 should be rejected"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must be > 0") || stderr.contains("invalid value"),
        "Should mention value must be > 0, got: {}",
        stderr
    );
}

/// window_size = -1 (negative) should be rejected
#[test]
fn test_ibs_window_size_negative_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("-1")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    cmd.assert().failure();
}

/// window_size = non-numeric should be rejected
#[test]
fn test_ibs_window_size_non_numeric_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("xyz")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    cmd.assert().failure();
}

// ── region/bed conflict ──────────────────────────────────────────────

/// Providing both --region and --bed should fail
#[test]
fn test_ibs_region_and_bed_conflict() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut bed_file = NamedTempFile::new().unwrap();
    writeln!(bed_file, "chr1\t1\t10000").unwrap();
    bed_file.flush().unwrap();

    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--bed").arg(bed_file.path())
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    cmd.assert().failure();
}

/// Providing neither --region nor --bed should fail
#[test]
fn test_ibs_no_region_no_bed() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    cmd.assert().failure();
}
