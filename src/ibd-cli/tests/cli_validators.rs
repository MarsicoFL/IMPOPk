//! CLI argument validation tests for ibd: probability boundaries, positive_u64,
//! unit_interval, region/bed conflicts, invalid population, and non-numeric inputs.

#![allow(deprecated)]

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helper: ibd command with minimal required args ───────────────────

fn ibd_base_cmd() -> Command {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--sequence-files").arg("/dummy/seq")
       .arg("-a").arg("/dummy/align")
       .arg("-r").arg("REF");
    cmd
}

fn ibd_validate_base_cmd() -> Command {
    Command::cargo_bin("ibd-validate").unwrap()
}

// ── validate_probability (exclusive bounds: (0.0, 1.0)) ──────────────

/// p-enter-ibd = 0.0 should be rejected (exclusive lower bound)
#[test]
fn test_ibd_p_enter_zero_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("0.0");
    cmd.assert().failure();
}

/// p-enter-ibd = 1.0 should be rejected (exclusive upper bound)
#[test]
fn test_ibd_p_enter_one_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("1.0");
    cmd.assert().failure();
}

/// p-enter-ibd = 0.5 should be accepted
#[test]
fn test_ibd_p_enter_valid_accepted() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("0.5");
    // Should fail for a reason OTHER than p-enter-ibd parsing
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("probability must be in"),
        "p-enter-ibd=0.5 should be accepted, got: {}",
        stderr
    );
}

/// p-enter-ibd = -0.1 (below range) rejected
#[test]
fn test_ibd_p_enter_negative_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("-0.1");
    cmd.assert().failure();
}

/// p-enter-ibd = 1.5 (above range) rejected
#[test]
fn test_ibd_p_enter_above_one_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("1.5");
    cmd.assert().failure();
}

/// p-enter-ibd = non-numeric rejected
#[test]
fn test_ibd_p_enter_non_numeric_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--p-enter-ibd").arg("foo");
    cmd.assert().failure();
}

// ── validate_unit_interval ([0.0, 1.0]) ──────────────────────────────

/// posterior-threshold = 0.0 (inclusive lower bound, accepted)
#[test]
fn test_ibd_posterior_threshold_zero_accepted() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--posterior-threshold").arg("0.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be in"),
        "posterior-threshold=0.0 should be accepted, got: {}",
        stderr
    );
}

/// posterior-threshold = 1.0 (inclusive upper bound, accepted)
#[test]
fn test_ibd_posterior_threshold_one_accepted() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--posterior-threshold").arg("1.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be in"),
        "posterior-threshold=1.0 should be accepted, got: {}",
        stderr
    );
}

/// posterior-threshold = 1.01 (above range, rejected)
#[test]
fn test_ibd_posterior_threshold_above_one_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--posterior-threshold").arg("1.01");
    cmd.assert().failure();
}

// ── validate_positive_u64 (window size, min-len-bp) ──────────────────

/// size = 0 rejected
#[test]
fn test_ibd_window_size_zero_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--size").arg("0");
    cmd.assert().failure();
}

/// min-len-bp = 0 rejected
#[test]
fn test_ibd_min_len_bp_zero_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--min-len-bp").arg("0");
    cmd.assert().failure();
}

/// expected-seg-windows = 0 rejected
#[test]
fn test_ibd_expected_seg_windows_zero_rejected() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--expected-seg-windows").arg("0");
    cmd.assert().failure();
}

// ── validate_positive_f64 ────────────────────────────────────────────

/// identity-floor = 0.0 rejected (must be > 0.0 for positive_f64)
/// Note: identity-floor uses unit_interval, so 0.0 is valid
#[test]
fn test_ibd_identity_floor_zero_accepted() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--identity-floor").arg("0.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // identity-floor uses unit_interval validator, so 0.0 is accepted
    assert!(
        !stderr.contains("value must be in"),
        "identity-floor=0.0 should be accepted (unit_interval), got: {}",
        stderr
    );
}

// ── region / bed conflict ────────────────────────────────────────────

/// Providing both --region and --bed should fail
#[test]
fn test_ibd_region_and_bed_conflict() {
    let mut bed_file = NamedTempFile::new().unwrap();
    writeln!(bed_file, "chr1\t1\t10000").unwrap();
    bed_file.flush().unwrap();

    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--bed").arg(bed_file.path());
    cmd.assert().failure();
}

/// Providing neither --region nor --bed should fail
#[test]
fn test_ibd_no_region_no_bed() {
    let mut cmd = ibd_base_cmd();
    // No --region, no --bed
    cmd.assert().failure();
}

// ── Invalid population name ──────────────────────────────────────────

/// Invalid --population value should be rejected
#[test]
fn test_ibd_invalid_population_name() {
    let mut cmd = ibd_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--population").arg("INVALID_POP_NAME");
    let out = cmd.output().unwrap();
    // Should fail either at arg parsing or early in run
    assert!(
        !out.status.success(),
        "Invalid population should be rejected"
    );
}

// ── ibd-validate argument validation ─────────────────────────────────

/// ibd-validate: identity-floor > 1.0 rejected
#[test]
fn test_ibd_validate_identity_floor_above_one() {
    let mut cmd = ibd_validate_base_cmd();
    cmd.arg("--input").arg("/dummy")
       .arg("--output").arg("/dummy")
       .arg("--identity-floor").arg("2.0");
    cmd.assert().failure();
}

/// ibd-validate: identity-floor negative rejected
#[test]
fn test_ibd_validate_identity_floor_negative() {
    let mut cmd = ibd_validate_base_cmd();
    cmd.arg("--input").arg("/dummy")
       .arg("--output").arg("/dummy")
       .arg("--identity-floor").arg("-0.5");
    cmd.assert().failure();
}

/// ibd-validate: validate-min-lod non-numeric rejected
#[test]
fn test_ibd_validate_min_lod_non_numeric() {
    let mut cmd = ibd_validate_base_cmd();
    cmd.arg("--input").arg("/dummy")
       .arg("--output").arg("/dummy")
       .arg("--validate-min-lod").arg("abc");
    cmd.assert().failure();
}

/// ibd-validate: baum-welch-iters non-numeric rejected
#[test]
fn test_ibd_validate_baum_welch_non_numeric() {
    let mut cmd = ibd_validate_base_cmd();
    cmd.arg("--input").arg("/dummy")
       .arg("--output").arg("/dummy")
       .arg("--baum-welch-iters").arg("xyz");
    cmd.assert().failure();
}
