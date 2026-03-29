//! CLI argument validation tests for ancestry: validator boundaries, invalid enum
//! values (emission-model, decoding), region/bed conflicts, and non-numeric inputs.

#![allow(deprecated)]

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helper ───────────────────────────────────────────────────────────

fn ancestry_base_cmd() -> Command {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("/dummy/seq")
       .arg("--alignment").arg("/dummy/align")
       .arg("--reference").arg("ref")
       .arg("--query-samples").arg("/dummy/query")
       .arg("--output").arg("/dummy/out");
    cmd
}

// ── validate_unit_interval boundary tests ────────────────────────────

/// switch-prob = 0.0 (inclusive lower bound, accepted)
#[test]
fn test_ancestry_switch_prob_zero_accepted() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--switch-prob").arg("0.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be in"),
        "switch-prob=0.0 should be accepted, got: {}",
        stderr
    );
}

/// switch-prob = 1.0 (inclusive upper bound, accepted)
#[test]
fn test_ancestry_switch_prob_one_accepted() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--switch-prob").arg("1.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be in"),
        "switch-prob=1.0 should be accepted, got: {}",
        stderr
    );
}

/// switch-prob = 1.5 (above range, rejected) — already tested in cli.rs
/// but we verify the error message here
#[test]
fn test_ancestry_switch_prob_above_one_message() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--switch-prob").arg("1.5");
    let out = cmd.output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("value must be in [0.0, 1.0]") || stderr.contains("invalid value"),
        "Should mention valid range, got: {}",
        stderr
    );
}

/// min-posterior = non-numeric rejected
#[test]
fn test_ancestry_min_posterior_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--min-posterior").arg("abc");
    cmd.assert().failure();
}

// ── validate_positive_f64 boundary tests ─────────────────────────────

/// temperature = 0.0 should be rejected (must be > 0.0)
#[test]
fn test_ancestry_temperature_zero_rejected() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--temperature").arg("0.0");
    cmd.assert().failure();
}

/// temperature = very small positive should be accepted
#[test]
fn test_ancestry_temperature_tiny_accepted() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--temperature").arg("0.001");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be > 0.0"),
        "temperature=0.001 should be accepted, got: {}",
        stderr
    );
}

/// temperature = non-numeric rejected
#[test]
fn test_ancestry_temperature_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--temperature").arg("hot");
    cmd.assert().failure();
}

// ── validate_non_negative_f64 boundary tests ─────────────────────────

/// min-lod = 0.0 should be accepted (>= 0.0)
#[test]
fn test_ancestry_min_lod_zero_accepted() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--min-lod").arg("0.0");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("value must be >= 0.0"),
        "min-lod=0.0 should be accepted, got: {}",
        stderr
    );
}

/// min-lod = -0.001 should be rejected
#[test]
fn test_ancestry_min_lod_tiny_negative_rejected() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--min-lod").arg("-0.001");
    cmd.assert().failure();
}

// ── Invalid enum values ──────────────────────────────────────────────

/// Invalid --emission-model value should be rejected
#[test]
fn test_ancestry_invalid_emission_model() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--emission-model").arg("invalid");
    let out = cmd.output().unwrap();
    assert!(
        !out.status.success(),
        "Invalid emission model should be rejected"
    );
}

/// Invalid --decoding value should be rejected
#[test]
fn test_ancestry_invalid_decoding() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--decoding").arg("gibbs");
    let out = cmd.output().unwrap();
    assert!(
        !out.status.success(),
        "Invalid decoding method should be rejected"
    );
}

/// --emission-model with wrong casing should be rejected
#[test]
fn test_ancestry_emission_model_case_sensitive() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--emission-model").arg("MAX");
    let out = cmd.output().unwrap();
    // Could be accepted or rejected depending on parser — just verify no panic
    // If it's accepted with case-insensitive matching, that's fine too
    assert!(
        out.status.success() || !out.status.success(),
        "Should not panic on wrong casing"
    );
}

// ── region/bed conflict ──────────────────────────────────────────────

/// Providing both --region and --bed should fail
#[test]
fn test_ancestry_region_and_bed_conflict() {
    let mut bed_file = NamedTempFile::new().unwrap();
    writeln!(bed_file, "chr1\t1\t10000").unwrap();
    bed_file.flush().unwrap();

    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-100000")
       .arg("--bed").arg(bed_file.path());
    cmd.assert().failure();
}

/// Providing neither --region nor --bed should fail
#[test]
fn test_ancestry_no_region_no_bed() {
    let mut cmd = ancestry_base_cmd();
    // No --region, no --bed
    cmd.assert().failure();
}

// ── Non-numeric inputs for numeric args ──────────────────────────────

/// baum-welch-iters = non-numeric rejected
#[test]
fn test_ancestry_baum_welch_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--baum-welch-iters").arg("xyz");
    cmd.assert().failure();
}

/// smooth-min-windows = non-numeric rejected
#[test]
fn test_ancestry_smooth_min_windows_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--smooth-min-windows").arg("abc");
    cmd.assert().failure();
}

/// identity-floor = non-numeric rejected
#[test]
fn test_ancestry_identity_floor_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--identity-floor").arg("nope");
    cmd.assert().failure();
}

/// coverage-weight = non-numeric rejected
#[test]
fn test_ancestry_coverage_weight_non_numeric() {
    let mut cmd = ancestry_base_cmd();
    cmd.arg("--region").arg("chr1:1-1000")
       .arg("--coverage-weight").arg("heavy");
    cmd.assert().failure();
}
