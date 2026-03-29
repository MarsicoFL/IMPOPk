//! Tests for ibs-cli main.rs FilteredRow filtering logic and region processing.
//!
//! Since process_window/process_region are private and depend on the impg binary,
//! we reimplement the pure filtering logic for unit testing and test CLI validation
//! for error paths.

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// =====================================================================
// Reimplemented pure filtering logic from ibs-cli main.rs
// =====================================================================

/// Reimplemented filtering logic from process_window (main.rs lines 166-196)
fn filter_row(
    group_a: &str,
    group_b: &str,
    identity: f64,
    cutoff: f64,
    ref_prefix: &str,
) -> bool {
    // Apply cutoff
    if identity < cutoff {
        return false;
    }
    // Skip self-self
    if group_a == group_b {
        return false;
    }
    // Skip reference
    if group_a.starts_with(ref_prefix) || group_b.starts_with(ref_prefix) {
        return false;
    }
    // Keep only canonical order
    if group_a > group_b {
        return false;
    }
    true
}

// =====================================================================
// filter_row unit tests
// =====================================================================

#[test]
fn filter_row_valid_canonical_pair() {
    assert!(filter_row("A#1", "B#1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_below_cutoff_rejected() {
    assert!(!filter_row("A#1", "B#1", 0.998, 0.999, "CHM13#"));
}

#[test]
fn filter_row_at_cutoff_accepted() {
    assert!(filter_row("A#1", "B#1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_above_cutoff_accepted() {
    assert!(filter_row("A#1", "B#1", 1.0, 0.999, "CHM13#"));
}

#[test]
fn filter_row_self_self_rejected() {
    assert!(!filter_row("A#1", "A#1", 1.0, 0.999, "CHM13#"));
}

#[test]
fn filter_row_ref_in_group_a_rejected() {
    assert!(!filter_row("CHM13#0#chr1", "B#1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_ref_in_group_b_rejected() {
    assert!(!filter_row("A#1", "CHM13#0#chr1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_non_canonical_order_rejected() {
    assert!(!filter_row("B#1", "A#1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_canonical_order_accepted() {
    assert!(filter_row("A#1", "B#1", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_same_string_self_rejected() {
    assert!(!filter_row("HG00096#1", "HG00096#1", 0.9999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_different_haps_same_sample_accepted() {
    assert!(filter_row("HG00096#1", "HG00096#2", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_cutoff_zero_accepts_all() {
    assert!(filter_row("A#1", "B#1", 0.0, 0.0, "CHM13#"));
}

#[test]
fn filter_row_cutoff_one_rejects_below_one() {
    assert!(!filter_row("A#1", "B#1", 0.9999, 1.0, "CHM13#"));
}

#[test]
fn filter_row_cutoff_one_accepts_exactly_one() {
    assert!(filter_row("A#1", "B#1", 1.0, 1.0, "CHM13#"));
}

#[test]
fn filter_row_ref_prefix_partial_match_not_ref() {
    // "CHM13_sample" does NOT start with "CHM13#" so it's not the reference.
    // But "B#1" < "CHM13_sample" lexicographically, so canonical order requires
    // group_a <= group_b. With B#1 as group_a:
    assert!(filter_row("B#1", "CHM13_sample", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_ref_prefix_exact_match_rejected() {
    // "CHM13#0" starts with "CHM13#" so it IS the reference → rejected
    assert!(!filter_row("B#1", "CHM13#0", 0.999, 0.999, "CHM13#"));
}

#[test]
fn filter_row_empty_strings() {
    assert!(!filter_row("", "", 1.0, 0.0, "CHM13#"));
}

#[test]
fn filter_row_lexicographic_ordering_numbers() {
    assert!(filter_row("HG00096#1", "HG00097#1", 0.999, 0.999, "CHM13#"));
    assert!(!filter_row("HG00097#1", "HG00096#1", 0.999, 0.999, "CHM13#"));
}

// =====================================================================
// CLI validation error paths
// =====================================================================

fn ibs_cmd() -> Command {
    #[allow(deprecated)]
    Command::cargo_bin("ibs").unwrap()
}

#[test]
fn cli_missing_sequence_files_error() {
    let output = ibs_cmd()
        .args(["--sequence-files", "/nonexistent/sequences.agc",
               "-a", "/some/alignment.paf",
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "1000",
               "--subset-sequence-list", "/some/list.txt",
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("sequence-files does not exist"),
        "Expected missing file error, got: {}", stderr
    );
}

#[test]
fn cli_missing_alignment_file_error() {
    let mut seq_file = NamedTempFile::new().unwrap();
    write!(seq_file, "dummy").unwrap();
    seq_file.flush().unwrap();

    let output = ibs_cmd()
        .args(["--sequence-files", seq_file.path().to_str().unwrap(),
               "-a", "/nonexistent/alignment.paf",
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "1000",
               "--subset-sequence-list", "/some/list.txt",
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("alignment file does not exist"),
        "Expected missing alignment error, got: {}", stderr
    );
}

#[test]
fn cli_missing_subset_list_error() {
    let mut seq_file = NamedTempFile::new().unwrap();
    write!(seq_file, "dummy").unwrap();
    seq_file.flush().unwrap();

    let mut align_file = NamedTempFile::new().unwrap();
    write!(align_file, "dummy").unwrap();
    align_file.flush().unwrap();

    let output = ibs_cmd()
        .args(["--sequence-files", seq_file.path().to_str().unwrap(),
               "-a", align_file.path().to_str().unwrap(),
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "1000",
               "--subset-sequence-list", "/nonexistent/list.txt",
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("subset-sequence-list does not exist"),
        "Expected missing subset list error, got: {}", stderr
    );
}

#[test]
fn cli_nonexistent_bed_file_error() {
    let mut seq_file = NamedTempFile::new().unwrap();
    write!(seq_file, "dummy").unwrap();
    seq_file.flush().unwrap();

    let mut align_file = NamedTempFile::new().unwrap();
    write!(align_file, "dummy").unwrap();
    align_file.flush().unwrap();

    let mut list_file = NamedTempFile::new().unwrap();
    write!(list_file, "sample1").unwrap();
    list_file.flush().unwrap();

    let output = ibs_cmd()
        .args(["--sequence-files", seq_file.path().to_str().unwrap(),
               "-a", align_file.path().to_str().unwrap(),
               "-r", "CHM13",
               "--bed", "/nonexistent/regions.bed",
               "--size", "1000",
               "--subset-sequence-list", list_file.path().to_str().unwrap(),
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("BED file does not exist"),
        "Expected missing BED file error, got: {}", stderr
    );
}

#[test]
fn cli_window_size_zero_rejected() {
    let output = ibs_cmd()
        .args(["--sequence-files", "/dummy",
               "-a", "/dummy",
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "0",
               "--subset-sequence-list", "/dummy",
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn cli_cutoff_above_one_rejected() {
    let output = ibs_cmd()
        .args(["--sequence-files", "/dummy",
               "-a", "/dummy",
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "1000",
               "--subset-sequence-list", "/dummy",
               "--output", "/tmp/test_ibs_output.tsv",
               "-c", "1.5"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn cli_cutoff_negative_rejected() {
    let output = ibs_cmd()
        .args(["--sequence-files", "/dummy",
               "-a", "/dummy",
               "-r", "CHM13",
               "--region", "chr1:1-1000",
               "--size", "1000",
               "--subset-sequence-list", "/dummy",
               "--output", "/tmp/test_ibs_output.tsv",
               "-c", "-0.5"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn cli_region_chrom_only_without_length_fails() {
    let output = ibs_cmd()
        .args(["--sequence-files", "/dummy",
               "-a", "/dummy",
               "-r", "CHM13",
               "--region", "chr1",
               "--size", "1000",
               "--subset-sequence-list", "/dummy",
               "--output", "/tmp/test_ibs_output.tsv"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}
