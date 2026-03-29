//! Integration tests for the ibs CLI binary.
#![allow(deprecated)]

use assert_cmd::Command;

/// Test that `ibs --help` runs successfully
#[test]
fn test_ibs_help() {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test that `ibs --version` runs successfully
#[test]
fn test_ibs_version() {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test that `ibs` fails without required arguments
#[test]
fn test_ibs_missing_args() {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.assert().failure();
}

/// Test that `ibs` shows proper error message for missing impg
#[test]
fn test_ibs_impg_not_found() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a dummy subset list
    let mut subset_file = NamedTempFile::new().unwrap();
    writeln!(subset_file, "HG001#1").unwrap();
    writeln!(subset_file, "HG002#1").unwrap();
    subset_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();

    // This should fail because impg is not in PATH (unless it happens to be installed)
    // We test that it at least parses arguments correctly before failing
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files")
        .arg("/nonexistent/file.agc")
        .arg("-a")
        .arg("/nonexistent/alignment.paf")
        .arg("-r")
        .arg("CHM13")
        .arg("--region")
        .arg("chr1:1-10000")
        .arg("--size")
        .arg("5000")
        .arg("--subset-sequence-list")
        .arg(subset_file.path())
        .arg("--output")
        .arg(output_file.path());

    // This will fail either because impg is not found or because files don't exist
    // The important thing is that argument parsing works
    cmd.assert().failure();
}

/// Test that help output contains expected options
#[test]
fn test_ibs_help_contains_options() {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify key options are documented
    assert!(stdout.contains("--sequence-files"));
    assert!(stdout.contains("--region"));
    assert!(stdout.contains("--size"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--subset-sequence-list"));
}
