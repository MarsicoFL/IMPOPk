//! Integration tests for the ibd and ibd-validate CLI binaries.

use assert_cmd::Command;

/// Test that `ibd --help` runs successfully
#[test]
fn test_ibd_help() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test that `ibd --version` runs successfully
#[test]
fn test_ibd_version() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test that `ibd` fails without required arguments
#[test]
fn test_ibd_missing_args() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.assert().failure();
}

/// Test that `ibd-validate --help` runs successfully
#[test]
fn test_ibd_validate_help() {
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test that `ibd-validate --version` runs successfully
#[test]
fn test_ibd_validate_version() {
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test that `ibd-validate` fails without required arguments
#[test]
fn test_ibd_validate_missing_args() {
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.assert().failure();
}

/// Test that ibd-validate works with a simple input file
#[test]
fn test_ibd_validate_with_input() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a simple IBS input file
    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(
        input_file,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    writeln!(input_file, "chr1\t1\t5000\tHG001#1\tHG002#1\t0.9999").unwrap();
    writeln!(input_file, "chr1\t5001\t10000\tHG001#1\tHG002#1\t0.9998").unwrap();
    writeln!(input_file, "chr1\t10001\t15000\tHG001#1\tHG002#1\t0.9997").unwrap();
    writeln!(input_file, "chr1\t15001\t20000\tHG001#1\tHG002#1\t0.9999").unwrap();
    writeln!(input_file, "chr1\t20001\t25000\tHG001#1\tHG002#1\t0.9998").unwrap();
    input_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input")
        .arg(input_file.path())
        .arg("--output")
        .arg(output_file.path());

    cmd.assert().success();

    // Verify the output file was created
    let content = std::fs::read_to_string(output_file.path()).unwrap();
    assert!(content.contains("chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity"));
}

/// Test that ibd-validate correctly parses haplotype IDs with coordinate suffixes
#[test]
fn test_ibd_validate_coordinate_suffix() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(
        input_file,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    // Haplotype IDs with coordinate suffixes (like real impg output)
    writeln!(
        input_file,
        "chr1\t1\t5000\tHG001#1#JBHDWB010000002.1:1-5000\tHG002#1#JBHDWB010000003.1:1-5000\t0.9999"
    )
    .unwrap();
    writeln!(
        input_file,
        "chr1\t5001\t10000\tHG001#1#JBHDWB010000002.1:5001-10000\tHG002#1#JBHDWB010000003.1:5001-10000\t0.9998"
    )
    .unwrap();
    writeln!(
        input_file,
        "chr1\t10001\t15000\tHG001#1#JBHDWB010000002.1:10001-15000\tHG002#1#JBHDWB010000003.1:10001-15000\t0.9997"
    )
    .unwrap();
    input_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input")
        .arg(input_file.path())
        .arg("--output")
        .arg(output_file.path());

    cmd.assert().success();

    // The output should have haplotypes WITHOUT coordinate suffixes
    let content = std::fs::read_to_string(output_file.path()).unwrap();
    // Header should be present
    assert!(content.contains("chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity"));
    // Should NOT contain the coordinate suffix pattern
    assert!(!content.contains("JBHDWB010000002.1:1-5000"));
}
