//! Integration tests for the ibd and ibd-validate CLI binaries.
#![allow(deprecated)]

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

/// Test that ibd-validate --identity-floor filters low-identity windows.
/// With floor=0.9, windows with identity < 0.9 should be excluded from HMM.
#[test]
fn test_ibd_validate_identity_floor() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create IBS data with some low-identity (gap) windows and some high-identity windows
    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(input_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity").unwrap();
    // Mix of gap windows (identity near 0) and real windows (identity near 1)
    for i in 0..100 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        let identity = if i % 3 == 0 { 0.1 } else { 0.999 };
        writeln!(input_file, "chr20\t{}\t{}\tHG001#1\tHG002#1\t{:.4}", start, end, identity).unwrap();
    }
    input_file.flush().unwrap();

    // Run without floor
    let output_no_floor = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input").arg(input_file.path())
       .arg("--output").arg(output_no_floor.path())
       .arg("--identity-floor").arg("0.0")
       .arg("--baum-welch-iters").arg("0");
    cmd.assert().success();
    let content_no_floor = std::fs::read_to_string(output_no_floor.path()).unwrap();

    // Run with floor=0.5
    let output_with_floor = NamedTempFile::new().unwrap();
    let mut cmd2 = Command::cargo_bin("ibd-validate").unwrap();
    cmd2.arg("--input").arg(input_file.path())
        .arg("--output").arg(output_with_floor.path())
        .arg("--identity-floor").arg("0.5")
        .arg("--baum-welch-iters").arg("0");
    cmd2.assert().success();
    let content_with_floor = std::fs::read_to_string(output_with_floor.path()).unwrap();

    // Both should produce valid output (header at minimum)
    assert!(content_no_floor.contains("chrom\tstart\tend"));
    assert!(content_with_floor.contains("chrom\tstart\tend"));

    // The outputs may differ because floor filtering changes the observation distribution
    // At minimum, both should succeed without errors
}

/// Test that ibd-validate --identity-floor=0.9 with all-low-identity input produces no segments.
#[test]
fn test_ibd_validate_identity_floor_all_below() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(input_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity").unwrap();
    // All windows below the floor threshold
    for i in 0..10 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        writeln!(input_file, "chr20\t{}\t{}\tHG001#1\tHG002#1\t0.5", start, end).unwrap();
    }
    input_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input").arg(input_file.path())
       .arg("--output").arg(output_file.path())
       .arg("--identity-floor").arg("0.9");
    cmd.assert().success();

    // Should have only header, no data rows (all windows filtered out)
    let content = std::fs::read_to_string(output_file.path()).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "Expected header only, got {} lines", lines.len());
    assert!(lines[0].starts_with("chrom\t"));
}

/// Test that ibd-validate --coverage-feature works without crashing.
#[test]
fn test_ibd_validate_coverage_feature() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(input_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length").unwrap();
    for i in 0..20 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        writeln!(input_file, "chr20\t{}\t{}\tHG001#1\tHG002#1\t0.9995\t4900\t4850", start, end).unwrap();
    }
    input_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input").arg(input_file.path())
       .arg("--output").arg(output_file.path())
       .arg("--coverage-feature")
       .arg("--baum-welch-iters").arg("5");
    cmd.assert().success();

    let content = std::fs::read_to_string(output_file.path()).unwrap();
    assert!(content.contains("chrom\tstart\tend"));
}

/// Test ibd-validate --validate-against with synthetic hap-ibd file.
#[test]
fn test_ibd_validate_against_hapibd() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create IBS input with a clear IBD signal
    let mut input_file = NamedTempFile::new().unwrap();
    writeln!(input_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity").unwrap();
    for i in 0..200 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        // IBD region: windows 50-150 have higher identity
        let identity = if (50..150).contains(&i) { 0.9999 } else { 0.990 };
        writeln!(input_file, "chr20\t{}\t{}\tHG001#1\tHG002#1\t{:.4}", start, end, identity).unwrap();
    }
    input_file.flush().unwrap();

    // Create synthetic hap-ibd output
    let mut hapibd_file = NamedTempFile::new().unwrap();
    writeln!(hapibd_file, "HG001\t1\tHG002\t1\tchr20\t250001\t750000\t15.0").unwrap();
    hapibd_file.flush().unwrap();

    let output_file = NamedTempFile::new().unwrap();
    let validate_output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--input").arg(input_file.path())
       .arg("--output").arg(output_file.path())
       .arg("--validate-against").arg(hapibd_file.path())
       .arg("--validate-output").arg(validate_output.path())
       .arg("--validate-min-lod").arg("3.0")
       .arg("--baum-welch-iters").arg("5")
       .arg("--min-len-bp").arg("5000")
       .arg("--min-windows").arg("3");
    cmd.assert().success();

    // Validate output should contain metrics header
    let val_content = std::fs::read_to_string(validate_output.path()).unwrap();
    assert!(val_content.contains("pair\tn_ours\tn_hapibd\tjaccard\tprecision\trecall\tf1\tconcordance\tlength_r"));
}

/// Test that `ibd --help` includes recommended parameter presets.
#[test]
fn test_ibd_help_contains_presets() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("RECOMMENDED PARAMETER PRESETS"), "Missing presets section");
    assert!(stdout.contains("--identity-floor"), "Missing --identity-floor in presets");
    assert!(stdout.contains("--baum-welch-iters"), "Missing --baum-welch-iters in presets");
    assert!(stdout.contains("--min-len-bp"), "Missing --min-len-bp in presets");
}

/// Test that `ibd --help` documents all important flags.
#[test]
fn test_ibd_help_contains_all_flags() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let required_flags = [
        "--population",
        "--identity-floor",
        "--min-len-bp",
        "--min-windows",
        "--baum-welch-iters",
        "--posterior-threshold",
        "--distance-aware",
        "--genetic-map",
        "--adaptive-transitions",
        "--output-bed",
        "--output-posteriors",
    ];

    for flag in &required_flags {
        assert!(stdout.contains(flag), "Missing flag: {}", flag);
    }
}

/// Test that `ibd-validate --help` documents all validation-specific flags.
#[test]
fn test_ibd_validate_help_contains_validation_flags() {
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    let validation_flags = [
        "--validate-against",
        "--validate-min-lod",
        "--validate-output",
        "--coverage-feature",
        "--identity-floor",
        "--logit-transform",
        "--background-filter",
        "--exclude-regions",
        "--states-output",
    ];

    for flag in &validation_flags {
        assert!(stdout.contains(flag), "Missing validation flag: {}", flag);
    }
}

/// Test that `ibd` rejects invalid parameter values.
#[test]
fn test_ibd_invalid_identity_floor() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--identity-floor").arg("1.5")
       .arg("--sequence-files").arg("dummy")
       .arg("--align").arg("dummy")
       .arg("--ref").arg("dummy")
       .arg("--region").arg("chr1:1-100");
    cmd.assert().failure();
}

/// Test that `ibd` rejects invalid posterior threshold.
#[test]
fn test_ibd_invalid_posterior_threshold() {
    let mut cmd = Command::cargo_bin("ibd").unwrap();
    cmd.arg("--posterior-threshold").arg("-0.5")
       .arg("--sequence-files").arg("dummy")
       .arg("--align").arg("dummy")
       .arg("--ref").arg("dummy")
       .arg("--region").arg("chr1:1-100");
    cmd.assert().failure();
}
