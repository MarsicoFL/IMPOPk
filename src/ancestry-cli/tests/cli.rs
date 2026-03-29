//! Integration tests for the ancestry CLI binary.
#![allow(deprecated)]

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

/// Test that `ancestry --help` runs successfully
#[test]
fn test_ancestry_help() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test that `ancestry --version` runs successfully
#[test]
fn test_ancestry_version() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

/// Test that `ancestry` fails without required arguments
#[test]
fn test_ancestry_missing_args() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.assert().failure();
}

/// Test that help output contains expected options
#[test]
fn test_ancestry_help_contains_options() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("--region"));
    assert!(stdout.contains("--window-size") || stdout.contains("--size"));
}

/// Test that help output contains validation and advanced flags
#[test]
fn test_ancestry_help_contains_advanced_flags() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("--validate-against"), "missing --validate-against");
    assert!(stdout.contains("--emission-model"), "missing --emission-model");
    assert!(stdout.contains("--decoding"), "missing --decoding");
    assert!(stdout.contains("--similarity-column"), "missing --similarity-column");
    assert!(stdout.contains("--baum-welch-iters"), "missing --baum-welch-iters");
    assert!(stdout.contains("--estimate-params"), "missing --estimate-params");
    assert!(stdout.contains("--coverage-feature"), "missing --coverage-feature");
    assert!(stdout.contains("--normalize-emissions"), "missing --normalize-emissions");
    assert!(stdout.contains("--genetic-map"), "missing --genetic-map");
    assert!(stdout.contains("--min-posterior"), "missing --min-posterior");
    assert!(stdout.contains("--min-lod"), "missing --min-lod");
    assert!(stdout.contains("--smooth-min-windows"), "missing --smooth-min-windows");
    assert!(stdout.contains("--cross-validate"), "missing --cross-validate");
    assert!(stdout.contains("--output-bed"), "missing --output-bed");
}

/// Helper to create test fixture files for ancestry CLI integration tests.
/// Returns (similarity_file, query_file, pop_file, dummy_seq, dummy_align)
fn create_ancestry_test_fixtures(n_windows: usize) -> (NamedTempFile, NamedTempFile, NamedTempFile, NamedTempFile, NamedTempFile) {
    // Similarity file: 2 populations (POP_A with ref_a1, ref_a2; POP_B with ref_b1, ref_b2)
    // Query: query#1
    let mut sim_file = NamedTempFile::new().unwrap();
    writeln!(sim_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length").unwrap();
    for i in 0..n_windows {
        let start = i as u64 * 5000 + 1;
        let end = (i as u64 + 1) * 5000;
        // In first half, POP_A has higher similarity; in second half, POP_B does
        let sim_a = if i < n_windows / 2 { 0.999 } else { 0.990 };
        let sim_b = if i < n_windows / 2 { 0.990 } else { 0.999 };
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_a1\t{:.4}\t4900\t4800", start, end, sim_a).unwrap();
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_a2\t{:.4}\t4900\t4850", start, end, sim_a - 0.001).unwrap();
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_b1\t{:.4}\t4900\t4900", start, end, sim_b).unwrap();
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_b2\t{:.4}\t4900\t4850", start, end, sim_b - 0.001).unwrap();
    }
    sim_file.flush().unwrap();

    // Query samples file
    let mut query_file = NamedTempFile::new().unwrap();
    writeln!(query_file, "query#1").unwrap();
    query_file.flush().unwrap();

    // Population file (TSV: pop_name\thaplotype_id)
    let mut pop_file = NamedTempFile::new().unwrap();
    writeln!(pop_file, "POP_A\tref_a1").unwrap();
    writeln!(pop_file, "POP_A\tref_a2").unwrap();
    writeln!(pop_file, "POP_B\tref_b1").unwrap();
    writeln!(pop_file, "POP_B\tref_b2").unwrap();
    pop_file.flush().unwrap();

    // Dummy required files (not used when --similarity-file is provided)
    let dummy_seq = NamedTempFile::new().unwrap();
    let dummy_align = NamedTempFile::new().unwrap();

    (sim_file, query_file, pop_file, dummy_seq, dummy_align)
}

/// Build a basic ancestry command with the required args and similarity file
fn ancestry_cmd_with_fixtures(
    sim_file: &NamedTempFile,
    query_file: &NamedTempFile,
    pop_file: &NamedTempFile,
    dummy_seq: &NamedTempFile,
    dummy_align: &NamedTempFile,
    output_file: &NamedTempFile,
    n_windows: usize,
) -> Command {
    let region_end = n_windows as u64 * 5000;
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg(dummy_seq.path())
       .arg("--alignment").arg(dummy_align.path())
       .arg("--reference").arg("ref")
       .arg("--region").arg(format!("chr20:1-{}", region_end))
       .arg("--query-samples").arg(query_file.path())
       .arg("--populations").arg(pop_file.path())
       .arg("--similarity-file").arg(sim_file.path())
       .arg("--output").arg(output_file.path())
       .arg("--threads").arg("1");
    cmd
}

/// Test basic ancestry inference with pre-computed similarity file
#[test]
fn test_ancestry_with_similarity_file() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.assert().success();

    let content = std::fs::read_to_string(output_file.path()).unwrap();
    assert!(!content.is_empty(), "Output should not be empty");
    // Should contain ancestry segments
    assert!(content.contains("query#1"), "Output should contain query sample");
}

/// Test --emission-model flag (max, mean, median)
#[test]
fn test_ancestry_emission_models() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);

    for model in &["max", "mean", "median"] {
        let output_file = NamedTempFile::new().unwrap();
        let mut cmd = ancestry_cmd_with_fixtures(
            &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
        );
        cmd.arg("--emission-model").arg(model);
        cmd.assert().success();

        let content = std::fs::read_to_string(output_file.path()).unwrap();
        assert!(!content.is_empty(), "Output with --emission-model {} should not be empty", model);
    }
}

/// Test --decoding flag (viterbi and posterior)
#[test]
fn test_ancestry_decoding_methods() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);

    for method in &["viterbi", "posterior"] {
        let output_file = NamedTempFile::new().unwrap();
        let mut cmd = ancestry_cmd_with_fixtures(
            &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
        );
        cmd.arg("--decoding").arg(method);
        cmd.assert().success();

        let content = std::fs::read_to_string(output_file.path()).unwrap();
        assert!(!content.is_empty(), "Output with --decoding {} should not be empty", method);
    }
}

/// Test --estimate-params auto-estimates temperature and switch-prob
#[test]
fn test_ancestry_estimate_params() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--estimate-params");
    cmd.assert().success();

    let content = std::fs::read_to_string(output_file.path()).unwrap();
    assert!(!content.is_empty());
}

/// Test --baum-welch-iters for EM training
#[test]
fn test_ancestry_baum_welch() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--baum-welch-iters").arg("5");
    cmd.assert().success();
}

/// Test --output-bed produces valid BED output
#[test]
fn test_ancestry_output_bed() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();
    let bed_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--output-bed").arg(bed_file.path());
    cmd.assert().success();

    let bed_content = std::fs::read_to_string(bed_file.path()).unwrap();
    assert!(!bed_content.is_empty(), "BED output should not be empty");
    // BED format: tab-separated, at least 3 columns (chrom, start, end)
    for line in bed_content.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        assert!(fields.len() >= 3, "BED line should have at least 3 fields: {}", line);
        // start and end should be numeric
        fields[1].parse::<u64>().expect("BED start should be numeric");
        fields[2].parse::<u64>().expect("BED end should be numeric");
    }
}

/// Test --posteriors-output produces per-window posterior probabilities
#[test]
fn test_ancestry_posteriors_output() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();
    let posteriors_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--posteriors-output").arg(posteriors_file.path());
    cmd.assert().success();

    let posteriors = std::fs::read_to_string(posteriors_file.path()).unwrap();
    assert!(!posteriors.is_empty(), "Posteriors output should not be empty");
    // Should have a header and data lines
    let lines: Vec<&str> = posteriors.lines().collect();
    assert!(lines.len() > 1, "Posteriors should have header + data");
}

/// Test --min-posterior filtering
#[test]
fn test_ancestry_min_posterior_filter() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);

    // Run with no filter
    let output_no_filter = NamedTempFile::new().unwrap();
    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_no_filter, 20,
    );
    cmd.arg("--min-posterior").arg("0.0");
    cmd.assert().success();
    let content_no = std::fs::read_to_string(output_no_filter.path()).unwrap();

    // Run with very high filter (should keep fewer or equal segments)
    let output_strict = NamedTempFile::new().unwrap();
    let mut cmd2 = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_strict, 20,
    );
    cmd2.arg("--min-posterior").arg("0.99");
    cmd2.assert().success();
    let content_strict = std::fs::read_to_string(output_strict.path()).unwrap();

    // Strict filter should have <= lines than no filter
    assert!(content_strict.lines().count() <= content_no.lines().count(),
        "Strict filter should produce fewer or equal segments");
}

/// Test --min-lod filtering
#[test]
fn test_ancestry_min_lod_filter() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);

    let output_file = NamedTempFile::new().unwrap();
    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--min-lod").arg("5.0");
    cmd.assert().success();
}

/// Test --smooth-min-windows removes short segments
#[test]
fn test_ancestry_smoothing() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(40);

    let output_file = NamedTempFile::new().unwrap();
    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 40,
    );
    cmd.arg("--smooth-min-windows").arg("3");
    cmd.assert().success();
}

/// Test --coverage-feature with alignment length data
#[test]
fn test_ancestry_coverage_feature() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--coverage-feature")
       .arg("--coverage-weight").arg("1.0");
    cmd.assert().success();
}

/// Test --similarity-column with jaccard.similarity
#[test]
fn test_ancestry_similarity_column() {
    // Create a similarity file with both identity and jaccard columns
    let mut sim_file = NamedTempFile::new().unwrap();
    writeln!(sim_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tjaccard.similarity").unwrap();
    for i in 0..20u64 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        let sim = 0.995;
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_a1\t{:.4}\t{:.4}", start, end, sim, sim - 0.01).unwrap();
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_b1\t{:.4}\t{:.4}", start, end, sim - 0.005, sim - 0.015).unwrap();
    }
    sim_file.flush().unwrap();

    let mut query_file = NamedTempFile::new().unwrap();
    writeln!(query_file, "query#1").unwrap();
    query_file.flush().unwrap();

    let mut pop_file = NamedTempFile::new().unwrap();
    writeln!(pop_file, "POP_A\tref_a1").unwrap();
    writeln!(pop_file, "POP_B\tref_b1").unwrap();
    pop_file.flush().unwrap();

    let dummy_seq = NamedTempFile::new().unwrap();
    let dummy_align = NamedTempFile::new().unwrap();
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--similarity-column").arg("jaccard.similarity");
    cmd.assert().success();
}

/// Test --temperature explicit value
#[test]
fn test_ancestry_explicit_temperature() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--temperature").arg("0.05")
       .arg("--switch-prob").arg("0.01");
    cmd.assert().success();
}

/// Test combined flags: estimate-params + emission-model + posterior decoding
#[test]
fn test_ancestry_combined_flags() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(30);
    let output_file = NamedTempFile::new().unwrap();
    let posteriors_file = NamedTempFile::new().unwrap();
    let bed_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 30,
    );
    cmd.arg("--estimate-params")
       .arg("--emission-model").arg("max")
       .arg("--decoding").arg("posterior")
       .arg("--posteriors-output").arg(posteriors_file.path())
       .arg("--output-bed").arg(bed_file.path())
       .arg("--min-posterior").arg("0.5")
       .arg("--smooth-min-windows").arg("2");
    cmd.assert().success();

    // All output files should exist and have content
    let content = std::fs::read_to_string(output_file.path()).unwrap();
    assert!(!content.is_empty());
    let posteriors = std::fs::read_to_string(posteriors_file.path()).unwrap();
    assert!(!posteriors.is_empty());
}

/// Test --validate-against with synthetic RFMix output
#[test]
fn test_ancestry_validate_against_rfmix() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();
    let validate_output = NamedTempFile::new().unwrap();

    // Create synthetic RFMix .msp.tsv file
    // Format: line 1 = population header, line 2 = column header, then data
    let mut rfmix_file = NamedTempFile::new().unwrap();
    writeln!(rfmix_file, "#Subpopulation order/codes: POP_A=0\tPOP_B=1").unwrap();
    writeln!(rfmix_file, "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tquery#1.0\tquery#1.1").unwrap();
    // First half POP_A (0), second half POP_B (1)
    writeln!(rfmix_file, "chr20\t1\t50000\t0.0\t1.0\t100\t0\t0").unwrap();
    writeln!(rfmix_file, "chr20\t50001\t100000\t1.0\t2.0\t100\t1\t1").unwrap();
    rfmix_file.flush().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--validate-against").arg(rfmix_file.path())
       .arg("--validate-output").arg(validate_output.path());
    cmd.assert().success();

    let val_content = std::fs::read_to_string(validate_output.path()).unwrap();
    assert!(!val_content.is_empty(), "Validation output should not be empty");
}

/// Test --emission-model top5 (TopK variant)
#[test]
fn test_ancestry_topk_emission_model() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    // top2 since we only have 2 refs per pop
    cmd.arg("--emission-model").arg("top2");
    cmd.assert().success();
}

/// Test --identity-floor filters low-identity windows
#[test]
fn test_ancestry_identity_floor() {
    // Create fixture data with some low-identity windows (alignment gaps)
    let mut sim_file = NamedTempFile::new().unwrap();
    writeln!(sim_file, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity").unwrap();
    for i in 0..20u64 {
        let start = i * 5000 + 1;
        let end = (i + 1) * 5000;
        // Windows 5-9 have very low identity (alignment gaps)
        let sim_a = if (5..10).contains(&i) { 0.1 } else { 0.999 };
        let sim_b = if (5..10).contains(&i) { 0.05 } else { 0.990 };
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_a1\t{:.4}", start, end, sim_a).unwrap();
        writeln!(sim_file, "chr20\t{}\t{}\tquery#1\tref_b1\t{:.4}", start, end, sim_b).unwrap();
    }
    sim_file.flush().unwrap();

    let mut query_file = NamedTempFile::new().unwrap();
    writeln!(query_file, "query#1").unwrap();
    query_file.flush().unwrap();

    let mut pop_file = NamedTempFile::new().unwrap();
    writeln!(pop_file, "POP_A\tref_a1").unwrap();
    writeln!(pop_file, "POP_B\tref_b1").unwrap();
    pop_file.flush().unwrap();

    let dummy_seq = NamedTempFile::new().unwrap();
    let dummy_align = NamedTempFile::new().unwrap();
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--identity-floor").arg("0.9");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    // Check stderr mentions filtering
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Identity floor"), "Should mention identity floor filtering");
    assert!(stderr.contains("removed"), "Should report removed windows");
}

/// Test --identity-floor=0.0 (no filtering, default)
#[test]
fn test_ancestry_identity_floor_disabled() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    cmd.arg("--identity-floor").arg("0.0");
    cmd.assert().success();

    // Should NOT mention identity floor in output since 0.0 disables it
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("ancestry"))
        .arg("--sequence-files").arg(dummy_seq.path())
        .arg("--alignment").arg(dummy_align.path())
        .arg("--reference").arg("ref")
        .arg("--region").arg("chr20:1-100000")
        .arg("--query-samples").arg(query_file.path())
        .arg("--populations").arg(pop_file.path())
        .arg("--similarity-file").arg(sim_file.path())
        .arg("--output").arg(output_file.path())
        .arg("--identity-floor").arg("0.0")
        .arg("--threads").arg("1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Identity floor"), "Should not mention identity floor when disabled");
}

/// Test invalid --switch-prob values are rejected
#[test]
fn test_ancestry_invalid_switch_prob() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("dummy")
       .arg("--alignment").arg("dummy")
       .arg("--reference").arg("ref")
       .arg("--region").arg("chr1:1-1000")
       .arg("--query-samples").arg("dummy")
       .arg("--output").arg("dummy")
       .arg("--switch-prob").arg("1.5");  // > 1.0, invalid
    cmd.assert().failure();
}

/// Test invalid --min-posterior values are rejected
#[test]
fn test_ancestry_invalid_min_posterior() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("dummy")
       .arg("--alignment").arg("dummy")
       .arg("--reference").arg("ref")
       .arg("--region").arg("chr1:1-1000")
       .arg("--query-samples").arg("dummy")
       .arg("--output").arg("dummy")
       .arg("--min-posterior").arg("-0.5");  // < 0.0, invalid
    cmd.assert().failure();
}

/// Test invalid --temperature values are rejected
#[test]
fn test_ancestry_invalid_temperature() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("dummy")
       .arg("--alignment").arg("dummy")
       .arg("--reference").arg("ref")
       .arg("--region").arg("chr1:1-1000")
       .arg("--query-samples").arg("dummy")
       .arg("--output").arg("dummy")
       .arg("--temperature").arg("-1.0");  // negative, invalid
    cmd.assert().failure();
}

/// Test invalid --identity-floor values are rejected
#[test]
fn test_ancestry_invalid_identity_floor() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("dummy")
       .arg("--alignment").arg("dummy")
       .arg("--reference").arg("ref")
       .arg("--region").arg("chr1:1-1000")
       .arg("--query-samples").arg("dummy")
       .arg("--output").arg("dummy")
       .arg("--identity-floor").arg("2.0");  // > 1.0, invalid
    cmd.assert().failure();
}

/// Test help output includes parameter presets
#[test]
fn test_ancestry_help_contains_presets() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--help");
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("RECOMMENDED PARAMETER PRESETS"), "missing parameter presets");
    assert!(stdout.contains("Production"), "missing production preset");
    assert!(stdout.contains("--identity-floor"), "missing --identity-floor flag");
    assert!(stdout.contains("93.4%"), "missing concordance result");
}

/// Test invalid --min-lod values are rejected
#[test]
fn test_ancestry_invalid_min_lod() {
    let mut cmd = Command::cargo_bin("ancestry").unwrap();
    cmd.arg("--sequence-files").arg("dummy")
       .arg("--alignment").arg("dummy")
       .arg("--reference").arg("ref")
       .arg("--region").arg("chr1:1-1000")
       .arg("--query-samples").arg("dummy")
       .arg("--output").arg("dummy")
       .arg("--min-lod").arg("-1.0");  // negative, invalid
    cmd.assert().failure();
}

/// Test identity-floor with 1.0 (all windows pass if they have any similarity)
#[test]
fn test_ancestry_identity_floor_one() {
    let (sim_file, query_file, pop_file, dummy_seq, dummy_align) = create_ancestry_test_fixtures(20);
    let output_file = NamedTempFile::new().unwrap();

    let mut cmd = ancestry_cmd_with_fixtures(
        &sim_file, &query_file, &pop_file, &dummy_seq, &dummy_align, &output_file, 20,
    );
    // 1.0 should filter most windows (only perfect identity passes)
    cmd.arg("--identity-floor").arg("1.0");

    let output = cmd.output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Identity floor"), "Should mention identity floor");
}
