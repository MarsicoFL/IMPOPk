//! CLI integration tests for --query-samples and --ref-samples filtering (A1).
//! Tests file validation, error messages, and flag combinations.

use assert_cmd::Command;
use std::io::Write;
use tempfile::{NamedTempFile, TempDir};

// ── Helper ──────────────────────────────────────────────────────────

/// Holds all required temp files so they stay alive during the test.
struct TestEnv {
    _dir: TempDir,
    seq_file: std::path::PathBuf,
    align_file: std::path::PathBuf,
    subset_file: std::path::PathBuf,
    output_file: std::path::PathBuf,
}

impl TestEnv {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let seq_file = dir.path().join("seq.agc");
        let align_file = dir.path().join("align.paf");
        let subset_file = dir.path().join("subset.txt");
        let output_file = dir.path().join("out.tsv");
        std::fs::write(&seq_file, "").unwrap();
        std::fs::write(&align_file, "").unwrap();
        std::fs::write(&subset_file, "HG01175#1\nHG01175#2\n").unwrap();
        TestEnv {
            _dir: dir,
            seq_file,
            align_file,
            subset_file,
            output_file,
        }
    }

    fn base_cmd(&self) -> Command {
        let mut cmd = Command::cargo_bin("ibs").unwrap();
        cmd.arg("--sequence-files")
            .arg(&self.seq_file)
            .arg("-a")
            .arg(&self.align_file)
            .arg("-r")
            .arg("REF")
            .arg("--size")
            .arg("10000")
            .arg("--subset-sequence-list")
            .arg(&self.subset_file)
            .arg("--output")
            .arg(&self.output_file)
            .arg("--region")
            .arg("chr1:1-10000");
        cmd
    }
}

fn write_sample_file(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{}", content).unwrap();
    f.flush().unwrap();
    f
}

// ── --query-samples flag ────────────────────────────────────────────

/// --query-samples with a nonexistent file should fail with clear error
#[test]
fn test_query_samples_nonexistent_file() {
    let env = TestEnv::new();
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples")
        .arg("/nonexistent/query_samples.txt");

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("query-samples file does not exist"),
        "Expected 'query-samples file does not exist' error, got: {}",
        stderr
    );
}

/// --ref-samples with a nonexistent file should fail with clear error
#[test]
fn test_ref_samples_nonexistent_file() {
    let env = TestEnv::new();
    let mut cmd = env.base_cmd();
    cmd.arg("--ref-samples")
        .arg("/nonexistent/ref_samples.txt");

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ref-samples file does not exist"),
        "Expected 'ref-samples file does not exist' error, got: {}",
        stderr
    );
}

/// --query-samples with valid file should load samples before failing on impg
#[test]
fn test_query_samples_valid_file_loads_ok() {
    let env = TestEnv::new();
    let samples = write_sample_file("HG01175\nNA19239\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples").arg(samples.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Loaded 2 query samples"),
        "Expected sample count in stderr, got: {}",
        stderr
    );
}

/// --ref-samples with valid file shows loaded count
#[test]
fn test_ref_samples_valid_file_loads_ok() {
    let env = TestEnv::new();
    let samples = write_sample_file("HG02257\nNA20845\nHG00099\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--ref-samples").arg(samples.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Loaded 3 reference samples"),
        "Expected reference sample count, got: {}",
        stderr
    );
}

/// Both --query-samples and --ref-samples with valid files shows cross-pair info
#[test]
fn test_both_flags_shows_cross_pair_stats() {
    let env = TestEnv::new();
    let query = write_sample_file("HG01175\nNA19239\n");
    let refs = write_sample_file("HG02257\nNA20845\nHG00099\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples")
        .arg(query.path())
        .arg("--ref-samples")
        .arg(refs.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    // 2 query × 3 ref → max 2*2 * 2*3 = 24 cross-haplotype pairs
    assert!(
        stderr.contains("Query-ref mode: 2 query"),
        "Expected cross-pair stats, got: {}",
        stderr
    );
    assert!(
        stderr.contains("24 cross-haplotype pairs"),
        "Expected 24 cross-haplotype pairs, got: {}",
        stderr
    );
}

/// --query-samples with empty file loads 0 samples
#[test]
fn test_query_samples_empty_file() {
    let env = TestEnv::new();
    let samples = write_sample_file("");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples").arg(samples.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Loaded 0 query samples"),
        "Expected 0 query samples, got: {}",
        stderr
    );
}

/// --query-samples file with only comments and blanks loads 0
#[test]
fn test_query_samples_comments_only() {
    let env = TestEnv::new();
    let samples = write_sample_file("# population AFR\n# more comments\n\n  \n");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples").arg(samples.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Loaded 0 query samples"),
        "Expected 0 query samples from comments-only file, got: {}",
        stderr
    );
}

/// --query-samples file with duplicates deduplicates correctly
#[test]
fn test_query_samples_deduplicates() {
    let env = TestEnv::new();
    let samples = write_sample_file("HG01175\nHG01175\nNA19239\nNA19239\nNA19239\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples").arg(samples.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    // HashSet deduplicates: 5 lines → 2 unique samples
    assert!(
        stderr.contains("Loaded 2 query samples"),
        "Expected 2 unique query samples after dedup, got: {}",
        stderr
    );
}

/// --query-samples without --ref-samples is valid (filters to pairs involving query)
#[test]
fn test_query_only_no_ref_is_valid() {
    let env = TestEnv::new();
    let query = write_sample_file("HG01175\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--query-samples").arg(query.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT see "Query-ref mode" since --ref-samples not provided
    assert!(
        !stderr.contains("Query-ref mode"),
        "Should not show cross-pair stats without --ref-samples, got: {}",
        stderr
    );
    assert!(
        stderr.contains("Loaded 1 query samples"),
        "Expected query sample count, got: {}",
        stderr
    );
}

/// --ref-samples without --query-samples is valid (filters to pairs involving ref)
#[test]
fn test_ref_only_no_query_is_valid() {
    let env = TestEnv::new();
    let refs = write_sample_file("NA19239\nHG02257\n");
    let mut cmd = env.base_cmd();
    cmd.arg("--ref-samples").arg(refs.path());

    let output = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Query-ref mode"),
        "Should not show cross-pair stats without --query-samples, got: {}",
        stderr
    );
    assert!(
        stderr.contains("Loaded 2 reference samples"),
        "Expected ref sample count, got: {}",
        stderr
    );
}
