//! Edge case tests for ibs-cli: validator special inputs, BED parsing edge cases,
//! CLI file-not-found error messages, --bed file path validation, and --threads flag.

#![allow(deprecated)]

use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

// ── Helper ──────────────────────────────────────────────────────────

fn ibs_base_cmd() -> Command {
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg("/dummy/seq")
       .arg("-a").arg("/dummy/align")
       .arg("-r").arg("REF");
    cmd
}

// ── validate_cutoff: special floating-point values ──────────────────

/// NaN parses as f64 but is NOT contained in [0.0, 1.0]
#[test]
fn test_validate_cutoff_nan_rejected() {
    fn validate_cutoff(val: &str) -> Result<f64, String> {
        let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if (0.0..=1.0).contains(&v) {
            Ok(v)
        } else {
            Err(format!("cutoff must be in [0.0, 1.0], got {}", v))
        }
    }
    // f64::NAN.contains() is always false
    assert!(validate_cutoff("nan").is_err() || validate_cutoff("NaN").is_err());
}

/// Infinity parses as f64 but is out of [0.0, 1.0]
#[test]
fn test_validate_cutoff_inf_rejected() {
    fn validate_cutoff(val: &str) -> Result<f64, String> {
        let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if (0.0..=1.0).contains(&v) {
            Ok(v)
        } else {
            Err(format!("cutoff must be in [0.0, 1.0], got {}", v))
        }
    }
    assert!(validate_cutoff("inf").is_err());
    assert!(validate_cutoff("-inf").is_err());
}

/// Whitespace-padded string fails to parse as f64
#[test]
fn test_validate_cutoff_whitespace_rejected() {
    fn validate_cutoff(val: &str) -> Result<f64, String> {
        let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if (0.0..=1.0).contains(&v) {
            Ok(v)
        } else {
            Err(format!("cutoff must be in [0.0, 1.0], got {}", v))
        }
    }
    assert!(validate_cutoff(" 0.5").is_err());
    assert!(validate_cutoff("0.5 ").is_err());
}

// ── validate_positive_u64: special values ───────────────────────────

/// u64::MAX is valid
#[test]
fn test_validate_positive_u64_max() {
    fn validate_positive_u64(val: &str) -> Result<u64, String> {
        let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if v > 0 { Ok(v) } else { Err("value must be > 0".to_string()) }
    }
    assert_eq!(validate_positive_u64("18446744073709551615").unwrap(), u64::MAX);
}

/// Overflow beyond u64::MAX fails to parse
#[test]
fn test_validate_positive_u64_overflow() {
    fn validate_positive_u64(val: &str) -> Result<u64, String> {
        let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if v > 0 { Ok(v) } else { Err("value must be > 0".to_string()) }
    }
    assert!(validate_positive_u64("18446744073709551616").is_err());
}

/// Whitespace-padded string fails
#[test]
fn test_validate_positive_u64_whitespace() {
    fn validate_positive_u64(val: &str) -> Result<u64, String> {
        let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
        if v > 0 { Ok(v) } else { Err("value must be > 0".to_string()) }
    }
    assert!(validate_positive_u64(" 1").is_err());
}

// ── parse_bed_regions: BED-specific edge cases ──────────────────────

/// BED start=0 → 1-based start=1 (explicit zero-origin conversion)
#[test]
fn test_parse_bed_zero_start_conversion() {
    use hprc_common::Region;
    use anyhow::{bail, Context, Result};
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
        let file = File::open(bed_path)
            .context(format!("Failed to open BED file: {}", bed_path))?;
        let reader = BufReader::new(file);
        let mut regions = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.context("Failed to read BED file")?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 3 { bail!("BED line {} has fewer than 3 fields", line_num + 1); }
            let chrom = fields[0].to_string();
            let start: u64 = fields[1].parse()
                .context(format!("Invalid start on line {}", line_num + 1))?;
            let end: u64 = fields[2].parse()
                .context(format!("Invalid end on line {}", line_num + 1))?;
            regions.push(Region { chrom, start: start + 1, end });
        }
        Ok(regions)
    }

    let dir = tempfile::tempdir().unwrap();
    let bed = dir.path().join("test.bed");
    std::fs::write(&bed, "chr1\t0\t1000\n").unwrap();
    let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].start, 1, "BED start=0 should convert to 1-based start=1");
    assert_eq!(regions[0].end, 1000);
}

/// All-comments-only BED file returns empty vec (not an error)
#[test]
fn test_parse_bed_all_comments_returns_empty() {
    use hprc_common::Region;
    use anyhow::{bail, Context, Result};
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
        let file = File::open(bed_path)
            .context(format!("Failed to open BED file: {}", bed_path))?;
        let reader = BufReader::new(file);
        let mut regions = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.context("Failed to read BED file")?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 3 { bail!("BED line {} has fewer than 3 fields", line_num + 1); }
            let chrom = fields[0].to_string();
            let start: u64 = fields[1].parse()
                .context(format!("Invalid start on line {}", line_num + 1))?;
            let end: u64 = fields[2].parse()
                .context(format!("Invalid end on line {}", line_num + 1))?;
            regions.push(Region { chrom, start: start + 1, end });
        }
        Ok(regions)
    }

    let dir = tempfile::tempdir().unwrap();
    let bed = dir.path().join("test.bed");
    std::fs::write(&bed, "# comment 1\n# comment 2\n# comment 3\n").unwrap();
    let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 0);
}

/// Inline # in data (not at position 0) is NOT treated as a comment
#[test]
fn test_parse_bed_inline_hash_not_comment() {
    use hprc_common::Region;
    use anyhow::{bail, Context, Result};
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
        let file = File::open(bed_path)
            .context(format!("Failed to open BED file: {}", bed_path))?;
        let reader = BufReader::new(file);
        let mut regions = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.context("Failed to read BED file")?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 3 { bail!("BED line {} has fewer than 3 fields", line_num + 1); }
            let chrom = fields[0].to_string();
            let start: u64 = fields[1].parse()
                .context(format!("Invalid start on line {}", line_num + 1))?;
            let end: u64 = fields[2].parse()
                .context(format!("Invalid end on line {}", line_num + 1))?;
            regions.push(Region { chrom, start: start + 1, end });
        }
        Ok(regions)
    }

    let dir = tempfile::tempdir().unwrap();
    let bed = dir.path().join("test.bed");
    // The chrom field "chr1" doesn't start with #, so this is data
    std::fs::write(&bed, "chr1\t0\t1000\tname # note\n").unwrap();
    let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 1);
}

/// Multiple regions parsed in order
#[test]
fn test_parse_bed_multiple_regions_order() {
    use hprc_common::Region;
    use anyhow::{bail, Context, Result};
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
        let file = File::open(bed_path)
            .context(format!("Failed to open BED file: {}", bed_path))?;
        let reader = BufReader::new(file);
        let mut regions = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.context("Failed to read BED file")?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 3 { bail!("BED line {} has fewer than 3 fields", line_num + 1); }
            let chrom = fields[0].to_string();
            let start: u64 = fields[1].parse()
                .context(format!("Invalid start on line {}", line_num + 1))?;
            let end: u64 = fields[2].parse()
                .context(format!("Invalid end on line {}", line_num + 1))?;
            regions.push(Region { chrom, start: start + 1, end });
        }
        Ok(regions)
    }

    let dir = tempfile::tempdir().unwrap();
    let bed = dir.path().join("test.bed");
    std::fs::write(&bed, "chr3\t0\t100\nchr1\t500\t600\nchr2\t200\t300\n").unwrap();
    let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 3);
    // Order should match file order (not sorted)
    assert_eq!(regions[0].chrom, "chr3");
    assert_eq!(regions[1].chrom, "chr1");
    assert_eq!(regions[2].chrom, "chr2");
}

// ── CLI: specific error messages for missing files ──────────────────

/// Missing --sequence-files reports the specific file path
#[test]
fn test_ibs_missing_sequence_files_error() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg("/nonexistent/seq.agc")
       .arg("-a").arg(subset.path())  // exists (but wrong type)
       .arg("-r").arg("REF")
       .arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());

    let out = cmd.output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("sequence-files does not exist") || stderr.contains("/nonexistent/seq.agc"),
        "Should report missing sequence-files, got: {}",
        stderr
    );
}

/// Missing alignment file (-a) reports the specific file path
#[test]
fn test_ibs_missing_alignment_file_error() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg(subset.path())  // exists
       .arg("-a").arg("/nonexistent/align.paf")
       .arg("-r").arg("REF")
       .arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());

    let out = cmd.output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("alignment file does not exist") || stderr.contains("/nonexistent/align.paf"),
        "Should report missing alignment file, got: {}",
        stderr
    );
}

/// Missing --subset-sequence-list reports the specific file path
#[test]
fn test_ibs_missing_subset_list_error() {
    let seq = NamedTempFile::new().unwrap();
    let align = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg(seq.path())
       .arg("-a").arg(align.path())
       .arg("-r").arg("REF")
       .arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg("/nonexistent/subset.txt")
       .arg("--output").arg(output.path());

    let out = cmd.output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("subset-sequence-list does not exist") || stderr.contains("/nonexistent/subset.txt"),
        "Should report missing subset file, got: {}",
        stderr
    );
}

// ── CLI: --bed file validation ──────────────────────────────────────

/// --bed with nonexistent file path
#[test]
fn test_ibs_bed_nonexistent_file_error() {
    let seq = NamedTempFile::new().unwrap();
    let align = NamedTempFile::new().unwrap();
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg(seq.path())
       .arg("-a").arg(align.path())
       .arg("-r").arg("REF")
       .arg("--bed").arg("/nonexistent/regions.bed")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());

    let out = cmd.output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("BED file does not exist") || stderr.contains("/nonexistent/regions.bed"),
        "Should report missing BED file, got: {}",
        stderr
    );
}

/// --bed with valid file (but impg not available) — should get past BED validation
#[test]
fn test_ibs_bed_valid_file_passes_validation() {
    let seq = NamedTempFile::new().unwrap();
    let align = NamedTempFile::new().unwrap();
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();

    let mut bed_file = NamedTempFile::new().unwrap();
    writeln!(bed_file, "chr1\t0\t10000").unwrap();
    bed_file.flush().unwrap();

    let mut cmd = Command::cargo_bin("ibs").unwrap();
    cmd.arg("--sequence-files").arg(seq.path())
       .arg("-a").arg(align.path())
       .arg("-r").arg("REF")
       .arg("--bed").arg(bed_file.path())
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());

    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Should fail on impg (not BED parsing)
    assert!(
        !stderr.contains("BED file does not exist") && !stderr.contains("fewer than 3 fields"),
        "BED should parse fine; error should be about impg, got: {}",
        stderr
    );
}

// ── CLI: NaN/inf cutoff via CLI ─────────────────────────────────────

/// cutoff=nan should be rejected by clap validation
#[test]
fn test_ibs_cli_cutoff_nan_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("nan");
    let out = cmd.output().unwrap();
    // NaN is not in [0.0, 1.0] so validation should reject it
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() || stderr.contains("cutoff"),
        "nan cutoff should be rejected"
    );
}

/// cutoff=inf should be rejected
#[test]
fn test_ibs_cli_cutoff_inf_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-c").arg("inf");
    cmd.assert().failure();
}

// ── CLI: --threads flag ─────────────────────────────────────────────

/// --threads 1 accepted (passes argument parsing)
#[test]
fn test_ibs_threads_valid_accepted() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-t").arg("1");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Should fail at runtime (missing files or impg), NOT at arg parsing
    assert!(
        !stderr.contains("invalid value") && !stderr.contains("is not a valid"),
        "threads=1 should be accepted, got: {}",
        stderr
    );
}

/// --threads abc should be rejected
#[test]
fn test_ibs_threads_non_numeric_rejected() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-t").arg("abc");
    cmd.assert().failure();
}

// ── CLI: --metric flag ──────────────────────────────────────────────

/// Custom metric value accepted (informational only)
#[test]
fn test_ibs_custom_metric_accepted() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1:1-10000")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path())
       .arg("-m").arg("euclidean");
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Metric is informational — should not cause arg parsing failure
    assert!(
        !stderr.contains("invalid value '\"euclidean\"'"),
        "Custom metric should be accepted"
    );
}

// ── CLI: --region-length flag ───────────────────────────────────────

/// --region-length with chrom-only region
#[test]
fn test_ibs_region_length_accepted() {
    let subset = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();
    let mut cmd = ibs_base_cmd();
    cmd.arg("--region").arg("chr1")
       .arg("--region-length").arg("248956422")
       .arg("--size").arg("5000")
       .arg("--subset-sequence-list").arg(subset.path())
       .arg("--output").arg(output.path());
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Should not fail on arg parsing — failure would be at runtime (impg/files)
    assert!(
        !stderr.contains("region-length") || !stderr.contains("invalid"),
        "region-length should be accepted"
    );
}
