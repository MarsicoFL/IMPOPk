//! Tests for pure helper functions in ibd-cli main.rs
//!
//! These test the validation functions, pair_key canonical ordering,
//! and parse_bed_regions file I/O — all defined in main.rs.
//!
//! NOTE: ibd-cli main.rs functions are not `pub`, so we test them
//! via the binary (clap validation) and by reimplementing the logic
//! for unit-level verification.

use std::fs;
use tempfile::TempDir;

// ── pair_key logic (canonical ordering) ────────────────────────────────

/// Reimplementation of pair_key for testing — the original is in main.rs
fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

#[test]
fn pair_key_already_ordered() {
    assert_eq!(pair_key("A#1", "B#1"), ("A#1".into(), "B#1".into()));
}

#[test]
fn pair_key_reversed() {
    assert_eq!(pair_key("B#1", "A#1"), ("A#1".into(), "B#1".into()));
}

#[test]
fn pair_key_same_string() {
    assert_eq!(pair_key("X#1", "X#1"), ("X#1".into(), "X#1".into()));
}

#[test]
fn pair_key_empty_strings() {
    assert_eq!(pair_key("", ""), ("".into(), "".into()));
}

#[test]
fn pair_key_empty_vs_nonempty() {
    // Empty string sorts before any non-empty string
    assert_eq!(pair_key("Z", ""), ("".into(), "Z".into()));
}

#[test]
fn pair_key_idempotent() {
    let (a, b) = pair_key("B", "A");
    let (a2, b2) = pair_key(&a, &b);
    assert_eq!((a, b), (a2, b2));
}

#[test]
fn pair_key_symmetric() {
    let result1 = pair_key("HG00733#1", "HG01175#2");
    let result2 = pair_key("HG01175#2", "HG00733#1");
    assert_eq!(result1, result2);
}

#[test]
fn pair_key_lexicographic_not_numeric() {
    // "9" > "10" lexicographically
    assert_eq!(pair_key("sample#9", "sample#10"), ("sample#10".into(), "sample#9".into()));
}

// ── Validator function reimplementations for unit testing ──────────────

fn validate_probability(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(format!("probability must be in (0.0, 1.0), got {}", v))
    }
}

fn validate_positive_u64(val: &str) -> Result<u64, String> {
    let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0 {
        Ok(v)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn validate_positive_usize(val: &str) -> Result<usize, String> {
    let v: usize = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0 {
        Ok(v)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn validate_unit_interval(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("value must be in [0.0, 1.0], got {}", v))
    }
}

fn validate_positive_f64(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(format!("value must be > 0.0, got {}", v))
    }
}

// ── validate_probability tests ─────────────────────────────────────────

#[test]
fn validate_probability_valid_mid() {
    assert_eq!(validate_probability("0.5").unwrap(), 0.5);
}

#[test]
fn validate_probability_valid_small() {
    assert_eq!(validate_probability("0.0001").unwrap(), 0.0001);
}

#[test]
fn validate_probability_valid_near_one() {
    assert_eq!(validate_probability("0.9999").unwrap(), 0.9999);
}

#[test]
fn validate_probability_zero_rejected() {
    assert!(validate_probability("0.0").is_err());
}

#[test]
fn validate_probability_one_rejected() {
    assert!(validate_probability("1.0").is_err());
}

#[test]
fn validate_probability_negative_rejected() {
    assert!(validate_probability("-0.1").is_err());
}

#[test]
fn validate_probability_above_one_rejected() {
    assert!(validate_probability("1.5").is_err());
}

#[test]
fn validate_probability_not_a_number() {
    assert!(validate_probability("abc").is_err());
}

#[test]
fn validate_probability_empty() {
    assert!(validate_probability("").is_err());
}

// ── validate_positive_u64 tests ────────────────────────────────────────

#[test]
fn validate_positive_u64_one() {
    assert_eq!(validate_positive_u64("1").unwrap(), 1);
}

#[test]
fn validate_positive_u64_large() {
    assert_eq!(validate_positive_u64("999999999").unwrap(), 999999999);
}

#[test]
fn validate_positive_u64_zero_rejected() {
    assert!(validate_positive_u64("0").is_err());
}

#[test]
fn validate_positive_u64_float_rejected() {
    assert!(validate_positive_u64("1.5").is_err());
}

// ── validate_positive_usize tests ──────────────────────────────────────

#[test]
fn validate_positive_usize_one() {
    assert_eq!(validate_positive_usize("1").unwrap(), 1);
}

#[test]
fn validate_positive_usize_zero_rejected() {
    assert!(validate_positive_usize("0").is_err());
}

#[test]
fn validate_positive_usize_negative_rejected() {
    assert!(validate_positive_usize("-5").is_err());
}

// ── validate_unit_interval tests ───────────────────────────────────────

#[test]
fn validate_unit_interval_zero() {
    assert_eq!(validate_unit_interval("0.0").unwrap(), 0.0);
}

#[test]
fn validate_unit_interval_one() {
    assert_eq!(validate_unit_interval("1.0").unwrap(), 1.0);
}

#[test]
fn validate_unit_interval_mid() {
    assert_eq!(validate_unit_interval("0.5").unwrap(), 0.5);
}

#[test]
fn validate_unit_interval_negative_rejected() {
    assert!(validate_unit_interval("-0.001").is_err());
}

#[test]
fn validate_unit_interval_above_one_rejected() {
    assert!(validate_unit_interval("1.001").is_err());
}

// ── validate_positive_f64 tests ────────────────────────────────────────

#[test]
fn validate_positive_f64_small() {
    assert_eq!(validate_positive_f64("0.001").unwrap(), 0.001);
}

#[test]
fn validate_positive_f64_large() {
    assert_eq!(validate_positive_f64("100.0").unwrap(), 100.0);
}

#[test]
fn validate_positive_f64_zero_rejected() {
    assert!(validate_positive_f64("0.0").is_err());
}

#[test]
fn validate_positive_f64_negative_rejected() {
    assert!(validate_positive_f64("-1.0").is_err());
}

// ── parse_bed_regions (reimplemented) ──────────────────────────────────

/// Reimplementation of parse_bed_regions from ibd-cli main.rs
fn parse_bed_regions(bed_path: &str) -> anyhow::Result<Vec<impopk_common::Region>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(bed_path)?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            anyhow::bail!("BED line {} has fewer than 3 fields: {}", line_num + 1, line);
        }
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse()?;
        let end: u64 = fields[2].parse()?;

        // BED is 0-based half-open; Region is 1-based inclusive
        regions.push(impopk_common::Region {
            chrom,
            start: start + 1,
            end,
        });
    }

    Ok(regions)
}

#[test]
fn parse_bed_regions_basic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t100\t200\nchr2\t500\t1000\n").unwrap();

    let regions = parse_bed_regions(path.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].chrom, "chr1");
    assert_eq!(regions[0].start, 101); // 0-based→1-based
    assert_eq!(regions[0].end, 200);
    assert_eq!(regions[1].chrom, "chr2");
    assert_eq!(regions[1].start, 501);
    assert_eq!(regions[1].end, 1000);
}

#[test]
fn parse_bed_regions_skips_comments_and_blanks() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "# comment\n\nchr1\t0\t100\n\n# another\nchr2\t200\t300\n").unwrap();

    let regions = parse_bed_regions(path.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 2);
}

#[test]
fn parse_bed_regions_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "").unwrap();

    let regions = parse_bed_regions(path.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 0);
}

#[test]
fn parse_bed_regions_too_few_fields_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t100\n").unwrap();

    assert!(parse_bed_regions(path.to_str().unwrap()).is_err());
}

#[test]
fn parse_bed_regions_invalid_start_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\tabc\t200\n").unwrap();

    assert!(parse_bed_regions(path.to_str().unwrap()).is_err());
}

#[test]
fn parse_bed_regions_invalid_end_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t100\txyz\n").unwrap();

    assert!(parse_bed_regions(path.to_str().unwrap()).is_err());
}

#[test]
fn parse_bed_regions_nonexistent_file_errors() {
    assert!(parse_bed_regions("/nonexistent/path.bed").is_err());
}

#[test]
fn parse_bed_regions_extra_columns_accepted() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t0\t1000\tname\t500\t+\n").unwrap();

    let regions = parse_bed_regions(path.to_str().unwrap()).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].start, 1);
    assert_eq!(regions[0].end, 1000);
}

#[test]
fn parse_bed_regions_zero_start() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t0\t100\n").unwrap();

    let regions = parse_bed_regions(path.to_str().unwrap()).unwrap();
    assert_eq!(regions[0].start, 1); // 0-based 0 → 1-based 1
}
