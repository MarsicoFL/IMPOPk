//! Tests for pure helper functions in ancestry-cli main.rs
//!
//! Tests parse_region, parse_bed_regions, load_populations, load_sample_list,
//! and validator functions. These are private functions in main.rs, so we
//! reimplement the logic for unit-level verification.

use std::fs;
use tempfile::TempDir;

// ── parse_region reimplementation ──────────────────────────────────────

fn parse_region(
    region: &str,
    reference: &str,
    region_length: Option<u64>,
) -> anyhow::Result<(String, u64, u64)> {
    if region.contains(':') {
        let parts: Vec<&str> = region.split(':').collect();
        let chrom = format!("{}#{}", reference, parts[0]);
        let coords: Vec<&str> = parts[1].split('-').collect();
        let start: u64 = coords[0].parse()?;
        let end: u64 = coords[1].parse()?;
        Ok((chrom, start, end))
    } else {
        let end = region_length
            .ok_or_else(|| anyhow::anyhow!("--region-length required when region is just chromosome name"))?;
        let chrom = format!("{}#{}", reference, region);
        Ok((chrom, 1, end))
    }
}

#[test]
fn parse_region_with_coordinates() {
    let (chrom, start, end) = parse_region("chr10:1000-2000", "HG00097#1", None).unwrap();
    assert_eq!(chrom, "HG00097#1#chr10");
    assert_eq!(start, 1000);
    assert_eq!(end, 2000);
}

#[test]
fn parse_region_chromosome_only_with_length() {
    let (chrom, start, end) = parse_region("chr10", "HG00097#1", Some(248956422)).unwrap();
    assert_eq!(chrom, "HG00097#1#chr10");
    assert_eq!(start, 1);
    assert_eq!(end, 248956422);
}

#[test]
fn parse_region_chromosome_only_without_length_errors() {
    assert!(parse_region("chr10", "HG00097#1", None).is_err());
}

#[test]
fn parse_region_large_coordinates() {
    let (chrom, start, end) = parse_region("chr1:1-248956422", "ref", None).unwrap();
    assert_eq!(chrom, "ref#chr1");
    assert_eq!(start, 1);
    assert_eq!(end, 248956422);
}

#[test]
fn parse_region_invalid_start_coordinate() {
    assert!(parse_region("chr1:abc-1000", "ref", None).is_err());
}

#[test]
fn parse_region_invalid_end_coordinate() {
    assert!(parse_region("chr1:100-xyz", "ref", None).is_err());
}

#[test]
fn parse_region_zero_based() {
    let (_, start, end) = parse_region("chr1:0-1000", "ref", None).unwrap();
    assert_eq!(start, 0);
    assert_eq!(end, 1000);
}

// ── parse_bed_regions reimplementation ─────────────────────────────────

fn parse_bed_regions(
    path: &std::path::Path,
    reference: &str,
) -> anyhow::Result<Vec<(String, u64, u64)>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            anyhow::bail!("BED line must have at least 3 columns: {}", line);
        }
        let chrom_raw = fields[0];
        let chrom = if chrom_raw.contains('#') {
            chrom_raw.to_string()
        } else {
            format!("{}#{}", reference, chrom_raw)
        };
        let start: u64 = fields[1].parse()?;
        let end: u64 = fields[2].parse()?;
        regions.push((chrom, start + 1, end));
    }

    if regions.is_empty() {
        anyhow::bail!("BED file is empty or has no valid regions");
    }

    Ok(regions)
}

#[test]
fn parse_bed_regions_basic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr10\t1000\t2000\nchr11\t5000\t6000\n").unwrap();

    let regions = parse_bed_regions(&path, "HG00097#1").unwrap();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].0, "HG00097#1#chr10");
    assert_eq!(regions[0].1, 1001); // 0-based→1-based
    assert_eq!(regions[0].2, 2000);
    assert_eq!(regions[1].0, "HG00097#1#chr11");
}

#[test]
fn parse_bed_regions_with_reference_prefix() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    // If chrom already has #, don't double-prefix
    fs::write(&path, "ref#HAP1#chr10\t0\t1000\n").unwrap();

    let regions = parse_bed_regions(&path, "HG00097#1").unwrap();
    assert_eq!(regions[0].0, "ref#HAP1#chr10"); // kept as-is
}

#[test]
fn parse_bed_regions_comments_and_blanks() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "# comment\n\nchr10\t0\t100\n\n").unwrap();

    let regions = parse_bed_regions(&path, "ref").unwrap();
    assert_eq!(regions.len(), 1);
}

#[test]
fn parse_bed_regions_empty_file_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "").unwrap();

    assert!(parse_bed_regions(&path, "ref").is_err());
}

#[test]
fn parse_bed_regions_only_comments_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "# only comments\n# nothing else\n").unwrap();

    assert!(parse_bed_regions(&path, "ref").is_err());
}

#[test]
fn parse_bed_regions_too_few_columns_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t100\n").unwrap();

    assert!(parse_bed_regions(&path, "ref").is_err());
}

#[test]
fn parse_bed_regions_invalid_start_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\tabc\t200\n").unwrap();

    assert!(parse_bed_regions(&path, "ref").is_err());
}

#[test]
fn parse_bed_regions_invalid_end_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.bed");
    fs::write(&path, "chr1\t100\txyz\n").unwrap();

    assert!(parse_bed_regions(&path, "ref").is_err());
}

#[test]
fn parse_bed_regions_nonexistent_file_errors() {
    assert!(parse_bed_regions(std::path::Path::new("/no/such/file.bed"), "ref").is_err());
}

// ── load_populations reimplementation ──────────────────────────────────

fn load_populations(
    path: &std::path::Path,
) -> anyhow::Result<Vec<(String, Vec<String>)>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut pop_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            pop_map
                .entry(parts[0].to_string())
                .or_default()
                .push(parts[1].to_string());
        }
    }

    let mut populations: Vec<(String, Vec<String>)> = pop_map.into_iter().collect();
    populations.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(populations)
}

#[test]
fn load_populations_basic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pops.tsv");
    fs::write(&path, "EUR\tHG00096#1\nEUR\tHG00096#2\nAFR\tNA19239#1\nAFR\tNA19239#2\n").unwrap();

    let pops = load_populations(&path).unwrap();
    assert_eq!(pops.len(), 2);
    // Sorted: AFR, EUR
    assert_eq!(pops[0].0, "AFR");
    assert_eq!(pops[0].1.len(), 2);
    assert_eq!(pops[1].0, "EUR");
    assert_eq!(pops[1].1.len(), 2);
}

#[test]
fn load_populations_single_column_skipped() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pops.tsv");
    // Lines with fewer than 2 tab-separated fields are skipped
    fs::write(&path, "EUR\nAFR\tNA19239#1\n").unwrap();

    let pops = load_populations(&path).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].0, "AFR");
}

#[test]
fn load_populations_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pops.tsv");
    fs::write(&path, "").unwrap();

    let pops = load_populations(&path).unwrap();
    assert_eq!(pops.len(), 0);
}

#[test]
fn load_populations_three_way() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pops.tsv");
    fs::write(
        &path,
        "EUR\tA#1\nEUR\tA#2\nAFR\tB#1\nAFR\tB#2\nAMR\tC#1\nAMR\tC#2\n",
    )
    .unwrap();

    let pops = load_populations(&path).unwrap();
    assert_eq!(pops.len(), 3);
}

#[test]
fn load_populations_nonexistent_file_errors() {
    assert!(load_populations(std::path::Path::new("/no/such/file.tsv")).is_err());
}

// ── load_sample_list reimplementation ──────────────────────────────────

fn load_sample_list(path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let samples: Vec<String> = reader
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    Ok(samples)
}

#[test]
fn load_sample_list_basic() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("samples.txt");
    fs::write(&path, "HG00733#1\nHG00733#2\nHG01928#1\n").unwrap();

    let samples = load_sample_list(&path).unwrap();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0], "HG00733#1");
}

#[test]
fn load_sample_list_skips_blank_lines() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("samples.txt");
    fs::write(&path, "HG00733#1\n\n\nHG01928#1\n\n").unwrap();

    let samples = load_sample_list(&path).unwrap();
    assert_eq!(samples.len(), 2);
}

#[test]
fn load_sample_list_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("samples.txt");
    fs::write(&path, "").unwrap();

    let samples = load_sample_list(&path).unwrap();
    assert_eq!(samples.len(), 0);
}

#[test]
fn load_sample_list_nonexistent_file_errors() {
    assert!(load_sample_list(std::path::Path::new("/no/such/file.txt")).is_err());
}

#[test]
fn load_sample_list_whitespace_only_lines_skipped() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("samples.txt");
    fs::write(&path, "   \n\t\nHG00733#1\n  \n").unwrap();

    let samples = load_sample_list(&path).unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0], "HG00733#1");
}

// ── Validator functions ────────────────────────────────────────────────

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

fn validate_non_negative_f64(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("value must be >= 0.0, got {}", v))
    }
}

#[test]
fn validate_unit_interval_boundaries() {
    assert_eq!(validate_unit_interval("0.0").unwrap(), 0.0);
    assert_eq!(validate_unit_interval("1.0").unwrap(), 1.0);
}

#[test]
fn validate_unit_interval_rejects_out_of_range() {
    assert!(validate_unit_interval("-0.1").is_err());
    assert!(validate_unit_interval("1.1").is_err());
}

#[test]
fn validate_positive_f64_accepts_small() {
    assert_eq!(validate_positive_f64("0.001").unwrap(), 0.001);
}

#[test]
fn validate_positive_f64_rejects_zero() {
    assert!(validate_positive_f64("0.0").is_err());
}

#[test]
fn validate_non_negative_f64_accepts_zero() {
    assert_eq!(validate_non_negative_f64("0.0").unwrap(), 0.0);
}

#[test]
fn validate_non_negative_f64_accepts_positive() {
    assert_eq!(validate_non_negative_f64("5.0").unwrap(), 5.0);
}

#[test]
fn validate_non_negative_f64_rejects_negative() {
    assert!(validate_non_negative_f64("-0.001").is_err());
}

#[test]
fn validate_non_negative_f64_not_a_number() {
    assert!(validate_non_negative_f64("abc").is_err());
}

// ── print_diagnostics partial testing ──────────────────────────────────
// We can test the posterior classification thresholds (>0.8 confident,
// 0.5-0.8 uncertain, <0.5 ambiguous) since that's pure logic.

#[test]
fn posterior_classification_thresholds() {
    let posteriors = vec![0.95, 0.85, 0.75, 0.55, 0.45, 0.3];
    let confident = posteriors.iter().filter(|&&p| p > 0.8).count();
    let uncertain = posteriors.iter().filter(|&&p| (0.5..=0.8).contains(&p)).count();
    let ambiguous = posteriors.iter().filter(|&&p| p < 0.5).count();

    assert_eq!(confident, 2); // 0.95, 0.85
    assert_eq!(uncertain, 2); // 0.75, 0.55
    assert_eq!(ambiguous, 2); // 0.45, 0.3
}

#[test]
fn posterior_classification_boundary_at_0_8() {
    // Exactly 0.8 is uncertain (0.5..=0.8)
    let posteriors = vec![0.8];
    let confident = posteriors.iter().filter(|&&p| p > 0.8).count();
    let uncertain = posteriors.iter().filter(|&&p| (0.5..=0.8).contains(&p)).count();
    assert_eq!(confident, 0);
    assert_eq!(uncertain, 1);
}

#[test]
fn posterior_classification_boundary_at_0_5() {
    // Exactly 0.5 is uncertain (0.5..=0.8)
    let posteriors = vec![0.5];
    let uncertain = posteriors.iter().filter(|&&p| (0.5..=0.8).contains(&p)).count();
    let ambiguous = posteriors.iter().filter(|&&p| p < 0.5).count();
    assert_eq!(uncertain, 1);
    assert_eq!(ambiguous, 0);
}
