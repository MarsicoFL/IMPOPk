//! Tests for Population per-variant constants, GeneticMap::from_file error paths,
//! and parse_hapibd_file I/O operations.

use impopk_ibd::hapibd::parse_hapibd_file;
use impopk_ibd::hmm::{GeneticMap, Population};
use std::io::Write;

// ============================================================================
// Population::diversity — per-variant exhaustive tests
// ============================================================================

#[test]
fn population_diversity_afr() {
    assert!((Population::AFR.diversity() - 0.00125).abs() < 1e-12);
}

#[test]
fn population_diversity_eur() {
    assert!((Population::EUR.diversity() - 0.00085).abs() < 1e-12);
}

#[test]
fn population_diversity_eas() {
    assert!((Population::EAS.diversity() - 0.00080).abs() < 1e-12);
}

#[test]
fn population_diversity_csa() {
    assert!((Population::CSA.diversity() - 0.00095).abs() < 1e-12);
}

#[test]
fn population_diversity_amr() {
    assert!((Population::AMR.diversity() - 0.00100).abs() < 1e-12);
}

#[test]
fn population_diversity_interpop() {
    assert!((Population::InterPop.diversity() - 0.00110).abs() < 1e-12);
}

#[test]
fn population_diversity_generic() {
    assert!((Population::Generic.diversity() - 0.00100).abs() < 1e-12);
}

/// AFR should have the highest diversity of the single-population variants.
#[test]
fn population_diversity_ordering() {
    let afr = Population::AFR.diversity();
    let eur = Population::EUR.diversity();
    let eas = Population::EAS.diversity();
    let csa = Population::CSA.diversity();
    assert!(afr > eur);
    assert!(afr > eas);
    assert!(afr > csa);
    // EAS should have the lowest
    assert!(eas < eur);
}

// ============================================================================
// Population::non_ibd_emission — per-variant tests
// ============================================================================

#[test]
fn non_ibd_emission_afr_10000() {
    let e = Population::AFR.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00125;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0, "std must be positive");
    assert!(e.std.is_finite());
}

#[test]
fn non_ibd_emission_eur_10000() {
    let e = Population::EUR.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00085;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

#[test]
fn non_ibd_emission_eas_10000() {
    let e = Population::EAS.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00080;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

#[test]
fn non_ibd_emission_csa_10000() {
    let e = Population::CSA.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00095;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

#[test]
fn non_ibd_emission_amr_10000() {
    let e = Population::AMR.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00100;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

#[test]
fn non_ibd_emission_interpop_10000() {
    let e = Population::InterPop.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00110;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

#[test]
fn non_ibd_emission_generic_10000() {
    let e = Population::Generic.non_ibd_emission(10_000);
    let expected_mean = 1.0 - 0.00100;
    assert!((e.mean - expected_mean).abs() < 1e-12);
    assert!(e.std > 0.0);
}

/// AFR should have higher diversity → lower mean identity and higher std.
#[test]
fn non_ibd_emission_afr_higher_diversity_than_eur() {
    let afr = Population::AFR.non_ibd_emission(10_000);
    let eur = Population::EUR.non_ibd_emission(10_000);
    assert!(afr.mean < eur.mean, "AFR should have lower mean identity");
    assert!(afr.std > eur.std, "AFR should have higher std");
}

/// Larger windows should produce smaller std (more averaging).
#[test]
fn non_ibd_emission_std_decreases_with_window_size() {
    let small = Population::EUR.non_ibd_emission(1_000);
    let medium = Population::EUR.non_ibd_emission(10_000);
    let large = Population::EUR.non_ibd_emission(100_000);
    assert!(small.std > medium.std);
    assert!(medium.std > large.std);
    // Mean should be the same regardless of window size
    assert!((small.mean - medium.mean).abs() < 1e-12);
    assert!((medium.mean - large.mean).abs() < 1e-12);
}

/// Verify std follows the formula: sqrt(pi / window_size * 3).
#[test]
fn non_ibd_emission_std_formula_check() {
    let ws = 5_000u64;
    let e = Population::EUR.non_ibd_emission(ws);
    let pi = Population::EUR.diversity();
    let expected_std = (pi / ws as f64 * 3.0).sqrt();
    assert!((e.std - expected_std).abs() < 1e-15);
}

// ============================================================================
// Population::from_str — exhaustive variant coverage
// ============================================================================

#[test]
fn population_from_str_case_insensitive() {
    assert_eq!(Population::from_str("afr"), Some(Population::AFR));
    assert_eq!(Population::from_str("Afr"), Some(Population::AFR));
    assert_eq!(Population::from_str("AFR"), Some(Population::AFR));
    assert_eq!(Population::from_str("eur"), Some(Population::EUR));
    assert_eq!(Population::from_str("eas"), Some(Population::EAS));
    assert_eq!(Population::from_str("csa"), Some(Population::CSA));
    assert_eq!(Population::from_str("amr"), Some(Population::AMR));
}

#[test]
fn population_from_str_interpop_aliases() {
    assert_eq!(Population::from_str("interpop"), Some(Population::InterPop));
    assert_eq!(Population::from_str("INTER"), Some(Population::InterPop));
}

#[test]
fn population_from_str_generic_aliases() {
    assert_eq!(Population::from_str("generic"), Some(Population::Generic));
    assert_eq!(Population::from_str("UNKNOWN"), Some(Population::Generic));
}

#[test]
fn population_from_str_invalid() {
    assert_eq!(Population::from_str(""), None);
    assert_eq!(Population::from_str("XYZ"), None);
    assert_eq!(Population::from_str("european"), None);
}

// ============================================================================
// GeneticMap::from_file — error path tests
// ============================================================================

#[test]
fn genetic_map_from_file_nonexistent() {
    let result = GeneticMap::from_file("/nonexistent/path/genetic_map.txt", "chr1");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Failed to open"), "Error should mention open failure: {}", err);
}

#[test]
fn genetic_map_from_file_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.map");
    std::fs::write(&path, "").unwrap();
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("No genetic map entries"), "Expected 'no entries' error: {}", err);
}

#[test]
fn genetic_map_from_file_only_comments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("comments_only.map");
    std::fs::write(&path, "# comment 1\n# comment 2\n# comment 3\n").unwrap();
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("No genetic map entries"), "Expected 'no entries' error: {}", err);
}

#[test]
fn genetic_map_from_file_wrong_chromosome() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chr2_only.map");
    std::fs::write(&path, "chr2\t1000\t1.0\t0.001\nchr2\t2000\t1.0\t0.002\n").unwrap();
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err(), "Should fail when no entries match target chromosome");
}

#[test]
fn genetic_map_from_file_valid_4col() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid.map");
    std::fs::write(
        &path,
        "chr10\t1000\t1.0\t0.001\nchr10\t5000\t1.0\t0.005\nchr10\t10000\t1.0\t0.010\n",
    )
    .unwrap();
    let result = GeneticMap::from_file(&path, "chr10");
    assert!(result.is_ok());
    let map = result.unwrap();
    assert_eq!(map.len(), 3);
}

#[test]
fn genetic_map_from_file_valid_3col() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid3.map");
    std::fs::write(&path, "1000\t1.0\t0.001\n5000\t1.0\t0.005\n").unwrap();
    let result = GeneticMap::from_file(&path, "anything"); // 3-col ignores chrom
    assert!(result.is_ok());
    let map = result.unwrap();
    assert_eq!(map.len(), 2);
}

#[test]
fn genetic_map_from_file_chr_prefix_normalization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prefix.map");
    std::fs::write(&path, "chr10\t1000\t1.0\t0.001\nchr10\t2000\t1.0\t0.002\n").unwrap();
    // Query with "10" (no prefix) should still match "chr10" in file
    let result = GeneticMap::from_file(&path, "10");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

#[test]
fn genetic_map_from_file_bad_cm_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_cm.map");
    std::fs::write(&path, "chr1\t1000\t1.0\tNOTANUMBER\n").unwrap();
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Invalid cM"), "Expected cM parse error: {}", err);
}

#[test]
fn genetic_map_from_file_mixed_chroms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.map");
    let content = "chr10\t1000\t1.0\t0.001\n\
                   chr11\t2000\t1.0\t0.002\n\
                   chr10\t3000\t1.0\t0.003\n\
                   chr12\t4000\t1.0\t0.004\n";
    std::fs::write(&path, content).unwrap();
    let result = GeneticMap::from_file(&path, "chr10");
    assert!(result.is_ok());
    let map = result.unwrap();
    assert_eq!(map.len(), 2, "Should only include chr10 entries");
}

#[test]
fn genetic_map_from_file_skips_blank_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blanks.map");
    let content = "\n\nchr1\t1000\t1.0\t0.001\n\nchr1\t2000\t1.0\t0.002\n\n";
    std::fs::write(&path, content).unwrap();
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 2);
}

// ============================================================================
// parse_hapibd_file — file I/O tests
// ============================================================================

#[test]
fn parse_hapibd_file_nonexistent() {
    let result = parse_hapibd_file("/nonexistent/hapibd.ibd");
    assert!(result.is_err());
}

#[test]
fn parse_hapibd_file_valid() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.ibd");
    let content = "HG00733\t1\tNA12878\t1\tchr20\t1000000\t5000000\t12.5\n\
                   HG00733\t2\tHG00514\t1\tchr20\t3000000\t7000000\t15.7\n";
    std::fs::write(&path, content).unwrap();
    let segs = parse_hapibd_file(&path).unwrap();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].sample1, "HG00733");
    assert_eq!(segs[0].sample2, "NA12878");
    assert!((segs[0].lod - 12.5).abs() < 1e-9);
}

#[test]
fn parse_hapibd_file_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.ibd");
    std::fs::write(&path, "").unwrap();
    let segs = parse_hapibd_file(&path).unwrap();
    assert!(segs.is_empty());
}

#[test]
fn parse_hapibd_file_with_comments_and_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("comments.ibd");
    let content = "# header comment\n\n\
                   HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n\
                   # inline comment\n\
                   \n\
                   HG003\t1\tHG004\t2\tchr1\t300\t400\t6.0\n";
    std::fs::write(&path, content).unwrap();
    let segs = parse_hapibd_file(&path).unwrap();
    assert_eq!(segs.len(), 2);
}

#[test]
fn parse_hapibd_file_skips_malformed_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("malformed.ibd");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0").unwrap();
    writeln!(f, "too_few_fields").unwrap();
    writeln!(f, "HG001\t1\tHG002\tBAD\tchr1\t100\t200\t5.0").unwrap(); // bad hap2
    writeln!(f, "HG003\t1\tHG004\t2\tchr1\t300\t400\t6.0").unwrap();
    let segs = parse_hapibd_file(&path).unwrap();
    assert_eq!(segs.len(), 2, "Should skip malformed lines");
}

// ============================================================================
// GeneticMap::from_file with baum_welch out-of-range positions
// ============================================================================

#[test]
fn genetic_map_interpolate_outside_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("narrow.map");
    // Map only covers 50000-100000
    std::fs::write(&path, "chr1\t50000\t1.0\t0.05\nchr1\t100000\t1.0\t0.10\n").unwrap();
    let map = GeneticMap::from_file(&path, "chr1").unwrap();
    // Interpolate before map range
    let cm_before = map.interpolate_cm(0);
    assert!(cm_before.is_finite());
    // Interpolate after map range
    let cm_after = map.interpolate_cm(200_000);
    assert!(cm_after.is_finite());
    // Interpolate within range
    let cm_mid = map.interpolate_cm(75_000);
    assert!(cm_mid.is_finite());
    assert!(cm_mid > 0.05 && cm_mid < 0.10);
}
