//! Tests for load_population_samples(), load_populations_from_dir(),
//! and segment_ancestry_lod() edge cases.
//!
//! These functions require file I/O, so we use tempfile/tempdir for isolation.

use hprc_ancestry_cli::ancestry::{load_population_samples, load_populations_from_dir, segment_ancestry_lod};
use hprc_ancestry_cli::hmm::{AncestralPopulation, AncestryHmmParams, AncestryObservation};
use std::collections::HashMap;

// =============================================================================
// load_population_samples tests
// =============================================================================

/// Loading a valid population file returns the sample IDs.
#[test]
fn test_load_population_samples_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("AFR.txt");
    std::fs::write(&path, "HG01884\nHG01885\nHG01886\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples, vec!["HG01884", "HG01885", "HG01886"]);
}

/// Comments (lines starting with #) and blank lines are skipped.
#[test]
fn test_load_population_samples_comments_and_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("EUR.txt");
    std::fs::write(&path, "# European samples\nHG00096\n\n# Another comment\nHG00097\n\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples, vec!["HG00096", "HG00097"]);
}

/// Empty file (no samples) returns an empty vec.
#[test]
fn test_load_population_samples_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.txt");
    std::fs::write(&path, "").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert!(samples.is_empty());
}

/// File with only comments returns empty vec.
#[test]
fn test_load_population_samples_only_comments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("comments.txt");
    std::fs::write(&path, "# just a comment\n# another comment\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert!(samples.is_empty());
}

/// Whitespace is trimmed from sample IDs.
#[test]
fn test_load_population_samples_whitespace_trimming() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spaced.txt");
    std::fs::write(&path, "  HG00096  \n\tHG00097\t\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples, vec!["HG00096", "HG00097"]);
}

/// Nonexistent file returns an error.
#[test]
fn test_load_population_samples_nonexistent() {
    let result = load_population_samples(std::path::Path::new("/tmp/nonexistent_population_file_xyz.txt"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to open"));
}

// =============================================================================
// load_populations_from_dir tests
// =============================================================================

/// Loading from a directory with multiple .txt files produces sorted populations.
#[test]
fn test_load_populations_from_dir_basic() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("EUR.txt"), "HG00096\nHG00097\n").unwrap();
    std::fs::write(dir.path().join("AFR.txt"), "HG01884\nHG01885\n").unwrap();
    std::fs::write(dir.path().join("AMR.txt"), "HG00733\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 3);
    // Should be sorted alphabetically
    assert_eq!(pops[0].name, "AFR");
    assert_eq!(pops[1].name, "AMR");
    assert_eq!(pops[2].name, "EUR");
    assert_eq!(pops[0].haplotypes, vec!["HG01884", "HG01885"]);
    assert_eq!(pops[1].haplotypes, vec!["HG00733"]);
    assert_eq!(pops[2].haplotypes, vec!["HG00096", "HG00097"]);
}

/// Non-.txt files in the directory are ignored.
#[test]
fn test_load_populations_from_dir_ignores_non_txt() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("EUR.txt"), "HG00096\n").unwrap();
    std::fs::write(dir.path().join("README.md"), "This is a readme\n").unwrap();
    std::fs::write(dir.path().join("data.csv"), "sample,pop\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "EUR");
}

/// Empty .txt files (no samples) are skipped (not included as populations).
#[test]
fn test_load_populations_from_dir_skips_empty_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("EUR.txt"), "HG00096\n").unwrap();
    std::fs::write(dir.path().join("EMPTY.txt"), "").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "EUR");
}

/// Empty directory returns empty populations list.
#[test]
fn test_load_populations_from_dir_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert!(pops.is_empty());
}

/// Nonexistent directory returns an error.
#[test]
fn test_load_populations_from_dir_nonexistent() {
    let result = load_populations_from_dir(std::path::Path::new("/tmp/nonexistent_dir_xyz_12345"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read directory"));
}

/// Files with only comments produce empty haplotype lists and are skipped.
#[test]
fn test_load_populations_from_dir_comment_only_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("COMMENTS.txt"), "# just a comment\n# another\n").unwrap();
    std::fs::write(dir.path().join("EUR.txt"), "HG00096\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "EUR");
}

/// Population name is derived from filename without extension.
#[test]
fn test_load_populations_from_dir_filename_as_pop_name() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("my_custom_pop.txt"), "sample1\nsample2\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "my_custom_pop");
}

// =============================================================================
// segment_ancestry_lod edge cases
// =============================================================================

fn make_pops_2way() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["pop_b#1".to_string()],
        },
    ]
}

fn make_obs(start: u64, end: u64, sim_a: f64, sim_b: f64) -> AncestryObservation {
    let mut sims = HashMap::new();
    sims.insert("pop_a#1".to_string(), sim_a);
    sims.insert("pop_b#1".to_string(), sim_b);
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "test".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

/// segment_ancestry_lod over a single window returns the same as per-window LOD.
#[test]
fn test_segment_ancestry_lod_single_window() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![make_obs(0, 5000, 0.95, 0.80)];

    let lod = segment_ancestry_lod(&obs, &params, 0, 0, 0);
    // For correct assignment (pop_a has higher sim), LOD should be positive
    assert!(lod > 0.0, "Single window, correct assignment: LOD={} should be > 0", lod);
}

/// segment_ancestry_lod sums across multiple windows.
#[test]
fn test_segment_ancestry_lod_sum_property() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![
        make_obs(0, 5000, 0.95, 0.80),
        make_obs(5000, 10000, 0.90, 0.85),
        make_obs(10000, 15000, 0.98, 0.70),
    ];

    let lod_full = segment_ancestry_lod(&obs, &params, 0, 0, 2);
    let lod_0 = segment_ancestry_lod(&obs, &params, 0, 0, 0);
    let lod_1 = segment_ancestry_lod(&obs, &params, 0, 1, 1);
    let lod_2 = segment_ancestry_lod(&obs, &params, 0, 2, 2);

    let sum = lod_0 + lod_1 + lod_2;
    assert!(
        (lod_full - sum).abs() < 1e-10,
        "LOD over full range ({}) should equal sum of individual windows ({})",
        lod_full,
        sum
    );
}

/// segment_ancestry_lod for wrong assignment should be negative.
#[test]
fn test_segment_ancestry_lod_wrong_assignment_negative() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    // All windows favor pop_a (state 0)
    let obs = vec![
        make_obs(0, 5000, 0.95, 0.80),
        make_obs(5000, 10000, 0.93, 0.82),
    ];

    // Assign to pop_b (state 1) — wrong
    let lod = segment_ancestry_lod(&obs, &params, 1, 0, 1);
    assert!(lod < 0.0, "Wrong assignment should give negative LOD, got {}", lod);
}

/// segment_ancestry_lod with equal similarities across windows: LOD ≈ 0.
#[test]
fn test_segment_ancestry_lod_equal_sims() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![
        make_obs(0, 5000, 0.90, 0.90),
        make_obs(5000, 10000, 0.90, 0.90),
    ];

    let lod = segment_ancestry_lod(&obs, &params, 0, 0, 1);
    assert!(
        lod.abs() < 0.1,
        "Equal similarities should give LOD near 0, got {}",
        lod
    );
}

/// segment_ancestry_lod for a sub-range (start_idx > 0).
#[test]
fn test_segment_ancestry_lod_subrange() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![
        make_obs(0, 5000, 0.50, 0.95),     // favors pop_b
        make_obs(5000, 10000, 0.95, 0.50),  // favors pop_a
        make_obs(10000, 15000, 0.95, 0.50), // favors pop_a
    ];

    // LOD for state 0 on windows 1-2 only (the pop_a windows)
    let lod = segment_ancestry_lod(&obs, &params, 0, 1, 2);
    assert!(lod > 0.0, "Sub-range favoring pop_a should have positive LOD for state 0, got {}", lod);

    // LOD for state 0 on window 0 only (the pop_b window)
    let lod_wrong = segment_ancestry_lod(&obs, &params, 0, 0, 0);
    assert!(lod_wrong < 0.0, "Window 0 favors pop_b, so LOD for state 0 should be negative, got {}", lod_wrong);
}

/// segment_ancestry_lod monotonicity: more confirming windows increase LOD.
#[test]
fn test_segment_ancestry_lod_monotonic_with_more_windows() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![
        make_obs(0, 5000, 0.95, 0.80),
        make_obs(5000, 10000, 0.95, 0.80),
        make_obs(10000, 15000, 0.95, 0.80),
    ];

    let lod_1 = segment_ancestry_lod(&obs, &params, 0, 0, 0);
    let lod_2 = segment_ancestry_lod(&obs, &params, 0, 0, 1);
    let lod_3 = segment_ancestry_lod(&obs, &params, 0, 0, 2);

    assert!(lod_2 > lod_1, "More confirming windows should increase LOD: {} > {}", lod_2, lod_1);
    assert!(lod_3 > lod_2, "Even more confirming windows should increase LOD: {} > {}", lod_3, lod_2);
}
