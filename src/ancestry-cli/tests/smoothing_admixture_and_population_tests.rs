//! Tests for smooth_states, count_smoothing_changes, estimate_admixture_proportions,
//! load_population_samples, load_populations_from_dir, glossophaga_populations,
//! and filter_segments_by_min_lod.

use hprc_ancestry_cli::ancestry::{
    count_smoothing_changes, estimate_admixture_proportions, filter_segments_by_min_lod,
    glossophaga_populations, load_population_samples, load_populations_from_dir, smooth_states,
    AncestrySegment,
};
use std::path::Path;

// ── smooth_states ───────────────────────────────────────────────────────

#[test]
fn smooth_empty() {
    assert_eq!(smooth_states(&[], 3), Vec::<usize>::new());
}

#[test]
fn smooth_single() {
    assert_eq!(smooth_states(&[0], 3), vec![0]);
}

#[test]
fn smooth_two_elements_not_smoothed() {
    assert_eq!(smooth_states(&[0, 1], 5), vec![0, 1]);
}

#[test]
fn smooth_min_run_1_is_noop() {
    let states = vec![0, 1, 0, 1, 0];
    assert_eq!(smooth_states(&states, 1), states);
}

#[test]
fn smooth_min_run_0_is_noop() {
    let states = vec![0, 1, 0, 1, 0, 1, 0];
    assert_eq!(smooth_states(&states, 0), states);
}

#[test]
fn smooth_short_run_surrounded_by_same_state() {
    // Short run of 1 surrounded by 0s → should be smoothed to 0
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn smooth_short_run_different_neighbors_not_smoothed() {
    // Short run of 1 between 0 and 2 → NOT smoothed (neighbors differ)
    let states = vec![0, 0, 1, 2, 2];
    let smoothed = smooth_states(&states, 3);
    // prev_state=0, next_state=2 → different → no smoothing
    assert_eq!(smoothed, vec![0, 0, 1, 2, 2]);
}

#[test]
fn smooth_keeps_long_runs() {
    // Run of length 5 with min_run=3 → should NOT be smoothed
    let states = vec![0, 0, 1, 1, 1, 1, 1, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states);
}

#[test]
fn smooth_multiple_short_runs() {
    let states = vec![0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    // State 1 (length 1) between 0s → smoothed
    // State 2 (length 1) between 0s → smoothed
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn smooth_at_start_not_smoothed() {
    // Short run at the very start → no previous state → NOT smoothed
    let states = vec![1, 0, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    // i=0 → run_len=1 < min_run=3, but i==0 → condition i>0 fails
    assert_eq!(smoothed, vec![1, 0, 0, 0, 0]);
}

#[test]
fn smooth_at_end_not_smoothed() {
    // Short run at the very end → no next state → NOT smoothed
    let states = vec![0, 0, 0, 0, 1];
    let smoothed = smooth_states(&states, 3);
    // run_end == smoothed.len() → condition run_end < smoothed.len() fails
    assert_eq!(smoothed, vec![0, 0, 0, 0, 1]);
}

#[test]
fn smooth_exactly_min_run_not_smoothed() {
    // Run of exactly min_run → NOT smoothed (run_len < min_run is false)
    let states = vec![0, 0, 0, 1, 1, 1, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states);
}

#[test]
fn smooth_homogeneous_is_noop() {
    let states = vec![2; 20];
    let smoothed = smooth_states(&states, 5);
    assert_eq!(smoothed, states);
}

#[test]
fn smooth_alternating_states() {
    // Rapid alternation: 0,1,0,1,0 with min_run=2
    // Each run has length 1 < min_run
    let states = vec![0, 0, 1, 0, 0, 1, 0, 0];
    let smoothed = smooth_states(&states, 2);
    // Single 1s between 0s → smoothed to 0
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn smooth_preserves_length() {
    let states = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let smoothed = smooth_states(&states, 5);
    assert_eq!(smoothed.len(), states.len());
}

// ── count_smoothing_changes ─────────────────────────────────────────────

#[test]
fn count_changes_empty() {
    assert_eq!(count_smoothing_changes(&[], &[]), 0);
}

#[test]
fn count_changes_identical() {
    let v = vec![0, 1, 2, 0, 1];
    assert_eq!(count_smoothing_changes(&v, &v), 0);
}

#[test]
fn count_changes_all_different() {
    let original = vec![0, 1, 2, 3, 4];
    let smoothed = vec![5, 6, 7, 8, 9];
    assert_eq!(count_smoothing_changes(&original, &smoothed), 5);
}

#[test]
fn count_changes_some_different() {
    let original = vec![0, 1, 0, 1, 0];
    let smoothed = vec![0, 0, 0, 0, 0];
    assert_eq!(count_smoothing_changes(&original, &smoothed), 2);
}

#[test]
fn count_changes_consistent_with_smooth() {
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    let n_changes = count_smoothing_changes(&states, &smoothed);
    assert_eq!(n_changes, 1); // The single 1 was changed to 0
}

// ── estimate_admixture_proportions ──────────────────────────────────────

fn make_segment(
    ancestry_name: &str,
    ancestry_idx: usize,
    start: u64,
    end: u64,
) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "test_sample".to_string(),
        ancestry_idx,
        ancestry_name: ancestry_name.to_string(),
        n_windows: ((end - start) / 10000) as usize,
        mean_similarity: 0.998,
        mean_posterior: Some(0.95),
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

#[test]
fn admixture_empty_segments() {
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&[], "sample1", &pop_names);
    assert_eq!(result.sample, "sample1");
    assert_eq!(result.total_length_bp, 0);
    assert_eq!(result.n_switches, 0);
    for name in &pop_names {
        assert_eq!(*result.proportions.get(name).unwrap(), 0.0);
    }
}

#[test]
fn admixture_single_population() {
    let segments = vec![make_segment("EUR", 0, 0, 100_000)];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);

    assert_eq!(result.total_length_bp, 100_000);
    assert!((result.proportions["EUR"] - 1.0).abs() < 1e-10);
    assert!((result.proportions["AFR"] - 0.0).abs() < 1e-10);
    assert_eq!(result.n_switches, 0);
}

#[test]
fn admixture_two_populations_equal_tracts() {
    let segments = vec![
        make_segment("AFR", 0, 0, 50_000),
        make_segment("EUR", 1, 50_000, 100_000),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);

    assert_eq!(result.total_length_bp, 100_000);
    assert!((result.proportions["AFR"] - 0.5).abs() < 1e-10);
    assert!((result.proportions["EUR"] - 0.5).abs() < 1e-10);
    assert_eq!(result.n_switches, 1);
}

#[test]
fn admixture_proportions_sum_to_one() {
    let segments = vec![
        make_segment("AFR", 0, 0, 30_000),
        make_segment("EUR", 1, 30_000, 70_000),
        make_segment("EAS", 2, 70_000, 100_000),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string(), "EAS".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);

    let total: f64 = result.proportions.values().sum();
    assert!((total - 1.0).abs() < 1e-10, "Proportions should sum to 1.0");
}

#[test]
fn admixture_multiple_switches() {
    let segments = vec![
        make_segment("AFR", 0, 0, 10_000),
        make_segment("EUR", 1, 10_000, 20_000),
        make_segment("AFR", 0, 20_000, 30_000),
        make_segment("EUR", 1, 30_000, 40_000),
        make_segment("AFR", 0, 40_000, 50_000),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);
    assert_eq!(result.n_switches, 4); // AFR→EUR, EUR→AFR, AFR→EUR, EUR→AFR
}

#[test]
fn admixture_no_switches_same_ancestry() {
    let segments = vec![
        make_segment("EUR", 1, 0, 50_000),
        make_segment("EUR", 1, 50_000, 100_000),
    ];
    let pop_names = vec!["EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);
    assert_eq!(result.n_switches, 0);
}

#[test]
fn admixture_pop_not_in_names_still_counted() {
    // Segment with ancestry name not in pop_names list
    let segments = vec![
        make_segment("AFR", 0, 0, 50_000),
        make_segment("UNKNOWN", 3, 50_000, 100_000),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);

    // UNKNOWN gets added via or_insert
    assert_eq!(result.total_length_bp, 100_000);
    assert!((result.proportions["AFR"] - 0.5).abs() < 1e-10);
    assert!(result.tract_lengths_bp.contains_key("UNKNOWN"));
}

#[test]
fn admixture_mean_tract_length() {
    let segments = vec![
        make_segment("AFR", 0, 0, 20_000),
        make_segment("EUR", 1, 20_000, 80_000),
        make_segment("AFR", 0, 80_000, 100_000),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "sample1", &pop_names);

    // 3 segments with lengths 20k, 60k, 20k → mean = 100k / 3 = 33333
    // n_switches = 2 (AFR→EUR, EUR→AFR)
    assert_eq!(result.n_switches, 2);
    assert!(result.mean_tract_length_bp > 0.0);
}

// ── glossophaga_populations ─────────────────────────────────────────────

#[test]
fn glossophaga_returns_populations() {
    let pops = glossophaga_populations();
    assert!(!pops.is_empty(), "Should return at least one population");
}

#[test]
fn glossophaga_populations_have_names_and_haplotypes() {
    let pops = glossophaga_populations();
    for pop in &pops {
        assert!(!pop.name.is_empty(), "Population name should not be empty");
        assert!(
            !pop.haplotypes.is_empty(),
            "Population {} should have haplotypes",
            pop.name
        );
    }
}

#[test]
fn glossophaga_haplotypes_are_unique_within_population() {
    let pops = glossophaga_populations();
    for pop in &pops {
        let mut seen = std::collections::HashSet::new();
        for hap in &pop.haplotypes {
            assert!(
                seen.insert(hap.clone()),
                "Duplicate haplotype {} in population {}",
                hap,
                pop.name
            );
        }
    }
}

// ── load_population_samples ─────────────────────────────────────────────

#[test]
fn load_pop_samples_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop.txt");
    std::fs::write(&path, "HG01175\nNA19239\nHG02257\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0], "HG01175");
}

#[test]
fn load_pop_samples_filters_comments_and_blanks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop.txt");
    std::fs::write(&path, "# header\nHG01175\n\n# comment\nNA19239\n  \n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples.len(), 2);
}

#[test]
fn load_pop_samples_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop.txt");
    std::fs::write(&path, "").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert!(samples.is_empty());
}

#[test]
fn load_pop_samples_nonexistent_file() {
    let result = load_population_samples(Path::new("/nonexistent/pop.txt"));
    assert!(result.is_err());
}

#[test]
fn load_pop_samples_trims_whitespace() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pop.txt");
    std::fs::write(&path, "  HG01175  \n\tNA19239\t\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0], "HG01175");
    assert_eq!(samples[1], "NA19239");
}

// ── load_populations_from_dir ───────────────────────────────────────────

#[test]
fn load_pops_from_dir_basic() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AFR.txt"), "HG01175\nNA19239\n").unwrap();
    std::fs::write(dir.path().join("EUR.txt"), "HG02257\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 2);
    // Sorted by name
    assert_eq!(pops[0].name, "AFR");
    assert_eq!(pops[1].name, "EUR");
    assert_eq!(pops[0].haplotypes.len(), 2);
    assert_eq!(pops[1].haplotypes.len(), 1);
}

#[test]
fn load_pops_from_dir_ignores_non_txt() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AFR.txt"), "HG01175\n").unwrap();
    std::fs::write(dir.path().join("README.md"), "not a population\n").unwrap();
    std::fs::write(dir.path().join("data.csv"), "also not\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "AFR");
}

#[test]
fn load_pops_from_dir_skips_empty_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AFR.txt"), "HG01175\n").unwrap();
    std::fs::write(dir.path().join("EMPTY.txt"), "").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "AFR");
}

#[test]
fn load_pops_from_dir_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert!(pops.is_empty());
}

#[test]
fn load_pops_from_dir_nonexistent() {
    let result = load_populations_from_dir(Path::new("/nonexistent/dir"));
    assert!(result.is_err());
}

#[test]
fn load_pops_from_dir_sorted_deterministically() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("EAS.txt"), "H1\n").unwrap();
    std::fs::write(dir.path().join("AFR.txt"), "H2\n").unwrap();
    std::fs::write(dir.path().join("CSA.txt"), "H3\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops[0].name, "AFR");
    assert_eq!(pops[1].name, "CSA");
    assert_eq!(pops[2].name, "EAS");
}

// ── filter_segments_by_min_lod ──────────────────────────────────────────

#[test]
fn filter_segments_empty() {
    let filtered = filter_segments_by_min_lod(vec![], 5.0);
    assert!(filtered.is_empty());
}

#[test]
fn filter_segments_all_pass() {
    let segments = vec![
        make_segment("AFR", 0, 0, 50_000),
        make_segment("EUR", 1, 50_000, 100_000),
    ];
    let filtered = filter_segments_by_min_lod(segments, 0.0);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn filter_segments_all_removed() {
    let segments = vec![
        make_segment("AFR", 0, 0, 50_000),
        make_segment("EUR", 1, 50_000, 100_000),
    ];
    // Default lod_score is 5.0, filter at 100.0
    let filtered = filter_segments_by_min_lod(segments, 100.0);
    assert!(filtered.is_empty());
}

#[test]
fn filter_segments_partial() {
    let mut seg_high = make_segment("AFR", 0, 0, 50_000);
    seg_high.lod_score = 10.0;
    let mut seg_low = make_segment("EUR", 1, 50_000, 100_000);
    seg_low.lod_score = 1.0;

    let segments = vec![seg_high, seg_low];
    let filtered = filter_segments_by_min_lod(segments, 5.0);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].ancestry_name, "AFR");
}

#[test]
fn filter_segments_exact_threshold() {
    let mut seg = make_segment("AFR", 0, 0, 50_000);
    seg.lod_score = 5.0;
    let filtered = filter_segments_by_min_lod(vec![seg], 5.0);
    // >= threshold should pass
    assert_eq!(filtered.len(), 1);
}

#[test]
fn filter_segments_negative_lod() {
    let mut seg = make_segment("AFR", 0, 0, 50_000);
    seg.lod_score = -2.0;
    let filtered = filter_segments_by_min_lod(vec![seg], 0.0);
    assert!(filtered.is_empty());
}

#[test]
fn filter_segments_negative_threshold() {
    let mut seg = make_segment("AFR", 0, 0, 50_000);
    seg.lod_score = -1.0;
    let filtered = filter_segments_by_min_lod(vec![seg], -2.0);
    assert_eq!(filtered.len(), 1); // -1.0 >= -2.0
}
