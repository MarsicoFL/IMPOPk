//! Tests for transition functions, coverage_ratio, segment_quality_score,
//! baum_welch early return, and estimate_emissions_logit edge cases.
//!
//! These cover untested edge cases identified in cycle 15 gap analysis.

use hprc_ibd::hmm::{
    coverage_ratio, distance_dependent_log_transition, recombination_aware_log_transition,
    segment_quality_score, GeneticMap, HmmParams, IbdSegmentWithPosterior, Population,
};

// ===========================================================================
// coverage_ratio tests
// ===========================================================================

#[test]
fn test_coverage_ratio_both_zero() {
    assert_eq!(coverage_ratio(0, 0), 0.0);
}

#[test]
fn test_coverage_ratio_equal() {
    assert_eq!(coverage_ratio(100, 100), 1.0);
}

#[test]
fn test_coverage_ratio_one_zero() {
    assert_eq!(coverage_ratio(100, 0), 0.0);
    assert_eq!(coverage_ratio(0, 100), 0.0);
}

#[test]
fn test_coverage_ratio_asymmetric() {
    let r = coverage_ratio(50, 100);
    assert!((r - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_symmetric() {
    // coverage_ratio(a, b) == coverage_ratio(b, a)
    assert_eq!(coverage_ratio(30, 70), coverage_ratio(70, 30));
}

#[test]
fn test_coverage_ratio_large_values() {
    let r = coverage_ratio(1_000_000_000, 2_000_000_000);
    assert!((r - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_one_each() {
    assert_eq!(coverage_ratio(1, 1), 1.0);
}

// ===========================================================================
// segment_quality_score tests
// ===========================================================================

fn make_segment(
    mean_posterior: f64,
    min_posterior: f64,
    max_posterior: f64,
    lod_score: f64,
    n_windows: usize,
) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: n_windows.saturating_sub(1),
        n_windows,
        mean_posterior,
        min_posterior,
        max_posterior,
        lod_score,
    }
}

#[test]
fn test_quality_score_perfect_segment() {
    let seg = make_segment(1.0, 1.0, 1.0, 50.0, 30);
    let score = segment_quality_score(&seg);
    // posterior_score = 40, consistency_score = 20, lod_score = 30, length_score = 10
    assert!((score - 100.0).abs() < 1e-10);
}

#[test]
fn test_quality_score_zero_posterior() {
    let seg = make_segment(0.0, 0.0, 0.0, 0.0, 0);
    let score = segment_quality_score(&seg);
    assert!((score - 0.0).abs() < 1e-10);
}

#[test]
fn test_quality_score_half_posterior() {
    let seg = make_segment(0.5, 0.5, 0.5, 10.0, 10);
    let score = segment_quality_score(&seg);
    // posterior: 0.5*40 = 20
    // consistency: (0.5/0.5) = 1.0 => 20
    // lod_per_window: 10/10 = 1.0 => 30
    // length: (10/20)*10 = 5
    assert!((score - 75.0).abs() < 1e-10);
}

#[test]
fn test_quality_score_low_consistency() {
    // min_posterior much lower than mean
    let seg = make_segment(0.8, 0.1, 1.0, 20.0, 20);
    let score = segment_quality_score(&seg);
    // posterior: 0.8*40 = 32
    // consistency: (0.1/0.8)*20 = 2.5
    // lod_per_window: 20/20 = 1.0 => 30
    // length: (20/20)*10 = 10
    let expected = 32.0 + 2.5 + 30.0 + 10.0;
    assert!((score - expected).abs() < 1e-10);
}

#[test]
fn test_quality_score_short_segment() {
    let seg = make_segment(0.9, 0.9, 0.9, 5.0, 2);
    let score = segment_quality_score(&seg);
    // posterior: 0.9*40 = 36
    // consistency: (0.9/0.9)*20 = 20
    // lod_per_window: 5/2 = 2.5, clamped to 1.0 => 30
    // length: (2/20)*10 = 1
    assert!((score - 87.0).abs() < 1e-10);
}

#[test]
fn test_quality_score_n_windows_zero() {
    // Edge case: n_windows == 0 => lod_per_window = 0, length = 0
    let seg = make_segment(0.5, 0.5, 0.5, 0.0, 0);
    let score = segment_quality_score(&seg);
    // posterior: 0.5*40 = 20, consistency: 20, lod: 0, length: 0
    assert!((score - 40.0).abs() < 1e-10);
}

#[test]
fn test_quality_score_clamps_to_100() {
    // Even with extreme values, max is 100
    let seg = make_segment(2.0, 2.0, 2.0, 1000.0, 1000);
    let score = segment_quality_score(&seg);
    assert!(score <= 100.0);
}

// ===========================================================================
// distance_dependent_log_transition tests
// ===========================================================================

#[test]
fn test_distance_dependent_zero_distance() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 0, 5000);
    // Should return default log transitions
    let expected_stay = params.transition[0][0].ln();
    let expected_switch = params.transition[0][1].ln();
    assert!((trans[0][0] - expected_stay).abs() < 1e-10);
    assert!((trans[0][1] - expected_switch).abs() < 1e-10);
}

#[test]
fn test_distance_dependent_zero_window_size() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 5000, 0);
    // Should return default log transitions
    let expected_stay = params.transition[0][0].ln();
    assert!((trans[0][0] - expected_stay).abs() < 1e-10);
}

#[test]
fn test_distance_dependent_nominal_distance() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 5000, 5000);
    // With distance == window_size, scale == 1, should be similar to base transitions
    // But continuous-time approximation may differ slightly
    assert!(trans[0][0] < 0.0); // log(stay) < 0
    assert!(trans[0][1] < 0.0); // log(switch) < 0
    assert!(trans[0][0] > trans[0][1]); // stay more likely than switch
}

#[test]
fn test_distance_dependent_large_distance() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans_small = distance_dependent_log_transition(&params, 5000, 5000);
    let trans_large = distance_dependent_log_transition(&params, 50000, 5000);
    // Larger distance → higher switch probability → higher log(switch)
    assert!(trans_large[0][1] > trans_small[0][1]);
}

#[test]
fn test_distance_dependent_rows_sum_to_one() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 10000, 5000);
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-8, "row sums to {}", sum);
    }
}

#[test]
fn test_distance_dependent_symmetry() {
    // Both rows should sum to 1 in probability space
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 7500, 5000);
    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    let row1_sum = trans[1][0].exp() + trans[1][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-8);
    assert!((row1_sum - 1.0).abs() < 1e-8);
}

// ===========================================================================
// recombination_aware_log_transition tests
// ===========================================================================

fn make_genetic_map() -> GeneticMap {
    // Uniform ~1 cM/Mb genetic map from 0-10Mb
    GeneticMap::new(vec![
        (0, 0.0),
        (10_000_000, 10.0),
    ])
}

#[test]
fn test_recomb_zero_window_size() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = make_genetic_map();
    let trans = recombination_aware_log_transition(&params, 1_000_000, 2_000_000, &gmap, 0);
    // Should fall back to default transitions
    let expected = params.transition[0][0].ln();
    assert!((trans[0][0] - expected).abs() < 1e-10);
}

#[test]
fn test_recomb_same_position() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = make_genetic_map();
    let trans = recombination_aware_log_transition(&params, 1_000_000, 1_000_000, &gmap, 5000);
    // Same position → fall back to default
    let expected = params.transition[0][0].ln();
    assert!((trans[0][0] - expected).abs() < 1e-10);
}

#[test]
fn test_recomb_rows_sum_to_one() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = make_genetic_map();
    let trans = recombination_aware_log_transition(&params, 1_000_000, 2_000_000, &gmap, 5000);
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-8, "row sums to {}", sum);
    }
}

#[test]
fn test_recomb_larger_distance_more_switching() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = make_genetic_map();
    let trans_close = recombination_aware_log_transition(&params, 1_000_000, 1_005_000, &gmap, 5000);
    let trans_far = recombination_aware_log_transition(&params, 1_000_000, 2_000_000, &gmap, 5000);
    // Farther positions → higher switch probability
    assert!(trans_far[0][1] > trans_close[0][1]);
}

#[test]
fn test_recomb_all_values_finite() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = make_genetic_map();
    let trans = recombination_aware_log_transition(&params, 500_000, 5_000_000, &gmap, 5000);
    for row in &trans {
        for &v in row {
            assert!(v.is_finite(), "got non-finite: {}", v);
        }
    }
}

// ===========================================================================
// baum_welch early return tests
// ===========================================================================

#[test]
fn test_baum_welch_too_few_observations() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let original_emission = params.emission.clone();
    let original_transition = params.transition;

    // Only 5 observations (< 10)
    let obs = vec![0.999, 0.998, 0.997, 0.996, 0.995];
    params.baum_welch(&obs, 20, 1e-6, None, 5000);

    // Parameters should be unchanged
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
    assert_eq!(params.emission[1].mean, original_emission[1].mean);
    assert_eq!(params.transition[0][0], original_transition[0][0]);
}

#[test]
fn test_baum_welch_exactly_nine_observations() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let original_emission = params.emission.clone();

    let obs = vec![0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992, 0.991];
    params.baum_welch(&obs, 20, 1e-6, None, 5000);

    // 9 < 10, so parameters should be unchanged
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
}

#[test]
fn test_baum_welch_ten_observations_runs() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let _original_emission_mean = params.emission[0].mean;

    // Exactly 10 observations - should run
    let obs = vec![0.999, 0.998, 0.997, 0.5, 0.6, 0.7, 0.998, 0.999, 0.5, 0.4];
    params.baum_welch(&obs, 20, 1e-6, None, 5000);

    // With mixed observations, emission parameters should change
    // (we can't predict exactly how, but they should differ from initial)
    // At minimum, baum_welch shouldn't panic
}

// ===========================================================================
// estimate_emissions_logit edge cases
// ===========================================================================

#[test]
fn test_estimate_emissions_logit_too_few() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let original_emission = params.emission.clone();

    // Only 5 logit-space observations (< 10)
    let logit_obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    params.estimate_emissions_logit(&logit_obs, None, 5000);

    // Should not change parameters
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
}

#[test]
fn test_estimate_emissions_logit_identical_values() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    // All identical values → variance < 1e-6
    let logit_obs = vec![3.0; 20];
    params.estimate_emissions_logit(&logit_obs, None, 5000);

    // Should still produce valid parameters (low-variance branch)
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

#[test]
fn test_estimate_emissions_logit_well_separated() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    // Two clear clusters in logit space
    let mut logit_obs = vec![1.0; 15]; // low cluster
    logit_obs.extend(vec![5.0; 15]);   // high cluster
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // Emissions should reflect the separation
    assert!(params.emission[0].mean != params.emission[1].mean);
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

#[test]
fn test_estimate_emissions_logit_with_population_prior() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    let logit_obs: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    params.estimate_emissions_logit(&logit_obs, Some(Population::AFR), 5000);

    // Should produce valid, finite parameters
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}
