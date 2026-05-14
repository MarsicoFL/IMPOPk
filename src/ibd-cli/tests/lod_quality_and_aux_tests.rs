//! Tests for LOD scores, quality scores, posterior std, state refinement,
//! auxiliary emissions, and coverage ratio edge cases.

use impopk_ibd::hmm::{
    compute_per_window_lod, coverage_ratio, estimate_auxiliary_emissions,
    extract_ibd_segments_with_posteriors, infer_ibd_with_aux_features, refine_states_with_posteriors,
    segment_lod_score, segment_posterior_std, segment_quality_score, HmmParams,
    IbdSegmentWithPosterior, Population,
};
// ============================================================================
// compute_per_window_lod
// ============================================================================

#[test]
fn test_per_window_lod_ibd_observation() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    // Observation closer to IBD emission mean (0.9997) than non-IBD (0.999)
    let obs = vec![0.9999];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    // Should be positive: evidence favors IBD
    assert!(lods[0] > 0.0, "LOD for IBD-like obs should be positive, got {}", lods[0]);
}

#[test]
fn test_per_window_lod_non_ibd_observation() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    // Observation far from IBD mean
    let obs = vec![0.5];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    // Should be negative: evidence favors non-IBD
    assert!(lods[0] < 0.0, "LOD for non-IBD obs should be negative, got {}", lods[0]);
}

#[test]
fn test_per_window_lod_multiple_observations() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.5, 0.998, 0.9999, 0.9997, 0.5];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 5);
    // First and last should be negative (non-IBD region)
    assert!(lods[0] < 0.0);
    assert!(lods[4] < 0.0);
    // Middle values near IBD mean should be positive
    assert!(lods[2] > 0.0);
}

#[test]
fn test_per_window_lod_empty() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs: Vec<f64> = vec![];
    let lods = compute_per_window_lod(&obs, &params);
    assert!(lods.is_empty());
}

#[test]
fn test_per_window_lod_at_emission_means() {
    // Test LOD at the emission means themselves
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let non_ibd_mean = params.emission[0].mean;
    let ibd_mean = params.emission[1].mean;

    let lods = compute_per_window_lod(&[non_ibd_mean, ibd_mean], &params);
    // At non-IBD mean, LOD should be negative (non-IBD pdf > IBD pdf)
    assert!(lods[0] < 0.0, "LOD at non-IBD mean should be negative, got {}", lods[0]);
    // At IBD mean, LOD should be positive (IBD pdf > non-IBD pdf)
    assert!(lods[1] > 0.0, "LOD at IBD mean should be positive, got {}", lods[1]);
}

// ============================================================================
// segment_lod_score
// ============================================================================

#[test]
fn test_segment_lod_score_single_window() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.9999];
    let lod = segment_lod_score(&obs, 0, 0, &params);
    // Single window segment LOD = per-window LOD
    let per_window = compute_per_window_lod(&obs, &params);
    assert!((lod - per_window[0]).abs() < 1e-10);
}

#[test]
fn test_segment_lod_score_multi_window() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.5, 0.9999, 0.9997, 0.9998, 0.5];
    // LOD for the IBD segment (indices 1-3)
    let lod = segment_lod_score(&obs, 1, 3, &params);
    // Should be positive (IBD region)
    assert!(lod > 0.0, "IBD segment LOD should be positive, got {}", lod);

    // Sum of per-window LODs for indices 1-3
    let per_window = compute_per_window_lod(&obs, &params);
    let expected: f64 = per_window[1..=3].iter().sum();
    assert!((lod - expected).abs() < 1e-10, "LOD {} != expected sum {}", lod, expected);
}

#[test]
fn test_segment_lod_score_start_after_end() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.999, 0.9999];
    // Invalid: start > end
    let lod = segment_lod_score(&obs, 1, 0, &params);
    assert_eq!(lod, 0.0);
}

#[test]
fn test_segment_lod_score_end_out_of_bounds() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.999, 0.9999];
    // Invalid: end >= observations.len()
    let lod = segment_lod_score(&obs, 0, 5, &params);
    assert_eq!(lod, 0.0);
}

#[test]
fn test_segment_lod_score_full_range() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995];
    let lod = segment_lod_score(&obs, 0, 2, &params);
    let per_window = compute_per_window_lod(&obs, &params);
    let expected: f64 = per_window.iter().sum();
    assert!((lod - expected).abs() < 1e-10);
}

// ============================================================================
// segment_quality_score
// ============================================================================

#[test]
fn test_quality_score_high_confidence() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 49,
        n_windows: 50,
        mean_posterior: 0.99,
        min_posterior: 0.95,
        max_posterior: 1.0,
        lod_score: 100.0, // LOD/window = 2.0 > 1.0
    };
    let q = segment_quality_score(&seg);
    // Posterior: 0.99 * 40 = 39.6
    // Consistency: (0.95/0.99) * 20 ≈ 19.2
    // LOD: min(2.0/1.0, 1.0) * 30 = 30
    // Length: min(50/20, 1.0) * 10 = 10
    // Total ≈ 98.8
    assert!(q > 90.0, "High confidence should score >90, got {}", q);
}

#[test]
fn test_quality_score_low_confidence() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 2,
        n_windows: 3,
        mean_posterior: 0.3,
        min_posterior: 0.1,
        max_posterior: 0.5,
        lod_score: 0.5, // LOD/window ≈ 0.17
    };
    let q = segment_quality_score(&seg);
    // Posterior: 0.3 * 40 = 12
    // Consistency: (0.1/0.3) * 20 ≈ 6.7
    // LOD: min(0.17/1.0, 1.0) * 30 ≈ 5.0
    // Length: min(3/20, 1.0) * 10 = 1.5
    // Total ≈ 25
    assert!(q < 40.0, "Low confidence should score <40, got {}", q);
    assert!(q > 10.0, "Should still have some score, got {}", q);
}

#[test]
fn test_quality_score_zero_windows() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 0,
        n_windows: 0,
        mean_posterior: 0.0,
        min_posterior: 0.0,
        max_posterior: 0.0,
        lod_score: 0.0,
    };
    let q = segment_quality_score(&seg);
    assert_eq!(q, 0.0, "Zero-everything segment should score 0");
}

#[test]
fn test_quality_score_perfect() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 99,
        n_windows: 100,
        mean_posterior: 1.0,
        min_posterior: 1.0,
        max_posterior: 1.0,
        lod_score: 200.0, // LOD/window = 2.0
    };
    let q = segment_quality_score(&seg);
    // Posterior: 1.0 * 40 = 40
    // Consistency: (1.0/1.0) * 20 = 20
    // LOD: min(2.0, 1.0) * 30 = 30
    // Length: min(100/20, 1.0) * 10 = 10
    // Total = 100
    assert!((q - 100.0).abs() < 1e-10, "Perfect segment should score 100, got {}", q);
}

#[test]
fn test_quality_score_clamped_at_100() {
    // Even with extreme values, score should not exceed 100
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 999,
        n_windows: 1000,
        mean_posterior: 2.0, // will be clamped to 1.0
        min_posterior: 2.0,
        max_posterior: 2.0,
        lod_score: 100000.0,
    };
    let q = segment_quality_score(&seg);
    assert!(q <= 100.0, "Quality score should be capped at 100, got {}", q);
}

// ============================================================================
// segment_posterior_std
// ============================================================================

#[test]
fn test_posterior_std_constant() {
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    let std = segment_posterior_std(&posteriors, 0, 4);
    assert!((std - 0.0).abs() < 1e-10, "Constant posteriors should have std=0, got {}", std);
}

#[test]
fn test_posterior_std_varying() {
    let posteriors = vec![0.8, 0.9, 1.0];
    let std = segment_posterior_std(&posteriors, 0, 2);
    // mean = 0.9, variance = ((0.8-0.9)^2 + (0.9-0.9)^2 + (1.0-0.9)^2) / 2 = 0.01
    // std = 0.1
    assert!((std - 0.1).abs() < 1e-10, "Expected std=0.1, got {}", std);
}

#[test]
fn test_posterior_std_single_window() {
    let posteriors = vec![0.95];
    let std = segment_posterior_std(&posteriors, 0, 0);
    assert_eq!(std, 0.0, "Single window should have std=0");
}

#[test]
fn test_posterior_std_start_after_end() {
    let posteriors = vec![0.8, 0.9, 1.0];
    let std = segment_posterior_std(&posteriors, 2, 0);
    assert_eq!(std, 0.0, "Invalid range should return 0");
}

#[test]
fn test_posterior_std_end_out_of_bounds() {
    let posteriors = vec![0.8, 0.9];
    let std = segment_posterior_std(&posteriors, 0, 5);
    assert_eq!(std, 0.0, "Out-of-bounds end should return 0");
}

#[test]
fn test_posterior_std_subset() {
    let posteriors = vec![0.1, 0.5, 0.9, 0.5, 0.1];
    // Only compute std for indices 1-3: [0.5, 0.9, 0.5]
    let std = segment_posterior_std(&posteriors, 1, 3);
    // mean = (0.5+0.9+0.5)/3 ≈ 0.633
    // var = ((0.5-0.633)^2 + (0.9-0.633)^2 + (0.5-0.633)^2) / 2
    let mean: f64 = (0.5 + 0.9 + 0.5) / 3.0;
    let var: f64 = ((0.5 - mean).powi(2) + (0.9 - mean).powi(2) + (0.5 - mean).powi(2)) / 2.0;
    let expected = var.sqrt();
    assert!((std - expected).abs() < 1e-10, "Expected std={}, got {}", expected, std);
}

// ============================================================================
// refine_states_with_posteriors
// ============================================================================

#[test]
fn test_refine_extend_into_high_posterior_neighbor() {
    // Viterbi says [0, 0, 1, 1, 0, 0] but posteriors[4] is high → extend
    let mut states = vec![0, 0, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.95, 0.95, 0.8, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Index 4 (posterior=0.8 >= 0.5 extend_threshold) is adjacent to IBD at index 3
    assert_eq!(states[4], 1, "Should extend IBD to high-posterior neighbor");
    assert_eq!(states[5], 0, "Should not extend further");
}

#[test]
fn test_refine_trim_low_posterior_edge() {
    // IBD segment with weak edge
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.1, 0.95, 0.95, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Index 1 (posterior=0.1 < 0.2 trim_threshold) is at edge of IBD segment
    assert_eq!(states[1], 0, "Should trim low-posterior edge");
    assert_eq!(states[2], 1, "Interior should remain IBD");
    assert_eq!(states[3], 1, "Interior should remain IBD");
}

#[test]
fn test_refine_no_change_when_consistent() {
    let mut states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.9, 0.1, 0.1];
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original, "Consistent states should not change");
}

#[test]
fn test_refine_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert!(states.is_empty());
}

#[test]
fn test_refine_mismatched_lengths() {
    // Posteriors shorter than states → should return without changes
    let mut states = vec![0, 1, 1, 0];
    let posteriors = vec![0.1, 0.9];
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original, "Mismatched lengths should not modify states");
}

#[test]
fn test_refine_cascading_extension() {
    // Extension should cascade: if extending one window makes another eligible
    let mut states = vec![0, 0, 0, 1, 0, 0, 0];
    let posteriors = vec![0.1, 0.6, 0.7, 0.95, 0.7, 0.6, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Index 4 is adjacent to IBD (3), posterior 0.7 >= 0.5 → extend
    // Then index 5 is adjacent to extended IBD (4), posterior 0.6 >= 0.5 → extend
    // Then index 2 is adjacent to IBD (3), posterior 0.7 >= 0.5 → extend
    // Then index 1 is adjacent to extended IBD (2), posterior 0.6 >= 0.5 → extend
    assert_eq!(states[1], 1, "Cascading extension left");
    assert_eq!(states[2], 1, "Extension left");
    assert_eq!(states[3], 1, "Original IBD");
    assert_eq!(states[4], 1, "Extension right");
    assert_eq!(states[5], 1, "Cascading extension right");
    assert_eq!(states[0], 0, "Too low posterior to extend");
    assert_eq!(states[6], 0, "Too low posterior to extend");
}

// ============================================================================
// extract_ibd_segments_with_posteriors
// ============================================================================

#[test]
fn test_extract_with_posteriors_basic() {
    let states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.85, 0.1, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 2, 0.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 2);
    assert_eq!(segments[0].end_idx, 4);
    assert_eq!(segments[0].n_windows, 3);
    assert!((segments[0].mean_posterior - 0.9).abs() < 0.01);
    assert!((segments[0].min_posterior - 0.85).abs() < 1e-10);
    assert!((segments[0].max_posterior - 0.95).abs() < 1e-10);
}

#[test]
fn test_extract_with_posteriors_below_min_mean() {
    let states = vec![0, 1, 1, 0];
    let posteriors = vec![0.1, 0.3, 0.4, 0.1]; // mean posterior = 0.35, below 0.5
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segments.is_empty(), "Segment with low mean posterior should be filtered");
}

#[test]
fn test_extract_with_posteriors_below_min_windows() {
    let states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 3, 0.5);
    assert!(segments.is_empty(), "Segment below min_windows should be filtered");
}

// ============================================================================
// estimate_auxiliary_emissions
// ============================================================================

#[test]
fn test_aux_emissions_clear_separation() {
    // Clear IBD and non-IBD posteriors with distinct auxiliary values
    let aux_obs = vec![0.3, 0.4, 0.35, 0.9, 0.95, 0.92];
    let posteriors = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // clean separation
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // non-IBD mean should be ~0.35
    assert!((non_ibd.mean - 0.35).abs() < 0.01,
        "non-IBD mean should be ~0.35, got {}", non_ibd.mean);
    // IBD mean should be ~0.923
    assert!((ibd.mean - 0.923).abs() < 0.01,
        "IBD mean should be ~0.923, got {}", ibd.mean);
    // IBD mean should be higher than non-IBD
    assert!(ibd.mean > non_ibd.mean);
}

#[test]
fn test_aux_emissions_soft_assignments() {
    // Soft posteriors (typical real-world scenario)
    let aux_obs = vec![0.5, 0.6, 0.8, 0.9];
    let posteriors = vec![0.2, 0.3, 0.7, 0.9];
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Both should have finite, positive std
    assert!(non_ibd.std > 0.0 && non_ibd.std.is_finite());
    assert!(ibd.std > 0.0 && ibd.std.is_finite());
    // Means should be finite
    assert!(non_ibd.mean.is_finite());
    assert!(ibd.mean.is_finite());
}

#[test]
fn test_aux_emissions_all_ibd() {
    // All posteriors = 1 → non-IBD state has no data → defaults
    let aux_obs = vec![0.9, 0.95, 0.92];
    let posteriors = vec![1.0, 1.0, 1.0];
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Non-IBD should use defaults (mean=0.5, std=0.2) since no data for state 0
    assert!((non_ibd.mean - 0.5).abs() < 1e-10, "Default non-IBD mean should be 0.5");
    assert!((non_ibd.std - 0.2).abs() < 1e-10, "Default non-IBD std should be 0.2");
    // IBD should compute from data
    let expected_mean = (0.9 + 0.95 + 0.92) / 3.0;
    assert!((ibd.mean - expected_mean).abs() < 1e-10);
}

#[test]
fn test_aux_emissions_all_non_ibd() {
    // All posteriors = 0 → IBD state has no data → defaults
    let aux_obs = vec![0.3, 0.35, 0.32];
    let posteriors = vec![0.0, 0.0, 0.0];
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // IBD should use defaults (mean=0.9, std=0.1) since no data for state 1
    assert!((ibd.mean - 0.9).abs() < 1e-10, "Default IBD mean should be 0.9");
    assert!((ibd.std - 0.1).abs() < 1e-10, "Default IBD std should be 0.1");
    // Non-IBD should compute from data
    let expected_mean = (0.3 + 0.35 + 0.32) / 3.0;
    assert!((non_ibd.mean - expected_mean).abs() < 1e-10);
}

#[test]
fn test_aux_emissions_single_observation() {
    let aux_obs = vec![0.7];
    let posteriors = vec![0.5];
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);
    // Both should have finite values
    assert!(non_ibd.mean.is_finite());
    assert!(ibd.mean.is_finite());
    assert!(non_ibd.std > 0.0);
    assert!(ibd.std > 0.0);
}

// ============================================================================
// coverage_ratio
// ============================================================================

#[test]
fn test_coverage_ratio_equal() {
    assert!((coverage_ratio(100, 100) - 1.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_both_zero() {
    assert_eq!(coverage_ratio(0, 0), 0.0);
}

#[test]
fn test_coverage_ratio_one_zero() {
    assert_eq!(coverage_ratio(100, 0), 0.0);
    assert_eq!(coverage_ratio(0, 100), 0.0);
}

#[test]
fn test_coverage_ratio_half() {
    assert!((coverage_ratio(50, 100) - 0.5).abs() < 1e-10);
    assert!((coverage_ratio(100, 50) - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_large_values() {
    let ratio = coverage_ratio(1_000_000_000, 2_000_000_000);
    assert!((ratio - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_one_bp_difference() {
    let ratio = coverage_ratio(999, 1000);
    assert!((ratio - 0.999).abs() < 1e-10);
}

// ============================================================================
// infer_ibd_with_aux_features
// ============================================================================

#[test]
fn test_infer_with_aux_too_few_observations() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.999, 0.9995]; // only 2 obs < 3
    let (result, aux) = infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 10000, 3, None);
    assert_eq!(result.states.len(), 2);
    assert!(result.states.iter().all(|&s| s == 0), "Too few obs should return all non-IBD");
    assert_eq!(result.log_likelihood, f64::NEG_INFINITY);
    assert!(aux.is_none());
}

#[test]
fn test_infer_with_aux_no_auxiliary() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.9999, 0.9998, 0.9997, 0.999, 0.998];
    let (result, aux) = infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 10000, 2, None);
    assert_eq!(result.states.len(), 8);
    assert!(result.log_likelihood.is_finite());
    assert!(aux.is_none(), "No aux data → aux should be None");
}

#[test]
fn test_infer_with_aux_features_provided() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.9999, 0.9998, 0.9997, 0.999, 0.998, 0.997, 0.9985];
    let aux = vec![0.3, 0.35, 0.5, 0.9, 0.95, 0.92, 0.4, 0.3, 0.32, 0.38];
    let (result, aux_em) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::EUR, 10000, 2, Some(&aux),
    );
    assert_eq!(result.states.len(), 10);
    assert!(result.log_likelihood.is_finite());
    assert!(aux_em.is_some(), "Aux data provided → aux_em should be Some");
    let [non_ibd_em, ibd_em] = aux_em.unwrap();
    assert!(non_ibd_em.std > 0.0);
    assert!(ibd_em.std > 0.0);
}

#[test]
fn test_infer_with_aux_mismatched_length() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.9999, 0.9998];
    let aux = vec![0.3, 0.35]; // wrong length
    let (result, aux_em) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::EUR, 10000, 2, Some(&aux),
    );
    assert_eq!(result.states.len(), 5);
    assert!(aux_em.is_none(), "Mismatched aux length → aux should be None");
}

#[test]
fn test_infer_with_aux_zero_bw_iters() {
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.9999, 0.9998, 0.9997, 0.999, 0.998, 0.997, 0.9985];
    let (result, _) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::EUR, 10000, 0, None,
    );
    assert_eq!(result.states.len(), 10);
    assert!(result.log_likelihood.is_finite());
}
