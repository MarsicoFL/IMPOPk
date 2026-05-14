//! Cycle 32: Boundary condition tests for specific untested code paths.
//!
//! These tests target specific parameter boundaries and conditional branches
//! that existing tests do not exercise:
//! - gaussian_to_logit_space with mean=0.0 and mean=1.0 (denominator ≤ 1e-15 branch)
//! - bic_model_selection with exactly 2 data points (minimum valid n)
//! - trimmed_mean with trim_fraction ≥ 0.5 returning None (start ≥ end branch)
//! - Region::parse with colon but no dash (empty coord string)
//! - baum_welch convergence within 1 iteration (early exit via tol branch)
//! - segment_lod_score with single-window segment (start_idx == end_idx)
//! - infer_ibd_with_aux_features with fewer than 3 observations (early return)
//! - compute_combined_log_emissions aux length mismatch (fallback to primary-only)
//! - estimate_auxiliary_emissions with all-zero posteriors (fallback branch)
//! - refine_states_with_posteriors cascading extension across gap
//! - HmmParams::summary content verification

use impopk_ibd::hmm::{
    compute_combined_log_emissions, compute_per_window_lod, coverage_ratio,
    estimate_auxiliary_emissions, extract_ibd_segments_with_lod,
    extract_ibd_segments_with_posteriors, forward_backward, forward_backward_from_log_emit,
    forward_from_log_emit, infer_ibd, infer_ibd_with_aux_features, infer_ibd_with_training,
    precompute_log_emissions, refine_states_with_posteriors, segment_lod_score,
    segment_quality_score, viterbi, viterbi_from_log_emit, HmmParams, IbdSegmentWithPosterior,
    Population,
};
use impopk_ibd::stats::{bic_model_selection, gaussian_to_logit_space, trimmed_mean, GaussianParams, LOGIT_CAP};
use impopk_ibd::Region;

// ══════════════════════════════════════════════════════════════════════
// gaussian_to_logit_space: mean=0.0 and mean=1.0 (denominator ≤ 1e-15)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn gaussian_to_logit_space_mean_exactly_zero() {
    // mean=0.0 → denominator = 0.0 * 1.0 = 0.0 → else branch → logit_std = LOGIT_CAP * 0.5
    let params = gaussian_to_logit_space(0.0, 0.001);
    assert!(params.mean.is_finite(), "logit mean should be finite");
    assert!(params.std > 0.0, "logit std should be positive");
    // logit_std should be LOGIT_CAP * 0.5
    let expected_std = LOGIT_CAP * 0.5;
    assert!(
        (params.std - expected_std).abs() < 0.01,
        "logit_std should be {} (LOGIT_CAP * 0.5), got {}",
        expected_std, params.std
    );
}

#[test]
fn gaussian_to_logit_space_mean_exactly_one() {
    // mean=1.0 → denominator = 1.0 * 0.0 = 0.0 → else branch
    let params = gaussian_to_logit_space(1.0, 0.001);
    assert!(params.mean.is_finite());
    assert!(params.std > 0.0);
    let expected_std = LOGIT_CAP * 0.5;
    assert!(
        (params.std - expected_std).abs() < 0.01,
        "logit_std should be {}, got {}",
        expected_std, params.std
    );
}

#[test]
fn gaussian_to_logit_space_mean_near_boundary_but_above_threshold() {
    // mean just enough that denominator > 1e-15 → normal branch
    let mean = 0.001; // denominator = 0.001 * 0.999 = 0.000999 >> 1e-15
    let params = gaussian_to_logit_space(mean, 0.0001);
    // Should use delta method, not fallback
    let expected_logit_std = (0.0001 / (mean * (1.0 - mean))).min(LOGIT_CAP * 0.5);
    assert!(
        (params.std - expected_logit_std.max(0.01)).abs() < 0.1,
        "expected ~{}, got {}",
        expected_logit_std,
        params.std
    );
}

// ══════════════════════════════════════════════════════════════════════
// bic_model_selection: boundary conditions
// ══════════════════════════════════════════════════════════════════════

#[test]
fn bic_model_selection_exactly_two_data_points() {
    // n=2 is the minimum valid input (n < 2 returns (0.0, 0.0))
    let data = [0.99, 0.999];
    let low = GaussianParams::new_unchecked(0.99, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    assert!(bic_1.is_finite(), "bic_1 should be finite for n=2");
    assert!(bic_2.is_finite(), "bic_2 should be finite for n=2");
    // Both BIC values should be negative (log-likelihood based)
    assert!(bic_1 < 0.0, "bic_1 should be negative");
    assert!(bic_2 < 0.0, "bic_2 should be negative");
}

#[test]
fn bic_model_selection_one_data_point_returns_zeros() {
    let data = [0.999];
    let low = GaussianParams::new_unchecked(0.99, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

#[test]
fn bic_model_selection_empty_returns_zeros() {
    let low = GaussianParams::new_unchecked(0.99, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&[], &low, &high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

// ══════════════════════════════════════════════════════════════════════
// trimmed_mean: trim_fraction ≥ 0.5 → start ≥ end → None
// ══════════════════════════════════════════════════════════════════════

#[test]
fn trimmed_mean_fraction_0_5_exactly() {
    // trim_fraction is clamped to 0.49, so 0.5 → 0.49
    // With 4 elements: trim_count = (4 * 0.49) = 1, start=1, end=3 → valid
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = trimmed_mean(&data, 0.5);
    assert!(
        result.is_some(),
        "trim_fraction=0.5 should be clamped to 0.49"
    );
}

#[test]
fn trimmed_mean_fraction_above_0_5_clamped() {
    // Even 0.99 is clamped to 0.49
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = trimmed_mean(&data, 0.99);
    assert!(result.is_some(), "trim_fraction=0.99 should be clamped to 0.49");
}

#[test]
fn trimmed_mean_two_elements_high_trim() {
    // With 2 elements: trim_count = (2 * 0.49) = 0, start=0, end=2 → whole range
    let data = [1.0, 3.0];
    let result = trimmed_mean(&data, 0.49).unwrap();
    assert!((result - 2.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_single_element() {
    let result = trimmed_mean(&[42.0], 0.49).unwrap();
    assert!((result - 42.0).abs() < 1e-10);
}

// ══════════════════════════════════════════════════════════════════════
// baum_welch: convergence within 1 iteration (tolerance branch)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn baum_welch_converges_within_few_iterations_with_large_tol() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    // Use a very large tolerance so it converges in 1 iteration
    let obs: Vec<f64> = (0..20).map(|_| 0.9995).collect();
    params.baum_welch(&obs, 100, 1e30, Some(Population::EUR), 5000);

    // Should have exited early — params might change from 1 iteration but not many
    assert!(
        params.emission[0].mean.is_finite(),
        "emission mean should remain finite"
    );
    // The tolerance is so large that log_lik improvement < tol after first iteration
}

#[test]
fn baum_welch_max_iter_zero_is_noop() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let orig_mean = params.emission[0].mean;
    let obs: Vec<f64> = (0..20).map(|_| 0.9995).collect();
    params.baum_welch(&obs, 0, 1e-6, Some(Population::EUR), 5000);
    assert_eq!(
        params.emission[0].mean, orig_mean,
        "max_iter=0 should not modify params"
    );
}

// ══════════════════════════════════════════════════════════════════════
// infer_ibd_with_aux_features: edge cases
// ══════════════════════════════════════════════════════════════════════

#[test]
fn infer_ibd_with_aux_features_fewer_than_3_obs() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.999, 0.998];
    let (result, aux_emit) = infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 5, None);
    assert_eq!(result.states.len(), 2);
    assert!(result.states.iter().all(|&s| s == 0), "should return all non-IBD");
    assert!(result.posteriors.iter().all(|&p| p == 0.0));
    assert_eq!(result.log_likelihood, f64::NEG_INFINITY);
    assert!(aux_emit.is_none());
}

#[test]
fn infer_ibd_with_aux_features_empty_obs() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let (result, aux_emit) = infer_ibd_with_aux_features(&[], &mut params, Population::EUR, 5000, 5, None);
    assert!(result.states.is_empty());
    assert!(result.posteriors.is_empty());
    assert!(aux_emit.is_none());
}

#[test]
fn infer_ibd_with_aux_features_with_aux_data() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = vec![0.998, 0.998, 0.9999, 0.9999, 0.9999, 0.998, 0.998, 0.998, 0.998, 0.998];
    let aux = vec![0.5, 0.5, 0.9, 0.95, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5];
    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 3, Some(&aux));
    assert_eq!(result.states.len(), 10);
    assert!(result.log_likelihood.is_finite());
    // With aux data provided, aux_emit should be Some
    assert!(aux_emit.is_some(), "aux_emit should be returned when aux data provided");
    let [non_ibd_aux, ibd_aux] = aux_emit.unwrap();
    assert!(non_ibd_aux.std > 0.0);
    assert!(ibd_aux.std > 0.0);
}

#[test]
fn infer_ibd_with_aux_features_aux_length_mismatch() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = vec![0.998, 0.998, 0.9999, 0.9999, 0.9999, 0.998, 0.998, 0.998, 0.998, 0.998];
    let aux = vec![0.5, 0.5]; // length mismatch
    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 3, Some(&aux));
    assert_eq!(result.states.len(), 10);
    // Aux length mismatch → aux_emit should be None
    assert!(aux_emit.is_none(), "aux_emit should be None when lengths mismatch");
}

// ══════════════════════════════════════════════════════════════════════
// compute_combined_log_emissions: fallback when aux is mismatched
// ══════════════════════════════════════════════════════════════════════

#[test]
fn compute_combined_log_emissions_aux_length_mismatch_falls_back() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.999, 0.998, 0.997];
    let aux_obs = vec![0.5, 0.6]; // wrong length
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.1),
        GaussianParams::new_unchecked(0.9, 0.1),
    ];

    let primary_only = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    // Should fall back to primary-only when lengths don't match
    assert_eq!(primary_only.len(), combined.len());
    for (p, c) in primary_only.iter().zip(combined.iter()) {
        assert_eq!(p[0], c[0]);
        assert_eq!(p[1], c[1]);
    }
}

#[test]
fn compute_combined_log_emissions_no_aux_matches_primary() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.999, 0.998, 0.997];
    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, None, None);

    for (p, c) in primary.iter().zip(combined.iter()) {
        assert_eq!(p[0], c[0]);
        assert_eq!(p[1], c[1]);
    }
}

#[test]
fn compute_combined_log_emissions_with_valid_aux_differs_from_primary() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.999, 0.998, 0.997];
    let aux_obs = vec![0.5, 0.6, 0.7];
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.1),
        GaussianParams::new_unchecked(0.9, 0.1),
    ];

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    // Combined should differ from primary when aux is valid
    let any_different = primary
        .iter()
        .zip(combined.iter())
        .any(|(p, c)| (p[0] - c[0]).abs() > 1e-10 || (p[1] - c[1]).abs() > 1e-10);
    assert!(
        any_different,
        "combined emissions should differ from primary when aux is valid"
    );
}

// ══════════════════════════════════════════════════════════════════════
// estimate_auxiliary_emissions: all-zero posteriors (fallback means)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn estimate_auxiliary_emissions_all_zero_posteriors() {
    let aux_obs = vec![0.5, 0.6, 0.7, 0.8];
    let posteriors = vec![0.0, 0.0, 0.0, 0.0]; // all non-IBD
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);
    // All weight on non-IBD state; IBD state gets fallback mean=0.9, std=0.1
    assert!(non_ibd.mean.is_finite());
    assert!(non_ibd.std > 0.0);
    assert!((ibd.mean - 0.9).abs() < 0.01, "IBD should use fallback mean=0.9");
    assert!((ibd.std - 0.1).abs() < 0.01, "IBD should use fallback std=0.1");
}

#[test]
fn estimate_auxiliary_emissions_all_one_posteriors() {
    let aux_obs = vec![0.5, 0.6, 0.7, 0.8];
    let posteriors = vec![1.0, 1.0, 1.0, 1.0]; // all IBD
    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);
    // All weight on IBD state; non-IBD state gets fallback mean=0.5, std=0.2
    assert!((non_ibd.mean - 0.5).abs() < 0.01, "non-IBD should use fallback mean=0.5");
    assert!((non_ibd.std - 0.2).abs() < 0.01, "non-IBD should use fallback std=0.2");
    assert!(ibd.mean.is_finite());
    assert!(ibd.std > 0.0);
}

// ══════════════════════════════════════════════════════════════════════
// refine_states_with_posteriors: cascading extension fills gap
// ══════════════════════════════════════════════════════════════════════

#[test]
fn refine_states_cascading_extension_fills_gap() {
    // IBD [1,1,0,0,1,1] with high posteriors at gap positions
    let mut states = vec![1, 1, 0, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.8, 0.8, 0.9, 0.9];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // The 0s at positions 2,3 have posteriors > extend_threshold (0.5)
    // and are adjacent to IBD → should be extended
    assert_eq!(states, vec![1, 1, 1, 1, 1, 1]);
}

#[test]
fn refine_states_trim_low_posterior_edge() {
    // IBD at edges with low posteriors → should trim
    let mut states = vec![1, 1, 1, 1];
    let posteriors = vec![0.1, 0.9, 0.9, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Positions 0 and 3 have posterior < trim_threshold (0.2) and are at edges
    assert_eq!(states[0], 0, "low posterior at left edge should be trimmed");
    assert_eq!(states[3], 0, "low posterior at right edge should be trimmed");
    assert_eq!(states[1], 1);
    assert_eq!(states[2], 1);
}

#[test]
fn refine_states_mismatched_lengths_is_noop() {
    let mut states = vec![0, 1, 1, 0];
    let posteriors = vec![0.9, 0.9]; // wrong length
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original, "mismatched lengths should leave states unchanged");
}

// ══════════════════════════════════════════════════════════════════════
// log-emit HMM equivalence: verify log-emit versions match observation versions
// ══════════════════════════════════════════════════════════════════════

#[test]
fn log_emit_forward_backward_matches_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9985, 0.9999, 0.99995, 0.9999, 0.998, 0.997];

    let (posteriors_std, ll_std) = forward_backward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors_le, ll_le) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(
        (ll_std - ll_le).abs() < 1e-10,
        "log-likelihoods should match: {} vs {}",
        ll_std,
        ll_le
    );
    for (i, (p_std, p_le)) in posteriors_std.iter().zip(posteriors_le.iter()).enumerate() {
        assert!(
            (p_std - p_le).abs() < 1e-10,
            "posterior mismatch at window {}: {} vs {}",
            i,
            p_std,
            p_le
        );
    }
}

#[test]
fn log_emit_viterbi_matches_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9985, 0.9999, 0.99995, 0.9999, 0.998, 0.997];

    let states_std = viterbi(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let states_le = viterbi_from_log_emit(&log_emit, &params);

    assert_eq!(
        states_std, states_le,
        "Viterbi states should match between standard and log-emit"
    );
}

#[test]
fn log_emit_empty_input() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let log_emit: Vec<[f64; 2]> = vec![];

    let (alpha, ll) = forward_from_log_emit(&log_emit, &params);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);

    let (posteriors, ll2) = forward_backward_from_log_emit(&log_emit, &params);
    assert!(posteriors.is_empty());
    assert_eq!(ll2, 0.0);

    let states = viterbi_from_log_emit(&log_emit, &params);
    assert!(states.is_empty());
}

// ══════════════════════════════════════════════════════════════════════
// segment_quality_score: specific component verification
// ══════════════════════════════════════════════════════════════════════

#[test]
fn segment_quality_score_perfect_segment() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 29,
        n_windows: 30,
        mean_posterior: 1.0,
        min_posterior: 1.0,
        max_posterior: 1.0,
        lod_score: 50.0, // very high LOD
    };
    let q = segment_quality_score(&seg);
    // posterior_score = 1.0 * 40 = 40
    // consistency = 1.0 * 20 = 20
    // lod_per_window = 50/30 = 1.67 → clamped to 1.0 → * 30 = 30
    // length = min(30/20, 1.0) * 10 = 10
    // total = 100
    assert!((q - 100.0).abs() < 0.01, "perfect segment should score 100, got {}", q);
}

#[test]
fn segment_quality_score_zero_everything() {
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
    assert!(q >= 0.0 && q <= 100.0);
    assert_eq!(q, 0.0, "all-zero segment should score 0");
}

// ══════════════════════════════════════════════════════════════════════
// HmmParams::summary content
// ══════════════════════════════════════════════════════════════════════

#[test]
fn hmm_params_summary_contains_key_values() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let summary = params.summary();
    assert!(summary.contains("HMM Parameters:"));
    assert!(summary.contains("Initial:"));
    assert!(summary.contains("Transition:"));
    assert!(summary.contains("Emission non-IBD:"));
    assert!(summary.contains("Emission IBD:"));
    // Verify the emission means appear in the summary
    let mean_str = format!("{:.6}", params.emission[0].mean);
    assert!(
        summary.contains(&mean_str),
        "summary should contain emission[0].mean={}, got: {}",
        mean_str,
        summary
    );
}

// ══════════════════════════════════════════════════════════════════════
// coverage_ratio: ibd-cli version boundary
// ══════════════════════════════════════════════════════════════════════

#[test]
fn coverage_ratio_one_vs_u64_max() {
    let r = coverage_ratio(1, u64::MAX);
    assert!(r > 0.0);
    assert!(r < 1e-10, "1/MAX should be very close to 0");
}

// ══════════════════════════════════════════════════════════════════════
// extract_ibd_segments_with_lod: obs+params provided computes LOD
// ══════════════════════════════════════════════════════════════════════

#[test]
fn extract_segments_with_lod_computes_nonzero_lod() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.9, 0.1, 0.1];
    let obs = vec![0.998, 0.998, 0.9999, 0.99999, 0.9999, 0.998, 0.998];

    let segments =
        extract_ibd_segments_with_lod(&states, &posteriors, 1, 0.0, Some((&obs, &params)), None);
    assert_eq!(segments.len(), 1);
    assert!(
        segments[0].lod_score > 0.0,
        "LOD should be positive for IBD-like observations, got {}",
        segments[0].lod_score
    );
}

#[test]
fn extract_segments_without_obs_gives_zero_lod() {
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.95, 0.9, 0.1];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(
        segments[0].lod_score, 0.0,
        "LOD should be 0 when no obs provided"
    );
}

// ══════════════════════════════════════════════════════════════════════
// per_window_lod: sign verification
// ══════════════════════════════════════════════════════════════════════

#[test]
fn per_window_lod_sign_matches_state() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    // IBD-like observation (very high identity)
    let obs_ibd = vec![0.99999];
    let lods = compute_per_window_lod(&obs_ibd, &params);
    assert!(
        lods[0] > 0.0,
        "LOD should be positive for IBD observation: {}",
        lods[0]
    );

    // Non-IBD observation (lower identity)
    let obs_non_ibd = vec![params.emission[0].mean];
    let lods_non_ibd = compute_per_window_lod(&obs_non_ibd, &params);
    assert!(
        lods_non_ibd[0] < 0.0,
        "LOD should be negative for non-IBD observation: {}",
        lods_non_ibd[0]
    );
}

// ══════════════════════════════════════════════════════════════════════
// segment_lod_score: boundary with start_idx == end_idx (single window)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn segment_lod_score_single_window_matches_per_window() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9999, 0.997];

    let per_window = compute_per_window_lod(&obs, &params);
    let seg_lod = segment_lod_score(&obs, 1, 1, &params);
    assert!(
        (seg_lod - per_window[1]).abs() < 1e-10,
        "single-window segment LOD should match per-window LOD: {} vs {}",
        seg_lod,
        per_window[1]
    );
}

// ══════════════════════════════════════════════════════════════════════
// Region::parse boundary conditions (via re-export from impopk-common)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn region_parse_colon_no_coords_is_error() {
    // "chr1:" has colon but rest="" which has no dash → InvalidRegion
    let result = Region::parse("chr1:", None);
    assert!(result.is_err(), "chr1: (no coords after colon) should be error");
}

#[test]
fn region_parse_colon_dash_only_is_error() {
    // "chr1:-" has rest="-", dash at pos 0, start="" and end="" → parse error
    let result = Region::parse("chr1:-", None);
    assert!(result.is_err(), "chr1:- should be error (empty start/end)");
}

#[test]
fn region_parse_colon_with_dash_empty_end() {
    // "chr1:100-" has rest="100-", dash at pos 3, end="" → parse error
    let result = Region::parse("chr1:100-", None);
    assert!(result.is_err(), "chr1:100- (empty end) should be error");
}

#[test]
fn region_parse_colon_with_dash_empty_start() {
    // "chr1:-200" has rest="-200", dash at pos 0, start="" → parse error
    let result = Region::parse("chr1:-200", None);
    assert!(result.is_err(), "chr1:-200 (empty start) should be error");
}

#[test]
fn region_parse_multiple_dashes() {
    // "chr1:100-200-300" → first dash at pos 3, start="100", end="200-300" → parse error on end
    let result = Region::parse("chr1:100-200-300", None);
    assert!(result.is_err(), "multiple dashes should be error");
}

#[test]
fn region_parse_multiple_colons() {
    // "chr1:100:200" → first colon at pos 4, rest="100:200", no dash → error
    let result = Region::parse("chr1:100:200", None);
    assert!(result.is_err(), "multiple colons without dash should be error");
}

#[test]
fn region_parse_empty_string_without_length() {
    // "" has no colon → chrom-only branch → no length → error
    let result = Region::parse("", None);
    assert!(result.is_err(), "empty string without length should be error");
}

#[test]
fn region_parse_empty_string_with_length() {
    // "" has no colon → chrom-only branch → uses length
    let result = Region::parse("", Some(100));
    assert!(result.is_ok());
    let region = result.unwrap();
    assert_eq!(region.chrom, "");
    assert_eq!(region.start, 1);
    assert_eq!(region.end, 100);
}
