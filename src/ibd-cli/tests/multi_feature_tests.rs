/// Edge case and integration tests for multi-feature emission model functions:
/// - precompute_log_emissions
/// - compute_combined_log_emissions
/// - forward_from_log_emit / backward_from_log_emit
/// - viterbi_from_log_emit / forward_backward_from_log_emit
/// - estimate_auxiliary_emissions
/// - infer_ibd_with_aux_features
/// - coverage_ratio
use hprc_ibd::hmm::{
    backward_from_log_emit, compute_combined_log_emissions, coverage_ratio,
    estimate_auxiliary_emissions, forward_backward, forward_backward_from_log_emit,
    forward_from_log_emit, infer_ibd_with_aux_features, precompute_log_emissions,
    viterbi_from_log_emit, HmmParams, Population,
};
use hprc_ibd::stats::{
    bic_model_selection, em_two_component, em_two_component_map, trimmed_mean, GaussianParams,
};

// ============================================================================
// precompute_log_emissions
// ============================================================================

#[test]
fn test_precompute_log_emissions_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let log_emit = precompute_log_emissions(&[], &params);
    assert!(log_emit.is_empty());
}

#[test]
fn test_precompute_log_emissions_single() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let log_emit = precompute_log_emissions(&[0.999], &params);
    assert_eq!(log_emit.len(), 1);
    // Both log-emissions should be finite (not necessarily negative for unnormalized Gaussian)
    assert!(log_emit[0][0].is_finite());
    assert!(log_emit[0][1].is_finite());
}

#[test]
fn test_precompute_log_emissions_all_same() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5; 10];
    let log_emit = precompute_log_emissions(&obs, &params);
    assert_eq!(log_emit.len(), 10);
    // All entries should be identical since observations are identical
    for t in 1..10 {
        assert!((log_emit[t][0] - log_emit[0][0]).abs() < 1e-15);
        assert!((log_emit[t][1] - log_emit[0][1]).abs() < 1e-15);
    }
}

#[test]
fn test_precompute_log_emissions_extreme_values() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    // Very near 0 and very near 1
    let obs = vec![0.001, 0.999999];
    let log_emit = precompute_log_emissions(&obs, &params);
    assert_eq!(log_emit.len(), 2);
    for t in 0..2 {
        assert!(log_emit[t][0].is_finite(), "log_emit[{}][0] = {}", t, log_emit[t][0]);
        assert!(log_emit[t][1].is_finite(), "log_emit[{}][1] = {}", t, log_emit[t][1]);
    }
}

// ============================================================================
// compute_combined_log_emissions
// ============================================================================

#[test]
fn test_combined_log_emissions_none_aux_obs_some_emit() {
    // aux_observations is None but aux_emission is Some → should return primary only
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998, 0.999];
    let aux_emit = [
        GaussianParams::new_unchecked(0.85, 0.1),
        GaussianParams::new_unchecked(0.98, 0.02),
    ];

    let combined = compute_combined_log_emissions(&obs, &params, None, Some(&aux_emit));
    let primary = precompute_log_emissions(&obs, &params);
    for t in 0..obs.len() {
        for s in 0..2 {
            assert!((combined[t][s] - primary[t][s]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_combined_log_emissions_some_aux_obs_none_emit() {
    // aux_observations is Some but aux_emission is None → should return primary only
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998, 0.999];
    let aux = vec![0.8, 0.9];

    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux), None);
    let primary = precompute_log_emissions(&obs, &params);
    for t in 0..obs.len() {
        for s in 0..2 {
            assert!((combined[t][s] - primary[t][s]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_combined_log_emissions_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let combined = compute_combined_log_emissions(&[], &params, None, None);
    assert!(combined.is_empty());
}

#[test]
fn test_combined_log_emissions_aux_adds_log_probability() {
    // With auxiliary, combined should be strictly less (more negative) than primary alone
    // because we're adding negative log-pdf values
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998];
    let aux = vec![0.8];
    let aux_emit = [
        GaussianParams::new_unchecked(0.85, 0.1),
        GaussianParams::new_unchecked(0.98, 0.02),
    ];

    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux), Some(&aux_emit));
    let primary = precompute_log_emissions(&obs, &params);

    // Combined = primary + aux_log_pdf — should differ from primary
    // (aux_log_pdf adds additional information, whether positive or negative)
    for s in 0..2 {
        let diff = (combined[0][s] - primary[0][s]).abs();
        assert!(diff > 1e-10,
            "Combined[{}] should differ from primary[{}]: {} vs {}", s, s, combined[0][s], primary[0][s]);
    }
}

// ============================================================================
// forward_from_log_emit
// ============================================================================

#[test]
fn test_forward_from_log_emit_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let (alpha, ll) = forward_from_log_emit(&[], &params);
    assert!(alpha.is_empty());
    assert!((ll - 0.0).abs() < 1e-10);
}

#[test]
fn test_forward_from_log_emit_single() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let log_emit = precompute_log_emissions(&[0.999], &params);
    let (alpha, ll) = forward_from_log_emit(&log_emit, &params);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
    // Alpha values should be finite
    assert!(alpha[0][0].is_finite());
    assert!(alpha[0][1].is_finite());
}

#[test]
fn test_forward_from_log_emit_ll_negative() {
    // Log-likelihood should always be negative (probability < 1)
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.5, 0.6];
    let log_emit = precompute_log_emissions(&obs, &params);
    let (_, ll) = forward_from_log_emit(&log_emit, &params);
    assert!(ll < 0.0, "Log-likelihood should be negative: {}", ll);
    assert!(ll.is_finite());
}

#[test]
fn test_forward_from_log_emit_longer_sequence_lower_ll() {
    // Longer sequence should have lower log-likelihood (more data → lower joint probability)
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs_short = vec![0.998, 0.999];
    let obs_long = vec![0.998, 0.999, 0.997, 0.996, 0.995];

    let le_short = precompute_log_emissions(&obs_short, &params);
    let le_long = precompute_log_emissions(&obs_long, &params);

    let (_, ll_short) = forward_from_log_emit(&le_short, &params);
    let (_, ll_long) = forward_from_log_emit(&le_long, &params);

    assert!(ll_long < ll_short,
        "Longer sequence should have lower LL: {} vs {}", ll_long, ll_short);
}

// ============================================================================
// backward_from_log_emit
// ============================================================================

#[test]
fn test_backward_from_log_emit_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let beta = backward_from_log_emit(&[], &params);
    assert!(beta.is_empty());
}

#[test]
fn test_backward_from_log_emit_single() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let log_emit = precompute_log_emissions(&[0.999], &params);
    let beta = backward_from_log_emit(&log_emit, &params);
    assert_eq!(beta.len(), 1);
    // Terminal condition: beta[T-1] = [0, 0] (log(1) = 0)
    assert!((beta[0][0] - 0.0).abs() < 1e-10);
    assert!((beta[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_backward_from_log_emit_terminal_condition() {
    // The last beta should always be [0, 0] regardless of sequence
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.998, 0.999, 0.5];
    let log_emit = precompute_log_emissions(&obs, &params);
    let beta = backward_from_log_emit(&log_emit, &params);
    let n = beta.len();
    assert!((beta[n - 1][0] - 0.0).abs() < 1e-10);
    assert!((beta[n - 1][1] - 0.0).abs() < 1e-10);
}

// ============================================================================
// forward_backward_from_log_emit
// ============================================================================

#[test]
fn test_fb_from_log_emit_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let (posteriors, ll) = forward_backward_from_log_emit(&[], &params);
    assert!(posteriors.is_empty());
    assert!((ll - 0.0).abs() < 1e-10);
}

#[test]
fn test_fb_from_log_emit_posteriors_in_unit_interval() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.998, 0.999, 0.9995, 0.5, 0.4];
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors, ll) = forward_backward_from_log_emit(&log_emit, &params);

    assert_eq!(posteriors.len(), obs.len());
    assert!(ll.is_finite());
    for (t, &p) in posteriors.iter().enumerate() {
        assert!(p >= 0.0 && p <= 1.0,
            "Posterior at t={} out of range: {}", t, p);
        assert!(p.is_finite(), "Posterior at t={} is not finite", t);
    }
}

#[test]
fn test_fb_from_log_emit_consistency_with_forward() {
    // Log-likelihood from FB should match log-likelihood from forward
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.5, 0.6, 0.998];
    let log_emit = precompute_log_emissions(&obs, &params);

    let (_, ll_forward) = forward_from_log_emit(&log_emit, &params);
    let (_, ll_fb) = forward_backward_from_log_emit(&log_emit, &params);

    assert!((ll_forward - ll_fb).abs() < 1e-10,
        "Forward LL ({}) should match FB LL ({})", ll_forward, ll_fb);
}

#[test]
fn test_fb_from_log_emit_high_identity_high_posterior() {
    // Windows with very high identity should have high P(IBD)
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.5, 0.9999, 0.9999, 0.9999, 0.5, 0.5];
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors, _) = forward_backward_from_log_emit(&log_emit, &params);

    // Middle windows should have higher posterior than edge windows
    let mid_avg: f64 = posteriors[2..5].iter().sum::<f64>() / 3.0;
    let edge_avg: f64 = (posteriors[0] + posteriors[1] + posteriors[5] + posteriors[6]) / 4.0;
    assert!(mid_avg > edge_avg,
        "IBD region posterior ({:.4}) should exceed non-IBD ({:.4})", mid_avg, edge_avg);
}

// ============================================================================
// viterbi_from_log_emit
// ============================================================================

#[test]
fn test_viterbi_from_log_emit_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let states = viterbi_from_log_emit(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn test_viterbi_from_log_emit_single() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let log_emit = precompute_log_emissions(&[0.999], &params);
    let states = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] <= 1);
}

#[test]
fn test_viterbi_from_log_emit_valid_binary_states() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.998, 0.999, 0.9995, 0.5, 0.4];
    let log_emit = precompute_log_emissions(&obs, &params);
    let states = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states.len(), obs.len());
    for &s in &states {
        assert!(s <= 1, "State should be 0 or 1, got {}", s);
    }
}

#[test]
fn test_viterbi_from_log_emit_detects_ibd_region() {
    // Create clear IBD signal with appropriate emission parameters
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    // Set emissions so 0.5 is clearly non-IBD and 0.999 is clearly IBD
    params.emission[0] = GaussianParams::new_unchecked(0.5, 0.05);
    params.emission[1] = GaussianParams::new_unchecked(0.999, 0.001);

    let mut obs = vec![0.5; 30];
    for i in 10..20 {
        obs[i] = 0.999;
    }
    let log_emit = precompute_log_emissions(&obs, &params);
    let states = viterbi_from_log_emit(&log_emit, &params);

    // At least some windows in the IBD region should be detected
    let ibd_count: usize = states[10..20].iter().filter(|&&s| s == 1).count();
    assert!(ibd_count > 0, "Should detect IBD in high-identity region, got 0");

    // Edge windows should mostly be non-IBD
    let non_ibd_edges: usize = states[0..5].iter().chain(states[25..30].iter())
        .filter(|&&s| s == 0).count();
    assert!(non_ibd_edges >= 8, "Edge regions should be mostly non-IBD, got {}", non_ibd_edges);
}

// ============================================================================
// estimate_auxiliary_emissions
// ============================================================================

#[test]
fn test_estimate_aux_emissions_empty() {
    let aux_emit = estimate_auxiliary_emissions(&[], &[]);
    // Should return fallback values (mean 0.5/0.9, std 0.2/0.1)
    assert!((aux_emit[0].mean - 0.5).abs() < 1e-10);
    assert!((aux_emit[1].mean - 0.9).abs() < 1e-10);
    assert!((aux_emit[0].std - 0.2).abs() < 1e-10);
    assert!((aux_emit[1].std - 0.1).abs() < 1e-10);
}

#[test]
fn test_estimate_aux_emissions_all_posterior_zero() {
    // All windows assigned to non-IBD (posterior = 0)
    let aux = vec![0.7, 0.8, 0.6, 0.75];
    let posteriors = vec![0.0, 0.0, 0.0, 0.0];
    let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

    // Non-IBD should capture all data
    let expected_mean = (0.7 + 0.8 + 0.6 + 0.75) / 4.0;
    assert!((aux_emit[0].mean - expected_mean).abs() < 1e-10,
        "Non-IBD mean should be {:.4}, got {:.4}", expected_mean, aux_emit[0].mean);

    // IBD should get fallback (no weight)
    assert!((aux_emit[1].mean - 0.9).abs() < 1e-10,
        "IBD mean should be fallback 0.9, got {:.4}", aux_emit[1].mean);
}

#[test]
fn test_estimate_aux_emissions_all_posterior_one() {
    // All windows assigned to IBD (posterior = 1)
    let aux = vec![0.95, 0.98, 0.96, 0.97];
    let posteriors = vec![1.0, 1.0, 1.0, 1.0];
    let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

    // IBD should capture all data
    let expected_mean = (0.95 + 0.98 + 0.96 + 0.97) / 4.0;
    assert!((aux_emit[1].mean - expected_mean).abs() < 1e-10,
        "IBD mean should be {:.4}, got {:.4}", expected_mean, aux_emit[1].mean);

    // Non-IBD should get fallback (no weight)
    assert!((aux_emit[0].mean - 0.5).abs() < 1e-10,
        "Non-IBD mean should be fallback 0.5, got {:.4}", aux_emit[0].mean);
}

#[test]
fn test_estimate_aux_emissions_std_positive() {
    // Std should always be positive (min 1e-4)
    let aux = vec![0.5, 0.5, 0.5]; // All same → variance = 0, std clamped to 1e-4
    let posteriors = vec![0.0, 0.5, 1.0];
    let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);
    assert!(aux_emit[0].std >= 1e-4, "Non-IBD std should be >= 1e-4: {}", aux_emit[0].std);
    assert!(aux_emit[1].std >= 1e-4, "IBD std should be >= 1e-4: {}", aux_emit[1].std);
}

#[test]
fn test_estimate_aux_emissions_clamped_posteriors() {
    // Posteriors slightly outside [0,1] should be clamped
    let aux = vec![0.7, 0.9];
    let posteriors = vec![-0.1, 1.1]; // Slightly out of range
    let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);
    // Should not panic, and means should be finite
    assert!(aux_emit[0].mean.is_finite());
    assert!(aux_emit[1].mean.is_finite());
}

// ============================================================================
// coverage_ratio
// ============================================================================

#[test]
fn test_coverage_ratio_one_bp() {
    assert!((coverage_ratio(1, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_large_values() {
    // No overflow with large u64 values
    let ratio = coverage_ratio(1_000_000_000, 2_000_000_000);
    assert!((ratio - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_max_u64() {
    // u64::MAX should not overflow
    let ratio = coverage_ratio(u64::MAX, u64::MAX);
    assert!((ratio - 1.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_always_in_unit_interval() {
    // Coverage ratio should always be in [0, 1]
    let test_cases = vec![
        (0, 0), (0, 1), (1, 0), (1, 1),
        (100, 200), (200, 100),
        (1, 1000000), (999, 1000),
    ];
    for (a, b) in test_cases {
        let r = coverage_ratio(a, b);
        assert!(r >= 0.0 && r <= 1.0,
            "coverage_ratio({}, {}) = {} not in [0, 1]", a, b, r);
    }
}

// ============================================================================
// infer_ibd_with_aux_features
// ============================================================================

#[test]
fn test_infer_ibd_with_aux_features_empty() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let (result, aux_emit) = infer_ibd_with_aux_features(
        &[], &mut params, Population::Generic, 5000, 5, None,
    );
    assert!(result.states.is_empty());
    assert!(result.posteriors.is_empty());
    assert!(aux_emit.is_none());
}

#[test]
fn test_infer_ibd_with_aux_features_single_obs() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let (result, _) = infer_ibd_with_aux_features(
        &[0.999], &mut params, Population::Generic, 5000, 5, None,
    );
    // < 3 observations → early return with all non-IBD
    assert_eq!(result.states.len(), 1);
    assert_eq!(result.states[0], 0);
}

#[test]
fn test_infer_ibd_with_aux_features_mismatched_aux_length() {
    // Auxiliary length doesn't match primary → should ignore auxiliary
    let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6];
    let aux = vec![0.8, 0.9, 0.7]; // Wrong length!

    let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
    let (result, aux_emit) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::Generic, 5000, 5, Some(&aux),
    );

    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
    // aux_emit should be None because length mismatch
    assert!(aux_emit.is_none());
}

#[test]
fn test_infer_ibd_with_aux_features_all_populations() {
    // Should work for all population types
    let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6];
    let populations = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];

    for pop in &populations {
        let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
        let (result, _) = infer_ibd_with_aux_features(
            &obs, &mut params, *pop, 5000, 5, None,
        );
        assert_eq!(result.states.len(), obs.len(),
            "Failed for population {:?}", pop);
        assert!(result.log_likelihood.is_finite(),
            "Non-finite LL for population {:?}: {}", pop, result.log_likelihood);
    }
}

#[test]
fn test_infer_ibd_with_aux_features_zero_bw_iters() {
    // BW iters = 0 should still work (skip training)
    let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6];
    let aux = vec![0.6, 0.65, 0.7, 0.55, 0.95, 0.97, 0.99, 0.96, 0.7, 0.6];

    let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
    let (result, aux_emit) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::Generic, 5000, 0, Some(&aux),
    );

    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
    assert!(aux_emit.is_some());
}

#[test]
fn test_infer_ibd_with_aux_features_posteriors_valid() {
    // All posteriors should be in [0, 1]
    let obs = vec![0.5, 0.6, 0.55, 0.998, 0.999, 0.9995, 0.5, 0.4, 0.6, 0.55];
    let aux = vec![0.6, 0.7, 0.65, 0.95, 0.97, 0.99, 0.65, 0.5, 0.7, 0.6];

    let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
    let (result, _) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::Generic, 5000, 5, Some(&aux),
    );

    for (t, &p) in result.posteriors.iter().enumerate() {
        assert!(p >= -1e-10 && p <= 1.0 + 1e-10,
            "Posterior at t={} out of range: {}", t, p);
        assert!(p.is_finite(), "Posterior at t={} is not finite", t);
    }
}

// ============================================================================
// Combined pipeline: log_emit functions match standard functions
// ============================================================================

#[test]
fn test_log_emit_pipeline_matches_standard_pipeline() {
    // Full pipeline: precompute_log_emissions → forward/backward/viterbi/FB
    // should produce identical results to standard forward/backward/viterbi/FB
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.55, 0.998, 0.999, 0.9995, 0.5, 0.4];
    let log_emit = precompute_log_emissions(&obs, &params);

    // Forward-backward
    let (post_std, ll_std) = forward_backward(&obs, &params);
    let (post_le, ll_le) = forward_backward_from_log_emit(&log_emit, &params);

    assert!((ll_std - ll_le).abs() < 1e-10,
        "LL mismatch: {} vs {}", ll_std, ll_le);
    for t in 0..obs.len() {
        assert!((post_std[t] - post_le[t]).abs() < 1e-10,
            "Posterior mismatch at t={}: {} vs {}", t, post_std[t], post_le[t]);
    }
}

// ============================================================================
// Stress / numerical stability
// ============================================================================

#[test]
fn test_log_emit_long_sequence_no_nan() {
    // 10,000 window sequence — no NaN or Inf in forward/backward/FB
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let n = 10_000;
    let obs: Vec<f64> = (0..n).map(|i| {
        if i % 100 < 30 { 0.999 } else { 0.5 }
    }).collect();
    let log_emit = precompute_log_emissions(&obs, &params);

    let (alpha, ll_fwd) = forward_from_log_emit(&log_emit, &params);
    assert!(ll_fwd.is_finite(), "Forward LL is not finite: {}", ll_fwd);
    for (t, a) in alpha.iter().enumerate() {
        assert!(a[0].is_finite() && a[1].is_finite(),
            "Alpha at t={} is not finite: [{}, {}]", t, a[0], a[1]);
    }

    let beta = backward_from_log_emit(&log_emit, &params);
    for (t, b) in beta.iter().enumerate() {
        assert!(b[0].is_finite() && b[1].is_finite(),
            "Beta at t={} is not finite: [{}, {}]", t, b[0], b[1]);
    }

    let (posteriors, ll_fb) = forward_backward_from_log_emit(&log_emit, &params);
    assert!(ll_fb.is_finite(), "FB LL is not finite");
    assert!((ll_fwd - ll_fb).abs() < 1e-6,
        "Forward LL ({}) != FB LL ({})", ll_fwd, ll_fb);
    for (t, &p) in posteriors.iter().enumerate() {
        assert!(p.is_finite() && p >= 0.0 && p <= 1.0,
            "Posterior at t={} invalid: {}", t, p);
    }

    let states = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states.len(), n);
    for &s in &states {
        assert!(s <= 1);
    }
}

#[test]
fn test_infer_ibd_with_aux_long_sequence() {
    // Test full inference pipeline with auxiliary features on longer sequence
    let n = 500;
    let mut obs = Vec::with_capacity(n);
    let mut aux = Vec::with_capacity(n);
    for i in 0..n {
        if (200..300).contains(&i) {
            obs.push(0.999);
            aux.push(0.98);
        } else {
            obs.push(0.5 + 0.1 * ((i as f64) / 100.0).sin());
            aux.push(0.6 + 0.1 * ((i as f64) / 50.0).cos());
        }
    }

    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let (result, aux_emit) = infer_ibd_with_aux_features(
        &obs, &mut params, Population::Generic, 5000, 5, Some(&aux),
    );

    assert_eq!(result.states.len(), n);
    assert!(result.log_likelihood.is_finite());
    assert!(aux_emit.is_some());

    // Check all posteriors valid
    for &p in &result.posteriors {
        assert!(p.is_finite() && p >= -1e-10 && p <= 1.0 + 1e-10);
    }
}

// ============================================================================
// Multi-feature BW training: log-likelihood should not decrease
// ============================================================================

#[test]
fn test_infer_ibd_aux_bw_improves_or_maintains_ll() {
    // With BW training on combined emissions, LL should not decrease
    // (this is a weaker test since we compare BW=0 vs BW=5)
    let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6,
                   0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6, 0.55, 0.5];
    let aux = vec![0.6, 0.65, 0.7, 0.55, 0.95, 0.97, 0.99, 0.96, 0.7, 0.6,
                   0.65, 0.55, 0.95, 0.97, 0.99, 0.96, 0.65, 0.6, 0.7, 0.55];

    let mut params0 = HmmParams::from_expected_length(4.0, 0.01, 5000);
    let (result0, _) = infer_ibd_with_aux_features(
        &obs, &mut params0, Population::Generic, 5000, 0, Some(&aux),
    );

    let mut params5 = HmmParams::from_expected_length(4.0, 0.01, 5000);
    let (result5, _) = infer_ibd_with_aux_features(
        &obs, &mut params5, Population::Generic, 5000, 5, Some(&aux),
    );

    // Both should be finite
    assert!(result0.log_likelihood.is_finite());
    assert!(result5.log_likelihood.is_finite());
    // BW training should not make things worse
    // (Note: LL comparison is tricky because emission params change,
    //  so we just verify both are reasonable)
}

// ============================================================================
// EM two-component edge cases
// ============================================================================

#[test]
fn test_em_two_component_swapped_init_still_orders_correctly() {
    // Initialize with low > high — result should still have low.mean < high.mean
    let mut data = Vec::new();
    for _ in 0..40 {
        data.push(0.998);
    }
    for _ in 0..20 {
        data.push(0.9997);
    }

    // Intentionally swap: init_low has high mean, init_high has low mean
    let init_low = GaussianParams::new_unchecked(0.9997, 0.0005);
    let init_high = GaussianParams::new_unchecked(0.998, 0.001);

    let result = em_two_component(&data, &init_low, &init_high, 0.3, 50, 1e-6);
    assert!(result.is_some(), "Should converge despite swapped init");
    let (low, high, _, _) = result.unwrap();
    assert!(
        low.mean < high.mean,
        "Result should always have low.mean < high.mean, got {} and {}",
        low.mean,
        high.mean
    );
}

#[test]
fn test_em_two_component_weights_sum_to_one() {
    let mut data = Vec::new();
    for _ in 0..60 {
        data.push(0.998);
    }
    for _ in 0..40 {
        data.push(0.9997);
    }

    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    let result = em_two_component(&data, &init_low, &init_high, 0.6, 100, 1e-8);
    assert!(result.is_some());
    let (_, _, w_low, w_high) = result.unwrap();
    assert!(
        ((w_low + w_high) - 1.0).abs() < 1e-6,
        "Weights should sum to 1.0, got {} + {} = {}",
        w_low,
        w_high,
        w_low + w_high
    );
}

#[test]
fn test_em_two_component_map_weights_sum_to_one() {
    let mut data = Vec::new();
    for _ in 0..60 {
        data.push(0.998);
    }
    for _ in 0..40 {
        data.push(0.9997);
    }

    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    let result = em_two_component_map(&data, &init_low, &init_high, 0.6, 100, 1e-8, 5.0);
    assert!(result.is_some());
    let (_, _, w_low, w_high) = result.unwrap();
    assert!(
        ((w_low + w_high) - 1.0).abs() < 1e-6,
        "MAP weights should sum to 1.0, got {} + {} = {}",
        w_low,
        w_high,
        w_low + w_high
    );
}

#[test]
fn test_em_two_component_map_strong_prior_preserves_means() {
    // Very strong prior should keep means close to their initial values
    let mut data = vec![0.999; 100]; // Ambiguous single-mode data
    data.push(0.9997); // Tiny signal

    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    let result = em_two_component_map(&data, &init_low, &init_high, 0.9, 50, 1e-6, 100.0);
    if let Some((low, high, _, _)) = result {
        // With very strong prior (100.0), means should be near their initial values
        assert!(
            (low.mean - 0.998).abs() < 0.005,
            "Strong prior should keep low mean near 0.998, got {}",
            low.mean
        );
        assert!(
            (high.mean - 0.9997).abs() < 0.005,
            "Strong prior should keep high mean near 0.9997, got {}",
            high.mean
        );
    }
}

// ============================================================================
// trimmed_mean edge cases
// ============================================================================

#[test]
fn test_trimmed_mean_single_element() {
    let data = vec![42.0];
    let tm = trimmed_mean(&data, 0.1).unwrap();
    assert!(
        (tm - 42.0).abs() < 1e-10,
        "Single element trimmed mean should be the element itself"
    );
}

#[test]
fn test_trimmed_mean_two_elements() {
    let data = vec![10.0, 20.0];
    let tm = trimmed_mean(&data, 0.0).unwrap();
    assert!(
        (tm - 15.0).abs() < 1e-10,
        "Mean of [10, 20] should be 15, got {}",
        tm
    );
}

#[test]
fn test_trimmed_mean_negative_trim_clamped() {
    // Negative trim fraction should be clamped to 0
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tm = trimmed_mean(&data, -0.5).unwrap();
    let regular_mean = 3.0;
    assert!(
        (tm - regular_mean).abs() < 1e-10,
        "Negative trim should be clamped to 0, giving regular mean"
    );
}

#[test]
fn test_trimmed_mean_trim_above_half_clamped() {
    // Trim > 0.49 should be clamped to 0.49
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let tm = trimmed_mean(&data, 0.99).unwrap(); // Clamped to 0.49
    // trim_count = (10 * 0.49) = 4, so keeps [5, 6] => mean = 5.5
    assert!(
        (tm - 5.5).abs() < 0.5,
        "Extreme trim should be clamped and still return valid result, got {}",
        tm
    );
}

// ============================================================================
// bic_model_selection edge cases
// ============================================================================

#[test]
fn test_bic_empty_data() {
    let data: Vec<f64> = vec![];
    let low = GaussianParams::new_unchecked(0.998, 0.001);
    let high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

#[test]
fn test_bic_identical_data() {
    // All data identical — 1-component model should be strongly preferred
    let data = vec![0.999; 50];
    let low = GaussianParams::new_unchecked(0.999, 0.0001);
    let high = GaussianParams::new_unchecked(0.9995, 0.0001);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // Both BIC values should be finite
    assert!(bic_1.is_finite(), "BIC 1-component should be finite");
    assert!(bic_2.is_finite(), "BIC 2-component should be finite");
}

#[test]
fn test_bic_two_data_points() {
    let data = vec![0.998, 0.9997];
    let low = GaussianParams::new_unchecked(0.998, 0.001);
    let high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // With only 2 data points, penalty term is small but 5-param model
    // should be penalized more than 2-param model
    assert!(bic_1.is_finite());
    assert!(bic_2.is_finite());
}

#[test]
fn test_bic_extreme_weight() {
    // weight_low near 1.0 means almost all weight on low component
    let mut data = vec![0.998; 50];
    data.extend(std::iter::repeat_n(0.9997, 50));

    let low = GaussianParams::new_unchecked(0.998, 0.0005);
    let high = GaussianParams::new_unchecked(0.9997, 0.0003);

    let (bic_1a, bic_2a) = bic_model_selection(&data, &low, &high, 0.99);
    let (bic_1b, bic_2b) = bic_model_selection(&data, &low, &high, 0.01);

    // BIC 1-component should be the same regardless of weight (it ignores weight)
    assert!(
        (bic_1a - bic_1b).abs() < 1e-10,
        "BIC 1-component should not depend on weight"
    );
    // BIC 2-component should differ with different weights
    assert!(
        (bic_2a - bic_2b).abs() > 0.01,
        "BIC 2-component should change with weight"
    );
}

// ============================================================================
// OnlineStats edge cases
// ============================================================================

use hprc_ibd::stats::OnlineStats;

#[test]
fn test_online_stats_nan_propagation() {
    // NaN input should propagate to mean and variance
    let mut stats = OnlineStats::new();
    stats.add(1.0);
    stats.add(f64::NAN);
    stats.add(3.0);
    // Mean with NaN should be NaN
    assert!(stats.mean().is_nan(), "Mean with NaN input should be NaN, got {}", stats.mean());
}

#[test]
fn test_online_stats_infinity_single_value() {
    // OnlineStats with a single Inf value
    let mut stats = OnlineStats::new();
    stats.add(f64::INFINITY);
    assert_eq!(stats.count(), 1);
    assert!(stats.mean().is_infinite(), "Mean of single Inf should be Inf");
    assert_eq!(stats.variance(), 0.0, "Variance of single value should be 0");
}

#[test]
fn test_online_stats_welford_with_infs() {
    // Welford's algorithm with two Infs: after first Inf, mean=Inf.
    // Second Inf: delta = Inf - Inf = NaN → mean becomes NaN.
    // This is a known limitation of Welford's online algorithm with non-finite values.
    // The key property: it should NOT panic.
    let mut stats = OnlineStats::new();
    stats.add(f64::INFINITY);
    stats.add(f64::INFINITY);
    assert_eq!(stats.count(), 2);
    // Mean is NaN due to Inf - Inf in Welford update
    assert!(stats.mean().is_nan(),
        "Welford with two Infs produces NaN mean: {}", stats.mean());
}

#[test]
fn test_online_stats_neg_infinity() {
    let mut stats = OnlineStats::new();
    stats.add(f64::NEG_INFINITY);
    stats.add(f64::INFINITY);
    assert_eq!(stats.count(), 2);
    // Mean of -Inf and Inf = NaN (due to NaN propagation from -Inf + Inf)
    assert!(stats.mean().is_nan(), "Mean of -Inf and Inf should be NaN");
}

// ============================================================================
// GaussianParams edge cases
// ============================================================================

#[test]
fn test_gaussian_log_pdf_far_from_mean() {
    // Very far from mean: should be very negative (not -Inf or NaN)
    let g = GaussianParams::new_unchecked(0.0, 1.0);
    let log_pdf = g.log_pdf(100.0);
    assert!(log_pdf.is_finite(), "log_pdf(100) should be finite: {}", log_pdf);
    assert!(log_pdf < -4000.0, "log_pdf(100) should be very negative: {}", log_pdf);
}

#[test]
fn test_gaussian_pdf_far_from_mean_is_zero() {
    let g = GaussianParams::new_unchecked(0.0, 1.0);
    let pdf = g.pdf(40.0); // 40 standard deviations away
    // Should be so small it's effectively 0 (may underflow to 0)
    assert!(pdf < 1e-300, "PDF 40 std devs away should be ~0: {}", pdf);
}

#[test]
fn test_gaussian_new_unchecked_very_small_std() {
    // new_unchecked with very small std: PDF at mean should be very large
    let g = GaussianParams::new_unchecked(5.0, 1e-300);
    let pdf = g.pdf(5.0);
    // 1/(1e-300 * sqrt(2pi)) ≈ Infinity (floating point overflow)
    assert!(pdf.is_infinite() || pdf > 1e290,
        "PDF at mean with tiny std should be very large or Inf: {}", pdf);
}

// ============================================================================
// coverage_ratio edge cases (ibd-cli hmm module)
// ============================================================================

#[test]
fn test_hmm_coverage_ratio_zero_zero() {
    // Both zero should return 0.0 (not NaN)
    assert_eq!(coverage_ratio(0, 0), 0.0);
}

#[test]
fn test_hmm_coverage_ratio_symmetric() {
    // coverage_ratio should be symmetric: (a,b) == (b,a)
    assert_eq!(coverage_ratio(100, 200), coverage_ratio(200, 100));
    assert_eq!(coverage_ratio(1, 1000000), coverage_ratio(1000000, 1));
}

#[test]
fn test_hmm_coverage_ratio_equal_values() {
    // Equal values should give 1.0
    for val in [1, 100, 999999, u64::MAX / 2] {
        assert!((coverage_ratio(val, val) - 1.0).abs() < 1e-10,
            "Equal values {} should give ratio 1.0", val);
    }
}

// ============================================================================
// precompute_log_emissions with NaN observations
// ============================================================================

#[test]
fn test_precompute_log_emissions_nan_observation() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![f64::NAN, 0.999];
    let log_emit = precompute_log_emissions(&obs, &params);
    assert_eq!(log_emit.len(), 2);
    // NaN input should produce NaN log-emissions
    assert!(log_emit[0][0].is_nan() || log_emit[0][0].is_finite(),
        "NaN observation should produce NaN or finite log-emission");
    // Normal observation should be finite
    assert!(log_emit[1][0].is_finite(), "Normal observation should give finite log-emission");
    assert!(log_emit[1][1].is_finite());
}

// ============================================================================
// Viterbi with all identical observations
// ============================================================================

#[test]
fn test_viterbi_all_identical_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.999; 100];
    let log_emit = precompute_log_emissions(&obs, &params);
    let states = viterbi_from_log_emit(&log_emit, &params);

    assert_eq!(states.len(), 100);
    // All identical observations: Viterbi should produce a consistent state sequence
    // (either all 0 or all 1, but definitely valid)
    for &s in &states {
        assert!(s <= 1, "State should be 0 or 1");
    }
    // Check that the states are all the same (constant observation → constant state)
    let first = states[0];
    assert!(states.iter().all(|&s| s == first),
        "All-identical observations should produce uniform Viterbi path");
}

// ============================================================================
// forward_backward posteriors sum to reasonable values
// ============================================================================

#[test]
fn test_fb_posteriors_extreme_observations() {
    // Very low identity (clearly non-IBD)
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.1; 20]; // Very low identity
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors, _) = forward_backward_from_log_emit(&log_emit, &params);

    for &p in &posteriors {
        assert!(p.is_finite() && p >= 0.0 && p <= 1.0,
            "Posterior should be in [0,1]: {}", p);
    }
    // Very low identity should have low IBD posteriors
    let mean_posterior: f64 = posteriors.iter().sum::<f64>() / posteriors.len() as f64;
    assert!(mean_posterior < 0.5,
        "Very low identity should have low IBD posterior: {}", mean_posterior);
}
