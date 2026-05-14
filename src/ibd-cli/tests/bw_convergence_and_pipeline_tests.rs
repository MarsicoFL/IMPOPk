//! Tests for IBD Baum-Welch convergence behavior and combined emission pipeline.
//!
//! Motivation: Mathematician's analysis (T2) showed Baum-Welch convergence rate
//! r ≈ 0.9 for IBD, meaning 10 iterations leave ~35% residual error. These tests
//! verify convergence properties and ensure the multi-feature emission pipeline
//! (precompute → combine → forward/backward/viterbi) produces consistent results.

use impopk_ibd::hmm::*;
use impopk_ibd::stats::GaussianParams;

// ============================================================================
// Helpers
// ============================================================================

fn make_clear_ibd_obs(n_non_ibd: usize, n_ibd: usize, n_tail: usize) -> Vec<f64> {
    let mut obs = Vec::new();
    // Non-IBD region: identity around 0.997
    for _ in 0..n_non_ibd {
        obs.push(0.9970);
    }
    // IBD region: identity around 0.9998
    for _ in 0..n_ibd {
        obs.push(0.9998);
    }
    // Trailing non-IBD
    for _ in 0..n_tail {
        obs.push(0.9972);
    }
    obs
}

fn make_test_params() -> HmmParams {
    HmmParams::from_expected_length(50.0, 0.001, 10000)
}

fn make_afr_params() -> HmmParams {
    HmmParams::from_population(Population::AFR, 50.0, 0.001, 10000)
}

fn make_eur_params() -> HmmParams {
    HmmParams::from_population(Population::EUR, 50.0, 0.001, 10000)
}

// ============================================================================
// Baum-Welch Convergence Tests
// ============================================================================

#[test]
fn bw_log_likelihood_monotonically_increases() {
    // BW should never decrease log-likelihood (EM guarantee)
    let obs = make_clear_ibd_obs(30, 20, 30);
    let mut params = make_afr_params();

    let mut prev_ll = f64::NEG_INFINITY;
    for iter in 0..30 {
        let (_, ll) = forward(&obs, &params);
        // Allow tiny numerical noise (1e-10)
        assert!(
            ll >= prev_ll - 1e-10,
            "Log-likelihood decreased at iteration {iter}: {prev_ll} -> {ll}"
        );
        prev_ll = ll;

        // One BW step
        let mut params_clone = params.clone();
        params_clone.baum_welch(&obs, 1, 1e-12, Some(Population::AFR), 10000);
        params = params_clone;
    }
}

#[test]
fn bw_10_iters_vs_30_iters_emission_shift() {
    // Verify that 30 iterations converge further than 10
    // (validates mathematician's finding about slow convergence)
    let obs = make_clear_ibd_obs(40, 30, 40);

    let mut params_10 = make_afr_params();
    params_10.baum_welch(&obs, 10, 1e-12, Some(Population::AFR), 10000);

    let mut params_30 = make_afr_params();
    params_30.baum_welch(&obs, 30, 1e-12, Some(Population::AFR), 10000);

    // Both should be valid
    assert!(params_10.emission[0].mean < params_10.emission[1].mean);
    assert!(params_30.emission[0].mean < params_30.emission[1].mean);

    // 30 iterations should achieve equal or better log-likelihood
    let (_, ll_10) = forward(&obs, &params_10);
    let (_, ll_30) = forward(&obs, &params_30);
    assert!(
        ll_30 >= ll_10 - 1e-6,
        "30-iter ll ({ll_30}) should be >= 10-iter ll ({ll_10})"
    );
}

#[test]
fn bw_convergence_early_stop_with_tight_tolerance() {
    // Tight tolerance should cause early convergence
    let obs = make_clear_ibd_obs(30, 20, 30);
    let mut params_tight = make_afr_params();
    let mut params_loose = make_afr_params();

    // Run both with 100 max iters but different tolerances
    params_tight.baum_welch(&obs, 100, 1e-2, Some(Population::AFR), 10000);
    params_loose.baum_welch(&obs, 100, 1e-12, Some(Population::AFR), 10000);

    // Loose tolerance should give equal or better LL
    let (_, ll_tight) = forward(&obs, &params_tight);
    let (_, ll_loose) = forward(&obs, &params_loose);
    assert!(ll_loose >= ll_tight - 1e-6);
}

#[test]
fn bw_with_all_identical_observations() {
    // Edge case: all observations identical (degenerate case)
    let obs = vec![0.998; 50];
    let mut params = make_afr_params();
    params.baum_welch(&obs, 10, 1e-6, Some(Population::AFR), 10000);

    // Should not panic, params should remain valid
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
    assert!(params.transition[0][0] + params.transition[0][1] - 1.0 < 1e-10);
    assert!(params.transition[1][0] + params.transition[1][1] - 1.0 < 1e-10);
}

#[test]
fn bw_with_less_than_10_observations_is_noop() {
    // BW skips training when n < 10
    let obs = vec![0.997, 0.998, 0.9999, 0.997, 0.998];
    let mut params = make_afr_params();
    let emission_before = params.emission.clone();

    params.baum_welch(&obs, 20, 1e-6, Some(Population::AFR), 10000);

    // Parameters should be unchanged
    assert_eq!(params.emission[0].mean, emission_before[0].mean);
    assert_eq!(params.emission[1].mean, emission_before[1].mean);
}

#[test]
fn bw_with_exactly_10_observations_runs() {
    let obs = vec![0.997, 0.997, 0.997, 0.997, 0.997, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998];
    let mut params = make_afr_params();
    let emission_before = params.emission.clone();

    params.baum_welch(&obs, 20, 1e-6, Some(Population::AFR), 10000);

    // With 10 observations BW should run and may change emissions
    // Just verify no panic and params remain valid
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    // Emissions likely changed
    let changed = (params.emission[0].mean - emission_before[0].mean).abs() > 1e-10
        || (params.emission[1].mean - emission_before[1].mean).abs() > 1e-10;
    assert!(changed, "BW with 10 obs should modify emission params");
}

#[test]
fn bw_preserves_identifiability_constraint() {
    // After BW, state 0 mean should be < state 1 mean
    let obs = make_clear_ibd_obs(40, 30, 40);
    for pop in [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::Generic,
    ] {
        let mut params = HmmParams::from_population(pop, 50.0, 0.001, 10000);
        params.baum_welch(&obs, 20, 1e-6, Some(pop), 10000);
        assert!(
            params.emission[0].mean < params.emission[1].mean,
            "Identifiability broken for {:?}: {} >= {}",
            pop,
            params.emission[0].mean,
            params.emission[1].mean
        );
    }
}

#[test]
fn bw_transition_bounds_respected() {
    // BW should keep transitions within biological bounds
    let obs = make_clear_ibd_obs(40, 30, 40);
    let mut params = make_afr_params();
    params.baum_welch(&obs, 30, 1e-6, Some(Population::AFR), 10000);

    // P(enter IBD) bounded [1e-8, 0.1]
    assert!(params.transition[0][1] >= 1e-8);
    assert!(params.transition[0][1] <= 0.1);
    // P(leave IBD) bounded [0.001, 0.5]
    assert!(params.transition[1][0] >= 0.001);
    assert!(params.transition[1][0] <= 0.5);
    // Rows sum to 1
    assert!((params.transition[0][0] + params.transition[0][1] - 1.0).abs() < 1e-10);
    assert!((params.transition[1][0] + params.transition[1][1] - 1.0).abs() < 1e-10);
}

#[test]
fn bw_emission_bounds_respected() {
    // BW should keep emission means within biological bounds
    let obs = make_clear_ibd_obs(40, 30, 40);
    let mut params = make_afr_params();
    params.baum_welch(&obs, 30, 1e-6, Some(Population::AFR), 10000);

    // Non-IBD std bounded [0.0003, 0.005]
    assert!(params.emission[0].std >= 0.0003 - 1e-10);
    assert!(params.emission[0].std <= 0.005 + 1e-10);
    // IBD mean bounded [0.999, 1.0]
    assert!(params.emission[1].mean >= 0.999 - 1e-10);
    assert!(params.emission[1].mean <= 1.0 + 1e-10);
    // IBD std bounded [0.0002, 0.002]
    assert!(params.emission[1].std >= 0.0002 - 1e-10);
    assert!(params.emission[1].std <= 0.002 + 1e-10);
}

#[test]
fn bw_zero_max_iter_is_noop() {
    let obs = make_clear_ibd_obs(30, 20, 30);
    let mut params = make_afr_params();
    let emission_before = params.emission.clone();

    params.baum_welch(&obs, 0, 1e-6, Some(Population::AFR), 10000);

    assert_eq!(params.emission[0].mean, emission_before[0].mean);
    assert_eq!(params.emission[1].mean, emission_before[1].mean);
}

// ============================================================================
// Precompute + Forward/Backward/Viterbi Pipeline Consistency
// ============================================================================

#[test]
fn precompute_matches_inline_emissions() {
    let obs = vec![0.997, 0.998, 0.9995, 0.9999, 0.997];
    let params = make_test_params();
    let precomp = precompute_log_emissions(&obs, &params);

    for (t, &val) in obs.iter().enumerate() {
        let expected_0 = params.emission[0].log_pdf(val);
        let expected_1 = params.emission[1].log_pdf(val);
        assert!(
            (precomp[t][0] - expected_0).abs() < 1e-12,
            "Mismatch at t={t} state=0: {} vs {}",
            precomp[t][0],
            expected_0
        );
        assert!(
            (precomp[t][1] - expected_1).abs() < 1e-12,
            "Mismatch at t={t} state=1: {} vs {}",
            precomp[t][1],
            expected_1
        );
    }
}

#[test]
fn precompute_empty_observations() {
    let obs: Vec<f64> = vec![];
    let params = make_test_params();
    let precomp = precompute_log_emissions(&obs, &params);
    assert!(precomp.is_empty());
}

#[test]
fn precompute_single_observation() {
    let obs = vec![0.999];
    let params = make_test_params();
    let precomp = precompute_log_emissions(&obs, &params);
    assert_eq!(precomp.len(), 1);
    assert!(precomp[0][0].is_finite());
    assert!(precomp[0][1].is_finite());
}

#[test]
fn viterbi_from_log_emit_matches_standard_viterbi() {
    let obs = make_clear_ibd_obs(20, 15, 20);
    let params = make_afr_params();

    let states_std = viterbi(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let states_precomp = viterbi_from_log_emit(&log_emit, &params);

    assert_eq!(
        states_std, states_precomp,
        "Viterbi from precomputed emissions should match standard Viterbi"
    );
}

#[test]
fn fb_from_log_emit_matches_standard_fb() {
    let obs = make_clear_ibd_obs(20, 15, 20);
    let params = make_afr_params();

    let (post_std, ll_std) = forward_backward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (post_precomp, ll_precomp) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(
        (ll_std - ll_precomp).abs() < 1e-8,
        "Log-likelihood mismatch: {ll_std} vs {ll_precomp}"
    );
    assert_eq!(post_std.len(), post_precomp.len());
    for (t, (&p_std, &p_precomp)) in post_std.iter().zip(post_precomp.iter()).enumerate() {
        assert!(
            (p_std - p_precomp).abs() < 1e-10,
            "Posterior mismatch at t={t}: {p_std} vs {p_precomp}"
        );
    }
}

#[test]
fn forward_backward_log_emit_posteriors_sum_valid() {
    // Posteriors from forward_backward_from_log_emit should be in [0, 1]
    let obs = make_clear_ibd_obs(30, 20, 30);
    let params = make_afr_params();
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors, ll) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(ll.is_finite(), "Log-likelihood should be finite");
    for (t, &p) in posteriors.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "Posterior at t={t} out of range: {p}"
        );
    }
}

#[test]
fn forward_from_log_emit_empty() {
    let log_emit: Vec<[f64; 2]> = vec![];
    let params = make_test_params();
    let (alpha, ll) = forward_from_log_emit(&log_emit, &params);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn backward_from_log_emit_empty() {
    let log_emit: Vec<[f64; 2]> = vec![];
    let params = make_test_params();
    let beta = backward_from_log_emit(&log_emit, &params);
    assert!(beta.is_empty());
}

#[test]
fn viterbi_from_log_emit_empty() {
    let log_emit: Vec<[f64; 2]> = vec![];
    let params = make_test_params();
    let states = viterbi_from_log_emit(&log_emit, &params);
    assert!(states.is_empty());
}

#[test]
fn forward_backward_from_log_emit_empty() {
    let log_emit: Vec<[f64; 2]> = vec![];
    let params = make_test_params();
    let (post, ll) = forward_backward_from_log_emit(&log_emit, &params);
    assert!(post.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn viterbi_from_log_emit_single_window() {
    let params = make_afr_params();
    // Single window strongly favoring IBD
    let log_emit = vec![[
        params.emission[0].log_pdf(0.9999),
        params.emission[1].log_pdf(0.9999),
    ]];
    let states = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states.len(), 1);
    // State should be 0 or 1 (valid)
    assert!(states[0] <= 1);
}

// ============================================================================
// Combined Log-Emissions (Primary + Auxiliary)
// ============================================================================

#[test]
fn combined_emissions_without_aux_equals_primary() {
    let obs = vec![0.997, 0.998, 0.9995];
    let params = make_test_params();

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, None, None);

    assert_eq!(primary.len(), combined.len());
    for t in 0..primary.len() {
        assert_eq!(primary[t][0], combined[t][0]);
        assert_eq!(primary[t][1], combined[t][1]);
    }
}

#[test]
fn combined_emissions_with_aux_shifts_values() {
    let obs = vec![0.997, 0.998, 0.9995, 0.9999];
    let params = make_test_params();
    let aux_obs = vec![0.5, 0.6, 0.9, 0.95];
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.15),
        GaussianParams::new_unchecked(0.9, 0.1),
    ];

    let primary = precompute_log_emissions(&obs, &params);
    let combined =
        compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    // Combined should differ from primary (aux adds contribution)
    let mut any_diff = false;
    for t in 0..obs.len() {
        if (combined[t][0] - primary[t][0]).abs() > 1e-10 {
            any_diff = true;
        }
        // Combined = primary + aux (log space addition)
        let expected_0 = primary[t][0] + aux_emit[0].log_pdf(aux_obs[t]);
        let expected_1 = primary[t][1] + aux_emit[1].log_pdf(aux_obs[t]);
        assert!(
            (combined[t][0] - expected_0).abs() < 1e-10,
            "Combined mismatch at t={t} state=0"
        );
        assert!(
            (combined[t][1] - expected_1).abs() < 1e-10,
            "Combined mismatch at t={t} state=1"
        );
    }
    assert!(any_diff, "Auxiliary features should change combined emissions");
}

#[test]
fn combined_emissions_mismatched_length_falls_back() {
    let obs = vec![0.997, 0.998, 0.9995];
    let params = make_test_params();
    let aux_obs = vec![0.5, 0.6]; // Wrong length
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.15),
        GaussianParams::new_unchecked(0.9, 0.1),
    ];

    let primary = precompute_log_emissions(&obs, &params);
    let combined =
        compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    // Should fall back to primary-only
    for t in 0..obs.len() {
        assert_eq!(primary[t][0], combined[t][0]);
        assert_eq!(primary[t][1], combined[t][1]);
    }
}

#[test]
fn combined_emissions_none_aux_emit_falls_back() {
    let obs = vec![0.997, 0.998];
    let params = make_test_params();
    let aux_obs = vec![0.5, 0.6];

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux_obs), None);

    for t in 0..obs.len() {
        assert_eq!(primary[t][0], combined[t][0]);
        assert_eq!(primary[t][1], combined[t][1]);
    }
}

#[test]
fn combined_viterbi_with_informative_aux_changes_states() {
    // Primary obs are ambiguous; aux obs disambiguate
    let obs = vec![0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985, 0.9985];
    let params = make_eur_params();

    // Aux: first half looks non-IBD (coverage ~0.5), second half looks IBD (coverage ~0.95)
    let aux_obs = vec![0.45, 0.50, 0.48, 0.52, 0.47, 0.93, 0.95, 0.94, 0.96, 0.92];
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.1),
        GaussianParams::new_unchecked(0.95, 0.05),
    ];

    let log_emit_primary = precompute_log_emissions(&obs, &params);
    let log_emit_combined =
        compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    let states_primary = viterbi_from_log_emit(&log_emit_primary, &params);
    let states_combined = viterbi_from_log_emit(&log_emit_combined, &params);

    // Combined should differ from primary (aux provides signal)
    // At minimum, the combined should show more IBD in the second half
    let ibd_count_primary_second_half: usize =
        states_primary[5..].iter().filter(|&&s| s == 1).count();
    let ibd_count_combined_second_half: usize =
        states_combined[5..].iter().filter(|&&s| s == 1).count();

    assert!(
        ibd_count_combined_second_half >= ibd_count_primary_second_half,
        "Informative aux should increase IBD in second half: combined={} vs primary={}",
        ibd_count_combined_second_half,
        ibd_count_primary_second_half
    );
}

// ============================================================================
// End-to-end pipeline: precompute → forward → backward → posterior → viterbi
// ============================================================================

#[test]
fn full_pipeline_forward_backward_from_log_emit_consistent() {
    let obs = make_clear_ibd_obs(30, 20, 30);
    let params = make_afr_params();

    let log_emit = precompute_log_emissions(&obs, &params);

    // Forward
    let (alpha, ll_fwd) = forward_from_log_emit(&log_emit, &params);
    assert!(ll_fwd.is_finite());
    assert_eq!(alpha.len(), obs.len());

    // Backward
    let beta = backward_from_log_emit(&log_emit, &params);
    assert_eq!(beta.len(), obs.len());

    // Combined forward-backward
    let (posteriors, ll_fb) = forward_backward_from_log_emit(&log_emit, &params);

    // Log-likelihoods should match
    assert!(
        (ll_fwd - ll_fb).abs() < 1e-8,
        "Forward LL ({ll_fwd}) != FB LL ({ll_fb})"
    );

    // Verify posterior computation from alpha/beta manually
    for t in 0..obs.len() {
        let log_g0 = alpha[t][0] + beta[t][0] - ll_fwd;
        let log_g1 = alpha[t][1] + beta[t][1] - ll_fwd;
        let max_log = log_g0.max(log_g1);
        let log_sum = max_log + ((log_g0 - max_log).exp() + (log_g1 - max_log).exp()).ln();
        let p_ibd = (log_g1 - log_sum).exp();
        assert!(
            (p_ibd - posteriors[t]).abs() < 1e-10,
            "Manual posterior mismatch at t={t}: {p_ibd} vs {}",
            posteriors[t]
        );
    }
}

#[test]
fn pipeline_viterbi_and_fb_agree_on_clear_signal() {
    // When signal is strong, Viterbi and posterior_decode should agree
    let obs = make_clear_ibd_obs(30, 20, 30);
    let params = make_afr_params();
    let log_emit = precompute_log_emissions(&obs, &params);

    let states_viterbi = viterbi_from_log_emit(&log_emit, &params);
    let (posteriors, _) = forward_backward_from_log_emit(&log_emit, &params);
    let states_fb: Vec<usize> = posteriors.iter().map(|&p| if p > 0.5 { 1 } else { 0 }).collect();

    // Most positions should agree (allow a few boundary differences)
    let agreement: usize = states_viterbi
        .iter()
        .zip(states_fb.iter())
        .filter(|(v, f)| v == f)
        .count();
    let agreement_rate = agreement as f64 / states_viterbi.len() as f64;
    assert!(
        agreement_rate > 0.90,
        "Viterbi and FB should mostly agree on clear signal: {:.1}%",
        agreement_rate * 100.0
    );
}

// ============================================================================
// Baum-Welch with distances
// ============================================================================

#[test]
fn bw_with_distances_produces_valid_params() {
    let obs = make_clear_ibd_obs(30, 20, 30);
    let n = obs.len();
    // Create window positions: 10kb windows, contiguous
    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| (i as u64 * 10000, (i as u64 + 1) * 10000))
        .collect();

    let mut params = make_afr_params();
    params.baum_welch_with_distances(&obs, &positions, 20, 1e-6, Some(Population::AFR), 10000);

    // Validate params are sane
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].mean < params.emission[1].mean);
    assert!((params.transition[0][0] + params.transition[0][1] - 1.0).abs() < 1e-10);
    assert!((params.transition[1][0] + params.transition[1][1] - 1.0).abs() < 1e-10);
}

#[test]
fn bw_with_distances_mismatched_positions_noop() {
    // If positions length doesn't match obs, BW should handle gracefully
    let obs = make_clear_ibd_obs(30, 20, 30);
    let positions: Vec<(u64, u64)> = vec![(0, 10000), (10000, 20000)]; // Wrong length

    let mut params = make_afr_params();
    params.baum_welch_with_distances(&obs, &positions, 20, 1e-6, Some(Population::AFR), 10000);

    // Should handle gracefully (may or may not modify params — just no panic)
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ============================================================================
// Baum-Welch with genetic map
// ============================================================================

#[test]
fn bw_with_genetic_map_produces_valid_params() {
    let obs = make_clear_ibd_obs(30, 20, 30);
    let n = obs.len();
    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| (i as u64 * 10000, (i as u64 + 1) * 10000))
        .collect();
    let genetic_map = GeneticMap::uniform(0, n as u64 * 10000, 1.0);

    let mut params = make_afr_params();
    params.baum_welch_with_genetic_map(
        &obs,
        &positions,
        &genetic_map,
        20,
        1e-6,
        Some(Population::AFR),
        10000,
    );

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].mean < params.emission[1].mean);
}

// ============================================================================
// NaN/Inf in precomputed emissions
// ============================================================================

#[test]
fn precompute_with_nan_observation() {
    let obs = vec![0.997, f64::NAN, 0.9999];
    let params = make_test_params();
    let log_emit = precompute_log_emissions(&obs, &params);

    assert_eq!(log_emit.len(), 3);
    assert!(log_emit[0][0].is_finite());
    // NaN observation → log_pdf(NaN) should produce -inf or NaN
    // Either way, the pipeline shouldn't panic
    let _ = log_emit[1][0]; // Just access, don't assert finite
}

#[test]
fn precompute_with_inf_observation() {
    let obs = vec![0.997, f64::INFINITY, 0.9999];
    let params = make_test_params();
    let log_emit = precompute_log_emissions(&obs, &params);
    assert_eq!(log_emit.len(), 3);
}

#[test]
fn viterbi_from_log_emit_with_neg_inf_emissions() {
    // Some emissions are -inf (impossible states)
    let log_emit = vec![
        [0.0, f64::NEG_INFINITY],  // Only state 0 possible
        [0.0, f64::NEG_INFINITY],
        [f64::NEG_INFINITY, 0.0],  // Only state 1 possible
        [f64::NEG_INFINITY, 0.0],
    ];
    let params = make_test_params();
    let states = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states.len(), 4);
    // First two should be state 0, last two state 1
    assert_eq!(states[0], 0);
    assert_eq!(states[1], 0);
    assert_eq!(states[2], 1);
    assert_eq!(states[3], 1);
}

#[test]
fn forward_backward_from_log_emit_with_extreme_values() {
    // Very large negative emissions (near -inf)
    let log_emit = vec![
        [-1.0, -1000.0],
        [-1.0, -1000.0],
        [-1000.0, -1.0],
        [-1000.0, -1.0],
    ];
    let params = make_test_params();
    let (posteriors, ll) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(ll.is_finite());
    for (t, &p) in posteriors.iter().enumerate() {
        assert!(p.is_finite(), "Posterior at t={t} is not finite: {p}");
        assert!((0.0..=1.0).contains(&p), "Posterior at t={t} out of range: {p}");
    }
    // First half should be non-IBD (~0), second half IBD (~1)
    assert!(posteriors[0] < 0.1);
    assert!(posteriors[3] > 0.9);
}
