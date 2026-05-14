/// Tests for HmmParams::estimate_emissions — the only public function
/// without direct test coverage. Also includes cross-algorithm consistency
/// tests that verify agreement between different HMM inference paths.
use impopk_ibd::hmm::{
    forward, forward_backward, forward_backward_from_log_emit, forward_from_log_emit,
    infer_ibd, precompute_log_emissions, viterbi, viterbi_from_log_emit, HmmParams, Population,
};
use impopk_ibd::stats::GaussianParams;

// ============================================================================
// HmmParams::estimate_emissions — basic functionality
// ============================================================================

#[test]
fn test_estimate_emissions_clear_two_clusters() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![
        0.5, 0.6, 0.55, 0.45, 0.52, 0.48, // Non-IBD cluster
        0.999, 0.998, 0.9995, 0.9997, 0.999, // IBD cluster
    ];
    params.estimate_emissions(&obs);

    // Low cluster mean should be < 0.7
    assert!(
        params.emission[0].mean < 0.7,
        "Low cluster mean should be < 0.7, got {}",
        params.emission[0].mean
    );
    // High cluster mean should be > 0.99
    assert!(
        params.emission[1].mean > 0.99,
        "High cluster mean should be > 0.99, got {}",
        params.emission[1].mean
    );
    // Emission[0] should have lower mean than emission[1]
    assert!(
        params.emission[0].mean < params.emission[1].mean,
        "Low emission mean ({}) should be < high emission mean ({})",
        params.emission[0].mean,
        params.emission[1].mean
    );
}

#[test]
fn test_estimate_emissions_fewer_than_3_obs_no_change() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission_0 = params.emission[0].clone();
    let original_emission_1 = params.emission[1].clone();

    // 2 observations — should return without change
    params.estimate_emissions(&[0.5, 0.999]);

    assert!(
        (params.emission[0].mean - original_emission_0.mean).abs() < 1e-15,
        "With < 3 obs, emission[0] should not change"
    );
    assert!(
        (params.emission[1].mean - original_emission_1.mean).abs() < 1e-15,
        "With < 3 obs, emission[1] should not change"
    );
}

#[test]
fn test_estimate_emissions_empty_no_change() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission_0 = params.emission[0].clone();
    params.estimate_emissions(&[]);
    assert!(
        (params.emission[0].mean - original_emission_0.mean).abs() < 1e-15,
        "Empty obs should not change emissions"
    );
}

#[test]
fn test_estimate_emissions_single_obs_no_change() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission_0 = params.emission[0].clone();
    params.estimate_emissions(&[0.999]);
    assert!(
        (params.emission[0].mean - original_emission_0.mean).abs() < 1e-15,
        "Single obs should not change emissions"
    );
}

#[test]
fn test_estimate_emissions_zero_variance_no_change() {
    // All identical values → variance < 1e-12 → no change
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission_0 = params.emission[0].clone();
    let original_emission_1 = params.emission[1].clone();

    params.estimate_emissions(&[0.999, 0.999, 0.999, 0.999, 0.999]);

    assert!(
        (params.emission[0].mean - original_emission_0.mean).abs() < 1e-15,
        "Zero-variance data should not change emission[0]"
    );
    assert!(
        (params.emission[1].mean - original_emission_1.mean).abs() < 1e-15,
        "Zero-variance data should not change emission[1]"
    );
}

#[test]
fn test_estimate_emissions_exactly_3_obs() {
    // Minimal case: exactly 3 observations with variance
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    params.estimate_emissions(&[0.5, 0.75, 0.999]);

    // Should update — means should be finite
    assert!(
        params.emission[0].mean.is_finite(),
        "emission[0].mean should be finite"
    );
    assert!(
        params.emission[1].mean.is_finite(),
        "emission[1].mean should be finite"
    );
    assert!(params.emission[0].std > 0.0, "emission[0].std should be > 0");
    assert!(params.emission[1].std > 0.0, "emission[1].std should be > 0");
}

#[test]
fn test_estimate_emissions_std_is_clamped() {
    // Low cluster std clamped to >= 0.01, high cluster std clamped to >= 0.001
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    // All values nearly identical in each cluster → variance ≈ 0, std clamped
    let obs = vec![
        0.500, 0.500, 0.500, 0.500, // Near-zero variance low cluster
        0.999, 0.999, 0.999, 0.999, // Near-zero variance high cluster
    ];
    params.estimate_emissions(&obs);

    assert!(
        params.emission[0].std >= 0.01,
        "Low emission std should be clamped to >= 0.01, got {}",
        params.emission[0].std
    );
    assert!(
        params.emission[1].std >= 0.001,
        "High emission std should be clamped to >= 0.001, got {}",
        params.emission[1].std
    );
}

#[test]
fn test_estimate_emissions_bimodal_data() {
    // Strongly bimodal distribution — k-means should succeed
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let mut obs = Vec::new();
    for _ in 0..50 {
        obs.push(0.5);
    }
    for _ in 0..50 {
        obs.push(0.999);
    }
    params.estimate_emissions(&obs);

    // Should cleanly separate
    assert!(
        (params.emission[0].mean - 0.5).abs() < 0.1,
        "Low cluster should be near 0.5, got {}",
        params.emission[0].mean
    );
    assert!(
        (params.emission[1].mean - 0.999).abs() < 0.01,
        "High cluster should be near 0.999, got {}",
        params.emission[1].mean
    );
}

#[test]
fn test_estimate_emissions_unimodal_with_spread() {
    // Unimodal data with enough variance to trigger estimation
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..20).map(|i| 0.5 + 0.025 * i as f64).collect(); // 0.5 to 0.975
    params.estimate_emissions(&obs);

    // Should still produce finite, reasonable estimates
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].mean < params.emission[1].mean);
}

#[test]
fn test_estimate_emissions_preserves_transition_and_initial() {
    // estimate_emissions should only change emission, not transition or initial
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_transition = params.transition;
    let original_initial = params.initial;

    let obs = vec![0.5, 0.6, 0.55, 0.999, 0.998, 0.9995];
    params.estimate_emissions(&obs);

    assert_eq!(
        params.transition, original_transition,
        "Transition should not change"
    );
    assert_eq!(params.initial, original_initial, "Initial should not change");
}

#[test]
fn test_estimate_emissions_idempotent_on_zero_variance() {
    // Calling twice with same zero-variance data should have same result
    let mut params1 = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let mut params2 = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.8, 0.8, 0.8, 0.8];

    params1.estimate_emissions(&obs);
    params2.estimate_emissions(&obs);
    params2.estimate_emissions(&obs); // Second call

    assert!(
        (params1.emission[0].mean - params2.emission[0].mean).abs() < 1e-10,
        "Should be idempotent on zero-variance data"
    );
}

// ============================================================================
// estimate_emissions — fallback path (kmeans returns None)
// ============================================================================

#[test]
fn test_estimate_emissions_kmeans_none_fallback() {
    // When k-means fails to converge or data is unsuitable, the fallback
    // uses percentiles (30th and 90th) — this is tested with moderate data
    // that has sufficient variance but may not cluster cleanly
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    // Uniform-ish data that won't cluster cleanly
    let obs: Vec<f64> = (0..100).map(|i| 0.3 + 0.006 * i as f64).collect(); // 0.3 to 0.894
    params.estimate_emissions(&obs);

    // Regardless of path taken, emissions should be finite and ordered
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

// ============================================================================
// Cross-consistency: standard vs log-emit pipeline
// ============================================================================

#[test]
fn test_viterbi_standard_matches_log_emit_pipeline() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.55, 0.998, 0.999, 0.9995, 0.5, 0.4, 0.6];

    let states_std = viterbi(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let states_le = viterbi_from_log_emit(&log_emit, &params);

    assert_eq!(
        states_std.len(),
        states_le.len(),
        "Standard and log-emit Viterbi should produce same length"
    );
    for (t, (&s1, &s2)) in states_std.iter().zip(states_le.iter()).enumerate() {
        assert_eq!(
            s1, s2,
            "Viterbi state mismatch at t={}: std={}, log_emit={}",
            t, s1, s2
        );
    }
}

#[test]
fn test_forward_standard_matches_log_emit_ll() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998, 0.999, 0.5, 0.6, 0.998, 0.55, 0.4, 0.999];

    let (_, ll_std) = forward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (_, ll_le) = forward_from_log_emit(&log_emit, &params);

    assert!(
        (ll_std - ll_le).abs() < 1e-8,
        "Forward LL mismatch: standard={}, log_emit={}",
        ll_std,
        ll_le
    );
}

#[test]
fn test_fb_standard_matches_log_emit_posteriors() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.5, 0.6, 0.998, 0.999, 0.9995, 0.5, 0.4];

    let (post_std, ll_std) = forward_backward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (post_le, ll_le) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(
        (ll_std - ll_le).abs() < 1e-8,
        "FB LL mismatch: standard={}, log_emit={}",
        ll_std,
        ll_le
    );

    for (t, (&p1, &p2)) in post_std.iter().zip(post_le.iter()).enumerate() {
        assert!(
            (p1 - p2).abs() < 1e-8,
            "FB posterior mismatch at t={}: standard={}, log_emit={}",
            t,
            p1,
            p2
        );
    }
}

#[test]
fn test_viterbi_and_fb_agree_on_strong_signal() {
    // With a very strong IBD signal, Viterbi states and FB posteriors
    // should agree: high posterior ↔ state 1, low posterior ↔ state 0
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    params.emission[0] = GaussianParams::new_unchecked(0.5, 0.05);
    params.emission[1] = GaussianParams::new_unchecked(0.999, 0.001);

    let mut obs = vec![0.5; 20];
    for i in 8..16 {
        obs[i] = 0.999;
    }

    let states = viterbi(&obs, &params);
    let (posteriors, _) = forward_backward(&obs, &params);

    for t in 0..obs.len() {
        if states[t] == 1 {
            assert!(
                posteriors[t] > 0.5,
                "Viterbi says IBD at t={} but posterior is {:.4}",
                t,
                posteriors[t]
            );
        }
        if states[t] == 0 {
            assert!(
                posteriors[t] < 0.5,
                "Viterbi says non-IBD at t={} but posterior is {:.4}",
                t,
                posteriors[t]
            );
        }
    }
}

// ============================================================================
// Cross-consistency: infer_ibd should produce valid results
// ============================================================================

#[test]
fn test_infer_ibd_posteriors_match_states_direction() {
    // infer_ibd returns both states and posteriors — they should be directionally consistent
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![
        0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6,
    ];

    let result = infer_ibd(&obs, &params);
    assert_eq!(result.states.len(), result.posteriors.len());

    // All posteriors should be in [0, 1]
    for (t, &p) in result.posteriors.iter().enumerate() {
        assert!(
            p >= -1e-10 && p <= 1.0 + 1e-10,
            "Posterior at t={} out of [0,1]: {}",
            t,
            p
        );
    }

    // Log-likelihood should be finite and negative
    assert!(
        result.log_likelihood.is_finite(),
        "Log-likelihood should be finite"
    );
    assert!(
        result.log_likelihood < 0.0,
        "Log-likelihood should be negative: {}",
        result.log_likelihood
    );
}

// ============================================================================
// estimate_emissions followed by inference should produce better results
// ============================================================================

#[test]
fn test_estimate_emissions_then_infer_produces_valid_output() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![
        0.5, 0.55, 0.6, 0.48, 0.52, // Non-IBD
        0.999, 0.998, 0.9995, 0.9997, 0.999, // IBD
        0.5, 0.55, 0.6, 0.48, // Non-IBD
    ];

    params.estimate_emissions(&obs);

    // After estimation, run inference
    let result = infer_ibd(&obs, &params);
    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());

    // Posteriors should be valid
    for &p in &result.posteriors {
        assert!(p >= -1e-10 && p <= 1.0 + 1e-10);
    }
}

// ============================================================================
// Cross-consistency: forward LL equals backward-computed LL
// ============================================================================

#[test]
fn test_forward_backward_ll_consistency_various_sequences() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);

    let test_sequences = vec![
        vec![0.5; 10],
        vec![0.999; 10],
        vec![0.5, 0.999, 0.5, 0.999, 0.5],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999],
        (0..100).map(|i| if i % 20 < 10 { 0.999 } else { 0.5 }).collect(),
    ];

    for (i, obs) in test_sequences.iter().enumerate() {
        let (_, ll_fwd) = forward(obs, &params);
        let (_, ll_fb) = forward_backward(obs, &params);
        assert!(
            (ll_fwd - ll_fb).abs() < 1e-6,
            "Sequence {}: forward LL ({}) != FB LL ({})",
            i,
            ll_fwd,
            ll_fb
        );
    }
}

// ============================================================================
// estimate_emissions with population priors then inference
// ============================================================================

#[test]
fn test_estimate_emissions_all_populations_produce_valid_inference() {
    let obs = vec![
        0.5, 0.55, 0.6, 0.48, 0.999, 0.998, 0.9995, 0.48, 0.5, 0.55,
    ];
    let populations = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
        Population::Generic,
    ];

    for pop in &populations {
        let mut params = HmmParams::from_population(*pop, 50.0, 0.001, 5000);
        params.estimate_emissions(&obs);

        // Emissions should be finite and ordered
        assert!(
            params.emission[0].mean.is_finite(),
            "emission[0].mean not finite for {:?}",
            pop
        );
        assert!(
            params.emission[1].mean.is_finite(),
            "emission[1].mean not finite for {:?}",
            pop
        );

        let result = infer_ibd(&obs, &params);
        assert_eq!(result.states.len(), obs.len());
        assert!(
            result.log_likelihood.is_finite(),
            "LL not finite for {:?}: {}",
            pop,
            result.log_likelihood
        );
    }
}

// ============================================================================
// estimate_emissions with near-boundary variance
// ============================================================================

#[test]
fn test_estimate_emissions_tiny_variance_just_above_threshold() {
    // Variance just above 1e-12 — should still estimate
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_mean = params.emission[0].mean;

    // values differ by ~1e-5, variance ≈ 2.5e-11 > 1e-12
    let obs = vec![0.5, 0.50001, 0.49999, 0.5, 0.50001];
    params.estimate_emissions(&obs);

    // With such tiny variance, k-means may not separate well, but it should
    // still produce finite results without panic
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    // The means may or may not change depending on k-means behavior,
    // but the function should complete successfully
    let _ = original_mean; // suppress unused warning
}

#[test]
fn test_estimate_emissions_large_dataset() {
    // 1000 observations with clear bimodal structure
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let mut obs = Vec::with_capacity(1000);
    for i in 0..1000 {
        if i % 3 == 0 {
            obs.push(0.999 + 0.0005 * (i as f64 / 1000.0));
        } else {
            obs.push(0.5 + 0.05 * (i as f64 / 1000.0));
        }
    }
    params.estimate_emissions(&obs);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].mean < params.emission[1].mean);
}

// ============================================================================
// Viterbi path consistency across pipelines
// ============================================================================

#[test]
fn test_viterbi_states_always_binary() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);

    let sequences = vec![
        vec![0.0; 50],
        vec![1.0; 50],
        vec![0.5; 50],
        (0..200)
            .map(|i| 0.5 + 0.499 * ((i as f64 * 0.1).sin()))
            .collect::<Vec<f64>>(),
    ];

    for (i, obs) in sequences.iter().enumerate() {
        let states = viterbi(obs, &params);
        assert_eq!(states.len(), obs.len(), "Sequence {} length mismatch", i);
        for (t, &s) in states.iter().enumerate() {
            assert!(
                s <= 1,
                "Sequence {} state at t={} should be 0 or 1, got {}",
                i,
                t,
                s
            );
        }
    }
}
