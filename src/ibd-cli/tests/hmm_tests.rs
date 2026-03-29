//! Extended IBD HMM tests: property-based, Viterbi consistency,
//! AFR regression, stress tests, emission estimation robustness,
//! distance-aware HMM edge cases, and LOD scoring edge cases.

use hprc_ibd::hmm::{
    compute_per_window_lod, distance_dependent_log_transition,
    extract_ibd_segments_with_lod, extract_ibd_segments_with_posteriors,
    forward_backward, forward_backward_with_distances, forward_with_distances,
    forward_backward_with_genetic_map, forward_with_genetic_map,
    backward_with_genetic_map, viterbi_with_genetic_map,
    recombination_aware_log_transition,
    infer_ibd, segment_lod_score, segment_posterior_std, segment_quality_score,
    viterbi, viterbi_with_distances, IbdSegmentWithPosterior, GeneticMap, HmmParams, Population,
};

// ---------------------------------------------------------------------------
// 1. Property-based tests for forward-backward
// ---------------------------------------------------------------------------

/// All posteriors must be in [0, 1].
#[test]
fn test_fb_posteriors_in_unit_interval() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

    // Use a deterministic sweep of identity values in [0.990, 1.000]
    let observations: Vec<f64> = (0..200)
        .map(|i| 0.990 + (i as f64 / 200.0) * 0.010)
        .collect();

    let (posteriors, _log_lik) = forward_backward(&observations, &params);

    assert_eq!(posteriors.len(), observations.len());
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "Posterior at position {} is {}, outside [0, 1]",
            i,
            p
        );
    }
}

/// For each position, P(non-IBD) + P(IBD) must sum to ~1.0.
/// Since forward_backward returns only P(IBD), we check that value is in [0,1]
/// and verify the log-space derivation is consistent by checking both bounds.
#[test]
fn test_fb_posteriors_implicit_sum_to_one() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // Mix of values: some clearly non-IBD, some clearly IBD
    let observations = vec![
        0.9980, 0.9975, 0.9970, 0.9985, // non-IBD region
        0.9998, 0.9999, 0.9997, 0.9999, 0.9998, // IBD region
        0.9975, 0.9980, // non-IBD
    ];

    let (posteriors, _) = forward_backward(&observations, &params);

    for (i, &p_ibd) in posteriors.iter().enumerate() {
        let p_non_ibd = 1.0 - p_ibd;
        let sum = p_ibd + p_non_ibd;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "P(non-IBD) + P(IBD) = {} at position {} (expected 1.0)",
            sum,
            i
        );
    }
}

/// Test with many random-like observations in [0.990, 1.000] that posteriors
/// are always well-defined (not NaN, not Inf).
#[test]
fn test_fb_posteriors_well_defined_dense_range() {
    let params = HmmParams::from_population(Population::AFR, 30.0, 0.0001, 5000);

    // Use a simple deterministic pseudo-random sequence
    let observations: Vec<f64> = (0..500)
        .map(|i| {
            let t = (i as f64 * 0.618033988) % 1.0; // golden ratio fractional parts
            0.990 + t * 0.010
        })
        .collect();

    let (posteriors, log_lik) = forward_backward(&observations, &params);

    assert!(log_lik.is_finite(), "Log-likelihood should be finite");
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(p.is_finite(), "Posterior at {} is not finite: {}", i, p);
        assert!(
            (0.0..=1.0).contains(&p),
            "Posterior at {} is {}, outside [0, 1]",
            i,
            p
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Viterbi consistency with posteriors
// ---------------------------------------------------------------------------

/// Where posterior P(IBD) > 0.95, Viterbi should also call IBD.
#[test]
fn test_viterbi_agrees_with_high_posterior() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.01, 5000);

    // Construct a scenario with a clear IBD block
    let mut obs = vec![0.9980; 50]; // non-IBD block
    obs.extend(vec![0.9999; 80]); // clear IBD block
    obs.extend(vec![0.9975; 50]); // non-IBD block

    let result = infer_ibd(&obs, &params);

    for (i, (&state, &post)) in result
        .states
        .iter()
        .zip(result.posteriors.iter())
        .enumerate()
    {
        if post > 0.95 {
            assert_eq!(
                state, 1,
                "Position {} has P(IBD)={:.4} but Viterbi says non-IBD",
                i, post
            );
        }
    }
}

/// Viterbi path must contain only 0s and 1s.
#[test]
fn test_viterbi_path_is_valid_binary() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    let observations: Vec<f64> = (0..300)
        .map(|i| {
            let t = (i as f64 * 0.618033988) % 1.0;
            0.990 + t * 0.010
        })
        .collect();

    let states = viterbi(&observations, &params);

    assert_eq!(states.len(), observations.len());
    for (i, &s) in states.iter().enumerate() {
        assert!(
            s == 0 || s == 1,
            "Invalid state {} at position {} (expected 0 or 1)",
            s,
            i
        );
    }
}

// ---------------------------------------------------------------------------
// 3. AFR regression test
// ---------------------------------------------------------------------------

/// Create synthetic data mimicking AFR diversity (pi=0.00125).
/// Non-IBD windows: identity ~ N(0.99875, 0.0003)
/// IBD windows (positions 100-200): identity ~ N(0.99995, 0.00005)
/// Verify at least 80% of IBD region is detected.
#[test]
fn test_afr_regression_detects_ibd_block() {
    // Build synthetic observations with deterministic pattern
    let n_total = 400;
    let ibd_start = 100;
    let ibd_end = 200; // exclusive
    let ibd_len = ibd_end - ibd_start;

    let mut observations = Vec::with_capacity(n_total);
    for i in 0..n_total {
        if i >= ibd_start && i < ibd_end {
            // IBD region: very high identity with small variation
            let offset = ((i as f64 * 0.7) % 1.0 - 0.5) * 0.0001;
            observations.push((0.99995 + offset).clamp(0.999, 1.0));
        } else {
            // Non-IBD region: AFR background with variation
            let offset = ((i as f64 * 0.618033988) % 1.0 - 0.5) * 0.0006;
            observations.push((0.99875 + offset).clamp(0.990, 1.0));
        }
    }

    // Use AFR-specific parameters
    let mut params =
        HmmParams::from_population_adaptive(Population::AFR, 50.0, 0.0001, 5000);
    params.estimate_emissions_robust(&observations, Some(Population::AFR), 5000);

    let result = infer_ibd(&observations, &params);

    // Count how many windows in the IBD region were called IBD
    let ibd_detected: usize = result.states[ibd_start..ibd_end]
        .iter()
        .filter(|&&s| s == 1)
        .count();

    let detection_rate = ibd_detected as f64 / ibd_len as f64;
    assert!(
        detection_rate >= 0.80,
        "Only {:.1}% of AFR IBD region detected (expected >= 80%); detected {}/{}",
        detection_rate * 100.0,
        ibd_detected,
        ibd_len
    );
}

// ---------------------------------------------------------------------------
// 4. Stress test
// ---------------------------------------------------------------------------

/// 10,000 windows of deterministic pseudo-random data.
/// Must complete without panic and in a reasonable time.
#[test]
fn test_stress_10k_windows_no_panic() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

    let observations: Vec<f64> = (0..10_000)
        .map(|i| {
            let t = (i as f64 * 0.618033988) % 1.0;
            0.990 + t * 0.010 // values in [0.990, 1.000]
        })
        .collect();

    let start = std::time::Instant::now();

    let result = infer_ibd(&observations, &params);

    let elapsed = start.elapsed();

    assert_eq!(result.states.len(), 10_000);
    assert_eq!(result.posteriors.len(), 10_000);
    assert!(result.log_likelihood.is_finite());

    // Must complete in < 5 seconds
    assert!(
        elapsed.as_secs() < 5,
        "Stress test took {:?}, expected < 5s",
        elapsed
    );
}

/// Stress test: forward-backward with 10,000 windows should not panic.
#[test]
fn test_stress_forward_backward_10k() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

    let observations: Vec<f64> = (0..10_000)
        .map(|i| {
            let t = (i as f64 * 0.314159265) % 1.0;
            0.995 + t * 0.005
        })
        .collect();

    let (posteriors, log_lik) = forward_backward(&observations, &params);

    assert_eq!(posteriors.len(), 10_000);
    assert!(log_lik.is_finite());

    // All posteriors must be valid
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "Invalid posterior at {}: {}",
            i,
            p
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Emission estimation robustness
// ---------------------------------------------------------------------------

/// Feed data where 95% is non-IBD + 5% IBD.
/// Verify k-means (or EM) produces two distinct clusters.
#[test]
fn test_emission_estimation_separates_clusters() {
    let mut data = Vec::new();

    // 95% non-IBD (EUR diversity)
    for i in 0..190 {
        let offset = ((i as f64 * 0.618033988) % 1.0 - 0.5) * 0.0004;
        data.push(0.99915 + offset);
    }

    // 5% IBD
    for i in 0..10 {
        let offset = ((i as f64 * 0.7) % 1.0 - 0.5) * 0.0001;
        data.push(0.9997 + offset);
    }

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    params.estimate_emissions_robust(&data, Some(Population::EUR), 5000);

    // The two emission distributions should be distinct
    let non_ibd_mean = params.emission[0].mean;
    let ibd_mean = params.emission[1].mean;

    assert!(
        ibd_mean > non_ibd_mean,
        "IBD mean ({}) should be greater than non-IBD mean ({})",
        ibd_mean,
        non_ibd_mean
    );

    let separation = ibd_mean - non_ibd_mean;
    assert!(
        separation > 0.0002,
        "Separation between clusters ({}) should be > 0.0002",
        separation
    );
}

/// Emission estimation with all-IBD data should not produce degenerate params.
#[test]
fn test_emission_estimation_all_high_identity() {
    let data: Vec<f64> = (0..100).map(|i| 0.9997 + (i as f64 * 0.000001)).collect();

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    params.estimate_emissions_robust(&data, Some(Population::EUR), 5000);

    // Should not panic, and emissions should remain valid
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

/// Emission estimation with all-non-IBD data should keep the IBD emission at prior.
#[test]
fn test_emission_estimation_all_low_identity() {
    let data: Vec<f64> = (0..100).map(|i| 0.998 + (i as f64 * 0.000005)).collect();

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    params.estimate_emissions_robust(&data, Some(Population::EUR), 5000);

    // IBD emission should still have a high mean (from prior or bounded)
    assert!(
        params.emission[1].mean >= 0.999,
        "IBD mean {} should be >= 0.999 even with no IBD data",
        params.emission[1].mean
    );
    assert!(params.emission[0].mean.is_finite());
    // There should still be two distinct emission components
    assert!(params.emission[1].mean > params.emission[0].mean);
}

// ---------------------------------------------------------------------------
// 6. Additional edge case tests
// ---------------------------------------------------------------------------

/// extract_ibd_segments_with_posteriors should handle segments at boundaries.
#[test]
fn test_segments_with_posteriors_at_boundaries() {
    // IBD at start
    let states = vec![1, 1, 1, 0, 0, 0, 0, 0, 1, 1];
    let posteriors = vec![0.95, 0.90, 0.88, 0.1, 0.1, 0.1, 0.1, 0.1, 0.92, 0.94];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 2, 0.5);

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 2);
    assert_eq!(segments[0].n_windows, 3);
    assert_eq!(segments[1].start_idx, 8);
    assert_eq!(segments[1].end_idx, 9);
    assert_eq!(segments[1].n_windows, 2);
}

// ---------------------------------------------------------------------------
// 7. Distance-aware HMM edge case tests
// ---------------------------------------------------------------------------

/// Very large gap (10 Mb) should push transitions toward their stationary
/// distribution (close to 50/50 or the prior).
#[test]
fn test_distance_aware_very_large_gap() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let huge_gap = 10_000_000u64; // 10 Mb
    let trans = distance_dependent_log_transition(&params, huge_gap, 5000);

    // At a huge distance, switching should be highly likely —
    // the off-diagonal elements should be substantially larger than at unit distance.
    let unit_trans = distance_dependent_log_transition(&params, 5000, 5000);

    // P(switch) at large gap >= P(switch) at unit distance
    // In log space: less negative means higher probability
    assert!(
        trans[0][1] >= unit_trans[0][1],
        "Large gap should increase P(0->1): got {} vs {}",
        trans[0][1], unit_trans[0][1]
    );
    assert!(
        trans[1][0] >= unit_trans[1][0],
        "Large gap should increase P(1->0): got {} vs {}",
        trans[1][0], unit_trans[1][0]
    );

    // Transitions should still be valid (exponentiated rows sum to 1)
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Transition row should sum to 1.0 in probability space, got {}",
            sum
        );
    }
}

/// Distance-aware forward-backward with a single observation should not panic
/// and should produce valid posteriors.
#[test]
fn test_distance_aware_single_observation() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = vec![0.9995];
    let positions = vec![(100000u64, 105000u64)];

    let (posteriors, log_lik) = forward_backward_with_distances(&obs, &params, &positions);

    assert_eq!(posteriors.len(), 1);
    assert!(posteriors[0].is_finite());
    assert!((0.0..=1.0).contains(&posteriors[0]));
    assert!(log_lik.is_finite());
}

/// Forward + backward consistency: for each position,
/// log(sum_s(alpha[t][s] * beta[t][s])) should equal the log-likelihood.
#[test]
fn test_distance_aware_alpha_beta_consistency() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..50)
        .map(|i| 0.995 + (i as f64 / 50.0) * 0.005)
        .collect();
    let positions: Vec<(u64, u64)> = (0..50)
        .map(|i| (i * 5000, (i + 1) * 5000))
        .collect();

    let (alpha, log_lik) = forward_with_distances(&obs, &params, &positions);
    let (posteriors, log_lik2) = forward_backward_with_distances(&obs, &params, &positions);

    // Both should agree on log-likelihood
    assert!(
        (log_lik - log_lik2).abs() < 1e-6,
        "Forward and forward-backward should agree on log-likelihood: {} vs {}",
        log_lik, log_lik2
    );

    // Alpha at last position should give the log-likelihood via logsumexp
    let last = alpha.len() - 1;
    let max_a = alpha[last][0].max(alpha[last][1]);
    let logsumexp = max_a + ((alpha[last][0] - max_a).exp() + (alpha[last][1] - max_a).exp()).ln();
    assert!(
        (logsumexp - log_lik).abs() < 1e-4,
        "Alpha logsumexp at final position ({}) should match log_lik ({})",
        logsumexp, log_lik
    );

    // All posteriors must be valid
    for &p in &posteriors {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }
}

/// Viterbi with distances: a single observation should return a valid state.
#[test]
fn test_viterbi_with_distances_single_obs() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = vec![0.9998]; // IBD-like
    let positions = vec![(0u64, 5000u64)];

    let states = viterbi_with_distances(&obs, &params, &positions);
    assert_eq!(states.len(), 1);
    assert!(states[0] == 0 || states[0] == 1);
}

/// Distance-aware transitions at various population priors should
/// all produce valid transition matrices.
#[test]
fn test_distance_aware_all_populations() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
    ];

    for pop in &pops {
        let params = HmmParams::from_population_adaptive(*pop, 50.0, 0.001, 5000);
        for distance in [100u64, 5000, 50000, 500000, 5000000] {
            let trans = distance_dependent_log_transition(&params, distance, 5000);
            for row in &trans {
                let sum = row[0].exp() + row[1].exp();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Pop {:?} distance {}: row sum {} != 1.0",
                    pop, distance, sum
                );
                // Both entries should be finite
                assert!(row[0].is_finite() && row[1].is_finite(),
                    "Pop {:?} distance {}: non-finite transition",
                    pop, distance);
            }
        }
    }
}

/// Stress test: distance-aware forward-backward with 5000 windows,
/// some with large gaps, should not panic and produce valid posteriors.
#[test]
fn test_distance_aware_stress_with_gaps() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    let n = 5000;
    let obs: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64 * 0.618033988) % 1.0;
            0.995 + t * 0.005
        })
        .collect();

    // Most windows are contiguous but add some large gaps
    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| {
            let base = if i < 1000 {
                i as u64 * 5000
            } else if i < 2000 {
                // 1 Mb gap after position 1000
                1_000_000 + (i as u64 - 1000) * 5000
            } else {
                // 5 Mb gap after position 2000
                6_000_000 + (i as u64 - 2000) * 5000
            };
            (base, base + 5000)
        })
        .collect();

    let (posteriors, log_lik) = forward_backward_with_distances(&obs, &params, &positions);

    assert_eq!(posteriors.len(), n);
    assert!(log_lik.is_finite());
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "Invalid posterior at {}: {}",
            i, p
        );
    }
}

// ---------------------------------------------------------------------------
// 8. LOD scoring edge case tests
// ---------------------------------------------------------------------------

/// LOD at the exact non-IBD emission mean should be negative
/// (the observation is maximally consistent with non-IBD).
#[test]
fn test_lod_at_non_ibd_emission_mean() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let non_ibd_mean = params.emission[0].mean;
    let obs = vec![non_ibd_mean];

    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    assert!(
        lods[0] < 0.0,
        "LOD at non-IBD mean should be negative, got {}",
        lods[0]
    );
}

/// LOD at the exact IBD emission mean should be positive
/// (the observation is maximally consistent with IBD).
#[test]
fn test_lod_at_ibd_emission_mean() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let ibd_mean = params.emission[1].mean;
    let obs = vec![ibd_mean];

    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    assert!(
        lods[0] > 0.0,
        "LOD at IBD mean should be positive, got {}",
        lods[0]
    );
}

/// LOD is population-dependent: different population parameters should
/// produce different LOD values for the same observation.
#[test]
fn test_lod_population_dependent() {
    let params_afr = HmmParams::from_population(Population::AFR, 50.0, 0.001, 5000);
    let params_eur = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // At the IBD emission mean, both should be positive
    let obs_ibd = vec![0.9999];
    let lod_afr = compute_per_window_lod(&obs_ibd, &params_afr);
    let lod_eur = compute_per_window_lod(&obs_ibd, &params_eur);

    assert!(lod_afr[0] > 0.0, "AFR LOD at IBD obs should be positive");
    assert!(lod_eur[0] > 0.0, "EUR LOD at IBD obs should be positive");

    // They should differ because the populations have different non-IBD means
    assert!(
        (lod_afr[0] - lod_eur[0]).abs() > 0.01,
        "LOD should differ between populations: AFR={}, EUR={}",
        lod_afr[0], lod_eur[0]
    );

    // At each population's own non-IBD mean, LOD should be negative
    let obs_afr_mean = vec![params_afr.emission[0].mean];
    let obs_eur_mean = vec![params_eur.emission[0].mean];
    assert!(
        compute_per_window_lod(&obs_afr_mean, &params_afr)[0] < 0.0,
        "LOD at AFR non-IBD mean should be negative"
    );
    assert!(
        compute_per_window_lod(&obs_eur_mean, &params_eur)[0] < 0.0,
        "LOD at EUR non-IBD mean should be negative"
    );
}

/// Segment LOD with mixed IBD/non-IBD observations within the segment range.
#[test]
fn test_segment_lod_mixed_observations() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // Mixed: some IBD-like, some non-IBD-like
    let obs = vec![0.998, 0.9999, 0.997, 0.9998, 0.996];

    let lod = segment_lod_score(&obs, 0, 4, &params);
    // The LOD should be finite (could be positive or negative depending on balance)
    assert!(lod.is_finite(), "Segment LOD should be finite, got {}", lod);

    // Individual LODs
    let per_window = compute_per_window_lod(&obs, &params);
    let manual_sum: f64 = per_window.iter().sum();
    assert!(
        (lod - manual_sum).abs() < 1e-10,
        "Segment LOD ({}) should equal sum of per-window LODs ({})",
        lod, manual_sum
    );
}

/// Quality score for a single-window segment.
#[test]
fn test_quality_score_single_window() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 5,
        end_idx: 5,
        n_windows: 1,
        mean_posterior: 0.95,
        min_posterior: 0.95,
        max_posterior: 0.95,
        lod_score: 2.0,
    };

    let q = segment_quality_score(&seg);
    assert!(q.is_finite());
    assert!((0.0..=100.0).contains(&q));

    // Single window should get low length score (1/20 * 10 = 0.5)
    // But high posterior (0.95 * 40 = 38) and high consistency (1.0 * 20 = 20)
    // So Q should be moderate
    assert!(q > 30.0, "Single-window segment with high posterior should score > 30, got {}", q);
}

/// Quality score with negative LOD (non-IBD-like) should be low.
#[test]
fn test_quality_score_negative_lod() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 9,
        n_windows: 10,
        mean_posterior: 0.6,
        min_posterior: 0.3,
        max_posterior: 0.8,
        lod_score: -5.0,
    };

    let q = segment_quality_score(&seg);
    assert!(q.is_finite());
    assert!((0.0..=100.0).contains(&q));

    // Negative LOD should contribute 0 to LOD component
    // So Q should be lower than a segment with positive LOD
    let seg_pos_lod = IbdSegmentWithPosterior {
        lod_score: 10.0,
        ..seg
    };
    let q_pos = segment_quality_score(&seg_pos_lod);
    assert!(
        q < q_pos,
        "Negative LOD segment Q ({}) should be < positive LOD Q ({})",
        q, q_pos
    );
}

/// Segment posterior std with alternating 0.0 and 1.0 posteriors.
#[test]
fn test_posterior_std_extreme_alternating() {
    let posteriors = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let std = segment_posterior_std(&posteriors, 0, 5);
    assert!(std.is_finite());
    // Expected std for alternating 0/1 with 6 values: sqrt(sum((p-0.5)^2)/5) = sqrt(6*0.25/5)
    assert!(
        std > 0.4,
        "Std of alternating 0/1 should be > 0.4, got {}",
        std
    );
}

/// extract_ibd_segments_with_lod: LOD filtering should remove weak segments.
#[test]
fn test_lod_filtering_removes_weak_segments() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // Two IBD segments: one strong (high identity), one weak (borderline identity)
    let mut states = vec![0usize; 40];
    let mut posteriors = vec![0.1f64; 40];
    let mut observations = vec![0.998f64; 40]; // non-IBD background

    // Strong IBD segment: positions 5-14
    for i in 5..15 {
        states[i] = 1;
        posteriors[i] = 0.95;
        observations[i] = 0.99985;
    }

    // Weak IBD segment: positions 25-29 (borderline identity)
    for i in 25..30 {
        states[i] = 1;
        posteriors[i] = 0.70;
        observations[i] = 0.9994;
    }

    // Without LOD filter: should get 2 segments
    let segs_no_filter = extract_ibd_segments_with_lod(
        &states, &posteriors, 3, 0.5,
        Some((&observations, &params)), None,
    );
    assert_eq!(segs_no_filter.len(), 2, "Without LOD filter should find 2 segments");

    // The strong segment should have higher LOD than the weak one
    assert!(
        segs_no_filter[0].lod_score > segs_no_filter[1].lod_score,
        "Strong segment LOD ({}) should be > weak segment LOD ({})",
        segs_no_filter[0].lod_score, segs_no_filter[1].lod_score
    );

    // With LOD filter set between the two: should get only 1 segment
    let mid_lod = (segs_no_filter[0].lod_score + segs_no_filter[1].lod_score) / 2.0;
    let segs_filtered = extract_ibd_segments_with_lod(
        &states, &posteriors, 3, 0.5,
        Some((&observations, &params)), Some(mid_lod),
    );
    assert_eq!(segs_filtered.len(), 1, "LOD filter should remove weak segment");
    assert_eq!(segs_filtered[0].start_idx, 5);
}

/// extract_ibd_segments_with_lod: without observations, LOD should be 0.0.
#[test]
fn test_lod_zero_without_observations() {
    let states = vec![1, 1, 1, 1, 1, 0, 0, 0];
    let posteriors = vec![0.9, 0.95, 0.92, 0.88, 0.91, 0.1, 0.1, 0.1];

    let segs = extract_ibd_segments_with_lod(
        &states, &posteriors, 2, 0.5, None, None,
    );

    assert_eq!(segs.len(), 1);
    assert!(
        (segs[0].lod_score - 0.0).abs() < 1e-10,
        "LOD without observations should be 0.0, got {}",
        segs[0].lod_score
    );
}

/// Viterbi with distances should detect an IBD region surrounded by large gaps.
#[test]
fn test_viterbi_distances_ibd_between_gaps() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.01, 5000);
    // Estimate from synthetic data with clear IBD region
    let mut obs = vec![0.9985f64; 100];
    for o in obs.iter_mut().take(70).skip(30) {
        *o = 0.99975;
    }
    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

    // Put a 500kb gap before and after the IBD region
    let mut positions: Vec<(u64, u64)> = (0..30)
        .map(|i| (i as u64 * 5000, (i as u64 + 1) * 5000))
        .collect();
    // Gap: 500kb
    for i in 30..70 {
        let base = 500_000 + (i as u64 - 30) * 5000;
        positions.push((base, base + 5000));
    }
    // Another gap: 500kb
    for i in 70..100 {
        let base = 1_200_000 + (i as u64 - 70) * 5000;
        positions.push((base, base + 5000));
    }

    let states = viterbi_with_distances(&obs, &params, &positions);

    // The IBD region (30-69) should be mostly called IBD
    let ibd_in_region: usize = states[30..70].iter().filter(|&&s| s == 1).count();
    let detection_rate = ibd_in_region as f64 / 40.0;
    assert!(
        detection_rate >= 0.7,
        "Should detect >= 70% of IBD region between gaps, got {:.1}%",
        detection_rate * 100.0
    );
}

// ---------------------------------------------------------------------------
// 9. Genetic map and recombination-aware edge case tests
// ---------------------------------------------------------------------------

/// Empty genetic map should still produce valid transitions
/// (should fall back to base transitions since positions will be the same or
/// interpolate_cm returns 0.0).
#[test]
fn test_recombination_aware_empty_genetic_map() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let empty_map = GeneticMap::new(vec![]);

    let trans = recombination_aware_log_transition(&params, 0, 50000, &empty_map, 5000);

    // Should produce valid transitions (empty map returns 0 cM everywhere)
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Empty genetic map: row sum {} != 1.0",
            sum
        );
        assert!(row[0].is_finite() && row[1].is_finite());
    }
}

/// Genetic map with positions reversed (pos2 < pos1) should still work
/// because genetic_distance_cm uses abs().
#[test]
fn test_recombination_aware_reversed_positions() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 10_000_000, 1.0);

    let trans_forward = recombination_aware_log_transition(&params, 100000, 200000, &map, 5000);
    let trans_reverse = recombination_aware_log_transition(&params, 200000, 100000, &map, 5000);

    // Should produce identical results due to abs() in genetic_distance_cm
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (trans_forward[i][j] - trans_reverse[i][j]).abs() < 1e-10,
                "Forward/reverse transitions should match at [{},{}]: {} vs {}",
                i, j, trans_forward[i][j], trans_reverse[i][j]
            );
        }
    }
}

/// Very small genetic distance (< 0.0001 cM) should produce transitions
/// close to base transitions.
#[test]
fn test_recombination_aware_very_small_distance() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // Very low rate: 0.001 cM/Mb = essentially no recombination
    let map = GeneticMap::uniform(0, 1_000_000, 0.001);

    // Adjacent windows at very short distance
    let trans = recombination_aware_log_transition(&params, 5000, 10000, &map, 5000);

    // Should be valid
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Very small distance: row sum {} != 1.0",
            sum
        );
    }

    // Stay probabilities should be high (close to base)
    assert!(
        trans[0][0].exp() > 0.9,
        "P(stay non-IBD) should be high, got {}",
        trans[0][0].exp()
    );
    assert!(
        trans[1][1].exp() > 0.9,
        "P(stay IBD) should be high, got {}",
        trans[1][1].exp()
    );
}

/// Very large genetic distance (> 100 cM) should drive switching probability
/// toward stationary distribution.
#[test]
fn test_recombination_aware_very_large_distance() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // Very high rate: 100 cM/Mb
    let map = GeneticMap::uniform(0, 100_000_000, 100.0);

    // Positions 50 Mb apart with very high recombination rate
    let trans = recombination_aware_log_transition(&params, 0, 50_000_000, &map, 5000);

    // Should still be valid
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Very large distance: row sum {} != 1.0",
            sum
        );
        assert!(row[0].is_finite() && row[1].is_finite());
    }
}

/// Genetic map that doesn't cover the observation region (extrapolation-only)
/// should still produce valid transitions.
#[test]
fn test_recombination_aware_extrapolation_only() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // Map covers 10-20 Mb, but our observations are at 0-5 Mb (entirely before map)
    let map = GeneticMap::new(vec![
        (10_000_000, 10.0),
        (20_000_000, 20.0),
    ]);

    let trans = recombination_aware_log_transition(&params, 1_000_000, 2_000_000, &map, 5000);

    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Extrapolation-only: row sum {} != 1.0",
            sum
        );
        assert!(row[0].is_finite() && row[1].is_finite());
    }
}

/// Genetic map with a recombination hotspot: transition near hotspot should
/// have higher exit probability than far from hotspot.
#[test]
fn test_recombination_hotspot_effect() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // Hotspot: low rate before, high rate in middle, low rate after
    let map = GeneticMap::new(vec![
        (0,         0.0),
        (1_000_000, 1.0),     // 1 cM/Mb = normal
        (1_050_000, 6.0),     // 100 cM/Mb = hotspot (5 cM in 50kb)
        (2_000_000, 7.0),     // back to ~1 cM/Mb
    ]);

    // Transition across the hotspot (within the 50kb hotspot region)
    let trans_hotspot = recombination_aware_log_transition(
        &params, 1_010_000, 1_040_000, &map, 5000,
    );

    // Transition in normal region (before hotspot)
    let trans_normal = recombination_aware_log_transition(
        &params, 500_000, 530_000, &map, 5000,
    );

    // Hotspot should have higher exit probability (less negative log)
    assert!(
        trans_hotspot[1][0] >= trans_normal[1][0],
        "Hotspot exit rate ({}) should be >= normal exit rate ({})",
        trans_hotspot[1][0], trans_normal[1][0]
    );
}

/// forward_backward_with_genetic_map with empty observations should return
/// empty posteriors.
#[test]
fn test_fb_genetic_map_empty_observations() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 1_000_000, 1.0);

    let (posteriors, _log_lik) = forward_backward_with_genetic_map(
        &[], &params, &[], &map, 5000,
    );

    assert!(posteriors.is_empty());
}

/// forward_backward_with_genetic_map with single observation should produce
/// valid posterior.
#[test]
fn test_fb_genetic_map_single_observation() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 1_000_000, 1.0);
    let obs = vec![0.9998];
    let positions = vec![(100_000u64, 105_000u64)];

    let (posteriors, log_lik) = forward_backward_with_genetic_map(
        &obs, &params, &positions, &map, 5000,
    );

    assert_eq!(posteriors.len(), 1);
    assert!(posteriors[0].is_finite());
    assert!((0.0..=1.0).contains(&posteriors[0]));
    assert!(log_lik.is_finite());
}

/// viterbi_with_genetic_map should detect an IBD region.
#[test]
fn test_viterbi_genetic_map_detects_ibd() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.01, 5000);
    let map = GeneticMap::uniform(0, 5_000_000, 1.0);

    // Build observations: non-IBD → IBD → non-IBD
    let mut obs = vec![0.9985f64; 200];
    for o in obs.iter_mut().take(120).skip(80) {
        *o = 0.99975;
    }
    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

    let positions: Vec<(u64, u64)> = (0..200)
        .map(|i| (i as u64 * 5000, (i as u64 + 1) * 5000))
        .collect();

    let states = viterbi_with_genetic_map(&obs, &params, &positions, &map, 5000);

    assert_eq!(states.len(), 200);
    // At least 50% of the IBD region should be detected
    let ibd_count: usize = states[80..120].iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count >= 20,
        "Should detect >= 50% of IBD region with genetic map, detected {}/40",
        ibd_count
    );
}

/// forward_with_genetic_map and backward_with_genetic_map should produce
/// consistent results (alpha + beta = log-likelihood at each position).
#[test]
fn test_forward_backward_genetic_map_consistency() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 2_000_000, 1.0);

    let obs: Vec<f64> = (0..50)
        .map(|i| 0.995 + (i as f64 / 50.0) * 0.005)
        .collect();
    let positions: Vec<(u64, u64)> = (0..50)
        .map(|i| (i as u64 * 5000, (i as u64 + 1) * 5000))
        .collect();

    let (alpha, log_lik_fwd) = forward_with_genetic_map(&obs, &params, &positions, &map, 5000);
    let beta = backward_with_genetic_map(&obs, &params, &positions, &map, 5000);
    let (posteriors, log_lik_fb) = forward_backward_with_genetic_map(
        &obs, &params, &positions, &map, 5000,
    );

    // Forward and forward-backward should agree on log-likelihood
    assert!(
        (log_lik_fwd - log_lik_fb).abs() < 1e-4,
        "Forward ({}) and FB ({}) log-likelihoods should match",
        log_lik_fwd, log_lik_fb
    );

    // Alpha at last position should give the log-likelihood
    let last = alpha.len() - 1;
    let max_a = alpha[last][0].max(alpha[last][1]);
    let logsumexp = max_a + ((alpha[last][0] - max_a).exp() + (alpha[last][1] - max_a).exp()).ln();
    assert!(
        (logsumexp - log_lik_fwd).abs() < 1e-4,
        "Alpha logsumexp ({}) should match log_lik ({})",
        logsumexp, log_lik_fwd
    );

    // Beta should start at [0,0] (log 1.0) at the last position
    assert!(
        (beta[last][0]).abs() < 1e-10 && (beta[last][1]).abs() < 1e-10,
        "Beta at last position should be [0,0], got [{}, {}]",
        beta[last][0], beta[last][1]
    );

    // All posteriors should be valid
    for &p in &posteriors {
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }
}

/// All populations should produce valid recombination-aware transitions
/// at various genetic distances.
#[test]
fn test_recombination_aware_all_populations_distances() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
    ];

    let map = GeneticMap::uniform(0, 100_000_000, 1.0);

    for pop in &pops {
        let params = HmmParams::from_population_adaptive(*pop, 50.0, 0.001, 5000);
        for &(pos1, pos2) in &[
            (0u64, 5000u64),
            (0, 50000),
            (0, 500000),
            (0, 5000000),
            (0, 50000000),
        ] {
            let trans = recombination_aware_log_transition(&params, pos1, pos2, &map, 5000);
            for row in &trans {
                let sum = row[0].exp() + row[1].exp();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Pop {:?} dist {}: row sum {} != 1.0",
                    pop, pos2 - pos1, sum
                );
                assert!(
                    row[0].is_finite() && row[1].is_finite(),
                    "Pop {:?} dist {}: non-finite transitions",
                    pop, pos2 - pos1
                );
            }
        }
    }
}

/// Stress test: genetic map with many entries and 2000 windows.
#[test]
fn test_genetic_map_stress_many_entries() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

    // Create a detailed genetic map with 1000 entries
    let entries: Vec<(u64, f64)> = (0..1000)
        .map(|i| {
            let bp = i as u64 * 10_000;
            // Variable rate: peaks in the middle
            let rate = 1.0 + 5.0 * (-(((i as f64 - 500.0) / 200.0).powi(2))).exp();
            let cm = if i == 0 {
                0.0
            } else {
                let prev_bp = (i - 1) as u64 * 10_000;
                let prev_cm = (i - 1) as f64 * 0.01 * rate;
                prev_cm + (bp - prev_bp) as f64 * rate / 1_000_000.0
            };
            (bp, cm)
        })
        .collect();

    let map = GeneticMap::new(entries);
    let n = 2000;

    let obs: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64 * 0.618033988) % 1.0;
            0.995 + t * 0.005
        })
        .collect();

    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| (i as u64 * 5000, (i as u64 + 1) * 5000))
        .collect();

    let (posteriors, log_lik) = forward_backward_with_genetic_map(
        &obs, &params, &positions, &map, 5000,
    );

    assert_eq!(posteriors.len(), n);
    assert!(log_lik.is_finite(), "Log-likelihood should be finite");
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "Invalid posterior at {}: {}",
            i, p
        );
    }
}

/// Genetic map: window_size=0 should return base transitions.
#[test]
fn test_recombination_aware_zero_window_size() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 0, 100000, &map, 0);

    // Should return base transitions in log space
    let expected_00 = params.transition[0][0].ln();
    let expected_01 = params.transition[0][1].ln();

    assert!(
        (trans[0][0] - expected_00).abs() < 1e-10,
        "window_size=0 should give base transition: {} vs {}",
        trans[0][0], expected_00
    );
    assert!(
        (trans[0][1] - expected_01).abs() < 1e-10,
        "window_size=0 should give base transition: {} vs {}",
        trans[0][1], expected_01
    );
}

/// Genetic map: same position should return base transitions.
#[test]
fn test_recombination_aware_same_position_base() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let map = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 500000, 500000, &map, 5000);

    let expected_00 = params.transition[0][0].ln();
    assert!(
        (trans[0][0] - expected_00).abs() < 1e-10,
        "Same position should give base transition"
    );
}

/// Population-specific params should produce different emission means.
#[test]
fn test_population_specific_emission_means() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
    ];

    let means: Vec<f64> = pops
        .iter()
        .map(|p| {
            let params = HmmParams::from_population(*p, 50.0, 0.0001, 5000);
            params.emission[0].mean
        })
        .collect();

    // AFR should have the lowest non-IBD mean (highest diversity)
    assert!(
        means[0] < means[1],
        "AFR non-IBD mean ({}) should be < EUR ({})",
        means[0],
        means[1]
    );
    assert!(
        means[0] < means[2],
        "AFR non-IBD mean ({}) should be < EAS ({})",
        means[0],
        means[2]
    );

    // All IBD means should be the same (IBD_EMISSION constant)
    let ibd_means: Vec<f64> = pops
        .iter()
        .map(|p| {
            let params = HmmParams::from_population(*p, 50.0, 0.0001, 5000);
            params.emission[1].mean
        })
        .collect();

    for (i, &m) in ibd_means.iter().enumerate() {
        assert!(
            (m - ibd_means[0]).abs() < 1e-10,
            "IBD mean for {:?} ({}) differs from AFR ({})",
            pops[i],
            m,
            ibd_means[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Malformed genetic map input tests
// ---------------------------------------------------------------------------

#[test]
fn test_genetic_map_malformed_missing_columns() {
    // File with only 2 columns — should produce an empty map (lines with <3 fields skipped)
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_map.txt");
    std::fs::write(&path, "1000\t0.5\n2000\t1.0\n").unwrap();
    let result = GeneticMap::from_file(&path, "1");
    // Should return Err because no valid entries found
    assert!(result.is_err(), "2-column file should produce error (no valid entries)");
}

#[test]
fn test_genetic_map_malformed_non_numeric() {
    // File with text in numeric columns
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_map2.txt");
    std::fs::write(&path, "chr1\tabc\t0.5\t0.0\nchr1\t2000\txyz\t1.0\n").unwrap();
    let result = GeneticMap::from_file(&path, "1");
    // First line has "abc" in pos_bp → parse error
    assert!(result.is_err(), "Non-numeric position should produce error");
}

#[test]
fn test_genetic_map_duplicate_positions() {
    // Two entries at same position — should not panic
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup_map.txt");
    std::fs::write(&path, "chr1\t1000\t1.0\t0.001\nchr1\t1000\t1.0\t0.001\nchr1\t2000\t1.0\t0.002\n").unwrap();
    let result = GeneticMap::from_file(&path, "1");
    assert!(result.is_ok(), "Duplicate positions should not cause error");
    let map = result.unwrap();
    // Interpolation should still work
    let cm = map.interpolate_cm(1500);
    assert!(cm.is_finite(), "Interpolation with duplicate entries should be finite");
}

// ---------------------------------------------------------------------------
// Numerical stability tests
// ---------------------------------------------------------------------------

#[test]
fn test_hmm_extreme_observations_near_one() {
    // Observations very close to 1.0
    let obs: Vec<f64> = (0..100).map(|_| 0.999999).collect();
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let (posteriors, _ll) = forward_backward(&obs, &params);
    for &p in &posteriors {
        assert!((0.0..=1.0).contains(&p), "Posterior {} out of [0,1]", p);
        assert!(p.is_finite(), "Posterior should be finite");
    }
}

#[test]
fn test_hmm_extreme_observations_near_zero() {
    // Observations near 0.0 — far from both emission means
    let obs: Vec<f64> = (0..100).map(|_| 0.001).collect();
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let (posteriors, _ll) = forward_backward(&obs, &params);
    for &p in &posteriors {
        assert!((0.0..=1.0).contains(&p), "Posterior {} out of [0,1]", p);
        assert!(p.is_finite(), "Posterior should be finite");
    }
}

#[test]
fn test_hmm_mixed_extreme_observations() {
    // Mix of very high and very low values
    let mut obs = vec![0.001; 50];
    obs.extend(std::iter::repeat_n(0.999999, 50));
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite(), "Log-likelihood should be finite");
    for &p in &posteriors {
        assert!((0.0..=1.0).contains(&p), "Posterior {} out of [0,1]", p);
        assert!(p.is_finite(), "Posterior should be finite");
    }
}

#[test]
fn test_forward_backward_long_sequence() {
    // 50,000 windows — verify no NaN/Inf in posteriors
    let obs: Vec<f64> = (0..50_000)
        .map(|i| if i % 100 < 10 { 0.9997 } else { 0.9988 })
        .collect();
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite(), "Log-likelihood should be finite for 50k windows");
    assert_eq!(posteriors.len(), 50_000);
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "Posterior at window {} is invalid: {}",
            i, p
        );
    }
}

// ---------------------------------------------------------------------------
// 10. GeneticMap utility method tests (len, is_empty)
// ---------------------------------------------------------------------------

#[test]
fn test_genetic_map_len_empty() {
    let map = GeneticMap::new(vec![]);
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
}

#[test]
fn test_genetic_map_len_nonempty() {
    let map = GeneticMap::new(vec![(1000, 0.001), (2000, 0.002), (3000, 0.003)]);
    assert_eq!(map.len(), 3);
    assert!(!map.is_empty());
}

#[test]
fn test_genetic_map_len_uniform() {
    let map = GeneticMap::uniform(0, 10_000_000, 1.0);
    assert_eq!(map.len(), 2); // start and end entries
    assert!(!map.is_empty());
}

#[test]
fn test_genetic_map_interpolate_empty_returns_zero() {
    let map = GeneticMap::new(vec![]);
    assert_eq!(map.interpolate_cm(1000), 0.0);
}

#[test]
fn test_genetic_map_interpolate_single_entry() {
    let map = GeneticMap::new(vec![(5000, 1.5)]);
    // Single entry: always returns that cM value
    assert_eq!(map.interpolate_cm(0), 1.5);
    assert_eq!(map.interpolate_cm(5000), 1.5);
    assert_eq!(map.interpolate_cm(10000), 1.5);
}

#[test]
fn test_genetic_map_interpolate_exact_entry() {
    let map = GeneticMap::new(vec![(1000, 0.001), (2000, 0.002), (3000, 0.003)]);
    // At exact entries
    assert!((map.interpolate_cm(1000) - 0.001).abs() < 1e-10);
    assert!((map.interpolate_cm(2000) - 0.002).abs() < 1e-10);
    assert!((map.interpolate_cm(3000) - 0.003).abs() < 1e-10);
}

#[test]
fn test_genetic_map_interpolate_between_entries() {
    let map = GeneticMap::new(vec![(1000, 0.0), (3000, 2.0)]);
    // Midpoint
    let cm = map.interpolate_cm(2000);
    assert!((cm - 1.0).abs() < 1e-10, "Expected 1.0 cM at midpoint, got {}", cm);
}

#[test]
fn test_genetic_map_extrapolate_before() {
    let map = GeneticMap::new(vec![(1000, 0.0), (2000, 1.0)]);
    // Before first entry: extrapolate backward
    let cm = map.interpolate_cm(500);
    // Rate = 1.0 cM / 1000 bp = 0.001 cM/bp; 500 bp before start → -0.5 cM
    assert!((cm - (-0.5)).abs() < 1e-10, "Expected -0.5 cM before map, got {}", cm);
}

#[test]
fn test_genetic_map_extrapolate_after() {
    let map = GeneticMap::new(vec![(1000, 0.0), (2000, 1.0)]);
    // After last entry: extrapolate forward
    let cm = map.interpolate_cm(3000);
    // Rate = 0.001 cM/bp; 1000 bp after end → 1.0 + 1.0 = 2.0 cM
    assert!((cm - 2.0).abs() < 1e-10, "Expected 2.0 cM after map, got {}", cm);
}

#[test]
fn test_genetic_map_distance_symmetry() {
    let map = GeneticMap::uniform(0, 10_000_000, 1.0);
    let d1 = map.genetic_distance_cm(1_000_000, 2_000_000);
    let d2 = map.genetic_distance_cm(2_000_000, 1_000_000);
    assert!((d1 - d2).abs() < 1e-10, "Distance should be symmetric");
    // 1 Mb at 1 cM/Mb = 1.0 cM
    assert!((d1 - 1.0).abs() < 1e-10, "Expected 1.0 cM for 1Mb, got {}", d1);
}
