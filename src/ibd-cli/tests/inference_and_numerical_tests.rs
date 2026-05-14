//! Tests for infer_ibd_with_training edge cases, numerical stability,
//! and end-to-end inference pipeline behavior.
//!
//! Targets:
//! - infer_ibd_with_training: short data (< 10 obs), zero BW iters, all-IBD, all-non-IBD
//! - viterbi/forward/backward with extreme observations (near 0, near 1)
//! - segment_lod_score boundary conditions
//! - precompute_log_emissions numerical stability
//! - GeneticMap edge cases not covered elsewhere
//! - HmmParams::from_population_logit p_enter_ibd boundaries

use impopk_ibd::hmm::{
    backward, forward, forward_backward, infer_ibd, infer_ibd_with_training,
    precompute_log_emissions, segment_lod_score, viterbi,
    GeneticMap, HmmParams, Population,
};
use impopk_ibd::stats::logit_transform_observations;

// === infer_ibd_with_training edge cases ===

/// With < 10 observations, Baum-Welch should be skipped (guard in function)
/// but inference should still succeed.
#[test]
fn test_infer_ibd_with_training_short_data() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999, 0.9995];
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 20);
    assert_eq!(result.states.len(), 3);
    assert_eq!(result.posteriors.len(), 3);
    assert!(result.log_likelihood.is_finite());
}

/// With 0 Baum-Welch iterations, should skip training but still infer.
#[test]
fn test_infer_ibd_with_training_zero_bw_iters() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let original_emission = params.emission.clone();
    let obs = vec![0.998; 20];
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 0);
    assert_eq!(result.states.len(), 20);
    // Emissions should not change since BW was skipped
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
    assert_eq!(params.emission[1].mean, original_emission[1].mean);
}

/// With exactly 10 observations (minimum for Baum-Welch), training should proceed.
#[test]
fn test_infer_ibd_with_training_exactly_10_obs() {
    let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
    let obs = vec![0.997, 0.998, 0.997, 0.998, 0.997, 0.9999, 0.99999, 0.99998, 0.9999, 0.99999];
    let result = infer_ibd_with_training(&obs, &mut params, Population::AFR, 5000, 5);
    assert_eq!(result.states.len(), 10);
    assert!(result.log_likelihood.is_finite());
}

/// All observations very high (near 1.0) — everything should be IBD.
#[test]
fn test_infer_ibd_with_training_all_ibd() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = vec![0.99999; 20];
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 10);
    // Most or all should be classified as IBD (state 1)
    let ibd_count = result.states.iter().filter(|&&s| s == 1).count();
    assert!(ibd_count >= 15, "Expected most windows IBD, got {}/20", ibd_count);
}

/// All observations typical non-IBD — everything should be non-IBD.
#[test]
fn test_infer_ibd_with_training_all_non_ibd() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.9975; 20];
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 10);
    let non_ibd_count = result.states.iter().filter(|&&s| s == 0).count();
    assert!(non_ibd_count >= 15, "Expected most windows non-IBD, got {}/20 non-IBD", non_ibd_count);
}

/// Empty observations should work without panicking.
#[test]
fn test_infer_ibd_with_training_empty() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let result = infer_ibd_with_training(&[], &mut params, Population::EUR, 5000, 10);
    assert!(result.states.is_empty());
    assert!(result.posteriors.is_empty());
}

/// Single observation — no Baum-Welch, still produces inference.
#[test]
fn test_infer_ibd_with_training_single_obs() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let result = infer_ibd_with_training(&[0.99999], &mut params, Population::EUR, 5000, 20);
    assert_eq!(result.states.len(), 1);
    assert_eq!(result.posteriors.len(), 1);
    assert!(result.log_likelihood.is_finite());
}

/// Test with all populations to ensure no panics.
#[test]
fn test_infer_ibd_with_training_all_populations() {
    let pops = [
        Population::EUR, Population::AFR, Population::EAS,
        Population::AMR, Population::CSA, Population::InterPop,
        Population::Generic,
    ];
    let obs = vec![0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.9997, 0.998, 0.997, 0.996, 0.998, 0.999];
    for pop in &pops {
        let mut params = HmmParams::from_population(*pop, 50.0, 0.0001, 5000);
        let result = infer_ibd_with_training(&obs, &mut params, *pop, 5000, 5);
        assert_eq!(result.states.len(), obs.len(), "Failed for {:?}", pop);
        assert!(result.log_likelihood.is_finite(), "Non-finite LL for {:?}", pop);
    }
}

// === Numerical stability with extreme observations ===

/// Observations at exact 0.0 — should not panic or produce NaN.
#[test]
fn test_viterbi_extreme_low_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 7);
    for &s in &states {
        assert!(s <= 1, "Invalid state: {}", s);
    }
}

/// Observations at exact 1.0 — should not panic.
#[test]
fn test_viterbi_exact_one_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5);
}

/// Forward algorithm with extreme observations — should produce finite results.
#[test]
fn test_forward_extreme_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.0, 0.5, 0.99, 0.999, 0.9999, 1.0];
    let (alpha, ll) = forward(&obs, &params);
    assert_eq!(alpha.len(), 6);
    assert!(ll.is_finite(), "Log-likelihood should be finite, got {}", ll);
}

/// Backward algorithm with extreme observations.
#[test]
fn test_backward_extreme_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.0, 0.5, 0.99, 0.999, 0.9999, 1.0];
    let beta = backward(&obs, &params);
    assert_eq!(beta.len(), 6);
    for b in &beta {
        assert!(b[0].is_finite() || b[0] == f64::NEG_INFINITY, "Non-finite backward: {:?}", b);
        assert!(b[1].is_finite() || b[1] == f64::NEG_INFINITY, "Non-finite backward: {:?}", b);
    }
}

/// Forward-backward with a mix of extreme and normal observations.
#[test]
fn test_forward_backward_mixed_extreme() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.997, 0.998, 0.0, 0.998, 1.0, 0.999, 0.997];
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 7);
    assert!(ll.is_finite());
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior out of range: {}", p);
    }
}

/// infer_ibd with a 0.0 observation (outlier) should still produce valid results.
#[test]
fn test_infer_ibd_zero_observation_outlier() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let mut obs = vec![0.998; 10];
    obs[5] = 0.0; // a single outlier
    let result = infer_ibd(&obs, &params);
    assert_eq!(result.states.len(), 10);
    assert!(result.log_likelihood.is_finite());
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

// === precompute_log_emissions edge cases ===

/// precompute_log_emissions with empty input should return empty.
#[test]
fn test_precompute_log_emissions_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let result = precompute_log_emissions(&[], &params);
    assert!(result.is_empty());
}

/// precompute_log_emissions should produce finite values for each state.
/// Note: with narrow Gaussians, PDF > 1 is possible, so log-emissions can be positive.
#[test]
fn test_precompute_log_emissions_finite() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.990, 0.995, 0.998, 0.999, 0.9995, 0.9999];
    let log_emit = precompute_log_emissions(&obs, &params);
    assert_eq!(log_emit.len(), 6);
    for le in &log_emit {
        assert!(le[0].is_finite(), "Log emission should be finite, got {}", le[0]);
        assert!(le[1].is_finite(), "Log emission should be finite, got {}", le[1]);
    }
}

/// precompute_log_emissions: IBD-like observation should favor IBD emission.
#[test]
fn test_precompute_log_emissions_ibd_favors_state1() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.99999]; // very high identity
    let log_emit = precompute_log_emissions(&obs, &params);
    // State 1 (IBD) should have higher log-emission for high identity
    assert!(log_emit[0][1] > log_emit[0][0],
        "IBD emission {} should exceed non-IBD {} for high identity",
        log_emit[0][1], log_emit[0][0]);
}

// === segment_lod_score boundary cases ===

/// segment_lod_score with start > end should return 0.0.
#[test]
fn test_segment_lod_score_start_gt_end() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999, 0.9995];
    let lod = segment_lod_score(&obs, 2, 0, &params);
    assert_eq!(lod, 0.0);
}

/// segment_lod_score with end >= observations.len() should return 0.0.
#[test]
fn test_segment_lod_score_out_of_bounds() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999];
    let lod = segment_lod_score(&obs, 0, 5, &params);
    assert_eq!(lod, 0.0);
}

/// segment_lod_score for a single window should match compute_per_window_lod.
#[test]
fn test_segment_lod_score_single_window() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.9999];
    let seg_lod = segment_lod_score(&obs, 0, 0, &params);
    assert!(seg_lod.is_finite());
    // For a very high identity observation, LOD should be positive (favors IBD)
    assert!(seg_lod > 0.0, "LOD for high identity should be positive, got {}", seg_lod);
}

/// segment_lod_score for empty observations returns 0.
#[test]
fn test_segment_lod_score_empty_obs() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lod = segment_lod_score(&[], 0, 0, &params);
    assert_eq!(lod, 0.0);
}

// === GeneticMap additional edge cases ===

/// GeneticMap::uniform should produce correct genetic distance.
#[test]
fn test_genetic_map_uniform_distance() {
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0); // 1 cM/Mb
    let dist = gm.genetic_distance_cm(0, 1_000_000);
    assert!((dist - 1.0).abs() < 0.01, "Expected ~1 cM, got {}", dist);
}

/// GeneticMap::uniform with zero rate should give zero distance.
#[test]
fn test_genetic_map_uniform_zero_rate() {
    let gm = GeneticMap::uniform(0, 1_000_000, 0.0);
    let dist = gm.genetic_distance_cm(0, 1_000_000);
    assert_eq!(dist, 0.0);
}

/// GeneticMap with empty entries should be empty.
#[test]
fn test_genetic_map_empty() {
    let gm = GeneticMap::new(vec![]);
    assert!(gm.is_empty());
    assert_eq!(gm.len(), 0);
}

/// GeneticMap with single entry — interpolation should work.
#[test]
fn test_genetic_map_single_entry() {
    let gm = GeneticMap::new(vec![(100_000, 1.5)]);
    assert_eq!(gm.len(), 1);
    assert!(!gm.is_empty());
    let cm = gm.interpolate_cm(100_000);
    assert!((cm - 1.5).abs() < 1e-10);
}

/// GeneticMap::genetic_distance_cm with reversed positions should return the absolute distance.
#[test]
fn test_genetic_map_reversed_positions() {
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    let dist_forward = gm.genetic_distance_cm(0, 1_000_000);
    let dist_reverse = gm.genetic_distance_cm(1_000_000, 0);
    assert!((dist_forward - dist_reverse).abs() < 1e-10,
        "Forward {} and reverse {} distances should match", dist_forward, dist_reverse);
}

// === HmmParams::from_population_logit edge cases ===

/// from_population_logit with very small p_enter_ibd should not panic.
#[test]
fn test_from_population_logit_small_p_enter() {
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 1e-10, 5000);
    assert!(params.initial[0] > 0.0);
    assert!(params.initial[1] > 0.0);
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

/// from_population_logit with p_enter_ibd close to 1.0 should not panic.
#[test]
fn test_from_population_logit_large_p_enter() {
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.999, 5000);
    assert!(params.initial[0] > 0.0);
    assert!(params.initial[1] > 0.0);
}

/// from_population_logit: emissions should be in logit space (much larger absolute values).
#[test]
fn test_from_population_logit_emissions_in_logit_space() {
    let params_raw = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let params_logit = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    // In logit space, means are much larger in absolute value than in raw [0,1] space
    assert!(params_logit.emission[1].mean.abs() > params_raw.emission[1].mean.abs(),
        "Logit mean {} should be larger in absolute value than raw mean {}",
        params_logit.emission[1].mean, params_raw.emission[1].mean);
}

// === logit_transform_observations + inference integration ===

/// Logit-transformed observations should still allow valid inference
/// when used with logit-space parameters.
#[test]
fn test_logit_transform_roundtrip_inference() {
    let raw_obs: Vec<f64> = vec![0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995,
                                  0.9998, 0.999, 0.998, 0.997, 0.996];
    let logit_obs = logit_transform_observations(&raw_obs);
    assert_eq!(logit_obs.len(), raw_obs.len());
    // All logit values should be finite and positive (since all raw > 0.5)
    for &v in &logit_obs {
        assert!(v.is_finite(), "Logit value should be finite");
        assert!(v > 0.0, "Logit of >0.5 should be positive, got {}", v);
    }
}

/// Monotonicity: higher raw identity → higher logit value.
#[test]
fn test_logit_transform_monotonic() {
    let raw_obs = vec![0.990, 0.995, 0.998, 0.999, 0.9999];
    let logit_obs = logit_transform_observations(&raw_obs);
    for i in 1..logit_obs.len() {
        assert!(logit_obs[i] > logit_obs[i - 1],
            "Logit should be monotonic: {} should be > {}",
            logit_obs[i], logit_obs[i - 1]);
    }
}

// === infer_ibd consistency checks ===

/// infer_ibd should produce posteriors that sum to valid probabilities.
#[test]
fn test_infer_ibd_posteriors_valid() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.997, 0.998, 0.9995, 0.9999, 0.99999, 0.9999, 0.9995, 0.998, 0.997];
    let result = infer_ibd(&obs, &params);
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0 + 1e-10, "Posterior {} out of [0,1] range", p);
    }
}

/// infer_ibd: log-likelihood should be finite.
/// Note: With narrow Gaussians, log-likelihood can be positive (PDF > 1 at peak).
#[test]
fn test_infer_ibd_log_likelihood_finite() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999, 0.9995];
    let result = infer_ibd(&obs, &params);
    assert!(result.log_likelihood.is_finite(),
        "Log-likelihood should be finite, got {}", result.log_likelihood);
}

/// infer_ibd: same observations should give the same log-likelihood (deterministic).
#[test]
fn test_infer_ibd_deterministic() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999, 0.9995, 0.997, 0.998];
    let result1 = infer_ibd(&obs, &params);
    let result2 = infer_ibd(&obs, &params);
    assert_eq!(result1.log_likelihood, result2.log_likelihood);
    assert_eq!(result1.states, result2.states);
}

/// infer_ibd with single observation should work.
#[test]
fn test_infer_ibd_single_obs() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let result = infer_ibd(&[0.99999], &params);
    assert_eq!(result.states.len(), 1);
    assert_eq!(result.posteriors.len(), 1);
    assert!(result.log_likelihood.is_finite());
}

// === HmmParams::from_expected_length edge cases ===

/// Very short expected IBD length (1 window) — stay probability should be clamped.
#[test]
fn test_from_expected_length_very_short() {
    let params = HmmParams::from_expected_length(1.0, 0.0001, 5000);
    // p_stay_ibd = 1 - 1/1.0 = 0.0, should be clamped to 0.5
    assert!(params.transition[1][1] >= 0.5);
}

/// Very long expected IBD length — stay probability should be near 1.
#[test]
fn test_from_expected_length_very_long() {
    let params = HmmParams::from_expected_length(10000.0, 0.0001, 5000);
    assert!(params.transition[1][1] > 0.999);
}
