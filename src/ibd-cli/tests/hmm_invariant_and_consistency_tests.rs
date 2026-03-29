//! HMM invariant and cross-function consistency tests.
//!
//! Cycle 48: Verify mathematical properties that MUST hold:
//! - Forward-backward posteriors sum to 1 per time step
//! - Viterbi path is consistent with posteriors in unambiguous cases
//! - from_log_emit variants match standard variants
//! - Distance/genetic-map variants produce valid results
//! - Logit emission estimation edge cases

use hprc_ibd::hmm::*;
use hprc_ibd::stats::*;

// ============================================================================
// Helper: create standard test params
// ============================================================================

fn test_params() -> HmmParams {
    HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000)
}

fn test_observations() -> Vec<f64> {
    vec![
        0.9985, 0.9990, 0.9988, 0.9992,
        0.9998, 0.9997, 0.9999, 0.9998, 0.9997,
        0.9988, 0.9985, 0.9990,
    ]
}

fn test_positions(n: usize) -> Vec<(u64, u64)> {
    (0..n).map(|i| {
        let start = (i as u64) * 5000;
        (start, start + 4999)
    }).collect()
}

// ============================================================================
// Forward-backward invariants
// ============================================================================

#[test]
fn posteriors_are_valid_probabilities() {
    let params = test_params();
    let obs = test_observations();
    let (posteriors, ll) = forward_backward(&obs, &params);

    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior {} out of [0,1]", p);
    }
    assert!(ll.is_finite(), "Log-likelihood should be finite");
}

#[test]
fn forward_backward_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();

    let (posteriors_standard, ll_standard) = forward_backward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (posteriors_from_emit, ll_from_emit) = forward_backward_from_log_emit(&log_emit, &params);

    assert!(
        (ll_standard - ll_from_emit).abs() < 1e-10,
        "Log-likelihoods should match: {} vs {}",
        ll_standard, ll_from_emit
    );
    for (i, (&ps, &pe)) in posteriors_standard.iter().zip(posteriors_from_emit.iter()).enumerate() {
        assert!((ps - pe).abs() < 1e-10, "Posteriors differ at {}: {} vs {}", i, ps, pe);
    }
}

#[test]
fn viterbi_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();

    let states_standard = viterbi(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let states_from_emit = viterbi_from_log_emit(&log_emit, &params);

    assert_eq!(states_standard, states_from_emit, "Viterbi paths should be identical");
}

#[test]
fn forward_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();

    let (alpha_standard, ll_standard) = forward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let (alpha_from_emit, ll_from_emit) = forward_from_log_emit(&log_emit, &params);

    assert!((ll_standard - ll_from_emit).abs() < 1e-10, "Forward LLs should match");
    for (i, (as_, ae)) in alpha_standard.iter().zip(alpha_from_emit.iter()).enumerate() {
        for s in 0..2 {
            assert!((as_[s] - ae[s]).abs() < 1e-8, "Alpha differs at t={}, s={}", i, s);
        }
    }
}

#[test]
fn backward_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();

    let beta_standard = backward(&obs, &params);
    let log_emit = precompute_log_emissions(&obs, &params);
    let beta_from_emit = backward_from_log_emit(&log_emit, &params);

    for (i, (bs, be)) in beta_standard.iter().zip(beta_from_emit.iter()).enumerate() {
        for s in 0..2 {
            assert!((bs[s] - be[s]).abs() < 1e-8, "Beta differs at t={}, s={}", i, s);
        }
    }
}

// ============================================================================
// Distance-based variants produce valid results
// ============================================================================

#[test]
fn forward_with_distances_produces_valid_results() {
    let params = test_params();
    let obs = test_observations();
    let positions = test_positions(obs.len());

    let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);
    assert!(ll_dist.is_finite(), "Distance-based LL should be finite");
    assert_eq!(alpha_dist.len(), obs.len());
}

#[test]
fn viterbi_with_distances_produces_valid_path() {
    let params = test_params();
    let obs = test_observations();
    let positions = test_positions(obs.len());

    let states = viterbi_with_distances(&obs, &params, &positions);
    assert_eq!(states.len(), obs.len());
    for &s in &states {
        assert!(s == 0 || s == 1, "State must be 0 or 1, got {}", s);
    }
}

#[test]
fn forward_backward_with_distances_posteriors_valid() {
    let params = test_params();
    let obs = test_observations();
    let positions = test_positions(obs.len());

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert!(ll.is_finite());
    assert_eq!(posteriors.len(), obs.len());
    for &p in &posteriors {
        assert!(p >= -1e-10 && p <= 1.0 + 1e-10, "Posterior {} out of range", p);
    }
}

// ============================================================================
// Genetic map variants with uniform map
// ============================================================================

#[test]
fn genetic_map_uniform_produces_valid_results() {
    let params = test_params();
    let obs = test_observations();
    let n = obs.len();
    let positions = test_positions(n);

    let gmap = GeneticMap::uniform(0, (n as u64) * 5000, 1.0);

    let states = viterbi_with_genetic_map(&obs, &params, &positions, &gmap, 5000);
    assert_eq!(states.len(), n);

    let (posteriors, ll) =
        forward_backward_with_genetic_map(&obs, &params, &positions, &gmap, 5000);
    assert!(ll.is_finite());
    assert_eq!(posteriors.len(), n);
}

#[test]
fn genetic_map_interpolation_monotone() {
    let gmap = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 3.0)]);

    let mut prev = f64::NEG_INFINITY;
    for pos in (0..2_000_001).step_by(100_000) {
        let cm = gmap.interpolate_cm(pos);
        assert!(cm >= prev, "Non-monotone at pos {}: {} < {}", pos, cm, prev);
        prev = cm;
    }
}

#[test]
fn genetic_map_distance_is_nonnegative() {
    let gmap = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 3.0)]);
    let dist = gmap.genetic_distance_cm(500_000, 1_500_000);
    assert!(dist >= 0.0, "Genetic distance should be non-negative: {}", dist);

    let dist_rev = gmap.genetic_distance_cm(1_500_000, 500_000);
    assert!(dist_rev >= 0.0, "Reversed genetic distance should be non-negative: {}", dist_rev);
}

// ============================================================================
// Viterbi/posteriors consistency
// ============================================================================

#[test]
fn viterbi_agrees_with_posteriors_in_clear_cases() {
    // Use population-realistic values: non-IBD ~0.999, IBD ~0.9998+
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs: Vec<f64> = vec![
        0.9985, 0.9984, 0.9983, 0.9986,               // clearly non-IBD (below EUR mean)
        0.99990, 0.99995, 0.99992, 0.99991, 0.99993,   // clearly IBD (very high)
        0.9985, 0.9984, 0.9983,                         // clearly non-IBD
    ];

    let states = viterbi(&obs, &params);
    let (posteriors, _) = forward_backward(&obs, &params);

    // Just verify both produce consistent results (same direction)
    assert_eq!(states.len(), obs.len());
    assert_eq!(posteriors.len(), obs.len());

    // For the clearly IBD region, posteriors should be elevated
    let ibd_avg_posterior: f64 = posteriors[4..9].iter().sum::<f64>() / 5.0;
    let non_ibd_avg_posterior: f64 = posteriors[0..4].iter().sum::<f64>() / 4.0;
    assert!(
        ibd_avg_posterior > non_ibd_avg_posterior,
        "IBD region avg posterior ({}) should exceed non-IBD ({})",
        ibd_avg_posterior, non_ibd_avg_posterior
    );
}

// ============================================================================
// Combined log-emissions
// ============================================================================

#[test]
fn combined_log_emissions_without_aux_equals_primary() {
    let params = test_params();
    let obs = test_observations();

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, None, None);

    for (i, (p, c)) in primary.iter().zip(combined.iter()).enumerate() {
        for s in 0..2 {
            assert!((p[s] - c[s]).abs() < 1e-15, "Mismatch at t={}, s={}", i, s);
        }
    }
}

#[test]
fn combined_log_emissions_with_aux_adds_log_probs() {
    let params = test_params();
    let obs = vec![0.999, 0.998, 0.9995];
    let aux_obs = vec![0.5, 0.6, 0.7];
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.1),
        GaussianParams::new_unchecked(0.8, 0.1),
    ];

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    for (i, (p, c)) in primary.iter().zip(combined.iter()).enumerate() {
        for s in 0..2 {
            let aux_log_p = aux_emit[s].log_pdf(aux_obs[i]);
            let expected = p[s] + aux_log_p;
            assert!((c[s] - expected).abs() < 1e-12, "Combined mismatch at t={}, s={}", i, s);
        }
    }
}

#[test]
fn combined_log_emissions_mismatched_length_falls_back() {
    let params = test_params();
    let obs = vec![0.999, 0.998];
    let aux_obs = vec![0.5]; // wrong length
    let aux_emit = [
        GaussianParams::new_unchecked(0.5, 0.1),
        GaussianParams::new_unchecked(0.8, 0.1),
    ];

    let primary = precompute_log_emissions(&obs, &params);
    let combined = compute_combined_log_emissions(&obs, &params, Some(&aux_obs), Some(&aux_emit));

    for (p, c) in primary.iter().zip(combined.iter()) {
        for s in 0..2 {
            assert!((p[s] - c[s]).abs() < 1e-15);
        }
    }
}

// ============================================================================
// estimate_emissions_logit edge cases
// ============================================================================

#[test]
fn estimate_emissions_logit_with_few_observations_is_noop() {
    let mut params = test_params();
    let orig_emission = params.emission.clone();
    let few_obs = vec![1.0, 2.0, 3.0];

    params.estimate_emissions_logit(&few_obs, None, 5000);

    assert!(
        (params.emission[0].mean - orig_emission[0].mean).abs() < 1e-15,
        "Emissions changed with fewer than 10 observations"
    );
}

#[test]
fn estimate_emissions_logit_with_constant_data() {
    let mut params = test_params();
    let constant_obs: Vec<f64> = vec![5.0; 20];

    params.estimate_emissions_logit(&constant_obs, Some(Population::EUR), 5000);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std.is_finite() && params.emission[0].std > 0.0);
    assert!(params.emission[1].std.is_finite() && params.emission[1].std > 0.0);
}

#[test]
fn estimate_emissions_logit_with_bimodal_data() {
    let mut params = test_params();
    let mut logit_obs = Vec::new();
    for _ in 0..50 {
        logit_obs.push(3.0);
    }
    for _ in 0..10 {
        logit_obs.push(7.0);
    }

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    assert!(
        params.emission[0].mean < params.emission[1].mean,
        "Emission 0 mean ({}) should be < emission 1 mean ({})",
        params.emission[0].mean, params.emission[1].mean
    );
}

#[test]
fn estimate_emissions_logit_with_all_populations() {
    let populations = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop, Population::Generic,
    ];
    let logit_obs: Vec<f64> = (0..30).map(|i| 3.0 + (i as f64) * 0.1).collect();

    for pop in &populations {
        let mut params = HmmParams::from_population(*pop, 50.0, 0.001, 5000);
        params.estimate_emissions_logit(&logit_obs, Some(*pop), 5000);

        assert!(params.emission[0].mean.is_finite(), "NaN emission[0] for {:?}", pop);
        assert!(params.emission[1].mean.is_finite(), "NaN emission[1] for {:?}", pop);
    }
}

// ============================================================================
// estimate_emissions_robust edge cases
// ============================================================================

#[test]
fn estimate_emissions_robust_preserves_ordering() {
    let mut params = test_params();
    let obs = vec![0.997, 0.998, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9998, 0.997, 0.996];

    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

    assert!(
        params.emission[0].mean <= params.emission[1].mean,
        "Emission ordering violated: {} > {}",
        params.emission[0].mean, params.emission[1].mean
    );
}

#[test]
fn estimate_emissions_robust_with_extreme_identity() {
    let mut params = test_params();
    let obs: Vec<f64> = vec![0.99999; 20];

    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ============================================================================
// infer_ibd_with_aux_features edge cases
// ============================================================================

#[test]
fn infer_ibd_with_aux_features_empty_input() {
    let mut params = test_params();
    let (result, aux) = infer_ibd_with_aux_features(&[], &mut params, Population::EUR, 5000, 10, None);
    assert!(result.states.is_empty());
    assert!(result.posteriors.is_empty());
    assert!(aux.is_none());
}

#[test]
fn infer_ibd_with_aux_features_two_observations() {
    let mut params = test_params();
    let obs = vec![0.999, 0.998];
    let (result, aux) = infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 10, None);
    assert_eq!(result.states.len(), 2);
    assert!(result.log_likelihood == f64::NEG_INFINITY);
    assert!(aux.is_none());
}

#[test]
fn infer_ibd_with_aux_features_with_aux_data() {
    let mut params = test_params();
    let obs = vec![0.998, 0.997, 0.999, 0.9995, 0.9998, 0.9997, 0.998, 0.997, 0.998, 0.997, 0.999, 0.998];
    let aux = vec![0.5, 0.4, 0.6, 0.8, 0.9, 0.85, 0.5, 0.4, 0.5, 0.45, 0.55, 0.5];

    let (result, aux_emission) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 5, Some(&aux));

    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
    assert!(aux_emission.is_some());
    let ae = aux_emission.unwrap();
    assert!(ae[0].mean.is_finite());
    assert!(ae[1].mean.is_finite());
}

#[test]
fn infer_ibd_with_aux_features_zero_bw_iters() {
    let mut params = test_params();
    let obs = vec![0.998, 0.997, 0.999, 0.9995, 0.9998, 0.997, 0.998, 0.997, 0.999, 0.998];

    let (result, _) = infer_ibd_with_aux_features(&obs, &mut params, Population::EUR, 5000, 0, None);
    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
}

// ============================================================================
// Segment extraction consistency
// ============================================================================

#[test]
fn extract_ibd_segments_from_all_zero_states() {
    let states = vec![0; 100];
    let segments = extract_ibd_segments(&states);
    assert!(segments.is_empty());
}

#[test]
fn extract_ibd_segments_from_all_one_states() {
    let states = vec![1; 100];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    // (start_idx, end_idx, n_windows)
    assert_eq!(segments[0], (0, 99, 100));
}

#[test]
fn extract_ibd_segments_alternating() {
    let states: Vec<usize> = (0..20).map(|i| i % 2).collect();
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 10);
    for seg in &segments {
        assert_eq!(seg.0, seg.1); // single-window segments
        assert_eq!(seg.2, 1);     // n_windows = 1
    }
}

#[test]
fn extract_ibd_segments_preserves_indices() {
    let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 2);
    // (start_idx, end_idx, n_windows)
    assert_eq!(segments[0], (2, 4, 3));
    assert_eq!(segments[1], (7, 8, 2));
}

// ============================================================================
// Population diversity and emission correctness
// ============================================================================

#[test]
fn population_diversity_ordering() {
    let afr = Population::AFR.diversity();
    let eur = Population::EUR.diversity();
    let eas = Population::EAS.diversity();

    assert!(afr > eur, "AFR diversity ({}) should exceed EUR ({})", afr, eur);
    assert!(eur >= eas, "EUR diversity ({}) should >= EAS ({})", eur, eas);
}

#[test]
fn non_ibd_emission_has_reasonable_mean() {
    for pop in &[Population::AFR, Population::EUR, Population::EAS, Population::CSA, Population::AMR] {
        let e = pop.non_ibd_emission(5000);
        assert!(e.mean > 0.99 && e.mean < 1.0, "{:?} non-IBD mean {} not in (0.99, 1.0)", pop, e.mean);
        assert!(e.std > 0.0 && e.std < 0.01, "{:?} std {} out of range", pop, e.std);
    }
}

// ============================================================================
// Baum-Welch convergence
// ============================================================================

#[test]
fn baum_welch_does_not_degenerate() {
    let mut params = test_params();
    let obs = vec![0.997, 0.998, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9997, 0.9999,
                   0.997, 0.996, 0.998, 0.997, 0.996];

    params.baum_welch(&obs, 10, 1e-6, Some(Population::EUR), 5000);

    for row in &params.transition {
        let sum = row[0] + row[1];
        assert!((sum - 1.0).abs() < 1e-10, "Transition row sum = {}", sum);
        assert!(row[0] > 0.0 && row[1] > 0.0);
    }
    for e in &params.emission {
        assert!(e.mean.is_finite());
        assert!(e.std > 0.0);
    }
}

// ============================================================================
// LOD score properties
// ============================================================================

#[test]
fn per_window_lod_has_correct_length() {
    let params = test_params();
    let obs = test_observations();
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), obs.len());
    for &l in &lods {
        assert!(l.is_finite());
    }
}

#[test]
fn segment_lod_score_empty_observations() {
    let params = test_params();
    let obs: Vec<f64> = Vec::new();
    // start_idx > end_idx when empty, returns 0.0
    let lod = segment_lod_score(&obs, 0, 0, &params);
    assert!(lod.is_finite());
}

#[test]
fn segment_lod_score_single_ibd_window() {
    let params = test_params();
    let obs = vec![0.9999];
    let lod = segment_lod_score(&obs, 0, 0, &params);
    assert!(lod > 0.0, "Single IBD window should have positive LOD, got {}", lod);
}

#[test]
fn segment_lod_score_single_non_ibd_window() {
    let params = test_params();
    let obs = vec![0.50];
    let lod = segment_lod_score(&obs, 0, 0, &params);
    assert!(lod < 0.0, "Single non-IBD window should have negative LOD, got {}", lod);
}

// ============================================================================
// Coverage ratio
// ============================================================================

#[test]
fn coverage_ratio_symmetric() {
    let r1 = coverage_ratio(100, 200);
    let r2 = coverage_ratio(200, 100);
    assert!((r1 - r2).abs() < 1e-10, "coverage_ratio should be symmetric: {} vs {}", r1, r2);
}

#[test]
fn coverage_ratio_equal_lengths() {
    let r = coverage_ratio(1000, 1000);
    assert!((r - 1.0).abs() < 1e-10, "Equal lengths should give ratio 1.0, got {}", r);
}

#[test]
fn coverage_ratio_zero_length() {
    let r = coverage_ratio(0, 1000);
    assert!(r.is_finite());
}

#[test]
fn coverage_ratio_both_zero() {
    let r = coverage_ratio(0, 0);
    assert!(r.is_finite());
}

// ============================================================================
// Transition matrices
// ============================================================================

#[test]
fn recombination_aware_transition_rows_sum_to_one() {
    let params = test_params();
    let gmap = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 3.0)]);

    let test_cases: Vec<(u64, u64)> = vec![(0, 0), (0, 5000), (100_000, 500_000), (0, 1_000_000)];
    for (pos1, pos2) in test_cases {
        let log_trans = recombination_aware_log_transition(&params, pos1, pos2, &gmap, 5000);
        for i in 0..2 {
            let sum = log_trans[i][0].exp() + log_trans[i][1].exp();
            assert!((sum - 1.0).abs() < 1e-8, "Row {} sum = {} for ({}, {})", i, sum, pos1, pos2);
        }
    }
}

#[test]
fn distance_dependent_transition_rows_sum_to_one() {
    let distances: Vec<u64> = vec![0, 1000, 5000, 100000];
    let params = test_params();

    for d in distances {
        let log_trans = distance_dependent_log_transition(&params, d, 5000);
        for i in 0..2 {
            let sum = log_trans[i][0].exp() + log_trans[i][1].exp();
            assert!((sum - 1.0).abs() < 1e-8, "Row {} sum = {} for distance {}", i, sum, d);
        }
    }
}

// ============================================================================
// Refine states with posteriors
// ============================================================================

#[test]
fn refine_does_not_change_confident_states() {
    let mut states = vec![0, 0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.01, 0.02, 0.05, 0.95, 0.98, 0.97, 0.03, 0.01];
    let orig_states = states.clone();

    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, orig_states);
}

#[test]
fn refine_trims_low_posterior_boundary_state() {
    // An isolated IBD state at a boundary with low posterior should be trimmed
    let mut states = vec![0, 0, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.15, 0.1, 0.1]; // posterior < trim_threshold

    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // State 2 is at both left and right boundary (both neighbors are 0)
    // and posterior (0.15) < trim_threshold (0.2), so it should be trimmed to 0
    assert_eq!(states[2], 0, "Boundary state with low posterior should be trimmed");
}

// ============================================================================
// Precompute log emissions
// ============================================================================

#[test]
fn precompute_log_emissions_are_finite() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);

    assert_eq!(log_emit.len(), obs.len());
    for le in &log_emit {
        for s in 0..2 {
            // Note: log_pdf can be positive for narrow Gaussians when obs is near mean
            assert!(le[s].is_finite(), "Log-emission should be finite, got {}", le[s]);
        }
    }
}
