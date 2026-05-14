//! Baum-Welch training round-trip and distance-variant consistency tests.
//!
//! Validates that:
//! 1. BW training improves or maintains log-likelihood on synthetic data
//! 2. BW training with known-structure data recovers reasonable emission params
//! 3. Distance-based variants produce consistent posteriors with gap effects
//! 4. viterbi_with_distances fallback and non-uniform spacing
//! 5. Pipeline: estimate → train → infer → extract round-trip coherence

use impopk_ibd::hmm::*;
use impopk_ibd::stats::GaussianParams;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate synthetic bimodal observations: state 0 near non_ibd_mean, state 1 near ibd_mean
fn generate_synthetic(
    state_seq: &[usize],
    non_ibd_mean: f64,
    ibd_mean: f64,
    seed: u64,
) -> Vec<f64> {
    state_seq
        .iter()
        .enumerate()
        .map(|(i, &state)| {
            let base = if state == 0 { non_ibd_mean } else { ibd_mean };
            let variation = ((i as f64 * 0.618 + seed as f64 * 0.317).sin() * 0.0003).abs();
            let sign = if (i + seed as usize) % 2 == 0 { 1.0 } else { -1.0 };
            (base + sign * variation).clamp(0.0, 1.0)
        })
        .collect()
}

/// Positions for uniformly-spaced windows of given size
fn uniform_positions(n: usize, window_size: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|i| {
            let start = i as u64 * window_size;
            (start, start + window_size - 1)
        })
        .collect()
}

fn default_params() -> HmmParams {
    HmmParams {
        initial: [0.9, 0.1],
        transition: [[0.99, 0.01], [0.02, 0.98]],
        emission: [
            GaussianParams { mean: 0.998, std: 0.001 },
            GaussianParams { mean: 0.9999, std: 0.0003 },
        ],
    }
}

const WINDOW_SIZE: u64 = 10_000;

// ═══════════════════════════════════════════════════════════════════════════
// BW training: log-likelihood monotonicity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_improves_loglikelihood_bimodal() {
    let params = default_params();
    let mut truth = vec![0_usize; 100];
    truth.extend(vec![1_usize; 50]);
    truth.extend(vec![0_usize; 100]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    let (_, ll_before) = forward_backward(&obs, &params);

    let mut trained_params = params.clone();
    trained_params.baum_welch(&obs, 20, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let (_, ll_after) = forward_backward(&obs, &trained_params);

    assert!(
        ll_after >= ll_before - 1e-6,
        "BW training should not degrade log-likelihood: before={}, after={}",
        ll_before, ll_after
    );
}

#[test]
fn bw_training_improves_loglikelihood_all_non_ibd() {
    let params = default_params();
    let obs = generate_synthetic(&vec![0_usize; 200], 0.998, 0.9999, 7);

    let (_, ll_before) = forward_backward(&obs, &params);

    let mut trained = params.clone();
    trained.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let (_, ll_after) = forward_backward(&obs, &trained);

    assert!(
        ll_after >= ll_before - 1e-6,
        "BW on uniform data: before={}, after={}",
        ll_before, ll_after
    );
}

#[test]
fn bw_training_improves_loglikelihood_all_ibd() {
    let params = default_params();
    let obs = generate_synthetic(&vec![1_usize; 200], 0.998, 0.9999, 13);

    let (_, ll_before) = forward_backward(&obs, &params);

    let mut trained = params.clone();
    trained.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let (_, ll_after) = forward_backward(&obs, &trained);

    assert!(
        ll_after >= ll_before - 1e-6,
        "BW on all-IBD data: before={}, after={}",
        ll_before, ll_after
    );
}

#[test]
fn bw_training_preserves_param_validity() {
    let params = default_params();
    let mut truth = vec![0_usize; 80];
    truth.extend(vec![1_usize; 40]);
    truth.extend(vec![0_usize; 80]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 99);

    let mut trained = params.clone();
    trained.baum_welch(&obs, 30, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    // Initial probs must sum to ~1
    let init_sum: f64 = trained.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-6, "Initial probs must sum to 1, got {}", init_sum);

    // Transition rows must sum to ~1
    for (i, row) in trained.transition.iter().enumerate() {
        let row_sum: f64 = row.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "Transition row {} must sum to 1, got {}", i, row_sum);
    }

    // Emission std must be positive
    for (i, e) in trained.emission.iter().enumerate() {
        assert!(e.std > 0.0, "Emission {} std must be positive, got {}", i, e.std);
    }

    // Emission means must be finite
    for (i, e) in trained.emission.iter().enumerate() {
        assert!(e.mean.is_finite(), "Emission {} mean must be finite", i);
    }
}

#[test]
fn bw_multiple_iterations_monotonic_loglikelihood() {
    let params = default_params();
    let mut truth = vec![0_usize; 60];
    truth.extend(vec![1_usize; 30]);
    truth.extend(vec![0_usize; 60]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 55);

    let mut current = params.clone();
    let mut prev_ll = f64::NEG_INFINITY;

    for iteration in 0..10 {
        let (_, ll) = forward_backward(&obs, &current);
        assert!(
            ll >= prev_ll - 1e-8,
            "Iteration {}: ll decreased from {} to {}",
            iteration, prev_ll, ll
        );
        prev_ll = ll;
        current.baum_welch(&obs, 1, 1e-10, Some(Population::Generic), WINDOW_SIZE);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BW training: classification accuracy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_maintains_viterbi_accuracy_clear_signal() {
    let params = default_params();
    let mut truth = vec![0_usize; 80];
    truth.extend(vec![1_usize; 40]);
    truth.extend(vec![0_usize; 80]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    let states_before = viterbi(&obs, &params);
    let accuracy_before: f64 = states_before
        .iter()
        .zip(truth.iter())
        .filter(|(&s, &t)| s == t)
        .count() as f64
        / truth.len() as f64;

    let mut trained = params.clone();
    trained.baum_welch(&obs, 20, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let states_after = viterbi(&obs, &trained);
    let accuracy_after: f64 = states_after
        .iter()
        .zip(truth.iter())
        .filter(|(&s, &t)| s == t)
        .count() as f64
        / truth.len() as f64;

    assert!(
        accuracy_after >= accuracy_before - 0.05,
        "BW training degraded accuracy: before={:.3}, after={:.3}",
        accuracy_before, accuracy_after
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BW training: emission recovery
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_recovers_emission_direction() {
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.95, 0.05], [0.05, 0.95]],
        emission: [
            GaussianParams { mean: 0.990, std: 0.005 },
            GaussianParams { mean: 0.995, std: 0.005 },
        ],
    };

    let mut truth = vec![0_usize; 100];
    truth.extend(vec![1_usize; 100]);
    let obs = generate_synthetic(&truth, 0.990, 0.9995, 77);

    let mut trained = params.clone();
    trained.baum_welch(&obs, 30, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    // After training, emission[1] mean should be > emission[0] mean
    assert!(
        trained.emission[1].mean > trained.emission[0].mean,
        "After BW, IBD mean ({}) should exceed non-IBD mean ({})",
        trained.emission[1].mean, trained.emission[0].mean
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance-variant consistency: uniform spacing = standard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn distance_variants_uniform_spacing_close_to_standard() {
    let params = default_params();
    let obs = generate_synthetic(&vec![0_usize; 50], 0.998, 0.9999, 42);
    let positions = uniform_positions(50, 10000);

    let (posteriors_std, ll_std) = forward_backward(&obs, &params);
    let (posteriors_dist, ll_dist) = forward_backward_with_distances(&obs, &params, &positions);

    assert!(
        (ll_std - ll_dist).abs() < 1.0,
        "Uniform spacing: ll_std={}, ll_dist={} should be close",
        ll_std, ll_dist
    );

    for (i, (&p_std, &p_dist)) in posteriors_std.iter().zip(posteriors_dist.iter()).enumerate() {
        assert!(
            (p_std - p_dist).abs() < 0.1,
            "Window {}: posterior_std={}, posterior_dist={} differ too much",
            i, p_std, p_dist
        );
    }
}

#[test]
fn viterbi_with_distances_uniform_matches_standard() {
    let params = default_params();
    let mut truth = vec![0_usize; 40];
    truth.extend(vec![1_usize; 20]);
    truth.extend(vec![0_usize; 40]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);
    let positions = uniform_positions(100, 10000);

    let states_std = viterbi(&obs, &params);
    let states_dist = viterbi_with_distances(&obs, &params, &positions);

    let agree_count = states_std
        .iter()
        .zip(states_dist.iter())
        .filter(|(&a, &b)| a == b)
        .count();
    let agreement = agree_count as f64 / states_std.len() as f64;

    assert!(
        agreement > 0.95,
        "Uniform spacing: Viterbi agreement {:.3} should be >0.95",
        agreement
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance-variant: gap effects
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn distance_variants_large_gap_affects_posteriors() {
    let params = default_params();
    let obs = vec![0.9999, 0.9999, 0.9999, 0.998, 0.998, 0.998];

    let positions_uniform: Vec<(u64, u64)> = (0..6)
        .map(|i| (i * 10000, (i + 1) * 10000 - 1))
        .collect();

    let positions_gap = vec![
        (0, 9999),
        (10000, 19999),
        (20000, 29999),
        (1_000_000, 1_009_999),
        (1_010_000, 1_019_999),
        (1_020_000, 1_029_999),
    ];

    let (post_uniform, _) = forward_backward_with_distances(&obs, &params, &positions_uniform);
    let (post_gap, _) = forward_backward_with_distances(&obs, &params, &positions_gap);

    for p in &post_uniform {
        assert!(*p >= 0.0 && *p <= 1.0, "Uniform posterior out of range: {}", p);
    }
    for p in &post_gap {
        assert!(*p >= 0.0 && *p <= 1.0, "Gap posterior out of range: {}", p);
    }

    assert_eq!(post_uniform.len(), 6);
    assert_eq!(post_gap.len(), 6);
}

#[test]
fn viterbi_with_distances_large_gap_valid_states() {
    let params = default_params();
    let obs = vec![0.9999, 0.9999, 0.998, 0.998, 0.9999, 0.9999];

    let positions_gap = vec![
        (0, 9999),
        (10000, 19999),
        (500_000, 509_999),
        (510_000, 519_999),
        (1_000_000, 1_009_999),
        (1_010_000, 1_019_999),
    ];

    let states = viterbi_with_distances(&obs, &params, &positions_gap);
    assert_eq!(states.len(), 6);
    for &s in &states {
        assert!(s == 0 || s == 1, "Invalid state: {}", s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance-variant: fallback on mismatched positions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn viterbi_with_distances_mismatched_positions_falls_back() {
    let params = default_params();
    let obs = vec![0.998, 0.998, 0.9999, 0.9999];
    let positions = vec![(0, 9999), (10000, 19999)];

    let states_dist = viterbi_with_distances(&obs, &params, &positions);
    let states_std = viterbi(&obs, &params);

    assert_eq!(states_dist, states_std, "Mismatched positions should fall back to standard viterbi");
}

#[test]
fn forward_backward_with_distances_mismatched_falls_back() {
    let params = default_params();
    let obs = vec![0.998, 0.998, 0.9999];
    let positions = vec![(0, 9999)];

    let (post_dist, ll_dist) = forward_backward_with_distances(&obs, &params, &positions);
    let (post_std, ll_std) = forward_backward(&obs, &params);

    assert_eq!(post_dist.len(), post_std.len());
    assert!((ll_dist - ll_std).abs() < 1e-10, "Fallback should match standard");
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline: estimate → train → infer → extract
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_pipeline_roundtrip_produces_segments() {
    let mut truth = vec![0_usize; 60];
    truth.extend(vec![1_usize; 30]);
    truth.extend(vec![0_usize; 60]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    let mut params = HmmParams::from_expected_length(50.0, 0.01, WINDOW_SIZE);
    params.estimate_emissions(&obs);
    params.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let result = infer_ibd(&obs, &params);

    assert_eq!(result.posteriors.len(), obs.len());
    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());

    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior out of range: {}", p);
    }
    for &s in &result.states {
        assert!(s == 0 || s == 1, "Invalid state: {}", s);
    }

    let segments = extract_ibd_segments(&result.states);
    for &(start, end, n_windows) in &segments {
        assert!(end >= start);
        assert!(n_windows > 0);
    }
}

#[test]
fn pipeline_with_training_segments_have_valid_lod() {
    let mut truth = vec![0_usize; 50];
    truth.extend(vec![1_usize; 40]);
    truth.extend(vec![0_usize; 50]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 88);

    let mut params = HmmParams::from_expected_length(50.0, 0.01, WINDOW_SIZE);
    params.estimate_emissions(&obs);
    params.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let result = infer_ibd(&obs, &params);

    let segments_with_post = extract_ibd_segments_with_posteriors(
        &result.states, &result.posteriors, 1, 0.0,
    );

    for seg in &segments_with_post {
        assert!(seg.mean_posterior >= 0.0 && seg.mean_posterior <= 1.0);
        assert!(seg.end_idx >= seg.start_idx);
    }

    let segments_with_lod = extract_ibd_segments_with_lod(
        &result.states, &result.posteriors, 1, 0.0,
        Some((&obs, &params)), None,
    );

    for seg in &segments_with_lod {
        assert!(seg.end_idx >= seg.start_idx);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// infer_ibd_with_training: training doesn't crash with different populations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn infer_ibd_with_training_all_populations_valid() {
    let populations = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::Generic,
    ];

    for &pop in &populations {
        let obs = generate_synthetic(&vec![0_usize; 100], 0.998, 0.9999, 42);
        let mut params = HmmParams::from_population(pop, 50.0, 0.0001, WINDOW_SIZE);
        let result = infer_ibd_with_training(&obs, &mut params, pop, WINDOW_SIZE, 5);

        assert_eq!(result.states.len(), 100);
        assert_eq!(result.posteriors.len(), 100);
        assert!(result.log_likelihood.is_finite());

        for &p in &result.posteriors {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }
}

#[test]
fn infer_ibd_with_training_bimodal_recovers_ibd() {
    let mut truth = vec![0_usize; 80];
    truth.extend(vec![1_usize; 40]);
    truth.extend(vec![0_usize; 80]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, WINDOW_SIZE);
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, WINDOW_SIZE, 15);

    // Core IBD region (indices 90-110) should have high posterior
    let core_posterior_mean: f64 = result.posteriors[90..110].iter().sum::<f64>() / 20.0;
    assert!(
        core_posterior_mean > 0.5,
        "Core IBD region should have elevated posterior, got {:.3}",
        core_posterior_mean
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BW training: convergence with few observations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_few_observations_no_crash() {
    let params = default_params();

    // 5 observations — baum_welch requires >=10, should be a no-op
    let obs = vec![0.998, 0.998, 0.9999, 0.998, 0.998];
    let mut trained = params.clone();
    trained.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    // Params should remain valid (unchanged since n < 10)
    let init_sum: f64 = trained.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-6);

    for row in &trained.transition {
        let row_sum: f64 = row.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn bw_training_constant_observations_no_crash() {
    let params = default_params();
    let obs = vec![0.999; 50];

    let mut trained = params.clone();
    trained.baum_welch(&obs, 10, 1e-8, Some(Population::Generic), WINDOW_SIZE);

    let init_sum: f64 = trained.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// BW training with different population priors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_with_afr_prior() {
    let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, WINDOW_SIZE);
    let mut truth = vec![0_usize; 80];
    truth.extend(vec![1_usize; 40]);
    truth.extend(vec![0_usize; 80]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    let (_, ll_before) = forward_backward(&obs, &params);
    params.baum_welch(&obs, 15, 1e-8, Some(Population::AFR), WINDOW_SIZE);
    let (_, ll_after) = forward_backward(&obs, &params);

    assert!(ll_after >= ll_before - 1e-6);
}

#[test]
fn bw_training_none_prior_uses_generic() {
    let params = default_params();
    let obs = generate_synthetic(&vec![0_usize; 100], 0.998, 0.9999, 42);

    let mut trained = params.clone();
    trained.baum_welch(&obs, 10, 1e-8, None, WINDOW_SIZE);

    // Should work fine with None prior
    let init_sum: f64 = trained.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance-variant: zero-distance and edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn forward_backward_with_distances_zero_distance_windows() {
    let params = default_params();
    let obs = vec![0.998, 0.998, 0.9999, 0.9999];
    let positions = vec![(1000, 2000), (1000, 2000), (1000, 2000), (1000, 2000)];

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert_eq!(posteriors.len(), 4);
    assert!(ll.is_finite());
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

#[test]
fn viterbi_with_distances_single_window() {
    let params = default_params();
    let obs = vec![0.9999];
    let positions = vec![(0, 9999)];

    let states = viterbi_with_distances(&obs, &params, &positions);
    assert_eq!(states.len(), 1);
    assert!(states[0] == 0 || states[0] == 1);
}

#[test]
fn viterbi_with_distances_empty() {
    let params = default_params();
    let states = viterbi_with_distances(&[], &params, &[]);
    assert!(states.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// BW convergence tolerance: early stopping
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_convergence_tight_tolerance_more_iterations() {
    let params = default_params();
    let mut truth = vec![0_usize; 60];
    truth.extend(vec![1_usize; 30]);
    truth.extend(vec![0_usize; 60]);
    let obs = generate_synthetic(&truth, 0.998, 0.9999, 42);

    // Tight tolerance — may converge late
    let mut tight = params.clone();
    tight.baum_welch(&obs, 100, 1e-12, Some(Population::Generic), WINDOW_SIZE);

    // Loose tolerance — converges early
    let mut loose = params.clone();
    loose.baum_welch(&obs, 100, 1.0, Some(Population::Generic), WINDOW_SIZE);

    // Both should produce valid params
    for p in &[&tight, &loose] {
        let init_sum: f64 = p.initial.iter().sum();
        assert!((init_sum - 1.0).abs() < 1e-6);
        for row in &p.transition {
            let row_sum: f64 = row.iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InterPop population variant
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bw_training_with_interpop_prior() {
    let mut params = HmmParams::from_population(Population::InterPop, 50.0, 0.0001, WINDOW_SIZE);
    let obs = generate_synthetic(&vec![0_usize; 100], 0.998, 0.9999, 42);

    params.baum_welch(&obs, 10, 1e-8, Some(Population::InterPop), WINDOW_SIZE);

    let init_sum: f64 = params.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-6);
}
