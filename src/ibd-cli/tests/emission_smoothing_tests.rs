/// Tests for IBD emission smoothing (smooth_log_emissions) and
/// the distance/genetic-map _from_log_emit variants.

use impopk_ibd::hmm::{
    precompute_log_emissions, smooth_log_emissions,
    viterbi, viterbi_from_log_emit,
    viterbi_with_distances, viterbi_with_distances_from_log_emit,
    forward_backward, forward_backward_from_log_emit,
    forward_backward_with_distances, forward_backward_with_distances_from_log_emit,
    forward_with_distances_from_log_emit, backward_with_distances_from_log_emit,
    forward_with_genetic_map_from_log_emit, backward_with_genetic_map_from_log_emit,
    viterbi_with_genetic_map_from_log_emit, forward_backward_with_genetic_map_from_log_emit,
    HmmParams, Population, GeneticMap,
};

fn test_params() -> HmmParams {
    HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000)
}

fn test_observations() -> Vec<f64> {
    vec![
        0.998, 0.997, 0.9985, 0.997, 0.998,   // non-IBD
        0.9998, 0.9999, 0.9997, 0.9998, 0.9999, // IBD
        0.997, 0.998, 0.9975, 0.997, 0.998,     // non-IBD
    ]
}

// ===== smooth_log_emissions tests =====

#[test]
fn test_smooth_log_emissions_context_zero_is_identity() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 0);
    assert_eq!(log_emit.len(), smoothed.len());
    for (orig, smooth) in log_emit.iter().zip(smoothed.iter()) {
        assert!((orig[0] - smooth[0]).abs() < 1e-12);
        assert!((orig[1] - smooth[1]).abs() < 1e-12);
    }
}

#[test]
fn test_smooth_log_emissions_empty() {
    let smoothed = smooth_log_emissions(&[], 2);
    assert!(smoothed.is_empty());
}

#[test]
fn test_smooth_log_emissions_single_element() {
    let log_emit = vec![[-1.0, -2.0]];
    let smoothed = smooth_log_emissions(&log_emit, 3);
    assert_eq!(smoothed.len(), 1);
    assert!((smoothed[0][0] - (-1.0)).abs() < 1e-12);
    assert!((smoothed[0][1] - (-2.0)).abs() < 1e-12);
}

#[test]
fn test_smooth_log_emissions_context_1_averaging() {
    // 3 elements, context=1: each element averages with neighbors
    let log_emit = vec![
        [-1.0, -2.0],
        [-3.0, -4.0],
        [-5.0, -6.0],
    ];
    let smoothed = smooth_log_emissions(&log_emit, 1);

    // t=0: average of [0,1] = [-2.0, -3.0]
    assert!((smoothed[0][0] - (-2.0)).abs() < 1e-12);
    assert!((smoothed[0][1] - (-3.0)).abs() < 1e-12);

    // t=1: average of [0,1,2] = [-3.0, -4.0]
    assert!((smoothed[1][0] - (-3.0)).abs() < 1e-12);
    assert!((smoothed[1][1] - (-4.0)).abs() < 1e-12);

    // t=2: average of [1,2] = [-4.0, -5.0]
    assert!((smoothed[2][0] - (-4.0)).abs() < 1e-12);
    assert!((smoothed[2][1] - (-5.0)).abs() < 1e-12);
}

#[test]
fn test_smooth_log_emissions_context_larger_than_n() {
    // context=10 but only 3 elements: all elements become the same average
    let log_emit = vec![
        [-1.0, -2.0],
        [-3.0, -4.0],
        [-5.0, -6.0],
    ];
    let smoothed = smooth_log_emissions(&log_emit, 10);

    let expected_s0 = (-1.0 + -3.0 + -5.0) / 3.0;
    let expected_s1 = (-2.0 + -4.0 + -6.0) / 3.0;

    for t in 0..3 {
        assert!((smoothed[t][0] - expected_s0).abs() < 1e-12);
        assert!((smoothed[t][1] - expected_s1).abs() < 1e-12);
    }
}

#[test]
fn test_smooth_log_emissions_preserves_length() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);

    for ctx in [1, 2, 3, 5] {
        let smoothed = smooth_log_emissions(&log_emit, ctx);
        assert_eq!(smoothed.len(), log_emit.len(), "context={}", ctx);
    }
}

#[test]
fn test_smooth_log_emissions_reduces_variance() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 2);

    // Smoothing should reduce the variance of emissions
    let orig_var: f64 = {
        let mean = log_emit.iter().map(|e| e[1]).sum::<f64>() / log_emit.len() as f64;
        log_emit.iter().map(|e| (e[1] - mean).powi(2)).sum::<f64>() / log_emit.len() as f64
    };
    let smooth_var: f64 = {
        let mean = smoothed.iter().map(|e| e[1]).sum::<f64>() / smoothed.len() as f64;
        smoothed.iter().map(|e| (e[1] - mean).powi(2)).sum::<f64>() / smoothed.len() as f64
    };

    assert!(smooth_var < orig_var, "Smoothed variance {} should be less than original {}", smooth_var, orig_var);
}

// ===== from_log_emit equivalence tests =====

#[test]
fn test_viterbi_from_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let states_std = viterbi(&obs, &params);
    let states_log = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states_std, states_log);
}

#[test]
fn test_fb_from_log_emit_matches_standard() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let (post_std, ll_std) = forward_backward(&obs, &params);
    let (post_log, ll_log) = forward_backward_from_log_emit(&log_emit, &params);
    assert!((ll_std - ll_log).abs() < 1e-6, "LL mismatch: {} vs {}", ll_std, ll_log);
    for (a, b) in post_std.iter().zip(post_log.iter()) {
        assert!((a - b).abs() < 1e-10, "Posterior mismatch: {} vs {}", a, b);
    }
}

// ===== distance-aware from_log_emit tests =====

#[test]
fn test_viterbi_with_distances_from_log_emit_matches() {
    let params = test_params();
    let obs = test_observations();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let states_std = viterbi_with_distances(&obs, &params, &positions);
    let states_log = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(states_std, states_log);
}

#[test]
fn test_fb_with_distances_from_log_emit_matches() {
    let params = test_params();
    let obs = test_observations();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let (post_std, ll_std) = forward_backward_with_distances(&obs, &params, &positions);
    let (post_log, ll_log) = forward_backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert!((ll_std - ll_log).abs() < 1e-6, "LL mismatch: {} vs {}", ll_std, ll_log);
    for (a, b) in post_std.iter().zip(post_log.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_forward_with_distances_from_log_emit_empty() {
    let params = test_params();
    let (alpha, ll) = forward_with_distances_from_log_emit(&[], &params, &[]);
    assert!(alpha.is_empty());
    assert!((ll - 0.0).abs() < 1e-12);
}

#[test]
fn test_backward_with_distances_from_log_emit_empty() {
    let params = test_params();
    let beta = backward_with_distances_from_log_emit(&[], &params, &[]);
    assert!(beta.is_empty());
}

#[test]
fn test_viterbi_with_distances_from_log_emit_position_mismatch_fallback() {
    // When positions length doesn't match, should fall back to standard viterbi_from_log_emit
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let positions = vec![(0, 4999)]; // length mismatch

    let states_fallback = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    let states_standard = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states_fallback, states_standard);
}

#[test]
fn test_fb_with_distances_from_log_emit_position_mismatch_fallback() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let positions = vec![(0, 4999)]; // length mismatch

    let (post_fallback, _) = forward_backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    let (post_standard, _) = forward_backward_from_log_emit(&log_emit, &params);
    for (a, b) in post_fallback.iter().zip(post_standard.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

// ===== genetic-map from_log_emit tests =====

fn test_genetic_map() -> GeneticMap {
    // Simple linear genetic map: (position_bp, position_cM)
    GeneticMap::new(vec![
        (0, 0.0), (10000, 0.01), (20000, 0.02), (30000, 0.03),
        (40000, 0.04), (50000, 0.05), (60000, 0.06), (70000, 0.07),
    ])
}

#[test]
fn test_viterbi_with_genetic_map_from_log_emit_empty() {
    let params = test_params();
    let gmap = test_genetic_map();
    let states = viterbi_with_genetic_map_from_log_emit(&[], &params, &[], &gmap, 5000);
    assert!(states.is_empty());
}

#[test]
fn test_fb_with_genetic_map_from_log_emit_empty() {
    let params = test_params();
    let gmap = test_genetic_map();
    let (post, ll) = forward_backward_with_genetic_map_from_log_emit(&[], &params, &[], &gmap, 5000);
    assert!(post.is_empty());
    assert!((ll - 0.0).abs() < 1e-12);
}

#[test]
fn test_forward_with_genetic_map_from_log_emit_empty() {
    let params = test_params();
    let gmap = test_genetic_map();
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&[], &params, &[], &gmap, 5000);
    assert!(alpha.is_empty());
    assert!((ll - 0.0).abs() < 1e-12);
}

#[test]
fn test_backward_with_genetic_map_from_log_emit_empty() {
    let params = test_params();
    let gmap = test_genetic_map();
    let beta = backward_with_genetic_map_from_log_emit(&[], &params, &[], &gmap, 5000);
    assert!(beta.is_empty());
}

#[test]
fn test_viterbi_with_genetic_map_from_log_emit_mismatch_fallback() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let gmap = test_genetic_map();
    let positions = vec![(0, 4999)]; // mismatch

    let states_fallback = viterbi_with_genetic_map_from_log_emit(&log_emit, &params, &positions, &gmap, 5000);
    let states_standard = viterbi_from_log_emit(&log_emit, &params);
    assert_eq!(states_fallback, states_standard);
}

#[test]
fn test_fb_with_genetic_map_from_log_emit_mismatch_fallback() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let gmap = test_genetic_map();
    let positions = vec![(0, 4999)]; // mismatch

    let (post_fallback, _) = forward_backward_with_genetic_map_from_log_emit(
        &log_emit, &params, &positions, &gmap, 5000);
    let (post_standard, _) = forward_backward_from_log_emit(&log_emit, &params);
    for (a, b) in post_fallback.iter().zip(post_standard.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

// ===== Smoothed emissions change Viterbi output =====

#[test]
fn test_smoothed_emissions_can_differ_from_unsmoothed() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 2);

    // At least one emission value should differ after smoothing
    let any_diff = log_emit.iter().zip(smoothed.iter())
        .any(|(orig, smooth)| (orig[0] - smooth[0]).abs() > 1e-12 || (orig[1] - smooth[1]).abs() > 1e-12);
    assert!(any_diff, "Smoothing should change at least some emissions");
}

#[test]
fn test_smoothed_viterbi_produces_valid_states() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 2);
    let states = viterbi_from_log_emit(&smoothed, &params);
    assert_eq!(states.len(), obs.len());
    for &s in &states {
        assert!(s == 0 || s == 1, "State must be 0 or 1, got {}", s);
    }
}

#[test]
fn test_smoothed_fb_posteriors_valid() {
    let params = test_params();
    let obs = test_observations();
    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 2);
    let (posteriors, ll) = forward_backward_from_log_emit(&smoothed, &params);
    assert_eq!(posteriors.len(), obs.len());
    assert!(ll.is_finite(), "Log-likelihood should be finite");
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior {} out of range", p);
    }
}

// ===== Distance-aware with smoothed emissions =====

#[test]
fn test_smoothed_viterbi_with_distances() {
    let params = test_params();
    let obs = test_observations();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 1);
    let states = viterbi_with_distances_from_log_emit(&smoothed, &params, &positions);
    assert_eq!(states.len(), obs.len());
}

#[test]
fn test_smoothed_fb_with_distances() {
    let params = test_params();
    let obs = test_observations();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 1);
    let (posteriors, ll) = forward_backward_with_distances_from_log_emit(
        &smoothed, &params, &positions);
    assert_eq!(posteriors.len(), obs.len());
    assert!(ll.is_finite());
}

// ===== Genetic-map with smoothed emissions =====

#[test]
fn test_smoothed_viterbi_with_genetic_map() {
    let params = test_params();
    let obs = test_observations();
    let gmap = test_genetic_map();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 1);
    let states = viterbi_with_genetic_map_from_log_emit(
        &smoothed, &params, &positions, &gmap, 5000);
    assert_eq!(states.len(), obs.len());
}

#[test]
fn test_smoothed_fb_with_genetic_map() {
    let params = test_params();
    let obs = test_observations();
    let gmap = test_genetic_map();
    let positions: Vec<(u64, u64)> = (0..obs.len())
        .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
        .collect();

    let log_emit = precompute_log_emissions(&obs, &params);
    let smoothed = smooth_log_emissions(&log_emit, 1);
    let (posteriors, ll) = forward_backward_with_genetic_map_from_log_emit(
        &smoothed, &params, &positions, &gmap, 5000);
    assert_eq!(posteriors.len(), obs.len());
    assert!(ll.is_finite());
}
