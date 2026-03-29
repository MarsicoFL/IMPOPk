//! Tests for HMM algorithms with genetic map integration:
//! - viterbi_with_genetic_map
//! - forward_with_genetic_map
//! - backward_with_genetic_map
//! - forward_backward_with_genetic_map
//! - recombination_aware_log_transition
//! - GeneticMap interpolation edge cases

use hprc_ibd::hmm::*;
use hprc_ibd::stats::GaussianParams;

/// Helper: create typical IBD HMM parameters
fn typical_ibd_params() -> HmmParams {
    HmmParams {
        initial: [0.95, 0.05],
        transition: [[0.99, 0.01], [0.05, 0.95]],
        emission: [
            GaussianParams::new_unchecked(0.99915, 0.001),
            GaussianParams::new_unchecked(0.9997, 0.0003),
        ],
    }
}

/// Helper: create a realistic genetic map
fn realistic_genetic_map() -> GeneticMap {
    // Simulate variable recombination rate across ~10 Mb
    GeneticMap::new(vec![
        (0, 0.0),
        (1_000_000, 1.0),    // 1 cM/Mb
        (2_000_000, 2.5),    // 1.5 cM/Mb (hotspot)
        (3_000_000, 3.0),    // 0.5 cM/Mb (coldspot)
        (5_000_000, 4.0),    // 0.5 cM/Mb
        (8_000_000, 7.0),    // 1 cM/Mb
        (10_000_000, 9.0),   // 1 cM/Mb
    ])
}

/// Helper: create window positions for n windows of given size
fn make_windows(n: usize, start: u64, size: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|i| {
            let s = start + i as u64 * size;
            (s, s + size)
        })
        .collect()
}

// ============================================================
// GeneticMap: interpolation edge cases
// ============================================================

#[test]
fn genetic_map_empty_returns_zero() {
    let gm = GeneticMap::new(vec![]);
    assert_eq!(gm.interpolate_cm(1000), 0.0);
}

#[test]
fn genetic_map_single_entry() {
    let gm = GeneticMap::new(vec![(1_000_000, 1.0)]);
    assert_eq!(gm.interpolate_cm(1_000_000), 1.0);
    // Any position returns the single entry's cM
    assert_eq!(gm.interpolate_cm(0), 1.0);
    assert_eq!(gm.interpolate_cm(5_000_000), 1.0);
}

#[test]
fn genetic_map_exact_entry_positions() {
    let gm = realistic_genetic_map();
    assert!((gm.interpolate_cm(0) - 0.0).abs() < 1e-10);
    assert!((gm.interpolate_cm(1_000_000) - 1.0).abs() < 1e-10);
    assert!((gm.interpolate_cm(2_000_000) - 2.5).abs() < 1e-10);
}

#[test]
fn genetic_map_linear_interpolation() {
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    // Midpoint should be 0.5 cM
    let cm = gm.interpolate_cm(500_000);
    assert!((cm - 0.5).abs() < 1e-10, "Midpoint should be 0.5 cM, got {}", cm);
    // Quarter point
    let cm_q = gm.interpolate_cm(250_000);
    assert!((cm_q - 0.25).abs() < 1e-10);
}

#[test]
fn genetic_map_extrapolation_before() {
    let gm = GeneticMap::new(vec![(1_000_000, 1.0), (2_000_000, 2.0)]);
    // Before first entry: extrapolate at rate 1 cM/Mb
    let cm = gm.interpolate_cm(500_000);
    assert!((cm - 0.5).abs() < 1e-10, "Extrapolated cM should be 0.5, got {}", cm);
}

#[test]
fn genetic_map_extrapolation_after() {
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    // After last entry: extrapolate at rate 1 cM/Mb
    let cm = gm.interpolate_cm(2_000_000);
    assert!((cm - 2.0).abs() < 1e-10, "Extrapolated cM should be 2.0, got {}", cm);
}

#[test]
fn genetic_map_distance_symmetric() {
    let gm = realistic_genetic_map();
    let d1 = gm.genetic_distance_cm(1_000_000, 5_000_000);
    let d2 = gm.genetic_distance_cm(5_000_000, 1_000_000);
    assert!((d1 - d2).abs() < 1e-10, "Distance should be symmetric");
}

#[test]
fn genetic_map_distance_zero_same_pos() {
    let gm = realistic_genetic_map();
    let d = gm.genetic_distance_cm(3_000_000, 3_000_000);
    assert!(d.abs() < 1e-10, "Distance to self should be 0");
}

#[test]
fn genetic_map_uniform_constructor() {
    let gm = GeneticMap::uniform(0, 10_000_000, 1.0);
    assert_eq!(gm.len(), 2);
    assert!(!gm.is_empty());
    let d = gm.genetic_distance_cm(0, 10_000_000);
    assert!((d - 10.0).abs() < 1e-10, "10 Mb at 1 cM/Mb should be 10 cM");
}

#[test]
fn genetic_map_hotspot_vs_coldspot() {
    let gm = realistic_genetic_map();
    // 1-2 Mb: hotspot at 1.5 cM/Mb
    let d_hot = gm.genetic_distance_cm(1_000_000, 2_000_000);
    // 2-3 Mb: coldspot at 0.5 cM/Mb
    let d_cold = gm.genetic_distance_cm(2_000_000, 3_000_000);
    assert!(
        d_hot > d_cold,
        "Hotspot distance ({}) should be > coldspot distance ({})",
        d_hot,
        d_cold
    );
}

// ============================================================
// viterbi_with_genetic_map
// ============================================================

#[test]
fn viterbi_gm_empty_observations() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let result = viterbi_with_genetic_map(&[], &params, &[], &gm, 10_000);
    assert!(result.is_empty());
}

#[test]
fn viterbi_gm_single_observation() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let windows = vec![(0u64, 10_000u64)];
    let result = viterbi_with_genetic_map(&[0.999], &params, &windows, &gm, 10_000);
    assert_eq!(result.len(), 1);
    assert!(result[0] < 2, "State must be 0 or 1");
}

#[test]
fn viterbi_gm_all_ibd_signal() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let n = 20;
    let obs = vec![0.9999; n]; // Very high identity → IBD
    let windows = make_windows(n, 0, 10_000);
    let result = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 10_000);
    assert_eq!(result.len(), n);
    // Most should be state 1 (IBD)
    let ibd_count = result.iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count > n / 2,
        "High identity should produce mostly IBD states: {}/{}",
        ibd_count,
        n
    );
}

#[test]
fn viterbi_gm_all_non_ibd_signal() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let n = 20;
    let obs = vec![0.998; n]; // Low identity → non-IBD
    let windows = make_windows(n, 0, 10_000);
    let result = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 10_000);
    assert_eq!(result.len(), n);
    let non_ibd_count = result.iter().filter(|&&s| s == 0).count();
    assert!(
        non_ibd_count > n / 2,
        "Low identity should produce mostly non-IBD states: {}/{}",
        non_ibd_count,
        n
    );
}

#[test]
fn viterbi_gm_valid_states() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 50;
    let obs: Vec<f64> = (0..n).map(|i| if i < 20 || i > 35 { 0.998 } else { 0.9998 }).collect();
    let windows = make_windows(n, 0, 100_000);
    let result = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 100_000);
    assert_eq!(result.len(), n);
    for &s in &result {
        assert!(s < 2, "State must be 0 or 1, got {}", s);
    }
}

#[test]
fn viterbi_gm_mismatched_positions_falls_back() {
    // When window_positions.len() != observations.len(), should fall back to regular viterbi
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let obs = vec![0.999; 10];
    let windows = vec![(0u64, 10_000u64); 5]; // Wrong length
    let result = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 10_000);
    assert_eq!(result.len(), 10, "Should still produce correct length output");
}

#[test]
fn viterbi_gm_matches_uniform_when_rate_constant() {
    // With uniform genetic map, viterbi_with_genetic_map should produce
    // same or very similar results to regular viterbi_with_distances
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 10_000_000, 1.0);
    let n = 30;
    let obs: Vec<f64> = (0..n)
        .map(|i| if (10..20).contains(&i) { 0.9998 } else { 0.998 })
        .collect();
    let windows = make_windows(n, 0, 10_000);

    let result_gm = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 10_000);
    let result_std = viterbi(&obs, &params);

    assert_eq!(result_gm.len(), result_std.len());
    // They should be very similar (might differ slightly due to distance calculation)
    let agreement = result_gm
        .iter()
        .zip(result_std.iter())
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        agreement >= n * 7 / 10,
        "Uniform genetic map should mostly agree with standard viterbi: {}/{} agree",
        agreement,
        n
    );
}

// ============================================================
// forward_with_genetic_map
// ============================================================

#[test]
fn forward_gm_empty() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let (alpha, ll) = forward_with_genetic_map(&[], &params, &[], &gm, 10_000);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn forward_gm_single() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let windows = vec![(0u64, 10_000u64)];
    let (alpha, ll) = forward_with_genetic_map(&[0.999], &params, &windows, &gm, 10_000);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

#[test]
fn forward_gm_log_likelihood_finite() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 50;
    let obs: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.9998 } else { 0.999 }).collect();
    let windows = make_windows(n, 0, 100_000);
    let (alpha, ll) = forward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);
    assert_eq!(alpha.len(), n);
    assert!(ll.is_finite(), "Log likelihood must be finite: {}", ll);
    // Note: log likelihood can be positive for narrow Gaussian emissions
    // (high density near mean). The key invariant is finiteness.
}

#[test]
fn forward_gm_mismatched_falls_back() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let obs = vec![0.999; 10];
    let windows = vec![(0u64, 10_000u64); 5]; // Wrong length
    let (alpha, ll) = forward_with_genetic_map(&obs, &params, &windows, &gm, 10_000);
    assert_eq!(alpha.len(), 10);
    assert!(ll.is_finite());
}

// ============================================================
// backward_with_genetic_map
// ============================================================

#[test]
fn backward_gm_empty() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let beta = backward_with_genetic_map(&[], &params, &[], &gm, 10_000);
    assert!(beta.is_empty());
}

#[test]
fn backward_gm_single() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let windows = vec![(0u64, 10_000u64)];
    let beta = backward_with_genetic_map(&[0.999], &params, &windows, &gm, 10_000);
    assert_eq!(beta.len(), 1);
    // Last element should be [0.0, 0.0] (log(1))
    assert!((beta[0][0] - 0.0).abs() < 1e-10);
    assert!((beta[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn backward_gm_last_entry_is_zero() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 30;
    let obs = vec![0.999; n];
    let windows = make_windows(n, 0, 100_000);
    let beta = backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);
    assert_eq!(beta.len(), n);
    assert!((beta[n - 1][0]).abs() < 1e-10);
    assert!((beta[n - 1][1]).abs() < 1e-10);
}

#[test]
fn backward_gm_all_finite() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 50;
    let obs: Vec<f64> = (0..n).map(|i| 0.998 + 0.002 * (i as f64 / n as f64)).collect();
    let windows = make_windows(n, 0, 100_000);
    let beta = backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);
    for (t, b) in beta.iter().enumerate() {
        assert!(b[0].is_finite(), "beta[{}][0] = {} is not finite", t, b[0]);
        assert!(b[1].is_finite(), "beta[{}][1] = {} is not finite", t, b[1]);
    }
}

// ============================================================
// forward_backward_with_genetic_map
// ============================================================

#[test]
fn fb_gm_empty() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let (post, ll) = forward_backward_with_genetic_map(&[], &params, &[], &gm, 10_000);
    assert!(post.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn fb_gm_posteriors_in_01() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 50;
    let obs: Vec<f64> = (0..n).map(|i| if i < 20 || i > 35 { 0.998 } else { 0.9998 }).collect();
    let windows = make_windows(n, 0, 100_000);
    let (post, ll) = forward_backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);

    assert_eq!(post.len(), n);
    assert!(ll.is_finite());
    for (t, &p) in post.iter().enumerate() {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Posterior at {} must be in [0,1]: {}",
            t,
            p
        );
    }
}

#[test]
fn fb_gm_high_identity_high_posterior() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 5_000_000, 1.0);
    let n = 30;
    let obs = vec![0.9999; n]; // Very high → IBD
    let windows = make_windows(n, 0, 100_000);
    let (post, _) = forward_backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);

    // Interior posteriors should be high
    for t in 5..25 {
        assert!(
            post[t] > 0.5,
            "High identity at position {} should have P(IBD) > 0.5: {}",
            t,
            post[t]
        );
    }
}

#[test]
fn fb_gm_low_identity_low_posterior() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 5_000_000, 1.0);
    let n = 30;
    let obs = vec![0.997; n]; // Low → non-IBD
    let windows = make_windows(n, 0, 100_000);
    let (post, _) = forward_backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);

    for t in 5..25 {
        assert!(
            post[t] < 0.5,
            "Low identity at position {} should have P(IBD) < 0.5: {}",
            t,
            post[t]
        );
    }
}

#[test]
fn fb_gm_consistency_with_viterbi() {
    // Viterbi path should be consistent with posterior > 0.5 in most positions
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let n = 40;
    let obs: Vec<f64> = (0..n)
        .map(|i| if (15..30).contains(&i) { 0.9998 } else { 0.997 })
        .collect();
    let windows = make_windows(n, 0, 100_000);

    let vit = viterbi_with_genetic_map(&obs, &params, &windows, &gm, 100_000);
    let (post, _) = forward_backward_with_genetic_map(&obs, &params, &windows, &gm, 100_000);

    let agreement = vit
        .iter()
        .zip(post.iter())
        .filter(|(&v, &p)| (v == 1) == (p > 0.5))
        .count();
    assert!(
        agreement >= n * 7 / 10,
        "Viterbi and posteriors should mostly agree: {}/{}",
        agreement,
        n
    );
}

// ============================================================
// recombination_aware_log_transition: properties
// ============================================================

#[test]
fn recombi_transition_rows_sum_to_one() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();

    for &(p1, p2) in &[(1_000_000u64, 1_100_000u64), (2_000_000, 3_000_000), (0, 5_000_000)] {
        let log_trans = recombination_aware_log_transition(&params, p1, p2, &gm, 10_000);
        for s in 0..2 {
            let sum = log_trans[s][0].exp() + log_trans[s][1].exp();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "Transition row {} should sum to ~1: {} (positions {}-{})",
                s,
                sum,
                p1,
                p2
            );
        }
    }
}

#[test]
fn recombi_transition_same_position() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let log_trans = recombination_aware_log_transition(&params, 1_000_000, 1_000_000, &gm, 10_000);
    // Same position → should use base transition
    let base_00 = params.transition[0][0].ln();
    let base_01 = params.transition[0][1].ln();
    assert!(
        (log_trans[0][0] - base_00).abs() < 1e-10,
        "Same position should use base transition"
    );
    assert!(
        (log_trans[0][1] - base_01).abs() < 1e-10,
        "Same position should use base transition"
    );
}

#[test]
fn recombi_transition_zero_window_size() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let log_trans = recombination_aware_log_transition(&params, 1_000_000, 2_000_000, &gm, 0);
    // Zero window size → should use base transition
    let base_00 = params.transition[0][0].ln();
    assert!(
        (log_trans[0][0] - base_00).abs() < 1e-10,
        "Zero window size should use base transition"
    );
}

#[test]
fn recombi_transition_larger_distance_more_switching() {
    let params = typical_ibd_params();
    let gm = GeneticMap::uniform(0, 100_000_000, 1.0);

    let log_trans_short = recombination_aware_log_transition(&params, 0, 100_000, &gm, 10_000);
    let log_trans_long = recombination_aware_log_transition(&params, 0, 10_000_000, &gm, 10_000);

    // Longer distance → higher switching probability → log_trans[0][1] should be less negative
    assert!(
        log_trans_long[0][1] > log_trans_short[0][1],
        "Longer distance should have higher switching prob: short={}, long={}",
        log_trans_short[0][1],
        log_trans_long[0][1]
    );
}

#[test]
fn recombi_transition_all_finite() {
    let params = typical_ibd_params();
    let gm = realistic_genetic_map();
    let log_trans = recombination_aware_log_transition(&params, 0, 5_000_000, &gm, 10_000);
    for s in 0..2 {
        for t in 0..2 {
            assert!(
                log_trans[s][t].is_finite(),
                "log_trans[{}][{}] = {} is not finite",
                s,
                t,
                log_trans[s][t]
            );
        }
    }
}
