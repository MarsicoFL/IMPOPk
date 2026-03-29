//! Edge-case tests for Baum-Welch distance/genetic-map variants and
//! distance_dependent_log_transition boundary branches.
//!
//! Targets genuinely untested paths:
//! - distance_dependent_log_transition with p_enter/p_exit at exact 0.0 and 1.0 boundaries
//! - baum_welch_with_distances mismatch fallback produces same result as standard baum_welch
//! - baum_welch_with_genetic_map mismatch fallback produces same result as standard baum_welch
//! - baum_welch_with_genetic_map non-finite log-likelihood early exit
//! - recombination_aware_log_transition with zero-cM fallback (constant genetic map)
//! - distance_dependent_log_transition large scale factor and tiny scale factor

use hprc_ibd::hmm::{
    distance_dependent_log_transition, recombination_aware_log_transition, GeneticMap, HmmParams,
};

// ── distance_dependent_log_transition: p_enter_base boundary branches ───

#[test]
fn dd_transition_p_enter_exactly_one() {
    // p_enter_base = 1.0 → rate_enter = INFINITY → p_enter clamped to 1-1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.0, 1.0], [0.02, 0.98]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    let p_enter = trans[0][1].exp();
    assert!(
        p_enter > 1.0 - 1e-5,
        "p_enter should be near 1.0 after clamping, got {}",
        p_enter
    );
    let row_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!(
        (row_sum - 1.0).abs() < 1e-6,
        "Row 0 should sum to 1.0, got {}",
        row_sum
    );
}

#[test]
fn dd_transition_p_enter_exactly_zero() {
    // p_enter_base = 0.0 → rate_enter = 0.0 → p_enter clamped to 1e-10
    let params = HmmParams {
        initial: [1.0, 0.0],
        transition: [[1.0, 0.0], [0.02, 0.98]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    let p_enter = trans[0][1].exp();
    assert!(
        p_enter < 1e-5,
        "p_enter should be near 0 after clamping, got {}",
        p_enter
    );
    assert!(p_enter > 0.0, "p_enter should be > 0 after clamping");
    let row_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!((row_sum - 1.0).abs() < 1e-6);
}

#[test]
fn dd_transition_p_exit_exactly_one() {
    // p_exit_base = 1.0 → rate_exit = INFINITY → p_exit clamped to 1-1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.999, 0.001], [1.0, 0.0]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    let p_exit = trans[1][0].exp();
    assert!(
        p_exit > 1.0 - 1e-5,
        "p_exit should be near 1.0, got {}",
        p_exit
    );
    let row_sum = trans[1][0].exp() + trans[1][1].exp();
    assert!((row_sum - 1.0).abs() < 1e-6);
}

#[test]
fn dd_transition_p_exit_exactly_zero() {
    // p_exit_base = 0.0 → rate_exit = 0.0 → p_exit clamped to 1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.999, 0.001], [0.0, 1.0]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    let p_exit = trans[1][0].exp();
    assert!(
        p_exit < 1e-5,
        "p_exit should be near 0, got {}",
        p_exit
    );
    assert!(p_exit > 0.0);
    let row_sum = trans[1][0].exp() + trans[1][1].exp();
    assert!((row_sum - 1.0).abs() < 1e-6);
}

#[test]
fn dd_transition_both_rates_at_one() {
    // Both p_enter = 1.0 and p_exit = 1.0 → both INFINITY rates, clamped
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.0, 1.0], [1.0, 0.0]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    // Both off-diagonal near 1.0
    assert!(trans[0][1].exp() > 1.0 - 1e-5);
    assert!(trans[1][0].exp() > 1.0 - 1e-5);
}

#[test]
fn dd_transition_both_rates_at_zero() {
    // Both p_enter = 0.0 and p_exit = 0.0 → both zero rates, clamped to 1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[1.0, 0.0], [0.0, 1.0]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    // Both off-diagonal near 0
    assert!(trans[0][1].exp() < 1e-5);
    assert!(trans[1][0].exp() < 1e-5);
}

#[test]
fn dd_transition_very_large_scale() {
    // distance_bp >> window_size → scale is very large → rates scale up
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 10_000_000, 5000);
    // With scale = 2000, even small base rates saturate
    let p_enter = trans[0][1].exp();
    let p_exit = trans[1][0].exp();
    // Both should be clamped or near their limits
    assert!(p_enter > 0.0 && p_enter <= 1.0);
    assert!(p_exit > 0.0 && p_exit <= 1.0);
    // Rows sum to 1
    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn dd_transition_tiny_scale() {
    // distance_bp < window_size → scale < 1.0 → rates scale down
    let params = HmmParams::from_expected_length(50.0, 0.001, 10_000);
    let base_trans = distance_dependent_log_transition(&params, 10_000, 10_000);
    let small_trans = distance_dependent_log_transition(&params, 100, 10_000);
    // With scale = 0.01, transition probabilities should be much smaller
    let base_p_enter = base_trans[0][1].exp();
    let small_p_enter = small_trans[0][1].exp();
    assert!(
        small_p_enter < base_p_enter,
        "Smaller scale should give smaller p_enter: {} vs {}",
        small_p_enter,
        base_p_enter
    );
}

// ── baum_welch_with_distances: mismatch fallback verifies params change ──

#[test]
fn bw_with_distances_mismatch_actually_runs_standard_bw() {
    // When positions.len() != obs.len(), it should fall back to standard baum_welch.
    // Verify that params actually change (emissions are updated).
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_nonibd_mean = params.emission[0].mean;
    let original_ibd_mean = params.emission[1].mean;

    // Create clear IBD signal: first 10 windows low identity, next 10 high
    let mut obs = vec![0.997; 10];
    obs.extend(vec![0.9999; 10]);
    // Mismatched positions (15 != 20)
    let positions: Vec<(u64, u64)> = (0..15).map(|i| (i * 5000, (i + 1) * 5000)).collect();

    params.baum_welch_with_distances(&obs, &positions, 20, 1e-8, None, 5000);

    // Standard BW should have updated at least one emission parameter
    let changed = (params.emission[0].mean - original_nonibd_mean).abs() > 1e-10
        || (params.emission[1].mean - original_ibd_mean).abs() > 1e-10;
    assert!(
        changed,
        "Mismatch fallback should still run standard BW and update emissions"
    );
}

#[test]
fn bw_with_distances_mismatch_same_as_standard() {
    // Verify that the mismatch fallback produces the same result as calling baum_welch directly.
    let base_params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();

    // Run via standard BW
    let mut params_standard = base_params.clone();
    params_standard.baum_welch(&obs, 10, 1e-8, None, 5000);

    // Run via mismatch fallback (7 positions != 20 observations)
    let mut params_fallback = base_params.clone();
    let positions: Vec<(u64, u64)> = (0..7).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    params_fallback.baum_welch_with_distances(&obs, &positions, 10, 1e-8, None, 5000);

    // Both should produce identical results
    assert!(
        (params_standard.emission[0].mean - params_fallback.emission[0].mean).abs() < 1e-12,
        "Non-IBD mean: standard={}, fallback={}",
        params_standard.emission[0].mean,
        params_fallback.emission[0].mean
    );
    assert!(
        (params_standard.emission[1].mean - params_fallback.emission[1].mean).abs() < 1e-12,
        "IBD mean: standard={}, fallback={}",
        params_standard.emission[1].mean,
        params_fallback.emission[1].mean
    );
}

// ── baum_welch_with_genetic_map: mismatch fallback same as standard ──

#[test]
fn bw_with_genetic_map_mismatch_same_as_standard() {
    let base_params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();

    let mut params_standard = base_params.clone();
    params_standard.baum_welch(&obs, 10, 1e-8, None, 5000);

    let mut params_fallback = base_params.clone();
    let positions: Vec<(u64, u64)> = (0..7).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (200000, 0.2)]);
    params_fallback.baum_welch_with_genetic_map(&obs, &positions, &gmap, 10, 1e-8, None, 5000);

    assert!(
        (params_standard.emission[0].mean - params_fallback.emission[0].mean).abs() < 1e-12,
        "Non-IBD mean: standard={}, fallback={}",
        params_standard.emission[0].mean,
        params_fallback.emission[0].mean
    );
    assert!(
        (params_standard.emission[1].mean - params_fallback.emission[1].mean).abs() < 1e-12,
    );
}

// ── baum_welch_with_genetic_map: non-finite log-likelihood early exit ──

#[test]
fn bw_with_genetic_map_nonfinite_ll_breaks_early() {
    // Construct degenerate params that produce non-finite log-likelihood.
    // Setting extreme emission params so forward pass produces -inf or NaN.
    let mut params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.99, 0.01], [0.02, 0.98]],
        emission: [
            // Both states have mean=0.0, std=1e-20 → log-pdf at any realistic obs → -inf
            hprc_ibd::stats::GaussianParams::new_unchecked(0.0, 1e-20),
            hprc_ibd::stats::GaussianParams::new_unchecked(0.0, 1e-20),
        ],
    };
    let obs: Vec<f64> = (0..20).map(|i| (i as f64) * 0.01 + 0.95).collect();
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (100000, 0.1)]);

    // Should not panic — the non-finite LL guard should break the loop
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 50, 1e-8, None, 5000);
    // Params should be finite after the call (even if not well-trained)
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ── recombination_aware_log_transition: zero-cM fallback ──

#[test]
fn recomb_transition_constant_genetic_map_uses_fallback() {
    // When the genetic map has constant cM (flat map), nominal_cm = 0.0 for
    // adjacent windows → triggers the else branch: window_size / 1_000_000
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    // Flat genetic map: all positions map to 0.0 cM
    let flat_map = GeneticMap::new(vec![(0, 0.0), (1_000_000, 0.0)]);

    let trans = recombination_aware_log_transition(&params, 100_000, 110_000, &flat_map, 5000);

    // Should produce valid log-probabilities (not NaN or -inf on diagonal)
    for row in &trans {
        for &val in row {
            assert!(val.is_finite(), "Transition value should be finite, got {}", val);
        }
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6, "Row should sum to 1.0, got {}", sum);
    }
}

#[test]
fn recomb_transition_flat_map_zero_cm_per_window_fallback() {
    // Verify the cm_per_window == 0 → scale = 1.0 fallback path
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    // Map where the nominal window [100000, 105000] maps to identical cM
    let map = GeneticMap::new(vec![
        (0, 0.0),
        (100_000, 1.0),  // all before 100k maps to 0→1cM
        (200_000, 1.0),  // 100k→200k is FLAT (0 cM distance)
        (300_000, 2.0),
    ]);

    // pos1=100000, window_size=5000 → mid1=100000, mid2=105000
    // nominal_cm = genetic_distance(100000, 105000) ≈ 0 since both in flat region
    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &map, 5000);

    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// ── baum_welch_with_distances: proper convergence with population prior ──

#[test]
fn bw_with_distances_population_prior_runs() {
    use hprc_ibd::hmm::Population;
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 10_000);
    let mut obs = vec![0.998; 30];
    for item in obs.iter_mut().take(20).skip(10) {
        *item = 0.9999;
    }
    let positions: Vec<(u64, u64)> = (0..30)
        .map(|i| (i * 10_000, (i + 1) * 10_000 - 1))
        .collect();
    params.baum_welch_with_distances(&obs, &positions, 5, 1e-6, Some(Population::AFR), 10_000);
    // Should not panic, emissions should be updated
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ── baum_welch_with_genetic_map: population prior ──

#[test]
fn bw_with_genetic_map_population_prior_runs() {
    use hprc_ibd::hmm::Population;
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 10_000);
    let mut obs = vec![0.998; 30];
    for item in obs.iter_mut().take(20).skip(10) {
        *item = 0.9999;
    }
    let positions: Vec<(u64, u64)> = (0..30)
        .map(|i| (i * 10_000, (i + 1) * 10_000 - 1))
        .collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (150_000, 0.15), (300_000, 0.30)]);
    params.baum_welch_with_genetic_map(
        &obs,
        &positions,
        &gmap,
        5,
        1e-6,
        Some(Population::EUR),
        10_000,
    );
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ── distance_dependent_log_transition: negative p_enter (out of valid range) ──

#[test]
fn dd_transition_negative_p_enter_treated_as_zero() {
    // If someone constructs params with negative transition[0][1], it should
    // be treated as <= 0.0 → rate = 0.0
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[-0.1, -0.1], [0.02, 0.98]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    // Should not panic or produce NaN
    for row in &trans {
        for &val in row {
            assert!(val.is_finite() || val == f64::NEG_INFINITY,
                "Expected finite or -inf, got {}", val);
        }
    }
}

#[test]
fn dd_transition_p_enter_above_one_treated_as_one() {
    // transition[0][1] > 1.0 → should be treated as >= 1.0 → INFINITY rate
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [[-0.5, 1.5], [0.02, 0.98]],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let trans = distance_dependent_log_transition(&params, 50_000, 10_000);
    let p_enter = trans[0][1].exp();
    assert!(
        p_enter > 1.0 - 1e-5,
        "p_enter > 1.0 should produce near-max transition, got {}",
        p_enter
    );
}

// ── baum_welch_with_distances: exactly 10 observations (minimum) ──

#[test]
fn bw_with_distances_exactly_10_runs() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9999; 10]; // Exactly 10 — should run BW, not early-return
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    params.baum_welch_with_distances(&obs, &positions, 10, 1e-8, None, 5000);
    // At minimum, BW should run — check it completed without panic
    // Emissions may or may not change with homogeneous input, but function ran
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

#[test]
fn bw_with_distances_9_skips() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original = params.emission.clone();
    let obs = vec![0.9999; 9]; // 9 < 10 → early return
    let positions: Vec<(u64, u64)> = (0..9).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    params.baum_welch_with_distances(&obs, &positions, 10, 1e-8, None, 5000);
    assert_eq!(params.emission[0].mean, original[0].mean);
    assert_eq!(params.emission[1].mean, original[1].mean);
}

// ── baum_welch_with_genetic_map: exactly 10 observations (minimum) ──

#[test]
fn bw_with_genetic_map_exactly_10_runs() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9999; 10];
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (50000, 0.05)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 10, 1e-8, None, 5000);
    assert!(params.emission[0].mean.is_finite());
}

#[test]
fn bw_with_genetic_map_9_skips() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original = params.emission.clone();
    let obs = vec![0.9999; 9];
    let positions: Vec<(u64, u64)> = (0..9).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (50000, 0.05)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 10, 1e-8, None, 5000);
    assert_eq!(params.emission[0].mean, original[0].mean);
}

// ── baum_welch_with_distances: convergence in 1 iteration ──

#[test]
fn bw_with_distances_max_iter_one() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    params.baum_welch_with_distances(&obs, &positions, 1, 1e-8, None, 5000);
    // Should run exactly 1 iteration without error
    assert!(params.emission[0].mean.is_finite());
}

// ── baum_welch_with_genetic_map: convergence in 1 iteration ──

#[test]
fn bw_with_genetic_map_max_iter_one() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (100000, 0.1)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 1, 1e-8, None, 5000);
    assert!(params.emission[0].mean.is_finite());
}

// ── baum_welch_with_genetic_map: max_iter = 0 → no iterations ──

#[test]
fn bw_with_genetic_map_zero_iterations() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original = params.emission.clone();
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (100000, 0.1)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 0, 1e-8, None, 5000);
    // Zero iterations → params unchanged
    assert_eq!(params.emission[0].mean, original[0].mean);
    assert_eq!(params.emission[1].mean, original[1].mean);
}

// ── baum_welch_with_distances: max_iter = 0 → no iterations ──

#[test]
fn bw_with_distances_zero_iterations() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original = params.emission.clone();
    let obs: Vec<f64> = (0..20).map(|i| if i < 10 { 0.997 } else { 0.9999 }).collect();
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000)).collect();
    params.baum_welch_with_distances(&obs, &positions, 0, 1e-8, None, 5000);
    assert_eq!(params.emission[0].mean, original[0].mean);
    assert_eq!(params.emission[1].mean, original[1].mean);
}
