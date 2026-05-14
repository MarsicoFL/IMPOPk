//! Edge-case and boundary tests for `blend_log_emissions_adaptive`.
//!
//! The algo_dev inline tests cover happy-path (strong/weak signal, uniform
//! margins, empty input, weight clamping). These tests target edge cases:
//! NaN/Inf handling, single-population rows, all-zero margins fallback,
//! base_weight extremes, kappa clamping bounds, mismatched lengths, and
//! NEG_INFINITY emissions.

use impopk_ancestry_cli::{blend_log_emissions, blend_log_emissions_adaptive};

// ============================================================================
// NaN / Inf safety
// ============================================================================

#[test]
fn adaptive_nan_in_profile_does_not_propagate() {
    let standard = vec![vec![-1.0, -2.0], vec![-1.5, -1.5]];
    let pairwise = vec![vec![f64::NAN, -2.0], vec![-0.5, -1.5]];
    let result = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
    // Window 0 pop 0: std is finite, profile is NaN → should use std
    assert!(result[0][0].is_finite(), "NaN profile should fall back to standard");
    // Window 0 pop 1: both finite → blended
    assert!(result[0][1].is_finite());
    // Window 1: both finite
    assert!(result[1][0].is_finite());
    assert!(result[1][1].is_finite());
}

#[test]
fn adaptive_inf_in_standard_uses_profile() {
    let standard = vec![vec![f64::NEG_INFINITY, -2.0]];
    let pairwise = vec![vec![-0.5, -1.5]];
    let result = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
    // std is -inf, profile finite → should use profile
    assert!(result[0][0].is_finite(), "Inf standard should fall back to profile");
    assert!(result[0][1].is_finite());
}

#[test]
fn adaptive_both_inf_returns_neg_infinity() {
    let standard = vec![vec![f64::NEG_INFINITY, -1.0]];
    let pairwise = vec![vec![f64::NEG_INFINITY, -2.0]];
    let result = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
    assert!(result[0][0] == f64::NEG_INFINITY, "Both -inf should yield -inf");
    assert!(result[0][1].is_finite());
}

// ============================================================================
// Single-population rows (len < 2 → margin = 0)
// ============================================================================

#[test]
fn adaptive_single_pop_rows_margin_zero() {
    // Single-element rows: margin computation returns 0
    let standard = vec![vec![-1.0], vec![-2.0], vec![-1.5]];
    let pairwise = vec![vec![-0.5], vec![-0.8], vec![-1.0]];
    // All margins are 0 → sorted_margins is empty → falls back to static blend
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.4);
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.4);
    for t in 0..3 {
        assert!(
            (adaptive[t][0] - static_blend[t][0]).abs() < 1e-12,
            "Single-pop row should fallback to static: t={t}"
        );
    }
}

// ============================================================================
// All-zero margins → fallback to static blend
// ============================================================================

#[test]
fn adaptive_all_zero_margins_equals_static() {
    // When all profile rows have equal values → margin = 0 for all → fallback
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -0.5],
    ];
    let pairwise = vec![
        vec![-1.0, -1.0], // margin = 0
        vec![-2.0, -2.0], // margin = 0
    ];
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.3);
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.3);
    for t in 0..2 {
        for s in 0..2 {
            assert!(
                (adaptive[t][s] - static_blend[t][s]).abs() < 1e-12,
                "Zero margins should equal static at [{t}][{s}]"
            );
        }
    }
}

// ============================================================================
// base_weight = 0.0 → standard only (regardless of adaptive)
// ============================================================================

#[test]
fn adaptive_base_weight_zero_returns_standard() {
    let standard = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-0.5, -1.5, -2.5],
    ];
    let pairwise = vec![
        vec![-5.0, -0.1, -3.0],
        vec![-0.1, -5.0, -3.0],
    ];
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.0);
    for t in 0..2 {
        for s in 0..3 {
            // base_weight=0 → w = 0 * kappa → clamped at 0 → w_std = 1.0
            assert!(
                (adaptive[t][s] - standard[t][s]).abs() < 1e-12,
                "Weight 0 should return standard at [{t}][{s}]"
            );
        }
    }
}

// ============================================================================
// base_weight = 1.0 → almost all pairwise but clamped at 0.95
// ============================================================================

#[test]
fn adaptive_base_weight_one_clamped_at_095() {
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let pairwise = vec![
        vec![-0.5, -3.0], // margin = 2.5
        vec![-0.5, -3.0], // margin = 2.5
    ];
    // Uniform margins → kappa = 1.0 → w = 1.0 * 1.0 → clamped at 0.95
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 1.0);
    // Expected: 0.05 * std + 0.95 * pairwise
    for t in 0..2 {
        for s in 0..2 {
            let expected = 0.05 * standard[t][s] + 0.95 * pairwise[t][s];
            assert!(
                (adaptive[t][s] - expected).abs() < 1e-10,
                "Weight 1.0 should clamp at 0.95 at [{t}][{s}]: got {}, expected {}",
                adaptive[t][s], expected
            );
        }
    }
}

// ============================================================================
// Kappa clamping boundaries (0.1 floor, 2.5 ceiling)
// ============================================================================

#[test]
fn adaptive_kappa_lower_bound_010() {
    // Construct scenario: one window with near-zero margin, others with large margin
    // so median_margin is large and the weak window's kappa = margin/median → < 0.1 → clamped to 0.1
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let pairwise = vec![
        vec![-0.5, -5.0],   // margin = 4.5 (very strong)
        vec![-0.5, -5.0],   // margin = 4.5
        vec![-1.0, -1.001], // margin = 0.001 (very weak → kappa = 0.001/4.5 ≈ 0.0002 → clamped to 0.1)
    ];
    let base_weight = 0.5;
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, base_weight);

    // Window 2 should use w = base_weight * 0.1 = 0.05
    let w = base_weight * 0.1;
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[2][s] + w * pairwise[2][s];
        assert!(
            (adaptive[2][s] - expected).abs() < 1e-10,
            "Kappa floor 0.1 at window 2 pop {s}: got {}, expected {}",
            adaptive[2][s], expected
        );
    }
}

#[test]
fn adaptive_kappa_upper_bound_250() {
    // One window with extremely high margin relative to median
    // Median is from sorted margins, so construct 3 windows where one is 10x the median
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let pairwise = vec![
        vec![-0.1, -10.0],  // margin = 9.9 → kappa = 9.9/0.5 = 19.8 → clamped to 2.5
        vec![-1.0, -1.5],   // margin = 0.5
        vec![-1.0, -1.5],   // margin = 0.5 → median = 0.5
    ];
    let base_weight = 0.3;
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, base_weight);

    // Window 0 should use w = base_weight * 2.5 = 0.75
    let w = base_weight * 2.5;
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (adaptive[0][s] - expected).abs() < 1e-10,
            "Kappa ceiling 2.5 at window 0 pop {s}: got {}, expected {}",
            adaptive[0][s], expected
        );
    }
}

// ============================================================================
// Mismatched lengths: standard longer than profile (zip truncates)
// ============================================================================

#[test]
fn adaptive_mismatched_lengths_truncates_to_shorter() {
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -0.5],
        vec![-2.0, -1.0], // extra window
    ];
    let pairwise = vec![
        vec![-0.5, -2.5],
        vec![-0.8, -1.2],
        // only 2 windows
    ];
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.3);
    // zip stops at min(3, 2) = 2
    assert_eq!(adaptive.len(), 2, "Should truncate to shorter input");
}

// ============================================================================
// All NEG_INFINITY profile emissions → margins all 0 → fallback to static
// ============================================================================

#[test]
fn adaptive_all_neg_inf_profile_falls_back_to_standard() {
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -0.5],
    ];
    let pairwise = vec![
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
    ];
    // Margins: best = -inf, second = -inf → both not finite → margin = 0
    // sorted_margins is empty → fallback to static blend
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.5);
    for t in 0..2 {
        for s in 0..2 {
            // static blend: std is finite, pairwise is -inf → returns std
            assert!(
                adaptive[t][s] == static_blend[t][s],
                "All -inf profile should equal static blend at [{t}][{s}]"
            );
        }
    }
}

// ============================================================================
// Large number of populations (10 pops)
// ============================================================================

#[test]
fn adaptive_ten_populations() {
    let k = 10;
    let n_windows = 5;
    let mut standard = Vec::with_capacity(n_windows);
    let mut pairwise = Vec::with_capacity(n_windows);
    for t in 0..n_windows {
        let mut std_row = vec![-2.0; k];
        let mut pw_row = vec![-2.0; k];
        // One population is "correct" per window
        std_row[t % k] = -0.5;
        pw_row[t % k] = -0.3;
        standard.push(std_row);
        pairwise.push(pw_row);
    }
    let result = blend_log_emissions_adaptive(&standard, &pairwise, 0.4);
    assert_eq!(result.len(), n_windows);
    for row in &result {
        assert_eq!(row.len(), k);
        for &v in row {
            assert!(v.is_finite(), "All values should be finite");
        }
    }
}

// ============================================================================
// Single window input
// ============================================================================

#[test]
fn adaptive_single_window() {
    let standard = vec![vec![-1.0, -2.0, -3.0]];
    let pairwise = vec![vec![-0.5, -2.5, -1.0]];
    // Single window: sorted_margins has 1 element → median = that element → kappa = 1.0
    // Should equal static blend
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.4);
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.4);
    for s in 0..3 {
        assert!(
            (adaptive[0][s] - static_blend[0][s]).abs() < 1e-12,
            "Single window should equal static blend at pop {s}"
        );
    }
}

// ============================================================================
// Monotonicity: adaptive preserves ranking of pairwise signal strength
// ============================================================================

#[test]
fn adaptive_monotonic_weight_with_signal_strength() {
    // 5 windows with increasing margin → pairwise weight should increase
    let standard = vec![vec![-1.0, -2.0]; 5];
    let pairwise = vec![
        vec![-0.9, -1.1],  // margin 0.2
        vec![-0.7, -1.3],  // margin 0.6
        vec![-0.5, -1.5],  // margin 1.0
        vec![-0.3, -1.7],  // margin 1.4
        vec![-0.1, -1.9],  // margin 1.8
    ];
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.4);
    // Deviation from standard should increase with margin
    let deviations: Vec<f64> = adaptive
        .iter()
        .enumerate()
        .map(|(t, row)| (row[0] - standard[t][0]).abs())
        .collect();
    for i in 1..5 {
        assert!(
            deviations[i] >= deviations[i - 1] - 1e-10,
            "Deviation should be monotonic: dev[{}]={:.6} < dev[{}]={:.6}",
            i, deviations[i], i - 1, deviations[i - 1]
        );
    }
}

// ============================================================================
// Median-margin computation: even number of positive margins
// ============================================================================

#[test]
fn adaptive_even_number_of_positive_margins() {
    // 4 windows with margins [1.0, 2.0, 3.0, 4.0]
    // Median index = len/2 = 2 → median = 3.0 (not averaged, integer division)
    let standard = vec![vec![-1.0, -2.0]; 4];
    let pairwise = vec![
        vec![-0.5, -1.5],  // margin = 1.0
        vec![-0.5, -2.5],  // margin = 2.0
        vec![-0.5, -3.5],  // margin = 3.0
        vec![-0.5, -4.5],  // margin = 4.0
    ];
    // With median=3.0:
    // kappa[0] = 1.0/3.0 ≈ 0.333 (above 0.1 floor)
    // kappa[3] = 4.0/3.0 ≈ 1.333 (below 2.5 ceiling)
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
    // Just verify all finite and different from static for non-median windows
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.5);
    let mut has_difference = false;
    for t in 0..4 {
        for s in 0..2 {
            assert!(adaptive[t][s].is_finite());
            if (adaptive[t][s] - static_blend[t][s]).abs() > 1e-10 {
                has_difference = true;
            }
        }
    }
    assert!(has_difference, "Non-uniform margins should differ from static blend");
}

// ============================================================================
// Mixed: some margins zero, some positive
// ============================================================================

#[test]
fn adaptive_mixed_zero_and_positive_margins() {
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let pairwise = vec![
        vec![-1.0, -1.0],  // margin = 0 (filtered out of sorted_margins)
        vec![-0.5, -2.5],  // margin = 2.0
        vec![-0.3, -3.0],  // margin = 2.7
    ];
    // sorted_margins = [2.0, 2.7], median = sorted_margins[1] = 2.7
    // Window 0: margin=0 → kappa = 0/2.7 = 0 → clamped to 0.1
    // Window 1: kappa = 2.0/2.7 ≈ 0.74
    // Window 2: kappa = 2.7/2.7 = 1.0
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.4);
    assert_eq!(adaptive.len(), 3);
    for row in &adaptive {
        for &v in row {
            assert!(v.is_finite());
        }
    }
    // Window 0 should have smallest pairwise influence (kappa=0.1)
    let dev_0 = (adaptive[0][0] - standard[0][0]).abs();
    let dev_2 = (adaptive[2][0] - standard[2][0]).abs();
    assert!(dev_0 < dev_2, "Zero-margin window should have less pairwise influence");
}

// ============================================================================
// Tiny median_margin near epsilon threshold (1e-10)
// ============================================================================

#[test]
fn adaptive_tiny_median_margin_uses_kappa_one() {
    // All margins are barely above zero but below 1e-10 → median_margin ≤ 1e-10 → kappa = 1.0
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -0.5],
    ];
    let pairwise = vec![
        vec![-1.0, -1.0 - 1e-12], // margin = 1e-12 > 0, passes filter
        vec![-0.5, -0.5 - 1e-12], // margin = 1e-12
    ];
    let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.3);
    // median_margin = 1e-12 < 1e-10 threshold → kappa = 1.0 for all → same as static
    let static_blend = blend_log_emissions(&standard, &pairwise, 0.3);
    for t in 0..2 {
        for s in 0..2 {
            assert!(
                (adaptive[t][s] - static_blend[t][s]).abs() < 1e-10,
                "Tiny median should use kappa=1 (equal to static) at [{t}][{s}]"
            );
        }
    }
}
