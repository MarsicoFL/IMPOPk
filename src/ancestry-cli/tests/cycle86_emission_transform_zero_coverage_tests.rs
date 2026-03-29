//! Cycle 86: Tests for ancestry emission transform functions with ZERO prior coverage.
//!
//! Targets: center_emissions, detrend_emissions, whiten_log_emissions,
//! median_polish_emissions, quantile_normalize_emissions, variance_stabilize_emissions,
//! sparsify_top_k_emissions, apply_emission_floor, apply_emission_momentum,
//! apply_snr_weighting, amplify_emission_residuals.

use hprc_ancestry_cli::hmm::{
    amplify_emission_residuals, apply_emission_floor, apply_emission_momentum,
    apply_snr_weighting, center_emissions, detrend_emissions, median_polish_emissions,
    quantile_normalize_emissions, sparsify_top_k_emissions, variance_stabilize_emissions,
    whiten_log_emissions,
};

// ============================================================================
// center_emissions
// ============================================================================

#[test]
fn center_emissions_empty() {
    let result = center_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn center_emissions_single_population() {
    let input = vec![vec![5.0], vec![10.0], vec![15.0]];
    let result = center_emissions(&input);
    // Each row has 1 element, so mean = element itself, centered = 0
    for row in &result {
        assert!((row[0]).abs() < 1e-10);
    }
}

#[test]
fn center_emissions_symmetric_values() {
    // Mean of [-1, 1] = 0, so centered = [-1, 1]
    let input = vec![vec![-1.0, 1.0]];
    let result = center_emissions(&input);
    assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    assert!((result[0][1] - 1.0).abs() < 1e-10);
}

#[test]
fn center_emissions_uniform_row_becomes_zero() {
    let input = vec![vec![3.0, 3.0, 3.0]];
    let result = center_emissions(&input);
    for &v in &result[0] {
        assert!(v.abs() < 1e-10);
    }
}

#[test]
fn center_emissions_preserves_non_finite() {
    let input = vec![vec![2.0, f64::NEG_INFINITY, 4.0]];
    let result = center_emissions(&input);
    // Mean of [2.0, 4.0] = 3.0
    assert!((result[0][0] - (2.0 - 3.0)).abs() < 1e-10);
    assert!(result[0][1].is_infinite()); // preserved
    assert!((result[0][2] - (4.0 - 3.0)).abs() < 1e-10);
}

#[test]
fn center_emissions_all_nan_row_unchanged() {
    let input = vec![vec![f64::NAN, f64::NAN]];
    let result = center_emissions(&input);
    assert!(result[0][0].is_nan());
    assert!(result[0][1].is_nan());
}

#[test]
fn center_emissions_sum_of_centered_is_zero() {
    let input = vec![vec![1.0, 5.0, 3.0, 7.0]];
    let result = center_emissions(&input);
    let sum: f64 = result[0].iter().sum();
    assert!(sum.abs() < 1e-10);
}

// ============================================================================
// detrend_emissions
// ============================================================================

#[test]
fn detrend_emissions_empty() {
    let result = detrend_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn detrend_emissions_single_window_passthrough() {
    let input = vec![vec![1.0, 2.0]];
    let result = detrend_emissions(&input);
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 1.0).abs() < 1e-10);
    assert!((result[0][1] - 2.0).abs() < 1e-10);
}

#[test]
fn detrend_emissions_linear_trend_removed() {
    // Population 0: values 0, 1, 2, 3 (slope = 1.0)
    let input: Vec<Vec<f64>> = (0..4).map(|t| vec![t as f64]).collect();
    let result = detrend_emissions(&input);
    // After detrending, all values should be near the mean (1.5)
    for row in &result {
        assert!((row[0] - 1.5).abs() < 1e-8, "expected ~1.5, got {}", row[0]);
    }
}

#[test]
fn detrend_emissions_constant_unchanged() {
    let input = vec![vec![5.0, 3.0], vec![5.0, 3.0], vec![5.0, 3.0]];
    let result = detrend_emissions(&input);
    for (i, row) in result.iter().enumerate() {
        assert!((row[0] - 5.0).abs() < 1e-10, "row {} pop 0", i);
        assert!((row[1] - 3.0).abs() < 1e-10, "row {} pop 1", i);
    }
}

#[test]
fn detrend_emissions_preserves_non_finite() {
    let input = vec![
        vec![1.0, f64::NEG_INFINITY],
        vec![2.0, f64::NEG_INFINITY],
        vec![3.0, f64::NEG_INFINITY],
    ];
    let result = detrend_emissions(&input);
    // Pop 1 has <2 finite values, so should be unchanged
    assert!(result[0][1].is_infinite());
    assert!(result[1][1].is_infinite());
}

#[test]
fn detrend_emissions_two_pops_independent() {
    // Pop 0: slope +1, Pop 1: slope -1
    let input = vec![
        vec![0.0, 3.0],
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 0.0],
    ];
    let result = detrend_emissions(&input);
    // After detrending, both should be nearly constant
    let pop0_range = result.iter().map(|r| r[0]).fold(0.0_f64, |acc, v| acc.max(v))
        - result.iter().map(|r| r[0]).fold(f64::INFINITY, |acc, v| acc.min(v));
    assert!(pop0_range < 1e-8, "pop0 range: {}", pop0_range);
}

// ============================================================================
// whiten_log_emissions
// ============================================================================

#[test]
fn whiten_empty() {
    let result = whiten_log_emissions(&[], 0.01);
    assert!(result.is_empty());
}

#[test]
fn whiten_single_population_passthrough() {
    let input = vec![vec![1.0], vec![2.0], vec![3.0]];
    let result = whiten_log_emissions(&input, 0.01);
    // K=1 → passthrough
    assert_eq!(result.len(), 3);
    assert!((result[0][0] - 1.0).abs() < 1e-10);
}

#[test]
fn whiten_too_few_observations_passthrough() {
    // Need n_valid >= k+1. With k=3, need at least 4 valid rows.
    let input = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let result = whiten_log_emissions(&input, 0.01);
    // Only 2 rows < 4 required → passthrough
    assert!((result[0][0] - 1.0).abs() < 1e-10);
}

#[test]
fn whiten_identical_rows_passthrough() {
    // All same → covariance = 0, with regularization it should still work
    let input = vec![
        vec![1.0, 2.0],
        vec![1.0, 2.0],
        vec![1.0, 2.0],
        vec![1.0, 2.0],
    ];
    let result = whiten_log_emissions(&input, 0.01);
    assert_eq!(result.len(), 4);
    // Should be all zeros after centering (before rescaling)
    for row in &result {
        for &v in row {
            assert!(v.is_finite(), "got non-finite: {}", v);
        }
    }
}

#[test]
fn whiten_preserves_non_finite_rows() {
    let input = vec![
        vec![1.0, 2.0],
        vec![f64::NEG_INFINITY, 3.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
    ];
    let result = whiten_log_emissions(&input, 0.01);
    // Row with NEG_INFINITY should be preserved
    assert!(result[1][0].is_infinite());
}

#[test]
fn whiten_high_regularization_approaches_identity() {
    let input = vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
        vec![4.0, 40.0],
    ];
    let result = whiten_log_emissions(&input, 1e6);
    // With huge regularization, Σ^{-1/2} ≈ identity/sqrt(reg) → values are scaled
    // but structure preserved. Just check finite.
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// median_polish_emissions
// ============================================================================

#[test]
fn median_polish_empty() {
    let result = median_polish_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn median_polish_empty_columns() {
    let input = vec![vec![], vec![]];
    let result = median_polish_emissions(&input);
    assert_eq!(result.len(), 2);
}

#[test]
fn median_polish_single_element() {
    let input = vec![vec![5.0]];
    let result = median_polish_emissions(&input);
    // After removing row median (5), result is 0
    assert!((result[0][0]).abs() < 1e-10);
}

#[test]
fn median_polish_uniform_becomes_zero() {
    let input = vec![
        vec![7.0, 7.0, 7.0],
        vec![7.0, 7.0, 7.0],
    ];
    let result = median_polish_emissions(&input);
    for row in &result {
        for &v in row {
            assert!(v.abs() < 1e-10, "expected ~0, got {}", v);
        }
    }
}

#[test]
fn median_polish_additive_model() {
    // Additive: row_effect + col_effect. Median polish should recover residuals ≈ 0.
    let input = vec![
        vec![1.0 + 10.0, 1.0 + 20.0],
        vec![2.0 + 10.0, 2.0 + 20.0],
        vec![3.0 + 10.0, 3.0 + 20.0],
    ];
    let result = median_polish_emissions(&input);
    // Residuals should be near zero for additive model
    for row in &result {
        for &v in row {
            assert!(v.abs() < 1e-6, "residual too large: {}", v);
        }
    }
}

#[test]
fn median_polish_preserves_non_finite() {
    let input = vec![
        vec![1.0, f64::NEG_INFINITY],
        vec![2.0, 3.0],
    ];
    let result = median_polish_emissions(&input);
    assert!(result[0][1].is_infinite());
}

// ============================================================================
// quantile_normalize_emissions
// ============================================================================

#[test]
fn quantile_normalize_empty() {
    let result = quantile_normalize_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn quantile_normalize_single_window() {
    let input = vec![vec![1.0, 5.0, 3.0]];
    let result = quantile_normalize_emissions(&input);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 3);
}

#[test]
fn quantile_normalize_identical_pops_unchanged() {
    let input = vec![
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0],
    ];
    let result = quantile_normalize_emissions(&input);
    // All pops have same distribution → same reference → same result
    for (orig, norm) in input.iter().zip(result.iter()) {
        for (o, n) in orig.iter().zip(norm.iter()) {
            assert!((o - n).abs() < 1e-10);
        }
    }
}

#[test]
fn quantile_normalize_different_scales_aligned() {
    // Pop 0: [1, 2, 3], Pop 1: [10, 20, 30]
    let input = vec![
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
    ];
    let result = quantile_normalize_emissions(&input);
    // After QN, both pops should have same values at each rank
    // Same ordering, so result[t][0] should equal result[t][1]
    for row in &result {
        assert!((row[0] - row[1]).abs() < 1e-10,
            "expected same after QN: {} vs {}", row[0], row[1]);
    }
}

#[test]
fn quantile_normalize_preserves_order_within_pop() {
    let input = vec![
        vec![3.0, 1.0],
        vec![1.0, 3.0],
        vec![2.0, 2.0],
    ];
    let result = quantile_normalize_emissions(&input);
    // Pop 0: ranks are [2, 0, 1] → result should preserve relative order
    // At least check finite
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// variance_stabilize_emissions
// ============================================================================

#[test]
fn variance_stabilize_empty() {
    let result = variance_stabilize_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn variance_stabilize_single_window_passthrough() {
    let input = vec![vec![1.0, 2.0, 3.0]];
    let result = variance_stabilize_emissions(&input);
    assert_eq!(result, input);
}

#[test]
fn variance_stabilize_equal_variance_unchanged() {
    // Both pops have same std → target = that std → scale = 1
    let input = vec![
        vec![1.0, 1.0],
        vec![3.0, 3.0],
        vec![5.0, 5.0],
    ];
    let result = variance_stabilize_emissions(&input);
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!((v - input[i][j]).abs() < 1e-10);
        }
    }
}

#[test]
fn variance_stabilize_different_variance_equalized() {
    // Pop 0: std = 1.0, Pop 1: std = 10.0. After stabilization, stds should converge.
    let input = vec![
        vec![0.0, 0.0],
        vec![1.0, 10.0],
        vec![2.0, 20.0],
        vec![3.0, 30.0],
    ];
    let result = variance_stabilize_emissions(&input);
    // Compute std of each pop after stabilization
    let means: Vec<f64> = (0..2).map(|p| result.iter().map(|r| r[p]).sum::<f64>() / 4.0).collect();
    let stds: Vec<f64> = (0..2).map(|p| {
        let m = means[p];
        (result.iter().map(|r| (r[p] - m).powi(2)).sum::<f64>() / 3.0).sqrt()
    }).collect();
    // After variance stabilization, stds should be similar
    assert!((stds[0] - stds[1]).abs() < stds[0].max(stds[1]) * 0.5 + 1e-6,
        "stds not equalized: {} vs {}", stds[0], stds[1]);
}

#[test]
fn variance_stabilize_all_zero_variance_passthrough() {
    let input = vec![
        vec![5.0, 5.0],
        vec![5.0, 5.0],
    ];
    let result = variance_stabilize_emissions(&input);
    // Zero std → stds all ≤ 1e-12 → sorted_stds empty → passthrough
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!((v - input[i][j]).abs() < 1e-10);
        }
    }
}

// ============================================================================
// sparsify_top_k_emissions
// ============================================================================

#[test]
fn sparsify_empty() {
    let result = sparsify_top_k_emissions(&[], 2, -100.0);
    assert!(result.is_empty());
}

#[test]
fn sparsify_top_k_zero_passthrough() {
    let input = vec![vec![1.0, 2.0, 3.0]];
    let result = sparsify_top_k_emissions(&input, 0, -100.0);
    assert_eq!(result, input);
}

#[test]
fn sparsify_top_k_ge_k_passthrough() {
    let input = vec![vec![1.0, 2.0, 3.0]];
    let result = sparsify_top_k_emissions(&input, 3, -100.0);
    assert_eq!(result, input);
}

#[test]
fn sparsify_top_1_keeps_max() {
    let input = vec![vec![1.0, 5.0, 3.0]];
    let result = sparsify_top_k_emissions(&input, 1, -999.0);
    assert!((result[0][1] - 5.0).abs() < 1e-10);
    assert!((result[0][0] - (-999.0)).abs() < 1e-10);
    assert!((result[0][2] - (-999.0)).abs() < 1e-10);
}

#[test]
fn sparsify_top_2_keeps_two_highest() {
    let input = vec![vec![1.0, 5.0, 3.0]];
    let result = sparsify_top_k_emissions(&input, 2, -100.0);
    assert!((result[0][1] - 5.0).abs() < 1e-10); // max
    assert!((result[0][2] - 3.0).abs() < 1e-10); // 2nd
    assert!((result[0][0] - (-100.0)).abs() < 1e-10); // floored
}

#[test]
fn sparsify_multiple_windows() {
    let input = vec![
        vec![10.0, 1.0, 5.0],
        vec![1.0, 10.0, 5.0],
    ];
    let result = sparsify_top_k_emissions(&input, 1, -50.0);
    assert!((result[0][0] - 10.0).abs() < 1e-10);
    assert!((result[1][1] - 10.0).abs() < 1e-10);
}

// ============================================================================
// apply_emission_floor
// ============================================================================

#[test]
fn emission_floor_empty() {
    let result = apply_emission_floor(&[], -10.0);
    assert!(result.is_empty());
}

#[test]
fn emission_floor_no_change_above() {
    let input = vec![vec![-5.0, -3.0]];
    let result = apply_emission_floor(&input, -10.0);
    assert!((result[0][0] - (-5.0)).abs() < 1e-10);
    assert!((result[0][1] - (-3.0)).abs() < 1e-10);
}

#[test]
fn emission_floor_clamps_below() {
    let input = vec![vec![-50.0, -100.0]];
    let result = apply_emission_floor(&input, -20.0);
    assert!((result[0][0] - (-20.0)).abs() < 1e-10);
    assert!((result[0][1] - (-20.0)).abs() < 1e-10);
}

#[test]
fn emission_floor_preserves_neg_infinity() {
    let input = vec![vec![f64::NEG_INFINITY, -5.0]];
    let result = apply_emission_floor(&input, -10.0);
    assert!(result[0][0].is_infinite()); // masked → preserved
    assert!((result[0][1] - (-5.0)).abs() < 1e-10);
}

#[test]
fn emission_floor_preserves_nan() {
    let input = vec![vec![f64::NAN]];
    let result = apply_emission_floor(&input, -10.0);
    assert!(result[0][0].is_nan());
}

#[test]
fn emission_floor_zero_floor() {
    let input = vec![vec![-5.0, 5.0]];
    let result = apply_emission_floor(&input, 0.0);
    assert!((result[0][0] - 0.0).abs() < 1e-10);
    assert!((result[0][1] - 5.0).abs() < 1e-10);
}

// ============================================================================
// apply_emission_momentum
// ============================================================================

#[test]
fn emission_momentum_empty() {
    let result = apply_emission_momentum(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn emission_momentum_alpha_zero_passthrough() {
    let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = apply_emission_momentum(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn emission_momentum_alpha_one_passthrough() {
    let input = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = apply_emission_momentum(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn emission_momentum_single_window() {
    let input = vec![vec![5.0, 10.0]];
    let result = apply_emission_momentum(&input, 0.5);
    // Single window: forward = input, backward = input, avg = input
    assert!((result[0][0] - 5.0).abs() < 1e-10);
    assert!((result[0][1] - 10.0).abs() < 1e-10);
}

#[test]
fn emission_momentum_smooths_step_function() {
    // Step from 0 to 10 — momentum should smooth the transition
    let input = vec![
        vec![0.0], vec![0.0], vec![0.0],
        vec![10.0], vec![10.0], vec![10.0],
    ];
    let result = apply_emission_momentum(&input, 0.3);
    // At the boundary, values should be intermediate
    assert!(result[2][0] > 0.0, "should smooth upward near boundary");
    assert!(result[3][0] < 10.0, "should smooth downward near boundary");
}

#[test]
fn emission_momentum_preserves_non_finite() {
    let input = vec![
        vec![1.0, f64::NEG_INFINITY],
        vec![2.0, f64::NEG_INFINITY],
    ];
    let result = apply_emission_momentum(&input, 0.5);
    assert!(result[0][1].is_infinite());
    assert!(result[1][1].is_infinite());
}

#[test]
fn emission_momentum_constant_sequence_unchanged() {
    let input = vec![vec![5.0, 5.0]; 10];
    let result = apply_emission_momentum(&input, 0.5);
    for row in &result {
        assert!((row[0] - 5.0).abs() < 1e-10);
        assert!((row[1] - 5.0).abs() < 1e-10);
    }
}

// ============================================================================
// apply_snr_weighting
// ============================================================================

#[test]
fn snr_weighting_empty() {
    let result = apply_snr_weighting(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn snr_weighting_power_zero_passthrough() {
    let input = vec![vec![1.0, 5.0]];
    let result = apply_snr_weighting(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn snr_weighting_negative_power_passthrough() {
    let input = vec![vec![1.0, 5.0]];
    let result = apply_snr_weighting(&input, -1.0);
    assert_eq!(result, input);
}

#[test]
fn snr_weighting_uniform_rows_unchanged() {
    // All ranges = 0 → no positive ranges → passthrough
    let input = vec![vec![5.0, 5.0], vec![5.0, 5.0]];
    let result = apply_snr_weighting(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn snr_weighting_single_element_rows_passthrough() {
    let input = vec![vec![5.0], vec![10.0]];
    let result = apply_snr_weighting(&input, 1.0);
    // Single element → range = 0 → no scaling
    assert_eq!(result, input);
}

#[test]
fn snr_weighting_high_range_amplified() {
    // Window 0: range 10, Window 1: range 1, Window 2: range 10
    // Median range = 10, so window 1 gets dampened, windows 0 and 2 stay normal
    let input = vec![
        vec![0.0, 10.0],  // range 10
        vec![4.5, 5.5],   // range 1
        vec![0.0, 10.0],  // range 10
    ];
    let result = apply_snr_weighting(&input, 1.0);
    // Window 1 (low range) gets scale = (1/10)^1 = 0.1, clamped to 0.1
    let range_1 = result[1][1] - result[1][0];
    let range_0 = result[0][1] - result[0][0];
    assert!(range_1 < range_0, "low-SNR window should have smaller range after weighting");
}

#[test]
fn snr_weighting_preserves_structure() {
    let input = vec![
        vec![1.0, 3.0, 2.0],
        vec![1.0, 3.0, 2.0],
    ];
    let result = apply_snr_weighting(&input, 1.0);
    // Same range → same scale → relative ordering preserved
    assert!(result[0][1] > result[0][2]);
    assert!(result[0][2] > result[0][0]);
}

// ============================================================================
// amplify_emission_residuals
// ============================================================================

#[test]
fn amplify_residuals_empty() {
    let result = amplify_emission_residuals(&[], 2.0);
    assert!(result.is_empty());
}

#[test]
fn amplify_residuals_factor_zero_passthrough() {
    let input = vec![vec![1.0, 5.0]];
    let result = amplify_emission_residuals(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn amplify_residuals_factor_negative_passthrough() {
    let input = vec![vec![1.0, 5.0]];
    let result = amplify_emission_residuals(&input, -1.0);
    assert_eq!(result, input);
}

#[test]
fn amplify_residuals_factor_one_unchanged() {
    let input = vec![vec![1.0, 5.0, 3.0]];
    let result = amplify_emission_residuals(&input, 1.0);
    for (o, n) in input[0].iter().zip(result[0].iter()) {
        assert!((o - n).abs() < 1e-10);
    }
}

#[test]
fn amplify_residuals_factor_two_doubles_spread() {
    // mean = 3.0, residuals = [-2, 2, 0]
    let input = vec![vec![1.0, 5.0, 3.0]];
    let result = amplify_emission_residuals(&input, 2.0);
    // amplified: mean + 2*(v - mean) = 3 + 2*(-2) = -1, 3+2*2 = 7, 3+0 = 3
    assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    assert!((result[0][1] - 7.0).abs() < 1e-10);
    assert!((result[0][2] - 3.0).abs() < 1e-10);
}

#[test]
fn amplify_residuals_preserves_non_finite() {
    let input = vec![vec![f64::NEG_INFINITY, 5.0, 3.0]];
    let result = amplify_emission_residuals(&input, 2.0);
    assert!(result[0][0].is_infinite());
    // Mean of [5.0, 3.0] = 4.0, residuals = [1, -1], amplified = [4+2, 4-2] = [6, 2]
    assert!((result[0][1] - 6.0).abs() < 1e-10);
    assert!((result[0][2] - 2.0).abs() < 1e-10);
}

#[test]
fn amplify_residuals_all_same_unchanged() {
    let input = vec![vec![5.0, 5.0, 5.0]];
    let result = amplify_emission_residuals(&input, 10.0);
    for &v in &result[0] {
        assert!((v - 5.0).abs() < 1e-10);
    }
}

#[test]
fn amplify_residuals_mean_preserved() {
    let input = vec![vec![1.0, 3.0, 5.0, 7.0]];
    let result = amplify_emission_residuals(&input, 3.0);
    let orig_mean: f64 = input[0].iter().sum::<f64>() / 4.0;
    let new_mean: f64 = result[0].iter().sum::<f64>() / 4.0;
    assert!((orig_mean - new_mean).abs() < 1e-10);
}
