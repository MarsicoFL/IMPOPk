//! Tier-1 edge case tests for untested emission transformation functions:
//! - whiten_log_emissions (ZCA whitening with Jacobi eigendecomposition)
//! - compute_heteroscedastic_temperatures (per-pop temperature estimation)
//! - precompute_heteroscedastic_log_emissions (per-pop softmax)
//! - compute_lookahead_transitions (sequence-aware transition modification)
//! - compute_cooccurrence_transitions (empirical transition learning)
//! - detrend_emissions (linear regression detrending)
//! - bayesian_shrink_emissions (mean shrinkage)
//! - apply_kurtosis_weighting (excess-kurtosis scaling)
//!
//! Focus: NaN/Inf safety, division-by-zero, empty inputs, singular matrices.

use hprc_ancestry_cli::hmm::{
    whiten_log_emissions, compute_heteroscedastic_temperatures,
    precompute_heteroscedastic_log_emissions, compute_lookahead_transitions,
    compute_cooccurrence_transitions, detrend_emissions, bayesian_shrink_emissions,
    apply_kurtosis_weighting,
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
};

// ===========================================================================
// Helpers
// ===========================================================================

fn make_params(n_pops: usize, switch_prob: f64) -> AncestryHmmParams {
    let pops: Vec<AncestralPopulation> = (0..n_pops)
        .map(|i| AncestralPopulation {
            name: format!("pop_{}", i),
            haplotypes: vec![format!("hap_{}_1", i), format!("hap_{}_2", i)],
        })
        .collect();
    AncestryHmmParams::new(pops, switch_prob)
}

fn make_obs(start: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 5000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

// ===========================================================================
// whiten_log_emissions
// ===========================================================================

#[test]
fn whiten_empty_input() {
    let result = whiten_log_emissions(&[], 0.01);
    assert!(result.is_empty());
}

#[test]
fn whiten_single_population() {
    // k=1 → early return (no whitening possible)
    let emissions = vec![vec![-1.0], vec![-2.0], vec![-0.5]];
    let result = whiten_log_emissions(&emissions, 0.01);
    assert_eq!(result, emissions);
}

#[test]
fn whiten_single_window() {
    // n=1, k=3 → n_valid < k+1 → early return
    let emissions = vec![vec![-1.0, -2.0, -0.5]];
    let result = whiten_log_emissions(&emissions, 0.01);
    assert_eq!(result, emissions);
}

#[test]
fn whiten_too_few_windows_for_covariance() {
    // n=3, k=3 → n_valid(3) < k+1(4) → early return
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-1.5, -1.0, -0.8],
        vec![-0.8, -1.5, -1.2],
    ];
    let result = whiten_log_emissions(&emissions, 0.01);
    assert_eq!(result, emissions);
}

#[test]
fn whiten_all_nan_rows() {
    // All rows contain NaN → n_valid=0 → early return
    let nan = f64::NAN;
    let emissions = vec![
        vec![nan, -1.0, -0.5],
        vec![-1.0, nan, -0.8],
        vec![-0.8, -1.5, nan],
        vec![nan, nan, nan],
        vec![-2.0, nan, -1.2],
    ];
    let result = whiten_log_emissions(&emissions, 0.01);
    // Rows with any non-finite value are excluded from covariance
    // Only rows where ALL values are finite count
    // Row 0: NaN → skip; Row 1: NaN → skip; Row 2: NaN → skip
    // Row 3: NaN → skip; Row 4: NaN → skip
    // n_valid = 0 < k+1 = 4 → early return
    assert_eq!(result.len(), emissions.len());
}

#[test]
fn whiten_neg_infinity_rows_preserved() {
    // NEG_INFINITY rows are passed through untouched
    let ni = f64::NEG_INFINITY;
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-1.5, -1.0, -0.8],
        vec![-0.8, -1.5, -1.2],
        vec![-2.0, -0.5, -1.0],
        vec![-1.2, ni, -0.9], // row with NEG_INFINITY
    ];
    let result = whiten_log_emissions(&emissions, 0.01);
    // The last row has NEG_INFINITY → returned as-is
    assert_eq!(result[4], emissions[4]);
    // First 4 rows are finite, n_valid=4, k=3 → 4 >= k+1=4 → whitening applied
    for i in 0..4 {
        for val in &result[i] {
            assert!(val.is_finite(), "whitened row {} has non-finite value", i);
        }
    }
}

#[test]
fn whiten_zero_variance_all_same() {
    // All finite rows identical → covariance is zero matrix → eigenvalues = 0
    // Regularization prevents division by zero
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-1.0, -2.0, -0.5],
        vec![-1.0, -2.0, -0.5],
        vec![-1.0, -2.0, -0.5],
    ];
    let result = whiten_log_emissions(&emissions, 0.01);
    // After centering, all centered values are 0, so result should be all zeros
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "zero-variance whitening produced non-finite");
            assert!(val.abs() < 1e-6, "zero-variance whitening should produce ~0, got {}", val);
        }
    }
}

#[test]
fn whiten_near_singular_covariance() {
    // Two populations perfectly correlated (pop_0 = -pop_1 + const)
    // This creates a near-singular covariance — regularization must handle it
    let emissions: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            let x = i as f64 * 0.1;
            vec![x, -x + 1.0]
        })
        .collect();
    let result = whiten_log_emissions(&emissions, 0.01);
    assert_eq!(result.len(), 10);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "near-singular whitening produced non-finite");
        }
    }
}

#[test]
fn whiten_regularization_zero() {
    // regularization=0 → internally clamped to 1e-10
    let emissions: Vec<Vec<f64>> = (0..8)
        .map(|i| vec![-1.0 + i as f64 * 0.3, -2.0 + i as f64 * 0.1])
        .collect();
    let result = whiten_log_emissions(&emissions, 0.0);
    assert_eq!(result.len(), 8);
    for row in &result {
        for &val in row {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn whiten_large_regularization() {
    // Very large regularization → eigenvalues dominated by regularization → near-identity whitening
    let emissions: Vec<Vec<f64>> = (0..8)
        .map(|i| vec![-1.0 + i as f64 * 0.3, -2.0 + i as f64 * 0.1])
        .collect();
    let result = whiten_log_emissions(&emissions, 1e6);
    assert_eq!(result.len(), 8);
    for row in &result {
        for &val in row {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn whiten_output_decorrelated() {
    // After whitening, output columns should have lower cross-correlation
    let emissions: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let x = i as f64 * 0.1;
            vec![x + 0.5 * (i as f64 * 0.7).sin(), x * 0.8 + 0.2]
        })
        .collect();

    let result = whiten_log_emissions(&emissions, 0.01);

    // Compute correlation of output columns
    let n = result.len() as f64;
    let mean_0: f64 = result.iter().map(|r| r[0]).sum::<f64>() / n;
    let mean_1: f64 = result.iter().map(|r| r[1]).sum::<f64>() / n;
    let cov: f64 = result.iter().map(|r| (r[0] - mean_0) * (r[1] - mean_1)).sum::<f64>() / n;
    let var_0: f64 = result.iter().map(|r| (r[0] - mean_0).powi(2)).sum::<f64>() / n;
    let var_1: f64 = result.iter().map(|r| (r[1] - mean_1).powi(2)).sum::<f64>() / n;

    let corr = if var_0 > 0.0 && var_1 > 0.0 {
        cov / (var_0.sqrt() * var_1.sqrt())
    } else {
        0.0
    };
    assert!(corr.abs() < 0.3, "whitened columns should be decorrelated, got corr={}", corr);
}

// ===========================================================================
// compute_heteroscedastic_temperatures
// ===========================================================================

#[test]
fn hetero_temps_empty_variances() {
    let result = compute_heteroscedastic_temperatures(&[], 1.0, 0.5);
    assert!(result.is_empty());
}

#[test]
fn hetero_temps_zero_gamma() {
    // gamma=0 → base_temp for all
    let result = compute_heteroscedastic_temperatures(&[1.0, 2.0, 0.5], 1.0, 0.0);
    assert_eq!(result, vec![1.0, 1.0, 1.0]);
}

#[test]
fn hetero_temps_negative_gamma() {
    // gamma < 0 → base_temp for all
    let result = compute_heteroscedastic_temperatures(&[1.0, 2.0], 1.0, -1.0);
    assert_eq!(result, vec![1.0, 1.0]);
}

#[test]
fn hetero_temps_all_zero_variances() {
    // All variances = 0 → all stds = 0 → no positive stds → base_temp for all
    let result = compute_heteroscedastic_temperatures(&[0.0, 0.0, 0.0], 1.0, 0.5);
    assert_eq!(result, vec![1.0, 1.0, 1.0]);
}

#[test]
fn hetero_temps_single_variance() {
    // Single positive variance → median = that std → ratio = 1 → factor = 1 → base_temp
    let result = compute_heteroscedastic_temperatures(&[4.0], 1.0, 0.5);
    assert!((result[0] - 1.0).abs() < 1e-10);
}

#[test]
fn hetero_temps_mixed_zero_and_positive() {
    // Some zero, some positive — zero variances get base_temp
    let result = compute_heteroscedastic_temperatures(&[0.0, 4.0, 0.0, 9.0], 1.0, 0.5);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[2], 1.0);
    // Non-zero get scaled
    assert!(result[1].is_finite());
    assert!(result[3].is_finite());
}

#[test]
fn hetero_temps_extreme_variance_clamped() {
    // Very high variance → high ratio → high factor → clamped to base*3.0
    let result = compute_heteroscedastic_temperatures(&[1.0, 10000.0], 1.0, 10.0);
    assert!(result[1] <= 3.0, "should be clamped to base*3.0, got {}", result[1]);
}

#[test]
fn hetero_temps_tiny_variance_clamped() {
    // Very low variance compared to median → low ratio → clamped to base*0.3
    let result = compute_heteroscedastic_temperatures(&[0.0001, 100.0, 100.0], 1.0, 10.0);
    assert!(result[0] >= 0.3, "should be clamped to base*0.3, got {}", result[0]);
}

#[test]
fn hetero_temps_nan_variance_propagates() {
    // BUG DOC: NaN variance → NaN sqrt → `NaN <= 0.0` is false in Rust →
    // proceeds to compute ratio → NaN propagates through.
    // This is a known NaN-propagation issue (filed for algo_dev).
    let result = compute_heteroscedastic_temperatures(&[f64::NAN, 4.0, 9.0], 1.0, 0.5);
    assert!(result[0].is_nan(), "NaN variance currently propagates NaN (known issue)");
    // Other pops should be unaffected
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

#[test]
fn hetero_temps_negative_variance_propagates() {
    // BUG DOC: Negative variance → NaN sqrt → `NaN <= 0.0` is false →
    // NaN propagates. Same root cause as NaN variance.
    let result = compute_heteroscedastic_temperatures(&[-1.0, 4.0, 9.0], 1.0, 0.5);
    assert!(result[0].is_nan(), "negative variance currently propagates NaN (known issue)");
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

// ===========================================================================
// compute_lookahead_transitions
// ===========================================================================

#[test]
fn lookahead_empty_emissions() {
    let params = make_params(3, 0.01);
    let result = compute_lookahead_transitions(&[], &[], &params, 5);
    assert!(result.is_empty());
}

#[test]
fn lookahead_zero_radius() {
    let params = make_params(3, 0.01);
    let emissions = vec![vec![-1.0, -2.0, -0.5]];
    let result = compute_lookahead_transitions(&emissions, &[0], &params, 0);
    assert!(result.is_empty());
}

#[test]
fn lookahead_single_pop() {
    // k=1 → k >= 2 is false → margin = 0 → no boost
    let params = make_params(1, 0.01);
    let emissions = vec![vec![-1.0]; 5];
    let states = vec![0; 5];
    let result = compute_lookahead_transitions(&emissions, &states, &params, 3);
    assert_eq!(result.len(), 5);
    // All transitions should be valid log-probabilities
    for trans in &result {
        for row in trans {
            for &val in row {
                assert!(val.is_finite());
            }
        }
    }
}

#[test]
fn lookahead_single_window() {
    // n=1 → lookahead window is empty for t=0 → no boost
    let params = make_params(3, 0.01);
    let emissions = vec![vec![-1.0, -2.0, -0.5]];
    let states = vec![0];
    let result = compute_lookahead_transitions(&emissions, &states, &params, 5);
    assert_eq!(result.len(), 1);
}

#[test]
fn lookahead_all_neg_infinity_emissions() {
    let params = make_params(3, 0.01);
    let ni = f64::NEG_INFINITY;
    let emissions = vec![vec![ni, ni, ni]; 5];
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_lookahead_transitions(&emissions, &states, &params, 3);
    assert_eq!(result.len(), 5);
    // All NEG_INFINITY → sums are all 0, no finite data → finite_count=5 but sums all zero
    for trans in &result {
        for row in trans {
            for &val in row {
                assert!(val.is_finite() || val == f64::NEG_INFINITY,
                    "should be finite or -inf, got {}", val);
            }
        }
    }
}

#[test]
fn lookahead_state_out_of_range() {
    // current state >= k → the condition `current < k` prevents boost
    let params = make_params(3, 0.01);
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-0.5, -1.0, -2.0],
        vec![-2.0, -0.5, -1.0],
    ];
    let states = vec![99, 0, 1]; // state 99 is out of range
    let result = compute_lookahead_transitions(&emissions, &states, &params, 3);
    assert_eq!(result.len(), 3);
    // Should not panic
}

#[test]
fn lookahead_last_window_no_lookahead() {
    // For the last window, t+1 >= n → no lookahead → base transitions
    let params = make_params(3, 0.01);
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-0.5, -1.0, -2.0],
    ];
    let states = vec![0, 1];
    let result = compute_lookahead_transitions(&emissions, &states, &params, 3);
    assert_eq!(result.len(), 2);
    // Last window should have unmodified base transitions
}

#[test]
fn lookahead_normalizes_rows() {
    // After boosting, row sums in probability space should equal 1
    let params = make_params(3, 0.01);
    let emissions = vec![
        vec![-1.0, -2.0, -0.5],
        vec![-0.5, -1.0, -2.0],
        vec![-2.0, -0.5, -1.0],
        vec![-1.0, -0.5, -2.0],
        vec![-0.5, -2.0, -1.0],
    ];
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_lookahead_transitions(&emissions, &states, &params, 3);
    for (t, trans) in result.iter().enumerate() {
        for (i, row) in trans.iter().enumerate() {
            let sum_prob: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!(
                (sum_prob - 1.0).abs() < 0.01,
                "row normalization failed at t={}, i={}: sum={}", t, i, sum_prob
            );
        }
    }
}

// ===========================================================================
// compute_cooccurrence_transitions
// ===========================================================================

#[test]
fn cooccurrence_empty_states() {
    let params = make_params(3, 0.01);
    let result = compute_cooccurrence_transitions(&[], &params, 1.0);
    // Should return valid base transitions
    assert_eq!(result.len(), 3);
    for row in &result {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn cooccurrence_single_state() {
    // Single state → no windows(2) → no transitions → base transitions
    let params = make_params(3, 0.01);
    let result = compute_cooccurrence_transitions(&[0], &params, 1.0);
    assert_eq!(result.len(), 3);
}

#[test]
fn cooccurrence_no_switches() {
    // All same state → no transitions counted → base transitions only
    let params = make_params(3, 0.01);
    let states = vec![1; 20];
    let result = compute_cooccurrence_transitions(&states, &params, 1.0);
    assert_eq!(result.len(), 3);
    for row in &result {
        let sum_prob: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum_prob - 1.0).abs() < 0.01, "sum={}", sum_prob);
    }
}

#[test]
fn cooccurrence_all_switches() {
    // Alternating states → many transitions
    let params = make_params(2, 0.01);
    let states: Vec<usize> = (0..20).map(|i| i % 2).collect();
    let result = compute_cooccurrence_transitions(&states, &params, 1.0);
    assert_eq!(result.len(), 2);
    for row in &result {
        let sum_prob: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum_prob - 1.0).abs() < 0.01, "sum={}", sum_prob);
    }
}

#[test]
fn cooccurrence_zero_weight() {
    // weight=0 → no bonus → pure base transitions
    let params = make_params(3, 0.01);
    let states = vec![0, 1, 2, 0, 1, 2];
    let result = compute_cooccurrence_transitions(&states, &params, 0.0);
    assert_eq!(result.len(), 3);
}

#[test]
fn cooccurrence_normalizes_rows() {
    let params = make_params(3, 0.01);
    let states = vec![0, 1, 0, 2, 1, 0, 2, 1, 0, 2];
    let result = compute_cooccurrence_transitions(&states, &params, 2.0);
    for (i, row) in result.iter().enumerate() {
        let sum_prob: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!(
            (sum_prob - 1.0).abs() < 0.01,
            "row {} not normalized: sum={}", i, sum_prob
        );
    }
}

// ===========================================================================
// detrend_emissions
// ===========================================================================

#[test]
fn detrend_empty() {
    let result = detrend_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn detrend_single_window() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = detrend_emissions(&emissions);
    assert_eq!(result, emissions);
}

#[test]
fn detrend_constant_emissions() {
    // No trend → detrending should not change values
    let emissions = vec![vec![-1.0, -2.0]; 10];
    let result = detrend_emissions(&emissions);
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                (val - emissions[i][j]).abs() < 1e-10,
                "constant emission changed at [{},{}]: {} vs {}",
                i, j, val, emissions[i][j]
            );
        }
    }
}

#[test]
fn detrend_linear_trend_removed() {
    // Perfect linear trend → should become flat (constant mean)
    let emissions: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![-1.0 + i as f64 * 0.5, -2.0 + i as f64 * 0.3])
        .collect();
    let result = detrend_emissions(&emissions);

    // After detrending, values should cluster near the mean
    let mean_0: f64 = result.iter().map(|r| r[0]).sum::<f64>() / 10.0;
    for row in &result {
        assert!(
            (row[0] - mean_0).abs() < 1e-8,
            "detrended value {} far from mean {}",
            row[0], mean_0
        );
    }
}

#[test]
fn detrend_neg_infinity_preserved() {
    // NEG_INFINITY values should remain NEG_INFINITY
    let ni = f64::NEG_INFINITY;
    let emissions = vec![
        vec![-1.0, ni],
        vec![-2.0, -1.5],
        vec![-3.0, ni],
        vec![-4.0, -2.5],
    ];
    let result = detrend_emissions(&emissions);
    assert_eq!(result[0][1], ni);
    assert_eq!(result[2][1], ni);
}

#[test]
fn detrend_preserves_mean() {
    // Detrending should preserve the global mean per population
    let emissions: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![-1.0 + i as f64 * 0.2 + (i as f64).sin() * 0.1])
        .collect();
    let orig_mean: f64 = emissions.iter().map(|r| r[0]).sum::<f64>() / 20.0;
    let result = detrend_emissions(&emissions);
    let new_mean: f64 = result.iter().map(|r| r[0]).sum::<f64>() / 20.0;
    assert!(
        (orig_mean - new_mean).abs() < 1e-6,
        "detrending changed mean: {} vs {}", orig_mean, new_mean
    );
}

// ===========================================================================
// bayesian_shrink_emissions
// ===========================================================================

#[test]
fn shrink_empty() {
    let result = bayesian_shrink_emissions(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn shrink_zero_alpha() {
    let emissions = vec![vec![-1.0, -2.0], vec![-3.0, -0.5]];
    let result = bayesian_shrink_emissions(&emissions, 0.0);
    assert_eq!(result, emissions);
}

#[test]
fn shrink_negative_alpha() {
    let emissions = vec![vec![-1.0, -2.0], vec![-3.0, -0.5]];
    let result = bayesian_shrink_emissions(&emissions, -0.5);
    assert_eq!(result, emissions);
}

#[test]
fn shrink_alpha_one_becomes_mean() {
    // alpha=1 → all windows become the global mean
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
        vec![-5.0, -6.0],
    ];
    let result = bayesian_shrink_emissions(&emissions, 1.0);
    let mean_0 = (-1.0 + -3.0 + -5.0) / 3.0;
    let mean_1 = (-2.0 + -4.0 + -6.0) / 3.0;
    for row in &result {
        assert!((row[0] - mean_0).abs() < 1e-10);
        assert!((row[1] - mean_1).abs() < 1e-10);
    }
}

#[test]
fn shrink_neg_infinity_untouched() {
    let ni = f64::NEG_INFINITY;
    let emissions = vec![
        vec![-1.0, ni],
        vec![-3.0, -2.0],
    ];
    let result = bayesian_shrink_emissions(&emissions, 0.5);
    assert_eq!(result[0][1], ni);
    assert!(result[1][1].is_finite());
}

#[test]
fn shrink_reduces_variance() {
    // Shrinkage should reduce the variance of finite values
    let emissions: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![-5.0 + i as f64 * 0.5])
        .collect();
    let var_before: f64 = {
        let mean: f64 = emissions.iter().map(|r| r[0]).sum::<f64>() / 20.0;
        emissions.iter().map(|r| (r[0] - mean).powi(2)).sum::<f64>() / 20.0
    };
    let result = bayesian_shrink_emissions(&emissions, 0.5);
    let var_after: f64 = {
        let mean: f64 = result.iter().map(|r| r[0]).sum::<f64>() / 20.0;
        result.iter().map(|r| (r[0] - mean).powi(2)).sum::<f64>() / 20.0
    };
    assert!(var_after < var_before, "shrinkage should reduce variance");
}

// ===========================================================================
// apply_kurtosis_weighting
// ===========================================================================

#[test]
fn kurtosis_too_few_windows() {
    // n < 4 → early return
    let emissions = vec![vec![-1.0, -2.0]; 3];
    let result = apply_kurtosis_weighting(&emissions, 0.5);
    assert_eq!(result, emissions);
}

#[test]
fn kurtosis_empty() {
    let result = apply_kurtosis_weighting(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn kurtosis_constant_emissions() {
    // All values identical → variance = 0 → kurtosis defaults to 3
    // All populations same kurtosis → scale = 1 → no change
    let emissions = vec![vec![-1.0, -2.0]; 10];
    let result = apply_kurtosis_weighting(&emissions, 0.5);
    for (i, row) in result.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                (val - emissions[i][j]).abs() < 1e-10,
                "constant emission changed at [{},{}]", i, j
            );
        }
    }
}

#[test]
fn kurtosis_with_neg_infinity() {
    let ni = f64::NEG_INFINITY;
    let mut emissions: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![-1.0 + i as f64 * 0.2, -2.0 + i as f64 * 0.1])
        .collect();
    emissions[3][0] = ni;
    let result = apply_kurtosis_weighting(&emissions, 0.5);
    assert_eq!(result.len(), 10);
    // NEG_INFINITY values should remain non-finite
    assert!(!result[3][0].is_finite());
}

#[test]
fn kurtosis_all_finite_output() {
    // Normal-like data → should produce all-finite output
    let emissions: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![
            -1.0 + (i as f64 * 0.7).sin(),
            -2.0 + (i as f64 * 0.3).cos(),
            -0.5 + (i as f64 * 0.5).sin() * 0.3,
        ])
        .collect();
    let result = apply_kurtosis_weighting(&emissions, 1.0);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "kurtosis weighting produced non-finite: {}", val);
        }
    }
}

// ===========================================================================
// precompute_heteroscedastic_log_emissions
// ===========================================================================

#[test]
fn hetero_emissions_empty_obs() {
    let params = make_params(3, 0.01);
    let result = precompute_heteroscedastic_log_emissions(&[], &params, &[1.0, 1.0, 1.0]);
    assert!(result.is_empty());
}

#[test]
fn hetero_emissions_single_valid_pop() {
    // Only one population has data → valid.len() <= 1 → that pop gets 0.0
    let params = make_params(3, 0.01);
    let obs = vec![make_obs(0, &[("hap_0_1", 0.95)])];
    let result = precompute_heteroscedastic_log_emissions(&obs, &params, &[1.0, 1.0, 1.0]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0][0], 0.0);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
    assert_eq!(result[0][2], f64::NEG_INFINITY);
}

#[test]
fn hetero_emissions_no_data_for_any_pop() {
    // Similarities don't match any haplotype → all pops have no data
    let params = make_params(3, 0.01);
    let obs = vec![make_obs(0, &[("unknown_hap", 0.95)])];
    let result = precompute_heteroscedastic_log_emissions(&obs, &params, &[1.0, 1.0, 1.0]);
    assert_eq!(result.len(), 1);
    // All NEG_INFINITY since no valid populations
    for &val in &result[0] {
        assert_eq!(val, f64::NEG_INFINITY);
    }
}

#[test]
fn hetero_emissions_uniform_temperatures() {
    // All temperatures equal → should behave like standard softmax
    let params = make_params(3, 0.01);
    let obs = vec![make_obs(0, &[
        ("hap_0_1", 0.95), ("hap_0_2", 0.90),
        ("hap_1_1", 0.80), ("hap_1_2", 0.75),
        ("hap_2_1", 0.85), ("hap_2_2", 0.82),
    ])];
    let result = precompute_heteroscedastic_log_emissions(&obs, &params, &[1.0, 1.0, 1.0]);
    assert_eq!(result.len(), 1);
    // Should be valid log probabilities summing to ~0 in exp space
    let sum_prob: f64 = result[0].iter().filter(|v| v.is_finite()).map(|&v| v.exp()).sum();
    assert!((sum_prob - 1.0).abs() < 0.01, "sum={}", sum_prob);
}

#[test]
fn hetero_emissions_extreme_temperature_ratio() {
    // Pop 0 has very low temp (sharp), pop 1 has very high temp (soft)
    let params = make_params(2, 0.01);
    let obs = vec![make_obs(0, &[
        ("hap_0_1", 0.95), ("hap_0_2", 0.90),
        ("hap_1_1", 0.80), ("hap_1_2", 0.75),
    ])];
    let result = precompute_heteroscedastic_log_emissions(&obs, &params, &[0.1, 10.0]);
    assert_eq!(result.len(), 1);
    // Both should be finite
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
    // Pop 0 (higher sim, lower temp) should dominate
    assert!(result[0][0] > result[0][1]);
}

#[test]
fn hetero_emissions_zero_similarity_excluded() {
    // Similarity = 0.0 → filtered out by `s > 0.0` check
    let params = make_params(2, 0.01);
    let obs = vec![make_obs(0, &[
        ("hap_0_1", 0.0), ("hap_0_2", 0.0),
        ("hap_1_1", 0.80), ("hap_1_2", 0.75),
    ])];
    let result = precompute_heteroscedastic_log_emissions(&obs, &params, &[1.0, 1.0]);
    assert_eq!(result.len(), 1);
    // Pop 0 has zero sim → excluded → only pop 1 valid → single valid → gets 0.0
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}
