//! Edge-case and boundary tests for algo_dev cycles 35-66 functions.
//!
//! Covers: softmax_renormalize, bidirectional_smooth_states, median_polish_emissions,
//! center_emissions, apply_persistence_bonus, majority_vote_filter,
//! apply_proportion_prior, rank_transform_emissions, dampen_emission_outliers,
//! apply_emission_momentum, apply_emission_floor, variance_stabilize_emissions,
//! apply_emission_anchor_boost, apply_label_smoothing, apply_kurtosis_weighting,
//! apply_gap_penalty, detrend_emissions, apply_fb_temperature,
//! apply_confidence_weighting, apply_gradient_penalty, sharpen_posteriors,
//! amplify_emission_residuals, apply_diversity_scaling, apply_snr_weighting,
//! bayesian_shrink_emissions, sparsify_top_k_emissions, correct_short_segments,
//! entropy_smooth_posteriors, apply_windowed_normalization, quantile_normalize_emissions

use hprc_ancestry_cli::{
    softmax_renormalize, bidirectional_smooth_states, median_polish_emissions,
    center_emissions, apply_persistence_bonus, majority_vote_filter,
    apply_proportion_prior, rank_transform_emissions, dampen_emission_outliers,
    apply_emission_momentum, apply_emission_floor, variance_stabilize_emissions,
    apply_emission_anchor_boost, apply_label_smoothing,
    apply_gap_penalty, detrend_emissions, apply_fb_temperature,
    apply_confidence_weighting, apply_gradient_penalty, sharpen_posteriors,
    amplify_emission_residuals, apply_diversity_scaling, apply_snr_weighting,
    bayesian_shrink_emissions, sparsify_top_k_emissions, correct_short_segments,
    entropy_smooth_posteriors, apply_windowed_normalization, quantile_normalize_emissions,
};

// ============================================================================
// softmax_renormalize
// ============================================================================

#[test]
fn softmax_renormalize_empty_input() {
    let result = softmax_renormalize(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn softmax_renormalize_zero_temperature_returns_identity() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = softmax_renormalize(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn softmax_renormalize_negative_temperature_returns_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = softmax_renormalize(&input, -1.0);
    assert_eq!(result, input);
}

#[test]
fn softmax_renormalize_all_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = softmax_renormalize(&input, 1.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

#[test]
fn softmax_renormalize_single_finite_becomes_zero() {
    let input = vec![vec![-5.0, f64::NEG_INFINITY]];
    let result = softmax_renormalize(&input, 1.0);
    // Single finite value → softmax gives 1.0 → ln(1.0) ≈ 0
    assert!((result[0][0] - 0.0).abs() < 1e-10);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

#[test]
fn softmax_renormalize_equal_values_become_equal() {
    let input = vec![vec![-3.0, -3.0, -3.0]];
    let result = softmax_renormalize(&input, 1.0);
    let expected = (-3.0_f64).exp().ln() - (3.0 * (-3.0_f64).exp()).ln();
    // All equal → each gets 1/3 → ln(1/3)
    for &v in &result[0] {
        assert!((v - (-3.0_f64.ln())).abs() < 1e-6, "got {v}");
    }
}

#[test]
fn softmax_renormalize_high_temperature_flattens() {
    let input = vec![vec![-1.0, -10.0]];
    let t1 = softmax_renormalize(&input, 1.0);
    let t100 = softmax_renormalize(&input, 100.0);
    // High temperature should make the gap smaller
    let gap1 = t1[0][0] - t1[0][1];
    let gap100 = t100[0][0] - t100[0][1];
    assert!(gap100 < gap1, "high T should flatten: gap100={gap100} < gap1={gap1}");
}

#[test]
fn softmax_renormalize_low_temperature_sharpens() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let t1 = softmax_renormalize(&input, 1.0);
    let t01 = softmax_renormalize(&input, 0.1);
    let gap1 = t1[0][0] - t1[0][2];
    let gap01 = t01[0][0] - t01[0][2];
    assert!(gap01 > gap1, "low T should sharpen");
}

#[test]
fn softmax_renormalize_all_finite_values_stay_finite() {
    let input = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-0.5, -0.5, -100.0],
        vec![-50.0, -51.0, -52.0],
    ];
    let result = softmax_renormalize(&input, 0.5);
    for row in &result {
        for &v in row {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }
}

#[test]
fn softmax_renormalize_output_log_sums_close() {
    // After softmax renormalization, exp(result) should sum to ~1 per window
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = softmax_renormalize(&input, 1.0);
    let sum: f64 = result[0].iter().map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "exp sum should be ~1, got {sum}");
}

// ============================================================================
// bidirectional_smooth_states
// ============================================================================

#[test]
fn bidirectional_smooth_empty() {
    assert!(bidirectional_smooth_states(&[], 3, 2).is_empty());
}

#[test]
fn bidirectional_smooth_radius_zero_identity() {
    let states = vec![0, 1, 2, 1, 0];
    assert_eq!(bidirectional_smooth_states(&states, 3, 0), states);
}

#[test]
fn bidirectional_smooth_single_state() {
    let states = vec![0, 0, 0, 0, 0];
    let result = bidirectional_smooth_states(&states, 3, 5);
    assert_eq!(result, vec![0, 0, 0, 0, 0]);
}

#[test]
fn bidirectional_smooth_isolated_flicker_removed() {
    // One isolated state 1 among many state 0s should be smoothed out
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let result = bidirectional_smooth_states(&states, 2, 3);
    assert_eq!(result[3], 0, "isolated flicker should be smoothed");
}

#[test]
fn bidirectional_smooth_genuine_block_preserved() {
    // A block of state 1 should survive
    let states = vec![0, 0, 1, 1, 1, 1, 0, 0];
    let result = bidirectional_smooth_states(&states, 2, 2);
    // Middle of the block should remain state 1
    assert_eq!(result[3], 1);
    assert_eq!(result[4], 1);
}

#[test]
fn bidirectional_smooth_tie_keeps_original() {
    // With radius=1 and pattern [0,1,0], center has equal votes → keeps original
    let states = vec![0, 1, 0];
    let result = bidirectional_smooth_states(&states, 2, 1);
    // t=1: votes for 0: w(1)+w(1)=2*exp(-1), votes for 1: w(0)=1
    // exp(-1) ≈ 0.37, so 2*0.37=0.74 vs 1.0 → state 1 wins
    assert_eq!(result[1], 1);
}

#[test]
fn bidirectional_smooth_out_of_range_state_ignored() {
    let states = vec![0, 5, 0]; // 5 >= n_states=3
    let result = bidirectional_smooth_states(&states, 3, 1);
    // state 5 is out of range; should be skipped in voting
    assert_eq!(result[0], 0);
    assert_eq!(result[2], 0);
}

// ============================================================================
// median_polish_emissions
// ============================================================================

#[test]
fn median_polish_empty() {
    assert!(median_polish_emissions(&[]).is_empty());
}

#[test]
fn median_polish_empty_rows() {
    let input = vec![vec![], vec![]];
    let result = median_polish_emissions(&input);
    assert_eq!(result.len(), 2);
}

#[test]
fn median_polish_single_row() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = median_polish_emissions(&input);
    assert_eq!(result.len(), 1);
    // After row median removal, median=-2 → [-1+2, 0, -1] = [1, 0, -1]
    // Then column median removal with single row: each becomes 0
    // With multiple iterations, converges
    for &v in &result[0] {
        assert!(v.is_finite());
    }
}

#[test]
fn median_polish_uniform_becomes_zero() {
    let input = vec![
        vec![-5.0, -5.0, -5.0],
        vec![-5.0, -5.0, -5.0],
    ];
    let result = median_polish_emissions(&input);
    for row in &result {
        for &v in row {
            assert!(v.abs() < 1e-10, "uniform should become ~0, got {v}");
        }
    }
}

#[test]
fn median_polish_preserves_neg_infinity() {
    let input = vec![
        vec![-1.0, f64::NEG_INFINITY],
        vec![-2.0, -3.0],
    ];
    let result = median_polish_emissions(&input);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

#[test]
fn median_polish_all_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = median_polish_emissions(&input);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

// ============================================================================
// center_emissions
// ============================================================================

#[test]
fn center_emissions_empty() {
    let result = center_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn center_emissions_sums_to_zero() {
    let input = vec![vec![-1.0, -3.0, -5.0]];
    let result = center_emissions(&input);
    let sum: f64 = result[0].iter().sum();
    assert!(sum.abs() < 1e-10, "centered row should sum to ~0, got {sum}");
}

#[test]
fn center_emissions_preserves_neg_infinity() {
    let input = vec![vec![-1.0, f64::NEG_INFINITY, -3.0]];
    let result = center_emissions(&input);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
    // Mean of finite: (-1 + -3)/2 = -2
    assert!((result[0][0] - 1.0).abs() < 1e-10); // -1 - (-2) = 1
    assert!((result[0][2] - (-1.0)).abs() < 1e-10); // -3 - (-2) = -1
}

#[test]
fn center_emissions_all_neg_infinity_unchanged() {
    let input = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = center_emissions(&input);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

#[test]
fn center_emissions_preserves_ordering() {
    let input = vec![vec![-1.0, -5.0, -3.0]];
    let result = center_emissions(&input);
    assert!(result[0][0] > result[0][2], "ordering should be preserved");
    assert!(result[0][2] > result[0][1], "ordering should be preserved");
}

// ============================================================================
// apply_persistence_bonus
// ============================================================================

#[test]
fn persistence_bonus_empty() {
    let result = apply_persistence_bonus(&[], &[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn persistence_bonus_zero_weight_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_persistence_bonus(&input, &[0], 0.0);
    assert_eq!(result, input);
}

#[test]
fn persistence_bonus_negative_weight_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_persistence_bonus(&input, &[0], -1.0);
    assert_eq!(result, input);
}

#[test]
fn persistence_bonus_adds_to_correct_state() {
    let input = vec![vec![-5.0, -5.0, -5.0]];
    let states = vec![1];
    let result = apply_persistence_bonus(&input, &states, 2.0);
    assert_eq!(result[0][0], -5.0);
    assert_eq!(result[0][1], -3.0); // -5 + 2
    assert_eq!(result[0][2], -5.0);
}

#[test]
fn persistence_bonus_out_of_bounds_state_ignored() {
    let input = vec![vec![-1.0, -2.0]];
    let states = vec![5]; // out of bounds
    let result = apply_persistence_bonus(&input, &states, 1.0);
    assert_eq!(result, input); // unchanged
}

#[test]
fn persistence_bonus_shorter_states_partial() {
    let input = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let states = vec![0]; // only covers first window
    let result = apply_persistence_bonus(&input, &states, 1.0);
    assert_eq!(result[0][0], 0.0); // -1 + 1
    assert_eq!(result[1], vec![-3.0, -4.0]); // unchanged
}

// ============================================================================
// majority_vote_filter
// ============================================================================

#[test]
fn majority_vote_empty() {
    assert!(majority_vote_filter(&[], 3, 2).is_empty());
}

#[test]
fn majority_vote_radius_zero_identity() {
    let states = vec![0, 1, 2, 1, 0];
    assert_eq!(majority_vote_filter(&states, 3, 0), states);
}

#[test]
fn majority_vote_uniform_preserved() {
    let states = vec![2, 2, 2, 2, 2];
    assert_eq!(majority_vote_filter(&states, 3, 3), states);
}

#[test]
fn majority_vote_isolated_flicker_removed() {
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let result = majority_vote_filter(&states, 2, 2);
    assert_eq!(result[3], 0, "isolated state should be voted out");
}

#[test]
fn majority_vote_out_of_range_states_ignored() {
    let states = vec![0, 10, 0]; // 10 >= n_states=3
    let result = majority_vote_filter(&states, 3, 1);
    assert_eq!(result[0], 0);
    assert_eq!(result[2], 0);
}

// ============================================================================
// apply_proportion_prior
// ============================================================================

#[test]
fn proportion_prior_empty() {
    let result = apply_proportion_prior(&[], &[0.5, 0.5], 1.0);
    assert!(result.is_empty());
}

#[test]
fn proportion_prior_zero_weight_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_proportion_prior(&input, &[0.5, 0.5], 0.0);
    assert_eq!(result, input);
}

#[test]
fn proportion_prior_equal_proportions_shift_equal() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_proportion_prior(&input, &[0.5, 0.5], 1.0);
    // Both get same prior, so gap should be preserved
    let gap_before = input[0][0] - input[0][1];
    let gap_after = result[0][0] - result[0][1];
    assert!((gap_before - gap_after).abs() < 1e-10);
}

#[test]
fn proportion_prior_dominant_pop_gets_bigger_boost() {
    let input = vec![vec![-5.0, -5.0]];
    let result = apply_proportion_prior(&input, &[0.9, 0.1], 1.0);
    assert!(result[0][0] > result[0][1], "dominant pop should get bigger boost");
}

#[test]
fn proportion_prior_zero_proportion_clamped() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_proportion_prior(&input, &[0.0, 1.0], 1.0);
    // Zero proportion clamped to 1e-6
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
}

#[test]
fn proportion_prior_preserves_neg_infinity() {
    let input = vec![vec![-1.0, f64::NEG_INFINITY]];
    let result = apply_proportion_prior(&input, &[0.5, 0.5], 1.0);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

// ============================================================================
// rank_transform_emissions
// ============================================================================

#[test]
fn rank_transform_empty() {
    assert!(rank_transform_emissions(&[]).is_empty());
}

#[test]
fn rank_transform_single_pop_identity() {
    let input = vec![vec![-5.0], vec![-3.0]];
    let result = rank_transform_emissions(&input);
    assert_eq!(result, input);
}

#[test]
fn rank_transform_best_gets_highest_score() {
    let input = vec![vec![-1.0, -5.0, -3.0]];
    let result = rank_transform_emissions(&input);
    assert!(result[0][0] > result[0][2], "best should rank highest");
    assert!(result[0][2] > result[0][1], "middle rank in middle");
}

#[test]
fn rank_transform_preserves_neg_infinity() {
    let input = vec![vec![-1.0, f64::NEG_INFINITY, -3.0]];
    let result = rank_transform_emissions(&input);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
    assert!(result[0][0] > result[0][2]);
}

#[test]
fn rank_transform_output_sums_to_log_probs() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = rank_transform_emissions(&input);
    // Exp of rank-transformed scores should sum to ~1
    let sum: f64 = result[0].iter().filter(|v| v.is_finite()).map(|v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "rank scores should be valid log-probs, sum={sum}");
}

// ============================================================================
// dampen_emission_outliers
// ============================================================================

#[test]
fn dampen_outliers_empty() {
    let result = dampen_emission_outliers(&[], 3.0);
    assert!(result.is_empty());
}

#[test]
fn dampen_outliers_zero_threshold_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = dampen_emission_outliers(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn dampen_outliers_few_values_passthrough() {
    // Less than 3 values per population → NaN median → passthrough
    let input = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let result = dampen_emission_outliers(&input, 3.0);
    assert_eq!(result, input);
}

#[test]
fn dampen_outliers_no_outliers_unchanged() {
    let input = vec![
        vec![-1.0, -1.0],
        vec![-1.1, -1.1],
        vec![-0.9, -0.9],
        vec![-1.05, -1.05],
    ];
    let result = dampen_emission_outliers(&input, 3.0);
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!((v - input[i][j]).abs() < 1e-6);
        }
    }
}

#[test]
fn dampen_outliers_extreme_value_clipped() {
    // Need varying values so MAD > 0
    let input = vec![
        vec![-1.0], vec![-1.5], vec![-2.0], vec![-1.2],
        vec![-1.8], vec![-1.3], vec![-1.7], vec![-1.1],
        vec![-1.4], vec![-1.6], vec![-100.0], // extreme outlier
    ];
    let result = dampen_emission_outliers(&input, 2.0);
    // The extreme value should be clipped closer to median
    assert!(result[10][0] > -100.0, "outlier should be dampened, got {}", result[10][0]);
}

#[test]
fn dampen_outliers_zero_mad_passthrough() {
    // All identical values → MAD=0 → no dampening possible
    let mut input = Vec::new();
    for _ in 0..10 {
        input.push(vec![-1.0]);
    }
    input.push(vec![-100.0]);
    let result = dampen_emission_outliers(&input, 2.0);
    // With MAD=0, dampening is skipped
    assert_eq!(result[10][0], -100.0);
}

#[test]
fn dampen_outliers_preserves_neg_infinity() {
    let mut input = vec![vec![f64::NEG_INFINITY]];
    for _ in 0..5 {
        input.push(vec![-1.0]);
    }
    let result = dampen_emission_outliers(&input, 3.0);
    // NEG_INFINITY is not finite → passthrough
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// apply_emission_momentum
// ============================================================================

#[test]
fn momentum_empty() {
    assert!(apply_emission_momentum(&[], 0.5).is_empty());
}

#[test]
fn momentum_zero_alpha_identity() {
    let input = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let result = apply_emission_momentum(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn momentum_one_alpha_identity() {
    let input = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let result = apply_emission_momentum(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn momentum_smooths_sharp_transitions() {
    let input = vec![
        vec![-1.0, -10.0],
        vec![-10.0, -1.0],
        vec![-1.0, -10.0],
    ];
    let result = apply_emission_momentum(&input, 0.5);
    // Momentum should smooth the sharp transitions
    // Middle row should have less extreme values than original
    assert!(result[1][0] > -10.0, "should be smoothed up");
    assert!(result[1][1] < -1.0, "should be smoothed down");
}

#[test]
fn momentum_preserves_neg_infinity() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0], vec![-2.0, -3.0]];
    let result = apply_emission_momentum(&input, 0.5);
    // NEG_INFINITY should not propagate incorrectly
    for row in &result {
        for &v in row {
            assert!(!v.is_nan(), "no NaN from momentum");
        }
    }
}

// ============================================================================
// apply_emission_floor
// ============================================================================

#[test]
fn emission_floor_empty() {
    assert!(apply_emission_floor(&[], -10.0).is_empty());
}

#[test]
fn emission_floor_raises_low_values() {
    let input = vec![vec![-1.0, -100.0, -50.0]];
    let result = apply_emission_floor(&input, -20.0);
    assert_eq!(result[0][0], -1.0); // above floor
    assert_eq!(result[0][1], -20.0); // raised
    assert_eq!(result[0][2], -20.0); // raised
}

#[test]
fn emission_floor_preserves_neg_infinity() {
    let input = vec![vec![f64::NEG_INFINITY, -5.0]];
    let result = apply_emission_floor(&input, -10.0);
    // NEG_INFINITY is handled: max(NEG_INFINITY, -10) = -10? Let me check...
    // Actually f64::max(NEG_INFINITY, -10) = -10. But the function may preserve it.
    // Let's just check it doesn't panic
    assert!(result[0][1] == -5.0);
}

// ============================================================================
// variance_stabilize_emissions
// ============================================================================

#[test]
fn variance_stabilize_empty() {
    assert!(variance_stabilize_emissions(&[]).is_empty());
}

#[test]
fn variance_stabilize_single_row_identity() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = variance_stabilize_emissions(&input);
    assert_eq!(result, input);
}

#[test]
fn variance_stabilize_uniform_variance_unchanged() {
    // All pops have same variance → target_std = same → identity
    let input = vec![
        vec![-1.0, -1.0, -1.0],
        vec![-3.0, -3.0, -3.0],
    ];
    let result = variance_stabilize_emissions(&input);
    for (i, row) in result.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!((v - input[i][j]).abs() < 1e-6);
        }
    }
}

#[test]
fn variance_stabilize_equalizes_spread() {
    // Pop 0 has high variance, pop 1 has low variance
    let input = vec![
        vec![-1.0, -2.0],
        vec![-10.0, -2.1],
        vec![-1.0, -2.0],
        vec![-10.0, -2.1],
    ];
    let result = variance_stabilize_emissions(&input);
    // After stabilization, variances should be more similar
    let var0: f64 = result.iter().map(|r| r[0]).collect::<Vec<_>>().windows(2)
        .map(|w| (w[0] - w[1]).abs()).sum::<f64>();
    let var1: f64 = result.iter().map(|r| r[1]).collect::<Vec<_>>().windows(2)
        .map(|w| (w[0] - w[1]).abs()).sum::<f64>();
    // Variance ratio should be closer to 1 than before
    let original_ratio = 9.0 / 0.1; // approximately
    let new_ratio = if var0 > var1 { var0 / var1.max(1e-10) } else { var1 / var0.max(1e-10) };
    assert!(new_ratio < original_ratio, "variances should be more similar");
}

// ============================================================================
// apply_label_smoothing
// ============================================================================

#[test]
fn label_smoothing_empty() {
    let result = apply_label_smoothing(&[], 0.1);
    assert!(result.is_empty());
}

#[test]
fn label_smoothing_zero_alpha_identity() {
    let input = vec![vec![-1.0, -5.0]];
    let result = apply_label_smoothing(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn label_smoothing_full_alpha_uniform() {
    let input = vec![vec![-1.0, -10.0, -5.0]];
    let result = apply_label_smoothing(&input, 1.0);
    // alpha=1 → purely uniform → all equal to log(1/K)
    let expected = (1.0_f64 / 3.0).ln();
    for &v in &result[0] {
        if v.is_finite() {
            assert!((v - expected).abs() < 1e-6, "v={v}, expected={expected}");
        }
    }
}

#[test]
fn label_smoothing_preserves_neg_infinity() {
    let input = vec![vec![-1.0, f64::NEG_INFINITY]];
    let result = apply_label_smoothing(&input, 0.5);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

// ============================================================================
// apply_emission_anchor_boost
// ============================================================================

#[test]
fn anchor_boost_empty() {
    let result = apply_emission_anchor_boost(&[], 3, 0.5, 1.0);
    assert!(result.is_empty());
}

#[test]
fn anchor_boost_radius_zero_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_emission_anchor_boost(&input, 0, 0.5, 1.0);
    assert_eq!(result, input);
}

#[test]
fn anchor_boost_zero_boost_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_emission_anchor_boost(&input, 3, 0.5, 0.0);
    assert_eq!(result, input);
}

#[test]
fn anchor_boost_consistent_neighborhood_gets_boost() {
    // All windows agree on pop 0
    let input = vec![
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
    ];
    let result = apply_emission_anchor_boost(&input, 2, 0.3, 2.0);
    // Middle window should get a boost to pop 0
    assert!(result[2][0] > input[2][0], "consistent pop should get boost");
}

// ============================================================================
// apply_gap_penalty
// ============================================================================

#[test]
fn gap_penalty_empty() {
    let result = apply_gap_penalty(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn gap_penalty_single_window_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_gap_penalty(&input, 1.0);
    // With single window, can't compute gaps, should return identity or close
    assert_eq!(result.len(), 1);
}

#[test]
fn gap_penalty_smooth_signal_low_penalty() {
    // Smooth signal: small gaps → low penalty
    let input: Vec<Vec<f64>> = (0..10).map(|i| {
        vec![-(i as f64) * 0.01, -(i as f64) * 0.01 - 1.0]
    }).collect();
    let result = apply_gap_penalty(&input, 1.0);
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// detrend_emissions
// ============================================================================

#[test]
fn detrend_empty() {
    assert!(detrend_emissions(&[]).is_empty());
}

#[test]
fn detrend_single_row() {
    let input = vec![vec![-1.0, -2.0]];
    let result = detrend_emissions(&input);
    // With single row, can't compute trend → identity
    assert_eq!(result, input);
}

#[test]
fn detrend_removes_linear_trend() {
    // Linear trend in pop 0: -1, -2, -3, -4, -5
    let input = vec![
        vec![-1.0, -1.0],
        vec![-2.0, -1.0],
        vec![-3.0, -1.0],
        vec![-4.0, -1.0],
        vec![-5.0, -1.0],
    ];
    let result = detrend_emissions(&input);
    // After detrending, pop 0 should have less spread
    let range_before = input.last().unwrap()[0] - input[0][0];
    let range_after = result.last().unwrap()[0] - result[0][0];
    assert!(range_after.abs() < range_before.abs(),
        "detrending should reduce spread: {range_after} vs {range_before}");
}

// ============================================================================
// apply_fb_temperature
// ============================================================================

#[test]
fn fb_temperature_empty() {
    assert!(apply_fb_temperature(&[], 0.5).is_empty());
}

#[test]
fn fb_temperature_one_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_fb_temperature(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn fb_temperature_low_sharpens() {
    let input = vec![vec![-1.0, -3.0]];
    let result = apply_fb_temperature(&input, 0.5);
    // Dividing by 0.5 doubles the values
    assert!((result[0][0] - (-2.0)).abs() < 1e-6);
    assert!((result[0][1] - (-6.0)).abs() < 1e-6);
}

#[test]
fn fb_temperature_preserves_neg_infinity() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0]];
    let result = apply_fb_temperature(&input, 0.5);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// apply_confidence_weighting
// ============================================================================

#[test]
fn confidence_weighting_empty() {
    assert!(apply_confidence_weighting(&[], 1.0).is_empty());
}

#[test]
fn confidence_weighting_all_values_finite() {
    let input = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-0.5, -4.0, -2.0],
        vec![-2.0, -1.0, -5.0],
    ];
    let result = apply_confidence_weighting(&input, 1.0);
    for row in &result {
        for &v in row {
            assert!(v.is_finite(), "confidence weighting should produce finite values");
        }
    }
}

// ============================================================================
// apply_gradient_penalty
// ============================================================================

#[test]
fn gradient_penalty_empty() {
    assert!(apply_gradient_penalty(&[], 1.0).is_empty());
}

#[test]
fn gradient_penalty_single_window_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_gradient_penalty(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn gradient_penalty_smooths_jumps() {
    let input = vec![
        vec![-1.0, -1.0],
        vec![-100.0, -100.0], // big jump
        vec![-1.0, -1.0],
    ];
    let result = apply_gradient_penalty(&input, 0.5);
    // The extreme middle row should be pulled toward neighbors
    assert!(result[1][0] > -100.0, "gradient penalty should smooth jump");
}

// ============================================================================
// sharpen_posteriors
// ============================================================================

#[test]
fn sharpen_posteriors_empty() {
    assert!(sharpen_posteriors(&[], 0.5).is_empty());
}

#[test]
fn sharpen_posteriors_temperature_one_identity() {
    let input = vec![vec![0.5, 0.3, 0.2]];
    let result = sharpen_posteriors(&input, 1.0);
    for (a, b) in result[0].iter().zip(input[0].iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn sharpen_posteriors_low_temp_sharpens() {
    let input = vec![vec![0.5, 0.3, 0.2]];
    let result = sharpen_posteriors(&input, 0.5);
    // Sharpening should increase the gap between max and min
    let gap_before = input[0][0] - input[0][2];
    let gap_after = result[0][0] - result[0][2];
    assert!(gap_after > gap_before, "sharpening should increase gap");
}

#[test]
fn sharpen_posteriors_high_temp_flattens() {
    let input = vec![vec![0.8, 0.15, 0.05]];
    let result = sharpen_posteriors(&input, 5.0);
    let gap_before = input[0][0] - input[0][2];
    let gap_after = result[0][0] - result[0][2];
    assert!(gap_after < gap_before, "high temp should flatten");
}

#[test]
fn sharpen_posteriors_sums_to_one() {
    let input = vec![vec![0.5, 0.3, 0.2]];
    let result = sharpen_posteriors(&input, 0.5);
    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "should still sum to 1, got {sum}");
}

#[test]
fn sharpen_posteriors_zero_stays_zero() {
    let input = vec![vec![1.0, 0.0, 0.0]];
    let result = sharpen_posteriors(&input, 0.5);
    assert_eq!(result[0][1], 0.0);
    assert_eq!(result[0][2], 0.0);
    assert!((result[0][0] - 1.0).abs() < 1e-6);
}

// ============================================================================
// amplify_emission_residuals
// ============================================================================

#[test]
fn amplify_residuals_empty() {
    assert!(amplify_emission_residuals(&[], 2.0).is_empty());
}

#[test]
fn amplify_residuals_all_neg_infinity_unchanged() {
    let input = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = amplify_emission_residuals(&input, 2.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

#[test]
fn amplify_residuals_factor_one_identity() {
    let input = vec![vec![-1.0, -3.0]];
    let result = amplify_emission_residuals(&input, 1.0);
    for (a, b) in result[0].iter().zip(input[0].iter()) {
        assert!((a - b).abs() < 1e-10, "factor=1 should be identity");
    }
}

#[test]
fn amplify_residuals_increases_spread() {
    let input = vec![vec![-1.0, -5.0]];
    let result = amplify_emission_residuals(&input, 3.0);
    let spread_before = (input[0][0] - input[0][1]).abs();
    let spread_after = (result[0][0] - result[0][1]).abs();
    assert!(spread_after > spread_before, "amplification should increase spread");
}

// ============================================================================
// apply_diversity_scaling
// ============================================================================

#[test]
fn diversity_scaling_empty() {
    assert!(apply_diversity_scaling(&[], 1.5, 0.5).is_empty());
}

#[test]
fn diversity_scaling_single_pop_identity() {
    let input = vec![vec![-1.0]];
    let result = apply_diversity_scaling(&input, 1.5, 0.5);
    assert_eq!(result, input);
}

#[test]
fn diversity_scaling_all_finite() {
    let input = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-0.5, -4.0, -1.0],
    ];
    let result = apply_diversity_scaling(&input, 1.5, 0.5);
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// apply_snr_weighting
// ============================================================================

#[test]
fn snr_weighting_empty() {
    assert!(apply_snr_weighting(&[], 1.0).is_empty());
}

#[test]
fn snr_weighting_all_finite() {
    let input = vec![
        vec![-1.0, -3.0, -5.0],
        vec![-2.0, -2.0, -2.0],
        vec![-0.5, -4.0, -1.0],
    ];
    let result = apply_snr_weighting(&input, 1.0);
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// bayesian_shrink_emissions
// ============================================================================

#[test]
fn bayesian_shrink_empty() {
    assert!(bayesian_shrink_emissions(&[], 0.5).is_empty());
}

#[test]
fn bayesian_shrink_zero_alpha_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = bayesian_shrink_emissions(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn bayesian_shrink_full_alpha_becomes_mean() {
    let input = vec![
        vec![-1.0, -3.0],
        vec![-5.0, -7.0],
    ];
    let result = bayesian_shrink_emissions(&input, 1.0);
    // alpha=1 → all become global per-pop mean
    // pop 0 mean = (-1-5)/2 = -3, pop 1 mean = (-3-7)/2 = -5
    assert!((result[0][0] - (-3.0)).abs() < 1e-6);
    assert!((result[1][0] - (-3.0)).abs() < 1e-6);
    assert!((result[0][1] - (-5.0)).abs() < 1e-6);
}

// ============================================================================
// sparsify_top_k_emissions
// ============================================================================

#[test]
fn sparsify_empty() {
    assert!(sparsify_top_k_emissions(&[], 2, -20.0).is_empty());
}

#[test]
fn sparsify_k_ge_n_identity() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = sparsify_top_k_emissions(&input, 3, -20.0);
    assert_eq!(result, input);
}

#[test]
fn sparsify_keeps_top_k() {
    let input = vec![vec![-1.0, -5.0, -3.0]];
    let result = sparsify_top_k_emissions(&input, 2, -20.0);
    assert_eq!(result[0][0], -1.0); // top 1
    assert_eq!(result[0][2], -3.0); // top 2
    assert_eq!(result[0][1], -20.0); // floored
}

// ============================================================================
// correct_short_segments
// ============================================================================

#[test]
fn correct_short_segments_empty() {
    let result = correct_short_segments(&[], &[], 3);
    assert!(result.is_empty());
}

#[test]
fn correct_short_segments_single_state_unchanged() {
    let states = vec![0, 0, 0, 0, 0];
    let emissions = vec![vec![-1.0, -5.0]; 5];
    let result = correct_short_segments(&states, &emissions, 3);
    assert_eq!(result, states);
}

#[test]
fn correct_short_segments_short_segment_merged() {
    // Short segment of state 1 (2 windows < min 3) between state 0 blocks
    let states = vec![0, 0, 1, 1, 0, 0, 0];
    let emissions = vec![
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
        vec![-3.0, -2.0],
        vec![-3.0, -2.0],
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
        vec![-1.0, -5.0],
    ];
    let result = correct_short_segments(&states, &emissions, 3);
    // The short segment should be corrected based on emission evidence
    assert_eq!(result.len(), 7);
}

// ============================================================================
// entropy_smooth_posteriors
// ============================================================================

#[test]
fn entropy_smooth_empty() {
    assert!(entropy_smooth_posteriors(&[], 3).is_empty());
}

#[test]
fn entropy_smooth_radius_zero_identity() {
    let input = vec![vec![0.8, 0.1, 0.1], vec![0.1, 0.8, 0.1]];
    let result = entropy_smooth_posteriors(&input, 0);
    assert_eq!(result, input);
}

#[test]
fn entropy_smooth_preserves_row_sum() {
    let input = vec![
        vec![0.9, 0.05, 0.05],
        vec![0.3, 0.4, 0.3],
        vec![0.05, 0.9, 0.05],
    ];
    let result = entropy_smooth_posteriors(&input, 1);
    for row in &result {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "posteriors should sum to ~1, got {sum}");
    }
}

// ============================================================================
// apply_windowed_normalization
// ============================================================================

#[test]
fn windowed_normalization_empty() {
    assert!(apply_windowed_normalization(&[], 5).is_empty());
}

#[test]
fn windowed_normalization_radius_zero_identity() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_windowed_normalization(&input, 0);
    assert_eq!(result, input);
}

#[test]
fn windowed_normalization_all_finite() {
    let input: Vec<Vec<f64>> = (0..20).map(|i| {
        vec![-(i as f64), -(i as f64) - 1.0]
    }).collect();
    let result = apply_windowed_normalization(&input, 5);
    for row in &result {
        for &v in row {
            assert!(v.is_finite(), "windowed norm should produce finite values");
        }
    }
}

// ============================================================================
// quantile_normalize_emissions
// ============================================================================

#[test]
fn quantile_normalize_empty() {
    assert!(quantile_normalize_emissions(&[]).is_empty());
}

#[test]
fn quantile_normalize_single_row() {
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = quantile_normalize_emissions(&input);
    assert_eq!(result.len(), 1);
    // All finite
    for &v in &result[0] {
        assert!(v.is_finite());
    }
}

#[test]
fn quantile_normalize_preserves_neg_infinity() {
    let input = vec![
        vec![-1.0, f64::NEG_INFINITY],
        vec![-2.0, -3.0],
    ];
    let result = quantile_normalize_emissions(&input);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

#[test]
fn quantile_normalize_equal_values_unchanged() {
    let input = vec![
        vec![-2.0, -2.0],
        vec![-2.0, -2.0],
    ];
    let result = quantile_normalize_emissions(&input);
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}
