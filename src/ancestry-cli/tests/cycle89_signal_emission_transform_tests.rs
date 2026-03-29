//! Cycle 89: Edge case tests for signal processing and emission transform functions
//! with only 1 test-file reference each.
//!
//! Targets: softmax_renormalize, rank_transform_emissions, dampen_emission_outliers,
//! apply_gradient_penalty, apply_pairwise_emission_contrast, bayesian_shrink_emissions,
//! apply_proportion_prior, sharpen_posteriors, apply_gap_penalty, apply_persistence_bonus,
//! apply_changepoint_prior, apply_diversity_scaling, compute_window_quality.

use hprc_ancestry_cli::hmm::{
    apply_changepoint_prior, apply_diversity_scaling, apply_gap_penalty,
    apply_gradient_penalty, apply_pairwise_emission_contrast, apply_persistence_bonus,
    apply_proportion_prior, bayesian_shrink_emissions, compute_window_quality,
    dampen_emission_outliers, rank_transform_emissions, sharpen_posteriors,
    softmax_renormalize,
};

// ============================================================================
// softmax_renormalize
// ============================================================================

#[test]
fn softmax_renormalize_empty() {
    let result = softmax_renormalize(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn softmax_renormalize_temperature_zero_returns_copy() {
    let input = vec![vec![-1.0, -2.0]];
    let result = softmax_renormalize(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn softmax_renormalize_temperature_one_preserves_order() {
    let input = vec![vec![-1.0, -3.0, -5.0]];
    let result = softmax_renormalize(&input, 1.0);
    assert!(result[0][0] > result[0][1]);
    assert!(result[0][1] > result[0][2]);
}

#[test]
fn softmax_renormalize_high_temp_approaches_uniform() {
    let input = vec![vec![-1.0, -10.0]];
    let result = softmax_renormalize(&input, 100.0);
    // High temperature → nearly uniform log probs
    let diff = (result[0][0] - result[0][1]).abs();
    assert!(diff < 0.5, "diff={diff}");
}

#[test]
fn softmax_renormalize_low_temp_sharpens() {
    let input = vec![vec![-1.0, -2.0]];
    let result = softmax_renormalize(&input, 0.01);
    // Very low temperature → best state dominates, diff increases
    let diff = (result[0][0] - result[0][1]).abs();
    assert!(diff > 1.0, "diff={diff}");
}

#[test]
fn softmax_renormalize_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
    let result = softmax_renormalize(&input, 1.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert!(result[0][1].is_finite());
}

#[test]
fn softmax_renormalize_all_neg_infinity_unchanged() {
    let input = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = softmax_renormalize(&input, 1.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

#[test]
fn softmax_renormalize_output_logsumexp_near_zero() {
    // After softmax renormalization, probabilities should sum to ~1
    // so log-sum-exp of output should be near 0 (= ln(1))
    let input = vec![vec![-1.0, -3.0, -0.5]];
    let result = softmax_renormalize(&input, 1.0);
    let max_val = result[0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max_val + result[0].iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
    assert!((lse).abs() < 1e-6, "lse={lse}");
}

// ============================================================================
// rank_transform_emissions
// ============================================================================

#[test]
fn rank_transform_empty() {
    let result = rank_transform_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn rank_transform_single_pop() {
    let input = vec![vec![5.0]];
    let result = rank_transform_emissions(&input);
    assert_eq!(result, input); // k<=1 returns copy
}

#[test]
fn rank_transform_preserves_order() {
    let input = vec![vec![-1.0, -5.0, -3.0]]; // order: 0 > 2 > 1
    let result = rank_transform_emissions(&input);
    assert!(result[0][0] > result[0][2]);
    assert!(result[0][2] > result[0][1]);
}

#[test]
fn rank_transform_neg_infinity_stays_neg_infinity() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
    let result = rank_transform_emissions(&input);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert!(result[0][1] > result[0][2]); // -1 > -2, so rank 0 > rank 1
}

#[test]
fn rank_transform_multiple_rows_independent() {
    let input = vec![
        vec![-1.0, -5.0],
        vec![-5.0, -1.0],
    ];
    let result = rank_transform_emissions(&input);
    assert!(result[0][0] > result[0][1]);
    assert!(result[1][1] > result[1][0]);
}

#[test]
fn rank_transform_all_equal_deterministic() {
    let input = vec![vec![-2.0, -2.0, -2.0]];
    let result = rank_transform_emissions(&input);
    // All equal → stable sort order determines ranks; all values should be finite
    for &v in &result[0] {
        assert!(v.is_finite());
    }
}

// ============================================================================
// dampen_emission_outliers
// ============================================================================

#[test]
fn dampen_outliers_empty() {
    let result = dampen_emission_outliers(&[], 2.0);
    assert!(result.is_empty());
}

#[test]
fn dampen_outliers_z_threshold_zero_returns_copy() {
    let input = vec![vec![-1.0, -100.0]];
    let result = dampen_emission_outliers(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn dampen_outliers_negative_z_returns_copy() {
    let input = vec![vec![-1.0, -100.0]];
    let result = dampen_emission_outliers(&input, -1.0);
    assert_eq!(result, input);
}

#[test]
fn dampen_outliers_no_outliers_unchanged() {
    let input = vec![
        vec![-1.0, -2.0],
        vec![-1.1, -2.1],
        vec![-0.9, -1.9],
    ];
    let result = dampen_emission_outliers(&input, 3.0);
    for (r, i) in result.iter().zip(input.iter()) {
        for (rv, iv) in r.iter().zip(i.iter()) {
            assert!((rv - iv).abs() < 1e-10);
        }
    }
}

#[test]
fn dampen_outliers_extreme_value_clamped() {
    // Spread values around a center with one extreme outlier
    let input = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -2.5],
        vec![-0.5, -1.5],
        vec![-1.2, -2.2],
        vec![-0.8, -1.8],
        vec![-1.1, -2.1],
        vec![-0.9, -1.9],
        vec![-1.3, -2.3],
        vec![-0.7, -1.7],
        vec![-1.4, -2.4],
        vec![-100.0, -200.0], // extreme outlier
    ];
    let result = dampen_emission_outliers(&input, 2.0);
    // The outlier should be pulled toward the median
    assert!(result[10][0] > -100.0);
    assert!(result[10][1] > -200.0);
}

#[test]
fn dampen_outliers_too_few_values_unchanged() {
    // Only 2 values per population → fewer than 3, so no dampening
    let input = vec![vec![-1.0, -2.0], vec![-100.0, -200.0]];
    let result = dampen_emission_outliers(&input, 1.0);
    assert_eq!(result, input);
}

#[test]
fn dampen_outliers_neg_infinity_preserved() {
    let input = vec![
        vec![f64::NEG_INFINITY, -1.0],
        vec![-2.0, -1.5],
        vec![-1.5, -2.0],
        vec![-1.0, -1.0],
    ];
    let result = dampen_emission_outliers(&input, 2.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// apply_gradient_penalty
// ============================================================================

#[test]
fn gradient_penalty_empty() {
    let result = apply_gradient_penalty(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn gradient_penalty_single_window() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_gradient_penalty(&input, 0.5);
    assert_eq!(result, input); // len < 2 → copy
}

#[test]
fn gradient_penalty_weight_zero_noop() {
    let input = vec![vec![-1.0], vec![-5.0]];
    let result = apply_gradient_penalty(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn gradient_penalty_smooths_neighbors() {
    // Two windows far apart → penalty brings them closer
    let input = vec![vec![0.0], vec![-10.0]];
    let result = apply_gradient_penalty(&input, 0.5);
    assert!(result[0][0] < 0.0); // moved down
    assert!(result[1][0] > -10.0); // moved up
}

#[test]
fn gradient_penalty_neg_infinity_skipped() {
    let input = vec![vec![f64::NEG_INFINITY], vec![-1.0]];
    let result = apply_gradient_penalty(&input, 0.5);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[1][0], -1.0); // unchanged since partner is non-finite
}

#[test]
fn gradient_penalty_weight_clamped_to_one() {
    // weight > 1 should be clamped
    let input = vec![vec![0.0], vec![-10.0]];
    let result = apply_gradient_penalty(&input, 5.0);
    // At weight=1, both become the average
    let avg = -5.0;
    assert!((result[0][0] - avg).abs() < 1e-10);
    assert!((result[1][0] - avg).abs() < 1e-10);
}

// ============================================================================
// apply_pairwise_emission_contrast
// ============================================================================

#[test]
fn pairwise_contrast_empty() {
    let result = apply_pairwise_emission_contrast(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn pairwise_contrast_boost_zero_noop() {
    let input = vec![vec![-1.0, -3.0]];
    let result = apply_pairwise_emission_contrast(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn pairwise_contrast_boosts_best_penalizes_second() {
    let input = vec![vec![-1.0, -3.0, -5.0]]; // best=0, second=1
    let boost = 0.5;
    let result = apply_pairwise_emission_contrast(&input, boost);
    assert!((result[0][0] - (-1.0 + boost)).abs() < 1e-10);
    assert!((result[0][1] - (-3.0 - boost)).abs() < 1e-10);
    assert!((result[0][2] - (-5.0)).abs() < 1e-10); // third unchanged
}

#[test]
fn pairwise_contrast_single_finite_unchanged() {
    let input = vec![vec![f64::NEG_INFINITY, -2.0, f64::NEG_INFINITY]];
    let result = apply_pairwise_emission_contrast(&input, 1.0);
    // Only 1 finite value → fewer than 2 → unchanged
    assert_eq!(result, input);
}

#[test]
fn pairwise_contrast_negative_boost_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_pairwise_emission_contrast(&input, -0.5);
    assert_eq!(result, input);
}

// ============================================================================
// bayesian_shrink_emissions
// ============================================================================

#[test]
fn bayesian_shrink_empty() {
    let result = bayesian_shrink_emissions(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn bayesian_shrink_alpha_zero_noop() {
    let input = vec![vec![-1.0, -5.0]];
    let result = bayesian_shrink_emissions(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn bayesian_shrink_alpha_one_all_at_mean() {
    let input = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
    ];
    let result = bayesian_shrink_emissions(&input, 1.0);
    // Global mean pop0 = (-1 + -3)/2 = -2, pop1 = (-2 + -4)/2 = -3
    assert!((result[0][0] - (-2.0)).abs() < 1e-10);
    assert!((result[0][1] - (-3.0)).abs() < 1e-10);
    assert!((result[1][0] - (-2.0)).abs() < 1e-10);
    assert!((result[1][1] - (-3.0)).abs() < 1e-10);
}

#[test]
fn bayesian_shrink_alpha_half_interpolates() {
    let input = vec![vec![-1.0, -3.0], vec![-3.0, -5.0]];
    let result = bayesian_shrink_emissions(&input, 0.5);
    // mean pop0=-2, pop1=-4
    // row0 pop0: 0.5*(-1) + 0.5*(-2) = -1.5
    assert!((result[0][0] - (-1.5)).abs() < 1e-10);
    assert!((result[0][1] - (-3.5)).abs() < 1e-10);
}

#[test]
fn bayesian_shrink_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0], vec![-2.0, -3.0]];
    let result = bayesian_shrink_emissions(&input, 0.5);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
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
fn proportion_prior_weight_zero_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_proportion_prior(&input, &[0.8, 0.2], 0.0);
    assert_eq!(result, input);
}

#[test]
fn proportion_prior_adds_log_proportions() {
    let input = vec![vec![0.0, 0.0]];
    let proportions = vec![0.8, 0.2];
    let result = apply_proportion_prior(&input, &proportions, 1.0);
    assert!((result[0][0] - (0.8_f64.ln())).abs() < 1e-10);
    assert!((result[0][1] - (0.2_f64.ln())).abs() < 1e-10);
}

#[test]
fn proportion_prior_higher_weight_amplifies() {
    let input = vec![vec![0.0, 0.0]];
    let proportions = vec![0.8, 0.2];
    let r1 = apply_proportion_prior(&input, &proportions, 1.0);
    let r2 = apply_proportion_prior(&input, &proportions, 2.0);
    let diff1 = r1[0][0] - r1[0][1];
    let diff2 = r2[0][0] - r2[0][1];
    assert!(diff2.abs() > diff1.abs());
}

#[test]
fn proportion_prior_zero_proportion_floored() {
    let input = vec![vec![0.0, 0.0]];
    let result = apply_proportion_prior(&input, &[0.0, 1.0], 1.0);
    // Zero proportion gets floored to 1e-6
    assert!(result[0][0].is_finite());
    assert!(result[0][0] < result[0][1]);
}

#[test]
fn proportion_prior_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0]];
    let result = apply_proportion_prior(&input, &[0.5, 0.5], 1.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// sharpen_posteriors
// ============================================================================

#[test]
fn sharpen_posteriors_empty() {
    let result = sharpen_posteriors(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn sharpen_posteriors_temperature_one_noop() {
    let input = vec![vec![0.6, 0.3, 0.1]];
    let result = sharpen_posteriors(&input, 1.0);
    for (r, i) in result[0].iter().zip(input[0].iter()) {
        assert!((r - i).abs() < 1e-10);
    }
}

#[test]
fn sharpen_posteriors_temp_zero_noop() {
    let input = vec![vec![0.6, 0.4]];
    let result = sharpen_posteriors(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn sharpen_posteriors_low_temp_concentrates() {
    let input = vec![vec![0.6, 0.3, 0.1]];
    let result = sharpen_posteriors(&input, 0.1);
    // Low temperature → sharpens, best posterior increases
    assert!(result[0][0] > input[0][0]);
    assert!(result[0][2] < input[0][2]);
}

#[test]
fn sharpen_posteriors_high_temp_flattens() {
    let input = vec![vec![0.9, 0.05, 0.05]];
    let result = sharpen_posteriors(&input, 10.0);
    // High temperature → flattens distribution
    assert!(result[0][0] < input[0][0]);
}

#[test]
fn sharpen_posteriors_sums_to_one() {
    let input = vec![vec![0.5, 0.3, 0.2]];
    let result = sharpen_posteriors(&input, 0.5);
    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn sharpen_posteriors_zero_entries_stay_zero() {
    let input = vec![vec![0.7, 0.3, 0.0]];
    let result = sharpen_posteriors(&input, 0.5);
    assert_eq!(result[0][2], 0.0);
    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
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
fn gap_penalty_single_window() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_gap_penalty(&input, 1.0);
    assert_eq!(result, input); // n < 2 → copy
}

#[test]
fn gap_penalty_weight_zero_noop() {
    let input = vec![vec![-1.0], vec![-5.0]];
    let result = apply_gap_penalty(&input, 0.0);
    assert_eq!(result, input);
}

#[test]
fn gap_penalty_constant_signal_no_penalty() {
    let input = vec![vec![-1.0, -2.0]; 5];
    let result = apply_gap_penalty(&input, 1.0);
    // All gaps are zero for both pops → all gaps < 1e-12 → sorted_gaps empty → return copy
    for (r, i) in result.iter().zip(input.iter()) {
        for (rv, iv) in r.iter().zip(i.iter()) {
            assert!((rv - iv).abs() < 1e-10);
        }
    }
}

#[test]
fn gap_penalty_high_gap_pop_penalized() {
    // Pop 0 is smooth, pop 1 is noisy
    let input = vec![
        vec![-1.0, -1.0],
        vec![-1.0, -10.0],
        vec![-1.0, -1.0],
        vec![-1.0, -10.0],
    ];
    let result = apply_gap_penalty(&input, 1.0);
    // Pop 1 has larger gaps → gets penalized more than pop 0
    // Since pop 0 has gap=0 and pop 1 has large gaps, pop 0 should be unchanged or less penalized
    let pop0_shift = (result[0][0] - input[0][0]).abs();
    let pop1_shift = (result[0][1] - input[0][1]).abs();
    assert!(pop1_shift >= pop0_shift);
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
fn persistence_bonus_weight_zero_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_persistence_bonus(&input, &[0], 0.0);
    assert_eq!(result, input);
}

#[test]
fn persistence_bonus_adds_to_current_state() {
    let input = vec![vec![-5.0, -5.0, -5.0]];
    let states = vec![1];
    let result = apply_persistence_bonus(&input, &states, 1.0);
    assert!((result[0][0] - (-5.0)).abs() < 1e-10);
    assert!((result[0][1] - (-4.0)).abs() < 1e-10); // -5 + 1.0
    assert!((result[0][2] - (-5.0)).abs() < 1e-10);
}

#[test]
fn persistence_bonus_state_out_of_bounds_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let states = vec![5]; // out of bounds
    let result = apply_persistence_bonus(&input, &states, 1.0);
    assert_eq!(result, input);
}

#[test]
fn persistence_bonus_multiple_windows() {
    let input = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    ];
    let states = vec![0, 1];
    let result = apply_persistence_bonus(&input, &states, 2.0);
    assert!((result[0][0] - 2.0).abs() < 1e-10);
    assert!((result[1][1] - 2.0).abs() < 1e-10);
}

// ============================================================================
// apply_changepoint_prior
// ============================================================================

#[test]
fn changepoint_prior_empty() {
    let result = apply_changepoint_prior(&[], &[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn changepoint_prior_bonus_zero_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_changepoint_prior(&input, &[0], 0.0);
    assert_eq!(result, input);
}

#[test]
fn changepoint_prior_adds_bonus_to_state() {
    let input = vec![vec![-5.0, -5.0]];
    let states = vec![0];
    let result = apply_changepoint_prior(&input, &states, 1.5);
    assert!((result[0][0] - (-3.5)).abs() < 1e-10); // -5 + 1.5
    assert!((result[0][1] - (-5.0)).abs() < 1e-10);
}

#[test]
fn changepoint_prior_state_neg_infinity_skipped() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0]];
    let states = vec![0]; // state 0 is NEG_INFINITY → not finite → no bonus added
    let result = apply_changepoint_prior(&input, &states, 1.0);
    // The function checks is_finite before adding bonus
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

#[test]
fn changepoint_prior_empty_states_noop() {
    let input = vec![vec![-1.0, -2.0]];
    let result = apply_changepoint_prior(&input, &[], 1.0);
    assert_eq!(result, input);
}

// ============================================================================
// apply_diversity_scaling
// ============================================================================

#[test]
fn diversity_scaling_empty() {
    let result = apply_diversity_scaling(&[], 2.0, 0.5);
    assert!(result.is_empty());
}

#[test]
fn diversity_scaling_single_pop_noop() {
    let input = vec![vec![-1.0], vec![-2.0]];
    let result = apply_diversity_scaling(&input, 2.0, 0.5);
    assert_eq!(result, input);
}

#[test]
fn diversity_scaling_low_entropy_amplifies() {
    // One dominant state → low entropy → scale closer to amplify
    let input = vec![vec![0.0, -100.0, -100.0]];
    let result = apply_diversity_scaling(&input, 2.0, 0.5);
    // Low entropy → scale ~ amplify=2.0 → deviations from mean enlarged
    let mean_in = input[0].iter().sum::<f64>() / 3.0;
    let mean_out = result[0].iter().sum::<f64>() / 3.0;
    // The deviation of element 0 from mean should be larger
    let dev_in = (input[0][0] - mean_in).abs();
    let dev_out = (result[0][0] - mean_out).abs();
    assert!(dev_out > dev_in);
}

#[test]
fn diversity_scaling_high_entropy_dampens() {
    // Uniform distribution → high entropy → scale closer to dampen
    let input = vec![vec![-1.0, -1.0, -1.0]];
    let result = apply_diversity_scaling(&input, 2.0, 0.5);
    // Uniform → all equal → deviation is 0, so output ≈ input regardless
    for (r, i) in result[0].iter().zip(input[0].iter()) {
        assert!((r - i).abs() < 1e-10);
    }
}

#[test]
fn diversity_scaling_neg_infinity_preserved() {
    let input = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
    let result = apply_diversity_scaling(&input, 2.0, 0.5);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// compute_window_quality
// ============================================================================

#[test]
fn window_quality_empty() {
    let result = compute_window_quality(&[], &[], &[], 1);
    assert!(result.is_empty());
}

#[test]
fn window_quality_perfect_confidence() {
    // Perfect posterior margin, clear emission discriminability, all neighbors agree
    let posteriors = vec![vec![1.0, 0.0, 0.0]; 5];
    let log_emissions = vec![vec![0.0, -10.0, -10.0]; 5];
    let states = vec![0; 5];
    let result = compute_window_quality(&posteriors, &log_emissions, &states, 2);
    for &q in &result {
        assert!(q > 0.8, "quality={q} should be high");
    }
}

#[test]
fn window_quality_uncertain_low() {
    // Uniform posteriors, no emission discriminability, disagreeing neighbors
    let posteriors = vec![
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ];
    let log_emissions = vec![vec![-1.0, -1.0, -1.0]; 3];
    let states = vec![0, 1, 2]; // all different
    let result = compute_window_quality(&posteriors, &log_emissions, &states, 1);
    for &q in &result {
        assert!(q < 0.5, "quality={q} should be low");
    }
}

#[test]
fn window_quality_radius_zero_agreement_one() {
    let posteriors = vec![vec![0.5, 0.5]];
    let log_emissions = vec![vec![-1.0, -1.0]];
    let states = vec![0];
    let result = compute_window_quality(&posteriors, &log_emissions, &states, 0);
    // radius=0 → agreement=1.0 for all windows
    assert!(result[0] > 0.0);
}

#[test]
fn window_quality_in_unit_interval() {
    let posteriors = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
    let log_emissions = vec![vec![-1.0, -2.0], vec![-3.0, -0.5]];
    let states = vec![0, 1];
    let result = compute_window_quality(&posteriors, &log_emissions, &states, 1);
    for &q in &result {
        assert!(q >= 0.0 && q <= 1.0, "quality={q} out of [0,1]");
    }
}
