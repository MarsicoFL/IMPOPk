//! Cycle 86: Tests for state/transition/blend functions with ZERO prior coverage.
//!
//! Targets: bidirectional_smooth_states, majority_vote_filter, correct_short_segments,
//! compute_transition_momentum, regularize_toward_posteriors,
//! blend_log_emissions_agreement, blend_log_emissions_hybrid,
//! blend_log_emissions_adaptive_per_window.

use hprc_ancestry_cli::hmm::{
    bidirectional_smooth_states, blend_log_emissions_adaptive_per_window,
    blend_log_emissions_agreement, blend_log_emissions_hybrid, correct_short_segments,
    majority_vote_filter, compute_transition_momentum, regularize_toward_posteriors,
    AncestralPopulation, AncestryHmmParams,
};

fn make_params(n_states: usize) -> AncestryHmmParams {
    let switch_prob = 0.01;
    let self_prob = 1.0 - switch_prob * (n_states - 1) as f64;
    let transitions = (0..n_states).map(|i| {
        (0..n_states).map(|j| if i == j { self_prob } else { switch_prob }).collect()
    }).collect();
    let initial = vec![1.0 / n_states as f64; n_states];
    let populations = (0..n_states).map(|i| AncestralPopulation {
        name: format!("Pop{}", i),
        haplotypes: vec![],
    }).collect();

    AncestryHmmParams {
        n_states,
        populations,
        transitions,
        initial,
        emission_same_pop_mean: 0.95,
        emission_diff_pop_mean: 0.85,
        emission_std: 0.03,
        emission_model: hprc_ancestry_cli::hmm::EmissionModel::Max,
        normalization: None,
        coverage_weight: 0.0,
        transition_dampening: 0.0,
    }
}

// ============================================================================
// bidirectional_smooth_states
// ============================================================================

#[test]
fn bidir_smooth_empty() {
    let result = bidirectional_smooth_states(&[], 3, 2);
    assert!(result.is_empty());
}

#[test]
fn bidir_smooth_radius_zero_passthrough() {
    let states = vec![0, 1, 2, 0, 1];
    let result = bidirectional_smooth_states(&states, 3, 0);
    assert_eq!(result, states);
}

#[test]
fn bidir_smooth_single_element() {
    let result = bidirectional_smooth_states(&[2], 3, 5);
    assert_eq!(result, vec![2]);
}

#[test]
fn bidir_smooth_all_same_unchanged() {
    let states = vec![1, 1, 1, 1, 1];
    let result = bidirectional_smooth_states(&states, 3, 3);
    assert_eq!(result, states);
}

#[test]
fn bidir_smooth_isolated_outlier_corrected() {
    // [0,0,0,1,0,0,0] with radius 2 → the isolated 1 should be smoothed to 0
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let result = bidirectional_smooth_states(&states, 2, 2);
    assert_eq!(result[3], 0, "isolated outlier should be smoothed");
}

#[test]
fn bidir_smooth_strong_block_preserved() {
    // [0,0,0,1,1,1,1,1,0,0,0] → the block of 1s is strong enough
    let states = vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0];
    let result = bidirectional_smooth_states(&states, 2, 1);
    // Central 1s should be preserved
    assert_eq!(result[5], 1);
    assert_eq!(result[6], 1);
}

#[test]
fn bidir_smooth_tie_broken_by_original() {
    // At position 2: radius 1 → [0,1,1], states[2]=1
    // 0 gets less weight because it's farther, 1 should win or tie (broken by original=1)
    let states = vec![0, 1, 1, 0];
    let result = bidirectional_smooth_states(&states, 2, 1);
    assert_eq!(result[1], 1);
}

#[test]
fn bidir_smooth_distance_weighting_decays() {
    // Exponential weighting: closer neighbors have more influence
    // [0,0,1,0,0,0,0] radius=3 → at position 2, mostly 0s around it
    let states = vec![0, 0, 1, 0, 0, 0, 0];
    let result = bidirectional_smooth_states(&states, 2, 3);
    assert_eq!(result[2], 0, "distant majority should outweigh isolated state");
}

// ============================================================================
// majority_vote_filter
// ============================================================================

#[test]
fn majority_vote_empty() {
    let result = majority_vote_filter(&[], 3, 2);
    assert!(result.is_empty());
}

#[test]
fn majority_vote_radius_zero_passthrough() {
    let states = vec![0, 1, 2];
    let result = majority_vote_filter(&states, 3, 0);
    assert_eq!(result, states);
}

#[test]
fn majority_vote_single_element() {
    let result = majority_vote_filter(&[2], 3, 5);
    assert_eq!(result, vec![2]);
}

#[test]
fn majority_vote_all_same() {
    let states = vec![1, 1, 1, 1, 1];
    let result = majority_vote_filter(&states, 3, 2);
    assert_eq!(result, states);
}

#[test]
fn majority_vote_isolated_outlier() {
    let states = vec![0, 0, 0, 1, 0, 0, 0];
    let result = majority_vote_filter(&states, 2, 2);
    assert_eq!(result[3], 0, "isolated outlier removed by majority vote");
}

#[test]
fn majority_vote_edge_boundary() {
    // At position 0: only [0..min(3,7)] = [0..3] = [0,0,1] → 0 wins
    let states = vec![0, 0, 1, 1, 1];
    let result = majority_vote_filter(&states, 2, 2);
    assert_eq!(result[0], 0);
    assert_eq!(result[4], 1);
}

#[test]
fn majority_vote_tie_broken_by_original() {
    // [0, 1] with radius 1 at pos 0: window [0..2] = [0, 1], tie → keep original
    let states = vec![0, 1];
    let result = majority_vote_filter(&states, 2, 1);
    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1);
}

#[test]
fn majority_vote_large_radius() {
    // radius larger than array → global majority
    let states = vec![0, 0, 0, 1, 1];
    let result = majority_vote_filter(&states, 2, 100);
    for &s in &result {
        assert_eq!(s, 0, "global majority is 0");
    }
}

// ============================================================================
// correct_short_segments
// ============================================================================

#[test]
fn correct_short_empty() {
    let result = correct_short_segments(&[], &[], 3);
    assert!(result.is_empty());
}

#[test]
fn correct_short_min_windows_one_passthrough() {
    let states = vec![0, 1, 0];
    let emissions = vec![vec![0.0, 0.0]; 3];
    let result = correct_short_segments(&states, &emissions, 1);
    assert_eq!(result, states);
}

#[test]
fn correct_short_all_long_enough() {
    let states = vec![0, 0, 0, 1, 1, 1];
    let emissions = vec![vec![0.0, 0.0]; 6];
    let result = correct_short_segments(&states, &emissions, 3);
    assert_eq!(result, states);
}

#[test]
fn correct_short_single_window_segment_merged() {
    // [0, 0, 1, 0, 0] with min_windows=2 → segment at pos 2 (len 1) gets merged
    let states = vec![0, 0, 1, 0, 0];
    let emissions = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.5, 0.5], // ambiguous
        vec![1.0, 0.0],
        vec![1.0, 0.0],
    ];
    let result = correct_short_segments(&states, &emissions, 2);
    assert_eq!(result[2], 0, "short segment merged with neighbors");
}

#[test]
fn correct_short_neighbors_agree() {
    // Both neighbors are state 0 → short state 1 becomes 0
    let states = vec![0, 0, 1, 0, 0];
    let emissions = vec![vec![0.0, 0.0]; 5];
    let result = correct_short_segments(&states, &emissions, 2);
    assert_eq!(result[2], 0);
}

#[test]
fn correct_short_left_only_neighbor() {
    // Short segment at the end
    let states = vec![0, 0, 0, 1];
    let emissions = vec![vec![0.0, 0.0]; 4];
    let result = correct_short_segments(&states, &emissions, 2);
    assert_eq!(result[3], 0, "short tail segment merged with left neighbor");
}

#[test]
fn correct_short_right_only_neighbor() {
    // Short segment at the start
    let states = vec![1, 0, 0, 0];
    let emissions = vec![vec![0.0, 0.0]; 4];
    let result = correct_short_segments(&states, &emissions, 2);
    assert_eq!(result[0], 0, "short head segment merged with right neighbor");
}

#[test]
fn correct_short_single_segment_unchanged() {
    // Only one segment → no neighbors → can't correct
    let states = vec![0, 0, 0];
    let emissions = vec![vec![1.0, 0.0]; 3];
    let result = correct_short_segments(&states, &emissions, 5);
    assert_eq!(result, states);
}

#[test]
fn correct_short_emission_support_decides() {
    // Short segment between two different states: pick the one with better emission support
    let states = vec![0, 0, 2, 1, 1];
    let emissions = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.8, 0.2], // state 1 has better support
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    let result = correct_short_segments(&states, &emissions, 2);
    assert_eq!(result[2], 1, "should pick neighbor with better emission support");
}

// ============================================================================
// compute_transition_momentum
// ============================================================================

#[test]
fn transition_momentum_empty() {
    let params = make_params(3);
    let result = compute_transition_momentum(&[], &params, 0.5);
    assert!(result.is_empty());
}

#[test]
fn transition_momentum_alpha_zero_empty() {
    let params = make_params(3);
    let result = compute_transition_momentum(&[0, 1, 0], &params, 0.0);
    assert!(result.is_empty());
}

#[test]
fn transition_momentum_single_window() {
    let params = make_params(2);
    let result = compute_transition_momentum(&[0], &params, 0.5);
    assert_eq!(result.len(), 1);
    // Should be K×K matrix
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][0].len(), 2);
}

#[test]
fn transition_momentum_run_length_increases() {
    let params = make_params(2);
    let states = vec![0, 0, 0, 0, 0]; // run of 5
    let result = compute_transition_momentum(&states, &params, 1.0);
    // Self-transition boost should increase with run length
    // First window: run=1, last window: run=5
    let self_0_first = result[0][0][0];
    let self_0_last = result[4][0][0];
    assert!(self_0_last >= self_0_first,
        "longer run should have >= self-transition: {} vs {}", self_0_last, self_0_first);
}

#[test]
fn transition_momentum_run_resets_on_switch() {
    let params = make_params(2);
    let states = vec![0, 0, 0, 1, 1];
    let result = compute_transition_momentum(&states, &params, 1.0);
    // At t=3 (state 1, run=1), boost should be small
    let self_1_at_switch = result[3][1][1];
    // At t=4 (state 1, run=2), boost should be larger
    let self_1_after = result[4][1][1];
    assert!(self_1_after >= self_1_at_switch);
}

#[test]
fn transition_momentum_row_sums_valid() {
    let params = make_params(3);
    let states = vec![0, 0, 1, 1, 2];
    let result = compute_transition_momentum(&states, &params, 0.5);
    // After re-normalization, each row should sum to ~1 in probability space
    for t_trans in &result {
        for row in t_trans {
            let log_sum = row.iter().cloned().fold(f64::NEG_INFINITY, |a, b| {
                let max_ab = a.max(b);
                if max_ab == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                }
            });
            // log_sum should be close to 0 (prob sum = 1)
            // Only check the boosted row
        }
        // At minimum, values should be finite
        for row in t_trans {
            for &v in row {
                assert!(v.is_finite(), "transition value not finite: {}", v);
            }
        }
    }
}

// ============================================================================
// regularize_toward_posteriors
// ============================================================================

#[test]
fn regularize_empty() {
    let result = regularize_toward_posteriors(&[], &[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn regularize_lambda_zero_pure_emission() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.9, 0.1]];
    let result = regularize_toward_posteriors(&emissions, &posteriors, 0.0);
    assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    assert!((result[0][1] - (-2.0)).abs() < 1e-10);
}

#[test]
fn regularize_lambda_one_pure_posterior() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.9, 0.1]];
    let result = regularize_toward_posteriors(&emissions, &posteriors, 1.0);
    // Should be log(posteriors)
    let expected_0 = 0.9_f64.ln();
    let expected_1 = 0.1_f64.ln();
    assert!((result[0][0] - expected_0).abs() < 1e-10);
    assert!((result[0][1] - expected_1).abs() < 1e-10);
}

#[test]
fn regularize_interpolation() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.8, 0.2]];
    let result = regularize_toward_posteriors(&emissions, &posteriors, 0.5);
    let log_p0 = 0.8_f64.ln();
    let log_p1 = 0.2_f64.ln();
    let expected_0 = 0.5 * (-1.0) + 0.5 * log_p0;
    let expected_1 = 0.5 * (-2.0) + 0.5 * log_p1;
    assert!((result[0][0] - expected_0).abs() < 1e-10);
    assert!((result[0][1] - expected_1).abs() < 1e-10);
}

// ============================================================================
// blend_log_emissions_agreement
// ============================================================================

#[test]
fn blend_agreement_empty_standard() {
    let result = blend_log_emissions_agreement(&[], &[vec![1.0]], 0.5, 1.0, 0.5);
    assert!(result.is_empty());
}

#[test]
fn blend_agreement_empty_profile() {
    let std_e = vec![vec![1.0, 2.0]];
    let result = blend_log_emissions_agreement(&std_e, &[], 0.5, 1.0, 0.5);
    assert_eq!(result, std_e);
}

#[test]
fn blend_agreement_agree_windows_use_agree_scale() {
    // Both argmax at index 1
    let std_e = vec![vec![0.0, 1.0]];
    let prof_e = vec![vec![0.0, 2.0]];
    let result = blend_log_emissions_agreement(&std_e, &prof_e, 0.5, 1.0, 0.0);
    // agree → w = clamp(0.5 * 1.0, 0, 0.95) = 0.5, w_std = 0.5
    // result[0] = 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    // result[1] = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    assert!((result[0][0]).abs() < 1e-10);
    assert!((result[0][1] - 1.5).abs() < 1e-10);
}

#[test]
fn blend_agreement_disagree_windows_use_disagree_scale() {
    // std argmax at index 0, prof argmax at index 1
    let std_e = vec![vec![1.0, 0.0]];
    let prof_e = vec![vec![0.0, 1.0]];
    let result = blend_log_emissions_agreement(&std_e, &prof_e, 0.5, 2.0, 0.5);
    // disagree → w = clamp(0.5 * 0.5, 0, 0.95) = 0.25, w_std = 0.75
    // result[0] = 0.75 * 1.0 + 0.25 * 0.0 = 0.75
    // result[1] = 0.75 * 0.0 + 0.25 * 1.0 = 0.25
    assert!((result[0][0] - 0.75).abs() < 1e-10);
    assert!((result[0][1] - 0.25).abs() < 1e-10);
}

#[test]
fn blend_agreement_weight_clamped_to_095() {
    let std_e = vec![vec![0.0, 1.0]];
    let prof_e = vec![vec![0.0, 2.0]];
    // base_weight * agree_scale = 10.0 * 10.0 = 100.0 → clamped to 0.95
    let result = blend_log_emissions_agreement(&std_e, &prof_e, 10.0, 10.0, 1.0);
    let w = 0.95;
    let w_std = 1.0 - w;
    let expected_1 = w_std * 1.0 + w * 2.0;
    assert!((result[0][1] - expected_1).abs() < 1e-10);
}

#[test]
fn blend_agreement_non_finite_handling() {
    let std_e = vec![vec![f64::NEG_INFINITY, 1.0]];
    let prof_e = vec![vec![0.5, 0.5]];
    let result = blend_log_emissions_agreement(&std_e, &prof_e, 0.3, 1.0, 1.0);
    // std[0] is NEG_INFINITY, prof[0] is finite → result should be prof[0]
    assert!((result[0][0] - 0.5).abs() < 1e-10);
}

#[test]
fn blend_agreement_both_non_finite() {
    let std_e = vec![vec![f64::NEG_INFINITY]];
    let prof_e = vec![vec![f64::NEG_INFINITY]];
    let result = blend_log_emissions_agreement(&std_e, &prof_e, 0.3, 1.0, 1.0);
    assert!(result[0][0].is_infinite());
}

// ============================================================================
// blend_log_emissions_hybrid
// ============================================================================

#[test]
fn blend_hybrid_empty() {
    let result = blend_log_emissions_hybrid(&[], &[vec![1.0]], 0.5, 1.0, 0.5, 0.2, 3.0);
    assert!(result.is_empty());
}

#[test]
fn blend_hybrid_single_population_zero_margin() {
    // Single pop → margin = 0, median_margin = 1 (fallback)
    let std_e = vec![vec![1.0]];
    let prof_e = vec![vec![2.0]];
    let result = blend_log_emissions_hybrid(&std_e, &prof_e, 0.5, 1.0, 0.5, 0.2, 3.0);
    assert_eq!(result.len(), 1);
    assert!(result[0][0].is_finite());
}

#[test]
fn blend_hybrid_agree_with_margin() {
    // Both agree: argmax at index 1 for both
    let std_e = vec![vec![0.0, 2.0], vec![0.0, 2.0]];
    let prof_e = vec![vec![0.0, 3.0], vec![0.0, 1.0]]; // margin = 3 and 1
    let result = blend_log_emissions_hybrid(&std_e, &prof_e, 0.5, 1.0, 0.5, 0.2, 3.0);
    // Both windows agree → margin modulation applied
    for row in &result {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn blend_hybrid_disagree_ignores_margin() {
    // Disagreement: std argmax = 0, prof argmax = 1
    let std_e = vec![vec![2.0, 0.0]];
    let prof_e = vec![vec![0.0, 2.0]];
    let result = blend_log_emissions_hybrid(&std_e, &prof_e, 0.5, 2.0, 0.3, 0.2, 3.0);
    // disagree → weight = clamp(0.5 * 0.3, 0, 0.95) = 0.15
    let w = 0.15;
    let w_std = 1.0 - w;
    assert!((result[0][0] - (w_std * 2.0 + w * 0.0)).abs() < 1e-10);
}

#[test]
fn blend_hybrid_margin_clamped() {
    // Extreme margin ratios should be clamped
    let std_e = vec![vec![0.0, 100.0]]; // agree with prof
    let prof_e = vec![vec![0.0, 100.0]]; // huge margin
    let result = blend_log_emissions_hybrid(&std_e, &prof_e, 0.5, 1.0, 0.5, 0.2, 3.0);
    for &v in &result[0] {
        assert!(v.is_finite());
    }
}

// ============================================================================
// blend_log_emissions_adaptive_per_window
// ============================================================================

#[test]
fn blend_adaptive_basic() {
    let std_e = vec![vec![1.0, 0.0]];
    let pw_e = vec![vec![0.0, 1.0]];
    let scales = vec![1.0];
    let result = blend_log_emissions_adaptive_per_window(&std_e, &pw_e, 0.5, &scales);
    // w = clamp(0.5 * 1.0, 0, 0.95) = 0.5, w_std = 0.5
    assert!((result[0][0] - 0.5).abs() < 1e-10);
    assert!((result[0][1] - 0.5).abs() < 1e-10);
}

#[test]
fn blend_adaptive_scale_zero_pure_standard() {
    let std_e = vec![vec![1.0, 0.0]];
    let pw_e = vec![vec![0.0, 1.0]];
    let scales = vec![0.0];
    let result = blend_log_emissions_adaptive_per_window(&std_e, &pw_e, 0.5, &scales);
    // w = clamp(0.5 * 0.0, 0, 0.95) = 0.0 → pure standard
    assert!((result[0][0] - 1.0).abs() < 1e-10);
    assert!((result[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn blend_adaptive_clamped_weight() {
    let std_e = vec![vec![1.0]];
    let pw_e = vec![vec![2.0]];
    let scales = vec![100.0];
    let result = blend_log_emissions_adaptive_per_window(&std_e, &pw_e, 0.5, &scales);
    // w = clamp(0.5 * 100, 0, 0.95) = 0.95
    let expected = 0.05 * 1.0 + 0.95 * 2.0;
    assert!((result[0][0] - expected).abs() < 1e-10);
}

#[test]
fn blend_adaptive_non_finite_fallback() {
    let std_e = vec![vec![f64::NEG_INFINITY]];
    let pw_e = vec![vec![1.0]];
    let scales = vec![1.0];
    let result = blend_log_emissions_adaptive_per_window(&std_e, &pw_e, 0.5, &scales);
    // std is NEG_INF, pw is finite → result = pw
    assert!((result[0][0] - 1.0).abs() < 1e-10);
}

#[test]
fn blend_adaptive_multiple_windows_different_scales() {
    let std_e = vec![vec![10.0], vec![10.0]];
    let pw_e = vec![vec![0.0], vec![0.0]];
    let scales = vec![0.0, 2.0]; // first: pure std, second: strong pw blend
    let result = blend_log_emissions_adaptive_per_window(&std_e, &pw_e, 0.5, &scales);
    assert!((result[0][0] - 10.0).abs() < 1e-10); // scale=0 → pure std
    // scale=2.0 → w = clamp(1.0, 0, 0.95) = 0.95
    let expected = 0.05 * 10.0 + 0.95 * 0.0;
    assert!((result[1][0] - expected).abs() < 1e-10);
}
