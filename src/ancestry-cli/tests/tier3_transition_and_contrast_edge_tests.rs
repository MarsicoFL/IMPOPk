//! Edge case tests for untested ancestry-cli transition & contrast functions:
//! - entropy_weighted_smooth_posteriors (entropy-based posterior smoothing)
//! - apply_pairwise_emission_contrast (top-2 emission boosting)
//! - adjust_pop_temperatures (pass-2 posterior-guided temp adjustment)
//! - regularize_toward_posteriors (wrapper around blend_posteriors_with_emissions)
//! - compute_transition_momentum (run-length-based self-transition boost)
//! - compute_segment_length_prior (min-segment-length switching penalty)
//! - compute_recency_transitions (exponentially decayed state histogram boost)
//! - compute_disagreement_transitions (emission-vs-state agreement/disagreement)
//!
//! Focus: empty inputs, zero/negative params, single-element, NaN/Inf, normalization.

use hprc_ancestry_cli::hmm::{
    entropy_weighted_smooth_posteriors, apply_pairwise_emission_contrast,
    adjust_pop_temperatures, regularize_toward_posteriors,
    compute_transition_momentum, compute_segment_length_prior,
    compute_recency_transitions, compute_disagreement_transitions,
    AncestralPopulation, AncestryHmmParams,
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

// ===========================================================================
// entropy_weighted_smooth_posteriors
// ===========================================================================

#[test]
fn ewsp_empty_posteriors() {
    let result = entropy_weighted_smooth_posteriors(&[], 3, 1.0);
    assert!(result.is_empty());
}

#[test]
fn ewsp_radius_zero_returns_copy() {
    let posteriors = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 0, 1.0);
    assert_eq!(result, posteriors);
}

#[test]
fn ewsp_single_window() {
    let posteriors = vec![vec![0.7, 0.3]];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 5, 1.0);
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - 0.7).abs() < 1e-9);
    assert!((result[0][1] - 0.3).abs() < 1e-9);
}

#[test]
fn ewsp_uniform_posteriors_max_entropy() {
    // All posteriors uniform → all entropy maximal → all weights equal
    let posteriors = vec![
        vec![0.5, 0.5],
        vec![0.5, 0.5],
        vec![0.5, 0.5],
    ];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 1.0);
    for row in &result {
        assert!((row[0] - 0.5).abs() < 1e-9);
        assert!((row[1] - 0.5).abs() < 1e-9);
    }
}

#[test]
fn ewsp_pure_posteriors_zero_entropy() {
    // All posteriors [1.0, 0.0] → entropy = 0 → h_max = 0 → returns copy
    let posteriors = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
    ];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 1.0);
    assert_eq!(result, posteriors);
}

#[test]
fn ewsp_beta_zero_equal_weights() {
    // beta=0 → exp(0) = 1 for all → uniform weighting
    let posteriors = vec![
        vec![0.9, 0.1],
        vec![0.1, 0.9],
        vec![0.5, 0.5],
    ];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 10, 0.0);
    // With uniform weights and radius=10 (covers all), each output = mean of all posteriors
    let mean0 = (0.9 + 0.1 + 0.5) / 3.0;
    let mean1 = (0.1 + 0.9 + 0.5) / 3.0;
    for row in &result {
        assert!((row[0] - mean0).abs() < 1e-9);
        assert!((row[1] - mean1).abs() < 1e-9);
    }
}

#[test]
fn ewsp_large_beta_confident_windows_dominate() {
    // Large beta → low-entropy windows get much higher weight
    let posteriors = vec![
        vec![0.99, 0.01], // very confident
        vec![0.5, 0.5],   // uncertain
        vec![0.01, 0.99], // very confident
    ];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 100.0);
    // Middle window should be pulled toward neighbors more than it pulls them
    assert!(result.len() == 3);
    // All rows should still sum to ~1
    for row in &result {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "row sum = {}", sum);
    }
}

#[test]
fn ewsp_negative_beta_inverts_weighting() {
    // Negative beta → high-entropy windows get *more* weight (inverted)
    let posteriors = vec![
        vec![0.99, 0.01],
        vec![0.5, 0.5],
    ];
    let result = entropy_weighted_smooth_posteriors(&posteriors, 1, -1.0);
    assert_eq!(result.len(), 2);
    // Should not panic
}

// ===========================================================================
// apply_pairwise_emission_contrast
// ===========================================================================

#[test]
fn pec_empty() {
    let result = apply_pairwise_emission_contrast(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn pec_boost_zero_returns_copy() {
    let log_e = vec![vec![-1.0, -2.0, -3.0]];
    let result = apply_pairwise_emission_contrast(&log_e, 0.0);
    assert_eq!(result, log_e);
}

#[test]
fn pec_negative_boost_returns_copy() {
    let log_e = vec![vec![-1.0, -2.0, -3.0]];
    let result = apply_pairwise_emission_contrast(&log_e, -0.5);
    assert_eq!(result, log_e);
}

#[test]
fn pec_single_finite_value_unchanged() {
    // Only one finite value → can't find top-2 → returns row unchanged
    let log_e = vec![vec![f64::NEG_INFINITY, -2.0, f64::NEG_INFINITY]];
    let result = apply_pairwise_emission_contrast(&log_e, 1.0);
    assert_eq!(result[0], log_e[0]);
}

#[test]
fn pec_boosts_best_penalizes_second() {
    let log_e = vec![vec![-1.0, -3.0, -5.0]];
    let result = apply_pairwise_emission_contrast(&log_e, 0.5);
    assert!((result[0][0] - (-1.0 + 0.5)).abs() < 1e-9); // best boosted
    assert!((result[0][1] - (-3.0 - 0.5)).abs() < 1e-9); // second penalized
    assert!((result[0][2] - (-5.0)).abs() < 1e-9); // third unchanged
}

#[test]
fn pec_all_neg_infinity_unchanged() {
    let log_e = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let result = apply_pairwise_emission_contrast(&log_e, 1.0);
    assert!(result[0].iter().all(|v| *v == f64::NEG_INFINITY));
}

#[test]
fn pec_two_populations_both_adjusted() {
    let log_e = vec![vec![-2.0, -4.0]];
    let result = apply_pairwise_emission_contrast(&log_e, 1.0);
    assert!((result[0][0] - (-2.0 + 1.0)).abs() < 1e-9);
    assert!((result[0][1] - (-4.0 - 1.0)).abs() < 1e-9);
}

#[test]
fn pec_tied_values_deterministic() {
    // Two tied values → sort is stable, result should be deterministic
    let log_e = vec![vec![-2.0, -2.0, -5.0]];
    let result = apply_pairwise_emission_contrast(&log_e, 0.3);
    // One of the -2.0 should be boosted, the other penalized
    let boosted_count = result[0].iter().filter(|&&v| (v - (-2.0 + 0.3)).abs() < 1e-9).count();
    let penalized_count = result[0].iter().filter(|&&v| (v - (-2.0 - 0.3)).abs() < 1e-9).count();
    assert_eq!(boosted_count, 1);
    assert_eq!(penalized_count, 1);
}

#[test]
fn pec_multiple_windows_independent() {
    let log_e = vec![
        vec![-1.0, -3.0],
        vec![-5.0, -2.0],
    ];
    let result = apply_pairwise_emission_contrast(&log_e, 1.0);
    // Window 0: best=-1.0 (idx 0), second=-3.0 (idx 1)
    assert!((result[0][0] - 0.0).abs() < 1e-9);
    assert!((result[0][1] - (-4.0)).abs() < 1e-9);
    // Window 1: best=-2.0 (idx 1), second=-5.0 (idx 0)
    assert!((result[1][1] - (-1.0)).abs() < 1e-9);
    assert!((result[1][0] - (-6.0)).abs() < 1e-9);
}

// ===========================================================================
// adjust_pop_temperatures
// ===========================================================================

#[test]
fn apt_empty_emissions() {
    let result = adjust_pop_temperatures(&[], &[vec![0.5, 0.5]], 1.0);
    assert!(result.is_empty());
}

#[test]
fn apt_empty_posteriors() {
    let log_e = vec![vec![-1.0, -2.0]];
    let result = adjust_pop_temperatures(&log_e, &[], 1.0);
    assert_eq!(result, log_e);
}

#[test]
fn apt_factor_zero_returns_copy() {
    let log_e = vec![vec![-1.0, -2.0]];
    let result = adjust_pop_temperatures(&log_e, &[vec![0.5, 0.5]], 0.0);
    assert_eq!(result, log_e);
}

#[test]
fn apt_negative_factor_returns_copy() {
    let log_e = vec![vec![-1.0, -2.0]];
    let result = adjust_pop_temperatures(&log_e, &[vec![0.5, 0.5]], -1.0);
    assert_eq!(result, log_e);
}

#[test]
fn apt_uniform_posteriors_no_change() {
    // mean_post = [0.5, 0.5], threshold = 1/2 = 0.5 → scale = 1.0 for all
    let log_e = vec![vec![-1.0, -3.0]];
    let posteriors = vec![vec![0.5, 0.5]];
    let result = adjust_pop_temperatures(&log_e, &posteriors, 1.0);
    // scale = 1 + 1*(0.5 - 0.5) = 1.0 → output = mean + (v-mean)*1.0 = v
    assert!((result[0][0] - (-1.0)).abs() < 1e-9);
    assert!((result[0][1] - (-3.0)).abs() < 1e-9);
}

#[test]
fn apt_confident_pop_dampened_uncertain_amplified() {
    // Pop0 has high posterior → dampened; pop1 low → amplified
    let log_e = vec![vec![-1.0, -3.0]];
    let posteriors = vec![vec![0.8, 0.2]];
    let result = adjust_pop_temperatures(&log_e, &posteriors, 1.0);
    let mean = (-1.0 + -3.0) / 2.0; // -2.0
    // Pop0 scale = 1 + 1*(0.5 - 0.8) = 0.7
    // Pop1 scale = 1 + 1*(0.5 - 0.2) = 1.3
    let expected_0 = mean + (-1.0 - mean) * 0.7;
    let expected_1 = mean + (-3.0 - mean) * 1.3;
    assert!((result[0][0] - expected_0).abs() < 1e-9);
    assert!((result[0][1] - expected_1).abs() < 1e-9);
}

#[test]
fn apt_zero_k_returns_copy() {
    let log_e: Vec<Vec<f64>> = vec![vec![]];
    let posteriors = vec![vec![]];
    let result = adjust_pop_temperatures(&log_e, &posteriors, 1.0);
    assert_eq!(result, log_e);
}

#[test]
fn apt_neg_infinity_preserved() {
    let log_e = vec![vec![f64::NEG_INFINITY, -2.0]];
    let posteriors = vec![vec![0.5, 0.5]];
    let result = adjust_pop_temperatures(&log_e, &posteriors, 1.0);
    assert!(result[0][0] == f64::NEG_INFINITY);
}

#[test]
fn apt_scale_floor_at_0_1() {
    // Very confident pop → scale would go negative without floor
    let log_e = vec![vec![-1.0, -3.0]];
    let posteriors = vec![vec![0.99, 0.01]]; // threshold = 0.5
    let result = adjust_pop_temperatures(&log_e, &posteriors, 100.0);
    // Pop0: scale = 1 + 100*(0.5-0.99) = 1 - 49 = -48 → clamped to 0.1
    // Pop1: scale = 1 + 100*(0.5-0.01) = 50.0
    let mean = -2.0;
    let expected_0 = mean + (-1.0 - mean) * 0.1;
    assert!((result[0][0] - expected_0).abs() < 1e-9, "got {}", result[0][0]);
}

// ===========================================================================
// regularize_toward_posteriors (delegates to blend_posteriors_with_emissions)
// ===========================================================================

#[test]
fn rtp_empty() {
    let result = regularize_toward_posteriors(&[], &[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn rtp_lambda_zero_returns_emissions() {
    let log_e = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.9, 0.1]];
    let result = regularize_toward_posteriors(&log_e, &posteriors, 0.0);
    assert!((result[0][0] - (-1.0)).abs() < 1e-9);
    assert!((result[0][1] - (-2.0)).abs() < 1e-9);
}

#[test]
fn rtp_lambda_one_uses_posteriors() {
    let log_e = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.9, 0.1]];
    let result = regularize_toward_posteriors(&log_e, &posteriors, 1.0);
    // lambda=1 → pure log-posterior
    assert!((result[0][0] - 0.9_f64.ln()).abs() < 1e-6);
}

// ===========================================================================
// compute_transition_momentum
// ===========================================================================

#[test]
fn tm_empty_states() {
    let params = make_params(3, 0.01);
    let result = compute_transition_momentum(&[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn tm_alpha_zero_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_transition_momentum(&[0, 0, 1], &params, 0.0);
    assert!(result.is_empty());
}

#[test]
fn tm_negative_alpha_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_transition_momentum(&[0, 0, 1], &params, -1.0);
    assert!(result.is_empty());
}

#[test]
fn tm_single_window() {
    let params = make_params(3, 0.01);
    let result = compute_transition_momentum(&[0], &params, 1.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 3); // k×k matrix
    assert_eq!(result[0][0].len(), 3);
}

#[test]
fn tm_longer_run_stronger_boost() {
    let params = make_params(2, 0.01);
    // State 0 repeated → increasing run length → increasing momentum
    let result = compute_transition_momentum(&[0, 0, 0, 0, 0], &params, 1.0);
    // Self-transition for state 0 should increase over time
    let self_trans: Vec<f64> = result.iter().map(|m| m[0][0]).collect();
    for i in 1..self_trans.len() {
        assert!(self_trans[i] >= self_trans[i - 1] - 1e-9,
            "self-trans should increase with run length: {} < {}", self_trans[i], self_trans[i-1]);
    }
}

#[test]
fn tm_state_switch_resets_run() {
    let params = make_params(2, 0.01);
    let result = compute_transition_momentum(&[0, 0, 0, 1, 1], &params, 1.0);
    // After switch to state 1 at t=3, run_length resets to 1
    // State 1's self-transition at t=3 should be smaller than at t=4
    assert!(result[4][1][1] > result[3][1][1] - 1e-9);
}

#[test]
fn tm_rows_normalized_in_log_space() {
    let params = make_params(3, 0.01);
    let result = compute_transition_momentum(&[0, 0, 1, 2, 0], &params, 2.0);
    for (t, matrix) in result.iter().enumerate() {
        for (row_idx, row) in matrix.iter().enumerate() {
            // exp-sum of log-transitions should be ~1.0 for the modified row
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01,
                "t={} row {} not normalized: sum={}", t, row_idx, sum);
        }
    }
}

#[test]
fn tm_out_of_range_state_no_panic() {
    let params = make_params(2, 0.01);
    // State 5 is out of range (k=2)
    let result = compute_transition_momentum(&[5, 0, 1], &params, 1.0);
    assert_eq!(result.len(), 3);
}

// ===========================================================================
// compute_segment_length_prior
// ===========================================================================

#[test]
fn slp_empty_states() {
    let params = make_params(3, 0.01);
    let result = compute_segment_length_prior(&[], &params, 5);
    assert!(result.is_empty());
}

#[test]
fn slp_min_length_one_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_segment_length_prior(&[0, 0, 1], &params, 1);
    assert!(result.is_empty());
}

#[test]
fn slp_min_length_zero_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_segment_length_prior(&[0, 0, 1], &params, 0);
    assert!(result.is_empty());
}

#[test]
fn slp_short_run_penalizes_switching() {
    let params = make_params(2, 0.5);
    // Single window of state 0 → run_length=1 < min_length=3
    let result = compute_segment_length_prior(&[0], &params, 3);
    assert_eq!(result.len(), 1);
    // Off-diagonal (0→1) should be penalized relative to base
    let base_log_01 = (0.5_f64).max(1e-20).ln();
    assert!(result[0][0][1] < base_log_01, "off-diag should be penalized");
}

#[test]
fn slp_long_run_no_penalty() {
    let params = make_params(2, 0.01);
    // 5 consecutive state 0 → run_length=5 >= min_length=3 → no penalty
    let result = compute_segment_length_prior(&[0, 0, 0, 0, 0], &params, 3);
    let base_log = params.transitions[0].iter().map(|&p| p.max(1e-20).ln()).collect::<Vec<_>>();
    // At t=4 (run=5 >= 3), row should be identical to base
    for j in 0..2 {
        assert!((result[4][0][j] - base_log[j]).abs() < 1e-9,
            "long run should not be penalized: {} vs {}", result[4][0][j], base_log[j]);
    }
}

#[test]
fn slp_penalty_decreases_with_run_length() {
    let params = make_params(2, 0.5);
    // Run of state 0: run lengths 1, 2, 3 (all < min_length=4)
    let result = compute_segment_length_prior(&[0, 0, 0], &params, 4);
    // Penalty at t=0 (run=1) > penalty at t=1 (run=2) > penalty at t=2 (run=3)
    // Higher penalty → lower off-diagonal value
    assert!(result[0][0][1] < result[1][0][1],
        "shorter run should have stronger penalty");
    assert!(result[1][0][1] < result[2][0][1],
        "shorter run should have stronger penalty");
}

#[test]
fn slp_rows_normalized() {
    let params = make_params(3, 0.01);
    let result = compute_segment_length_prior(&[0, 0, 1, 2, 0], &params, 3);
    for matrix in &result {
        for row in matrix {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01, "row not normalized: sum={}", sum);
        }
    }
}

// ===========================================================================
// compute_recency_transitions
// ===========================================================================

#[test]
fn rt_empty_states() {
    let params = make_params(3, 0.01);
    let result = compute_recency_transitions(&[], &params, 0.9);
    assert!(result.is_empty());
}

#[test]
fn rt_alpha_zero_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_recency_transitions(&[0, 0, 1], &params, 0.0);
    assert!(result.is_empty());
}

#[test]
fn rt_negative_alpha_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_recency_transitions(&[0, 1, 2], &params, -0.5);
    assert!(result.is_empty());
}

#[test]
fn rt_single_state_repeated() {
    let params = make_params(2, 0.01);
    let result = compute_recency_transitions(&[0, 0, 0], &params, 0.9);
    assert_eq!(result.len(), 3);
    // State 0 accumulates weight → self-transition should grow
    let self_trans: Vec<f64> = result.iter().map(|m| m[0][0]).collect();
    // First window has less history than third
    assert!(self_trans[2] >= self_trans[0] - 1e-9);
}

#[test]
fn rt_alternating_states_balanced() {
    let params = make_params(2, 0.5);
    // Both states appear equally → histogram should be roughly balanced
    let result = compute_recency_transitions(&[0, 1, 0, 1, 0, 1], &params, 0.9);
    assert_eq!(result.len(), 6);
    // At last window, both states should have similar self-transition boosts
}

#[test]
fn rt_out_of_range_state_safe() {
    let params = make_params(2, 0.01);
    let result = compute_recency_transitions(&[5, 0, 1], &params, 0.9);
    assert_eq!(result.len(), 3);
}

#[test]
fn rt_rows_normalized() {
    let params = make_params(3, 0.01);
    let result = compute_recency_transitions(&[0, 1, 2, 0, 1], &params, 0.9);
    for matrix in &result {
        for row in matrix {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01, "row not normalized: sum={}", sum);
        }
    }
}

#[test]
fn rt_alpha_near_one_long_memory() {
    let params = make_params(2, 0.01);
    let result = compute_recency_transitions(&[0, 0, 0, 1, 1, 1], &params, 0.99);
    assert_eq!(result.len(), 6);
    // At t=5 state 1 has recent weight but state 0 also retains some from early windows
}

// ===========================================================================
// compute_disagreement_transitions
// ===========================================================================

#[test]
fn dt_empty_emissions() {
    let params = make_params(3, 0.01);
    let result = compute_disagreement_transitions(&[], &[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn dt_weight_zero_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_disagreement_transitions(
        &[vec![-1.0, -2.0, -3.0]], &[0], &params, 0.0);
    assert!(result.is_empty());
}

#[test]
fn dt_negative_weight_returns_empty() {
    let params = make_params(3, 0.01);
    let result = compute_disagreement_transitions(
        &[vec![-1.0, -2.0, -3.0]], &[0], &params, -1.0);
    assert!(result.is_empty());
}

#[test]
fn dt_agreement_boosts_self_transition() {
    let params = make_params(3, 0.01);
    // Emission argmax = 0, current state = 0 → agreement
    let log_e = vec![vec![-1.0, -3.0, -5.0]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);
    let base_self = params.transitions[0][0].max(1e-20).ln();
    // After normalization, self-transition should still be boosted relative to others
    assert!(result[0][0][0] > result[0][0][1], "agreement should boost self-transition");
}

#[test]
fn dt_disagreement_boosts_target_transition() {
    let params = make_params(3, 0.01);
    // Emission argmax = 1, current state = 0 → disagreement
    let log_e = vec![vec![-5.0, -1.0, -3.0]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&log_e, &states, &params, 2.0);
    // Transition from 0→1 should be boosted
    let base_01 = params.transitions[0][1].max(1e-20).ln();
    // After normalization it's hard to check absolute value, but 0→1 should be larger than 0→2
    assert!(result[0][0][1] > result[0][0][2], "disagreement should boost target state");
}

#[test]
fn dt_states_shorter_than_emissions() {
    let params = make_params(2, 0.01);
    let log_e = vec![vec![-1.0, -2.0], vec![-2.0, -1.0], vec![-1.5, -1.5]];
    let states = vec![0]; // only 1 state for 3 windows
    let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);
    assert_eq!(result.len(), 3); // produces output for all emission windows
}

#[test]
fn dt_all_neg_infinity_emissions() {
    let params = make_params(2, 0.01);
    let log_e = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);
    assert_eq!(result.len(), 1);
    // emission_argmax falls back to current state (0), so it's agreement
}

#[test]
fn dt_rows_normalized() {
    let params = make_params(3, 0.01);
    let log_e = vec![
        vec![-1.0, -3.0, -2.0],
        vec![-2.0, -1.0, -3.0],
        vec![-3.0, -2.0, -1.0],
    ];
    let states = vec![0, 1, 2];
    let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);
    for matrix in &result {
        for row in matrix {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01, "row not normalized: sum={}", sum);
        }
    }
}

#[test]
fn dt_out_of_range_state_safe() {
    let params = make_params(2, 0.01);
    let log_e = vec![vec![-1.0, -2.0]];
    let states = vec![5]; // out of range
    let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);
    assert_eq!(result.len(), 1);
}
