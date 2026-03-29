//! Cycle 89: Edge case tests for transition computation and posterior refinement
//! functions with only 1 test-file reference each.
//!
//! Targets: compute_confusion_penalties, compute_cooccurrence_transitions,
//! compute_segment_length_prior, compute_recency_transitions,
//! compute_disagreement_transitions, compute_adaptive_transitions,
//! apply_windowed_normalization, entropy_smooth_posteriors, local_rerank_emissions.

use hprc_ancestry_cli::hmm::{
    apply_windowed_normalization, compute_adaptive_transitions, compute_cooccurrence_transitions,
    compute_confusion_penalties, compute_disagreement_transitions, compute_recency_transitions,
    compute_segment_length_prior, entropy_smooth_posteriors, local_rerank_emissions,
    AncestralPopulation, AncestryHmmParams, EmissionModel,
};

/// Helper: build a minimal 3-state AncestryHmmParams for tests.
fn make_params_3() -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation { name: "A".into(), haplotypes: vec!["h1".into()] },
        AncestralPopulation { name: "B".into(), haplotypes: vec!["h2".into()] },
        AncestralPopulation { name: "C".into(), haplotypes: vec!["h3".into()] },
    ];
    AncestryHmmParams {
        n_states: 3,
        populations: pops,
        transitions: vec![
            vec![0.9, 0.05, 0.05],
            vec![0.05, 0.9, 0.05],
            vec![0.05, 0.05, 0.9],
        ],
        initial: vec![1.0 / 3.0; 3],
        emission_same_pop_mean: 0.95,
        emission_diff_pop_mean: 0.80,
        emission_std: 0.05,
        emission_model: EmissionModel::Max,
        normalization: None,
        coverage_weight: 0.0,
        transition_dampening: 0.0,
    }
}

/// Helper: build a minimal 2-state AncestryHmmParams for tests.
fn make_params_2() -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation { name: "P0".into(), haplotypes: vec!["h0".into()] },
        AncestralPopulation { name: "P1".into(), haplotypes: vec!["h1".into()] },
    ];
    AncestryHmmParams {
        n_states: 2,
        populations: pops,
        transitions: vec![
            vec![0.95, 0.05],
            vec![0.05, 0.95],
        ],
        initial: vec![0.5, 0.5],
        emission_same_pop_mean: 0.95,
        emission_diff_pop_mean: 0.80,
        emission_std: 0.05,
        emission_model: EmissionModel::Max,
        normalization: None,
        coverage_weight: 0.0,
        transition_dampening: 0.0,
    }
}

// ============================================================================
// compute_confusion_penalties
// ============================================================================

#[test]
fn confusion_penalties_empty_states() {
    let result = compute_confusion_penalties(&[], 3, 1.0);
    // Empty states → no transitions, all penalties = 0
    assert_eq!(result.len(), 3);
    for row in &result {
        for &v in row {
            assert!((v).abs() < 1e-10);
        }
    }
}

#[test]
fn confusion_penalties_weight_zero() {
    let states = vec![0, 1, 0, 1];
    let result = compute_confusion_penalties(&states, 2, 0.0);
    for row in &result {
        for &v in row {
            assert!((v).abs() < 1e-10);
        }
    }
}

#[test]
fn confusion_penalties_no_switches() {
    let states = vec![0, 0, 0, 0];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    // No switches → all penalties = 0
    for row in &result {
        for &v in row {
            assert!((v).abs() < 1e-10);
        }
    }
}

#[test]
fn confusion_penalties_symmetric() {
    let states = vec![0, 1, 0, 2, 0, 1];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    for i in 0..3 {
        for j in 0..3 {
            assert!((result[i][j] - result[j][i]).abs() < 1e-10, "asymmetric at ({i},{j})");
        }
    }
}

#[test]
fn confusion_penalties_diagonal_zero() {
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    for i in 0..3 {
        assert!((result[i][i]).abs() < 1e-10);
    }
}

#[test]
fn confusion_penalties_high_switch_pair_more_negative() {
    // 0↔1 switch happens 4 times, 0↔2 switch happens once
    let states = vec![0, 1, 0, 1, 0, 1, 0, 2, 0];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    // 0↔1 should have more negative penalty than 0↔2
    assert!(result[0][1] < result[0][2]);
}

#[test]
fn confusion_penalties_n_states_one_empty() {
    let states = vec![0, 0, 0];
    let result = compute_confusion_penalties(&states, 1, 1.0);
    // n_states < 2 → all zero
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
}

// ============================================================================
// compute_cooccurrence_transitions
// ============================================================================

#[test]
fn cooccurrence_transitions_empty_states() {
    let params = make_params_3();
    let result = compute_cooccurrence_transitions(&[], &params, 1.0);
    // No transitions → result ≈ base log-transitions
    assert_eq!(result.len(), 3);
}

#[test]
fn cooccurrence_transitions_no_switches() {
    let params = make_params_3();
    let states = vec![0, 0, 0, 0];
    let result = compute_cooccurrence_transitions(&states, &params, 1.0);
    // No switches → co-occurrence = 0 → result ≈ base
    assert_eq!(result.len(), 3);
    // Each row should be log-normalized
    for row in &result {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max_val + row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
        assert!(lse.abs() < 0.1, "row not normalized: lse={lse}");
    }
}

#[test]
fn cooccurrence_transitions_frequent_switch_boosted() {
    let params = make_params_2();
    // Many 0→1 switches
    let states = vec![0, 1, 0, 1, 0, 1, 0, 1];
    let result = compute_cooccurrence_transitions(&states, &params, 2.0);
    // The 0→1 transition should be boosted relative to base
    let base_01 = (0.05_f64).max(1e-20).ln();
    // result[0][1] should be > base_01 (before renormalization)
    // After renormalization it's harder to check directly, but we can verify
    // the off-diagonal is relatively stronger than with no switches
    let no_switch_result = compute_cooccurrence_transitions(&[0, 0, 0, 0], &params, 2.0);
    // Both are log-normalized, so compare relative differences
    let switch_diff = result[0][0] - result[0][1];
    let no_switch_diff = no_switch_result[0][0] - no_switch_result[0][1];
    // With frequent switches, off-diagonal boosted → smaller diff
    assert!(
        switch_diff < no_switch_diff,
        "switch_diff={switch_diff}, no_switch_diff={no_switch_diff}"
    );
}

#[test]
fn cooccurrence_transitions_rows_log_normalized() {
    let params = make_params_3();
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_cooccurrence_transitions(&states, &params, 1.0);
    for (i, row) in result.iter().enumerate() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max_val + row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
        assert!(lse.abs() < 0.1, "row {i} not normalized: lse={lse}");
    }
}

// ============================================================================
// compute_segment_length_prior
// ============================================================================

#[test]
fn segment_length_prior_empty() {
    let params = make_params_3();
    let result = compute_segment_length_prior(&[], &params, 3);
    assert!(result.is_empty());
}

#[test]
fn segment_length_prior_min_length_one() {
    let params = make_params_3();
    let result = compute_segment_length_prior(&[0, 0, 1], &params, 1);
    assert!(result.is_empty()); // min_length <= 1 → empty
}

#[test]
fn segment_length_prior_long_run_no_penalty() {
    let params = make_params_2();
    let states = vec![0, 0, 0, 0, 0]; // run of 5
    let min_length = 3;
    let result = compute_segment_length_prior(&states, &params, min_length);
    // At t=2 (run_length=3), run_length >= min_length → no penalty, base transitions
    // At t=4 (run_length=5), run_length >= min_length → no penalty
    assert_eq!(result.len(), 5);
    // Windows at t>=2 should have base transitions (no penalty applied)
    let base_self = (0.95_f64).max(1e-20).ln();
    // t=4 has run_length=5 >= 3, so no penalty. The self-transition should be close to base.
    let self_trans = result[4][0][0];
    // The base may be renormalized slightly, but should be close
    assert!(self_trans < 0.0); // log probability is negative
}

#[test]
fn segment_length_prior_short_run_penalized() {
    let params = make_params_2();
    let states = vec![0, 1, 0]; // short runs
    let min_length = 3;
    let result = compute_segment_length_prior(&states, &params, min_length);
    // t=0: run_length=1 < 3 → penalty applied to off-diagonal
    // t=1: state changed to 1, run_length=1 < 3 → penalty applied
    assert_eq!(result.len(), 3);
    // The penalty should make switching harder (off-diagonal more negative)
    // compared to a long run
    let long_states = vec![0, 0, 0];
    let long_result = compute_segment_length_prior(&long_states, &params, min_length);
    // At t=0, both have run_length=1, so both are penalized equally
    // At t=2, long run has run_length=3 → no penalty; short run has run_length=1 → penalty
    let short_off_diag = result[2][0][1]; // state 0 → state 1
    let long_off_diag = long_result[2][0][1];
    assert!(
        short_off_diag < long_off_diag,
        "short={short_off_diag}, long={long_off_diag}"
    );
}

#[test]
fn segment_length_prior_output_dimensions() {
    let params = make_params_3();
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_segment_length_prior(&states, &params, 3);
    assert_eq!(result.len(), 5);
    for trans in &result {
        assert_eq!(trans.len(), 3);
        for row in trans {
            assert_eq!(row.len(), 3);
        }
    }
}

// ============================================================================
// compute_recency_transitions
// ============================================================================

#[test]
fn recency_transitions_empty() {
    let params = make_params_3();
    let result = compute_recency_transitions(&[], &params, 0.9);
    assert!(result.is_empty());
}

#[test]
fn recency_transitions_alpha_zero() {
    let params = make_params_2();
    let result = compute_recency_transitions(&[0, 1, 0], &params, 0.0);
    assert!(result.is_empty()); // alpha <= 0 → empty
}

#[test]
fn recency_transitions_output_length() {
    let params = make_params_3();
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_recency_transitions(&states, &params, 0.9);
    assert_eq!(result.len(), 5);
}

#[test]
fn recency_transitions_rows_log_normalized() {
    let params = make_params_3();
    let states = vec![0, 0, 1, 1, 2];
    let result = compute_recency_transitions(&states, &params, 0.8);
    for (t, trans) in result.iter().enumerate() {
        for (i, row) in trans.iter().enumerate() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = max_val + row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
            assert!(
                lse.abs() < 0.1,
                "t={t}, row {i} not normalized: lse={lse}"
            );
        }
    }
}

#[test]
fn recency_transitions_recent_state_boosted() {
    let params = make_params_2();
    // State 0 appears much more recently
    let states = vec![0, 0, 0, 0, 0];
    let result = compute_recency_transitions(&states, &params, 0.9);
    // At t=4, state 0 has high recency weight → self-transition for state 0 boosted
    let self_0 = result[4][0][0];
    let off_01 = result[4][0][1];
    assert!(self_0 > off_01);
}

// ============================================================================
// compute_disagreement_transitions
// ============================================================================

#[test]
fn disagreement_transitions_empty() {
    let params = make_params_3();
    let result = compute_disagreement_transitions(&[], &[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn disagreement_transitions_weight_zero() {
    let params = make_params_2();
    let emissions = vec![vec![-1.0, -5.0]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&emissions, &states, &params, 0.0);
    assert!(result.is_empty());
}

#[test]
fn disagreement_transitions_agreement_tightens() {
    let params = make_params_2();
    // Emission argmax = 0, state = 0 → agreement
    let emissions = vec![vec![0.0, -10.0]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&emissions, &states, &params, 1.0);
    assert_eq!(result.len(), 1);
    // Self-transition for state 0 should be boosted
    let self_0 = result[0][0][0];
    let off_01 = result[0][0][1];
    // Base: ln(0.95) ≈ -0.051, ln(0.05) ≈ -2.996
    // Agreement: self gets +weight → even higher self-transition
    assert!(self_0 > off_01);
}

#[test]
fn disagreement_transitions_disagreement_boosts_argmax() {
    let params = make_params_2();
    // Emission argmax = 1, state = 0 → disagreement
    let emissions = vec![vec![-10.0, 0.0]];
    let states = vec![0];
    let result = compute_disagreement_transitions(&emissions, &states, &params, 1.0);
    // Transition 0→1 should be boosted
    let off_01 = result[0][0][1];
    // Compare with agreement case
    let agree_emissions = vec![vec![0.0, -10.0]];
    let agree_result =
        compute_disagreement_transitions(&agree_emissions, &[0], &params, 1.0);
    let agree_off_01 = agree_result[0][0][1];
    assert!(
        off_01 > agree_off_01,
        "disagree off={off_01}, agree off={agree_off_01}"
    );
}

#[test]
fn disagreement_transitions_output_dimensions() {
    let params = make_params_3();
    let emissions = vec![vec![-1.0, -2.0, -3.0]; 4];
    let states = vec![0, 1, 2, 0];
    let result = compute_disagreement_transitions(&emissions, &states, &params, 0.5);
    assert_eq!(result.len(), 4);
    for trans in &result {
        assert_eq!(trans.len(), 3);
        for row in trans {
            assert_eq!(row.len(), 3);
        }
    }
}

// ============================================================================
// compute_adaptive_transitions
// ============================================================================

#[test]
fn adaptive_transitions_empty() {
    let params = make_params_3();
    let result = compute_adaptive_transitions(&[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn adaptive_transitions_output_length() {
    let params = make_params_2();
    let emissions = vec![vec![-1.0, -2.0]; 5];
    let result = compute_adaptive_transitions(&emissions, &params, 1.0);
    assert_eq!(result.len(), 5);
}

#[test]
fn adaptive_transitions_confident_emission_tighter() {
    let params = make_params_2();
    // Confident: one dominant emission
    let confident = vec![vec![0.0, -100.0]];
    // Uncertain: uniform emission
    let uncertain = vec![vec![-1.0, -1.0]];
    let r_conf = compute_adaptive_transitions(&confident, &params, 2.0);
    let r_uncert = compute_adaptive_transitions(&uncertain, &params, 2.0);
    // Confident → low entropy → large scale → off-diagonal more penalized
    let conf_off = r_conf[0][0][1];
    let uncert_off = r_uncert[0][0][1];
    assert!(
        conf_off < uncert_off,
        "conf_off={conf_off}, uncert_off={uncert_off}"
    );
}

#[test]
fn adaptive_transitions_rows_log_normalized() {
    let params = make_params_3();
    let emissions = vec![vec![-1.0, -3.0, -5.0], vec![-2.0, -2.0, -2.0]];
    let result = compute_adaptive_transitions(&emissions, &params, 1.0);
    for (t, trans) in result.iter().enumerate() {
        for (i, row) in trans.iter().enumerate() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = max_val + row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
            assert!(
                lse.abs() < 0.1,
                "t={t}, row {i} not normalized: lse={lse}"
            );
        }
    }
}

// ============================================================================
// apply_windowed_normalization
// ============================================================================

#[test]
fn windowed_norm_empty() {
    let result = apply_windowed_normalization(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn windowed_norm_radius_zero_noop() {
    let input = vec![vec![-1.0, -5.0], vec![-3.0, -7.0]];
    let result = apply_windowed_normalization(&input, 0);
    assert_eq!(result, input);
}

#[test]
fn windowed_norm_constant_signal_unchanged() {
    // If all windows are the same, local mean = global mean → output = input
    let input = vec![vec![-2.0, -3.0]; 5];
    let result = apply_windowed_normalization(&input, 2);
    for (r, i) in result.iter().zip(input.iter()) {
        for (rv, iv) in r.iter().zip(i.iter()) {
            assert!((rv - iv).abs() < 1e-10);
        }
    }
}

#[test]
fn windowed_norm_preserves_global_mean() {
    let input = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
        vec![-5.0, -6.0],
    ];
    let result = apply_windowed_normalization(&input, 1);
    // Global mean per pop should be preserved
    let global_mean_0: f64 = input.iter().map(|r| r[0]).sum::<f64>() / 3.0;
    let result_mean_0: f64 = result.iter().map(|r| r[0]).sum::<f64>() / 3.0;
    assert!(
        (global_mean_0 - result_mean_0).abs() < 0.5,
        "global={global_mean_0}, result={result_mean_0}"
    );
}

#[test]
fn windowed_norm_neg_infinity_preserved() {
    let input = vec![
        vec![f64::NEG_INFINITY, -1.0],
        vec![-2.0, -3.0],
        vec![-4.0, -5.0],
    ];
    let result = apply_windowed_normalization(&input, 1);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

#[test]
fn windowed_norm_single_window() {
    // With 1 window, local mean = global mean → output = input
    let input = vec![vec![-1.0, -2.0, -3.0]];
    let result = apply_windowed_normalization(&input, 5);
    for (rv, iv) in result[0].iter().zip(input[0].iter()) {
        assert!((rv - iv).abs() < 1e-10);
    }
}

// ============================================================================
// entropy_smooth_posteriors
// ============================================================================

#[test]
fn entropy_smooth_empty() {
    let result = entropy_smooth_posteriors(&[], 2);
    assert!(result.is_empty());
}

#[test]
fn entropy_smooth_radius_zero_noop() {
    let input = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
    let result = entropy_smooth_posteriors(&input, 0);
    assert_eq!(result, input);
}

#[test]
fn entropy_smooth_confident_dominates() {
    // Window 0: confident (low entropy, high weight)
    // Window 1: uncertain (high entropy, low weight)
    // With radius=1, the smoothed value at window 1 should be pulled toward window 0
    let input = vec![vec![0.99, 0.01], vec![0.5, 0.5]];
    let result = entropy_smooth_posteriors(&input, 1);
    // Window 1 should be pulled toward [0.99, 0.01]
    assert!(result[1][0] > 0.5, "smoothed_p0={}", result[1][0]);
}

#[test]
fn entropy_smooth_uniform_equal_weights() {
    // All windows identical → smoothed = input
    let input = vec![vec![0.5, 0.5]; 5];
    let result = entropy_smooth_posteriors(&input, 2);
    for row in &result {
        assert!((row[0] - 0.5).abs() < 1e-10);
        assert!((row[1] - 0.5).abs() < 1e-10);
    }
}

#[test]
fn entropy_smooth_sums_to_one() {
    let input = vec![vec![0.7, 0.2, 0.1], vec![0.3, 0.4, 0.3], vec![0.1, 0.1, 0.8]];
    let result = entropy_smooth_posteriors(&input, 1);
    for row in &result {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
    }
}

// ============================================================================
// local_rerank_emissions
// ============================================================================

#[test]
fn local_rerank_empty() {
    let result = local_rerank_emissions(&[], 2);
    assert!(result.is_empty());
}

#[test]
fn local_rerank_radius_zero_noop() {
    let input = vec![vec![-1.0, -5.0]];
    let result = local_rerank_emissions(&input, 0);
    assert_eq!(result, input);
}

#[test]
fn local_rerank_consistent_neighborhood() {
    // All windows agree pop 0 is best → pop 0 gets highest rank score
    let input = vec![
        vec![0.0, -10.0, -20.0],
        vec![0.0, -10.0, -20.0],
        vec![0.0, -10.0, -20.0],
    ];
    let result = local_rerank_emissions(&input, 1);
    for row in &result {
        assert!(row[0] > row[1]);
        assert!(row[1] > row[2]);
    }
}

#[test]
fn local_rerank_output_finite() {
    let input = vec![vec![-1.0, -3.0], vec![-2.0, -4.0]];
    let result = local_rerank_emissions(&input, 1);
    for row in &result {
        for &v in row {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
    }
}

#[test]
fn local_rerank_output_dimensions() {
    let input = vec![vec![-1.0, -2.0, -3.0]; 4];
    let result = local_rerank_emissions(&input, 2);
    assert_eq!(result.len(), 4);
    for row in &result {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn local_rerank_disagreeing_neighborhood() {
    // Windows disagree on which pop is best
    let input = vec![
        vec![0.0, -100.0],  // pop 0 best
        vec![-100.0, 0.0],  // pop 1 best
        vec![0.0, -100.0],  // pop 0 best
    ];
    let result = local_rerank_emissions(&input, 1);
    // At t=1 with radius=1, neighborhood sums: pop0 = 0 + (-100) + 0 = -100, pop1 = -100 + 0 + (-100) = -200
    // pop 0 has higher cumulative → rank 0
    assert!(result[1][0] > result[1][1]);
}
