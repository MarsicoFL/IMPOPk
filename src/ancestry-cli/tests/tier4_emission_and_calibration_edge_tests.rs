// Tier 4 edge case tests: emission transforms, calibration, confusion penalties,
// window quality, and temperature scaling functions.
//
// Covers 12 previously-untested functions in ancestry-cli/src/hmm.rs:
//   - scale_temperature_for_copying
//   - apply_focused_masking
//   - compute_adaptive_pairwise_scales
//   - blend_log_emissions_adaptive_per_window
//   - apply_margin_persistence
//   - apply_variance_penalty
//   - apply_flank_informed_bonus
//   - compute_calibration_boosts
//   - apply_calibration_boosts
//   - compute_confusion_penalties
//   - apply_confusion_penalties
//   - compute_window_quality

use hprc_ancestry_cli::hmm::{
    apply_calibration_boosts, apply_confusion_penalties, apply_flank_informed_bonus,
    apply_focused_masking, apply_margin_persistence, apply_variance_penalty,
    blend_log_emissions_adaptive_per_window, compute_adaptive_pairwise_scales,
    compute_boundary_boost_transitions, compute_calibration_boosts,
    compute_confusion_penalties, compute_window_quality,
    scale_temperature_for_copying, AncestralPopulation, AncestryHmmParams,
};

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
// scale_temperature_for_copying
// ===========================================================================

#[test]
fn stfc_single_population_returns_input() {
    // n_populations <= 1 should return population_temperature unchanged
    assert_eq!(scale_temperature_for_copying(0.5, 1, 10), 0.5);
}

#[test]
fn stfc_single_haplotype_returns_input() {
    assert_eq!(scale_temperature_for_copying(0.5, 3, 1), 0.5);
}

#[test]
fn stfc_haps_equal_pops_returns_input() {
    // n_haplotypes <= n_populations → return unchanged
    assert_eq!(scale_temperature_for_copying(0.5, 5, 5), 0.5);
}

#[test]
fn stfc_correction_factor_applied() {
    // 3 pops, 30 haps → correction = ln(3)/ln(30) ≈ 0.323
    let result = scale_temperature_for_copying(1.0, 3, 30);
    let expected = (3.0_f64.ln() / 30.0_f64.ln()).clamp(0.0001, 0.5);
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn stfc_clamped_below() {
    // Very small temperature input → clamped to 0.0001
    let result = scale_temperature_for_copying(0.0001, 2, 100);
    assert!(result >= 0.0001);
}

#[test]
fn stfc_clamped_above() {
    // Large temperature with correction near 1 → clamped to 0.5
    let result = scale_temperature_for_copying(10.0, 5, 6);
    assert!(result <= 0.5);
}

#[test]
fn stfc_zero_temperature() {
    let result = scale_temperature_for_copying(0.0, 3, 30);
    assert!(result >= 0.0001); // clamped
}

// ===========================================================================
// apply_focused_masking
// ===========================================================================

#[test]
fn afm_empty_emissions() {
    let result = apply_focused_masking(&[], &[], 0.1, 2);
    assert!(result.is_empty());
}

#[test]
fn afm_zero_threshold_returns_unchanged() {
    let le = vec![vec![-1.0, -2.0, -3.0]];
    let post = vec![vec![0.5, 0.3, 0.2]];
    let result = apply_focused_masking(&le, &post, 0.0, 2);
    assert_eq!(result, le);
}

#[test]
fn afm_negative_threshold_returns_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.7, 0.3]];
    let result = apply_focused_masking(&le, &post, -0.5, 2);
    assert_eq!(result, le);
}

#[test]
fn afm_empty_posteriors_returns_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let result = apply_focused_masking(&le, &[], 0.1, 2);
    assert_eq!(result, le);
}

#[test]
fn afm_masks_below_threshold() {
    // 4 pops, threshold=0.2, min_active=2
    // post = [0.5, 0.3, 0.1, 0.1] → 2 above threshold (0.5, 0.3) >= min_active=2
    let le = vec![vec![-1.0, -2.0, -3.0, -4.0]];
    let post = vec![vec![0.5, 0.3, 0.1, 0.1]];
    let result = apply_focused_masking(&le, &post, 0.2, 2);
    assert_eq!(result[0][0], -1.0); // kept
    assert_eq!(result[0][1], -2.0); // kept
    assert!(result[0][2].is_infinite()); // masked
    assert!(result[0][3].is_infinite()); // masked
}

#[test]
fn afm_keeps_top_n_when_too_few_survive() {
    // 4 pops, threshold=0.4, min_active=2
    // post = [0.35, 0.30, 0.20, 0.15] → 0 above threshold < effective_min=2
    // → keep top-2 by posterior (idx 0, 1)
    let le = vec![vec![-1.0, -2.0, -3.0, -4.0]];
    let post = vec![vec![0.35, 0.30, 0.20, 0.15]];
    let result = apply_focused_masking(&le, &post, 0.4, 2);
    assert_eq!(result[0][0], -1.0); // top-1
    assert_eq!(result[0][1], -2.0); // top-2
    assert!(result[0][2].is_infinite()); // masked
    assert!(result[0][3].is_infinite()); // masked
}

#[test]
fn afm_min_active_floor_is_2() {
    // min_active=0 → effective_min = max(0, 2) = 2
    let le = vec![vec![-1.0, -2.0, -3.0]];
    let post = vec![vec![0.8, 0.1, 0.1]];
    // Only 1 above threshold=0.5 → < effective_min=2 → keeps top-2
    let result = apply_focused_masking(&le, &post, 0.5, 0);
    assert_eq!(result[0][0], -1.0);
    assert_eq!(result[0][1], -2.0);
    assert!(result[0][2].is_infinite());
}

#[test]
fn afm_multiple_windows_independent() {
    let le = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-4.0, -5.0, -6.0],
    ];
    let post = vec![
        vec![0.6, 0.3, 0.1],
        vec![0.1, 0.1, 0.8],
    ];
    let result = apply_focused_masking(&le, &post, 0.25, 2);
    // Window 0: 0.6, 0.3 above threshold → mask idx 2
    assert!(result[0][2].is_infinite());
    assert_eq!(result[0][0], -1.0);
    // Window 1: only 0.8 above threshold → too few → keep top-2 (idx 2, 0 or 1)
    assert_eq!(result[1][2], -6.0); // top-1 by posterior
}

// ===========================================================================
// compute_adaptive_pairwise_scales
// ===========================================================================

#[test]
fn caps_empty() {
    let result = compute_adaptive_pairwise_scales(&[], 0.2, 1.0);
    assert!(result.is_empty());
}

#[test]
fn caps_single_window() {
    // Single window can't have a meaningful gap distribution
    let le = vec![vec![-1.0, -3.0, -2.0]];
    let result = compute_adaptive_pairwise_scales(&le, 0.2, 1.0);
    assert_eq!(result.len(), 1);
    assert!(result[0] >= 0.2 - 1e-10 && result[0] <= 1.0 + 1e-10);
}

#[test]
fn caps_all_equal_emissions() {
    // All equal → gaps are 0 → ratio 0 → max_scale
    let le = vec![vec![-2.0, -2.0, -2.0]; 5];
    let result = compute_adaptive_pairwise_scales(&le, 0.1, 0.9);
    for &s in &result {
        assert!((s - 0.9).abs() < 1e-10); // max_scale for zero gaps
    }
}

#[test]
fn caps_large_gap_gets_low_scale() {
    // Window with huge gap → high gap/median → ratio ≥ 1 → min_scale
    // Window with zero gap → ratio 0 → max_scale
    let le = vec![
        vec![0.0, -100.0], // large gap → low scale
        vec![-1.0, -1.0],  // zero gap → high scale
    ];
    let result = compute_adaptive_pairwise_scales(&le, 0.0, 1.0);
    assert!(result[0] < result[1]); // large gap → smaller scale
}

#[test]
fn caps_neg_infinity_filtered() {
    // NEG_INFINITY filtered when computing gap
    let le = vec![vec![-1.0, f64::NEG_INFINITY, -3.0]];
    let result = compute_adaptive_pairwise_scales(&le, 0.0, 1.0);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_finite());
}

#[test]
fn caps_output_in_range() {
    let le = vec![
        vec![-1.0, -2.0, -5.0],
        vec![-1.0, -1.5, -3.0],
        vec![-1.0, -10.0, -10.0],
    ];
    let result = compute_adaptive_pairwise_scales(&le, 0.3, 0.8);
    for &s in &result {
        assert!(s >= 0.3 - 1e-10 && s <= 0.8 + 1e-10);
    }
}

// ===========================================================================
// blend_log_emissions_adaptive_per_window
// ===========================================================================

#[test]
fn bleapw_empty() {
    let result = blend_log_emissions_adaptive_per_window(&[], &[], 0.5, &[]);
    assert!(result.is_empty());
}

#[test]
fn bleapw_scale_zero_means_standard_only() {
    let std = vec![vec![-1.0, -2.0]];
    let pw = vec![vec![-5.0, -6.0]];
    let scales = vec![0.0];
    let result = blend_log_emissions_adaptive_per_window(&std, &pw, 0.5, &scales);
    // effective_weight = 0.5 * 0.0 = 0.0 → all standard
    assert_eq!(result[0][0], -1.0);
    assert_eq!(result[0][1], -2.0);
}

#[test]
fn bleapw_high_scale_blends_more_pairwise() {
    let std = vec![vec![-1.0, -2.0]];
    let pw = vec![vec![-3.0, -4.0]];
    let lo = vec![0.2];
    let hi = vec![1.0];
    let result_lo = blend_log_emissions_adaptive_per_window(&std, &pw, 0.5, &lo);
    let result_hi = blend_log_emissions_adaptive_per_window(&std, &pw, 0.5, &hi);
    // Higher scale → more pairwise → values closer to pw
    let diff_lo = (result_lo[0][0] - (-3.0)).abs();
    let diff_hi = (result_hi[0][0] - (-3.0)).abs();
    assert!(diff_hi < diff_lo); // high scale result is closer to pairwise
}

#[test]
fn bleapw_effective_weight_clamped_at_095() {
    // base_weight=1.0, scale=10.0 → effective_weight clamped to 0.95
    let std = vec![vec![-1.0]];
    let pw = vec![vec![-2.0]];
    let scales = vec![10.0];
    let result = blend_log_emissions_adaptive_per_window(&std, &pw, 1.0, &scales);
    let expected = 0.05 * (-1.0) + 0.95 * (-2.0);
    assert!((result[0][0] - expected).abs() < 1e-10);
}

#[test]
fn bleapw_standard_neg_inf_uses_pairwise() {
    let std = vec![vec![f64::NEG_INFINITY, -2.0]];
    let pw = vec![vec![-5.0, -6.0]];
    let scales = vec![1.0];
    let result = blend_log_emissions_adaptive_per_window(&std, &pw, 0.5, &scales);
    // std is NEG_INFINITY → falls through to pairwise
    assert_eq!(result[0][0], -5.0);
}

#[test]
fn bleapw_both_neg_inf() {
    let std = vec![vec![f64::NEG_INFINITY]];
    let pw = vec![vec![f64::NEG_INFINITY]];
    let scales = vec![1.0];
    let result = blend_log_emissions_adaptive_per_window(&std, &pw, 0.5, &scales);
    assert!(result[0][0].is_infinite());
}

// ===========================================================================
// apply_margin_persistence
// ===========================================================================

#[test]
fn amp_empty() {
    let result = apply_margin_persistence(&[], &[], 0.1, 1.0);
    assert!(result.is_empty());
}

#[test]
fn amp_zero_bonus_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.8, 0.2]];
    let result = apply_margin_persistence(&le, &post, 0.1, 0.0);
    assert_eq!(result, le);
}

#[test]
fn amp_negative_bonus_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.8, 0.2]];
    let result = apply_margin_persistence(&le, &post, 0.1, -1.0);
    assert_eq!(result, le);
}

#[test]
fn amp_threshold_gte_1_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.8, 0.2]];
    let result = apply_margin_persistence(&le, &post, 1.0, 1.0);
    assert_eq!(result, le);
}

#[test]
fn amp_margin_below_threshold_unchanged() {
    // margin = 0.6 - 0.4 = 0.2 < threshold=0.5 → no change
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.6, 0.4]];
    let result = apply_margin_persistence(&le, &post, 0.5, 1.0);
    assert_eq!(result, le);
}

#[test]
fn amp_margin_above_threshold_boosts_argmax() {
    // margin = 0.8 - 0.2 = 0.6 > threshold=0.1
    // strength = 1.0 * (0.6 - 0.1) / (1.0 - 0.1) = 0.5/0.9 ≈ 0.556
    let le = vec![vec![-1.0, -2.0]];
    let post = vec![vec![0.8, 0.2]];
    let result = apply_margin_persistence(&le, &post, 0.1, 1.0);
    assert!(result[0][0] > -1.0); // argmax boosted
    assert_eq!(result[0][1], -2.0); // other unchanged
}

#[test]
fn amp_neg_infinity_argmax_not_boosted() {
    // If argmax emission is NEG_INFINITY, the boost is skipped (not finite)
    let le = vec![vec![f64::NEG_INFINITY, -2.0]];
    let post = vec![vec![0.9, 0.1]];
    let result = apply_margin_persistence(&le, &post, 0.1, 1.0);
    assert!(result[0][0].is_infinite());
}

#[test]
fn amp_single_pop_unchanged() {
    // k < 2 → return unchanged
    let le = vec![vec![-1.0]];
    let post = vec![vec![1.0]];
    let result = apply_margin_persistence(&le, &post, 0.1, 1.0);
    assert_eq!(result, le);
}

// ===========================================================================
// apply_variance_penalty
// ===========================================================================

#[test]
fn avp_empty_emissions() {
    let result = apply_variance_penalty(&[], &[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn avp_zero_weight_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let var = vec![vec![0.5, 0.8]];
    let result = apply_variance_penalty(&le, &var, 0.0);
    assert_eq!(result, le);
}

#[test]
fn avp_negative_weight_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let var = vec![vec![0.5, 0.8]];
    let result = apply_variance_penalty(&le, &var, -1.0);
    assert_eq!(result, le);
}

#[test]
fn avp_all_zero_variances_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let var = vec![vec![0.0, 0.0]];
    let result = apply_variance_penalty(&le, &var, 1.0);
    assert_eq!(result, le);
}

#[test]
fn avp_penalizes_high_variance() {
    // Two pops with different variances
    let le = vec![
        vec![-1.0, -1.0],
        vec![-1.0, -1.0],
    ];
    let var = vec![
        vec![0.1, 0.5], // pop 1 has 5x variance
        vec![0.5, 0.1],
    ];
    let result = apply_variance_penalty(&le, &var, 1.0);
    // Higher variance → more penalty → more negative
    assert!(result[0][1] < result[0][0]); // pop 1 penalized more in window 0
    assert!(result[1][0] < result[1][1]); // pop 0 penalized more in window 1
}

#[test]
fn avp_neg_infinity_preserved() {
    let le = vec![vec![f64::NEG_INFINITY, -1.0]];
    let var = vec![vec![0.5, 0.5]];
    let result = apply_variance_penalty(&le, &var, 1.0);
    assert!(result[0][0].is_infinite());
    assert!(result[0][1] < -1.0); // penalized
}

#[test]
fn avp_empty_variances_returns_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let result = apply_variance_penalty(&le, &[], 1.0);
    assert_eq!(result, le);
}

// ===========================================================================
// apply_flank_informed_bonus
// ===========================================================================

#[test]
fn afib_empty_emissions() {
    let result = apply_flank_informed_bonus(&[], &[], 2, 1.0, 3);
    assert!(result.is_empty());
}

#[test]
fn afib_zero_radius_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let states = vec![0];
    let result = apply_flank_informed_bonus(&le, &states, 0, 1.0, 2);
    assert_eq!(result, le);
}

#[test]
fn afib_zero_bonus_unchanged() {
    let le = vec![vec![-1.0, -2.0]];
    let states = vec![0];
    let result = apply_flank_informed_bonus(&le, &states, 2, 0.0, 2);
    assert_eq!(result, le);
}

#[test]
fn afib_negative_bonus_unchanged() {
    let le = vec![vec![-1.0, -2.0]; 5];
    let states = vec![0, 0, 0, 0, 0];
    let result = apply_flank_informed_bonus(&le, &states, 1, -1.0, 2);
    assert_eq!(result, le);
}

#[test]
fn afib_both_flanks_agree_applies_bonus() {
    // [0, 0, X, 0, 0] with radius=2 → window 2 flanks both 0 → bonus to state 0
    let le = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0], // target window
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let states = vec![0, 0, 1, 0, 0]; // flanks agree on 0
    let result = apply_flank_informed_bonus(&le, &states, 2, 0.5, 2);
    assert!((result[2][0] - (-1.0 + 0.5)).abs() < 1e-10); // state 0 boosted
    assert_eq!(result[2][1], -2.0); // state 1 unchanged
}

#[test]
fn afib_flanks_disagree_no_bonus() {
    // [0, 0, X, 1, 1] → left=0, right=1 → disagree → no bonus
    let le = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let states = vec![0, 0, 0, 1, 1];
    let result = apply_flank_informed_bonus(&le, &states, 1, 0.5, 2);
    assert_eq!(result[2], le[2]); // no bonus (flanks disagree)
}

#[test]
fn afib_first_window_no_left_flank() {
    // Window 0 has no left flank → no bonus
    let le = vec![vec![-1.0, -2.0]; 3];
    let states = vec![0, 0, 0];
    let result = apply_flank_informed_bonus(&le, &states, 1, 0.5, 2);
    assert_eq!(result[0], le[0]); // no left flank → no bonus
}

#[test]
fn afib_last_window_no_right_flank() {
    // Last window has no right flank → no bonus
    let le = vec![vec![-1.0, -2.0]; 3];
    let states = vec![0, 0, 0];
    let result = apply_flank_informed_bonus(&le, &states, 1, 0.5, 2);
    let last = le.len() - 1;
    assert_eq!(result[last], le[last]);
}

#[test]
fn afib_neg_infinity_not_boosted() {
    let le = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![f64::NEG_INFINITY, -2.0], // target — state 0 is NEG_INFINITY
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];
    let states = vec![0, 0, 0, 0, 0]; // flanks agree on 0
    let result = apply_flank_informed_bonus(&le, &states, 1, 0.5, 2);
    assert!(result[2][0].is_infinite()); // NEG_INFINITY not boosted
}

// ===========================================================================
// compute_calibration_boosts
// ===========================================================================

#[test]
fn ccb_empty_states() {
    let result = compute_calibration_boosts(&[], &[0.5, 0.5], 2, 0.5);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn ccb_zero_scale_factor() {
    let result = compute_calibration_boosts(&[0, 1, 0, 1], &[0.5, 0.5], 2, 0.0);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn ccb_negative_scale_factor() {
    let result = compute_calibration_boosts(&[0, 1, 0, 1], &[0.5, 0.5], 2, -1.0);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn ccb_balanced_no_boost() {
    // 50/50 observed matches 50/50 expected → boosts near 0
    let states = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let props = vec![0.5, 0.5];
    let result = compute_calibration_boosts(&states, &props, 2, 0.5);
    assert!(result[0].abs() < 0.01);
    assert!(result[1].abs() < 0.01);
}

#[test]
fn ccb_underrepresented_gets_positive_boost() {
    // All states=0, but proportions say 50/50 → state 1 underrepresented
    let states = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let props = vec![0.5, 0.5];
    let result = compute_calibration_boosts(&states, &props, 2, 0.5);
    assert!(result[1] > 0.0); // underrepresented → positive boost
    assert!(result[0] < 0.0); // overrepresented → negative boost
}

#[test]
fn ccb_overrepresented_gets_negative_boost() {
    // All states=0 with prior=[0.1, 0.9] → state 0 overrepresented
    let states = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let props = vec![0.1, 0.9];
    let result = compute_calibration_boosts(&states, &props, 2, 0.5);
    assert!(result[0] < 0.0); // over-represented
    assert!(result[1] > 0.0); // under-represented
}

#[test]
fn ccb_out_of_range_state_ignored() {
    // State=5 but n_states=2 → only valid states counted
    let states = vec![0, 5, 0, 5];
    let props = vec![0.5, 0.5];
    let result = compute_calibration_boosts(&states, &props, 2, 0.5);
    assert!(result.len() == 2);
}

// ===========================================================================
// apply_calibration_boosts
// ===========================================================================

#[test]
fn acb_empty_emissions() {
    let result = apply_calibration_boosts(&[], &[1.0, -1.0]);
    assert!(result.is_empty());
}

#[test]
fn acb_empty_boosts() {
    let le = vec![vec![-1.0, -2.0]];
    let result = apply_calibration_boosts(&le, &[]);
    assert_eq!(result, le);
}

#[test]
fn acb_applies_boosts() {
    let le = vec![vec![-1.0, -2.0]];
    let boosts = vec![0.5, -0.3];
    let result = apply_calibration_boosts(&le, &boosts);
    assert!((result[0][0] - (-0.5)).abs() < 1e-10);
    assert!((result[0][1] - (-2.3)).abs() < 1e-10);
}

#[test]
fn acb_neg_infinity_preserved() {
    let le = vec![vec![f64::NEG_INFINITY, -2.0]];
    let boosts = vec![1.0, 1.0];
    let result = apply_calibration_boosts(&le, &boosts);
    assert!(result[0][0].is_infinite());
    assert!((result[0][1] - (-1.0)).abs() < 1e-10);
}

#[test]
fn acb_boost_index_beyond_row_uses_zero() {
    // boosts has 3 entries but row has 2 → extra boost ignored safely
    let le = vec![vec![-1.0, -2.0]];
    let boosts = vec![0.5, -0.5, 999.0];
    let result = apply_calibration_boosts(&le, &boosts);
    assert!((result[0][0] - (-0.5)).abs() < 1e-10);
    assert!((result[0][1] - (-2.5)).abs() < 1e-10);
}

#[test]
fn acb_multi_window() {
    let le = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
    ];
    let boosts = vec![0.1, -0.1];
    let result = apply_calibration_boosts(&le, &boosts);
    assert!((result[0][0] - (-0.9)).abs() < 1e-10);
    assert!((result[1][1] - (-4.1)).abs() < 1e-10);
}

// ===========================================================================
// compute_confusion_penalties
// ===========================================================================

#[test]
fn ccp_empty_states() {
    let result = compute_confusion_penalties(&[], 3, 0.5);
    assert_eq!(result, vec![vec![0.0; 3]; 3]);
}

#[test]
fn ccp_single_state_element() {
    let result = compute_confusion_penalties(&[0], 3, 0.5);
    assert_eq!(result, vec![vec![0.0; 3]; 3]);
}

#[test]
fn ccp_zero_weight() {
    let result = compute_confusion_penalties(&[0, 1, 0, 1], 2, 0.0);
    assert_eq!(result, vec![vec![0.0; 2]; 2]);
}

#[test]
fn ccp_no_switches() {
    // All same state → no transitions → no penalties
    let result = compute_confusion_penalties(&[0, 0, 0, 0, 0], 3, 0.5);
    assert_eq!(result, vec![vec![0.0; 3]; 3]);
}

#[test]
fn ccp_frequent_switches_get_larger_penalty() {
    // 0↔1 switches frequently, 0↔2 never switches
    let states = vec![0, 1, 0, 1, 0, 1, 0, 2];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    // 0↔1 should have more negative penalty than 0↔2
    assert!(result[0][1] < result[0][2]);
    // Symmetrized
    assert!((result[0][1] - result[1][0]).abs() < 1e-10);
}

#[test]
fn ccp_penalties_are_nonpositive() {
    let states = vec![0, 1, 2, 0, 1, 2, 1, 0];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    for row in &result {
        for &val in row {
            assert!(val <= 0.0);
        }
    }
}

#[test]
fn ccp_diagonal_always_zero() {
    let states = vec![0, 1, 2, 0, 1, 2];
    let result = compute_confusion_penalties(&states, 3, 1.0);
    for i in 0..3 {
        assert_eq!(result[i][i], 0.0);
    }
}

#[test]
fn ccp_out_of_range_states_safe() {
    let states = vec![0, 5, 1, 10]; // states 5, 10 out of range for n=3
    let result = compute_confusion_penalties(&states, 3, 1.0);
    assert_eq!(result.len(), 3);
}

// ===========================================================================
// apply_confusion_penalties
// ===========================================================================

#[test]
fn acp_zero_penalties_normalizes() {
    let params = make_params(3, 0.1);
    let penalties = vec![vec![0.0; 3]; 3];
    let result = apply_confusion_penalties(&params, &penalties);
    // With zero penalties, should get log of original transitions (normalized)
    for row in &result {
        let sum: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn acp_negative_penalty_reduces_transition() {
    let params = make_params(2, 0.2);
    let mut penalties = vec![vec![0.0; 2]; 2];
    penalties[0][1] = -2.0; // heavy penalty for 0→1
    let result = apply_confusion_penalties(&params, &penalties);
    // Transition 0→1 should be less likely with penalty
    let base_log_01 = params.transitions[0][1].max(1e-20).ln();
    // After penalty and renormalization, the off-diag should be smaller in prob space
    let penalized_prob = result[0][1].exp();
    assert!(penalized_prob < params.transitions[0][1]);
}

#[test]
fn acp_rows_sum_to_one_in_prob() {
    let params = make_params(4, 0.15);
    let penalties = vec![
        vec![0.0, -0.5, -1.0, -0.2],
        vec![-0.3, 0.0, -0.1, -0.8],
        vec![-1.0, -0.5, 0.0, -0.3],
        vec![-0.2, -0.2, -0.2, 0.0],
    ];
    let result = apply_confusion_penalties(&params, &penalties);
    for row in &result {
        let sum: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "Row sum = {}", sum);
    }
}

// ===========================================================================
// compute_window_quality
// ===========================================================================

#[test]
fn cwq_empty() {
    let result = compute_window_quality(&[], &[], &[], 2);
    assert!(result.is_empty());
}

#[test]
fn cwq_single_window_max_quality() {
    // Single pop → margin=1.0, single window → agreement=1.0
    let post = vec![vec![1.0]];
    let le = vec![vec![-1.0]];
    let states = vec![0];
    let result = compute_window_quality(&post, &le, &states, 1);
    assert_eq!(result.len(), 1);
    // margin=1.0 (only 1 pop), disc=0.0 (single element), agreement=1.0
    // quality = 0.4*1.0 + 0.3*0.0 + 0.3*1.0 = 0.7
    assert!((result[0] - 0.7).abs() < 1e-6);
}

#[test]
fn cwq_confident_window() {
    // Clear winner in posteriors and emissions, neighbors agree
    let post = vec![
        vec![0.9, 0.05, 0.05],
        vec![0.9, 0.05, 0.05],
        vec![0.9, 0.05, 0.05],
    ];
    let le = vec![
        vec![-0.1, -5.0, -5.0],
        vec![-0.1, -5.0, -5.0],
        vec![-0.1, -5.0, -5.0],
    ];
    let states = vec![0, 0, 0];
    let result = compute_window_quality(&post, &le, &states, 1);
    // All windows should have high quality
    assert!(result[1] > 0.7);
}

#[test]
fn cwq_uncertain_window() {
    // All pops equal in posteriors and emissions
    let post = vec![vec![0.333, 0.333, 0.334]];
    let le = vec![vec![-1.1, -1.1, -1.1]];
    let states = vec![0];
    let result = compute_window_quality(&post, &le, &states, 1);
    // margin ≈ 0.001, disc ≈ 0, agreement=1 (single window)
    // quality = 0.4*~0 + 0.3*0 + 0.3*1 ≈ 0.3
    assert!(result[0] < 0.5);
}

#[test]
fn cwq_disagreeing_neighbors() {
    // Longer sequence: middle window disagrees, edges agree
    let post = vec![
        vec![0.9, 0.1],
        vec![0.9, 0.1],
        vec![0.9, 0.1],
        vec![0.9, 0.1],
        vec![0.9, 0.1],
    ];
    let le = vec![
        vec![-0.5, -3.0],
        vec![-0.5, -3.0],
        vec![-0.5, -3.0],
        vec![-0.5, -3.0],
        vec![-0.5, -3.0],
    ];
    let states = vec![0, 0, 1, 0, 0]; // middle disagrees with all neighbors
    let result = compute_window_quality(&post, &le, &states, 1);
    // Window 2 has 0% agreement vs window 1 has ~50-100% agreement
    assert!(result[2] < result[1]);
}

#[test]
fn cwq_output_in_range() {
    let post = vec![
        vec![0.5, 0.3, 0.2],
        vec![0.1, 0.8, 0.1],
        vec![0.4, 0.4, 0.2],
    ];
    let le = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-3.0, -0.5, -4.0],
        vec![-1.5, -1.5, -3.0],
    ];
    let states = vec![0, 1, 0];
    let result = compute_window_quality(&post, &le, &states, 1);
    for &q in &result {
        assert!(q >= 0.0 && q <= 1.0);
    }
}

#[test]
fn cwq_radius_zero_full_agreement() {
    let post = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
    let le = vec![vec![-0.5, -3.0], vec![-3.0, -0.5]];
    let states = vec![0, 1];
    let result = compute_window_quality(&post, &le, &states, 0);
    // radius=0 → agreement=1.0 for all windows
    // quality should be higher than with radius>0 and disagreement
    for &q in &result {
        assert!(q > 0.0);
    }
}

// ===========================================================================
// compute_boundary_boost_transitions
// ===========================================================================

#[test]
fn cbbt_empty_states() {
    let params = make_params(3, 0.1);
    let result = compute_boundary_boost_transitions(&[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn cbbt_no_boundaries_base_transitions() {
    let params = make_params(2, 0.1);
    let states = vec![0, 0, 0, 0];
    let result = compute_boundary_boost_transitions(&states, &params, 1.0);
    assert_eq!(result.len(), 4);
    // All non-boundary → same as base transitions
    // Check rows sum to 1 in prob space
    for mat in &result {
        for row in mat {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn cbbt_boundary_eases_switching() {
    let params = make_params(2, 0.1);
    let states = vec![0, 0, 1, 1]; // boundary at positions 1,2
    let result = compute_boundary_boost_transitions(&states, &params, 1.0);
    // Position 0: non-boundary
    let non_boundary_switch = result[0][0][1].exp();
    // Position 1 or 2: boundary
    let boundary_switch = result[1][0][1].exp();
    // Boundary should make switching easier (higher prob)
    assert!(boundary_switch > non_boundary_switch);
}

#[test]
fn cbbt_rows_normalized() {
    let params = make_params(3, 0.2);
    let states = vec![0, 1, 2, 0, 1];
    let result = compute_boundary_boost_transitions(&states, &params, 2.0);
    for mat in &result {
        for row in mat {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-5, "Row sum = {}", sum);
        }
    }
}
