//! Edge-case tests for `blend_log_emissions_hybrid` (T78 §4.2).
//!
//! Complements `hybrid_pairwise_tests.rs` (13 tests by algo_dev) with
//! boundary, NaN/Inf, degenerate-input, and stress scenarios.

use impopk_ancestry_cli::blend_log_emissions_hybrid;

// ============================================================================
// Negative parameter clamping
// ============================================================================

#[test]
fn hybrid_negative_base_weight_clamped_to_zero() {
    let std = vec![vec![-1.0, -2.0]];
    let pw = vec![vec![-0.5, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, -0.5, 1.5, 0.2, 0.2, 3.0);
    // w = (-0.5 * 1.5 * ratio).clamp(0.0, 0.95) = 0.0 → pure standard
    for s in 0..2 {
        assert!(
            (result[0][s] - std[0][s]).abs() < 1e-12,
            "negative base_weight: pop {s}: got {}, expected {}",
            result[0][s], std[0][s]
        );
    }
}

#[test]
fn hybrid_negative_agree_scale_clamped_to_zero() {
    // Agree case, but agree_scale < 0 → product < 0 → clamped to 0
    let std = vec![vec![-0.5, -2.0]];
    let pw = vec![vec![-0.3, -2.5]]; // agree, both argmax=0
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, -1.0, 0.2, 0.2, 3.0);
    for s in 0..2 {
        assert!(
            (result[0][s] - std[0][s]).abs() < 1e-12,
            "negative agree_scale: pop {s}: got {}, expected {}",
            result[0][s], std[0][s]
        );
    }
}

#[test]
fn hybrid_negative_disagree_scale_clamped_to_zero() {
    // Disagree case, but disagree_scale < 0 → clamped to 0
    let std = vec![vec![-0.5, -2.0]]; // argmax=0
    let pw = vec![vec![-2.0, -0.3]]; // argmax=1 (disagree)
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, -1.0, 0.2, 3.0);
    for s in 0..2 {
        assert!(
            (result[0][s] - std[0][s]).abs() < 1e-12,
            "negative disagree_scale: pop {s}: got {}, expected {}",
            result[0][s], std[0][s]
        );
    }
}

// ============================================================================
// NaN / Inf in full rows
// ============================================================================

#[test]
fn hybrid_all_nan_profile_row() {
    // Profile is all-NaN → NaN.is_finite()=false → fallback to standard
    let std = vec![vec![-1.0, -2.0]];
    let pw = vec![vec![f64::NAN, f64::NAN]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    for s in 0..2 {
        assert_eq!(result[0][s], std[0][s], "all-NaN profile: fallback to standard");
    }
}

#[test]
fn hybrid_all_nan_standard_row() {
    // Standard is all-NaN → fallback to profile
    let std = vec![vec![f64::NAN, f64::NAN]];
    let pw = vec![vec![-0.3, -1.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    for s in 0..2 {
        assert_eq!(result[0][s], pw[0][s], "all-NaN standard: fallback to profile");
    }
}

#[test]
fn hybrid_both_all_nan() {
    // Both all-NaN → NEG_INFINITY
    let std = vec![vec![f64::NAN, f64::NAN]];
    let pw = vec![vec![f64::NAN, f64::NAN]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    for s in 0..2 {
        assert_eq!(
            result[0][s],
            f64::NEG_INFINITY,
            "both all-NaN: NEG_INFINITY"
        );
    }
}

#[test]
fn hybrid_pos_inf_in_profile_row() {
    // +Inf in profile → is_finite()=false → margin=0 for that row
    // Also blend falls back to standard for that element
    let std = vec![vec![-1.0, -2.0]];
    let pw = vec![vec![f64::INFINITY, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // Pop 0: profile is +Inf (not finite) → fallback to standard = -1.0
    assert_eq!(result[0][0], -1.0, "+Inf profile: fallback to standard");
}

#[test]
fn hybrid_pos_inf_in_standard_row() {
    // +Inf in standard → is_finite()=false → fallback to profile
    let std = vec![vec![f64::INFINITY, -2.0]];
    let pw = vec![vec![-0.5, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result[0][0], -0.5, "+Inf standard: fallback to profile");
}

#[test]
fn hybrid_both_pos_inf_same_element() {
    // Both +Inf → neither finite → NEG_INFINITY
    let std = vec![vec![f64::INFINITY, -2.0]];
    let pw = vec![vec![f64::INFINITY, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result[0][0], f64::NEG_INFINITY, "both +Inf: NEG_INFINITY");
}

// ============================================================================
// Empty/degenerate inner rows
// ============================================================================

#[test]
fn hybrid_empty_inner_rows() {
    // Empty rows: argmax = None, agree = false, margin = 0
    let std = vec![vec![], vec![-1.0, -2.0]];
    let pw = vec![vec![], vec![-0.5, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert!(result[0].is_empty(), "empty row produces empty output");
    assert_eq!(result.len(), 2, "zip truncation preserves window count");
}

#[test]
fn hybrid_single_element_rows_always_agree() {
    // Single-element rows: margin = 0, argmax always index 0 → agree
    let std = vec![vec![-1.0], vec![-2.0], vec![-3.0]];
    let pw = vec![vec![-0.5], vec![-1.5], vec![-2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.4, 1.0, 0.2, 0.2, 3.0);
    // All margins = 0 → no positive margins → median = 1.0 (neutral)
    // margin_ratio = 0/1.0 = 0 → clamped to margin_clamp_lo = 0.2
    let w = (0.4_f64 * 1.0 * 0.2).clamp(0.0, 0.95); // = 0.08
    for t in 0..3 {
        let expected = (1.0 - w) * std[t][0] + w * pw[t][0];
        assert!(
            (result[t][0] - expected).abs() < 1e-12,
            "single-element window {t}: got {}, expected {}",
            result[t][0], expected
        );
    }
}

// ============================================================================
// Clamp bounds edge cases
// ============================================================================

#[test]
fn hybrid_margin_clamp_lo_equals_hi() {
    // clamp_lo == clamp_hi → all agree windows get same margin ratio
    let std = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pw = vec![
        vec![-0.3, -3.0], // agree, margin = 2.7
        vec![-0.3, -0.5], // agree, margin = 0.2
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.0, 0.2, 1.0, 1.0);
    // Both clamped to 1.0, so w = 0.3 * 1.0 * 1.0 = 0.3 for both
    let w = 0.3;
    for t in 0..2 {
        for s in 0..2 {
            let expected = (1.0 - w) * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "clamp_lo==hi: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

#[test]
fn hybrid_zero_margin_clamp_lo() {
    // clamp_lo = 0 → weak-margin agree windows get near-zero weight
    let std = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pw = vec![
        vec![-0.3, -0.300001], // margin ≈ 1e-6, agree
        vec![-0.3, -2.3],      // margin = 2.0, agree
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.0, 3.0);
    // median = 2.0, window 0 ratio = ~5e-7 → clamp(0.0, 3.0) = ~5e-7
    // w ≈ 0.3 * 1.5 * 5e-7 ≈ 2.25e-7 → effectively near-zero weight
    // Window 1: ratio = 2.0/2.0 = 1.0, w = 0.3 * 1.5 * 1.0 = 0.45
    let w1 = 0.45;
    for s in 0..2 {
        // Window 0: nearly pure standard (w ≈ 0)
        assert!(
            (result[0][s] - std[0][s]).abs() < 1e-5,
            "zero clamp_lo: window 0 near standard, pop {s}: diff = {}",
            (result[0][s] - std[0][s]).abs()
        );
        // Window 1: full weight
        let expected = (1.0 - w1) * std[1][s] + w1 * pw[1][s];
        assert!(
            (result[1][s] - expected).abs() < 1e-12,
            "zero clamp_lo: window 1 pop {s}: got {}, expected {}",
            result[1][s], expected
        );
    }
}

// ============================================================================
// base_weight boundary
// ============================================================================

#[test]
fn hybrid_zero_base_weight_returns_standard() {
    // base_weight = 0 → w = 0 for both agree/disagree → result = standard
    let std = vec![vec![-0.5, -2.0], vec![-2.0, -0.5]];
    let pw = vec![vec![-0.3, -2.5], vec![-0.3, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.0, 1.5, 0.2, 0.2, 3.0);
    for t in 0..2 {
        for s in 0..2 {
            assert!(
                (result[t][s] - std[t][s]).abs() < 1e-12,
                "zero base_weight: [{t}][{s}]: got {}, expected {}",
                result[t][s], std[t][s]
            );
        }
    }
}

#[test]
fn hybrid_base_weight_one_agree_scale_one_clamped() {
    // base_weight=1.0, agree_scale=1.0 → w = 1.0 * ratio → can exceed 0.95 → clamped
    let std = vec![vec![-0.5, -2.0]];
    let pw = vec![vec![-0.3, -2.5]]; // agree, margin = 2.2, median = 2.2, ratio = 1.0
    let result = blend_log_emissions_hybrid(&std, &pw, 1.0, 1.0, 0.2, 0.2, 3.0);
    // w = (1.0 * 1.0 * 1.0).clamp(0.0, 0.95) = 0.95
    let w = 0.95;
    for s in 0..2 {
        let expected = (1.0 - w) * std[0][s] + w * pw[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "base=1 agree=1: pop {s}: got {}, expected {}",
            result[0][s], expected
        );
    }
}

// ============================================================================
// Median margin computation
// ============================================================================

#[test]
fn hybrid_tiny_median_margin_near_epsilon() {
    // All margins very close to 1e-10 → triggers the "median_margin > 1e-10" branch barely
    let std = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pw = vec![
        vec![-0.3, -0.3 - 2e-10], // margin = 2e-10 (agree)
        vec![-0.3, -0.3 - 2e-10], // margin = 2e-10 (agree)
        vec![-0.3, -0.3 - 2e-10], // margin = 2e-10 (agree)
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // margins all = 2e-10 > 0 → included. median = 2e-10. median > 1e-10 → use ratio
    // ratio = 2e-10 / 2e-10 = 1.0, w = 0.3 * 1.5 * 1.0 = 0.45
    let w = 0.45;
    for t in 0..3 {
        for s in 0..2 {
            let expected = (1.0 - w) * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-9,
                "tiny median: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

#[test]
fn hybrid_median_margin_below_epsilon_uses_neutral() {
    // median_margin ≤ 1e-10 → use ratio = 1.0 (neutral)
    let std = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    // margins so tiny they're below epsilon once filtered
    let pw = vec![
        vec![-0.3, -0.3 - 5e-11], // margin = 5e-11 (agree)
        vec![-0.3, -0.3 - 5e-11], // margin = 5e-11 (agree)
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // margins = [5e-11, 5e-11]. Both > 0 → included. median = 5e-11
    // 5e-11 <= 1e-10 → ratio = 1.0 (neutral fallback)
    let w = (0.3_f64 * 1.5 * 1.0).clamp(0.0, 0.95); // = 0.45
    for t in 0..2 {
        for s in 0..2 {
            let expected = (1.0 - w) * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-9,
                "below-epsilon median: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

#[test]
fn hybrid_even_count_positive_margins_picks_floor_median() {
    // 4 positive margins → median = sorted[2] (index 4/2 = 2)
    let std = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pw = vec![
        vec![-0.3, -0.5],  // margin = 0.2, agree
        vec![-0.3, -1.3],  // margin = 1.0, agree
        vec![-0.3, -2.3],  // margin = 2.0, agree
        vec![-0.3, -3.3],  // margin = 3.0, agree
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.0, 0.2, 0.2, 3.0);
    // sorted = [0.2, 1.0, 2.0, 3.0], median = sorted[4/2] = sorted[2] = 2.0
    let median = 2.0;
    let ratios: [f64; 4] = [0.2 / median, 1.0 / median, 2.0 / median, 3.0 / median];
    for (t, &ratio_raw) in ratios.iter().enumerate() {
        let ratio = ratio_raw.clamp(0.2, 3.0);
        let w = (0.3_f64 * 1.0 * ratio).clamp(0.0, 0.95);
        for s in 0..2 {
            let expected = (1.0 - w) * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "even-count median: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

// ============================================================================
// Argmax tie-breaking
// ============================================================================

#[test]
fn hybrid_argmax_tie_total_cmp_picks_last() {
    // Two identical max values: max_by picks the last one
    // If both models have ties resolved the same way → agree
    let std = vec![vec![-1.0, -1.0, -2.0]]; // tie between 0 and 1, max_by → index 1
    let pw = vec![vec![-1.0, -1.0, -2.0]];  // same tie → index 1 → agree
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // Both pick index 1 → agree. margin = 0 → no positive margins → median = 1.0
    // ratio = 0/1.0 = 0 → clamped to 0.2
    let w = (0.3_f64 * 1.5 * 0.2).clamp(0.0, 0.95); // = 0.09
    for s in 0..3 {
        let expected = (1.0 - w) * std[0][s] + w * pw[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "tie agree: pop {s}: got {}, expected {}",
            result[0][s], expected
        );
    }
}

#[test]
fn hybrid_argmax_tie_different_patterns_disagree() {
    // std has tie at [0,1], pw has tie at [1,2] → different last-pick → disagree
    let std = vec![vec![-1.0, -1.0, -2.0]]; // max_by picks index 1
    let pw = vec![vec![-2.0, -1.0, -1.0]];  // max_by picks index 2
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // Disagree: w = 0.3 * 0.2 = 0.06
    let w = 0.06;
    for s in 0..3 {
        let expected = (1.0 - w) * std[0][s] + w * pw[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "tie disagree: pop {s}: got {}, expected {}",
            result[0][s], expected
        );
    }
}

// ============================================================================
// Mismatched lengths
// ============================================================================

#[test]
fn hybrid_more_standard_windows_truncated() {
    // Standard has 3 windows, profile has 2 → zip gives 2
    let std = vec![vec![-0.5, -2.0], vec![-0.5, -2.0], vec![-0.5, -2.0]];
    let pw = vec![vec![-0.3, -2.5], vec![-0.3, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result.len(), 2, "zip truncation to shorter length");
}

#[test]
fn hybrid_more_profile_windows_truncated() {
    // Profile has 3 windows, standard has 2 → zip gives 2
    let std = vec![vec![-0.5, -2.0], vec![-0.5, -2.0]];
    let pw = vec![vec![-0.3, -2.5], vec![-0.3, -2.5], vec![-0.3, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result.len(), 2, "zip truncation to shorter length");
}

#[test]
fn hybrid_mismatched_inner_row_lengths() {
    // std row has 3 pops, pw row has 2 → zip gives 2 elements
    let std = vec![vec![-0.5, -2.0, -3.0]];
    let pw = vec![vec![-0.3, -2.5]];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result[0].len(), 2, "inner zip truncation");
}

// ============================================================================
// Profile all-identical values (zero margin)
// ============================================================================

#[test]
fn hybrid_profile_all_identical_zero_margin() {
    // Profile has identical values → margin = 0 for all windows
    let std = vec![vec![-0.5, -2.0], vec![-0.5, -2.0]];
    let pw = vec![vec![-1.0, -1.0], vec![-1.0, -1.0]]; // margin = 0 everywhere
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // All margins = 0 → no positive margins → median = 1.0
    // argmax_bt picks last index (1) via total_cmp tie → disagree with std (argmax=0)
    // w = 0.3 * 0.2 = 0.06
    let w = 0.06;
    for t in 0..2 {
        for s in 0..2 {
            let expected = (1.0 - w) * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "all-identical profile: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

// ============================================================================
// Stress test
// ============================================================================

#[test]
fn hybrid_50_pops_100_windows_all_finite() {
    let n_pops = 50;
    let n_windows = 100;
    let mut std_em = Vec::with_capacity(n_windows);
    let mut pw_em = Vec::with_capacity(n_windows);
    for t in 0..n_windows {
        let mut std_row = vec![-3.0; n_pops];
        let mut pw_row = vec![-3.0; n_pops];
        // Make population (t % n_pops) the argmax for both
        std_row[t % n_pops] = -0.5;
        pw_row[t % n_pops] = -0.3;
        // Second-best slightly worse in profile for margin
        pw_row[(t + 1) % n_pops] = -1.0;
        std_em.push(std_row);
        pw_em.push(pw_row);
    }
    let result = blend_log_emissions_hybrid(&std_em, &pw_em, 0.3, 1.5, 0.2, 0.2, 3.0);
    assert_eq!(result.len(), n_windows);
    for t in 0..n_windows {
        assert_eq!(result[t].len(), n_pops);
        for s in 0..n_pops {
            assert!(
                result[t][s].is_finite(),
                "stress [{t}][{s}] not finite: {}",
                result[t][s]
            );
        }
    }
}

// ============================================================================
// Symmetry and identity
// ============================================================================

#[test]
fn hybrid_identical_inputs_weighted_identity() {
    // When standard == profile, result should be identical (regardless of weight)
    let data = vec![vec![-0.5, -1.5, -2.5], vec![-2.0, -0.3, -1.0]];
    let result = blend_log_emissions_hybrid(&data, &data, 0.4, 1.5, 0.2, 0.2, 3.0);
    for t in 0..2 {
        for s in 0..3 {
            assert!(
                (result[t][s] - data[t][s]).abs() < 1e-12,
                "identity: [{t}][{s}]: got {}, expected {}",
                result[t][s], data[t][s]
            );
        }
    }
}

// ============================================================================
// NaN in margin computation
// ============================================================================

#[test]
fn hybrid_nan_in_profile_margin_computation() {
    // Mix of NaN and finite in profile row → best/second tracking with NaN
    // NaN > everything in total_cmp → best = NaN, not finite → margin = 0
    let std = vec![vec![-0.5, -2.0, -3.0]];
    let pw = vec![vec![-0.3, f64::NAN, -2.5]]; // NaN in middle
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // NaN in total_cmp is greatest → best = NaN, not finite → margin = 0
    // argmax_bt = index 1 (NaN), argmax_std = 0 → disagree
    // Element 1: pw is NaN (not finite) → fallback to standard
    assert_eq!(result[0][1], std[0][1], "NaN element falls back to standard");
    assert!(result[0].len() == 3);
}

// ============================================================================
// Margin where only one window has positive margin
// ============================================================================

#[test]
fn hybrid_single_positive_margin_is_median() {
    // 3 windows but only 1 has positive margin → median = that margin
    let std = vec![
        vec![-0.5, -2.0], // argmax = 0
        vec![-0.5, -2.0], // argmax = 0
        vec![-0.5, -2.0], // argmax = 0
    ];
    let pw = vec![
        vec![-0.3, -0.3], // margin = 0 (tie), argmax = 1 (last via total_cmp) → disagree
        vec![-0.3, -0.3], // margin = 0, disagree
        vec![-0.3, -1.3], // margin = 1.0, argmax = 0 → agree
    ];
    let result = blend_log_emissions_hybrid(&std, &pw, 0.3, 1.5, 0.2, 0.2, 3.0);
    // Only window 2 has positive margin (1.0) → median = 1.0
    // Window 2 (agree): ratio = 1.0/1.0 = 1.0, w = 0.3 * 1.5 * 1.0 = 0.45
    // Windows 0,1 (disagree): w = 0.3 * 0.2 = 0.06
    let weights = [0.06, 0.06, 0.45];
    for (t, &w) in weights.iter().enumerate() {
        let w_std = 1.0 - w;
        for s in 0..2 {
            let expected = w_std * std[t][s] + w * pw[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "single positive margin: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}
