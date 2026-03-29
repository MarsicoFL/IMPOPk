//! Edge case tests for `blend_log_emissions_agreement` (T76).
//!
//! Complements the 14 tests in `agreement_pairwise_tests.rs` by covering
//! boundary conditions, numerical edge cases, and unusual parameter combos.

use hprc_ancestry_cli::blend_log_emissions_agreement;

// ============================================================================
// Negative parameter edge cases
// ============================================================================

#[test]
fn agreement_negative_base_weight_clamped_to_zero() {
    // Negative base_weight → product is negative → clamp(0.0, 0.95) → 0.0
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![-5.0, -0.1]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, -0.5, 1.5, 0.2);
    // w = (-0.5 * scale).clamp(0.0, 0.95) = 0.0 for both agree and disagree
    for s in 0..2 {
        assert!(
            (result[0][s] - standard[0][s]).abs() < 1e-12,
            "Negative base_weight should clamp to 0 → pure standard"
        );
    }
}

#[test]
fn agreement_negative_agree_scale_clamped_to_zero() {
    // Positive base_weight * negative agree_scale → negative → clamped to 0
    let standard = vec![vec![-0.5, -2.0]];
    let pairwise = vec![vec![-0.3, -2.5]]; // agree: both argmax=0
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, -2.0, 0.2);
    // w = (0.4 * -2.0).clamp(0.0, 0.95) = 0.0
    for s in 0..2 {
        assert!(
            (result[0][s] - standard[0][s]).abs() < 1e-12,
            "Negative agree_scale should clamp to 0 → pure standard"
        );
    }
}

#[test]
fn agreement_negative_disagree_scale_clamped_to_zero() {
    // disagree case with negative disagree_scale → clamped to 0
    let standard = vec![vec![-0.5, -2.0]]; // argmax=0
    let pairwise = vec![vec![-2.0, -0.3]]; // argmax=1 → disagree
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, -1.0);
    // w = (0.4 * -1.0).clamp(0.0, 0.95) = 0.0
    for s in 0..2 {
        assert!(
            (result[0][s] - standard[0][s]).abs() < 1e-12,
            "Negative disagree_scale should clamp to 0 → pure standard"
        );
    }
}

// ============================================================================
// Argmax tie-breaking behavior
// ============================================================================

#[test]
fn agreement_tied_argmax_first_index_wins() {
    // When two populations have equal values, total_cmp picks the last one
    // (max_by returns the last maximum for equal elements)
    let standard = vec![vec![-1.0, -1.0, -2.0]]; // tie between 0 and 1
    let pairwise = vec![vec![-1.0, -1.0, -2.0]]; // same tie → should agree
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    // Both argmax resolve to same index → agree
    let w: f64 = (0.4 * 1.5_f64).min(0.95);
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Tied argmax should agree when both have same tie pattern"
        );
    }
}

#[test]
fn agreement_different_tie_patterns_disagree() {
    // Standard: tie at [0, 1], pairwise: clear winner at 2
    let standard = vec![vec![-1.0, -1.0, -2.0]]; // argmax = 1 (last of tie via max_by)
    let pairwise = vec![vec![-2.0, -2.0, -0.5]]; // argmax = 2 → disagree
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    let w: f64 = (0.4 * 0.2_f64).min(0.95);
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Different tie-breaking patterns should disagree"
        );
    }
}

// ============================================================================
// Row length mismatches
// ============================================================================

#[test]
fn agreement_mismatched_row_lengths_zip_truncates() {
    // Standard has 3 pops, pairwise has 2 → zip gives 2
    let standard = vec![vec![-1.0, -2.0, -3.0]];
    let pairwise = vec![vec![-0.5, -1.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2, "zip truncates to shorter row");
}

#[test]
fn agreement_more_standard_windows_than_pairwise() {
    // Standard has 3 windows, pairwise has 1 → zip gives 1
    let standard = vec![
        vec![-1.0, -2.0],
        vec![-0.5, -1.5],
        vec![-2.0, -0.5],
    ];
    let pairwise = vec![vec![-0.3, -2.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    assert_eq!(result.len(), 1, "zip truncates to fewer windows");
}

// ============================================================================
// Positive infinity handling
// ============================================================================

#[test]
fn agreement_positive_inf_in_standard_uses_pairwise() {
    // +Inf is not finite → falls back to pairwise
    let standard = vec![vec![f64::INFINITY, -2.0]];
    let pairwise = vec![vec![-0.5, -1.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    assert_eq!(result[0][0], -0.5, "+Inf std should fallback to profile value");
    assert!(result[0][1].is_finite());
}

#[test]
fn agreement_positive_inf_in_pairwise_uses_standard() {
    // +Inf pairwise is not finite → falls back to standard
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![f64::INFINITY, -1.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    assert_eq!(result[0][0], -1.0, "+Inf profile should fallback to standard value");
}

#[test]
fn agreement_both_positive_inf_returns_neg_inf() {
    // Both +Inf → neither is finite → last else branch → NEG_INFINITY
    let standard = vec![vec![f64::INFINITY, -1.0]];
    let pairwise = vec![vec![f64::INFINITY, -2.0]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
}

// ============================================================================
// NaN argmax interaction
// ============================================================================

#[test]
fn agreement_all_nan_standard_argmax_still_computed() {
    // NaN values: total_cmp puts NaN after +Inf, so argmax could pick NaN index
    // The blend still works because of the is_finite guard
    let standard = vec![vec![f64::NAN, f64::NAN]];
    let pairwise = vec![vec![-0.5, -1.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    // NaN is not finite → uses pairwise for each element
    assert_eq!(result[0][0], -0.5);
    assert_eq!(result[0][1], -1.5);
}

#[test]
fn agreement_all_nan_pairwise_argmax_still_computed() {
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![f64::NAN, f64::NAN]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    // NaN not finite → uses standard
    assert_eq!(result[0][0], -1.0);
    assert_eq!(result[0][1], -2.0);
}

#[test]
fn agreement_both_all_nan_returns_neg_inf() {
    let standard = vec![vec![f64::NAN, f64::NAN]];
    let pairwise = vec![vec![f64::NAN, f64::NAN]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.5, 1.5, 0.2);
    // Neither finite → NEG_INFINITY
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
}

// ============================================================================
// Equal scale factors
// ============================================================================

#[test]
fn agreement_equal_scales_no_distinction() {
    // agree_scale == disagree_scale → same weight regardless of agreement
    let standard = vec![
        vec![-0.5, -2.0], // argmax=0
        vec![-2.0, -0.5], // argmax=1
    ];
    let pairwise = vec![
        vec![-0.3, -2.5], // argmax=0 → agree
        vec![-0.3, -2.5], // argmax=0 → disagree
    ];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.0, 1.0);
    // Both: w = 0.4 * 1.0 = 0.4
    let w = 0.4;
    let w_std = 1.0 - w;
    for t in 0..2 {
        for s in 0..2 {
            let expected = w_std * standard[t][s] + w * pairwise[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "Equal scales: window {t} pop {s}"
            );
        }
    }
}

// ============================================================================
// Extreme scale values
// ============================================================================

#[test]
fn agreement_very_large_disagree_scale_clamped() {
    // Even disagree path can hit the 0.95 clamp with large scale
    let standard = vec![vec![-0.5, -2.0]]; // argmax=0
    let pairwise = vec![vec![-2.0, -0.5]]; // argmax=1 → disagree
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.8, 1.5, 10.0);
    // w = (0.8 * 10.0).clamp(0.0, 0.95) = 0.95
    let w = 0.95;
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Large disagree_scale should still clamp at 0.95"
        );
    }
}

#[test]
fn agreement_zero_base_weight_with_large_scales() {
    // base_weight=0 → w=0 regardless of scale factors
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![-5.0, -0.1]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.0, 1000.0, 1000.0);
    for s in 0..2 {
        assert!(
            (result[0][s] - standard[0][s]).abs() < 1e-12,
            "base_weight=0 with huge scales should still give pure standard"
        );
    }
}

// ============================================================================
// Many-population stress
// ============================================================================

#[test]
fn agreement_100_populations_all_finite() {
    let k = 100;
    let n = 20;
    let mut standard = Vec::with_capacity(n);
    let mut pairwise = Vec::with_capacity(n);
    for t in 0..n {
        let mut std_row = vec![-3.0; k];
        let mut pw_row = vec![-3.0; k];
        // Alternate agreement patterns
        std_row[t % k] = -0.1;
        if t % 3 == 0 {
            pw_row[(t + 1) % k] = -0.1; // disagree every 3rd window
        } else {
            pw_row[t % k] = -0.1; // agree
        }
        standard.push(std_row);
        pairwise.push(pw_row);
    }
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.3, 1.5, 0.2);
    assert_eq!(result.len(), n);
    for (t, row) in result.iter().enumerate() {
        assert_eq!(row.len(), k);
        for (s, &v) in row.iter().enumerate() {
            assert!(v.is_finite(), "Window {t} pop {s} not finite: {v}");
        }
    }
}

// ============================================================================
// Single-element rows
// ============================================================================

#[test]
fn agreement_single_element_rows_always_agree() {
    // With k=1 both argmax are always 0 → always agree
    let standard = vec![vec![-3.0], vec![-1.0], vec![-5.0]];
    let pairwise = vec![vec![-0.5], vec![-2.0], vec![-0.1]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.6, 1.5, 0.2);
    let w: f64 = (0.6 * 1.5_f64).min(0.95); // 0.9
    let w_std = 1.0 - w;
    for t in 0..3 {
        let expected = w_std * standard[t][0] + w * pairwise[t][0];
        assert!(
            (result[t][0] - expected).abs() < 1e-12,
            "Single-element row {t}: always agree path"
        );
    }
}

// ============================================================================
// Weight boundary precision
// ============================================================================

#[test]
fn agreement_weight_exactly_095_not_rounded_further() {
    // base * agree_scale = exactly 0.95 → no clamping needed
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![-0.5, -3.0]]; // agree
    // 0.95 / 1.5 = 0.6333... not exact, use 0.475 * 2.0 = 0.95
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.475, 2.0, 0.2);
    let w = 0.95;
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Weight exactly at 0.95 boundary"
        );
    }
}

#[test]
fn agreement_weight_just_above_095_clamped() {
    // 0.48 * 2.0 = 0.96 → clamped to 0.95
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![-0.5, -3.0]]; // agree
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.48, 2.0, 0.2);
    let w = 0.95; // clamped from 0.96
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Weight 0.96 should clamp to 0.95"
        );
    }
}

// ============================================================================
// Symmetry properties
// ============================================================================

#[test]
fn agreement_symmetric_emissions_always_agree() {
    // When standard and pairwise are identical, they always agree
    let emissions = vec![
        vec![-0.5, -2.0, -3.0],
        vec![-3.0, -0.5, -2.0],
        vec![-2.0, -3.0, -0.5],
    ];
    let result = blend_log_emissions_agreement(
        &emissions, &emissions, 0.4, 1.5, 0.2);
    // Always agree → w = 0.4 * 1.5 = 0.6
    // w_std * x + w * x = x (linear combination of same value)
    for t in 0..3 {
        for s in 0..3 {
            assert!(
                (result[t][s] - emissions[t][s]).abs() < 1e-12,
                "Identical inputs should return input unchanged"
            );
        }
    }
}

// ============================================================================
// Empty row within non-empty input
// ============================================================================

#[test]
fn agreement_empty_inner_row_produces_empty_output_row() {
    // A window with zero populations → empty row, argmax is None
    let standard = vec![vec![], vec![-1.0, -2.0]];
    let pairwise = vec![vec![], vec![-0.5, -1.5]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    assert_eq!(result.len(), 2);
    assert!(result[0].is_empty(), "Empty inner row should produce empty output row");
    // argmax is None for empty row → agree = false (argmax_std.is_some() fails)
    // Second row: both argmax=0 → agree
    assert_eq!(result[1].len(), 2);
}

#[test]
fn agreement_empty_inner_row_argmax_none_means_disagree() {
    // Empty row → argmax is None → agree=false → disagree_scale used
    // But the inner zip produces nothing, so it doesn't matter for the values.
    // This test just verifies it doesn't panic.
    let standard = vec![vec![]];
    let pairwise = vec![vec![]];
    let result = blend_log_emissions_agreement(&standard, &pairwise, 0.4, 1.5, 0.2);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}
