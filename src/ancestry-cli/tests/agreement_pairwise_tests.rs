//! Tests for `blend_log_emissions_agreement` (T76 bias-aware variant).
//!
//! Agreement-based blending uses argmax agreement between standard and pairwise
//! emissions as the quality signal, rather than margin-based confidence (which
//! amplifies coherent bias via the Stein paradox).

use hprc_ancestry_cli::blend_log_emissions_agreement;

// ============================================================================
// Basic agreement / disagreement behavior
// ============================================================================

#[test]
fn agreement_upweights_when_models_agree() {
    // Both standard and pairwise point to pop 0
    let standard = vec![vec![-0.5, -2.0, -3.0]];
    let pairwise = vec![vec![-0.3, -2.5, -3.5]];
    // agree_scale=1.5, disagree_scale=0.2
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.4, 1.5, 0.2);
    // w = 0.4 * 1.5 = 0.6
    let w = 0.6;
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Agree: pop {s} got {}, expected {}", result[0][s], expected
        );
    }
}

#[test]
fn agreement_downweights_when_models_disagree() {
    // Standard points to pop 0, pairwise points to pop 1
    let standard = vec![vec![-0.5, -2.0, -3.0]];
    let pairwise = vec![vec![-2.0, -0.3, -3.5]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.4, 1.5, 0.2);
    // w = 0.4 * 0.2 = 0.08
    let w = 0.08;
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Disagree: pop {s} got {}, expected {}", result[0][s], expected
        );
    }
}

#[test]
fn agreement_mixed_windows() {
    // Window 0: agree (both → pop 0), Window 1: disagree (std → pop 1, pw → pop 0)
    let standard = vec![
        vec![-0.5, -2.0],  // argmax = 0
        vec![-2.0, -0.5],  // argmax = 1
    ];
    let pairwise = vec![
        vec![-0.3, -1.5],  // argmax = 0 → agree
        vec![-0.3, -1.5],  // argmax = 0 → disagree
    ];
    let base = 0.5;
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, base, 1.5, 0.2);

    // Window 0: agree → w = 0.5 * 1.5 = 0.75
    let w0 = 0.75;
    for s in 0..2 {
        let expected = (1.0 - w0) * standard[0][s] + w0 * pairwise[0][s];
        assert!((result[0][s] - expected).abs() < 1e-12);
    }
    // Window 1: disagree → w = 0.5 * 0.2 = 0.1
    let w1 = 0.1;
    for s in 0..2 {
        let expected = (1.0 - w1) * standard[1][s] + w1 * pairwise[1][s];
        assert!((result[1][s] - expected).abs() < 1e-12);
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn agreement_empty_input() {
    let standard: Vec<Vec<f64>> = vec![];
    let pairwise: Vec<Vec<f64>> = vec![];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.2);
    assert!(result.is_empty());
}

#[test]
fn agreement_empty_profile_returns_standard() {
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise: Vec<Vec<f64>> = vec![];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.2);
    assert_eq!(result, standard);
}

#[test]
fn agreement_weight_clamped_at_095() {
    // agree_scale very large → should clamp at 0.95
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![-0.5, -3.0]]; // agree: both argmax=0
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.8, 5.0, 0.2);
    // w = 0.8 * 5.0 = 4.0 → clamped to 0.95
    let w = 0.95;
    let w_std = 1.0 - w;
    for s in 0..2 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "Weight should clamp at 0.95"
        );
    }
}

#[test]
fn agreement_base_weight_zero_returns_standard() {
    let standard = vec![vec![-1.0, -2.0], vec![-0.5, -1.5]];
    let pairwise = vec![vec![-5.0, -0.1], vec![-0.1, -5.0]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.0, 1.5, 0.2);
    for t in 0..2 {
        for s in 0..2 {
            assert!(
                (result[t][s] - standard[t][s]).abs() < 1e-12,
                "Weight 0 should return standard"
            );
        }
    }
}

#[test]
fn agreement_nan_in_profile_uses_standard() {
    let standard = vec![vec![-1.0, -2.0]];
    let pairwise = vec![vec![f64::NAN, -2.0]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.2);
    // std[0] is finite, profile[0] is NaN → should use std
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
}

#[test]
fn agreement_neg_inf_standard_uses_profile() {
    let standard = vec![vec![f64::NEG_INFINITY, -2.0]];
    let pairwise = vec![vec![-0.5, -1.5]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.2);
    assert!(result[0][0].is_finite(), "NEG_INFINITY std should fallback to profile");
}

#[test]
fn agreement_both_neg_inf_returns_neg_inf() {
    let standard = vec![vec![f64::NEG_INFINITY, -1.0]];
    let pairwise = vec![vec![f64::NEG_INFINITY, -2.0]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.2);
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert!(result[0][1].is_finite());
}

#[test]
fn agreement_single_pop_always_agrees() {
    // With single population, argmax is always 0 for both → agree
    let standard = vec![vec![-1.0], vec![-2.0]];
    let pairwise = vec![vec![-0.5], vec![-0.8]];
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.4, 1.5, 0.2);
    // Always agree → w = 0.4 * 1.5 = 0.6
    let w = 0.6;
    let w_std = 1.0 - w;
    for t in 0..2 {
        let expected = w_std * standard[t][0] + w * pairwise[t][0];
        assert!(
            (result[t][0] - expected).abs() < 1e-12,
            "Single pop should always agree"
        );
    }
}

#[test]
fn agreement_ten_populations() {
    let k = 10;
    let n = 5;
    let mut standard = Vec::with_capacity(n);
    let mut pairwise = Vec::with_capacity(n);
    for t in 0..n {
        let mut std_row = vec![-2.0; k];
        let mut pw_row = vec![-2.0; k];
        std_row[t % k] = -0.5;
        pw_row[t % k] = -0.3;
        standard.push(std_row);
        pairwise.push(pw_row);
    }
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.4, 1.5, 0.2);
    assert_eq!(result.len(), n);
    for row in &result {
        assert_eq!(row.len(), k);
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn agreement_disagree_scale_zero_returns_standard_on_disagree() {
    // disagree_scale=0 → w=0 on disagreement → pure standard
    let standard = vec![vec![-0.5, -2.0]]; // argmax=0
    let pairwise = vec![vec![-2.0, -0.5]]; // argmax=1 → disagree
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, 0.5, 1.5, 0.0);
    // w = 0.5 * 0.0 = 0.0
    for s in 0..2 {
        assert!(
            (result[0][s] - standard[0][s]).abs() < 1e-12,
            "disagree_scale=0 should return pure standard on disagreement"
        );
    }
}

#[test]
fn agreement_scales_independently_per_window() {
    // Verify each window gets its own agree/disagree check
    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // argmax=0
        vec![-3.0, -0.5, -2.0],  // argmax=1
        vec![-2.0, -3.0, -0.5],  // argmax=2
    ];
    let pairwise = vec![
        vec![-0.3, -2.5, -3.5],  // argmax=0 → agree
        vec![-0.3, -2.5, -3.5],  // argmax=0 → disagree (std=1)
        vec![-2.5, -3.5, -0.3],  // argmax=2 → agree
    ];
    let base = 0.4;
    let result = blend_log_emissions_agreement(
        &standard, &pairwise, base, 2.0, 0.1);

    // Window 0: agree → w = 0.4*2.0 = 0.8
    let w0 = 0.8;
    // Window 1: disagree → w = 0.4*0.1 = 0.04
    let w1 = 0.04;
    // Window 2: agree → w = 0.4*2.0 = 0.8
    let w2 = 0.8;

    for s in 0..3 {
        let e0 = (1.0 - w0) * standard[0][s] + w0 * pairwise[0][s];
        assert!((result[0][s] - e0).abs() < 1e-12, "Window 0 pop {s}");

        let e1 = (1.0 - w1) * standard[1][s] + w1 * pairwise[1][s];
        assert!((result[1][s] - e1).abs() < 1e-12, "Window 1 pop {s}");

        let e2 = (1.0 - w2) * standard[2][s] + w2 * pairwise[2][s];
        assert!((result[2][s] - e2).abs() < 1e-12, "Window 2 pop {s}");
    }
}
