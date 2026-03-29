//! Tests for `blend_log_emissions_hybrid` (T78 §4.2).
//!
//! Hybrid blending combines agreement gating (T76) with margin scaling (adaptive):
//! - Agreement windows: weight = base × agree_scale × margin_ratio
//! - Disagreement windows: weight = base × disagree_scale (flat, no margin)

use hprc_ancestry_cli::blend_log_emissions_hybrid;

// ============================================================================
// Basic behavior
// ============================================================================

#[test]
fn hybrid_agree_with_strong_margin_gets_high_weight() {
    // Both models agree on pop 0, strong margin in pairwise
    let standard = vec![vec![-0.5, -2.0, -3.0]];
    let pairwise = vec![vec![-0.3, -2.5, -3.5]]; // margin = 2.2

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.5, 0.2, 0.2, 3.0);

    // median_margin = 2.2 (only one window), ratio = 1.0
    // w = 0.3 * 1.5 * 1.0 = 0.45
    let w = 0.45;
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "pop {s}: got {}, expected {}", result[0][s], expected
        );
    }
}

#[test]
fn hybrid_disagree_gets_flat_downweight_no_margin() {
    // Standard → pop 0, pairwise → pop 1 (disagreement)
    let standard = vec![vec![-0.5, -2.0, -3.0]];
    let pairwise = vec![vec![-2.0, -0.3, -3.5]]; // strong margin, but disagree

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.5, 0.2, 0.2, 3.0);

    // Disagreement: w = 0.3 * 0.2 = 0.06 (no margin scaling)
    let w = 0.06;
    let w_std = 1.0 - w;
    for s in 0..3 {
        let expected = w_std * standard[0][s] + w * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-12,
            "pop {s}: got {}, expected {}", result[0][s], expected
        );
    }
}

#[test]
fn hybrid_margin_modulates_agree_windows() {
    // 3 windows: all agree on pop 0, but different margin strengths
    let standard = vec![
        vec![-0.5, -2.0],  // argmax = 0
        vec![-0.5, -2.0],  // argmax = 0
        vec![-0.5, -2.0],  // argmax = 0
    ];
    let pairwise = vec![
        vec![-0.3, -0.5],  // margin = 0.2 (weak)
        vec![-0.3, -1.3],  // margin = 1.0 (medium — this is the median)
        vec![-0.3, -2.3],  // margin = 2.0 (strong)
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.0, 0.2, 0.2, 3.0);

    // Margins: [0.2, 1.0, 2.0], sorted=[0.2, 1.0, 2.0], median=1.0
    // Window 0: ratio = 0.2/1.0 = 0.2, w = 0.3 * 1.0 * 0.2 = 0.06
    // Window 1: ratio = 1.0/1.0 = 1.0, w = 0.3 * 1.0 * 1.0 = 0.3
    // Window 2: ratio = 2.0/1.0 = 2.0, w = 0.3 * 1.0 * 2.0 = 0.6

    let expected_weights = [0.06, 0.3, 0.6];
    for (t, &w) in expected_weights.iter().enumerate() {
        let w_std = 1.0 - w;
        for s in 0..2 {
            let expected = w_std * standard[t][s] + w * pairwise[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "window {t}, pop {s}: got {}, expected {}", result[t][s], expected
            );
        }
    }
}

#[test]
fn hybrid_mixed_agree_disagree() {
    // Window 0: agree (both → pop 0), Window 1: disagree (std→1, pw→0)
    let standard = vec![
        vec![-0.5, -2.0],  // argmax = 0
        vec![-2.0, -0.5],  // argmax = 1
    ];
    let pairwise = vec![
        vec![-0.3, -1.3],  // argmax = 0, margin = 1.0
        vec![-0.3, -1.3],  // argmax = 0, margin = 1.0 (disagree with std)
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.4, 1.5, 0.2, 0.2, 3.0);

    // median_margin = 1.0, both margins = 1.0 → ratio = 1.0
    // Window 0 (agree): w = 0.4 * 1.5 * 1.0 = 0.6
    // Window 1 (disagree): w = 0.4 * 0.2 = 0.08
    let weights = [0.6, 0.08];
    for (t, &w) in weights.iter().enumerate() {
        let w_std = 1.0 - w;
        for s in 0..2 {
            let expected = w_std * standard[t][s] + w * pairwise[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "window {t}, pop {s}: got {}, expected {}", result[t][s], expected
            );
        }
    }
}

// ============================================================================
// Margin clamping
// ============================================================================

#[test]
fn hybrid_margin_ratio_clamped_below() {
    // All agree, one window has very weak margin relative to median
    let standard = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pairwise = vec![
        vec![-0.3, -0.30001],  // margin ≈ 0.00001 (tiny)
        vec![-0.3, -2.3],      // margin = 2.0
        vec![-0.3, -2.3],      // margin = 2.0
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.0, 0.2, 0.2, 3.0);

    // median = 2.0, ratio for window 0 = 0.00001/2.0 ≈ 0 → clamped to 0.2
    let w0 = 0.3 * 1.0 * 0.2; // = 0.06 (clamped ratio)
    let w_std0 = 1.0 - w0;
    for s in 0..2 {
        let expected = w_std0 * standard[0][s] + w0 * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-10,
            "clamped-low: pop {s}: got {}, expected {}", result[0][s], expected
        );
    }
}

#[test]
fn hybrid_margin_ratio_clamped_above() {
    // All agree, one window has very strong margin relative to median
    let standard = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
    ];
    let pairwise = vec![
        vec![-0.3, -10.3],    // margin = 10.0 (huge)
        vec![-0.3, -0.5],     // margin = 0.2
        vec![-0.3, -0.5],     // margin = 0.2
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.0, 0.2, 0.2, 3.0);

    // median = 0.2, ratio for window 0 = 10.0/0.2 = 50 → clamped to 3.0
    let w0 = (0.3_f64 * 1.0 * 3.0).min(0.95); // = 0.9
    let w_std0 = 1.0 - w0;
    for s in 0..2 {
        let expected = w_std0 * standard[0][s] + w0 * pairwise[0][s];
        assert!(
            (result[0][s] - expected).abs() < 1e-10,
            "clamped-high: pop {s}: got {}, expected {}", result[0][s], expected
        );
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn hybrid_empty_inputs() {
    let empty: Vec<Vec<f64>> = vec![];
    let some = vec![vec![-1.0, -2.0]];
    assert!(blend_log_emissions_hybrid(&empty, &some, 0.3, 1.5, 0.2, 0.2, 3.0).is_empty());
    assert!(blend_log_emissions_hybrid(&some, &empty, 0.3, 1.5, 0.2, 0.2, 3.0) == some);
}

#[test]
fn hybrid_single_population() {
    // Single pop: margin = 0 always, ratio = 1.0 (neutral fallback)
    let standard = vec![vec![-0.5]];
    let pairwise = vec![vec![-0.3]];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.4, 1.5, 0.2, 0.2, 3.0);

    // Single pop: argmax agree (both → pop 0), margin = 0
    // No positive margins → median_margin = 1.0 (neutral), margin[0] = 0
    // margin_ratio = 0/1.0 = 0.0 → clamped to 0.2
    let w = (0.4_f64 * 1.5 * 0.2).clamp(0.0, 0.95); // = 0.12
    let expected = (1.0 - w) * (-0.5) + w * (-0.3);
    assert!(
        (result[0][0] - expected).abs() < 1e-12,
        "single pop: got {}, expected {}", result[0][0], expected
    );
}

#[test]
fn hybrid_weight_clamp_at_095() {
    // Very high base_weight × agree_scale × margin_ratio → clamp at 0.95
    let standard = vec![vec![-0.5, -2.0]];
    let pairwise = vec![vec![-0.3, -2.5]];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.9, 2.0, 0.2, 0.2, 3.0);

    // w = 0.9 * 2.0 * 1.0 = 1.8 → clamped to 0.95
    let w = 0.95;
    let w_std = 1.0 - w;
    let expected0 = w_std * (-0.5) + w * (-0.3);
    assert!(
        (result[0][0] - expected0).abs() < 1e-12,
        "clamp 0.95: got {}, expected {}", result[0][0], expected0
    );
}

#[test]
fn hybrid_nan_inf_handling() {
    let standard = vec![vec![f64::NAN, -2.0]];
    let pairwise = vec![vec![-0.3, f64::NEG_INFINITY]];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.5, 0.2, 0.2, 3.0);

    // Pop 0: std is NaN → fall back to pairwise (-0.3)
    assert_eq!(result[0][0], -0.3);
    // Pop 1: pairwise is -inf → fall back to standard (-2.0)
    assert_eq!(result[0][1], -2.0);
}

#[test]
fn hybrid_uniform_margins_equal_agreement() {
    // When all margins are identical, ratio = 1.0, so hybrid reduces to agreement
    let standard = vec![
        vec![-0.5, -1.5],
        vec![-0.5, -1.5],
    ];
    let pairwise = vec![
        vec![-0.3, -1.3], // margin = 1.0, agree
        vec![-0.3, -1.3], // margin = 1.0, agree
    ];

    let hybrid = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.5, 0.2, 0.2, 3.0);

    use hprc_ancestry_cli::blend_log_emissions_agreement;
    let agreement = blend_log_emissions_agreement(
        &standard, &pairwise, 0.3, 1.5, 0.2);

    for t in 0..2 {
        for s in 0..2 {
            assert!(
                (hybrid[t][s] - agreement[t][s]).abs() < 1e-12,
                "uniform margins: hybrid should equal agreement at [{t}][{s}]"
            );
        }
    }
}

#[test]
fn hybrid_disagree_ignores_margin_differences() {
    // Two disagree windows with different margins should get same weight
    let standard = vec![
        vec![-0.5, -2.0],  // argmax = 0
        vec![-0.5, -2.0],  // argmax = 0
    ];
    let pairwise = vec![
        vec![-3.0, -0.3],  // argmax = 1 (disagree), margin = 2.7
        vec![-1.0, -0.3],  // argmax = 1 (disagree), margin = 0.7
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.4, 1.5, 0.2, 0.2, 3.0);

    // Both disagree: w = 0.4 * 0.2 = 0.08 for both, regardless of margin
    let w = 0.08;
    let w_std = 1.0 - w;
    for t in 0..2 {
        for s in 0..2 {
            let expected = w_std * standard[t][s] + w * pairwise[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-12,
                "disagree margin-independent: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}

#[test]
fn hybrid_per_window_independence() {
    // Each window's weight is computed independently based on its own agreement
    // and the global median margin
    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // argmax = 0
        vec![-3.0, -0.5, -2.0],  // argmax = 1
        vec![-2.0, -3.0, -0.5],  // argmax = 2
    ];
    let pairwise = vec![
        vec![-0.3, -2.5, -3.5],  // argmax = 0 (agree), margin = 2.2
        vec![-0.3, -2.5, -3.5],  // argmax = 0 (disagree), margin = 2.2
        vec![-3.0, -2.5, -0.5],  // argmax = 2 (agree), margin = 2.0
    ];

    let result = blend_log_emissions_hybrid(
        &standard, &pairwise, 0.3, 1.5, 0.2, 0.2, 3.0);

    // margins = [2.2, 2.2, 2.0], sorted = [2.0, 2.2, 2.2], median = 2.2
    // Window 0 (agree): ratio = 2.2/2.2 = 1.0, w = 0.3 * 1.5 * 1.0 = 0.45
    // Window 1 (disagree): w = 0.3 * 0.2 = 0.06
    // Window 2 (agree): ratio = 2.0/2.2 ≈ 0.909, w = 0.3 * 1.5 * 0.909 ≈ 0.409
    let expected_weights: [f64; 3] = [0.45, 0.06, 0.3 * 1.5 * (2.0 / 2.2)];
    for (t, &w_raw) in expected_weights.iter().enumerate() {
        let w = w_raw.clamp(0.0, 0.95);
        let w_std = 1.0 - w;
        for s in 0..3 {
            let expected = w_std * standard[t][s] + w * pairwise[t][s];
            assert!(
                (result[t][s] - expected).abs() < 1e-10,
                "per-window independence: [{t}][{s}]: got {}, expected {}",
                result[t][s], expected
            );
        }
    }
}
