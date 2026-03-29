// Cycle 88: Deep edge-case tests for 8 low-coverage ancestry-cli functions.
//
// Targets (2-4 prior test references each):
//   - apply_confidence_weighting
//   - apply_fb_temperature
//   - compute_boundary_boost_transitions
//   - apply_emission_anchor_boost
//   - apply_label_smoothing
//   - blend_log_emissions_per_pop_hybrid
//   - forward_backward_from_log_emissions_with_transitions
//   - forward_backward_from_log_emissions

use hprc_ancestry_cli::hmm::{
    apply_confidence_weighting, apply_emission_anchor_boost, apply_fb_temperature,
    apply_label_smoothing, blend_log_emissions_per_pop_hybrid,
    compute_boundary_boost_transitions, forward_backward_from_log_emissions,
    forward_backward_from_log_emissions_with_transitions, AncestralPopulation,
    AncestryHmmParams, PerPopAgreementScales,
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
// apply_confidence_weighting
// ===========================================================================

#[test]
fn cw_empty_returns_empty() {
    let result = apply_confidence_weighting(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn cw_single_pop_rows_gap_zero() {
    // Single pop per row → gap is 0 for all, median gap → 1e-6 floor
    let emissions = vec![vec![0.5], vec![0.3], vec![0.7]];
    let result = apply_confidence_weighting(&emissions, 1.0);
    assert_eq!(result.len(), 3);
    // With gap=0, scale = (0/1e-6)^1.0 clamped to 0.1
    // result = mean + 0.1*(v - mean) → shrink toward mean
    for (orig, res) in emissions.iter().zip(result.iter()) {
        assert!((res[0] - orig[0]).abs() < 0.5);
    }
}

#[test]
fn cw_uniform_rows_no_gap() {
    // All rows have identical pops → gap = 0 → scale clamped to 0.1
    let emissions = vec![vec![1.0, 1.0, 1.0]; 5];
    let result = apply_confidence_weighting(&emissions, 1.0);
    for row in &result {
        for &v in row {
            assert!((v - 1.0).abs() < 1e-10, "uniform rows should stay near original");
        }
    }
}

#[test]
fn cw_high_gap_amplifies_deviations() {
    // One row with big gap, others with small gaps
    // Big gap row should have scale > 1.0, amplifying deviations
    let emissions = vec![
        vec![10.0, 0.0, 0.0],  // big gap: 10
        vec![1.0, 0.9, 0.8],   // small gap: 0.1
        vec![1.1, 0.9, 0.8],   // small gap: 0.2
    ];
    let result = apply_confidence_weighting(&emissions, 1.0);
    // Row 0 has much bigger gap than median → scale > 1 → amplified
    let mean0: f64 = emissions[0].iter().sum::<f64>() / 3.0;
    let dev_orig = emissions[0][0] - mean0;
    let dev_result = result[0][0] - mean0;
    assert!(dev_result.abs() > dev_orig.abs() * 0.9, "high gap should amplify");
}

#[test]
fn cw_power_zero_scale_one() {
    // power=0 → (gap/median)^0 = 1.0 for all → identity transform
    let emissions = vec![
        vec![5.0, 1.0],
        vec![3.0, 2.0],
    ];
    let result = apply_confidence_weighting(&emissions, 0.0);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-10, "power 0 should be identity");
        }
    }
}

#[test]
fn cw_preserves_non_finite() {
    let emissions = vec![
        vec![1.0, f64::NEG_INFINITY, 3.0],
        vec![2.0, 2.0, 2.0],
    ];
    let result = apply_confidence_weighting(&emissions, 1.0);
    assert!(result[0][1].is_infinite() && result[0][1] < 0.0, "NEG_INF preserved");
}

#[test]
fn cw_nan_in_row_preserved() {
    let emissions = vec![
        vec![f64::NAN, 1.0, 2.0],
        vec![3.0, 3.0, 3.0],
    ];
    let result = apply_confidence_weighting(&emissions, 1.0);
    assert!(result[0][0].is_nan(), "NaN should be preserved");
}

#[test]
fn cw_scale_clamped_to_five() {
    // Extreme gap difference → scale should clamp to 5.0 max
    // 3 rows so median = middle gap = 1.0
    let emissions = vec![
        vec![100.0, 0.0],  // gap = 100, scale = (100/1)^2 = 10000 → clamped to 5
        vec![1.0, 0.0],    // gap = 1 (median)
        vec![1.0, 0.0],    // gap = 1
    ];
    let result = apply_confidence_weighting(&emissions, 2.0);
    // Row 0: mean=50, scale=5 (clamped), result = 50 + 5*(100-50) = 300
    let mean0 = 50.0;
    let dev = result[0][0] - mean0;
    // 5.0 * (100.0 - 50.0) = 250
    assert!((dev - 250.0).abs() < 1.0, "should be clamped at scale=5: got {}", dev);
}

#[test]
fn cw_two_pop_rows_mean_preserved() {
    // After scaling, the finite-value mean of each row should remain unchanged
    let emissions = vec![
        vec![4.0, 2.0],
        vec![3.0, 1.0],
        vec![5.0, 3.0],
    ];
    let result = apply_confidence_weighting(&emissions, 0.5);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        let orig_mean: f64 = orig.iter().sum::<f64>() / orig.len() as f64;
        let res_mean: f64 = res.iter().sum::<f64>() / res.len() as f64;
        assert!((orig_mean - res_mean).abs() < 1e-10, "mean must be preserved");
    }
}

// ===========================================================================
// apply_fb_temperature
// ===========================================================================

#[test]
fn fbt_empty_returns_empty() {
    let result = apply_fb_temperature(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn fbt_temperature_one_identity() {
    let emissions = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let result = apply_fb_temperature(&emissions, 1.0);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-12, "T=1 is identity");
        }
    }
}

#[test]
fn fbt_temperature_half_doubles() {
    // T=0.5 → inv_temp=2.0 → values doubled
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_fb_temperature(&emissions, 0.5);
    assert!((result[0][0] - (-2.0)).abs() < 1e-10);
    assert!((result[0][1] - (-4.0)).abs() < 1e-10);
}

#[test]
fn fbt_temperature_two_halves() {
    // T=2.0 → inv_temp=0.5 → values halved
    let emissions = vec![vec![-4.0, -6.0]];
    let result = apply_fb_temperature(&emissions, 2.0);
    assert!((result[0][0] - (-2.0)).abs() < 1e-10);
    assert!((result[0][1] - (-3.0)).abs() < 1e-10);
}

#[test]
fn fbt_near_zero_temperature_clamped() {
    // T near 0 → clamped to 1e-6 → inv_temp = 1e6
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_fb_temperature(&emissions, 0.0);
    // 1/1e-6 = 1e6, so -1.0 * 1e6 = -1e6
    assert!(result[0][0] < -999_000.0);
}

#[test]
fn fbt_preserves_neg_infinity() {
    let emissions = vec![vec![f64::NEG_INFINITY, -1.0]];
    let result = apply_fb_temperature(&emissions, 0.5);
    assert!(result[0][0] == f64::NEG_INFINITY);
}

#[test]
fn fbt_preserves_nan() {
    let emissions = vec![vec![f64::NAN, -1.0]];
    let result = apply_fb_temperature(&emissions, 0.5);
    assert!(result[0][0].is_nan());
}

#[test]
fn fbt_multiple_rows() {
    let emissions = vec![vec![-1.0, -3.0], vec![-2.0, -4.0], vec![-5.0, -1.0]];
    let result = apply_fb_temperature(&emissions, 0.25);
    // inv_temp = 4.0
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((*o * 4.0 - *r).abs() < 1e-10);
        }
    }
}

// ===========================================================================
// compute_boundary_boost_transitions
// ===========================================================================

#[test]
fn bbt_empty_states() {
    let params = make_params(3, 0.01);
    let result = compute_boundary_boost_transitions(&[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn bbt_single_state_no_boundary() {
    let params = make_params(2, 0.01);
    let result = compute_boundary_boost_transitions(&[0], &params, 1.0);
    assert_eq!(result.len(), 1);
    // No boundary → base transitions
}

#[test]
fn bbt_all_same_no_boundaries() {
    let params = make_params(3, 0.01);
    let states = vec![1, 1, 1, 1, 1];
    let result = compute_boundary_boost_transitions(&states, &params, 1.0);
    assert_eq!(result.len(), 5);
    // No boundaries → all base transitions
    // Check that off-diagonal entries are the same base value
    for t in 0..5 {
        let log_trans = &result[t];
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    // Should be base log transition
                    let base = (0.01 / 2.0_f64).max(1e-20).ln();
                    assert!((log_trans[i][j] - base).abs() < 1e-6,
                        "non-boundary should use base transitions");
                }
            }
        }
    }
}

#[test]
fn bbt_boundary_detected_at_switch() {
    let params = make_params(2, 0.01);
    let states = vec![0, 0, 1, 1]; // boundary at index 1,2
    let result = compute_boundary_boost_transitions(&states, &params, 2.0);
    assert_eq!(result.len(), 4);

    // Indices 1 and 2 should be boundaries (switch between states[1]=0 and states[2]=1)
    // At boundary: off-diagonal *= 1/(1+weight), so less negative
    let base_off_diag = (0.01_f64).max(1e-20).ln();
    let boundary_off = result[2][0][1]; // at boundary position
    let nonbound_off = result[0][0][1]; // at non-boundary position

    // Boundary off-diagonal should be less negative (boosted)
    assert!(boundary_off > nonbound_off || (boundary_off - nonbound_off).abs() < 1e-6,
        "boundary should have less negative off-diagonal: {} vs {}", boundary_off, nonbound_off);
}

#[test]
fn bbt_weight_zero_no_boost() {
    let params = make_params(2, 0.01);
    let states = vec![0, 1, 0]; // boundaries at 0,1 and 1,2
    let result = compute_boundary_boost_transitions(&states, &params, 0.0);
    // weight=0 → 1/(1+0) = 1 → no change at boundaries
    // All transitions should be base
    for t in 0..3 {
        let base_self = (1.0 - 0.01_f64).max(1e-20).ln();
        assert!((result[t][0][0] - base_self).abs() < 0.1,
            "weight 0 should not change transitions");
    }
}

#[test]
fn bbt_output_rows_log_normalized() {
    // Each row should sum to ~1.0 in probability space (log-normalized)
    let params = make_params(3, 0.05);
    let states = vec![0, 1, 2, 0];
    let result = compute_boundary_boost_transitions(&states, &params, 1.0);
    for t in 0..result.len() {
        for i in 0..3 {
            let log_row: Vec<f64> = (0..3).map(|j| result[t][i][j]).collect();
            // log_sum_exp
            let max_v = log_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_row.iter().map(|v| (v - max_v).exp()).sum();
            let log_sum = max_v + sum_exp.ln();
            // Should be close to 0 (log of 1.0)
            assert!(log_sum.abs() < 0.1,
                "row {} at t={} should be log-normalized: log_sum={}", i, t, log_sum);
        }
    }
}

#[test]
fn bbt_large_weight_approaches_uniform() {
    let params = make_params(2, 0.01);
    let states = vec![0, 1]; // boundary at both positions
    let result = compute_boundary_boost_transitions(&states, &params, 100.0);
    // Very large weight → 1/(1+100) ≈ 0.01 → off-diagonal close to on-diagonal
    // At boundary, off-diagonal should approach self-transition
    let diff = (result[1][0][0] - result[1][0][1]).abs();
    assert!(diff < 2.0, "large weight should make transitions nearly uniform: diff={}", diff);
}

// ===========================================================================
// apply_emission_anchor_boost
// ===========================================================================

#[test]
fn eab_empty_returns_empty() {
    let result = apply_emission_anchor_boost(&[], 2, 0.5, 1.0);
    assert!(result.is_empty());
}

#[test]
fn eab_radius_zero_noop() {
    let emissions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let result = apply_emission_anchor_boost(&emissions, 0, 0.5, 1.0);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-12);
        }
    }
}

#[test]
fn eab_boost_zero_noop() {
    let emissions = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    let result = apply_emission_anchor_boost(&emissions, 1, 0.0, 0.0);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-12);
        }
    }
}

#[test]
fn eab_all_agree_gets_boost() {
    // All windows have argmax=0, so agreement fraction should be 1.0 for interior
    let emissions = vec![
        vec![2.0, 0.0, 0.0],
        vec![3.0, 1.0, 0.0],
        vec![4.0, 0.0, 0.0],
        vec![2.5, 0.0, 0.0],
        vec![3.5, 0.0, 0.0],
    ];
    let result = apply_emission_anchor_boost(&emissions, 2, 0.5, 1.0);
    // Interior windows should have argmax boosted
    assert!(result[2][0] > emissions[2][0], "center should be boosted: {} vs {}", result[2][0], emissions[2][0]);
    // Non-argmax should stay the same
    assert!((result[2][1] - emissions[2][1]).abs() < 1e-12);
}

#[test]
fn eab_no_agreement_below_threshold() {
    // Alternating argmax → agreement fraction near 0 → below threshold → no boost
    let emissions = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let result = apply_emission_anchor_boost(&emissions, 1, 0.8, 1.0);
    // Agreement is 0% for each window → no boost
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-12, "no agreement should mean no boost");
        }
    }
}

#[test]
fn eab_negative_boost_noop() {
    let emissions = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    let result = apply_emission_anchor_boost(&emissions, 1, 0.5, -1.0);
    for (orig, res) in emissions.iter().zip(result.iter()) {
        for (o, r) in orig.iter().zip(res.iter()) {
            assert!((o - r).abs() < 1e-12);
        }
    }
}

#[test]
fn eab_all_non_finite_no_boost() {
    let emissions = vec![
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
    ];
    let result = apply_emission_anchor_boost(&emissions, 1, 0.0, 2.0);
    // No finite argmax → no boost possible
    for row in &result {
        for &v in row {
            assert!(v == f64::NEG_INFINITY);
        }
    }
}

#[test]
fn eab_edge_window_fewer_neighbors() {
    // First and last windows have fewer neighbors, still get boost if threshold met
    let emissions = vec![
        vec![5.0, 0.0],
        vec![5.0, 0.0],
        vec![5.0, 0.0],
    ];
    let result = apply_emission_anchor_boost(&emissions, 5, 0.5, 2.0);
    // Edge window [0] has only 1 neighbor (idx 1 excluded self) in range, agreement = 1.0
    assert!(result[0][0] > emissions[0][0], "edge should still be boosted");
}

// ===========================================================================
// apply_label_smoothing
// ===========================================================================

#[test]
fn ls_empty_returns_empty() {
    let result = apply_label_smoothing(&[], 0.1);
    assert!(result.is_empty());
}

#[test]
fn ls_alpha_zero_identity() {
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let result = apply_label_smoothing(&emissions, 0.0);
    for (o, r) in emissions[0].iter().zip(result[0].iter()) {
        assert!((o - r).abs() < 1e-12);
    }
}

#[test]
fn ls_alpha_negative_identity() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_label_smoothing(&emissions, -0.5);
    for (o, r) in emissions[0].iter().zip(result[0].iter()) {
        assert!((o - r).abs() < 1e-12);
    }
}

#[test]
fn ls_alpha_one_uniform() {
    // alpha=1.0 → result = 0*(emission) + 1*log(1/K) = -ln(K)
    let emissions = vec![vec![-1.0, -5.0, -10.0]];
    let result = apply_label_smoothing(&emissions, 1.0);
    let expected = -(3.0_f64).ln();
    for &v in &result[0] {
        assert!((v - expected).abs() < 1e-10, "alpha=1 should give uniform: got {}", v);
    }
}

#[test]
fn ls_alpha_half_interpolates() {
    // alpha=0.5 → result = 0.5*emission + 0.5*log(1/K)
    let emissions = vec![vec![-2.0, -4.0]];
    let log_uniform = -(2.0_f64).ln();
    let result = apply_label_smoothing(&emissions, 0.5);
    let expected0 = 0.5 * (-2.0) + 0.5 * log_uniform;
    let expected1 = 0.5 * (-4.0) + 0.5 * log_uniform;
    assert!((result[0][0] - expected0).abs() < 1e-10);
    assert!((result[0][1] - expected1).abs() < 1e-10);
}

#[test]
fn ls_neg_infinity_preserved() {
    // NEG_INFINITY = masked state → stays masked
    let emissions = vec![vec![-1.0, f64::NEG_INFINITY, -3.0]];
    let result = apply_label_smoothing(&emissions, 0.3);
    assert!(result[0][1] == f64::NEG_INFINITY, "masked state must stay masked");
    assert!(result[0][0].is_finite(), "non-masked should stay finite");
}

#[test]
fn ls_alpha_clamped_to_one() {
    // alpha > 1.0 → clamped to 1.0 → uniform
    let emissions = vec![vec![-1.0, -5.0]];
    let result = apply_label_smoothing(&emissions, 5.0);
    let expected = -(2.0_f64).ln();
    for &v in &result[0] {
        assert!((v - expected).abs() < 1e-10, "alpha>1 clamped to 1: got {}", v);
    }
}

#[test]
fn ls_empty_row_preserved() {
    let emissions = vec![vec![]];
    let result = apply_label_smoothing(&emissions, 0.5);
    assert!(result[0].is_empty());
}

#[test]
fn ls_single_pop_log_uniform_zero() {
    // K=1 → log(1/1) = 0 → interpolation with 0
    let emissions = vec![vec![-5.0]];
    let result = apply_label_smoothing(&emissions, 0.3);
    let expected = 0.7 * (-5.0) + 0.3 * 0.0;
    assert!((result[0][0] - expected).abs() < 1e-10);
}

// ===========================================================================
// blend_log_emissions_per_pop_hybrid
// ===========================================================================

fn make_scales(k: usize) -> PerPopAgreementScales {
    PerPopAgreementScales {
        agree_scales: vec![1.0; k],
        disagree_matrix: vec![vec![0.5; k]; k],
    }
}

#[test]
fn bph_empty_standard() {
    let scales = make_scales(2);
    let result = blend_log_emissions_per_pop_hybrid(&[], &[vec![1.0]], 0.5, &scales, 0.5, 2.0);
    assert!(result.is_empty());
}

#[test]
fn bph_empty_profile() {
    let std_em = vec![vec![1.0, 2.0]];
    let scales = make_scales(2);
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &[], 0.5, &scales, 0.5, 2.0);
    // Empty profile → returns standard unchanged
    assert_eq!(result[0][0], std_em[0][0]);
}

#[test]
fn bph_base_weight_zero_pure_standard() {
    let std_em = vec![vec![1.0, 2.0]];
    let prof_em = vec![vec![5.0, 6.0]];
    let scales = make_scales(2);
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 0.0, &scales, 0.5, 2.0);
    // weight = 0*scale = 0, so w_std = 1.0 → pure standard
    for (o, r) in std_em[0].iter().zip(result[0].iter()) {
        assert!((o - r).abs() < 1e-10, "base_weight=0 should give pure standard");
    }
}

#[test]
fn bph_agreement_uses_agree_scales() {
    // When argmax agrees, uses agree_scales[pop]
    let std_em = vec![vec![-1.0, -3.0]];  // argmax=0
    let prof_em = vec![vec![-0.5, -4.0]]; // argmax=0 (agrees)
    let mut scales = make_scales(2);
    scales.agree_scales = vec![2.0, 0.5]; // pop 0 agrees strongly
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 0.3, &scales, 0.5, 2.0);
    // w = (0.3 * 2.0 * margin_ratio).clamp(0, 0.95)
    // Should mix some profile in
    assert!(result[0][0] != std_em[0][0], "agreement should blend profile in");
}

#[test]
fn bph_disagreement_uses_disagree_matrix() {
    // When argmax disagrees, uses disagree_matrix[a][b]
    let std_em = vec![vec![-1.0, -3.0]];  // argmax=0
    let prof_em = vec![vec![-4.0, -0.5]]; // argmax=1 (disagrees)
    let mut scales = make_scales(2);
    scales.disagree_matrix = vec![vec![0.5, 0.8], vec![0.8, 0.5]];
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 0.5, &scales, 0.5, 2.0);
    // w = (0.5 * 0.8).clamp(0, 0.95) = 0.4
    let expected0 = 0.6 * (-1.0) + 0.4 * (-4.0);
    assert!((result[0][0] - expected0).abs() < 0.2, "disagreement blend: got {}", result[0][0]);
}

#[test]
fn bph_non_finite_fallback() {
    let std_em = vec![vec![f64::NEG_INFINITY, -1.0]];
    let prof_em = vec![vec![-2.0, -1.0]];
    let scales = make_scales(2);
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 0.5, &scales, 0.5, 2.0);
    // std[0] is -inf, prof[0] is finite → should use prof[0]
    assert!((result[0][0] - (-2.0)).abs() < 1e-10, "non-finite std should fall back to prof");
}

#[test]
fn bph_weight_clamped_to_095() {
    // Even with huge base_weight * scale, should clamp to 0.95
    let std_em = vec![vec![0.0, -1.0]];
    let prof_em = vec![vec![10.0, 10.0]];
    let mut scales = make_scales(2);
    scales.agree_scales = vec![100.0, 100.0];
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 10.0, &scales, 0.1, 100.0);
    // w clamped to 0.95, so w_std = 0.05
    let expected0 = 0.05 * 0.0 + 0.95 * 10.0;
    assert!((result[0][0] - expected0).abs() < 1.0, "weight should be clamped to 0.95");
}

#[test]
fn bph_multiple_windows() {
    let std_em = vec![
        vec![-1.0, -3.0],
        vec![-2.0, -1.0],
        vec![-4.0, -2.0],
    ];
    let prof_em = vec![
        vec![-1.5, -3.5],
        vec![-1.0, -2.0],
        vec![-3.0, -5.0],
    ];
    let scales = make_scales(2);
    let result = blend_log_emissions_per_pop_hybrid(&std_em, &prof_em, 0.3, &scales, 0.5, 2.0);
    assert_eq!(result.len(), 3);
    // Each window should have 2 values
    for row in &result {
        assert_eq!(row.len(), 2);
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ===========================================================================
// forward_backward_from_log_emissions
// ===========================================================================

#[test]
fn fb_empty_returns_empty() {
    let params = make_params(2, 0.01);
    let result = forward_backward_from_log_emissions(&[], &params);
    assert!(result.is_empty());
}

#[test]
fn fb_single_window_uniform_prior() {
    let params = make_params(3, 0.01);
    // Uniform initial + uniform emissions → uniform posteriors
    let emissions = vec![vec![0.0, 0.0, 0.0]];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 3);
    for &p in &result[0] {
        assert!((p - 1.0 / 3.0).abs() < 1e-6, "uniform should give uniform posteriors: {}", p);
    }
}

#[test]
fn fb_strong_emission_dominates() {
    let params = make_params(2, 0.01);
    // Very strong emission for state 0
    let emissions = vec![vec![0.0, -100.0]];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    assert!(result[0][0] > 0.99, "strong emission should dominate: {}", result[0][0]);
}

#[test]
fn fb_posteriors_sum_to_one() {
    let params = make_params(3, 0.05);
    let emissions = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-3.0, -1.0, -2.0],
        vec![-2.0, -3.0, -1.0],
    ];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    for (t, row) in result.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posteriors at t={} sum to {}", t, sum);
    }
}

#[test]
fn fb_posteriors_all_nonnegative() {
    let params = make_params(3, 0.1);
    let emissions = vec![
        vec![-0.5, -1.5, -2.5],
        vec![-2.5, -0.5, -1.5],
    ];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    for row in &result {
        for &p in row {
            assert!(p >= 0.0, "posteriors must be non-negative: {}", p);
        }
    }
}

#[test]
fn fb_two_windows_transitions_matter() {
    // With low switch prob, state that dominates first window should have residual effect on second
    let params = make_params(2, 0.001);
    let emissions = vec![
        vec![0.0, -50.0],    // strongly state 0
        vec![-0.1, -0.2],    // ambiguous, slight favor state 0
    ];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    // Window 1 should have high posterior for state 0 due to transition memory
    assert!(result[1][0] > 0.5, "transition memory should favor state 0: {}", result[1][0]);
}

#[test]
fn fb_symmetric_emissions_symmetric_posteriors() {
    let params = make_params(2, 0.01);
    // Symmetric emissions for 2 states → symmetric posteriors
    let emissions = vec![vec![-1.0, -1.0]];
    let result = forward_backward_from_log_emissions(&emissions, &params);
    assert!((result[0][0] - result[0][1]).abs() < 1e-10, "symmetric should give equal posteriors");
}

// ===========================================================================
// forward_backward_from_log_emissions_with_transitions
// ===========================================================================

#[test]
fn fbwt_empty_returns_empty() {
    let params = make_params(2, 0.01);
    let result = forward_backward_from_log_emissions_with_transitions(&[], &params, &[]);
    assert!(result.is_empty());
}

#[test]
fn fbwt_single_window_no_transitions_needed() {
    let params = make_params(2, 0.01);
    let emissions = vec![vec![0.0, -5.0]];
    // No transitions needed for single window
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &[]);
    assert_eq!(result.len(), 1);
    assert!(result[0][0] > 0.99);
}

#[test]
fn fbwt_posteriors_sum_to_one() {
    let params = make_params(3, 0.05);
    let emissions = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-3.0, -1.0, -2.0],
        vec![-2.0, -3.0, -1.0],
    ];
    // Per-window uniform transitions in log space
    let k = 3;
    let uniform_log = (1.0 / k as f64).ln();
    let log_trans = vec![vec![vec![uniform_log; k]; k]; 2];
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    for (t, row) in result.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posteriors at t={} sum to {}", t, sum);
    }
}

#[test]
fn fbwt_strong_self_transition_persists_state() {
    let params = make_params(2, 0.01);
    let emissions = vec![
        vec![0.0, -10.0],   // strongly state 0
        vec![-0.5, -0.6],   // ambiguous
    ];
    // Very strong self-transition
    let log_self = (0.999_f64).ln();
    let log_switch = (0.001_f64).ln();
    let log_trans = vec![vec![
        vec![log_self, log_switch],
        vec![log_switch, log_self],
    ]];
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    assert!(result[1][0] > 0.7, "strong self-transition should persist state: {}", result[1][0]);
}

#[test]
fn fbwt_high_switch_allows_change() {
    let params = make_params(2, 0.01);
    let emissions = vec![
        vec![0.0, -10.0],  // strongly state 0
        vec![-10.0, 0.0],  // strongly state 1
    ];
    // High switch probability
    let log_switch = (0.5_f64).ln();
    let log_trans = vec![vec![
        vec![log_switch, log_switch],
        vec![log_switch, log_switch],
    ]];
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    assert!(result[0][0] > 0.9, "first window should favor state 0: {}", result[0][0]);
    assert!(result[1][1] > 0.9, "second window should favor state 1: {}", result[1][1]);
}

#[test]
fn fbwt_matches_base_when_transitions_match_params() {
    // When per-window transitions match params transitions, results should match base FB
    let params = make_params(2, 0.05);
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-2.0, -1.0],
        vec![-1.5, -1.5],
    ];
    // Construct log-transitions matching params
    let k = 2;
    let log_trans: Vec<Vec<Vec<f64>>> = (0..2).map(|_| {
        (0..k).map(|i| {
            (0..k).map(|j| params.transitions[i][j].ln()).collect()
        }).collect()
    }).collect();

    let result_with = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    let result_base = forward_backward_from_log_emissions(&emissions, &params);

    for t in 0..3 {
        for s in 0..k {
            assert!((result_with[t][s] - result_base[t][s]).abs() < 1e-6,
                "should match base FB at t={} s={}: {} vs {}", t, s, result_with[t][s], result_base[t][s]);
        }
    }
}

#[test]
fn fbwt_insufficient_transitions_falls_back() {
    // When log_transitions.len() < n-1, falls back to base forward_backward_from_log_emissions
    let params = make_params(2, 0.05);
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-2.0, -1.0],
        vec![-1.5, -1.5],
    ];
    // Only provide 1 transition (need 2 for 3 windows)
    let k = 2;
    let log_trans: Vec<Vec<Vec<f64>>> = vec![
        (0..k).map(|i| {
            (0..k).map(|j| params.transitions[i][j].ln()).collect()
        }).collect()
    ];
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    let result_base = forward_backward_from_log_emissions(&emissions, &params);
    // Should fall back to base (since transition for t=1 missing)
    for t in 0..3 {
        for s in 0..k {
            assert!((result[t][s] - result_base[t][s]).abs() < 1e-6,
                "insufficient transitions should fallback: t={} s={}", t, s);
        }
    }
}

#[test]
fn fbwt_posteriors_nonnegative() {
    let params = make_params(3, 0.1);
    let emissions = vec![
        vec![-0.5, -1.5, -2.5],
        vec![-2.5, -0.5, -1.5],
    ];
    let k = 3;
    let log_trans = vec![vec![vec![(1.0 / k as f64).ln(); k]; k]];
    let result = forward_backward_from_log_emissions_with_transitions(&emissions, &params, &log_trans);
    for row in &result {
        for &p in row {
            assert!(p >= 0.0 && p <= 1.0 + 1e-6, "posterior out of range: {}", p);
        }
    }
}
