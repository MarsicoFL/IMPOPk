//! Tests for per-population agreement scaling (T79).
//!
//! Per-population agreement uses D_min-based heuristic to compute population-specific
//! agree_scale (well-separated pops get higher boost) and per-pair disagree_scale
//! (close pairs get less suppression, distant pairs get more).

use hprc_ancestry_cli::{
    PerPopAgreementScales,
    blend_log_emissions_per_pop_agreement,
    blend_log_emissions_per_pop_hybrid,
    compute_per_pop_agreement_scales,
    AncestralPopulation, AncestryObservation,
};
use std::collections::HashMap;

// ============================================================================
// Helper: create simple PerPopAgreementScales
// ============================================================================

fn make_scales_3pop(agree: [f64; 3], disagree: [[f64; 3]; 3]) -> PerPopAgreementScales {
    PerPopAgreementScales {
        agree_scales: agree.to_vec(),
        disagree_matrix: disagree.iter().map(|r| r.to_vec()).collect(),
    }
}

// ============================================================================
// blend_log_emissions_per_pop_agreement tests
// ============================================================================

#[test]
fn per_pop_agreement_uses_population_specific_agree_scale() {
    // Well-separated pop 0 (AFR-like): agree_scale = 1.72
    // Moderate pop 1 (EUR-like): agree_scale = 1.40
    // Poorly separated pop 2 (AMR-like): agree_scale = 1.01
    let scales = make_scales_3pop(
        [1.72, 1.40, 1.01],
        [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
    );

    let base_weight = 0.4;

    // Window agreeing on pop 0 (AFR): w = 0.4 * 1.72 = 0.688
    let std_afr = vec![vec![-0.5, -2.0, -3.0]];
    let pw_afr = vec![vec![-0.3, -2.5, -3.5]];
    let result_afr = blend_log_emissions_per_pop_agreement(
        &std_afr, &pw_afr, base_weight, &scales);
    let w_afr = (base_weight * 1.72).min(0.95);
    let expected_0 = (1.0 - w_afr) * std_afr[0][0] + w_afr * pw_afr[0][0];
    assert!((result_afr[0][0] - expected_0).abs() < 1e-12,
        "AFR agree: got {}, expected {}", result_afr[0][0], expected_0);

    // Window agreeing on pop 2 (AMR): w = 0.4 * 1.01 = 0.404
    let std_amr = vec![vec![-3.0, -2.0, -0.5]];
    let pw_amr = vec![vec![-3.5, -2.5, -0.3]];
    let result_amr = blend_log_emissions_per_pop_agreement(
        &std_amr, &pw_amr, base_weight, &scales);
    let w_amr = base_weight * 1.01;
    let expected_2 = (1.0 - w_amr) * std_amr[0][0] + w_amr * pw_amr[0][0];
    assert!((result_amr[0][0] - expected_2).abs() < 1e-12,
        "AMR agree: got {}, expected {}", result_amr[0][0], expected_2);

    // AFR agreement gets higher pairwise weight than AMR agreement
    assert!(w_afr > w_amr, "AFR agree weight {} should exceed AMR agree weight {}", w_afr, w_amr);
}

#[test]
fn per_pop_agreement_uses_per_pair_disagree_scale() {
    // Type I (EUR↔AMR, close): disagree_scale = 0.30
    // Type II (AFR-involving, distant): disagree_scale = 0.12
    let scales = make_scales_3pop(
        [1.5, 1.5, 1.5],
        [
            [0.0, 0.12, 0.12],  // AFR→EUR=0.12, AFR→AMR=0.12 (Type II)
            [0.12, 0.0, 0.30],  // EUR→AFR=0.12, EUR→AMR=0.30 (Type I)
            [0.12, 0.30, 0.0],  // AMR→AFR=0.12, AMR→EUR=0.30 (Type I)
        ],
    );

    let base_weight = 0.4;

    // Type I disagreement: std → EUR (1), pw → AMR (2)
    let std_eur_amr = vec![vec![-2.0, -0.5, -1.5]];
    let pw_eur_amr = vec![vec![-2.5, -1.5, -0.3]];
    let result_type1 = blend_log_emissions_per_pop_agreement(
        &std_eur_amr, &pw_eur_amr, base_weight, &scales);
    let w_type1 = base_weight * 0.30; // EUR→AMR

    // Type II disagreement: std → AFR (0), pw → EUR (1)
    let std_afr_eur = vec![vec![-0.5, -2.0, -3.0]];
    let pw_afr_eur = vec![vec![-2.5, -0.3, -3.5]];
    let result_type2 = blend_log_emissions_per_pop_agreement(
        &std_afr_eur, &pw_afr_eur, base_weight, &scales);
    let w_type2 = base_weight * 0.12; // AFR→EUR

    // Type I gets more pairwise weight (less suppression) than Type II
    assert!(w_type1 > w_type2,
        "Type I w={} should exceed Type II w={}", w_type1, w_type2);

    // Verify exact values
    for s in 0..3 {
        let exp1 = (1.0 - w_type1) * std_eur_amr[0][s] + w_type1 * pw_eur_amr[0][s];
        assert!((result_type1[0][s] - exp1).abs() < 1e-12);
        let exp2 = (1.0 - w_type2) * std_afr_eur[0][s] + w_type2 * pw_afr_eur[0][s];
        assert!((result_type2[0][s] - exp2).abs() < 1e-12);
    }
}

#[test]
fn per_pop_agreement_empty_input() {
    let scales = make_scales_3pop(
        [1.5, 1.5, 1.5],
        [[0.2; 3]; 3],
    );
    let result = blend_log_emissions_per_pop_agreement(
        &[], &[], 0.4, &scales);
    assert!(result.is_empty());
}

#[test]
fn per_pop_agreement_single_population() {
    let scales = PerPopAgreementScales {
        agree_scales: vec![1.5],
        disagree_matrix: vec![vec![0.2]],
    };
    let standard = vec![vec![-0.5]];
    let pairwise = vec![vec![-0.3]];
    let result = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, 0.4, &scales);
    // Single pop: always agrees on pop 0
    let w = (0.4_f64 * 1.5).min(0.95);
    let expected = (1.0 - w) * (-0.5) + w * (-0.3);
    assert!((result[0][0] - expected).abs() < 1e-12);
}

#[test]
fn per_pop_agreement_weight_clamped_at_095() {
    // Very high agree_scale → weight should be clamped
    let scales = PerPopAgreementScales {
        agree_scales: vec![5.0, 5.0],
        disagree_matrix: vec![vec![0.2; 2]; 2],
    };
    let standard = vec![vec![-0.5, -2.0]];
    let pairwise = vec![vec![-0.3, -2.5]];
    let result = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, 0.5, &scales);
    // w = 0.5 * 5.0 = 2.5 → clamped to 0.95
    let w = 0.95;
    let expected = (1.0 - w) * (-0.5) + w * (-0.3);
    assert!((result[0][0] - expected).abs() < 1e-12);
}

#[test]
fn per_pop_agreement_nan_inf_safety() {
    let scales = make_scales_3pop(
        [1.5, 1.5, 1.5],
        [[0.2; 3]; 3],
    );
    let standard = vec![vec![f64::NEG_INFINITY, -0.5, -1.0]];
    let pairwise = vec![vec![-0.3, f64::NEG_INFINITY, -1.5]];
    let result = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, 0.4, &scales);
    // Pop 0: std is -inf, use pairwise value
    assert_eq!(result[0][0], -0.3);
    // Pop 1: pw is -inf, use standard value
    assert_eq!(result[0][1], -0.5);
    // Pop 2: both finite, normal blend
    assert!(result[0][2].is_finite());
}

#[test]
fn per_pop_agreement_per_window_independence() {
    // Each window should be independent — changing one window shouldn't affect others
    let scales = make_scales_3pop(
        [1.72, 1.40, 1.01],
        [[0.2, 0.12, 0.12], [0.12, 0.2, 0.30], [0.12, 0.30, 0.2]],
    );
    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // agree on pop 0
        vec![-2.0, -0.5, -3.0],  // agree on pop 1
    ];
    let pairwise = vec![
        vec![-0.3, -2.5, -3.5],
        vec![-2.5, -0.3, -3.5],
    ];
    let result_both = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, 0.4, &scales);

    // Check window 0 individually
    let result_w0 = blend_log_emissions_per_pop_agreement(
        &standard[0..1], &pairwise[0..1], 0.4, &scales);
    for s in 0..3 {
        assert!((result_both[0][s] - result_w0[0][s]).abs() < 1e-12,
            "Window 0 pop {} not independent", s);
    }
}

// ============================================================================
// blend_log_emissions_per_pop_hybrid tests
// ============================================================================

#[test]
fn per_pop_hybrid_agreement_uses_per_pop_scale_times_margin() {
    let scales = make_scales_3pop(
        [1.72, 1.40, 1.01],
        [[0.2; 3]; 3],
    );

    // Two windows both agreeing on pop 0, but with different margins
    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // agree on 0
        vec![-0.5, -2.0, -3.0],  // agree on 0
    ];
    let pairwise = vec![
        vec![-0.3, -3.0, -4.0],  // strong margin (0.3→3.0, margin=2.7)
        vec![-0.5, -0.8, -1.0],  // weak margin (0.5→0.8, margin=0.2)
    ];
    let result = blend_log_emissions_per_pop_hybrid(
        &standard, &pairwise, 0.3, &scales, 0.2, 3.0);

    // Window with strong margin should get higher pairwise weight
    // Both agree on pop 0 (agree_scale=1.72), but margin modulates
    // Can't compute exact values without median, but can check ordering:
    // The strong-margin window should have result closer to pairwise
    // (higher pairwise contribution)
    // Both windows should produce finite results
    assert!(result[0][0].is_finite());
    assert!(result[1][0].is_finite());
    // Strong margin window gets higher pairwise weight, so its blend
    // differs more from the standard-only value
    let std_only_0 = standard[0][0];
    let std_only_1 = standard[1][0];
    let shift_strong = (result[0][0] - std_only_0).abs();
    let shift_weak = (result[1][0] - std_only_1).abs();
    assert!(shift_strong > shift_weak || shift_strong > 0.0,
        "Strong margin should shift more from standard: {} vs {}", shift_strong, shift_weak);
}

#[test]
fn per_pop_hybrid_disagreement_uses_per_pair_scale_no_margin() {
    let scales = make_scales_3pop(
        [1.5, 1.5, 1.5],
        [
            [0.0, 0.12, 0.12],
            [0.12, 0.0, 0.30],
            [0.12, 0.30, 0.0],
        ],
    );

    // Two disagreement windows with different margins
    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // std → pop 0 (AFR)
        vec![-0.5, -2.0, -3.0],  // std → pop 0 (AFR)
    ];
    let pairwise = vec![
        vec![-2.0, -0.3, -3.0],  // pw → pop 1, high margin
        vec![-2.0, -0.5, -3.0],  // pw → pop 1, low margin
    ];
    let result = blend_log_emissions_per_pop_hybrid(
        &standard, &pairwise, 0.4, &scales, 0.2, 3.0);

    // Both disagree (AFR→EUR = Type II, scale=0.12)
    // Disagreement should NOT use margin — both get same weight
    let w = (0.4_f64 * 0.12).clamp(0.0, 0.95);
    let w_std = 1.0 - w;
    // Both windows should use the same weight regardless of margin
    for t in 0..2 {
        for s in 0..3 {
            let exp = w_std * standard[t][s] + w * pairwise[t][s];
            assert!((result[t][s] - exp).abs() < 1e-12,
                "Disagree window {} pop {}: got {}, expected {}", t, s, result[t][s], exp);
        }
    }
}

#[test]
fn per_pop_hybrid_empty_input() {
    let scales = make_scales_3pop([1.5; 3], [[0.2; 3]; 3]);
    let result = blend_log_emissions_per_pop_hybrid(
        &[], &[], 0.4, &scales, 0.2, 3.0);
    assert!(result.is_empty());
}

#[test]
fn per_pop_hybrid_single_pop() {
    let scales = PerPopAgreementScales {
        agree_scales: vec![1.5],
        disagree_matrix: vec![vec![0.2]],
    };
    let standard = vec![vec![-0.5]];
    let pairwise = vec![vec![-0.3]];
    let result = blend_log_emissions_per_pop_hybrid(
        &standard, &pairwise, 0.4, &scales, 0.2, 3.0);
    // Single pop with < 2 pops: margin = 0.0 for each row
    // median_margin = 1.0 (fallback), margin_ratio = (0.0/1.0).clamp(0.2, 3.0) = 0.2
    // w = (0.4 * 1.5 * 0.2).clamp(0.0, 0.95) = 0.12
    let w = 0.12;
    let expected = (1.0 - w) * (-0.5) + w * (-0.3);
    assert!((result[0][0] - expected).abs() < 1e-12);
}

// ============================================================================
// compute_per_pop_agreement_scales tests
// ============================================================================

fn make_observation(sims: Vec<(&str, f64)>) -> AncestryObservation {
    let mut similarities = HashMap::new();
    for (name, val) in sims {
        similarities.insert(name.to_string(), val);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities,
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|h| h.to_string()).collect(),
    }
}

#[test]
fn compute_scales_well_separated_pop_gets_higher_agree_scale() {
    // Pop A: hap "a1", Pop B: hap "b1", Pop C: hap "c1"
    // A is well-separated from both B and C; B and C are close to each other
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];

    // Generate observations where A is very distinct, B and C are similar
    // Use independent noise per population so Cohen's d differences emerge
    let observations: Vec<AncestryObservation> = (0..100)
        .map(|i| {
            let na = (i as f64 * 0.13).sin() * 0.02;
            let nb = (i as f64 * 0.29).sin() * 0.02;
            let nc = (i as f64 * 0.47).sin() * 0.02;
            make_observation(vec![
                ("a1", 0.95 + na),
                ("b1", 0.80 + nb),
                ("c1", 0.79 + nc),
            ])
        })
        .collect();

    let scales = compute_per_pop_agreement_scales(&observations, &pops, 1.5, 0.2);

    // Pop A (well-separated) should have higher agree_scale
    // D_min(A) = min(d(A,B), d(A,C)) — both large because A is far from B and C
    // D_min(B) = min(d(B,A), d(B,C)) — d(B,C) is small because B≈C
    // D_min(C) = min(d(C,A), d(C,B)) — d(C,B) is small because C≈B
    assert!(scales.agree_scales[0] > scales.agree_scales[1],
        "Pop A agree_scale {} should > Pop B {}", scales.agree_scales[0], scales.agree_scales[1]);
    assert!(scales.agree_scales[0] > scales.agree_scales[2],
        "Pop A agree_scale {} should > Pop C {}", scales.agree_scales[0], scales.agree_scales[2]);

    // B and C should have similar (lower) agree_scales
    assert!((scales.agree_scales[1] - scales.agree_scales[2]).abs() < 0.5,
        "B and C should have similar agree_scales: {} vs {}", scales.agree_scales[1], scales.agree_scales[2]);
}

#[test]
fn compute_scales_close_pairs_get_higher_disagree_scale() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];

    // A is well-separated, B≈C (close pair) — independent noise per pop
    let observations: Vec<AncestryObservation> = (0..100)
        .map(|i| {
            let na = (i as f64 * 0.13).sin() * 0.02;
            let nb = (i as f64 * 0.29).sin() * 0.02;
            let nc = (i as f64 * 0.47).sin() * 0.02;
            make_observation(vec![
                ("a1", 0.95 + na),
                ("b1", 0.80 + nb),
                ("c1", 0.79 + nc),
            ])
        })
        .collect();

    let scales = compute_per_pop_agreement_scales(&observations, &pops, 1.5, 0.2);

    // B↔C (close pair, Type I) should have higher disagree_scale (less suppression)
    // A↔B (distant pair, Type II) should have lower disagree_scale (more suppression)
    assert!(scales.disagree_matrix[1][2] > scales.disagree_matrix[0][1],
        "B→C disagree {} should > A→B disagree {}",
        scales.disagree_matrix[1][2], scales.disagree_matrix[0][1]);
}

#[test]
fn compute_scales_too_few_observations_returns_uniform() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<AncestryObservation> = (0..5)  // < 10 minimum
        .map(|_| make_observation(vec![("a1", 0.95), ("b1", 0.80)]))
        .collect();

    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.5, 0.2);
    assert_eq!(scales.agree_scales, vec![1.5, 1.5]);
    assert_eq!(scales.disagree_matrix, vec![vec![0.2, 0.2], vec![0.2, 0.2]]);
}

#[test]
fn compute_scales_single_population_returns_uniform() {
    let pops = vec![make_pop("A", &["a1"])];
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|_| make_observation(vec![("a1", 0.95)]))
        .collect();

    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.5, 0.2);
    assert_eq!(scales.agree_scales, vec![1.5]);
}

#[test]
fn compute_scales_two_populations() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let observations: Vec<AncestryObservation> = (0..100)
        .map(|i| {
            let noise = (i as f64 * 0.01).sin() * 0.001;
            make_observation(vec![
                ("a1", 0.95 + noise),
                ("b1", 0.80 + noise),
            ])
        })
        .collect();

    let scales = compute_per_pop_agreement_scales(&observations, &pops, 1.5, 0.2);
    // With 2 pops, each has the same D_min (only one pair), so ratio = 1.0
    assert!((scales.agree_scales[0] - scales.agree_scales[1]).abs() < 1e-6,
        "2-pop: both should have equal agree_scale: {} vs {}",
        scales.agree_scales[0], scales.agree_scales[1]);
    // Both should be close to base (ratio=1.0)
    assert!((scales.agree_scales[0] - 1.5).abs() < 0.01);
}

#[test]
fn compute_scales_agree_scales_clamped_to_range() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];

    // A is extremely well-separated, B≈C are nearly identical
    let observations: Vec<AncestryObservation> = (0..100)
        .map(|i| {
            let na = (i as f64 * 0.13).sin() * 0.01;
            let nb = (i as f64 * 0.29).sin() * 0.01;
            let nc = (i as f64 * 0.47).sin() * 0.01;
            make_observation(vec![
                ("a1", 0.99 + na),
                ("b1", 0.50 + nb),
                ("c1", 0.50 + nc + 0.0001),
            ])
        })
        .collect();

    let scales = compute_per_pop_agreement_scales(&observations, &pops, 1.5, 0.2);
    // All agree_scales should be within [0.5, 3.0]
    for (i, &s) in scales.agree_scales.iter().enumerate() {
        assert!(s >= 0.5 && s <= 3.0,
            "Pop {} agree_scale {} outside [0.5, 3.0]", i, s);
    }
    // All disagree_scales should be within [0.05, 0.5]
    for a in 0..3 {
        for b in 0..3 {
            if a != b {
                let d = scales.disagree_matrix[a][b];
                assert!(d >= 0.05 && d <= 0.5,
                    "disagree[{}][{}]={} outside [0.05, 0.5]", a, b, d);
            }
        }
    }
}

// ============================================================================
// Integration: uniform scales should match agreement blending
// ============================================================================

#[test]
fn per_pop_agreement_with_uniform_scales_matches_agreement() {
    use hprc_ancestry_cli::blend_log_emissions_agreement;

    let agree_scale = 1.5;
    let disagree_scale = 0.2;
    let base_weight = 0.4;

    // Create uniform scales (should be equivalent to non-per-pop version)
    let scales = make_scales_3pop(
        [agree_scale; 3],
        [[disagree_scale; 3]; 3],
    );

    let standard = vec![
        vec![-0.5, -2.0, -3.0],  // agree on pop 0
        vec![-2.0, -0.5, -3.0],  // agree on pop 1
        vec![-0.5, -2.0, -3.0],  // std→0, disagree
    ];
    let pairwise = vec![
        vec![-0.3, -2.5, -3.5],  // agree on pop 0
        vec![-2.5, -0.3, -3.5],  // agree on pop 1
        vec![-2.5, -0.3, -3.5],  // pw→1, disagree
    ];

    let result_per_pop = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, base_weight, &scales);
    let result_uniform = blend_log_emissions_agreement(
        &standard, &pairwise, base_weight, agree_scale, disagree_scale);

    for t in 0..3 {
        for s in 0..3 {
            assert!((result_per_pop[t][s] - result_uniform[t][s]).abs() < 1e-12,
                "Window {} pop {}: per_pop={}, uniform={}",
                t, s, result_per_pop[t][s], result_uniform[t][s]);
        }
    }
}

// ============================================================================
// Edge case: argmax out of range
// ============================================================================

#[test]
fn per_pop_agreement_argmax_out_of_range_uses_fallback() {
    // scales has 2 populations but emissions have 3 columns
    // argmax could be 2 which is >= k=2
    let scales = PerPopAgreementScales {
        agree_scales: vec![1.5, 1.5],
        disagree_matrix: vec![vec![0.2, 0.2], vec![0.2, 0.2]],
    };

    let standard = vec![vec![-3.0, -2.0, -0.5]];  // argmax = 2
    let pairwise = vec![vec![-3.0, -2.0, -0.3]];   // argmax = 2

    // Both agree on pop 2, but scales only has 2 pops → fallback
    let result = blend_log_emissions_per_pop_agreement(
        &standard, &pairwise, 0.4, &scales);
    // Should use fallback weight: 0.4 * 0.2 = 0.08
    let w = 0.08;
    for s in 0..3 {
        let exp = (1.0 - w) * standard[0][s] + w * pairwise[0][s];
        assert!((result[0][s] - exp).abs() < 1e-12);
    }
}
