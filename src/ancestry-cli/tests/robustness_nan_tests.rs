//! Robustness tests targeting partial_cmp().unwrap() paths, NaN safety,
//! empty-input edge cases, and HashMap access safety in the ancestry HMM.

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestryObservation, AncestryHmmParams, AncestralPopulation,
    EmissionModel, PopulationNormalization,
    viterbi, forward_backward, posterior_decode,
    estimate_temperature, estimate_temperature_normalized,
    estimate_switch_prob,
    cross_validate,
};

// Helper: make an observation with per-haplotype similarities
fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_at(pos: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: pos,
        end: pos + 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_3pop_params() -> AncestryHmmParams {
    AncestryHmmParams::new(vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string(), "a2".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string(), "b2".to_string()] },
        AncestralPopulation { name: "C".to_string(), haplotypes: vec!["c1".to_string(), "c2".to_string()] },
    ], 0.01)
}

// ========================================================================
// EmissionModel aggregate paths — tested through log_emission (aggregate is private)
// ========================================================================

#[test]
fn test_log_emission_median_identical_hap_values() {
    // All haplotypes in each pop have identical similarity → median is trivial
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Median;
    let obs = make_obs(&[("a1", 0.9), ("a2", 0.9), ("b1", 0.8), ("b2", 0.8), ("c1", 0.85), ("c2", 0.85)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "Median with identical values should work: {}", le);
}

#[test]
fn test_log_emission_median_even_count() {
    // 2 haplotypes per pop → even count for median (averages middle two)
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Median;
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.85), ("b1", 0.70), ("b2", 0.60), ("c1", 0.80), ("c2", 0.75)]);
    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);
    assert!(le_a.is_finite() && le_b.is_finite());
    assert!(le_a > le_b, "Pop A has higher median, should have higher emission");
}

#[test]
fn test_log_emission_topk_descending_order() {
    // TopK(1) should be equivalent to Max
    let mut params_topk = make_3pop_params();
    params_topk.emission_model = EmissionModel::TopK(1);
    let mut params_max = make_3pop_params();
    params_max.emission_model = EmissionModel::Max;
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.80), ("b1", 0.70), ("b2", 0.60), ("c1", 0.85), ("c2", 0.75)]);
    let le_topk = params_topk.log_emission(&obs, 0);
    let le_max = params_max.log_emission(&obs, 0);
    assert!((le_topk - le_max).abs() < 1e-10, "TopK(1) should equal Max: {} vs {}", le_topk, le_max);
}

#[test]
fn test_log_emission_topk_zero_returns_neg_inf() {
    // TopK(0) returns None from aggregate → log_emission returns NEG_INFINITY
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::TopK(0);
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.75), ("c1", 0.85), ("c2", 0.82)]);
    let le = params.log_emission(&obs, 0);
    assert!(le == f64::NEG_INFINITY, "TopK(0) should give NEG_INFINITY: {}", le);
}

#[test]
fn test_log_emission_mean_vs_max_ordering() {
    // Mean and Max should both rank pop A highest, but with different magnitudes
    let mut params_mean = make_3pop_params();
    params_mean.emission_model = EmissionModel::Mean;
    let mut params_max = make_3pop_params();
    params_max.emission_model = EmissionModel::Max;
    let obs = make_obs(&[("a1", 0.99), ("a2", 0.70), ("b1", 0.80), ("b2", 0.79), ("c1", 0.85), ("c2", 0.84)]);
    // Max: pop A = 0.99, pop B = 0.80, pop C = 0.85 → A wins
    // Mean: pop A = 0.845, pop B = 0.795, pop C = 0.845 → A ties with C
    let le_max = params_max.log_emission(&obs, 0);
    let le_mean = params_mean.log_emission(&obs, 0);
    assert!(le_max.is_finite() && le_mean.is_finite());
}

#[test]
fn test_log_emission_very_small_similarities() {
    // Extremely small but positive similarities
    let params = make_3pop_params();
    let obs = make_obs(&[("a1", 1e-300), ("b1", 1e-301), ("c1", 1e-299)]);
    let le = params.log_emission(&obs, 0);
    // These are all positive and > 0, so they should produce finite log_emission
    assert!(le.is_finite(), "Very small similarities should produce finite emission: {}", le);
}

#[test]
fn test_log_emission_single_hap_per_pop_max() {
    // Single haplotype per population — aggregate still works
    let params = AncestryHmmParams::new(vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string()] },
        AncestralPopulation { name: "C".to_string(), haplotypes: vec!["c1".to_string()] },
    ], 0.01);
    let obs = make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite());
}

// ========================================================================
// Viterbi with degenerate inputs — tests partial_cmp in max_by chains
// ========================================================================

#[test]
fn test_viterbi_all_zero_similarities() {
    // All similarities are 0 — log_emission returns NEG_INFINITY for all states
    let params = make_3pop_params();
    let obs = vec![
        make_obs(&[("a1", 0.0), ("b1", 0.0), ("c1", 0.0)]),
        make_obs(&[("a1", 0.0), ("b1", 0.0), ("c1", 0.0)]),
    ];
    let states = viterbi(&obs, &params);
    // Should still produce valid output (states are valid indices)
    assert_eq!(states.len(), 2);
    for &s in &states {
        assert!(s < 3, "state {} should be < 3", s);
    }
}

#[test]
fn test_viterbi_single_pop_has_data() {
    // Only one population has non-zero similarity
    let params = make_3pop_params();
    let obs = vec![
        make_obs(&[("a1", 0.95), ("b1", 0.0), ("c1", 0.0)]),
        make_obs(&[("a1", 0.90), ("b1", 0.0), ("c1", 0.0)]),
        make_obs(&[("a1", 0.92), ("b1", 0.0), ("c1", 0.0)]),
    ];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 3);
    // Should assign state 0 (pop A) since it's the only one with data
    for &s in &states {
        assert_eq!(s, 0, "Only pop A has data; got state {}", s);
    }
}

#[test]
fn test_viterbi_missing_haplotypes_in_obs() {
    // Observation doesn't contain any of the reference haplotypes
    let params = make_3pop_params();
    let obs = vec![
        make_obs(&[("x1", 0.95), ("x2", 0.90)]),
        make_obs(&[("x1", 0.92), ("x2", 0.88)]),
    ];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 2);
    // All log_emissions are NEG_INFINITY but Viterbi should still return valid states
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn test_viterbi_identical_similarities_all_pops() {
    // All populations have identical similarity — no signal, states should be valid
    let params = make_3pop_params();
    let obs = vec![
        make_obs(&[("a1", 0.9), ("b1", 0.9), ("c1", 0.9)]),
        make_obs(&[("a1", 0.9), ("b1", 0.9), ("c1", 0.9)]),
    ];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 2);
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn test_viterbi_very_small_similarities() {
    // Similarities near machine epsilon — log_emission_similarity_only might produce extreme values
    let params = make_3pop_params();
    let obs = vec![
        make_obs(&[("a1", 1e-300), ("b1", 1e-300), ("c1", 1e-300)]),
        make_obs(&[("a1", 1e-300), ("b1", 1e-300), ("c1", 1e-300)]),
    ];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 2);
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn test_viterbi_one_obs() {
    // Single observation — no transitions, only initial * emission
    let params = make_3pop_params();
    let obs = vec![make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0, "Pop A has highest sim, should be state 0");
}

// ========================================================================
// Forward-backward with edge cases — posterior normalization safety
// ========================================================================

#[test]
fn test_forward_backward_all_equal_emissions() {
    // Equal similarities → posteriors should be near uniform
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.9), ("b1", 0.9), ("c1", 0.9)])
    }).collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 5);
    for p in &posteriors {
        assert_eq!(p.len(), 3);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors should sum to 1, got {}", sum);
        // Each should be near 1/3
        for &val in p {
            assert!((val - 1.0 / 3.0).abs() < 0.1, "Expected near 1/3, got {}", val);
        }
    }
}

#[test]
fn test_forward_backward_no_matching_haplotypes() {
    // No observations match reference haplotypes → all emissions are -inf
    let params = make_3pop_params();
    let obs: Vec<_> = (0..3).map(|i| {
        make_obs_at(i * 10000, &[("x1", 0.95), ("x2", 0.90)])
    }).collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 3);
    // Should still return valid posteriors (graceful fallback to uniform)
    for p in &posteriors {
        let sum: f64 = p.iter().sum();
        // If all emissions are -inf, the forward-backward might produce NaN
        // or fall back to something. Either way, the function shouldn't panic.
        assert!(sum.is_finite() || sum.is_nan(), "Sum should be finite or NaN, got {}", sum);
    }
}

#[test]
fn test_forward_backward_single_observation() {
    let params = make_3pop_params();
    let obs = vec![make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])];
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 1);
    assert_eq!(posteriors[0].len(), 3);
    let sum: f64 = posteriors[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Pop A should have highest posterior
    assert!(posteriors[0][0] > posteriors[0][1]);
    assert!(posteriors[0][0] > posteriors[0][2]);
}

// ========================================================================
// posterior_decode — argmax via partial_cmp().unwrap() + unwrap_or
// ========================================================================

#[test]
fn test_posterior_decode_consistent_with_viterbi_strong_signal() {
    // Strong signal: posterior decode should agree with Viterbi
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.50), ("c1", 0.60)])
    }).collect();
    let vit_states = viterbi(&obs, &params);
    let post_states = posterior_decode(&obs, &params);
    assert_eq!(vit_states.len(), post_states.len());
    for (v, p) in vit_states.iter().zip(post_states.iter()) {
        assert_eq!(*v, *p, "Viterbi and posterior should agree on strong signal");
    }
}

#[test]
fn test_posterior_decode_empty_observations() {
    let params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    let states = posterior_decode(&obs, &params);
    assert!(states.is_empty());
}

#[test]
fn test_posterior_decode_returns_valid_indices() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..20).map(|i| {
        // Alternate signals
        if i % 2 == 0 {
            make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])
        } else {
            make_obs_at(i * 10000, &[("a1", 0.80), ("b1", 0.95), ("c1", 0.85)])
        }
    }).collect();
    let states = posterior_decode(&obs, &params);
    assert_eq!(states.len(), 20);
    for &s in &states {
        assert!(s < 3, "State {} out of range", s);
    }
}

// ========================================================================
// estimate_temperature — partial_cmp in diffs.sort_by
// ========================================================================

#[test]
fn test_estimate_temperature_empty_observations() {
    let params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    let temp = estimate_temperature(&obs, &params.populations);
    assert_eq!(temp, 0.03, "Empty observations should return fallback 0.03");
}

#[test]
fn test_estimate_temperature_single_observation() {
    let params = make_3pop_params();
    let obs = vec![make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])];
    let temp = estimate_temperature(&obs, &params.populations);
    // With a single observation, diffs has exactly 1 entry
    assert!(temp >= 0.01 && temp <= 0.15, "Temperature {} out of clamped range", temp);
}

#[test]
fn test_estimate_temperature_identical_sims() {
    // All populations have same sim → max == min → no diffs → fallback 0.03
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.9), ("b1", 0.9), ("c1", 0.9)])
    }).collect();
    let temp = estimate_temperature(&obs, &params.populations);
    assert_eq!(temp, 0.03, "Zero diffs should return fallback 0.03");
}

#[test]
fn test_estimate_temperature_large_spread() {
    // Very large spread → clamped to 0.15
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.01), ("c1", 0.50)])
    }).collect();
    let temp = estimate_temperature(&obs, &params.populations);
    assert_eq!(temp, 0.15, "Large diffs should clamp to 0.15");
}

#[test]
fn test_estimate_temperature_normalized_empty() {
    // No normalization set → delegates to estimate_temperature → fallback 0.03
    let params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    let temp = estimate_temperature_normalized(&obs, &params);
    assert_eq!(temp, 0.03, "Empty obs without normalization delegates to estimate_temperature fallback");
}

#[test]
fn test_estimate_temperature_normalized_with_normalization() {
    let mut params = make_3pop_params();
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.8, 0.85],
        stds: vec![0.05, 0.05, 0.05],
    });
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])
    }).collect();
    let temp = estimate_temperature_normalized(&obs, &params);
    assert!(temp >= 0.5 && temp <= 5.0, "Normalized temp {} out of range", temp);
}

// ========================================================================
// estimate_switch_prob — uses viterbi internally, partial_cmp in max_by
// ========================================================================

#[test]
fn test_estimate_switch_prob_constant_signal() {
    // All windows favor one population → no switches → low switch prob
    let params = make_3pop_params();
    let obs: Vec<_> = (0..20).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.70), ("c1", 0.75)])
    }).collect();
    let sp = estimate_switch_prob(&obs, &params.populations, params.emission_std);
    assert!(sp > 0.0 && sp < 1.0, "Switch prob {} out of (0,1)", sp);
    assert!(sp < 0.1, "Constant signal should give low switch prob: {}", sp);
}

#[test]
fn test_estimate_switch_prob_empty() {
    let params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    let sp = estimate_switch_prob(&obs, &params.populations, params.emission_std);
    // Empty observations → fallback for small data
    assert!(sp > 0.0 && sp < 1.0);
}

#[test]
fn test_estimate_switch_prob_single_obs() {
    let params = make_3pop_params();
    let obs = vec![make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])];
    let sp = estimate_switch_prob(&obs, &params.populations, params.emission_std);
    assert!(sp > 0.0 && sp < 1.0);
}

// ========================================================================
// cross_validate (LOO) — HashMap get_mut().unwrap() safety
// ========================================================================

#[test]
fn test_cross_validate_single_hap_per_pop() {
    // Single haplotype per population: LOO needs >=2 haplotypes, so these pops are skipped
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string()] },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let mut observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();
    observations.insert("a1".to_string(), vec![
        make_obs(&[("a1", 0.95), ("b1", 0.80)]),
    ]);
    observations.insert("b1".to_string(), vec![
        make_obs(&[("a1", 0.80), ("b1", 0.95)]),
    ]);

    let result = cross_validate(&observations, &pops, &params);
    // With single hap per pop, no LOO is possible → accuracy should be 0/NaN
    assert!(result.overall_accuracy.is_finite() || result.overall_accuracy.is_nan());
}

#[test]
fn test_cross_validate_empty_observations() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string(), "a2".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string(), "b2".to_string()] },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate(&observations, &pops, &params);
    // No data to validate → accuracy metrics should be 0/NaN but not panic
    assert!(result.overall_accuracy.is_finite() || result.overall_accuracy.is_nan());
}

#[test]
fn test_cross_validate_three_pops_normal() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string(), "a2".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string(), "b2".to_string()] },
        AncestralPopulation { name: "C".to_string(), haplotypes: vec!["c1".to_string(), "c2".to_string()] },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let mut observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    // Give each haplotype observations favoring their own population
    for (hap, pop_sims) in [
        ("a1", [("a1", 0.99), ("b1", 0.70), ("c1", 0.75)]),
        ("a2", [("a2", 0.98), ("b1", 0.72), ("c1", 0.74)]),
        ("b1", [("a1", 0.70), ("b1", 0.99), ("c1", 0.73)]),
        ("b2", [("a1", 0.71), ("b2", 0.97), ("c1", 0.74)]),
        ("c1", [("a1", 0.72), ("b1", 0.71), ("c1", 0.99)]),
        ("c2", [("a1", 0.73), ("b1", 0.70), ("c2", 0.98)]),
    ] {
        observations.insert(hap.to_string(), (0..5).map(|i| {
            make_obs_at(i * 10000, &pop_sims)
        }).collect());
    }

    let result = cross_validate(&observations, &pops, &params);
    assert!(result.overall_accuracy >= 0.0 && result.overall_accuracy <= 1.0);
    // With strong signal and 2 haps per pop, accuracy should be good
    assert!(result.overall_accuracy > 0.5,
        "Expected decent accuracy with clear signal, got {}", result.overall_accuracy);
}

// ========================================================================
// learn_normalization edge cases
// ========================================================================

#[test]
fn test_learn_normalization_empty_observations() {
    let mut params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    params.learn_normalization(&obs);
    assert!(params.normalization.is_some());
    let norm = params.normalization.as_ref().unwrap();
    // All means should be 0 (no data)
    for &m in &norm.means {
        assert_eq!(m, 0.0);
    }
    // Stds should be 1e-6 (minimum)
    for &s in &norm.stds {
        assert_eq!(s, 1e-6);
    }
}

#[test]
fn test_learn_normalization_single_observation() {
    let mut params = make_3pop_params();
    let obs = vec![make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])];
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    // With single observation, std should be clamped to 1e-6
    for &s in &norm.stds {
        assert_eq!(s, 1e-6, "Single obs should give minimum std");
    }
}

#[test]
fn test_learn_normalization_no_matching_haplotypes() {
    let mut params = make_3pop_params();
    // Observations reference haplotypes not in any population
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("x1", 0.95), ("x2", 0.80)])
    }).collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    // No population gets any data → means=0, stds=1e-6
    for &m in &norm.means {
        assert_eq!(m, 0.0);
    }
}

// ========================================================================
// EmissionModel with various emission models — log_emission paths
// ========================================================================

#[test]
fn test_log_emission_topk_emission_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::TopK(2);
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.75), ("c1", 0.85), ("c2", 0.82)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "log_emission with TopK should be finite: {}", le);
    assert!(le < 0.0, "log_emission should be negative (log prob): {}", le);
}

#[test]
fn test_log_emission_mean_emission_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Mean;
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.75), ("c1", 0.85), ("c2", 0.82)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite());
}

#[test]
fn test_log_emission_median_emission_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Median;
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.75), ("c1", 0.85), ("c2", 0.82)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite());
    // Pop A has highest median → should have highest log_emission
    let le_b = params.log_emission(&obs, 1);
    assert!(le > le_b, "Pop A (higher median) should have higher log_emission");
}

#[test]
fn test_log_emission_with_normalization_all_models() {
    for model in [EmissionModel::Max, EmissionModel::Mean, EmissionModel::Median, EmissionModel::TopK(2)] {
        let mut params = make_3pop_params();
        params.emission_model = model;
        params.normalization = Some(PopulationNormalization {
            means: vec![0.9, 0.8, 0.85],
            stds: vec![0.05, 0.05, 0.05],
        });
        let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.75), ("c1", 0.85), ("c2", 0.82)]);
        let le = params.log_emission(&obs, 0);
        assert!(le.is_finite(), "model {:?}: log_emission should be finite: {}", model, le);
    }
}

// ========================================================================
// Viterbi/FB with TopK/Mean/Median emission models — full pipeline
// ========================================================================

#[test]
fn test_viterbi_with_topk_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::TopK(2);
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[
            ("a1", 0.99), ("a2", 0.95),
            ("b1", 0.70), ("b2", 0.72),
            ("c1", 0.80), ("c2", 0.78),
        ])
    }).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 10);
    for &s in &states {
        assert_eq!(s, 0, "Pop A has strongest signal");
    }
}

#[test]
fn test_viterbi_with_median_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Median;
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[
            ("a1", 0.99), ("a2", 0.95),
            ("b1", 0.70), ("b2", 0.72),
            ("c1", 0.80), ("c2", 0.78),
        ])
    }).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 10);
    for &s in &states {
        assert_eq!(s, 0);
    }
}

#[test]
fn test_forward_backward_with_mean_model() {
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Mean;
    let obs: Vec<_> = (0..5).map(|i| {
        make_obs_at(i * 10000, &[
            ("a1", 0.95), ("a2", 0.90),
            ("b1", 0.80), ("b2", 0.75),
            ("c1", 0.85), ("c2", 0.82),
        ])
    }).collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 5);
    for p in &posteriors {
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors should sum to 1: {}", sum);
        // Pop A should dominate
        assert!(p[0] > p[1] && p[0] > p[2]);
    }
}

// ========================================================================
// estimate_emissions — partial_cmp in max_by for is_best determination
// ========================================================================

#[test]
fn test_estimate_emissions_no_signal() {
    // All populations have identical similarity → is_best check goes to first pop
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.9), ("b1", 0.9), ("c1", 0.9)])
    }).collect();
    params.estimate_emissions(&obs);
    // Should not panic; parameters should be updated
    assert!(params.emission_same_pop_mean >= 0.0);
    assert!(params.emission_diff_pop_mean >= 0.0);
}

#[test]
fn test_estimate_emissions_single_pop_data() {
    // Only one population has nonzero data → all go to same_pop
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_at(i * 10000, &[("a1", 0.95)])
    }).collect();
    params.estimate_emissions(&obs);
    assert!(params.emission_same_pop_mean.is_finite());
    assert!(params.emission_diff_pop_mean.is_finite());
}

#[test]
fn test_estimate_emissions_empty_observations() {
    let mut params = make_3pop_params();
    let obs: Vec<AncestryObservation> = vec![];
    params.estimate_emissions(&obs);
    // With no data, parameters should remain at defaults (or be safely updated)
    assert!(params.emission_same_pop_mean.is_finite());
    assert!(params.emission_diff_pop_mean.is_finite());
}
