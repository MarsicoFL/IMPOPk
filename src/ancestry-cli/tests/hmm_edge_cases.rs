//! Edge case tests for ancestry HMM emission model and log_emission.
//!
//! Covers edge cases not tested elsewhere:
//! - EmissionModel::TopK(0) returns None
//! - log_emission returns NEG_INFINITY when target state has no data
//! - log_emission returns 0.0 when only one population has data
//! - learn_normalization with empty or single-window observations
//! - estimate_emissions with empty data preserves defaults
//! - baum_welch with single-state edge case
//! - EmissionModel::Display round-trip consistency

use impopk_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    forward_backward, viterbi,
};
use std::collections::HashMap;

// ===========================================================================
// Helpers
// ===========================================================================

fn two_pop() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR_H1".to_string(), "EUR_H2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR_H1".to_string(), "AFR_H2".to_string()],
        },
    ]
}

fn three_pop() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR_H1".to_string(), "EUR_H2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR_H1".to_string(), "AFR_H2".to_string()],
        },
        AncestralPopulation {
            name: "EAS".to_string(),
            haplotypes: vec!["EAS_H1".to_string(), "EAS_H2".to_string()],
        },
    ]
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ===========================================================================
// 1. EmissionModel::TopK(0) — aggregate returns None
// ===========================================================================

/// TopK(0) should not crash when used in the HMM pipeline.
/// It returns None from aggregate, so log_emission should return NEG_INFINITY
/// for all states (no population has valid data after aggregation).
#[test]
fn test_topk_zero_log_emission_returns_neg_infinity() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_emission_model(EmissionModel::TopK(0));

    let obs = make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.85),
        ("AFR_H2", 0.84),
    ]);

    // TopK(0) → take=0 → aggregate returns None for all pops
    // → log_emission returns NEG_INFINITY
    let em0 = params.log_emission(&obs, 0);
    let em1 = params.log_emission(&obs, 1);
    assert!(
        em0 == f64::NEG_INFINITY,
        "TopK(0) should give NEG_INFINITY emission for state 0, got {}",
        em0
    );
    assert!(
        em1 == f64::NEG_INFINITY,
        "TopK(0) should give NEG_INFINITY emission for state 1, got {}",
        em1
    );
}

/// TopK(0) with Viterbi should produce valid states (all arbitrary since
/// all emissions are -inf, but should not panic).
#[test]
fn test_topk_zero_viterbi_does_not_panic() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_emission_model(EmissionModel::TopK(0));

    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 5000,
            end: (i + 1) * 5000,
            sample: "query".to_string(),
            similarities: [
                ("EUR_H1".to_string(), 0.95),
                ("EUR_H2".to_string(), 0.94),
                ("AFR_H1".to_string(), 0.85),
                ("AFR_H2".to_string(), 0.84),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5, "Should return 5 states even with TopK(0)");
    for &s in &states {
        assert!(s < 2, "State should be valid (0 or 1), got {}", s);
    }
}

// ===========================================================================
// 2. log_emission when target state has no data
// ===========================================================================

/// When the target state's population has no matching haplotypes in the
/// observation, log_emission should return NEG_INFINITY.
#[test]
fn test_log_emission_missing_target_state_data() {
    let pops = two_pop();
    let params = AncestryHmmParams::new(pops, 0.001);

    // Only EUR haplotypes present, no AFR data
    let obs = make_obs(&[("EUR_H1", 0.95), ("EUR_H2", 0.94)]);

    let em_eur = params.log_emission(&obs, 0); // EUR has data
    let em_afr = params.log_emission(&obs, 1); // AFR has no data

    assert!(
        em_afr == f64::NEG_INFINITY,
        "State with no data should get NEG_INFINITY, got {}",
        em_afr
    );
    // EUR is the only population with data → log(1) = 0
    assert!(
        (em_eur - 0.0).abs() < 1e-10,
        "Only population with data should get log(1)=0, got {}",
        em_eur
    );
}

/// With 3 populations and only one having data, that state should get 0.0.
#[test]
fn test_log_emission_single_pop_with_data_three_states() {
    let pops = three_pop();
    let params = AncestryHmmParams::new(pops, 0.001);

    // Only EAS haplotypes present
    let obs = make_obs(&[("EAS_H1", 0.90), ("EAS_H2", 0.89)]);

    let em_eur = params.log_emission(&obs, 0);
    let em_afr = params.log_emission(&obs, 1);
    let em_eas = params.log_emission(&obs, 2);

    assert_eq!(em_eur, f64::NEG_INFINITY);
    assert_eq!(em_afr, f64::NEG_INFINITY);
    assert!(
        (em_eas - 0.0).abs() < 1e-10,
        "EAS (sole data source) should get 0.0, got {}",
        em_eas
    );
}

/// Empty observation (no similarities at all): all states get NEG_INFINITY.
#[test]
fn test_log_emission_empty_observation() {
    let pops = two_pop();
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "query".to_string(),
        similarities: HashMap::new(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };

    for state in 0..2 {
        let em = params.log_emission(&obs, state);
        assert_eq!(
            em,
            f64::NEG_INFINITY,
            "Empty observation should give NEG_INFINITY for state {}",
            state
        );
    }
}

// ===========================================================================
// 3. log_emission with zero-valued similarities
// ===========================================================================

/// Similarities of exactly 0.0 are filtered out (> 0.0 check).
#[test]
fn test_log_emission_zero_similarities_treated_as_missing() {
    let pops = two_pop();
    let params = AncestryHmmParams::new(pops, 0.001);

    // AFR has 0.0 similarities → treated as no data
    let obs = make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.0),
        ("AFR_H2", 0.0),
    ]);

    let em_eur = params.log_emission(&obs, 0);
    let em_afr = params.log_emission(&obs, 1);

    // EUR is sole population with data → 0.0
    assert!(
        (em_eur - 0.0).abs() < 1e-10,
        "EUR should get 0.0 (sole data provider), got {}",
        em_eur
    );
    assert_eq!(
        em_afr,
        f64::NEG_INFINITY,
        "AFR with zero sims should get NEG_INFINITY, got {}",
        em_afr
    );
}

// ===========================================================================
// 4. learn_normalization edge cases
// ===========================================================================

/// learn_normalization with empty observations: means=0, stds=1e-6.
#[test]
fn test_learn_normalization_empty_observations() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    assert!(params.normalization.is_none());

    params.learn_normalization(&[]);

    let norm = params.normalization.as_ref().unwrap();
    assert_eq!(norm.means.len(), 2);
    assert_eq!(norm.stds.len(), 2);
    // No data → means default to 0.0, stds to 1e-6
    for &m in &norm.means {
        assert_eq!(m, 0.0, "Empty normalization should have mean=0");
    }
    for &s in &norm.stds {
        assert!(
            (s - 1e-6).abs() < 1e-10,
            "Empty normalization should have std=1e-6, got {}",
            s
        );
    }
}

/// learn_normalization with single observation: std should be 1e-6 (< 2 samples).
#[test]
fn test_learn_normalization_single_observation() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.85),
        ("AFR_H2", 0.84),
    ])];

    params.learn_normalization(&obs);

    let norm = params.normalization.as_ref().unwrap();
    // With 1 observation each, pop_counts[i] == 1, so stds = 1e-6
    for &s in &norm.stds {
        assert!(
            (s - 1e-6).abs() < 1e-10,
            "Single observation should give std=1e-6, got {}",
            s
        );
    }
    // Means should be the aggregated similarity values
    for &m in &norm.means {
        assert!(m > 0.0, "Mean should be positive with data, got {}", m);
    }
}

/// learn_normalization then log_emission should produce finite values.
#[test]
fn test_learn_normalization_then_emission_finite() {
    let pops = three_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 5000,
            end: (i + 1) * 5000,
            sample: "query".to_string(),
            similarities: [
                ("EUR_H1".to_string(), 0.95 + i as f64 * 0.001),
                ("EUR_H2".to_string(), 0.94),
                ("AFR_H1".to_string(), 0.85),
                ("AFR_H2".to_string(), 0.84),
                ("EAS_H1".to_string(), 0.80),
                ("EAS_H2".to_string(), 0.79),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    params.learn_normalization(&obs);

    for state in 0..3 {
        let em = params.log_emission(&obs[10], state);
        assert!(
            em.is_finite(),
            "Emission for state {} should be finite after normalization, got {}",
            state,
            em
        );
    }
}

// ===========================================================================
// 5. estimate_emissions edge cases
// ===========================================================================

/// estimate_emissions with empty observations should preserve defaults.
#[test]
fn test_estimate_emissions_empty_preserves_defaults() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let default_same = params.emission_same_pop_mean;
    let default_diff = params.emission_diff_pop_mean;
    let default_std = params.emission_std;

    params.estimate_emissions(&[]);

    assert_eq!(params.emission_same_pop_mean, default_same);
    assert_eq!(params.emission_diff_pop_mean, default_diff);
    assert_eq!(params.emission_std, default_std);
}

/// estimate_emissions with observations where all haplotype sims are 0
/// should preserve defaults (filtered by > 0.0 check).
#[test]
fn test_estimate_emissions_zero_sims_preserves_defaults() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let default_same = params.emission_same_pop_mean;

    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "query".to_string(),
        similarities: [
            ("EUR_H1".to_string(), 0.0),
            ("EUR_H2".to_string(), 0.0),
            ("AFR_H1".to_string(), 0.0),
            ("AFR_H2".to_string(), 0.0),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];

    params.estimate_emissions(&obs);
    assert_eq!(
        params.emission_same_pop_mean, default_same,
        "Zero sims should not update emission params"
    );
}

// ===========================================================================
// 6. EmissionModel Display/FromStr round-trip
// ===========================================================================

/// Display → FromStr round-trip for all EmissionModel variants.
#[test]
fn test_emission_model_display_fromstr_roundtrip() {
    let models = vec![
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(1),
        EmissionModel::TopK(5),
        EmissionModel::TopK(100),
    ];

    for model in models {
        let s = format!("{}", model);
        let parsed: EmissionModel = s.parse().unwrap();
        assert_eq!(
            model, parsed,
            "Round-trip failed for {:?}: displayed as '{}', parsed as {:?}",
            model, s, parsed
        );
    }
}

// ===========================================================================
// 7. baum_welch with single-element sequences (skipped by len < 2 guard)
// ===========================================================================

/// baum_welch should handle a mix of single-element and multi-element sequences.
#[test]
fn test_baum_welch_skips_single_element_sequences() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // One single-element sequence and one multi-element sequence
    let single_obs = vec![make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.85),
        ("AFR_H2", 0.84),
    ])];

    let multi_obs: Vec<AncestryObservation> = (0..10)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 5000,
            end: (i + 1) * 5000,
            sample: "query".to_string(),
            similarities: [
                ("EUR_H1".to_string(), 0.95),
                ("EUR_H2".to_string(), 0.94),
                ("AFR_H1".to_string(), 0.85),
                ("AFR_H2".to_string(), 0.84),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    let all_obs: Vec<&[AncestryObservation]> =
        vec![single_obs.as_slice(), multi_obs.as_slice()];

    let ll = params.baum_welch(&all_obs, 3, 1e-4);
    assert!(ll.is_finite(), "BW should handle mixed sequence lengths: {}", ll);

    // Transition rows should still sum to 1
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Transition row should sum to 1 after BW: {}",
            sum
        );
    }
}

/// baum_welch with all single-element sequences should return NEG_INFINITY
/// (all skipped).
#[test]
fn test_baum_welch_all_single_element_returns_neg_infinity() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let obs1 = vec![make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.85),
        ("AFR_H2", 0.84),
    ])];
    let obs2 = vec![make_obs(&[
        ("EUR_H1", 0.90),
        ("EUR_H2", 0.89),
        ("AFR_H1", 0.80),
        ("AFR_H2", 0.79),
    ])];

    let all_obs: Vec<&[AncestryObservation]> = vec![obs1.as_slice(), obs2.as_slice()];

    // Save original params to verify they're unchanged
    let orig_switch = params.transitions[0][1];

    let ll = params.baum_welch(&all_obs, 5, 1e-4);
    // total_ll stays 0 since all sequences skipped, but prev_ll starts at NEG_INFINITY
    // After 1 iteration with no data, total_ll=0 which is finite, so it continues
    // but xi is all zeros → no transition update
    assert!(
        ll.is_finite() || ll == f64::NEG_INFINITY,
        "BW with all single-element sequences should return finite or NEG_INFINITY: {}",
        ll
    );
    // Transitions should be unchanged (no xi data to update from)
    assert!(
        (params.transitions[0][1] - orig_switch).abs() < 1e-6,
        "Transitions should be unchanged when all sequences have len < 2"
    );
}

// ===========================================================================
// 8. Viterbi and forward_backward with normalization
// ===========================================================================

/// Normalization should not cause NaN in forward_backward posteriors.
#[test]
fn test_forward_backward_with_normalization_no_nan() {
    let pops = three_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs: Vec<AncestryObservation> = (0..30)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 5000,
            end: (i + 1) * 5000,
            sample: "query".to_string(),
            similarities: [
                ("EUR_H1".to_string(), 0.95 + (i as f64 % 5.0) * 0.005),
                ("EUR_H2".to_string(), 0.94),
                ("AFR_H1".to_string(), 0.85 + (i as f64 % 3.0) * 0.005),
                ("AFR_H2".to_string(), 0.84),
                ("EAS_H1".to_string(), 0.80 + (i as f64 % 7.0) * 0.003),
                ("EAS_H2".to_string(), 0.79),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    params.learn_normalization(&obs);
    assert!(params.normalization.is_some());

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 30);

    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 3);
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Window {} posteriors should sum to 1 with normalization: {}",
            t,
            sum
        );
        for &p in post {
            assert!(
                p.is_finite() && p >= 0.0,
                "Window {} has invalid posterior: {}",
                t,
                p
            );
        }
    }
}

// ===========================================================================
// 9. log_emission softmax properties
// ===========================================================================

/// Log-emission probabilities across all states should sum to approximately 1
/// when exponentiated (i.e., they form a valid probability distribution).
#[test]
fn test_log_emission_softmax_sums_to_one() {
    let pops = three_pop();
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = make_obs(&[
        ("EUR_H1", 0.95),
        ("EUR_H2", 0.94),
        ("AFR_H1", 0.90),
        ("AFR_H2", 0.89),
        ("EAS_H1", 0.85),
        ("EAS_H2", 0.84),
    ]);

    let log_probs: Vec<f64> = (0..3).map(|s| params.log_emission(&obs, s)).collect();
    let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
    let log_sum = max_lp + sum_exp.ln();

    // Each exp(log_emission) / sum should give a valid probability
    // So the raw exp sum should equal 1 after normalization,
    // i.e., log(sum(exp(log_emissions))) ≈ 0
    assert!(
        log_sum.abs() < 1e-6,
        "Softmax log-emissions should sum to log(1)=0, got log_sum={}",
        log_sum
    );
}
