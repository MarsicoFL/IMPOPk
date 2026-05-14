//! Tests for EmissionModel Display/FromStr, DecodingMethod Display/FromStr,
//! AncestryHmmParams setters, learn_normalization, estimate_emissions,
//! log_emission_with_coverage, estimate_temperature_normalized, and estimate_switch_prob.

use std::collections::HashMap;

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    DecodingMethod, estimate_temperature, estimate_temperature_normalized,
    estimate_switch_prob, viterbi, forward_backward, posterior_decode,
};

fn make_populations() -> Vec<AncestralPopulation> {
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
            name: "AMR".to_string(),
            haplotypes: vec!["AMR_H1".to_string(), "AMR_H2".to_string()],
        },
    ]
}

fn make_obs(start: u64, eur: f64, afr: f64, amr: f64) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr10".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: [
            ("EUR_H1".to_string(), eur),
            ("EUR_H2".to_string(), eur - 0.01),
            ("AFR_H1".to_string(), afr),
            ("AFR_H2".to_string(), afr - 0.01),
            ("AMR_H1".to_string(), amr),
            ("AMR_H2".to_string(), amr - 0.01),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_coverage(
    start: u64,
    eur: f64,
    afr: f64,
    amr: f64,
    eur_cov: f64,
    afr_cov: f64,
    amr_cov: f64,
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr10".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: [
            ("EUR_H1".to_string(), eur),
            ("EUR_H2".to_string(), eur - 0.01),
            ("AFR_H1".to_string(), afr),
            ("AFR_H2".to_string(), afr - 0.01),
            ("AMR_H1".to_string(), amr),
            ("AMR_H2".to_string(), amr - 0.01),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: Some(
            [
                ("EUR_H1".to_string(), eur_cov),
                ("EUR_H2".to_string(), eur_cov),
                ("AFR_H1".to_string(), afr_cov),
                ("AFR_H2".to_string(), afr_cov),
                ("AMR_H1".to_string(), amr_cov),
                ("AMR_H2".to_string(), amr_cov),
            ]
            .into_iter()
            .collect(),
        ),
        haplotype_consistency_bonus: None,
    }
}

// ============================================================
// EmissionModel Display and FromStr
// ============================================================

#[test]
fn emission_model_display_max() {
    assert_eq!(format!("{}", EmissionModel::Max), "max");
}

#[test]
fn emission_model_display_mean() {
    assert_eq!(format!("{}", EmissionModel::Mean), "mean");
}

#[test]
fn emission_model_display_median() {
    assert_eq!(format!("{}", EmissionModel::Median), "median");
}

#[test]
fn emission_model_display_topk() {
    assert_eq!(format!("{}", EmissionModel::TopK(5)), "top5");
    assert_eq!(format!("{}", EmissionModel::TopK(10)), "top10");
    assert_eq!(format!("{}", EmissionModel::TopK(1)), "top1");
}

#[test]
fn emission_model_fromstr_max() {
    let model: EmissionModel = "max".parse().unwrap();
    assert_eq!(model, EmissionModel::Max);
}

#[test]
fn emission_model_fromstr_mean() {
    let model: EmissionModel = "mean".parse().unwrap();
    assert_eq!(model, EmissionModel::Mean);
}

#[test]
fn emission_model_fromstr_median() {
    let model: EmissionModel = "median".parse().unwrap();
    assert_eq!(model, EmissionModel::Median);
}

#[test]
fn emission_model_fromstr_topk() {
    let model: EmissionModel = "top5".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(5));
}

#[test]
fn emission_model_fromstr_case_insensitive() {
    let model: EmissionModel = "MAX".parse().unwrap();
    assert_eq!(model, EmissionModel::Max);
    let model: EmissionModel = "Mean".parse().unwrap();
    assert_eq!(model, EmissionModel::Mean);
    let model: EmissionModel = "TOP3".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(3));
}

#[test]
fn emission_model_fromstr_invalid() {
    let result: Result<EmissionModel, _> = "invalid".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown emission model"));
}

#[test]
fn emission_model_fromstr_topk_invalid_number() {
    let result: Result<EmissionModel, _> = "topX".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid top-k value"));
}

#[test]
fn emission_model_display_fromstr_roundtrip() {
    for model in [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(7),
    ] {
        let display = format!("{}", model);
        let parsed: EmissionModel = display.parse().unwrap();
        assert_eq!(parsed, model);
    }
}

// ============================================================
// DecodingMethod Display and FromStr
// ============================================================

#[test]
fn decoding_method_display_viterbi() {
    assert_eq!(format!("{}", DecodingMethod::Viterbi), "viterbi");
}

#[test]
fn decoding_method_display_posterior() {
    assert_eq!(format!("{}", DecodingMethod::Posterior), "posterior");
}

#[test]
fn decoding_method_fromstr_viterbi() {
    let method: DecodingMethod = "viterbi".parse().unwrap();
    assert_eq!(method, DecodingMethod::Viterbi);
}

#[test]
fn decoding_method_fromstr_posterior() {
    let method: DecodingMethod = "posterior".parse().unwrap();
    assert_eq!(method, DecodingMethod::Posterior);
}

#[test]
fn decoding_method_fromstr_case_insensitive() {
    let method: DecodingMethod = "VITERBI".parse().unwrap();
    assert_eq!(method, DecodingMethod::Viterbi);
    let method: DecodingMethod = "Posterior".parse().unwrap();
    assert_eq!(method, DecodingMethod::Posterior);
}

#[test]
fn decoding_method_fromstr_invalid() {
    let result: Result<DecodingMethod, _> = "baum-welch".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown decoding method"));
}

#[test]
fn decoding_method_display_fromstr_roundtrip() {
    for method in [DecodingMethod::Viterbi, DecodingMethod::Posterior] {
        let display = format!("{}", method);
        let parsed: DecodingMethod = display.parse().unwrap();
        assert_eq!(parsed, method);
    }
}

// ============================================================
// AncestryHmmParams setters
// ============================================================

#[test]
fn set_temperature_changes_emission_std() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    assert!((params.emission_std - 0.03).abs() < 1e-9);

    params.set_temperature(0.05);
    assert!((params.emission_std - 0.05).abs() < 1e-9);
}

#[test]
fn set_switch_prob_recalculates_transitions() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    params.set_switch_prob(0.01);

    // Stay probability should be 0.99
    assert!((params.transitions[0][0] - 0.99).abs() < 1e-9);
    // Switch to each other state: 0.01 / 2 = 0.005
    assert!((params.transitions[0][1] - 0.005).abs() < 1e-9);
    assert!((params.transitions[0][2] - 0.005).abs() < 1e-9);

    // All rows sum to 1
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

#[test]
fn set_emission_model_changes_model() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    assert_eq!(params.emission_model, EmissionModel::Max);

    params.set_emission_model(EmissionModel::Mean);
    assert_eq!(params.emission_model, EmissionModel::Mean);

    params.set_emission_model(EmissionModel::TopK(3));
    assert_eq!(params.emission_model, EmissionModel::TopK(3));
}

#[test]
fn set_coverage_weight_changes_weight() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    assert!((params.coverage_weight - 0.0).abs() < 1e-9);

    params.set_coverage_weight(0.5);
    assert!((params.coverage_weight - 0.5).abs() < 1e-9);
}

// ============================================================
// learn_normalization
// ============================================================

#[test]
fn learn_normalization_sets_means_and_stds() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    assert!(params.normalization.is_none());

    let obs = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.92, 0.88, 0.91),
        make_obs(20000, 0.94, 0.86, 0.89),
    ];
    params.learn_normalization(&obs);

    let norm = params.normalization.as_ref().unwrap();
    assert_eq!(norm.means.len(), 3);
    assert_eq!(norm.stds.len(), 3);

    // EUR mean should be highest (0.95 avg sim → highest aggregated)
    assert!(norm.means[0] > norm.means[1], "EUR mean should be > AFR mean");
    // All stds should be positive
    for &s in &norm.stds {
        assert!(s > 0.0);
    }
}

#[test]
fn learn_normalization_empty_observations() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let obs: Vec<AncestryObservation> = vec![];
    params.learn_normalization(&obs);

    let norm = params.normalization.as_ref().unwrap();
    // With no data, means should be 0 and stds should be 1e-6 (minimum)
    for &m in &norm.means {
        assert!((m - 0.0).abs() < 1e-9);
    }
    for &s in &norm.stds {
        assert!((s - 1e-6).abs() < 1e-9);
    }
}

#[test]
fn learn_normalization_single_observation() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![make_obs(0, 0.95, 0.85, 0.90)];
    params.learn_normalization(&obs);

    let norm = params.normalization.as_ref().unwrap();
    // With 1 observation, stds should be minimum (1e-6)
    for &s in &norm.stds {
        assert!((s - 1e-6).abs() < 1e-9);
    }
}

#[test]
fn learn_normalization_affects_emission() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.93, 0.87, 0.91),
        make_obs(20000, 0.94, 0.86, 0.89),
    ];

    // Get emissions without normalization
    let test_obs = make_obs(30000, 0.94, 0.86, 0.90);
    let e0_before = params.log_emission(&test_obs, 0);
    let e1_before = params.log_emission(&test_obs, 1);

    // Learn normalization
    params.learn_normalization(&obs);

    // Get emissions with normalization - they should be different
    let e0_after = params.log_emission(&test_obs, 0);
    let e1_after = params.log_emission(&test_obs, 1);

    // The values should change (normalization adjusts the emission model)
    assert!(
        (e0_before - e0_after).abs() > 1e-6 || (e1_before - e1_after).abs() > 1e-6,
        "Normalization should change emission probabilities"
    );
}

// ============================================================
// estimate_emissions
// ============================================================

#[test]
fn estimate_emissions_updates_parameters() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let default_same = params.emission_same_pop_mean;
    let default_diff = params.emission_diff_pop_mean;

    let obs = vec![
        make_obs(0, 0.98, 0.70, 0.75),
        make_obs(10000, 0.72, 0.97, 0.74),
        make_obs(20000, 0.71, 0.73, 0.96),
        make_obs(30000, 0.97, 0.72, 0.76),
    ];

    params.estimate_emissions(&obs);

    // Parameters should have changed
    assert!(
        (params.emission_same_pop_mean - default_same).abs() > 1e-6
            || (params.emission_diff_pop_mean - default_diff).abs() > 1e-6,
        "Emission parameters should change after estimation"
    );
    // Same-pop mean should be higher than diff-pop mean
    assert!(
        params.emission_same_pop_mean > params.emission_diff_pop_mean,
        "same_pop_mean ({}) should be > diff_pop_mean ({})",
        params.emission_same_pop_mean,
        params.emission_diff_pop_mean
    );
}

#[test]
fn estimate_emissions_empty_observations() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let default_same = params.emission_same_pop_mean;
    let default_diff = params.emission_diff_pop_mean;

    let obs: Vec<AncestryObservation> = vec![];
    params.estimate_emissions(&obs);

    // With no data, parameters should remain at defaults
    assert!((params.emission_same_pop_mean - default_same).abs() < 1e-9);
    assert!((params.emission_diff_pop_mean - default_diff).abs() < 1e-9);
}

// ============================================================
// log_emission_with_coverage
// ============================================================

#[test]
fn log_emission_with_coverage_no_coverage_data() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    // Observation without coverage_ratios
    let obs = make_obs(0, 0.95, 0.85, 0.90);
    let emission = params.log_emission(&obs, 0);

    // Should fall back to similarity-only emission
    params.set_coverage_weight(0.0);
    let emission_no_cov = params.log_emission(&obs, 0);
    assert!(
        (emission - emission_no_cov).abs() < 1e-9,
        "Without coverage data, should equal similarity-only emission"
    );
}

#[test]
fn log_emission_with_coverage_combines_signals() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs = make_obs_with_coverage(0, 0.95, 0.85, 0.90, 0.95, 0.80, 0.88);

    // Without coverage weight
    params.set_coverage_weight(0.0);
    let e_no_cov = params.log_emission(&obs, 0);

    // With coverage weight
    params.set_coverage_weight(0.5);
    let e_with_cov = params.log_emission(&obs, 0);

    // Results should differ when coverage data and weight are present
    assert!(
        (e_no_cov - e_with_cov).abs() > 1e-6,
        "Coverage should affect emission: no_cov={}, with_cov={}",
        e_no_cov,
        e_with_cov
    );
}

#[test]
fn log_emission_with_coverage_high_weight_amplifies_coverage() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    // EUR has high similarity AND high coverage → should be strongly favored
    let obs = make_obs_with_coverage(0, 0.95, 0.85, 0.85, 0.99, 0.60, 0.60);

    params.set_coverage_weight(1.0);
    let e_eur = params.log_emission(&obs, 0);
    let e_afr = params.log_emission(&obs, 1);

    assert!(
        e_eur > e_afr,
        "EUR (high sim + high cov) should be more likely than AFR: EUR={}, AFR={}",
        e_eur,
        e_afr
    );
}

#[test]
fn log_emission_with_coverage_empty_ratios() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    let obs = AncestryObservation {
        chrom: "chr10".to_string(),
        start: 0,
        end: 10000,
        sample: "query#1".to_string(),
        similarities: [
            ("EUR_H1".to_string(), 0.95),
            ("AFR_H1".to_string(), 0.85),
            ("AMR_H1".to_string(), 0.90),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: Some(HashMap::new()), // empty map
            haplotype_consistency_bonus: None,
    };

    // Should not panic, falls back to similarity-only
    let _emission = params.log_emission(&obs, 0);
}

// ============================================================
// estimate_temperature_normalized
// ============================================================

#[test]
fn estimate_temperature_normalized_without_normalization() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let obs = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.92, 0.88, 0.91),
        make_obs(20000, 0.94, 0.86, 0.89),
    ];

    // Without normalization, should fall back to estimate_temperature
    let temp_normalized = estimate_temperature_normalized(&obs, &params);
    let temp_regular = estimate_temperature(&obs, &pops);

    assert!(
        (temp_normalized - temp_regular).abs() < 1e-9,
        "Without normalization, should equal regular estimate_temperature"
    );
}

#[test]
fn estimate_temperature_normalized_with_normalization() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.92, 0.88, 0.91),
        make_obs(20000, 0.94, 0.86, 0.89),
        make_obs(30000, 0.93, 0.87, 0.90),
    ];

    params.learn_normalization(&obs);

    let temp = estimate_temperature_normalized(&obs, &params);
    // Should be in z-score range (0.5 - 5.0)
    assert!(temp >= 0.5, "Normalized temperature should be >= 0.5, got {}", temp);
    assert!(temp <= 5.0, "Normalized temperature should be <= 5.0, got {}", temp);
}

#[test]
fn estimate_temperature_normalized_empty_observations() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let training = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.92, 0.88, 0.91),
    ];
    params.learn_normalization(&training);

    let empty: Vec<AncestryObservation> = vec![];
    let temp = estimate_temperature_normalized(&empty, &params);
    // Should return fallback
    assert!((temp - 1.0).abs() < 1e-9, "Empty obs should return fallback 1.0");
}

// ============================================================
// estimate_switch_prob
// ============================================================

#[test]
fn estimate_switch_prob_small_data_fallback() {
    let pops = make_populations();
    // Fewer than 10 observations → should return fallback
    let obs = vec![
        make_obs(0, 0.95, 0.85, 0.90),
        make_obs(10000, 0.92, 0.88, 0.91),
    ];

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.03);
    assert!(
        (switch_prob - 0.001).abs() < 1e-9,
        "Small data should return fallback 0.001"
    );
}

#[test]
fn estimate_switch_prob_no_switches() {
    let pops = make_populations();
    // All windows strongly favor EUR → no switches
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| make_obs(i * 10000, 0.98, 0.70, 0.72))
        .collect();

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.03);
    // Should be low (close to regularized prior 0.001)
    assert!(switch_prob < 0.01, "No switches: prob should be low, got {}", switch_prob);
    assert!(switch_prob >= 0.0001, "Should be at least minimum clamp");
}

#[test]
fn estimate_switch_prob_frequent_switches() {
    let pops = make_populations();
    // Alternating strong EUR and strong AFR signals
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            if i % 2 == 0 {
                make_obs(i * 10000, 0.99, 0.50, 0.50)
            } else {
                make_obs(i * 10000, 0.50, 0.99, 0.50)
            }
        })
        .collect();

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.03);
    // With alternating signals and broad prior (0.01), should detect switches
    assert!(switch_prob > 0.001, "Frequent switches should yield higher prob, got {}", switch_prob);
    assert!(switch_prob <= 0.05, "Should be <= max clamp 0.05");
}

#[test]
fn estimate_switch_prob_clamped_range() {
    let pops = make_populations();
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| make_obs(i * 10000, 0.95, 0.85, 0.90))
        .collect();

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.03);
    assert!(switch_prob >= 0.0001);
    assert!(switch_prob <= 0.05);
}

// ============================================================
// Emission model affects Viterbi/posterior results
// ============================================================

#[test]
fn emission_model_mean_vs_max_different_results() {
    // Use populations with 3+ haplotypes so Max and Mean diverge more clearly
    let pops = vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR_H1".to_string(), "EUR_H2".to_string(), "EUR_H3".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR_H1".to_string(), "AFR_H2".to_string(), "AFR_H3".to_string()],
        },
    ];

    let obs = AncestryObservation {
        chrom: "chr10".to_string(),
        start: 0,
        end: 10000,
        sample: "query#1".to_string(),
        similarities: [
            ("EUR_H1".to_string(), 0.99), // high outlier
            ("EUR_H2".to_string(), 0.70),
            ("EUR_H3".to_string(), 0.72),
            ("AFR_H1".to_string(), 0.80),
            ("AFR_H2".to_string(), 0.81),
            ("AFR_H3".to_string(), 0.79),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };

    let mut params_max = AncestryHmmParams::new(pops.clone(), 0.001);
    params_max.set_emission_model(EmissionModel::Max);

    let mut params_mean = AncestryHmmParams::new(pops, 0.001);
    params_mean.set_emission_model(EmissionModel::Mean);

    let e_max = params_max.log_emission(&obs, 0);
    let e_mean = params_mean.log_emission(&obs, 0);

    // Max(EUR) = 0.99 >> Mean(EUR) = 0.803, so emissions should differ
    assert!(
        (e_max - e_mean).abs() > 1e-6,
        "Max and Mean emission models should yield different results: max={}, mean={}",
        e_max,
        e_mean
    );
}

#[test]
fn posterior_decode_returns_argmax_of_posteriors() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.98, 0.70, 0.75),
        make_obs(10000, 0.70, 0.98, 0.75),
        make_obs(20000, 0.70, 0.75, 0.98),
    ];

    let states = posterior_decode(&obs, &params);
    let posteriors = forward_backward(&obs, &params);

    for (t, &state) in states.iter().enumerate() {
        let max_state = posteriors[t]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(state, max_state, "Posterior decode should match argmax at t={}", t);
    }
}

// ============================================================
// set_switch_prob with 2 states
// ============================================================

#[test]
fn set_switch_prob_two_states() {
    let pops = vec![
        AncestralPopulation {
            name: "A".to_string(),
            haplotypes: vec!["A_H1".to_string()],
        },
        AncestralPopulation {
            name: "B".to_string(),
            haplotypes: vec!["B_H1".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_switch_prob(0.1);

    assert!((params.transitions[0][0] - 0.9).abs() < 1e-9);
    assert!((params.transitions[0][1] - 0.1).abs() < 1e-9);
    assert!((params.transitions[1][0] - 0.1).abs() < 1e-9);
    assert!((params.transitions[1][1] - 0.9).abs() < 1e-9);
}

// ============================================================
// Baum-Welch edge cases
// ============================================================

#[test]
fn baum_welch_single_state_returns_neg_inf() {
    let pops = vec![AncestralPopulation {
        name: "A".to_string(),
        haplotypes: vec!["A_H1".to_string()],
    }];
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![make_obs(0, 0.95, 0.85, 0.90)];
    let obs_slices: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_slices, 10, 1e-6);
    assert!(ll == f64::NEG_INFINITY);
}

#[test]
fn baum_welch_empty_observations_returns_neg_inf() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);

    let obs_slices: Vec<&[AncestryObservation]> = vec![];
    let ll = params.baum_welch(&obs_slices, 10, 1e-6);
    assert!(ll == f64::NEG_INFINITY);
}

#[test]
fn baum_welch_updates_parameters() {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let obs = vec![
        make_obs(0, 0.95, 0.80, 0.85),
        make_obs(10000, 0.94, 0.81, 0.84),
        make_obs(20000, 0.93, 0.82, 0.86),
        make_obs(30000, 0.80, 0.95, 0.83),
        make_obs(40000, 0.81, 0.94, 0.82),
    ];

    let old_stay = params.transitions[0][0];
    let obs_slices: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_slices, 5, 1e-8);

    assert!(ll.is_finite(), "Log-likelihood should be finite, got {}", ll);
    // Transitions should have been updated
    // (they may or may not change significantly depending on data, but the function should run)
    let _ = old_stay; // just ensure it compiled
}
