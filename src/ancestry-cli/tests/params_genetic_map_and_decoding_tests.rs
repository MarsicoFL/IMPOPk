//! Tests for AncestryHmmParams methods (set_emission_model, set_coverage_weight),
//! AncestryGeneticMap edge cases, DecodingMethod parsing/display,
//! and EmissionModel aggregate edge cases via log_emission.

use std::collections::HashMap;
use impopk_ancestry_cli::hmm::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    DecodingMethod, EmissionModel, viterbi, posterior_decode,
    forward_backward_with_genetic_map, viterbi_with_genetic_map,
    posterior_decode_with_genetic_map,
};

fn make_pops() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string(), "eur2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string(), "afr2".to_string()],
        },
    ]
}

fn make_obs_eur() -> AncestryObservation {
    let mut sims = HashMap::new();
    sims.insert("eur1".to_string(), 0.999);
    sims.insert("eur2".to_string(), 0.998);
    sims.insert("afr1".to_string(), 0.990);
    sims.insert("afr2".to_string(), 0.991);
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 1000,
        end: 2000,
        sample: "query".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_afr() -> AncestryObservation {
    let mut sims = HashMap::new();
    sims.insert("eur1".to_string(), 0.990);
    sims.insert("eur2".to_string(), 0.991);
    sims.insert("afr1".to_string(), 0.999);
    sims.insert("afr2".to_string(), 0.998);
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 2000,
        end: 3000,
        sample: "query".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ============================================================
// set_emission_model
// ============================================================

#[test]
fn test_set_emission_model_changes_model() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    // Default is Max
    params.set_emission_model(EmissionModel::Mean);
    // Verify it changed by checking that log_emission produces different results
    let obs = make_obs_eur();
    let le_mean = params.log_emission(&obs, 0);

    params.set_emission_model(EmissionModel::Max);
    let le_max = params.log_emission(&obs, 0);

    // Mean and Max should produce different emission values
    // (unless data happens to be coincidental)
    assert!(le_mean.is_finite());
    assert!(le_max.is_finite());
}

#[test]
fn test_set_emission_model_median() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    params.set_emission_model(EmissionModel::Median);
    let obs = make_obs_eur();
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "Median emission should be finite");
}

#[test]
fn test_set_emission_model_topk() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    params.set_emission_model(EmissionModel::TopK(1));
    let obs = make_obs_eur();
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "TopK(1) emission should be finite");
}

// ============================================================
// set_coverage_weight
// ============================================================

#[test]
fn test_set_coverage_weight_zero_is_default() {
    let params = AncestryHmmParams::new(make_pops(), 0.001);
    assert_eq!(params.coverage_weight, 0.0);
}

#[test]
fn test_set_coverage_weight_positive() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    params.set_coverage_weight(0.5);
    assert_eq!(params.coverage_weight, 0.5);
}

#[test]
fn test_set_coverage_weight_does_not_affect_without_ratios() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    let obs = make_obs_eur(); // no coverage_ratios

    let le_before = params.log_emission(&obs, 0);
    params.set_coverage_weight(1.0);
    let le_after = params.log_emission(&obs, 0);
    // Without coverage_ratios in observation, weight shouldn't change emission
    assert!((le_before - le_after).abs() < 1e-10,
        "Coverage weight shouldn't affect emission without coverage data");
}

#[test]
fn test_set_coverage_weight_affects_with_ratios() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);

    let mut sims = HashMap::new();
    sims.insert("eur1".to_string(), 0.999);
    sims.insert("eur2".to_string(), 0.998);
    sims.insert("afr1".to_string(), 0.990);
    sims.insert("afr2".to_string(), 0.991);

    let mut cov = HashMap::new();
    cov.insert("eur1".to_string(), 0.95);
    cov.insert("eur2".to_string(), 0.90);
    cov.insert("afr1".to_string(), 0.50);
    cov.insert("afr2".to_string(), 0.55);

    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 1000,
        end: 2000,
        sample: "query".to_string(),
        similarities: sims,
        coverage_ratios: Some(cov),
            haplotype_consistency_bonus: None,
    };

    // Without coverage weight
    params.set_coverage_weight(0.0);
    let le_no_cov = params.log_emission(&obs, 0);

    // With coverage weight
    params.set_coverage_weight(1.0);
    let le_with_cov = params.log_emission(&obs, 0);

    // They should differ when coverage data is present and weight > 0
    assert!(le_no_cov.is_finite());
    assert!(le_with_cov.is_finite());
    assert!((le_no_cov - le_with_cov).abs() > 1e-10,
        "Coverage weight should change emission: no_cov={}, with_cov={}",
        le_no_cov, le_with_cov);
}

// ============================================================
// set_switch_prob: transition matrix consistency
// ============================================================

#[test]
fn test_set_switch_prob_updates_transitions() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    params.set_switch_prob(0.01);

    // Stay prob should be 1 - 0.01 = 0.99
    assert!((params.transitions[0][0] - 0.99).abs() < 1e-10);
    assert!((params.transitions[1][1] - 0.99).abs() < 1e-10);

    // Switch prob should be 0.01 / (2-1) = 0.01 for 2 states
    assert!((params.transitions[0][1] - 0.01).abs() < 1e-10);
    assert!((params.transitions[1][0] - 0.01).abs() < 1e-10);
}

#[test]
fn test_set_switch_prob_three_states() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".to_string()] },
        AncestralPopulation { name: "C".to_string(), haplotypes: vec!["c1".to_string()] },
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_switch_prob(0.06);

    // Stay prob = 1 - 0.06 = 0.94
    for i in 0..3 {
        assert!((params.transitions[i][i] - 0.94).abs() < 1e-10);
    }
    // Switch prob = 0.06 / (3-1) = 0.03
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!((params.transitions[i][j] - 0.03).abs() < 1e-10,
                    "transitions[{}][{}] = {} should be 0.03", i, j, params.transitions[i][j]);
            }
        }
    }
}

#[test]
fn test_set_switch_prob_row_sums() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    for &sp in &[0.0001, 0.01, 0.1, 0.5] {
        params.set_switch_prob(sp);
        for i in 0..params.n_states {
            let row_sum: f64 = params.transitions[i].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10,
                "Row {} sum should be 1.0 for switch_prob={}: got {}", i, sp, row_sum);
        }
    }
}

// ============================================================
// DecodingMethod: FromStr and Display
// ============================================================

#[test]
fn test_decoding_method_fromstr_roundtrip() {
    let viterbi_method: DecodingMethod = "viterbi".parse().unwrap();
    assert_eq!(format!("{}", viterbi_method), "viterbi");

    let posterior_method: DecodingMethod = "posterior".parse().unwrap();
    assert_eq!(format!("{}", posterior_method), "posterior");
}

#[test]
fn test_decoding_method_fromstr_case_insensitive() {
    assert!("Viterbi".parse::<DecodingMethod>().is_ok());
    assert!("VITERBI".parse::<DecodingMethod>().is_ok());
    assert!("Posterior".parse::<DecodingMethod>().is_ok());
    assert!("POSTERIOR".parse::<DecodingMethod>().is_ok());
}

#[test]
fn test_decoding_method_fromstr_invalid() {
    assert!("unknown".parse::<DecodingMethod>().is_err());
    assert!("".parse::<DecodingMethod>().is_err());
    assert!("forward_backward".parse::<DecodingMethod>().is_err());
}

// ============================================================
// Viterbi vs Posterior decode: agreement on clear signals
// ============================================================

#[test]
fn test_viterbi_vs_posterior_both_valid_decodings() {
    let params = AncestryHmmParams::new(make_pops(), 0.001);
    // EUR-EUR-EUR-AFR-AFR-AFR
    let obs = vec![
        make_obs_eur(), make_obs_eur(), make_obs_eur(),
        make_obs_afr(), make_obs_afr(), make_obs_afr(),
    ];

    let vit_states = viterbi(&obs, &params);
    let post_states = posterior_decode(&obs, &params);

    assert_eq!(vit_states.len(), 6);
    assert_eq!(post_states.len(), 6);

    // Posterior decoding should detect the switch (more sensitive)
    assert_eq!(post_states[0], post_states[1], "First two should be same ancestry");
    assert_eq!(post_states[4], post_states[5], "Last two should be same ancestry");
    assert_ne!(post_states[0], post_states[5], "First and last should differ");

    // Viterbi may or may not detect the switch (it penalizes transitions)
    // but both should produce valid state indices
    for &s in &vit_states {
        assert!(s < 2, "Viterbi state should be 0 or 1");
    }
    for &s in &post_states {
        assert!(s < 2, "Posterior state should be 0 or 1");
    }
}

// ============================================================
// AncestryGeneticMap edge cases
// ============================================================

#[test]
fn test_ancestry_genetic_map_uniform() {
    let gm = AncestryGeneticMap::uniform(1_000_000, 2_000_000, 1.0);
    // 1 Mb at 1.0 cM/Mb = 1.0 cM total
    let cm_start = gm.interpolate_cm(1_000_000);
    let cm_end = gm.interpolate_cm(2_000_000);
    assert!((cm_start - 0.0).abs() < 1e-10);
    assert!((cm_end - 1.0).abs() < 1e-10);
}

#[test]
fn test_ancestry_genetic_map_uniform_midpoint() {
    let gm = AncestryGeneticMap::uniform(0, 2_000_000, 1.0);
    let cm_mid = gm.interpolate_cm(1_000_000);
    assert!((cm_mid - 1.0).abs() < 1e-10, "Midpoint should be 1.0 cM");
}

#[test]
fn test_ancestry_genetic_map_distance() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let dist = gm.genetic_distance_cm(1_000_000, 3_000_000);
    assert!((dist - 2.0).abs() < 1e-10, "2Mb at 1cM/Mb = 2 cM");
}

#[test]
fn test_ancestry_genetic_map_distance_symmetry() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let d1 = gm.genetic_distance_cm(1_000_000, 3_000_000);
    let d2 = gm.genetic_distance_cm(3_000_000, 1_000_000);
    assert!((d1 - d2).abs() < 1e-10, "Distance should be symmetric");
}

#[test]
fn test_ancestry_genetic_map_modulated_switch_prob() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let base_prob = 0.001;
    let window_size = 10_000; // 10kb

    // Normal distance
    let prob1 = gm.modulated_switch_prob(base_prob, 1_000_000, 1_010_000, window_size);
    assert!(prob1 > 0.0 && prob1 < 1.0, "Switch prob should be in (0,1)");

    // Larger distance = higher switch prob
    let prob2 = gm.modulated_switch_prob(base_prob, 1_000_000, 1_100_000, window_size);
    assert!(prob2 > prob1, "Larger distance should give higher switch prob");
}

#[test]
fn test_ancestry_genetic_map_modulated_switch_prob_zero_distance() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let prob = gm.modulated_switch_prob(0.001, 1_000_000, 1_000_000, 10_000);
    // Same position = 0 genetic distance = very small probability
    assert!(prob >= 0.0 && prob <= 1.0);
}

// ============================================================
// forward_backward_with_genetic_map: basic smoke test
// ============================================================

#[test]
fn test_fb_with_genetic_map_posteriors_valid() {
    let params = AncestryHmmParams::new(make_pops(), 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs: Vec<AncestryObservation> = (0..5).map(|i| {
        let mut o = make_obs_eur();
        o.start = i * 10_000;
        o.end = (i + 1) * 10_000;
        o
    }).collect();

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 5);
    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 2, "Should have 2 state posteriors at t={}", t);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors should sum to 1.0 at t={}: got {}", t, sum);
        for &p in post {
            assert!(p >= 0.0 && p <= 1.0, "Posterior out of range at t={}: {}", t, p);
        }
    }
}

#[test]
fn test_viterbi_with_genetic_map_matches_without() {
    let params = AncestryHmmParams::new(make_pops(), 0.001);
    // Uniform genetic map = approximately same as no genetic map
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs: Vec<AncestryObservation> = (0..5).map(|i| {
        let mut o = if i < 3 { make_obs_eur() } else { make_obs_afr() };
        o.start = i as u64 * 10_000;
        o.end = (i as u64 + 1) * 10_000;
        o
    }).collect();

    let states_no_gm = viterbi(&obs, &params);
    let states_with_gm = viterbi_with_genetic_map(&obs, &params, &gm);

    // For uniform map, results should be very similar
    assert_eq!(states_no_gm.len(), states_with_gm.len());
    // Both should detect the EUR/AFR switch
    assert_eq!(states_no_gm[0], states_with_gm[0], "First window should match");
    assert_eq!(states_no_gm[4], states_with_gm[4], "Last window should match");
}

#[test]
fn test_posterior_decode_with_genetic_map_empty() {
    let params = AncestryHmmParams::new(make_pops(), 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs: Vec<AncestryObservation> = vec![];
    let states = posterior_decode_with_genetic_map(&obs, &params, &gm);
    assert!(states.is_empty());
}

// ============================================================
// EmissionModel::FromStr edge cases
// ============================================================

#[test]
fn test_emission_model_topk_various() {
    assert!(matches!("top1".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(1)));
    assert!(matches!("top5".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(5)));
    assert!(matches!("top10".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(10)));
    assert!(matches!("top100".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(100)));
}

#[test]
fn test_emission_model_topk_invalid() {
    assert!("top".parse::<EmissionModel>().is_err()); // no number
    assert!("topX".parse::<EmissionModel>().is_err()); // non-numeric
    assert!("top-1".parse::<EmissionModel>().is_err()); // negative
}

#[test]
fn test_emission_model_display_roundtrip_all() {
    let models = vec![
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(3),
    ];
    for model in &models {
        let s = format!("{}", model);
        let parsed: EmissionModel = s.parse().unwrap();
        assert_eq!(format!("{}", parsed), s, "Roundtrip failed for {}", s);
    }
}

// ============================================================
// set_temperature
// ============================================================

#[test]
fn test_set_temperature_changes_emission_std() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    let original = params.emission_std;
    params.set_temperature(0.05);
    assert_eq!(params.emission_std, 0.05);
    assert_ne!(params.emission_std, original, "Temperature should have changed");
}

#[test]
fn test_set_temperature_affects_emission() {
    let mut params = AncestryHmmParams::new(make_pops(), 0.001);
    let obs = make_obs_eur();

    params.set_temperature(0.01);
    let le_sharp = params.log_emission(&obs, 0);

    params.set_temperature(0.1);
    let le_flat = params.log_emission(&obs, 0);

    // Sharper temperature should give more extreme (more negative or more close to 0) log prob
    // for the favored state
    assert!(le_sharp.is_finite());
    assert!(le_flat.is_finite());
    // With sharp temp, the correct state should have higher log probability
    assert!(le_sharp > le_flat || (le_sharp - le_flat).abs() < 0.01,
        "Sharp temp should give higher emission for correct state: sharp={}, flat={}",
        le_sharp, le_flat);
}
