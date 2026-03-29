//! Tests for AncestryGeneticMap modulated_switch_prob boundary conditions,
//! estimate_emissions with realistic data, and log_emission through softmax
//! emission pathway (covering log_emission_similarity_only indirectly).
//!
//! Identified as coverage gaps in cycle 15 analysis.

use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    forward_backward_with_genetic_map,
};
use std::collections::HashMap;

// ===========================================================================
// Helpers
// ===========================================================================

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs(start: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    let mut similarities = HashMap::new();
    for &(k, v) in sims {
        similarities.insert(k.to_string(), v);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query".to_string(),
        similarities,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_3pop_params() -> AncestryHmmParams {
    let pops = vec![
        make_pop("EUR", &["eur1", "eur2"]),
        make_pop("AFR", &["afr1", "afr2"]),
        make_pop("AMR", &["amr1", "amr2"]),
    ];
    AncestryHmmParams::new(pops, 0.01)
}

// ===========================================================================
// AncestryGeneticMap::modulated_switch_prob edge cases
// ===========================================================================

#[test]
fn test_modulated_switch_prob_very_small_base() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let result = gm.modulated_switch_prob(1e-8, 10_000_000, 20_000_000, 10_000);
    // Should be clamped to at least 1e-6
    assert!(result >= 1e-6);
    assert!(result.is_finite());
}

#[test]
fn test_modulated_switch_prob_base_zero() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let result = gm.modulated_switch_prob(0.0, 10_000_000, 20_000_000, 10_000);
    // base_switch_prob * adjusted / base_haldane — with 0 base, should clamp to >= 1e-6
    assert!(result >= 1e-6);
    assert!(result <= 0.5);
}

#[test]
fn test_modulated_switch_prob_same_position() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let result = gm.modulated_switch_prob(0.01, 50_000_000, 50_000_000, 10_000);
    // dist_cm = 0, adjusted = 0, result should be very small (clamped to 1e-6)
    assert!(result >= 1e-6);
    assert!(result <= 0.5);
}

#[test]
fn test_modulated_switch_prob_very_large_distance() {
    let gm = AncestryGeneticMap::uniform(0, 200_000_000, 1.0);
    let result = gm.modulated_switch_prob(0.01, 0, 200_000_000, 10_000);
    // Very large distance, should clamp to 0.5
    assert!(result <= 0.5);
    assert!(result >= 1e-6);
    assert!(result.is_finite());
}

#[test]
fn test_modulated_switch_prob_tiny_window() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    // window_size = 1 bp → expected_dist_morgans = 1e-6/100 = 1e-8 < 1e-10
    let result = gm.modulated_switch_prob(0.01, 1_000_000, 2_000_000, 1);
    // Should fall into the expected_dist_morgans < 1e-10 branch
    assert!(result.is_finite());
    assert!(result > 0.0);
}

#[test]
fn test_modulated_switch_prob_high_rate_map() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 50.0); // 50 cM/Mb
    let result = gm.modulated_switch_prob(0.01, 10_000_000, 20_000_000, 10_000);
    // High recombination → higher switch probability, clamped to 0.5
    assert!(result >= 1e-6);
    assert!(result <= 0.5);
}

// ===========================================================================
// AncestryGeneticMap::interpolate_cm edge cases
// ===========================================================================

#[test]
fn test_interpolate_extrapolate_before_map() {
    let gm = AncestryGeneticMap::uniform(10_000_000, 20_000_000, 1.0);
    // Position before map start should extrapolate
    let cm = gm.interpolate_cm(5_000_000);
    // At 1 cM/Mb, 5Mb before start → should be negative
    assert!(cm < 0.0);
}

#[test]
fn test_interpolate_extrapolate_after_map() {
    let gm = AncestryGeneticMap::uniform(10_000_000, 20_000_000, 1.0);
    // Position after map end should extrapolate
    let cm = gm.interpolate_cm(25_000_000);
    // At 1 cM/Mb, 5Mb after end (which has 10 cM) → ~15 cM
    assert!(cm > 10.0);
}

#[test]
fn test_interpolate_at_entry_positions() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    // Start: 0 cM, End: 10 cM
    assert!((gm.interpolate_cm(0) - 0.0).abs() < 1e-10);
    assert!((gm.interpolate_cm(10_000_000) - 10.0).abs() < 1e-10);
}

#[test]
fn test_interpolate_midpoint() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let cm = gm.interpolate_cm(5_000_000);
    assert!((cm - 5.0).abs() < 1e-10);
}

// ===========================================================================
// estimate_emissions tests
// ===========================================================================

#[test]
fn test_estimate_emissions_with_clear_signal() {
    let mut params = make_3pop_params();

    // Create observations where EUR haps are most similar
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(i * 10000, &[
            ("eur1", 0.99), ("eur2", 0.98),
            ("afr1", 0.80), ("afr2", 0.79),
            ("amr1", 0.85), ("amr2", 0.84),
        ])
    }).collect();

    params.estimate_emissions(&obs);

    // same_pop_mean (best matches = EUR) should be higher than diff_pop_mean
    assert!(params.emission_same_pop_mean > params.emission_diff_pop_mean);
    assert!(params.emission_std > 0.0);
}

#[test]
fn test_estimate_emissions_empty_observations() {
    let mut params = make_3pop_params();
    let original_same = params.emission_same_pop_mean;
    let original_diff = params.emission_diff_pop_mean;

    params.estimate_emissions(&[]);

    // Should preserve defaults
    assert_eq!(params.emission_same_pop_mean, original_same);
    assert_eq!(params.emission_diff_pop_mean, original_diff);
}

#[test]
fn test_estimate_emissions_no_similarity_data() {
    let mut params = make_3pop_params();
    let original_same = params.emission_same_pop_mean;
    let original_diff = params.emission_diff_pop_mean;

    // Observations with no matching haplotype names
    let obs = vec![make_obs(0, &[("unknown1", 0.99), ("unknown2", 0.80)])];
    params.estimate_emissions(&obs);

    // No data matched → defaults preserved
    assert_eq!(params.emission_same_pop_mean, original_same);
    assert_eq!(params.emission_diff_pop_mean, original_diff);
}

// ===========================================================================
// log_emission (softmax pathway) tests
// ===========================================================================

#[test]
fn test_log_emission_highest_sim_gets_highest_prob() {
    let params = make_3pop_params();

    let obs = make_obs(0, &[
        ("eur1", 0.99), ("eur2", 0.98),
        ("afr1", 0.80), ("afr2", 0.79),
        ("amr1", 0.85), ("amr2", 0.84),
    ]);

    let emit_eur = params.log_emission(&obs, 0); // EUR
    let emit_afr = params.log_emission(&obs, 1); // AFR
    let emit_amr = params.log_emission(&obs, 2); // AMR

    // EUR has highest similarity → highest emission
    assert!(emit_eur > emit_afr);
    assert!(emit_eur > emit_amr);
}

#[test]
fn test_log_emission_no_data_for_state() {
    let params = make_3pop_params();

    // Only EUR haps have data, AFR and AMR have none
    let obs = make_obs(0, &[("eur1", 0.99), ("eur2", 0.98)]);

    let emit_afr = params.log_emission(&obs, 1);
    // No data for AFR → should return NEG_INFINITY
    assert_eq!(emit_afr, f64::NEG_INFINITY);
}

#[test]
fn test_log_emission_all_populations_equal_sim() {
    let params = make_3pop_params();

    let obs = make_obs(0, &[
        ("eur1", 0.95), ("eur2", 0.95),
        ("afr1", 0.95), ("afr2", 0.95),
        ("amr1", 0.95), ("amr2", 0.95),
    ]);

    let emit_eur = params.log_emission(&obs, 0);
    let emit_afr = params.log_emission(&obs, 1);
    let emit_amr = params.log_emission(&obs, 2);

    // All equal → emissions should be nearly equal
    assert!((emit_eur - emit_afr).abs() < 0.1);
    assert!((emit_eur - emit_amr).abs() < 0.1);
}

#[test]
fn test_log_emission_single_hap_per_pop() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.01);

    let obs = make_obs(0, &[("a1", 0.99), ("b1", 0.80)]);

    let emit_a = params.log_emission(&obs, 0);
    let emit_b = params.log_emission(&obs, 1);

    assert!(emit_a > emit_b);
    assert!(emit_a.is_finite());
    assert!(emit_b.is_finite());
}

// ===========================================================================
// forward_backward_with_genetic_map comprehensive tests
// ===========================================================================

#[test]
fn test_fb_genetic_map_posteriors_sum_to_one() {
    let params = make_3pop_params();
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    let observations: Vec<AncestryObservation> = (0..5).map(|i| {
        make_obs(i * 10000, &[
            ("eur1", 0.99), ("eur2", 0.98),
            ("afr1", 0.80), ("afr2", 0.79),
            ("amr1", 0.85), ("amr2", 0.84),
        ])
    }).collect();

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);

    assert_eq!(posteriors.len(), 5);
    for row in &posteriors {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posterior sum = {}", sum);
    }
}

#[test]
fn test_fb_genetic_map_empty_observations() {
    let params = make_3pop_params();
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&[], &params, &gm);
    assert!(posteriors.is_empty());
}

#[test]
fn test_fb_genetic_map_single_observation() {
    let params = make_3pop_params();
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    let obs = vec![make_obs(50_000_000, &[
        ("eur1", 0.99), ("eur2", 0.98),
        ("afr1", 0.80), ("afr2", 0.79),
        ("amr1", 0.85), ("amr2", 0.84),
    ])];

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 1);

    let sum: f64 = posteriors[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // EUR should have highest posterior
    assert!(posteriors[0][0] > posteriors[0][1]);
    assert!(posteriors[0][0] > posteriors[0][2]);
}

#[test]
fn test_fb_genetic_map_ancestry_switch() {
    let params = make_3pop_params();
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    // First half: EUR ancestry, second half: AFR ancestry
    let mut observations = Vec::new();
    for i in 0..5 {
        observations.push(make_obs(i * 10000, &[
            ("eur1", 0.99), ("eur2", 0.98),
            ("afr1", 0.70), ("afr2", 0.69),
            ("amr1", 0.75), ("amr2", 0.74),
        ]));
    }
    for i in 5..10 {
        observations.push(make_obs(i * 10000, &[
            ("eur1", 0.70), ("eur2", 0.69),
            ("afr1", 0.99), ("afr2", 0.98),
            ("amr1", 0.75), ("amr2", 0.74),
        ]));
    }

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    assert_eq!(posteriors.len(), 10);

    // First window should favor EUR (state 0)
    assert!(posteriors[0][0] > posteriors[0][1]);
    // Last window should favor AFR (state 1)
    assert!(posteriors[9][1] > posteriors[9][0]);
}
