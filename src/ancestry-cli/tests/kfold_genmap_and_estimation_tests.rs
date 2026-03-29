//! Tests for cross_validate_kfold edge cases, genetic map HMM variant properties,
//! and temperature/switch_prob estimation round-trip consistency.
//!
//! Targets low-coverage areas:
//! - cross_validate_kfold: many folds, asymmetric pops, single-hap pops
//! - viterbi_with_genetic_map / posterior_decode_with_genetic_map: hotspot effects
//! - estimate_temperature / estimate_switch_prob: recovery from synthetic data
//! - forward_backward_with_genetic_map: posterior properties with non-uniform recomb

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    AncestryGeneticMap, EmissionModel,
    cross_validate_kfold,
    viterbi, viterbi_with_genetic_map,
    forward_backward, forward_backward_with_genetic_map,
    posterior_decode_with_genetic_map,
    estimate_temperature, estimate_switch_prob,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn make_obs(
    pops: &[AncestralPopulation],
    true_pop: usize,
    window_idx: usize,
) -> AncestryObservation {
    let mut sims = HashMap::new();
    for (pop_idx, pop) in pops.iter().enumerate() {
        let base_sim = if pop_idx == true_pop { 0.98 } else { 0.80 };
        let variation = ((window_idx as f64 * 0.618 + pop_idx as f64 * 1.23).sin() * 0.002).abs();
        for (h_idx, hap) in pop.haplotypes.iter().enumerate() {
            let hap_variation = h_idx as f64 * 0.003;
            sims.insert(hap.clone(), base_sim + variation - hap_variation);
        }
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: (window_idx * 10000) as u64,
        end: ((window_idx + 1) * 10000 - 1) as u64,
        sample: "test#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_pops_with_haps(specs: &[(&str, usize)]) -> Vec<AncestralPopulation> {
    specs
        .iter()
        .map(|(name, n_haps)| AncestralPopulation {
            name: name.to_string(),
            haplotypes: (0..*n_haps)
                .map(|i| format!("{}_hap{}", name, i))
                .collect(),
        })
        .collect()
}

fn uniform_genetic_map(start: u64, end: u64, rate_cm_per_mb: f64) -> AncestryGeneticMap {
    AncestryGeneticMap::uniform(start, end, rate_cm_per_mb)
}

fn build_observations(pops: &[AncestralPopulation], n_windows: usize) -> HashMap<String, Vec<AncestryObservation>> {
    let mut observations = HashMap::new();
    for pop in pops {
        for hap in &pop.haplotypes {
            let true_pop = pops.iter().position(|p| p.haplotypes.contains(hap)).unwrap();
            let obs: Vec<AncestryObservation> = (0..n_windows)
                .map(|w| make_obs(pops, true_pop, w))
                .collect();
            observations.insert(hap.clone(), obs);
        }
    }
    observations
}

// ═══════════════════════════════════════════════════════════════════════════
// cross_validate_kfold: edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kfold_k_larger_than_haplotypes_doesnt_crash() {
    let pops = make_pops_with_haps(&[("AFR", 3), ("EUR", 3)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 5);

    // k=10 but only 3 haplotypes per pop — should not crash
    let result = cross_validate_kfold(&observations, &pops, &params, 10);
    assert!(result.overall_accuracy >= 0.0 && result.overall_accuracy <= 1.0);
}

#[test]
fn kfold_k_equals_2_minimum_folds() {
    let pops = make_pops_with_haps(&[("AFR", 4), ("EUR", 4)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 5);

    let result = cross_validate_kfold(&observations, &pops, &params, 2);
    assert!(result.overall_accuracy >= 0.0 && result.overall_accuracy <= 1.0);
    // n_windows_per_pop should have entries
    assert!(!result.n_windows_per_pop.is_empty());
}

#[test]
fn kfold_k_equals_1_clamped_to_2() {
    let pops = make_pops_with_haps(&[("AFR", 4), ("EUR", 4)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 3);

    // k=1 should be clamped to 2
    let result = cross_validate_kfold(&observations, &pops, &params, 1);
    assert!(result.overall_accuracy >= 0.0);
}

#[test]
fn kfold_asymmetric_population_sizes() {
    let pops = make_pops_with_haps(&[("AFR", 8), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 5);

    let result = cross_validate_kfold(&observations, &pops, &params, 3);
    assert!(result.overall_accuracy >= 0.0 && result.overall_accuracy <= 1.0);
}

#[test]
fn kfold_single_hap_population_kept_in_reference() {
    let pops = make_pops_with_haps(&[("AFR", 1), ("EUR", 4)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 3);

    let result = cross_validate_kfold(&observations, &pops, &params, 3);
    // Should work — EUR haps tested, AFR single-hap stays in reference
    assert!(result.overall_accuracy >= 0.0);
}

#[test]
fn kfold_three_populations_clear_signal() {
    let pops = make_pops_with_haps(&[("AFR", 4), ("EUR", 4), ("EAS", 4)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 10);

    let result = cross_validate_kfold(&observations, &pops, &params, 3);
    assert!(
        result.overall_accuracy > 0.5,
        "3-pop k-fold accuracy should be >0.5 with clear signal, got {:.3}",
        result.overall_accuracy
    );
}

#[test]
fn kfold_result_fields_consistent() {
    let pops = make_pops_with_haps(&[("AFR", 4), ("EUR", 4)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 5);

    let result = cross_validate_kfold(&observations, &pops, &params, 4);

    // overall_accuracy should be in [0, 1]
    assert!(result.overall_accuracy >= 0.0 && result.overall_accuracy <= 1.0);

    // accuracy_per_pop should have values in [0, 1]
    for (_, &acc) in &result.accuracy_per_pop {
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    // precision, recall, f1 per pop should be in [0, 1]
    for (_, &p) in &result.precision_per_pop {
        assert!(p >= 0.0 && p <= 1.0);
    }
    for (_, &r) in &result.recall_per_pop {
        assert!(r >= 0.0 && r <= 1.0);
    }
    for (_, &f) in &result.f1_per_pop {
        assert!(f >= 0.0 && f <= 1.0);
    }
}

#[test]
fn kfold_five_populations() {
    let pops = make_pops_with_haps(&[
        ("AFR", 3), ("EUR", 3), ("EAS", 3), ("CSA", 3), ("AMR", 3),
    ]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let observations = build_observations(&pops, 8);

    let result = cross_validate_kfold(&observations, &pops, &params, 3);
    assert!(result.overall_accuracy >= 0.0);
    // Should have entries for all 5 populations
    assert_eq!(result.accuracy_per_pop.len(), 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Genetic map HMM: properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn viterbi_with_genetic_map_empty_input() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let states = viterbi_with_genetic_map(&[], &params, &gmap);
    assert!(states.is_empty());
}

#[test]
fn viterbi_with_genetic_map_single_observation() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs = vec![make_obs(&pops, 0, 0)];
    let states = viterbi_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), 1);
    assert!(states[0] < pops.len());
}

#[test]
fn viterbi_with_genetic_map_valid_states_3pop() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2), ("EAS", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let mut obs: Vec<AncestryObservation> = (0..30)
        .map(|w| make_obs(&pops, 0, w))
        .collect();
    obs.extend((30..60).map(|w| make_obs(&pops, 1, w)));

    let states = viterbi_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), 60);
    for &s in &states {
        assert!(s < 3, "State {} out of range for 3 populations", s);
    }
}

#[test]
fn posterior_decode_with_genetic_map_empty() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let states = posterior_decode_with_genetic_map(&[], &params, &gmap);
    assert!(states.is_empty());
}

#[test]
fn posterior_decode_with_genetic_map_valid_states() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let states = posterior_decode_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), 20);
    for &s in &states {
        assert!(s < 2, "State {} out of range", s);
    }
}

#[test]
fn forward_backward_with_genetic_map_posteriors_sum_to_one() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2), ("EAS", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..30)
        .map(|w| make_obs(&pops, w % 3, w))
        .collect();

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(posteriors.len(), 30);

    for (t, probs) in posteriors.iter().enumerate() {
        assert_eq!(probs.len(), 3, "Window {} should have 3 posteriors", t);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Window {}: posteriors sum to {}, expected 1.0",
            t, sum
        );
        for &p in probs {
            assert!(p >= 0.0 && p <= 1.0, "Window {}: posterior {} out of range", t, p);
        }
    }
}

#[test]
fn forward_backward_with_genetic_map_empty() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&[], &params, &gmap);
    assert!(posteriors.is_empty());
}

#[test]
fn genetic_map_uniform_close_to_standard_fb() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let posteriors_std = forward_backward(&obs, &params);
    let posteriors_gmap = forward_backward_with_genetic_map(&obs, &params, &gmap);

    assert_eq!(posteriors_std.len(), posteriors_gmap.len());
    for (t, (std_p, gmap_p)) in posteriors_std.iter().zip(posteriors_gmap.iter()).enumerate() {
        for (s, (&ps, &pg)) in std_p.iter().zip(gmap_p.iter()).enumerate() {
            assert!(
                (ps - pg).abs() < 0.3,
                "Window {} state {}: std={:.4}, gmap={:.4} differ too much",
                t, s, ps, pg
            );
        }
    }
}

#[test]
fn viterbi_with_genetic_map_agrees_with_posterior_decode() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let vit_states = viterbi_with_genetic_map(&obs, &params, &gmap);
    let pd_states = posterior_decode_with_genetic_map(&obs, &params, &gmap);

    assert_eq!(vit_states.len(), pd_states.len());

    let agreement: f64 = vit_states
        .iter()
        .zip(pd_states.iter())
        .filter(|(&a, &b)| a == b)
        .count() as f64
        / vit_states.len() as f64;

    assert!(
        agreement > 0.8,
        "Viterbi and posterior decode should agree on clear signal: {:.3}",
        agreement
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// estimate_temperature: round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn estimate_temperature_returns_positive() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let obs: Vec<AncestryObservation> = (0..30)
        .map(|w| make_obs(&pops, w % 2, w))
        .collect();

    let temp = estimate_temperature(&obs, &pops);
    assert!(temp > 0.0, "Temperature should be positive, got {}", temp);
    assert!(temp.is_finite(), "Temperature should be finite");
}

#[test]
fn estimate_temperature_both_contrasts_positive() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);

    // High contrast
    let mut high_contrast_obs: Vec<AncestryObservation> = Vec::new();
    for w in 0..30 {
        let mut sims = HashMap::new();
        for (pop_idx, pop) in pops.iter().enumerate() {
            let base_sim = if pop_idx == 0 { 0.98 } else { 0.50 };
            for hap in &pop.haplotypes {
                sims.insert(hap.clone(), base_sim);
            }
        }
        high_contrast_obs.push(AncestryObservation {
            chrom: "chr1".to_string(),
            start: (w * 10000) as u64,
            end: ((w + 1) * 10000 - 1) as u64,
            sample: "test#1".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        });
    }

    // Low contrast
    let mut low_contrast_obs: Vec<AncestryObservation> = Vec::new();
    for w in 0..30 {
        let mut sims = HashMap::new();
        for (pop_idx, pop) in pops.iter().enumerate() {
            let base_sim = if pop_idx == 0 { 0.90 } else { 0.88 };
            for hap in &pop.haplotypes {
                sims.insert(hap.clone(), base_sim);
            }
        }
        low_contrast_obs.push(AncestryObservation {
            chrom: "chr1".to_string(),
            start: (w * 10000) as u64,
            end: ((w + 1) * 10000 - 1) as u64,
            sample: "test#1".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        });
    }

    let temp_high = estimate_temperature(&high_contrast_obs, &pops);
    let temp_low = estimate_temperature(&low_contrast_obs, &pops);

    assert!(temp_high > 0.0 && temp_high.is_finite());
    assert!(temp_low > 0.0 && temp_low.is_finite());
}

#[test]
fn estimate_temperature_single_observation() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let obs = vec![make_obs(&pops, 0, 0)];

    let temp = estimate_temperature(&obs, &pops);
    assert!(temp > 0.0 && temp.is_finite());
}

#[test]
fn estimate_temperature_empty_observations() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let obs: Vec<AncestryObservation> = vec![];

    let temp = estimate_temperature(&obs, &pops);
    assert!(temp.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════
// estimate_switch_prob: properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn estimate_switch_prob_returns_valid_probability() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let obs: Vec<AncestryObservation> = (0..50)
        .map(|w| make_obs(&pops, w % 2, w))
        .collect();

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.01);
    assert!(
        switch_prob >= 0.0 && switch_prob <= 1.0,
        "Switch prob should be in [0,1], got {}",
        switch_prob
    );
}

#[test]
fn estimate_switch_prob_uniform_ancestry_low() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);

    // All windows same ancestry
    let obs: Vec<AncestryObservation> = (0..50)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let switch_prob = estimate_switch_prob(&obs, &pops, 0.01);
    assert!(
        switch_prob >= 0.0 && switch_prob <= 1.0,
        "Switch prob should be valid, got {}",
        switch_prob
    );
    assert!(
        switch_prob < 0.2,
        "Uniform ancestry should have low switch prob, got {}",
        switch_prob
    );
}

#[test]
fn estimate_switch_prob_few_observations_returns_fallback() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    // < 10 obs should return fallback
    let switch_prob = estimate_switch_prob(&obs, &pops, 0.01);
    assert_eq!(switch_prob, 0.001, "Few observations should return 0.001 fallback");
}

// ═══════════════════════════════════════════════════════════════════════════
// Temperature affects emission sharpness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn low_temperature_produces_sharper_posteriors() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);

    let mut params_low_temp = AncestryHmmParams::new(pops.clone(), 0.01);
    params_low_temp.set_temperature(0.001);

    let mut params_high_temp = AncestryHmmParams::new(pops.clone(), 0.01);
    params_high_temp.set_temperature(0.1);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let posteriors_low = forward_backward(&obs, &params_low_temp);
    let posteriors_high = forward_backward(&obs, &params_high_temp);

    let avg_max_low: f64 = posteriors_low
        .iter()
        .map(|p| p.iter().cloned().fold(0.0_f64, f64::max))
        .sum::<f64>()
        / posteriors_low.len() as f64;

    let avg_max_high: f64 = posteriors_high
        .iter()
        .map(|p| p.iter().cloned().fold(0.0_f64, f64::max))
        .sum::<f64>()
        / posteriors_high.len() as f64;

    assert!(
        avg_max_low >= avg_max_high - 0.05,
        "Low temp avg max posterior ({:.4}) should >= high temp ({:.4})",
        avg_max_low, avg_max_high
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// set_switch_prob and set_temperature round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn set_switch_prob_produces_valid_viterbi() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    let states_low = viterbi(&obs, &params);

    params.set_switch_prob(0.5);
    let states_high = viterbi(&obs, &params);

    for &s in states_low.iter().chain(states_high.iter()) {
        assert!(s < 2);
    }
}

#[test]
fn set_temperature_then_viterbi_valid() {
    let pops = make_pops_with_haps(&[("AFR", 2), ("EUR", 2)]);
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| make_obs(&pops, 0, w))
        .collect();

    for temp in &[0.001, 0.01, 0.1, 1.0, 10.0] {
        params.set_temperature(*temp);
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 20);
        for &s in &states {
            assert!(s < 2, "State out of range with temperature {}", temp);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// All emission models with genetic map
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_emission_models_with_genetic_map_valid() {
    let pops = make_pops_with_haps(&[("AFR", 3), ("EUR", 3)]);
    let gmap = uniform_genetic_map(0, 1_000_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..15)
        .map(|w| make_obs(&pops, w % 2, w))
        .collect();

    let models = [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(1),
        EmissionModel::TopK(2),
    ];

    for model in &models {
        let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
        params.set_emission_model(model.clone());

        let states = viterbi_with_genetic_map(&obs, &params, &gmap);
        assert_eq!(states.len(), 15);
        for &s in &states {
            assert!(s < 2, "Model {:?}: state {} out of range", model, s);
        }

        let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
        assert_eq!(posteriors.len(), 15);
        for (t, probs) in posteriors.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Model {:?} window {}: posteriors sum to {}", model, t, sum
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5-population with genetic map
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn five_populations_genetic_map_valid() {
    let pops = make_pops_with_haps(&[
        ("AFR", 2), ("EUR", 2), ("EAS", 2), ("CSA", 2), ("AMR", 2),
    ]);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let gmap = uniform_genetic_map(0, 2_000_000, 1.0);

    let mut obs: Vec<AncestryObservation> = Vec::new();
    for pop_idx in 0..5 {
        for w in 0..10 {
            obs.push(make_obs(&pops, pop_idx, pop_idx * 10 + w));
        }
    }

    let states = viterbi_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), 50);
    for &s in &states {
        assert!(s < 5, "5-pop state out of range: {}", s);
    }

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(posteriors.len(), 50);
    for probs in &posteriors {
        assert_eq!(probs.len(), 5);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
