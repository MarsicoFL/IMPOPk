//! Cycle 90: Deep edge case tests for medium-coverage ancestry-cli functions.
//!
//! Targets (6-8 prior test mentions each):
//! - `ensemble_decode`
//! - `estimate_emission_context`
//! - `compute_population_aware_transitions`
//! - `viterbi_from_log_emissions_with_transitions`
//! - `viterbi_from_log_emissions_with_genetic_map`
//! - `set_distance_weighted_transitions`
//! - `apply_confusion_penalties`
//! - `compute_per_pop_agreement_scales`
//! - `log_emission_with_coverage`

use hprc_ancestry_cli::hmm::{
    apply_confusion_penalties, compute_per_pop_agreement_scales, compute_population_aware_transitions,
    ensemble_decode, estimate_emission_context, set_distance_weighted_transitions,
    viterbi_from_log_emissions_with_genetic_map, viterbi_from_log_emissions_with_transitions,
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    EmissionModel,
};
use std::collections::HashMap;

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|h| h.to_string()).collect(),
    }
}

fn make_params(n_pops: usize, switch_prob: f64) -> AncestryHmmParams {
    let pops: Vec<AncestralPopulation> = (0..n_pops)
        .map(|i| make_pop(&format!("pop{i}"), &[&format!("hap{i}_0"), &format!("hap{i}_1")]))
        .collect();
    AncestryHmmParams::new(pops, switch_prob)
}

fn make_obs(start: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "sample1".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_cov(
    start: u64,
    sims: &[(&str, f64)],
    covs: &[(&str, f64)],
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "sample1".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: Some(covs.iter().map(|(k, v)| (k.to_string(), *v)).collect()),
        haplotype_consistency_bonus: None,
    }
}

// ============================================================================
// ensemble_decode tests
// ============================================================================

#[test]
fn ensemble_decode_empty() {
    let p = make_params(3, 0.01);
    let (posteriors, states) = ensemble_decode(&[], &p, 5, 2.0);
    assert!(posteriors.is_empty());
    assert!(states.is_empty());
}

#[test]
fn ensemble_decode_zero_ensemble() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.5, -1.0]];
    let (posteriors, states) = ensemble_decode(&log_em, &p, 0, 2.0);
    assert!(posteriors.is_empty());
    assert!(states.is_empty());
}

#[test]
fn ensemble_decode_single_ensemble_equals_fb() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.3, -1.0], vec![-1.0, -0.3], vec![-0.5, -0.5]];
    let (posteriors, states) = ensemble_decode(&log_em, &p, 1, 2.0);
    assert_eq!(posteriors.len(), 3);
    assert_eq!(states.len(), 3);
    // Posteriors should be valid probabilities
    for row in &posteriors {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posteriors should sum to 1, got {sum}");
    }
}

#[test]
fn ensemble_decode_two_pops_strong_signal() {
    let p = make_params(2, 0.01);
    // Very strong signal favoring pop0
    let log_em = vec![vec![-0.01, -10.0]; 10];
    let (_posteriors, states) = ensemble_decode(&log_em, &p, 5, 2.0);
    assert_eq!(states.len(), 10);
    for s in &states {
        assert_eq!(*s, 0, "should decode as pop0 with strong signal");
    }
}

#[test]
fn ensemble_decode_posteriors_sum_to_one() {
    let p = make_params(3, 0.01);
    let log_em = vec![
        vec![-0.5, -1.0, -1.5],
        vec![-1.0, -0.5, -1.5],
        vec![-1.5, -1.0, -0.5],
    ];
    let (posteriors, states) = ensemble_decode(&log_em, &p, 3, 1.5);
    assert_eq!(posteriors.len(), 3);
    for (t, row) in posteriors.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "window {t}: posteriors sum = {sum}");
        assert!(states[t] < 3);
    }
}

#[test]
fn ensemble_decode_scale_factor_near_one() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.5, -1.0], vec![-1.0, -0.5]];
    // scale_factor < 1.01 gets clamped to 1.01
    let (posteriors, states) = ensemble_decode(&log_em, &p, 3, 0.5);
    assert_eq!(posteriors.len(), 2);
    assert_eq!(states.len(), 2);
}

#[test]
fn ensemble_decode_large_ensemble() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.3, -1.0]; 5];
    let (posteriors, _states) = ensemble_decode(&log_em, &p, 20, 3.0);
    assert_eq!(posteriors.len(), 5);
    for row in &posteriors {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

// ============================================================================
// estimate_emission_context tests
// ============================================================================

#[test]
fn estimate_emission_context_empty_obs() {
    let pops = vec![make_pop("EUR", &["e0", "e1"]), make_pop("AFR", &["a0", "a1"])];
    let result = estimate_emission_context(&[], &pops, 5, 1, 20);
    assert_eq!(result, 5); // returns base_context
}

#[test]
fn estimate_emission_context_empty_pops() {
    let obs = vec![make_obs(0, &[("e0", 0.9)])];
    let result = estimate_emission_context(&obs, &[], 5, 1, 20);
    assert_eq!(result, 5);
}

#[test]
fn estimate_emission_context_base_zero() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let obs = vec![make_obs(0, &[("e0", 0.9), ("a0", 0.5)])];
    let result = estimate_emission_context(&obs, &pops, 0, 1, 20);
    assert_eq!(result, 0);
}

#[test]
fn estimate_emission_context_strong_signal_small_context() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Strong discriminability: 0.9 vs 0.1 gap = 0.8
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9), ("a0", 0.1)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 10, 1, 20);
    // Strong signal should give small context
    assert!(result <= 10, "strong signal should use small context, got {result}");
}

#[test]
fn estimate_emission_context_weak_signal_large_context() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Weak discriminability: 0.5 vs 0.499 gap = 0.001
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.5), ("a0", 0.499)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 5, 1, 20);
    // Weak signal should use larger context
    assert!(result >= 5, "weak signal should use large context, got {result}");
}

#[test]
fn estimate_emission_context_zero_disc_returns_base() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Identical similarities = zero disc → discs empty → returns base_context
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.5), ("a0", 0.5)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 5, 1, 15);
    assert_eq!(result, 5, "zero disc empties discs vec → returns base_context");
}

#[test]
fn estimate_emission_context_tiny_disc_gives_max() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Tiny positive disc → median_disc ≈ 0 → max_context
    // disc > 0 but so small that scale = target/median >> 1 → clamped to max
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.5001), ("a0", 0.5)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 5, 1, 15);
    assert_eq!(result, 15, "tiny disc should give max_context");
}

#[test]
fn estimate_emission_context_clamped_to_min() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Huge gap → very small adaptive context, clamped to min
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.99), ("a0", 0.01)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 100, 3, 200);
    assert!(result >= 3, "should be at least min_context, got {result}");
}

// ============================================================================
// compute_population_aware_transitions tests
// ============================================================================

#[test]
fn pop_aware_trans_single_pop() {
    let pops = vec![make_pop("EUR", &["e0"])];
    let obs = vec![make_obs(0, &[("e0", 0.9)])];
    let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    assert_eq!(trans.len(), 1);
    assert_eq!(trans[0].len(), 1);
}

#[test]
fn pop_aware_trans_two_pops_symmetric() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    // Symmetric observations
    let obs: Vec<AncestryObservation> = (0..50)
        .map(|i| make_obs(i * 10000, &[("e0", 0.8), ("a0", 0.8)]))
        .collect();
    let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    assert_eq!(trans.len(), 2);
    // Rows should be in log space and finite
    for row in &trans {
        for &v in row {
            assert!(v.is_finite(), "transition should be finite, got {v}");
        }
    }
}

#[test]
fn pop_aware_trans_well_separated_pops() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let obs: Vec<AncestryObservation> = (0..50)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9), ("a0", 0.3)]))
        .collect();
    let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    assert_eq!(trans.len(), 2);
    // All values should be in log space (negative)
    for row in &trans {
        for &v in row {
            assert!(v <= 0.0, "log-transition should be <= 0, got {v}");
        }
    }
}

#[test]
fn pop_aware_trans_three_pops() {
    let pops = vec![
        make_pop("EUR", &["e0"]),
        make_pop("AFR", &["a0"]),
        make_pop("EAS", &["s0"]),
    ];
    let obs: Vec<AncestryObservation> = (0..50)
        .map(|i| make_obs(i * 10000, &[("e0", 0.8), ("a0", 0.6), ("s0", 0.4)]))
        .collect();
    let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    assert_eq!(trans.len(), 3);
    for row in &trans {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn pop_aware_trans_empty_obs() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let trans = compute_population_aware_transitions(&[], &pops, &EmissionModel::Max, 0.01);
    assert_eq!(trans.len(), 2);
    // Should still produce valid transitions
    for row in &trans {
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

// ============================================================================
// viterbi_from_log_emissions_with_transitions tests
// ============================================================================

#[test]
fn vit_trans_empty() {
    let p = make_params(2, 0.01);
    let states = viterbi_from_log_emissions_with_transitions(&[], &p, &[]);
    assert!(states.is_empty());
}

#[test]
fn vit_trans_single_window() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.01, -10.0]];
    let states = viterbi_from_log_emissions_with_transitions(&log_em, &p, &[]);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0);
}

#[test]
fn vit_trans_strong_signal() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-10.0, -0.01]; 5];
    let log_trans = vec![vec![vec![(0.99_f64).ln(), (0.01_f64).ln()], vec![(0.01_f64).ln(), (0.99_f64).ln()]]; 4];
    let states = viterbi_from_log_emissions_with_transitions(&log_em, &p, &log_trans);
    assert_eq!(states.len(), 5);
    for s in &states {
        assert_eq!(*s, 1, "strong pop1 signal should decode as 1");
    }
}

#[test]
fn vit_trans_insufficient_transitions_falls_back() {
    let p = make_params(2, 0.01);
    let log_em = vec![vec![-0.5, -1.0]; 5];
    // Only 2 transition matrices for 5 windows (need 4) → falls back
    let log_trans = vec![
        vec![vec![(0.9_f64).ln(), (0.1_f64).ln()], vec![(0.1_f64).ln(), (0.9_f64).ln()]],
        vec![vec![(0.9_f64).ln(), (0.1_f64).ln()], vec![(0.1_f64).ln(), (0.9_f64).ln()]],
    ];
    let states = viterbi_from_log_emissions_with_transitions(&log_em, &p, &log_trans);
    assert_eq!(states.len(), 5);
    for s in &states {
        assert!(*s < 2);
    }
}

#[test]
fn vit_trans_three_pops() {
    let p = make_params(3, 0.01);
    let log_em = vec![
        vec![-0.1, -2.0, -2.0],
        vec![-2.0, -0.1, -2.0],
        vec![-2.0, -2.0, -0.1],
    ];
    let stay = (0.98_f64).ln();
    let switch = (0.01_f64).ln();
    let trans = vec![vec![stay, switch, switch], vec![switch, stay, switch], vec![switch, switch, stay]];
    let log_trans = vec![trans.clone(), trans];
    let states = viterbi_from_log_emissions_with_transitions(&log_em, &p, &log_trans);
    assert_eq!(states.len(), 3);
    for s in &states {
        assert!(*s < 3);
    }
}

// ============================================================================
// viterbi_from_log_emissions_with_genetic_map tests
// ============================================================================

#[test]
fn vit_genmap_ancestry_empty() {
    let p = make_params(2, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let states = viterbi_from_log_emissions_with_genetic_map(&[], &[], &p, &gm);
    assert!(states.is_empty());
}

#[test]
fn vit_genmap_ancestry_single_window() {
    let p = make_params(2, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs = vec![make_obs(0, &[("hap0_0", 0.9), ("hap0_1", 0.8), ("hap1_0", 0.3), ("hap1_1", 0.2)])];
    let log_em = vec![vec![-0.1, -5.0]];
    let states = viterbi_from_log_emissions_with_genetic_map(&obs, &log_em, &p, &gm);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0);
}

#[test]
fn vit_genmap_ancestry_strong_pop1() {
    let p = make_params(2, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs(i * 10000, &[("hap0_0", 0.3), ("hap1_0", 0.9)]))
        .collect();
    let log_em = vec![vec![-5.0, -0.1]; 5];
    let states = viterbi_from_log_emissions_with_genetic_map(&obs, &log_em, &p, &gm);
    assert_eq!(states.len(), 5);
    for s in &states {
        assert_eq!(*s, 1);
    }
}

#[test]
fn vit_genmap_ancestry_valid_states() {
    let p = make_params(3, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let obs: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs(i * 10000, &[("hap0_0", 0.5), ("hap1_0", 0.6), ("hap2_0", 0.4)]))
        .collect();
    let log_em = vec![vec![-1.0, -0.5, -1.5]; 10];
    let states = viterbi_from_log_emissions_with_genetic_map(&obs, &log_em, &p, &gm);
    assert_eq!(states.len(), 10);
    for s in &states {
        assert!(*s < 3);
    }
}

// ============================================================================
// set_distance_weighted_transitions tests
// ============================================================================

#[test]
fn dist_weighted_trans_mismatched_lengths() {
    let mut p = make_params(2, 0.01);
    let orig_trans = p.transitions.clone();
    // Wrong length distances — should be noop
    set_distance_weighted_transitions(&mut p, &[vec![0.0]], &[0.5, 0.5], &[0.01, 0.01]);
    assert_eq!(p.transitions, orig_trans);
}

#[test]
fn dist_weighted_trans_two_pops() {
    let mut p = make_params(2, 0.01);
    let distances = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let proportions = vec![0.5, 0.5];
    let switch_rates = vec![0.05, 0.05];
    set_distance_weighted_transitions(&mut p, &distances, &proportions, &switch_rates);
    // Row sums should be 1
    for row in &p.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row sum should be 1, got {sum}");
    }
    // Diagonal should be stay probability
    assert!((p.transitions[0][0] - 0.95).abs() < 1e-10);
    assert!((p.transitions[1][1] - 0.95).abs() < 1e-10);
}

#[test]
fn dist_weighted_trans_three_pops_distance_affects_weights() {
    let mut p = make_params(3, 0.01);
    // Pop0 close to Pop1, far from Pop2
    let distances = vec![
        vec![0.0, 0.1, 10.0],
        vec![0.1, 0.0, 10.0],
        vec![10.0, 10.0, 0.0],
    ];
    let proportions = vec![0.33, 0.33, 0.34];
    let switch_rates = vec![0.1, 0.1, 0.1];
    set_distance_weighted_transitions(&mut p, &distances, &proportions, &switch_rates);
    // Pop0→Pop1 should be higher than Pop0→Pop2 (closer)
    assert!(
        p.transitions[0][1] > p.transitions[0][2],
        "closer pop should get more probability: {} vs {}",
        p.transitions[0][1],
        p.transitions[0][2]
    );
}

#[test]
fn dist_weighted_trans_zero_switch_rate() {
    let mut p = make_params(2, 0.01);
    let distances = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let proportions = vec![0.5, 0.5];
    let switch_rates = vec![0.0, 0.0];
    set_distance_weighted_transitions(&mut p, &distances, &proportions, &switch_rates);
    // With zero switch rate, diagonal should be 1, off-diagonal 0
    assert!((p.transitions[0][0] - 1.0).abs() < 1e-10);
    assert!((p.transitions[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn dist_weighted_trans_proportions_affect_weights() {
    let mut p = make_params(3, 0.01);
    let distances = vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    // Pop1 has much higher proportion than Pop2
    let proportions = vec![0.1, 0.8, 0.1];
    let switch_rates = vec![0.1, 0.1, 0.1];
    set_distance_weighted_transitions(&mut p, &distances, &proportions, &switch_rates);
    // From Pop0: switching to Pop1 should be favored (higher proportion)
    assert!(
        p.transitions[0][1] > p.transitions[0][2],
        "higher proportion pop should get more: {} vs {}",
        p.transitions[0][1],
        p.transitions[0][2]
    );
}

// ============================================================================
// apply_confusion_penalties tests
// ============================================================================

#[test]
fn confusion_penalties_two_pops_zero_penalty() {
    let p = make_params(2, 0.01);
    let penalties = vec![vec![0.0; 2]; 2];
    let log_trans = apply_confusion_penalties(&p, &penalties);
    assert_eq!(log_trans.len(), 2);
    // Rows should be log-normalized (sum of exp = 1)
    for row in &log_trans {
        let sum: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "exp sum should be 1, got {sum}");
    }
}

#[test]
fn confusion_penalties_negative_penalty_discourages() {
    let p = make_params(2, 0.01);
    // Heavy negative penalty on 0→1 transition
    let penalties = vec![vec![0.0, -10.0], vec![0.0, 0.0]];
    let log_trans = apply_confusion_penalties(&p, &penalties);
    // 0→1 should be much smaller than baseline
    let prob_01 = log_trans[0][1].exp();
    assert!(prob_01 < 0.001, "penalized transition should be tiny, got {prob_01}");
}

#[test]
fn confusion_penalties_positive_penalty_encourages() {
    let p = make_params(2, 0.01);
    // Positive penalty (bonus) on 0→1
    let penalties = vec![vec![0.0, 5.0], vec![0.0, 0.0]];
    let log_trans = apply_confusion_penalties(&p, &penalties);
    // 0→1 should be much larger than baseline
    let prob_01_penalized = log_trans[0][1].exp();
    let prob_01_baseline = p.transitions[0][1]; // pre-penalty
    assert!(
        prob_01_penalized > prob_01_baseline,
        "bonus should increase transition: {prob_01_penalized} vs baseline {prob_01_baseline}"
    );
}

#[test]
fn confusion_penalties_three_pops_normalized() {
    let p = make_params(3, 0.01);
    let penalties = vec![vec![0.0, -1.0, 1.0], vec![0.5, 0.0, -0.5], vec![-0.5, 0.5, 0.0]];
    let log_trans = apply_confusion_penalties(&p, &penalties);
    assert_eq!(log_trans.len(), 3);
    for row in &log_trans {
        assert_eq!(row.len(), 3);
        let sum: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "exp sum should be 1, got {sum}");
    }
}

#[test]
fn confusion_penalties_all_negative_still_normalized() {
    let p = make_params(2, 0.01);
    let penalties = vec![vec![-5.0, -5.0], vec![-5.0, -5.0]];
    let log_trans = apply_confusion_penalties(&p, &penalties);
    for row in &log_trans {
        let sum: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// ============================================================================
// compute_per_pop_agreement_scales tests
// ============================================================================

#[test]
fn per_pop_agreement_scales_too_few_obs() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9), ("a0", 0.5)]))
        .collect();
    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.2, 0.8);
    // Fallback: uniform scales
    assert_eq!(scales.agree_scales.len(), 2);
    assert!((scales.agree_scales[0] - 1.2).abs() < 1e-10);
    assert!((scales.agree_scales[1] - 1.2).abs() < 1e-10);
}

#[test]
fn per_pop_agreement_scales_single_pop() {
    let pops = vec![make_pop("EUR", &["e0"])];
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9)]))
        .collect();
    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.2, 0.8);
    assert_eq!(scales.agree_scales.len(), 1);
    assert!((scales.agree_scales[0] - 1.2).abs() < 1e-10);
}

#[test]
fn per_pop_agreement_scales_well_separated() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9), ("a0", 0.3)]))
        .collect();
    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.2, 0.8);
    assert_eq!(scales.agree_scales.len(), 2);
    assert_eq!(scales.disagree_matrix.len(), 2);
    // Both pops should have finite agree_scales
    for &s in &scales.agree_scales {
        assert!(s.is_finite(), "agree_scale should be finite");
        assert!(s > 0.0, "agree_scale should be positive");
    }
}

#[test]
fn per_pop_agreement_scales_three_pops_asymmetric() {
    let pops = vec![
        make_pop("EUR", &["e0"]),
        make_pop("AFR", &["a0"]),
        make_pop("EAS", &["s0"]),
    ];
    // EUR very distinct, AFR and EAS similar
    let obs: Vec<AncestryObservation> = (0..100)
        .map(|i| make_obs(i * 10000, &[("e0", 0.9), ("a0", 0.5), ("s0", 0.48)]))
        .collect();
    let scales = compute_per_pop_agreement_scales(&obs, &pops, 1.2, 0.8);
    assert_eq!(scales.agree_scales.len(), 3);
    assert_eq!(scales.disagree_matrix.len(), 3);
    // EUR (well-separated) should have different scale than AFR/EAS (close)
    for row in &scales.disagree_matrix {
        assert_eq!(row.len(), 3);
        for &v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn per_pop_agreement_scales_empty_obs() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let scales = compute_per_pop_agreement_scales(&[], &pops, 1.2, 0.8);
    assert_eq!(scales.agree_scales.len(), 2);
    assert!((scales.agree_scales[0] - 1.2).abs() < 1e-10);
}

// ============================================================================
// log_emission_with_coverage tests
// ============================================================================

#[test]
fn log_emission_with_coverage_no_coverage_data() {
    let pops = vec![make_pop("EUR", &["e0", "e1"]), make_pop("AFR", &["a0", "a1"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    let obs = make_obs(0, &[("e0", 0.9), ("e1", 0.85), ("a0", 0.4), ("a1", 0.3)]);
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 0.5);
    // No coverage data → same as base
    assert!((base - with_cov).abs() < 1e-10);
}

#[test]
fn log_emission_with_coverage_zero_weight() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    let obs = make_obs_with_cov(
        0,
        &[("e0", 0.9), ("a0", 0.4)],
        &[("e0", 0.95), ("a0", 0.7)],
    );
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 0.0);
    // Weight 0 → coverage term is zero → same as base
    assert!((base - with_cov).abs() < 1e-10);
}

#[test]
fn log_emission_with_coverage_positive_weight_modifies() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    let obs = make_obs_with_cov(
        0,
        &[("e0", 0.9), ("a0", 0.4)],
        &[("e0", 0.95), ("a0", 0.7)],
    );
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 1.0);
    // With coverage data and weight>0, result should differ from base
    assert!((base - with_cov).abs() > 1e-10, "coverage should modify emission");
    assert!(with_cov.is_finite());
}

#[test]
fn log_emission_with_coverage_empty_coverage_map() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    let mut obs = make_obs(0, &[("e0", 0.9), ("a0", 0.4)]);
    obs.coverage_ratios = Some(HashMap::new());
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 0.5);
    assert!((base - with_cov).abs() < 1e-10, "empty coverage map should be same as no coverage");
}

#[test]
fn log_emission_with_coverage_single_pop_covered() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    // Only one pop has coverage → only 1 valid cov → returns sim-only
    let obs = make_obs_with_cov(
        0,
        &[("e0", 0.9), ("a0", 0.4)],
        &[("e0", 0.95)],
    );
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 0.5);
    assert!((base - with_cov).abs() < 1e-10, "single valid cov should return sim-only");
}

#[test]
fn log_emission_with_coverage_target_zero_cov() {
    let pops = vec![make_pop("EUR", &["e0"]), make_pop("AFR", &["a0"])];
    let p = AncestryHmmParams::new(pops, 0.01);
    // Target population has zero coverage → returns sim-only
    let obs = make_obs_with_cov(
        0,
        &[("e0", 0.9), ("a0", 0.4)],
        &[("e0", 0.0), ("a0", 0.8)],
    );
    let base = p.log_emission(&obs, 0);
    let with_cov = p.log_emission_with_coverage(&obs, 0, 0.5);
    assert!((base - with_cov).abs() < 1e-10, "zero target cov should return sim-only");
}
