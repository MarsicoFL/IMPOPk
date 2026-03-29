//! End-to-end round-trip tests for ancestry HMM inference.
//!
//! Strategy: construct populations with known haplotypes → generate synthetic
//! observations where each window has clear ancestry signal → run Viterbi /
//! forward-backward → verify the recovered state sequence matches ground truth.
//!
//! These tests verify the full ancestry pipeline works end-to-end.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    EmissionModel, forward_backward, posterior_decode, viterbi,
    CrossValidationResult, cross_validate,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn make_populations(n_pops: usize) -> Vec<AncestralPopulation> {
    let pop_names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n_pops)
        .map(|i| {
            let name = pop_names[i % pop_names.len()].to_string();
            AncestralPopulation {
                name: name.clone(),
                haplotypes: vec![
                    format!("{}_hap1", name),
                    format!("{}_hap2", name),
                ],
            }
        })
        .collect()
}

/// Generate synthetic observation where the `true_pop_idx` population has
/// highest similarity (0.98) and others have lower similarity (0.80).
fn make_observation(
    pops: &[AncestralPopulation],
    true_pop_idx: usize,
    window_idx: usize,
) -> AncestryObservation {
    let mut sims = HashMap::new();

    for (pop_idx, pop) in pops.iter().enumerate() {
        let base_sim = if pop_idx == true_pop_idx { 0.98 } else { 0.80 };
        // Add small deterministic variation
        let variation = ((window_idx as f64 * 0.618 + pop_idx as f64 * 1.23).sin() * 0.002).abs();
        for (h_idx, hap) in pop.haplotypes.iter().enumerate() {
            let hap_variation = h_idx as f64 * 0.005;
            sims.insert(hap.clone(), base_sim + variation - hap_variation);
        }
    }

    AncestryObservation {
        chrom: "chr1".to_string(),
        start: (window_idx * 10000) as u64,
        end: ((window_idx + 1) * 10000 - 1) as u64,
        sample: "test_sample#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

/// Generate a sequence of observations following a ground truth state sequence.
fn generate_ancestry_observations(
    pops: &[AncestralPopulation],
    ground_truth: &[usize],
) -> Vec<AncestryObservation> {
    ground_truth
        .iter()
        .enumerate()
        .map(|(i, &state)| make_observation(pops, state, i))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: uniform ancestry (all windows from same population)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_uniform_ancestry_2pop() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // All windows from pop 0 (AFR)
    let gt = vec![0_usize; 50];
    let obs = generate_ancestry_observations(&pops, &gt);

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 50);

    let correct = states.iter().filter(|&&s| s == 0).count();
    assert_eq!(correct, 50, "uniform pop0 should all be called pop0");
}

#[test]
fn roundtrip_uniform_ancestry_3pop() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // All windows from pop 2 (EAS)
    let gt = vec![2_usize; 40];
    let obs = generate_ancestry_observations(&pops, &gt);

    let states = viterbi(&obs, &params);

    let correct = states.iter().filter(|&&s| s == 2).count();
    assert!(
        correct >= 35,
        "uniform pop2 with 3 pops should mostly recover, got {}/40",
        correct
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: single ancestry switch
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_single_switch_2pop() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // 30 windows of pop0, 30 windows of pop1
    let mut gt = vec![0_usize; 30];
    gt.extend(vec![1_usize; 30]);
    let obs = generate_ancestry_observations(&pops, &gt);

    let states = viterbi(&obs, &params);

    // Check first 25 windows are pop0
    let first_correct = states[0..25].iter().filter(|&&s| s == 0).count();
    assert!(first_correct >= 20, "first segment should be pop0: {}/25", first_correct);

    // Check last 25 windows are pop1
    let last_correct = states[35..60].iter().filter(|&&s| s == 1).count();
    assert!(last_correct >= 20, "last segment should be pop1: {}/25", last_correct);
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: two switches (three-segment pattern)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_two_switches_3pop() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // Pop0(20) → Pop1(20) → Pop2(20)
    let mut gt = Vec::new();
    gt.extend(vec![0_usize; 20]);
    gt.extend(vec![1_usize; 20]);
    gt.extend(vec![2_usize; 20]);

    let obs = generate_ancestry_observations(&pops, &gt);
    let states = viterbi(&obs, &params);

    // Verify each segment's core (away from boundaries)
    let core0 = states[5..15].iter().filter(|&&s| s == 0).count();
    let core1 = states[25..35].iter().filter(|&&s| s == 1).count();
    let core2 = states[45..55].iter().filter(|&&s| s == 2).count();

    assert!(core0 >= 7, "core pop0 should be recovered: {}/10", core0);
    assert!(core1 >= 7, "core pop1 should be recovered: {}/10", core1);
    assert!(core2 >= 7, "core pop2 should be recovered: {}/10", core2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: Viterbi and posterior_decode agreement
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_viterbi_posterior_agreement() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let gt = vec![0_usize; 50];
    let obs = generate_ancestry_observations(&pops, &gt);

    let vit_states = viterbi(&obs, &params);
    let post_states = posterior_decode(&obs, &params);

    assert_eq!(vit_states.len(), post_states.len());

    // With clear signal, Viterbi and posterior decode should agree on most windows
    let agree = vit_states.iter().zip(post_states.iter()).filter(|(a, b)| a == b).count();
    assert!(
        agree as f64 / vit_states.len() as f64 >= 0.9,
        "Viterbi and posterior decode should agree >= 90%, got {}/{}",
        agree,
        vit_states.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: forward-backward posteriors properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_posteriors_sum_to_one() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let gt = vec![1_usize; 30];
    let obs = generate_ancestry_observations(&pops, &gt);

    let posteriors = forward_backward(&obs, &params);

    assert_eq!(posteriors.len(), 30);
    for (t, row) in posteriors.iter().enumerate() {
        assert_eq!(row.len(), 3, "posteriors should have n_states columns");
        let row_sum: f64 = row.iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "posteriors at t={} should sum to 1, got {}",
            t,
            row_sum
        );
    }
}

#[test]
fn roundtrip_posteriors_highest_for_true_pop() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // All windows from pop 1
    let gt = vec![1_usize; 30];
    let obs = generate_ancestry_observations(&pops, &gt);

    let posteriors = forward_backward(&obs, &params);

    // For the core region, posterior for pop1 should be highest
    for t in 5..25 {
        let max_pop = posteriors[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_pop, 1,
            "at t={}, highest posterior should be pop1, got pop{}",
            t, max_pop
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: emission model variants all work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_all_emission_models() {
    let pops = make_populations(2);

    for model in [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(1),
        EmissionModel::TopK(2),
    ] {
        let mut params = AncestryHmmParams::new(pops.clone(), 0.001);
        params.set_emission_model(model);

        let gt = vec![0_usize; 30];
        let obs = generate_ancestry_observations(&pops, &gt);

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 30, "model {:?} should produce valid states", model);

        // All states should be valid indices
        for &s in &states {
            assert!(s < pops.len(), "state {} out of range for model {:?}", s, model);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: temperature affects confidence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_low_temperature_sharper_posteriors() {
    let pops = make_populations(2);

    let gt = vec![0_usize; 30];
    let obs = generate_ancestry_observations(&pops, &gt);

    // Low temperature (more confident)
    let mut params_low = AncestryHmmParams::new(pops.clone(), 0.001);
    params_low.set_temperature(0.01);

    // High temperature (less confident)
    let mut params_high = AncestryHmmParams::new(pops.clone(), 0.001);
    params_high.set_temperature(0.1);

    let post_low = forward_backward(&obs, &params_low);
    let post_high = forward_backward(&obs, &params_high);

    // Low temperature should produce more extreme (sharper) posteriors
    let avg_max_low: f64 = post_low.iter()
        .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .sum::<f64>() / post_low.len() as f64;

    let avg_max_high: f64 = post_high.iter()
        .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .sum::<f64>() / post_high.len() as f64;

    assert!(
        avg_max_low >= avg_max_high - 0.01,
        "low temperature should give sharper posteriors: low={:.4} vs high={:.4}",
        avg_max_low,
        avg_max_high
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: empty and single-element inputs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_empty_observations() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs: Vec<AncestryObservation> = Vec::new();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 0);

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 0);
}

#[test]
fn roundtrip_single_observation() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let obs = vec![make_observation(&pops, 1, 0)];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 3, "state must be valid index");

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 1);
    let row_sum: f64 = posteriors[0].iter().sum();
    assert!((row_sum - 1.0).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: deterministic output
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_deterministic() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let gt = vec![0_usize; 20];
    let obs = generate_ancestry_observations(&pops, &gt);

    let states1 = viterbi(&obs, &params);
    let states2 = viterbi(&obs, &params);
    assert_eq!(states1, states2, "Viterbi must be deterministic");

    let post1 = forward_backward(&obs, &params);
    let post2 = forward_backward(&obs, &params);
    for (r1, r2) in post1.iter().zip(post2.iter()) {
        for (p1, p2) in r1.iter().zip(r2.iter()) {
            assert_eq!(*p1, *p2, "forward-backward must be deterministic");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: 5-population scenario
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_five_populations() {
    let pops = make_populations(5);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // Pop0(15) → Pop2(15) → Pop4(15) → Pop1(15) → Pop3(15)
    let mut gt = Vec::new();
    gt.extend(vec![0_usize; 15]);
    gt.extend(vec![2_usize; 15]);
    gt.extend(vec![4_usize; 15]);
    gt.extend(vec![1_usize; 15]);
    gt.extend(vec![3_usize; 15]);

    let obs = generate_ancestry_observations(&pops, &gt);
    let states = viterbi(&obs, &params);

    assert_eq!(states.len(), 75);

    // All states should be valid
    for &s in &states {
        assert!(s < 5, "state {} out of range for 5 pops", s);
    }

    // Core of first segment should be pop0
    let core0 = states[3..12].iter().filter(|&&s| s == 0).count();
    assert!(core0 >= 5, "core pop0 segment: {}/9", core0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: concordance with ground truth (strong signal)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_strong_signal_high_concordance() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    // Simple two-segment pattern
    let mut gt = vec![0_usize; 40];
    gt.extend(vec![1_usize; 40]);

    let obs = generate_ancestry_observations(&pops, &gt);
    let states = viterbi(&obs, &params);

    let concordant = states.iter().zip(gt.iter()).filter(|(a, b)| a == b).count();
    let concordance = concordant as f64 / gt.len() as f64;
    assert!(
        concordance >= 0.85,
        "strong 2-pop signal should give >= 85% concordance, got {:.1}%",
        concordance * 100.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validation: public API edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cross_validate_all_pops_single_hap_skips_all() {
    // All populations with only 1 haplotype → LOO skips everything
    let pops = vec![
        AncestralPopulation {
            name: "A".to_string(),
            haplotypes: vec!["A_h1".to_string()],
        },
        AncestralPopulation {
            name: "B".to_string(),
            haplotypes: vec!["B_h1".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate(&observations, &pops, &params);
    assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
    // With 0.0 accuracy (no data), has_bias returns true since 0.0 < 0.5
    // This is expected: skipped pops get accuracy=0.0 which triggers the bias check
    assert!(result.has_bias(), "0.0 accuracy should be flagged as bias");
}

#[test]
fn cross_validate_result_fields_consistent() {
    // Verify that cross_validate produces internally consistent results
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let mut observations = HashMap::new();
    for (pop_idx, pop) in pops.iter().enumerate() {
        for hap in &pop.haplotypes {
            let obs: Vec<AncestryObservation> = (0..5)
                .map(|i| {
                    let mut o = make_observation(&pops, pop_idx, i);
                    o.sample = hap.clone();
                    o
                })
                .collect();
            observations.insert(hap.clone(), obs);
        }
    }

    let result = cross_validate(&observations, &pops, &params);

    // Verify: overall_accuracy = total_correct / total_windows
    let total_windows: usize = result.n_windows_per_pop.values().sum();
    if total_windows > 0 {
        let total_correct: usize = result.confusion.iter()
            .filter(|((t, p), _)| t == p)
            .map(|(_, &c)| c)
            .sum();
        let expected_acc = total_correct as f64 / total_windows as f64;
        assert!(
            (result.overall_accuracy - expected_acc).abs() < 1e-6,
            "overall accuracy should be total_correct/total_windows"
        );
    }

    // Precision and recall should be in [0,1] for all populations
    for pop in &pops {
        if let Some(&prec) = result.precision_per_pop.get(&pop.name) {
            assert!((0.0..=1.0).contains(&prec), "precision out of range for {}", pop.name);
        }
        if let Some(&rec) = result.recall_per_pop.get(&pop.name) {
            assert!((0.0..=1.0).contains(&rec), "recall out of range for {}", pop.name);
        }
        if let Some(&f1) = result.f1_per_pop.get(&pop.name) {
            assert!((0.0..=1.0).contains(&f1), "f1 out of range for {}", pop.name);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validation: confusion_matrix_tsv
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn confusion_matrix_tsv_empty_confusion() {
    let result = CrossValidationResult {
        accuracy_per_pop: [("X".to_string(), 0.0)].into(),
        overall_accuracy: 0.0,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: [("X".to_string(), 0.0)].into(),
        recall_per_pop: [("X".to_string(), 0.0)].into(),
        f1_per_pop: [("X".to_string(), 0.0)].into(),
    };

    let tsv = result.confusion_matrix_tsv();
    assert!(tsv.contains("true_pop\tpred_pop\tcount"));
    assert!(tsv.contains("X\tX\t0"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validation: has_bias
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn has_bias_empty_accuracy_map() {
    let result = CrossValidationResult {
        accuracy_per_pop: HashMap::new(),
        overall_accuracy: 0.0,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };
    assert!(!result.has_bias(), "empty map should not report bias");
}

#[test]
fn has_bias_all_perfect() {
    let result = CrossValidationResult {
        accuracy_per_pop: [
            ("A".to_string(), 1.0),
            ("B".to_string(), 0.95),
            ("C".to_string(), 0.88),
        ].into(),
        overall_accuracy: 0.94,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };
    assert!(!result.has_bias(), "all >= 0.5 should not be biased");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validate integration: LOO with clear signal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cross_validate_loo_perfect_signal() {
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let mut observations = HashMap::new();
    // Generate observations for all haplotypes
    for (pop_idx, pop) in pops.iter().enumerate() {
        for hap in &pop.haplotypes {
            let obs = (0..10)
                .map(|i| make_observation(&pops, pop_idx, i))
                .collect::<Vec<_>>();
            // Update sample field to match haplotype ID
            let obs: Vec<AncestryObservation> = obs.into_iter().map(|mut o| {
                o.sample = hap.clone();
                o
            }).collect();
            observations.insert(hap.clone(), obs);
        }
    }

    let result = cross_validate(&observations, &pops, &params);

    // With clear signal, LOO should work well
    assert!(result.overall_accuracy >= 0.5);
    assert!(!result.has_bias());

    // Both pops should have been tested
    for pop in &pops {
        assert!(
            *result.n_windows_per_pop.get(&pop.name).unwrap_or(&0) > 0,
            "pop {} should have windows tested",
            pop.name
        );
    }
}
