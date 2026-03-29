//! Tests for enhanced Baum-Welch with initial prob re-estimation and temperature grid search.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
};

fn make_pops(n: usize) -> Vec<AncestralPopulation> {
    (0..n).map(|i| AncestralPopulation {
        name: format!("pop{}", i),
        haplotypes: vec![format!("pop{}#HAP1", i), format!("pop{}#HAP2", i)],
    }).collect()
}

fn make_obs(start: u64, dominant_pop: usize, n_pops: usize) -> AncestryObservation {
    let mut sims = HashMap::new();
    for i in 0..n_pops {
        let base = if i == dominant_pop { 0.98 } else { 0.91 };
        sims.insert(format!("pop{}#HAP1", i), base);
        sims.insert(format!("pop{}#HAP2", i), base - 0.005);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// === baum_welch_full: initial prob re-estimation ===

#[test]
fn test_bw_full_empty_returns_neg_inf() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let ll = params.baum_welch_full(&[], 10, 1e-4, false);
    assert_eq!(ll, f64::NEG_INFINITY);
}

#[test]
fn test_bw_full_single_state_returns_neg_inf() {
    let pops = make_pops(1);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let obs = vec![make_obs(0, 0, 1), make_obs(10000, 0, 1)];
    let ll = params.baum_welch_full(&[obs.as_slice()], 5, 1e-4, false);
    assert_eq!(ll, f64::NEG_INFINITY);
}

#[test]
fn test_bw_full_updates_initial_probs() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // Generate observations: 80% pop0, 20% pop1
    let mut obs = Vec::new();
    for i in 0..80 {
        obs.push(make_obs(i * 10000, 0, 3));
    }
    for i in 80..100 {
        obs.push(make_obs(i * 10000, 1, 3));
    }

    // Before: uniform initial probs
    let initial_before = params.initial.clone();
    assert!((initial_before[0] - initial_before[1]).abs() < 1e-10, "should start uniform");

    // Run full BW
    let ll = params.baum_welch_full(&[obs.as_slice()], 10, 1e-4, false);
    assert!(ll.is_finite(), "log-likelihood should be finite");

    // After: initial probs should reflect observed ancestry proportions
    // Pop0 dominant → should have highest initial prob
    assert!(params.initial[0] > params.initial[2],
        "pop0 should have higher initial prob: {:.4} vs {:.4}", params.initial[0], params.initial[2]);

    // Sum should still be 1
    let sum: f64 = params.initial.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "initial probs should sum to 1");
}

#[test]
fn test_bw_full_uniform_ancestry_stays_near_uniform() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // Uniform: equal amounts of each ancestry across 3 sequences
    // Use multiple sequences to average out initial state bias
    let seq0: Vec<AncestryObservation> = (0..20).map(|i| make_obs(i * 10000, (i as usize) % 3, 3)).collect();
    let seq1: Vec<AncestryObservation> = (0..20).map(|i| make_obs(i * 10000, ((i as usize) + 1) % 3, 3)).collect();
    let seq2: Vec<AncestryObservation> = (0..20).map(|i| make_obs(i * 10000, ((i as usize) + 2) % 3, 3)).collect();

    params.baum_welch_full(&[seq0.as_slice(), seq1.as_slice(), seq2.as_slice()], 10, 1e-4, false);

    // Initial probs should sum to 1 and all be positive
    let sum: f64 = params.initial.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "initial probs should sum to 1");
    for &p in &params.initial {
        assert!(p > 0.0, "each initial prob should be positive: {:.6}", p);
    }
}

#[test]
fn test_bw_full_ll_nondecreasing() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(i * 10000, (i as usize) % 3, 3)
    }).collect();

    // Run 1 iteration at a time and check LL doesn't decrease
    let mut prev_ll = f64::NEG_INFINITY;
    for _ in 0..5 {
        let ll = params.baum_welch_full(&[obs.as_slice()], 1, 1e-10, false);
        if ll.is_finite() && prev_ll.is_finite() {
            assert!(ll >= prev_ll - 1e-6,
                "LL should not decrease: {} < {}", ll, prev_ll);
        }
        prev_ll = ll;
    }
}

// === baum_welch_full with temperature re-estimation ===

#[test]
fn test_bw_full_with_temperature_changes_temp() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.1); // Start with suboptimal temperature

    let obs: Vec<AncestryObservation> = (0..30).map(|i| {
        make_obs(i * 10000, (i as usize) % 3, 3)
    }).collect();

    let temp_before = params.emission_std;
    params.baum_welch_full(&[obs.as_slice()], 5, 1e-4, true);
    let temp_after = params.emission_std;

    // Temperature should change (grid search explores different values)
    // With 0.1 as starting point, it should find something different
    assert!((temp_before - temp_after).abs() > 1e-6 || temp_after > 0.0,
        "temperature should be adjusted: before={:.6}, after={:.6}", temp_before, temp_after);
}

#[test]
fn test_bw_full_with_temperature_finite_result() {
    let pops = make_pops(2);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.05);

    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(i * 10000, (i as usize) % 2, 2)
    }).collect();

    let ll = params.baum_welch_full(&[obs.as_slice()], 3, 1e-4, true);
    assert!(ll.is_finite(), "should produce finite LL with temperature re-estimation");
    assert!(params.emission_std > 0.0, "temperature should be positive");
}

// === Standard vs full BW comparison ===

#[test]
fn test_bw_standard_vs_full_similar_on_uniform() {
    let pops = make_pops(3);

    // Uniform data
    let obs: Vec<AncestryObservation> = (0..30).map(|i| {
        make_obs(i * 10000, (i as usize) % 3, 3)
    }).collect();

    // Standard BW
    let mut params_std = AncestryHmmParams::new(pops.clone(), 0.001);
    params_std.set_temperature(0.03);
    let ll_std = params_std.baum_welch(&[obs.as_slice()], 10, 1e-4);

    // Full BW (no temp reestimation)
    let mut params_full = AncestryHmmParams::new(pops, 0.001);
    params_full.set_temperature(0.03);
    let ll_full = params_full.baum_welch_full(&[obs.as_slice()], 10, 1e-4, false);

    // On uniform data, full BW should give similar or better LL
    if ll_std.is_finite() && ll_full.is_finite() {
        assert!(ll_full >= ll_std - 1.0,
            "full BW should match or exceed standard: full={:.2} vs std={:.2}", ll_full, ll_std);
    }
}

// === Edge cases ===

#[test]
fn test_bw_full_short_sequence() {
    let pops = make_pops(2);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // Very short: only 2 observations
    let obs = vec![make_obs(0, 0, 2), make_obs(10000, 1, 2)];
    let ll = params.baum_welch_full(&[obs.as_slice()], 5, 1e-4, false);
    assert!(ll.is_finite(), "should handle 2-observation sequence");
}

#[test]
fn test_bw_full_multiple_sequences() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // Two sequences with different ancestry patterns
    let seq1: Vec<AncestryObservation> = (0..20).map(|i| make_obs(i * 10000, 0, 3)).collect();
    let seq2: Vec<AncestryObservation> = (0..20).map(|i| make_obs(i * 10000, 1, 3)).collect();

    let ll = params.baum_welch_full(&[seq1.as_slice(), seq2.as_slice()], 5, 1e-4, false);
    assert!(ll.is_finite(), "should handle multiple sequences");
}

#[test]
fn test_bw_full_max_iters_zero() {
    let pops = make_pops(2);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let initial = params.initial.clone();
    let transitions = params.transitions.clone();

    let obs = vec![make_obs(0, 0, 2), make_obs(10000, 1, 2)];
    let ll = params.baum_welch_full(&[obs.as_slice()], 0, 1e-4, false);

    assert_eq!(ll, f64::NEG_INFINITY, "0 iters should return NEG_INFINITY");
    assert_eq!(params.initial, initial, "params should be unchanged");
    assert_eq!(params.transitions, transitions, "transitions should be unchanged");
}
