//! Tests for estimate_temperature_with_spread — temperature estimation with
//! optional multi-population spread ratio boost (k≥3).
//!
//! This function computes base temperature from the upper-quartile mean of
//! per-window (max_sim - min_sim), then for k≥3 populations applies a spread
//! ratio boost when the widest pairwise median is much larger than the closest.

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryObservation,
    estimate_temperature, estimate_temperature_with_spread,
};

// ── Helpers ──

fn make_pops(names: &[&str], n_haps: usize) -> Vec<AncestralPopulation> {
    names.iter().map(|&name| AncestralPopulation {
        name: name.to_string(),
        haplotypes: (0..n_haps)
            .map(|i| format!("{}#HAP{}", name, i + 1))
            .collect(),
    }).collect()
}

fn make_obs_with_sims(
    start: u64,
    pops: &[AncestralPopulation],
    pop_sims: &[f64],
) -> AncestryObservation {
    let mut sims = HashMap::new();
    for (pop_idx, pop) in pops.iter().enumerate() {
        for hap in &pop.haplotypes {
            // Slight variation per haplotype
            let base = pop_sims[pop_idx];
            sims.insert(hap.clone(), base);
        }
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

// ── Tests ──

#[test]
fn empty_observations_returns_default() {
    let pops = make_pops(&["AFR", "EUR"], 2);
    let temp = estimate_temperature_with_spread(&[], &pops, None);
    assert!((temp - 0.03).abs() < 1e-10, "empty obs should return default 0.03: {}", temp);
}

#[test]
fn single_observation_returns_valid_temperature() {
    let pops = make_pops(&["AFR", "EUR"], 2);
    let obs = vec![make_obs_with_sims(0, &pops, &[0.97, 0.92])];
    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    assert!(temp > 0.0 && temp <= 0.15, "should be in valid range: {}", temp);
}

#[test]
fn two_populations_no_spread_boost() {
    // With k=2, there's no spread ratio boost — just base temperature
    let pops = make_pops(&["AFR", "EUR"], 2);
    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.005;
        make_obs_with_sims(i * 10000, &pops, &[0.97 + noise, 0.90 + noise])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    // Base temp should reflect the ~0.07 gap between populations
    assert!(temp > 0.01, "temp should reflect population gap: {}", temp);
    assert!(temp <= 0.15, "temp should be within clamp: {}", temp);
}

#[test]
fn three_populations_close_pair_gets_spread_boost() {
    // 3 pops: AFR far from EUR/EAS, but EUR and EAS very close.
    // Spread ratio = widest/closest should be large → boost applied.
    let pops = make_pops(&["AFR", "EUR", "EAS"], 2);

    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.002;
        // AFR: 0.97, EUR: 0.92, EAS: 0.915 — EUR-EAS gap is tiny (0.005)
        make_obs_with_sims(i * 10000, &pops, &[0.97 + noise, 0.92 + noise, 0.915 + noise])
    }).collect();

    let temp_3pop = estimate_temperature_with_spread(&obs, &pops, None);

    // Compare with 2-pop (just AFR+EUR) — no spread boost
    let pops2 = make_pops(&["AFR", "EUR"], 2);
    let obs2: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.002;
        make_obs_with_sims(i * 10000, &pops2, &[0.97 + noise, 0.92 + noise])
    }).collect();
    let temp_2pop = estimate_temperature_with_spread(&obs2, &pops2, None);

    // 3-pop with close pair should have higher temp (spread boost)
    assert!(temp_3pop > temp_2pop,
        "3-pop with close pair should boost temp: 3pop={}, 2pop={}", temp_3pop, temp_2pop);
}

#[test]
fn three_populations_equidistant_no_excessive_boost() {
    // 3 equidistant pops → spread ratio ≈ 1 → no boost (< 2.0 threshold)
    let pops = make_pops(&["AFR", "EUR", "EAS"], 2);

    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.002;
        // Equal spacing: 0.97, 0.93, 0.89 — all gaps ≈ 0.04
        make_obs_with_sims(i * 10000, &pops, &[0.97 + noise, 0.93 + noise, 0.89 + noise])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    // Should be reasonable — equidistant pops don't trigger large boost
    assert!(temp >= 0.0005 && temp <= 0.15, "temp should be within clamp: {}", temp);
}

#[test]
fn estimate_temperature_wrapper_matches_none_spread() {
    // estimate_temperature(obs, pops) == estimate_temperature_with_spread(obs, pops, None)
    let pops = make_pops(&["AFR", "EUR", "EAS"], 2);
    let obs: Vec<AncestryObservation> = (0..50).map(|i| {
        let noise = (i as f64 * 0.5).sin() * 0.003;
        make_obs_with_sims(i * 10000, &pops, &[0.97 + noise, 0.91 + noise, 0.94 + noise])
    }).collect();

    let temp1 = estimate_temperature(&obs, &pops);
    let temp2 = estimate_temperature_with_spread(&obs, &pops, None);
    assert!((temp1 - temp2).abs() < 1e-15, "wrapper should match: {} vs {}", temp1, temp2);
}

#[test]
fn raw_spread_source_affects_boost() {
    // When raw_for_spread is provided, spread ratio uses raw data instead of
    // the (possibly smoothed) main observations.
    let pops = make_pops(&["AFR", "EUR", "EAS"], 2);

    // Main obs: close pops (smoothed away the difference)
    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.001;
        make_obs_with_sims(i * 10000, &pops, &[0.95 + noise, 0.94 + noise, 0.935 + noise])
    }).collect();

    // Raw obs: large pairwise spread (unsmoothed)
    let raw: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.002;
        make_obs_with_sims(i * 10000, &pops, &[0.97 + noise, 0.92 + noise, 0.915 + noise])
    }).collect();

    let temp_no_raw = estimate_temperature_with_spread(&obs, &pops, None);
    let temp_with_raw = estimate_temperature_with_spread(&obs, &pops, Some(&raw));

    // With raw data showing bigger pairwise spread, boost should be larger
    assert!(temp_with_raw > temp_no_raw,
        "raw spread source should increase boost: with_raw={}, no_raw={}",
        temp_with_raw, temp_no_raw);
}

#[test]
fn temperature_clamped_low() {
    // Very similar populations → very small diffs → temp clamped to 0.0005
    let pops = make_pops(&["AFR", "EUR"], 2);
    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        // Nearly identical: gap = 0.0001
        make_obs_with_sims(i * 10000, &pops, &[0.95, 0.9499])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    assert!(temp >= 0.0005, "temp should be clamped at lower bound: {}", temp);
}

#[test]
fn temperature_clamped_high() {
    // Huge separation → temp clamped to 0.15
    let pops = make_pops(&["AFR", "EUR", "EAS"], 2);
    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        // Extreme separation + large spread ratio
        let noise = (i as f64 * 0.3).sin() * 0.01;
        make_obs_with_sims(i * 10000, &pops, &[0.99 + noise, 0.50 + noise, 0.495 + noise])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    assert!(temp <= 0.15, "temp should be clamped at upper bound: {}", temp);
}

#[test]
fn no_valid_diffs_returns_default() {
    // All pops have identical similarity → max == min → no diffs recorded
    let pops = make_pops(&["AFR", "EUR"], 2);
    let obs: Vec<AncestryObservation> = (0..50).map(|i| {
        make_obs_with_sims(i * 10000, &pops, &[0.95, 0.95])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    // With max == min for all windows, all_diffs is empty → default 0.03
    assert!((temp - 0.03).abs() < 1e-10, "identical pops should return default: {}", temp);
}

#[test]
fn single_population_returns_default() {
    // Single population → can't compute pairwise diffs
    let pops = make_pops(&["AFR"], 2);
    let obs: Vec<AncestryObservation> = (0..50).map(|i| {
        make_obs_with_sims(i * 10000, &pops, &[0.95])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    // pop_sims.len() < 2 → no diffs → default
    assert!((temp - 0.03).abs() < 1e-10, "single pop should return default: {}", temp);
}

#[test]
fn five_populations_spread_ratio() {
    // 5 populations with varying distances — spread ratio boost should kick in
    let pops = make_pops(&["AFR", "EUR", "EAS", "CSA", "AMR"], 2);

    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let noise = (i as f64 * 0.3).sin() * 0.002;
        // AFR far, EUR/AMR close, EAS/CSA in between
        make_obs_with_sims(i * 10000, &pops, &[
            0.97 + noise, 0.92 + noise, 0.89 + noise, 0.885 + noise, 0.918 + noise
        ])
    }).collect();

    let temp = estimate_temperature_with_spread(&obs, &pops, None);
    assert!(temp >= 0.0005 && temp <= 0.15, "5-pop temp should be in valid range: {}", temp);
    // EUR-AMR gap ≈ 0.002, AFR-EAS gap ≈ 0.08 → spread ratio > 2 → boost
    assert!(temp > 0.01, "should have spread boost with 5 pops: {}", temp);
}
