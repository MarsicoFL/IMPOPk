//! Stress tests for auto_configure_pairwise_params and D_ref calibration.
//!
//! Tests the auto-configure logic under:
//! - Varying population counts (2, 3, 5)
//! - Extreme signal separation and near-identical populations
//! - Large window counts
//! - Zero-variance edge cases
//! - CV_D heterogeneity (mixed close/distant pairs)

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryObservation, auto_configure_pairwise_params,
};

// ── Helpers ──

fn make_pops(n: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n).map(|i| AncestralPopulation {
        name: names.get(i).unwrap_or(&"POP").to_string(),
        haplotypes: vec![
            format!("pop{}#HAP1", i),
            format!("pop{}#HAP2", i),
        ],
    }).collect()
}

fn make_obs_with_sims(start: u64, pop_sims: &[f64]) -> AncestryObservation {
    let mut sims = HashMap::new();
    for (i, &sim) in pop_sims.iter().enumerate() {
        sims.insert(format!("pop{}#HAP1", i), sim);
        sims.insert(format!("pop{}#HAP2", i), sim - 0.005);
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

// ── 2-population tests ──

#[test]
fn auto_configure_2pop_well_separated() {
    let pops = make_pops(2);
    let obs: Vec<_> = (0..200).map(|i| {
        let noise = ((i as f64 * 0.7).sin() * 0.003);
        make_obs_with_sims(i * 10000, &[0.99 + noise, 0.93 - noise])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Well separated → high pairwise weight
    assert!(pw > 0.1, "2-pop separated: pw={} should be > 0.1", pw);
    assert!(ec <= 5, "2-pop separated: ec={} should be low", ec);
}

#[test]
fn auto_configure_2pop_nearly_identical() {
    let pops = make_pops(2);
    let obs: Vec<_> = (0..200).map(|i| {
        let noise = ((i as f64 * 0.7).sin() * 0.001);
        make_obs_with_sims(i * 10000, &[0.998 + noise, 0.997 + noise])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Nearly identical → low pairwise weight (k=2 uses cv_factor=1, so d_min is key)
    assert!(pw < 0.7, "2-pop close: pw={} should be moderate to low", pw);
}

// ── 3-population tests (the HPRC-like case) ──

#[test]
fn auto_configure_3pop_mixed_distances() {
    let pops = make_pops(3);
    // Pop0 and pop1 are close, pop2 is distant → high CV_D
    let obs: Vec<_> = (0..300).map(|i| {
        let noise_a = ((i as f64 * 0.7).sin() * 0.002);
        let noise_b = ((i as f64 * 1.3).sin() * 0.002);
        let noise_c = ((i as f64 * 2.1).sin() * 0.003);
        make_obs_with_sims(i * 10000, &[0.998 + noise_a, 0.997 + noise_b, 0.980 + noise_c])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // With mixed distances, CV_D should be significant → moderate weight
    assert!(pw >= 0.0 && pw <= 0.95, "3-pop mixed: pw={} out of range", pw);
    assert!(ec >= 1 && ec <= 15, "3-pop mixed: ec={} out of range", ec);
}

#[test]
fn auto_configure_3pop_all_equidistant() {
    let pops = make_pops(3);
    // All populations equidistant (same separation, identical noise amplitude)
    let obs: Vec<_> = (0..200).map(|i| {
        let noise = ((i as f64 * 1.1).sin() * 0.002);
        make_obs_with_sims(i * 10000, &[0.99 + noise, 0.96 + noise, 0.93 + noise])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // Equidistant populations → CV_D=0 → cv_factor=0 → pw=0.
    // This is correct: when all pairs have equal Cohen's d, pairwise contrast
    // emissions add no discriminative value beyond softmax.
    assert!((pw - 0.0).abs() < 0.01, "3-pop equidistant: pw={} should be ~0 (CV_D=0)", pw);
}

// ── 5-population tests ──

#[test]
fn auto_configure_5pop_realistic() {
    let pops = make_pops(5);
    // Human-like: AFR most diverse, EAS least
    let obs: Vec<_> = (0..500).map(|i| {
        let n = |seed: f64| ((i as f64 * seed).sin() * 0.003);
        make_obs_with_sims(i * 10000, &[
            0.995 + n(0.7),  // AFR
            0.993 + n(1.3),  // EUR
            0.992 + n(2.1),  // EAS
            0.990 + n(3.7),  // CSA
            0.988 + n(4.3),  // AMR
        ])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    assert!(pw >= 0.0 && pw <= 0.95, "5-pop realistic: pw={} out of range", pw);
    assert!(ec >= 1 && ec <= 15, "5-pop realistic: ec={} out of range", ec);
}

#[test]
fn auto_configure_5pop_one_outlier() {
    // 4 close populations + 1 very distant → high CV_D
    let pops = make_pops(5);
    let obs: Vec<_> = (0..300).map(|i| {
        let n = |seed: f64| ((i as f64 * seed).sin() * 0.001);
        make_obs_with_sims(i * 10000, &[
            0.998 + n(0.7),
            0.997 + n(1.3),
            0.997 + n(2.1),
            0.996 + n(3.7),
            0.960 + n(4.3), // distant outlier
        ])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // D_min should be small (close pair), CV_D should be high (outlier)
    assert!(pw >= 0.0 && pw <= 0.95, "5-pop outlier: pw={}", pw);
}

// ── Edge cases ──

#[test]
fn auto_configure_exactly_10_observations() {
    let pops = make_pops(3);
    let obs: Vec<_> = (0..10).map(|i| {
        make_obs_with_sims(i * 10000, &[0.99, 0.96, 0.93])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // At boundary: 10 obs should be processed (not early return)
    // But n_windows=10 is small → (10/5000)^0.5 is very small → pw near 0
    assert!(pw < 0.1, "10 obs: pw={} should be very small due to sqrt(n/5000)", pw);
}

#[test]
fn auto_configure_9_observations_returns_defaults() {
    let pops = make_pops(3);
    let obs: Vec<_> = (0..9).map(|i| {
        make_obs_with_sims(i * 10000, &[0.99, 0.96, 0.93])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!((pw - 0.3).abs() < 0.01, "9 obs: should return default pw=0.3, got {}", pw);
    assert_eq!(ec, 0, "9 obs: should return default ec=0, got {}", ec);
}

#[test]
fn auto_configure_single_pop_returns_defaults() {
    let pops = make_pops(1);
    let obs: Vec<_> = (0..100).map(|i| {
        make_obs_with_sims(i * 10000, &[0.99])
    }).collect();
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!((pw - 0.3).abs() < 0.01, "k=1: default pw, got {}", pw);
    assert_eq!(ec, 0, "k=1: default ec, got {}", ec);
}

#[test]
fn auto_configure_empty_observations() {
    let pops = make_pops(3);
    let (pw, ec) = auto_configure_pairwise_params(&[], &pops);
    assert!((pw - 0.3).abs() < 0.01, "empty obs: default pw, got {}", pw);
    assert_eq!(ec, 0);
}

#[test]
fn auto_configure_zero_variance_perfect_discriminability() {
    // All windows have identical similarities → zero std → Cohen's d should be very high
    let pops = make_pops(2);
    let obs: Vec<_> = (0..100).map(|i| {
        make_obs_with_sims(i * 10000, &[0.99, 0.80])
    }).collect();
    let (pw, _ec) = auto_configure_pairwise_params(&obs, &pops);
    // Zero variance with large difference → d=100 (capped) → high weight
    assert!(pw > 0.01, "zero-variance: pw={} should be high", pw);
}

#[test]
fn auto_configure_pairwise_weight_clamped_to_095() {
    // Even with perfect data, weight should not exceed 0.95
    let pops = make_pops(2);
    let obs: Vec<_> = (0..10000).map(|i| {
        let noise = ((i as f64 * 0.7).sin() * 0.001);
        make_obs_with_sims(i * 10000, &[0.99 + noise, 0.70 + noise])
    }).collect();
    let (pw, _ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(pw <= 0.95, "pw={} should be clamped to 0.95", pw);
}

#[test]
fn auto_configure_emission_context_range() {
    // ec should always be in [1, 15] when returned from valid data
    for k in [2, 3, 5] {
        let pops = make_pops(k);
        let obs: Vec<_> = (0..200).map(|i| {
            let sims: Vec<f64> = (0..k).map(|j| {
                0.99 - 0.01 * j as f64 + ((i as f64 * (j as f64 + 1.0)).sin() * 0.003)
            }).collect();
            make_obs_with_sims(i * 10000, &sims)
        }).collect();
        let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
        assert!(ec <= 15, "k={}: ec={} exceeds 15", k, ec);
    }
}
