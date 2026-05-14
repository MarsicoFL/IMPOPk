//! Tests for auto_configure_pairwise_params robustness with diverse input patterns.
//!
//! Originally tested Tukey fence outlier filtering (T70), which was removed after
//! B81 proved it HURTS on HPRC PAF data (IQR≈0, catches 11-13% of normal windows,
//! inflating D_min 17-18x). These tests now verify that auto_configure_pairwise_params
//! produces valid output across challenging input distributions:
//! - Mixed normal + extreme-outlier windows
//! - Few windows (below/at statistical thresholds)
//! - Uniform variance, near-zero signal, bimodal distributions
//! - Various outlier fractions

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryObservation, auto_configure_pairwise_params,
};

// ── Helpers ──

fn make_pops(n: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n)
        .map(|i| AncestralPopulation {
            name: names.get(i).unwrap_or(&"POP").to_string(),
            haplotypes: vec![
                format!("pop{}#HAP1", i),
                format!("pop{}#HAP2", i),
            ],
        })
        .collect()
}

fn make_obs(start: u64, pop_sims: &[f64]) -> AncestryObservation {
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

// ── Auto-configure robustness tests ──

#[test]
fn outlier_windows_included_affect_dmin() {
    // 3 populations with moderate separation. 95 normal windows + 5 extreme SD windows.
    // All windows are included (no filtering). Each pop gets INDEPENDENT noise so CV_D > 0.
    let pops = make_pops(3);

    let normal: Vec<AncestryObservation> = (0..95).map(|i| {
        let t = i as f64;
        let n0 = (t * 0.3).sin() * 0.005;
        let n1 = (t * 0.7).sin() * 0.005;
        let n2 = (t * 1.1).sin() * 0.005;
        make_obs(i * 10000, &[0.97 + n0, 0.90 + n1, 0.94 + n2])
    }).collect();

    let extreme: Vec<AncestryObservation> = (0..5).map(|i| {
        make_obs((95 + i) * 10000, &[0.99, 0.10, 0.50])
    }).collect();

    let mut with_outliers = normal.clone();
    with_outliers.extend(extreme);

    let (pw_filtered, ec_filtered) = auto_configure_pairwise_params(&with_outliers, &pops);
    let (pw_normal, ec_normal) = auto_configure_pairwise_params(&normal, &pops);

    // Both should produce valid output (not fallback defaults).
    assert!(pw_filtered > 0.0, "pw_filtered should be positive: {}", pw_filtered);
    assert!(pw_normal > 0.0, "pw_normal should be positive: {}", pw_normal);
    assert!(ec_filtered > 0, "ec_filtered should be > 0: {}", ec_filtered);
    assert!(ec_normal > 0, "ec_normal should be > 0: {}", ec_normal);
}

#[test]
fn fewer_than_20_windows_all_included() {
    // With fewer than 20 windows, all are included (no statistical thresholds).
    // Extreme variance windows contribute to D_min estimation.
    let pops = make_pops(2);

    let mut obs: Vec<AncestryObservation> = (0..15).map(|i| {
        let noise = (i as f64 * 0.5).sin() * 0.001;
        make_obs(i * 10000, &[0.97 + noise, 0.92 + noise])
    }).collect();

    // Add 4 extreme windows (total = 19, still < 20)
    for i in 0..4 {
        obs.push(make_obs((15 + i) * 10000, &[0.99, 0.10]));
    }

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Should still produce valid output including the extreme windows
    assert!(pw > 0.0, "pw should be positive even with outliers: {}", pw);
    // ec should be low because extreme windows inflate the separation
    assert!(ec >= 1, "ec should be at least 1: {}", ec);
}

#[test]
fn exactly_20_windows_valid_output() {
    // At exactly 20 windows, should produce valid D_min estimates.
    let pops = make_pops(2);

    let mut obs: Vec<AncestryObservation> = (0..18).map(|i| {
        let noise = (i as f64 * 0.4).sin() * 0.001;
        make_obs(i * 10000, &[0.96 + noise, 0.92 + noise])
    }).collect();

    // 2 extreme outlier windows → total 20
    obs.push(make_obs(18 * 10000, &[0.99, 0.05]));
    obs.push(make_obs(19 * 10000, &[0.99, 0.05]));

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Should produce valid output including outlier windows
    assert!(pw > 0.0, "pw should be positive: {}", pw);
    assert!(ec >= 1, "ec should be at least 1: {}", ec);
}

#[test]
fn uniform_variance_no_exclusion() {
    // When all windows have very similar variance, all contribute equally.
    // Use independent noise for CV_D > 0.
    let pops = make_pops(3);

    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        let t = i as f64;
        let n0 = (t * 0.3).sin() * 0.003;
        let n1 = (t * 0.7).sin() * 0.003;
        let n2 = (t * 1.1).sin() * 0.003;
        make_obs(i * 10000, &[0.97 + n0, 0.90 + n1, 0.94 + n2])
    }).collect();

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(pw > 0.0, "pw should be positive: {}", pw);
    assert!(ec >= 1, "ec should be at least 1: {}", ec);
}

#[test]
fn near_zero_iqr_with_single_extreme_outlier() {
    // B81 scenario: most windows have near-zero pop-sim variance (no signal),
    // but a few windows have huge variance. Without filtering, the few high-signal
    // windows are included but diluted by the 97 no-signal windows.
    let pops = make_pops(2);

    // 97 windows with nearly identical pop_sims (no discrimination)
    let mut obs: Vec<AncestryObservation> = (0..97).map(|i| {
        let noise = (i as f64 * 0.2).sin() * 0.00001;
        make_obs(i * 10000, &[0.95 + noise, 0.95 + noise])
    }).collect();

    // 3 windows with real signal (these are the "useful" ones)
    obs.push(make_obs(97 * 10000, &[0.98, 0.90]));
    obs.push(make_obs(98 * 10000, &[0.97, 0.91]));
    obs.push(make_obs(99 * 10000, &[0.99, 0.89]));

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // In this scenario (B81), the useful windows are diluted by the 97
    // near-zero-signal windows. The result should still be valid (not panic),
    // though D_min estimation will be dominated by the no-signal majority.
    assert!(pw >= 0.0, "pw should be non-negative: {}", pw);
    assert!(ec <= 15, "ec should be within clamp range: {}", ec);
}

#[test]
fn all_windows_extreme_returns_valid_output() {
    // If every window has extreme variance, they all contribute to
    // D_min estimation — consistent high separation across the genome.
    let pops = make_pops(2);

    let obs: Vec<AncestryObservation> = (0..50).map(|i| {
        // All windows have huge separation between pops
        let shift = (i as f64 * 0.2).sin() * 0.02;
        make_obs(i * 10000, &[0.99 + shift, 0.60 + shift])
    }).collect();

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // All windows have similar high separation → D_min reflects consistent signal
    assert!(pw > 0.0, "pw should be positive: {}", pw);
    assert!(ec >= 1, "ec should be at least 1: {}", ec);
}

#[test]
fn moderate_outlier_fraction_still_works() {
    // ~20% outlier windows mixed with normal windows.
    // Use independent noise per population for CV_D > 0.
    let pops = make_pops(3);

    let mut obs: Vec<AncestryObservation> = (0..80).map(|i| {
        let t = i as f64;
        let n0 = (t * 0.3).sin() * 0.005;
        let n1 = (t * 0.7).sin() * 0.005;
        let n2 = (t * 1.1).sin() * 0.005;
        make_obs(i * 10000, &[0.96 + n0, 0.89 + n1, 0.93 + n2])
    }).collect();

    // 20 extreme "SD" windows
    for i in 0..20 {
        let shift = (i as f64 * 0.5).sin() * 0.01;
        obs.push(make_obs((80 + i) * 10000, &[0.99 + shift, 0.10 + shift, 0.50 + shift]));
    }

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(pw > 0.0, "pw should be positive with 20% outliers: {}", pw);
    assert!(ec >= 1, "ec should be at least 1 with 20% outliers: {}", ec);
}

#[test]
fn separation_preserves_correct_ordering() {
    // The relative ec ordering should be:
    // well-separated → lower ec; poorly-separated → higher ec.
    // Use independent noise so D_min is finite and reflects true separation.
    let pops = make_pops(2);

    // Well-separated base + outliers (0.08 mean gap)
    let mut separated: Vec<AncestryObservation> = (0..90).map(|i| {
        let t = i as f64;
        let n0 = (t * 0.3).sin() * 0.010;
        let n1 = (t * 0.7).sin() * 0.010;
        make_obs(i * 10000, &[0.97 + n0, 0.89 + n1])
    }).collect();
    for i in 0..10 {
        separated.push(make_obs((90 + i) * 10000, &[0.99, 0.10]));
    }

    // Poorly-separated base + outliers (0.01 mean gap)
    let mut close: Vec<AncestryObservation> = (0..90).map(|i| {
        let t = i as f64;
        let n0 = (t * 0.3).sin() * 0.010;
        let n1 = (t * 0.7).sin() * 0.010;
        make_obs(i * 10000, &[0.95 + n0, 0.94 + n1])
    }).collect();
    for i in 0..10 {
        close.push(make_obs((90 + i) * 10000, &[0.99, 0.10]));
    }

    let (_, ec_sep) = auto_configure_pairwise_params(&separated, &pops);
    let (_, ec_close) = auto_configure_pairwise_params(&close, &pops);

    // Higher D_min → lower ec; lower D_min → higher ec
    assert!(ec_sep <= ec_close,
        "Well-separated (ec={}) should have lower or equal ec than poorly-separated (ec={})",
        ec_sep, ec_close);
}

#[test]
fn gradual_variance_increase_valid_output() {
    // Windows with gradually increasing variance. All windows contribute,
    // including the extreme tail.
    let pops = make_pops(2);

    let obs: Vec<AncestryObservation> = (0..100).map(|i| {
        // Gradually increase separation: first 90 are close, last 10 are extreme
        let sep = if i < 90 {
            0.01 + 0.001 * (i as f64 / 90.0)
        } else {
            0.30 + 0.05 * ((i - 90) as f64) // huge jump
        };
        make_obs(i * 10000, &[0.95 + sep / 2.0, 0.95 - sep / 2.0])
    }).collect();

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Should produce valid output. All windows contribute including
    // the extreme tail, which inflates D_min estimation.
    assert!(pw > 0.0, "pw should be positive: {}", pw);
    assert!(ec >= 1, "ec should be at least 1: {}", ec);
}

#[test]
fn bimodal_variance_distribution() {
    // Two clusters of windows: low-variance (ancestry signal) and high-variance (SD regions).
    // Both clusters contribute to D_min estimation.
    let pops = make_pops(3);

    // Cluster 1: normal ancestry signal (70 windows)
    let mut obs: Vec<AncestryObservation> = (0..70).map(|i| {
        let noise = (i as f64 * 0.4).sin() * 0.002;
        make_obs(i * 10000, &[0.97 + noise, 0.92 + noise, 0.95 + noise])
    }).collect();

    // Cluster 2: SD regions with wild variance (30 windows)
    for i in 0..30 {
        let phase = (i as f64 * 0.7).sin();
        obs.push(make_obs((70 + i) * 10000, &[
            0.95 + 0.04 * phase,
            0.30 + 0.20 * phase,
            0.60 + 0.10 * phase,
        ]));
    }

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Both clusters contribute; SD cluster inflates D_min
    assert!(pw > 0.0, "pw should be positive: {}", pw);
    assert!(ec >= 1 && ec <= 15, "ec should be in valid range: {}", ec);
}

#[test]
fn ten_windows_returns_valid_output() {
    // Exactly 10 windows (minimum for non-default return).
    let pops = make_pops(2);

    let obs: Vec<AncestryObservation> = (0..10).map(|i| {
        let noise = (i as f64 * 0.5).sin() * 0.002;
        make_obs(i * 10000, &[0.97 + noise, 0.91 + noise])
    }).collect();

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // 10 windows, enough for D_min computation
    assert!(pw > 0.0, "pw should be positive with 10 windows: {}", pw);
}

#[test]
fn nine_windows_returns_default() {
    // 9 windows → below minimum threshold → returns default (0.3, 0)
    let pops = make_pops(2);

    let obs: Vec<AncestryObservation> = (0..9).map(|i| {
        make_obs(i * 10000, &[0.97, 0.91])
    }).collect();

    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!((pw - 0.3).abs() < 1e-10, "pw should be default 0.3 with < 10 windows: {}", pw);
    assert_eq!(ec, 0, "ec should be default 0 with < 10 windows: {}", ec);
}
