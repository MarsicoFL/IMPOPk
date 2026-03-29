//! Tests for the D_min-based emission_context formula (cd66691).
//!
//! The new formula: ec = (0.1 / D_min).round().clamp(1, 15)
//! replaces the broken median_disc approach that always gave ~0.0003
//! regardless of data regime (B52 benchmark).
//!
//! Tests verify:
//! - Correct ec for known D_min regimes (sim, HPRC, boundary)
//! - ec clamp behavior at extremes
//! - Differentiation between sim-like and HPRC-like data
//! - pairwise_weight sensitivity to n_windows scaling
//! - cv_factor k=2 vs k≥3 behavior
//! - D_ref=0.014 calibration point

use std::collections::HashMap;
use hprc_ancestry_cli::{
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

/// Build an observation with explicit per-population similarities.
/// Each pop gets two haplotypes: HAP1=sim, HAP2=sim-0.005.
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

/// Generate observations where pop_sims[i] = base[i] + amp * sin(seed_i * t).
/// Returns (obs, approximate_d_min) where d_min is analytically estimated.
fn make_obs_with_noise(
    n: usize,
    bases: &[f64],
    noise_amp: f64,
) -> Vec<AncestryObservation> {
    let k = bases.len();
    (0..n)
        .map(|t| {
            let sims: Vec<f64> = (0..k)
                .map(|j| {
                    let seed = (j as f64 + 1.0) * 0.7;
                    bases[j] + noise_amp * (t as f64 * seed).sin()
                })
                .collect();
            make_obs(t as u64 * 10000, &sims)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// D_min → ec mapping: precision tests
// ec = (0.1 / D_min).round().clamp(1, 15)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ec_formula_zero_noise_gives_ec_1() {
    // Zero noise → std=0 → d=100 (capped) → ec = (0.1/100).round() = 0 → clamp to 1
    let pops = make_pops(2);
    let obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.93]))
        .collect();
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert_eq!(ec, 1, "zero noise → d=100 → ec should clamp to 1, got {}", ec);
}

#[test]
fn ec_formula_large_gap_small_noise_gives_low_ec() {
    // gap=0.06, noise_amp=0.003 → d ≈ 0.06/(0.003*sqrt(2)) ≈ 14.1
    // ec = (0.1/14.1).round() = 0 → clamp to 1
    let pops = make_pops(2);
    let obs = make_obs_with_noise(500, &[0.99, 0.93], 0.003);
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(ec <= 2, "large gap + small noise: ec={} should be ≤ 2", ec);
}

#[test]
fn ec_formula_sim_regime_d_min_around_025() {
    // Sim-like: d ≈ 0.025 → ec = (0.1/0.025).round() = 4
    // gap=0.001, noise_amp≈0.028 → d ≈ 0.001/(0.028*sqrt(2)) ≈ 0.025
    let pops = make_pops(2);
    let obs = make_obs_with_noise(1000, &[0.990, 0.989], 0.028);
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(
        ec >= 2 && ec <= 8,
        "sim-regime (D_min≈0.025): ec={} should be in [2,8]",
        ec
    );
}

#[test]
fn ec_formula_hprc_regime_d_min_around_003() {
    // HPRC-like: d ≈ 0.003 → ec = (0.1/0.003).round() = 33 → clamp to 15
    // gap very small relative to noise
    let pops = make_pops(3);
    // Pop means: 0.970, 0.9698, 0.9695 → smallest gap=0.0003
    // noise_amp=0.015 → d ≈ 0.0003/(0.015*sqrt(2)) ≈ 0.014
    // Actually for 3 pops, D_min is the min across all 3 pairs.
    // Pair(0,1): gap=0.0002, d≈0.0002/0.021≈0.0094
    // Pair(0,2): gap=0.0005, d≈0.0005/0.021≈0.024
    // Pair(1,2): gap=0.0003, d≈0.0003/0.021≈0.014
    // D_min ≈ 0.0094 → ec = (0.1/0.0094) ≈ 11
    let obs = make_obs_with_noise(500, &[0.970, 0.9698, 0.9695], 0.015);
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(
        ec >= 5,
        "HPRC-regime (tiny gaps, big noise): ec={} should be ≥ 5",
        ec
    );
}

#[test]
fn ec_formula_very_small_d_min_clamps_to_15() {
    // D_min near 1e-6 → ec = (0.1/1e-6) = 100000 → clamp to 15
    let pops = make_pops(2);
    // Nearly identical means with large noise: gap≈0, amp=0.01
    let obs: Vec<_> = (0..500)
        .map(|i| {
            let noise = (i as f64 * 0.7).sin() * 0.01;
            make_obs(i * 10000, &[0.990 + noise, 0.990 + noise * 0.999])
        })
        .collect();
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert_eq!(
        ec, 15,
        "near-zero D_min: ec should clamp to 15, got {}",
        ec
    );
}

#[test]
fn ec_formula_moderate_d_min_around_01() {
    // D_min ≈ 0.1 → ec = (0.1/0.1).round() = 1
    // gap=0.003, noise=0.021 → d ≈ 0.003/(0.021*sqrt(2)) ≈ 0.101
    let pops = make_pops(2);
    let obs = make_obs_with_noise(500, &[0.993, 0.990], 0.021);
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(
        ec <= 3,
        "D_min≈0.1: ec={} should be ≤ 3",
        ec
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regime differentiation: sim vs HPRC produce different ec values
// (the key regression the fix addresses — old code always gave ec=15)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sim_and_hprc_produce_different_ec() {
    let pops3 = make_pops(3);

    // Sim-like: well-separated populations
    let sim_obs = make_obs_with_noise(500, &[0.99, 0.96, 0.93], 0.003);
    let (_pw_sim, ec_sim) = auto_configure_pairwise_params(&sim_obs, &pops3);

    // HPRC-like: very close populations
    let hprc_obs = make_obs_with_noise(500, &[0.970, 0.9698, 0.9695], 0.015);
    let (_pw_hprc, ec_hprc) = auto_configure_pairwise_params(&hprc_obs, &pops3);

    // Sim should have lower ec (strong signal → less smoothing)
    // HPRC should have higher ec (weak signal → more smoothing)
    assert!(
        ec_hprc > ec_sim,
        "HPRC ec={} should be > sim ec={} (regime differentiation)",
        ec_hprc, ec_sim
    );
}

#[test]
fn sim_and_hprc_produce_different_pairwise_weight_k2() {
    // Use k=2 (cv_factor=1) so pw depends purely on D_min and n_windows.
    let pops = make_pops(2);

    // Sim-like: large gap, small noise → high D_min
    let sim_obs = make_obs_with_noise(500, &[0.99, 0.93], 0.003);
    let (pw_sim, _) = auto_configure_pairwise_params(&sim_obs, &pops);

    // HPRC-like: tiny gap, large noise → D_min well below 0.014
    // gap=0.0002, noise=0.030 → d ≈ 0.0002/0.030 ≈ 0.007
    let hprc_obs = make_obs_with_noise(500, &[0.9702, 0.970], 0.030);
    let (pw_hprc, _) = auto_configure_pairwise_params(&hprc_obs, &pops);

    // Sim has larger D_min → (d_min/0.014).min(1.0) is higher → higher pw
    assert!(
        pw_sim > pw_hprc,
        "sim pw={} should be > hprc pw={} (D_min is larger for sim)",
        pw_sim, pw_hprc
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// pairwise_weight: n_windows scaling factor
// w_star = 0.7 * cv_factor * (d_min/0.014).min(1) * (n/5000)^0.5.min(1)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pairwise_weight_increases_with_more_windows() {
    let pops = make_pops(2);

    // Same data pattern but different window counts
    let obs_200 = make_obs_with_noise(200, &[0.99, 0.93], 0.003);
    let (pw_200, _) = auto_configure_pairwise_params(&obs_200, &pops);

    let obs_2000 = make_obs_with_noise(2000, &[0.99, 0.93], 0.003);
    let (pw_2000, _) = auto_configure_pairwise_params(&obs_2000, &pops);

    assert!(
        pw_2000 >= pw_200,
        "more windows should increase pw: pw_2000={} vs pw_200={}",
        pw_2000, pw_200
    );
}

#[test]
fn pairwise_weight_saturates_at_5000_windows() {
    let pops = make_pops(2);

    let obs_5000 = make_obs_with_noise(5000, &[0.99, 0.93], 0.003);
    let (pw_5000, _) = auto_configure_pairwise_params(&obs_5000, &pops);

    let obs_10000 = make_obs_with_noise(10000, &[0.99, 0.93], 0.003);
    let (pw_10000, _) = auto_configure_pairwise_params(&obs_10000, &pops);

    // sqrt(n/5000).min(1) saturates at n=5000
    let ratio = pw_10000 / pw_5000.max(1e-10);
    assert!(
        ratio < 1.1,
        "pw should saturate at 5000 windows: pw_10000={}, pw_5000={}",
        pw_10000, pw_5000
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// cv_factor: k=2 vs k≥3
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_factor_k2_always_1() {
    // For k=2: cv_factor=1.0, so only D_min and n_windows matter
    let pops = make_pops(2);

    // Two well-separated pops → D_min large, cv_factor=1
    let obs = make_obs_with_noise(500, &[0.99, 0.93], 0.003);
    let (pw, _) = auto_configure_pairwise_params(&obs, &pops);

    // For k=2 with large D_min: w* = 0.7 * 1.0 * 1.0 * sqrt(500/5000)
    // = 0.7 * sqrt(0.1) = 0.7 * 0.316 ≈ 0.221
    // (D_min is huge so (d_min/0.014).min(1.0) = 1.0)
    assert!(
        pw > 0.1,
        "k=2 with high D_min: pw={} should be > 0.1 (cv_factor=1)",
        pw
    );
}

#[test]
fn cv_factor_k3_equidistant_has_low_cv_d() {
    // For k≥3: cv_factor = cv_d / (0.4 + cv_d)
    // Equidistant pops with same noise seed → CV_D should be relatively low.
    // Note: different sin frequencies per pop create slightly different std(diff),
    // so CV_D won't be exactly 0, but cv_factor should still suppress pw.
    let pops = make_pops(3);
    let obs = make_obs_with_noise(500, &[0.99, 0.96, 0.93], 0.003);
    let (pw, _) = auto_configure_pairwise_params(&obs, &pops);

    // With equidistant spacing, CV_D is low → cv_factor small → pw moderate
    // (D_min is very high so (d/D_ref) caps at 1.0)
    assert!(
        pw < 0.3,
        "k=3 equidistant: pw={} should be low due to small cv_factor",
        pw
    );
}

#[test]
fn cv_factor_k3_heterogeneous_gives_nonzero_weight() {
    // One close pair + one distant pair → high CV_D → cv_factor > 0
    let pops = make_pops(3);
    let obs = make_obs_with_noise(500, &[0.998, 0.997, 0.960], 0.002);
    let (pw, _) = auto_configure_pairwise_params(&obs, &pops);

    // High CV_D from mixed distances → pairwise_weight > 0
    assert!(
        pw > 0.01,
        "k=3 heterogeneous: pw={} should be > 0.01 (high CV_D)",
        pw
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// D_ref=0.014 calibration point
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn d_ref_calibration_d_min_above_014_caps_contribution() {
    // When D_min > 0.014, the (d_min/0.014).min(1.0) factor caps at 1.0
    // So increasing D_min beyond 0.014 doesn't increase pw
    let pops = make_pops(2);

    // D_min well above 0.014
    let obs_high = make_obs_with_noise(500, &[0.99, 0.93], 0.003);
    let (pw_high, _) = auto_configure_pairwise_params(&obs_high, &pops);

    // D_min even higher (less noise → higher d)
    let obs_higher = make_obs_with_noise(500, &[0.99, 0.93], 0.001);
    let (pw_higher, _) = auto_configure_pairwise_params(&obs_higher, &pops);

    // Both should be similar since (d_min/0.014) is capped at 1.0 for both
    let diff = (pw_high - pw_higher).abs();
    assert!(
        diff < 0.05,
        "D_min well above D_ref: pw should saturate. high={}, higher={}, diff={}",
        pw_high, pw_higher, diff
    );
}

#[test]
fn d_ref_calibration_d_min_below_014_scales_linearly() {
    // When D_min < 0.014, pw scales linearly with D_min via (d_min/0.014)
    let pops = make_pops(2);

    // gap=0.0002, noise=0.020 → d ≈ 0.0002/(0.020*1.414) ≈ 0.007 (< 0.014)
    let obs_med = make_obs_with_noise(500, &[0.9902, 0.990], 0.020);
    let (pw_med, _) = auto_configure_pairwise_params(&obs_med, &pops);

    // gap=0.0002, noise=0.040 → d ≈ 0.0002/(0.040*1.414) ≈ 0.0035 (< 0.014)
    let obs_noisy = make_obs_with_noise(500, &[0.9902, 0.990], 0.040);
    let (pw_noisy, _) = auto_configure_pairwise_params(&obs_noisy, &pops);

    assert!(
        pw_med > pw_noisy,
        "Lower noise → higher D_min → higher pw: med={}, noisy={}",
        pw_med, pw_noisy
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ec monotonicity with D_min
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ec_monotonically_decreases_with_increasing_d_min() {
    // As D_min increases, ec = 0.1/D_min decreases
    let pops = make_pops(2);

    // Generate observations with decreasing noise → increasing D_min
    let noise_levels = [0.050, 0.020, 0.010, 0.005, 0.002];
    let mut prev_ec = 16usize;

    for &noise in &noise_levels {
        let obs = make_obs_with_noise(500, &[0.995, 0.990], noise);
        let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);
        assert!(
            ec <= prev_ec,
            "ec should decrease as noise decreases (D_min increases): noise={}, ec={}, prev_ec={}",
            noise, ec, prev_ec
        );
        prev_ec = ec;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NaN / Inf safety
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn auto_configure_nan_in_similarities_handled() {
    // Some observations have NaN similarities — should not crash
    let pops = make_pops(2);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.93]))
        .collect();
    // Inject NaN into one observation
    obs[50]
        .similarities
        .insert("pop0#HAP1".to_string(), f64::NAN);
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Should still return valid results
    assert!(pw.is_finite(), "pw should be finite even with NaN input, got {}", pw);
    assert!(ec >= 1 && ec <= 15, "ec should be in [1,15], got {}", ec);
}

#[test]
fn auto_configure_inf_in_similarities_handled() {
    let pops = make_pops(2);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.93]))
        .collect();
    obs[50]
        .similarities
        .insert("pop0#HAP1".to_string(), f64::INFINITY);
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(pw.is_finite(), "pw should be finite even with Inf input");
    assert!(ec >= 1 && ec <= 15, "ec should be in [1,15], got {}", ec);
}

// ═══════════════════════════════════════════════════════════════════════════
// Stability under repeated calls
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn auto_configure_deterministic() {
    // Same input → same output (no internal randomness)
    let pops = make_pops(3);
    let obs = make_obs_with_noise(300, &[0.99, 0.96, 0.93], 0.005);

    let (pw1, ec1) = auto_configure_pairwise_params(&obs, &pops);
    let (pw2, ec2) = auto_configure_pairwise_params(&obs, &pops);

    assert_eq!(pw1, pw2, "auto_configure should be deterministic: pw");
    assert_eq!(ec1, ec2, "auto_configure should be deterministic: ec");
}

// ═══════════════════════════════════════════════════════════════════════════
// Missing haplotype coverage
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn auto_configure_partial_haplotype_coverage() {
    // Some observations missing haplotypes for one population → pop_sims.len() < k
    // These windows should be skipped gracefully
    let pops = make_pops(3);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.96, 0.93]))
        .collect();
    // Remove all pop2 haplotypes from half the observations
    for o in obs.iter_mut().take(50) {
        o.similarities.remove("pop2#HAP1");
        o.similarities.remove("pop2#HAP2");
    }
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // Should work with 50 complete observations (≥ 10 threshold)
    assert!(pw.is_finite(), "partial coverage: pw should be finite");
    assert!(ec >= 1 && ec <= 15, "partial coverage: ec={} out of range", ec);
}

#[test]
fn auto_configure_all_observations_incomplete() {
    // No observation has all populations → pairwise_signed_diffs all empty → defaults
    let pops = make_pops(3);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.96, 0.93]))
        .collect();
    // Remove pop2 from ALL observations
    for o in obs.iter_mut() {
        o.similarities.remove("pop2#HAP1");
        o.similarities.remove("pop2#HAP2");
    }
    let (pw, _ec) = auto_configure_pairwise_params(&obs, &pops);
    // pair(0,2) and pair(1,2) have no data. pair(0,1) should still have data.
    // Actually pop_sims only includes pops where a haplotype exists,
    // but pop_sims.len() would be 2, not 3 = k, so these windows are skipped.
    // Verify it doesn't crash.
    assert!(pw.is_finite(), "all incomplete: pw should be finite");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4-population stress test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ec_formula_4pop_hierarchical_distances() {
    // (AFR, EUR) close, (EAS, CSA) close, but the two clusters distant
    // D_min = smallest pair = within-cluster → high ec
    // CV_D = high (within-cluster d << between-cluster d)
    let pops = make_pops(4);
    let obs = make_obs_with_noise(500, &[0.990, 0.9895, 0.975, 0.9745], 0.005);
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // Within-cluster gap=0.0005, between-cluster gap≈0.015
    // D_min from within-cluster pair: d ≈ 0.0005/(0.005*sqrt(2)) ≈ 0.071
    // ec ≈ (0.1/0.071) ≈ 1.4 → round to 1
    assert!(ec >= 1, "4-pop hierarchical: ec={} should be ≥ 1", ec);
    assert!(ec <= 15, "4-pop hierarchical: ec={} out of range", ec);

    // CV_D should be high → pw should be nonzero
    assert!(pw > 0.0, "4-pop hierarchical: pw={} should be > 0 (high CV_D)", pw);
}

#[test]
fn ec_formula_4pop_uniformly_spaced() {
    // All populations uniformly spaced → all pairs have similar d → CV_D low
    let pops = make_pops(4);
    let obs = make_obs_with_noise(500, &[0.99, 0.97, 0.95, 0.93], 0.003);
    let (_pw, ec) = auto_configure_pairwise_params(&obs, &pops);

    // Smallest pair gap = 0.02 with noise=0.003 → d ≈ 0.02/(0.003*sqrt(2)) ≈ 4.7
    // ec = (0.1/4.7).round() = 0 → clamp to 1
    assert_eq!(ec, 1, "4-pop uniform: ec={} should be 1 (large D_min)", ec);
}

// ═══════════════════════════════════════════════════════════════════════════
// Boundary: exactly 10 complete observations (minimum for pair filter)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ec_formula_exactly_10_complete_windows() {
    // 10 complete + 90 incomplete → pair_diffs have exactly 10 entries (minimum)
    let pops = make_pops(2);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.93]))
        .collect();
    // Remove pop1 from all but first 10
    for o in obs.iter_mut().skip(10) {
        o.similarities.remove("pop1#HAP1");
        o.similarities.remove("pop1#HAP2");
    }
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    assert!(pw.is_finite(), "10 complete windows: pw should be finite");
    assert!(ec >= 1 && ec <= 15, "10 complete windows: ec={} out of range", ec);
}

#[test]
fn ec_formula_9_complete_windows_returns_defaults_for_pair() {
    // 9 complete observations for one pair → that pair filtered out
    // For k=2, this means no pairs → defaults
    let pops = make_pops(2);
    let mut obs: Vec<_> = (0..100)
        .map(|i| make_obs(i * 10000, &[0.99, 0.93]))
        .collect();
    // Remove pop1 from all but first 9
    for o in obs.iter_mut().skip(9) {
        o.similarities.remove("pop1#HAP1");
        o.similarities.remove("pop1#HAP2");
    }
    let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
    // With only 9 complete windows, pair_diffs has 9 entries → filtered out
    // pair_cohens_d is empty → returns defaults (0.3, 0)
    assert!(
        (pw - 0.3).abs() < 0.01,
        "9 complete windows for k=2: should return default pw=0.3, got {}",
        pw
    );
    assert_eq!(ec, 0, "9 complete windows for k=2: should return default ec=0, got {}", ec);
}
