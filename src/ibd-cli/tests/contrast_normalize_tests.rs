//! Tests for per-haplotype contrast normalization in IBD detection.

use std::collections::HashMap;

// We test the ContrastBaselines logic by importing the binary's types.
// Since ContrastBaselines is in main.rs (not lib.rs), we replicate the core
// logic here for unit testing.

/// Per-haplotype, per-window baseline statistics for contrast normalization.
#[derive(Debug, Clone)]
struct ContrastBaselines {
    hap_window: HashMap<(String, u64), (f64, usize)>,
    global_window: HashMap<u64, (f64, usize)>,
}

#[derive(Debug, Clone)]
struct WindowRecord {
    #[allow(dead_code)]
    chrom: String,
    start: u64,
    #[allow(dead_code)]
    end: u64,
    identity: f64,
}

impl ContrastBaselines {
    fn from_pair_data(pair_data: &HashMap<(String, String), Vec<WindowRecord>>) -> Self {
        let mut hap_window: HashMap<(String, u64), (f64, usize)> = HashMap::new();
        let mut global_window: HashMap<u64, (f64, usize)> = HashMap::new();

        for ((hap_a, hap_b), records) in pair_data {
            for rec in records {
                let entry_a = hap_window
                    .entry((hap_a.clone(), rec.start))
                    .or_insert((0.0, 0));
                entry_a.0 += rec.identity;
                entry_a.1 += 1;

                let entry_b = hap_window
                    .entry((hap_b.clone(), rec.start))
                    .or_insert((0.0, 0));
                entry_b.0 += rec.identity;
                entry_b.1 += 1;

                let entry_g = global_window
                    .entry(rec.start)
                    .or_insert((0.0, 0));
                entry_g.0 += rec.identity;
                entry_g.1 += 1;
            }
        }

        ContrastBaselines { hap_window, global_window }
    }

    fn hap_baseline(&self, hap: &str, window_start: u64) -> Option<f64> {
        self.hap_window
            .get(&(hap.to_string(), window_start))
            .map(|(sum, count)| sum / *count as f64)
    }

    fn global_mean(&self, window_start: u64) -> Option<f64> {
        self.global_window
            .get(&window_start)
            .map(|(sum, count)| sum / *count as f64)
    }

    fn adjust(&self, hap_a: &str, hap_b: &str, window_start: u64, identity: f64) -> f64 {
        let mu = match self.global_mean(window_start) {
            Some(m) => m,
            None => return identity,
        };
        let base_a = self.hap_baseline(hap_a, window_start).unwrap_or(mu);
        let base_b = self.hap_baseline(hap_b, window_start).unwrap_or(mu);
        let adjusted = identity - (base_a - mu) - (base_b - mu);
        adjusted.clamp(0.0, 1.0)
    }
}

fn make_record(start: u64, identity: f64) -> WindowRecord {
    WindowRecord {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        identity,
    }
}

// ============================================================================
// Core normalization tests
// ============================================================================

#[test]
fn test_baselines_from_symmetric_data() {
    // All pairs have the same identity → baselines = identity, adjustment = identity
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Global mean = 0.999
    let mu = bl.global_mean(0).unwrap();
    assert!((mu - 0.999).abs() < 1e-10);

    // Each haplotype appears in 2 pairs, both with 0.999
    assert!((bl.hap_baseline("A#1", 0).unwrap() - 0.999).abs() < 1e-10);
    assert!((bl.hap_baseline("B#1", 0).unwrap() - 0.999).abs() < 1e-10);

    // Adjustment should preserve identity when all baselines are equal
    let adj = bl.adjust("A#1", "B#1", 0, 0.999);
    assert!((adj - 0.999).abs() < 1e-10);
}

#[test]
fn test_baselines_remove_haplotype_bias() {
    // Haplotype A has systematically higher identity than B
    // A-C: 0.9995, A-D: 0.9995 → A baseline = 0.9995
    // B-C: 0.9985, B-D: 0.9985 → B baseline = 0.9985
    // A-B: 0.999 (to be adjusted)
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.9995)],
    );
    pair_data.insert(
        ("A#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.9995)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.9985)],
    );
    pair_data.insert(
        ("B#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.9985)],
    );
    pair_data.insert(
        ("C#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // A has positive deviation, B has negative deviation
    let mu = bl.global_mean(0).unwrap();
    let base_a = bl.hap_baseline("A#1", 0).unwrap();
    let base_b = bl.hap_baseline("B#1", 0).unwrap();
    assert!(base_a > mu, "A should be above global mean");
    assert!(base_b < mu, "B should be below global mean");

    // The adjusted A-B identity should be shifted toward global mean
    // removing the systematic A-high / B-low bias
    let raw = 0.999;
    let adj = bl.adjust("A#1", "B#1", 0, raw);

    // After removing A's positive excess and B's negative excess,
    // the adjustment should move the value closer to mu
    assert!(
        (adj - mu).abs() < (raw - mu).abs() + 0.001,
        "Adjustment should reduce bias: adj={}, mu={}, raw={}",
        adj, mu, raw
    );
}

#[test]
fn test_adjustment_preserves_ibd_signal() {
    // Pair A-B has IBD signal (identity = 0.9997), others are non-IBD (0.999)
    // After normalization, A-B should still show excess identity
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.9997)], // IBD signal
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("A#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("B#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("C#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    let adj_ibd = bl.adjust("A#1", "B#1", 0, 0.9997);
    let adj_non = bl.adjust("C#1", "D#1", 0, 0.999);

    // IBD pair should still have higher adjusted identity than non-IBD pair
    assert!(
        adj_ibd > adj_non,
        "IBD signal must survive normalization: adj_ibd={}, adj_non={}",
        adj_ibd, adj_non
    );

    // The IBD excess should actually be amplified relative to the non-IBD pair,
    // because A and B's baselines are inflated by the IBD signal (which makes
    // their baseline higher than it "should" be), so the contrast slightly
    // underestimates the IBD excess. But the key property is that the signal
    // is preserved.
    let raw_gap = 0.9997 - 0.999;
    let adj_gap = adj_ibd - adj_non;
    // With only 4 haplotypes, the IBD pair inflates both A and B's baselines,
    // attenuating the adjusted gap. With larger panels (n≥10), attenuation < 20%.
    // For this small panel (n=4), the gap is ~1/3 of raw — still clearly positive.
    assert!(
        adj_gap > raw_gap * 0.2,
        "Adjusted gap should retain signal: adj_gap={}, raw_gap={}",
        adj_gap, raw_gap
    );
}

#[test]
fn test_empty_pair_data() {
    let pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let bl = ContrastBaselines::from_pair_data(&pair_data);

    assert!(bl.hap_window.is_empty());
    assert!(bl.global_window.is_empty());

    // adjust should return raw identity when no baselines available
    let adj = bl.adjust("A#1", "B#1", 0, 0.999);
    assert!((adj - 0.999).abs() < 1e-10);
}

#[test]
fn test_single_pair() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // With single pair, each haplotype baseline = global mean = 0.999
    assert!((bl.global_mean(0).unwrap() - 0.999).abs() < 1e-10);
    assert!((bl.hap_baseline("A#1", 0).unwrap() - 0.999).abs() < 1e-10);

    // Adjustment should be identity (no bias to remove)
    let adj = bl.adjust("A#1", "B#1", 0, 0.999);
    assert!((adj - 0.999).abs() < 1e-10);
}

#[test]
fn test_multiple_windows() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![
            make_record(0, 0.999),
            make_record(10000, 0.998),
            make_record(20000, 0.997),
        ],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![
            make_record(0, 0.998),
            make_record(10000, 0.999),
            make_record(20000, 0.998),
        ],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Each window should have independent baselines
    assert!(bl.global_mean(0).is_some());
    assert!(bl.global_mean(10000).is_some());
    assert!(bl.global_mean(20000).is_some());

    // At window 0: A-B=0.999, A-C=0.998 → μ=0.9985
    // A baseline at 0: (0.999 + 0.998) / 2 = 0.9985
    let mu_0 = bl.global_mean(0).unwrap();
    assert!((mu_0 - 0.9985).abs() < 1e-10);
}

#[test]
fn test_adjustment_clamps_to_unit_interval() {
    // Extreme case where adjustment could push below 0 or above 1
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.001)], // Very low identity
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)], // Very high identity
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Should clamp result to [0, 1]
    let adj = bl.adjust("A#1", "B#1", 0, 0.001);
    assert!(adj >= 0.0 && adj <= 1.0, "Must be in [0,1], got {}", adj);

    let adj2 = bl.adjust("A#1", "C#1", 0, 0.999);
    assert!(adj2 >= 0.0 && adj2 <= 1.0, "Must be in [0,1], got {}", adj2);
}

#[test]
fn test_unknown_haplotype_falls_back_to_global() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Unknown haplotype Z#1 should fall back to global mean for its baseline
    let adj = bl.adjust("Z#1", "B#1", 0, 0.999);
    // Z's baseline = global mean, B's baseline = 0.999 = global mean
    // adjustment = 0.999 - (0.999 - 0.999) - (0.999 - 0.999) = 0.999
    assert!((adj - 0.999).abs() < 1e-10);
}

#[test]
fn test_unknown_window_returns_raw() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Unknown window position → no global mean → return raw identity
    let adj = bl.adjust("A#1", "B#1", 99999, 0.995);
    assert!((adj - 0.995).abs() < 1e-10);
}

#[test]
fn test_normalization_reduces_variance_across_populations() {
    // Simulate inter-population scenario:
    // AFR haplotypes have lower baseline identity (higher diversity)
    // EUR haplotypes have higher baseline identity
    // After normalization, the variance should be reduced
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();

    // EUR-EUR pairs (high baseline ~0.9992)
    pair_data.insert(
        ("EUR1#1".to_string(), "EUR2#1".to_string()),
        vec![make_record(0, 0.9992)],
    );
    // AFR-AFR pairs (lower baseline ~0.9988)
    pair_data.insert(
        ("AFR1#1".to_string(), "AFR2#1".to_string()),
        vec![make_record(0, 0.9988)],
    );
    // EUR-AFR pairs (intermediate ~0.9990)
    pair_data.insert(
        ("EUR1#1".to_string(), "AFR1#1".to_string()),
        vec![make_record(0, 0.9990)],
    );
    pair_data.insert(
        ("EUR2#1".to_string(), "AFR2#1".to_string()),
        vec![make_record(0, 0.9990)],
    );
    pair_data.insert(
        ("EUR1#1".to_string(), "AFR2#1".to_string()),
        vec![make_record(0, 0.9990)],
    );
    pair_data.insert(
        ("EUR2#1".to_string(), "AFR1#1".to_string()),
        vec![make_record(0, 0.9990)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // EUR baselines should be higher than AFR baselines
    let eur1_bl = bl.hap_baseline("EUR1#1", 0).unwrap();
    let afr1_bl = bl.hap_baseline("AFR1#1", 0).unwrap();
    assert!(eur1_bl > afr1_bl, "EUR should have higher baseline: EUR={}, AFR={}", eur1_bl, afr1_bl);

    // After adjustment, EUR-EUR and AFR-AFR should converge toward same value
    let adj_eur = bl.adjust("EUR1#1", "EUR2#1", 0, 0.9992);
    let adj_afr = bl.adjust("AFR1#1", "AFR2#1", 0, 0.9988);

    let raw_gap = (0.9992 - 0.9988_f64).abs();
    let adj_gap = (adj_eur - adj_afr).abs();
    assert!(
        adj_gap < raw_gap,
        "Normalization should reduce EUR-AFR gap: raw={}, adj={}",
        raw_gap, adj_gap
    );
}

#[test]
fn test_contrast_preserves_relative_ordering() {
    // If pair A-B has higher identity than A-C at a window,
    // this ordering should be preserved after normalization
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.9995)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.9990)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.9990)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    let adj_ab = bl.adjust("A#1", "B#1", 0, 0.9995);
    let adj_ac = bl.adjust("A#1", "C#1", 0, 0.9990);

    assert!(
        adj_ab > adj_ac,
        "Relative ordering must be preserved: adj_ab={}, adj_ac={}",
        adj_ab, adj_ac
    );
}

#[test]
fn test_baseline_counts_are_correct() {
    // With 4 haplotypes and all 6 pairs, each haplotype appears in 3 pairs
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let haps = ["A#1", "B#1", "C#1", "D#1"];
    for i in 0..4 {
        for j in (i + 1)..4 {
            pair_data.insert(
                (haps[i].to_string(), haps[j].to_string()),
                vec![make_record(0, 0.999)],
            );
        }
    }

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Each haplotype appears in 3 pairs → count = 3
    for hap in &haps {
        let (_, count) = bl.hap_window.get(&(hap.to_string(), 0)).unwrap();
        assert_eq!(*count, 3, "Hap {} should appear in 3 pairs", hap);
    }

    // Global: 6 pairs total
    let (_, global_count) = bl.global_window.get(&0).unwrap();
    assert_eq!(*global_count, 6);
}

#[test]
fn test_adjustment_formula_algebraically() {
    // Verify: adjusted = identity - (base_A - μ) - (base_B - μ)
    //                   = identity - base_A - base_B + 2μ
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.998)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.999)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.997)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let mu = bl.global_mean(0).unwrap();
    let base_a = bl.hap_baseline("A#1", 0).unwrap();
    let base_b = bl.hap_baseline("B#1", 0).unwrap();

    let adj = bl.adjust("A#1", "B#1", 0, 0.998);
    let expected = 0.998 - (base_a - mu) - (base_b - mu);

    assert!(
        (adj - expected).abs() < 1e-10,
        "Formula verification: adj={}, expected={}",
        adj, expected
    );

    // Also verify: identity - base_A - base_B + 2*mu
    let expected2 = 0.998 - base_a - base_b + 2.0 * mu;
    assert!(
        (adj - expected2).abs() < 1e-10,
        "Algebraic equivalence: adj={}, expected2={}",
        adj, expected2
    );
}
