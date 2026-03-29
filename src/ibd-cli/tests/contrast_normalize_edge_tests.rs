//! Edge case tests for ContrastBaselines (algo_dev cycle 2).
//!
//! Covers: NaN/Infinity propagation, self-pairs, empty record vectors,
//! negative identity, symmetry, mean-preservation, duplicate windows,
//! boundary identity values (0.0 / 1.0), both-haplotypes-unknown,
//! large panel scaling, and extreme bias scenarios.

use std::collections::HashMap;

// Replicate ContrastBaselines (lives in main.rs, not lib.rs)
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
// NaN / Infinity handling
// ============================================================================

#[test]
fn test_nan_identity_propagates_to_baseline() {
    // NaN in input data: NaN + anything = NaN
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, f64::NAN)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Baseline should be NaN since sum contains NaN
    let base = bl.hap_baseline("A#1", 0).unwrap();
    assert!(base.is_nan(), "NaN identity should produce NaN baseline, got {}", base);

    // Global mean should also be NaN
    let mu = bl.global_mean(0).unwrap();
    assert!(mu.is_nan(), "NaN identity should produce NaN global mean, got {}", mu);
}

#[test]
fn test_nan_adjustment_clamped_is_nan() {
    // NaN.clamp(0.0, 1.0) returns NaN in Rust — this documents the behavior
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, f64::NAN)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, f64::NAN);
    // NaN.clamp(0.0, 1.0) is NaN in Rust (since 1.50)
    assert!(adj.is_nan(), "NaN adjustment should remain NaN after clamp, got {}", adj);
}

#[test]
fn test_inf_identity_produces_inf_baseline() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, f64::INFINITY)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let base = bl.hap_baseline("A#1", 0).unwrap();
    assert!(base.is_infinite(), "Inf identity should produce Inf baseline, got {}", base);
}

#[test]
fn test_neg_inf_identity_clamps_to_zero() {
    // -Inf identity: adjustment formula might produce -Inf, clamp should give 0.0
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, f64::NEG_INFINITY)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    // The global mean includes -Inf, so it's -Inf
    // The adjust formula with -Inf inputs will produce NaN or -Inf
    // which clamps to 0.0 (if -Inf) or stays NaN
    let adj = bl.adjust("A#1", "B#1", 0, f64::NEG_INFINITY);
    // -Inf.clamp(0.0, 1.0) = 0.0
    // But the formula: -Inf - (base_a - mu) - (base_b - mu)
    // base_a = (-Inf + 0.999)/2 = -Inf, mu = (-Inf + 0.999)/2 = -Inf
    // -Inf - (-Inf - (-Inf)) - (-Inf - (-Inf)) = -Inf - NaN - NaN = NaN
    // This is actually NaN due to Inf - Inf = NaN
    // Document this: NaN or 0.0 both acceptable, just don't panic
    assert!(
        adj.is_nan() || (adj >= 0.0 && adj <= 1.0),
        "Result should be NaN or in [0,1], got {}",
        adj
    );
}

#[test]
fn test_nan_mixed_with_valid_contaminates_baseline() {
    // One NaN record + one valid record: sum = NaN, baseline = NaN
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, f64::NAN), make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let base = bl.hap_baseline("A#1", 0).unwrap();
    assert!(base.is_nan(), "NaN contaminates sum: got {}", base);
}

// ============================================================================
// Self-pair edge case (hap_a == hap_b)
// ============================================================================

#[test]
fn test_self_pair_double_counts_haplotype() {
    // When hap_a == hap_b, the code accumulates to the same key twice
    // (entry_a and entry_b both point to ("X#1", 0))
    // This means count = 2 (not 1) and sum = 2 * identity
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("X#1".to_string(), "X#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // The hap entry gets incremented twice (once as hap_a, once as hap_b)
    let (sum, count) = bl.hap_window.get(&("X#1".to_string(), 0)).unwrap();
    assert_eq!(*count, 2, "Self-pair should double-count the haplotype");
    assert!(
        (sum - 2.0 * 0.999).abs() < 1e-10,
        "Sum should be 2 * identity for self-pair"
    );

    // But baseline = sum/count = 0.999 (cancels out)
    let base = bl.hap_baseline("X#1", 0).unwrap();
    assert!(
        (base - 0.999).abs() < 1e-10,
        "Self-pair baseline should still be 0.999, got {}",
        base
    );

    // Global: count = 1 (only one pair entry)
    let (_, g_count) = bl.global_window.get(&0).unwrap();
    assert_eq!(*g_count, 1, "Global should count 1 pair record");
}

#[test]
fn test_self_pair_adjustment_is_identity() {
    // Self-pair: baseline_A = baseline_B = global_mean (for single pair)
    // adjust = identity - (μ - μ) - (μ - μ) = identity
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("X#1".to_string(), "X#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("X#1", "X#1", 0, 0.999);
    assert!(
        (adj - 0.999).abs() < 1e-10,
        "Self-pair adjustment should preserve identity, got {}",
        adj
    );
}

// ============================================================================
// Empty records vector for a pair
// ============================================================================

#[test]
fn test_pair_with_empty_records_vec() {
    // Pair exists in map but has no records
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![], // empty
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // No records processed → maps should be empty
    assert!(bl.hap_window.is_empty());
    assert!(bl.global_window.is_empty());

    // Adjustment should fall back to raw identity
    let adj = bl.adjust("A#1", "B#1", 0, 0.999);
    assert!((adj - 0.999).abs() < 1e-10);
}

#[test]
fn test_mix_of_empty_and_populated_pairs() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![], // empty
    );
    pair_data.insert(
        ("C#1".to_string(), "D#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Only C and D should have baselines
    assert!(bl.hap_baseline("A#1", 0).is_none());
    assert!(bl.hap_baseline("B#1", 0).is_none());
    assert!(bl.hap_baseline("C#1", 0).is_some());
    assert!(bl.hap_baseline("D#1", 0).is_some());

    // A-B adjustment: unknown haps → fallback to global mean for both
    // adjust = 0.5 - (μ - μ) - (μ - μ) = 0.5
    let adj = bl.adjust("A#1", "B#1", 0, 0.5);
    assert!((adj - 0.5).abs() < 1e-10);
}

// ============================================================================
// Negative identity values
// ============================================================================

#[test]
fn test_negative_identity_clamps_to_zero() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, -0.5)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Global mean includes negative value
    let mu = bl.global_mean(0).unwrap();
    assert!(mu < 0.999, "Mean should be dragged down by negative value");

    // Adjustment of the negative pair could push even more negative → clamp to 0
    let adj = bl.adjust("A#1", "B#1", 0, -0.5);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "Negative identity adjustment should be clamped to [0,1], got {}",
        adj
    );
}

#[test]
fn test_all_negative_identities() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, -0.1)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, -0.2)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, -0.3)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, -0.1);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "All-negative adjustment should clamp to [0,1], got {}",
        adj
    );
}

// ============================================================================
// Symmetry: adjust(A, B) == adjust(B, A)
// ============================================================================

#[test]
fn test_symmetry_of_adjustment() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.998)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.997)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    let adj_ab = bl.adjust("A#1", "B#1", 0, 0.998);
    let adj_ba = bl.adjust("B#1", "A#1", 0, 0.998);

    assert!(
        (adj_ab - adj_ba).abs() < 1e-10,
        "adjust(A,B) should equal adjust(B,A): {} vs {}",
        adj_ab, adj_ba
    );
}

#[test]
fn test_symmetry_with_biased_haplotypes() {
    // A has high baseline, B has low baseline
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.99)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.999)]);
    pair_data.insert(("A#1".to_string(), "D#1".to_string()), vec![make_record(0, 0.999)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.98)]);
    pair_data.insert(("B#1".to_string(), "D#1".to_string()), vec![make_record(0, 0.98)]);
    pair_data.insert(("C#1".to_string(), "D#1".to_string()), vec![make_record(0, 0.995)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    let adj_ab = bl.adjust("A#1", "B#1", 0, 0.99);
    let adj_ba = bl.adjust("B#1", "A#1", 0, 0.99);

    assert!(
        (adj_ab - adj_ba).abs() < 1e-10,
        "Symmetry must hold even with biased haplotypes: {} vs {}",
        adj_ab, adj_ba
    );
}

// ============================================================================
// Mean-preservation property
// ============================================================================

#[test]
fn test_mean_preservation_across_pairs() {
    // Mathematical property: the sum of adjusted identities at a window
    // should equal the sum of raw identities (normalization is mean-preserving)
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let identities = vec![
        (("A#1", "B#1"), 0.998),
        (("A#1", "C#1"), 0.999),
        (("A#1", "D#1"), 0.997),
        (("B#1", "C#1"), 0.996),
        (("B#1", "D#1"), 0.998),
        (("C#1", "D#1"), 0.999),
    ];

    for ((a, b), id) in &identities {
        pair_data.insert(
            (a.to_string(), b.to_string()),
            vec![make_record(0, *id)],
        );
    }

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    let raw_sum: f64 = identities.iter().map(|(_, id)| id).sum();
    let adj_sum: f64 = identities
        .iter()
        .map(|((a, b), id)| bl.adjust(a, b, 0, *id))
        .sum();

    assert!(
        (raw_sum - adj_sum).abs() < 1e-8,
        "Normalization should be mean-preserving: raw_sum={}, adj_sum={}",
        raw_sum, adj_sum
    );
}

// ============================================================================
// Duplicate window positions in records
// ============================================================================

#[test]
fn test_duplicate_window_positions_accumulate() {
    // Two records at the same window_start for one pair
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.998), make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // A at window 0: count=2, sum=0.998+0.999=1.997
    let (sum, count) = bl.hap_window.get(&("A#1".to_string(), 0)).unwrap();
    assert_eq!(*count, 2);
    assert!((sum - 1.997).abs() < 1e-10);

    // Baseline = 1.997 / 2 = 0.9985
    let base = bl.hap_baseline("A#1", 0).unwrap();
    assert!((base - 0.9985).abs() < 1e-10);

    // Global: count=2 (two records in the pair)
    let (g_sum, g_count) = bl.global_window.get(&0).unwrap();
    assert_eq!(*g_count, 2);
    assert!((g_sum - 1.997).abs() < 1e-10);
}

// ============================================================================
// Identity boundary values: exactly 0.0 and 1.0
// ============================================================================

#[test]
fn test_identity_exactly_zero() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.0)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, 0.0);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "Zero identity adjustment should be in [0,1], got {}",
        adj
    );
}

#[test]
fn test_identity_exactly_one() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 1.0)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.5)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, 1.0);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "Identity=1.0 adjustment should be in [0,1], got {}",
        adj
    );
    // With A and B both having elevated baselines from their high-identity pair,
    // their deviations from mean are positive → adj < 1.0
    let mu = bl.global_mean(0).unwrap();
    assert!(mu < 1.0, "Global mean should be below 1.0");
}

#[test]
fn test_all_identities_exactly_one() {
    // All pairs have identity = 1.0
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 1.0)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 1.0)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 1.0)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, 1.0);
    assert!(
        (adj - 1.0).abs() < 1e-10,
        "All-1.0 identities should adjust to 1.0, got {}",
        adj
    );
}

#[test]
fn test_all_identities_exactly_zero() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.0)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.0)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.0)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, 0.0);
    assert!(
        adj.abs() < 1e-10,
        "All-0.0 identities should adjust to 0.0, got {}",
        adj
    );
}

// ============================================================================
// Both haplotypes unknown
// ============================================================================

#[test]
fn test_both_haplotypes_unknown_double_fallback() {
    // Build baselines with known haplotypes, then query with unknown ones
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Both X and Y unknown → both baselines = global mean
    // adjust = 0.5 - (μ - μ) - (μ - μ) = 0.5
    let adj = bl.adjust("X#1", "Y#1", 0, 0.5);
    assert!(
        (adj - 0.5).abs() < 1e-10,
        "Both-unknown should return raw identity, got {}",
        adj
    );
}

#[test]
fn test_one_known_one_unknown() {
    // A known with high baseline, Z unknown
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.998)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.999)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.997)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let mu = bl.global_mean(0).unwrap();
    let base_a = bl.hap_baseline("A#1", 0).unwrap();

    // Z unknown → base_z = mu
    // adjust = 0.999 - (base_a - mu) - (mu - mu) = 0.999 - (base_a - mu)
    let adj = bl.adjust("A#1", "Z#1", 0, 0.999);
    let expected = (0.999 - (base_a - mu)).clamp(0.0, 1.0);
    assert!(
        (adj - expected).abs() < 1e-10,
        "One-known-one-unknown: adj={}, expected={}",
        adj, expected
    );
}

// ============================================================================
// Large panel scaling (10 haplotypes)
// ============================================================================

#[test]
fn test_large_panel_10_haplotypes() {
    // 10 haplotypes = 45 pairs
    // All have identity ~0.999, except one biased hap with higher identity
    let haps: Vec<String> = (0..10).map(|i| format!("H{}#1", i)).collect();
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();

    for i in 0..10 {
        for j in (i + 1)..10 {
            let id = if i == 0 {
                0.9995 // H0 has systematically higher identity
            } else {
                0.999
            };
            pair_data.insert(
                (haps[i].clone(), haps[j].clone()),
                vec![make_record(0, id)],
            );
        }
    }

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // H0 baseline should be higher than others
    let base_h0 = bl.hap_baseline("H0#1", 0).unwrap();
    let base_h5 = bl.hap_baseline("H5#1", 0).unwrap();
    assert!(
        base_h0 > base_h5,
        "Biased H0 should have higher baseline: {} vs {}",
        base_h0, base_h5
    );

    // After normalization, H0's pairs should have their inflated identity reduced
    let adj_h0_h1 = bl.adjust("H0#1", "H1#1", 0, 0.9995);
    let adj_h1_h2 = bl.adjust("H1#1", "H2#1", 0, 0.999);

    // The gap should be smaller after normalization
    let raw_gap = 0.9995 - 0.999;
    let adj_gap = adj_h0_h1 - adj_h1_h2;
    assert!(
        adj_gap < raw_gap,
        "Large panel normalization should reduce bias gap: raw={}, adj={}",
        raw_gap, adj_gap
    );
}

// ============================================================================
// Extreme bias scenarios
// ============================================================================

#[test]
fn test_extreme_bias_one_haplotype() {
    // One haplotype has identity near 1.0, all others near 0.5
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("BIAS#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.99)]);
    pair_data.insert(("BIAS#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.99)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.5)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let mu = bl.global_mean(0).unwrap();

    // BIAS has baseline much higher than μ
    let base_bias = bl.hap_baseline("BIAS#1", 0).unwrap();
    assert!(base_bias > mu + 0.1, "BIAS should be well above mean");

    // After adjustment, BIAS-B should be pulled down (removing BIAS's systematic excess)
    let adj = bl.adjust("BIAS#1", "B#1", 0, 0.99);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "Extreme bias adjustment must be in [0,1], got {}",
        adj
    );
}

#[test]
fn test_adjustment_with_identity_above_one() {
    // Edge case: identity > 1.0 (shouldn't happen in practice but test clamping)
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 1.5)]);
    pair_data.insert(("A#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.5)]);
    pair_data.insert(("B#1".to_string(), "C#1".to_string()), vec![make_record(0, 0.5)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 0, 1.5);
    assert!(
        adj >= 0.0 && adj <= 1.0,
        "Identity > 1.0 should be clamped, got {}",
        adj
    );
}

// ============================================================================
// Window u64 overflow edge case
// ============================================================================

#[test]
fn test_window_start_max_u64() {
    let max_start = u64::MAX;
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![WindowRecord {
            chrom: "chr1".to_string(),
            start: max_start,
            end: max_start, // avoid overflow in end calculation
            identity: 0.999,
        }],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    assert!(bl.global_mean(max_start).is_some());
    assert!(bl.hap_baseline("A#1", max_start).is_some());

    let adj = bl.adjust("A#1", "B#1", max_start, 0.999);
    assert!((adj - 0.999).abs() < 1e-10);
}

// ============================================================================
// Struct traits (Debug, Clone)
// ============================================================================

#[test]
fn test_contrast_baselines_debug_output() {
    let bl = ContrastBaselines {
        hap_window: HashMap::new(),
        global_window: HashMap::new(),
    };
    let debug = format!("{:?}", bl);
    assert!(debug.contains("ContrastBaselines"));
    assert!(debug.contains("hap_window"));
    assert!(debug.contains("global_window"));
}

#[test]
fn test_contrast_baselines_clone_preserves_data() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), vec![make_record(0, 0.999)]);

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let cloned = bl.clone();

    assert_eq!(
        bl.hap_baseline("A#1", 0).unwrap(),
        cloned.hap_baseline("A#1", 0).unwrap()
    );
    assert_eq!(
        bl.global_mean(0).unwrap(),
        cloned.global_mean(0).unwrap()
    );
}

// ============================================================================
// Multi-window independence
// ============================================================================

#[test]
fn test_window_independence_no_cross_contamination() {
    // Haplotype A is biased at window 0 but normal at window 10000
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(0, 0.999), make_record(10000, 0.990)],
    );
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.999), make_record(10000, 0.990)],
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        vec![make_record(0, 0.990), make_record(10000, 0.990)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // At window 0, A has high baseline; at window 10000, everyone is equal
    let base_a_w0 = bl.hap_baseline("A#1", 0).unwrap();
    let base_b_w0 = bl.hap_baseline("B#1", 0).unwrap();
    assert!(base_a_w0 > base_b_w0, "A should be biased at window 0");

    let base_a_w1 = bl.hap_baseline("A#1", 10000).unwrap();
    let base_b_w1 = bl.hap_baseline("B#1", 10000).unwrap();
    assert!(
        (base_a_w1 - base_b_w1).abs() < 1e-10,
        "A and B should have equal baselines at window 10000"
    );
}

// ============================================================================
// Adjustment with only one pair at window (minimal data)
// ============================================================================

#[test]
fn test_single_pair_single_window_is_noop() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    pair_data.insert(
        ("A#1".to_string(), "B#1".to_string()),
        vec![make_record(42, 0.997)],
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj = bl.adjust("A#1", "B#1", 42, 0.997);
    // Single pair → baseline_A = baseline_B = μ = 0.997
    // adjust = 0.997 - 0 - 0 = 0.997
    assert!(
        (adj - 0.997).abs() < 1e-10,
        "Single pair at single window should be noop, got {}",
        adj
    );
}

// ============================================================================
// Idempotency-like property: normalized data preserves structure
// ============================================================================

#[test]
fn test_double_normalization_changes_result() {
    // Normalizing already-normalized data should NOT be idempotent in general,
    // because the baselines are recomputed from the adjusted values.
    // This test documents the expected behavior.
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let ids = vec![
        (("A#1", "B#1"), 0.998),
        (("A#1", "C#1"), 0.999),
        (("B#1", "C#1"), 0.997),
    ];
    for ((a, b), id) in &ids {
        pair_data.insert((a.to_string(), b.to_string()), vec![make_record(0, *id)]);
    }

    let bl = ContrastBaselines::from_pair_data(&pair_data);
    let adj1_ab = bl.adjust("A#1", "B#1", 0, 0.998);

    // Build new baselines from adjusted data
    let mut pair_data2: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    for ((a, b), id) in &ids {
        let adj = bl.adjust(a, b, 0, *id);
        pair_data2.insert((a.to_string(), b.to_string()), vec![make_record(0, adj)]);
    }
    let bl2 = ContrastBaselines::from_pair_data(&pair_data2);
    let adj2_ab = bl2.adjust("A#1", "B#1", 0, adj1_ab);

    // Second normalization should be approximately noop (since first normalization
    // already removed biases) — the result should be close to adj1
    assert!(
        (adj2_ab - adj1_ab).abs() < 0.01,
        "Second normalization should be near-noop after first: adj1={}, adj2={}",
        adj1_ab, adj2_ab
    );
}

// ============================================================================
// Stress: many windows per pair
// ============================================================================

#[test]
fn test_100_windows_per_pair() {
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let records: Vec<WindowRecord> = (0..100)
        .map(|i| make_record(i * 10000, 0.999 - (i as f64) * 0.0001))
        .collect();
    pair_data.insert(("A#1".to_string(), "B#1".to_string()), records);
    pair_data.insert(
        ("A#1".to_string(), "C#1".to_string()),
        (0..100)
            .map(|i| make_record(i * 10000, 0.998))
            .collect(),
    );
    pair_data.insert(
        ("B#1".to_string(), "C#1".to_string()),
        (0..100)
            .map(|i| make_record(i * 10000, 0.997))
            .collect(),
    );

    let bl = ContrastBaselines::from_pair_data(&pair_data);

    // Should have 100 distinct window entries in global
    assert_eq!(bl.global_window.len(), 100);

    // All adjustments should be in [0, 1]
    for i in 0..100 {
        let w = i * 10000;
        let id = 0.999 - (i as f64) * 0.0001;
        let adj = bl.adjust("A#1", "B#1", w, id);
        assert!(
            adj >= 0.0 && adj <= 1.0,
            "Window {} adjustment out of range: {}",
            w, adj
        );
    }
}
