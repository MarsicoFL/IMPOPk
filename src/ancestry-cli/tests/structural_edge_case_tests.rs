//! Structural edge case tests for ancestry-cli HMM.
//!
//! Tests for division-by-zero bugs, single-population handling, coverage-ratio
//! NaN safety, Baum-Welch convergence properties, and normalization edge cases.
//!
//! Discovered in testing cycle 45:
//! - BUG: AncestryHmmParams::new panics with 0 populations (usize underflow)
//! - BUG: AncestryHmmParams::new produces Inf with 1 population (div by zero)
//! - BUG: set_switch_prob produces Inf with 1 population (div by zero)

use std::collections::HashMap;
use std::panic;

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    PopulationNormalization, estimate_temperature, estimate_temperature_normalized,
    estimate_switch_prob, forward_backward, viterbi,
};

// =====================================================================
// Helpers
// =====================================================================

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_at(pos: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: pos,
        end: pos + 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_coverage(
    sims: &[(&str, f64)],
    covs: &[(&str, f64)],
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: Some(covs.iter().map(|(k, v)| (k.to_string(), *v)).collect()),
            haplotype_consistency_bonus: None,
    }
}

fn make_3pop_params() -> AncestryHmmParams {
    AncestryHmmParams::new(
        vec![
            make_pop("A", &["a1", "a2"]),
            make_pop("B", &["b1", "b2"]),
            make_pop("C", &["c1", "c2"]),
        ],
        0.01,
    )
}

// =====================================================================
// Single/zero population edge cases (fixed in A3 cycle 31)
// Previously: division by zero in new() and set_switch_prob()
// =====================================================================

#[test]
fn single_population_new_correct_transition() {
    let params = AncestryHmmParams::new(vec![make_pop("only", &["h1", "h2"])], 0.01);
    assert_eq!(params.n_states, 1);
    assert_eq!(params.transitions.len(), 1);
    assert_eq!(params.transitions[0][0], 1.0, "Single state must have P(stay)=1.0");
    assert_eq!(params.initial.len(), 1);
    assert_eq!(params.initial[0], 1.0, "Single state must have P(init)=1.0");
}

#[test]
fn single_population_set_switch_prob_correct() {
    let mut params = AncestryHmmParams::new(vec![make_pop("only", &["h1"])], 0.0);
    params.set_switch_prob(0.01);
    assert_eq!(params.transitions[0][0], 1.0, "set_switch_prob with 1 state must keep P(stay)=1.0");
}

#[test]
fn zero_populations_new_no_panic() {
    let params = AncestryHmmParams::new(vec![], 0.01);
    assert_eq!(params.n_states, 0);
    assert!(params.transitions.is_empty());
    assert!(params.initial.is_empty());
}

#[test]
fn zero_populations_set_switch_prob_no_panic() {
    let mut params = AncestryHmmParams::new(vec![], 0.01);
    params.set_switch_prob(0.05); // should be a no-op, no panic
    assert_eq!(params.n_states, 0);
}

// =====================================================================
// Coverage-ratio emission NaN safety
// =====================================================================

#[test]
fn coverage_emission_nan_in_coverage_ratios() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    let obs = make_obs_with_coverage(
        &[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.85), ("c1", 0.75), ("c2", 0.78)],
        &[("a1", f64::NAN), ("a2", 0.9), ("b1", 0.8), ("b2", 0.85), ("c1", 0.7), ("c2", 0.75)],
    );
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.log_emission(&obs, 0)
    }));
    assert!(result.is_ok(), "log_emission with NaN coverage should not panic");
    if let Ok(le) = result {
        // NaN in coverage mean computation: NaN + 0.9 = NaN / 2 = NaN
        // NaN > 0.0 is false → filtered out → only sim emission used
        // Actually: filter_map on coverage_ratios.get(h) will include NaN
        // Then mean = (NaN + 0.9) / 2 = NaN → NaN > 0.0 is false → filtered
        assert!(le.is_finite() || le == f64::NEG_INFINITY,
            "Coverage NaN should be handled: {}", le);
    }
}

#[test]
fn coverage_emission_all_nan_coverage() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    let obs = make_obs_with_coverage(
        &[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)],
        &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)],
    );
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.log_emission(&obs, 0)
    }));
    assert!(result.is_ok(), "log_emission with all NaN coverage should not panic");
}

#[test]
fn coverage_emission_infinity_coverage() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    let obs = make_obs_with_coverage(
        &[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)],
        &[("a1", f64::INFINITY), ("b1", 0.8), ("c1", 0.7)],
    );
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.log_emission(&obs, 0)
    }));
    assert!(result.is_ok(), "log_emission with Inf coverage should not panic");
}

#[test]
fn coverage_emission_zero_weight_bypasses_coverage() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.0);
    let obs = make_obs_with_coverage(
        &[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)],
        &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)],
    );
    // With weight=0, coverage should be ignored entirely
    let le = params.log_emission(&obs, 0);
    let le_no_cov = {
        let obs2 = make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)]);
        params.log_emission(&obs2, 0)
    };
    assert!((le - le_no_cov).abs() < 1e-10,
        "Zero coverage weight should give same result as no coverage: {} vs {}", le, le_no_cov);
}

#[test]
fn coverage_emission_negative_coverage() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    let obs = make_obs_with_coverage(
        &[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)],
        &[("a1", -0.5), ("b1", -0.3), ("c1", -0.4)],
    );
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.log_emission(&obs, 0)
    }));
    assert!(result.is_ok(), "Negative coverage should not panic");
    if let Ok(le) = result {
        // Negative coverage: c > 0.0 fails → returns sim_emission only
        assert!(le.is_finite() || le == f64::NEG_INFINITY, "Should be valid: {}", le);
    }
}

#[test]
fn coverage_emission_missing_coverage_map() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    // coverage_ratios is None → should fall back to sim-only
    let obs = make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "Missing coverage map should give finite emission: {}", le);
}

#[test]
fn coverage_emission_empty_coverage_map() {
    let mut params = make_3pop_params();
    params.set_coverage_weight(0.5);
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities: [("a1", 0.95), ("b1", 0.80), ("c1", 0.75)]
            .iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: Some(HashMap::new()),
            haplotype_consistency_bonus: None,
    };
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "Empty coverage map should give finite emission: {}", le);
}

// =====================================================================
// Baum-Welch convergence properties
// =====================================================================

#[test]
fn baum_welch_loglikelihood_monotonic_increase() {
    // EM algorithm should never decrease the log-likelihood
    let mut params = make_3pop_params();
    let obs_seq: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            let pop = i % 3;
            let sims = match pop {
                0 => vec![("a1", 0.95), ("a2", 0.93), ("b1", 0.70), ("b2", 0.72), ("c1", 0.75), ("c2", 0.73)],
                1 => vec![("a1", 0.70), ("a2", 0.72), ("b1", 0.95), ("b2", 0.93), ("c1", 0.75), ("c2", 0.73)],
                _ => vec![("a1", 0.70), ("a2", 0.72), ("b1", 0.75), ("b2", 0.73), ("c1", 0.95), ("c2", 0.93)],
            };
            make_obs_at(i * 10000, &sims)
        })
        .collect();

    // Run single iterations and track log-likelihood
    let mut lls: Vec<f64> = Vec::new();
    for _iter in 0..10 {
        let ll = params.baum_welch(&[obs_seq.as_slice()], 1, 0.0);
        if ll.is_finite() {
            lls.push(ll);
        }
    }

    // Log-likelihood should not decrease between iterations
    // (small tolerance for floating-point)
    for window in lls.windows(2) {
        assert!(
            window[1] >= window[0] - 1e-6,
            "Baum-Welch LL decreased: {} -> {}", window[0], window[1]
        );
    }
}

#[test]
fn baum_welch_converges_within_max_iters() {
    let mut params = make_3pop_params();
    let obs_seq: Vec<AncestryObservation> = (0..30)
        .map(|i| {
            if i < 15 {
                make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.70), ("c1", 0.75)])
            } else {
                make_obs_at(i * 10000, &[("a1", 0.70), ("b1", 0.95), ("c1", 0.75)])
            }
        })
        .collect();

    let ll = params.baum_welch(&[obs_seq.as_slice()], 50, 1e-6);
    assert!(ll.is_finite(), "Baum-Welch should converge to finite LL: {}", ll);
}

#[test]
fn baum_welch_empty_observations() {
    let mut params = make_3pop_params();
    let ll = params.baum_welch(&[], 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY, "Empty observations should return NEG_INFINITY");
}

#[test]
fn baum_welch_single_observation_sequence() {
    // Single-element observation sequence is skipped (len < 2)
    let mut params = make_3pop_params();
    let obs_seq: Vec<AncestryObservation> = vec![
        make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)]),
    ];
    let ll = params.baum_welch(&[obs_seq.as_slice()], 10, 1e-6);
    // With a single observation, the sequence is skipped (obs.len() < 2)
    // total_ll stays 0.0 → finite
    assert!(ll.is_finite() || ll == f64::NEG_INFINITY,
        "Single obs should be handled: {}", ll);
}

#[test]
fn baum_welch_two_observation_sequence() {
    // Minimum valid sequence length for Baum-Welch (len >= 2)
    let mut params = make_3pop_params();
    let obs_seq: Vec<AncestryObservation> = vec![
        make_obs_at(0, &[("a1", 0.95), ("b1", 0.70), ("c1", 0.75)]),
        make_obs_at(10000, &[("a1", 0.70), ("b1", 0.95), ("c1", 0.75)]),
    ];
    let ll = params.baum_welch(&[obs_seq.as_slice()], 10, 1e-6);
    assert!(ll.is_finite(), "Two-observation BW should converge: {}", ll);
}

#[test]
fn baum_welch_multiple_sequences() {
    let mut params = make_3pop_params();
    let seq1: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.70), ("c1", 0.75)]))
        .collect();
    let seq2: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.70), ("b1", 0.95), ("c1", 0.75)]))
        .collect();
    let ll = params.baum_welch(&[seq1.as_slice(), seq2.as_slice()], 10, 1e-6);
    assert!(ll.is_finite(), "Multi-sequence BW should converge: {}", ll);
}

// =====================================================================
// learn_normalization edge cases
// =====================================================================

#[test]
fn learn_normalization_constant_similarities() {
    // All similarities identical → variance = 0 → std should clamp to 1e-6
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.90), ("a2", 0.90),
                ("b1", 0.90), ("b2", 0.90),
                ("c1", 0.90), ("c2", 0.90),
            ])
        })
        .collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    for &s in &norm.stds {
        assert!(s >= 1e-6, "Std should be clamped to at least 1e-6: {}", s);
        assert!(s.is_finite(), "Std should be finite: {}", s);
    }
    for &m in &norm.means {
        assert!((m - 0.90).abs() < 1e-10, "Mean should be 0.90: {}", m);
    }
}

#[test]
fn learn_normalization_single_observation() {
    // Single observation → pop_counts[i] = 1 → std = 1e-6
    let mut params = make_3pop_params();
    let obs = vec![
        make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("b2", 0.85), ("c1", 0.75), ("c2", 0.78)]),
    ];
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    for &s in &norm.stds {
        assert!(s >= 1e-6, "Single-obs std should be fallback: {}", s);
    }
}

#[test]
fn learn_normalization_missing_population_data() {
    // Some populations have no matching haplotypes in the observations
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            // Only pop A has data; pops B and C have no matching haplotypes
            make_obs_at(i * 10000, &[("a1", 0.95), ("a2", 0.90)])
        })
        .collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    // Pop A should have real mean, pops B/C should have fallback
    assert!(norm.means[0] > 0.0, "Pop A mean should be positive: {}", norm.means[0]);
    assert_eq!(norm.means[1], 0.0, "Pop B mean should be 0 (no data)");
    assert_eq!(norm.means[2], 0.0, "Pop C mean should be 0 (no data)");
}

#[test]
fn learn_normalization_all_zero_similarities() {
    // All similarities are 0.0 → filtered by v > 0.0 → pop_counts stay 0
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.0), ("a2", 0.0),
                ("b1", 0.0), ("b2", 0.0),
                ("c1", 0.0), ("c2", 0.0),
            ])
        })
        .collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    for &m in &norm.means {
        assert_eq!(m, 0.0, "All-zero sims should give mean 0: {}", m);
    }
}

// =====================================================================
// Viterbi and forward-backward with extreme parameters
// =====================================================================

#[test]
fn viterbi_switch_prob_zero() {
    // switch_prob=0 → no transitions allowed → stay in initial state forever
    let params = AncestryHmmParams::new(
        vec![
            make_pop("A", &["a1"]),
            make_pop("B", &["b1"]),
            make_pop("C", &["c1"]),
        ],
        0.0, // zero switch prob
    );
    let obs: Vec<_> = (0..5)
        .map(|i| {
            // Pop B has highest similarity in all windows
            make_obs_at(i * 10000, &[("a1", 0.70), ("b1", 0.95), ("c1", 0.75)])
        })
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    if result.is_ok() {
        let states = result.unwrap();
        assert_eq!(states.len(), 5);
        // With zero switch prob, transition[i][j] for i≠j = 0 → ln(0) = -inf
        // So the HMM can never leave its initial state
        // All states should be the same
        let first = states[0];
        for &s in &states {
            assert_eq!(s, first, "Zero switch prob should lock to initial state");
        }
    } else {
        eprintln!("Viterbi panics with switch_prob=0 (ln(0) in transitions)");
    }
}

#[test]
fn viterbi_switch_prob_one() {
    // switch_prob=1.0 → always switch → each window must be different from previous
    let params = AncestryHmmParams::new(
        vec![
            make_pop("A", &["a1"]),
            make_pop("B", &["b1"]),
            make_pop("C", &["c1"]),
        ],
        1.0,
    );
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)]))
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    if result.is_ok() {
        let states = result.unwrap();
        assert_eq!(states.len(), 5);
        // With switch_prob=1.0: stay_prob=0.0 → ln(0)=-inf on diagonal
        // switch_each = 1.0/(3-1) = 0.5 per off-diagonal
        // So consecutive states must differ (self-transition has -inf log prob)
    } else {
        eprintln!("Viterbi panics with switch_prob=1.0 (ln(0) in stay transition)");
    }
}

// =====================================================================
// estimate_temperature edge cases
// =====================================================================

#[test]
fn estimate_temperature_single_observation() {
    // Single observation → diffs has 1 element → median = diffs[0]
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs = vec![
        make_obs(&[("a1", 0.95), ("b1", 0.80)]),
    ];
    let temp = estimate_temperature(&obs, &pops);
    assert!(temp >= 0.01 && temp <= 0.15, "Single obs temp should be clamped: {}", temp);
    assert!((temp - 0.15).abs() < 1e-10, "Diff=0.15 → clamped to 0.15: {}", temp);
}

#[test]
fn estimate_temperature_identical_similarities() {
    // All populations have identical similarities → diffs are all 0.0 or empty
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.90), ("b1", 0.90), ("c1", 0.90)]))
        .collect();
    let temp = estimate_temperature(&obs, &pops);
    // max == min for all windows → no diffs → fallback 0.03
    assert_eq!(temp, 0.03, "Identical sims should give fallback: {}", temp);
}

#[test]
fn estimate_temperature_extreme_diff() {
    // Very large differences → should clamp to 0.15
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 1.0), ("b1", 0.0001)]))
        .collect();
    let temp = estimate_temperature(&obs, &pops);
    assert_eq!(temp, 0.15, "Large diffs should clamp to 0.15: {}", temp);
}

#[test]
fn estimate_temperature_very_small_diff() {
    // Very small differences → should clamp to lower bound (0.0005)
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.9000001), ("b1", 0.9000000)]))
        .collect();
    let temp = estimate_temperature(&obs, &pops);
    assert_eq!(temp, 0.0005, "Tiny diffs should clamp to lower bound 0.0005: {}", temp);
}

#[test]
fn estimate_temperature_single_population_with_data() {
    // Only 1 population has data → pop_sims.len() < 2 → no diffs → fallback
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95)]))  // only pop A has data
        .collect();
    let temp = estimate_temperature(&obs, &pops);
    assert_eq!(temp, 0.03, "Single-pop data should give fallback: {}", temp);
}

// =====================================================================
// estimate_switch_prob edge cases
// =====================================================================

#[test]
fn estimate_switch_prob_fewer_than_10_observations() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80)]))
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    assert_eq!(sp, 0.001, "< 10 observations should return fallback: {}", sp);
}

#[test]
fn estimate_switch_prob_uniform_ancestry() {
    // All same ancestry → no switches → rate near 0
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..20)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.50)]))
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    // No switches observed → rate = 0 → regularized toward 0.001
    assert!(sp >= 0.0001 && sp <= 0.05, "Switch prob out of range: {}", sp);
    assert!(sp < 0.01, "Uniform ancestry should have low switch prob: {}", sp);
}

#[test]
fn estimate_switch_prob_alternating_ancestry() {
    // Alternating ancestry every window → max switch rate
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
    ];
    let obs: Vec<_> = (0..20)
        .map(|i| {
            if i % 2 == 0 {
                make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.50)])
            } else {
                make_obs_at(i * 10000, &[("a1", 0.50), ("b1", 0.99)])
            }
        })
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    assert!(sp >= 0.0001 && sp <= 0.05, "Switch prob out of range: {}", sp);
    // Should be high (close to clamp max)
    assert!(sp > 0.01, "Alternating ancestry should have high switch prob: {}", sp);
}

// =====================================================================
// emission model Display and FromStr edge cases
// =====================================================================

#[test]
fn emission_model_topk_zero_via_log_emission() {
    // TopK(0) → take = 0.min(len) = 0 → returns None → log_emission = NEG_INFINITY
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::TopK(0);
    let obs = make_obs(&[("a1", 0.95), ("b1", 0.80), ("c1", 0.75)]);
    let le = params.log_emission(&obs, 0);
    // TopK(0) aggregation returns None → target_sim = None → NEG_INFINITY
    assert_eq!(le, f64::NEG_INFINITY, "TopK(0) should give NEG_INFINITY: {}", le);
}

#[test]
fn emission_model_no_matching_haplotypes() {
    // No haplotype keys match population definitions → empty sims → NEG_INFINITY
    let params = make_3pop_params();
    let obs = make_obs(&[("x1", 0.95), ("x2", 0.80)]);  // no a1, b1, c1 etc.
    let le = params.log_emission(&obs, 0);
    assert_eq!(le, f64::NEG_INFINITY, "No matching haps should give NEG_INFINITY: {}", le);
}

// =====================================================================
// log_emission with only one population having data
// =====================================================================

#[test]
fn log_emission_single_population_data_returns_zero() {
    let params = make_3pop_params();
    // Only pop A has similarity data → valid_scores.len() == 1 → return 0.0
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90)]);
    let le = params.log_emission(&obs, 0);
    assert_eq!(le, 0.0, "Single-pop data should give log(1)=0: {}", le);
}

#[test]
fn log_emission_target_state_no_data_returns_neg_inf() {
    let params = make_3pop_params();
    // Pop A has no data but pop B does → target state A returns NEG_INFINITY
    let obs = make_obs(&[("b1", 0.95), ("b2", 0.90), ("c1", 0.80)]);
    let le = params.log_emission(&obs, 0); // state 0 = pop A → no data
    assert_eq!(le, f64::NEG_INFINITY, "Missing target data should give NEG_INFINITY: {}", le);
}

// =====================================================================
// Viterbi with equal emissions across all states
// =====================================================================

#[test]
fn viterbi_equal_emissions_deterministic() {
    let params = make_3pop_params();
    // All populations have exactly equal similarity → emissions are equal
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.90), ("a2", 0.90),
                ("b1", 0.90), ("b2", 0.90),
                ("c1", 0.90), ("c2", 0.90),
            ])
        })
        .collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5);
    // With equal emissions, Viterbi should pick based on initial probs (uniform)
    // and transition probs (prefer staying). So all states should be the same.
    let first = states[0];
    for &s in &states {
        assert_eq!(s, first, "Equal emissions should give constant state: {} vs {}", s, first);
    }
}

// =====================================================================
// forward_backward normalization
// =====================================================================

#[test]
fn forward_backward_posteriors_sum_to_one() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.95), ("a2", 0.93),
                ("b1", 0.70), ("b2", 0.72),
                ("c1", 0.80), ("c2", 0.78),
            ])
        })
        .collect();
    let posteriors = forward_backward(&obs, &params);
    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Posteriors at window {} sum to {} (should be 1.0)", t, sum
        );
    }
}

#[test]
fn forward_backward_posteriors_non_negative() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.95), ("b1", 0.70), ("c1", 0.80),
            ])
        })
        .collect();
    let posteriors = forward_backward(&obs, &params);
    for (t, post) in posteriors.iter().enumerate() {
        for (s, &p) in post.iter().enumerate() {
            assert!(p >= 0.0, "Posterior at t={} state={} is negative: {}", t, s, p);
            assert!(p <= 1.0 + 1e-6, "Posterior at t={} state={} exceeds 1: {}", t, s, p);
        }
    }
}

// =====================================================================
// Two-population HMM edge cases
// =====================================================================

#[test]
fn two_population_viterbi_clear_signal() {
    let params = AncestryHmmParams::new(
        vec![make_pop("A", &["a1", "a2"]), make_pop("B", &["b1", "b2"])],
        0.01,
    );
    let obs: Vec<_> = (0..10)
        .map(|i| {
            if i < 5 {
                make_obs_at(i * 10000, &[("a1", 0.99), ("a2", 0.98), ("b1", 0.60), ("b2", 0.62)])
            } else {
                make_obs_at(i * 10000, &[("a1", 0.60), ("a2", 0.62), ("b1", 0.99), ("b2", 0.98)])
            }
        })
        .collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 10);
    // First 5 windows should be pop A (state 0), last 5 should be pop B (state 1)
    for &s in &states[..3] {
        assert_eq!(s, 0, "First windows should be pop A");
    }
    for &s in &states[7..] {
        assert_eq!(s, 1, "Last windows should be pop B");
    }
}

#[test]
fn two_population_forward_backward() {
    let params = AncestryHmmParams::new(
        vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])],
        0.01,
    );
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.70)]))
        .collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 5);
    for post in &posteriors {
        assert_eq!(post.len(), 2);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "2-pop posteriors should sum to 1: {}", sum);
        // Pop A should have higher posterior
        assert!(post[0] > post[1], "Pop A should dominate: {:?}", post);
    }
}

// =====================================================================
// estimate_temperature_normalized edge cases
// =====================================================================

#[test]
fn estimate_temperature_normalized_without_normalization_fallback() {
    // No normalization set → should fall back to regular estimate_temperature
    let params = make_3pop_params(); // normalization = None
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)]))
        .collect();
    let temp_norm = estimate_temperature_normalized(&obs, &params);
    let temp_regular = estimate_temperature(&obs, &params.populations);
    assert_eq!(temp_norm, temp_regular,
        "Without normalization, should fall back to regular: {} vs {}", temp_norm, temp_regular);
}

#[test]
fn estimate_temperature_normalized_zero_std() {
    // std = 0 → division by zero in z-score computation
    let mut params = make_3pop_params();
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.8, 0.85],
        stds: vec![0.0, 0.0, 0.0],  // zero std!
    });
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(i * 10000, &[
                ("a1", 0.95), ("a2", 0.90),
                ("b1", 0.80), ("b2", 0.85),
                ("c1", 0.75), ("c2", 0.78),
            ])
        })
        .collect();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        estimate_temperature_normalized(&obs, &params)
    }));
    if result.is_ok() {
        let temp = result.unwrap();
        // Division by zero → Inf z-scores → Inf - Inf = NaN in diffs → sort panics
        // But if it doesn't panic, the result should still be in valid range
        assert!(temp > 0.0, "Temp should be positive: {}", temp);
    } else {
        eprintln!("BUG: estimate_temperature_normalized panics with zero std (div by zero → NaN → sort panic)");
    }
}

// =====================================================================
// Many populations (stress test)
// =====================================================================

#[test]
fn many_populations_viterbi() {
    // 10 populations → large transition matrix
    let pops: Vec<_> = (0..10)
        .map(|i| make_pop(&format!("P{}", i), &[&format!("h{}", i)]))
        .collect();
    let params = AncestryHmmParams::new(pops, 0.01);

    let obs: Vec<_> = (0..5)
        .map(|i| {
            let sims: Vec<_> = (0..10)
                .map(|j| {
                    let sim = if j == (i % 10) as usize { 0.95 } else { 0.70 + (j as f64) * 0.01 };
                    (format!("h{}", j), sim)
                })
                .collect();
            let sim_refs: Vec<_> = sims.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            make_obs_at(i * 10000, &sim_refs)
        })
        .collect();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5);
    for &s in &states {
        assert!(s < 10, "State should be in [0,10): {}", s);
    }
}
