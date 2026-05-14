//! NaN and Infinity safety tests for ancestry-cli HMM functions.
//!
//! These tests verify that the HMM pipeline handles degenerate float values
//! (NaN, Infinity, -Infinity) without panicking. In pangenome data, NaN can
//! appear when identity computation fails (e.g., zero-length alignment)
//! and Infinity can arise from log(0) in certain edge cases.
//!
//! Previously, 15 calls to `partial_cmp().unwrap()` in hmm.rs could panic on NaN
//! because `f64::NAN.partial_cmp(&x)` returns `None`. These have been replaced
//! with `total_cmp()` which handles NaN safely. These tests verify the fix.

use std::panic;

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    PopulationNormalization, estimate_temperature, estimate_temperature_normalized,
    estimate_switch_prob, forward_backward, posterior_decode, viterbi,
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

// =====================================================================
// log_emission with NaN in similarities — Max model
// =====================================================================

#[test]
fn log_emission_nan_similarity_max_model_does_not_panic() {
    // Max model: fold(0.0, f64::max) swallows NaN because max(x, NaN) = x
    // So NaN similarities effectively become 0.0 → filtered by s > 0.0 guard
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", f64::NAN),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    let result = panic::catch_unwind(|| params.log_emission(&obs, 0));
    assert!(result.is_ok(), "log_emission should not panic on NaN with Max model");
    let le = result.unwrap();
    assert!(le.is_finite(), "log_emission with one NaN hap should still be finite: {}", le);
}

#[test]
fn log_emission_all_nan_similarities_max_model() {
    // All similarities are NaN → fold(0.0, max) gives 0.0 → not > 0.0 → NEG_INFINITY
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", f64::NAN),
        ("a2", f64::NAN),
        ("b1", f64::NAN),
        ("b2", f64::NAN),
        ("c1", f64::NAN),
        ("c2", f64::NAN),
    ]);
    let result = panic::catch_unwind(|| params.log_emission(&obs, 0));
    assert!(result.is_ok(), "log_emission should not panic when all similarities are NaN");
}

// =====================================================================
// log_emission with NaN — Median model (sort panics on NaN)
// =====================================================================

#[test]
fn log_emission_nan_similarity_median_model_safety() {
    // Median model uses total_cmp() for sorting — NaN sorts to the end safely.
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::Median;
    let obs = make_obs(&[
        ("a1", f64::NAN),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| params.log_emission(&obs, 0)));
    assert!(result.is_ok(), "Median model should not panic on NaN after total_cmp fix");
    let le = result.unwrap();
    assert!(le.is_finite() || le == f64::NEG_INFINITY,
        "Median with NaN should be finite or NEG_INFINITY: {}", le);
}

// =====================================================================
// log_emission with NaN — TopK model (sort panics on NaN)
// =====================================================================

#[test]
fn log_emission_nan_similarity_topk_model_safety() {
    // TopK model uses total_cmp() for sorting — NaN sorts safely.
    let mut params = make_3pop_params();
    params.emission_model = EmissionModel::TopK(2);
    let obs = make_obs(&[
        ("a1", f64::NAN),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| params.log_emission(&obs, 0)));
    assert!(result.is_ok(), "TopK model should not panic on NaN after total_cmp fix");
    let le = result.unwrap();
    assert!(le.is_finite() || le == f64::NEG_INFINITY,
        "TopK with NaN should be finite or NEG_INFINITY: {}", le);
}

// =====================================================================
// log_emission with Infinity in similarities
// =====================================================================

#[test]
fn log_emission_infinity_similarity_does_not_panic() {
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", f64::INFINITY),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    let result = panic::catch_unwind(|| params.log_emission(&obs, 0));
    assert!(result.is_ok(), "log_emission should not panic on INFINITY similarity");
}

#[test]
fn log_emission_neg_infinity_similarity() {
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", f64::NEG_INFINITY),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    // NEG_INFINITY < 0.0 so it gets filtered by max > 0.0 or s > 0.0 guard
    let result = panic::catch_unwind(|| params.log_emission(&obs, 0));
    assert!(result.is_ok(), "log_emission should not panic on NEG_INFINITY similarity");
}

// =====================================================================
// log_emission with negative similarities
// =====================================================================

#[test]
fn log_emission_negative_similarity_filtered() {
    // Negative similarities are filtered by s > 0.0 guard
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", -0.5),
        ("a2", 0.95),
        ("b1", 0.80),
        ("b2", 0.85),
        ("c1", 0.75),
        ("c2", 0.78),
    ]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "Negative sim should be filtered: {}", le);
}

#[test]
fn log_emission_all_negative_returns_neg_infinity() {
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", -0.5),
        ("a2", -0.3),
        ("b1", -0.4),
        ("b2", -0.2),
        ("c1", -0.6),
        ("c2", -0.1),
    ]);
    let le = params.log_emission(&obs, 0);
    assert_eq!(le, f64::NEG_INFINITY, "All negative should give NEG_INFINITY");
}

// =====================================================================
// Viterbi with NaN emissions — argmax partial_cmp().unwrap() risk
// =====================================================================

#[test]
fn viterbi_nan_in_one_similarity_max_model() {
    // With Max model, NaN is swallowed by fold(0.0, f64::max)
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 2 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.70),
                    ("b2", 0.72),
                    ("c1", 0.80),
                    ("c2", 0.78),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    assert!(result.is_ok(), "Viterbi should not panic on NaN with Max model");
    let states = result.unwrap();
    assert_eq!(states.len(), 5);
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn viterbi_all_nan_similarities() {
    // All NaN → all emissions -inf → Viterbi should still return valid states
    let params = make_3pop_params();
    let obs: Vec<_> = (0..3)
        .map(|i| make_obs_at(i * 10000, &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)]))
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    assert!(result.is_ok(), "Viterbi should handle all-NaN without panic");
    let states = result.unwrap();
    assert_eq!(states.len(), 3);
}

// =====================================================================
// Forward-backward with NaN
// =====================================================================

#[test]
fn forward_backward_nan_in_similarity() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 2 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.70),
                    ("c1", 0.80),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| forward_backward(&obs, &params));
    assert!(result.is_ok(), "forward_backward should not panic on NaN with Max model");
}

// =====================================================================
// posterior_decode with NaN
// =====================================================================

#[test]
fn posterior_decode_nan_in_similarity() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 2 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.70),
                    ("c1", 0.80),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| posterior_decode(&obs, &params));
    assert!(result.is_ok(), "posterior_decode should not panic on NaN");
    if let Ok(states) = result {
        assert_eq!(states.len(), 5);
    }
}

// =====================================================================
// estimate_temperature with NaN similarities
// =====================================================================

#[test]
fn estimate_temperature_nan_in_similarities_safety() {
    // estimate_temperature uses total_cmp() for sorting diffs — NaN-safe.
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 5 { f64::NAN } else { 0.95 }),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| estimate_temperature(&obs, &params.populations));
    assert!(result.is_ok(), "estimate_temperature should not panic on NaN after total_cmp fix");
    let temp = result.unwrap();
    assert!(temp >= 0.01 && temp <= 0.15, "temp out of range: {}", temp);
}

#[test]
fn estimate_temperature_all_nan_similarities() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)]))
        .collect();
    let result = panic::catch_unwind(|| estimate_temperature(&obs, &params.populations));
    assert!(result.is_ok(), "estimate_temperature should not panic on all-NaN after total_cmp fix");
    let temp = result.unwrap();
    assert_eq!(temp, 0.03, "All NaN should return fallback: {}", temp);
}

// =====================================================================
// estimate_temperature_normalized with NaN
// =====================================================================

#[test]
fn estimate_temperature_normalized_nan_safety() {
    let mut params = make_3pop_params();
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.8, 0.85],
        stds: vec![0.05, 0.05, 0.05],
    });
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 3 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        estimate_temperature_normalized(&obs, &params)
    }));
    assert!(result.is_ok(), "estimate_temperature_normalized should not panic on NaN after total_cmp fix");
    let temp = result.unwrap();
    assert!(temp > 0.0, "temp should be positive: {}", temp);
}

// =====================================================================
// estimate_switch_prob with NaN
// =====================================================================

#[test]
fn estimate_switch_prob_nan_in_similarity() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 5 { f64::NAN } else { 0.95 }),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| {
        estimate_switch_prob(&obs, &params.populations, params.emission_std)
    });
    assert!(result.is_ok(), "estimate_switch_prob should not panic on NaN after total_cmp fix");
    let sp = result.unwrap();
    assert!(sp > 0.0 && sp < 1.0, "switch_prob out of range: {}", sp);
}

// =====================================================================
// estimate_emissions with NaN — the partial_cmp().unwrap() in max_by
// =====================================================================

#[test]
fn estimate_emissions_nan_similarity_safety() {
    // RISK: estimate_emissions line 264: .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    // NaN from fold(0.0, f64::max) is swallowed (max returns non-NaN).
    // So this should be safe with Max model.
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", f64::NAN),
                    ("a2", 0.95),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.estimate_emissions(&obs);
    }));
    assert!(result.is_ok(), "estimate_emissions should not panic with NaN (Max model swallows NaN)");
}

#[test]
fn estimate_emissions_all_nan_similarity() {
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)],
            )
        })
        .collect();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.estimate_emissions(&obs);
    }));
    assert!(result.is_ok(), "estimate_emissions should handle all-NaN without panic");
}

// =====================================================================
// Temperature API edge cases (bypassing CLI validation)
// =====================================================================

#[test]
fn set_temperature_zero_via_api() {
    // CLI rejects temperature=0, but API allows it.
    // Division by zero in softmax: ((s - max) / 0.0).exp()
    let mut params = make_3pop_params();
    params.set_temperature(0.0);
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("c1", 0.85)]);

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| params.log_emission(&obs, 0)));
    // Division by zero produces Inf/NaN but shouldn't panic
    assert!(result.is_ok(), "log_emission with temp=0 should not panic");
    if let Ok(le) = result {
        // With temp=0, (s-max)/0 = 0/0=NaN for max, and neg/0=-Inf for others
        // exp(NaN)=NaN, exp(-Inf)=0, ln(NaN)=NaN → result is NaN or -Inf
        // This is mathematically degenerate but shouldn't crash
        assert!(
            le.is_nan() || le.is_infinite() || le.is_finite(),
            "Any float result is acceptable for temp=0: {}",
            le
        );
    }
}

#[test]
fn set_temperature_negative_via_api() {
    // Negative temperature inverts the softmax (lowest similarity gets highest prob)
    let mut params = make_3pop_params();
    params.set_temperature(-0.03);
    let obs = make_obs(&[("a1", 0.95), ("a2", 0.90), ("b1", 0.80), ("c1", 0.85)]);

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| params.log_emission(&obs, 0)));
    assert!(result.is_ok(), "log_emission with negative temp should not panic");
}

#[test]
fn set_temperature_very_large() {
    // Very large temperature → uniform distribution (all softmax weights equal)
    let mut params = make_3pop_params();
    params.set_temperature(1e6);
    let obs = make_obs(&[
        ("a1", 0.99),
        ("a2", 0.95),
        ("b1", 0.50),
        ("b2", 0.48),
        ("c1", 0.75),
        ("c2", 0.72),
    ]);
    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);
    let le_c = params.log_emission(&obs, 2);

    // With very high temperature, all emissions should be near -ln(3) ≈ -1.099
    let expected = -(3.0_f64).ln();
    assert!((le_a - expected).abs() < 0.01, "High temp should give near-uniform: {}", le_a);
    assert!((le_b - expected).abs() < 0.01, "High temp should give near-uniform: {}", le_b);
    assert!((le_c - expected).abs() < 0.01, "High temp should give near-uniform: {}", le_c);
}

#[test]
fn set_temperature_very_small_positive() {
    // Very small temperature → winner-take-all
    let mut params = make_3pop_params();
    params.set_temperature(1e-10);
    let obs = make_obs(&[
        ("a1", 0.99),
        ("a2", 0.95),
        ("b1", 0.50),
        ("b2", 0.48),
        ("c1", 0.75),
        ("c2", 0.72),
    ]);
    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);

    // Pop A (highest) should get log(1) ≈ 0, pop B should get very negative
    assert!(le_a > le_b, "Small temp should strongly favor highest similarity");
    assert!(le_a > -0.01, "Winner should get near-zero log prob: {}", le_a);
}

// =====================================================================
// Baum-Welch with NaN
// =====================================================================

#[test]
fn baum_welch_nan_observation_does_not_panic() {
    let mut params = make_3pop_params();
    let obs_seq: Vec<AncestryObservation> = (0..10)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 5 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.70),
                    ("c1", 0.80),
                ],
            )
        })
        .collect();
    let obs_slices: Vec<&[AncestryObservation]> = vec![obs_seq.as_slice()];
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.baum_welch(&obs_slices, 5, 1e-6)
    }));
    // Baum-Welch internally calls forward_backward which uses emissions
    // With Max model, NaN is swallowed, so this should be safe
    assert!(result.is_ok(), "Baum-Welch should not panic on NaN with Max model");
}

// =====================================================================
// learn_normalization with NaN
// =====================================================================

#[test]
fn learn_normalization_nan_similarity() {
    let mut params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 3 { f64::NAN } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.80),
                    ("b2", 0.85),
                    ("c1", 0.75),
                    ("c2", 0.78),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        params.learn_normalization(&obs);
    }));
    assert!(
        result.is_ok(),
        "learn_normalization should not panic on NaN similarity"
    );
    if let Ok(()) = result {
        let norm = params.normalization.as_ref().unwrap();
        for &m in &norm.means {
            assert!(m.is_finite(), "Normalization mean should be finite: {}", m);
        }
        for &s in &norm.stds {
            assert!(s.is_finite() && s > 0.0, "Normalization std should be finite positive: {}", s);
        }
    }
}

// =====================================================================
// Infinity in similarities
// =====================================================================

#[test]
fn viterbi_infinity_similarity_no_longer_panics() {
    // Previously panicked because INFINITY in softmax produced NaN scores and
    // partial_cmp().unwrap() failed. Fixed by replacing with total_cmp().
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 2 { f64::INFINITY } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    assert!(result.is_ok(), "Viterbi should not panic on INFINITY after total_cmp fix");
    let states = result.unwrap();
    assert_eq!(states.len(), 5);
}

#[test]
fn forward_backward_infinity_similarity() {
    // forward_backward with INFINITY — safe after total_cmp fix.
    let params = make_3pop_params();
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[
                    ("a1", if i == 2 { f64::INFINITY } else { 0.95 }),
                    ("a2", 0.90),
                    ("b1", 0.80),
                    ("c1", 0.85),
                ],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| forward_backward(&obs, &params));
    assert!(result.is_ok(), "forward_backward should not panic on INFINITY after total_cmp fix");
}

// =====================================================================
// Two-population edge case (minimal valid state count)
// =====================================================================

#[test]
fn two_population_nan_in_one() {
    let params = AncestryHmmParams::new(
        vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])],
        0.01,
    );
    let obs: Vec<_> = (0..5)
        .map(|i| {
            make_obs_at(
                i * 10000,
                &[("a1", if i == 2 { f64::NAN } else { 0.95 }), ("b1", 0.80)],
            )
        })
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    assert!(result.is_ok(), "Two-pop viterbi should handle NaN");
}

// =====================================================================
// Subnormal float values
// =====================================================================

#[test]
fn log_emission_subnormal_similarities() {
    // Subnormal (denormalized) floats — extremely small but not zero
    let params = make_3pop_params();
    let obs = make_obs(&[
        ("a1", 5e-324), // smallest positive f64
        ("b1", 5e-324),
        ("c1", 5e-324),
    ]);
    let le = params.log_emission(&obs, 0);
    // Subnormals are > 0.0, so they should pass the guard
    assert!(
        le.is_finite() || le == f64::NEG_INFINITY,
        "Subnormal should produce valid emission: {}",
        le
    );
}

// =====================================================================
// Mixed NaN and valid across windows
// =====================================================================

#[test]
fn viterbi_alternating_nan_and_valid_windows() {
    let params = make_3pop_params();
    let obs: Vec<_> = (0..10)
        .map(|i| {
            if i % 3 == 0 {
                // Every 3rd window has NaN
                make_obs_at(i * 10000, &[("a1", f64::NAN), ("b1", f64::NAN), ("c1", f64::NAN)])
            } else {
                make_obs_at(i * 10000, &[("a1", 0.95), ("b1", 0.80), ("c1", 0.85)])
            }
        })
        .collect();
    let result = panic::catch_unwind(|| viterbi(&obs, &params));
    assert!(result.is_ok(), "Viterbi should handle alternating NaN/valid windows");
    let states = result.unwrap();
    assert_eq!(states.len(), 10);
}
