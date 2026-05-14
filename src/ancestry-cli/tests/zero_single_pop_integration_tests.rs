//! Integration tests for zero/single population edge cases after fix 35e90a0.
//!
//! Verifies that the AncestryHmmParams zero/single population fix
//! (transitions[0][0] = 1.0 for k=1, empty params for k=0) interacts
//! correctly with all downstream HMM functions:
//! - viterbi, forward_backward (core decoders)
//! - baum_welch, baum_welch_full (parameter learning)
//! - learn_normalization, estimate_emissions (emission estimation)
//! - set_switch_prob, set_initial_probs, set_proportional_transitions (param setters)
//! - set_temperature, set_emission_model, set_coverage_weight (param setters)
//!
//! The zero-population case is pathological (CLI validates ≥2 pops), but these
//! tests ensure library-level safety for programmatic use.

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    forward_backward, viterbi,
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

fn single_pop_params() -> AncestryHmmParams {
    AncestryHmmParams::new(vec![make_pop("EUR", &["h1", "h2"])], 0.01)
}

fn zero_pop_params() -> AncestryHmmParams {
    AncestryHmmParams::new(vec![], 0.01)
}

// =====================================================================
// Single-population: viterbi
// =====================================================================

#[test]
fn single_pop_viterbi_all_state_zero() {
    let params = single_pop_params();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[("h1", 0.95), ("h2", 0.90)])).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5);
    assert!(states.iter().all(|&s| s == 0), "Only state 0 exists");
}

#[test]
fn single_pop_viterbi_empty_observations() {
    let params = single_pop_params();
    let states = viterbi(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn single_pop_viterbi_single_observation() {
    let params = single_pop_params();
    let obs = vec![make_obs(&[("h1", 0.8)])];
    let states = viterbi(&obs, &params);
    assert_eq!(states, vec![0]);
}

#[test]
fn single_pop_viterbi_no_similarity_data() {
    // Observation with no matching haplotypes — emission should still be finite
    let params = single_pop_params();
    let obs = vec![make_obs(&[("unknown_hap", 0.9)])];
    let states = viterbi(&obs, &params);
    assert_eq!(states, vec![0]);
}

// =====================================================================
// Single-population: forward_backward
// =====================================================================

#[test]
fn single_pop_forward_backward_posteriors_all_one() {
    let params = single_pop_params();
    let obs: Vec<_> = (0..3).map(|_| make_obs(&[("h1", 0.95)])).collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 3);
    for post in &posteriors {
        assert_eq!(post.len(), 1);
        assert!((post[0] - 1.0).abs() < 1e-10, "Single state posterior must be 1.0, got {}", post[0]);
    }
}

#[test]
fn single_pop_forward_backward_empty() {
    let params = single_pop_params();
    let posteriors = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
}

// =====================================================================
// Single-population: baum_welch
// =====================================================================

#[test]
fn single_pop_baum_welch_returns_neg_infinity() {
    // baum_welch guards with k < 2
    let mut params = single_pop_params();
    let obs = vec![make_obs(&[("h1", 0.9)]), make_obs(&[("h1", 0.85)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_refs, 10, 1e-6);
    assert!(ll == f64::NEG_INFINITY, "BW with k=1 must return -inf");
    // Transition should be unchanged
    assert_eq!(params.transitions[0][0], 1.0);
}

#[test]
fn single_pop_baum_welch_full_returns_neg_infinity() {
    let mut params = single_pop_params();
    let obs = vec![make_obs(&[("h1", 0.9)]), make_obs(&[("h1", 0.85)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch_full(&obs_refs, 10, 1e-6, true);
    assert!(ll == f64::NEG_INFINITY);
}

// =====================================================================
// Single-population: set_switch_prob idempotency
// =====================================================================

#[test]
fn single_pop_set_switch_prob_multiple_calls() {
    let mut params = single_pop_params();
    for sp in &[0.0, 0.001, 0.5, 1.0] {
        params.set_switch_prob(*sp);
        assert_eq!(params.transitions[0][0], 1.0,
            "set_switch_prob({}) must keep P(stay)=1.0 for single state", sp);
    }
}

// =====================================================================
// Single-population: learn_normalization and estimate_emissions
// =====================================================================

#[test]
fn single_pop_learn_normalization() {
    let mut params = single_pop_params();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[("h1", 0.95), ("h2", 0.88)])).collect();
    params.learn_normalization(&obs);
    assert!(params.normalization.is_some());
}

#[test]
fn single_pop_estimate_emissions() {
    let mut params = single_pop_params();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[("h1", 0.95), ("h2", 0.88)])).collect();
    // With a single pop, all similarities are "same_pop" — diff_pop_sims is empty
    params.estimate_emissions(&obs);
    // emission_same_pop_mean should be updated
    assert!(params.emission_same_pop_mean > 0.0);
}

#[test]
fn single_pop_learn_normalization_empty_obs() {
    let mut params = single_pop_params();
    params.learn_normalization(&[]);
    // Should produce normalization with zero counts → mean=0, std=1e-6
    let norm = params.normalization.unwrap();
    assert_eq!(norm.means.len(), 1);
    assert_eq!(norm.means[0], 0.0);
}

// =====================================================================
// Single-population: set_initial_probs
// =====================================================================

#[test]
fn single_pop_set_initial_probs() {
    let mut params = single_pop_params();
    params.set_initial_probs(&[0.5]);
    assert!((params.initial[0] - 1.0).abs() < 1e-10, "Normalized single initial must be 1.0");
}

#[test]
fn single_pop_set_initial_probs_wrong_length_no_op() {
    let mut params = single_pop_params();
    let orig = params.initial[0];
    params.set_initial_probs(&[0.3, 0.7]);
    assert_eq!(params.initial[0], orig, "Wrong-length priors must be no-op");
}

// =====================================================================
// Single-population: set_proportional_transitions
// =====================================================================

#[test]
fn single_pop_set_proportional_transitions() {
    let mut params = single_pop_params();
    params.set_proportional_transitions(&[1.0], &[0.05]);
    // With 1 state: stay_prob = 1 - 0.05 = 0.95, but there's no other state to switch to
    // The loop only has i==j, so transitions[0][0] = stay_prob
    assert!((params.transitions[0][0] - 0.95).abs() < 1e-10);
}

// =====================================================================
// Single-population: temperature and model setters
// =====================================================================

#[test]
fn single_pop_set_temperature() {
    let mut params = single_pop_params();
    params.set_temperature(0.05);
    // Should not panic; emission model still works
    let obs = make_obs(&[("h1", 0.9)]);
    let le = params.log_emission(&obs, 0);
    assert!(le.is_finite(), "log_emission after set_temperature must be finite");
}

#[test]
fn single_pop_set_emission_model_all_variants() {
    let mut params = single_pop_params();
    let obs = make_obs(&[("h1", 0.9), ("h2", 0.85)]);
    for model in &[EmissionModel::Max, EmissionModel::Mean, EmissionModel::Median] {
        params.set_emission_model(model.clone());
        let le = params.log_emission(&obs, 0);
        assert!(le.is_finite(), "log_emission with {:?} model must be finite", model);
    }
}

// =====================================================================
// Zero-population: constructor properties
// =====================================================================

#[test]
fn zero_pop_params_are_empty() {
    let params = zero_pop_params();
    assert_eq!(params.n_states, 0);
    assert!(params.transitions.is_empty());
    assert!(params.initial.is_empty());
    assert!(params.populations.is_empty());
}

// =====================================================================
// Zero-population: setters are no-ops
// =====================================================================

#[test]
fn zero_pop_set_switch_prob_no_panic() {
    let mut params = zero_pop_params();
    params.set_switch_prob(0.5);
    assert_eq!(params.n_states, 0);
    assert!(params.transitions.is_empty());
}

#[test]
fn zero_pop_set_initial_probs_no_op() {
    let mut params = zero_pop_params();
    params.set_initial_probs(&[]);
    // Empty priors sum = 0, so guard sum > 0 prevents update
    assert!(params.initial.is_empty());
}

#[test]
fn zero_pop_set_proportional_transitions_no_op() {
    let mut params = zero_pop_params();
    params.set_proportional_transitions(&[], &[]);
    assert!(params.transitions.is_empty());
}

#[test]
fn zero_pop_set_temperature_no_panic() {
    let mut params = zero_pop_params();
    params.set_temperature(0.05);
    // Just verifying no panic
}

// =====================================================================
// Zero-population: baum_welch short-circuits
// =====================================================================

#[test]
fn zero_pop_baum_welch_returns_neg_infinity() {
    let mut params = zero_pop_params();
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_refs, 10, 1e-6);
    assert!(ll == f64::NEG_INFINITY);
}

#[test]
fn zero_pop_baum_welch_full_returns_neg_infinity() {
    let mut params = zero_pop_params();
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch_full(&obs_refs, 10, 1e-6, false);
    assert!(ll == f64::NEG_INFINITY);
}

// =====================================================================
// Zero-population: learn_normalization and estimate_emissions
// =====================================================================

#[test]
fn zero_pop_learn_normalization_empty() {
    let mut params = zero_pop_params();
    params.learn_normalization(&[]);
    // With n_states=0, all loops iterate over empty ranges — no panic
    let norm = params.normalization.unwrap();
    assert!(norm.means.is_empty());
    assert!(norm.stds.is_empty());
}

#[test]
fn zero_pop_estimate_emissions_no_panic() {
    let mut params = zero_pop_params();
    let obs = vec![make_obs(&[("h1", 0.9)])];
    // populations is empty so inner loop body never executes
    params.estimate_emissions(&obs);
}

// =====================================================================
// Zero-population: viterbi and forward_backward with empty observations
// =====================================================================

#[test]
fn zero_pop_viterbi_empty_obs() {
    let params = zero_pop_params();
    let states = viterbi(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn zero_pop_forward_backward_empty_obs() {
    let params = zero_pop_params();
    let posteriors = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
}

// =====================================================================
// Transition from k=2 to k=1: downgrade scenario (re-create with fewer pops)
// =====================================================================

#[test]
fn downgrade_two_to_one_pop_transitions_correct() {
    // Simulate a scenario where analysis starts with 2 pops but one is dropped
    let params2 = AncestryHmmParams::new(
        vec![make_pop("EUR", &["h1"]), make_pop("AFR", &["h2"])],
        0.01,
    );
    assert_eq!(params2.n_states, 2);
    assert!((params2.transitions[0][0] - 0.99).abs() < 1e-10);

    // Now re-create with just 1 pop
    let params1 = AncestryHmmParams::new(
        vec![make_pop("EUR", &["h1"])],
        0.01,
    );
    assert_eq!(params1.n_states, 1);
    assert_eq!(params1.transitions[0][0], 1.0);
    assert_eq!(params1.initial[0], 1.0);
}
