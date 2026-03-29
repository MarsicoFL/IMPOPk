//! Tests for CrossValidationResult::print_summary (smoke), ancestry pipeline
//! edge cases, and AncestryHmmParams numerical stability.
//!
//! Targets:
//! - print_summary: verify it doesn't panic with various inputs
//! - AncestryHmmParams::baum_welch: short data, convergence behavior
//! - posterior_decode vs viterbi consistency
//! - AncestryHmmParams::learn_normalization edge cases
//! - estimate_admixture_proportions edge cases

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, AncestrySegment,
    CrossValidationResult, cross_validate,
    estimate_admixture_proportions,
};
use hprc_ancestry_cli::hmm::{
    forward_backward, posterior_decode, viterbi,
};

/// Helper to create a minimal CrossValidationResult for testing.
fn make_cv_result(pops: &[&str], accuracy: f64) -> CrossValidationResult {
    let mut accuracy_per_pop = HashMap::new();
    let mut precision_per_pop = HashMap::new();
    let mut recall_per_pop = HashMap::new();
    let mut f1_per_pop = HashMap::new();
    let mut n_windows_per_pop = HashMap::new();
    let mut confusion = HashMap::new();

    for &pop in pops {
        accuracy_per_pop.insert(pop.to_string(), accuracy);
        precision_per_pop.insert(pop.to_string(), accuracy);
        recall_per_pop.insert(pop.to_string(), accuracy);
        f1_per_pop.insert(pop.to_string(), accuracy);
        n_windows_per_pop.insert(pop.to_string(), 100);
        for &pred_pop in pops {
            let count = if pop == pred_pop { 90 } else { 5 };
            confusion.insert((pop.to_string(), pred_pop.to_string()), count);
        }
    }

    CrossValidationResult {
        overall_accuracy: accuracy,
        accuracy_per_pop,
        precision_per_pop,
        recall_per_pop,
        f1_per_pop,
        n_windows_per_pop,
        confusion,
    }
}

fn make_obs(chrom: &str, start: u64, end: u64, sample: &str, sims: Vec<(&str, f64)>) -> AncestryObservation {
    let similarities: HashMap<String, f64> = sims
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    AncestryObservation {
        chrom: chrom.to_string(),
        start,
        end,
        sample: sample.to_string(),
        similarities,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|h| h.to_string()).collect(),
    }
}

fn make_segment(chrom: &str, start: u64, end: u64, sample: &str, ancestry_idx: usize, ancestry_name: &str, n_windows: usize) -> AncestrySegment {
    AncestrySegment {
        chrom: chrom.to_string(),
        start,
        end,
        sample: sample.to_string(),
        ancestry_idx,
        ancestry_name: ancestry_name.to_string(),
        n_windows,
        mean_similarity: 0.95,
        mean_posterior: Some(0.9),
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

// === print_summary smoke tests ===

/// print_summary should not panic with a normal result.
#[test]
fn test_print_summary_normal_2pop() {
    let result = make_cv_result(&["EUR", "AFR"], 0.85);
    result.print_summary(); // Should not panic
}

/// print_summary should not panic with 3 populations.
#[test]
fn test_print_summary_normal_3pop() {
    let result = make_cv_result(&["EUR", "AFR", "AMR"], 0.92);
    result.print_summary();
}

/// print_summary should not panic with zero accuracy.
#[test]
fn test_print_summary_zero_accuracy() {
    let result = make_cv_result(&["EUR", "AFR"], 0.0);
    result.print_summary();
}

/// print_summary should not panic with perfect accuracy.
#[test]
fn test_print_summary_perfect_accuracy() {
    let result = make_cv_result(&["EUR", "AFR"], 1.0);
    result.print_summary();
}

/// print_summary should not panic with empty confusion matrix entries.
#[test]
fn test_print_summary_missing_confusion_entries() {
    let mut result = make_cv_result(&["EUR", "AFR"], 0.5);
    result.confusion.clear(); // Remove all confusion entries
    result.print_summary(); // Should handle missing entries gracefully (unwrap_or)
}

/// print_summary should not panic with single population.
#[test]
fn test_print_summary_single_pop() {
    let result = make_cv_result(&["EUR"], 1.0);
    result.print_summary();
}

/// print_summary from actual cross_validate result.
#[test]
fn test_print_summary_from_cross_validate() {
    let pops = vec![
        make_pop("A", &["a1", "a2", "a3"]),
        make_pop("B", &["b1", "b2", "b3"]),
    ];
    // cross_validate expects observations keyed by haplotype name
    let mut obs_map: HashMap<String, Vec<AncestryObservation>> = HashMap::new();
    for hap in &["a1", "a2", "a3", "b1", "b2", "b3"] {
        let obs: Vec<AncestryObservation> = (0..10).map(|i| {
            make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, hap,
                vec![("a1", 0.95), ("a2", 0.93), ("a3", 0.94), ("b1", 0.80), ("b2", 0.79), ("b3", 0.78)])
        }).collect();
        obs_map.insert(hap.to_string(), obs);
    }

    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let result = cross_validate(&obs_map, &pops, &params);
    result.print_summary(); // Should not panic
}

// === AncestryHmmParams edge cases ===

/// AncestryHmmParams with single population should work.
#[test]
fn test_ancestry_hmm_single_population() {
    let pops = vec![make_pop("A", &["a1", "a2"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    assert_eq!(params.n_states, 1);
}

/// learn_normalization with empty observations should not panic.
#[test]
fn test_learn_normalization_empty() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.learn_normalization(&[]);
    // Should not panic or change anything significantly
}

/// learn_normalization with single observation.
#[test]
fn test_learn_normalization_single_obs() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![make_obs("chr1", 0, 10000, "q", vec![("a1", 0.9), ("b1", 0.8)])];
    params.learn_normalization(&obs);
}

/// estimate_emissions should not panic with varied data.
#[test]
fn test_estimate_emissions_varied() {
    let pops = vec![make_pop("A", &["a1", "a2"]), make_pop("B", &["b1", "b2"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let observations: Vec<AncestryObservation> = (0..30).map(|i| {
        if i < 15 {
            make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
                vec![("a1", 0.95), ("a2", 0.93), ("b1", 0.80), ("b2", 0.79)])
        } else {
            make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
                vec![("a1", 0.80), ("a2", 0.78), ("b1", 0.95), ("b2", 0.94)])
        }
    }).collect();
    params.estimate_emissions(&observations);
}

// === posterior_decode vs viterbi consistency ===

/// For well-separated data, posterior_decode and viterbi should agree.
#[test]
fn test_posterior_viterbi_agreement_well_separated() {
    let pops = vec![
        make_pop("A", &["a1", "a2"]),
        make_pop("B", &["b1", "b2"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    // Clear separation: first half strongly A, second half strongly B
    let mut observations = Vec::new();
    for i in 0..10 {
        observations.push(make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
            vec![("a1", 0.99), ("a2", 0.98), ("b1", 0.70), ("b2", 0.69)]));
    }
    for i in 10..20 {
        observations.push(make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
            vec![("a1", 0.70), ("a2", 0.69), ("b1", 0.99), ("b2", 0.98)]));
    }

    let vit_states = viterbi(&observations, &params);
    let post_states = posterior_decode(&observations, &params);

    assert_eq!(vit_states.len(), 20);
    assert_eq!(post_states.len(), 20);

    // First half should be state 0 (A)
    for i in 0..8 {
        assert_eq!(vit_states[i], 0, "Viterbi window {} should be A(0)", i);
        assert_eq!(post_states[i], 0, "Posterior window {} should be A(0)", i);
    }
    // Second half should be state 1 (B)
    for i in 12..20 {
        assert_eq!(vit_states[i], 1, "Viterbi window {} should be B(1)", i);
        assert_eq!(post_states[i], 1, "Posterior window {} should be B(1)", i);
    }
}

/// forward_backward posteriors should sum to approximately 1.0 at each window.
#[test]
fn test_forward_backward_posteriors_sum_to_one() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.01);

    let observations: Vec<AncestryObservation> = (0..10).map(|i| {
        make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
            vec![("a1", 0.9), ("b1", 0.85), ("c1", 0.80)])
    }).collect();

    let posteriors = forward_backward(&observations, &params);
    assert_eq!(posteriors.len(), 10);
    for (i, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 3);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "Window {} posteriors sum to {} instead of 1.0", i, sum);
        for &p in post {
            assert!(p >= 0.0 && p <= 1.0 + 1e-10,
                "Window {} has invalid posterior {}", i, p);
        }
    }
}

// === estimate_admixture_proportions edge cases ===

/// Empty segments should return zero proportions.
#[test]
fn test_estimate_admixture_empty_segments() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let proportions = estimate_admixture_proportions(&[], "sample1", &pop_names);
    assert_eq!(proportions.proportions.len(), 2);
}

/// All segments same ancestry should give ~100% for that population.
#[test]
fn test_estimate_admixture_all_same_ancestry() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];
    let segments = vec![
        make_segment("chr1", 0, 100000, "s1", 1, "AFR", 10),
        make_segment("chr1", 100000, 200000, "s1", 1, "AFR", 10),
    ];
    let proportions = estimate_admixture_proportions(&segments, "s1", &pop_names);
    let afr_prop = *proportions.proportions.get("AFR").unwrap_or(&0.0);
    assert!((afr_prop - 1.0).abs() < 1e-10, "AFR should be 100%, got {}", afr_prop);
}

/// Mixed segments should give proportional admixture.
#[test]
fn test_estimate_admixture_mixed() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let segments = vec![
        make_segment("chr1", 0, 60000, "s1", 0, "EUR", 6),
        make_segment("chr1", 60000, 100000, "s1", 1, "AFR", 4),
    ];
    let proportions = estimate_admixture_proportions(&segments, "s1", &pop_names);
    let eur_prop = *proportions.proportions.get("EUR").unwrap_or(&0.0);
    let afr_prop = *proportions.proportions.get("AFR").unwrap_or(&0.0);
    assert!(eur_prop > afr_prop, "EUR ({}) should have higher proportion than AFR ({})", eur_prop, afr_prop);
    let total: f64 = proportions.proportions.values().sum();
    assert!((total - 1.0).abs() < 1e-6, "Proportions should sum to 1.0, got {}", total);
}

// === viterbi and forward_backward with empty observations ===

#[test]
fn test_viterbi_empty_observations() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let states = viterbi(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn test_forward_backward_empty_observations() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let posteriors = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
}

/// Viterbi with single observation should return a single state.
#[test]
fn test_viterbi_single_observation() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![make_obs("chr1", 0, 10000, "q", vec![("a1", 0.95), ("b1", 0.80)])];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 2);
}

/// Baum-Welch with very few observations should not panic.
#[test]
fn test_ancestry_baum_welch_few_obs() {
    let pops = vec![make_pop("A", &["a1", "a2"]), make_pop("B", &["b1", "b2"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![
        make_obs("chr1", 0, 10000, "q", vec![("a1", 0.95), ("a2", 0.94), ("b1", 0.80), ("b2", 0.79)]),
        make_obs("chr1", 10000, 20000, "q", vec![("a1", 0.93), ("a2", 0.92), ("b1", 0.81), ("b2", 0.80)]),
    ];
    let obs_slice: &[AncestryObservation] = &obs;
    params.baum_welch(&[obs_slice], 5, 1e-6);
    // Should not panic; parameters may or may not update
}

/// Baum-Welch with empty observations list should return NEG_INFINITY.
#[test]
fn test_ancestry_baum_welch_empty() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let ll = params.baum_welch(&[], 10, 1e-6);
    assert!(ll == f64::NEG_INFINITY);
}

/// Baum-Welch should improve log-likelihood over iterations for reasonable data.
#[test]
fn test_ancestry_baum_welch_improves_ll() {
    let pops = vec![make_pop("A", &["a1", "a2"]), make_pop("B", &["b1", "b2"])];
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let mut obs = Vec::new();
    for i in 0..30 {
        if i < 15 {
            obs.push(make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
                vec![("a1", 0.95), ("a2", 0.93), ("b1", 0.80), ("b2", 0.79)]));
        } else {
            obs.push(make_obs("chr1", i * 10000, (i + 1) * 10000 - 1, "q",
                vec![("a1", 0.80), ("a2", 0.78), ("b1", 0.95), ("b2", 0.94)]));
        }
    }

    let obs_slice: &[AncestryObservation] = &obs;

    // Run 1 iteration
    let mut params_1iter = params.clone();
    let ll_1 = params_1iter.baum_welch(&[obs_slice], 1, 1e-6);

    // Run 10 iterations
    let ll_10 = params.baum_welch(&[obs_slice], 10, 1e-6);

    assert!(ll_10.is_finite());
    assert!(ll_1.is_finite());
    // More iterations should yield at least as good log-likelihood
    assert!(ll_10 >= ll_1 - 1e-6,
        "10-iter LL {} should be >= 1-iter LL {}", ll_10, ll_1);
}
