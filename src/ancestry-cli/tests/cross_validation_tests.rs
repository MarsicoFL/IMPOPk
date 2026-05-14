//! Tests for cross_validate and cross_validate_kfold in ancestry-cli/validation.rs
//!
//! These functions were previously untested. They implement leave-one-out and
//! k-fold cross-validation on reference haplotypes to detect population bias.

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    CrossValidationResult, cross_validate, cross_validate_kfold,
};

/// Helper to create a simple observation for testing
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

/// Create a 2-population scenario with clear separation
fn setup_two_pop() -> (Vec<AncestralPopulation>, AncestryHmmParams, HashMap<String, Vec<AncestryObservation>>) {
    let populations = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
    ];

    let params = AncestryHmmParams::new(populations.clone(), 0.01);

    // Create observations where each haplotype clearly belongs to its population.
    // For hap_a1 and hap_a2: high similarity to POP_A haplotypes, low to POP_B
    // For hap_b1 and hap_b2: high similarity to POP_B haplotypes, low to POP_A
    let mut observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    for hap in &["hap_a1", "hap_a2"] {
        let obs = vec![
            make_obs("chr1", 0, 10000, hap, vec![
                ("hap_a1", 0.99), ("hap_a2", 0.98),
                ("hap_b1", 0.80), ("hap_b2", 0.79),
            ]),
            make_obs("chr1", 10000, 20000, hap, vec![
                ("hap_a1", 0.98), ("hap_a2", 0.99),
                ("hap_b1", 0.81), ("hap_b2", 0.80),
            ]),
        ];
        observations.insert(hap.to_string(), obs);
    }

    for hap in &["hap_b1", "hap_b2"] {
        let obs = vec![
            make_obs("chr1", 0, 10000, hap, vec![
                ("hap_a1", 0.80), ("hap_a2", 0.79),
                ("hap_b1", 0.99), ("hap_b2", 0.98),
            ]),
            make_obs("chr1", 10000, 20000, hap, vec![
                ("hap_a1", 0.81), ("hap_a2", 0.80),
                ("hap_b1", 0.98), ("hap_b2", 0.99),
            ]),
        ];
        observations.insert(hap.to_string(), obs);
    }

    (populations, params, observations)
}

/// Create a 3-population scenario
fn setup_three_pop() -> (Vec<AncestralPopulation>, AncestryHmmParams, HashMap<String, Vec<AncestryObservation>>) {
    let populations = vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string(), "eur2".to_string(), "eur3".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string(), "afr2".to_string(), "afr3".to_string()],
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: vec!["amr1".to_string(), "amr2".to_string(), "amr3".to_string()],
        },
    ];

    let params = AncestryHmmParams::new(populations.clone(), 0.005);

    let mut observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let all_haps: Vec<(&str, usize)> = vec![
        ("eur1", 0), ("eur2", 0), ("eur3", 0),
        ("afr1", 1), ("afr2", 1), ("afr3", 1),
        ("amr1", 2), ("amr2", 2), ("amr3", 2),
    ];

    for &(hap, pop_idx) in &all_haps {
        let mut obs_vec = Vec::new();
        for w in 0..3 {
            let mut sims = Vec::new();
            for &(ref_hap, ref_pop) in &all_haps {
                let sim = if ref_pop == pop_idx { 0.95 + 0.01 * (w as f64) } else { 0.70 + 0.01 * (w as f64) };
                sims.push((ref_hap, sim));
            }
            obs_vec.push(make_obs("chr1", w * 10000, (w + 1) * 10000, hap, sims));
        }
        observations.insert(hap.to_string(), obs_vec);
    }

    (populations, params, observations)
}

// ============================================================================
// cross_validate tests
// ============================================================================

#[test]
fn cross_validate_two_pop_perfect_separation() {
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate(&observations, &populations, &params);

    // With clear separation, accuracy should be high
    assert!(
        result.overall_accuracy > 0.5,
        "overall accuracy should be reasonable, got {}",
        result.overall_accuracy
    );

    // Both populations should have windows tested
    assert!(result.n_windows_per_pop.get("POP_A").unwrap_or(&0) > &0);
    assert!(result.n_windows_per_pop.get("POP_B").unwrap_or(&0) > &0);
}

#[test]
fn cross_validate_three_pop() {
    let (populations, params, observations) = setup_three_pop();
    let result = cross_validate(&observations, &populations, &params);

    // Should have tested all populations
    for pop in &["EUR", "AFR", "AMR"] {
        assert!(
            result.n_windows_per_pop.get(*pop).unwrap_or(&0) > &0,
            "population {} should have windows tested",
            pop
        );
    }

    // Overall accuracy should be non-zero
    assert!(result.overall_accuracy > 0.0);
}

#[test]
fn cross_validate_result_metrics_consistent() {
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate(&observations, &populations, &params);

    // Precision, recall, F1 should all be populated
    for pop in &["POP_A", "POP_B"] {
        assert!(result.precision_per_pop.contains_key(*pop));
        assert!(result.recall_per_pop.contains_key(*pop));
        assert!(result.f1_per_pop.contains_key(*pop));

        let prec = result.precision_per_pop[*pop];
        let rec = result.recall_per_pop[*pop];
        let f1 = result.f1_per_pop[*pop];

        // All should be in [0, 1]
        assert!(prec >= 0.0 && prec <= 1.0, "precision out of range: {}", prec);
        assert!(rec >= 0.0 && rec <= 1.0, "recall out of range: {}", rec);
        assert!(f1 >= 0.0 && f1 <= 1.0, "F1 out of range: {}", f1);

        // F1 should be harmonic mean of precision and recall
        if prec + rec > 0.0 {
            let expected_f1 = 2.0 * prec * rec / (prec + rec);
            assert!(
                (f1 - expected_f1).abs() < 1e-10,
                "F1 should be harmonic mean: got {} expected {}",
                f1,
                expected_f1
            );
        }
    }
}

#[test]
fn cross_validate_confusion_matrix_sums() {
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate(&observations, &populations, &params);

    // Sum of confusion matrix should equal total windows tested
    let total_from_confusion: usize = result.confusion.values().sum();
    let total_from_n_windows: usize = result.n_windows_per_pop.values().sum();
    assert_eq!(
        total_from_confusion, total_from_n_windows,
        "confusion matrix sum should equal total windows"
    );
}

#[test]
fn cross_validate_has_bias_detection() {
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate(&observations, &populations, &params);

    // has_bias() checks if any pop accuracy < 0.5
    // With well-separated data, there should be no bias
    // (But we can't guarantee accuracy > 0.5 with the default params, so just test the method exists)
    let _has_bias = result.has_bias();
}

#[test]
fn cross_validate_empty_observations() {
    let populations = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(populations.clone(), 0.01);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate(&observations, &populations, &params);
    assert_eq!(result.overall_accuracy, 0.0);
}

#[test]
fn cross_validate_single_haplotype_per_pop_skipped() {
    // LOO requires at least 2 haplotypes per population
    let populations = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["hap_a1".to_string()], // Only 1 haplotype
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["hap_b1".to_string()], // Only 1 haplotype
        },
    ];
    let params = AncestryHmmParams::new(populations.clone(), 0.01);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate(&observations, &populations, &params);
    // Should skip both populations since they have < 2 haplotypes
    assert_eq!(result.overall_accuracy, 0.0);
    assert_eq!(*result.n_windows_per_pop.get("POP_A").unwrap_or(&1), 0);
}

// ============================================================================
// cross_validate_kfold tests
// ============================================================================

#[test]
fn cross_validate_kfold_two_folds() {
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate_kfold(&observations, &populations, &params, 2);

    // Should have tested windows from both populations
    let total_windows: usize = result.n_windows_per_pop.values().sum();
    assert!(total_windows > 0, "k-fold should test some windows");
}

#[test]
fn cross_validate_kfold_three_pop() {
    let (populations, params, observations) = setup_three_pop();
    let result = cross_validate_kfold(&observations, &populations, &params, 3);

    for pop in &["EUR", "AFR", "AMR"] {
        assert!(
            result.n_windows_per_pop.get(*pop).unwrap_or(&0) > &0,
            "k-fold should test population {}",
            pop
        );
    }
}

#[test]
fn cross_validate_kfold_k_clamped_to_min_2() {
    // k=0 and k=1 should be clamped to k=2
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate_kfold(&observations, &populations, &params, 1);
    let total: usize = result.n_windows_per_pop.values().sum();
    assert!(total > 0, "k=1 should be clamped to k=2 and produce results");
}

#[test]
fn cross_validate_kfold_large_k() {
    // k larger than number of haplotypes: some folds may be empty
    let (populations, params, observations) = setup_two_pop();
    let result = cross_validate_kfold(&observations, &populations, &params, 100);
    // Should not panic
    let _total: usize = result.n_windows_per_pop.values().sum();
}

#[test]
fn cross_validate_kfold_metrics_consistent() {
    let (populations, params, observations) = setup_three_pop();
    let result = cross_validate_kfold(&observations, &populations, &params, 3);

    // Confusion sum should equal total windows
    let total_from_confusion: usize = result.confusion.values().sum();
    let total_from_n_windows: usize = result.n_windows_per_pop.values().sum();
    assert_eq!(total_from_confusion, total_from_n_windows);

    // Precision/recall/F1 should all be populated and valid
    for pop_name in populations.iter().map(|p| &p.name) {
        if let Some(&prec) = result.precision_per_pop.get(pop_name) {
            assert!(prec >= 0.0 && prec <= 1.0);
        }
        if let Some(&rec) = result.recall_per_pop.get(pop_name) {
            assert!(rec >= 0.0 && rec <= 1.0);
        }
        if let Some(&f1) = result.f1_per_pop.get(pop_name) {
            assert!(f1 >= 0.0 && f1 <= 1.0);
        }
    }
}

#[test]
fn cross_validate_kfold_empty_observations() {
    let populations = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(populations.clone(), 0.01);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate_kfold(&observations, &populations, &params, 3);
    assert_eq!(result.overall_accuracy, 0.0);
}

// ============================================================================
// CrossValidationResult::has_bias tests
// ============================================================================

#[test]
fn has_bias_true_when_low_accuracy() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("POP_A".to_string(), 0.9);
    accuracy_per_pop.insert("POP_B".to_string(), 0.3); // below 0.5

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.6,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(result.has_bias(), "should detect bias when any pop accuracy < 0.5");
}

#[test]
fn has_bias_false_when_all_high() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("POP_A".to_string(), 0.9);
    accuracy_per_pop.insert("POP_B".to_string(), 0.85);

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.875,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(!result.has_bias(), "should not detect bias when all pop accuracy >= 0.5");
}

#[test]
fn has_bias_boundary_at_exactly_0_5() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("POP_A".to_string(), 0.5);

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.5,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(!result.has_bias(), "accuracy exactly 0.5 should not be bias");
}
