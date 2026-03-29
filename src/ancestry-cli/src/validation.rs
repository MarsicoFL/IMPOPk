//! Cross-validation for ancestry HMM
//!
//! Implements leave-one-out and k-fold cross-validation using reference haplotypes
//! to detect potential population bias in the model.

use std::collections::HashMap;
use crate::hmm::{AncestryHmmParams, AncestryObservation, AncestralPopulation, viterbi};

/// Results from cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Accuracy for each population (fraction correctly classified)
    pub accuracy_per_pop: HashMap<String, f64>,
    /// Overall accuracy across all populations
    pub overall_accuracy: f64,
    /// Number of windows tested per population
    pub n_windows_per_pop: HashMap<String, usize>,
    /// Confusion counts: (true_pop, predicted_pop) -> count
    pub confusion: HashMap<(String, String), usize>,
    /// Per-population precision: TP / (TP + FP)
    pub precision_per_pop: HashMap<String, f64>,
    /// Per-population recall: TP / (TP + FN)
    pub recall_per_pop: HashMap<String, f64>,
    /// Per-population F1 score: 2 * precision * recall / (precision + recall)
    pub f1_per_pop: HashMap<String, f64>,
}

impl CrossValidationResult {
    /// Compute precision, recall, F1 from confusion matrix
    fn compute_metrics(
        confusion: &HashMap<(String, String), usize>,
        pop_names: &[String],
    ) -> (HashMap<String, f64>, HashMap<String, f64>, HashMap<String, f64>) {
        let mut precision = HashMap::new();
        let mut recall = HashMap::new();
        let mut f1 = HashMap::new();

        for pop in pop_names {
            // True positives: predicted as pop AND truly pop
            let tp = *confusion.get(&(pop.clone(), pop.clone())).unwrap_or(&0) as f64;

            // False positives: predicted as pop but truly something else
            let fp: f64 = pop_names.iter()
                .filter(|true_pop| *true_pop != pop)
                .map(|true_pop| *confusion.get(&(true_pop.clone(), pop.clone())).unwrap_or(&0) as f64)
                .sum();

            // False negatives: truly pop but predicted as something else
            let fn_: f64 = pop_names.iter()
                .filter(|pred_pop| *pred_pop != pop)
                .map(|pred_pop| *confusion.get(&(pop.clone(), pred_pop.clone())).unwrap_or(&0) as f64)
                .sum();

            let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let rec = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1_score = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };

            precision.insert(pop.clone(), prec);
            recall.insert(pop.clone(), rec);
            f1.insert(pop.clone(), f1_score);
        }

        (precision, recall, f1)
    }

    /// Print a summary of cross-validation results
    pub fn print_summary(&self) {
        eprintln!("\n=== Cross-Validation Results ===");
        eprintln!("Overall accuracy: {:.1}%", self.overall_accuracy * 100.0);

        // Sorted population names for consistent output
        let mut pops: Vec<_> = self.accuracy_per_pop.keys().cloned().collect();
        pops.sort();

        eprintln!("\nPer-population metrics:");
        eprintln!("  {:>15}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}",
            "Population", "Accuracy", "Precision", "Recall", "F1", "N");
        for pop in &pops {
            let acc = self.accuracy_per_pop.get(pop).unwrap_or(&0.0);
            let prec = self.precision_per_pop.get(pop).unwrap_or(&0.0);
            let rec = self.recall_per_pop.get(pop).unwrap_or(&0.0);
            let f1 = self.f1_per_pop.get(pop).unwrap_or(&0.0);
            let n = self.n_windows_per_pop.get(pop).unwrap_or(&0);
            eprintln!("  {:>15}  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>7.3}  {:>6}",
                pop, acc * 100.0, prec * 100.0, rec * 100.0, f1, n);
        }

        // Print confusion matrix
        eprintln!("\nConfusion matrix (rows=true, cols=predicted):");

        // Header
        eprint!("            ");
        for p in &pops {
            eprint!("{:>12}", p);
        }
        eprintln!();

        // Rows
        for true_pop in &pops {
            eprint!("{:>12}", true_pop);
            for pred_pop in &pops {
                let count = self.confusion.get(&(true_pop.clone(), pred_pop.clone())).unwrap_or(&0);
                eprint!("{:>12}", count);
            }
            eprintln!();
        }
    }

    /// Format confusion matrix as TSV string
    pub fn confusion_matrix_tsv(&self) -> String {
        let mut pops: Vec<_> = self.accuracy_per_pop.keys().cloned().collect();
        pops.sort();

        let mut out = String::new();

        // Header
        out.push_str("true_pop\tpred_pop\tcount\n");

        for true_pop in &pops {
            for pred_pop in &pops {
                let count = self.confusion.get(&(true_pop.clone(), pred_pop.clone())).unwrap_or(&0);
                out.push_str(&format!("{}\t{}\t{}\n", true_pop, pred_pop, count));
            }
        }

        out
    }

    /// Check if there's significant bias (any population < 50% accuracy)
    pub fn has_bias(&self) -> bool {
        self.accuracy_per_pop.values().any(|&acc| acc < 0.5)
    }
}

/// Perform leave-one-out cross-validation on reference haplotypes.
///
/// For each population with 2+ haplotypes:
/// 1. Use one haplotype as "query" (pretend it's a test sample)
/// 2. Use the other haplotype(s) as reference for that population
/// 3. Run the HMM and check if it correctly assigns to the true population
///
/// This helps detect if the model is biased towards certain populations.
pub fn cross_validate(
    observations: &HashMap<String, Vec<AncestryObservation>>,
    populations: &[AncestralPopulation],
    params: &AncestryHmmParams,
) -> CrossValidationResult {
    let mut correct_per_pop: HashMap<String, usize> = HashMap::new();
    let mut total_per_pop: HashMap<String, usize> = HashMap::new();
    let mut confusion: HashMap<(String, String), usize> = HashMap::new();

    // Initialize counters
    for pop in populations {
        correct_per_pop.insert(pop.name.clone(), 0);
        total_per_pop.insert(pop.name.clone(), 0);
    }

    // For each population, try using each of its haplotypes as query
    for (true_pop_idx, true_pop) in populations.iter().enumerate() {
        if true_pop.haplotypes.len() < 2 {
            continue; // Need at least 2 haplotypes for LOO
        }

        for test_hap in &true_pop.haplotypes {
            // Check if we have observations for this haplotype
            if let Some(obs) = observations.get(test_hap) {
                if obs.is_empty() {
                    continue;
                }

                // Run Viterbi (using original params - the other haplotype from same pop is still in references)
                let states = viterbi(obs, params);

                // Count correct assignments
                for &state in &states {
                    *total_per_pop.get_mut(&true_pop.name).unwrap() += 1;

                    let pred_pop = &populations[state].name;
                    *confusion.entry((true_pop.name.clone(), pred_pop.clone())).or_insert(0) += 1;

                    if state == true_pop_idx {
                        *correct_per_pop.get_mut(&true_pop.name).unwrap() += 1;
                    }
                }
            }
        }
    }

    // Calculate accuracies
    let mut accuracy_per_pop = HashMap::new();
    let mut total_correct = 0usize;
    let mut total_windows = 0usize;

    let pop_names: Vec<String> = populations.iter().map(|p| p.name.clone()).collect();

    for pop in populations {
        let correct = *correct_per_pop.get(&pop.name).unwrap_or(&0);
        let total = *total_per_pop.get(&pop.name).unwrap_or(&0);

        let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        accuracy_per_pop.insert(pop.name.clone(), acc);

        total_correct += correct;
        total_windows += total;
    }

    let overall_accuracy = if total_windows > 0 {
        total_correct as f64 / total_windows as f64
    } else {
        0.0
    };

    let n_windows_per_pop: HashMap<String, usize> = total_per_pop;

    let (precision_per_pop, recall_per_pop, f1_per_pop) =
        CrossValidationResult::compute_metrics(&confusion, &pop_names);

    CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy,
        n_windows_per_pop,
        confusion,
        precision_per_pop,
        recall_per_pop,
        f1_per_pop,
    }
}

/// Perform stratified k-fold cross-validation on reference haplotypes.
///
/// Splits each population's haplotypes into k folds (maintaining population proportions),
/// then for each fold, uses that fold's haplotypes as queries and the rest as reference.
///
/// # Arguments
/// * `observations` - All observation data keyed by sample/haplotype ID
/// * `populations` - Population definitions with haplotype lists
/// * `params` - HMM parameters (switch_prob, temperature, etc.)
/// * `k` - Number of folds (default 5)
///
/// # Returns
/// Aggregated CrossValidationResult across all folds
pub fn cross_validate_kfold(
    observations: &HashMap<String, Vec<AncestryObservation>>,
    populations: &[AncestralPopulation],
    params: &AncestryHmmParams,
    k: usize,
) -> CrossValidationResult {
    let k = k.max(2); // At least 2 folds

    let mut total_confusion: HashMap<(String, String), usize> = HashMap::new();
    let mut total_per_pop: HashMap<String, usize> = HashMap::new();
    let mut correct_per_pop: HashMap<String, usize> = HashMap::new();

    for pop in populations {
        total_per_pop.insert(pop.name.clone(), 0);
        correct_per_pop.insert(pop.name.clone(), 0);
    }

    // For each fold
    for fold in 0..k {
        // Build fold-specific populations: for each population, assign haplotypes to
        // test or reference based on fold index
        let mut test_haplotypes: Vec<(usize, String)> = Vec::new(); // (true_pop_idx, hap_id)
        let mut fold_populations: Vec<AncestralPopulation> = Vec::new();

        for (pop_idx, pop) in populations.iter().enumerate() {
            let n_haps = pop.haplotypes.len();
            if n_haps < 2 {
                // Can't split, keep all in reference
                fold_populations.push(pop.clone());
                continue;
            }

            // Stratified split: assign haplotypes to folds round-robin
            let mut ref_haps = Vec::new();
            for (i, hap) in pop.haplotypes.iter().enumerate() {
                if i % k == fold {
                    test_haplotypes.push((pop_idx, hap.clone()));
                } else {
                    ref_haps.push(hap.clone());
                }
            }

            // If all haplotypes ended up in test, move one back to reference
            if ref_haps.is_empty() && !test_haplotypes.is_empty() {
                let last = test_haplotypes.iter()
                    .rposition(|(idx, _)| *idx == pop_idx)
                    .unwrap();
                let (_, hap) = test_haplotypes.remove(last);
                ref_haps.push(hap);
            }

            fold_populations.push(AncestralPopulation {
                name: pop.name.clone(),
                haplotypes: ref_haps,
            });
        }

        // Skip fold if no test haplotypes
        if test_haplotypes.is_empty() {
            continue;
        }

        // Build HMM params with fold-specific populations
        let mut fold_params = AncestryHmmParams::new(fold_populations, params.transitions[0][1]);
        fold_params.set_temperature(params.emission_std);

        // Test each held-out haplotype
        for (true_pop_idx, test_hap) in &test_haplotypes {
            if let Some(obs) = observations.get(test_hap) {
                if obs.is_empty() {
                    continue;
                }

                let states = viterbi(obs, &fold_params);

                for &state in &states {
                    let true_name = &populations[*true_pop_idx].name;
                    let pred_name = &populations[state].name;

                    *total_per_pop.get_mut(true_name).unwrap() += 1;
                    *total_confusion.entry((true_name.clone(), pred_name.clone())).or_insert(0) += 1;

                    if state == *true_pop_idx {
                        *correct_per_pop.get_mut(true_name).unwrap() += 1;
                    }
                }
            }
        }
    }

    // Calculate metrics
    let mut accuracy_per_pop = HashMap::new();
    let mut total_correct = 0usize;
    let mut total_windows = 0usize;

    let pop_names: Vec<String> = populations.iter().map(|p| p.name.clone()).collect();

    for pop in populations {
        let correct = *correct_per_pop.get(&pop.name).unwrap_or(&0);
        let total = *total_per_pop.get(&pop.name).unwrap_or(&0);

        let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        accuracy_per_pop.insert(pop.name.clone(), acc);

        total_correct += correct;
        total_windows += total;
    }

    let overall_accuracy = if total_windows > 0 {
        total_correct as f64 / total_windows as f64
    } else {
        0.0
    };

    let (precision_per_pop, recall_per_pop, f1_per_pop) =
        CrossValidationResult::compute_metrics(&total_confusion, &pop_names);

    CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy,
        n_windows_per_pop: total_per_pop,
        confusion: total_confusion,
        precision_per_pop,
        recall_per_pop,
        f1_per_pop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_populations() -> Vec<AncestralPopulation> {
        vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string(), "pop_a#2".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string(), "pop_b#2".to_string()],
            },
        ]
    }

    #[test]
    fn test_cross_validation_result_has_bias() {
        let pops = vec!["pop_a".to_string(), "pop_b".to_string()];
        let confusion = HashMap::new();
        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);

        let mut result = CrossValidationResult {
            accuracy_per_pop: HashMap::new(),
            overall_accuracy: 0.75,
            n_windows_per_pop: HashMap::new(),
            confusion,
            precision_per_pop: prec,
            recall_per_pop: rec,
            f1_per_pop: f1,
        };

        result.accuracy_per_pop.insert("pop_a".to_string(), 0.9);
        result.accuracy_per_pop.insert("pop_b".to_string(), 0.6);
        assert!(!result.has_bias()); // both >= 0.5

        result.accuracy_per_pop.insert("pop_b".to_string(), 0.4);
        assert!(result.has_bias()); // pop_b < 0.5
    }

    #[test]
    fn test_compute_metrics() {
        let mut confusion = HashMap::new();
        // pop_a: 8 correct, 2 predicted as pop_b
        confusion.insert(("pop_a".to_string(), "pop_a".to_string()), 8);
        confusion.insert(("pop_a".to_string(), "pop_b".to_string()), 2);
        // pop_b: 7 correct, 3 predicted as pop_a
        confusion.insert(("pop_b".to_string(), "pop_b".to_string()), 7);
        confusion.insert(("pop_b".to_string(), "pop_a".to_string()), 3);

        let pops = vec!["pop_a".to_string(), "pop_b".to_string()];
        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);

        // pop_a precision: TP=8, FP=3 (pop_b predicted as pop_a) -> 8/11 ≈ 0.727
        assert!((prec["pop_a"] - 8.0 / 11.0).abs() < 1e-6);
        // pop_a recall: TP=8, FN=2 (pop_a predicted as pop_b) -> 8/10 = 0.8
        assert!((rec["pop_a"] - 0.8).abs() < 1e-6);

        // pop_b precision: TP=7, FP=2 (pop_a predicted as pop_b) -> 7/9 ≈ 0.778
        assert!((prec["pop_b"] - 7.0 / 9.0).abs() < 1e-6);
        // pop_b recall: TP=7, FN=3 (pop_b predicted as pop_a) -> 7/10 = 0.7
        assert!((rec["pop_b"] - 0.7).abs() < 1e-6);

        // F1 scores
        for pop in &pops {
            let expected_f1 = 2.0 * prec[pop] * rec[pop] / (prec[pop] + rec[pop]);
            assert!((f1[pop] - expected_f1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_confusion_matrix_tsv() {
        let mut confusion = HashMap::new();
        confusion.insert(("pop_a".to_string(), "pop_a".to_string()), 10);
        confusion.insert(("pop_a".to_string(), "pop_b".to_string()), 2);
        confusion.insert(("pop_b".to_string(), "pop_a".to_string()), 1);
        confusion.insert(("pop_b".to_string(), "pop_b".to_string()), 9);

        let pops = vec!["pop_a".to_string(), "pop_b".to_string()];
        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);

        let result = CrossValidationResult {
            accuracy_per_pop: [("pop_a".to_string(), 0.833), ("pop_b".to_string(), 0.9)].into(),
            overall_accuracy: 0.864,
            n_windows_per_pop: [("pop_a".to_string(), 12), ("pop_b".to_string(), 10)].into(),
            confusion,
            precision_per_pop: prec,
            recall_per_pop: rec,
            f1_per_pop: f1,
        };

        let tsv = result.confusion_matrix_tsv();
        assert!(tsv.contains("true_pop\tpred_pop\tcount"));
        assert!(tsv.contains("pop_a\tpop_a\t10"));
        assert!(tsv.contains("pop_b\tpop_b\t9"));
    }

    #[test]
    fn test_cross_validate_loo_with_clear_signal() {
        // Test leave-one-out cross-validation directly
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);

        let mut observations = HashMap::new();

        // pop_a haplotypes: high sim to pop_a refs, low to pop_b
        observations.insert("pop_a#1".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_a#1".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.98), ("pop_a#2".to_string(), 0.96),
                    ("pop_b#1".to_string(), 0.80), ("pop_b#2".to_string(), 0.79),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_a#2".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_a#2".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.97), ("pop_a#2".to_string(), 0.95),
                    ("pop_b#1".to_string(), 0.81), ("pop_b#2".to_string(), 0.80),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_b#1".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#1".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.79), ("pop_a#2".to_string(), 0.80),
                    ("pop_b#1".to_string(), 0.97), ("pop_b#2".to_string(), 0.96),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_b#2".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#2".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.80), ("pop_a#2".to_string(), 0.81),
                    ("pop_b#1".to_string(), 0.96), ("pop_b#2".to_string(), 0.95),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);

        let result = cross_validate(&observations, &pops, &params);

        // With clear signal, LOO should achieve high accuracy
        assert!(result.overall_accuracy >= 0.5,
            "LOO accuracy should be >= 0.5 with clear signal, got {}", result.overall_accuracy);
        assert!(!result.has_bias());
        assert!(result.n_windows_per_pop.values().all(|&n| n > 0),
            "Each population should have some windows tested");
    }

    #[test]
    fn test_cross_validate_loo_no_observations() {
        // If observations map is empty, should handle gracefully
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);
        let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

        let result = cross_validate(&observations, &pops, &params);
        assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validate_loo_single_haplotype_pop() {
        // Population with only 1 haplotype should be skipped in LOO
        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()], // only 1 — will be skipped
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string(), "pop_b#2".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops.clone(), 0.001);

        let mut observations = HashMap::new();
        observations.insert("pop_b#1".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#1".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.80),
                    ("pop_b#1".to_string(), 0.97), ("pop_b#2".to_string(), 0.96),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_b#2".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#2".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.80),
                    ("pop_b#1".to_string(), 0.96), ("pop_b#2".to_string(), 0.95),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);

        let result = cross_validate(&observations, &pops, &params);
        // pop_a should have 0 windows tested (skipped)
        assert_eq!(*result.n_windows_per_pop.get("pop_a").unwrap_or(&0), 0);
        // pop_b should have windows tested
        assert!(*result.n_windows_per_pop.get("pop_b").unwrap_or(&0) > 0);
    }

    #[test]
    fn test_kfold_k_equals_1_clamped_to_2() {
        // k=1 should be clamped to k=2 (minimum)
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);
        let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

        // Should not panic — k=1 → k=2 internally
        let result = cross_validate_kfold(&observations, &pops, &params, 1);
        assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kfold_with_empty_observations() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);
        let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

        let result = cross_validate_kfold(&observations, &pops, &params, 3);
        assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
        assert!(result.confusion.is_empty());
    }

    #[test]
    fn test_compute_metrics_empty_confusion() {
        let confusion: HashMap<(String, String), usize> = HashMap::new();
        let pops = vec!["A".to_string(), "B".to_string()];
        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);
        // All should be 0 when no data
        assert!((prec["A"] - 0.0).abs() < 1e-10);
        assert!((rec["A"] - 0.0).abs() < 1e-10);
        assert!((f1["A"] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_metrics_perfect_classification() {
        let mut confusion = HashMap::new();
        confusion.insert(("A".to_string(), "A".to_string()), 50);
        confusion.insert(("B".to_string(), "B".to_string()), 30);
        let pops = vec!["A".to_string(), "B".to_string()];

        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);
        assert!((prec["A"] - 1.0).abs() < 1e-10);
        assert!((rec["A"] - 1.0).abs() < 1e-10);
        assert!((f1["A"] - 1.0).abs() < 1e-10);
        assert!((prec["B"] - 1.0).abs() < 1e-10);
        assert!((rec["B"] - 1.0).abs() < 1e-10);
        assert!((f1["B"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_has_bias_edge_cases() {
        let pops = vec!["A".to_string()];
        let confusion = HashMap::new();
        let (prec, rec, f1) = CrossValidationResult::compute_metrics(&confusion, &pops);

        let result = CrossValidationResult {
            accuracy_per_pop: [("A".to_string(), 0.5)].into(),
            overall_accuracy: 0.5,
            n_windows_per_pop: HashMap::new(),
            confusion,
            precision_per_pop: prec,
            recall_per_pop: rec,
            f1_per_pop: f1,
        };
        assert!(!result.has_bias()); // exactly 0.5 is not bias

        let result2 = CrossValidationResult {
            accuracy_per_pop: [("A".to_string(), 0.499)].into(),
            overall_accuracy: 0.499,
            n_windows_per_pop: HashMap::new(),
            confusion: HashMap::new(),
            precision_per_pop: HashMap::new(),
            recall_per_pop: HashMap::new(),
            f1_per_pop: HashMap::new(),
        };
        assert!(result2.has_bias()); // just below 0.5
    }

    #[test]
    fn test_kfold_with_clear_signal() {
        // Create populations with clear, distinguishable signal
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);

        // Create observations where pop_a haplotypes are clearly pop_a-like
        let mut observations = HashMap::new();

        // pop_a#1: high similarity to pop_a refs, low to pop_b
        observations.insert("pop_a#1".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_a#1".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.98), ("pop_a#2".to_string(), 0.96),
                    ("pop_b#1".to_string(), 0.80), ("pop_b#2".to_string(), 0.79),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_a#2".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_a#2".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.97), ("pop_a#2".to_string(), 0.95),
                    ("pop_b#1".to_string(), 0.81), ("pop_b#2".to_string(), 0.80),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        // pop_b#1, #2: high similarity to pop_b refs
        observations.insert("pop_b#1".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#1".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.79), ("pop_a#2".to_string(), 0.80),
                    ("pop_b#1".to_string(), 0.97), ("pop_b#2".to_string(), 0.96),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);
        observations.insert("pop_b#2".to_string(), vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "pop_b#2".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.80), ("pop_a#2".to_string(), 0.81),
                    ("pop_b#1".to_string(), 0.96), ("pop_b#2".to_string(), 0.95),
                ].into(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            },
        ]);

        let result = cross_validate_kfold(&observations, &pops, &params, 2);

        // With clear signal, accuracy should be high
        assert!(result.overall_accuracy >= 0.5,
            "Expected high accuracy with clear signal, got {}", result.overall_accuracy);
        assert!(!result.has_bias());

        // F1 scores should exist for both populations
        assert!(result.f1_per_pop.contains_key("pop_a"));
        assert!(result.f1_per_pop.contains_key("pop_b"));
    }
}
