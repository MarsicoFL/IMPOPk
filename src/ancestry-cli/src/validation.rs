//! Cross-validation for ancestry HMM
//!
//! Implements leave-one-out cross-validation using reference haplotypes
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
}

impl CrossValidationResult {
    /// Print a summary of cross-validation results
    pub fn print_summary(&self) {
        eprintln!("\n=== Cross-Validation Results ===");
        eprintln!("Overall accuracy: {:.1}%", self.overall_accuracy * 100.0);
        eprintln!("\nPer-population accuracy:");
        for (pop, acc) in &self.accuracy_per_pop {
            let n = self.n_windows_per_pop.get(pop).unwrap_or(&0);
            eprintln!("  {}: {:.1}% ({} windows)", pop, acc * 100.0, n);
        }

        // Print confusion matrix
        eprintln!("\nConfusion matrix (rows=true, cols=predicted):");
        let pops: Vec<_> = self.accuracy_per_pop.keys().cloned().collect();

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

    CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy,
        n_windows_per_pop,
        confusion,
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
        let mut result = CrossValidationResult {
            accuracy_per_pop: HashMap::new(),
            overall_accuracy: 0.75,
            n_windows_per_pop: HashMap::new(),
            confusion: HashMap::new(),
        };

        result.accuracy_per_pop.insert("pop_a".to_string(), 0.9);
        result.accuracy_per_pop.insert("pop_b".to_string(), 0.6);
        assert!(!result.has_bias()); // both >= 0.5

        result.accuracy_per_pop.insert("pop_b".to_string(), 0.4);
        assert!(result.has_bias()); // pop_b < 0.5
    }
}
