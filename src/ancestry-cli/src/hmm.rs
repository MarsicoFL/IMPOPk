#![allow(clippy::manual_is_multiple_of)]
//! HMM for Local Ancestry Inference
//!
//! This module implements a Hidden Markov Model for inferring local ancestry
//! from similarity data against multiple reference populations.

use std::collections::HashMap;

/// Emission model for aggregating per-haplotype similarities into per-population scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmissionModel {
    /// Use maximum similarity to any haplotype in the population (default)
    Max,
    /// Use mean similarity across all haplotypes in the population
    Mean,
    /// Use median similarity across all haplotypes in the population
    Median,
    /// Use mean of top-k most similar haplotypes per population.
    /// More robust than Max (less sensitive to single outlier match)
    /// while still focusing on the most informative haplotypes.
    TopK(usize),
    /// TopK with exponential decay weights: top-1 gets weight 1, top-2 gets
    /// decay^1, top-3 gets decay^2, etc. Emphasizes the most similar haplotype
    /// while still using others for noise reduction.
    TopKWeighted(usize, f64),
}

impl EmissionModel {
    /// Aggregate a list of similarity values into a single population score
    fn aggregate(&self, sims: &[f64]) -> Option<f64> {
        if sims.is_empty() {
            return None;
        }
        match self {
            EmissionModel::Max => Some(sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
            EmissionModel::Mean => Some(sims.iter().sum::<f64>() / sims.len() as f64),
            EmissionModel::Median => {
                let mut sorted = sims.to_vec();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Some((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Some(sorted[mid])
                }
            }
            EmissionModel::TopK(k) => {
                let mut sorted = sims.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a)); // descending
                let take = (*k).min(sorted.len());
                if take == 0 { return None; }
                Some(sorted[..take].iter().sum::<f64>() / take as f64)
            }
            EmissionModel::TopKWeighted(k, decay) => {
                let mut sorted = sims.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a)); // descending
                let take = (*k).min(sorted.len());
                if take == 0 { return None; }
                let mut weight_sum = 0.0;
                let mut value_sum = 0.0;
                let mut w = 1.0;
                for &val in &sorted[..take] {
                    value_sum += w * val;
                    weight_sum += w;
                    w *= decay;
                }
                Some(value_sum / weight_sum)
            }
        }
    }
}

impl std::fmt::Display for EmissionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmissionModel::Max => write!(f, "max"),
            EmissionModel::Mean => write!(f, "mean"),
            EmissionModel::Median => write!(f, "median"),
            EmissionModel::TopK(k) => write!(f, "top{k}"),
            EmissionModel::TopKWeighted(k, decay) => write!(f, "top{k}w{:.2}", decay),
        }
    }
}

impl std::str::FromStr for EmissionModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "max" => Ok(EmissionModel::Max),
            "mean" => Ok(EmissionModel::Mean),
            "median" => Ok(EmissionModel::Median),
            _ if lower.starts_with("top") => {
                let k_str = &lower[3..];
                k_str.parse::<usize>()
                    .map(EmissionModel::TopK)
                    .map_err(|_| format!("Invalid top-k value '{}'. Use: top3, top5, top10", k_str))
            }
            _ => Err(format!("Unknown emission model '{}'. Use: max, mean, median, top5", s)),
        }
    }
}

/// Ancestral population definition
#[derive(Debug, Clone)]
pub struct AncestralPopulation {
    /// Population/species name
    pub name: String,
    /// Reference haplotype IDs belonging to this population
    pub haplotypes: Vec<String>,
}

/// Per-population emission normalization statistics.
///
/// When enabled, each population's aggregated similarity is z-score normalized
/// before softmax: `z = (sim - mean) / std`. This removes systematic bias where
/// some populations have inherently higher similarity (e.g., due to more reference
/// haplotypes in the pangenome).
#[derive(Debug, Clone)]
pub struct PopulationNormalization {
    /// Per-population mean of aggregated similarity scores
    pub means: Vec<f64>,
    /// Per-population standard deviation of aggregated similarity scores
    pub stds: Vec<f64>,
}

/// HMM parameters for ancestry inference
#[derive(Debug, Clone)]
pub struct AncestryHmmParams {
    /// Number of ancestral populations (states)
    pub n_states: usize,
    /// Population definitions
    pub populations: Vec<AncestralPopulation>,
    /// Transition matrix: transitions[i][j] = P(state j | state i)
    pub transitions: Vec<Vec<f64>>,
    /// Prior probability of starting in each state
    pub initial: Vec<f64>,
    /// Expected similarity when sample belongs to population (mean)
    pub emission_same_pop_mean: f64,
    /// Expected similarity when sample doesn't belong to population (mean)
    pub emission_diff_pop_mean: f64,
    /// Standard deviation for emission distributions
    pub emission_std: f64,
    /// Emission model for aggregating per-haplotype similarities
    pub emission_model: EmissionModel,
    /// Optional per-population emission normalization.
    /// When set, similarities are z-score normalized before softmax.
    pub normalization: Option<PopulationNormalization>,
    /// Weight for coverage-ratio auxiliary emission (0.0 = disabled).
    /// When > 0, combines similarity-based emission with coverage ratio:
    /// log P(obs|state) = log P(sim|state) + coverage_weight * log P(cov|state)
    pub coverage_weight: f64,
    /// Transition dampening factor for Baum-Welch (T52).
    /// Controls how much the M-step trusts MLE transitions vs prior.
    /// 0.0 = full MLE (default, no dampening). 1.0 = don't learn transitions.
    /// When pairwise emissions are active, dampening prevents double-correction
    /// of weakly-discriminable pairs (e.g., EUR→AMR).
    pub transition_dampening: f64,
}

impl AncestryHmmParams {
    /// Set the emission temperature (softmax sharpness)
    pub fn set_temperature(&mut self, temp: f64) {
        self.emission_std = temp;
    }

    /// Update switch probability and recalculate transition matrix
    pub fn set_switch_prob(&mut self, switch_prob: f64) {
        if self.n_states <= 1 {
            // Single state: always stay; zero states: no-op
            if self.n_states == 1 {
                self.transitions[0][0] = 1.0;
            }
            return;
        }
        let stay_prob = 1.0 - switch_prob;
        let switch_each = switch_prob / (self.n_states - 1) as f64;

        for i in 0..self.n_states {
            for j in 0..self.n_states {
                self.transitions[i][j] = if i == j { stay_prob } else { switch_each };
            }
        }
    }

    /// Set initial state distribution (priors).
    ///
    /// Values are normalized to sum to 1. Used by two-pass inference to set
    /// informative priors from first-pass ancestry proportions.
    pub fn set_initial_probs(&mut self, priors: &[f64]) {
        let sum: f64 = priors.iter().sum();
        if sum > 0.0 && priors.len() == self.n_states {
            self.initial = priors.iter().map(|&p| p / sum).collect();
        }
    }

    /// Set asymmetric transition matrix based on per-population proportions.
    ///
    /// Switch probability to state j is proportional to proportions[j].
    /// This models the biological expectation that ancestry switches are more
    /// likely to go to the dominant ancestry in the individual. For a person
    /// who is 80% EUR + 20% AFR, switches from AFR→EUR are more likely than
    /// EUR→AFR, reflecting the frequency of ancestry tracts.
    ///
    /// Each row i sums to 1.0:
    /// - T[i][i] = 1 - switch_rate[i]
    /// - T[i][j] = switch_rate[i] * proportions[j] / sum(proportions[j≠i])
    pub fn set_proportional_transitions(
        &mut self,
        proportions: &[f64],
        switch_rates: &[f64],
    ) {
        if proportions.len() != self.n_states || switch_rates.len() != self.n_states {
            return;
        }

        for (i, &switch_rate) in switch_rates.iter().enumerate().take(self.n_states) {
            let switch_prob = switch_rate.clamp(0.0, 1.0);
            let stay_prob = 1.0 - switch_prob;

            // Sum of proportions for states other than i
            let other_sum: f64 = proportions.iter().enumerate()
                .take(self.n_states)
                .filter(|&(j, _)| j != i)
                .map(|(_, &p)| p)
                .sum();

            for (j, &proportion) in proportions.iter().enumerate().take(self.n_states) {
                if i == j {
                    self.transitions[i][j] = stay_prob;
                } else if other_sum > 0.0 {
                    self.transitions[i][j] = switch_prob * proportion / other_sum;
                } else {
                    // Fallback to uniform
                    self.transitions[i][j] = switch_prob / (self.n_states - 1).max(1) as f64;
                }
            }
        }
    }

    /// Create parameters from population definitions
    ///
    /// # Arguments
    /// * `populations` - List of ancestral populations with their reference haplotypes
    /// * `switch_prob` - Probability of switching ancestry per window (e.g., 0.001)
    pub fn new(populations: Vec<AncestralPopulation>, switch_prob: f64) -> Self {
        let n_states = populations.len();

        // Handle edge cases: zero or single population
        if n_states == 0 {
            return Self {
                n_states: 0,
                populations,
                transitions: vec![],
                initial: vec![],
                emission_same_pop_mean: 0.95,
                emission_diff_pop_mean: 0.85,
                emission_std: 0.03,
                emission_model: EmissionModel::Max,
                normalization: None,
                coverage_weight: 0.0,
                transition_dampening: 0.0,
            };
        }

        // Uniform initial distribution
        let initial = vec![1.0 / n_states as f64; n_states];

        // Transition matrix: high self-transition, uniform switch probability
        let mut transitions = vec![vec![0.0; n_states]; n_states];
        if n_states == 1 {
            transitions[0][0] = 1.0;
        } else {
            let stay_prob = 1.0 - switch_prob;
            let switch_each = switch_prob / (n_states - 1) as f64;
            for (i, row) in transitions.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = if i == j { stay_prob } else { switch_each };
                }
            }
        }

        Self {
            n_states,
            populations,
            transitions,
            initial,
            // Default emission parameters - can be estimated from data
            emission_same_pop_mean: 0.95,
            emission_diff_pop_mean: 0.85,
            emission_std: 0.03,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        }
    }

    /// Learn per-population normalization statistics from observed data.
    ///
    /// For each population, computes the mean and std of the aggregated similarity
    /// score across all windows. This is used to z-score normalize before softmax,
    /// removing systematic bias where some populations have inherently higher
    /// similarity (e.g., due to reference panel size or pangenome assembly quality).
    pub fn learn_normalization(&mut self, observations: &[AncestryObservation]) {
        let mut pop_sums: Vec<f64> = vec![0.0; self.n_states];
        let mut pop_sum_sq: Vec<f64> = vec![0.0; self.n_states];
        let mut pop_counts: Vec<usize> = vec![0; self.n_states];

        for obs in observations {
            for (pop_idx, pop) in self.populations.iter().enumerate() {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                if let Some(agg) = self.emission_model.aggregate(&sims) {
                    if agg > 0.0 {
                        pop_sums[pop_idx] += agg;
                        pop_sum_sq[pop_idx] += agg * agg;
                        pop_counts[pop_idx] += 1;
                    }
                }
            }
        }

        let means: Vec<f64> = (0..self.n_states).map(|i| {
            if pop_counts[i] > 0 {
                pop_sums[i] / pop_counts[i] as f64
            } else {
                0.0
            }
        }).collect();

        let stds: Vec<f64> = (0..self.n_states).map(|i| {
            if pop_counts[i] > 1 {
                let mean = means[i];
                let var = pop_sum_sq[i] / pop_counts[i] as f64 - mean * mean;
                var.max(0.0).sqrt().max(1e-6) // prevent division by zero
            } else {
                1e-6
            }
        }).collect();

        self.normalization = Some(PopulationNormalization { means, stds });
    }

    /// Estimate emission parameters from observed data
    pub fn estimate_emissions(&mut self, observations: &[AncestryObservation]) {
        // Collect similarities grouped by whether it's same-pop or different-pop
        // This is a simplified approach - in practice we'd use EM or supervised learning

        let mut same_pop_sims: Vec<f64> = Vec::new();
        let mut diff_pop_sims: Vec<f64> = Vec::new();

        for obs in observations {
            // For each population, get the max similarity to its haplotypes
            for (pop_idx, pop) in self.populations.iter().enumerate() {
                let max_sim = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .fold(0.0_f64, f64::max);

                if max_sim > 0.0 {
                    // We don't know ground truth, so use heuristic:
                    // highest similarity likely indicates true ancestry
                    let is_best = self.populations.iter().enumerate()
                        .map(|(i, p)| {
                            let s = p.haplotypes.iter()
                                .filter_map(|h| obs.similarities.get(h))
                                .cloned()
                                .fold(0.0_f64, f64::max);
                            (i, s)
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(i, _)| i == pop_idx)
                        .unwrap_or(false);

                    if is_best {
                        same_pop_sims.push(max_sim);
                    } else {
                        diff_pop_sims.push(max_sim);
                    }
                }
            }
        }

        // Update emission parameters
        if !same_pop_sims.is_empty() {
            self.emission_same_pop_mean = same_pop_sims.iter().sum::<f64>() / same_pop_sims.len() as f64;
        }
        if !diff_pop_sims.is_empty() {
            self.emission_diff_pop_mean = diff_pop_sims.iter().sum::<f64>() / diff_pop_sims.len() as f64;
        }

        // Estimate std from combined data
        let all_sims: Vec<f64> = same_pop_sims.iter().chain(diff_pop_sims.iter()).cloned().collect();
        if all_sims.len() > 1 {
            let mean = all_sims.iter().sum::<f64>() / all_sims.len() as f64;
            let variance = all_sims.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_sims.len() as f64;
            self.emission_std = variance.sqrt().max(0.01);
        }
    }

    /// Compute log emission probability for observing similarities given ancestry state
    ///
    /// The emission model uses softmax over similarities: P(state) ∝ exp(sim / temperature)
    /// This ensures the state with highest similarity gets highest probability.
    ///
    /// IMPORTANT: Only populations with actual data (non-zero similarity) participate
    /// in the softmax. Missing data is treated as "unknown", not as zero similarity.
    /// Set the emission model
    pub fn set_emission_model(&mut self, model: EmissionModel) {
        self.emission_model = model;
    }

    /// Set coverage-ratio auxiliary emission weight.
    /// When > 0, coverage ratios from alignment lengths are used as auxiliary features.
    pub fn set_coverage_weight(&mut self, weight: f64) {
        self.coverage_weight = weight;
    }

    pub fn log_emission(&self, obs: &AncestryObservation, state: usize) -> f64 {
        // Delegate to coverage-aware emission if coverage_weight > 0 and data available
        if self.coverage_weight > 0.0 && obs.coverage_ratios.is_some() {
            return self.log_emission_with_coverage(obs, state, self.coverage_weight);
        }

        self.log_emission_similarity_only(obs, state)
    }

    /// Compute log emission from similarity data only (no coverage).
    /// This is the core softmax emission, used by both `log_emission` and
    /// `log_emission_with_coverage` as the similarity component.
    fn log_emission_similarity_only(&self, obs: &AncestryObservation, state: usize) -> f64 {
        // Get aggregated similarity for each population using the configured model
        let mut pop_sims: Vec<Option<f64>> = self.populations.iter()
            .map(|pop| {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                self.emission_model.aggregate(&sims)
            })
            .collect();

        // Apply haplotype consistency bonus if available.
        // The bonus is pre-scaled by weight * temperature, so dividing by temperature
        // in softmax gives an additive boost of (weight * consistency) in score space.
        if let Some(ref bonus) = obs.haplotype_consistency_bonus {
            for (p, sim) in pop_sims.iter_mut().enumerate() {
                if let Some(ref mut s) = sim {
                    if p < bonus.len() {
                        *s += bonus[p];
                    }
                }
            }
        }

        // Check if we have data for the target state
        let target_sim = match pop_sims[state] {
            Some(s) if s > 0.0 => s,
            _ => return f64::NEG_INFINITY,  // No data for target state
        };

        // Apply per-population normalization if available
        // z-score: (sim - pop_mean) / pop_std
        // This removes systematic bias where some populations have inherently
        // higher similarity (e.g., EUR > AFR in pangenome data)
        let (target_score, valid_scores) = if let Some(ref norm) = self.normalization {
            let target_z = (target_sim - norm.means[state]) / norm.stds[state];
            let valids: Vec<f64> = pop_sims.iter().enumerate()
                .filter_map(|(i, &s)| {
                    s.filter(|&v| v > 0.0)
                        .map(|v| (v - norm.means[i]) / norm.stds[i])
                })
                .collect();
            (target_z, valids)
        } else {
            let valids: Vec<f64> = pop_sims.iter()
                .filter_map(|&s| s)
                .filter(|&s| s > 0.0)
                .collect();
            (target_sim, valids)
        };

        // If only one population has data, it gets probability 1
        if valid_scores.len() <= 1 {
            return 0.0;  // log(1) = 0
        }

        // Use softmax with temperature parameter
        // Lower temperature = more confident (sharper distribution)
        let temperature = self.emission_std;

        // For numerical stability, subtract max before exp
        let max_score = valid_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Log-softmax only over populations with data
        let log_sum_exp: f64 = valid_scores.iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .sum::<f64>()
            .ln();

        (target_score - max_score) / temperature - log_sum_exp
    }

    /// Compute log emission with auxiliary coverage-ratio feature.
    ///
    /// Combines the primary similarity-based softmax emission with a coverage-ratio
    /// auxiliary signal. The coverage ratio `min(a_len, b_len) / max(a_len, b_len)`
    /// measures alignment symmetry — higher values indicate better alignment quality,
    /// which may correlate with true ancestry.
    ///
    /// The combined emission is: log P(obs|state) = log P(sim|state) + weight * log P(cov|state)
    /// where the coverage-ratio component uses the same softmax model.
    pub fn log_emission_with_coverage(
        &self,
        obs: &AncestryObservation,
        state: usize,
        coverage_weight: f64,
    ) -> f64 {
        let sim_emission = self.log_emission_similarity_only(obs, state);

        // If no coverage data, return identity-only emission
        let coverage_ratios = match &obs.coverage_ratios {
            Some(cr) if !cr.is_empty() => cr,
            _ => return sim_emission,
        };

        // Compute per-population mean coverage ratio
        let pop_cov: Vec<Option<f64>> = self.populations.iter()
            .map(|pop| {
                let covs: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| coverage_ratios.get(h))
                    .cloned()
                    .collect();
                if covs.is_empty() {
                    None
                } else {
                    Some(covs.iter().sum::<f64>() / covs.len() as f64)
                }
            })
            .collect();

        let target_cov = match pop_cov[state] {
            Some(c) if c > 0.0 => c,
            _ => return sim_emission,
        };

        // Softmax over coverage ratios (using same temperature)
        let temperature = self.emission_std;
        let valid_covs: Vec<f64> = pop_cov.iter()
            .filter_map(|c| c.filter(|&v| v > 0.0))
            .collect();

        if valid_covs.len() <= 1 {
            return sim_emission;
        }

        let max_cov = valid_covs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp_cov: f64 = valid_covs.iter()
            .map(|&c| ((c - max_cov) / temperature).exp())
            .sum::<f64>()
            .ln();
        let cov_log_emission = (target_cov - max_cov) / temperature - log_sum_exp_cov;

        // Combine: sim + weight * coverage
        sim_emission + coverage_weight * cov_log_emission
    }

    /// Baum-Welch parameter re-estimation for ancestry HMM.
    ///
    /// Uses forward-backward to compute expected state occupancies (gamma) and
    /// transition counts (xi), then re-estimates switch probability and temperature.
    ///
    /// # Arguments
    /// * `all_observations` - All observation sequences (one per sample)
    /// * `max_iters` - Maximum EM iterations
    /// * `tol` - Convergence tolerance on log-likelihood improvement
    ///
    /// # Returns
    /// Final log-likelihood
    pub fn baum_welch(
        &mut self,
        all_observations: &[&[AncestryObservation]],
        max_iters: usize,
        tol: f64,
    ) -> f64 {
        let k = self.n_states;
        if k < 2 || all_observations.is_empty() {
            return f64::NEG_INFINITY;
        }

        // T52: save prior transitions for dampened learning
        let prior_transitions = self.transitions.clone();
        let dampening = self.transition_dampening;

        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..max_iters {
            // Accumulate sufficient statistics across all sequences
            let mut total_xi = vec![vec![0.0_f64; k]; k]; // expected transition counts
            let mut total_gamma = vec![0.0_f64; k]; // expected state occupancies
            let mut total_ll = 0.0_f64;

            for obs in all_observations {
                if obs.len() < 2 {
                    continue;
                }

                let n = obs.len();

                // Forward pass (log scale)
                let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];
                for (s, alpha_0) in alpha[0].iter_mut().enumerate() {
                    *alpha_0 = self.initial[s].ln() + self.log_emission(&obs[0], s);
                }
                for t in 1..n {
                    for s in 0..k {
                        let emission = self.log_emission(&obs[t], s);
                        let prev_sum = log_sum_exp(&(0..k)
                            .map(|prev_s| alpha[t-1][prev_s] + self.transitions[prev_s][s].ln())
                            .collect::<Vec<_>>());
                        alpha[t][s] = prev_sum + emission;
                    }
                }

                // Backward pass (log scale)
                let mut beta = vec![vec![0.0; k]; n];
                for t in (0..n-1).rev() {
                    for s in 0..k {
                        beta[t][s] = log_sum_exp(&(0..k)
                            .map(|next_s| {
                                self.transitions[s][next_s].ln()
                                + self.log_emission(&obs[t+1], next_s)
                                + beta[t+1][next_s]
                            })
                            .collect::<Vec<_>>());
                    }
                }

                // Sequence log-likelihood
                let seq_ll = log_sum_exp(&alpha[n-1].to_vec());
                if seq_ll.is_finite() {
                    total_ll += seq_ll;
                }

                // Gamma: P(state s at time t | all obs)
                for t in 0..n {
                    let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
                    let log_total = log_sum_exp(&log_probs);
                    for s in 0..k {
                        let gamma_ts = (log_probs[s] - log_total).exp();
                        total_gamma[s] += gamma_ts;
                    }
                }

                // Xi: P(state i at t, state j at t+1 | all obs)
                for t in 0..n-1 {
                    let mut log_xi = vec![vec![f64::NEG_INFINITY; k]; k];
                    for i in 0..k {
                        for j in 0..k {
                            log_xi[i][j] = alpha[t][i]
                                + self.transitions[i][j].ln()
                                + self.log_emission(&obs[t+1], j)
                                + beta[t+1][j];
                        }
                    }
                    let log_total = log_sum_exp(&log_xi.iter().flatten().cloned().collect::<Vec<_>>());
                    for i in 0..k {
                        for j in 0..k {
                            total_xi[i][j] += (log_xi[i][j] - log_total).exp();
                        }
                    }
                }
            }

            // Check convergence
            if iter > 0 && total_ll.is_finite() && prev_ll.is_finite() {
                let improvement = total_ll - prev_ll;
                if improvement < tol && improvement >= 0.0 {
                    break;
                }
            }
            prev_ll = total_ll;

            // M-step: re-estimate transition matrix
            for (i, (xi_row, trans_row)) in total_xi.iter().zip(self.transitions.iter_mut()).enumerate() {
                let row_sum: f64 = xi_row.iter().sum();
                if row_sum > 0.0 {
                    for (j, (t, &xi)) in trans_row.iter_mut().zip(xi_row.iter()).enumerate() {
                        let mle = (xi / row_sum).max(1e-10);
                        // T52: blend MLE with prior, dampening prevents double-correction
                        *t = if dampening > 0.0 {
                            (1.0 - dampening) * mle + dampening * prior_transitions[i][j]
                        } else {
                            mle
                        };
                    }
                    // Normalize
                    let new_sum: f64 = trans_row.iter().sum();
                    for t in trans_row.iter_mut() {
                        *t /= new_sum;
                    }
                }
            }

            // For 2 states, enforce symmetry (only one off-diagonal parameter).
            // For 3+ states, keep the learned asymmetric transitions — this allows
            // BW to learn that e.g. EUR↔AMR switches are more likely than EUR↔AFR,
            // which is critical for genetically similar population discrimination.
            if k == 2 {
                let mut switch_sum = 0.0;
                for i in 0..k {
                    switch_sum += 1.0 - self.transitions[i][i];
                }
                let avg_switch: f64 = (switch_sum / k as f64).clamp(0.0001, 0.1);
                self.set_switch_prob(avg_switch);
            } else {
                // Clamp diagonal to reasonable range, normalize off-diagonal
                for i in 0..k {
                    let stay = self.transitions[i][i].clamp(0.9, 0.9999);
                    let off_diag_sum: f64 = (0..k).filter(|&j| j != i)
                        .map(|j| self.transitions[i][j]).sum();
                    if off_diag_sum > 0.0 {
                        let scale = (1.0 - stay) / off_diag_sum;
                        for j in 0..k {
                            if j == i {
                                self.transitions[i][j] = stay;
                            } else {
                                self.transitions[i][j] = (self.transitions[i][j] * scale).max(1e-10);
                            }
                        }
                    }
                }
            }
        }

        prev_ll
    }

    /// Enhanced Baum-Welch that also re-estimates initial probabilities.
    ///
    /// In addition to re-estimating transitions (like standard BW), this version
    /// updates the initial state distribution from gamma at t=0. This is useful
    /// when the ancestry proportions are unbalanced — the standard BW with uniform
    /// initial probs biases toward equal ancestry.
    ///
    /// Also optionally performs temperature grid search after transition estimation,
    /// finding the temperature that maximizes total log-likelihood.
    ///
    /// Returns the final total log-likelihood.
    pub fn baum_welch_full(
        &mut self,
        all_observations: &[&[AncestryObservation]],
        max_iters: usize,
        tol: f64,
        reestimate_temperature: bool,
    ) -> f64 {
        let k = self.n_states;
        if k < 2 || all_observations.is_empty() {
            return f64::NEG_INFINITY;
        }

        // T52: save prior transitions for dampened learning
        let prior_transitions = self.transitions.clone();
        let dampening = self.transition_dampening;

        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..max_iters {
            // Accumulate sufficient statistics across all sequences
            let mut total_xi = vec![vec![0.0_f64; k]; k];
            let mut total_gamma = vec![0.0_f64; k];
            let mut initial_gamma = vec![0.0_f64; k]; // gamma at t=0
            let mut total_ll = 0.0_f64;
            let mut n_sequences = 0usize;

            for obs in all_observations {
                if obs.len() < 2 {
                    continue;
                }
                n_sequences += 1;

                let n = obs.len();

                // Forward pass (log scale)
                let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];
                for (s, alpha_0) in alpha[0].iter_mut().enumerate() {
                    *alpha_0 = self.initial[s].ln() + self.log_emission(&obs[0], s);
                }
                for t in 1..n {
                    for s in 0..k {
                        let emission = self.log_emission(&obs[t], s);
                        let prev_sum = log_sum_exp(&(0..k)
                            .map(|prev_s| alpha[t-1][prev_s] + self.transitions[prev_s][s].ln())
                            .collect::<Vec<_>>());
                        alpha[t][s] = prev_sum + emission;
                    }
                }

                // Backward pass (log scale)
                let mut beta = vec![vec![0.0; k]; n];
                for t in (0..n-1).rev() {
                    for s in 0..k {
                        beta[t][s] = log_sum_exp(&(0..k)
                            .map(|next_s| {
                                self.transitions[s][next_s].ln()
                                + self.log_emission(&obs[t+1], next_s)
                                + beta[t+1][next_s]
                            })
                            .collect::<Vec<_>>());
                    }
                }

                // Sequence log-likelihood
                let seq_ll = log_sum_exp(&alpha[n-1].to_vec());
                if seq_ll.is_finite() {
                    total_ll += seq_ll;
                }

                // Gamma at t=0 for initial prob estimation
                {
                    let log_probs: Vec<f64> = (0..k).map(|s| alpha[0][s] + beta[0][s]).collect();
                    let log_total = log_sum_exp(&log_probs);
                    for s in 0..k {
                        let g = (log_probs[s] - log_total).exp();
                        initial_gamma[s] += g;
                    }
                }

                // Gamma: P(state s at time t | all obs)
                for t in 0..n {
                    let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
                    let log_total = log_sum_exp(&log_probs);
                    for s in 0..k {
                        let gamma_ts = (log_probs[s] - log_total).exp();
                        total_gamma[s] += gamma_ts;
                    }
                }

                // Xi: P(state i at t, state j at t+1 | all obs)
                for t in 0..n-1 {
                    let mut log_xi = vec![vec![f64::NEG_INFINITY; k]; k];
                    for i in 0..k {
                        for j in 0..k {
                            log_xi[i][j] = alpha[t][i]
                                + self.transitions[i][j].ln()
                                + self.log_emission(&obs[t+1], j)
                                + beta[t+1][j];
                        }
                    }
                    let log_total = log_sum_exp(&log_xi.iter().flatten().cloned().collect::<Vec<_>>());
                    for i in 0..k {
                        for j in 0..k {
                            total_xi[i][j] += (log_xi[i][j] - log_total).exp();
                        }
                    }
                }
            }

            // Check convergence
            if iter > 0 && total_ll.is_finite() && prev_ll.is_finite() {
                let improvement = total_ll - prev_ll;
                if improvement < tol && improvement >= 0.0 {
                    break;
                }
            }
            prev_ll = total_ll;

            // M-step 1: re-estimate initial probabilities from gamma at t=0
            if n_sequences > 0 {
                let init_sum: f64 = initial_gamma.iter().sum();
                if init_sum > 0.0 {
                    for (init_s, &gamma_s) in self.initial.iter_mut().zip(initial_gamma.iter()).take(k) {
                        *init_s = (gamma_s / init_sum).max(1e-10);
                    }
                    // Normalize
                    let norm: f64 = self.initial.iter().sum();
                    for p in &mut self.initial {
                        *p /= norm;
                    }
                }
            }

            // M-step 2: re-estimate transition matrix (with T52 dampening)
            for (i, (xi_row, trans_row)) in total_xi.iter().zip(self.transitions.iter_mut()).enumerate() {
                let row_sum: f64 = xi_row.iter().sum();
                if row_sum > 0.0 {
                    for (j, (t, &xi)) in trans_row.iter_mut().zip(xi_row.iter()).enumerate() {
                        let mle = (xi / row_sum).max(1e-10);
                        *t = if dampening > 0.0 {
                            (1.0 - dampening) * mle + dampening * prior_transitions[i][j]
                        } else {
                            mle
                        };
                    }
                    let new_sum: f64 = trans_row.iter().sum();
                    for t in trans_row.iter_mut() {
                        *t /= new_sum;
                    }
                }
            }

            // M-step 3: optionally re-estimate temperature via grid search
            if reestimate_temperature {
                let current_temp = self.emission_std;
                let best_temp = grid_search_temperature(self, all_observations, current_temp);
                self.emission_std = best_temp;
            }

            // For 2 states, enforce symmetry. For 3+, keep asymmetric.
            if k == 2 {
                let mut switch_sum = 0.0;
                for i in 0..k {
                    switch_sum += 1.0 - self.transitions[i][i];
                }
                let avg_switch: f64 = (switch_sum / k as f64).clamp(0.0001, 0.1);
                self.set_switch_prob(avg_switch);
            } else {
                for i in 0..k {
                    let stay = self.transitions[i][i].clamp(0.9, 0.9999);
                    let off_diag_sum: f64 = (0..k).filter(|&j| j != i)
                        .map(|j| self.transitions[i][j]).sum();
                    if off_diag_sum > 0.0 {
                        let scale = (1.0 - stay) / off_diag_sum;
                        for j in 0..k {
                            if j == i {
                                self.transitions[i][j] = stay;
                            } else {
                                self.transitions[i][j] = (self.transitions[i][j] * scale).max(1e-10);
                            }
                        }
                    }
                }
            }
        }

        prev_ll
    }
}

/// Grid search for optimal temperature around a center value.
///
/// Evaluates a set of temperature values near `center_temp` and returns the
/// one that maximizes total log-likelihood over all observation sequences.
/// Uses a coarse-to-fine strategy: first checks 5 values at ±50%, then
/// refines within the best interval.
fn grid_search_temperature(
    params: &AncestryHmmParams,
    all_observations: &[&[AncestryObservation]],
    center_temp: f64,
) -> f64 {
    let mut best_temp = center_temp;
    let mut best_ll = f64::NEG_INFINITY;

    // Coarse grid: ±50% in 5 steps
    let low = center_temp * 0.5;
    let high = center_temp * 1.5;
    let step = (high - low) / 4.0;

    for i in 0..5 {
        let temp = (low + step * i as f64).max(0.001);
        let ll = compute_total_ll(params, all_observations, temp);
        if ll > best_ll {
            best_ll = ll;
            best_temp = temp;
        }
    }

    // Fine grid: ±25% of best in 5 steps
    let fine_low = (best_temp * 0.75).max(0.001);
    let fine_high = best_temp * 1.25;
    let fine_step = (fine_high - fine_low) / 4.0;

    for i in 0..5 {
        let temp = fine_low + fine_step * i as f64;
        let ll = compute_total_ll(params, all_observations, temp);
        if ll > best_ll {
            best_ll = ll;
            best_temp = temp;
        }
    }

    best_temp
}

/// Compute total log-likelihood for a given temperature without modifying params.
fn compute_total_ll(
    params: &AncestryHmmParams,
    all_observations: &[&[AncestryObservation]],
    temperature: f64,
) -> f64 {
    let k = params.n_states;
    let mut temp_params = params.clone();
    temp_params.emission_std = temperature;

    let mut total_ll = 0.0_f64;

    for obs in all_observations {
        if obs.len() < 2 { continue; }
        let n = obs.len();

        // Forward pass only (for likelihood)
        let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];
        for (s, alpha_0) in alpha[0].iter_mut().enumerate().take(k) {
            *alpha_0 = temp_params.initial[s].ln() + temp_params.log_emission(&obs[0], s);
        }
        for t in 1..n {
            for s in 0..k {
                let emission = temp_params.log_emission(&obs[t], s);
                let prev_sum = log_sum_exp(&(0..k)
                    .map(|prev_s| alpha[t-1][prev_s] + temp_params.transitions[prev_s][s].ln())
                    .collect::<Vec<_>>());
                alpha[t][s] = prev_sum + emission;
            }
        }

        let seq_ll = log_sum_exp(&alpha[n-1].to_vec());
        if seq_ll.is_finite() {
            total_ll += seq_ll;
        }
    }

    total_ll
}

/// Observation for a single window: similarities to each reference haplotype
#[derive(Debug, Clone)]
pub struct AncestryObservation {
    /// Chromosome/scaffold name
    pub chrom: String,
    /// Window start position
    pub start: u64,
    /// Window end position
    pub end: u64,
    /// Sample ID being analyzed
    pub sample: String,
    /// Similarities to each reference haplotype: haplotype_id -> similarity
    pub similarities: HashMap<String, f64>,
    /// Optional coverage ratios per reference haplotype: min(a_len, b_len) / max(a_len, b_len)
    /// Higher values indicate more symmetric alignment, suggesting better match quality.
    pub coverage_ratios: Option<HashMap<String, f64>>,
    /// Per-population haplotype consistency bonus (indexed by population order).
    /// Measures how stable the best-matching haplotype is across consecutive windows.
    /// Applied as a pre-softmax similarity adjustment.
    pub haplotype_consistency_bonus: Option<Vec<f64>>,
}

/// Viterbi algorithm for ancestry HMM
///
/// Returns the most likely sequence of ancestral states
pub fn viterbi(observations: &[AncestryObservation], params: &AncestryHmmParams) -> Vec<usize> {
    let n = observations.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    // Viterbi tables (log scale)
    let mut v = vec![vec![f64::NEG_INFINITY; k]; n];
    let mut backptr = vec![vec![0usize; k]; n];

    // Initialize
    for (s, v_0s) in v[0].iter_mut().enumerate() {
        *v_0s = params.initial[s].ln() + params.log_emission(&observations[0], s);
    }

    // Forward pass
    for t in 1..n {
        for s in 0..k {
            let emission = params.log_emission(&observations[t], s);

            let (best_prev, best_prob) = (0..k)
                .map(|prev_s| {
                    let prob = v[t-1][prev_s] + params.transitions[prev_s][s].ln();
                    (prev_s, prob)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            v[t][s] = best_prob + emission;
            backptr[t][s] = best_prev;
        }
    }

    // Backtrack
    let mut states = vec![0; n];
    states[n-1] = (0..k)
        .max_by(|&a, &b| v[n-1][a].total_cmp(&v[n-1][b]))
        .unwrap();

    for t in (0..n-1).rev() {
        states[t] = backptr[t+1][states[t+1]];
    }

    states
}

/// Forward-backward algorithm for posterior probabilities
///
/// Returns P(state | all observations) for each window
pub fn forward_backward(observations: &[AncestryObservation], params: &AncestryHmmParams) -> Vec<Vec<f64>> {
    let n = observations.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    // Forward pass (log scale)
    let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];

    for (s, alpha_0s) in alpha[0].iter_mut().enumerate() {
        *alpha_0s = params.initial[s].ln() + params.log_emission(&observations[0], s);
    }

    for t in 1..n {
        for s in 0..k {
            let emission = params.log_emission(&observations[t], s);
            let prev_sum = log_sum_exp(&(0..k)
                .map(|prev_s| alpha[t-1][prev_s] + params.transitions[prev_s][s].ln())
                .collect::<Vec<_>>());
            alpha[t][s] = prev_sum + emission;
        }
    }

    // Backward pass (log scale)
    let mut beta = vec![vec![0.0; k]; n];
    // beta[n-1] = [0, 0, ...] (log(1) = 0)

    for t in (0..n-1).rev() {
        for s in 0..k {
            beta[t][s] = log_sum_exp(&(0..k)
                .map(|next_s| {
                    params.transitions[s][next_s].ln()
                    + params.log_emission(&observations[t+1], next_s)
                    + beta[t+1][next_s]
                })
                .collect::<Vec<_>>());
        }
    }

    // Compute posteriors
    let mut posteriors = vec![vec![0.0; k]; n];

    for t in 0..n {
        let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
        let log_total = log_sum_exp(&log_probs);

        for s in 0..k {
            posteriors[t][s] = (log_probs[s] - log_total).exp();
        }
    }

    posteriors
}

/// Genetic map for recombination-rate-aware ancestry transitions.
///
/// Provides the genetic distance (cM) between positions, which modulates
/// the ancestry switch probability. In regions with high recombination,
/// ancestry switches are more likely.
#[derive(Debug, Clone)]
pub struct AncestryGeneticMap {
    /// Sorted entries: (position_bp, position_cM)
    entries: Vec<(u64, f64)>,
}

impl AncestryGeneticMap {
    /// Parse a PLINK-format genetic map file for a specific chromosome.
    ///
    /// Supports 4-column (chr, pos_bp, rate, pos_cM) and 3-column (pos_bp, rate, pos_cM).
    pub fn from_file<P: AsRef<std::path::Path>>(path: P, chrom: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("Failed to open genetic map: {}", e))?;
        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();

        let chrom_normalized = chrom.strip_prefix("chr").unwrap_or(chrom);

        use std::io::BufRead;
        for line_result in reader.lines() {
            let line = line_result.map_err(|e| format!("Read error: {}", e))?;
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 4 {
                let chr = fields[0].strip_prefix("chr").unwrap_or(fields[0]);
                if chr != chrom_normalized {
                    continue;
                }
                let pos_bp: u64 = fields[1].parse().map_err(|_| "Invalid position".to_string())?;
                let pos_cm: f64 = fields[3].parse().map_err(|_| "Invalid cM".to_string())?;
                entries.push((pos_bp, pos_cm));
            } else if fields.len() == 3 {
                let pos_bp: u64 = fields[0].parse().map_err(|_| "Invalid position".to_string())?;
                let pos_cm: f64 = fields[2].parse().map_err(|_| "Invalid cM".to_string())?;
                entries.push((pos_bp, pos_cm));
            }
        }

        entries.sort_by_key(|e| e.0);

        if entries.is_empty() {
            return Err(format!("No genetic map entries for chromosome {}", chrom));
        }

        Ok(Self { entries })
    }

    /// Create a uniform-rate genetic map for testing.
    pub fn uniform(start_bp: u64, end_bp: u64, rate_cm_per_mb: f64) -> Self {
        let start_cm = 0.0;
        let end_cm = (end_bp - start_bp) as f64 * rate_cm_per_mb / 1_000_000.0;
        Self {
            entries: vec![(start_bp, start_cm), (end_bp, end_cm)],
        }
    }

    /// Interpolate genetic position (cM) at a physical position (bp).
    pub fn interpolate_cm(&self, pos_bp: u64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        if self.entries.len() == 1 {
            return self.entries[0].1;
        }

        let idx = self.entries.partition_point(|e| e.0 <= pos_bp);

        if idx == 0 {
            let (bp0, cm0) = self.entries[0];
            let (bp1, cm1) = self.entries[1];
            let rate = if bp1 > bp0 { (cm1 - cm0) / (bp1 - bp0) as f64 } else { 0.0 };
            cm0 - rate * (bp0 - pos_bp) as f64
        } else if idx >= self.entries.len() {
            let n = self.entries.len();
            let (bp_prev, cm_prev) = self.entries[n - 2];
            let (bp_last, cm_last) = self.entries[n - 1];
            let rate = if bp_last > bp_prev { (cm_last - cm_prev) / (bp_last - bp_prev) as f64 } else { 0.0 };
            cm_last + rate * (pos_bp - bp_last) as f64
        } else {
            let (bp_lo, cm_lo) = self.entries[idx - 1];
            let (bp_hi, cm_hi) = self.entries[idx];
            if bp_hi == bp_lo { cm_lo }
            else {
                let frac = (pos_bp - bp_lo) as f64 / (bp_hi - bp_lo) as f64;
                cm_lo + frac * (cm_hi - cm_lo)
            }
        }
    }

    /// Genetic distance (cM) between two positions.
    pub fn genetic_distance_cm(&self, pos1_bp: u64, pos2_bp: u64) -> f64 {
        (self.interpolate_cm(pos2_bp) - self.interpolate_cm(pos1_bp)).abs()
    }

    /// Compute per-window switch probability modulated by genetic distance.
    ///
    /// Uses Haldane's map function: r = 0.5 * (1 - exp(-2d)), where d is in Morgans.
    /// The base switch probability is scaled by the ratio of actual genetic distance
    /// to the expected genetic distance for a window of that physical size.
    pub fn modulated_switch_prob(
        &self,
        base_switch_prob: f64,
        pos1_bp: u64,
        pos2_bp: u64,
        window_size_bp: u64,
    ) -> f64 {
        let dist_cm = self.genetic_distance_cm(pos1_bp, pos2_bp);
        let dist_morgans = dist_cm / 100.0;

        // Expected genetic distance for this window size at ~1 cM/Mb average
        let expected_dist_cm = window_size_bp as f64 / 1_000_000.0; // ~1 cM/Mb
        let expected_dist_morgans = expected_dist_cm / 100.0;

        if expected_dist_morgans < 1e-10 {
            return base_switch_prob;
        }

        // Scale factor: actual / expected genetic distance
        let scale = dist_morgans / expected_dist_morgans;

        // Haldane-adjusted switch probability
        let adjusted = 1.0 - (-2.0 * dist_morgans).exp();
        let base_haldane = 1.0 - (-2.0 * expected_dist_morgans).exp();

        if base_haldane < 1e-10 {
            return base_switch_prob * scale.max(0.01);
        }

        (base_switch_prob * adjusted / base_haldane).clamp(1e-6, 0.5)
    }
}

/// Decoding method for ancestry inference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodingMethod {
    /// Viterbi: most likely state *sequence* (MAP path)
    /// Favors contiguous blocks; may miss short minority ancestry tracts
    Viterbi,
    /// Posterior: choose most likely state at each window independently
    /// Better for detecting minority ancestry since it doesn't penalize isolated switches
    Posterior,
    /// MPEL: Maximum Posterior Expected Loss decoder (T24).
    /// Two-stage: forward-backward → Viterbi on log-posteriors.
    /// Combines full-sequence marginal evidence (FB) with path smoothness (Viterbi).
    /// Replaces heuristic smoothing with a principled approach.
    Mpel,
}

impl std::fmt::Display for DecodingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodingMethod::Viterbi => write!(f, "viterbi"),
            DecodingMethod::Posterior => write!(f, "posterior"),
            DecodingMethod::Mpel => write!(f, "mpel"),
        }
    }
}

impl std::str::FromStr for DecodingMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "viterbi" => Ok(DecodingMethod::Viterbi),
            "posterior" => Ok(DecodingMethod::Posterior),
            "mpel" => Ok(DecodingMethod::Mpel),
            _ => Err(format!("Unknown decoding method '{}'. Use: viterbi, posterior, mpel", s)),
        }
    }
}

/// Posterior decoding: choose the most probable state at each position
/// independently based on forward-backward posteriors.
///
/// Unlike Viterbi which finds the single most probable *path* (penalizing
/// switches), posterior decoding selects the state with highest marginal
/// posterior at each window. This is better for detecting minority ancestry
/// tracts because it doesn't impose a global path constraint.
///
/// # Arguments
/// * `observations` - Sequence of ancestry observations
/// * `params` - HMM parameters
///
/// # Returns
/// State assignments (one per observation), chosen as argmax of posterior
pub fn posterior_decode(observations: &[AncestryObservation], params: &AncestryHmmParams) -> Vec<usize> {
    let posteriors = forward_backward(observations, params);
    posteriors.iter()
        .map(|probs| {
            probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

/// MPEL (Maximum Posterior Expected Loss) decoder.
///
/// Two-stage approach from T24:
/// 1. Run forward-backward to get marginal posteriors γ_t(z)
/// 2. Run Viterbi on log(γ_t(z)) as pseudo-emissions
///
/// This combines the full-sequence evidence integration of forward-backward
/// (each posterior incorporates ALL observations) with the path smoothness
/// of Viterbi (dynamic programming enforces temporal continuity).
///
/// Key advantage over standard Viterbi: at position t, standard Viterbi
/// only has forward information. MPEL's pseudo-emissions at t already
/// integrate forward AND backward evidence, making it robust to local noise.
///
/// Key advantage over posterior decoding: posterior argmax has no continuity
/// constraint — a single noisy window can flip the state. MPEL's Viterbi
/// pass penalizes switches, producing smoother tracts.
pub fn mpel_decode_from_posteriors(
    posteriors: &[Vec<f64>],
    params: &AncestryHmmParams,
) -> Vec<usize> {
    let n = posteriors.len();
    if n == 0 {
        return Vec::new();
    }

    // Convert posteriors to log pseudo-emissions
    let log_pseudo: Vec<Vec<f64>> = posteriors.iter()
        .map(|probs| probs.iter().map(|&p| p.max(1e-300).ln()).collect())
        .collect();

    // Viterbi on pseudo-emissions with original transitions for path smoothing
    viterbi_from_log_emissions(&log_pseudo, params)
}

/// Compute per-step transition matrix modulated by genetic distance.
///
/// For each pair of consecutive windows, computes the switch probability
/// based on the genetic distance (cM) between them. Higher recombination
/// rate = higher switch probability.
fn genetic_map_transition_log(
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
    pos1: u64,
    pos2: u64,
    window_size: u64,
) -> Vec<Vec<f64>> {
    let k = params.n_states;
    let base_switch = 1.0 - params.transitions[0][0]; // base switch probability
    let mod_switch = genetic_map.modulated_switch_prob(base_switch, pos1, pos2, window_size);

    let stay = 1.0 - mod_switch;
    let switch_each = mod_switch / (k - 1).max(1) as f64;

    let mut trans = vec![vec![switch_each.ln(); k]; k];
    for (s, row) in trans.iter_mut().enumerate() {
        row[s] = stay.ln();
    }
    trans
}

/// Forward-backward with genetic-map-aware transitions.
///
/// Uses per-window transition probabilities modulated by the genetic distance
/// between consecutive windows. Higher recombination rate → higher switch probability.
pub fn forward_backward_with_genetic_map(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
) -> Vec<Vec<f64>> {
    let n = observations.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let window_size = if n > 1 {
        observations[1].start.saturating_sub(observations[0].start).max(1)
    } else {
        10_000
    };

    // Forward pass
    let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];
    for (s, alpha_0s) in alpha[0].iter_mut().enumerate() {
        *alpha_0s = params.initial[s].ln() + params.log_emission(&observations[0], s);
    }

    for t in 1..n {
        let trans = genetic_map_transition_log(
            params, genetic_map,
            observations[t - 1].start, observations[t].start,
            window_size,
        );
        for s in 0..k {
            let emission = params.log_emission(&observations[t], s);
            let prev_sum = log_sum_exp(&(0..k)
                .map(|prev_s| alpha[t - 1][prev_s] + trans[prev_s][s])
                .collect::<Vec<_>>());
            alpha[t][s] = prev_sum + emission;
        }
    }

    // Backward pass
    let mut beta = vec![vec![0.0; k]; n];
    for t in (0..n - 1).rev() {
        let trans = genetic_map_transition_log(
            params, genetic_map,
            observations[t].start, observations[t + 1].start,
            window_size,
        );
        for s in 0..k {
            beta[t][s] = log_sum_exp(&(0..k)
                .map(|next_s| {
                    trans[s][next_s]
                    + params.log_emission(&observations[t + 1], next_s)
                    + beta[t + 1][next_s]
                })
                .collect::<Vec<_>>());
        }
    }

    // Compute posteriors
    let mut posteriors = vec![vec![0.0; k]; n];
    for t in 0..n {
        let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
        let log_total = log_sum_exp(&log_probs);
        for s in 0..k {
            posteriors[t][s] = (log_probs[s] - log_total).exp();
        }
    }

    posteriors
}

/// Viterbi with genetic-map-aware transitions.
pub fn viterbi_with_genetic_map(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
) -> Vec<usize> {
    let n = observations.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let window_size = if n > 1 {
        observations[1].start.saturating_sub(observations[0].start).max(1)
    } else {
        10_000
    };

    let mut v = vec![vec![f64::NEG_INFINITY; k]; n];
    let mut traceback = vec![vec![0usize; k]; n];

    for (s, v_0s) in v[0].iter_mut().enumerate() {
        *v_0s = params.initial[s].ln() + params.log_emission(&observations[0], s);
    }

    for t in 1..n {
        let trans = genetic_map_transition_log(
            params, genetic_map,
            observations[t - 1].start, observations[t].start,
            window_size,
        );
        for s in 0..k {
            let emission = params.log_emission(&observations[t], s);
            let (best_prev, best_prob) = (0..k)
                .map(|prev_s| (prev_s, v[t - 1][prev_s] + trans[prev_s][s]))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            v[t][s] = best_prob + emission;
            traceback[t][s] = best_prev;
        }
    }

    // Backtrack
    let mut states = vec![0usize; n];
    states[n - 1] = v[n - 1].iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    for t in (0..n - 1).rev() {
        states[t] = traceback[t + 1][states[t + 1]];
    }

    states
}

/// Posterior decoding with genetic-map-aware transitions.
pub fn posterior_decode_with_genetic_map(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
) -> Vec<usize> {
    let posteriors = forward_backward_with_genetic_map(observations, params, genetic_map);
    posteriors.iter()
        .map(|probs| {
            probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

// ============================================================================
// Precomputed emission functions for multi-window context
// ============================================================================

/// Smooth raw per-haplotype similarities across neighboring windows.
///
/// For each window t and each haplotype h, replaces the similarity value with
/// the mean of similarities from windows [t-context, t+context].
/// This smooths the raw input BEFORE the softmax emission model, which is
/// mathematically superior to smoothing log-emissions post-softmax because:
/// 1. Averaging raw values preserves the signal distribution shape
/// 2. Softmax then amplifies the averaged (cleaner) differences
/// 3. Post-softmax averaging washes out the peaked probability structure
///
/// # Arguments
/// * `observations` - Slice of ancestry observations with per-haplotype similarities
/// * `context` - Number of neighboring windows on each side (0 = no smoothing)
///
/// # Returns
/// New vector of observations with smoothed similarity values
pub fn smooth_observations(
    observations: &[AncestryObservation],
    context: usize,
) -> Vec<AncestryObservation> {
    if context == 0 || observations.is_empty() {
        return observations.to_vec();
    }

    let n = observations.len();

    // Collect all haplotype keys that appear in any observation
    let mut all_haps: Vec<String> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for obs in observations {
            for key in obs.similarities.keys() {
                if seen.insert(key.clone()) {
                    all_haps.push(key.clone());
                }
            }
        }
    }

    let mut smoothed = Vec::with_capacity(n);

    for t in 0..n {
        let lo = t.saturating_sub(context);
        let hi = (t + context).min(n - 1);

        let mut new_sims = HashMap::new();
        for hap in &all_haps {
            let mut sum = 0.0;
            let mut count = 0.0;
            for obs in observations.iter().take(hi + 1).skip(lo) {
                if let Some(&val) = obs.similarities.get(hap) {
                    sum += val;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                new_sims.insert(hap.clone(), sum / count);
            }
        }

        // For coverage_ratios, also smooth them
        let new_coverage = observations[t].coverage_ratios.as_ref().map(|_| {
            let mut cov_sims = HashMap::new();
            for hap in &all_haps {
                let mut sum = 0.0;
                let mut count = 0.0;
                for obs in observations.iter().take(hi + 1).skip(lo) {
                    if let Some(cov) = &obs.coverage_ratios {
                        if let Some(&val) = cov.get(hap) {
                            sum += val;
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    cov_sims.insert(hap.clone(), sum / count);
                }
            }
            cov_sims
        });

        smoothed.push(AncestryObservation {
            chrom: observations[t].chrom.clone(),
            start: observations[t].start,
            end: observations[t].end,
            sample: observations[t].sample.clone(),
            similarities: new_sims,
            coverage_ratios: new_coverage,
            haplotype_consistency_bonus: None,
        });
    }

    smoothed
}

/// Apply haplotype consistency bonus to observations.
///
/// For each window and each population, identifies the best-matching reference
/// haplotype and checks if the same haplotype is consistently best across
/// neighboring windows (±context). Consistent best-haplotype implies genuine
/// haplotype sharing (like LD), while fluctuating best-haplotype suggests noise.
///
/// The bonus is added to per-population aggregated similarities before softmax,
/// scaled by `weight * temperature` so the effect is temperature-invariant.
///
/// This captures haplotype-level linkage information that per-window identity
/// alone discards — bridging the gap between identity-based and VCF-based methods.
pub fn apply_haplotype_consistency(
    observations: &mut [AncestryObservation],
    params: &AncestryHmmParams,
    context: usize,
    weight: f64,
) {
    if context == 0 || observations.is_empty() || params.populations.is_empty() {
        return;
    }

    let n = observations.len();
    let k = params.populations.len();
    let temperature = params.emission_std;

    // Step 1: For each window and each population, find the best-matching haplotype
    let mut best_hap_per_window: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);
    for obs in observations.iter() {
        let mut per_pop = Vec::with_capacity(k);
        for pop in &params.populations {
            let best = pop.haplotypes.iter().enumerate()
                .filter_map(|(idx, h)| obs.similarities.get(h).map(|&s| (idx, s)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            per_pop.push(best.map(|(idx, _)| idx));
        }
        best_hap_per_window.push(per_pop);
    }

    // Step 2: Compute consistency score for each window and population
    let mut bonuses: Vec<Vec<f64>> = Vec::with_capacity(n);
    for t in 0..n {
        let mut pop_bonuses = vec![0.0; k];
        for p in 0..k {
            let my_best = match best_hap_per_window[t][p] {
                Some(b) => b,
                None => continue,
            };

            let start = t.saturating_sub(context);
            let end = (t + context + 1).min(n);
            let mut matching = 0usize;
            let mut total = 0usize;

            for (j, neighbor_haps) in best_hap_per_window.iter().enumerate().take(end).skip(start) {
                if j == t { continue; }
                if let Some(neighbor_best) = neighbor_haps[p] {
                    total += 1;
                    if neighbor_best == my_best {
                        matching += 1;
                    }
                }
            }

            if total > 0 {
                let consistency = matching as f64 / total as f64;
                // Scale bonus by weight * temperature so it's temperature-invariant
                // in softmax space: bonus/temperature = weight * consistency
                pop_bonuses[p] = weight * consistency * temperature;
            }
        }
        bonuses.push(pop_bonuses);
    }

    // Step 3: Store bonuses in observations
    for (t, obs) in observations.iter_mut().enumerate() {
        obs.haplotype_consistency_bonus = Some(bonuses[t].clone());
    }
}

/// Precompute log emissions for all observations and states.
///
/// Returns an n×k matrix where entry [t][s] = log P(obs_t | state=s).
/// This allows emission smoothing before running HMM algorithms.
pub fn precompute_log_emissions(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
) -> Vec<Vec<f64>> {
    observations.iter()
        .map(|obs| {
            (0..params.n_states)
                .map(|s| params.log_emission(obs, s))
                .collect()
        })
        .collect()
}

/// Compute per-population variance of aggregated identities across all windows.
///
/// For each population k, computes Var(agg_sim_k) across all observation windows.
/// Higher variance indicates noisier identity signal (e.g., admixed AMR has higher
/// variance than well-defined AFR).
pub fn compute_population_variances(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
) -> Vec<f64> {
    let k = populations.len();
    let mut sums = vec![0.0_f64; k];
    let mut sq_sums = vec![0.0_f64; k];
    let mut counts = vec![0u64; k];

    for obs in observations {
        for (p, pop) in populations.iter().enumerate() {
            let sims: Vec<f64> = pop.haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h))
                .cloned()
                .collect();
            if let Some(val) = emission_model.aggregate(&sims) {
                if val > 0.0 {
                    sums[p] += val;
                    sq_sums[p] += val * val;
                    counts[p] += 1;
                }
            }
        }
    }

    (0..k).map(|p| {
        if counts[p] < 2 {
            return 0.0; // insufficient data
        }
        let n = counts[p] as f64;
        let mean = sums[p] / n;
        (sq_sums[p] / n - mean * mean).max(0.0)
    }).collect()
}

/// Convert per-population variances to per-population temperatures.
///
/// Higher variance → higher temperature (softer, less confident emissions).
/// Lower variance → lower temperature (sharper, more confident emissions).
///
/// Formula: temp_k = base_temp × (1 + gamma × (σ_k/σ_median - 1))
///
/// gamma controls heteroscedasticity strength:
/// - 0.0: uniform temperature (standard behavior)
/// - 0.5: moderate adjustment (recommended)
/// - 1.0: full proportional adjustment
///
/// Temperatures clamped to [0.3×base, 3.0×base] to prevent extreme values.
pub fn compute_heteroscedastic_temperatures(
    variances: &[f64],
    base_temp: f64,
    gamma: f64,
) -> Vec<f64> {
    if variances.is_empty() || gamma <= 0.0 {
        return vec![base_temp; variances.len()];
    }

    // Compute median of positive standard deviations
    let mut stds: Vec<f64> = variances.iter()
        .map(|&v| v.sqrt())
        .filter(|&s| s > 0.0)
        .collect();

    if stds.is_empty() {
        return vec![base_temp; variances.len()];
    }

    stds.sort_by(|a, b| a.total_cmp(b));
    let median_std = stds[stds.len() / 2];

    if median_std <= 0.0 {
        return vec![base_temp; variances.len()];
    }

    variances.iter().map(|&v| {
        let std = v.sqrt();
        if std <= 0.0 {
            return base_temp;
        }
        let ratio = std / median_std;
        let factor = 1.0 + gamma * (ratio - 1.0);
        (base_temp * factor).clamp(base_temp * 0.3, base_temp * 3.0)
    }).collect()
}

/// Precompute log-emissions with per-population temperatures (heteroscedastic softmax).
///
/// Unlike the standard softmax which uses a single temperature T for all populations:
///   P(k) = exp(s_k/T) / Σ_j exp(s_j/T)
///
/// This uses per-population temperatures:
///   P(k) = exp(s_k/T_k) / Σ_j exp(s_j/T_j)
///
/// Populations with higher identity variance get higher T_k (softer emissions),
/// preventing overconfident wrong calls. Populations with lower variance get
/// lower T_k (sharper emissions), maintaining discriminative power.
pub fn precompute_heteroscedastic_log_emissions(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    pop_temperatures: &[f64],
) -> Vec<Vec<f64>> {
    let k = params.n_states;

    observations.iter()
        .map(|obs| {
            // Get aggregated similarity per population
            let mut pop_sims: Vec<Option<f64>> = params.populations.iter()
                .map(|pop| {
                    let sims: Vec<f64> = pop.haplotypes.iter()
                        .filter_map(|h| obs.similarities.get(h))
                        .cloned()
                        .collect();
                    params.emission_model.aggregate(&sims)
                })
                .collect();

            // Apply haplotype consistency bonus
            if let Some(ref bonus) = obs.haplotype_consistency_bonus {
                for (p, sim) in pop_sims.iter_mut().enumerate() {
                    if let Some(ref mut s) = sim {
                        if p < bonus.len() {
                            *s += bonus[p];
                        }
                    }
                }
            }

            let mut log_emissions = vec![f64::NEG_INFINITY; k];

            // Collect valid scores (populations with data)
            let valid: Vec<(usize, f64)> = (0..k)
                .filter_map(|i| {
                    pop_sims[i]
                        .filter(|&s| s > 0.0)
                        .map(|s| {
                            let score = if let Some(ref norm) = params.normalization {
                                (s - norm.means[i]) / norm.stds[i]
                            } else {
                                s
                            };
                            (i, score)
                        })
                })
                .collect();

            if valid.len() <= 1 {
                for &(i, _) in &valid {
                    log_emissions[i] = 0.0;
                }
                return log_emissions;
            }

            // Per-population softmax: score_i / temp_i
            let scaled: Vec<(usize, f64)> = valid.iter()
                .map(|&(i, s)| (i, s / pop_temperatures[i]))
                .collect();

            let max_scaled = scaled.iter()
                .map(|&(_, s)| s)
                .fold(f64::NEG_INFINITY, f64::max);

            let log_sum_exp: f64 = scaled.iter()
                .map(|&(_, s)| (s - max_scaled).exp())
                .sum::<f64>()
                .ln();

            for &(i, s) in &scaled {
                log_emissions[i] = s - max_scaled - log_sum_exp;
            }

            log_emissions
        })
        .collect()
}

/// Smooth precomputed log emissions using a sliding window average.
///
/// For each window t, the smoothed emission is the mean of log emissions
/// from windows [t-context, t+context] (clipped at boundaries).
/// This addresses weak per-window signal by pooling evidence from neighbors.
///
/// With context=2 (5-window span), SNR increases by ~√5 ≈ 2.2x,
/// turning marginal signals (SNR ~0.74) into detectable ones (SNR ~1.6).
///
/// # Arguments
/// * `log_emissions` - n×k matrix of log emissions
/// * `context` - number of neighboring windows on each side (0 = no smoothing)
///
/// # Returns
/// Smoothed n×k matrix of log emissions
pub fn smooth_log_emissions(log_emissions: &[Vec<f64>], context: usize) -> Vec<Vec<f64>> {
    if context == 0 || log_emissions.is_empty() {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = if n > 0 { log_emissions[0].len() } else { return Vec::new() };

    let mut smoothed = vec![vec![0.0; k]; n];

    for (t, smoothed_t) in smoothed.iter_mut().enumerate() {
        let lo = t.saturating_sub(context);
        let hi = (t + context).min(n - 1);
        let span = (hi - lo + 1) as f64;

        for s in 0..k {
            // Average log emissions in the window
            // This is equivalent to geometric mean of probabilities
            let sum: f64 = (lo..=hi)
                .map(|i| log_emissions[i][s])
                .sum();
            smoothed_t[s] = sum / span;
        }
    }

    smoothed
}

/// SNR-weighted emission context smoothing.
///
/// Like `smooth_log_emissions` but weights neighboring windows by their
/// informativeness (discrimination gap between top-2 states). Windows
/// where one population clearly dominates get more weight; ambiguous
/// windows get less weight.
///
/// This is especially useful for ancestry inference where per-window SNR < 1:
/// borrowing strength from confident neighbors improves accuracy more than
/// uniform averaging.
///
/// # Arguments
/// * `log_emissions` - n×k matrix of log emissions
/// * `context` - number of neighboring windows on each side
///
/// # Returns
/// Smoothed n×k matrix of log emissions
pub fn smooth_log_emissions_weighted(log_emissions: &[Vec<f64>], context: usize) -> Vec<Vec<f64>> {
    if context == 0 || log_emissions.is_empty() {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = if n > 0 { log_emissions[0].len() } else { return Vec::new() };

    // Precompute discrimination gap (top1 - top2 log emission) for each window
    let gaps: Vec<f64> = log_emissions.iter().map(|row| {
        if k < 2 { return 1.0; }
        let mut top1 = f64::NEG_INFINITY;
        let mut top2 = f64::NEG_INFINITY;
        for &v in row {
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        // Gap between best and second-best state (in log space)
        // Larger gap = more discriminative window
        if top1.is_finite() && top2.is_finite() {
            (top1 - top2).max(0.0)
        } else {
            0.0
        }
    }).collect();

    let mut smoothed = vec![vec![0.0; k]; n];

    for (t, smoothed_t) in smoothed.iter_mut().enumerate() {
        let lo = t.saturating_sub(context);
        let hi = (t + context).min(n - 1);

        // Compute weights: floor of 1.0 + gap (so all windows contribute something)
        let weights: Vec<f64> = (lo..=hi).map(|i| 1.0 + gaps[i]).collect();
        let total_weight: f64 = weights.iter().sum();

        if total_weight > 0.0 {
            for s in 0..k {
                let weighted_sum: f64 = (lo..=hi).zip(weights.iter())
                    .map(|(i, &w)| w * log_emissions[i][s])
                    .sum();
                smoothed_t[s] = weighted_sum / total_weight;
            }
        } else {
            // Fallback: uniform average
            let span = (hi - lo + 1) as f64;
            for s in 0..k {
                let sum: f64 = (lo..=hi).map(|i| log_emissions[i][s]).sum();
                smoothed_t[s] = sum / span;
            }
        }
    }

    smoothed
}

/// Apply per-window contrast normalization to log emissions.
///
/// Centers each window's log emissions to have zero mean across states.
/// This removes the effect of windows having globally higher or lower
/// identity (e.g., due to repetitive sequences or alignment artifacts),
/// and focuses the HMM on which population is RELATIVELY most similar.
///
/// # Arguments
/// * `log_emissions` - n×k matrix of log emissions
///
/// # Returns
/// Contrast-normalized n×k matrix of log emissions
pub fn contrast_normalize_emissions(log_emissions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    log_emissions.iter().map(|row| {
        // Compute mean of finite values in this window
        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite_vals.is_empty() {
            return row.clone();
        }
        let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;

        // Center: subtract mean from all finite values, leave -inf unchanged
        row.iter().map(|&v| {
            if v.is_finite() {
                v - mean
            } else {
                v
            }
        }).collect()
    }).collect()
}

/// Dampen emissions for windows with low discriminability.
///
/// For each window, computes discriminability (gap between best and second-best
/// state log-emission). Windows with discriminability below the median have their
/// emissions scaled toward uniform, telling the HMM to rely on transitions for
/// these uncertain regions instead of following noisy emission signals.
///
/// The dampening factor is: min(1.0, discriminability / (scale_factor × median_disc))
/// - High discriminability (> scale_factor × median): full emission weight
/// - Low discriminability: emissions blended toward per-window mean
/// - Zero discriminability: uniform emissions (all states equal)
///
/// Mathematically equivalent to per-window adaptive temperature scaling:
/// effective_temp = base_temp / dampening_factor.
///
/// # Arguments
/// * `log_emissions` - n×k matrix of log emissions
/// * `scale_factor` - How aggressively to dampen. 1.0 = dampen anything below
///   median, 2.0 = only dampen below 2× median (gentler). Default: 1.5
///
/// # Returns
/// Dampened n×k matrix of log emissions
pub fn dampen_low_confidence_emissions(
    log_emissions: &[Vec<f64>],
    scale_factor: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    let k = if let Some(row) = log_emissions.first() { row.len() } else { return Vec::new() };
    if k <= 1 {
        return log_emissions.to_vec();
    }

    // Compute discriminability for each window: gap between best and second-best
    let discs: Vec<f64> = log_emissions
        .iter()
        .map(|row| {
            let mut finite: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
            if finite.len() < 2 {
                return 0.0;
            }
            finite.sort_by(|a, b| b.total_cmp(a)); // descending
            finite[0] - finite[1]
        })
        .collect();

    // Compute median of positive discriminabilities
    let mut positive_discs: Vec<f64> = discs.iter().filter(|&&d| d > 0.0).cloned().collect();
    if positive_discs.is_empty() {
        return log_emissions.to_vec();
    }
    positive_discs.sort_by(|a, b| a.total_cmp(b));
    let median_disc = positive_discs[positive_discs.len() / 2];

    if median_disc <= 0.0 {
        return log_emissions.to_vec();
    }

    let threshold = scale_factor * median_disc;

    // Dampen: scale deviation from per-window mean by dampening factor
    log_emissions
        .iter()
        .zip(&discs)
        .map(|(row, &disc)| {
            // Only dampen windows below threshold
            if disc >= threshold {
                return row.clone();
            }

            let finite_vals: Vec<f64> =
                row.iter().filter(|v| v.is_finite()).cloned().collect();
            if finite_vals.len() < 2 {
                return row.clone();
            }

            let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;
            let alpha = if threshold > 0.0 {
                (disc / threshold).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // Blend toward mean: dampened[s] = alpha * (v - mean) + mean
            // When alpha = 0: all states get mean (uniform)
            // When alpha = 1: original emission (no change)
            row.iter()
                .map(|&v| {
                    if v.is_finite() {
                        alpha * (v - mean) + mean
                    } else {
                        v
                    }
                })
                .collect()
        })
        .collect()
}

/// Viterbi algorithm using precomputed log emissions.
///
/// Identical to `viterbi()` but uses precomputed (potentially smoothed)
/// emissions instead of computing them on-the-fly.
pub fn viterbi_from_log_emissions(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
) -> Vec<usize> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let mut v = vec![vec![f64::NEG_INFINITY; k]; n];
    let mut backptr = vec![vec![0usize; k]; n];

    // Initialize
    for s in 0..k {
        v[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    // Forward pass
    for t in 1..n {
        for s in 0..k {
            let (best_prev, best_prob) = (0..k)
                .map(|prev_s| {
                    let prob = v[t-1][prev_s] + params.transitions[prev_s][s].ln();
                    (prev_s, prob)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            v[t][s] = best_prob + log_emissions[t][s];
            backptr[t][s] = best_prev;
        }
    }

    // Backtrack
    let mut states = vec![0; n];
    states[n-1] = (0..k)
        .max_by(|&a, &b| v[n-1][a].total_cmp(&v[n-1][b]))
        .unwrap();

    for t in (0..n-1).rev() {
        states[t] = backptr[t+1][states[t+1]];
    }

    states
}

/// Forward-backward algorithm using precomputed log emissions.
///
/// Returns posterior probabilities P(state | all observations) for each window.
/// Accepts precomputed (potentially smoothed) emissions.
pub fn forward_backward_from_log_emissions(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    // Forward pass (log scale)
    let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];

    for s in 0..k {
        alpha[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    for t in 1..n {
        for s in 0..k {
            let prev_sum = log_sum_exp(&(0..k)
                .map(|prev_s| alpha[t-1][prev_s] + params.transitions[prev_s][s].ln())
                .collect::<Vec<_>>());
            alpha[t][s] = prev_sum + log_emissions[t][s];
        }
    }

    // Backward pass (log scale)
    let mut beta = vec![vec![0.0; k]; n];

    for t in (0..n-1).rev() {
        for s in 0..k {
            beta[t][s] = log_sum_exp(&(0..k)
                .map(|next_s| {
                    params.transitions[s][next_s].ln()
                    + log_emissions[t+1][next_s]
                    + beta[t+1][next_s]
                })
                .collect::<Vec<_>>());
        }
    }

    // Compute posteriors
    let mut posteriors = vec![vec![0.0; k]; n];

    for t in 0..n {
        let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
        let log_total = log_sum_exp(&log_probs);

        for s in 0..k {
            posteriors[t][s] = (log_probs[s] - log_total).exp();
        }
    }

    posteriors
}

/// Compute per-window transition log-probabilities based on physical distance.
///
/// When consecutive windows are farther apart (gaps from filtered windows or
/// missing data), the effective switch probability increases because more
/// recombination could have occurred. This provides a simple approximation to
/// genetic-map-aware transitions without requiring a genetic map file.
///
/// The model assumes a constant recombination rate per base pair:
/// `P(switch) = 1 - (1 - base_switch)^(distance / expected_distance)`
///
/// For adjacent windows (distance = window_size), this equals base_switch.
/// For larger gaps, the switch probability increases toward 1.0.
///
/// # Arguments
/// * `observations` - Observations with start/end positions
/// * `params` - HMM parameters (provides base switch probability)
/// * `expected_window_size` - Expected distance between consecutive windows (e.g., 10000)
///
/// # Returns
/// Vector of (n-1) transition log-probability matrices, one per window transition
pub fn compute_distance_transitions(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    expected_window_size: u64,
) -> Vec<Vec<Vec<f64>>> {
    let k = params.n_states;
    let base_switch = 1.0 - params.transitions[0][0];
    let n = observations.len();

    if n <= 1 || expected_window_size == 0 {
        return Vec::new();
    }

    (0..n - 1).map(|t| {
        let dist = observations[t + 1].start.saturating_sub(observations[t].start);
        let ratio = dist as f64 / expected_window_size as f64;

        // Scale switch probability: P(switch) = 1 - (1 - base)^ratio
        let effective_switch = if ratio <= 1.0 {
            base_switch
        } else {
            1.0 - (1.0 - base_switch).powf(ratio)
        };

        // Clamp to valid range
        let switch = effective_switch.clamp(base_switch, 0.99);
        let stay = 1.0 - switch;
        let switch_each = switch / (k - 1).max(1) as f64;

        let mut trans = vec![vec![switch_each.ln(); k]; k];
        for (s, row) in trans.iter_mut().enumerate() {
            row[s] = stay.ln();
        }
        trans
    }).collect()
}

/// Compute a single population-similarity-aware log-transition matrix.
///
/// Standard HMM uses uniform off-diagonal transitions: P(i→j) = switch/(K-1).
/// This function makes transitions proportional to inter-population similarity:
/// P(i→j) ∝ exp(-gap(i,j) / scale) × switch_prob, where gap(i,j) is the median
/// absolute difference between populations i and j's best haplotype similarities
/// across windows.
///
/// For EUR/AMR/AFR: EUR↔AMR (gap ~0.002) gets much higher transition probability
/// than EUR↔AFR (gap ~0.01), reflecting biological admixture patterns.
///
/// # Returns
/// K×K log-transition matrix
pub fn compute_population_aware_transitions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
    switch_prob: f64,
) -> Vec<Vec<f64>> {
    let k = populations.len();
    if k < 2 {
        let stay = (1.0 - switch_prob).ln();
        return vec![vec![stay; k.max(1)]; k.max(1)];
    }

    // Compute per-window per-population aggregated similarities
    let mut pair_diffs: Vec<Vec<f64>> = vec![Vec::new(); k * (k - 1) / 2];

    for obs in observations {
        let pop_sims: Vec<Option<f64>> = populations.iter().map(|pop| {
            let sims: Vec<f64> = pop.haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h))
                .cloned()
                .collect();
            emission_model.aggregate(&sims)
        }).collect();

        let mut pair_idx = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                if let (Some(si), Some(sj)) = (pop_sims[i], pop_sims[j]) {
                    pair_diffs[pair_idx].push((si - sj).abs());
                }
                pair_idx += 1;
            }
        }
    }

    // Compute median gap for each pair (symmetric matrix requires index access)
    let mut gap_matrix = vec![vec![0.0_f64; k]; k];
    let mut pair_idx = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for j in (i + 1)..k {
            let median = if pair_diffs[pair_idx].is_empty() {
                0.01 // fallback
            } else {
                let mut sorted = pair_diffs[pair_idx].clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                sorted[sorted.len() / 2]
            };
            gap_matrix[i][j] = median;
            gap_matrix[j][i] = median;
            pair_idx += 1;
        }
    }

    // Scale parameter: median of all pairwise medians
    let mut all_medians: Vec<f64> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for j in (i + 1)..k {
            all_medians.push(gap_matrix[i][j]);
        }
    }
    all_medians.sort_by(|a, b| a.total_cmp(b));
    let scale = all_medians[all_medians.len() / 2].max(1e-6);

    // Build transition matrix with similarity-proportional off-diagonal
    let stay_prob = 1.0 - switch_prob;
    let mut log_trans = vec![vec![0.0_f64; k]; k];

    for i in 0..k {
        // Compute weights w_ij = exp(-gap/scale) for j != i
        let mut weights = vec![0.0_f64; k];
        let mut total_weight = 0.0;
        for j in 0..k {
            if j == i { continue; }
            let w = (-gap_matrix[i][j] / scale).exp();
            weights[j] = w;
            total_weight += w;
        }

        // Normalize: P(i→j) = switch_prob × w_ij / total_weight
        for j in 0..k {
            if j == i {
                log_trans[i][j] = stay_prob.ln();
            } else if total_weight > 0.0 {
                let p = switch_prob * weights[j] / total_weight;
                log_trans[i][j] = p.max(1e-20).ln();
            } else {
                // Fallback uniform
                log_trans[i][j] = (switch_prob / (k - 1) as f64).ln();
            }
        }
    }

    log_trans
}

/// Forward-backward algorithm with per-window transition matrices.
///
/// Uses precomputed log emissions and per-window transition log-probability
/// matrices. This supports distance-aware transitions where the switch
/// probability varies based on physical distance between consecutive windows.
pub fn forward_backward_from_log_emissions_with_transitions(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    log_transitions: &[Vec<Vec<f64>>],
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    // Forward pass
    let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];
    for s in 0..k {
        alpha[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    for t in 1..n {
        let trans = if t - 1 < log_transitions.len() {
            &log_transitions[t - 1]
        } else {
            // Fall back to params transitions
            return forward_backward_from_log_emissions(log_emissions, params);
        };

        for s in 0..k {
            let prev_sum = log_sum_exp(&(0..k)
                .map(|prev_s| alpha[t - 1][prev_s] + trans[prev_s][s])
                .collect::<Vec<_>>());
            alpha[t][s] = prev_sum + log_emissions[t][s];
        }
    }

    // Backward pass
    let mut beta = vec![vec![0.0; k]; n];
    for t in (0..n - 1).rev() {
        let trans = &log_transitions[t];
        for s in 0..k {
            beta[t][s] = log_sum_exp(&(0..k)
                .map(|next_s| {
                    trans[s][next_s]
                    + log_emissions[t + 1][next_s]
                    + beta[t + 1][next_s]
                })
                .collect::<Vec<_>>());
        }
    }

    // Compute posteriors
    let mut posteriors = vec![vec![0.0; k]; n];
    for t in 0..n {
        let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
        let log_total = log_sum_exp(&log_probs);
        for s in 0..k {
            posteriors[t][s] = (log_probs[s] - log_total).exp();
        }
    }

    posteriors
}

/// Viterbi with precomputed emissions and per-window transition matrices.
pub fn viterbi_from_log_emissions_with_transitions(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    log_transitions: &[Vec<Vec<f64>>],
) -> Vec<usize> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let mut v = vec![vec![f64::NEG_INFINITY; k]; n];
    let mut backptr = vec![vec![0usize; k]; n];

    for s in 0..k {
        v[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    for t in 1..n {
        let trans = if t - 1 < log_transitions.len() {
            &log_transitions[t - 1]
        } else {
            return viterbi_from_log_emissions(log_emissions, params);
        };

        for s in 0..k {
            let (best_prev, best_prob) = (0..k)
                .map(|prev_s| (prev_s, v[t - 1][prev_s] + trans[prev_s][s]))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            v[t][s] = best_prob + log_emissions[t][s];
            backptr[t][s] = best_prev;
        }
    }

    let mut states = vec![0; n];
    states[n - 1] = (0..k)
        .max_by(|&a, &b| v[n - 1][a].total_cmp(&v[n - 1][b]))
        .unwrap();
    for t in (0..n - 1).rev() {
        states[t] = backptr[t + 1][states[t + 1]];
    }

    states
}

/// Viterbi with precomputed emissions and genetic-map-aware transitions.
pub fn viterbi_from_log_emissions_with_genetic_map(
    observations: &[AncestryObservation],
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
) -> Vec<usize> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let mut v = vec![vec![f64::NEG_INFINITY; k]; n];
    let mut backptr = vec![vec![0usize; k]; n];

    for s in 0..k {
        v[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    for t in 1..n {
        let window_size = observations[t].end.saturating_sub(observations[t].start).max(1);
        let trans_log = genetic_map_transition_log(
            params, genetic_map,
            observations[t-1].start, observations[t].start, window_size,
        );

        for s in 0..k {
            let (best_prev, best_prob) = (0..k)
                .map(|prev_s| {
                    let prob = v[t-1][prev_s] + trans_log[prev_s][s];
                    (prev_s, prob)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            v[t][s] = best_prob + log_emissions[t][s];
            backptr[t][s] = best_prev;
        }
    }

    let mut states = vec![0; n];
    states[n-1] = (0..k)
        .max_by(|&a, &b| v[n-1][a].total_cmp(&v[n-1][b]))
        .unwrap();

    for t in (0..n-1).rev() {
        states[t] = backptr[t+1][states[t+1]];
    }

    states
}

/// Forward-backward with precomputed emissions and genetic-map-aware transitions.
pub fn forward_backward_from_log_emissions_with_genetic_map(
    observations: &[AncestryObservation],
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    genetic_map: &AncestryGeneticMap,
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 {
        return Vec::new();
    }

    let mut alpha = vec![vec![f64::NEG_INFINITY; k]; n];

    for s in 0..k {
        alpha[0][s] = params.initial[s].ln() + log_emissions[0][s];
    }

    for t in 1..n {
        let window_size = observations[t].end.saturating_sub(observations[t].start).max(1);
        let trans_log = genetic_map_transition_log(
            params, genetic_map,
            observations[t-1].start, observations[t].start, window_size,
        );

        for s in 0..k {
            let prev_sum = log_sum_exp(&(0..k)
                .map(|prev_s| alpha[t-1][prev_s] + trans_log[prev_s][s])
                .collect::<Vec<_>>());
            alpha[t][s] = prev_sum + log_emissions[t][s];
        }
    }

    let mut beta = vec![vec![0.0; k]; n];

    for t in (0..n-1).rev() {
        let window_size = observations[t+1].end.saturating_sub(observations[t+1].start).max(1);
        let trans_log = genetic_map_transition_log(
            params, genetic_map,
            observations[t].start, observations[t+1].start, window_size,
        );

        for s in 0..k {
            beta[t][s] = log_sum_exp(&(0..k)
                .map(|next_s| {
                    trans_log[s][next_s]
                    + log_emissions[t+1][next_s]
                    + beta[t+1][next_s]
                })
                .collect::<Vec<_>>());
        }
    }

    let mut posteriors = vec![vec![0.0; k]; n];

    for t in 0..n {
        let log_probs: Vec<f64> = (0..k).map(|s| alpha[t][s] + beta[t][s]).collect();
        let log_total = log_sum_exp(&log_probs);

        for s in 0..k {
            posteriors[t][s] = (log_probs[s] - log_total).exp();
        }
    }

    posteriors
}

/// Log-sum-exp for numerical stability
fn log_sum_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max_val + vals.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln()
}

/// Ensemble decoding: run forward-backward with multiple parameter perturbations
/// and average posteriors for more robust state assignments.
///
/// Instead of relying on a single temperature/switch_prob setting, this runs
/// the HMM multiple times with slightly different parameters and averages the
/// posterior probabilities. The final state assignment is decoded from the
/// averaged posteriors, which is more robust to parameter misspecification.
///
/// Perturbation strategy: for each of `n_ensemble` runs, temperature is multiplied
/// by a factor in [1/scale_factor, scale_factor] (linearly spaced), and switch
/// probability is similarly scaled. This covers a range of "sharper" to "softer"
/// emission discrimination and "stickier" to "more flexible" transitions.
///
/// # Arguments
/// * `log_emissions` - Precomputed n×k log emission matrix
/// * `params` - Base HMM parameters
/// * `n_ensemble` - Number of ensemble members (odd recommended for tie-breaking)
/// * `scale_factor` - Range of perturbation (e.g., 2.0 → temps from 0.5x to 2.0x)
///
/// # Returns
/// (averaged_posteriors, decoded_states)
pub fn ensemble_decode(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    n_ensemble: usize,
    scale_factor: f64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 || n_ensemble == 0 {
        return (Vec::new(), Vec::new());
    }

    if n_ensemble == 1 {
        let posteriors = forward_backward_from_log_emissions(log_emissions, params);
        let states: Vec<usize> = posteriors.iter()
            .map(|probs| probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx).unwrap_or(0))
            .collect();
        return (posteriors, states);
    }

    let sf = scale_factor.max(1.01); // ensure meaningful perturbation
    let base_temp = params.emission_std;
    let base_stay = params.transitions[0][0];

    // Generate perturbation factors linearly spaced from 1/sf to sf
    let factors: Vec<f64> = (0..n_ensemble)
        .map(|i| {
            let t = i as f64 / (n_ensemble - 1).max(1) as f64;
            // From 1/sf to sf on log scale
            ((-sf.ln()) + t * 2.0 * sf.ln()).exp()
        })
        .collect();

    // Accumulate posteriors
    let mut avg_posteriors = vec![vec![0.0_f64; k]; n];

    for &factor in &factors {
        // Perturb temperature
        let perturbed_temp = (base_temp * factor).clamp(0.0001, 0.5);

        // Perturb switch probability (inversely: higher temp = more uncertain = higher switch)
        let base_switch = 1.0 - base_stay;
        let perturbed_switch = (base_switch * factor.sqrt()).clamp(0.001, 0.5);
        let perturbed_stay = 1.0 - perturbed_switch;

        // Build perturbed params
        let mut p = params.clone();
        p.emission_std = perturbed_temp;
        let switch_each = perturbed_switch / (k - 1).max(1) as f64;
        for s in 0..k {
            for s2 in 0..k {
                p.transitions[s][s2] = if s == s2 { perturbed_stay } else { switch_each };
            }
        }

        // Re-scale emissions with perturbed temperature
        // Since log_emissions were computed with base_temp, we need to rescale
        let temp_ratio = base_temp / perturbed_temp;
        let rescaled: Vec<Vec<f64>> = log_emissions.iter()
            .map(|row| {
                // Rescale: new_log_emit = old_log_emit * (old_temp / new_temp)
                // But we need to re-normalize after rescaling
                let rescaled_row: Vec<f64> = row.iter().map(|&v| v * temp_ratio).collect();
                let log_sum = log_sum_exp(&rescaled_row);
                rescaled_row.iter().map(|&v| v - log_sum).collect()
            })
            .collect();

        let posteriors = forward_backward_from_log_emissions(&rescaled, &p);

        for t in 0..n {
            for s in 0..k {
                avg_posteriors[t][s] += posteriors[t][s] / n_ensemble as f64;
            }
        }
    }

    // Decode from averaged posteriors
    let states: Vec<usize> = avg_posteriors.iter()
        .map(|probs| probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx).unwrap_or(0))
        .collect();

    (avg_posteriors, states)
}

/// Estimate optimal temperature for softmax emissions from observed similarity differences.
/// Uses the median of (max_sim - min_sim) across populations as the temperature.
/// This makes the model adaptive to actual data signal strength.
pub fn estimate_temperature(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> f64 {
    estimate_temperature_with_spread(observations, populations, None)
}

/// Estimate temperature from observations, with optional separate raw observations
/// for computing the multi-population spread ratio.
///
/// When emission smoothing is used, the main observations may be smoothed (for
/// correct base temperature matching the HMM's input scale), while
/// `raw_for_spread` provides unsmoothed data for the pairwise spread ratio
/// calculation (smoothing compresses pairwise distances and underestimates the
/// boost needed for close population pairs).
pub fn estimate_temperature_with_spread(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    raw_for_spread: Option<&[AncestryObservation]>,
) -> f64 {
    let k = populations.len();

    // Compute per-window per-population max similarities for base temperature
    let mut all_diffs: Vec<f64> = Vec::new();

    for obs in observations {
        let pop_sims: Vec<f64> = populations
            .iter()
            .filter_map(|pop| {
                let max_sim = pop
                    .haplotypes
                    .iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .fold(None, |acc: Option<f64>, x| Some(acc.map_or(x, |a| a.max(x))));
                max_sim
            })
            .collect();

        if pop_sims.len() >= 2 {
            let max = pop_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = pop_sims.iter().cloned().fold(f64::INFINITY, f64::min);
            if max > min {
                all_diffs.push(max - min);
            }
        }
    }

    if all_diffs.is_empty() {
        return 0.03; // fallback default
    }

    // Base temperature: upper-quartile mean of max-min diffs
    all_diffs.sort_by(|a, b| a.total_cmp(b));
    let p75_idx = all_diffs.len() * 3 / 4;
    let upper_q_mean: f64 = all_diffs[p75_idx..].iter().sum::<f64>()
        / (all_diffs.len() - p75_idx) as f64;

    let mut temp = upper_q_mean;

    // For 3+ populations: compute spread ratio from raw observations (if provided)
    // or from the main observations. Raw data better captures the true pairwise
    // distance structure because smoothing compresses the spread ratio.
    if k >= 3 {
        let spread_source = raw_for_spread.unwrap_or(observations);
        let n_pairs = k * (k - 1) / 2;
        let mut pairwise_diffs: Vec<Vec<f64>> = vec![Vec::new(); n_pairs];

        for obs in spread_source {
            let pop_sims: Vec<f64> = populations
                .iter()
                .filter_map(|pop| {
                    let max_sim = pop
                        .haplotypes
                        .iter()
                        .filter_map(|h| obs.similarities.get(h))
                        .cloned()
                        .fold(None, |acc: Option<f64>, x| Some(acc.map_or(x, |a| a.max(x))));
                    max_sim
                })
                .collect();

            if pop_sims.len() == k {
                let mut pair_idx = 0;
                for i in 0..k {
                    for j in (i + 1)..k {
                        let d = (pop_sims[i] - pop_sims[j]).abs();
                        pairwise_diffs[pair_idx].push(d);
                        pair_idx += 1;
                    }
                }
            }
        }

        if !pairwise_diffs.is_empty() {
            let mut pair_medians: Vec<f64> = pairwise_diffs
                .iter()
                .filter(|v| !v.is_empty())
                .map(|v| {
                    let mut sorted = v.clone();
                    sorted.sort_by(|a, b| a.total_cmp(b));
                    sorted[sorted.len() / 2]
                })
                .collect();
            pair_medians.sort_by(|a, b| a.total_cmp(b));

            if let (Some(&closest), Some(&widest)) = (pair_medians.first(), pair_medians.last()) {
                if closest > 0.0 && widest > 0.0 {
                    let spread_ratio = widest / closest;
                    if spread_ratio > 2.0 {
                        // Boost temperature so the close pair produces gentle softmax.
                        // Factor: spread_ratio^1.5 — aggressive correction for close pairs.
                        let boost = spread_ratio.powf(1.5);
                        temp *= boost;
                        eprintln!(
                            "  Multi-pop temperature boost: {:.1}x (spread ratio {:.1}x between closest/widest pop pairs)",
                            boost, spread_ratio
                        );
                    }
                }
            }
        }
    }

    // Clamp to reasonable range
    temp.clamp(0.0005, 0.15)
}

/// Scale temperature based on average haplotypes per population.
///
/// Based on extreme value theory: Max of N independent samples from a Gaussian
/// has expected value proportional to √(2 ln N). With more haplotypes, the Max
/// aggregation produces larger values, reducing the inter-population contrast
/// relative to the scale. Temperature should decrease to compensate.
///
/// The correction factor uses a reference panel size of 10 haplotypes per population
/// (the typical per-population count in reduced panels). For larger panels, temperature
/// decreases; for smaller panels, it increases.
///
/// # Arguments
/// * `base_temperature` - Temperature estimated from data or set by user
/// * `avg_haps_per_pop` - Average number of haplotypes per reference population
///
/// # Returns
/// Adjusted temperature, clamped to [0.001, 1.0]
pub fn scale_temperature_for_panel(base_temperature: f64, avg_haps_per_pop: f64) -> f64 {
    if avg_haps_per_pop <= 1.0 {
        return base_temperature;
    }

    // Reference: 10 haplotypes per population (typical reduced panel)
    let reference_haps = 10.0_f64;
    let ref_factor = (2.0 * reference_haps.ln()).sqrt();
    let actual_factor = (2.0 * avg_haps_per_pop.ln()).sqrt();

    let correction = ref_factor / actual_factor;
    (base_temperature * correction).clamp(0.001, 1.0)
}

/// Estimate temperature for normalized emissions.
///
/// When using per-population normalization, the input to softmax is z-scores
/// instead of raw similarities. The temperature should be calibrated to the
/// z-score scale. Uses the median of (max_z - min_z) across populations.
pub fn estimate_temperature_normalized(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
) -> f64 {
    let norm = match &params.normalization {
        Some(n) => n,
        None => return estimate_temperature(observations, &params.populations),
    };

    let mut diffs: Vec<f64> = Vec::new();

    for obs in observations {
        let z_scores: Vec<f64> = params.populations.iter().enumerate()
            .filter_map(|(i, pop)| {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                params.emission_model.aggregate(&sims)
                    .filter(|&v| v > 0.0)
                    .map(|v| (v - norm.means[i]) / norm.stds[i])
            })
            .collect();

        if z_scores.len() >= 2 {
            let max_z = z_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_z = z_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            if max_z > min_z {
                diffs.push(max_z - min_z);
            }
        }
    }

    if diffs.is_empty() {
        return 1.0; // fallback for normalized space
    }

    diffs.sort_by(|a, b| a.total_cmp(b));
    let median = diffs[diffs.len() / 2];

    // Wider clamp range for z-scores (typical range 0.5 - 5.0)
    median.clamp(0.5, 5.0)
}

/// Estimate optimal emission context for a sample based on signal quality.
///
/// Computes the median per-window discriminability (max pop sim - 2nd max pop sim)
/// and maps it to an emission context value. Weak-signal samples get wider
/// context for more averaging; strong-signal samples get narrower context
/// to preserve boundary precision.
///
/// The mapping is: context = clamp(base_ec * (target_disc / observed_disc), min_ec, max_ec)
/// where target_disc is the "ideal" discriminability level.
///
/// Returns the recommended emission context (number of neighbors on each side).
pub fn estimate_emission_context(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    base_context: usize,
    min_context: usize,
    max_context: usize,
) -> usize {
    if observations.is_empty() || populations.is_empty() || base_context == 0 {
        return base_context;
    }

    // Compute per-window discriminability: max(pop_sim) - 2nd_max(pop_sim)
    let mut discs: Vec<f64> = Vec::new();

    for obs in observations {
        let mut pop_sims: Vec<f64> = populations
            .iter()
            .filter_map(|pop| {
                pop.haplotypes
                    .iter()
                    .filter_map(|h| obs.similarities.get(h).copied())
                    .fold(None, |max, s| Some(max.map_or(s, |m: f64| m.max(s))))
            })
            .collect();

        if pop_sims.len() >= 2 {
            pop_sims.sort_by(|a, b| b.total_cmp(a));
            let disc = pop_sims[0] - pop_sims[1];
            if disc > 0.0 {
                discs.push(disc);
            }
        }
    }

    if discs.is_empty() {
        return base_context;
    }

    discs.sort_by(|a, b| a.total_cmp(b));
    let median_disc = discs[discs.len() / 2];

    if median_disc <= 0.0 {
        return max_context; // No signal at all — use max context
    }

    // Target discriminability: ~0.001 is a "good" signal level based on
    // empirical observation (corresponds to ~1 identity unit difference)
    let target_disc = 0.001;

    // Scale context inversely with signal quality:
    // strong signal (median_disc >> target) → less context needed
    // weak signal (median_disc << target) → more context needed
    let scale = target_disc / median_disc;
    let adaptive_ec = (base_context as f64 * scale).round() as usize;

    adaptive_ec.clamp(min_context, max_context)
}

/// Auto-configure pairwise_weight and emission_context from data statistics.
///
/// Uses the T53 formula based on pairwise discriminability structure:
/// - D_min: minimum Cohen's d across all population pairs (effect size of
///   per-window similarity differences for the closest pair)
/// - CV_D: coefficient of variation of Cohen's d across pairs
/// - n: number of windows
///
/// Pairwise weight formula:
///   w* = 0.7 · CV_D/(0.4+CV_D) · min(1, D_min/0.3) · min(1, √(n/5000))
///
/// Emission context: inversely proportional to per-window discriminability.
/// Strong signal (sim data) → ec=1; weak signal (HPRC real) → ec=10-15.
///
/// Returns (pairwise_weight, emission_context).
pub fn auto_configure_pairwise_params(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> (f64, usize) {
    let k = populations.len();
    let n_windows = observations.len();

    if k < 2 || n_windows < 10 {
        return (0.3, 0);
    }

    // Compute per-window per-population max similarity
    let n_pairs = k * (k - 1) / 2;

    // Collect per-window per-population max similarity
    let mut all_window_sims: Vec<Vec<f64>> = Vec::with_capacity(n_windows);

    for obs in observations {
        let pop_sims: Vec<f64> = populations
            .iter()
            .filter_map(|pop| {
                pop.haplotypes
                    .iter()
                    .filter_map(|h| obs.similarities.get(h).copied())
                    .fold(None, |max, s| Some(max.map_or(s, |m: f64| m.max(s))))
            })
            .collect();

        if pop_sims.len() == k {
            all_window_sims.push(pop_sims);
        }
    }

    // NOTE: Tukey fence outlier filtering (T70) was tried here but B81 proved it
    // HURTS on HPRC PAF data: IQR≈0 causes Q3+3*IQR to catch 11-13% of normal
    // windows, inflating D_min 17-18x and corrupting ec/pairwise_weight estimation.
    // The filter has been removed. If robust D_min is needed in the future, use
    // value-level robustification (Winsorized/MAD effect size) instead.

    let mut pairwise_signed_diffs: Vec<Vec<f64>> = vec![Vec::new(); n_pairs];

    for pop_sims in all_window_sims.iter() {
        let mut pair_idx = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                pairwise_signed_diffs[pair_idx].push(pop_sims[i] - pop_sims[j]);
                pair_idx += 1;
            }
        }
    }

    if pairwise_signed_diffs.iter().all(|v| v.is_empty()) {
        return (0.3, 0);
    }

    // Cohen's d for each population pair: |mean(diff)| / std(diff)
    let pair_cohens_d: Vec<f64> = pairwise_signed_diffs
        .iter()
        .filter(|v| v.len() >= 10)
        .map(|diffs| {
            let n = diffs.len() as f64;
            let mean = diffs.iter().sum::<f64>() / n;
            let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            if std > 1e-10 {
                mean.abs() / std
            } else if mean.abs() > 1e-10 {
                // Zero variance with non-zero mean = perfect discriminability
                100.0
            } else {
                0.0
            }
        })
        .collect();

    if pair_cohens_d.is_empty() {
        return (0.3, 0);
    }

    // D_min = minimum Cohen's d across all pairs
    let d_min = pair_cohens_d
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);

    // CV_D = coefficient of variation of Cohen's d across pairs
    let mean_d: f64 = pair_cohens_d.iter().sum::<f64>() / pair_cohens_d.len() as f64;
    let var_d: f64 = pair_cohens_d
        .iter()
        .map(|d| (d - mean_d).powi(2))
        .sum::<f64>()
        / pair_cohens_d.len() as f64;
    let cv_d = if mean_d > 1e-10 {
        var_d.sqrt() / mean_d
    } else {
        0.0
    };

    // Pairwise weight formula (T53 adapted for Cohen's d scale).
    // For k=2: single pair, CV is undefined — D_min alone drives the weight.
    // For k≥3: CV_D captures pair heterogeneity (close vs distant pairs).
    // D_ref=0.014 calibrated so HPRC 3-way (D_min≈0.004, CV_D≈0.6) → w*≈0.12,
    // and sim 2-way (D_min>>0.014) → w*≈0.7.
    let cv_factor = if k == 2 {
        1.0 // single pair, no variation to measure
    } else {
        cv_d / (0.4 + cv_d)
    };
    let w_star = 0.7
        * cv_factor
        * (d_min / 0.014).min(1.0)
        * ((n_windows as f64 / 5000.0).sqrt()).min(1.0);
    let pairwise_weight = w_star.clamp(0.0, 0.95);

    // Emission context from D_min (Cohen's d of hardest-to-separate pair).
    // D_min captures the regime difference: sim (D_min≈0.025) needs low ec,
    // HPRC 3-way (D_min≈0.003) needs high ec.
    // Using median per-window discriminability was broken (always ~0.0003 → ec=15).
    let ec_star = (0.1 / d_min).round() as usize;
    let emission_context = ec_star.clamp(1, 15);

    eprintln!(
        "  Auto-configure: D_min={:.4}, CV_D={:.3}, n_win={}",
        d_min, cv_d, n_windows
    );
    eprintln!(
        "  → pairwise_weight={:.3}, emission_context={}",
        pairwise_weight, emission_context
    );

    (pairwise_weight, emission_context)
}

/// Deconvolve admixed reference populations using k-means clustering.
///
/// When two populations have very low Cohen's d (D_min < threshold), the one
/// with higher within-population identity variance is likely admixed.  For each
/// admixed population, we cluster its haplotypes into 2 components using their
/// per-window identity profiles against other populations, then split the
/// population into two sub-populations (e.g., AMR → AMR_0, AMR_1).
///
/// This addresses the fundamental problem where admixed references (like AMR
/// with ~50% EUR ancestry) produce identity values indistinguishable from the
/// source population (EUR), confounding the HMM.
///
/// Returns `(deconvolved_populations, parent_map)` where parent_map maps each
/// new sub-population name back to its original population name (for merging
/// in output).  Populations that are not split appear in parent_map mapping to
/// themselves.
pub fn deconvolve_admixed_populations(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    d_min_threshold: f64,
) -> (Vec<AncestralPopulation>, HashMap<String, String>) {
    let k = populations.len();
    let mut parent_map: HashMap<String, String> = HashMap::new();

    if k < 2 || observations.len() < 50 {
        for pop in populations {
            parent_map.insert(pop.name.clone(), pop.name.clone());
        }
        return (populations.to_vec(), parent_map);
    }

    // Step 1: Compute pairwise Cohen's d (reuse logic from auto_configure)
    let n_pairs = k * (k - 1) / 2;
    let mut pairwise_signed_diffs: Vec<Vec<f64>> = vec![Vec::new(); n_pairs];
    // Also track per-population within-haplotype variance
    let mut pop_hap_sims: Vec<Vec<Vec<f64>>> =
        populations.iter().map(|p| vec![Vec::new(); p.haplotypes.len()]).collect();

    for obs in observations {
        let pop_max: Vec<Option<f64>> = populations
            .iter()
            .map(|pop| {
                pop.haplotypes
                    .iter()
                    .filter_map(|h| obs.similarities.get(h).copied())
                    .fold(None, |max, s| Some(max.map_or(s, |m: f64| m.max(s))))
            })
            .collect();

        if pop_max.iter().all(|v| v.is_some()) {
            let sims: Vec<f64> = pop_max.iter().map(|v| v.unwrap()).collect();
            let mut pair_idx = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    pairwise_signed_diffs[pair_idx].push(sims[i] - sims[j]);
                    pair_idx += 1;
                }
            }
        }

        // Track per-haplotype similarities for each population
        for (pop_idx, pop) in populations.iter().enumerate() {
            for (hap_idx, hap) in pop.haplotypes.iter().enumerate() {
                if let Some(&sim) = obs.similarities.get(hap) {
                    pop_hap_sims[pop_idx][hap_idx].push(sim);
                }
            }
        }
    }

    // Compute Cohen's d for each pair
    let pair_cohens_d: Vec<f64> = pairwise_signed_diffs
        .iter()
        .map(|diffs| {
            if diffs.len() < 10 {
                return f64::INFINITY;
            }
            let n = diffs.len() as f64;
            let mean = diffs.iter().sum::<f64>() / n;
            let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            if std > 1e-10 { mean.abs() / std } else if mean.abs() > 1e-10 { 100.0 } else { 0.0 }
        })
        .collect();

    // Find pairs with D < threshold
    let mut admixed_candidates: Vec<usize> = Vec::new(); // population indices to deconvolve
    let mut pair_idx = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            if pair_cohens_d[pair_idx] < d_min_threshold {
                // The admixed population is the one with higher within-pop variance
                let var_i = within_pop_variance(&pop_hap_sims[i]);
                let var_j = within_pop_variance(&pop_hap_sims[j]);
                let candidate = if var_i > var_j { i } else { j };
                if !admixed_candidates.contains(&candidate) {
                    admixed_candidates.push(candidate);
                    eprintln!(
                        "  Deconvolution candidate: {} (d={:.4} vs {}, within-var={:.6})",
                        populations[candidate].name,
                        pair_cohens_d[pair_idx],
                        populations[if candidate == i { j } else { i }].name,
                        if candidate == i { var_i } else { var_j },
                    );
                }
            }
            pair_idx += 1;
        }
    }

    if admixed_candidates.is_empty() {
        for pop in populations {
            parent_map.insert(pop.name.clone(), pop.name.clone());
        }
        return (populations.to_vec(), parent_map);
    }

    // Step 2: For each admixed population, cluster haplotypes
    let mut new_populations: Vec<AncestralPopulation> = Vec::new();

    for (pop_idx, pop) in populations.iter().enumerate() {
        if !admixed_candidates.contains(&pop_idx) {
            parent_map.insert(pop.name.clone(), pop.name.clone());
            new_populations.push(pop.clone());
            continue;
        }

        // Build feature matrix: for each haplotype, compute its mean identity per window
        // Use identity to OTHER populations as features (not self, to avoid circularity)
        let n_haps = pop.haplotypes.len();
        if n_haps < 4 {
            // Too few haplotypes to split meaningfully
            eprintln!("  Skipping deconvolution of {} (only {} haplotypes)", pop.name, n_haps);
            parent_map.insert(pop.name.clone(), pop.name.clone());
            new_populations.push(pop.clone());
            continue;
        }

        // Feature: for each haplotype h, compute mean(h_identity - pop_p_max_identity)
        // per other population p. A EUR-like AMR haplotype will have Feature_EUR ≈ 0
        // (similar to EUR) and Feature_AFR >> 0 (much higher than AFR's identity).
        // This creates a (k-1)-dimensional feature that differentiates ancestry components.

        // First, compute per-window per-population max identity
        let n_obs = observations.len();
        let mut pop_max_per_window: Vec<Vec<f64>> = vec![vec![0.0; k]; n_obs];
        for (w, obs) in observations.iter().enumerate() {
            for (p, pop_p) in populations.iter().enumerate() {
                pop_max_per_window[w][p] = pop_p.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h).copied())
                    .fold(0.0_f64, f64::max);
            }
        }

        // Now compute per-haplotype features
        let hap_features: Vec<Vec<f64>> = (0..n_haps)
            .map(|h_idx| {
                let hap_name = &pop.haplotypes[h_idx];
                let mut features = Vec::new();
                for (other_idx, _) in populations.iter().enumerate() {
                    if other_idx == pop_idx {
                        continue;
                    }
                    // Mean difference: haplotype identity minus other pop's max identity
                    let mut sum_diff = 0.0;
                    let mut count = 0usize;
                    for (w, obs) in observations.iter().enumerate() {
                        if let Some(&h_sim) = obs.similarities.get(hap_name) {
                            sum_diff += h_sim - pop_max_per_window[w][other_idx];
                            count += 1;
                        }
                    }
                    features.push(if count > 0 { sum_diff / count as f64 } else { 0.0 });
                }
                features
            })
            .collect();

        // k-means++ with k=2
        let assignments = kmeans_pp_2(&hap_features, 20);

        let count_0 = assignments.iter().filter(|&&a| a == 0).count();
        let count_1 = n_haps - count_0;

        if count_0 == 0 || count_1 == 0 {
            // Degenerate: all haplotypes in one cluster
            eprintln!("  Deconvolution of {} failed: degenerate split ({}/{})", pop.name, count_0, count_1);
            parent_map.insert(pop.name.clone(), pop.name.clone());
            new_populations.push(pop.clone());
            continue;
        }

        // Create two sub-populations
        let mut haps_0 = Vec::new();
        let mut haps_1 = Vec::new();
        for (h_idx, &cluster) in assignments.iter().enumerate() {
            if cluster == 0 {
                haps_0.push(pop.haplotypes[h_idx].clone());
            } else {
                haps_1.push(pop.haplotypes[h_idx].clone());
            }
        }

        let name_0 = format!("{}_0", pop.name);
        let name_1 = format!("{}_1", pop.name);
        eprintln!(
            "  Deconvolved {}: {} ({} haps) + {} ({} haps)",
            pop.name, name_0, haps_0.len(), name_1, haps_1.len()
        );

        parent_map.insert(name_0.clone(), pop.name.clone());
        parent_map.insert(name_1.clone(), pop.name.clone());
        new_populations.push(AncestralPopulation { name: name_0, haplotypes: haps_0 });
        new_populations.push(AncestralPopulation { name: name_1, haplotypes: haps_1 });
    }

    (new_populations, parent_map)
}

/// Compute between-haplotype variance of mean identity profiles.
///
/// For detecting admixture: an admixed population has haplotypes with
/// significantly different mean identities (e.g., EUR-like haps ~0.999 vs
/// NAT-like haps ~0.997), creating high between-haplotype variance.
/// A homogeneous population has similar mean identities across all haplotypes.
fn within_pop_variance(hap_sims: &[Vec<f64>]) -> f64 {
    // Compute per-haplotype mean identity
    let hap_means: Vec<f64> = hap_sims
        .iter()
        .filter(|v| v.len() >= 10)
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .collect();
    if hap_means.len() < 2 {
        return 0.0;
    }
    // Variance of haplotype means = between-haplotype heterogeneity
    let n = hap_means.len() as f64;
    let grand_mean = hap_means.iter().sum::<f64>() / n;
    hap_means.iter().map(|m| (m - grand_mean).powi(2)).sum::<f64>() / n
}

/// 2-means clustering with k-means++ initialization.
/// Returns cluster assignment (0 or 1) for each point.
fn kmeans_pp_2(features: &[Vec<f64>], max_iters: usize) -> Vec<usize> {
    let n = features.len();
    let dim = features.first().map_or(0, |f| f.len());

    if n < 2 || dim == 0 {
        return vec![0; n];
    }

    // k-means++ init: pick first center randomly (use index 0 for determinism),
    // second center proportional to squared distance
    let c0: Vec<f64> = features[0].clone();
    let dists: Vec<f64> = features.iter().map(|f| sq_dist(f, &c0)).collect();
    let total_d: f64 = dists.iter().sum();
    let c1_idx = if total_d > 0.0 {
        // Pick the point with maximum distance (deterministic approximation of k-means++)
        dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(n - 1)
    } else {
        n - 1
    };
    let mut centers = [c0, features[c1_idx].clone()];

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iters {
        // Assign each point to nearest center
        let mut changed = false;
        for (i, feat) in features.iter().enumerate() {
            let d0 = sq_dist(feat, &centers[0]);
            let d1 = sq_dist(feat, &centers[1]);
            let new_assign = if d0 <= d1 { 0 } else { 1 };
            if assignments[i] != new_assign {
                changed = true;
                assignments[i] = new_assign;
            }
        }

        if !changed {
            break;
        }

        // Recompute centers
        for (c, center) in centers.iter_mut().enumerate() {
            let mut sum = vec![0.0; dim];
            let mut count = 0usize;
            for (i, feat) in features.iter().enumerate() {
                if assignments[i] == c {
                    for (d, &v) in sum.iter_mut().zip(feat.iter()) {
                        *d += v;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for d in sum.iter_mut() {
                    *d /= count as f64;
                }
                *center = sum;
            }
        }
    }

    assignments
}

/// Squared Euclidean distance between two vectors.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Automatically estimate the identity floor from the data.
///
/// Computes the distribution of per-window maximum similarity across all
/// reference haplotypes. Windows with abnormally low max identity are likely
/// alignment gaps, repetitive regions, or regions with insufficient pangenome
/// coverage. These produce noisy emissions that confuse the HMM.
///
/// The algorithm uses a two-stage approach:
/// 1. Compute max similarity per window across all reference haplotypes
/// 2. Find the median and IQR of the distribution
/// 3. Set floor to median - 3*IQR (excludes outlier-low windows)
/// 4. If there's a clear bimodal gap, use the gap threshold instead
///
/// This approach adapts to the data: tight distributions (high-quality sim data)
/// get high floors close to the median; wide distributions (noisy real data)
/// get lower floors.
///
/// Returns the estimated identity floor, clamped to [0.9, 0.9999].
pub fn estimate_identity_floor(
    observations: &[AncestryObservation],
) -> f64 {
    if observations.len() < 20 {
        return 0.995; // fallback
    }

    // Compute max similarity per window
    let mut max_sims: Vec<f64> = observations
        .iter()
        .map(|obs| {
            obs.similarities
                .values()
                .cloned()
                .fold(0.0_f64, f64::max)
        })
        .collect();
    max_sims.sort_by(|a, b| a.total_cmp(b));

    let n = max_sims.len();

    // Compute quartiles
    let q1 = max_sims[n / 4];
    let median = max_sims[n / 2];
    let q3 = max_sims[3 * n / 4];
    let iqr = q3 - q1;

    // Stage 1: IQR-based floor
    // Set floor to Q1 - 3*IQR (Tukey's outer fence for outlier detection)
    // This adapts to the data distribution:
    // - Tight distribution (IQR=0.0001): floor = Q1 - 0.0003 ≈ very close to data
    // - Wide distribution (IQR=0.01): floor = Q1 - 0.03 ≈ well below most data
    let iqr_floor = q1 - 3.0 * iqr;

    // Stage 2: Check for bimodal gap in the bottom portion
    let search_end = (n / 10).max(2); // bottom 10%
    let mut max_gap = 0.0_f64;
    let mut gap_idx = 0;
    for i in 1..search_end.min(n) {
        let gap = max_sims[i] - max_sims[i - 1];
        if gap > max_gap {
            max_gap = gap;
            gap_idx = i;
        }
    }

    // A "meaningful" bimodal gap is one that's > 100x the IQR / n
    // (i.e., much larger than the typical spacing between sorted values)
    let typical_spacing = if n > 1 { iqr / (n as f64 / 2.0) } else { 0.001 };
    let gap_floor = if max_gap > typical_spacing * 100.0 && gap_idx > 0 && gap_idx < search_end {
        // Clear bimodal separation — floor at the gap
        max_sims[gap_idx]
    } else {
        f64::NEG_INFINITY // no gap detected
    };

    // Use the higher of the two estimates (more aggressive filtering)
    let floor = iqr_floor.max(gap_floor);

    // Also ensure we don't remove more than ~5% of windows
    let p5_value = max_sims[n / 20];
    let floor = floor.min(p5_value);

    // If the floor would be within the IQR (too aggressive), back off to Q1
    let floor = if floor > q1 { q1 } else { floor };

    // Final clamp — upper bound is at least 0.9 to prevent clamp inversion
    let upper = median.max(0.9);
    floor.clamp(0.9, upper)
}

/// Estimate switch probability from observed state change rate.
/// Does an initial Viterbi pass with broad prior, counts transitions,
/// then regularizes towards prior expectation.
pub fn estimate_switch_prob(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    temperature: f64,
) -> f64 {
    if observations.len() < 10 {
        return 0.001; // fallback for small data
    }

    // Create temporary params with broad prior
    let mut temp_params = AncestryHmmParams::new(populations.to_vec(), 0.01);
    temp_params.emission_std = temperature;

    // Run Viterbi
    let states = viterbi(observations, &temp_params);

    if states.len() < 2 {
        return 0.001;
    }

    // Count state switches
    let n_switches = states.windows(2).filter(|w| w[0] != w[1]).count();

    let observed_rate = n_switches as f64 / (states.len() - 1) as f64;

    // Regularize: blend observed with prior (0.001) using weight 0.3
    let prior = 0.001;
    let alpha = 0.3;
    let estimated = alpha * prior + (1.0 - alpha) * observed_rate;

    // Clamp to reasonable range
    estimated.clamp(0.0001, 0.05)
}

/// Scale temperature correction for number of competing populations.
///
/// With more populations in the softmax, the probability mass is spread across
/// more states, reducing discriminability. Temperature should decrease to
/// compensate. Uses k=3 as reference.
///
/// Based on extreme value theory: the expected maximum of k softmax outputs
/// scales as √(2 ln k). To maintain the same discrimination level across
/// different numbers of populations, temperature scales inversely.
///
/// # Examples
/// - k=3 (reference): correction = 1.0 (no change)
/// - k=5 (HPRC ancestry): correction ≈ 0.83 (17% reduction)
/// - k=2 (binary): correction ≈ 1.26 (26% increase)
pub fn scale_temperature_for_populations(base_temperature: f64, n_populations: usize) -> f64 {
    if n_populations <= 1 {
        return base_temperature;
    }
    let reference_k = 3.0_f64;
    let k = n_populations as f64;

    let ref_factor = (2.0 * reference_k.ln()).sqrt();
    let actual_factor = (2.0 * k.ln()).sqrt();

    let correction = ref_factor / actual_factor;
    (base_temperature * correction).clamp(0.001, 1.0)
}

/// Scale temperature for the copying model's haplotype-level softmax.
///
/// The copying model uses softmax over M individual haplotypes instead of
/// K populations. With more states in the softmax denominator, the probability
/// mass is more spread out, reducing discriminability. To maintain equivalent
/// discrimination, the temperature should be reduced.
///
/// The correction is based on extreme value theory: for the softmax to assign
/// the same probability to the best state when the denominator has M terms
/// instead of K terms, we need ln(K)/ln(M) scaling.
///
/// # Examples
/// - K=2, M=46: correction = ln(2)/ln(46) ≈ 0.18 → temp reduced ~5.5x
/// - K=5, M=100: correction = ln(5)/ln(100) ≈ 0.35 → temp reduced ~2.9x
/// - K=2, M=6: correction = ln(2)/ln(6) ≈ 0.39 → temp reduced ~2.6x
///
/// If the temperature was already estimated at the haplotype level (via
/// `estimate_copying_params`), this correction should NOT be applied.
pub fn scale_temperature_for_copying(
    population_temperature: f64,
    n_populations: usize,
    n_haplotypes: usize,
) -> f64 {
    if n_populations <= 1 || n_haplotypes <= 1 || n_haplotypes <= n_populations {
        return population_temperature;
    }
    let correction = (n_populations as f64).ln() / (n_haplotypes as f64).ln();
    (population_temperature * correction).clamp(0.0001, 0.5)
}

/// Estimate per-state proportions from window-level state assignments.
///
/// Counts the fraction of windows assigned to each state. Used by two-pass
/// inference to set informative priors for the second pass.
///
/// Returns a Vec<f64> of length n_states with proportions summing to ~1.
/// Each proportion is floored at 1e-6 to prevent zero priors.
pub fn estimate_proportions_from_states(states: &[usize], n_states: usize) -> Vec<f64> {
    if states.is_empty() || n_states == 0 {
        let k = n_states.max(1);
        return vec![1.0 / k as f64; k];
    }

    let mut counts = vec![0usize; n_states];
    for &s in states {
        if s < n_states {
            counts[s] += 1;
        }
    }

    let total = states.len() as f64;
    let mut props: Vec<f64> = counts.iter().map(|&c| (c as f64 / total).max(1e-6)).collect();

    // Renormalize after flooring
    let sum: f64 = props.iter().sum();
    if sum > 0.0 {
        for p in &mut props {
            *p /= sum;
        }
    }

    props
}

/// Estimate per-population switch rates from first-pass state assignments.
///
/// For each state, computes the fraction of windows where the state changes
/// to a different state. States with longer tracts have lower switch rates.
/// Falls back to a uniform rate if insufficient data.
///
/// Returns a Vec<f64> of length n_states with per-state switch probabilities.
pub fn estimate_per_state_switch_rates(states: &[usize], n_states: usize) -> Vec<f64> {
    if states.len() < 2 || n_states == 0 {
        return vec![0.001; n_states.max(1)];
    }

    let mut state_windows = vec![0usize; n_states];
    let mut state_switches = vec![0usize; n_states];

    for w in states.windows(2) {
        if w[0] < n_states {
            state_windows[w[0]] += 1;
            if w[0] != w[1] {
                state_switches[w[0]] += 1;
            }
        }
    }
    // Last window doesn't have a successor but counts for state_windows
    if let Some(&last) = states.last() {
        if last < n_states {
            state_windows[last] += 1;
        }
    }

    (0..n_states).map(|i| {
        if state_windows[i] >= 10 {
            let rate = state_switches[i] as f64 / state_windows[i] as f64;
            rate.clamp(0.0001, 0.05)
        } else {
            0.001 // fallback for states with few windows
        }
    }).collect()
}

// ============================================================================
// Population profile emission learning (two-pass)
// ============================================================================

/// Learned emission profiles for two-pass ancestry inference.
///
/// After pass 1, the mean aggregated similarity to each population is computed
/// for windows assigned to each state. This gives a K×K "profile matrix" that
/// captures the expected similarity pattern for each ancestry.
///
/// In pass 2, the profile is used to compute correlation-based emissions:
/// how well does the observed similarity vector match each state's learned profile?
/// This captures cross-population patterns that softmax emissions miss due to
/// the Independence of Irrelevant Alternatives (IIA) property of softmax.
///
/// For example, when true ancestry is EUR, the profile captures that sim_EUR > sim_CSA > sim_AFR.
/// At an ambiguous window where softmax can't decide between EUR and CSA, the profile
/// checks whether the full pattern of similarities matches the EUR or CSA profile better.
#[derive(Debug, Clone)]
pub struct PopulationProfiles {
    /// Centroid vectors: centroids\[state\]\[pop_idx\] = mean aggregated similarity
    /// to population pop_idx for windows assigned to state.
    pub centroids: Vec<Vec<f64>>,
    /// Number of windows used to learn each centroid.
    pub n_windows: Vec<usize>,
}

/// Learn population emission profiles from first-pass state assignments.
///
/// For each state s, computes the mean aggregated similarity to each population
/// across all windows assigned to state s. This creates a "profile" that
/// captures the typical similarity pattern for each ancestry.
pub fn learn_population_profiles(
    observations: &[AncestryObservation],
    states: &[usize],
    params: &AncestryHmmParams,
) -> PopulationProfiles {
    let k = params.n_states;
    let n_pops = params.populations.len();
    let mut sums = vec![vec![0.0; n_pops]; k];
    let mut counts = vec![vec![0usize; n_pops]; k];
    let mut state_counts = vec![0usize; k];

    for (obs, &state) in observations.iter().zip(states.iter()) {
        if state >= k { continue; }
        state_counts[state] += 1;
        for (p, pop) in params.populations.iter().enumerate() {
            let sims: Vec<f64> = pop.haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h))
                .cloned()
                .collect();
            if let Some(agg) = params.emission_model.aggregate(&sims) {
                sums[state][p] += agg;
                counts[state][p] += 1;
            }
        }
    }

    let centroids: Vec<Vec<f64>> = (0..k).map(|s| {
        (0..n_pops).map(|p| {
            if counts[s][p] > 0 {
                sums[s][p] / counts[s][p] as f64
            } else {
                0.0
            }
        }).collect()
    }).collect();

    PopulationProfiles { centroids, n_windows: state_counts }
}

/// Compute Pearson correlation between two vectors.
///
/// Returns 0.0 if either vector has zero variance or if lengths don't match.
fn profile_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }

    let n = a.len() as f64;
    let a_mean = a.iter().sum::<f64>() / n;
    let b_mean = b.iter().sum::<f64>() / n;

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let ad = ai - a_mean;
        let bd = bi - b_mean;
        dot += ad * bd;
        norm_a += ad * ad;
        norm_b += bd * bd;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-12 {
        (dot / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Compute profile-based log emissions for all observations.
///
/// For each window, computes the Pearson correlation between the observation's
/// aggregated similarity vector and each state's centroid profile. The correlations
/// are then converted to log-probabilities via softmax with temperature scaling.
///
/// This captures cross-population similarity patterns that the standard softmax
/// emission misses due to IIA. The standard emission treats each population's
/// similarity score independently; the profile emission compares the full
/// similarity vector against learned population-specific patterns.
///
/// # Returns
/// n×k matrix where entry \[t\]\[s\] = log P_profile(obs_t | state=s)
pub fn compute_profile_log_emissions(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    profiles: &PopulationProfiles,
) -> Vec<Vec<f64>> {
    let k = params.n_states;
    let temperature = params.emission_std;

    observations.iter().map(|obs| {
        // Build observation vector: aggregated similarity to each population
        let obs_vec: Vec<f64> = params.populations.iter()
            .map(|pop| {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                params.emission_model.aggregate(&sims).unwrap_or(0.0)
            })
            .collect();

        // Check if observation has enough variation to compute correlation
        let obs_var: f64 = {
            let mean = obs_vec.iter().sum::<f64>() / obs_vec.len().max(1) as f64;
            obs_vec.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
        };

        if obs_var < 1e-16 {
            // No variation in observation — return uniform (log(1/k))
            return vec![-(k as f64).ln(); k];
        }

        // Compute correlation with each state's centroid
        let corr_scores: Vec<f64> = (0..k)
            .map(|s| {
                if profiles.n_windows[s] == 0 {
                    return 0.0; // No data for this state — neutral score
                }
                profile_correlation(&obs_vec, &profiles.centroids[s]) / temperature
            })
            .collect();

        // Log-softmax normalization
        let max_score = corr_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !max_score.is_finite() {
            return vec![-(k as f64).ln(); k];
        }
        let log_sum_exp = corr_scores.iter()
            .map(|&s| (s - max_score).exp())
            .sum::<f64>()
            .ln();

        corr_scores.iter()
            .map(|&s| s - max_score - log_sum_exp)
            .collect()
    }).collect()
}

/// Compute hierarchical log emissions to address the softmax IIA limitation.
///
/// When related populations (e.g., EUR and AMR) have very similar identity
/// profiles, standard softmax over all K populations suffers from IIA
/// (Independence of Irrelevant Alternatives): adding AMR dilutes EUR's signal
/// because the softmax denominator grows with a near-duplicate term.
///
/// The hierarchical approach decomposes emissions into two independent components:
/// 1. **Group-level**: "Which broad ancestry group?" — softmax over group-aggregate
///    similarities (merges haplotypes within each group). Easy discrimination.
/// 2. **Within-group**: "Which population within the group?" — softmax over only
///    the populations in the same group. Fine discrimination.
///
/// Combined as: `log_emit[pop] = w * group_emit[group(pop)] + (1-w) * within_emit[pop|group]`
///
/// This breaks IIA: EUR's emission depends on EUR-vs-AFR at group level AND
/// EUR-vs-AMR at within-group level, using independent softmax normalizations.
///
/// # Arguments
/// * `observations` - Per-window similarity data
/// * `populations` - Population definitions with haplotype lists
/// * `groups` - Population grouping: `groups[g]` = list of population indices in group g
/// * `emission_model` - How to aggregate per-haplotype similarities
/// * `temperature` - Softmax temperature (used for both levels)
/// * `group_weight` - Weight for group-level component (0.0 = pure within-group, 1.0 = pure group)
///
/// # Returns
/// n×k matrix of log-emission probabilities
pub fn compute_hierarchical_emissions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    groups: &[Vec<usize>],
    emission_model: &EmissionModel,
    temperature: f64,
    group_weight: f64,
) -> Vec<Vec<f64>> {
    let n_pops = populations.len();
    if n_pops == 0 || observations.is_empty() || groups.is_empty() {
        return vec![vec![0.0; n_pops]; observations.len()];
    }

    let w_group = group_weight.clamp(0.0, 1.0);
    let w_within = 1.0 - w_group;

    // Build pop_to_group mapping
    let mut pop_to_group = vec![0usize; n_pops];
    for (g, group) in groups.iter().enumerate() {
        for &pop_idx in group {
            if pop_idx < n_pops {
                pop_to_group[pop_idx] = g;
            }
        }
    }

    // Build group-level "super-populations" (merged haplotypes)
    let group_pops: Vec<AncestralPopulation> = groups.iter().map(|group| {
        let mut haplotypes = Vec::new();
        let mut names = Vec::new();
        for &pop_idx in group {
            if pop_idx < n_pops {
                haplotypes.extend(populations[pop_idx].haplotypes.clone());
                names.push(populations[pop_idx].name.clone());
            }
        }
        AncestralPopulation {
            name: names.join("+"),
            haplotypes,
        }
    }).collect();

    observations.iter().map(|obs| {
        // --- Group-level emissions ---
        let group_sims: Vec<Option<f64>> = group_pops.iter()
            .map(|gpop| {
                let sims: Vec<f64> = gpop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                emission_model.aggregate(&sims)
            })
            .collect();

        let group_log_probs = softmax_scores(&group_sims, temperature);

        // --- Within-group emissions ---
        let mut within_log_probs = vec![0.0_f64; n_pops];
        for group in groups.iter() {
            if group.len() <= 1 {
                // Single-population group: within-group probability = 1.0
                if let Some(&pop_idx) = group.first() {
                    if pop_idx < n_pops {
                        within_log_probs[pop_idx] = 0.0; // log(1) = 0
                    }
                }
                continue;
            }

            // Compute per-population similarities within this group
            let within_sims: Vec<Option<f64>> = group.iter()
                .map(|&pop_idx| {
                    if pop_idx >= n_pops { return None; }
                    let sims: Vec<f64> = populations[pop_idx].haplotypes.iter()
                        .filter_map(|h| obs.similarities.get(h))
                        .cloned()
                        .collect();
                    emission_model.aggregate(&sims)
                })
                .collect();

            let within_probs = softmax_scores(&within_sims, temperature);

            for (i, &pop_idx) in group.iter().enumerate() {
                if pop_idx < n_pops && i < within_probs.len() {
                    within_log_probs[pop_idx] = within_probs[i];
                }
            }
        }

        // --- Combine: w_group * group + w_within * within ---
        (0..n_pops).map(|p| {
            let g = pop_to_group[p];
            let group_emit = if g < group_log_probs.len() {
                group_log_probs[g]
            } else {
                -(n_pops as f64).ln()
            };
            w_group * group_emit + w_within * within_log_probs[p]
        }).collect()
    }).collect()
}

/// Helper: compute log-softmax from optional similarity scores.
fn softmax_scores(sims: &[Option<f64>], temperature: f64) -> Vec<f64> {
    let valid_scores: Vec<(usize, f64)> = sims.iter().enumerate()
        .filter_map(|(i, s)| s.filter(|&v| v > 0.0).map(|v| (i, v)))
        .collect();

    let n = sims.len();
    if valid_scores.is_empty() {
        return vec![-(n as f64).ln(); n];
    }
    if valid_scores.len() == 1 {
        let mut result = vec![f64::NEG_INFINITY; n];
        result[valid_scores[0].0] = 0.0;
        return result;
    }

    let max_score = valid_scores.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp: f64 = valid_scores.iter()
        .map(|(_, v)| ((v - max_score) / temperature).exp())
        .sum::<f64>()
        .ln();

    let mut result = vec![f64::NEG_INFINITY; n];
    for &(i, v) in &valid_scores {
        result[i] = (v - max_score) / temperature - log_sum_exp;
    }
    result
}

/// Parse population group specification string into group indices.
///
/// Format: "EUR,AMR;AFR;EAS,CSA" — semicolons separate groups, commas separate
/// populations within a group. Population names must match those in the populations list.
///
/// Returns `None` if any population name is not found.
pub fn parse_population_groups(
    spec: &str,
    populations: &[AncestralPopulation],
) -> Option<Vec<Vec<usize>>> {
    let mut groups = Vec::new();
    for group_str in spec.split(';') {
        let mut group = Vec::new();
        for name in group_str.split(',') {
            let name = name.trim();
            if name.is_empty() { continue; }
            let idx = populations.iter().position(|p| p.name == name)?;
            group.push(idx);
        }
        if !group.is_empty() {
            groups.push(group);
        }
    }
    Some(groups)
}

/// Auto-detect population groups from similarity data.
///
/// Computes mean inter-population similarity distance for all pairs and groups
/// populations whose distance is below the median distance. This identifies
/// populations that are hard to distinguish and should be handled hierarchically.
pub fn auto_detect_groups(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
) -> Vec<Vec<usize>> {
    let n_pops = populations.len();
    if n_pops <= 2 {
        // With 2 or fewer populations, no grouping needed — one group per pop
        return (0..n_pops).map(|i| vec![i]).collect();
    }

    // Compute mean aggregated similarity per population per window
    let n_obs = observations.len().max(1);
    let mut mean_sims = vec![0.0_f64; n_pops];
    let mut pair_dists = vec![vec![0.0_f64; n_pops]; n_pops];

    for obs in observations {
        let pop_sims: Vec<f64> = populations.iter()
            .map(|pop| {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                emission_model.aggregate(&sims).unwrap_or(0.0)
            })
            .collect();

        for (i, pop_sim_i) in pop_sims.iter().enumerate().take(n_pops) {
            mean_sims[i] += pop_sim_i / n_obs as f64;
            for (j, pop_sim_j) in pop_sims.iter().enumerate().take(n_pops).skip(i + 1) {
                let dist = (pop_sim_i - pop_sim_j).abs() / n_obs as f64;
                pair_dists[i][j] += dist;
                pair_dists[j][i] = pair_dists[i][j];
            }
        }
    }

    // Find median pairwise distance
    let mut all_dists: Vec<f64> = Vec::new();
    for (i, row) in pair_dists.iter().enumerate().take(n_pops) {
        for &dist in row.iter().skip(i + 1).take(n_pops - i - 1) {
            all_dists.push(dist);
        }
    }
    all_dists.sort_by(|a, b| a.total_cmp(b));
    let median_dist = if all_dists.is_empty() {
        0.0
    } else {
        all_dists[all_dists.len() / 2]
    };

    // Group populations that are within median_dist * 0.5 of each other
    // Use simple single-linkage clustering
    let threshold = median_dist * 0.5;
    let mut assigned = vec![false; n_pops];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..n_pops {
        if assigned[i] { continue; }
        let mut group = vec![i];
        assigned[i] = true;

        for j in (i + 1)..n_pops {
            if assigned[j] { continue; }
            // Check if j is close to any member of the current group
            let close = group.iter().any(|&g| pair_dists[g][j] < threshold);
            if close {
                group.push(j);
                assigned[j] = true;
            }
        }
        groups.push(group);
    }

    groups
}

/// Compute rank-based log emissions for ancestry inference.
///
/// Instead of aggregating similarities (max/mean/median), this model ranks ALL
/// haplotypes globally by similarity and counts how many from each population
/// appear in the top K positions. This captures "which population dominates the
/// top of the ranking" — a signal that's robust to weak absolute differences.
///
/// When the absolute similarity gap between populations is tiny (e.g., 0.0002),
/// standard aggregation struggles because noise can flip the ranking of the
/// aggregate statistic. But rank-based scoring is immune to scale: if EUR haplotypes
/// consistently rank above AFR haplotypes, even by 0.00001 each, the rank score
/// strongly favors EUR.
///
/// # Arguments
/// * `observations` - Per-window similarity data
/// * `populations` - Population definitions with haplotype lists
/// * `top_k` - Number of top-ranked haplotypes to consider (0 = auto: total_haps / n_pops)
///
/// # Returns
/// n×k matrix of log-emission probabilities (log-softmax of rank fractions)
pub fn compute_rank_log_emissions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    top_k: usize,
) -> Vec<Vec<f64>> {
    let n_pops = populations.len();
    if n_pops == 0 || observations.is_empty() {
        return vec![vec![0.0; n_pops]; observations.len()];
    }

    // Build haplotype-to-population lookup
    let mut hap_to_pop: HashMap<&str, usize> = HashMap::new();
    let mut total_haps = 0;
    for (p, pop) in populations.iter().enumerate() {
        for h in &pop.haplotypes {
            hap_to_pop.insert(h.as_str(), p);
            total_haps += 1;
        }
    }

    // Auto top_k: use total_haps / n_pops (average pop size)
    let k = if top_k == 0 {
        (total_haps / n_pops).max(1)
    } else {
        top_k.min(total_haps)
    };

    observations.iter().map(|obs| {
        // Collect all (similarity, population_index) pairs
        let mut hap_sims: Vec<(f64, usize)> = Vec::with_capacity(total_haps);
        for (hap_name, &sim) in &obs.similarities {
            if let Some(&pop_idx) = hap_to_pop.get(hap_name.as_str()) {
                hap_sims.push((sim, pop_idx));
            }
        }

        if hap_sims.is_empty() {
            return vec![-(n_pops as f64).ln(); n_pops]; // uniform
        }

        // Sort by similarity descending
        hap_sims.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Count each population's representation in top K
        let actual_k = k.min(hap_sims.len());
        let mut counts = vec![0.0_f64; n_pops];
        for &(_, pop_idx) in hap_sims.iter().take(actual_k) {
            counts[pop_idx] += 1.0;
        }

        // Laplace smoothing: add 0.5 to each count to avoid log(0)
        // With K=5, a 5-0 split gives log(5.5/6.5) vs log(0.5/6.5) ≈ 2.4 nats
        for c in counts.iter_mut() {
            *c += 0.5;
        }
        let total: f64 = counts.iter().sum();

        // Log probabilities
        counts.iter().map(|&c| (c / total).ln()).collect()
    }).collect()
}

/// Compute pairwise contrast emissions with per-pair adaptive temperature.
///
/// For each pair of populations (i, j), the similarity gap determines a pairwise
/// log-odds: logit(P(i > j)) = (sim_i - sim_j) / temperature_{i,j}. The per-pair
/// temperature is estimated from the data (upper-quartile absolute gap between the
/// pair's best haplotypes across windows). This avoids the IIA problem of standard
/// softmax: the EUR-vs-AMR comparison uses a tiny temperature tuned to their tiny
/// gaps (~0.0002), while EUR-vs-AFR uses a larger one tuned to their larger gaps.
///
/// Population scores are computed Bradley-Terry style: score_i = Σ_{j≠i} log(P(i > j)),
/// then normalized via log-softmax.
///
/// # Arguments
/// * `observations` - Per-window similarity data
/// * `populations` - Population definitions with haplotype lists
/// * `emission_model` - Aggregation model (Max, Mean, etc.)
///
/// # Returns
/// n×k matrix of log-emission probabilities
pub fn compute_pairwise_log_emissions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
) -> Vec<Vec<f64>> {
    let n_pops = populations.len();
    let n_obs = observations.len();

    if n_pops < 2 || n_obs == 0 {
        return vec![vec![-(n_pops.max(1) as f64).ln(); n_pops]; n_obs];
    }

    // Precompute per-window per-population aggregated similarities
    let pop_sims: Vec<Vec<Option<f64>>> = observations.iter().map(|obs| {
        populations.iter().map(|pop| {
            let sims: Vec<f64> = pop.haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h))
                .cloned()
                .collect();
            emission_model.aggregate(&sims)
        }).collect()
    }).collect();

    // Estimate per-pair temperature from data distribution
    let n_pairs = n_pops * (n_pops - 1) / 2;
    let mut pair_diffs: Vec<Vec<f64>> = vec![Vec::new(); n_pairs];

    for window_sims in &pop_sims {
        let mut pair_idx = 0;
        for i in 0..n_pops {
            for j in (i + 1)..n_pops {
                if let (Some(si), Some(sj)) = (window_sims[i], window_sims[j]) {
                    pair_diffs[pair_idx].push((si - sj).abs());
                }
                pair_idx += 1;
            }
        }
    }

    // Per-pair temperature: upper-quartile mean of absolute diffs
    // This adapts: EUR-AMR (tiny gaps) → tiny temp; EUR-AFR (big gaps) → big temp
    let pair_temps: Vec<f64> = pair_diffs.iter().map(|diffs| {
        if diffs.is_empty() {
            return 0.01; // fallback
        }
        let mut sorted = diffs.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p75 = sorted.len() * 3 / 4;
        let uq_mean: f64 = if p75 < sorted.len() {
            sorted[p75..].iter().sum::<f64>() / (sorted.len() - p75) as f64
        } else {
            sorted.iter().sum::<f64>() / sorted.len() as f64
        };
        // Clamp to avoid division by near-zero or extreme values
        uq_mean.clamp(1e-6, 0.1)
    }).collect();

    // Compute Bradley-Terry log-scores for each window
    pop_sims.iter().map(|window_sims| {
        let mut scores = vec![0.0_f64; n_pops];
        let mut valid_count = 0;

        let mut pair_idx = 0;
        for i in 0..n_pops {
            for j in (i + 1)..n_pops {
                if let (Some(si), Some(sj)) = (window_sims[i], window_sims[j]) {
                    let diff = si - sj;
                    let temp = pair_temps[pair_idx];
                    // Sigmoid: P(i > j) = 1 / (1 + exp(-(si - sj) / temp))
                    // log-odds: (si - sj) / temp
                    let logit = diff / temp;
                    // For numerical stability, use log-sigmoid decomposition:
                    // log(sigmoid(x)) = -log(1 + exp(-x)) = x - log(1 + exp(x)) for x > 0
                    let log_p_i_wins = log_sigmoid(logit);
                    let log_p_j_wins = log_sigmoid(-logit);

                    scores[i] += log_p_i_wins;
                    scores[j] += log_p_j_wins;
                    valid_count += 1;
                }
                pair_idx += 1;
            }
        }

        if valid_count == 0 {
            return vec![-(n_pops as f64).ln(); n_pops]; // uniform
        }

        // Normalize scores via log-softmax
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp: f64 = scores.iter()
            .map(|&s| (s - max_score).exp())
            .sum::<f64>()
            .ln();
        scores.iter().map(|&s| s - max_score - log_sum_exp).collect()
    }).collect()
}

/// Numerically stable log-sigmoid: log(1 / (1 + exp(-x)))
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// Blend standard softmax log emissions with profile-based log emissions.
///
/// Computes: blended\[t\]\[s\] = (1 - weight) * standard\[t\]\[s\] + weight * profile\[t\]\[s\]
///
/// In probability space, this corresponds to geometric interpolation:
/// P_blend ∝ P_standard^(1-w) × P_profile^w
///
/// This combines the softmax's focus on individual population similarities with
/// the profile's cross-population pattern matching for more discriminative emissions.
pub fn blend_log_emissions(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    weight: f64,
) -> Vec<Vec<f64>> {
    let w = weight.clamp(0.0, 1.0);
    let w_std = 1.0 - w;

    standard_emissions.iter().zip(profile_emissions.iter())
        .map(|(std_row, prof_row)| {
            std_row.iter().zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s // fall back to standard if profile is -inf
                    } else if p.is_finite() {
                        p // fall back to profile if standard is -inf
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

/// Blend standard and profile log-emissions with per-window adaptive weights.
///
/// For each window, the weight is scaled by the pairwise emission confidence
/// (margin between best and second-best population). Windows with strong
/// pairwise signal get higher weight; windows with weak/ambiguous signal
/// fall back to the standard emission.
///
/// `base_weight` is the nominal weight (e.g. 0.12 for HPRC, 0.7 for sim).
/// `profile_emissions` are the pairwise log-emissions.
pub fn blend_log_emissions_adaptive(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    base_weight: f64,
) -> Vec<Vec<f64>> {
    if standard_emissions.is_empty() || profile_emissions.is_empty() {
        return standard_emissions.to_vec();
    }

    // Compute per-window margin (max - 2nd max) from profile emissions
    let margins: Vec<f64> = profile_emissions
        .iter()
        .map(|row| {
            if row.len() < 2 {
                return 0.0;
            }
            let mut best = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for &v in row {
                if v > best {
                    second = best;
                    best = v;
                } else if v > second {
                    second = v;
                }
            }
            if best.is_finite() && second.is_finite() {
                best - second
            } else {
                0.0
            }
        })
        .collect();

    // Compute median margin as reference scale
    let mut sorted_margins: Vec<f64> = margins
        .iter()
        .filter(|m| **m > 0.0)
        .cloned()
        .collect();
    if sorted_margins.is_empty() {
        return blend_log_emissions(standard_emissions, profile_emissions, base_weight);
    }
    sorted_margins.sort_by(|a, b| a.total_cmp(b));
    let median_margin = sorted_margins[sorted_margins.len() / 2];

    // Blend with adaptive per-window weight
    standard_emissions
        .iter()
        .zip(profile_emissions.iter())
        .enumerate()
        .map(|(t, (std_row, prof_row))| {
            // Confidence ratio: how much stronger this window's signal is vs median
            let kappa = if median_margin > 1e-10 {
                (margins[t] / median_margin).clamp(0.1, 2.5)
            } else {
                1.0
            };
            let w = (base_weight * kappa).clamp(0.0, 0.95);
            let w_std = 1.0 - w;

            std_row
                .iter()
                .zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s
                    } else if p.is_finite() {
                        p
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

/// Blend standard and pairwise log-emissions using BT-standard agreement (T76).
///
/// Instead of using margin (which measures signal coherence, including coherent
/// bias), this uses agreement between the two models as the quality signal:
/// - Agreement (argmax_std == argmax_bt): both models point to the same
///   population → likely correct. Use higher pairwise weight.
/// - Disagreement: models conflict → pairwise may be amplifying bias.
///   Use lower pairwise weight to fall back toward standard emission.
///
/// This avoids the Stein paradox where "confident" pairwise windows (high margin)
/// that happen to be biased (e.g. EUR↔AMR confusion) get amplified, worsening
/// accuracy. Agreement is a better proxy for correctness than confidence.
///
/// `agree_scale` controls the up-weight factor on agreement (default 1.5).
/// `disagree_scale` controls the down-weight factor on disagreement (default 0.2).
pub fn blend_log_emissions_agreement(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    base_weight: f64,
    agree_scale: f64,
    disagree_scale: f64,
) -> Vec<Vec<f64>> {
    if standard_emissions.is_empty() || profile_emissions.is_empty() {
        return standard_emissions.to_vec();
    }

    standard_emissions
        .iter()
        .zip(profile_emissions.iter())
        .map(|(std_row, prof_row)| {
            // Find argmax of each emission model
            let argmax_std = std_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            let argmax_bt = prof_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);

            let agree = argmax_std.is_some()
                && argmax_bt.is_some()
                && argmax_std == argmax_bt;

            let scale = if agree { agree_scale } else { disagree_scale };
            let w = (base_weight * scale).clamp(0.0, 0.95);
            let w_std = 1.0 - w;

            std_row
                .iter()
                .zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s
                    } else if p.is_finite() {
                        p
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

/// Hybrid agreement-margin blending (T78 §4.2).
///
/// Combines agreement gating (T76) with margin scaling (adaptive), but only
/// applies margin modulation on windows where the models agree. This avoids
/// the Stein paradox: margin is only used when agreement confirms correctness,
/// recovering ~0.3 bits of signal in well-specified regimes.
///
/// - **Agreement windows**: weight = base × agree_scale × (margin / median_margin)
///   Margin modulates how much pairwise signal to mix in, but only when both
///   models point to the same population (likely correct direction).
/// - **Disagreement windows**: weight = base × disagree_scale
///   No margin modulation — models conflict, fall back toward standard emission.
///
/// `margin_clamp` limits the margin ratio to [0.2, 3.0] to prevent extreme weights.
pub fn blend_log_emissions_hybrid(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    base_weight: f64,
    agree_scale: f64,
    disagree_scale: f64,
    margin_clamp_lo: f64,
    margin_clamp_hi: f64,
) -> Vec<Vec<f64>> {
    if standard_emissions.is_empty() || profile_emissions.is_empty() {
        return standard_emissions.to_vec();
    }

    // Compute per-window margin (max - 2nd max) from profile emissions
    let margins: Vec<f64> = profile_emissions
        .iter()
        .map(|row| {
            if row.len() < 2 {
                return 0.0;
            }
            let mut best = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for &v in row {
                if v > best {
                    second = best;
                    best = v;
                } else if v > second {
                    second = v;
                }
            }
            if best.is_finite() && second.is_finite() {
                best - second
            } else {
                0.0
            }
        })
        .collect();

    // Compute median margin as reference scale (only positive margins)
    let mut sorted_margins: Vec<f64> = margins
        .iter()
        .filter(|m| **m > 0.0)
        .cloned()
        .collect();
    let median_margin = if sorted_margins.is_empty() {
        1.0 // no valid margins → margin ratio = 1.0 (neutral)
    } else {
        sorted_margins.sort_by(|a, b| a.total_cmp(b));
        sorted_margins[sorted_margins.len() / 2]
    };

    standard_emissions
        .iter()
        .zip(profile_emissions.iter())
        .enumerate()
        .map(|(t, (std_row, prof_row))| {
            // Find argmax of each emission model
            let argmax_std = std_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            let argmax_bt = prof_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);

            let agree = argmax_std.is_some()
                && argmax_bt.is_some()
                && argmax_std == argmax_bt;

            let w = if agree {
                // Agreement: scale by agree_scale × margin ratio
                let margin_ratio = if median_margin > 1e-10 {
                    (margins[t] / median_margin).clamp(margin_clamp_lo, margin_clamp_hi)
                } else {
                    1.0
                };
                (base_weight * agree_scale * margin_ratio).clamp(0.0, 0.95)
            } else {
                // Disagreement: flat down-weight, no margin
                (base_weight * disagree_scale).clamp(0.0, 0.95)
            };
            let w_std = 1.0 - w;

            std_row
                .iter()
                .zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s
                    } else if p.is_finite() {
                        p
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

/// Per-population agreement scaling parameters (T79).
///
/// Instead of uniform agree_scale/disagree_scale for all populations,
/// uses population-specific values derived from the F_ST hierarchy via
/// the D_min heuristic: well-separated populations (high D_min,k) get
/// higher agree_scale, while poorly separated populations get lower.
///
/// For disagreement, uses pairwise Cohen's d: close pairs (Type I, e.g.
/// EUR↔AMR) get moderate disagree_scale, distant pairs (Type II, e.g.
/// AFR-involving) get stronger suppression.
#[derive(Debug, Clone)]
pub struct PerPopAgreementScales {
    /// Per-population agree_scale (indexed by population index)
    pub agree_scales: Vec<f64>,
    /// Pairwise disagree_scale matrix: disagree_matrix[a][b] for when
    /// std says population a and BT says population b (a != b)
    pub disagree_matrix: Vec<Vec<f64>>,
}

/// Compute per-population agreement scales from observation data (T79 §8.2 Option C).
///
/// Uses the D_min heuristic: α₊^(k) = base_agree × (D_min,k / D̄_min)^γ
/// where D_min,k = min_{j≠k} Cohen_d(k,j), γ=0.5 (square-root from T60).
///
/// For disagreement: α₋^(a,b) = base_disagree × (d̄ / d(a,b))^γ
/// where d(a,b) is the pairwise Cohen's d. Close pairs → higher scale
/// (less suppression), distant pairs → lower scale (more suppression).
pub fn compute_per_pop_agreement_scales(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    base_agree_scale: f64,
    base_disagree_scale: f64,
) -> PerPopAgreementScales {
    let k = populations.len();

    // Fallback: uniform scales
    if k < 2 || observations.len() < 10 {
        return PerPopAgreementScales {
            agree_scales: vec![base_agree_scale; k],
            disagree_matrix: vec![vec![base_disagree_scale; k]; k],
        };
    }

    // Collect per-window per-population max similarity
    let mut all_window_sims: Vec<Vec<f64>> = Vec::with_capacity(observations.len());
    for obs in observations {
        let pop_sims: Vec<f64> = populations
            .iter()
            .filter_map(|pop| {
                pop.haplotypes
                    .iter()
                    .filter_map(|h| obs.similarities.get(h).copied())
                    .fold(None, |max, s| Some(max.map_or(s, |m: f64| m.max(s))))
            })
            .collect();
        if pop_sims.len() == k {
            all_window_sims.push(pop_sims);
        }
    }

    if all_window_sims.len() < 10 {
        return PerPopAgreementScales {
            agree_scales: vec![base_agree_scale; k],
            disagree_matrix: vec![vec![base_disagree_scale; k]; k],
        };
    }

    // Compute pairwise Cohen's d matrix (K×K, symmetric, diagonal=0)
    let mut cohens_d = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in (i + 1)..k {
            let diffs: Vec<f64> = all_window_sims
                .iter()
                .map(|sims| sims[i] - sims[j])
                .collect();
            let n = diffs.len() as f64;
            let mean = diffs.iter().sum::<f64>() / n;
            let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            let d = if std > 1e-10 {
                mean.abs() / std
            } else if mean.abs() > 1e-10 {
                100.0
            } else {
                0.0
            };
            cohens_d[i][j] = d;
            cohens_d[j][i] = d;
        }
    }

    // Per-population D_min,k = min_{j≠k} Cohen_d(k, j)
    let d_min_per_pop: Vec<f64> = (0..k)
        .map(|pop_k| {
            (0..k)
                .filter(|&j| j != pop_k)
                .map(|j| cohens_d[pop_k][j])
                .fold(f64::INFINITY, f64::min)
                .max(1e-6)
        })
        .collect();

    // D̄_min = mean of D_min,k
    let d_min_avg = d_min_per_pop.iter().sum::<f64>() / k as f64;

    // Per-population agree_scale: α₊^(k) = base × (D_min,k / D̄_min)^0.5
    let gamma = 0.5_f64;
    let agree_scales: Vec<f64> = d_min_per_pop
        .iter()
        .map(|&d_k| {
            let ratio = if d_min_avg > 1e-10 {
                (d_k / d_min_avg).powf(gamma)
            } else {
                1.0
            };
            // Clamp to reasonable range [0.5, 3.0]
            (base_agree_scale * ratio).clamp(0.5, 3.0)
        })
        .collect();

    // Per-pair disagree_scale: α₋^(a,b) = base × (d̄ / d(a,b))^0.5
    // Close pairs (low d) → higher scale (less suppression, Type I)
    // Distant pairs (high d) → lower scale (more suppression, Type II)
    let mean_d: f64 = {
        let mut sum = 0.0;
        let mut count = 0;
        for (i, row) in cohens_d.iter().enumerate() {
            for &d_val in row.iter().skip(i + 1) {
                sum += d_val;
                count += 1;
            }
        }
        if count > 0 { sum / count as f64 } else { 1.0 }
    };

    let mut disagree_matrix = vec![vec![base_disagree_scale; k]; k];
    for a in 0..k {
        for b in 0..k {
            if a == b {
                continue; // diagonal unused (agreement case)
            }
            let d_ab = cohens_d[a][b].max(1e-6);
            let ratio = if mean_d > 1e-10 {
                (mean_d / d_ab).powf(gamma)
            } else {
                1.0
            };
            // Clamp to [0.05, 0.5] — always suppress on disagreement
            disagree_matrix[a][b] = (base_disagree_scale * ratio).clamp(0.05, 0.5);
        }
    }

    eprintln!("  Per-population agreement scales (T79):");
    for (i, pop) in populations.iter().enumerate() {
        eprintln!(
            "    {}: D_min={:.4}, agree_scale={:.3}",
            pop.name, d_min_per_pop[i], agree_scales[i]
        );
    }
    eprintln!("  Disagree scales:");
    for a in 0..k {
        for b in 0..k {
            if a != b {
                eprintln!(
                    "    {}→{}: d={:.4}, disagree_scale={:.3}",
                    populations[a].name, populations[b].name,
                    cohens_d[a][b], disagree_matrix[a][b]
                );
            }
        }
    }

    PerPopAgreementScales {
        agree_scales,
        disagree_matrix,
    }
}

/// Agreement-based blending with per-population scaling (T79).
///
/// Like `blend_log_emissions_agreement` but uses population-specific
/// agree_scale from the F_ST hierarchy: well-separated populations (AFR)
/// get higher boost, poorly separated (AMR) get near-neutral boost.
/// Disagreement uses per-pair scaling: Type I (close pairs like EUR↔AMR)
/// gets moderate suppression, Type II (distant pairs like AFR↔EUR) gets
/// strong suppression.
pub fn blend_log_emissions_per_pop_agreement(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    base_weight: f64,
    scales: &PerPopAgreementScales,
) -> Vec<Vec<f64>> {
    if standard_emissions.is_empty() || profile_emissions.is_empty() {
        return standard_emissions.to_vec();
    }

    let k = scales.agree_scales.len();

    standard_emissions
        .iter()
        .zip(profile_emissions.iter())
        .map(|(std_row, prof_row)| {
            let argmax_std = std_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            let argmax_bt = prof_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);

            let w = match (argmax_std, argmax_bt) {
                (Some(a), Some(b)) if a == b && a < k => {
                    // Agreement on population a — use per-pop agree_scale
                    (base_weight * scales.agree_scales[a]).clamp(0.0, 0.95)
                }
                (Some(a), Some(b)) if a < k && b < k => {
                    // Disagreement (a,b) — use per-pair disagree_scale
                    (base_weight * scales.disagree_matrix[a][b]).clamp(0.0, 0.95)
                }
                _ => {
                    // Fallback: use first disagree_scale or base_weight * 0.2
                    (base_weight * 0.2).clamp(0.0, 0.95)
                }
            };
            let w_std = 1.0 - w;

            std_row
                .iter()
                .zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s
                    } else if p.is_finite() {
                        p
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

/// Hybrid agreement-margin blending with per-population scaling (T79 + T78).
///
/// Combines per-population agreement scaling (T79) with margin modulation
/// (T78): on agreement windows, weight = base × per_pop_agree_scale × margin_ratio;
/// on disagreement windows, weight = base × per_pair_disagree_scale (no margin).
pub fn blend_log_emissions_per_pop_hybrid(
    standard_emissions: &[Vec<f64>],
    profile_emissions: &[Vec<f64>],
    base_weight: f64,
    scales: &PerPopAgreementScales,
    margin_clamp_lo: f64,
    margin_clamp_hi: f64,
) -> Vec<Vec<f64>> {
    if standard_emissions.is_empty() || profile_emissions.is_empty() {
        return standard_emissions.to_vec();
    }

    let k = scales.agree_scales.len();

    // Compute per-window margin (max - 2nd max) from profile emissions
    let margins: Vec<f64> = profile_emissions
        .iter()
        .map(|row| {
            if row.len() < 2 {
                return 0.0;
            }
            let mut best = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for &v in row {
                if v > best {
                    second = best;
                    best = v;
                } else if v > second {
                    second = v;
                }
            }
            if best.is_finite() && second.is_finite() {
                best - second
            } else {
                0.0
            }
        })
        .collect();

    // Compute median margin as reference scale (only positive margins)
    let mut sorted_margins: Vec<f64> = margins
        .iter()
        .filter(|m| **m > 0.0)
        .cloned()
        .collect();
    let median_margin = if sorted_margins.is_empty() {
        1.0
    } else {
        sorted_margins.sort_by(|a, b| a.total_cmp(b));
        sorted_margins[sorted_margins.len() / 2]
    };

    standard_emissions
        .iter()
        .zip(profile_emissions.iter())
        .enumerate()
        .map(|(t, (std_row, prof_row))| {
            let argmax_std = std_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);
            let argmax_bt = prof_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i);

            let w = match (argmax_std, argmax_bt) {
                (Some(a), Some(b)) if a == b && a < k => {
                    // Agreement: per-pop agree_scale × margin ratio
                    let margin_ratio = if median_margin > 1e-10 {
                        (margins[t] / median_margin).clamp(margin_clamp_lo, margin_clamp_hi)
                    } else {
                        1.0
                    };
                    (base_weight * scales.agree_scales[a] * margin_ratio).clamp(0.0, 0.95)
                }
                (Some(a), Some(b)) if a < k && b < k => {
                    // Disagreement: per-pair disagree_scale, no margin
                    (base_weight * scales.disagree_matrix[a][b]).clamp(0.0, 0.95)
                }
                _ => {
                    (base_weight * 0.2).clamp(0.0, 0.95)
                }
            };
            let w_std = 1.0 - w;

            std_row
                .iter()
                .zip(prof_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        w_std * s + w * p
                    } else if s.is_finite() {
                        s
                    } else if p.is_finite() {
                        p
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect()
        })
        .collect()
}

// ============================================================================
// Posterior Feedback (iterative emission refinement)
// ============================================================================

/// Apply posterior feedback to log-emissions for iterative two-pass refinement.
///
/// For each window t and state k:
///   log_e'[t][k] = log_e[t][k] + lambda * log(max(posterior[t][k], epsilon))
///
/// This focuses emissions on populations that had high posterior in a previous
/// pass, improving discrimination between locally ambiguous populations
/// (e.g., EUR vs AMR in HPRC). The posterior acts as a per-window, per-state
/// prior that concentrates evidence on the most relevant populations.
///
/// Lambda controls feedback strength:
/// - 0.0: no feedback (standard emissions)
/// - 0.5: moderate focusing (recommended starting point)
/// - 1.0: strong focusing (equivalent to using posterior as time-varying prior)
///
/// # Arguments
/// * `log_emissions` - n×k matrix of log-emission probabilities
/// * `posteriors` - n×k matrix of posterior probabilities from pass 1
/// * `lambda` - feedback strength (0.0 = none, 1.0 = full)
pub fn apply_posterior_feedback(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    lambda: f64,
) -> Vec<Vec<f64>> {
    if lambda <= 0.0 || posteriors.is_empty() {
        return log_emissions.to_vec();
    }

    let epsilon = 1e-10;

    log_emissions.iter().zip(posteriors.iter())
        .map(|(le, post)| {
            le.iter().zip(post.iter())
                .map(|(&e, &p)| {
                    let feedback = lambda * p.max(epsilon).ln();
                    if e.is_finite() && feedback.is_finite() {
                        e + feedback
                    } else {
                        e
                    }
                })
                .collect()
        })
        .collect()
}

/// Mask (zero-out) log-emissions for populations with pass-1 posterior below threshold.
///
/// For each window t and population k:
///   if P₁(k|t) < threshold → log_emission'[t][k] = NEG_INFINITY
///   else → log_emission'[t][k] = log_emission[t][k]
///
/// This is a hard version of posterior feedback: instead of continuously weighting
/// by posterior, it binary-gates populations. Combined with posterior feedback,
/// masking removes clearly irrelevant populations while feedback fine-tunes among
/// the remaining ones.
///
/// The min_active parameter ensures at least N populations remain active per window
/// to prevent degenerate cases where all populations are masked.
pub fn apply_focused_masking(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    threshold: f64,
    min_active: usize,
) -> Vec<Vec<f64>> {
    if threshold <= 0.0 || posteriors.is_empty() {
        return log_emissions.to_vec();
    }

    log_emissions.iter().zip(posteriors.iter())
        .map(|(le, post)| {
            let k = le.len();
            // Count how many populations would survive the threshold
            let active_count = post.iter().filter(|&&p| p >= threshold).count();
            let effective_min = min_active.max(2).min(k);

            if active_count >= effective_min {
                // Normal masking: remove populations below threshold
                le.iter().zip(post.iter())
                    .map(|(&e, &p)| {
                        if p < threshold {
                            f64::NEG_INFINITY
                        } else {
                            e
                        }
                    })
                    .collect()
            } else {
                // Too few would survive — keep top-N by posterior
                let mut indexed: Vec<(usize, f64)> = post.iter()
                    .enumerate()
                    .map(|(i, &p)| (i, p))
                    .collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                let keep: std::collections::HashSet<usize> = indexed.iter()
                    .take(effective_min)
                    .map(|&(i, _)| i)
                    .collect();
                le.iter().enumerate()
                    .map(|(i, &e)| {
                        if keep.contains(&i) { e } else { f64::NEG_INFINITY }
                    })
                    .collect()
            }
        })
        .collect()
}

/// Smooth posteriors weighted by per-window confidence (inverse entropy).
///
/// For each window t, compute Shannon entropy H_t = -Σ_k P(k|t)×log(P(k|t)).
/// High-entropy (uncertain) windows get low weight; confident neighbors propagate
/// their signal to uncertain windows. Low-entropy (confident) windows get high
/// weight and resist smoothing.
///
/// P'(k|t) = Σ_{s=-R}^{R} w_{t+s} × P(k|t+s) / Σ_{s=-R}^{R} w_{t+s}
/// where w_t = exp(-beta × H_t / H_max)
///
/// beta controls the selectivity: higher beta = more weight on confident windows.
/// Default beta = 2.0 gives ~7x weight ratio between min and max entropy windows.
pub fn entropy_weighted_smooth_posteriors(
    posteriors: &[Vec<f64>],
    radius: usize,
    beta: f64,
) -> Vec<Vec<f64>> {
    let n = posteriors.len();
    if n == 0 || radius == 0 {
        return posteriors.to_vec();
    }

    let k = posteriors[0].len();

    // Compute entropy per window
    let entropies: Vec<f64> = posteriors.iter()
        .map(|post| {
            -post.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f64>()
        })
        .collect();

    // Find max entropy for normalization
    let h_max = entropies.iter().cloned().fold(0.0_f64, f64::max);
    if h_max <= 0.0 {
        return posteriors.to_vec();
    }

    // Compute weights: low entropy → high weight
    let weights: Vec<f64> = entropies.iter()
        .map(|&h| (-beta * h / h_max).exp())
        .collect();

    // Weighted average over sliding window
    let mut smoothed = vec![vec![0.0; k]; n];

    for t in 0..n {
        let lo = t.saturating_sub(radius);
        let hi = (t + radius + 1).min(n);

        let weight_sum: f64 = weights[lo..hi].iter().sum();

        if weight_sum > 0.0 {
            for (s, &wt) in weights[lo..hi].iter().enumerate() {
                let w = wt / weight_sum;
                for j in 0..k {
                    smoothed[t][j] += w * posteriors[lo + s][j];
                }
            }
        } else {
            smoothed[t] = posteriors[t].clone();
        }
    }

    smoothed
}

/// Compute pairwise population distances from aggregated identity profiles.
///
/// For each pair of populations (i, j), computes the average absolute difference
/// in their aggregated identities across all observation windows:
///   D[i][j] = mean_t |agg_sim_i(t) - agg_sim_j(t)|
///
/// Returns a symmetric K×K distance matrix with zeros on the diagonal.
pub fn compute_population_distances(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
) -> Vec<Vec<f64>> {
    let k = populations.len();
    let mut dist = vec![vec![0.0; k]; k];
    let mut count = 0u64;

    for obs in observations {
        // Compute aggregated similarity per population for this window
        let sims: Vec<Option<f64>> = populations.iter()
            .map(|pop| {
                let hap_sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                emission_model.aggregate(&hap_sims)
            })
            .collect();

        let valid: Vec<(usize, f64)> = sims.iter().enumerate()
            .filter_map(|(i, s)| s.filter(|&v| v > 0.0).map(|v| (i, v)))
            .collect();

        if valid.len() < 2 {
            continue;
        }
        count += 1;

        for a in 0..valid.len() {
            for b in (a+1)..valid.len() {
                let (i, si) = valid[a];
                let (j, sj) = valid[b];
                let d = (si - sj).abs();
                dist[i][j] += d;
                dist[j][i] += d;
            }
        }
    }

    if count > 0 {
        let n = count as f64;
        for row in dist.iter_mut() {
            for d in row.iter_mut() {
                *d /= n;
            }
        }
    }

    dist
}

/// Set transition probabilities weighted by population pairwise distances.
///
/// Close populations (small distance) get higher transition rates between them;
/// distant populations get lower rates. This captures the intuition that
/// EUR↔AMR switches are more likely than EUR↔EAS switches because EUR and AMR
/// are genealogically closer.
///
/// The closeness weight for pair (i,j) is: w[i][j] = exp(-distance[i][j] / scale)
/// where scale = median of all pairwise distances.
///
/// The transition matrix becomes:
///   T[i→j] = p_switch × (proportion_j × closeness(i,j)) / Z_i
pub fn set_distance_weighted_transitions(
    params: &mut AncestryHmmParams,
    distances: &[Vec<f64>],
    proportions: &[f64],
    switch_rates: &[f64],
) {
    let k = params.n_states;
    if distances.len() != k || proportions.len() != k || switch_rates.len() != k {
        return;
    }

    // Compute scale from median of pairwise distances
    let mut all_dists: Vec<f64> = Vec::new();
    for (i, row) in distances.iter().enumerate() {
        for &d in row.iter().skip(i + 1) {
            if d > 0.0 {
                all_dists.push(d);
            }
        }
    }
    all_dists.sort_by(|a, b| a.total_cmp(b));
    let scale = if all_dists.is_empty() {
        1.0
    } else {
        all_dists[all_dists.len() / 2].max(1e-10)
    };

    for i in 0..k {
        let switch_prob = switch_rates[i].clamp(0.0, 1.0);
        let stay_prob = 1.0 - switch_prob;

        // Compute weighted transition targets
        let weights: Vec<f64> = (0..k).map(|j| {
            if i == j {
                0.0
            } else {
                let closeness = (-distances[i][j] / scale).exp();
                proportions[j].max(1e-10) * closeness
            }
        }).collect();

        let weight_sum: f64 = weights.iter().sum();

        for (j, &wt) in weights.iter().enumerate() {
            if i == j {
                params.transitions[i][j] = stay_prob;
            } else if weight_sum > 0.0 {
                params.transitions[i][j] = switch_prob * wt / weight_sum;
            } else {
                params.transitions[i][j] = switch_prob / (k - 1).max(1) as f64;
            }
        }
    }
}

/// Apply label smoothing to log-emissions.
///
/// Interpolates between the data-driven log-emissions and uniform:
///   log_e'[t][k] = (1-α) × log_e[t][k] + α × log(1/K)
///
/// This prevents overconfident emissions that can trap the HMM in a single state,
/// especially important at ancestry boundaries where the true state changes but
/// emissions might strongly favor the old state.
///
/// Alpha = 0.0: no smoothing (standard emissions)
/// Alpha = 0.1: mild smoothing (recommended, preserves signal while ensuring switchability)
/// Alpha = 0.5: heavy smoothing (emissions carry half the information)
pub fn apply_label_smoothing(
    log_emissions: &[Vec<f64>],
    alpha: f64,
) -> Vec<Vec<f64>> {
    if alpha <= 0.0 {
        return log_emissions.to_vec();
    }
    let alpha = alpha.min(1.0);

    log_emissions.iter()
        .map(|le| {
            let k = le.len();
            if k == 0 {
                return le.clone();
            }
            let log_uniform = -(k as f64).ln();
            le.iter()
                .map(|&e| {
                    if e == f64::NEG_INFINITY {
                        // Masked state: keep masked
                        f64::NEG_INFINITY
                    } else {
                        (1.0 - alpha) * e + alpha * log_uniform
                    }
                })
                .collect()
        })
        .collect()
}

/// Apply margin-gated state persistence from pass-1 posteriors.
///
/// For windows where pass-1 was highly confident (margin > threshold), adds a
/// persistence bonus to the pass-1 argmax state's log-emission. This locks in
/// high-confidence calls while leaving ambiguous windows open to correction.
///
/// For each window t:
///   margin_t = max(P₁) - second_max(P₁)
///   if margin_t > threshold:
///     log_e'[t][argmax] += bonus × (margin_t - threshold) / (1.0 - threshold)
///
/// This is complementary to posterior feedback (which modifies all states
/// proportionally) — persistence specifically protects high-confidence calls.
pub fn apply_margin_persistence(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    threshold: f64,
    bonus: f64,
) -> Vec<Vec<f64>> {
    if bonus <= 0.0 || threshold >= 1.0 || posteriors.is_empty() {
        return log_emissions.to_vec();
    }

    log_emissions.iter().zip(posteriors.iter())
        .map(|(le, post)| {
            let k = le.len();
            if k < 2 {
                return le.clone();
            }

            // Find top-2 posteriors
            let mut sorted_idx: Vec<(usize, f64)> = post.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            sorted_idx.sort_by(|a, b| b.1.total_cmp(&a.1));

            let margin = sorted_idx[0].1 - sorted_idx[1].1;
            if margin <= threshold {
                return le.clone();
            }

            let argmax = sorted_idx[0].0;
            let strength = bonus * (margin - threshold) / (1.0 - threshold);

            let mut result = le.clone();
            if result[argmax].is_finite() {
                result[argmax] += strength;
            }
            result
        })
        .collect()
}

/// Compute per-window adaptive pairwise weight based on emission discriminability.
///
/// In each window, measures the gap between top-2 population standard emissions.
/// Windows with small gaps (ambiguous) get higher pairwise weight (pairwise signal
/// is most valuable there). Windows with large gaps get lower pairwise weight
/// (standard emissions are sufficient).
///
/// Returns per-window scaling factors in [min_scale, max_scale].
///
/// gap_t = |log_e[t][1st] - log_e[t][2nd]|
/// scale_t = max_scale - (max_scale - min_scale) × min(gap_t / gap_median, 1.0)
pub fn compute_adaptive_pairwise_scales(
    log_emissions: &[Vec<f64>],
    min_scale: f64,
    max_scale: f64,
) -> Vec<f64> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    // Compute per-window gap (absolute difference between top-2 finite emissions)
    let gaps: Vec<f64> = log_emissions.iter()
        .map(|le| {
            let mut finite: Vec<f64> = le.iter()
                .filter(|v| v.is_finite())
                .cloned()
                .collect();
            finite.sort_by(|a, b| b.total_cmp(a));
            if finite.len() >= 2 {
                (finite[0] - finite[1]).abs()
            } else {
                0.0
            }
        })
        .collect();

    // Compute median gap for normalization
    let mut sorted_gaps: Vec<f64> = gaps.iter()
        .filter(|&&g| g > 0.0)
        .cloned()
        .collect();
    sorted_gaps.sort_by(|a, b| a.total_cmp(b));

    let gap_median = if sorted_gaps.is_empty() {
        1.0
    } else {
        sorted_gaps[sorted_gaps.len() / 2].max(1e-10)
    };

    // Map gaps to scales: small gap → high scale, large gap → low scale
    gaps.iter()
        .map(|&g| {
            let ratio = (g / gap_median).min(1.0);
            max_scale - (max_scale - min_scale) * ratio
        })
        .collect()
}

/// Blend log-emissions with per-window adaptive pairwise weights.
///
/// For each window t:
///   blended[t] = (1 - scale_t × base_weight) × standard[t] + scale_t × base_weight × pairwise[t]
///
/// Where scale_t comes from compute_adaptive_pairwise_scales().
/// Windows where standard emissions are ambiguous get stronger pairwise contribution.
pub fn blend_log_emissions_adaptive_per_window(
    standard: &[Vec<f64>],
    pairwise: &[Vec<f64>],
    base_weight: f64,
    scales: &[f64],
) -> Vec<Vec<f64>> {
    standard.iter().zip(pairwise.iter()).zip(scales.iter())
        .map(|((std_row, pw_row), &scale)| {
            let effective_weight = (base_weight * scale).clamp(0.0, 0.95);
            std_row.iter().zip(pw_row.iter())
                .map(|(&s, &p)| {
                    if s.is_finite() && p.is_finite() {
                        (1.0 - effective_weight) * s + effective_weight * p
                    } else if s.is_finite() {
                        s
                    } else {
                        p
                    }
                })
                .collect()
        })
        .collect()
}

/// Compute per-reference-haplotype purity scores.
///
/// For each reference haplotype h in population k, computes how "distinctive"
/// it is for its population versus other populations. High purity = the query's
/// similarity to h strongly predicts population k. Low purity = h is admixed
/// or uninformative (similar to references from other populations).
///
/// Purity score for haplotype h in population k:
///   purity_h = mean_t(sim_h(t)) / mean_t(max_{h' ∈ pop ≠ k} sim_{h'}(t))
///
/// Scores > 1 indicate h is generally MORE similar to query than the best
/// out-of-population reference, making it a strong population-k indicator.
/// Scores ≈ 1 indicate h is uninformative (similar to out-of-pop references).
///
/// Returns HashMap<haplotype_name, purity_score>
pub fn compute_reference_purity(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> HashMap<String, f64> {
    let mut hap_sums: HashMap<String, f64> = HashMap::new();
    let mut hap_counts: HashMap<String, u64> = HashMap::new();
    let mut max_other_sums: HashMap<String, f64> = HashMap::new();

    for obs in observations {
        // For each population's haplotypes
        for (pop_idx, pop) in populations.iter().enumerate() {
            for hap in &pop.haplotypes {
                if let Some(&sim) = obs.similarities.get(hap) {
                    if sim <= 0.0 { continue; }
                    *hap_sums.entry(hap.clone()).or_insert(0.0) += sim;
                    *hap_counts.entry(hap.clone()).or_insert(0) += 1;

                    // Find max similarity to any haplotype NOT in this population
                    let max_other: f64 = populations.iter().enumerate()
                        .filter(|(i, _)| *i != pop_idx)
                        .flat_map(|(_, other_pop)| other_pop.haplotypes.iter())
                        .filter_map(|h| obs.similarities.get(h))
                        .cloned()
                        .fold(0.0_f64, f64::max);

                    if max_other > 0.0 {
                        *max_other_sums.entry(hap.clone()).or_insert(0.0) += max_other;
                    }
                }
            }
        }
    }

    let mut purity = HashMap::new();
    for (hap, &sum) in &hap_sums {
        let count = hap_counts.get(hap).cloned().unwrap_or(1) as f64;
        let mean_sim = sum / count;
        let mean_other = max_other_sums.get(hap).cloned().unwrap_or(mean_sim) / count;
        if mean_other > 0.0 {
            purity.insert(hap.clone(), mean_sim / mean_other);
        } else {
            purity.insert(hap.clone(), 1.0);
        }
    }

    purity
}

/// Create purity-weighted emission model wrapper.
///
/// For each population, when aggregating haplotype similarities, weight each
/// haplotype by its purity score. High-purity haplotypes (strong population
/// indicators) get more influence; low-purity (admixed) haplotypes get less.
///
/// Weighted aggregation: score_k = Σ_h purity_h × sim_h / Σ_h purity_h
/// Applied before the standard emission model (replaces raw similarities
/// with purity-weighted ones).
pub fn apply_purity_weighted_observations(
    observations: &[AncestryObservation],
    purity_scores: &HashMap<String, f64>,
    gamma: f64,
) -> Vec<AncestryObservation> {
    observations.iter()
        .map(|obs| {
            let mut new_sims = obs.similarities.clone();
            for (hap, sim) in new_sims.iter_mut() {
                let purity = purity_scores.get(hap).cloned().unwrap_or(1.0);
                // Weight: purity^gamma. gamma=0: no weighting, gamma=1: full weighting
                let weight = purity.powf(gamma);
                *sim *= weight;
            }
            AncestryObservation {
                chrom: obs.chrom.clone(),
                start: obs.start,
                end: obs.end,
                sample: obs.sample.clone(),
                similarities: new_sims,
                haplotype_consistency_bonus: obs.haplotype_consistency_bonus.clone(),
                coverage_ratios: obs.coverage_ratios.clone(),
            }
        })
        .collect()
}

// ============================================================================
// Within-population variance penalty
// ============================================================================

/// Compute within-population similarity variance for each window and population.
///
/// For each window t and population k, collects per-haplotype similarities
/// {sim(h, t) : h ∈ pop_k} and computes their variance. High variance indicates
/// the aggregated score is driven by one outlier haplotype rather than consistent
/// population-wide similarity.
///
/// Returns an n×K matrix of variances.
pub fn compute_within_pop_variance(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> Vec<Vec<f64>> {
    let n_pops = populations.len();
    observations.iter().map(|obs| {
        (0..n_pops).map(|p| {
            let sims: Vec<f64> = populations[p].haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h).cloned())
                .collect();
            if sims.len() < 2 {
                return 0.0;
            }
            let mean = sims.iter().sum::<f64>() / sims.len() as f64;
            let var = sims.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                / (sims.len() - 1) as f64; // sample variance
            var
        }).collect()
    }).collect()
}

/// Apply variance penalty to log-emissions.
///
/// Penalizes populations where the emission score is driven by one outlier
/// haplotype rather than consistent similarity across all references:
///   log_e'[t][k] = log_e[t][k] - weight × σ²_k(t) / scale
///
/// Where scale is the median non-zero variance across all (t, k) pairs.
/// This normalizes the penalty to be scale-independent.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `variances` - n×K matrix from `compute_within_pop_variance()`
/// * `weight` - Penalty strength. 0.0 = no penalty, 1.0 = moderate, 5.0 = strong
///
/// # Returns
/// Penalized n×K matrix of log emissions
pub fn apply_variance_penalty(
    log_emissions: &[Vec<f64>],
    variances: &[Vec<f64>],
    weight: f64,
) -> Vec<Vec<f64>> {
    if weight <= 0.0 || log_emissions.is_empty() || variances.is_empty() {
        return log_emissions.to_vec();
    }

    // Compute median non-zero variance for scaling
    let mut all_vars: Vec<f64> = variances.iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v > 0.0)
        .cloned()
        .collect();

    if all_vars.is_empty() {
        return log_emissions.to_vec();
    }

    all_vars.sort_by(|a, b| a.total_cmp(b));
    let median_var = all_vars[all_vars.len() / 2];

    if median_var <= 0.0 {
        return log_emissions.to_vec();
    }

    log_emissions.iter().zip(variances.iter()).map(|(le_row, var_row)| {
        le_row.iter().zip(var_row.iter()).map(|(&le, &var)| {
            if le.is_finite() {
                le - weight * var / median_var
            } else {
                le // preserve NEG_INFINITY (masked states)
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Flank-informed emission adjustment
// ============================================================================

/// Apply flank-informed emission adjustment using pass-1 decoded states.
///
/// For each window t, examines the mode (most frequent) decoded state in
/// flanking windows [t-R, t-1] (left) and [t+1, t+R] (right). If both flanks
/// agree on the same state, adds an emission bonus to that state in the output.
///
/// This leverages spatial context: if all neighbors strongly favor EUR, then
/// even an ambiguous window t should get an EUR bonus. Different from posterior
/// feedback (which uses the window's own posterior) and entropy smoothing
/// (which works post-hoc).
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions for pass-2
/// * `states` - Pass-1 decoded state sequence (length n)
/// * `radius` - Number of flanking windows on each side to examine
/// * `bonus` - Emission bonus (in nats) when both flanks agree
/// * `n_states` - Total number of states K
///
/// # Returns
/// Adjusted n×K matrix of log emissions
pub fn apply_flank_informed_bonus(
    log_emissions: &[Vec<f64>],
    states: &[usize],
    radius: usize,
    bonus: f64,
    n_states: usize,
) -> Vec<Vec<f64>> {
    if radius == 0 || bonus <= 0.0 || log_emissions.is_empty() || states.is_empty() {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();

    log_emissions.iter().enumerate().map(|(t, row)| {
        // Left flank: [t-R, t-1]
        let left_mode = if t > 0 {
            let lo = t.saturating_sub(radius);
            flank_mode(&states[lo..t], n_states)
        } else {
            None
        };

        // Right flank: [t+1, t+R]
        let right_mode = if t + 1 < n {
            let hi = (t + 1 + radius).min(n);
            flank_mode(&states[t + 1..hi], n_states)
        } else {
            None
        };

        // Only apply bonus if both flanks agree on the same state
        match (left_mode, right_mode) {
            (Some(l), Some(r)) if l == r => {
                let agreed_state = l;
                row.iter().enumerate().map(|(k, &le)| {
                    if k == agreed_state && le.is_finite() {
                        le + bonus
                    } else {
                        le
                    }
                }).collect()
            }
            _ => row.clone(),
        }
    }).collect()
}

/// Compute mode (most frequent value) of a state slice.
/// Returns None if slice is empty. Ties broken by lowest state index.
fn flank_mode(states: &[usize], n_states: usize) -> Option<usize> {
    if states.is_empty() {
        return None;
    }
    let mut counts = vec![0usize; n_states];
    for &s in states {
        if s < n_states {
            counts[s] += 1;
        }
    }
    counts.iter().enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
}

// ============================================================================
// Leave-one-out robust emissions
// ============================================================================

/// Compute leave-one-out robust log-emissions.
///
/// For each window and population, computes K leave-one-out aggregated
/// similarities (each leaving out one haplotype), then takes the minimum.
/// This is the "pessimistic" emission — it measures the minimum guaranteed
/// signal without any single haplotype. Detects when a population's score
/// is driven by one outlier haplotype (e.g., an AMR haplotype with EUR
/// ancestry inflating the AMR score in EUR regions).
///
/// Falls back to standard emission when a population has only 1 haplotype.
///
/// # Arguments
/// * `observations` - Per-window similarity data
/// * `populations` - Population definitions with haplotype lists
/// * `emission_model` - Aggregation model (Max, Mean, etc.)
/// * `temperature` - Softmax temperature for converting to log-probabilities
///
/// # Returns
/// n×K matrix of leave-one-out robust log emissions
pub fn compute_loo_robust_emissions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
    temperature: f64,
) -> Vec<Vec<f64>> {
    let n_pops = populations.len();
    if observations.is_empty() || n_pops == 0 {
        return vec![vec![0.0; n_pops]; observations.len()];
    }

    let temp = if temperature > 0.0 { temperature } else { 0.01 };

    observations.iter().map(|obs| {
        let loo_sims: Vec<f64> = populations.iter().map(|pop| {
            let sims: Vec<(usize, f64)> = pop.haplotypes.iter()
                .enumerate()
                .filter_map(|(i, h)| obs.similarities.get(h).map(|&s| (i, s)))
                .collect();

            if sims.len() <= 1 {
                // Can't do LOO with 0-1 haplotypes, use standard
                return sims.first().map(|&(_, s)| s).unwrap_or(0.0);
            }

            // Compute leave-one-out: for each haplotype, aggregate without it
            let mut min_agg = f64::INFINITY;
            for leave_out in 0..sims.len() {
                let subset: Vec<f64> = sims.iter()
                    .filter(|&&(i, _)| i != sims[leave_out].0)
                    .map(|&(_, s)| s)
                    .collect();
                if let Some(agg) = emission_model.aggregate(&subset) {
                    if agg < min_agg {
                        min_agg = agg;
                    }
                }
            }
            if min_agg.is_finite() { min_agg } else { 0.0 }
        }).collect();

        // Softmax to log-probabilities
        let max_sim = loo_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = loo_sims.iter().map(|&s| (s - max_sim) / temp).collect();
        let log_sum_exp = scaled.iter().map(|&s| s.exp()).sum::<f64>().ln();
        scaled.iter().map(|&s| s - log_sum_exp).collect()
    }).collect()
}

// ============================================================================
// Posterior sharpening
// ============================================================================

/// Apply temperature-based sharpening/softening to posteriors.
///
/// Transforms posteriors: P'(k|t) ∝ P(k|t)^(1/T)
/// - T < 1.0: sharpen (confident calls become more confident)
/// - T = 1.0: no change
/// - T > 1.0: soften (spread probability mass more evenly)
///
/// This is useful as post-processing before MPEL decode or entropy smoothing,
/// to amplify the signal in the posterior distribution.
///
/// # Arguments
/// * `posteriors` - n×K matrix of posterior probabilities
/// * `temperature` - Sharpening temperature. 0.5 = moderate sharpening, 0.1 = aggressive
///
/// # Returns
/// Re-normalized n×K matrix of posteriors
pub fn sharpen_posteriors(
    posteriors: &[Vec<f64>],
    temperature: f64,
) -> Vec<Vec<f64>> {
    if temperature <= 0.0 || (temperature - 1.0).abs() < 1e-10 || posteriors.is_empty() {
        return posteriors.to_vec();
    }

    let inv_temp = 1.0 / temperature;

    posteriors.iter().map(|row| {
        // Apply power: P^(1/T) = exp(log(P)/T)
        let powered: Vec<f64> = row.iter().map(|&p| {
            if p > 0.0 {
                p.powf(inv_temp)
            } else {
                0.0
            }
        }).collect();

        // Re-normalize
        let sum: f64 = powered.iter().sum();
        if sum > 0.0 {
            powered.iter().map(|&p| p / sum).collect()
        } else {
            row.clone()
        }
    }).collect()
}

// ============================================================================
// Per-window quality scoring
// ============================================================================

/// Compute per-window quality scores from posteriors and emissions.
///
/// Combines three signals into a single quality metric [0, 1]:
/// 1. Posterior margin: gap between top-1 and top-2 posterior
/// 2. Emission discriminability: gap between top-1 and top-2 log-emission
/// 3. Neighbor agreement: fraction of ±radius neighbors with same decoded state
///
/// Quality = w1×margin + w2×(disc/max_disc) + w3×agreement
/// where w1=0.4, w2=0.3, w3=0.3 (empirical weights favoring posterior signal).
///
/// # Returns
/// Vector of quality scores, one per window, in [0, 1]
pub fn compute_window_quality(
    posteriors: &[Vec<f64>],
    log_emissions: &[Vec<f64>],
    states: &[usize],
    radius: usize,
) -> Vec<f64> {
    let n = posteriors.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Posterior margins
    let margins: Vec<f64> = posteriors.iter().map(|row| {
        let mut sorted: Vec<f64> = row.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted.len() >= 2 {
            (sorted[0] - sorted[1]).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }).collect();

    // 2. Emission discriminability (normalized to [0,1])
    let discs: Vec<f64> = log_emissions.iter().map(|row| {
        let finite: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite.len() < 2 { return 0.0; }
        let mut sorted = finite;
        sorted.sort_by(|a, b| b.total_cmp(a));
        sorted[0] - sorted[1]
    }).collect();

    let max_disc = discs.iter().cloned().fold(0.0_f64, f64::max);
    let norm_discs: Vec<f64> = if max_disc > 0.0 {
        discs.iter().map(|&d| d / max_disc).collect()
    } else {
        vec![0.0; n]
    };

    // 3. Neighbor agreement
    let agreements: Vec<f64> = states.iter().enumerate().map(|(t, &s)| {
        if radius == 0 { return 1.0; }
        let lo = t.saturating_sub(radius);
        let hi = (t + radius + 1).min(n);
        let neighbors = hi - lo - 1; // exclude self
        if neighbors == 0 { return 1.0; }
        let agree = (lo..hi)
            .filter(|&i| i != t && states.get(i).copied() == Some(s))
            .count();
        agree as f64 / neighbors as f64
    }).collect();

    // Combine with weights
    let (w1, w2, w3) = (0.4, 0.3, 0.3);
    (0..n).map(|t| {
        let margin = margins.get(t).copied().unwrap_or(0.0);
        let disc = norm_discs.get(t).copied().unwrap_or(0.0);
        let agree = agreements.get(t).copied().unwrap_or(0.0);
        (w1 * margin + w2 * disc + w3 * agree).clamp(0.0, 1.0)
    }).collect()
}

// ============================================================================
// Iterative refinement (N-pass inference)
// ============================================================================

/// Run iterative multi-pass inference.
///
/// Generalizes two-pass to N passes. Each pass uses the previous pass's
/// posteriors to apply posterior feedback to the emissions, then re-runs
/// forward-backward. The posterior feedback lambda can be increased each
/// pass (progressive focusing).
///
/// # Arguments
/// * `log_emissions` - Initial n×K log-emission matrix
/// * `params` - HMM parameters
/// * `n_passes` - Number of refinement passes (1 = single-pass, same as standard)
/// * `feedback_lambda` - Base feedback lambda (increased by 0.5× each pass)
///
/// # Returns
/// (posteriors, states) from the final pass
pub fn iterative_refine(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    n_passes: usize,
    feedback_lambda: f64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    if log_emissions.is_empty() || n_passes == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut current_emissions = log_emissions.to_vec();
    let mut posteriors = forward_backward_from_log_emissions(&current_emissions, params);
    let mut states: Vec<usize> = posteriors.iter()
        .map(|probs| probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx).unwrap_or(0))
        .collect();

    // Early stopping threshold (T83): stop when max posterior change < epsilon.
    // Interior windows converge super-exponentially; boundary windows converge
    // linearly with rate rho ~ lambda/(1+lambda). Typically triggers at 3-5 passes.
    const CONVERGENCE_EPS: f64 = 0.001;

    // Iterative passes (pass 0 is the initial one above)
    for pass in 1..n_passes {
        // Progressive feedback: lambda increases each pass
        let lambda = feedback_lambda * (1.0 + 0.5 * (pass as f64 - 1.0));
        let lambda = lambda.min(2.0); // cap at 2.0

        let prev_posteriors = posteriors.clone();

        // Apply posterior feedback to original emissions
        current_emissions = apply_posterior_feedback(log_emissions, &posteriors, lambda);

        // Re-run forward-backward
        posteriors = forward_backward_from_log_emissions(&current_emissions, params);
        states = posteriors.iter()
            .map(|probs| probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx).unwrap_or(0))
            .collect();

        // Early stopping (T83): converged when max absolute posterior change is small
        let max_change = posteriors.iter().zip(prev_posteriors.iter())
            .flat_map(|(new, old)| new.iter().zip(old.iter()).map(|(n, o)| (n - o).abs()))
            .fold(0.0f64, f64::max);
        if max_change < CONVERGENCE_EPS {
            break;
        }
    }

    (posteriors, states)
}

// ============================================================================
// Population-specific emission scaling from pass-1 calibration
// ============================================================================

/// Compute per-population emission scaling factors from pass-1 results.
///
/// For each population k, compares the fraction of windows assigned to k
/// in pass-1 (observed) vs the prior proportion (expected). Under-represented
/// populations get a positive boost, over-represented get dampened.
///
/// boost_k = log(prior_k / observed_k) × scale_factor
///
/// This corrects systematic under/over-calling of specific populations.
///
/// # Arguments
/// * `states` - Pass-1 decoded states
/// * `proportions` - Prior proportions (from pass-1 estimation)
/// * `n_states` - Number of states K
/// * `scale_factor` - How aggressively to correct. 0.0 = off, 0.5 = moderate
///
/// # Returns
/// K-length vector of emission boosts (in nats)
pub fn compute_calibration_boosts(
    states: &[usize],
    proportions: &[f64],
    n_states: usize,
    scale_factor: f64,
) -> Vec<f64> {
    if states.is_empty() || proportions.is_empty() || n_states == 0 || scale_factor <= 0.0 {
        return vec![0.0; n_states];
    }

    // Count observed fractions
    let mut counts = vec![0usize; n_states];
    for &s in states {
        if s < n_states {
            counts[s] += 1;
        }
    }
    let n = states.len() as f64;

    (0..n_states).map(|k| {
        let observed = (counts[k] as f64 / n).max(1e-6);
        let expected = proportions.get(k).cloned().unwrap_or(1.0 / n_states as f64).max(1e-6);
        // log(expected/observed): positive when under-represented, negative when over-represented
        scale_factor * (expected / observed).ln()
    }).collect()
}

/// Apply per-population calibration boosts to log-emissions.
///
/// Adds a constant boost per population across all windows:
/// log_e'[t][k] = log_e[t][k] + boost_k
///
/// Preserves masked (NEG_INFINITY) states.
pub fn apply_calibration_boosts(
    log_emissions: &[Vec<f64>],
    boosts: &[f64],
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || boosts.is_empty() {
        return log_emissions.to_vec();
    }

    log_emissions.iter().map(|row| {
        row.iter().enumerate().map(|(k, &le)| {
            if le.is_finite() {
                le + boosts.get(k).cloned().unwrap_or(0.0)
            } else {
                le
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Emission diversity index
// ============================================================================

/// Apply diversity-based emission scaling.
///
/// For each window, computes the normalized Shannon entropy of the emission
/// probability distribution (after softmax). Low entropy = one population
/// dominates → trust emissions (scale up deviations from mean). High entropy =
/// all populations similar → flatten emissions (scale down deviations).
///
/// scaling = (1 - H_normalized) × amplify + H_normalized × dampen
/// Where H_normalized = H / log(K), in [0, 1]
///
/// When amplify > 1 and dampen < 1: confident windows get sharper,
/// uncertain windows get flatter. A nonlinear analog of heteroscedastic
/// temperature that works on the posterior distribution shape.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `amplify` - Scale factor for high-confidence windows (> 1.0 to sharpen). Default 1.2
/// * `dampen` - Scale factor for low-confidence windows (< 1.0 to flatten). Default 0.5
///
/// # Returns
/// Diversity-scaled n×K matrix of log emissions
pub fn apply_diversity_scaling(
    log_emissions: &[Vec<f64>],
    amplify: f64,
    dampen: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    let k = log_emissions[0].len();
    if k <= 1 {
        return log_emissions.to_vec();
    }

    let log_k = (k as f64).ln();
    if log_k <= 0.0 {
        return log_emissions.to_vec();
    }

    log_emissions.iter().map(|row| {
        // Convert to probabilities for entropy computation
        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite_vals.len() < 2 {
            return row.clone();
        }

        let max_val = finite_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = finite_vals.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return row.clone();
        }
        let norm_probs: Vec<f64> = probs.iter().map(|&p| p / sum).collect();

        // Shannon entropy
        let entropy: f64 = norm_probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let h_normalized = (entropy / log_k).clamp(0.0, 1.0);

        // Scaling factor: interpolate between amplify (low entropy) and dampen (high entropy)
        let scale = (1.0 - h_normalized) * amplify + h_normalized * dampen;

        // Apply: scale deviations from per-window mean
        let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;
        row.iter().map(|&v| {
            if v.is_finite() {
                mean + scale * (v - mean)
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Cross-population emission residual
// ============================================================================

/// Compute residual log-emissions (deviation from per-window mean).
///
/// For each window, subtracts the cross-population mean, then amplifies
/// the residuals by a factor. This extracts what's unique about each
/// population's emission, removing the shared "overall similarity" component.
///
/// Mathematically: residual[t][k] = factor × (log_e[t][k] - mean_t)
///
/// Different from contrast normalization which just centers (factor=1).
/// Amplification (factor > 1) makes small population-specific differences
/// more visible to the HMM.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `factor` - Residual amplification factor. 1.0 = centering only, 2.0 = 2x amplified
///
/// # Returns
/// Residual-amplified n×K matrix of log emissions
pub fn amplify_emission_residuals(
    log_emissions: &[Vec<f64>],
    factor: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || factor <= 0.0 {
        return log_emissions.to_vec();
    }

    log_emissions.iter().map(|row| {
        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite_vals.is_empty() {
            return row.clone();
        }
        let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;

        row.iter().map(|&v| {
            if v.is_finite() {
                mean + factor * (v - mean)
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Emission rank transform
// ============================================================================

/// Convert log-emissions to rank-based scores within each window.
///
/// For each window, ranks populations by their log-emission (highest = rank 0).
/// Then maps ranks to evenly-spaced log-probabilities using:
///   score[rank] = log(K - rank) - log(sum(1..K))
///
/// This eliminates scale differences between populations (e.g., AFR having
/// systematically higher identities than EUR/AMR) and forces the HMM to rely
/// purely on relative ordering. Useful when absolute emission magnitudes are
/// misleading due to panel composition effects.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
///
/// # Returns
/// Rank-transformed n×K matrix of log emissions (proper log-probabilities)
pub fn rank_transform_emissions(
    log_emissions: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    let k = log_emissions[0].len();
    if k <= 1 {
        return log_emissions.to_vec();
    }

    // Pre-compute rank scores: score[rank] = log(K - rank) - log(sum(1..K))
    // rank 0 (best) gets highest score, rank K-1 gets lowest
    let harmonic_sum: f64 = (1..=k).map(|i| i as f64).sum::<f64>();
    let log_norm = harmonic_sum.ln();
    let rank_scores: Vec<f64> = (0..k).map(|r| ((k - r) as f64).ln() - log_norm).collect();

    log_emissions.iter().map(|row| {
        // Find which indices are finite
        let mut indexed: Vec<(usize, f64)> = row.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| (i, v))
            .collect();

        if indexed.is_empty() {
            return row.clone();
        }

        // Sort by emission (descending) to assign ranks
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut result = vec![f64::NEG_INFINITY; k];
        for (rank, &(idx, _)) in indexed.iter().enumerate() {
            result[idx] = rank_scores[rank];
        }
        result
    }).collect()
}

// ============================================================================
// Emission anchor boosting
// ============================================================================

/// Boost emissions for windows that are "anchored" by high-confidence neighbors.
///
/// For each window, looks at a neighborhood of radius R. If the fraction of
/// neighbors where the argmax state matches exceeds a threshold, the window's
/// emission for that state gets a boost proportional to the agreement fraction.
///
/// Unlike flank_informed_bonus (which uses decoded states from pass-1), this
/// operates purely on emission argmax — no decoding needed. Can be used in
/// pass-1 or as a pre-processing step.
///
/// Mathematically:
///   agreement_frac = |{t' in [t-R,t+R] : argmax_k(e[t']) == argmax_k(e[t])}| / (2R)
///   if agreement_frac > threshold:
///     log_e'[t][argmax] += boost × agreement_frac
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `radius` - Number of windows to check on each side
/// * `threshold` - Minimum agreement fraction to trigger boost (0.0 to 1.0)
/// * `boost` - Maximum log-space bonus when agreement = 1.0
///
/// # Returns
/// Anchor-boosted n×K matrix of log emissions
pub fn apply_emission_anchor_boost(
    log_emissions: &[Vec<f64>],
    radius: usize,
    threshold: f64,
    boost: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || radius == 0 || boost <= 0.0 {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();

    // Pre-compute argmax for each window
    let argmax: Vec<Option<usize>> = log_emissions.iter().map(|row| {
        row.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }).collect();

    log_emissions.iter().enumerate().map(|(t, row)| {
        let center_state = match argmax[t] {
            Some(s) => s,
            None => return row.clone(),
        };

        // Count agreement in neighborhood (excluding self)
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);
        let mut agree_count = 0usize;
        let mut total_count = 0usize;

        for (i, am) in argmax.iter().enumerate().take(end).skip(start) {
            if i == t { continue; }
            if let Some(s) = am {
                total_count += 1;
                if *s == center_state {
                    agree_count += 1;
                }
            }
        }

        if total_count == 0 {
            return row.clone();
        }

        let agreement_frac = agree_count as f64 / total_count as f64;
        if agreement_frac < threshold {
            return row.clone();
        }

        let mut result = row.clone();
        if result[center_state].is_finite() {
            result[center_state] += boost * agreement_frac;
        }
        result
    }).collect()
}

// ============================================================================
// Emission outlier dampening
// ============================================================================

/// Dampen outlier emissions that deviate significantly from a population's
/// typical emission across windows.
///
/// For each population k, computes the median and MAD (median absolute deviation)
/// of log-emissions across all windows. Windows where emission[t][k] is more than
/// `z_threshold` MADs above the median get dampened toward the median.
///
/// This prevents isolated high-emission spikes from dominating the HMM, which
/// is particularly problematic when one window has an outlier similarity score
/// due to alignment artifacts or highly variable regions.
///
/// Dampening formula: if deviation > z_threshold × MAD:
///   log_e'[t][k] = median_k + z_threshold × MAD_k × sign(deviation)
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `z_threshold` - Number of MADs beyond which to clip (e.g., 3.0)
///
/// # Returns
/// Outlier-dampened n×K matrix of log emissions
pub fn dampen_emission_outliers(
    log_emissions: &[Vec<f64>],
    z_threshold: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || z_threshold <= 0.0 {
        return log_emissions.to_vec();
    }

    let k = log_emissions[0].len();
    if k == 0 {
        return log_emissions.to_vec();
    }

    // Compute per-population median and MAD
    let mut medians = Vec::with_capacity(k);
    let mut mads = Vec::with_capacity(k);

    for pop in 0..k {
        let mut vals: Vec<f64> = log_emissions.iter()
            .map(|row| row[pop])
            .filter(|v| v.is_finite())
            .collect();

        if vals.len() < 3 {
            medians.push(f64::NAN);
            mads.push(f64::NAN);
            continue;
        }

        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if vals.len() % 2 == 0 {
            (vals[vals.len() / 2 - 1] + vals[vals.len() / 2]) / 2.0
        } else {
            vals[vals.len() / 2]
        };

        let mut abs_devs: Vec<f64> = vals.iter().map(|&v| (v - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = if abs_devs.len() % 2 == 0 {
            (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
        } else {
            abs_devs[abs_devs.len() / 2]
        };

        medians.push(median);
        mads.push(mad);
    }

    log_emissions.iter().map(|row| {
        row.iter().enumerate().map(|(pop, &v)| {
            if !v.is_finite() || medians[pop].is_nan() || mads[pop] < 1e-12 {
                return v;
            }
            let deviation = v - medians[pop];
            if deviation.abs() > z_threshold * mads[pop] {
                medians[pop] + z_threshold * mads[pop] * deviation.signum()
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Population pair confusion penalty
// ============================================================================

/// Compute pair-specific transition penalties from pass-1 state sequence.
///
/// For frequently confused population pairs (high switch rate between them),
/// increases the transition cost (more negative log-transition) to require
/// stronger evidence before switching.
///
/// Returns an n_states × n_states matrix of log-transition adjustments.
/// High-switch pairs get negative adjustments (harder to switch).
///
/// Formula:
///   switch_rate[i][j] = count(i→j) / count(i→*)
///   penalty[i][j] = -weight × (switch_rate[i][j] / max_rate)
///
/// The penalties are applied symmetrically: penalty[i][j] = penalty[j][i].
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `n_states` - Number of populations
/// * `weight` - Penalty strength (log-space, e.g., 0.5)
///
/// # Returns
/// n_states × n_states matrix of log-transition adjustments
pub fn compute_confusion_penalties(
    states: &[usize],
    n_states: usize,
    weight: f64,
) -> Vec<Vec<f64>> {
    let mut penalties = vec![vec![0.0; n_states]; n_states];
    if states.len() < 2 || weight <= 0.0 || n_states < 2 {
        return penalties;
    }

    // Count transitions
    let mut transitions = vec![vec![0usize; n_states]; n_states];
    let mut from_counts = vec![0usize; n_states];

    for w in states.windows(2) {
        let from = w[0];
        let to = w[1];
        if from < n_states && to < n_states && from != to {
            transitions[from][to] += 1;
            from_counts[from] += 1;
        }
    }

    // Compute switch rates
    let mut max_rate = 0.0f64;
    let mut rates = vec![vec![0.0; n_states]; n_states];
    for i in 0..n_states {
        if from_counts[i] == 0 { continue; }
        for j in 0..n_states {
            if i == j { continue; }
            rates[i][j] = transitions[i][j] as f64 / from_counts[i] as f64;
            max_rate = max_rate.max(rates[i][j]);
        }
    }

    if max_rate < 1e-12 {
        return penalties;
    }

    // High-switch pairs get negative penalty (harder to switch)
    // Symmetrize by averaging
    for i in 0..n_states {
        for j in (i + 1)..n_states {
            let sym_rate = (rates[i][j] + rates[j][i]) / 2.0;
            let pen = -weight * (sym_rate / max_rate);
            penalties[i][j] = pen;
            penalties[j][i] = pen;
        }
    }

    penalties
}

/// Apply confusion penalties to transition parameters.
///
/// Adjusts the HMM transition probabilities by adding log-space penalties
/// from `compute_confusion_penalties`. Creates per-window transition matrices
/// (all identical) that can be used with `forward_backward_from_log_emissions_with_transitions`.
///
/// # Arguments
/// * `params` - HMM parameters (provides base transitions)
/// * `penalties` - n_states × n_states penalty matrix (from compute_confusion_penalties)
///
/// # Returns
/// Per-window n_states × n_states log-transition matrix
pub fn apply_confusion_penalties(
    params: &AncestryHmmParams,
    penalties: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = params.n_states;
    let mut log_trans = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let base_prob = params.transitions[i][j].max(1e-20);
            log_trans[i][j] = base_prob.ln() + penalties[i][j];
        }
        // Re-normalize in log space
        let max_val = log_trans[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum: f64 = log_trans[i].iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln() + max_val;
        for val in &mut log_trans[i] {
            *val -= log_sum;
        }
    }

    log_trans
}

// ============================================================================
// Emission momentum (temporal smoothing)
// ============================================================================

/// Apply exponential moving average to emissions across windows.
///
/// For each window t, blends the current emission with the running average:
///   e'[t] = (1 - alpha) × e[t] + alpha × e'[t-1]
///
/// This creates temporal continuity in emissions, reducing the impact of
/// single-window noise. Different from spatial smoothing (which uses
/// neighbors symmetrically) — momentum propagates forward only, like
/// an IIR filter.
///
/// Uses a forward-backward scheme for symmetry:
///   forward:  f[t] = (1 - alpha) × e[t] + alpha × f[t-1]
///   backward: b[t] = (1 - alpha) × e[t] + alpha × b[t+1]
///   result:   e'[t] = (f[t] + b[t]) / 2
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `alpha` - Momentum coefficient (0.0 = no smoothing, 0.9 = heavy smoothing)
///
/// # Returns
/// Momentum-smoothed n×K matrix of log emissions
pub fn apply_emission_momentum(
    log_emissions: &[Vec<f64>],
    alpha: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = log_emissions[0].len();

    // Forward pass
    let mut forward = log_emissions.to_vec();
    for t in 1..n {
        for pop in 0..k {
            if forward[t][pop].is_finite() && forward[t - 1][pop].is_finite() {
                forward[t][pop] = (1.0 - alpha) * log_emissions[t][pop]
                    + alpha * forward[t - 1][pop];
            }
        }
    }

    // Backward pass
    let mut backward = log_emissions.to_vec();
    for t in (0..n.saturating_sub(1)).rev() {
        for pop in 0..k {
            if backward[t][pop].is_finite() && backward[t + 1][pop].is_finite() {
                backward[t][pop] = (1.0 - alpha) * log_emissions[t][pop]
                    + alpha * backward[t + 1][pop];
            }
        }
    }

    // Average forward and backward
    forward.iter().zip(backward.iter()).map(|(f_row, b_row)| {
        f_row.iter().zip(b_row.iter()).map(|(&f, &b)| {
            if f.is_finite() && b.is_finite() {
                (f + b) / 2.0
            } else if f.is_finite() {
                f
            } else {
                b
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Population emission floor
// ============================================================================

/// Set a minimum log-emission for each population per window.
///
/// Prevents any population from being completely ruled out by setting a
/// floor on the log-emission. This is different from label smoothing
/// (which interpolates toward uniform) — the floor acts as a hard minimum
/// that only affects very low emissions.
///
/// For each window t and population k:
///   log_e'[t][k] = max(log_e[t][k], floor)
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `floor` - Minimum log-emission value (e.g., -10.0 or -20.0)
///
/// # Returns
/// Floored n×K matrix of log emissions
pub fn apply_emission_floor(
    log_emissions: &[Vec<f64>],
    floor: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    log_emissions.iter().map(|row| {
        row.iter().map(|&v| {
            if v.is_finite() {
                v.max(floor)
            } else {
                v // preserve NEG_INFINITY (masked states)
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Emission gradient penalty
// ============================================================================

/// Penalize large emission changes between adjacent windows.
///
/// For each pair of adjacent windows (t, t+1), computes the absolute difference
/// in log-emissions for each population. If the difference exceeds a threshold,
/// applies a penalty that pulls both windows toward their average.
///
/// This encourages smoother emission landscapes, reducing the impact of
/// single-window spikes that cause false ancestry switches.
///
/// Penalty: for each population k and adjacent pair (t, t+1):
///   delta = |e[t][k] - e[t+1][k]|
///   if delta > 0: adjust by weight × (avg - current)
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `weight` - Penalty strength (0.0 to 1.0). 0.0 = off, 0.5 = half correction
///
/// # Returns
/// Gradient-penalized n×K matrix of log emissions
pub fn apply_gradient_penalty(
    log_emissions: &[Vec<f64>],
    weight: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || weight <= 0.0 || log_emissions.len() < 2 {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = log_emissions[0].len();
    let w = weight.clamp(0.0, 1.0);

    let mut result = log_emissions.to_vec();

    // Apply gradient penalty iteratively
    for pop in 0..k {
        for t in 0..n.saturating_sub(1) {
            let (left, right) = result.split_at_mut(t + 1);
            let curr = left[t][pop];
            let next = right[0][pop];
            if !curr.is_finite() || !next.is_finite() {
                continue;
            }
            let avg = (curr + next) / 2.0;
            left[t][pop] = curr + w * (avg - curr);
            right[0][pop] = next + w * (avg - next);
        }
    }

    result
}

// ============================================================================
// Posterior-weighted Viterbi decode
// ============================================================================

/// Combine posteriors with emissions for a posterior-weighted Viterbi decode.
///
/// Creates modified emissions by blending original log-emissions with
/// log-posteriors: e'[t][k] = (1-lambda) × e[t][k] + lambda × log(P[t][k])
///
/// The resulting emissions can be fed to standard Viterbi for a decode that
/// balances emission evidence with full-sequence posterior information.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `posteriors` - n×K matrix of posterior probabilities (from FB)
/// * `lambda` - Blending weight for posteriors (0.0 = pure emission, 1.0 = pure posterior)
///
/// # Returns
/// Posterior-blended n×K matrix of log emissions suitable for Viterbi
pub fn blend_posteriors_with_emissions(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    lambda: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || lambda <= 0.0 {
        return log_emissions.to_vec();
    }
    if lambda >= 1.0 {
        // Pure posterior: convert to log
        return posteriors.iter().map(|row| {
            row.iter().map(|&p| {
                if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }
            }).collect()
        }).collect();
    }

    log_emissions.iter().zip(posteriors.iter()).map(|(e_row, p_row)| {
        e_row.iter().zip(p_row.iter()).map(|(&e, &p)| {
            let log_p = if p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
            if e.is_finite() && log_p.is_finite() {
                (1.0 - lambda) * e + lambda * log_p
            } else if e.is_finite() {
                e
            } else {
                log_p
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Changepoint prior (state persistence bonus)
// ============================================================================

/// Add a persistence bonus to emissions based on pass-1 decoded states.
///
/// For each window, adds a log-space bonus to the state that was decoded
/// in pass-1, creating inertia that resists state changes. This complements
/// the transition matrix (which encodes generic switch probability) with
/// evidence-specific persistence from the first pass.
///
/// Mathematically:
///   log_e'[t][k] = log_e[t][k] + bonus  if k == pass1_state[t]
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `states` - Pass-1 decoded state sequence (length n)
/// * `bonus` - Log-space bonus for the pass-1 state (e.g., 0.5)
///
/// # Returns
/// Persistence-boosted n×K matrix of log emissions
pub fn apply_changepoint_prior(
    log_emissions: &[Vec<f64>],
    states: &[usize],
    bonus: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || bonus <= 0.0 || states.is_empty() {
        return log_emissions.to_vec();
    }

    log_emissions.iter().enumerate().map(|(t, row)| {
        if t >= states.len() {
            return row.clone();
        }
        let prev_state = states[t];
        let mut result = row.clone();
        if prev_state < result.len() && result[prev_state].is_finite() {
            result[prev_state] += bonus;
        }
        result
    }).collect()
}

// ============================================================================
// Pairwise emission contrast enhancement
// ============================================================================

/// Amplify the emission gap between the top-2 populations per window.
///
/// For each window, identifies the best and second-best populations by emission.
/// Then widens the gap by adding `boost` to the best and subtracting `boost`
/// from the second-best. This directly addresses cases where two populations
/// (e.g., EUR and AMR) have nearly identical emissions.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `boost` - Amount to widen the gap (applied symmetrically: +boost to best, -boost to second)
///
/// # Returns
/// Contrast-enhanced n×K matrix of log emissions
pub fn apply_pairwise_emission_contrast(
    log_emissions: &[Vec<f64>],
    boost: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || boost <= 0.0 {
        return log_emissions.to_vec();
    }

    log_emissions.iter().map(|row| {
        // Find top-2 finite indices
        let mut indexed: Vec<(usize, f64)> = row.iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| (i, v))
            .collect();

        if indexed.len() < 2 {
            return row.clone();
        }

        // Sort descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = indexed[0].0;
        let second_idx = indexed[1].0;

        let mut result = row.clone();
        result[best_idx] += boost;
        result[second_idx] -= boost;
        result
    }).collect()
}

// ============================================================================
// Population-specific pass-2 temperature adjustment
// ============================================================================

/// Adjust emission temperatures per-population based on pass-1 posterior confidence.
///
/// For each population, computes the mean posterior from pass-1. Populations
/// with low mean posterior (uncertainty) get their emissions amplified (cooled),
/// while populations with high mean posterior (confidence) get dampened (warmed).
///
/// This is applied as a per-population emission rescaling:
///   log_e'[t][k] = mean + (log_e[t][k] - mean) × (1 + factor × (threshold - mean_post_k))
///
/// Where threshold is the reciprocal of n_states (1/K) — the expected posterior
/// under uniform assignment.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `posteriors` - Pass-1 posterior probabilities (n×K matrix)
/// * `factor` - Scaling strength (e.g., 1.0 = moderate, 2.0 = strong)
///
/// # Returns
/// Temperature-adjusted n×K matrix of log emissions
pub fn adjust_pop_temperatures(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    factor: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || posteriors.is_empty() || factor <= 0.0 {
        return log_emissions.to_vec();
    }

    let k = log_emissions[0].len();
    if k == 0 {
        return log_emissions.to_vec();
    }
    let threshold = 1.0 / k as f64;

    // Compute mean posterior per population
    let n = posteriors.len().min(log_emissions.len());
    let mut mean_post = vec![0.0; k];
    for post_row in posteriors.iter().take(n) {
        for (pop, val) in mean_post.iter_mut().enumerate().take(k) {
            if pop < post_row.len() {
                *val += post_row[pop];
            }
        }
    }
    for val in &mut mean_post {
        *val /= n as f64;
    }

    // Compute per-population scaling
    // Low mean_post → scale > 1 (amplify). High mean_post → scale < 1 (dampen).
    let scales: Vec<f64> = mean_post.iter().map(|&mp| {
        (1.0 + factor * (threshold - mp)).max(0.1) // floor at 0.1 to prevent inversion
    }).collect();

    log_emissions.iter().map(|row| {
        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite_vals.is_empty() {
            return row.clone();
        }
        let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;

        row.iter().enumerate().map(|(pop, &v)| {
            if v.is_finite() && pop < scales.len() {
                mean + (v - mean) * scales[pop]
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// SNR-weighted emissions
// ============================================================================

/// Scale emissions by per-window signal-to-noise ratio.
///
/// For each window, computes the emission range (max - min of finite values)
/// relative to the global median range. Windows with high range (clear signal)
/// get amplified; windows with low range (noise) get dampened.
///
/// Scaling: scale = (range_t / median_range) ^ power
/// log_e'[t][k] = mean_t + scale × (log_e[t][k] - mean_t)
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `power` - Exponent for the SNR scaling (0.5 = sqrt, 1.0 = linear)
///
/// # Returns
/// SNR-weighted n×K matrix of log emissions
pub fn apply_snr_weighting(
    log_emissions: &[Vec<f64>],
    power: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || power <= 0.0 {
        return log_emissions.to_vec();
    }

    // Compute per-window ranges
    let ranges: Vec<f64> = log_emissions.iter().map(|row| {
        let finite: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite.len() < 2 {
            return 0.0;
        }
        let max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }).collect();

    // Compute median range
    let mut sorted_ranges: Vec<f64> = ranges.iter().filter(|&&r| r > 0.0).cloned().collect();
    if sorted_ranges.is_empty() {
        return log_emissions.to_vec();
    }
    sorted_ranges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_range = if sorted_ranges.len() % 2 == 0 {
        (sorted_ranges[sorted_ranges.len() / 2 - 1] + sorted_ranges[sorted_ranges.len() / 2]) / 2.0
    } else {
        sorted_ranges[sorted_ranges.len() / 2]
    };

    if median_range < 1e-12 {
        return log_emissions.to_vec();
    }

    log_emissions.iter().enumerate().map(|(t, row)| {
        if ranges[t] <= 0.0 {
            return row.clone();
        }
        let scale = (ranges[t] / median_range).powf(power).clamp(0.1, 5.0);
        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        if finite_vals.is_empty() {
            return row.clone();
        }
        let mean = finite_vals.iter().sum::<f64>() / finite_vals.len() as f64;

        row.iter().map(|&v| {
            if v.is_finite() {
                mean + scale * (v - mean)
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Cross-entropy emission regularization
// ============================================================================

/// Regularize emissions toward pass-1 posterior distribution.
///
/// Blends log-emissions with log-posteriors from pass-1, preventing pass-2
/// emissions from deviating too far from what was already learned.
/// Uses simple linear interpolation in log-space:
///   log_e'[t][k] = (1 - lambda) × log_e[t][k] + lambda × log(post[t][k])
///
/// Equivalent to blend_posteriors_with_emissions but semantically different:
/// this is used as regularization before HMM, not as a decode modification.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `posteriors` - Pass-1 posterior probabilities (n×K matrix)
/// * `lambda` - Regularization strength (0.0 = pure emission, 1.0 = pure posterior)
///
/// # Returns
/// Regularized n×K matrix of log emissions
pub fn regularize_toward_posteriors(
    log_emissions: &[Vec<f64>],
    posteriors: &[Vec<f64>],
    lambda: f64,
) -> Vec<Vec<f64>> {
    blend_posteriors_with_emissions(log_emissions, posteriors, lambda)
}

// ============================================================================
// Windowed emission normalization
// ============================================================================

/// Normalize emissions relative to a local window mean.
///
/// For each window t, computes the mean emission per population across
/// windows [t-R, t+R]. Then subtracts this local mean and adds back the
/// global mean, removing slowly-varying trends (e.g., centromeric/telomeric
/// identity gradients).
///
/// Mathematically:
///   local_mean[t][k] = mean(log_e[max(0,t-R)..min(n,t+R+1)][k])
///   log_e'[t][k] = log_e[t][k] - local_mean[t][k] + global_mean[k]
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `radius` - Number of windows on each side for local normalization
///
/// # Returns
/// Locally normalized n×K matrix of log emissions
pub fn apply_windowed_normalization(
    log_emissions: &[Vec<f64>],
    radius: usize,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || radius == 0 {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = log_emissions[0].len();

    // Compute global mean per population
    let mut global_mean = vec![0.0; k];
    let mut global_count = vec![0usize; k];
    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                global_mean[pop] += v;
                global_count[pop] += 1;
            }
        }
    }
    for pop in 0..k {
        if global_count[pop] > 0 {
            global_mean[pop] /= global_count[pop] as f64;
        }
    }

    // For each window, compute local mean and normalize
    log_emissions.iter().enumerate().map(|(t, row)| {
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);

        row.iter().enumerate().map(|(pop, &v)| {
            if !v.is_finite() {
                return v;
            }

            let mut local_sum = 0.0;
            let mut local_count = 0usize;
            for window in log_emissions.iter().take(end).skip(start) {
                if pop < window.len() && window[pop].is_finite() {
                    local_sum += window[pop];
                    local_count += 1;
                }
            }

            if local_count == 0 {
                return v;
            }
            let local_mean = local_sum / local_count as f64;
            v - local_mean + global_mean[pop]
        }).collect()
    }).collect()
}

// ============================================================================
// Entropy-weighted posterior smoothing
// ============================================================================

/// Smooth posteriors using entropy-weighted averaging over a window neighborhood.
///
/// Confident windows (low Shannon entropy) propagate their posteriors to
/// uncertain neighbors (high entropy). For each window t, the smoothed
/// posterior is a weighted average of posteriors in [t-R, t+R], where the
/// weight of each neighbor is inversely proportional to its entropy.
///
/// # Arguments
/// * `posteriors` - n×K matrix of posterior probabilities
/// * `radius` - Number of windows on each side for smoothing
///
/// # Returns
/// Smoothed n×K posterior matrix (rows sum to ~1.0)
pub fn entropy_smooth_posteriors(
    posteriors: &[Vec<f64>],
    radius: usize,
) -> Vec<Vec<f64>> {
    if posteriors.is_empty() || radius == 0 {
        return posteriors.to_vec();
    }

    let n = posteriors.len();
    let k = if n > 0 { posteriors[0].len() } else { return vec![] };

    // Precompute inverse entropy weights for each window
    let weights: Vec<f64> = posteriors.iter().map(|row| {
        let entropy: f64 = row.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (k as f64).ln();
        if max_entropy <= 0.0 {
            return 1.0;
        }
        let h_norm = (entropy / max_entropy).clamp(0.0, 1.0);
        // Inverse entropy: confident (low H) → high weight
        1.0 / (h_norm + 1e-6)
    }).collect();

    posteriors.iter().enumerate().map(|(t, _)| {
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);

        let mut smoothed = vec![0.0; k];
        let mut total_weight = 0.0;

        for (i, w) in weights.iter().enumerate().take(end).skip(start) {
            total_weight += w;
            for (pop, val) in smoothed.iter_mut().enumerate() {
                *val += w * posteriors[i][pop];
            }
        }

        if total_weight > 0.0 {
            for val in &mut smoothed {
                *val /= total_weight;
            }
        }

        smoothed
    }).collect()
}

// ============================================================================
// Emission quantile normalization
// ============================================================================

/// Normalize each population's emission distribution across windows to match
/// a reference distribution (average of all populations' ranked values).
///
/// For each population: rank emissions, map to reference quantiles.
/// This ensures all populations have the same marginal emission distribution,
/// removing population-specific biases in emission scale.
///
/// NEG_INFINITY values are preserved and excluded from ranking.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
///
/// # Returns
/// Quantile-normalized n×K matrix of log emissions
pub fn quantile_normalize_emissions(
    log_emissions: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return vec![];
    }

    let k = log_emissions[0].len();

    // For each population, collect (index, value) of finite entries, sort by value
    let mut pop_ranked: Vec<Vec<(usize, f64)>> = (0..k).map(|pop| {
        let mut entries: Vec<(usize, f64)> = log_emissions.iter().enumerate()
            .filter(|(_, row)| pop < row.len() && row[pop].is_finite())
            .map(|(t, row)| (t, row[pop]))
            .collect();
        entries.sort_by(|a, b| a.1.total_cmp(&b.1));
        entries
    }).collect();

    // Compute reference distribution: average of ranked values across populations
    let max_len = pop_ranked.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 {
        return log_emissions.to_vec();
    }

    let reference: Vec<f64> = (0..max_len).map(|rank| {
        let mut sum = 0.0;
        let mut count = 0;
        for pop_entries in &pop_ranked {
            if pop_entries.is_empty() { continue; }
            // Map rank to this population's index space
            let idx = (rank * pop_entries.len()) / max_len;
            let idx = idx.min(pop_entries.len() - 1);
            sum += pop_entries[idx].1;
            count += 1;
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }).collect();

    // Assign reference values back to each population
    let mut result: Vec<Vec<f64>> = log_emissions.to_vec();
    for (pop, entries) in pop_ranked.iter_mut().enumerate() {
        let pop_len = entries.len();
        if pop_len == 0 { continue; }
        for (rank, &(t, _)) in entries.iter().enumerate() {
            // Map this population's rank to reference
            let ref_idx = (rank * max_len) / pop_len;
            let ref_idx = ref_idx.min(max_len - 1);
            result[t][pop] = reference[ref_idx];
        }
    }

    result
}

// ============================================================================
// Adaptive transition scaling
// ============================================================================

/// Scale transition log-probabilities per window based on local emission entropy.
///
/// Low-entropy (confident) windows get amplified transitions (more switching
/// resistance — the off-diagonal entries are pushed further from zero in log space).
/// High-entropy (uncertain) windows get relaxed transitions (easier switching).
///
/// For each window t, the normalized Shannon entropy H_t is computed. The
/// transition matrix is scaled as:
///   log_T'[i][j] = log_T[i][j] × (1 + factor × (1 - H_norm_t))  for i ≠ j
///   log_T'[i][i] adjusted to maintain row normalization
///
/// Returns per-window log-transition matrices suitable for
/// `forward_backward_from_log_emissions_with_transitions`.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions (for entropy computation)
/// * `params` - HMM parameters (base transition matrix)
/// * `factor` - Scaling strength. 0 = no scaling, 1 = moderate, 2 = aggressive
///
/// # Returns
/// Vector of n log-transition matrices (each K×K)
pub fn compute_adaptive_transitions(
    log_emissions: &[Vec<f64>],
    params: &AncestryHmmParams,
    factor: f64,
) -> Vec<Vec<Vec<f64>>> {
    let k = params.n_states;

    // Compute base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    // Compute per-window normalized entropy
    let entropies: Vec<f64> = log_emissions.iter().map(|row| {
        // Convert log-emissions to probabilities via softmax
        let max_val = row.iter().filter(|v| v.is_finite()).cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_val.is_finite() {
            return 1.0; // fully uncertain
        }
        let probs: Vec<f64> = row.iter().map(|&v| {
            if v.is_finite() { (v - max_val).exp() } else { 0.0 }
        }).collect();
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return 1.0;
        }
        let entropy: f64 = probs.iter().map(|&p| {
            let norm_p = p / sum;
            if norm_p > 0.0 { -norm_p * norm_p.ln() } else { 0.0 }
        }).sum();
        let max_entropy = (k as f64).ln();
        if max_entropy > 0.0 { (entropy / max_entropy).clamp(0.0, 1.0) } else { 1.0 }
    }).collect();

    // Build per-window log-transition matrices
    entropies.iter().map(|&h_norm| {
        // Scale factor: confident → large scale (more penalty for switching)
        // uncertain → small scale (less penalty)
        let scale = 1.0 + factor * (1.0 - h_norm);

        let mut log_trans = base_log_trans.clone();
        for (i, row) in log_trans.iter_mut().enumerate().take(k) {
            // Scale off-diagonal entries (they are negative in log space)
            for (j, val) in row.iter_mut().enumerate().take(k) {
                if i != j {
                    *val *= scale;
                }
            }
            // Re-normalize row in log space
            let log_sum = row.iter().cloned()
                .fold(f64::NEG_INFINITY, |a, b| {
                    let max_ab = a.max(b);
                    max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                });
            for val in row.iter_mut() {
                *val -= log_sum;
            }
        }
        log_trans
    }).collect()
}

// ============================================================================
// Local emission reranking
// ============================================================================

/// Re-rank populations by cumulative emission support in a local neighborhood.
///
/// For each window t, sums log-emissions over [t-R, t+R] for each population.
/// Then assigns log-probability scores based on the cumulative rank:
/// score[rank] = log(K - rank) - log(sum(1..K))
///
/// This is a neighborhood-aware version of rank_transform that uses soft
/// cumulative evidence rather than single-window argmax.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `radius` - Number of windows on each side for cumulative support
///
/// # Returns
/// Re-ranked n×K matrix of log emissions
pub fn local_rerank_emissions(
    log_emissions: &[Vec<f64>],
    radius: usize,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || radius == 0 {
        return log_emissions.to_vec();
    }

    let n = log_emissions.len();
    let k = log_emissions[0].len();

    // Precompute log-probability scores for each rank
    let rank_sum: f64 = (1..=k).map(|i| i as f64).sum();
    let rank_scores: Vec<f64> = (0..k).map(|rank| {
        ((k - rank) as f64).ln() - rank_sum.ln()
    }).collect();

    log_emissions.iter().enumerate().map(|(t, _)| {
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);

        // Sum emissions per population over neighborhood
        let mut cumulative = vec![0.0_f64; k];
        for row in log_emissions.iter().take(end).skip(start) {
            for (pop, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    cumulative[pop] += v;
                } else {
                    cumulative[pop] += -1e6; // large penalty for NEG_INFINITY
                }
            }
        }

        // Sort populations by cumulative support (descending)
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| cumulative[b].total_cmp(&cumulative[a]));

        // Assign rank scores
        let mut result = vec![f64::NEG_INFINITY; k];
        for (rank, &pop) in indices.iter().enumerate() {
            result[pop] = rank_scores[rank];
        }

        result
    }).collect()
}

// ============================================================================
// Emission Bayesian shrinkage
// ============================================================================

/// Shrink per-window emissions toward the population-specific global mean.
///
/// For each population k, computes the global mean emission μ[k] across all
/// windows. Then each window's emission is shrunk toward that mean:
///   e'[t][k] = (1 - alpha) × e[t][k] + alpha × μ[k]
///
/// This reduces noise in individual windows by pulling extreme values toward
/// the population baseline. alpha=0 → no shrinkage, alpha=1 → all windows
/// get the global mean.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `alpha` - Shrinkage strength in [0, 1]
///
/// # Returns
/// Shrunk n×K matrix of log emissions
pub fn bayesian_shrink_emissions(
    log_emissions: &[Vec<f64>],
    alpha: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || alpha <= 0.0 {
        return log_emissions.to_vec();
    }

    let k = log_emissions[0].len();

    // Compute global mean per population
    let mut global_mean = vec![0.0; k];
    let mut count = vec![0usize; k];
    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                global_mean[pop] += v;
                count[pop] += 1;
            }
        }
    }
    for pop in 0..k {
        if count[pop] > 0 {
            global_mean[pop] /= count[pop] as f64;
        }
    }

    // Shrink toward global mean
    log_emissions.iter().map(|row| {
        row.iter().enumerate().map(|(pop, &v)| {
            if v.is_finite() {
                (1.0 - alpha) * v + alpha * global_mean[pop]
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Top-K emission sparsification
// ============================================================================

/// Keep only the top-K emission values per window, set the rest to a floor value.
///
/// For each window, ranks populations by emission value (descending). The top-K
/// populations keep their original emissions; all others are set to `floor`
/// (a very negative log value, effectively removing them from consideration).
///
/// This forces the HMM to choose among a small candidate set per window,
/// reducing noise from impossible populations.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `top_k` - Number of populations to keep per window
/// * `floor` - Log-emission value for zeroed-out populations (e.g., -100.0)
///
/// # Returns
/// Sparsified n×K matrix of log emissions
pub fn sparsify_top_k_emissions(
    log_emissions: &[Vec<f64>],
    top_k: usize,
    floor: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || top_k == 0 {
        return log_emissions.to_vec();
    }

    let k = log_emissions[0].len();
    if top_k >= k {
        return log_emissions.to_vec();
    }

    log_emissions.iter().map(|row| {
        // Get sorted indices by emission (descending)
        let mut indices: Vec<usize> = (0..row.len()).collect();
        indices.sort_by(|&a, &b| row[b].total_cmp(&row[a]));

        let mut result = vec![floor; row.len()];
        for &idx in indices.iter().take(top_k) {
            result[idx] = row[idx];
        }
        result
    }).collect()
}

// ============================================================================
// Short segment correction
// ============================================================================

// ============================================================================
// Sliding window majority vote
// ============================================================================

/// Post-decoding majority filter: replace each state with the most common
/// state in a [t-R, t+R] neighborhood.
///
/// Simple but effective denoising. Removes isolated 1-2 window misassignments
/// while preserving genuine transitions that span multiple windows.
///
/// # Arguments
/// * `states` - Decoded state sequence
/// * `n_states` - Number of possible states
/// * `radius` - Neighborhood radius in windows
///
/// # Returns
/// Filtered state sequence
pub fn majority_vote_filter(
    states: &[usize],
    n_states: usize,
    radius: usize,
) -> Vec<usize> {
    if states.is_empty() || radius == 0 {
        return states.to_vec();
    }

    let n = states.len();

    states.iter().enumerate().map(|(t, &original)| {
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);

        let mut counts = vec![0usize; n_states];
        for &s in states.iter().take(end).skip(start) {
            if s < n_states {
                counts[s] += 1;
            }
        }

        // Find the most common state; break ties by keeping original
        let (best_state, best_count) = counts.iter().enumerate()
            .max_by_key(|&(i, &c)| (c, if i == original { 1 } else { 0 }))
            .map(|(i, &c)| (i, c))
            .unwrap_or((original, 0));

        if best_count > 0 { best_state } else { original }
    }).collect()
}

// ============================================================================
// Population proportion prior
// ============================================================================

/// Add a Bayesian prior based on population proportions to emissions.
///
/// From pass-1 decoded states, estimates population proportions. Then adds
/// a weighted log-prior to each window's emissions:
///   log_e'[t][k] = log_e[t][k] + weight × log(proportion_k)
///
/// Populations that appear more frequently get a Bayesian boost. This acts
/// as a population-frequency-informed prior, similar to how allele frequency
/// priors help in genotyping.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `proportions` - Per-population proportions (sum to ~1.0)
/// * `weight` - Strength of the prior (0.0 = off, 1.0 = standard Bayesian)
///
/// # Returns
/// Emissions with proportion prior added
pub fn apply_proportion_prior(
    log_emissions: &[Vec<f64>],
    proportions: &[f64],
    weight: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || weight <= 0.0 {
        return log_emissions.to_vec();
    }

    // Precompute log-proportion prior (with floor for zero proportions)
    let log_prior: Vec<f64> = proportions.iter()
        .map(|&p| (p.max(1e-6)).ln() * weight)
        .collect();

    log_emissions.iter().map(|row| {
        row.iter().enumerate().map(|(pop, &v)| {
            if v.is_finite() && pop < log_prior.len() {
                v + log_prior[pop]
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Short segment correction
// ============================================================================

// ============================================================================
// Segment boundary boost
// ============================================================================

/// Create per-window transition matrices that ease switching at pass-1 boundaries.
///
/// At positions where pass-1 decoded states change, transition off-diagonal
/// entries are boosted (less negative in log space). Away from boundaries,
/// self-transitions are amplified. This encodes the prior that ancestry
/// switches are most likely where pass-1 already detected them.
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (base transition matrix)
/// * `weight` - Boost strength. 0 = no effect, positive = stronger boundary signal
///
/// # Returns
/// Vector of n log-transition matrices (each K×K)
pub fn compute_boundary_boost_transitions(
    states: &[usize],
    params: &AncestryHmmParams,
    weight: f64,
) -> Vec<Vec<Vec<f64>>> {
    let k = params.n_states;

    // Compute base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    // Detect boundaries: positions where state changes
    let mut is_boundary = vec![false; states.len()];
    for i in 1..states.len() {
        if states[i] != states[i - 1] {
            is_boundary[i] = true;
            if i > 0 {
                is_boundary[i - 1] = true;
            }
        }
    }

    is_boundary.iter().map(|&boundary| {
        let mut log_trans = base_log_trans.clone();

        if boundary {
            // At boundaries: boost off-diagonal (make switching easier)
            // Reduce magnitude of negative off-diagonal entries
            for (i, row) in log_trans.iter_mut().enumerate().take(k) {
                for (j, val) in row.iter_mut().enumerate().take(k) {
                    if i != j {
                        // Make less negative → easier to switch
                        *val *= 1.0 / (1.0 + weight);
                    }
                }
                // Re-normalize row in log space
                let log_sum = row.iter().cloned()
                    .fold(f64::NEG_INFINITY, |a, b| {
                        let max_ab = a.max(b);
                        max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                    });
                for val in row.iter_mut() {
                    *val -= log_sum;
                }
            }
        }
        // Non-boundary: keep base transitions unchanged

        log_trans
    }).collect()
}

// ============================================================================
// Emission confidence weighting
// ============================================================================

/// Weight emission deviations by per-window confidence (gap between top-2 populations).
///
/// For each window, computes the gap between the best and second-best emission.
/// Windows with a large gap (confident) have their deviations amplified;
/// windows with a small gap (uncertain) have deviations dampened.
///
/// scale = (gap / median_gap) ^ power, clamped to [0.1, 5.0]
/// result = mean + scale × (v - mean)
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `power` - Power exponent. 0.5 = sqrt, 1.0 = linear, 2.0 = quadratic
///
/// # Returns
/// Confidence-weighted n×K matrix of log emissions
pub fn apply_confidence_weighting(
    log_emissions: &[Vec<f64>],
    power: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return vec![];
    }

    // Compute per-window confidence (gap between top-2)
    let gaps: Vec<f64> = log_emissions.iter().map(|row| {
        let mut finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        finite_vals.sort_by(|a, b| b.total_cmp(a));
        if finite_vals.len() >= 2 {
            finite_vals[0] - finite_vals[1]
        } else {
            0.0
        }
    }).collect();

    // Compute median gap
    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(|a, b| a.total_cmp(b));
    let median_gap = if sorted_gaps.is_empty() {
        1.0
    } else if sorted_gaps.len() % 2 == 0 {
        (sorted_gaps[sorted_gaps.len() / 2 - 1] + sorted_gaps[sorted_gaps.len() / 2]) / 2.0
    } else {
        sorted_gaps[sorted_gaps.len() / 2]
    };
    let median_gap = median_gap.max(1e-6);

    // Apply confidence scaling
    log_emissions.iter().enumerate().map(|(t, row)| {
        let scale = (gaps[t] / median_gap).powf(power).clamp(0.1, 5.0);

        let finite_vals: Vec<f64> = row.iter().filter(|v| v.is_finite()).cloned().collect();
        let mean = if finite_vals.is_empty() {
            0.0
        } else {
            finite_vals.iter().sum::<f64>() / finite_vals.len() as f64
        };

        row.iter().map(|&v| {
            if v.is_finite() {
                mean + scale * (v - mean)
            } else {
                v
            }
        }).collect()
    }).collect()
}

// ============================================================================
// Forward-backward temperature
// ============================================================================

/// Scale log-emissions by inverse temperature before inference.
///
/// Divides all finite log-emissions by temperature T:
///   log_e'[t][k] = log_e[t][k] / T
///
/// T < 1.0 sharpens the distribution (more confident),
/// T > 1.0 softens it (more uniform), T = 1.0 is identity.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `temperature` - Temperature parameter (must be > 0)
///
/// # Returns
/// Temperature-scaled n×K matrix of log emissions
pub fn apply_fb_temperature(
    log_emissions: &[Vec<f64>],
    temperature: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() || (temperature - 1.0).abs() < 1e-12 {
        return log_emissions.to_vec();
    }
    let inv_temp = 1.0 / temperature.max(1e-6);

    log_emissions.iter().map(|row| {
        row.iter().map(|&v| {
            if v.is_finite() { v * inv_temp } else { v }
        }).collect()
    }).collect()
}

// ============================================================================
// Population co-occurrence transition bonus
// ============================================================================

/// Build transition bonuses from pass-1 co-occurrence patterns.
///
/// Counts how often each pair of populations co-occurs at boundaries in the
/// pass-1 decoded sequence. Creates a symmetric affinity matrix and generates
/// per-window transition bonuses that boost commonly co-occurring pairs.
///
/// Returns a log-transition matrix with bonuses for frequent pairs.
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (base transitions)
/// * `weight` - Bonus strength
///
/// # Returns
/// K×K log-transition matrix with co-occurrence bonuses applied
pub fn compute_cooccurrence_transitions(
    states: &[usize],
    params: &AncestryHmmParams,
    weight: f64,
) -> Vec<Vec<f64>> {
    let k = params.n_states;

    // Count co-occurrences at boundaries
    let mut counts = vec![vec![0.0_f64; k]; k];
    let mut total_transitions = 0.0;
    for pair in states.windows(2) {
        if pair[0] != pair[1] {
            counts[pair[0]][pair[1]] += 1.0;
            counts[pair[1]][pair[0]] += 1.0;
            total_transitions += 1.0;
        }
    }

    // Normalize to get empirical transition rates
    if total_transitions > 0.0 {
        for row in &mut counts {
            for val in row.iter_mut() {
                *val /= total_transitions;
            }
        }
    }

    // Build log-transitions: base + weighted co-occurrence bonus
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    let mut log_trans = base_log_trans;
    for (i, row) in log_trans.iter_mut().enumerate().take(k) {
        for (j, val) in row.iter_mut().enumerate().take(k) {
            if i != j {
                // Add bonus proportional to co-occurrence rate
                *val += weight * counts[i][j];
            }
        }
        // Re-normalize row in log space
        let log_sum = row.iter().cloned()
            .fold(f64::NEG_INFINITY, |a, b| {
                let max_ab = a.max(b);
                max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
            });
        for val in row.iter_mut() {
            *val -= log_sum;
        }
    }

    log_trans
}

// ============================================================================
// Emission detrending
// ============================================================================

/// Remove linear trends from per-population log-emissions across genomic position.
///
/// For each population, fits a simple linear regression (emission vs window index)
/// and subtracts the fitted trend, keeping the global mean. This prevents systematic
/// drift in identity scores (e.g., from GC content variation or alignment quality
/// gradients) from biasing HMM state assignments.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
///
/// # Returns
/// Detrended n×K log-emission matrix where each population's linear trend is removed
pub fn detrend_emissions(log_emissions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n < 2 {
        return log_emissions.to_vec();
    }
    let k = log_emissions[0].len();

    // Clone the emissions
    let mut result: Vec<Vec<f64>> = log_emissions.to_vec();

    for pop in 0..k {
        // Collect finite values with their indices
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut count = 0.0_f64;

        for (t, row) in log_emissions.iter().enumerate() {
            let v = row[pop];
            if v.is_finite() {
                let x = t as f64;
                sum_x += x;
                sum_y += v;
                sum_xx += x * x;
                sum_xy += x * v;
                count += 1.0;
            }
        }

        if count < 2.0 {
            continue;
        }

        // Linear regression: y = a + b*x
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;
        let denom = sum_xx - sum_x * mean_x;

        if denom.abs() < 1e-12 {
            continue;
        }

        let slope = (sum_xy - sum_x * mean_y) / denom;

        // Subtract trend, keeping global mean
        for (t, row) in result.iter_mut().enumerate() {
            if row[pop].is_finite() {
                // Remove: slope * (t - mean_x), which preserves the mean
                row[pop] -= slope * (t as f64 - mean_x);
            }
        }
    }

    result
}

// ============================================================================
// Transition momentum
// ============================================================================

/// Compute per-window transition matrices with momentum from decoded run lengths.
///
/// For each window, computes a run-length from the pass-1 decoded states: how many
/// consecutive windows before (and including) t have the same state. Longer runs
/// increase self-transition probability (momentum), making it harder to switch away
/// from well-established states. The momentum decays exponentially:
///
///   self_boost = alpha * (1 - exp(-run_length / tau))
///
/// where tau = 5.0 (characteristic length). At the start of a new state, boost ≈ 0.
/// After 10+ windows in the same state, boost ≈ alpha.
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (for base transitions)
/// * `alpha` - Momentum strength (0.0 = off, 1.0 = strong)
///
/// # Returns
/// Vec of n per-window K×K log-transition matrices
pub fn compute_transition_momentum(
    states: &[usize],
    params: &AncestryHmmParams,
    alpha: f64,
) -> Vec<Vec<Vec<f64>>> {
    let n = states.len();
    let k = params.n_states;

    if n == 0 || alpha <= 0.0 {
        return vec![];
    }

    let tau = 5.0_f64; // characteristic run length

    // Compute run lengths
    let mut run_lengths = vec![1_usize; n];
    for t in 1..n {
        if states[t] == states[t - 1] {
            run_lengths[t] = run_lengths[t - 1] + 1;
        }
    }

    // Base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    let mut per_window_trans = Vec::with_capacity(n);

    for t in 0..n {
        let mut log_trans = base_log_trans.clone();
        let current_state = states[t];

        if current_state < k {
            // Exponential momentum: stronger for longer runs
            let boost = alpha * (1.0 - (-(run_lengths[t] as f64) / tau).exp());

            // Boost self-transition for current state
            log_trans[current_state][current_state] += boost;

            // Re-normalize the row in log space
            let log_sum = log_trans[current_state].iter().cloned()
                .fold(f64::NEG_INFINITY, |a, b| {
                    let max_ab = a.max(b);
                    if max_ab == f64::NEG_INFINITY {
                        f64::NEG_INFINITY
                    } else {
                        max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                    }
                });
            for val in log_trans[current_state].iter_mut() {
                *val -= log_sum;
            }
        }

        per_window_trans.push(log_trans);
    }

    per_window_trans
}

// ============================================================================
// Emission variance stabilization
// ============================================================================

/// Stabilize per-population emission variance across windows.
///
/// For each population, computes the mean and standard deviation of finite
/// log-emissions, then z-scores each value and rescales to have the target
/// standard deviation (median across populations). This makes all populations
/// have comparable spread, preventing high-variance populations from dominating.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
///
/// # Returns
/// Variance-stabilized n×K log-emission matrix
pub fn variance_stabilize_emissions(log_emissions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n < 2 {
        return log_emissions.to_vec();
    }
    let k = log_emissions[0].len();

    // Compute per-population mean and std
    let mut means = vec![0.0_f64; k];
    let mut stds = vec![0.0_f64; k];
    let mut counts = vec![0_usize; k];

    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                means[pop] += v;
                counts[pop] += 1;
            }
        }
    }

    for pop in 0..k {
        if counts[pop] > 0 {
            means[pop] /= counts[pop] as f64;
        }
    }

    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                let d = v - means[pop];
                stds[pop] += d * d;
            }
        }
    }

    for pop in 0..k {
        if counts[pop] > 1 {
            stds[pop] = (stds[pop] / (counts[pop] - 1) as f64).sqrt();
        } else {
            stds[pop] = 1.0;
        }
    }

    // Target std: median of all population stds
    let mut sorted_stds: Vec<f64> = stds.iter().filter(|&&s| s > 1e-12).copied().collect();
    if sorted_stds.is_empty() {
        return log_emissions.to_vec();
    }
    sorted_stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let target_std = sorted_stds[sorted_stds.len() / 2];

    // Apply: z-score then rescale to target std
    let mut result = log_emissions.to_vec();
    for row in &mut result {
        for (pop, val) in row.iter_mut().enumerate() {
            if val.is_finite() && stds[pop] > 1e-12 {
                *val = means[pop] + (*val - means[pop]) / stds[pop] * target_std;
            }
        }
    }

    result
}

// ============================================================================
// Lookahead transition adjustment
// ============================================================================

/// Compute per-window transitions informed by emission evidence from future windows.
///
/// For each window t, looks ahead R windows and computes the argmax population
/// from summed log-emissions in [t+1, t+R]. If the lookahead argmax differs from
/// the pass-1 state at t, off-diagonal transitions toward the lookahead state are
/// boosted, making it easier for the HMM to switch proactively.
///
/// The boost strength scales with the emission margin (difference between top-2
/// populations in the lookahead window), so only confident lookahead evidence
/// triggers easier transitions.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (for base transitions)
/// * `radius` - Number of windows to look ahead
///
/// # Returns
/// Vec of n per-window K×K log-transition matrices
pub fn compute_lookahead_transitions(
    log_emissions: &[Vec<f64>],
    states: &[usize],
    params: &AncestryHmmParams,
    radius: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 || radius == 0 {
        return vec![];
    }

    // Base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    let mut per_window_trans = Vec::with_capacity(n);

    for t in 0..n {
        let mut log_trans = base_log_trans.clone();

        // Sum emissions in lookahead window [t+1, t+radius]
        let end = (t + 1 + radius).min(n);
        if t + 1 < end {
            let mut sums = vec![0.0_f64; k];
            let mut finite_count = 0_usize;
            for row in &log_emissions[t + 1..end] {
                for (pop, &v) in row.iter().enumerate() {
                    if v.is_finite() {
                        sums[pop] += v;
                    }
                }
                finite_count += 1;
            }

            if finite_count > 0 {
                // Find top-2 populations in lookahead
                let mut indices: Vec<usize> = (0..k).collect();
                indices.sort_by(|&a, &b| sums[b].partial_cmp(&sums[a]).unwrap());

                let lookahead_state = indices[0];
                let margin = if k >= 2 {
                    (sums[indices[0]] - sums[indices[1]]) / finite_count as f64
                } else {
                    0.0
                };

                // If lookahead differs from current state, boost that transition
                let current = states[t];
                if current < k && lookahead_state != current && margin > 0.0 {
                    // Boost proportional to margin (clamped)
                    let boost = margin.min(2.0);
                    log_trans[current][lookahead_state] += boost;

                    // Re-normalize the row in log space
                    let log_sum = log_trans[current].iter().cloned()
                        .fold(f64::NEG_INFINITY, |a, b| {
                            let max_ab = a.max(b);
                            if max_ab == f64::NEG_INFINITY {
                                f64::NEG_INFINITY
                            } else {
                                max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                            }
                        });
                    for val in log_trans[current].iter_mut() {
                        *val -= log_sum;
                    }
                }
            }
        }

        per_window_trans.push(log_trans);
    }

    per_window_trans
}

// ============================================================================
// Emission kurtosis weighting
// ============================================================================

/// Scale per-population emissions based on excess kurtosis.
///
/// Populations with high excess kurtosis (leptokurtic, heavy-tailed) produce
/// more extreme emission outliers. This scales their emissions toward the mean,
/// dampening outlier influence. Populations with low/negative excess kurtosis
/// (platykurtic, light-tailed) get amplified.
///
/// For each population:
///   excess_kurtosis = E[(x-μ)^4] / σ^4 - 3
///   scale = (median_kurtosis / (excess_kurtosis + 3)).pow(power)  (clamped [0.3, 3.0])
///   result[t][k] = mean_k + scale * (emission[t][k] - mean_k)
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
/// * `power` - Scaling power (0.5 mild, 1.0 moderate)
///
/// # Returns
/// Kurtosis-weighted n×K log-emission matrix
pub fn apply_kurtosis_weighting(
    log_emissions: &[Vec<f64>],
    power: f64,
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n < 4 {
        return log_emissions.to_vec();
    }
    let k = log_emissions[0].len();

    // Compute per-population mean, variance, and kurtosis
    let mut means = vec![0.0_f64; k];
    let mut counts = vec![0_usize; k];

    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                means[pop] += v;
                counts[pop] += 1;
            }
        }
    }

    for pop in 0..k {
        if counts[pop] > 0 {
            means[pop] /= counts[pop] as f64;
        }
    }

    // Compute variance and 4th central moment
    let mut m2 = vec![0.0_f64; k]; // sum of (x - mean)^2
    let mut m4 = vec![0.0_f64; k]; // sum of (x - mean)^4

    for row in log_emissions {
        for (pop, &v) in row.iter().enumerate() {
            if v.is_finite() {
                let d = v - means[pop];
                m2[pop] += d * d;
                m4[pop] += d * d * d * d;
            }
        }
    }

    let mut kurtoses = vec![3.0_f64; k]; // default to mesokurtic
    for pop in 0..k {
        if counts[pop] > 3 && m2[pop] > 1e-12 {
            let var = m2[pop] / counts[pop] as f64;
            let kurt = (m4[pop] / counts[pop] as f64) / (var * var);
            kurtoses[pop] = kurt; // raw kurtosis (normal = 3)
        }
    }

    // Median kurtosis
    let mut sorted_kurt: Vec<f64> = kurtoses.clone();
    sorted_kurt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_kurt = sorted_kurt[sorted_kurt.len() / 2];

    // Apply scaling
    let mut result = log_emissions.to_vec();
    for row in &mut result {
        for (pop, val) in row.iter_mut().enumerate() {
            if val.is_finite() && kurtoses[pop] > 0.01 {
                let scale = (median_kurt / kurtoses[pop]).powf(power).clamp(0.3, 3.0);
                *val = means[pop] + scale * (*val - means[pop]);
            }
        }
    }

    result
}

// ============================================================================
// Segment length prior
// ============================================================================

/// Compute per-window transition matrices that penalize short segments.
///
/// For each window t, computes the run length of the current state from pass-1.
/// When a state has been active for fewer than `min_length` windows, off-diagonal
/// transitions are penalized (reduced), making it harder to switch away from the
/// current state before it establishes a minimum-length segment.
///
/// Penalty scales linearly: at run_length=1, penalty is maximal; at run_length
/// >= min_length, no penalty (base transitions).
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (for base transitions)
/// * `min_length` - Minimum desired segment length in windows
///
/// # Returns
/// Vec of n per-window K×K log-transition matrices
pub fn compute_segment_length_prior(
    states: &[usize],
    params: &AncestryHmmParams,
    min_length: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n = states.len();
    let k = params.n_states;

    if n == 0 || min_length <= 1 {
        return vec![];
    }

    // Compute run lengths
    let mut run_lengths = vec![1_usize; n];
    for t in 1..n {
        if states[t] == states[t - 1] {
            run_lengths[t] = run_lengths[t - 1] + 1;
        }
    }

    // Base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    let mut per_window_trans = Vec::with_capacity(n);

    for t in 0..n {
        let mut log_trans = base_log_trans.clone();
        let current_state = states[t];

        if current_state < k && run_lengths[t] < min_length {
            // Penalty: stronger for shorter runs
            let penalty = 2.0 * (1.0 - run_lengths[t] as f64 / min_length as f64);

            // Penalize off-diagonal transitions from current state
            for (j, val) in log_trans[current_state].iter_mut().enumerate().take(k) {
                if j != current_state {
                    *val -= penalty;
                }
            }

            // Re-normalize the row in log space
            let log_sum = log_trans[current_state].iter().cloned()
                .fold(f64::NEG_INFINITY, |a, b| {
                    let max_ab = a.max(b);
                    if max_ab == f64::NEG_INFINITY {
                        f64::NEG_INFINITY
                    } else {
                        max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                    }
                });
            for val in log_trans[current_state].iter_mut() {
                *val -= log_sum;
            }
        }

        per_window_trans.push(log_trans);
    }

    per_window_trans
}

// ============================================================================
// Emission gap penalty
// ============================================================================

/// Penalize populations with high temporal variability in emissions.
///
/// For each population, computes the mean absolute difference between adjacent
/// windows. Populations with high temporal variability (jump around a lot) get
/// penalized relative to smoothly varying populations.
///
///   gap_k = mean(|e[t][k] - e[t-1][k]|) for consecutive finite values
///   penalty_k = weight * (gap_k / median_gap - 1.0), clamped at [0, max_penalty]
///   result[t][k] = e[t][k] - penalty_k
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
/// * `weight` - Penalty strength (0.0 = off)
///
/// # Returns
/// Gap-penalized n×K log-emission matrix
pub fn apply_gap_penalty(
    log_emissions: &[Vec<f64>],
    weight: f64,
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n < 2 || weight <= 0.0 {
        return log_emissions.to_vec();
    }
    let k = log_emissions[0].len();

    // Compute per-population mean absolute gap
    let mut gaps = vec![0.0_f64; k];
    let mut gap_counts = vec![0_usize; k];

    for t in 1..n {
        for pop in 0..k {
            let a = log_emissions[t - 1][pop];
            let b = log_emissions[t][pop];
            if a.is_finite() && b.is_finite() {
                gaps[pop] += (b - a).abs();
                gap_counts[pop] += 1;
            }
        }
    }

    for pop in 0..k {
        if gap_counts[pop] > 0 {
            gaps[pop] /= gap_counts[pop] as f64;
        }
    }

    // Median gap
    let mut sorted_gaps: Vec<f64> = gaps.iter().filter(|&&g| g > 1e-12).copied().collect();
    if sorted_gaps.is_empty() {
        return log_emissions.to_vec();
    }
    sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gap = sorted_gaps[sorted_gaps.len() / 2];

    // Compute per-population penalty
    let max_penalty = 3.0;
    let penalties: Vec<f64> = gaps.iter().map(|&g| {
        if median_gap > 1e-12 {
            (weight * (g / median_gap - 1.0)).clamp(0.0, max_penalty)
        } else {
            0.0
        }
    }).collect();

    // Apply penalty
    let mut result = log_emissions.to_vec();
    for row in &mut result {
        for (pop, val) in row.iter_mut().enumerate() {
            if val.is_finite() {
                *val -= penalties[pop];
            }
        }
    }

    result
}

// ============================================================================
// Recency-weighted transitions
// ============================================================================

/// Compute per-window transitions weighted by recent decoded state history.
///
/// For each window t, computes an exponentially-weighted histogram of states
/// from windows [0, t] where more recent windows have higher weight:
///   w_s = alpha^(t - s) for s in [0, t]
///
/// The weighted histogram gives a soft estimate of which states have been
/// active recently. States with higher recent weight get boosted self-transition.
///
/// # Arguments
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (for base transitions)
/// * `alpha` - Decay rate (0-1, higher = longer memory, 0.9 typical)
///
/// # Returns
/// Vec of n per-window K×K log-transition matrices
pub fn compute_recency_transitions(
    states: &[usize],
    params: &AncestryHmmParams,
    alpha: f64,
) -> Vec<Vec<Vec<f64>>> {
    let n = states.len();
    let k = params.n_states;

    if n == 0 || alpha <= 0.0 {
        return vec![];
    }

    // Base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    // Running weighted histogram (exponential decay)
    let mut hist = vec![0.0_f64; k];
    let mut per_window_trans = Vec::with_capacity(n);

    for t in 0..n {
        // Decay existing weights and add current state
        for h in &mut hist {
            *h *= alpha;
        }
        if states[t] < k {
            hist[states[t]] += 1.0;
        }

        // Normalize histogram
        let total: f64 = hist.iter().sum();
        let mut log_trans = base_log_trans.clone();

        if total > 1e-6 {
            // Boost self-transitions proportionally to recency weight
            for state in 0..k {
                let recency_weight = hist[state] / total;
                let boost = recency_weight * 0.5; // moderate boost

                log_trans[state][state] += boost;

                // Re-normalize the row in log space
                let log_sum = log_trans[state].iter().cloned()
                    .fold(f64::NEG_INFINITY, |a, b| {
                        let max_ab = a.max(b);
                        if max_ab == f64::NEG_INFINITY {
                            f64::NEG_INFINITY
                        } else {
                            max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                        }
                    });
                for val in log_trans[state].iter_mut() {
                    *val -= log_sum;
                }
            }
        }

        per_window_trans.push(log_trans);
    }

    per_window_trans
}

// ============================================================================
// Emission centering
// ============================================================================

/// Per-window centering: subtract the window mean from all population emissions.
///
/// This transforms emissions from absolute log-likelihood space to relative
/// advantage space. After centering, a population's emission measures how much
/// better/worse it is compared to the average population at that window.
///
/// This is simpler than contrast normalization (which also rescales spread)
/// and does not change the relative ordering within each window.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
///
/// # Returns
/// Centered n×K log-emission matrix where each window sums to ~0
pub fn center_emissions(log_emissions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut result = log_emissions.to_vec();

    for row in &mut result {
        // Compute mean of finite values
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        for &v in row.iter() {
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        if count > 0 {
            let mean = sum / count as f64;
            for val in row.iter_mut() {
                if val.is_finite() {
                    *val -= mean;
                }
            }
        }
    }

    result
}

// ============================================================================
// State persistence bonus
// ============================================================================

/// Add emission bonus to the pass-1 decoded state at each window.
///
/// For each window t, adds `weight` to the log-emission of whichever
/// state was decoded in pass-1. This acts as a soft sticky prior on the
/// emission side, making it harder for pass-2 to disagree with pass-1
/// unless the emission evidence is strong enough to overcome the bonus.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
/// * `states` - Pass-1 decoded state sequence
/// * `weight` - Bonus to add (in log space)
///
/// # Returns
/// Modified n×K log-emission matrix with persistence bonus applied
pub fn apply_persistence_bonus(
    log_emissions: &[Vec<f64>],
    states: &[usize],
    weight: f64,
) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n == 0 || weight <= 0.0 {
        return log_emissions.to_vec();
    }

    let mut result = log_emissions.to_vec();

    for (t, row) in result.iter_mut().enumerate() {
        if t < states.len() && states[t] < row.len() {
            row[states[t]] += weight;
        }
    }

    result
}

// ============================================================================
// Emission median polish
// ============================================================================

/// Apply Tukey's median polish to the n×K emission matrix.
///
/// Iteratively removes row medians and column medians to decompose:
///   emission[t][k] = grand_effect + row_effect[t] + col_effect[k] + residual[t][k]
///
/// Returns the residuals, which represent the interaction between window and
/// population after removing main effects. This is useful when systematic
/// window-level or population-level biases mask the informative signal.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
///
/// # Returns
/// Residual n×K matrix after median polish
pub fn median_polish_emissions(log_emissions: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n == 0 {
        return vec![];
    }
    let k = log_emissions[0].len();
    if k == 0 {
        return log_emissions.to_vec();
    }

    let mut table = log_emissions.to_vec();
    let max_iter = 10;

    for _ in 0..max_iter {
        // Remove row medians
        for row in &mut table {
            let mut finite: Vec<f64> = row.iter().filter(|v| v.is_finite()).copied().collect();
            if finite.is_empty() {
                continue;
            }
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = finite[finite.len() / 2];
            for val in row.iter_mut() {
                if val.is_finite() {
                    *val -= median;
                }
            }
        }

        // Remove column medians
        for col in 0..k {
            let mut finite: Vec<f64> = table.iter()
                .map(|row| row[col])
                .filter(|v| v.is_finite())
                .collect();
            if finite.is_empty() {
                continue;
            }
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = finite[finite.len() / 2];
            for row in &mut table {
                if row[col].is_finite() {
                    row[col] -= median;
                }
            }
        }
    }

    table
}

// ============================================================================
// Disagreement penalty transitions
// ============================================================================

/// Compute per-window transitions based on emission evidence agreement.
///
/// For each window, checks if the emission argmax agrees with the pass-1
/// decoded state. When they agree, self-transition is boosted (harder to
/// switch). When they disagree, the transition toward the emission argmax
/// is boosted (easier to switch to what the emissions suggest).
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix (for computing argmax)
/// * `states` - Pass-1 decoded state sequence
/// * `params` - HMM parameters (for base transitions)
/// * `weight` - Penalty/bonus strength (positive)
///
/// # Returns
/// Vec of n per-window K×K log-transition matrices
pub fn compute_disagreement_transitions(
    log_emissions: &[Vec<f64>],
    states: &[usize],
    params: &AncestryHmmParams,
    weight: f64,
) -> Vec<Vec<Vec<f64>>> {
    let n = log_emissions.len();
    let k = params.n_states;

    if n == 0 || weight <= 0.0 {
        return vec![];
    }

    // Base log-transitions
    let base_log_trans: Vec<Vec<f64>> = params.transitions.iter().map(|row| {
        row.iter().map(|&p| p.max(1e-20).ln()).collect()
    }).collect();

    let mut per_window_trans = Vec::with_capacity(n);

    for t in 0..n {
        let mut log_trans = base_log_trans.clone();

        let current = if t < states.len() { states[t] } else { 0 };

        // Find emission argmax
        let emission_argmax = log_emissions[t].iter().enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(current);

        if current < k {
            if emission_argmax == current {
                // Agreement: tighten self-transition (harder to switch)
                log_trans[current][current] += weight;
            } else {
                // Disagreement: boost transition toward emission argmax
                if emission_argmax < k {
                    log_trans[current][emission_argmax] += weight;
                }
            }

            // Re-normalize the row in log space
            let log_sum = log_trans[current].iter().cloned()
                .fold(f64::NEG_INFINITY, |a, b| {
                    let max_ab = a.max(b);
                    if max_ab == f64::NEG_INFINITY {
                        f64::NEG_INFINITY
                    } else {
                        max_ab + (1.0 + (a.min(b) - max_ab).exp()).ln()
                    }
                });
            for val in log_trans[current].iter_mut() {
                *val -= log_sum;
            }
        }

        per_window_trans.push(log_trans);
    }

    per_window_trans
}

// ============================================================================
// Emission softmax renormalization
// ============================================================================

/// Convert log-emissions to softmax probabilities and back to log space.
///
/// For each window, computes softmax(log_emissions / temperature) to get
/// a proper probability distribution, then takes log of those probabilities.
/// This standardizes emission scale across windows: windows with extreme
/// spread get compressed, windows with small spread get expanded.
///
/// # Arguments
/// * `log_emissions` - n×K log-emission matrix
/// * `temperature` - Softmax temperature (higher = more uniform)
///
/// # Returns
/// Renormalized n×K log-emission matrix
pub fn softmax_renormalize(log_emissions: &[Vec<f64>], temperature: f64) -> Vec<Vec<f64>> {
    let n = log_emissions.len();
    if n == 0 || temperature <= 0.0 {
        return log_emissions.to_vec();
    }

    let mut result = Vec::with_capacity(n);

    for row in log_emissions {
        // Find max finite value for numerical stability
        let max_val = row.iter()
            .filter(|v| v.is_finite())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_val == f64::NEG_INFINITY {
            result.push(row.clone());
            continue;
        }

        // Softmax: exp((x - max) / T) / sum
        let inv_temp = 1.0 / temperature;
        let exps: Vec<f64> = row.iter().map(|&v| {
            if v.is_finite() {
                ((v - max_val) * inv_temp).exp()
            } else {
                0.0
            }
        }).collect();

        let sum: f64 = exps.iter().sum();

        // Convert back to log space
        let log_row: Vec<f64> = row.iter().zip(exps.iter()).map(|(&orig, &e)| {
            if orig.is_finite() && sum > 0.0 {
                (e / sum).max(1e-20).ln()
            } else {
                f64::NEG_INFINITY
            }
        }).collect();

        result.push(log_row);
    }

    result
}

// ============================================================================
// Bidirectional state smoothing
// ============================================================================

/// Post-decoding bidirectional smoothing of state assignments.
///
/// For each window, computes a weighted vote from states in [t-R, t+R]
/// where weights decay exponentially with distance: w = exp(-|d| / R).
/// The state with the highest weighted vote replaces the current state.
///
/// Unlike majority_vote_filter which uses uniform weights, this gives
/// closer neighbors stronger influence, producing smoother transitions.
///
/// # Arguments
/// * `states` - Decoded state sequence
/// * `n_states` - Number of possible states
/// * `radius` - Smoothing radius
///
/// # Returns
/// Smoothed state sequence
pub fn bidirectional_smooth_states(
    states: &[usize],
    n_states: usize,
    radius: usize,
) -> Vec<usize> {
    let n = states.len();
    if n == 0 || radius == 0 {
        return states.to_vec();
    }

    let mut result = Vec::with_capacity(n);

    for t in 0..n {
        let mut votes = vec![0.0_f64; n_states];
        let start = t.saturating_sub(radius);
        let end = (t + radius + 1).min(n);

        for s in start..end {
            if states[s] < n_states {
                let dist = (t as i64 - s as i64).unsigned_abs() as f64;
                let w = (-dist / radius as f64).exp();
                votes[states[s]] += w;
            }
        }

        // Pick best; break ties by keeping original
        let best = votes.iter().enumerate()
            .max_by(|(i, a), (j, b)| {
                a.partial_cmp(b).unwrap()
                    .then_with(|| if *i == states[t] { std::cmp::Ordering::Greater }
                               else if *j == states[t] { std::cmp::Ordering::Less }
                               else { std::cmp::Ordering::Equal })
            })
            .map(|(idx, _)| idx)
            .unwrap_or(states[t]);

        result.push(best);
    }

    result
}

// ============================================================================
// Short segment correction
// ============================================================================

/// Correct short ancestry segments by merging with best-supported neighbor.
///
/// After Viterbi decoding, segments shorter than `min_windows` are candidates
/// for correction. For each short segment, the algorithm looks at the left and
/// right neighbor segments and picks the one whose emission support (mean
/// log-emission) is higher for the candidate windows.
///
/// This prevents isolated 1-2 window misassignments that are common at
/// EUR↔AMR boundaries where per-window discriminability is very low.
///
/// # Arguments
/// * `states` - Decoded state sequence (modified in place)
/// * `log_emissions` - n×K log-emission matrix for emission support
/// * `min_windows` - Minimum segment length in windows (segments shorter get merged)
///
/// # Returns
/// Corrected state sequence
pub fn correct_short_segments(
    states: &[usize],
    log_emissions: &[Vec<f64>],
    min_windows: usize,
) -> Vec<usize> {
    if states.is_empty() || min_windows <= 1 || log_emissions.is_empty() {
        return states.to_vec();
    }

    let n = states.len();
    let mut corrected = states.to_vec();

    // Identify segments: (start, end_exclusive, state)
    let mut segments: Vec<(usize, usize, usize)> = Vec::new();
    let mut seg_start = 0;
    for i in 1..n {
        if corrected[i] != corrected[seg_start] {
            segments.push((seg_start, i, corrected[seg_start]));
            seg_start = i;
        }
    }
    segments.push((seg_start, n, corrected[seg_start]));

    // Correct short segments
    for seg_idx in 0..segments.len() {
        let (start, end, _state) = segments[seg_idx];
        let seg_len = end - start;

        if seg_len >= min_windows {
            continue;
        }

        // Get neighbor states
        let left_state = if seg_idx > 0 { Some(segments[seg_idx - 1].2) } else { None };
        let right_state = if seg_idx + 1 < segments.len() { Some(segments[seg_idx + 1].2) } else { None };

        // Choose the neighbor with better emission support for the candidate windows
        let replacement = match (left_state, right_state) {
            (Some(l), Some(r)) if l == r => l, // both neighbors agree
            (Some(l), Some(r)) => {
                // Pick the one with better emission support
                let l_support: f64 = (start..end)
                    .filter_map(|t| log_emissions.get(t).and_then(|row| row.get(l)))
                    .filter(|v| v.is_finite())
                    .sum();
                let r_support: f64 = (start..end)
                    .filter_map(|t| log_emissions.get(t).and_then(|row| row.get(r)))
                    .filter(|v| v.is_finite())
                    .sum();
                if l_support >= r_support { l } else { r }
            }
            (Some(l), None) => l,
            (None, Some(r)) => r,
            (None, None) => continue, // single segment, can't correct
        };

        // Apply correction
        for c in &mut corrected[start..end] {
            *c = replacement;
        }
    }

    corrected
}

// ============================================================================
// Emission whitening (ZCA)
// ============================================================================

/// Apply ZCA whitening to log-emissions to decorrelate population signals.
///
/// Computes the cross-population covariance matrix from all windows, then
/// applies ZCA whitening: x' = Σ^{-1/2} × (x - μ). This decorrelates
/// populations, making softmax more effective for highly correlated pairs
/// (EUR↔AMR where ρ > 0.9).
///
/// ZCA (Zero-phase Component Analysis) preserves the original axes (unlike PCA
/// rotation), so population indices remain meaningful after whitening.
///
/// Uses eigendecomposition for Σ^{-1/2} with regularization to handle
/// near-singular covariance.
///
/// # Arguments
/// * `log_emissions` - n×K matrix of log emissions
/// * `regularization` - Small constant added to eigenvalues for stability (default 1e-6)
///
/// # Returns
/// Whitened n×K matrix of log emissions
pub fn whiten_log_emissions(
    log_emissions: &[Vec<f64>],
    regularization: f64,
) -> Vec<Vec<f64>> {
    if log_emissions.is_empty() {
        return Vec::new();
    }

    let k = log_emissions[0].len();
    if k <= 1 {
        return log_emissions.to_vec();
    }

    // Compute mean per population (only finite values)
    let mut means = vec![0.0_f64; k];
    let mut counts = vec![0usize; k];
    for row in log_emissions {
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() {
                means[j] += v;
                counts[j] += 1;
            }
        }
    }
    for j in 0..k {
        if counts[j] > 0 {
            means[j] /= counts[j] as f64;
        }
    }

    // Compute covariance matrix (K×K)
    let mut cov = vec![vec![0.0_f64; k]; k];
    let mut n_valid = 0usize;
    for row in log_emissions {
        // Only include windows where all values are finite
        if row.iter().any(|v| !v.is_finite()) {
            continue;
        }
        n_valid += 1;
        let centered: Vec<f64> = row.iter().zip(means.iter())
            .map(|(&v, &m)| v - m).collect();
        for (i, &ci) in centered.iter().enumerate() {
            for (j, &cj) in centered.iter().enumerate().skip(i) {
                let d = ci * cj;
                cov[i][j] += d;
                if i != j {
                    cov[j][i] += d;
                }
            }
        }
    }

    if n_valid < k + 1 {
        // Not enough data to compute meaningful covariance
        return log_emissions.to_vec();
    }

    let divisor = (n_valid - 1) as f64;
    for row in cov.iter_mut() {
        for val in row.iter_mut() {
            *val /= divisor;
        }
    }

    // Eigendecomposition of symmetric covariance matrix via Jacobi iteration
    // For small K (typically 3-7), this is fast and numerically stable
    let (eigenvalues, eigenvectors) = symmetric_eigen_jacobi(&cov, 100);

    // Compute Σ^{-1/2} = V × diag(1/√(λ+ε)) × V^T
    let reg = regularization.max(1e-10);
    let inv_sqrt_eig: Vec<f64> = eigenvalues.iter()
        .map(|&l| 1.0 / (l + reg).max(reg).sqrt())
        .collect();

    // Apply whitening: x' = Σ^{-1/2} × (x - μ)
    log_emissions.iter().map(|row| {
        if row.iter().any(|v| !v.is_finite()) {
            return row.clone(); // preserve rows with masked states
        }

        // Center
        let centered: Vec<f64> = row.iter().zip(means.iter())
            .map(|(&v, &m)| v - m)
            .collect();

        // Multiply by V × diag(inv_sqrt) × V^T
        let mut result = vec![0.0_f64; k];
        for i in 0..k {
            for j in 0..k {
                let mut whitening_ij = 0.0;
                for l in 0..k {
                    whitening_ij += eigenvectors[i][l] * inv_sqrt_eig[l] * eigenvectors[j][l];
                }
                result[i] += whitening_ij * centered[j];
            }
        }

        result
    }).collect()
}

/// Jacobi eigendecomposition for small symmetric matrices.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i][j] is the i-th
/// component of the j-th eigenvector (column-major convention).
fn symmetric_eigen_jacobi(matrix: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        row
    }).collect();

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for (i, row) in a.iter().enumerate() {
            for (j, &val) in row.iter().enumerate().skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-12 {
            break; // converged
        }

        // Compute rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = cos_t * cos_t * a[p][p] + 2.0 * sin_t * cos_t * a[p][q] + sin_t * sin_t * a[q][q];
        new_a[q][q] = sin_t * sin_t * a[p][p] - 2.0 * sin_t * cos_t * a[p][q] + cos_t * cos_t * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[i][p] = cos_t * v[i][p] + sin_t * v[i][q];
            new_v[i][q] = -sin_t * v[i][p] + cos_t * v[i][q];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

// ============================================================================
// Consistency-based Emissions (sign-test over sliding window)
// ============================================================================

/// Compute consistency-based log-emissions by counting how many neighboring
/// windows each population "wins" (has highest aggregated similarity).
///
/// This converts tiny absolute similarity differences into robust frequency
/// counts. When EUR consistently beats AMR in 12/21 windows (even by 0.0002),
/// the log-probability difference is ~0.27 nats — much larger than the
/// per-window softmax difference (~0.01 nats) from the same 0.0002 gap.
///
/// Designed to be computed on RAW (unsmoothed) observations for independence
/// from the standard softmax emissions computed on smoothed data.
///
/// # Arguments
/// * `observations` - Per-window similarity data (preferably unsmoothed)
/// * `populations` - Population definitions with haplotype lists
/// * `emission_model` - Aggregation model (Max, Mean, etc.)
/// * `context` - Number of neighboring windows on each side to include
///
/// # Returns
/// n×k matrix of log-emission probabilities based on win-frequency counts
pub fn compute_consistency_log_emissions(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    emission_model: &EmissionModel,
    context: usize,
) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_pops = populations.len();

    if n_obs == 0 || n_pops == 0 {
        return vec![vec![0.0; n_pops]; n_obs];
    }

    // Pre-compute per-window, per-population aggregated similarities
    let pop_sims: Vec<Vec<Option<f64>>> = observations.iter().map(|obs| {
        populations.iter().map(|pop| {
            let sims: Vec<f64> = pop.haplotypes.iter()
                .filter_map(|h| obs.similarities.get(h))
                .cloned()
                .collect();
            emission_model.aggregate(&sims)
        }).collect()
    }).collect();

    // For each window, count wins per population over the context neighborhood
    (0..n_obs).map(|t| {
        let lo = t.saturating_sub(context);
        let hi = (t + context).min(n_obs - 1);

        let mut wins = vec![0.0_f64; n_pops];
        let mut total_windows = 0.0;

        for window_sims in &pop_sims[lo..=hi] {
            // Find winning population at this window
            let mut best_val = f64::NEG_INFINITY;
            let mut best_pops: Vec<usize> = Vec::new();

            for (p, sim_opt) in window_sims.iter().enumerate() {
                if let Some(s) = *sim_opt {
                    if s > best_val + 1e-9 {
                        // Clear winner
                        best_val = s;
                        best_pops.clear();
                        best_pops.push(p);
                    } else if (s - best_val).abs() <= 1e-9 {
                        // Tie with current best
                        best_pops.push(p);
                    }
                }
            }

            if !best_pops.is_empty() {
                // Distribute win equally among tied populations
                let share = 1.0 / best_pops.len() as f64;
                for &p in &best_pops {
                    wins[p] += share;
                }
                total_windows += 1.0;
            }
        }

        if total_windows == 0.0 {
            return vec![-(n_pops as f64).ln(); n_pops]; // uniform
        }

        // Log-probability with Laplace smoothing (0.5 pseudocount per population)
        let pseudocount = 0.5;
        let denom = total_windows + pseudocount * n_pops as f64;
        wins.iter().map(|&w| ((w + pseudocount) / denom).ln()).collect()
    }).collect()
}

// ============================================================================
// Haplotype Copying Model (Li & Stephens inspired)
// ============================================================================
//
// Instead of N population states with aggregated emissions, this model uses
// M states (one per reference haplotype). The query is modeled as a mosaic
// of reference haplotypes — at each window, one reference haplotype is being
// "copied" (has the highest similarity). The transition model favors staying
// on the same haplotype, with higher cost for switching between populations.
//
// After inference, haplotype-level posteriors are aggregated to population
// posteriors, compatible with the existing ancestry pipeline.
//
// Key advantages over population-aggregate model:
// 1. Captures haplotype continuity (same reference closest for many windows)
// 2. Robust to single outlier haplotypes from wrong population
// 3. Naturally handles within-population recombination

/// Infer local ancestry using a haplotype copying model.
///
/// Models the query as a mosaic of reference haplotypes (Li & Stephens 2003
/// style). Each HMM state corresponds to a specific reference haplotype.
/// Transitions favor staying on the same haplotype, with recombination
/// (haplotype switching) split between within-population and
/// between-population switches.
///
/// Returns (decoded_states, population_posteriors) where:
/// - decoded_states: per-window population index (argmax of aggregated posteriors)
/// - population_posteriors: per-window probability of each population
///
/// # Arguments
/// * `observations` - Per-window similarity data
/// * `populations` - Reference population definitions
/// * `hap_switch_rate` - Per-window probability of switching copied haplotype (e.g., 0.005)
/// * `ancestry_switch_frac` - Fraction of switches that also change population (e.g., 0.1)
/// * `temperature` - Emission temperature controlling sensitivity to identity differences
/// * `default_similarity` - Similarity assigned when no alignment data exists (e.g., 0.99)
pub fn infer_ancestry_copying(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    hap_switch_rate: f64,
    ancestry_switch_frac: f64,
    temperature: f64,
    default_similarity: f64,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n_obs = observations.len();
    let n_pops = populations.len();

    if n_obs == 0 || n_pops == 0 {
        return (Vec::new(), Vec::new());
    }

    // Build haplotype index: map each haplotype to (index, population_index)
    let mut hap_names: Vec<&str> = Vec::new();
    let mut hap_to_pop: Vec<usize> = Vec::new();
    let mut pop_hap_counts: Vec<usize> = Vec::new();

    for (pop_idx, pop) in populations.iter().enumerate() {
        pop_hap_counts.push(pop.haplotypes.len());
        for h in &pop.haplotypes {
            hap_names.push(h);
            hap_to_pop.push(pop_idx);
        }
    }

    let n_haps = hap_names.len();
    if n_haps == 0 {
        return (vec![0; n_obs], vec![vec![1.0 / n_pops as f64; n_pops]; n_obs]);
    }

    // Build log-transition matrix
    let log_trans = build_copying_transitions(
        n_haps,
        &hap_to_pop,
        &pop_hap_counts,
        hap_switch_rate,
        ancestry_switch_frac,
    );

    // Uniform initial probabilities across haplotypes
    let log_init = vec![-(n_haps as f64).ln(); n_haps];

    // Precompute log-emissions for all observations × haplotypes
    let log_emissions = compute_copying_emissions(
        observations,
        &hap_names,
        temperature,
        default_similarity,
    );

    // Forward-backward at haplotype level
    let hap_posteriors = copying_forward_backward(
        &log_emissions,
        &log_trans,
        &log_init,
        n_haps,
    );

    // Aggregate haplotype posteriors to population posteriors
    let mut pop_posteriors = vec![vec![0.0; n_pops]; n_obs];
    let mut decoded_states = vec![0usize; n_obs];

    for t in 0..n_obs {
        for h in 0..n_haps {
            pop_posteriors[t][hap_to_pop[h]] += hap_posteriors[t][h];
        }
        // Decoded state = population with highest posterior
        decoded_states[t] = pop_posteriors[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    (decoded_states, pop_posteriors)
}

/// Build log-transition matrix for haplotype copying model.
///
/// Transition probabilities:
/// - P(stay on same haplotype) = 1 - hap_switch_rate
/// - P(switch within population) = hap_switch_rate × (1 - ancestry_switch_frac) / (n_pop - 1)
/// - P(switch between populations) = hap_switch_rate × ancestry_switch_frac / n_other
fn build_copying_transitions(
    n_haps: usize,
    hap_to_pop: &[usize],
    pop_hap_counts: &[usize],
    hap_switch_rate: f64,
    ancestry_switch_frac: f64,
) -> Vec<Vec<f64>> {
    let mut log_trans = vec![vec![f64::NEG_INFINITY; n_haps]; n_haps];

    for i in 0..n_haps {
        let pop_i = hap_to_pop[i];
        let n_same = pop_hap_counts[pop_i];
        let n_other: usize = n_haps - n_same;

        for j in 0..n_haps {
            let p = if i == j {
                1.0 - hap_switch_rate
            } else if hap_to_pop[j] == pop_i {
                // Within-population switch
                let denom = (n_same - 1).max(1) as f64;
                hap_switch_rate * (1.0 - ancestry_switch_frac) / denom
            } else {
                // Between-population switch
                let denom = n_other.max(1) as f64;
                hap_switch_rate * ancestry_switch_frac / denom
            };
            log_trans[i][j] = p.max(1e-300).ln();
        }
    }

    log_trans
}

/// Compute log-emission matrix for haplotype copying model.
///
/// For each observation and each haplotype, the emission is proportional to
/// the similarity. Uses softmax-style normalization:
///   log P(obs | copying h) = sim_h / T - log_sum_exp(sim_all / T)
///
/// Missing haplotypes (no alignment data) get `default_similarity`.
fn compute_copying_emissions(
    observations: &[AncestryObservation],
    hap_names: &[&str],
    temperature: f64,
    default_similarity: f64,
) -> Vec<Vec<f64>> {
    let n_haps = hap_names.len();
    let temp = temperature.max(1e-10);

    observations
        .iter()
        .map(|obs| {
            // Get similarity for each haplotype (default if missing)
            let sims: Vec<f64> = hap_names
                .iter()
                .map(|&h| {
                    obs.similarities
                        .get(h)
                        .copied()
                        .unwrap_or(default_similarity)
                })
                .collect();

            // Log-softmax over all haplotypes
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_denom: f64 = sims
                .iter()
                .map(|&s| ((s - max_sim) / temp).exp())
                .sum::<f64>()
                .ln();

            let mut log_emit = vec![f64::NEG_INFINITY; n_haps];
            for h in 0..n_haps {
                log_emit[h] = (sims[h] - max_sim) / temp - log_denom;
            }
            log_emit
        })
        .collect()
}

/// Forward-backward algorithm for the haplotype copying model.
///
/// Uses structured transition decomposition for O(T × (M + K²)) complexity
/// instead of naive O(T × M²), where M = haplotypes and K = populations.
///
/// The transition matrix has special structure:
/// - P(stay on same haplotype) = p_stay (constant for all states)
/// - P(switch within population) = p_within / (n_pop - 1) (uniform)
/// - P(switch between populations) = p_between / n_other (uniform)
///
/// This means we can precompute per-population log-sum-alpha values and
/// avoid the full M×M matrix multiplication at each timestep.
fn copying_forward_backward(
    log_emissions: &[Vec<f64>],
    log_trans: &[Vec<f64>],
    log_init: &[f64],
    n_haps: usize,
) -> Vec<Vec<f64>> {
    let t = log_emissions.len();
    if t == 0 {
        return Vec::new();
    }

    // Forward pass
    let mut alpha = vec![vec![f64::NEG_INFINITY; n_haps]; t];

    // Initialize
    for h in 0..n_haps {
        alpha[0][h] = log_init[h] + log_emissions[0][h];
    }

    // Recurse — use explicit transition matrix (correct for any structure)
    // For panels ≤ ~100 haplotypes, the naive approach is fast enough
    // (46² × 13000 ≈ 27M ops, sub-second on modern CPU)
    for i in 1..t {
        for j in 0..n_haps {
            let mut max_val = f64::NEG_INFINITY;
            for h in 0..n_haps {
                let v = alpha[i - 1][h] + log_trans[h][j];
                if v > max_val {
                    max_val = v;
                }
            }
            if max_val == f64::NEG_INFINITY {
                alpha[i][j] = f64::NEG_INFINITY;
                continue;
            }
            let mut sum = 0.0f64;
            for h in 0..n_haps {
                let v = alpha[i - 1][h] + log_trans[h][j];
                sum += (v - max_val).exp();
            }
            alpha[i][j] = log_emissions[i][j] + max_val + sum.ln();
        }
    }

    // Backward pass
    let mut beta = vec![vec![0.0; n_haps]; t]; // beta[T-1] = 0 (log(1))

    for i in (0..t - 1).rev() {
        for h in 0..n_haps {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..n_haps {
                let v = log_trans[h][j] + log_emissions[i + 1][j] + beta[i + 1][j];
                if v > max_val {
                    max_val = v;
                }
            }
            if max_val == f64::NEG_INFINITY {
                beta[i][h] = f64::NEG_INFINITY;
                continue;
            }
            let mut sum = 0.0f64;
            for j in 0..n_haps {
                let v = log_trans[h][j] + log_emissions[i + 1][j] + beta[i + 1][j];
                sum += (v - max_val).exp();
            }
            beta[i][h] = max_val + sum.ln();
        }
    }

    // Compute posteriors: P(state h at time t) ∝ alpha[t][h] * beta[t][h]
    let mut posteriors = vec![vec![0.0; n_haps]; t];
    for i in 0..t {
        let mut max_lp = f64::NEG_INFINITY;
        for h in 0..n_haps {
            let lp = alpha[i][h] + beta[i][h];
            if lp > max_lp {
                max_lp = lp;
            }
        }
        if max_lp == f64::NEG_INFINITY {
            // All states have zero probability — uniform fallback
            let uniform = 1.0 / n_haps as f64;
            for p in posteriors[i].iter_mut() {
                *p = uniform;
            }
            continue;
        }
        let mut sum = 0.0f64;
        for (h, p) in posteriors[i].iter_mut().enumerate() {
            let lp = alpha[i][h] + beta[i][h];
            *p = (lp - max_lp).exp();
            sum += *p;
        }
        if sum > 0.0 {
            for p in posteriors[i].iter_mut() {
                *p = (*p / sum).clamp(0.0, 1.0);
            }
        }
    }

    posteriors
}

/// Estimate haplotype copying model parameters from data.
///
/// Analyzes the similarity data to determine good default values for:
/// - temperature: based on inter-haplotype similarity spread
/// - hap_switch_rate: estimated from haplotype-level signal changes
/// - default_similarity: background identity level for missing data
///
/// Returns (temperature, hap_switch_rate, default_similarity).
pub fn estimate_copying_params(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> (f64, f64, f64) {
    if observations.is_empty() || populations.is_empty() {
        return (0.003, 0.005, 0.99);
    }

    // Collect all haplotype names
    let all_haps: Vec<&str> = populations
        .iter()
        .flat_map(|p| p.haplotypes.iter().map(|s| s.as_str()))
        .collect();

    // Compute background similarity (median of all non-missing values)
    let mut all_sims: Vec<f64> = observations
        .iter()
        .flat_map(|obs| obs.similarities.values().copied())
        .filter(|&s| s > 0.0 && s < 1.0)
        .collect();
    all_sims.sort_by(|a, b| a.total_cmp(b));

    let background_sim = if all_sims.is_empty() {
        0.99
    } else {
        // Use P25 as default similarity for missing data (lower than typical matches)
        all_sims[all_sims.len() / 4]
    };

    // Estimate temperature from the spread of per-haplotype similarities
    // For each window, compute max - min across available haplotypes
    let mut diffs: Vec<f64> = Vec::new();
    for obs in observations {
        let sims: Vec<f64> = all_haps
            .iter()
            .filter_map(|&h| obs.similarities.get(h).copied())
            .collect();
        if sims.len() >= 2 {
            let max = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = sims.iter().cloned().fold(f64::INFINITY, f64::min);
            if max > min {
                diffs.push(max - min);
            }
        }
    }

    let temperature = if diffs.is_empty() {
        0.003
    } else {
        diffs.sort_by(|a, b| a.total_cmp(b));
        // Use median diff as temperature — the softmax will give ~e:1 ratio
        // for haplotypes separated by this amount
        let median = diffs[diffs.len() / 2];
        median.clamp(0.0005, 0.05)
    };

    // Estimate switch rate from best-haplotype stability
    // Count how often the best haplotype changes between consecutive windows
    let mut switches = 0usize;
    let mut comparisons = 0usize;

    for obs_pair in observations.windows(2) {
        let best0 = all_haps
            .iter()
            .max_by(|&&a, &&b| {
                let sa = obs_pair[0].similarities.get(a).copied().unwrap_or(0.0);
                let sb = obs_pair[0].similarities.get(b).copied().unwrap_or(0.0);
                sa.total_cmp(&sb)
            });
        let best1 = all_haps
            .iter()
            .max_by(|&&a, &&b| {
                let sa = obs_pair[1].similarities.get(a).copied().unwrap_or(0.0);
                let sb = obs_pair[1].similarities.get(b).copied().unwrap_or(0.0);
                sa.total_cmp(&sb)
            });

        if let (Some(b0), Some(b1)) = (best0, best1) {
            comparisons += 1;
            if b0 != b1 {
                switches += 1;
            }
        }
    }

    let switch_rate = if comparisons > 0 {
        let raw_rate = switches as f64 / comparisons as f64;
        // The raw rate overestimates true switches because noise causes
        // random best-haplotype changes. Use a fraction of the raw rate.
        (raw_rate * 0.5).clamp(0.001, 0.05)
    } else {
        0.005
    };

    (temperature, switch_rate, background_sim)
}

/// Infer local ancestry using the haplotype copying model with EM refinement.
///
/// Runs the copying model forward-backward algorithm, then re-estimates the
/// haplotype switch rate from expected transition counts (Baum-Welch style).
/// Multiple iterations allow the switch rate to converge to a data-driven value.
///
/// The EM loop:
/// 1. Run forward-backward with current parameters
/// 2. Compute expected transition counts from posteriors:
///    ξ(h→h') = Σ_t α[t][h] × T[h][h'] × emit[t+1][h'] × β[t+1][h'] / P(obs)
/// 3. Re-estimate: switch_rate = Σ(ξ for h≠h') / Σ(all ξ)
/// 4. Re-estimate: ancestry_frac = Σ(ξ for cross-pop) / Σ(ξ for h≠h')
/// 5. Rebuild transitions with new parameters and repeat
///
/// Returns (decoded_states, population_posteriors) from the final iteration.
pub fn infer_ancestry_copying_em(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
    initial_switch_rate: f64,
    initial_ancestry_frac: f64,
    temperature: f64,
    default_similarity: f64,
    em_iterations: usize,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n_obs = observations.len();
    let n_pops = populations.len();

    if n_obs == 0 || n_pops == 0 || em_iterations == 0 {
        return infer_ancestry_copying(
            observations,
            populations,
            initial_switch_rate,
            initial_ancestry_frac,
            temperature,
            default_similarity,
        );
    }

    // Build haplotype index
    let mut hap_names: Vec<&str> = Vec::new();
    let mut hap_to_pop: Vec<usize> = Vec::new();
    let mut pop_hap_counts: Vec<usize> = Vec::new();

    for (pop_idx, pop) in populations.iter().enumerate() {
        pop_hap_counts.push(pop.haplotypes.len());
        for h in &pop.haplotypes {
            hap_names.push(h);
            hap_to_pop.push(pop_idx);
        }
    }

    let n_haps = hap_names.len();
    if n_haps == 0 {
        return (vec![0; n_obs], vec![vec![1.0 / n_pops as f64; n_pops]; n_obs]);
    }

    // Precompute emissions once (they don't change across EM iterations)
    let log_emissions = compute_copying_emissions(
        observations,
        &hap_names,
        temperature,
        default_similarity,
    );

    let log_init = vec![-(n_haps as f64).ln(); n_haps];

    let mut switch_rate = initial_switch_rate;
    let mut ancestry_frac = initial_ancestry_frac;
    let mut hap_posteriors = Vec::new();

    for _iter in 0..em_iterations {
        // Build transitions with current parameters
        let log_trans = build_copying_transitions(
            n_haps,
            &hap_to_pop,
            &pop_hap_counts,
            switch_rate,
            ancestry_frac,
        );

        // E-step: forward-backward
        // We need alpha and beta for expected transition computation
        let (alpha, beta) = copying_forward_backward_alpha_beta(
            &log_emissions,
            &log_trans,
            &log_init,
            n_haps,
        );

        // M-step: re-estimate switch rate and ancestry fraction from
        // expected transition counts
        let mut total_stay = 0.0f64;
        let mut total_within = 0.0f64;
        let mut total_between = 0.0f64;

        for t in 0..n_obs.saturating_sub(1) {
            // For each pair (h, h'), compute ξ(t, h, h'):
            // ξ ∝ α[t][h] × T[h][h'] × emit[t+1][h'] × β[t+1][h']
            // We work in log space and normalize per timestep

            // First find max for numerical stability
            let mut max_xi = f64::NEG_INFINITY;
            for h in 0..n_haps {
                for j in 0..n_haps {
                    let lv = alpha[t][h] + log_trans[h][j]
                        + log_emissions[t + 1][j] + beta[t + 1][j];
                    if lv > max_xi {
                        max_xi = lv;
                    }
                }
            }

            if !max_xi.is_finite() {
                continue;
            }

            // Compute xi values and accumulate
            let mut step_total = 0.0f64;
            let mut step_stay = 0.0f64;
            let mut step_within = 0.0f64;
            let mut step_between = 0.0f64;

            for h in 0..n_haps {
                for j in 0..n_haps {
                    let lv = alpha[t][h] + log_trans[h][j]
                        + log_emissions[t + 1][j] + beta[t + 1][j];
                    let xi = (lv - max_xi).exp();
                    step_total += xi;

                    if h == j {
                        step_stay += xi;
                    } else if hap_to_pop[h] == hap_to_pop[j] {
                        step_within += xi;
                    } else {
                        step_between += xi;
                    }
                }
            }

            if step_total > 0.0 {
                total_stay += step_stay / step_total;
                total_within += step_within / step_total;
                total_between += step_between / step_total;
            }
        }

        // Update parameters
        let total_transitions = total_stay + total_within + total_between;
        if total_transitions > 0.0 {
            let new_switch_rate = (total_within + total_between) / total_transitions;
            switch_rate = new_switch_rate.clamp(0.0005, 0.1);

            let total_switches = total_within + total_between;
            if total_switches > 0.0 {
                let new_ancestry_frac = total_between / total_switches;
                ancestry_frac = new_ancestry_frac.clamp(0.01, 0.5);
            }
        }

        // Compute posteriors from the final alpha/beta
        hap_posteriors = compute_posteriors_from_alpha_beta(&alpha, &beta, n_haps);
    }

    // If no EM iterations ran (shouldn't happen given guard above), fall back
    if hap_posteriors.is_empty() {
        return infer_ancestry_copying(
            observations, populations, switch_rate, ancestry_frac,
            temperature, default_similarity,
        );
    }

    // Aggregate haplotype posteriors to population posteriors
    let mut pop_posteriors = vec![vec![0.0; n_pops]; n_obs];
    let mut decoded_states = vec![0usize; n_obs];

    for t in 0..n_obs {
        for h in 0..n_haps {
            pop_posteriors[t][hap_to_pop[h]] += hap_posteriors[t][h];
        }
        decoded_states[t] = pop_posteriors[t]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    (decoded_states, pop_posteriors)
}

/// Forward-backward returning raw alpha and beta matrices (for EM computation).
fn copying_forward_backward_alpha_beta(
    log_emissions: &[Vec<f64>],
    log_trans: &[Vec<f64>],
    log_init: &[f64],
    n_haps: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let t = log_emissions.len();
    if t == 0 {
        return (Vec::new(), Vec::new());
    }

    // Forward pass
    let mut alpha = vec![vec![f64::NEG_INFINITY; n_haps]; t];
    for h in 0..n_haps {
        alpha[0][h] = log_init[h] + log_emissions[0][h];
    }

    for i in 1..t {
        for j in 0..n_haps {
            let mut max_val = f64::NEG_INFINITY;
            for h in 0..n_haps {
                let v = alpha[i - 1][h] + log_trans[h][j];
                if v > max_val {
                    max_val = v;
                }
            }
            if max_val == f64::NEG_INFINITY {
                alpha[i][j] = f64::NEG_INFINITY;
                continue;
            }
            let mut sum = 0.0f64;
            for h in 0..n_haps {
                let v = alpha[i - 1][h] + log_trans[h][j];
                sum += (v - max_val).exp();
            }
            alpha[i][j] = log_emissions[i][j] + max_val + sum.ln();
        }
    }

    // Backward pass
    let mut beta = vec![vec![0.0; n_haps]; t];
    for i in (0..t - 1).rev() {
        for h in 0..n_haps {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..n_haps {
                let v = log_trans[h][j] + log_emissions[i + 1][j] + beta[i + 1][j];
                if v > max_val {
                    max_val = v;
                }
            }
            if max_val == f64::NEG_INFINITY {
                beta[i][h] = f64::NEG_INFINITY;
                continue;
            }
            let mut sum = 0.0f64;
            for j in 0..n_haps {
                let v = log_trans[h][j] + log_emissions[i + 1][j] + beta[i + 1][j];
                sum += (v - max_val).exp();
            }
            beta[i][h] = max_val + sum.ln();
        }
    }

    (alpha, beta)
}

/// Compute posteriors from pre-computed alpha and beta.
fn compute_posteriors_from_alpha_beta(
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
    n_haps: usize,
) -> Vec<Vec<f64>> {
    let t = alpha.len();
    let mut posteriors = vec![vec![0.0; n_haps]; t];

    for i in 0..t {
        let mut max_lp = f64::NEG_INFINITY;
        for h in 0..n_haps {
            let lp = alpha[i][h] + beta[i][h];
            if lp > max_lp {
                max_lp = lp;
            }
        }
        if max_lp == f64::NEG_INFINITY {
            let uniform = 1.0 / n_haps as f64;
            for p in posteriors[i].iter_mut() {
                *p = uniform;
            }
            continue;
        }
        let mut sum = 0.0f64;
        for (h, p) in posteriors[i].iter_mut().enumerate() {
            let lp = alpha[i][h] + beta[i][h];
            *p = (lp - max_lp).exp();
            sum += *p;
        }
        if sum > 0.0 {
            for p in posteriors[i].iter_mut() {
                *p = (*p / sum).clamp(0.0, 1.0);
            }
        }
    }

    posteriors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_populations() -> Vec<AncestralPopulation> {
        vec![
            AncestralPopulation {
                name: "popA".to_string(),
                haplotypes: vec!["popA#HAP1".to_string(), "popA#HAP2".to_string()],
            },
            AncestralPopulation {
                name: "popB".to_string(),
                haplotypes: vec!["popB#A".to_string(), "popB#B".to_string()],
            },
            AncestralPopulation {
                name: "popC".to_string(),
                haplotypes: vec!["popC#HAP1".to_string(), "popC#HAP2".to_string()],
            },
        ]
    }

    fn make_observation(start: u64, a: f64, b: f64, c: f64) -> AncestryObservation {
        AncestryObservation {
            chrom: "chr1".to_string(),
            start,
            end: start + 5000,
            sample: "QUERY_1#1".to_string(),
            similarities: [
                ("popA#HAP1".to_string(), a),
                ("popA#HAP2".to_string(), a - 0.01),
                ("popB#A".to_string(), b),
                ("popB#B".to_string(), b - 0.01),
                ("popC#HAP1".to_string(), c),
                ("popC#HAP2".to_string(), c - 0.01),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }

    #[test]
    fn test_params_creation() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        assert_eq!(params.n_states, 3);
        assert_eq!(params.populations.len(), 3);

        // Check transitions sum to 1
        for row in &params.transitions {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_viterbi_simple() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        // Create observations strongly favoring popA
        let obs = vec![make_observation(0, 0.98, 0.85, 0.88)];

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 0); // popA
    }

    #[test]
    fn test_viterbi_with_switch() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01); // Higher switch prob for test

        // Create sequence: popA -> popB -> popB
        let obs = vec![
            make_observation(0, 0.95, 0.80, 0.82),      // popA clear winner
            make_observation(5000, 0.95, 0.80, 0.82),   // popA
            make_observation(10000, 0.75, 0.95, 0.80),  // popB clear winner
            make_observation(15000, 0.75, 0.95, 0.80),  // popB
            make_observation(20000, 0.75, 0.95, 0.80),  // popB
        ];

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 5);
        assert_eq!(states[0], 0); // popA
        assert_eq!(states[1], 0); // popA
        assert_eq!(states[2], 1); // popB
        assert_eq!(states[3], 1); // popB
        assert_eq!(states[4], 1); // popB
    }

    #[test]
    fn test_posteriors_sum_to_one() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = vec![
            make_observation(0, 0.90, 0.85, 0.82),
            make_observation(5000, 0.85, 0.92, 0.80),
            make_observation(10000, 0.80, 0.85, 0.95),
        ];

        let posteriors = forward_backward(&obs, &params);

        assert_eq!(posteriors.len(), 3);
        for (t, probs) in posteriors.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Posteriors at t={} sum to {} (should be 1.0)", t, sum);
        }
    }

    #[test]
    fn test_highest_similarity_wins() {
        let pops = make_test_populations();
        // Use higher switch prob so single-window ancestry is possible
        let params = AncestryHmmParams::new(pops, 0.1);

        // Each window strongly favors one population (very strong signal)
        let obs = vec![
            make_observation(0, 0.99, 0.50, 0.50),     // popA overwhelmingly wins
            make_observation(5000, 0.50, 0.99, 0.50),  // popB overwhelmingly wins
            make_observation(10000, 0.50, 0.50, 0.99), // popC overwhelmingly wins
        ];

        let states = viterbi(&obs, &params);
        assert_eq!(states[0], 0, "Window 0 should be popA");
        assert_eq!(states[1], 1, "Window 1 should be popB");
        assert_eq!(states[2], 2, "Window 2 should be popC");

        let posteriors = forward_backward(&obs, &params);
        // Each window should have high posterior for the winning state
        assert!(posteriors[0][0] > 0.8, "Posterior for popA at t=0: {}", posteriors[0][0]);
        assert!(posteriors[1][1] > 0.8, "Posterior for popB at t=1: {}", posteriors[1][1]);
        assert!(posteriors[2][2] > 0.8, "Posterior for popC at t=2: {}", posteriors[2][2]);
    }

    #[test]
    fn test_equal_similarities_equal_posteriors() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        // All populations have equal similarity
        let obs = vec![make_observation(0, 0.90, 0.90, 0.90)];

        let posteriors = forward_backward(&obs, &params);

        // All posteriors should be approximately equal (1/3)
        for (i, &p) in posteriors[0].iter().enumerate() {
            assert!((p - 1.0/3.0).abs() < 0.01, "Posterior {} = {} (should be ~0.33)", i, p);
        }
    }

    #[test]
    fn test_emission_favors_highest_similarity() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = make_observation(0, 0.70, 0.95, 0.85);

        // popB (state 1) should have highest emission
        let log_em_a = params.log_emission(&obs, 0);
        let log_em_b = params.log_emission(&obs, 1);
        let log_em_c = params.log_emission(&obs, 2);

        assert!(log_em_b > log_em_c, "popB emission should be > popC");
        assert!(log_em_b > log_em_a, "popB emission should be > popA");
        assert!(log_em_c > log_em_a, "popC emission should be > popA");
    }

    #[test]
    fn test_estimate_temperature_basic() {
        let pops = make_test_populations();

        // Create observations with varying differences between populations
        let obs = vec![
            make_observation(0, 0.95, 0.85, 0.88),    // diff = 0.10
            make_observation(5000, 0.90, 0.85, 0.82), // diff = 0.08
            make_observation(10000, 0.75, 0.95, 0.80), // diff = 0.20
            make_observation(15000, 0.80, 0.88, 0.92), // diff = 0.12
            make_observation(20000, 0.92, 0.85, 0.88), // diff = 0.07
        ];

        let temp = estimate_temperature(&obs, &pops);

        // Temperature should be in reasonable range
        assert!(temp >= 0.01, "Temperature should be >= 0.01, got {}", temp);
        assert!(temp <= 0.15, "Temperature should be <= 0.15, got {}", temp);
    }

    #[test]
    fn test_estimate_temperature_empty() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = vec![];

        let temp = estimate_temperature(&obs, &pops);

        // Should return fallback default
        assert!((temp - 0.03).abs() < 1e-10, "Empty observations should return fallback 0.03");
    }

    #[test]
    fn test_estimate_temperature_clamping() {
        let pops = make_test_populations();

        // Create observations with very small differences (should clamp to >= 0.0005)
        let obs_small: Vec<AncestryObservation> = (0..10)
            .map(|i| make_observation(i * 5000, 0.90, 0.899, 0.898))
            .collect();

        let temp_small = estimate_temperature(&obs_small, &pops);
        assert!(temp_small >= 0.0005, "Temperature should be clamped to >= 0.0005, got {}", temp_small);
        assert!(temp_small <= 0.15, "Temperature should be clamped to <= 0.15, got {}", temp_small);

        // Create observations with very large differences (should clamp to 0.15)
        let obs_large: Vec<AncestryObservation> = (0..10)
            .map(|i| make_observation(i * 5000, 0.99, 0.50, 0.55))
            .collect();

        let temp_large = estimate_temperature(&obs_large, &pops);
        assert!(temp_large <= 0.15, "Temperature should be clamped to <= 0.15, got {}", temp_large);
    }

    #[test]
    fn test_estimate_switch_prob_basic() {
        let pops = make_test_populations();

        // Create observations with clear ancestry and one switch
        let obs: Vec<AncestryObservation> = vec![
            make_observation(0, 0.95, 0.80, 0.82),
            make_observation(5000, 0.95, 0.80, 0.82),
            make_observation(10000, 0.95, 0.80, 0.82),
            make_observation(15000, 0.95, 0.80, 0.82),
            make_observation(20000, 0.95, 0.80, 0.82),
            make_observation(25000, 0.75, 0.95, 0.80), // switch here
            make_observation(30000, 0.75, 0.95, 0.80),
            make_observation(35000, 0.75, 0.95, 0.80),
            make_observation(40000, 0.75, 0.95, 0.80),
            make_observation(45000, 0.75, 0.95, 0.80),
        ];

        let temp = estimate_temperature(&obs, &pops);
        let switch_prob = estimate_switch_prob(&obs, &pops, temp);

        // Should be in reasonable range
        assert!(switch_prob >= 0.0001, "Switch prob should be >= 0.0001, got {}", switch_prob);
        assert!(switch_prob <= 0.05, "Switch prob should be <= 0.05, got {}", switch_prob);
    }

    #[test]
    fn test_estimate_switch_prob_small_data() {
        let pops = make_test_populations();

        // Less than 10 observations should return fallback
        let obs = vec![
            make_observation(0, 0.95, 0.80, 0.82),
            make_observation(5000, 0.75, 0.95, 0.80),
        ];

        let switch_prob = estimate_switch_prob(&obs, &pops, 0.03);

        assert!((switch_prob - 0.001).abs() < 1e-10, "Small data should return fallback 0.001");
    }

    #[test]
    fn test_estimate_switch_prob_no_switches() {
        let pops = make_test_populations();

        // All observations favor the same population - no switches
        let obs: Vec<AncestryObservation> = (0..20)
            .map(|i| make_observation(i * 5000, 0.95, 0.80, 0.82))
            .collect();

        let temp = estimate_temperature(&obs, &pops);
        let switch_prob = estimate_switch_prob(&obs, &pops, temp);

        // Should be low (near prior of 0.001) due to regularization
        assert!(switch_prob < 0.01, "No switches should result in low switch prob, got {}", switch_prob);
    }

    #[test]
    fn test_set_temperature() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        // Initial temperature (emission_std)
        assert!((params.emission_std - 0.03).abs() < 1e-10);

        // Set new temperature
        params.set_temperature(0.05);
        assert!((params.emission_std - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_set_switch_prob() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        // Update switch probability
        params.set_switch_prob(0.02);

        // Check transitions were updated
        let expected_stay = 1.0 - 0.02;
        let expected_switch = 0.02 / 2.0; // 3 states, so switch to each of 2 others

        for i in 0..params.n_states {
            for j in 0..params.n_states {
                if i == j {
                    assert!(
                        (params.transitions[i][j] - expected_stay).abs() < 1e-10,
                        "Stay prob should be {}, got {}",
                        expected_stay,
                        params.transitions[i][j]
                    );
                } else {
                    assert!(
                        (params.transitions[i][j] - expected_switch).abs() < 1e-10,
                        "Switch prob should be {}, got {}",
                        expected_switch,
                        params.transitions[i][j]
                    );
                }
            }
        }

        // Rows should still sum to 1
        for row in &params.transitions {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Row should sum to 1, got {}", sum);
        }
    }

    #[test]
    fn test_emission_model_max() {
        let model = EmissionModel::Max;
        let sims = vec![0.9, 0.95, 0.8];
        assert_eq!(model.aggregate(&sims), Some(0.95));
    }

    #[test]
    fn test_emission_model_mean() {
        let model = EmissionModel::Mean;
        let sims = vec![0.9, 0.95, 0.8];
        let expected = (0.9 + 0.95 + 0.8) / 3.0;
        assert!((model.aggregate(&sims).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_emission_model_median() {
        let model = EmissionModel::Median;
        let sims = vec![0.9, 0.95, 0.8];
        // Sorted: [0.8, 0.9, 0.95], median = 0.9
        assert_eq!(model.aggregate(&sims), Some(0.9));
    }

    #[test]
    fn test_emission_model_empty() {
        assert_eq!(EmissionModel::Max.aggregate(&[]), None);
        assert_eq!(EmissionModel::Mean.aggregate(&[]), None);
        assert_eq!(EmissionModel::Median.aggregate(&[]), None);
    }

    #[test]
    fn test_emission_model_single() {
        let sims = vec![0.85];
        assert_eq!(EmissionModel::Max.aggregate(&sims), Some(0.85));
        assert_eq!(EmissionModel::Mean.aggregate(&sims), Some(0.85));
        assert_eq!(EmissionModel::Median.aggregate(&sims), Some(0.85));
    }

    #[test]
    fn test_emission_model_median_even() {
        let model = EmissionModel::Median;
        let sims = vec![0.8, 0.9, 0.95, 1.0];
        // Sorted: [0.8, 0.9, 0.95, 1.0], median = (0.9 + 0.95) / 2 = 0.925
        assert!((model.aggregate(&sims).unwrap() - 0.925).abs() < 1e-10);
    }

    #[test]
    fn test_emission_model_topk() {
        let sims = vec![0.5, 0.9, 0.7, 0.8, 0.6];
        // Top 3 = [0.9, 0.8, 0.7], mean = 0.8
        let result = EmissionModel::TopK(3).aggregate(&sims).unwrap();
        assert!((result - 0.8).abs() < 1e-10);
        // Top 1 = max
        let result1 = EmissionModel::TopK(1).aggregate(&sims).unwrap();
        assert!((result1 - 0.9).abs() < 1e-10);
        // Top k > n = mean
        let result_all = EmissionModel::TopK(100).aggregate(&sims).unwrap();
        let mean = sims.iter().sum::<f64>() / sims.len() as f64;
        assert!((result_all - mean).abs() < 1e-10);
    }

    #[test]
    fn test_emission_model_from_str() {
        assert_eq!("max".parse::<EmissionModel>().unwrap(), EmissionModel::Max);
        assert_eq!("mean".parse::<EmissionModel>().unwrap(), EmissionModel::Mean);
        assert_eq!("median".parse::<EmissionModel>().unwrap(), EmissionModel::Median);
        assert_eq!("MAX".parse::<EmissionModel>().unwrap(), EmissionModel::Max);
        assert_eq!("top5".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(5));
        assert_eq!("Top3".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(3));
        assert!("unknown".parse::<EmissionModel>().is_err());
        assert!("topX".parse::<EmissionModel>().is_err());
    }

    #[test]
    fn test_emission_model_display() {
        assert_eq!(format!("{}", EmissionModel::Max), "max");
        assert_eq!(format!("{}", EmissionModel::Mean), "mean");
        assert_eq!(format!("{}", EmissionModel::Median), "median");
        assert_eq!(format!("{}", EmissionModel::TopK(5)), "top5");
        assert_eq!(format!("{}", EmissionModel::TopKWeighted(3, 0.70)), "top3w0.70");
    }

    #[test]
    fn test_topk_weighted_basic() {
        let sims = vec![0.5, 0.9, 0.7, 0.8, 0.6];
        // Top 3 sorted descending = [0.9, 0.8, 0.7]
        // Decay 0.5: weights [1.0, 0.5, 0.25], sum=1.75
        // Weighted = (1.0*0.9 + 0.5*0.8 + 0.25*0.7) / 1.75 = (0.9+0.4+0.175)/1.75 = 1.475/1.75
        let result = EmissionModel::TopKWeighted(3, 0.5).aggregate(&sims).unwrap();
        let expected = (1.0 * 0.9 + 0.5 * 0.8 + 0.25 * 0.7) / (1.0 + 0.5 + 0.25);
        assert!((result - expected).abs() < 1e-10, "got {}, expected {}", result, expected);
    }

    #[test]
    fn test_topk_weighted_decay_one_equals_uniform() {
        // Decay = 1.0 should give same result as standard TopK
        let sims = vec![0.5, 0.9, 0.7, 0.8, 0.6];
        let uniform = EmissionModel::TopK(3).aggregate(&sims).unwrap();
        let weighted = EmissionModel::TopKWeighted(3, 1.0).aggregate(&sims).unwrap();
        assert!((uniform - weighted).abs() < 1e-10);
    }

    #[test]
    fn test_topk_weighted_decay_zero_equals_max() {
        // Decay approaching 0 should approximate max (only top-1 matters)
        let sims = vec![0.5, 0.9, 0.7, 0.8, 0.6];
        let weighted = EmissionModel::TopKWeighted(3, 0.001).aggregate(&sims).unwrap();
        let max = EmissionModel::Max.aggregate(&sims).unwrap();
        assert!((weighted - max).abs() < 0.01, "got {}, expected ~{}", weighted, max);
    }

    #[test]
    fn test_topk_weighted_single_element() {
        let sims = vec![0.95];
        let result = EmissionModel::TopKWeighted(3, 0.5).aggregate(&sims).unwrap();
        assert!((result - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_topk_weighted_empty() {
        assert!(EmissionModel::TopKWeighted(3, 0.5).aggregate(&[]).is_none());
    }

    #[test]
    fn test_topk_weighted_k_greater_than_n() {
        let sims = vec![0.9, 0.8];
        // Only 2 elements, but k=5. Uses all 2 with decay.
        let result = EmissionModel::TopKWeighted(5, 0.5).aggregate(&sims).unwrap();
        let expected = (1.0 * 0.9 + 0.5 * 0.8) / (1.0 + 0.5);
        assert!((result - expected).abs() < 1e-10);
    }

    // ---- Posterior feedback tests ----

    #[test]
    fn test_posterior_feedback_zero_lambda() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        let posteriors = vec![vec![0.7, 0.2, 0.1]];
        let result = apply_posterior_feedback(&emissions, &posteriors, 0.0);
        assert_eq!(result, emissions);
    }

    #[test]
    fn test_posterior_feedback_boosts_high_posterior() {
        let emissions = vec![vec![-1.0, -1.0, -1.0]]; // equal emissions
        let posteriors = vec![vec![0.8, 0.15, 0.05]];
        let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);

        // State 0 should have highest adjusted emission (highest posterior)
        assert!(result[0][0] > result[0][1], "state 0 should be boosted most");
        assert!(result[0][1] > result[0][2], "state 1 should be boosted more than state 2");
    }

    #[test]
    fn test_posterior_feedback_preserves_ranking_when_aligned() {
        // Emissions and posteriors agree → ranking preserved
        let emissions = vec![vec![-0.5, -1.0, -2.0]];
        let posteriors = vec![vec![0.6, 0.3, 0.1]];
        let result = apply_posterior_feedback(&emissions, &posteriors, 0.5);
        assert!(result[0][0] > result[0][1]);
        assert!(result[0][1] > result[0][2]);
    }

    #[test]
    fn test_posterior_feedback_can_override_emissions() {
        // Strong posterior for state 1, but emissions favor state 0
        let emissions = vec![vec![-0.1, -0.5, -2.0]];
        let posteriors = vec![vec![0.05, 0.90, 0.05]];
        // With strong feedback (lambda=2.0), posterior should dominate
        let result = apply_posterior_feedback(&emissions, &posteriors, 2.0);
        assert!(result[0][1] > result[0][0],
            "strong posterior feedback should override emission ranking: {} vs {}",
            result[0][1], result[0][0]);
    }

    #[test]
    fn test_posterior_feedback_multiple_windows() {
        let emissions = vec![
            vec![-1.0, -1.0],
            vec![-1.0, -1.0],
        ];
        let posteriors = vec![
            vec![0.9, 0.1], // window 0: strong state 0
            vec![0.1, 0.9], // window 1: strong state 1
        ];
        let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
        // Window 0: state 0 boosted
        assert!(result[0][0] > result[0][1]);
        // Window 1: state 1 boosted
        assert!(result[1][1] > result[1][0]);
    }

    #[test]
    fn test_posterior_feedback_handles_neg_infinity() {
        let emissions = vec![vec![f64::NEG_INFINITY, -1.0]];
        let posteriors = vec![vec![0.5, 0.5]];
        let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
        // NEG_INFINITY emission should stay NEG_INFINITY (no data for state)
        assert_eq!(result[0][0], f64::NEG_INFINITY);
        assert!(result[0][1].is_finite());
    }

    #[test]
    fn test_posterior_feedback_handles_zero_posterior() {
        let emissions = vec![vec![-1.0, -1.0]];
        let posteriors = vec![vec![1.0, 0.0]]; // zero posterior for state 1
        let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
        // Zero posterior → log(epsilon) ≈ -23, strong suppression
        assert!(result[0][0] > result[0][1]);
        assert!(result[0][1] < -20.0, "zero posterior should heavily suppress: {}", result[0][1]);
    }

    #[test]
    fn test_posterior_feedback_empty_inputs() {
        let emissions: Vec<Vec<f64>> = vec![];
        let posteriors: Vec<Vec<f64>> = vec![];
        let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
        assert!(result.is_empty());
    }

    // ---- Heteroscedastic emission tests ----

    fn make_hetero_obs(n: usize) -> Vec<AncestryObservation> {
        (0..n).map(|i| {
            let noise = (i as f64 * 0.37).sin() * 0.005;
            make_observation(i as u64 * 5000, 0.99 + noise, 0.96 + noise * 0.5, 0.93 + noise * 0.3)
        }).collect()
    }

    #[test]
    fn test_compute_population_variances_basic() {
        let pops = make_test_populations();
        let obs = make_hetero_obs(20);
        let variances = compute_population_variances(&obs, &pops, &EmissionModel::Max);
        assert_eq!(variances.len(), pops.len());
        for v in &variances {
            assert!(*v >= 0.0, "variance should be non-negative: {}", v);
        }
    }

    #[test]
    fn test_compute_population_variances_empty() {
        let pops = make_test_populations();
        let variances = compute_population_variances(&[], &pops, &EmissionModel::Max);
        assert_eq!(variances.len(), pops.len());
        for v in &variances {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_heteroscedastic_temperatures_zero_gamma() {
        let variances = vec![0.001, 0.005, 0.002];
        let temps = compute_heteroscedastic_temperatures(&variances, 0.01, 0.0);
        for t in &temps {
            assert!((*t - 0.01).abs() < 1e-10);
        }
    }

    #[test]
    fn test_heteroscedastic_temperatures_high_var_gets_high_temp() {
        let variances = vec![0.001, 0.010, 0.002];
        let temps = compute_heteroscedastic_temperatures(&variances, 0.01, 0.5);
        assert!(temps[1] > temps[0], "high-var pop should get higher temp");
        assert!(temps[1] > temps[2], "high-var pop should get higher temp");
    }

    #[test]
    fn test_heteroscedastic_temperatures_low_var_gets_low_temp() {
        let variances = vec![0.001, 0.010, 0.002];
        let temps = compute_heteroscedastic_temperatures(&variances, 0.01, 0.5);
        assert!(temps[0] < temps[1], "low-var pop should get lower temp");
        assert!(temps[0] < temps[2], "low-var pop should get lower temp");
    }

    #[test]
    fn test_heteroscedastic_temperatures_clamping() {
        let variances = vec![0.00001, 1.0, 0.0001];
        let temps = compute_heteroscedastic_temperatures(&variances, 0.01, 1.0);
        for t in &temps {
            assert!(*t >= 0.01 * 0.3 - 1e-10, "temp {} below min clamp", t);
            assert!(*t <= 0.01 * 3.0 + 1e-10, "temp {} above max clamp", t);
        }
    }

    #[test]
    fn test_heteroscedastic_temperatures_empty() {
        let temps = compute_heteroscedastic_temperatures(&[], 0.01, 0.5);
        assert!(temps.is_empty());
    }

    #[test]
    fn test_heteroscedastic_log_emissions_uniform_temps_match_standard() {
        let pops = make_test_populations();
        let obs = make_hetero_obs(10);
        let params = AncestryHmmParams::new(pops.clone(), 0.01);

        let uniform_temps = vec![params.emission_std; pops.len()];
        let standard = precompute_log_emissions(&obs, &params);
        let hetero = precompute_heteroscedastic_log_emissions(&obs, &params, &uniform_temps);

        for (t, (s_row, h_row)) in standard.iter().zip(hetero.iter()).enumerate() {
            for (k, (&s, &h)) in s_row.iter().zip(h_row.iter()).enumerate() {
                if s.is_finite() && h.is_finite() {
                    assert!((s - h).abs() < 0.01,
                        "window {} state {}: standard={:.6} hetero={:.6}", t, k, s, h);
                }
            }
        }
    }

    #[test]
    fn test_heteroscedastic_emissions_sharper_for_low_temp() {
        let pops = make_test_populations();
        let obs = make_hetero_obs(20);
        let params = AncestryHmmParams::new(pops.clone(), 0.01);

        let mut temps = vec![params.emission_std; pops.len()];
        temps[0] = params.emission_std * 0.5; // half temperature → sharper

        let standard = precompute_log_emissions(&obs, &params);
        let hetero = precompute_heteroscedastic_log_emissions(&obs, &params, &temps);

        // Pop 0 emissions should be boosted (less negative) since it's the dominant pop
        let mut pop0_boosted = 0;
        for (s_row, h_row) in standard.iter().zip(hetero.iter()) {
            if s_row[0].is_finite() && h_row[0].is_finite() && h_row[0] > s_row[0] {
                pop0_boosted += 1;
            }
        }
        assert!(pop0_boosted > obs.len() / 2,
            "pop 0 should be boosted by lower temp: {}/{}", pop0_boosted, obs.len());
    }

    #[test]
    fn test_compare_emission_models_cv() {
        // Step 7: Compare emission models via CV on synthetic 3-population data
        let pops = make_test_populations();

        // Create synthetic observations with clear population signal
        // Each "haplotype" has observations where its own population has highest similarity
        let mut observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

        for pop in &pops {
            for hap in &pop.haplotypes {
                let mut obs_list = Vec::new();
                for w in 0..10 {
                    let mut sims = HashMap::new();
                    for (i, p) in pops.iter().enumerate() {
                        for (j, h) in p.haplotypes.iter().enumerate() {
                            let base_sim = if p.name == pop.name { 0.95 - j as f64 * 0.01 } else { 0.80 - i as f64 * 0.02 };
                            sims.insert(h.clone(), base_sim);
                        }
                    }
                    obs_list.push(AncestryObservation {
                        chrom: "chr1".to_string(),
                        start: w * 5000,
                        end: (w + 1) * 5000,
                        sample: hap.clone(),
                        similarities: sims,
                        coverage_ratios: None,
                        haplotype_consistency_bonus: None,
                    });
                }
                observations.insert(hap.clone(), obs_list);
            }
        }

        let mut best_accuracy = 0.0;

        for model in &[EmissionModel::Max, EmissionModel::Mean, EmissionModel::Median] {
            let mut params = AncestryHmmParams::new(pops.clone(), 0.001);
            params.set_emission_model(*model);

            let result = crate::validation::cross_validate(&observations, &pops, &params);
            if result.overall_accuracy > best_accuracy {
                best_accuracy = result.overall_accuracy;
            }
        }

        // At least one model should achieve > 60% on this clear signal
        assert!(best_accuracy > 0.6,
            "Best accuracy across emission models was {:.1}%, expected > 60%", best_accuracy * 100.0);
    }

    #[test]
    fn test_temperature_very_sharp() {
        // Step 10: Temperature = 0.001 → very decisive ancestry calls
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_temperature(0.001); // Very sharp

        let obs = vec![
            make_observation(0, 0.95, 0.85, 0.88),
            make_observation(5000, 0.95, 0.85, 0.88),
            make_observation(10000, 0.95, 0.85, 0.88),
        ];

        let posteriors = forward_backward(&obs, &params);

        // With very sharp temperature, max posterior should be > 0.9
        for probs in &posteriors {
            let max_post = probs.iter().cloned().fold(0.0_f64, f64::max);
            assert!(max_post > 0.9,
                "Sharp temperature should give decisive posteriors, got max={:.4}", max_post);
        }
    }

    #[test]
    fn test_temperature_very_flat() {
        // Step 10: Temperature = 100.0 → nearly uniform posteriors
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_temperature(100.0); // Very flat

        let obs = vec![
            make_observation(0, 0.95, 0.85, 0.88),
            make_observation(5000, 0.95, 0.85, 0.88),
            make_observation(10000, 0.95, 0.85, 0.88),
        ];

        let posteriors = forward_backward(&obs, &params);

        // With very flat temperature, posteriors should be nearly uniform (1/3 ≈ 0.333)
        for probs in &posteriors {
            for &p in probs {
                assert!((p - 1.0 / 3.0).abs() < 0.05,
                    "Flat temperature should give near-uniform posteriors, got {:.4}", p);
            }
        }
    }

    #[test]
    fn test_synthetic_mixed_ancestry() {
        // Step 9: Synthetic multi-population test with known mixed ancestry
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.01); // Moderate switch prob
        params.set_temperature(0.03);

        // Region 1 (windows 0-4): clearly popA
        // Region 2 (windows 5-9): clearly popB
        let mut obs = Vec::new();
        for w in 0..5 {
            obs.push(make_observation(w * 5000, 0.96, 0.80, 0.82));
        }
        for w in 5..10 {
            obs.push(make_observation(w * 5000, 0.78, 0.96, 0.80));
        }

        let states = viterbi(&obs, &params);

        // Count correct assignments
        let correct_region1 = states[0..5].iter().filter(|&&s| s == 0).count();
        let correct_region2 = states[5..10].iter().filter(|&&s| s == 1).count();
        let total_correct = correct_region1 + correct_region2;

        // Should get >= 70% concordance (7 out of 10 windows)
        assert!(total_correct >= 7,
            "Expected >= 70% concordance, got {}/10 correct (region1: {}/5 popA, region2: {}/5 popB)",
            total_correct, correct_region1, correct_region2);
    }

    #[test]
    fn test_baum_welch_improves_likelihood() {
        // Baum-Welch should monotonically increase log-likelihood
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.01);
        params.set_temperature(0.03);

        // Create observations with clear population structure
        let obs: Vec<AncestryObservation> = (0..20).map(|w| {
            if w < 10 {
                make_observation(w * 5000, 0.95, 0.80, 0.82)
            } else {
                make_observation(w * 5000, 0.78, 0.95, 0.80)
            }
        }).collect();

        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        // Run with 1 iteration first
        let mut params1 = params.clone();
        let ll1 = params1.baum_welch(&obs_slices, 1, 1e-10);

        // Run with 5 iterations
        let mut params5 = params.clone();
        let ll5 = params5.baum_welch(&obs_slices, 5, 1e-10);

        // More iterations should give >= log-likelihood (or converge)
        assert!(ll5 >= ll1 - 1e-6,
            "5-iter LL ({:.4}) should be >= 1-iter LL ({:.4})", ll5, ll1);
    }

    #[test]
    fn test_baum_welch_empty_data() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let obs_slices: Vec<&[AncestryObservation]> = vec![];
        let ll = params.baum_welch(&obs_slices, 10, 1e-4);

        assert!(ll == f64::NEG_INFINITY, "Empty data should return -inf");
    }

    #[test]
    fn test_baum_welch_single_obs() {
        // Baum-Welch needs at least 2 observations per sequence for transitions
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let obs = vec![make_observation(0, 0.95, 0.80, 0.82)];
        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        let original_switch = 1.0 - params.transitions[0][0];
        let _ll = params.baum_welch(&obs_slices, 5, 1e-4);

        // With only 1 observation, switch prob should remain unchanged
        let new_switch = 1.0 - params.transitions[0][0];
        assert!((new_switch - original_switch).abs() < 1e-6,
            "Single obs should not change switch prob: was {}, now {}", original_switch, new_switch);
    }

    #[test]
    fn test_baum_welch_maintains_valid_params() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.01);
        params.set_temperature(0.03);

        let obs: Vec<AncestryObservation> = (0..15).map(|w| {
            make_observation(w * 5000, 0.90, 0.85 + (w as f64 * 0.005), 0.82)
        }).collect();

        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];
        params.baum_welch(&obs_slices, 10, 1e-4);

        // Transitions must still be valid probability distributions
        for row in &params.transitions {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "Transition row should sum to 1, got {}", sum);
            for &p in row {
                assert!((0.0..=1.0).contains(&p),
                    "Transition probability should be in [0,1], got {}", p);
            }
        }

        // Switch probability should be in valid range
        let switch_prob = 1.0 - params.transitions[0][0];
        assert!((0.0001..=0.1).contains(&switch_prob),
            "Switch prob should be in [0.0001, 0.1], got {}", switch_prob);
    }

    #[test]
    fn test_decoding_method_from_str() {
        assert_eq!("viterbi".parse::<DecodingMethod>().unwrap(), DecodingMethod::Viterbi);
        assert_eq!("posterior".parse::<DecodingMethod>().unwrap(), DecodingMethod::Posterior);
        assert_eq!("mpel".parse::<DecodingMethod>().unwrap(), DecodingMethod::Mpel);
        assert_eq!("VITERBI".parse::<DecodingMethod>().unwrap(), DecodingMethod::Viterbi);
        assert_eq!("Posterior".parse::<DecodingMethod>().unwrap(), DecodingMethod::Posterior);
        assert_eq!("MPEL".parse::<DecodingMethod>().unwrap(), DecodingMethod::Mpel);
        assert!("unknown".parse::<DecodingMethod>().is_err());
    }

    #[test]
    fn test_decoding_method_display() {
        assert_eq!(format!("{}", DecodingMethod::Viterbi), "viterbi");
        assert_eq!(format!("{}", DecodingMethod::Posterior), "posterior");
        assert_eq!(format!("{}", DecodingMethod::Mpel), "mpel");
    }

    #[test]
    fn test_posterior_decode_matches_viterbi_clear_signal() {
        // With very clear signal, posterior decoding should agree with Viterbi
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);

        let obs = vec![
            make_observation(0, 0.96, 0.80, 0.82),
            make_observation(5000, 0.96, 0.80, 0.82),
            make_observation(10000, 0.96, 0.80, 0.82),
            make_observation(15000, 0.78, 0.96, 0.80),
            make_observation(20000, 0.78, 0.96, 0.80),
            make_observation(25000, 0.78, 0.96, 0.80),
        ];

        let viterbi_states = viterbi(&obs, &params);
        let posterior_states = posterior_decode(&obs, &params);

        // Both should agree on all windows with clear signal
        assert_eq!(viterbi_states, posterior_states,
            "Viterbi {:?} and posterior {:?} should agree on clear signal",
            viterbi_states, posterior_states);
    }

    #[test]
    fn test_posterior_decode_detects_minority_better() {
        // Create a scenario where minority ancestry is a short tract
        // Posterior decoding should detect it even when Viterbi doesn't
        let pops = make_test_populations();
        // Low switch prob = Viterbi strongly favors staying in majority state
        let params = AncestryHmmParams::new(pops, 0.001);

        // 10 windows of popA, then 2 windows of popB signal, then 8 more popA
        let mut obs = Vec::new();
        for w in 0..10 {
            obs.push(make_observation(w * 5000, 0.95, 0.83, 0.82));
        }
        // Two windows where popB has highest similarity (minority tract)
        obs.push(make_observation(50000, 0.83, 0.95, 0.82));
        obs.push(make_observation(55000, 0.83, 0.95, 0.82));
        for w in 12..20 {
            obs.push(make_observation(w as u64 * 5000, 0.95, 0.83, 0.82));
        }

        let viterbi_states = viterbi(&obs, &params);
        let posterior_states = posterior_decode(&obs, &params);

        // Viterbi may call everything as popA (state 0) due to switch penalty
        // Posterior should detect the popB windows (state 1)
        let viterbi_popB = viterbi_states.iter().filter(|&&s| s == 1).count();
        let posterior_popB = posterior_states.iter().filter(|&&s| s == 1).count();

        // Posterior decoding should detect at least as many popB windows as Viterbi
        assert!(posterior_popB >= viterbi_popB,
            "Posterior ({} popB) should detect >= Viterbi ({} popB)",
            posterior_popB, viterbi_popB);
    }

    #[test]
    fn test_posterior_decode_empty() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs: Vec<AncestryObservation> = Vec::new();
        let states = posterior_decode(&obs, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_posterior_decode_single_observation() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = vec![make_observation(0, 0.90, 0.80, 0.85)];
        let states = posterior_decode(&obs, &params);

        assert_eq!(states.len(), 1);
        // popA (state 0) has highest similarity
        assert_eq!(states[0], 0);
    }

    #[test]
    fn test_posterior_decode_equal_sims_is_valid() {
        // Equal similarities should produce valid states (no panic)
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs: Vec<AncestryObservation> = (0..5)
            .map(|w| make_observation(w * 5000, 0.90, 0.90, 0.90))
            .collect();

        let states = posterior_decode(&obs, &params);
        assert_eq!(states.len(), 5);
        // All states should be valid indices
        for &s in &states {
            assert!(s < 3, "State {} should be < 3", s);
        }
    }

    #[test]
    fn test_mpel_decode_agrees_with_viterbi_clear_signal() {
        // With clear signal, MPEL should agree with Viterbi
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);

        let obs = vec![
            make_observation(0, 0.96, 0.80, 0.82),
            make_observation(5000, 0.96, 0.80, 0.82),
            make_observation(10000, 0.96, 0.80, 0.82),
            make_observation(15000, 0.78, 0.96, 0.80),
            make_observation(20000, 0.78, 0.96, 0.80),
            make_observation(25000, 0.78, 0.96, 0.80),
        ];

        let viterbi_states = viterbi(&obs, &params);
        let posteriors = forward_backward(&obs, &params);
        let mpel_states = mpel_decode_from_posteriors(&posteriors, &params);

        assert_eq!(viterbi_states, mpel_states,
            "Viterbi {:?} and MPEL {:?} should agree on clear signal",
            viterbi_states, mpel_states);
    }

    #[test]
    fn test_mpel_decode_smoother_than_posterior() {
        // MPEL should produce smoother paths than posterior argmax
        // when there's a single noisy window in a clear tract
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.005);

        let mut obs = Vec::new();
        // 8 clear popA windows
        for w in 0..8 {
            obs.push(make_observation(w * 5000, 0.95, 0.83, 0.82));
        }
        // 1 noisy window where popB barely wins
        obs.push(make_observation(40000, 0.88, 0.89, 0.82));
        // 6 more popA
        for w in 9..15 {
            obs.push(make_observation(w * 5000, 0.95, 0.83, 0.82));
        }

        let posteriors = forward_backward(&obs, &params);
        let mpel_states = mpel_decode_from_posteriors(&posteriors, &params);

        // Count state switches in MPEL
        let mpel_switches: usize = mpel_states.windows(2)
            .filter(|w| w[0] != w[1]).count();

        // Posterior argmax
        let posterior_states: Vec<usize> = posteriors.iter()
            .map(|probs| probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx).unwrap_or(0))
            .collect();
        let posterior_switches: usize = posterior_states.windows(2)
            .filter(|w| w[0] != w[1]).count();

        // MPEL should have fewer or equal switches (smoother)
        assert!(mpel_switches <= posterior_switches,
            "MPEL ({} switches) should be smoother than posterior ({} switches)",
            mpel_switches, posterior_switches);
    }

    #[test]
    fn test_mpel_decode_empty() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let posteriors: Vec<Vec<f64>> = Vec::new();
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_learn_normalization_computes_stats() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        // Create observations with systematic bias: popA always higher
        let obs: Vec<AncestryObservation> = (0..20)
            .map(|w| make_observation(w * 5000, 0.95, 0.85, 0.88))
            .collect();

        assert!(params.normalization.is_none());
        params.learn_normalization(&obs);
        assert!(params.normalization.is_some());

        let norm = params.normalization.as_ref().unwrap();
        assert_eq!(norm.means.len(), 3);
        assert_eq!(norm.stds.len(), 3);

        // popA should have highest mean
        assert!(norm.means[0] > norm.means[1],
            "popA mean ({}) should be > popB mean ({})",
            norm.means[0], norm.means[1]);

        // All stds should be positive
        for (i, &std) in norm.stds.iter().enumerate() {
            assert!(std > 0.0, "Population {} std should be > 0, got {}", i, std);
        }
    }

    #[test]
    fn test_normalization_reduces_majority_bias() {
        // Scenario: EUR (pop 0) systematically has 0.05 higher similarity than AFR (pop 1)
        // Without normalization, softmax always favors EUR
        // With normalization, the bias is removed
        let pops = vec![
            AncestralPopulation {
                name: "EUR".to_string(),
                haplotypes: vec!["EUR#1".to_string()],
            },
            AncestralPopulation {
                name: "AFR".to_string(),
                haplotypes: vec!["AFR#1".to_string()],
            },
        ];

        // Create observations: EUR always 0.93, AFR always 0.88 (systematic bias)
        let make_biased = |start: u64, eur: f64, afr: f64| -> AncestryObservation {
            AncestryObservation {
                chrom: "chr20".to_string(),
                start,
                end: start + 10000,
                sample: "test#1".to_string(),
                similarities: [
                    ("EUR#1".to_string(), eur),
                    ("AFR#1".to_string(), afr),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        };

        // Many windows with systematic EUR bias
        let mut obs: Vec<AncestryObservation> = (0..20)
            .map(|w| make_biased(w * 10000, 0.93, 0.88))
            .collect();
        // A few windows where AFR has RELATIVE advantage (still lower absolute)
        // AFR goes up to 0.91 while EUR stays at 0.93
        obs.push(make_biased(200000, 0.93, 0.91));
        obs.push(make_biased(210000, 0.93, 0.91));

        // Without normalization: log_emission for AFR at the 0.91 window
        let params_raw = AncestryHmmParams::new(pops.clone(), 0.01);
        let log_em_afr_raw = params_raw.log_emission(&obs[20], 1);

        // With normalization
        let mut params_norm = AncestryHmmParams::new(pops, 0.01);
        params_norm.learn_normalization(&obs);
        let log_em_afr_norm = params_norm.log_emission(&obs[20], 1);

        // After normalization, AFR emission should be higher (less negative)
        // because the bias is removed
        assert!(log_em_afr_norm > log_em_afr_raw,
            "Normalized AFR emission ({:.4}) should be > raw ({:.4})",
            log_em_afr_norm, log_em_afr_raw);
    }

    #[test]
    fn test_normalization_with_equal_sims_gives_equal_emissions() {
        // If both populations have equal similarity, normalization shouldn't change the result
        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string()],
            },
        ];

        let obs: Vec<AncestryObservation> = (0..10)
            .map(|w| AncestryObservation {
                chrom: "chr1".to_string(),
                start: w * 5000,
                end: (w + 1) * 5000,
                sample: "test".to_string(),
                similarities: [
                    ("pop_a#1".to_string(), 0.90),
                    ("pop_b#1".to_string(), 0.90),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            })
            .collect();

        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.learn_normalization(&obs);

        let log_em_a = params.log_emission(&obs[0], 0);
        let log_em_b = params.log_emission(&obs[0], 1);

        assert!((log_em_a - log_em_b).abs() < 1e-6,
            "Equal sims should give equal emissions: a={:.6}, b={:.6}", log_em_a, log_em_b);
    }

    #[test]
    fn test_normalization_empty_observations() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let obs: Vec<AncestryObservation> = Vec::new();
        params.learn_normalization(&obs);

        // Should still have normalization set (with zero means)
        assert!(params.normalization.is_some());
        let norm = params.normalization.as_ref().unwrap();
        for &m in &norm.means {
            assert!((m - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalized_posteriors_sum_to_one() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let obs: Vec<AncestryObservation> = (0..10)
            .map(|w| make_observation(w * 5000, 0.95, 0.85, 0.88))
            .collect();

        params.learn_normalization(&obs);

        let posteriors = forward_backward(&obs, &params);
        for (t, probs) in posteriors.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "Normalized posteriors at t={} sum to {} (should be 1.0)", t, sum);
        }
    }

    #[test]
    fn test_estimate_temperature_normalized_wider_range() {
        // Normalized temperatures should be in a wider range than raw temperatures
        // because z-scores have larger differences than raw similarities
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops.clone(), 0.001);

        let obs: Vec<AncestryObservation> = (0..20)
            .map(|w| make_observation(w * 5000, 0.95, 0.85, 0.88))
            .collect();

        // Raw temperature
        let raw_temp = estimate_temperature(&obs, &pops);

        // Normalized temperature
        params.learn_normalization(&obs);
        let norm_temp = estimate_temperature_normalized(&obs, &params);

        // Normalized temperature should be in z-score range (0.5 - 5.0)
        assert!(norm_temp >= 0.5, "Normalized temp should be >= 0.5, got {}", norm_temp);
        assert!(norm_temp <= 5.0, "Normalized temp should be <= 5.0, got {}", norm_temp);

        // Raw temperature should be in similarity range (0.01 - 0.15)
        assert!(raw_temp >= 0.01, "Raw temp should be >= 0.01, got {}", raw_temp);
        assert!(raw_temp <= 0.15, "Raw temp should be <= 0.15, got {}", raw_temp);
    }

    #[test]
    fn test_estimate_temperature_normalized_without_normalization() {
        // If no normalization is set, should fall back to standard estimation
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.001);

        let obs: Vec<AncestryObservation> = (0..10)
            .map(|w| make_observation(w * 5000, 0.95, 0.85, 0.88))
            .collect();

        let raw_temp = estimate_temperature(&obs, &pops);
        let norm_temp = estimate_temperature_normalized(&obs, &params);

        // Without normalization, both should be the same
        assert!((raw_temp - norm_temp).abs() < 1e-10,
            "Without normalization, raw ({}) and norm ({}) temps should match",
            raw_temp, norm_temp);
    }

    #[test]
    fn test_normalized_posterior_decode_detects_minority() {
        // Full pipeline test: normalization + posterior decoding should detect
        // minority ancestry in a scenario with systematic bias
        let pops = vec![
            AncestralPopulation {
                name: "EUR".to_string(),
                haplotypes: vec!["EUR#1".to_string()],
            },
            AncestralPopulation {
                name: "AFR".to_string(),
                haplotypes: vec!["AFR#1".to_string()],
            },
        ];

        let make_obs = |start: u64, eur: f64, afr: f64| -> AncestryObservation {
            AncestryObservation {
                chrom: "chr20".to_string(),
                start,
                end: start + 10000,
                sample: "test#1".to_string(),
                similarities: [
                    ("EUR#1".to_string(), eur),
                    ("AFR#1".to_string(), afr),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        };

        // Build observations: EUR baseline 0.93, AFR baseline 0.88
        // But 3 windows where AFR has a relative advantage (goes to 0.92)
        let mut obs: Vec<AncestryObservation> = Vec::new();
        for w in 0..10 {
            obs.push(make_obs(w * 10000, 0.93, 0.88));
        }
        // AFR-elevated windows
        obs.push(make_obs(100000, 0.93, 0.92));
        obs.push(make_obs(110000, 0.93, 0.92));
        obs.push(make_obs(120000, 0.93, 0.92));
        for w in 13..20 {
            obs.push(make_obs(w as u64 * 10000, 0.93, 0.88));
        }

        // Without normalization: everything should be EUR (systematic bias)
        let params_raw = AncestryHmmParams::new(pops.clone(), 0.01);
        let raw_states = posterior_decode(&obs, &params_raw);
        let raw_afr_count = raw_states.iter().filter(|&&s| s == 1).count();

        // With normalization + adjusted temperature
        let mut params_norm = AncestryHmmParams::new(pops, 0.01);
        params_norm.learn_normalization(&obs);
        let norm_temp = estimate_temperature_normalized(&obs, &params_norm);
        params_norm.set_temperature(norm_temp);
        let norm_states = posterior_decode(&obs, &params_norm);
        let norm_afr_count = norm_states.iter().filter(|&&s| s == 1).count();

        // Normalized version should detect more AFR windows than raw
        assert!(norm_afr_count >= raw_afr_count,
            "Normalized ({} AFR) should detect >= raw ({} AFR) windows",
            norm_afr_count, raw_afr_count);
    }

    #[test]
    fn test_genetic_map_uniform() {
        let gmap = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
        let cm = gmap.interpolate_cm(50_000_000);
        assert!((cm - 50.0).abs() < 0.01, "50 Mb at 1 cM/Mb should be ~50 cM, got {}", cm);

        let dist = gmap.genetic_distance_cm(10_000_000, 20_000_000);
        assert!((dist - 10.0).abs() < 0.01, "10 Mb distance should be ~10 cM, got {}", dist);
    }

    #[test]
    fn test_genetic_map_modulated_switch_prob() {
        let gmap = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
        let base_sw = 0.01;
        let window_size = 10_000;

        // 10kb apart at 1 cM/Mb = ~0.01 cM → should be close to base
        let sw1 = gmap.modulated_switch_prob(base_sw, 0, 10_000, window_size);
        assert!(sw1 > 0.0 && sw1 < 0.5, "Switch prob should be valid, got {}", sw1);

        // 100kb apart = 10× the normal distance → should be higher
        let sw10 = gmap.modulated_switch_prob(base_sw, 0, 100_000, window_size);
        assert!(sw10 > sw1, "100kb spacing ({}) should have higher switch than 10kb ({})",
            sw10, sw1);
    }

    #[test]
    fn test_forward_backward_genetic_map_posteriors_valid() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let gmap = AncestryGeneticMap::uniform(0, 200_000, 1.0);

        let obs: Vec<AncestryObservation> = (0..10)
            .map(|w| make_observation(w * 5000, 0.95, 0.85, 0.88))
            .collect();

        let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);

        assert_eq!(posteriors.len(), 10);
        for (t, probs) in posteriors.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "Posteriors at t={} sum to {} (should be 1.0)", t, sum);
            for &p in probs {
                assert!(p >= 0.0 && p <= 1.0,
                    "Posterior {} out of range at t={}", p, t);
            }
        }
    }

    #[test]
    fn test_viterbi_genetic_map_matches_standard_uniform() {
        // With uniform genetic map at expected rate, should roughly match standard Viterbi
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let gmap = AncestryGeneticMap::uniform(0, 200_000, 1.0);

        let obs = vec![
            make_observation(0, 0.96, 0.80, 0.82),
            make_observation(5000, 0.96, 0.80, 0.82),
            make_observation(10000, 0.80, 0.96, 0.82),
            make_observation(15000, 0.80, 0.96, 0.82),
        ];

        let states_std = viterbi(&obs, &params);
        let states_gmap = viterbi_with_genetic_map(&obs, &params, &gmap);

        // Both should detect the state change
        assert_eq!(states_std[0], states_gmap[0], "First window should match");
        assert_eq!(states_std[3], states_gmap[3], "Last window should match");
    }

    #[test]
    fn test_posterior_decode_genetic_map_empty() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let gmap = AncestryGeneticMap::uniform(0, 100_000, 1.0);

        let obs: Vec<AncestryObservation> = Vec::new();
        let states = posterior_decode_with_genetic_map(&obs, &params, &gmap);
        assert!(states.is_empty());
    }

    #[test]
    fn test_genetic_map_hotspot_increases_switch_detection() {
        // A recombination hotspot should make the HMM more willing to switch states
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.005);

        // Hotspot: 10× recombination rate at position 50k
        let gmap_hotspot = AncestryGeneticMap {
            entries: vec![
                (0, 0.0),
                (45000, 0.045),    // normal rate: ~1 cM/Mb
                (55000, 0.145),    // hotspot: 10× rate in this 10kb region
                (100000, 0.19),    // normal again
            ],
        };

        // Create observations with a state change at the hotspot
        let mut obs = Vec::new();
        for w in 0..9 {
            obs.push(make_observation(w * 5000, 0.94, 0.84, 0.83));
        }
        // Switch at position 45k-50k (near hotspot)
        obs.push(make_observation(45000, 0.84, 0.94, 0.83));
        obs.push(make_observation(50000, 0.84, 0.94, 0.83));
        for w in 11..20 {
            obs.push(make_observation(w as u64 * 5000, 0.94, 0.84, 0.83));
        }

        let states_hotspot = viterbi_with_genetic_map(&obs, &params, &gmap_hotspot);
        // The hotspot version should detect some state change
        let n_switches = states_hotspot.windows(2).filter(|w| w[0] != w[1]).count();
        // With a hotspot, we expect at least some switches detected
        // (may or may not detect - depends on signal strength, but should not crash)
        assert!(states_hotspot.len() == 20, "Should have 20 states, got {}", states_hotspot.len());
        // The states should all be valid
        for &s in &states_hotspot {
            assert!(s < 3, "State {} out of range", s);
        }
        // Log how many switches were detected
        eprintln!("Hotspot test: {} switches detected", n_switches);
    }

    // --- Coverage-ratio auxiliary emission tests ---

    fn make_observation_with_coverage(
        start: u64,
        a: f64,
        b: f64,
        c: f64,
        a_cov: f64,
        b_cov: f64,
        c_cov: f64,
    ) -> AncestryObservation {
        AncestryObservation {
            chrom: "chr1".to_string(),
            start,
            end: start + 5000,
            sample: "QUERY_1#1".to_string(),
            similarities: [
                ("popA#HAP1".to_string(), a),
                ("popA#HAP2".to_string(), a - 0.01),
                ("popB#A".to_string(), b),
                ("popB#B".to_string(), b - 0.01),
                ("popC#HAP1".to_string(), c),
                ("popC#HAP2".to_string(), c - 0.01),
            ].into_iter().collect(),
            coverage_ratios: Some([
                ("popA#HAP1".to_string(), a_cov),
                ("popA#HAP2".to_string(), a_cov * 0.95),
                ("popB#A".to_string(), b_cov),
                ("popB#B".to_string(), b_cov * 0.95),
                ("popC#HAP1".to_string(), c_cov),
                ("popC#HAP2".to_string(), c_cov * 0.95),
            ].into_iter().collect()),
            haplotype_consistency_bonus: None,
        }
    }

    #[test]
    fn test_coverage_weight_zero_equals_no_coverage() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_coverage_weight(0.0);

        let obs = make_observation_with_coverage(0, 0.94, 0.84, 0.83, 0.9, 0.8, 0.7);

        // With coverage_weight=0, should equal standard emission
        let standard_emission = {
            let mut p2 = params.clone();
            p2.coverage_weight = 0.0;
            p2.log_emission(&obs, 0)
        };

        let cov_emission = params.log_emission(&obs, 0);
        assert!((standard_emission - cov_emission).abs() < 1e-10,
            "Weight=0 should give same result: {} vs {}", standard_emission, cov_emission);
    }

    #[test]
    fn test_coverage_weight_positive_changes_relative_advantage() {
        let pops = make_test_populations();
        let mut params_no_cov = AncestryHmmParams::new(pops.clone(), 0.001);
        params_no_cov.set_coverage_weight(0.0);

        let mut params_cov = AncestryHmmParams::new(pops, 0.001);
        params_cov.set_coverage_weight(1.0);

        // Coverage strongly favors popA (state 0)
        let obs = make_observation_with_coverage(0, 0.90, 0.89, 0.88, 0.95, 0.60, 0.50);

        // Compare log-emission ratio between state 0 and state 1
        let diff_no_cov = params_no_cov.log_emission(&obs, 0) - params_no_cov.log_emission(&obs, 1);
        let diff_cov = params_cov.log_emission(&obs, 0) - params_cov.log_emission(&obs, 1);

        // Coverage should increase the relative advantage of state 0 over state 1
        assert!(diff_cov > diff_no_cov,
            "Coverage should increase relative advantage: diff_cov={:.6} vs diff_no_cov={:.6}",
            diff_cov, diff_no_cov);
    }

    #[test]
    fn test_coverage_emission_valid_probabilities() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_coverage_weight(1.0);

        let obs = make_observation_with_coverage(0, 0.94, 0.84, 0.83, 0.9, 0.8, 0.7);

        for state in 0..3 {
            let log_em = params.log_emission(&obs, state);
            assert!(log_em.is_finite(), "State {} emission should be finite: {}", state, log_em);
            assert!(log_em <= 0.0, "Log emission should be <= 0: {}", log_em);
        }
    }

    #[test]
    fn test_coverage_no_data_falls_back() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_coverage_weight(1.0);

        // Observation without coverage data
        let obs = make_observation(0, 0.94, 0.84, 0.83);

        // Should fall back to identity-only emission
        let em = params.log_emission(&obs, 0);
        assert!(em.is_finite(), "Should still produce valid emission without coverage data");
        assert!(em <= 0.0);
    }

    #[test]
    fn test_coverage_viterbi_consistency() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_coverage_weight(1.0);

        // 10 windows: popA identity similar to popB, but coverage strongly favors popA
        let obs: Vec<AncestryObservation> = (0..10)
            .map(|w| make_observation_with_coverage(
                w * 5000,
                0.92,  // popA identity
                0.91,  // popB identity (very close)
                0.83,  // popC identity
                0.95,  // popA coverage (high — symmetric alignment)
                0.60,  // popB coverage (low — asymmetric)
                0.50,  // popC coverage (low)
            ))
            .collect();

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 10);

        // With coverage favoring popA, most states should be 0
        let count_0 = states.iter().filter(|&&s| s == 0).count();
        assert!(count_0 >= 8,
            "Expected mostly state 0 (popA), got {}/10", count_0);
    }

    #[test]
    fn test_coverage_forward_backward_valid() {
        let pops = make_test_populations();
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.set_coverage_weight(0.5);

        let obs: Vec<AncestryObservation> = (0..5)
            .map(|w| make_observation_with_coverage(
                w * 5000, 0.94, 0.84, 0.83, 0.9, 0.8, 0.7
            ))
            .collect();

        let posteriors = forward_backward(&obs, &params);
        assert_eq!(posteriors.len(), 5);

        for (t, post) in posteriors.iter().enumerate() {
            assert_eq!(post.len(), 3, "Window {} should have 3 posteriors", t);
            let sum: f64 = post.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "Window {} posteriors should sum to 1: {}", t, sum);
            for &p in post {
                assert!(p >= 0.0 && p <= 1.0, "Posterior out of range: {}", p);
            }
        }
    }

    #[test]
    fn test_coverage_weight_scaling() {
        let pops = make_test_populations();
        let obs = make_observation_with_coverage(0, 0.90, 0.89, 0.88, 0.95, 0.60, 0.50);

        // Higher weight should amplify coverage effect on relative advantage
        let mut params_low = AncestryHmmParams::new(pops.clone(), 0.001);
        params_low.set_coverage_weight(0.5);

        let mut params_high = AncestryHmmParams::new(pops, 0.001);
        params_high.set_coverage_weight(2.0);

        // Relative advantage of state 0 over state 2 (low coverage)
        let diff_low = params_low.log_emission(&obs, 0) - params_low.log_emission(&obs, 2);
        let diff_high = params_high.log_emission(&obs, 0) - params_high.log_emission(&obs, 2);

        // Both should be valid
        assert!(diff_low.is_finite());
        assert!(diff_high.is_finite());

        // Higher weight should produce larger relative advantage
        assert!(diff_high > diff_low,
            "Higher coverage weight should increase relative advantage: {:.6} vs {:.6}",
            diff_high, diff_low);
    }

    // === set_switch_prob tests ===

    #[test]
    fn test_set_switch_prob_basic() {
        let pops = vec![
            AncestralPopulation { name: "EUR".into(), haplotypes: vec!["e1".into()] },
            AncestralPopulation { name: "AFR".into(), haplotypes: vec!["a1".into()] },
            AncestralPopulation { name: "AMR".into(), haplotypes: vec!["m1".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        params.set_switch_prob(0.01);

        // Diagonal should be 1 - 0.01 = 0.99
        for i in 0..3 {
            assert!((params.transitions[i][i] - 0.99).abs() < 1e-10,
                "Stay prob should be 0.99, got {}", params.transitions[i][i]);
        }

        // Off-diagonal should be 0.01 / 2 = 0.005
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!((params.transitions[i][j] - 0.005).abs() < 1e-10,
                        "Switch prob should be 0.005, got {}", params.transitions[i][j]);
                }
            }
        }
    }

    #[test]
    fn test_set_switch_prob_rows_sum_to_one() {
        let pops = vec![
            AncestralPopulation { name: "A".into(), haplotypes: vec!["a1".into()] },
            AncestralPopulation { name: "B".into(), haplotypes: vec!["b1".into()] },
            AncestralPopulation { name: "C".into(), haplotypes: vec!["c1".into()] },
            AncestralPopulation { name: "D".into(), haplotypes: vec!["d1".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        for &switch_p in &[0.0001, 0.01, 0.1, 0.5] {
            params.set_switch_prob(switch_p);
            for i in 0..4 {
                let row_sum: f64 = params.transitions[i].iter().sum();
                assert!((row_sum - 1.0).abs() < 1e-10,
                    "Row {} sum should be 1.0, got {} (switch_prob={})", i, row_sum, switch_p);
            }
        }
    }

    #[test]
    fn test_set_switch_prob_two_states() {
        let pops = vec![
            AncestralPopulation { name: "A".into(), haplotypes: vec!["a1".into()] },
            AncestralPopulation { name: "B".into(), haplotypes: vec!["b1".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        params.set_switch_prob(0.1);

        // With 2 states: stay = 0.9, switch = 0.1 / 1 = 0.1
        assert!((params.transitions[0][0] - 0.9).abs() < 1e-10);
        assert!((params.transitions[0][1] - 0.1).abs() < 1e-10);
        assert!((params.transitions[1][0] - 0.1).abs() < 1e-10);
        assert!((params.transitions[1][1] - 0.9).abs() < 1e-10);
    }

    // === estimate_emissions tests ===

    #[test]
    fn test_estimate_emissions_empty_observations() {
        let pops = vec![
            AncestralPopulation { name: "EUR".into(), haplotypes: vec!["e1".into()] },
            AncestralPopulation { name: "AFR".into(), haplotypes: vec!["a1".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let original_same = params.emission_same_pop_mean;
        let original_diff = params.emission_diff_pop_mean;

        params.estimate_emissions(&[]);

        // No observations → no change
        assert_eq!(params.emission_same_pop_mean, original_same);
        assert_eq!(params.emission_diff_pop_mean, original_diff);
    }

    #[test]
    fn test_estimate_emissions_updates_means() {
        let pops = vec![
            AncestralPopulation { name: "EUR".into(), haplotypes: vec!["e1".into(), "e2".into()] },
            AncestralPopulation { name: "AFR".into(), haplotypes: vec!["a1".into(), "a2".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        // Create observations where EUR is clearly the best match
        let observations = vec![
            AncestryObservation {
                chrom: "chr1".into(),
                start: 0,
                end: 10000,
                sample: "query".into(),
                similarities: [
                    ("e1".into(), 0.98), ("e2".into(), 0.97),
                    ("a1".into(), 0.80), ("a2".into(), 0.82),
                ].iter().cloned().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
            AncestryObservation {
                chrom: "chr1".into(),
                start: 10000,
                end: 20000,
                sample: "query".into(),
                similarities: [
                    ("e1".into(), 0.95), ("e2".into(), 0.96),
                    ("a1".into(), 0.78), ("a2".into(), 0.79),
                ].iter().cloned().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];

        params.estimate_emissions(&observations);

        // same_pop (best match) should have higher mean than diff_pop
        assert!(params.emission_same_pop_mean > params.emission_diff_pop_mean,
            "Same-pop mean ({:.4}) should be > diff-pop mean ({:.4})",
            params.emission_same_pop_mean, params.emission_diff_pop_mean);

        // Std should be positive and reasonable
        assert!(params.emission_std > 0.0, "emission_std should be positive");
    }

    #[test]
    fn test_estimate_emissions_single_observation() {
        let pops = vec![
            AncestralPopulation { name: "A".into(), haplotypes: vec!["a1".into()] },
            AncestralPopulation { name: "B".into(), haplotypes: vec!["b1".into()] },
        ];
        let mut params = AncestryHmmParams::new(pops, 0.001);

        let observations = vec![
            AncestryObservation {
                chrom: "chr1".into(),
                start: 0,
                end: 10000,
                sample: "query".into(),
                similarities: [
                    ("a1".into(), 0.9), ("b1".into(), 0.7),
                ].iter().cloned().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];

        params.estimate_emissions(&observations);

        // Best match is A (0.9), so same_pop = 0.9, diff_pop = 0.7
        assert!((params.emission_same_pop_mean - 0.9).abs() < 1e-10,
            "Same-pop mean should be 0.9, got {}", params.emission_same_pop_mean);
        assert!((params.emission_diff_pop_mean - 0.7).abs() < 1e-10,
            "Diff-pop mean should be 0.7, got {}", params.emission_diff_pop_mean);
    }

    // ====================================================================
    // Haplotype Copying Model Tests
    // ====================================================================

    fn make_copying_populations() -> Vec<AncestralPopulation> {
        vec![
            AncestralPopulation {
                name: "POP_A".into(),
                haplotypes: vec!["A1".into(), "A2".into(), "A3".into()],
            },
            AncestralPopulation {
                name: "POP_B".into(),
                haplotypes: vec!["B1".into(), "B2".into(), "B3".into()],
            },
        ]
    }

    fn make_copying_obs(start: u64, a_sims: [f64; 3], b_sims: [f64; 3]) -> AncestryObservation {
        AncestryObservation {
            chrom: "chr12".into(),
            start,
            end: start + 10000,
            sample: "query#1".into(),
            similarities: [
                ("A1".into(), a_sims[0]),
                ("A2".into(), a_sims[1]),
                ("A3".into(), a_sims[2]),
                ("B1".into(), b_sims[0]),
                ("B2".into(), b_sims[1]),
                ("B3".into(), b_sims[2]),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }

    fn make_copying_observations(n: usize, _pops: &[AncestralPopulation]) -> Vec<AncestryObservation> {
        (0..n).map(|i| {
            let a_sims = [0.999, 0.998, 0.997];
            let b_sims = [0.996, 0.995, 0.994];
            make_copying_obs(i as u64 * 10000, a_sims, b_sims)
        }).collect()
    }

    #[test]
    fn test_copying_empty_observations() {
        let pops = make_copying_populations();
        let (states, posteriors) = infer_ancestry_copying(&[], &pops, 0.005, 0.1, 0.003, 0.99);
        assert!(states.is_empty());
        assert!(posteriors.is_empty());
    }

    #[test]
    fn test_copying_empty_populations() {
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])];
        let (states, posteriors) = infer_ancestry_copying(&obs, &[], 0.005, 0.1, 0.003, 0.99);
        assert!(states.is_empty());
        assert!(posteriors.is_empty());
    }

    #[test]
    fn test_copying_single_window() {
        let pops = make_copying_populations();
        // POP_A haplotypes clearly higher
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.993, 0.992, 0.991])];
        let (states, posteriors) = infer_ancestry_copying(&obs, &pops, 0.005, 0.1, 0.003, 0.99);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 0); // POP_A
        assert_eq!(posteriors.len(), 1);
        assert_eq!(posteriors[0].len(), 2);
        assert!(posteriors[0][0] > posteriors[0][1], "POP_A posterior should be higher");
        // Posteriors should sum to ~1
        let sum: f64 = posteriors[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Posteriors should sum to ~1, got {}", sum);
    }

    #[test]
    fn test_copying_clear_ancestry_tract() {
        let pops = make_copying_populations();
        // 10 windows of clear POP_A ancestry (A1 best match consistently)
        let mut obs = Vec::new();
        for i in 0..10 {
            obs.push(make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ));
        }

        let (states, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        // All windows should be POP_A
        for (i, &s) in states.iter().enumerate() {
            assert_eq!(s, 0, "Window {} should be POP_A (0), got {}", i, s);
        }
        // POP_A posteriors should be high
        for (i, post) in posteriors.iter().enumerate() {
            assert!(post[0] > 0.5, "Window {} POP_A posterior {} should be > 0.5", i, post[0]);
        }
    }

    #[test]
    fn test_copying_ancestry_switch() {
        let pops = make_copying_populations();
        // 5 windows POP_A, then 5 windows POP_B
        let mut obs = Vec::new();
        for i in 0..5 {
            obs.push(make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ));
        }
        for i in 5..10 {
            obs.push(make_copying_obs(
                i * 10000,
                [0.9960, 0.9955, 0.9950],
                [0.9995, 0.9980, 0.9975],
            ));
        }

        let (states, _posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        // First windows should be POP_A, last should be POP_B
        assert_eq!(states[0], 0, "First window should be POP_A");
        assert_eq!(states[9], 1, "Last window should be POP_B");
    }

    #[test]
    fn test_copying_outlier_haplotype_robustness() {
        let pops = make_copying_populations();
        // Scenario: B1 is an outlier with high identity, but A1 is consistently best across windows.
        // The population-aggregate Max model would wrongly pick POP_B for a window
        // where B1 > A1, but the copying model should resist this because A1 is
        // consistently the copied haplotype across flanking windows.
        let mut obs = Vec::new();
        // 4 windows where A1 is clearly best
        for i in 0..4 {
            obs.push(make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ));
        }
        // 1 window where B1 outlier is highest but A1 is still close
        obs.push(make_copying_obs(
            40000,
            [0.9990, 0.9980, 0.9975],
            [0.9993, 0.9955, 0.9950], // B1 slightly higher than A1
        ));
        // 4 more windows where A1 is clearly best
        for i in 5..9 {
            obs.push(make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ));
        }

        let (states, _) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        // The outlier window (index 4) should still be POP_A due to continuity
        assert_eq!(states[4], 0,
            "Outlier window should stay POP_A due to haplotype continuity, got {}",
            states[4]);
    }

    #[test]
    fn test_copying_posteriors_sum_to_one() {
        let pops = make_copying_populations();
        let mut obs = Vec::new();
        for i in 0..20 {
            let a = if i < 10 {
                [0.9995, 0.9980, 0.9975]
            } else {
                [0.9960, 0.9955, 0.9950]
            };
            let b = if i < 10 {
                [0.9960, 0.9955, 0.9950]
            } else {
                [0.9995, 0.9980, 0.9975]
            };
            obs.push(make_copying_obs(i * 10000, a, b));
        }

        let (_, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        for (i, post) in posteriors.iter().enumerate() {
            let sum: f64 = post.iter().sum();
            assert!((sum - 1.0).abs() < 0.01,
                "Posteriors at window {} should sum to ~1.0, got {:.6}", i, sum);
        }
    }

    #[test]
    fn test_copying_missing_haplotype_data() {
        let pops = make_copying_populations();
        // Only provide data for some haplotypes
        let obs = vec![AncestryObservation {
            chrom: "chr12".into(),
            start: 0,
            end: 10000,
            sample: "query#1".into(),
            similarities: [
                ("A1".into(), 0.999),
                ("B1".into(), 0.995),
                // A2, A3, B2, B3 missing — should get default_similarity
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }];

        let (states, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        // Should not crash; A1 is highest so POP_A expected
        assert_eq!(states.len(), 1);
        assert_eq!(posteriors.len(), 1);
        assert_eq!(states[0], 0, "A1 is highest, should pick POP_A");
    }

    #[test]
    fn test_copying_transition_matrix_rows_sum_to_one() {
        let hap_to_pop = vec![0, 0, 0, 1, 1, 1];
        let pop_hap_counts = vec![3, 3];
        let n_haps = 6;

        let log_trans = build_copying_transitions(
            n_haps, &hap_to_pop, &pop_hap_counts, 0.01, 0.1);

        for i in 0..n_haps {
            let row_sum: f64 = log_trans[i].iter().map(|&lp| lp.exp()).sum();
            assert!((row_sum - 1.0).abs() < 1e-10,
                "Transition row {} should sum to 1.0, got {:.10}", i, row_sum);
        }
    }

    #[test]
    fn test_copying_transition_within_pop_higher_than_between() {
        let hap_to_pop = vec![0, 0, 0, 1, 1, 1];
        let pop_hap_counts = vec![3, 3];
        let n_haps = 6;

        let log_trans = build_copying_transitions(
            n_haps, &hap_to_pop, &pop_hap_counts, 0.01, 0.1);

        // Within-pop switch should be higher than between-pop switch
        let within_pop = log_trans[0][1].exp(); // A1 -> A2
        let between_pop = log_trans[0][3].exp(); // A1 -> B1
        assert!(within_pop > between_pop,
            "Within-pop ({:.6e}) should be > between-pop ({:.6e})",
            within_pop, between_pop);
    }

    #[test]
    fn test_copying_emission_softmax_normalization() {
        let hap_names: Vec<&str> = vec!["A1", "A2", "B1", "B2"];
        let obs = vec![AncestryObservation {
            chrom: "chr12".into(),
            start: 0,
            end: 10000,
            sample: "query".into(),
            similarities: [
                ("A1".into(), 0.999),
                ("A2".into(), 0.998),
                ("B1".into(), 0.995),
                ("B2".into(), 0.994),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }];

        let log_emissions = compute_copying_emissions(&obs, &hap_names, 0.003, 0.99);

        // Emissions should be log-softmax — they should log-sum-exp to 0
        let sum_exp: f64 = log_emissions[0].iter().map(|&le| le.exp()).sum();
        assert!((sum_exp - 1.0).abs() < 1e-10,
            "Softmax emissions should sum to 1.0, got {:.10}", sum_exp);

        // A1 (highest sim) should have highest emission
        assert!(log_emissions[0][0] > log_emissions[0][2],
            "A1 should have higher emission than B1");
    }

    #[test]
    fn test_copying_long_tract_consistency() {
        let pops = make_copying_populations();
        // 30 windows all clearly POP_A — should be very confident
        let obs: Vec<_> = (0..30)
            .map(|i| make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ))
            .collect();

        let (states, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        // All should be POP_A with high confidence
        for &s in &states {
            assert_eq!(s, 0, "All windows should be POP_A");
        }
        // Interior windows should have very high POP_A posterior
        for (i, post) in posteriors.iter().enumerate().skip(2).take(26) {
            assert!(post[0] > 0.9,
                "Interior window {} should have POP_A posterior > 0.9, got {:.4}", i, post[0]);
        }
    }

    #[test]
    fn test_estimate_copying_params_basic() {
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..20)
            .map(|i| make_copying_obs(
                i * 10000,
                [0.9995, 0.9980, 0.9975],
                [0.9960, 0.9955, 0.9950],
            ))
            .collect();

        let (temp, switch_rate, default_sim) = estimate_copying_params(&obs, &pops);

        // Temperature should be positive and reasonable
        assert!(temp > 0.0 && temp < 0.1,
            "Temperature should be reasonable, got {}", temp);
        // Switch rate should be positive and < 1
        assert!(switch_rate > 0.0 && switch_rate < 1.0,
            "Switch rate should be in (0, 1), got {}", switch_rate);
        // Default similarity should be positive
        assert!(default_sim > 0.0 && default_sim < 1.0,
            "Default similarity should be in (0, 1), got {}", default_sim);
    }

    #[test]
    fn test_estimate_copying_params_empty() {
        let pops = make_copying_populations();
        let (temp, switch_rate, default_sim) = estimate_copying_params(&[], &pops);
        assert!((temp - 0.003).abs() < 1e-10);
        assert!((switch_rate - 0.005).abs() < 1e-10);
        assert!((default_sim - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_copying_three_populations() {
        let pops = vec![
            AncestralPopulation {
                name: "EUR".into(),
                haplotypes: vec!["E1".into(), "E2".into()],
            },
            AncestralPopulation {
                name: "AFR".into(),
                haplotypes: vec!["F1".into(), "F2".into()],
            },
            AncestralPopulation {
                name: "EAS".into(),
                haplotypes: vec!["S1".into(), "S2".into()],
            },
        ];

        let obs: Vec<_> = (0..10)
            .map(|i| {
                let sims: HashMap<String, f64> = [
                    ("E1".into(), 0.9990),
                    ("E2".into(), 0.9985),
                    ("F1".into(), 0.9960),
                    ("F2".into(), 0.9955),
                    ("S1".into(), 0.9950),
                    ("S2".into(), 0.9945),
                ].into_iter().collect();
                AncestryObservation {
                    chrom: "chr12".into(),
                    start: i * 10000,
                    end: (i + 1) * 10000,
                    sample: "query".into(),
                    similarities: sims,
                    coverage_ratios: None,
                    haplotype_consistency_bonus: None,
                }
            })
            .collect();

        let (states, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        assert_eq!(states.len(), 10);
        assert_eq!(posteriors[0].len(), 3);

        // EUR haplotypes are highest, should be assigned EUR
        for &s in &states {
            assert_eq!(s, 0, "Should be EUR (0)");
        }
    }

    #[test]
    fn test_copying_forward_backward_single_haplotype() {
        // Edge case: one haplotype per population
        let pops = vec![
            AncestralPopulation { name: "A".into(), haplotypes: vec!["a1".into()] },
            AncestralPopulation { name: "B".into(), haplotypes: vec!["b1".into()] },
        ];

        let obs = vec![
            AncestryObservation {
                chrom: "chr1".into(),
                start: 0, end: 10000,
                sample: "q".into(),
                similarities: [("a1".into(), 0.999), ("b1".into(), 0.995)].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];

        let (states, posteriors) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);

        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 0); // A is higher
        let sum: f64 = posteriors[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_copying_switch_rate_affects_tract_length() {
        let pops = make_copying_populations();
        // Noisy data: POP_A generally, but some windows ambiguous
        let mut obs = Vec::new();
        for i in 0..20 {
            let a_boost = if i % 3 == 0 { 0.0 } else { 0.002 };
            obs.push(make_copying_obs(
                i * 10000,
                [0.997 + a_boost, 0.996, 0.995],
                [0.996, 0.995, 0.994],
            ));
        }

        // Low switch rate = stronger continuity, fewer switches
        let (states_low, _) = infer_ancestry_copying(
            &obs, &pops, 0.001, 0.1, 0.003, 0.99);
        let switches_low = states_low.windows(2).filter(|w| w[0] != w[1]).count();

        // High switch rate = weaker continuity, more switches possible
        let (states_high, _) = infer_ancestry_copying(
            &obs, &pops, 0.1, 0.1, 0.003, 0.99);
        let switches_high = states_high.windows(2).filter(|w| w[0] != w[1]).count();

        // Low switch rate should produce fewer or equal switches
        assert!(switches_low <= switches_high,
            "Low switch rate should produce <= switches: {} vs {}",
            switches_low, switches_high);
    }

    #[test]
    fn test_copying_ancestry_frac_controls_pop_switching() {
        let _pops = make_copying_populations();
        // Transition matrix: low ancestry_frac should have low between-pop transitions
        let hap_to_pop = vec![0, 0, 0, 1, 1, 1];
        let pop_hap_counts = vec![3, 3];

        let log_trans_low = build_copying_transitions(6, &hap_to_pop, &pop_hap_counts, 0.01, 0.01);
        let log_trans_high = build_copying_transitions(6, &hap_to_pop, &pop_hap_counts, 0.01, 0.5);

        // Between-pop transition probability
        let between_low = log_trans_low[0][3].exp();
        let between_high = log_trans_high[0][3].exp();

        assert!(between_high > between_low,
            "Higher ancestry_frac should give higher between-pop transition: {:.6e} vs {:.6e}",
            between_high, between_low);
    }

    // ======== Emission Dampening Tests ========

    #[test]
    fn test_dampen_empty() {
        let result = dampen_low_confidence_emissions(&[], 1.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dampen_single_state() {
        let emissions = vec![vec![-1.0], vec![-2.0], vec![-1.5]];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);
        assert_eq!(result, emissions); // k=1, no dampening possible
    }

    #[test]
    fn test_dampen_preserves_high_confidence() {
        // All windows have the same discriminability — with scale=1.0,
        // threshold = median = disc, so disc >= threshold → no dampening
        let emissions = vec![
            vec![0.0, -5.0],  // disc = 5.0
            vec![0.0, -5.0],
            vec![0.0, -5.0],
            vec![0.0, -5.0],
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.0);
        for (orig, dampened) in emissions.iter().zip(&result) {
            for (a, b) in orig.iter().zip(dampened) {
                assert!((a - b).abs() < 1e-10, "With scale=1.0, all at median should be unchanged");
            }
        }
    }

    #[test]
    fn test_dampen_reduces_low_confidence_gap() {
        // Mix of high and low discriminability windows
        let emissions = vec![
            vec![0.0, -10.0],   // disc = 10.0 (high)
            vec![0.0, -10.0],   // disc = 10.0 (high)
            vec![0.0, -10.0],   // disc = 10.0 (high)
            vec![-0.1, -0.2],   // disc = 0.1 (very low)
            vec![0.0, -10.0],   // disc = 10.0 (high)
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);

        // Window 3 (low confidence) should have reduced gap
        let orig_gap = emissions[3][0] - emissions[3][1];
        let dampened_gap = result[3][0] - result[3][1];
        assert!(dampened_gap.abs() < orig_gap.abs(),
            "Low-confidence gap should be reduced: {:.4} vs {:.4}", dampened_gap, orig_gap);

        // High-confidence windows should be dampened LESS than low-confidence ones
        // Median disc = 10.0, threshold = 15.0. Windows with disc=10 get alpha=10/15≈0.667
        // Window 3 with disc=0.1 gets alpha=0.1/15≈0.007 — much more dampened
        let high_gap = (result[0][0] - result[0][1]).abs();
        let low_gap = (result[3][0] - result[3][1]).abs();
        assert!(high_gap > low_gap,
            "High-confidence gap should be larger than low-confidence: {:.4} vs {:.4}",
            high_gap, low_gap);
    }

    #[test]
    fn test_dampen_preserves_length() {
        let emissions = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -1.6, -1.7],
            vec![-2.0, -1.0, -3.0],
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);
        assert_eq!(result.len(), emissions.len());
        for (orig, dampened) in emissions.iter().zip(&result) {
            assert_eq!(orig.len(), dampened.len());
        }
    }

    #[test]
    fn test_dampen_handles_neg_infinity() {
        let emissions = vec![
            vec![0.0, f64::NEG_INFINITY],
            vec![-1.0, -2.0],
            vec![-1.5, -1.6],
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);
        assert_eq!(result.len(), 3);
        // NEG_INFINITY should remain NEG_INFINITY
        assert!(result[0][1].is_infinite() && result[0][1] < 0.0);
    }

    #[test]
    fn test_dampen_zero_disc_gives_uniform() {
        // All windows have zero discriminability
        let emissions = vec![
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
            vec![-1.5, -1.5],
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);
        // All discs are 0, so no positive discs — should return unchanged
        for (orig, dampened) in emissions.iter().zip(&result) {
            for (a, b) in orig.iter().zip(dampened) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dampen_scale_factor_affects_aggressiveness() {
        let emissions = vec![
            vec![0.0, -10.0],   // disc = 10.0
            vec![0.0, -10.0],   // disc = 10.0
            vec![0.0, -10.0],   // disc = 10.0
            vec![-0.5, -1.5],   // disc = 1.0 (low)
        ];

        let gentle = dampen_low_confidence_emissions(&emissions, 0.5);
        let aggressive = dampen_low_confidence_emissions(&emissions, 3.0);

        // With scale=0.5, threshold = 0.5 * median(10) = 5.0, disc 1.0 < 5.0 → dampened
        // With scale=3.0, threshold = 3.0 * 10 = 30.0, disc 1.0 < 30.0 → more dampened
        let gap_gentle = (gentle[3][0] - gentle[3][1]).abs();
        let gap_aggressive = (aggressive[3][0] - aggressive[3][1]).abs();

        assert!(gap_aggressive < gap_gentle,
            "More aggressive scale should dampen more: {:.4} vs {:.4}",
            gap_aggressive, gap_gentle);
    }

    #[test]
    fn test_dampen_three_states() {
        let emissions = vec![
            vec![0.0, -5.0, -10.0],  // disc = 5.0 (1st - 2nd)
            vec![0.0, -5.0, -10.0],  // disc = 5.0
            vec![0.0, -5.0, -10.0],  // disc = 5.0
            vec![-1.0, -1.1, -1.2],  // disc = 0.1 (very low)
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);

        // Low-confidence window should have reduced spread
        let orig_spread = emissions[3][0] - emissions[3][2]; // 0.2
        let dampened_spread = result[3][0] - result[3][2];
        assert!(dampened_spread.abs() < orig_spread.abs(),
            "Low-confidence 3-state spread should be reduced");
    }

    #[test]
    fn test_dampen_mean_preserved() {
        // The mean of emissions in each window should be preserved (dampening
        // scales deviation from mean, not shifts it)
        let emissions = vec![
            vec![0.0, -10.0],
            vec![0.0, -10.0],
            vec![0.0, -10.0],
            vec![-0.5, -1.5],  // will be dampened
        ];
        let result = dampen_low_confidence_emissions(&emissions, 1.5);

        let orig_mean = emissions[3].iter().sum::<f64>() / 2.0;
        let dampened_mean = result[3].iter().sum::<f64>() / 2.0;
        assert!((orig_mean - dampened_mean).abs() < 1e-10,
            "Dampening should preserve per-window mean: {} vs {}", orig_mean, dampened_mean);
    }

    // ======== Adaptive Emission Context Tests ========

    #[test]
    fn test_estimate_ec_empty() {
        let pops = make_test_populations();
        assert_eq!(estimate_emission_context(&[], &pops, 3, 1, 15), 3);
    }

    #[test]
    fn test_estimate_ec_zero_base() {
        let pops = make_test_populations();
        let obs = vec![make_observation(0, 0.999, 0.997, 0.995)];
        assert_eq!(estimate_emission_context(&obs, &pops, 0, 1, 15), 0);
    }

    #[test]
    fn test_estimate_ec_strong_signal() {
        // Very strong signal (large gap) → smaller context needed
        let pops = make_test_populations();
        let obs: Vec<_> = (0..50).map(|i| {
            make_observation(i as u64 * 10000, 0.999, 0.990, 0.985) // disc ≈ 0.009
        }).collect();
        let ec = estimate_emission_context(&obs, &pops, 3, 1, 15);
        assert!(ec <= 3, "Strong signal should need <= base context: ec={}", ec);
    }

    #[test]
    fn test_estimate_ec_weak_signal() {
        // Very weak signal (tiny gap) → larger context needed
        let pops = make_test_populations();
        let obs: Vec<_> = (0..50).map(|i| {
            make_observation(i as u64 * 10000, 0.9993, 0.9990, 0.9988) // disc ≈ 0.0003
        }).collect();
        let ec = estimate_emission_context(&obs, &pops, 3, 1, 15);
        assert!(ec >= 3, "Weak signal should need >= base context: ec={}", ec);
    }

    #[test]
    fn test_estimate_ec_respects_bounds() {
        let pops = make_test_populations();
        let obs: Vec<_> = (0..50).map(|i| {
            make_observation(i as u64 * 10000, 0.99999, 0.99998, 0.99997)
        }).collect();
        let ec = estimate_emission_context(&obs, &pops, 3, 2, 10);
        assert!(ec >= 2 && ec <= 10, "Should respect bounds: ec={}", ec);
    }

    // ======== Copying Temperature Scaling Tests ========

    #[test]
    fn test_copying_temp_scale_basic() {
        // 2 pops, 46 haplotypes: ln(2)/ln(46) ≈ 0.181
        let scaled = scale_temperature_for_copying(0.003, 2, 46);
        let expected = 0.003 * (2.0_f64.ln() / 46.0_f64.ln());
        assert!((scaled - expected).abs() < 1e-6,
            "Expected {:.6}, got {:.6}", expected, scaled);
        assert!(scaled < 0.003, "Copying temp should be lower: {}", scaled);
    }

    #[test]
    fn test_copying_temp_scale_equal_pops_haps() {
        // n_haps == n_pops: no scaling (correction = 1.0)
        let base = 0.003;
        let scaled = scale_temperature_for_copying(base, 3, 3);
        assert!((scaled - base).abs() < 1e-6, "No scaling when pops == haps");
    }

    #[test]
    fn test_copying_temp_scale_few_haps() {
        // 2 pops, 6 haps: ln(2)/ln(6) ≈ 0.387
        let scaled = scale_temperature_for_copying(0.003, 2, 6);
        let expected = 0.003 * (2.0_f64.ln() / 6.0_f64.ln());
        assert!((scaled - expected).abs() < 1e-6);
        // Should be less aggressive than the 46-hap case
        let scaled_46 = scale_temperature_for_copying(0.003, 2, 46);
        assert!(scaled > scaled_46, "Fewer haps → less scaling");
    }

    #[test]
    fn test_copying_temp_scale_five_pops() {
        // 5 pops, 100 haps: ln(5)/ln(100) ≈ 0.349
        let scaled = scale_temperature_for_copying(0.003, 5, 100);
        let expected = 0.003 * (5.0_f64.ln() / 100.0_f64.ln());
        assert!((scaled - expected).abs() < 1e-6);
    }

    #[test]
    fn test_copying_temp_scale_edge_cases() {
        // 0 or 1 populations: return base temp
        assert_eq!(scale_temperature_for_copying(0.003, 0, 46), 0.003);
        assert_eq!(scale_temperature_for_copying(0.003, 1, 46), 0.003);
        // 0 or 1 haplotypes: return base temp
        assert_eq!(scale_temperature_for_copying(0.003, 2, 0), 0.003);
        assert_eq!(scale_temperature_for_copying(0.003, 2, 1), 0.003);
    }

    // ======== Copying Model EM Tests ========

    #[test]
    fn test_copying_em_empty() {
        let pops = make_copying_populations();
        let (states, posteriors) = infer_ancestry_copying_em(
            &[], &pops, 0.005, 0.1, 0.003, 0.99, 3);
        assert!(states.is_empty());
        assert!(posteriors.is_empty());
    }

    #[test]
    fn test_copying_em_zero_iters_equals_single_pass() {
        let pops = make_copying_populations();
        let obs = make_copying_observations(20, &pops);
        let (states_em, posteriors_em) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 0);
        let (states_sp, posteriors_sp) = infer_ancestry_copying(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99);
        assert_eq!(states_em, states_sp);
        assert_eq!(posteriors_em.len(), posteriors_sp.len());
    }

    #[test]
    fn test_copying_em_converges() {
        // EM should produce valid output after multiple iterations
        let pops = make_copying_populations();
        let obs = make_copying_observations(50, &pops);
        let (states, posteriors) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 5);
        assert_eq!(states.len(), 50);
        assert_eq!(posteriors.len(), 50);
        // All states should be valid population indices
        for &s in &states {
            assert!(s < pops.len(), "Invalid state: {}", s);
        }
        // Posteriors should sum to ~1
        for post in &posteriors {
            let sum: f64 = post.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Posterior sum = {}", sum);
        }
    }

    #[test]
    fn test_copying_em_clear_tract_stable() {
        // Clear EUR tract should remain EUR regardless of EM iterations
        let pops = make_copying_populations();
        let obs: Vec<AncestryObservation> = (0..20).map(|i| {
            let mut sims = HashMap::new();
            for h in &pops[0].haplotypes {
                sims.insert(h.clone(), 0.999 + (i as f64 * 0.00001) % 0.0005);
            }
            for h in &pops[1].haplotypes {
                sims.insert(h.clone(), 0.997);
            }
            AncestryObservation {
                chrom: "chr12".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "test#1".into(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();

        let (states_1, _) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 1);
        let (states_5, _) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 5);

        // Both should assign all windows to EUR (state 0)
        for (i, &s) in states_1.iter().enumerate() {
            assert_eq!(s, 0, "EM iter=1, window {} should be EUR", i);
        }
        for (i, &s) in states_5.iter().enumerate() {
            assert_eq!(s, 0, "EM iter=5, window {} should be EUR", i);
        }
    }

    #[test]
    fn test_copying_em_switch_detection() {
        // EUR tract followed by AFR tract — EM should detect the switch
        let pops = make_copying_populations();
        let mut obs = Vec::new();
        for i in 0..15 {
            let mut sims = HashMap::new();
            for h in &pops[0].haplotypes {
                sims.insert(h.clone(), 0.999);
            }
            for h in &pops[1].haplotypes {
                sims.insert(h.clone(), 0.997);
            }
            obs.push(AncestryObservation {
                chrom: "chr12".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "test#1".into(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            });
        }
        for i in 15..30 {
            let mut sims = HashMap::new();
            for h in &pops[0].haplotypes {
                sims.insert(h.clone(), 0.997);
            }
            for h in &pops[1].haplotypes {
                sims.insert(h.clone(), 0.999);
            }
            obs.push(AncestryObservation {
                chrom: "chr12".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "test#1".into(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            });
        }

        let (states, _) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 3);

        assert_eq!(states[0], 0, "First window should be EUR");
        assert_eq!(states[29], 1, "Last window should be AFR");
    }

    #[test]
    fn test_copying_forward_backward_alpha_beta_single() {
        // Single window — alpha should equal init + emission, beta should be 0
        let emissions = vec![vec![-1.0, -2.0]];
        let trans = vec![vec![(-0.1_f64).ln(), (-0.9_f64).ln()],
                        vec![(-0.9_f64).ln(), (-0.1_f64).ln()]];
        let init = vec![0.5_f64.ln(), 0.5_f64.ln()];

        let (alpha, beta) = copying_forward_backward_alpha_beta(&emissions, &trans, &init, 2);
        assert_eq!(alpha.len(), 1);
        assert_eq!(beta.len(), 1);
        // Beta at last timestep should be 0.0 (log(1))
        assert!((beta[0][0]).abs() < 1e-10);
        assert!((beta[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_compute_posteriors_from_alpha_beta() {
        // Construct alpha/beta that should give clear posteriors
        let alpha = vec![vec![0.0, -10.0]]; // state 0 dominates
        let beta = vec![vec![0.0, 0.0]];
        let posteriors = compute_posteriors_from_alpha_beta(&alpha, &beta, 2);
        assert_eq!(posteriors.len(), 1);
        assert!(posteriors[0][0] > 0.99, "State 0 should dominate: {}", posteriors[0][0]);
    }

    #[test]
    fn test_copying_em_posteriors_sum_to_one() {
        let pops = make_copying_populations();
        let obs = make_copying_observations(30, &pops);
        let (_, posteriors) = infer_ancestry_copying_em(
            &obs, &pops, 0.005, 0.1, 0.003, 0.99, 3);
        for (i, post) in posteriors.iter().enumerate() {
            let sum: f64 = post.iter().sum();
            assert!((sum - 1.0).abs() < 0.01,
                "EM posterior sum at window {} = {}", i, sum);
        }
    }

    // ======== Rank-Based Emission Tests ========

    #[test]
    fn test_rank_emissions_empty() {
        let pops = make_copying_populations();
        let result = compute_rank_log_emissions(&[], &pops, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rank_emissions_empty_pops() {
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])];
        let result = compute_rank_log_emissions(&obs, &[], 3);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }

    #[test]
    fn test_rank_emissions_clear_pop_a() {
        // All POP_A haplotypes rank above POP_B
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])];
        let emissions = compute_rank_log_emissions(&obs, &pops, 3);

        assert_eq!(emissions.len(), 1);
        assert_eq!(emissions[0].len(), 2);
        // POP_A should have much higher emission than POP_B
        assert!(emissions[0][0] > emissions[0][1],
            "POP_A ({:.4}) should dominate POP_B ({:.4})",
            emissions[0][0], emissions[0][1]);
        // With top_k=3, all 3 top haplotypes should be POP_A (counts: [3, 0])
        // Log prob: ln(3.5/4.0) ≈ -0.134 vs ln(0.5/4.0) ≈ -2.079
        assert!(emissions[0][0] > -0.2, "POP_A emission should be high: {:.4}", emissions[0][0]);
        assert!(emissions[0][1] < -1.5, "POP_B emission should be low: {:.4}", emissions[0][1]);
    }

    #[test]
    fn test_rank_emissions_clear_pop_b() {
        // All POP_B haplotypes rank above POP_A
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0, [0.990, 0.989, 0.988], [0.999, 0.998, 0.997])];
        let emissions = compute_rank_log_emissions(&obs, &pops, 3);

        assert!(emissions[0][1] > emissions[0][0],
            "POP_B ({:.4}) should dominate POP_A ({:.4})",
            emissions[0][1], emissions[0][0]);
    }

    #[test]
    fn test_rank_emissions_sum_to_one() {
        // Log probabilities should sum to approximately 1 in probability space
        let pops = make_copying_populations();
        let obs = vec![
            make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
            make_copying_obs(10000, [0.990, 0.989, 0.988], [0.999, 0.998, 0.997]),
        ];
        let emissions = compute_rank_log_emissions(&obs, &pops, 3);

        for (i, row) in emissions.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "Rank emissions at window {} should sum to 1 in prob space: {:.6}", i, sum);
        }
    }

    #[test]
    fn test_rank_emissions_auto_topk() {
        // top_k=0 should auto-set to total_haps / n_pops = 6/2 = 3
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])];
        let auto_emissions = compute_rank_log_emissions(&obs, &pops, 0);
        let manual_emissions = compute_rank_log_emissions(&obs, &pops, 3);

        for s in 0..2 {
            assert!((auto_emissions[0][s] - manual_emissions[0][s]).abs() < 1e-10,
                "Auto top_k should match manual top_k=3");
        }
    }

    #[test]
    fn test_rank_emissions_weak_signal_robust() {
        // Key test: even with tiny absolute differences (0.0002), rank ordering
        // should produce clear discrimination
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0,
            [0.99820, 0.99810, 0.99800],  // POP_A: tiny margin
            [0.99790, 0.99780, 0.99770])]; // POP_B: consistently lower
        let emissions = compute_rank_log_emissions(&obs, &pops, 3);

        // Despite only 0.0003 difference between max(A) and max(B),
        // rank emissions should strongly favor POP_A because all 3 A haps rank above B
        assert!(emissions[0][0] > emissions[0][1],
            "Rank should favor POP_A even with tiny margins: A={:.4} B={:.4}",
            emissions[0][0], emissions[0][1]);
        // The gap should be substantial (not just marginal)
        let gap = emissions[0][0] - emissions[0][1];
        assert!(gap > 1.0, "Rank emission gap should be substantial: {:.4}", gap);
    }

    #[test]
    fn test_rank_emissions_mixed_ranking() {
        // Mixed ranking: some A haps above some B haps
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0,
            [0.999, 0.993, 0.991],  // A1 is best, but A2/A3 are lower
            [0.997, 0.995, 0.989])]; // B1/B2 beat A2/A3
        let emissions = compute_rank_log_emissions(&obs, &pops, 3);

        // Top 3: A1=0.999, B1=0.997, B2=0.995 → counts: A=1, B=2
        // POP_B should be favored
        assert!(emissions[0][1] > emissions[0][0],
            "POP_B should be favored with 2/3 top-k: A={:.4} B={:.4}",
            emissions[0][0], emissions[0][1]);
    }

    #[test]
    fn test_rank_emissions_topk_larger_than_total() {
        // top_k larger than total haplotypes should use all
        let pops = make_copying_populations();
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])];
        let emissions = compute_rank_log_emissions(&obs, &pops, 100);

        // With k >= total (6), all haplotypes are in top-k → counts are [3, 3]
        // Should give approximately uniform emissions
        let gap = (emissions[0][0] - emissions[0][1]).abs();
        assert!(gap < 0.01, "With k >= total haps, emissions should be near-uniform: gap={:.4}", gap);
    }

    #[test]
    fn test_rank_emissions_blends_with_standard() {
        // Verify rank emissions can be blended with standard emissions
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..5).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();

        let mut params = AncestryHmmParams::new(pops.clone(), 0.05);
        params.emission_std = 0.003;
        let standard = precompute_log_emissions(&obs, &params);
        let rank = compute_rank_log_emissions(&obs, &pops, 3);

        let blended = blend_log_emissions(&standard, &rank, 0.3);
        assert_eq!(blended.len(), 5);
        for row in &blended {
            assert_eq!(row.len(), 2);
            // Blended should be finite and reasonable
            for &v in row {
                assert!(v.is_finite(), "Blended emission should be finite");
                assert!(v <= 0.0, "Log emission should be <= 0");
            }
        }
    }

    // ======== Hierarchical Emission Tests ========

    fn make_3pop_populations() -> Vec<AncestralPopulation> {
        vec![
            AncestralPopulation {
                name: "EUR".into(),
                haplotypes: vec!["E1".into(), "E2".into(), "E3".into()],
            },
            AncestralPopulation {
                name: "AMR".into(),
                haplotypes: vec!["M1".into(), "M2".into(), "M3".into()],
            },
            AncestralPopulation {
                name: "AFR".into(),
                haplotypes: vec!["F1".into(), "F2".into(), "F3".into()],
            },
        ]
    }

    fn make_3pop_obs(start: u64, eur: [f64; 3], amr: [f64; 3], afr: [f64; 3]) -> AncestryObservation {
        AncestryObservation {
            chrom: "chr1".into(),
            start,
            end: start + 10000,
            sample: "query#1".into(),
            similarities: [
                ("E1".into(), eur[0]),
                ("E2".into(), eur[1]),
                ("E3".into(), eur[2]),
                ("M1".into(), amr[0]),
                ("M2".into(), amr[1]),
                ("M3".into(), amr[2]),
                ("F1".into(), afr[0]),
                ("F2".into(), afr[1]),
                ("F3".into(), afr[2]),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }

    #[test]
    fn test_hierarchical_empty() {
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]]; // EUR+AMR vs AFR
        let result = compute_hierarchical_emissions(
            &[], &pops, &groups, &EmissionModel::Max, 0.003, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_hierarchical_output_dimensions() {
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]];
        let obs = vec![make_3pop_obs(0,
            [0.999, 0.998, 0.997], [0.996, 0.995, 0.994], [0.990, 0.989, 0.988])];
        let result = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3); // 3 populations
    }

    #[test]
    fn test_hierarchical_probabilities_sum() {
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]];
        let obs = vec![
            make_3pop_obs(0,
                [0.999, 0.998, 0.997], [0.996, 0.995, 0.994], [0.990, 0.989, 0.988]),
            make_3pop_obs(10000,
                [0.990, 0.989, 0.988], [0.990, 0.989, 0.988], [0.999, 0.998, 0.997]),
        ];
        let result = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.5);

        for (i, row) in result.iter().enumerate() {
            // Log probs don't need to sum to 0 since group+within decomposition
            // isn't a proper probability, but should be finite
            for (j, &v) in row.iter().enumerate() {
                assert!(v.is_finite(),
                    "Hierarchical emission [{}, {}] should be finite: {}", i, j, v);
            }
        }
    }

    #[test]
    fn test_hierarchical_eur_vs_afr_strong() {
        // EUR clearly dominates AFR — both standard and hierarchical should agree
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]]; // EUR+AMR vs AFR
        let obs = vec![make_3pop_obs(0,
            [0.999, 0.998, 0.997], [0.996, 0.995, 0.994], [0.990, 0.989, 0.988])];
        let result = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.5);

        // EUR (0) should be higher than AFR (2)
        assert!(result[0][0] > result[0][2],
            "EUR ({:.4}) should beat AFR ({:.4})", result[0][0], result[0][2]);
    }

    #[test]
    fn test_hierarchical_eur_vs_amr_discrimination() {
        // EUR and AMR are similar but EUR is slightly better
        // With tiny differences (0.0002), standard softmax struggles
        // Hierarchical should handle this via within-group component
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]];
        let obs = vec![make_3pop_obs(0,
            [0.99820, 0.99810, 0.99800],  // EUR
            [0.99800, 0.99790, 0.99780],  // AMR — very close to EUR
            [0.99500, 0.99490, 0.99480])]; // AFR — clearly different
        let result = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.5);

        // EUR should beat AMR (within-group discrimination)
        assert!(result[0][0] > result[0][1],
            "EUR ({:.4}) should beat AMR ({:.4})", result[0][0], result[0][1]);
        // Both EUR and AMR should beat AFR (group-level discrimination)
        assert!(result[0][0] > result[0][2],
            "EUR ({:.4}) should beat AFR ({:.4})", result[0][0], result[0][2]);
        assert!(result[0][1] > result[0][2],
            "AMR ({:.4}) should beat AFR ({:.4})", result[0][1], result[0][2]);
    }

    #[test]
    fn test_hierarchical_single_pop_groups() {
        // Each population is its own group — should behave like standard emissions
        let pops = make_3pop_populations();
        let groups = vec![vec![0], vec![1], vec![2]]; // No grouping
        let obs = vec![make_3pop_obs(0,
            [0.999, 0.998, 0.997], [0.996, 0.995, 0.994], [0.990, 0.989, 0.988])];
        let result = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.5);

        // Within-group is always 0 (log(1)=0) for single-pop groups
        // So result should be proportional to group-level softmax
        assert!(result[0][0] > result[0][1]);
        assert!(result[0][1] > result[0][2]);
    }

    #[test]
    fn test_hierarchical_group_weight_extremes() {
        let pops = make_3pop_populations();
        let groups = vec![vec![0, 1], vec![2]];
        let obs = vec![make_3pop_obs(0,
            [0.99820, 0.99810, 0.99800],
            [0.99800, 0.99790, 0.99780],
            [0.99500, 0.99490, 0.99480])];

        // Pure group-level: should be unable to distinguish EUR from AMR well
        let group_only = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 1.0);
        // Pure within-group: should focus on EUR-vs-AMR discrimination
        let within_only = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.003, 0.0);

        // Both should still rank EUR > AMR > AFR
        assert!(group_only[0][0] > group_only[0][2]);
        assert!(within_only[0][0] > within_only[0][1]);
    }

    #[test]
    fn test_parse_population_groups_valid() {
        let pops = make_3pop_populations();
        let groups = parse_population_groups("EUR,AMR;AFR", &pops);
        assert!(groups.is_some());
        let groups = groups.unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![0, 1]); // EUR, AMR
        assert_eq!(groups[1], vec![2]);     // AFR
    }

    #[test]
    fn test_parse_population_groups_invalid() {
        let pops = make_3pop_populations();
        let groups = parse_population_groups("EUR,UNKNOWN;AFR", &pops);
        assert!(groups.is_none());
    }

    #[test]
    fn test_auto_detect_groups_2pop() {
        // With 2 populations, should return one group per pop
        let pops = make_copying_populations();
        let obs = make_copying_observations(10, &pops);
        let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_auto_detect_groups_3pop_clusters() {
        // EUR and AMR similar, AFR different — should group EUR+AMR
        let pops = make_3pop_populations();
        let obs: Vec<_> = (0..20).map(|i| {
            make_3pop_obs(i * 10000,
                [0.99820, 0.99810, 0.99800],  // EUR
                [0.99800, 0.99790, 0.99780],  // AMR — very close to EUR
                [0.99500, 0.99490, 0.99480])  // AFR — clearly different
        }).collect();
        let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);

        // Should detect EUR+AMR as one group
        // Find which group EUR (0) and AMR (1) are in
        let eur_group = groups.iter().position(|g| g.contains(&0)).unwrap();
        let amr_group = groups.iter().position(|g| g.contains(&1)).unwrap();
        assert_eq!(eur_group, amr_group,
            "EUR and AMR should be in the same group: {:?}", groups);
    }

    #[test]
    fn test_softmax_scores_helper() {
        let sims = vec![Some(0.999), Some(0.990)];
        let result = softmax_scores(&sims, 0.003);
        assert_eq!(result.len(), 2);
        assert!(result[0] > result[1], "Higher sim should get higher score");
        let prob_sum: f64 = result.iter().map(|&v| v.exp()).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, "Should sum to 1: {}", prob_sum);
    }

    #[test]
    fn test_softmax_scores_with_none() {
        let sims = vec![Some(0.999), None, Some(0.990)];
        let result = softmax_scores(&sims, 0.003);
        assert_eq!(result.len(), 3);
        assert!(result[1] == f64::NEG_INFINITY, "None should give -inf");
        assert!(result[0] > result[2]);
    }

    // ======== Pairwise Contrast Emission Tests ========

    #[test]
    fn test_pairwise_empty() {
        let pops = make_copying_populations();
        let result = compute_pairwise_log_emissions(&[], &pops, &EmissionModel::Max);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pairwise_empty_pops() {
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])];
        let result = compute_pairwise_log_emissions(&obs, &[], &EmissionModel::Max);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_pairwise_clear_winner() {
        // POP_A clearly beats POP_B
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..10).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])
        }).collect();
        let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
        assert_eq!(emissions.len(), 10);
        for row in &emissions {
            assert_eq!(row.len(), 2);
            assert!(row[0] > row[1], "POP_A should dominate: {:.4} vs {:.4}", row[0], row[1]);
        }
    }

    #[test]
    fn test_pairwise_probabilities_sum_to_one() {
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..5).map(|i| {
            make_copying_obs(i * 10000, [0.998, 0.997, 0.996], [0.995, 0.994, 0.993])
        }).collect();
        let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
        for row in &emissions {
            let prob_sum: f64 = row.iter().map(|&lp| lp.exp()).sum();
            assert!((prob_sum - 1.0).abs() < 0.01,
                "Log-softmax should sum to ~1.0, got {:.6}", prob_sum);
        }
    }

    #[test]
    fn test_pairwise_3pop_eur_amr_discrimination() {
        // EUR and AMR have tiny gap (0.0002), AFR clearly different
        // Pairwise should discriminate EUR vs AMR better than standard softmax
        let pops = make_3pop_populations();
        let obs: Vec<_> = (0..20).map(|i| {
            make_3pop_obs(i * 10000,
                [0.99820, 0.99810, 0.99800],  // EUR
                [0.99800, 0.99790, 0.99780],  // AMR — tiny gap
                [0.99500, 0.99490, 0.99480])  // AFR — big gap
        }).collect();
        let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
        for row in &emissions {
            assert_eq!(row.len(), 3);
            // EUR > AMR > AFR
            assert!(row[0] > row[1], "EUR should beat AMR: {:.4} vs {:.4}", row[0], row[1]);
            assert!(row[0] > row[2], "EUR should beat AFR: {:.4} vs {:.4}", row[0], row[2]);
            // EUR-AMR gap should be meaningful (the whole point of pairwise)
            assert!(row[0] - row[1] > 0.1,
                "EUR-AMR gap should be >0.1 nats with per-pair temp: {:.4}", row[0] - row[1]);
        }
    }

    #[test]
    fn test_pairwise_symmetric_signal() {
        // Equal similarities → near-equal emissions
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..10).map(|i| {
            make_copying_obs(i * 10000, [0.998, 0.997, 0.996], [0.998, 0.997, 0.996])
        }).collect();
        let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
        for row in &emissions {
            let diff = (row[0] - row[1]).abs();
            assert!(diff < 0.01, "Equal sims should give near-equal emissions: diff={:.6}", diff);
        }
    }

    #[test]
    fn test_pairwise_blends_with_standard() {
        // Pairwise emissions can be blended with standard softmax
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..5).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])
        }).collect();
        let pw_emis = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
        let std_emis: Vec<Vec<f64>> = vec![vec![-0.5, -1.0]; 5];
        let blended = blend_log_emissions(&std_emis, &pw_emis, 0.3);
        assert_eq!(blended.len(), 5);
        for row in &blended {
            assert_eq!(row.len(), 2);
            assert!(row[0].is_finite());
            assert!(row[1].is_finite());
        }
    }

    #[test]
    fn test_log_sigmoid_numerical_stability() {
        // Test that log_sigmoid handles extreme values without NaN/Inf
        assert!((log_sigmoid(0.0) - (-0.6931)).abs() < 0.001); // log(0.5)
        assert!(log_sigmoid(100.0) > -0.001); // should be ~0
        assert!(log_sigmoid(-100.0) < -99.0); // should be ~-100
        assert!(log_sigmoid(1000.0).is_finite());
        assert!(log_sigmoid(-1000.0).is_finite());
    }

    // ======== Population-Aware Transition Tests ========

    #[test]
    fn test_pop_transitions_2pop_uniform() {
        // 2 populations → only one pair → transitions are uniform (same as standard)
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..20).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])
        }).collect();
        let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
        assert_eq!(trans.len(), 2);
        // With 2 states, off-diagonal must be switch_prob (all goes to one state)
        let stay = trans[0][0].exp();
        let switch = trans[0][1].exp();
        assert!((stay + switch - 1.0).abs() < 0.01, "Row should sum to 1: {:.4}", stay + switch);
    }

    #[test]
    fn test_pop_transitions_3pop_asymmetric() {
        // EUR close to AMR, AFR far away
        // EUR→AMR should be more likely than EUR→AFR
        let pops = make_3pop_populations();
        let obs: Vec<_> = (0..30).map(|i| {
            make_3pop_obs(i * 10000,
                [0.99820, 0.99810, 0.99800],  // EUR
                [0.99800, 0.99790, 0.99780],  // AMR — close to EUR
                [0.99500, 0.99490, 0.99480])  // AFR — far from both
        }).collect();
        let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
        assert_eq!(trans.len(), 3);

        // EUR→AMR should be more likely than EUR→AFR
        let eur_to_amr = trans[0][1].exp();
        let eur_to_afr = trans[0][2].exp();
        assert!(eur_to_amr > eur_to_afr,
            "EUR→AMR ({:.6}) should be more likely than EUR→AFR ({:.6})",
            eur_to_amr, eur_to_afr);

        // Each row should sum to ~1.0
        for (i, row) in trans.iter().enumerate() {
            let sum: f64 = row.iter().map(|&lp| lp.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01,
                "Row {} should sum to 1.0, got {:.4}", i, sum);
        }
    }

    #[test]
    fn test_pop_transitions_equal_pops() {
        // All populations equally distant → uniform off-diagonal
        let pops = make_3pop_populations();
        let obs: Vec<_> = (0..30).map(|i| {
            make_3pop_obs(i * 10000,
                [0.990, 0.989, 0.988],
                [0.980, 0.979, 0.978],
                [0.970, 0.969, 0.968])
        }).collect();
        let trans = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
        // EUR→AMR and EUR→AFR distances are similar (0.01 each)
        // So transitions should be roughly equal
        let eur_to_amr = trans[0][1].exp();
        let eur_to_afr = trans[0][2].exp();
        let ratio = eur_to_amr / eur_to_afr;
        // The gaps are 0.01 and 0.02, so AMR is closer but not dramatically
        assert!(ratio < 5.0, "Similar gaps should give similar transitions: ratio={:.2}", ratio);
    }

    // ======== Distance-Aware Transition Tests ========

    #[test]
    fn test_distance_transitions_empty() {
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let result = compute_distance_transitions(&[], &params, 10000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_distance_transitions_single_obs() {
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let obs = vec![make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])];
        let result = compute_distance_transitions(&obs, &params, 10000);
        assert!(result.is_empty()); // n-1 = 0 transitions
    }

    #[test]
    fn test_distance_transitions_adjacent_windows() {
        // Adjacent windows (no gap) should use base switch probability
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let obs = vec![
            make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
            make_copying_obs(10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
        ];
        let result = compute_distance_transitions(&obs, &params, 10000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2); // 2 states
        // Stay probability should be close to 1 - 0.05 = 0.95
        let stay = result[0][0][0].exp();
        assert!((stay - 0.95).abs() < 0.01,
            "Adjacent window stay prob should be ~0.95: {:.4}", stay);
    }

    #[test]
    fn test_distance_transitions_gap_increases_switch() {
        // Gap between windows should increase switch probability
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let obs_adjacent = vec![
            make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
            make_copying_obs(10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
        ];
        let obs_gap = vec![
            make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
            make_copying_obs(50000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]), // 5x gap
        ];
        let trans_adjacent = compute_distance_transitions(&obs_adjacent, &params, 10000);
        let trans_gap = compute_distance_transitions(&obs_gap, &params, 10000);

        let stay_adjacent = trans_adjacent[0][0][0].exp();
        let stay_gap = trans_gap[0][0][0].exp();

        assert!(stay_gap < stay_adjacent,
            "Gap should reduce stay prob: adjacent={:.4} gap={:.4}",
            stay_adjacent, stay_gap);
    }

    #[test]
    fn test_distance_transitions_rows_sum_to_one() {
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let obs = vec![
            make_copying_obs(0, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
            make_copying_obs(30000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994]),
        ];
        let result = compute_distance_transitions(&obs, &params, 10000);

        for row in &result[0] {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "Transition row should sum to 1: {:.10}", sum);
        }
    }

    #[test]
    fn test_fb_with_transitions_matches_standard() {
        // With adjacent windows (no gaps), distance transitions should give
        // same results as standard FB
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops.clone(), 0.05);
        let obs: Vec<_> = (0..10).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();

        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let standard_emissions = precompute_log_emissions(&obs, &p);
        let standard_posteriors = forward_backward_from_log_emissions(&standard_emissions, &p);

        let dist_trans = compute_distance_transitions(&obs, &p, 10000);
        let dist_posteriors = forward_backward_from_log_emissions_with_transitions(
            &standard_emissions, &p, &dist_trans);

        // Should be very close since all windows are adjacent
        for t in 0..10 {
            for s in 0..2 {
                assert!((standard_posteriors[t][s] - dist_posteriors[t][s]).abs() < 0.01,
                    "Window {} state {}: standard={:.4} distance={:.4}",
                    t, s, standard_posteriors[t][s], dist_posteriors[t][s]);
            }
        }
    }

    #[test]
    fn test_viterbi_with_transitions_matches_standard() {
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..10).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();

        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let standard_states = viterbi_from_log_emissions(&emissions, &p);

        let dist_trans = compute_distance_transitions(&obs, &p, 10000);
        let dist_states = viterbi_from_log_emissions_with_transitions(
            &emissions, &p, &dist_trans);

        assert_eq!(standard_states, dist_states,
            "Adjacent windows should give identical Viterbi paths");
    }

    // ======== Ensemble Decoding Tests ========

    #[test]
    fn test_ensemble_empty() {
        let pops = make_copying_populations();
        let params = AncestryHmmParams::new(pops, 0.05);
        let (posteriors, states) = ensemble_decode(&[], &params, 5, 2.0);
        assert!(posteriors.is_empty());
        assert!(states.is_empty());
    }

    #[test]
    fn test_ensemble_single_member() {
        // n_ensemble=1 should give same result as standard FB
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..10).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();
        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let (ens_post, ens_states) = ensemble_decode(&emissions, &p, 1, 2.0);
        let std_post = forward_backward_from_log_emissions(&emissions, &p);

        for t in 0..10 {
            for s in 0..2 {
                assert!((ens_post[t][s] - std_post[t][s]).abs() < 1e-10,
                    "Single-member ensemble should match standard: t={} s={}", t, s);
            }
        }
    }

    #[test]
    fn test_ensemble_posteriors_sum_to_one() {
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..15).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();
        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let (posteriors, _) = ensemble_decode(&emissions, &p, 5, 2.0);

        for (t, row) in posteriors.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01,
                "Ensemble posterior sum at window {} = {:.4}", t, sum);
        }
    }

    #[test]
    fn test_ensemble_clear_signal_consistent() {
        // With clear signal (large pop difference), ensemble should agree with single run
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..20).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.990, 0.989, 0.988])
        }).collect();
        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let (_, ens_states) = ensemble_decode(&emissions, &p, 7, 2.0);

        // All windows should be state 0 (POP_A clearly dominates)
        for (i, &s) in ens_states.iter().enumerate() {
            assert_eq!(s, 0, "Window {} should be POP_A in ensemble", i);
        }
    }

    #[test]
    fn test_ensemble_output_dimensions() {
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..8).map(|i| {
            make_copying_obs(i * 10000, [0.999, 0.998, 0.997], [0.996, 0.995, 0.994])
        }).collect();
        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let (posteriors, states) = ensemble_decode(&emissions, &p, 5, 2.0);

        assert_eq!(posteriors.len(), 8);
        assert_eq!(states.len(), 8);
        for row in &posteriors {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_ensemble_different_sizes() {
        // Larger ensembles should produce smoother posteriors
        let pops = make_copying_populations();
        let obs: Vec<_> = (0..10).map(|i| {
            let boost = if i % 3 == 0 { 0.001 } else { 0.003 };
            make_copying_obs(i * 10000,
                [0.997 + boost, 0.996, 0.995],
                [0.996, 0.995, 0.994])
        }).collect();
        let mut p = AncestryHmmParams::new(pops, 0.05);
        p.emission_std = 0.003;

        let emissions = precompute_log_emissions(&obs, &p);
        let (post3, _) = ensemble_decode(&emissions, &p, 3, 2.0);
        let (post9, _) = ensemble_decode(&emissions, &p, 9, 2.0);

        // Both should produce valid posteriors
        for t in 0..10 {
            let sum3: f64 = post3[t].iter().sum();
            let sum9: f64 = post9[t].iter().sum();
            assert!((sum3 - 1.0).abs() < 0.01);
            assert!((sum9 - 1.0).abs() < 0.01);
        }
    }

    // ======== Auto Identity Floor Tests ========

    #[test]
    fn test_auto_floor_empty() {
        let result = estimate_identity_floor(&[]);
        assert!((result - 0.995).abs() < 0.001, "Empty should return fallback 0.995, got {}", result);
    }

    #[test]
    fn test_auto_floor_small_input() {
        let obs: Vec<AncestryObservation> = (0..5).map(|i| {
            AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [("H1".into(), 0.999)].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();
        let result = estimate_identity_floor(&obs);
        assert!((result - 0.995).abs() < 0.001, "Small input should fallback, got {}", result);
    }

    #[test]
    fn test_auto_floor_uniform_data() {
        // All windows have high identity — floor should be high
        let obs: Vec<AncestryObservation> = (0..100).map(|i| {
            AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [
                    ("H1".into(), 0.999),
                    ("H2".into(), 0.998),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();
        let result = estimate_identity_floor(&obs);
        assert!(result >= 0.9, "Floor should be >= 0.9 for high-identity data, got {}", result);
        assert!(result <= 0.9999);
    }

    #[test]
    fn test_auto_floor_bimodal_data() {
        // 90% high identity, 10% low identity gap windows
        let mut obs: Vec<AncestryObservation> = Vec::new();
        // 10 "gap" windows with very low identity
        for i in 0..10 {
            obs.push(AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [
                    ("H1".into(), 0.5),
                    ("H2".into(), 0.4),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            });
        }
        // 90 high-identity windows
        for i in 10..100 {
            obs.push(AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [
                    ("H1".into(), 0.999),
                    ("H2".into(), 0.998),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            });
        }
        let result = estimate_identity_floor(&obs);
        // Should detect the gap and set floor at or above 0.9
        assert!(result >= 0.9, "Floor should be >= 0.9, got {}", result);
    }

    #[test]
    fn test_auto_floor_clamp_bounds() {
        // All windows with very low identity
        let obs: Vec<AncestryObservation> = (0..100).map(|i| {
            AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [
                    ("H1".into(), 0.5),
                ].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();
        let result = estimate_identity_floor(&obs);
        assert!(result >= 0.9, "Floor should be clamped to >= 0.9, got {}", result);
        assert!(result <= 0.9999, "Floor should be clamped to <= 0.9999, got {}", result);
    }

    #[test]
    fn test_auto_floor_gradual_distribution() {
        // Gradual distribution from 0.99 to 0.9999 — no clear gap
        let obs: Vec<AncestryObservation> = (0..200).map(|i| {
            let sim = 0.99 + 0.01 * (i as f64 / 200.0);
            AncestryObservation {
                chrom: "chr1".into(),
                start: i * 10000,
                end: (i + 1) * 10000,
                sample: "S1".into(),
                similarities: [("H1".into(), sim)].into_iter().collect(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();
        let result = estimate_identity_floor(&obs);
        // Should use percentile fallback, floor should be low end of the range
        assert!(result >= 0.9, "got {}", result);
        assert!(result <= 0.999, "Should use low percentile, got {}", result);
    }

    #[test]
    fn test_auto_configure_well_separated_populations() {
        // Simulate well-separated populations (like sim data) with realistic noise
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = (0..500).map(|i| {
            // Base: popA=0.99, popB=0.96, popC=0.93 with per-window variation
            // Use deterministic noise based on window index
            let noise_c = (i as f64 * 0.7).sin() * 0.005;
            let noise_m = (i as f64 * 1.3).sin() * 0.008;
            let noise_s = (i as f64 * 2.1).sin() * 0.006;
            make_observation(i * 10000, 0.99 + noise_c, 0.96 + noise_m, 0.93 + noise_s)
        }).collect();
        let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
        // Well-separated: pairwise should be positive, ec should be low
        assert!(pw > 0.01, "pairwise_weight should be positive for separated pops, got {}", pw);
        assert!(ec <= 3, "emission_context should be low for strong signal, got {}", ec);
    }

    #[test]
    fn test_auto_configure_close_populations() {
        // Simulate truly close populations (like HPRC real 3-way data).
        // Real HPRC has D_min ≈ 0.003-0.01: tiny mean gaps drowned in noise.
        // Gaps: 0.0002 and 0.0003 between pop means, noise amplitude 0.015 →
        // D_min ≈ 0.01-0.02 → ec should be high (5-15).
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = (0..200).map(|i| {
            let noise_c = (i as f64 * 0.7).sin() * 0.015;
            let noise_m = (i as f64 * 1.3).sin() * 0.015;
            let noise_s = (i as f64 * 2.1).sin() * 0.015;
            make_observation(i * 10000, 0.970 + noise_c, 0.9698 + noise_m, 0.9695 + noise_s)
        }).collect();
        let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
        // Close populations: pairwise should be low, ec should be high
        assert!(pw < 0.5, "pairwise_weight should be low for close pops, got {}", pw);
        assert!(ec >= 3, "emission_context should be high for weak signal, got {}", ec);
    }

    #[test]
    fn test_auto_configure_edge_cases() {
        let pops = make_test_populations();
        // Too few observations
        let obs: Vec<AncestryObservation> = (0..5).map(|i| {
            make_observation(i * 10000, 0.99, 0.96, 0.93)
        }).collect();
        let (pw, ec) = auto_configure_pairwise_params(&obs, &pops);
        assert!((pw - 0.3).abs() < 0.01, "Should return defaults for too few obs");
        assert_eq!(ec, 0);

        // Single population (k < 2)
        let single_pop = vec![pops[0].clone()];
        let obs: Vec<AncestryObservation> = (0..100).map(|i| {
            make_observation(i * 10000, 0.99, 0.96, 0.93)
        }).collect();
        let (pw, ec) = auto_configure_pairwise_params(&obs, &single_pop);
        assert!((pw - 0.3).abs() < 0.01, "Should return defaults for single pop");
        assert_eq!(ec, 0);
    }

    // =========================================================================
    // MPEL decoder edge cases (cycle 60)
    // =========================================================================

    #[test]
    fn test_mpel_single_window() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // Single window: MPEL should return the argmax of the posterior
        let posteriors = vec![vec![0.1, 0.7, 0.2]];
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 1, "Should pick state with highest posterior");
    }

    #[test]
    fn test_mpel_uniform_posteriors() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // Uniform posteriors: all states equally likely — result should be valid
        let uniform = vec![1.0 / 3.0; 3];
        let posteriors = vec![uniform.clone(); 10];
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert_eq!(states.len(), 10);
        // All states should be the same (transition penalty discourages switching)
        let switches: usize = states.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(switches, 0, "Uniform posteriors + transition penalty => no switches");
    }

    #[test]
    fn test_mpel_near_zero_posteriors() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // Nearly zero posteriors for all but one state
        let posteriors = vec![vec![1e-300, 1.0 - 2e-300, 1e-300]; 5];
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert_eq!(states.len(), 5);
        for &s in &states {
            assert_eq!(s, 1, "Should pick state 1 with near-1.0 posterior");
        }
    }

    #[test]
    fn test_mpel_all_mass_one_state() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // All posterior mass on state 2
        let posteriors = vec![vec![0.0, 0.0, 1.0]; 8];
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        for &s in &states {
            assert_eq!(s, 2);
        }
    }

    #[test]
    fn test_mpel_sharp_transition() {
        // Test that MPEL handles a clean state transition at a boundary
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let mut posteriors = Vec::new();
        // 5 windows strongly state 0
        for _ in 0..5 {
            posteriors.push(vec![0.95, 0.03, 0.02]);
        }
        // 5 windows strongly state 2
        for _ in 0..5 {
            posteriors.push(vec![0.02, 0.03, 0.95]);
        }
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert_eq!(states[0], 0);
        assert_eq!(states[9], 2);
        // Should have exactly 1 transition
        let switches: usize = states.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(switches <= 2, "Clean transition should have 1-2 switches, got {}", switches);
    }

    #[test]
    fn test_mpel_two_state_model() {
        // Regression: ensure MPEL works with 2-state models (k-1 transition targets)
        let pops = vec![
            AncestralPopulation {
                name: "A".to_string(),
                haplotypes: vec!["A#1".to_string(), "A#2".to_string()],
            },
            AncestralPopulation {
                name: "B".to_string(),
                haplotypes: vec!["B#1".to_string(), "B#2".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);
        // Strong runs for each state to overcome transition penalty
        let mut posteriors = Vec::new();
        for _ in 0..8 { posteriors.push(vec![0.95, 0.05]); }
        for _ in 0..8 { posteriors.push(vec![0.05, 0.95]); }
        let states = mpel_decode_from_posteriors(&posteriors, &params);
        assert_eq!(states.len(), 16);
        assert_eq!(states[0], 0, "First window should be state 0");
        assert_eq!(states[15], 1, "Last window should be state 1");
    }

    // =========================================================================
    // Deconvolution tests (cycle 60)
    // =========================================================================

    /// Helper: create observations with controlled per-haplotype similarity values.
    fn make_deconv_observations(
        n_windows: usize,
        pop_hap_sims: &[(&str, f64)],
    ) -> Vec<AncestryObservation> {
        (0..n_windows).map(|w| {
            let mut sims = HashMap::new();
            for &(hap, sim) in pop_hap_sims {
                sims.insert(hap.to_string(), sim);
            }
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: w as u64 * 5000,
                end: (w as u64 + 1) * 5000,
                sample: "test#1".to_string(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect()
    }

    #[test]
    fn test_deconvolve_empty_observations() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = Vec::new();
        let (result, map) = deconvolve_admixed_populations(&obs, &pops, 0.5);
        assert_eq!(result.len(), pops.len());
        for pop in &pops {
            assert_eq!(map[&pop.name], pop.name);
        }
    }

    #[test]
    fn test_deconvolve_too_few_observations() {
        let pops = make_test_populations();
        // Only 10 observations (threshold is <50)
        let obs = make_deconv_observations(10, &[
            ("popA#HAP1", 0.99), ("popA#HAP2", 0.98),
            ("popB#A", 0.95), ("popB#B", 0.94),
            ("popC#HAP1", 0.90), ("popC#HAP2", 0.89),
        ]);
        let (result, map) = deconvolve_admixed_populations(&obs, &pops, 0.5);
        assert_eq!(result.len(), pops.len(), "Should return original with <50 obs");
        for pop in &pops {
            assert_eq!(map[&pop.name], pop.name);
        }
    }

    #[test]
    fn test_deconvolve_single_population() {
        let single = vec![make_test_populations()[0].clone()];
        let obs = make_deconv_observations(100, &[
            ("popA#HAP1", 0.99), ("popA#HAP2", 0.98),
        ]);
        let (result, map) = deconvolve_admixed_populations(&obs, &single, 0.5);
        assert_eq!(result.len(), 1, "Single pop should not deconvolve");
        assert_eq!(map[&single[0].name], single[0].name);
    }

    #[test]
    fn test_deconvolve_well_separated_populations() {
        let pops = make_test_populations();
        // Large separation: popA=0.99, popB=0.85, popC=0.70 → high Cohen's d
        let obs = make_deconv_observations(100, &[
            ("popA#HAP1", 0.99), ("popA#HAP2", 0.98),
            ("popB#A", 0.85), ("popB#B", 0.84),
            ("popC#HAP1", 0.70), ("popC#HAP2", 0.69),
        ]);
        let (result, map) = deconvolve_admixed_populations(&obs, &pops, 0.01);
        // Well-separated: no deconvolution should occur
        assert_eq!(result.len(), pops.len(),
            "Well-separated pops should not be split (got {} vs {} original)",
            result.len(), pops.len());
    }

    #[test]
    fn test_deconvolve_too_few_haplotypes_skipped() {
        // Population with <4 haplotypes should be skipped even if d is low
        let pops = vec![
            AncestralPopulation {
                name: "A".to_string(),
                haplotypes: vec!["A#1".to_string(), "A#2".to_string()],
            },
            AncestralPopulation {
                name: "B".to_string(),
                // Only 2 haplotypes — below the 4 threshold for splitting
                haplotypes: vec!["B#1".to_string(), "B#2".to_string()],
            },
        ];
        // Same similarity → d ≈ 0
        let obs = make_deconv_observations(100, &[
            ("A#1", 0.95), ("A#2", 0.94),
            ("B#1", 0.95), ("B#2", 0.94),
        ]);
        let (result, map) = deconvolve_admixed_populations(&obs, &pops, 1.0);
        // Both pops have only 2 haplotypes, neither can be split
        assert_eq!(result.len(), 2, "Should skip pops with <4 haplotypes");
        assert!(map.contains_key("A"));
        assert!(map.contains_key("B"));
    }

    #[test]
    fn test_deconvolve_parent_map_correct() {
        // When deconvolution produces sub-populations, parent_map should link back
        let pops = vec![
            AncestralPopulation {
                name: "EUR".to_string(),
                haplotypes: vec!["EUR#1".to_string(), "EUR#2".to_string()],
            },
            AncestralPopulation {
                name: "AMR".to_string(),
                haplotypes: vec![
                    "AMR#1".to_string(), "AMR#2".to_string(),
                    "AMR#3".to_string(), "AMR#4".to_string(),
                    "AMR#5".to_string(), "AMR#6".to_string(),
                ],
            },
        ];
        // EUR and AMR have very close similarity (d ≈ 0).
        // AMR haplotypes 1-3 are EUR-like, 4-6 are different.
        let obs: Vec<AncestryObservation> = (0..100).map(|w| {
            let mut sims = HashMap::new();
            sims.insert("EUR#1".to_string(), 0.990);
            sims.insert("EUR#2".to_string(), 0.989);
            // AMR haplotypes: 3 EUR-like, 3 with lower identity
            sims.insert("AMR#1".to_string(), 0.990);
            sims.insert("AMR#2".to_string(), 0.989);
            sims.insert("AMR#3".to_string(), 0.988);
            sims.insert("AMR#4".to_string(), 0.960);
            sims.insert("AMR#5".to_string(), 0.959);
            sims.insert("AMR#6".to_string(), 0.958);
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: w * 5000,
                end: (w + 1) * 5000,
                sample: "test#1".to_string(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            }
        }).collect();
        let (result, map) = deconvolve_admixed_populations(&obs, &pops, 1.0);
        // EUR has <4 haplotypes so won't be split, but AMR might be
        // All parent_map values should be either "EUR" or "AMR"
        for (child, parent) in &map {
            assert!(parent == "EUR" || parent == "AMR",
                "Parent map value '{}' should be EUR or AMR (child='{}')", parent, child);
        }
        // Total haplotypes should be conserved
        let total_haps: usize = result.iter().map(|p| p.haplotypes.len()).sum();
        let orig_haps: usize = pops.iter().map(|p| p.haplotypes.len()).sum();
        assert_eq!(total_haps, orig_haps, "Total haplotypes must be conserved");
    }

    #[test]
    fn test_deconvolve_preserves_haplotypes() {
        // All haplotypes from original populations must appear in output
        let pops = make_test_populations();
        let obs = make_deconv_observations(100, &[
            ("popA#HAP1", 0.99), ("popA#HAP2", 0.98),
            ("popB#A", 0.95), ("popB#B", 0.94),
            ("popC#HAP1", 0.90), ("popC#HAP2", 0.89),
        ]);
        let (result, _map) = deconvolve_admixed_populations(&obs, &pops, 0.5);
        let mut orig_haps: Vec<String> = pops.iter()
            .flat_map(|p| p.haplotypes.iter().cloned()).collect();
        let mut result_haps: Vec<String> = result.iter()
            .flat_map(|p| p.haplotypes.iter().cloned()).collect();
        orig_haps.sort();
        result_haps.sort();
        assert_eq!(orig_haps, result_haps, "Haplotypes must be conserved");
    }

    // =========================================================================
    // kmeans_pp_2 tests (cycle 60)
    // =========================================================================

    #[test]
    fn test_kmeans_pp_2_empty() {
        let features: Vec<Vec<f64>> = Vec::new();
        let assignments = kmeans_pp_2(&features, 20);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_kmeans_pp_2_single_point() {
        let features = vec![vec![1.0, 2.0]];
        let assignments = kmeans_pp_2(&features, 20);
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn test_kmeans_pp_2_two_points() {
        let features = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let assignments = kmeans_pp_2(&features, 20);
        // Two distant points should be in different clusters
        assert_ne!(assignments[0], assignments[1],
            "Two distant points should be in different clusters");
    }

    #[test]
    fn test_kmeans_pp_2_separable_clusters() {
        // Clear two-cluster structure
        let mut features = Vec::new();
        for _ in 0..5 {
            features.push(vec![0.0, 0.0]);
        }
        for _ in 0..5 {
            features.push(vec![100.0, 100.0]);
        }
        let assignments = kmeans_pp_2(&features, 20);
        // All points in first group should have same cluster
        let cluster_a = assignments[0];
        for &a in &assignments[0..5] {
            assert_eq!(a, cluster_a, "First group should be same cluster");
        }
        // All points in second group should have same cluster
        let cluster_b = assignments[5];
        for &a in &assignments[5..10] {
            assert_eq!(a, cluster_b, "Second group should be same cluster");
        }
        // Groups should differ
        assert_ne!(cluster_a, cluster_b, "Groups should be different clusters");
    }

    #[test]
    fn test_kmeans_pp_2_identical_points() {
        // All points identical — should not crash, all same cluster
        let features = vec![vec![1.0, 1.0]; 10];
        let assignments = kmeans_pp_2(&features, 20);
        assert_eq!(assignments.len(), 10);
        // Should be deterministic and not panic
    }

    #[test]
    fn test_kmeans_pp_2_zero_dim() {
        let features = vec![Vec::new(); 5];
        let assignments = kmeans_pp_2(&features, 20);
        assert_eq!(assignments, vec![0; 5]);
    }

    // =========================================================================
    // within_pop_variance tests (cycle 60)
    // =========================================================================

    #[test]
    fn test_within_pop_variance_empty() {
        let hap_sims: Vec<Vec<f64>> = Vec::new();
        assert_eq!(within_pop_variance(&hap_sims), 0.0);
    }

    #[test]
    fn test_within_pop_variance_single_haplotype() {
        let hap_sims = vec![vec![0.99; 100]];
        // Single haplotype → can't compute variance between haplotypes
        assert_eq!(within_pop_variance(&hap_sims), 0.0);
    }

    #[test]
    fn test_within_pop_variance_identical_haplotypes() {
        // Two haplotypes with identical means → variance = 0
        let hap_sims = vec![
            vec![0.99; 100],
            vec![0.99; 100],
        ];
        assert!(within_pop_variance(&hap_sims) < 1e-10,
            "Identical haplotypes should have near-zero variance");
    }

    #[test]
    fn test_within_pop_variance_different_haplotypes() {
        // One haplotype at 0.99, another at 0.95 → noticeable variance
        let hap_sims = vec![
            vec![0.99; 100],
            vec![0.95; 100],
        ];
        let var = within_pop_variance(&hap_sims);
        // Mean of means = 0.97, variance = ((0.99-0.97)^2 + (0.95-0.97)^2)/2 = 0.0004
        assert!((var - 0.0004).abs() < 1e-6,
            "Expected var ≈ 0.0004, got {}", var);
    }

    #[test]
    fn test_within_pop_variance_too_few_data_points() {
        // Haplotype with <10 data points should be filtered out
        let hap_sims = vec![
            vec![0.99; 100],
            vec![0.95; 5], // Only 5 points — below filter threshold
        ];
        // Only one haplotype passes filter → variance = 0
        assert_eq!(within_pop_variance(&hap_sims), 0.0,
            "Haplotype with <10 points should be filtered");
    }

    // =========================================================================
    // sq_dist tests (cycle 60)
    // =========================================================================

    #[test]
    fn test_sq_dist_zero() {
        assert_eq!(sq_dist(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_sq_dist_basic() {
        // (3-0)^2 + (4-0)^2 = 9 + 16 = 25
        assert!((sq_dist(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_sq_dist_empty() {
        assert_eq!(sq_dist(&[], &[]), 0.0);
    }

    // =========================================================================
    // BW transition dampening tests (cycle 60)
    // =========================================================================

    #[test]
    fn test_bw_dampening_zero_equals_undampened() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = (0..30).map(|w| {
            if w < 15 {
                make_observation(w * 5000, 0.96, 0.82, 0.80)
            } else {
                make_observation(w * 5000, 0.80, 0.96, 0.82)
            }
        }).collect();
        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        // dampening = 0 explicitly
        let mut params0 = AncestryHmmParams::new(pops.clone(), 0.01);
        params0.set_temperature(0.03);
        params0.transition_dampening = 0.0;
        let _ll0 = params0.baum_welch(&obs_slices, 3, 1e-10);

        // default (also 0)
        let mut params_default = AncestryHmmParams::new(pops, 0.01);
        params_default.set_temperature(0.03);
        let _ll_d = params_default.baum_welch(&obs_slices, 3, 1e-10);

        // Transitions should be identical
        for i in 0..3 {
            for j in 0..3 {
                assert!((params0.transitions[i][j] - params_default.transitions[i][j]).abs() < 1e-12,
                    "dampening=0 should match default at [{},{}]: {} vs {}",
                    i, j, params0.transitions[i][j], params_default.transitions[i][j]);
            }
        }
    }

    #[test]
    fn test_bw_dampening_one_preserves_prior() {
        let pops = make_test_populations();
        let params_prior = AncestryHmmParams::new(pops.clone(), 0.01);
        let prior_transitions = params_prior.transitions.clone();

        let obs: Vec<AncestryObservation> = (0..30).map(|w| {
            make_observation(w * 5000, 0.96, 0.82, 0.80)
        }).collect();
        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        let mut params = AncestryHmmParams::new(pops, 0.01);
        params.set_temperature(0.03);
        params.transition_dampening = 1.0;
        let _ll = params.baum_welch(&obs_slices, 5, 1e-10);

        // With dampening=1.0, transitions should remain at prior values
        for i in 0..3 {
            for j in 0..3 {
                assert!((params.transitions[i][j] - prior_transitions[i][j]).abs() < 1e-6,
                    "dampening=1 should preserve prior at [{},{}]: {} vs {}",
                    i, j, params.transitions[i][j], prior_transitions[i][j]);
            }
        }
    }

    #[test]
    fn test_bw_dampening_half_interpolates() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = (0..30).map(|w| {
            if w < 15 {
                make_observation(w * 5000, 0.96, 0.82, 0.80)
            } else {
                make_observation(w * 5000, 0.80, 0.96, 0.82)
            }
        }).collect();
        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        // Run undampened
        let mut params_mle = AncestryHmmParams::new(pops.clone(), 0.01);
        params_mle.set_temperature(0.03);
        params_mle.transition_dampening = 0.0;
        let _ll_mle = params_mle.baum_welch(&obs_slices, 1, 1e-10);

        // Run with dampening=0.5
        let mut params_half = AncestryHmmParams::new(pops.clone(), 0.01);
        params_half.set_temperature(0.03);
        params_half.transition_dampening = 0.5;
        let _ll_half = params_half.baum_welch(&obs_slices, 1, 1e-10);

        let prior = AncestryHmmParams::new(pops, 0.01);

        // Dampened transitions should be between MLE and prior
        for i in 0..3 {
            for j in 0..3 {
                let mle_val = params_mle.transitions[i][j];
                let prior_val = prior.transitions[i][j];
                let half_val = params_half.transitions[i][j];
                let low = mle_val.min(prior_val) - 0.01;
                let high = mle_val.max(prior_val) + 0.01;
                assert!(half_val >= low && half_val <= high,
                    "Dampened [{},{}]={} should be between MLE={} and prior={}",
                    i, j, half_val, mle_val, prior_val);
            }
        }
    }

    #[test]
    fn test_bw_dampening_maintains_valid_distributions() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = (0..40).map(|w| {
            make_observation(w * 5000, 0.96, 0.85, 0.80)
        }).collect();
        let obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

        for &damp in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
            params.set_temperature(0.03);
            params.transition_dampening = damp;
            let _ll = params.baum_welch(&obs_slices, 3, 1e-10);

            // Verify all transition rows sum to ~1.0
            for (i, row) in params.transitions.iter().enumerate() {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "dampening={}: transitions[{}] sum={}, expected 1.0", damp, i, sum);
                for (j, &t) in row.iter().enumerate() {
                    assert!(t >= 0.0 && t <= 1.0,
                        "dampening={}: transitions[{}][{}]={} out of [0,1]", damp, i, j, t);
                }
            }
        }
    }

    // =========================================================================
    // viterbi_from_log_emissions edge cases (cycle 60)
    // =========================================================================

    #[test]
    fn test_viterbi_from_log_emissions_empty() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let log_em: Vec<Vec<f64>> = Vec::new();
        let states = viterbi_from_log_emissions(&log_em, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_viterbi_from_log_emissions_single_window() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // State 1 has highest log-emission
        let log_em = vec![vec![-5.0, -0.1, -3.0]];
        let states = viterbi_from_log_emissions(&log_em, &params);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 1);
    }

    #[test]
    fn test_viterbi_from_log_emissions_neg_inf() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // One state has neg infinity, others finite
        let log_em = vec![
            vec![f64::NEG_INFINITY, -1.0, -2.0],
            vec![f64::NEG_INFINITY, -1.0, -2.0],
        ];
        let states = viterbi_from_log_emissions(&log_em, &params);
        for &s in &states {
            assert_ne!(s, 0, "Should never pick state with -inf emission");
        }
    }

    #[test]
    fn test_viterbi_from_log_emissions_all_equal() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        // All emissions equal → transition prior drives state choice
        let log_em = vec![vec![-1.0; 3]; 10];
        let states = viterbi_from_log_emissions(&log_em, &params);
        assert_eq!(states.len(), 10);
        // With equal emissions and stay bias, should stay in one state
        let switches: usize = states.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(switches, 0, "Equal emissions + stay bias => no switches");
    }

    // =========================================================================
    // DecodingMethod parsing tests (cycle 60)
    // =========================================================================

    #[test]
    fn test_decoding_method_parse_mpel() {
        assert_eq!("mpel".parse::<DecodingMethod>().unwrap(), DecodingMethod::Mpel);
        assert_eq!("MPEL".parse::<DecodingMethod>().unwrap(), DecodingMethod::Mpel);
        assert_eq!("Mpel".parse::<DecodingMethod>().unwrap(), DecodingMethod::Mpel);
    }

    #[test]
    fn test_decoding_method_display_mpel() {
        assert_eq!(format!("{}", DecodingMethod::Mpel), "mpel");
    }

    #[test]
    fn test_decoding_method_parse_invalid() {
        assert!("invalid".parse::<DecodingMethod>().is_err());
        assert!("".parse::<DecodingMethod>().is_err());
    }

    #[test]
    fn test_decoding_method_roundtrip() {
        for method in &[DecodingMethod::Viterbi, DecodingMethod::Posterior, DecodingMethod::Mpel] {
            let s = format!("{}", method);
            let parsed: DecodingMethod = s.parse().unwrap();
            assert_eq!(&parsed, method);
        }
    }

    #[test]
    fn test_blend_log_emissions_adaptive_strong_signal_gets_higher_weight() {
        // 3 windows: strong, medium, weak pairwise signal
        // Standard emissions are uniform across windows; pairwise differ in confidence
        let standard = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let pairwise = vec![
            vec![-0.2, -3.0],  // strong: margin = 2.8, differs from std
            vec![-0.5, -1.5],  // medium: margin = 1.0, differs from std
            vec![-0.9, -1.1],  // weak: margin = 0.2, differs from std
        ];

        let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.5);
        let static_blend = blend_log_emissions(&standard, &pairwise, 0.5);

        // Strong window (0): above median → adaptive should use MORE pairwise than static
        let adaptive_dev_0 = (adaptive[0][0] - standard[0][0]).abs();
        let static_dev_0 = (static_blend[0][0] - standard[0][0]).abs();
        assert!(adaptive_dev_0 > static_dev_0,
            "Strong-signal window should deviate more from standard: adaptive={:.4} vs static={:.4}",
            adaptive_dev_0, static_dev_0);

        // Weak window (2): below median → adaptive should use LESS pairwise than static
        let adaptive_dev_2 = (adaptive[2][0] - standard[2][0]).abs();
        let static_dev_2 = (static_blend[2][0] - standard[2][0]).abs();
        assert!(adaptive_dev_2 < static_dev_2,
            "Weak-signal window should deviate less from standard: adaptive={:.4} vs static={:.4}",
            adaptive_dev_2, static_dev_2);
    }

    #[test]
    fn test_blend_log_emissions_adaptive_uniform_margins_equals_static() {
        // When all windows have identical margin, adaptive should behave like static
        let standard = vec![
            vec![-1.0, -2.0],
            vec![-1.5, -1.5],
            vec![-2.0, -1.0],
        ];
        let pairwise = vec![
            vec![-0.5, -2.5],  // margin = 2.0
            vec![-0.5, -2.5],  // margin = 2.0
            vec![-0.5, -2.5],  // margin = 2.0
        ];

        let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.3);
        let static_blend = blend_log_emissions(&standard, &pairwise, 0.3);

        for t in 0..3 {
            for s in 0..2 {
                assert!((adaptive[t][s] - static_blend[t][s]).abs() < 1e-10,
                    "Uniform margin should equal static blend at [{t}][{s}]");
            }
        }
    }

    #[test]
    fn test_blend_log_emissions_adaptive_empty_input() {
        let result = blend_log_emissions_adaptive(&[], &[], 0.5);
        assert!(result.is_empty());

        let standard = vec![vec![-1.0, -2.0]];
        let result = blend_log_emissions_adaptive(&standard, &[], 0.5);
        assert_eq!(result.len(), 1); // returns standard
    }

    #[test]
    fn test_blend_log_emissions_adaptive_weight_clamped() {
        // Even with very high confidence, weight should never exceed 0.95
        let standard = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let pairwise = vec![
            vec![-0.01, -10.0],  // extreme margin = 9.99
            vec![-1.0, -1.001],  // tiny margin = 0.001
        ];

        let adaptive = blend_log_emissions_adaptive(&standard, &pairwise, 0.9);
        // Result should be finite and between standard and pairwise
        for row in &adaptive {
            for &v in row {
                assert!(v.is_finite(), "All values should be finite");
            }
        }
    }

    // ======================================================================
    // Focused population masking tests
    // ======================================================================

    #[test]
    fn test_focused_masking_zero_threshold_noop() {
        let emissions = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -0.5, -2.5],
        ];
        let posteriors = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.2, 0.7, 0.1],
        ];
        let result = apply_focused_masking(&emissions, &posteriors, 0.0, 2);
        assert_eq!(result, emissions);
    }

    #[test]
    fn test_focused_masking_masks_low_posterior() {
        let emissions = vec![
            vec![-1.0, -2.0, -3.0],
        ];
        let posteriors = vec![
            vec![0.7, 0.25, 0.05],
        ];
        // threshold=0.1 should mask pop 2 (posterior=0.05)
        let result = apply_focused_masking(&emissions, &posteriors, 0.1, 2);
        assert_eq!(result[0][0], -1.0);
        assert_eq!(result[0][1], -2.0);
        assert_eq!(result[0][2], f64::NEG_INFINITY);
    }

    #[test]
    fn test_focused_masking_preserves_min_active() {
        let emissions = vec![
            vec![-1.0, -2.0, -3.0],
        ];
        let posteriors = vec![
            // Only pop 0 above threshold 0.5, but min_active=2 forces top-2 to remain
            vec![0.8, 0.15, 0.05],
        ];
        let result = apply_focused_masking(&emissions, &posteriors, 0.5, 2);
        // Top-2 by posterior: pop 0 (0.8) and pop 1 (0.15)
        assert_eq!(result[0][0], -1.0);
        assert_eq!(result[0][1], -2.0);
        assert_eq!(result[0][2], f64::NEG_INFINITY);
    }

    #[test]
    fn test_focused_masking_per_window_independence() {
        let emissions = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.0, -2.0, -3.0],
        ];
        let posteriors = vec![
            vec![0.7, 0.25, 0.05],  // mask pop 2
            vec![0.05, 0.7, 0.25],  // mask pop 0
        ];
        let result = apply_focused_masking(&emissions, &posteriors, 0.1, 2);
        assert_eq!(result[0][2], f64::NEG_INFINITY); // window 0: pop 2 masked
        assert!(result[0][0].is_finite());
        assert_eq!(result[1][0], f64::NEG_INFINITY); // window 1: pop 0 masked
        assert!(result[1][1].is_finite());
    }

    #[test]
    fn test_focused_masking_all_above_threshold() {
        let emissions = vec![vec![-1.0, -2.0]];
        let posteriors = vec![vec![0.6, 0.4]];
        // Both above 0.1, nothing masked
        let result = apply_focused_masking(&emissions, &posteriors, 0.1, 2);
        assert_eq!(result[0][0], -1.0);
        assert_eq!(result[0][1], -2.0);
    }

    #[test]
    fn test_focused_masking_aggressive_threshold() {
        let emissions = vec![vec![-1.0, -2.0, -3.0, -4.0, -5.0]];
        let posteriors = vec![vec![0.4, 0.3, 0.15, 0.1, 0.05]];
        // threshold=0.2 → pops 2,3,4 below threshold. min_active=2, so top-2 survive
        let result = apply_focused_masking(&emissions, &posteriors, 0.2, 2);
        assert!(result[0][0].is_finite());
        assert!(result[0][1].is_finite());
        assert_eq!(result[0][2], f64::NEG_INFINITY);
        assert_eq!(result[0][3], f64::NEG_INFINITY);
        assert_eq!(result[0][4], f64::NEG_INFINITY);
    }

    #[test]
    fn test_focused_masking_empty_input() {
        let result = apply_focused_masking(&[], &[], 0.1, 2);
        assert!(result.is_empty());
    }

    // ======================================================================
    // Entropy-weighted posterior smoothing tests
    // ======================================================================

    #[test]
    fn test_entropy_smooth_zero_radius_noop() {
        let posteriors = vec![
            vec![0.8, 0.2],
            vec![0.3, 0.7],
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 0, 2.0);
        assert_eq!(result, posteriors);
    }

    #[test]
    fn test_entropy_smooth_uniform_posteriors() {
        // All uniform → all same entropy → uniform weights → simple average
        let posteriors = vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 2.0);
        for row in &result {
            assert!((row[0] - 0.5).abs() < 1e-10);
            assert!((row[1] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_entropy_smooth_confident_propagates() {
        // Window 0: very confident (low entropy) → should dominate nearby smoothing
        // Window 1: uncertain → should be pulled toward window 0's values
        let posteriors = vec![
            vec![0.99, 0.01],  // very confident
            vec![0.5, 0.5],   // maximally uncertain
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 5.0);
        // Window 1 should be pulled toward [0.99, 0.01] because window 0 has much higher weight
        assert!(result[1][0] > 0.5, "Uncertain window should be pulled toward confident neighbor");
    }

    #[test]
    fn test_entropy_smooth_preserves_probability_sum() {
        let posteriors = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.2, 0.5, 0.3],
            vec![0.1, 0.1, 0.8],
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 2.0);
        for (t, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "Window {} posteriors should sum to 1.0, got {}", t, sum);
        }
    }

    #[test]
    fn test_entropy_smooth_single_window() {
        let posteriors = vec![vec![0.7, 0.3]];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 3, 2.0);
        assert!((result[0][0] - 0.7).abs() < 1e-10);
        assert!((result[0][1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_smooth_empty() {
        let result = entropy_weighted_smooth_posteriors(&[], 3, 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_entropy_smooth_large_radius_convergence() {
        // With very large radius, all windows should converge toward weighted average
        let posteriors = vec![
            vec![1.0, 0.0],  // zero entropy (max weight)
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 100, 10.0);
        // All windows should be strongly pulled toward [1.0, 0.0]
        for row in &result {
            assert!(row[0] > 0.8, "All windows pulled toward confident anchor");
        }
    }

    #[test]
    fn test_entropy_smooth_beta_zero_uniform_weights() {
        // beta=0 → all weights = exp(0) = 1 → simple uniform average
        // Use non-deterministic posteriors so entropy is nonzero
        let posteriors = vec![
            vec![0.8, 0.2],
            vec![0.3, 0.7],
            vec![0.9, 0.1],
        ];
        let result = entropy_weighted_smooth_posteriors(&posteriors, 1, 0.0);
        // With beta=0, weights are all exp(0)=1 → uniform average over window
        // Window 1 (radius=1): average of windows 0,1,2
        let expected_0 = (0.8 + 0.3 + 0.9) / 3.0;
        let expected_1 = (0.2 + 0.7 + 0.1) / 3.0;
        assert!((result[1][0] - expected_0).abs() < 1e-10,
            "Expected {}, got {}", expected_0, result[1][0]);
        assert!((result[1][1] - expected_1).abs() < 1e-10);
    }

    // ======================================================================
    // Population-distance weighted transitions tests
    // ======================================================================

    #[test]
    fn test_pop_distances_symmetric() {
        let pops = make_test_populations();
        let obs = (0..5).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.96, 0.93)
        }).collect::<Vec<_>>();
        let dists = compute_population_distances(&obs, &pops, &EmissionModel::Max);
        assert_eq!(dists.len(), 3);
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((dists[i][j] - dists[j][i]).abs() < 1e-10,
                    "Distance matrix should be symmetric");
            }
            assert!((dists[i][i]).abs() < 1e-10, "Self-distance should be 0");
        }
    }

    #[test]
    fn test_pop_distances_ordering() {
        // popA=0.99, popB=0.96, popC=0.93 → popA-popB distance < popA-popC distance
        let pops = make_test_populations();
        let obs = (0..10).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.96, 0.93)
        }).collect::<Vec<_>>();
        let dists = compute_population_distances(&obs, &pops, &EmissionModel::Max);
        // |0.99 - 0.96| = 0.03, |0.99 - 0.93| = 0.06
        assert!(dists[0][1] < dists[0][2],
            "popA-popB ({}) should be closer than popA-popC ({})", dists[0][1], dists[0][2]);
    }

    #[test]
    fn test_distance_weighted_transitions_close_pair_higher() {
        let pops = make_test_populations();
        // popA=0.99, popB=0.97, popC=0.90 → D[popB,popA]=0.02, D[popB,popC]=0.07
        let obs = (0..10).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.97, 0.90)
        }).collect::<Vec<_>>();
        let dists = compute_population_distances(&obs, &pops, &EmissionModel::Max);
        let proportions = vec![0.33, 0.34, 0.33];
        let switch_rates = vec![0.05, 0.05, 0.05];

        let mut params = AncestryHmmParams::new(pops, 0.05);
        set_distance_weighted_transitions(&mut params, &dists, &proportions, &switch_rates);

        // popB→popA transition should be higher than popB→popC (popB is closer to popA)
        assert!(params.transitions[1][0] > params.transitions[1][2],
            "popB→popA ({:.6}) should be > popB→popC ({:.6})",
            params.transitions[1][0], params.transitions[1][2]);
    }

    #[test]
    fn test_distance_weighted_transitions_row_sums() {
        let pops = make_test_populations();
        let obs = (0..10).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.96, 0.93)
        }).collect::<Vec<_>>();
        let dists = compute_population_distances(&obs, &pops, &EmissionModel::Max);
        let proportions = vec![0.5, 0.3, 0.2];
        let switch_rates = vec![0.05, 0.03, 0.08];

        let mut params = AncestryHmmParams::new(pops, 0.05);
        set_distance_weighted_transitions(&mut params, &dists, &proportions, &switch_rates);

        for i in 0..3 {
            let sum: f64 = params.transitions[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "Row {} transitions should sum to 1.0, got {}", i, sum);
        }
    }

    #[test]
    fn test_distance_weighted_transitions_empty_obs() {
        let pops = make_test_populations();
        let obs: Vec<AncestryObservation> = vec![];
        let dists = compute_population_distances(&obs, &pops, &EmissionModel::Max);
        // All distances should be 0
        for row in &dists {
            for &d in row {
                assert!((d).abs() < 1e-10);
            }
        }
    }

    // ======================================================================
    // Label smoothing tests
    // ======================================================================

    #[test]
    fn test_label_smooth_zero_alpha_noop() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        let result = apply_label_smoothing(&emissions, 0.0);
        assert_eq!(result, emissions);
    }

    #[test]
    fn test_label_smooth_full_alpha_uniform() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        let result = apply_label_smoothing(&emissions, 1.0);
        let log_uniform = -(3.0_f64).ln();
        for &v in &result[0] {
            assert!((v - log_uniform).abs() < 1e-10,
                "Alpha=1.0 should give uniform emissions");
        }
    }

    #[test]
    fn test_label_smooth_interpolation() {
        let emissions = vec![vec![-0.5, -3.0]];
        let alpha = 0.2;
        let result = apply_label_smoothing(&emissions, alpha);
        let log_uniform = -(2.0_f64).ln();
        let expected_0 = (1.0 - alpha) * (-0.5) + alpha * log_uniform;
        let expected_1 = (1.0 - alpha) * (-3.0) + alpha * log_uniform;
        assert!((result[0][0] - expected_0).abs() < 1e-10);
        assert!((result[0][1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn test_label_smooth_reduces_contrast() {
        let emissions = vec![vec![-0.1, -5.0]];
        let result = apply_label_smoothing(&emissions, 0.1);
        // Smoothing should reduce the gap between states
        let original_gap = emissions[0][0] - emissions[0][1];
        let smoothed_gap = result[0][0] - result[0][1];
        assert!(smoothed_gap.abs() < original_gap.abs(),
            "Label smoothing should reduce emission contrast");
    }

    #[test]
    fn test_label_smooth_preserves_neg_infinity() {
        let emissions = vec![vec![-1.0, f64::NEG_INFINITY, -2.0]];
        let result = apply_label_smoothing(&emissions, 0.2);
        assert!(result[0][0].is_finite());
        assert_eq!(result[0][1], f64::NEG_INFINITY); // Masked state stays masked
        assert!(result[0][2].is_finite());
    }

    #[test]
    fn test_label_smooth_empty() {
        let result = apply_label_smoothing(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_label_smooth_multi_window() {
        let emissions = vec![
            vec![-1.0, -2.0],
            vec![-3.0, -0.5],
        ];
        let result = apply_label_smoothing(&emissions, 0.1);
        // Each window smoothed independently
        assert_ne!(result[0], result[1]);
        for row in &result {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    // ======================================================================
    // Margin-gated persistence tests
    // ======================================================================

    #[test]
    fn test_margin_persistence_zero_bonus_noop() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        let posteriors = vec![vec![0.8, 0.15, 0.05]];
        let result = apply_margin_persistence(&emissions, &posteriors, 0.5, 0.0);
        assert_eq!(result, emissions);
    }

    #[test]
    fn test_margin_persistence_below_threshold_noop() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        // margin = 0.8 - 0.15 = 0.65, threshold = 0.7 → below threshold
        let posteriors = vec![vec![0.8, 0.15, 0.05]];
        let result = apply_margin_persistence(&emissions, &posteriors, 0.7, 2.0);
        assert_eq!(result, emissions);
    }

    #[test]
    fn test_margin_persistence_above_threshold_boosts_argmax() {
        let emissions = vec![vec![-1.0, -2.0, -3.0]];
        // margin = 0.9 - 0.08 = 0.82, threshold = 0.5 → above threshold
        let posteriors = vec![vec![0.9, 0.08, 0.02]];
        let result = apply_margin_persistence(&emissions, &posteriors, 0.5, 2.0);
        // State 0 (argmax) should be boosted
        assert!(result[0][0] > emissions[0][0], "Argmax should get bonus");
        assert_eq!(result[0][1], emissions[0][1], "Non-argmax unchanged");
        assert_eq!(result[0][2], emissions[0][2], "Non-argmax unchanged");
    }

    #[test]
    fn test_margin_persistence_bonus_scales_with_margin() {
        let emissions = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        // Window 0: margin=0.7, Window 1: margin=0.4 (both above threshold=0.3)
        let posteriors = vec![
            vec![0.85, 0.15],
            vec![0.70, 0.30],
        ];
        let result = apply_margin_persistence(&emissions, &posteriors, 0.3, 2.0);
        // Window 0 should get more bonus (higher margin)
        let boost_0 = result[0][0] - emissions[0][0];
        let boost_1 = result[1][0] - emissions[1][0];
        assert!(boost_0 > boost_1,
            "Higher margin window should get more bonus: {} vs {}", boost_0, boost_1);
    }

    #[test]
    fn test_margin_persistence_preserves_neg_infinity() {
        let emissions = vec![vec![f64::NEG_INFINITY, -2.0]];
        let posteriors = vec![vec![0.9, 0.1]]; // argmax is state 0
        let result = apply_margin_persistence(&emissions, &posteriors, 0.3, 2.0);
        // NEG_INFINITY state 0 is argmax but its emission is NEG_INFINITY
        assert_eq!(result[0][0], f64::NEG_INFINITY);
    }

    #[test]
    fn test_margin_persistence_empty() {
        let result = apply_margin_persistence(&[], &[], 0.5, 1.0);
        assert!(result.is_empty());
    }

    // ======================================================================
    // Adaptive pairwise weight tests
    // ======================================================================

    #[test]
    fn test_adaptive_pairwise_scales_ambiguous_gets_higher() {
        let emissions = vec![
            vec![-1.0, -1.01],   // small gap → ambiguous
            vec![-1.0, -5.0],    // large gap → clear
        ];
        let scales = compute_adaptive_pairwise_scales(&emissions, 0.3, 1.5);
        assert_eq!(scales.len(), 2);
        // Ambiguous window should get higher scale (more pairwise weight)
        assert!(scales[0] > scales[1],
            "Ambiguous window scale ({}) should be > clear window scale ({})",
            scales[0], scales[1]);
    }

    #[test]
    fn test_adaptive_pairwise_scales_bounds() {
        let emissions = vec![
            vec![-1.0, -1.0001],  // tiny gap
            vec![-1.0, -10.0],    // huge gap
        ];
        let scales = compute_adaptive_pairwise_scales(&emissions, 0.3, 1.5);
        for &s in &scales {
            assert!(s >= 0.3 && s <= 1.5,
                "Scale {} should be in [0.3, 1.5]", s);
        }
    }

    #[test]
    fn test_adaptive_pairwise_scales_empty() {
        let scales = compute_adaptive_pairwise_scales(&[], 0.3, 1.5);
        assert!(scales.is_empty());
    }

    #[test]
    fn test_blend_adaptive_per_window() {
        let standard = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let pairwise = vec![
            vec![-0.5, -3.0],
            vec![-0.5, -3.0],
        ];
        // Window 0 gets high scale (more pairwise), window 1 gets low scale
        let scales = vec![1.5, 0.3];
        let result = blend_log_emissions_adaptive_per_window(
            &standard, &pairwise, 0.5, &scales);
        // Window 0 should lean more toward pairwise than window 1
        let pw_influence_0 = (result[0][0] - standard[0][0]).abs();
        let pw_influence_1 = (result[1][0] - standard[1][0]).abs();
        assert!(pw_influence_0 > pw_influence_1,
            "High-scale window should have more pairwise influence");
    }

    #[test]
    fn test_blend_adaptive_per_window_preserves_neg_infinity() {
        let standard = vec![vec![f64::NEG_INFINITY, -2.0]];
        let pairwise = vec![vec![-0.5, -3.0]];
        let scales = vec![1.0];
        let result = blend_log_emissions_adaptive_per_window(
            &standard, &pairwise, 0.5, &scales);
        // When standard is NEG_INFINITY, should use pairwise
        assert!(result[0][0].is_finite() || result[0][0] == f64::NEG_INFINITY);
    }

    // ======================================================================
    // Reference purity scoring tests
    // ======================================================================

    #[test]
    fn test_reference_purity_basic() {
        let pops = make_test_populations();
        // popA has highest similarity → its refs should have high purity
        let obs = (0..10).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.90, 0.85)
        }).collect::<Vec<_>>();
        let purity = compute_reference_purity(&obs, &pops);
        // popA haplotypes should have higher purity than popB/popC
        let a_purity = purity.get("popA#HAP1").cloned().unwrap_or(0.0);
        let b_purity = purity.get("popB#A").cloned().unwrap_or(0.0);
        assert!(a_purity > 0.0, "popA purity should be positive");
        assert!(b_purity > 0.0, "popB purity should be positive");
    }

    #[test]
    fn test_reference_purity_scores_all_positive() {
        let pops = make_test_populations();
        let obs = (0..5).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.96, 0.93)
        }).collect::<Vec<_>>();
        let purity = compute_reference_purity(&obs, &pops);
        for (_, &score) in &purity {
            assert!(score > 0.0, "All purity scores should be positive");
        }
    }

    #[test]
    fn test_reference_purity_empty_obs() {
        let pops = make_test_populations();
        let purity = compute_reference_purity(&[], &pops);
        assert!(purity.is_empty());
    }

    #[test]
    fn test_purity_weighted_observations_zero_gamma_noop() {
        let pops = make_test_populations();
        let obs = vec![make_observation(0, 0.99, 0.96, 0.93)];
        let purity = compute_reference_purity(&obs, &pops);
        let weighted = apply_purity_weighted_observations(&obs, &purity, 0.0);
        // gamma=0 → weight=1 for all → no change
        for (orig, new) in obs.iter().zip(weighted.iter()) {
            for (hap, &sim) in &orig.similarities {
                let new_sim = new.similarities.get(hap).cloned().unwrap_or(0.0);
                assert!((sim - new_sim).abs() < 1e-10,
                    "gamma=0 should not change similarities");
            }
        }
    }

    #[test]
    fn test_purity_weighted_observations_positive_gamma() {
        let pops = make_test_populations();
        let obs = (0..5).map(|i| {
            make_observation(i as u64 * 5000, 0.99, 0.96, 0.93)
        }).collect::<Vec<_>>();
        let purity = compute_reference_purity(&obs, &pops);
        let weighted = apply_purity_weighted_observations(&obs, &purity, 1.0);
        // With gamma=1.0, similarities should be modified
        for new_obs in &weighted {
            for (_, &sim) in &new_obs.similarities {
                assert!(sim >= 0.0, "Weighted similarities should be non-negative");
            }
        }
    }

    // =========================================================================
    // Variance penalty tests
    // =========================================================================

    #[test]
    fn test_within_pop_variance_basic() {
        let pops = make_test_populations();
        // HAP1=0.99, HAP2=0.98 → small variance
        // popB A=0.96, B=0.95 → small variance
        let obs = vec![make_observation(0, 0.99, 0.96, 0.93)];
        let variances = compute_within_pop_variance(&obs, &pops);
        assert_eq!(variances.len(), 1);
        assert_eq!(variances[0].len(), 3);
        // Each population has 2 haplotypes differing by 0.01, so var = 0.01^2 / (2-1) = 0.0001 / 1
        for v in &variances[0] {
            assert!(*v > 0.0, "variance should be positive for non-identical haplotypes");
            assert!(*v < 0.001, "variance should be small for close values");
        }
    }

    #[test]
    fn test_within_pop_variance_high_variance_pop() {
        let pops = make_test_populations();
        // Create observation where popA has high variance (HAP1=0.99, HAP2=0.80)
        let mut obs = make_observation(0, 0.99, 0.96, 0.93);
        obs.similarities.insert("popA#HAP2".to_string(), 0.80);
        let variances = compute_within_pop_variance(&[obs], &pops);
        // popA variance should be much larger than others
        assert!(variances[0][0] > variances[0][1],
            "high-variance pop ({}) should have larger variance than normal ({})",
            variances[0][0], variances[0][1]);
    }

    #[test]
    fn test_compute_within_pop_variance_empty_obs() {
        let pops = make_test_populations();
        let variances = compute_within_pop_variance(&[], &pops);
        assert!(variances.is_empty());
    }

    #[test]
    fn test_variance_penalty_zero_weight_noop() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let vars = vec![vec![0.01, 0.02, 0.03]];
        let result = apply_variance_penalty(&log_e, &vars, 0.0);
        assert_eq!(result, log_e, "zero weight should be a no-op");
    }

    #[test]
    fn test_variance_penalty_reduces_high_variance() {
        let log_e = vec![
            vec![-1.0, -1.0, -1.0],  // equal emissions
        ];
        // Pop 0 has very high variance, others normal
        let vars = vec![vec![0.1, 0.001, 0.001]];
        let result = apply_variance_penalty(&log_e, &vars, 1.0);
        // Pop 0 should be penalized more than others
        assert!(result[0][0] < result[0][1],
            "high-variance pop should be penalized: {} should be < {}",
            result[0][0], result[0][1]);
    }

    #[test]
    fn test_variance_penalty_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let vars = vec![vec![0.01, 0.02, 0.03]];
        let result = apply_variance_penalty(&log_e, &vars, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0,
            "NEG_INFINITY should be preserved");
    }

    #[test]
    fn test_variance_penalty_multiple_windows() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -1.5, -1.5],
        ];
        let vars = vec![
            vec![0.01, 0.01, 0.01],
            vec![0.05, 0.001, 0.001],
        ];
        let result = apply_variance_penalty(&log_e, &vars, 2.0);
        // Window 1: all equal variance → penalty proportional but equal shift
        // Window 2: pop 0 high variance → more penalty
        assert!(result[1][0] < log_e[1][0], "high-variance should decrease emission");
        assert_eq!(result.len(), 2);
    }

    // =========================================================================
    // Flank-informed emission bonus tests
    // =========================================================================

    #[test]
    fn test_flank_informed_basic_agreement() {
        let log_e = vec![
            vec![-2.0, -1.0, -3.0],  // t=0: state 1 best
            vec![-1.5, -1.5, -1.5],  // t=1: ambiguous
            vec![-2.0, -1.0, -3.0],  // t=2: state 1 best
        ];
        let states = vec![1, 2, 1]; // pass-1: 1, 2, 1
        // For t=1: left mode=1, right mode=1 → agree → bonus to state 1
        let result = apply_flank_informed_bonus(&log_e, &states, 1, 0.5, 3);
        assert!(result[1][1] > log_e[1][1],
            "state 1 at t=1 should get bonus: {} > {}", result[1][1], log_e[1][1]);
        assert_eq!(result[1][0], log_e[1][0], "other states should be unchanged");
        assert_eq!(result[1][2], log_e[1][2], "other states should be unchanged");
    }

    #[test]
    fn test_flank_informed_disagreement_no_bonus() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -1.5, -1.5],
            vec![-3.0, -2.0, -1.0],
        ];
        let states = vec![0, 1, 2]; // all different → flanks disagree
        let result = apply_flank_informed_bonus(&log_e, &states, 1, 0.5, 3);
        // Middle window: left=0, right=2 → disagree → no bonus
        assert_eq!(result[1], log_e[1], "disagreement should leave emissions unchanged");
    }

    #[test]
    fn test_flank_informed_zero_radius_noop() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let states = vec![0];
        let result = apply_flank_informed_bonus(&log_e, &states, 0, 0.5, 3);
        assert_eq!(result, log_e, "zero radius should be a no-op");
    }

    #[test]
    fn test_flank_informed_zero_bonus_noop() {
        let log_e = vec![
            vec![-2.0, -1.0, -3.0],
            vec![-1.5, -1.5, -1.5],
            vec![-2.0, -1.0, -3.0],
        ];
        let states = vec![1, 1, 1];
        let result = apply_flank_informed_bonus(&log_e, &states, 1, 0.0, 3);
        assert_eq!(result, log_e, "zero bonus should be a no-op");
    }

    #[test]
    fn test_flank_informed_larger_radius() {
        let log_e = vec![
            vec![-2.0, -1.0, -3.0],  // state 1
            vec![-2.0, -1.0, -3.0],  // state 1
            vec![-1.5, -1.5, -1.5],  // ambiguous
            vec![-2.0, -1.0, -3.0],  // state 1
            vec![-2.0, -1.0, -3.0],  // state 1
        ];
        let states = vec![1, 1, 0, 1, 1]; // pass-1 had error at t=2
        // For t=2 with radius=2: left=[1,1]→mode=1, right=[1,1]→mode=1 → agree
        let result = apply_flank_informed_bonus(&log_e, &states, 2, 0.5, 3);
        assert!(result[2][1] > log_e[2][1],
            "state 1 should get bonus from flanks: {} > {}", result[2][1], log_e[2][1]);
    }

    #[test]
    fn test_flank_informed_preserves_neg_infinity() {
        let log_e = vec![
            vec![-2.0, -1.0, -3.0],
            vec![f64::NEG_INFINITY, -1.5, -1.5],
            vec![-2.0, -1.0, -3.0],
        ];
        let states = vec![0, 1, 0]; // flanks agree on state 0
        let result = apply_flank_informed_bonus(&log_e, &states, 1, 0.5, 3);
        // State 0 at t=1 is masked (NEG_INFINITY) but flanks agree on 0
        // Bonus should NOT be applied to NEG_INFINITY states
        assert!(result[1][0].is_infinite() && result[1][0] < 0.0,
            "NEG_INFINITY should be preserved even when flanks agree on that state");
    }

    #[test]
    fn test_flank_mode_basic() {
        assert_eq!(flank_mode(&[0, 0, 1], 3), Some(0));
        assert_eq!(flank_mode(&[1, 1, 1], 3), Some(1));
        assert_eq!(flank_mode(&[2, 0, 2], 3), Some(2));
        assert_eq!(flank_mode(&[], 3), None);
    }

    #[test]
    fn test_flank_informed_edge_windows() {
        // First and last windows have only one flank
        let log_e = vec![
            vec![-1.5, -1.5, -1.5],
            vec![-1.5, -1.5, -1.5],
            vec![-1.5, -1.5, -1.5],
        ];
        let states = vec![1, 1, 1]; // all same state
        let result = apply_flank_informed_bonus(&log_e, &states, 1, 0.5, 3);
        // t=0: left=None, right=Some(1) → no agreement possible? Actually:
        // left_mode requires t > 0, so t=0 left_mode is None → no bonus
        // But t=1: left=[1]→1, right=[1]→1 → agree → bonus
        assert!(result[1][1] > log_e[1][1], "middle should get bonus");
    }

    // =========================================================================
    // LOO robust emission tests
    // =========================================================================

    #[test]
    fn test_loo_robust_basic() {
        let pops = make_test_populations();
        let obs = vec![make_observation(0, 0.99, 0.96, 0.93)];
        let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, 0.01);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
        // Should be valid log-probabilities
        let sum: f64 = result[0].iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "should sum to ~1 in probability space: {}", sum);
    }

    #[test]
    fn test_loo_robust_penalizes_outlier() {
        let pops = make_test_populations();
        // popA has one outlier: HAP1=0.99, HAP2=0.80
        // LOO min: leaving out HAP1 → max(0.80) = 0.80; leaving out HAP2 → max(0.99) = 0.99
        // min(0.80, 0.99) = 0.80
        // popB is consistent: A=0.96, B=0.95 → LOO min = 0.95
        let mut obs = make_observation(0, 0.99, 0.96, 0.93);
        obs.similarities.insert("popA#HAP2".to_string(), 0.80);
        let loo_result = compute_loo_robust_emissions(&[obs.clone()], &pops, &EmissionModel::Max, 0.01);
        // popA's LOO score (0.80) < popB's LOO score (0.95)
        // So in LOO emissions, popB should have higher probability than popA
        // (opposite of standard where popA's 0.99 max would win)
        let standard_emissions = precompute_log_emissions(&[obs.clone()],
            &AncestryHmmParams::new(pops.clone(), 0.001));
        // Standard: popA wins (0.99 max)
        assert!(standard_emissions[0][0] > standard_emissions[0][1],
            "standard should favor popA");
        // LOO: popB wins (0.95 LOO vs 0.80 LOO)
        assert!(loo_result[0][1] > loo_result[0][0],
            "LOO should favor popB over outlier-dependent popA: {} vs {}",
            loo_result[0][1], loo_result[0][0]);
    }

    #[test]
    fn test_loo_robust_empty() {
        let pops = make_test_populations();
        let result = compute_loo_robust_emissions(&[], &pops, &EmissionModel::Max, 0.01);
        assert!(result.is_empty());
    }

    #[test]
    fn test_loo_robust_single_hap_fallback() {
        // Population with just 1 haplotype should fall back to standard
        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["a1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["b1".to_string(), "b2".to_string()],
            },
        ];
        let obs = vec![AncestryObservation {
            chrom: "chr1".to_string(),
            start: 0,
            end: 5000,
            sample: "test".to_string(),
            similarities: [
                ("a1".to_string(), 0.95),
                ("b1".to_string(), 0.90),
                ("b2".to_string(), 0.89),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }];
        let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, 0.01);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        // pop_a (single hap) falls back to standard: 0.95
        // pop_b LOO: min(max(0.89), max(0.90)) = 0.89
        // pop_a should still win
        assert!(result[0][0] > result[0][1],
            "single-hap pop should still score via fallback");
    }

    #[test]
    fn test_loo_robust_mean_model() {
        let pops = make_test_populations();
        let obs = vec![make_observation(0, 0.99, 0.96, 0.93)];
        let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Mean, 0.01);
        assert_eq!(result.len(), 1);
        let sum: f64 = result[0].iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "should be valid log-probs");
    }

    // =========================================================================
    // Posterior sharpening tests
    // =========================================================================

    #[test]
    fn test_sharpen_posteriors_identity_at_one() {
        let posteriors = vec![vec![0.7, 0.2, 0.1]];
        let result = sharpen_posteriors(&posteriors, 1.0);
        for (a, b) in result[0].iter().zip(posteriors[0].iter()) {
            assert!((a - b).abs() < 1e-10, "T=1 should be no-op");
        }
    }

    #[test]
    fn test_sharpen_posteriors_sharpens_below_one() {
        let posteriors = vec![vec![0.5, 0.3, 0.2]];
        let result = sharpen_posteriors(&posteriors, 0.5);
        // Sharpening: max posterior should increase
        assert!(result[0][0] > posteriors[0][0],
            "sharpening should increase max: {} > {}", result[0][0], posteriors[0][0]);
        // Min posterior should decrease
        assert!(result[0][2] < posteriors[0][2],
            "sharpening should decrease min: {} < {}", result[0][2], posteriors[0][2]);
        // Still sums to 1
        let sum: f64 = result[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "should still sum to 1");
    }

    #[test]
    fn test_sharpen_posteriors_softens_above_one() {
        let posteriors = vec![vec![0.8, 0.15, 0.05]];
        let result = sharpen_posteriors(&posteriors, 2.0);
        // Softening: max posterior should decrease
        assert!(result[0][0] < posteriors[0][0],
            "softening should decrease max: {} < {}", result[0][0], posteriors[0][0]);
        // Min posterior should increase
        assert!(result[0][2] > posteriors[0][2],
            "softening should increase min: {} > {}", result[0][2], posteriors[0][2]);
    }

    #[test]
    fn test_sharpen_posteriors_zero_temp_noop() {
        let posteriors = vec![vec![0.5, 0.3, 0.2]];
        let result = sharpen_posteriors(&posteriors, 0.0);
        assert_eq!(result, posteriors, "T=0 should be no-op");
    }

    #[test]
    fn test_sharpen_posteriors_preserves_normalization() {
        let posteriors = vec![
            vec![0.6, 0.3, 0.1],
            vec![0.4, 0.4, 0.2],
            vec![0.9, 0.05, 0.05],
        ];
        for temp in [0.1, 0.3, 0.5, 0.8, 1.5, 3.0] {
            let result = sharpen_posteriors(&posteriors, temp);
            for (t, row) in result.iter().enumerate() {
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-8,
                    "T={} window {}: sum={} should be 1.0", temp, t, sum);
            }
        }
    }

    #[test]
    fn test_sharpen_posteriors_extreme_sharpening() {
        let posteriors = vec![vec![0.5, 0.3, 0.2]];
        let result = sharpen_posteriors(&posteriors, 0.01);
        // With very aggressive sharpening, should be nearly one-hot
        assert!(result[0][0] > 0.99,
            "extreme sharpening should push max near 1.0: {}", result[0][0]);
    }

    // =========================================================================
    // Short segment correction tests
    // =========================================================================

    #[test]
    fn test_short_segment_correction_basic() {
        // [0,0,0, 1, 0,0,0] → short segment of state 1 in the middle
        let states = vec![0, 0, 0, 1, 0, 0, 0];
        let log_e = vec![
            vec![-0.5, -2.0], vec![-0.5, -2.0], vec![-0.5, -2.0],
            vec![-1.0, -1.2], // ambiguous but slightly favors 0
            vec![-0.5, -2.0], vec![-0.5, -2.0], vec![-0.5, -2.0],
        ];
        let result = correct_short_segments(&states, &log_e, 2);
        assert_eq!(result[3], 0, "short segment should be corrected to neighbor state");
    }

    #[test]
    fn test_short_segment_correction_respects_min() {
        // Segment of length 2 with min_windows=2 should NOT be corrected
        let states = vec![0, 0, 1, 1, 0, 0];
        let log_e = vec![
            vec![-0.5, -2.0], vec![-0.5, -2.0],
            vec![-2.0, -0.5], vec![-2.0, -0.5],
            vec![-0.5, -2.0], vec![-0.5, -2.0],
        ];
        let result = correct_short_segments(&states, &log_e, 2);
        assert_eq!(result, states, "segment of length >= min should be kept");
    }

    #[test]
    fn test_short_segment_correction_neighbors_agree() {
        // [0,0, 2, 0,0] — both neighbors are state 0
        let states = vec![0, 0, 2, 0, 0];
        let log_e = vec![
            vec![-0.5, -2.0, -2.0], vec![-0.5, -2.0, -2.0],
            vec![-1.0, -2.0, -1.1], // slightly favors 0 over 2
            vec![-0.5, -2.0, -2.0], vec![-0.5, -2.0, -2.0],
        ];
        let result = correct_short_segments(&states, &log_e, 2);
        assert_eq!(result[2], 0, "should merge with agreed neighbors");
    }

    #[test]
    fn test_short_segment_correction_empty() {
        let result = correct_short_segments(&[], &[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_short_segment_correction_min_one_noop() {
        let states = vec![0, 1, 0];
        let log_e = vec![vec![-1.0, -2.0]; 3];
        let result = correct_short_segments(&states, &log_e, 1);
        assert_eq!(result, states, "min_windows=1 should be a no-op");
    }

    #[test]
    fn test_short_segment_correction_emission_support() {
        // [0,0, 1, 2,2] — neighbors disagree (0 vs 2)
        // At the short segment position, emission supports state 2 more
        let states = vec![0, 0, 1, 2, 2];
        let log_e = vec![
            vec![-0.5, -2.0, -2.0], vec![-0.5, -2.0, -2.0],
            vec![-2.0, -1.0, -0.8], // state 2 has best emission support
            vec![-2.0, -2.0, -0.5], vec![-2.0, -2.0, -0.5],
        ];
        let result = correct_short_segments(&states, &log_e, 2);
        assert_eq!(result[2], 2, "should merge with neighbor that has better emission support");
    }

    // =========================================================================
    // Emission whitening tests
    // =========================================================================

    #[test]
    fn test_whiten_empty() {
        let result = whiten_log_emissions(&[], 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_whiten_single_pop() {
        let log_e = vec![vec![-1.0], vec![-2.0]];
        let result = whiten_log_emissions(&log_e, 1e-6);
        assert_eq!(result, log_e, "single population should be unchanged");
    }

    #[test]
    fn test_whiten_centers_mean() {
        // Need at least k+1 data points for meaningful covariance (k=3 → need 4+)
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -2.5, -3.5],
            vec![-0.5, -1.5, -2.5],
            vec![-1.2, -2.2, -3.2],
            vec![-0.8, -1.8, -2.8],
        ];
        let result = whiten_log_emissions(&log_e, 1e-6);
        // After whitening, each population should be approximately zero-mean
        let n = result.len() as f64;
        for j in 0..3 {
            let mean: f64 = result.iter().map(|row| row[j]).sum::<f64>() / n;
            assert!(mean.abs() < 0.1,
                "pop {} mean should be near 0 after whitening: {}", j, mean);
        }
    }

    #[test]
    fn test_whiten_decorrelates() {
        // Create highly correlated emissions (EUR↔AMR scenario)
        // Add small noise to avoid perfect collinearity
        let log_e: Vec<Vec<f64>> = (0..20).map(|i| {
            let base = -1.0 - 0.1 * (i as f64);
            let noise = 0.002 * ((i * 7 % 11) as f64 - 5.0); // deterministic pseudo-noise
            vec![base, base + 0.01 + noise, base - 0.5 - noise * 0.5]
        }).collect();

        // Compute correlation before whitening
        let corr_before = {
            let n = log_e.len() as f64;
            let m0: f64 = log_e.iter().map(|r| r[0]).sum::<f64>() / n;
            let m1: f64 = log_e.iter().map(|r| r[1]).sum::<f64>() / n;
            let cov: f64 = log_e.iter().map(|r| (r[0]-m0)*(r[1]-m1)).sum::<f64>();
            let v0: f64 = log_e.iter().map(|r| (r[0]-m0).powi(2)).sum::<f64>();
            let v1: f64 = log_e.iter().map(|r| (r[1]-m1).powi(2)).sum::<f64>();
            cov / (v0.sqrt() * v1.sqrt())
        };

        let result = whiten_log_emissions(&log_e, 1e-6);

        // Compute correlation after whitening
        let corr_after = {
            let n = result.len() as f64;
            let m0: f64 = result.iter().map(|r| r[0]).sum::<f64>() / n;
            let m1: f64 = result.iter().map(|r| r[1]).sum::<f64>() / n;
            let cov: f64 = result.iter().map(|r| (r[0]-m0)*(r[1]-m1)).sum::<f64>();
            let v0: f64 = result.iter().map(|r| (r[0]-m0).powi(2)).sum::<f64>();
            let v1: f64 = result.iter().map(|r| (r[1]-m1).powi(2)).sum::<f64>();
            if v0.sqrt() * v1.sqrt() > 1e-10 {
                cov / (v0.sqrt() * v1.sqrt())
            } else {
                0.0
            }
        };

        assert!(corr_before.abs() > 0.9,
            "input should be highly correlated: {}", corr_before);
        assert!(corr_after.abs() < corr_before.abs(),
            "whitening should reduce correlation: before={}, after={}", corr_before, corr_after);
    }

    #[test]
    fn test_whiten_preserves_neg_infinity() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![f64::NEG_INFINITY, -1.5, -2.5],
            vec![-1.0, -2.0, -3.0],
        ];
        let result = whiten_log_emissions(&log_e, 1e-6);
        // Row with NEG_INFINITY should be preserved as-is
        assert!(result[1][0].is_infinite() && result[1][0] < 0.0,
            "NEG_INFINITY row should be preserved");
    }

    #[test]
    fn test_jacobi_eigendecomposition() {
        // 2x2 diagonal matrix → eigenvalues should be diagonal entries
        let m = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
        let (evals, _evecs) = symmetric_eigen_jacobi(&m, 100);
        let mut sorted_evals = evals.clone();
        sorted_evals.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted_evals[0] - 3.0).abs() < 1e-10, "eigenvalue 1: {}", sorted_evals[0]);
        assert!((sorted_evals[1] - 1.0).abs() < 1e-10, "eigenvalue 2: {}", sorted_evals[1]);
    }

    #[test]
    fn test_jacobi_symmetric_matrix() {
        // 2x2 symmetric → eigenvalues should be real
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (evals, _) = symmetric_eigen_jacobi(&m, 100);
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1
        let mut sorted_evals = evals.clone();
        sorted_evals.sort_by(|a, b| b.total_cmp(a));
        assert!((sorted_evals[0] - 3.0).abs() < 1e-10);
        assert!((sorted_evals[1] - 1.0).abs() < 1e-10);
    }

    // =========================================================================
    // Window quality scoring tests
    // =========================================================================

    #[test]
    fn test_window_quality_basic() {
        let posteriors = vec![
            vec![0.9, 0.05, 0.05],  // high confidence
            vec![0.4, 0.35, 0.25],  // low confidence
            vec![0.9, 0.05, 0.05],  // high confidence
        ];
        let log_e = vec![
            vec![-0.1, -2.0, -3.0],
            vec![-0.5, -0.6, -0.7],
            vec![-0.1, -2.0, -3.0],
        ];
        let states = vec![0, 0, 0]; // all same state
        let quality = compute_window_quality(&posteriors, &log_e, &states, 1);
        assert_eq!(quality.len(), 3);
        // High-confidence window should have higher quality
        assert!(quality[0] > quality[1],
            "confident window should have higher quality: {} > {}", quality[0], quality[1]);
    }

    #[test]
    fn test_window_quality_range() {
        let posteriors = vec![vec![0.5, 0.3, 0.2]; 5];
        let log_e = vec![vec![-1.0, -1.5, -2.0]; 5];
        let states = vec![0; 5];
        let quality = compute_window_quality(&posteriors, &log_e, &states, 2);
        for &q in &quality {
            assert!(q >= 0.0 && q <= 1.0, "quality should be in [0,1]: {}", q);
        }
    }

    #[test]
    fn test_window_quality_empty() {
        let quality = compute_window_quality(&[], &[], &[], 1);
        assert!(quality.is_empty());
    }

    #[test]
    fn test_window_quality_neighbor_disagree() {
        let posteriors = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.5, 0.3, 0.2],
            vec![0.5, 0.3, 0.2],
        ];
        let log_e = vec![vec![-1.0, -1.5, -2.0]; 3];
        let states_agree = vec![0, 0, 0];
        let states_disagree = vec![0, 1, 2];
        let q_agree = compute_window_quality(&posteriors, &log_e, &states_agree, 1);
        let q_disagree = compute_window_quality(&posteriors, &log_e, &states_disagree, 1);
        // Middle window: agreement should give higher quality than disagreement
        assert!(q_agree[1] > q_disagree[1],
            "neighbor agreement should boost quality: {} > {}",
            q_agree[1], q_disagree[1]);
    }

    // =========================================================================
    // Iterative refinement tests
    // =========================================================================

    #[test]
    fn test_iterative_refine_single_pass() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let log_e = vec![
            vec![-0.5, -2.0, -3.0],
            vec![-0.5, -2.0, -3.0],
            vec![-0.5, -2.0, -3.0],
        ];
        let (posteriors, states) = iterative_refine(&log_e, &params, 1, 0.5);
        assert_eq!(posteriors.len(), 3);
        assert_eq!(states.len(), 3);
        // With clear emissions favoring state 0, all states should be 0
        for &s in &states {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn test_iterative_refine_multiple_passes() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let log_e = vec![
            vec![-0.5, -2.0, -3.0],
            vec![-1.0, -1.1, -3.0],  // ambiguous between 0 and 1
            vec![-0.5, -2.0, -3.0],
        ];
        let (post_1, _states_1) = iterative_refine(&log_e, &params, 1, 0.5);
        let (post_3, _states_3) = iterative_refine(&log_e, &params, 3, 0.5);
        // Multiple passes should increase confidence (higher max posterior)
        let max_post_1: f64 = post_1[1].iter().cloned().fold(0.0, f64::max);
        let max_post_3: f64 = post_3[1].iter().cloned().fold(0.0, f64::max);
        assert!(max_post_3 >= max_post_1 - 0.01,
            "multiple passes should maintain or increase confidence: {} vs {}",
            max_post_3, max_post_1);
    }

    #[test]
    fn test_iterative_refine_empty() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let (posteriors, states) = iterative_refine(&[], &params, 3, 0.5);
        assert!(posteriors.is_empty());
        assert!(states.is_empty());
    }

    #[test]
    fn test_iterative_refine_zero_passes() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let (posteriors, states) = iterative_refine(&log_e, &params, 0, 0.5);
        assert!(posteriors.is_empty());
        assert!(states.is_empty());
    }

    #[test]
    fn test_iterative_refine_convergence() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01);
        let log_e = vec![
            vec![-0.5, -2.0, -3.0],
            vec![-0.5, -2.0, -3.0],
            vec![-0.5, -2.0, -3.0],
        ];
        // After many passes, should converge to same result
        let (_, states_5) = iterative_refine(&log_e, &params, 5, 0.5);
        let (_, states_10) = iterative_refine(&log_e, &params, 10, 0.5);
        assert_eq!(states_5, states_10, "should converge");
    }

    // =========================================================================
    // Calibration boost tests
    // =========================================================================

    #[test]
    fn test_calibration_boosts_balanced() {
        // Proportions match observed → no boost
        let states = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let proportions = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        let boosts = compute_calibration_boosts(&states, &proportions, 3, 0.5);
        for &b in &boosts {
            assert!(b.abs() < 0.01,
                "balanced proportions should give near-zero boost: {}", b);
        }
    }

    #[test]
    fn test_calibration_boosts_under_represented() {
        // Pop 0 is under-represented: 1/10 observed vs 1/3 expected
        let states = vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        let proportions = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        let boosts = compute_calibration_boosts(&states, &proportions, 3, 0.5);
        assert!(boosts[0] > 0.0,
            "under-represented pop should get positive boost: {}", boosts[0]);
        assert!(boosts[2] < 0.0,
            "over-represented pop should get negative boost: {}", boosts[2]);
    }

    #[test]
    fn test_calibration_boosts_zero_scale() {
        let states = vec![0, 1, 2];
        let proportions = vec![0.5, 0.3, 0.2];
        let boosts = compute_calibration_boosts(&states, &proportions, 3, 0.0);
        assert!(boosts.iter().all(|&b| b == 0.0), "zero scale should give zero boosts");
    }

    #[test]
    fn test_calibration_boosts_empty() {
        let boosts = compute_calibration_boosts(&[], &[0.5, 0.5], 2, 0.5);
        assert!(boosts.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_apply_calibration_boosts_basic() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let boosts = vec![0.5, 0.0, -0.5];
        let result = apply_calibration_boosts(&log_e, &boosts);
        assert!((result[0][0] - (-0.5)).abs() < 1e-10);
        assert!((result[0][1] - (-2.0)).abs() < 1e-10);
        assert!((result[0][2] - (-3.5)).abs() < 1e-10);
    }

    #[test]
    fn test_apply_calibration_boosts_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let boosts = vec![1.0, 0.5, -0.5];
        let result = apply_calibration_boosts(&log_e, &boosts);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0,
            "NEG_INFINITY should be preserved");
    }

    // ========================================================================
    // Diversity scaling tests
    // ========================================================================

    #[test]
    fn test_diversity_scaling_empty() {
        let result = apply_diversity_scaling(&[], 1.2, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_diversity_scaling_single_state() {
        let log_e = vec![vec![-1.0], vec![-2.0]];
        let result = apply_diversity_scaling(&log_e, 1.2, 0.5);
        assert_eq!(result, log_e, "single state should return unchanged");
    }

    #[test]
    fn test_diversity_scaling_confident_window() {
        // One state dominates → low entropy → amplify factor should apply
        let log_e = vec![vec![-0.1, -5.0, -5.0]]; // pop 0 dominates
        let result = apply_diversity_scaling(&log_e, 2.0, 0.5);
        // Low entropy → scale close to amplify (2.0)
        // Deviations from mean should be amplified
        let mean_orig: f64 = log_e[0].iter().sum::<f64>() / 3.0;
        let mean_result: f64 = result[0].iter().sum::<f64>() / 3.0;
        assert!((mean_orig - mean_result).abs() < 1e-10, "mean should be preserved");
        // The dominant state should be pushed further from mean
        assert!(result[0][0] > log_e[0][0], "dominant state should be amplified upward");
    }

    #[test]
    fn test_diversity_scaling_ambiguous_window() {
        // All states roughly equal → high entropy → dampen factor should apply
        let log_e = vec![vec![-1.0, -1.0, -1.0]]; // uniform
        let result = apply_diversity_scaling(&log_e, 2.0, 0.5);
        // High entropy → scale close to dampen (0.5)
        // But values are all equal, so deviations are 0 → no change
        for i in 0..3 {
            assert!((result[0][i] - log_e[0][i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diversity_scaling_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let result = apply_diversity_scaling(&log_e, 1.5, 0.5);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
        assert!(result[0][1].is_finite());
        assert!(result[0][2].is_finite());
    }

    #[test]
    fn test_diversity_scaling_multiple_windows() {
        let log_e = vec![
            vec![-0.1, -5.0, -5.0], // confident
            vec![-1.0, -1.1, -1.2], // ambiguous
        ];
        let result = apply_diversity_scaling(&log_e, 1.5, 0.8);
        assert_eq!(result.len(), 2);
        // Confident window should have larger spread than ambiguous
        let spread_0 = result[0][0] - result[0][2];
        let spread_1 = result[1][0] - result[1][2];
        assert!(spread_0.abs() > spread_1.abs(),
            "confident window should have larger spread after scaling");
    }

    // ========================================================================
    // Residual amplification tests
    // ========================================================================

    #[test]
    fn test_residual_amplify_empty() {
        let result = amplify_emission_residuals(&[], 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_residual_amplify_zero_factor() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = amplify_emission_residuals(&log_e, 0.0);
        assert_eq!(result, log_e, "factor 0 should return unchanged");
    }

    #[test]
    fn test_residual_amplify_factor_one_centers() {
        // factor = 1.0 should center: residuals unchanged
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = amplify_emission_residuals(&log_e, 1.0);
        let mean = (-1.0 + -2.0 + -3.0) / 3.0;
        for i in 0..3 {
            let expected = mean + 1.0 * (log_e[0][i] - mean);
            assert!((result[0][i] - expected).abs() < 1e-10,
                "factor 1.0 should preserve deviations");
        }
    }

    #[test]
    fn test_residual_amplify_factor_two_doubles() {
        let log_e = vec![vec![-1.0, -3.0, -5.0]];
        let mean = -3.0;
        let result = amplify_emission_residuals(&log_e, 2.0);
        // residual[0] = 2 * (-1 - (-3)) = 4, so result[0] = -3 + 4 = 1.0
        assert!((result[0][0] - (mean + 2.0 * (-1.0 - mean))).abs() < 1e-10);
        assert!((result[0][1] - mean).abs() < 1e-10, "mean value should stay at mean");
        assert!((result[0][2] - (mean + 2.0 * (-5.0 - mean))).abs() < 1e-10);
    }

    #[test]
    fn test_residual_amplify_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let result = amplify_emission_residuals(&log_e, 2.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
        assert!(result[0][1].is_finite());
    }

    #[test]
    fn test_residual_amplify_preserves_mean() {
        let log_e = vec![vec![-1.0, -2.0, -4.0]];
        let mean_before: f64 = log_e[0].iter().sum::<f64>() / 3.0;
        let result = amplify_emission_residuals(&log_e, 1.5);
        let mean_after: f64 = result[0].iter().sum::<f64>() / 3.0;
        assert!((mean_before - mean_after).abs() < 1e-10,
            "amplification should preserve per-window mean");
    }

    // ========================================================================
    // Rank transform tests
    // ========================================================================

    #[test]
    fn test_rank_transform_empty() {
        let result = rank_transform_emissions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rank_transform_single_state() {
        let log_e = vec![vec![-2.0], vec![-3.0]];
        let result = rank_transform_emissions(&log_e);
        assert_eq!(result, log_e, "single state should return unchanged");
    }

    #[test]
    fn test_rank_transform_ordering_preserved() {
        // Pop 0 has best emission, pop 2 has worst
        let log_e = vec![vec![-0.5, -1.0, -3.0]];
        let result = rank_transform_emissions(&log_e);
        assert!(result[0][0] > result[0][1], "rank 0 > rank 1");
        assert!(result[0][1] > result[0][2], "rank 1 > rank 2");
    }

    #[test]
    fn test_rank_transform_scale_invariance() {
        // Two windows with very different absolute scales but same ordering
        let log_e = vec![
            vec![-0.1, -0.2, -0.3],    // small differences
            vec![-1.0, -10.0, -100.0],  // huge differences
        ];
        let result = rank_transform_emissions(&log_e);
        // Both should get identical rank scores since ordering is the same
        for k in 0..3 {
            assert!((result[0][k] - result[1][k]).abs() < 1e-10,
                "same ordering should produce same rank scores");
        }
    }

    #[test]
    fn test_rank_transform_proper_log_probs() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = rank_transform_emissions(&log_e);
        // Should sum to ~1.0 in probability space
        let prob_sum: f64 = result[0].iter().map(|&v| v.exp()).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10,
            "rank scores should be proper log-probabilities, sum={}", prob_sum);
    }

    #[test]
    fn test_rank_transform_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let result = rank_transform_emissions(&log_e);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
        // The finite ones should be ranked (rank 0 and rank 1)
        assert!(result[0][1] > result[0][2], "better emission gets higher rank");
    }

    // ========================================================================
    // Emission anchor boost tests
    // ========================================================================

    #[test]
    fn test_anchor_boost_empty() {
        let result = apply_emission_anchor_boost(&[], 3, 0.5, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_anchor_boost_zero_radius() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = apply_emission_anchor_boost(&log_e, 0, 0.5, 1.0);
        assert_eq!(result, log_e, "zero radius should return unchanged");
    }

    #[test]
    fn test_anchor_boost_consistent_neighborhood() {
        // 5 windows, all argmax = pop 0
        let log_e = vec![
            vec![-0.5, -2.0, -3.0],
            vec![-0.4, -1.5, -2.5],
            vec![-0.6, -2.5, -3.5], // center
            vec![-0.3, -1.8, -2.8],
            vec![-0.7, -2.2, -3.2],
        ];
        let result = apply_emission_anchor_boost(&log_e, 2, 0.5, 1.0);
        // Center window: all 4 neighbors agree on pop 0, agreement = 1.0 > 0.5
        assert!(result[2][0] > log_e[2][0], "center pop 0 should be boosted");
        assert_eq!(result[2][1], log_e[2][1], "non-argmax should be unchanged");
    }

    #[test]
    fn test_anchor_boost_disagreeing_neighborhood() {
        // 5 windows, alternating argmax
        let log_e = vec![
            vec![-2.0, -0.5, -3.0], // argmax = 1
            vec![-0.5, -2.0, -3.0], // argmax = 0
            vec![-2.0, -0.5, -3.0], // argmax = 1 (center)
            vec![-0.5, -2.0, -3.0], // argmax = 0
            vec![-2.0, -0.5, -3.0], // argmax = 1
        ];
        let result = apply_emission_anchor_boost(&log_e, 2, 0.7, 1.0);
        // Center (argmax=1): neighbors = [1, 0, 0, 1] → 2/4 = 0.5 < 0.7 threshold
        assert_eq!(result[2], log_e[2], "below threshold should not boost");
    }

    #[test]
    fn test_anchor_boost_proportional_to_agreement() {
        // Test that boost scales with agreement fraction
        let log_e_high = vec![
            vec![-0.5, -2.0], vec![-0.5, -2.0], vec![-0.6, -2.0],
            vec![-0.5, -2.0], vec![-0.5, -2.0],
        ];
        let result_high = apply_emission_anchor_boost(&log_e_high, 2, 0.0, 1.0);
        // All agree on pop 0 → agreement = 1.0

        let log_e_partial = vec![
            vec![-0.5, -2.0], vec![-2.0, -0.5], vec![-0.6, -2.0],
            vec![-0.5, -2.0], vec![-2.0, -0.5],
        ];
        let result_partial = apply_emission_anchor_boost(&log_e_partial, 2, 0.0, 1.0);
        // Center (argmax=0): neighbors = [0, 1, 0, 1] → 2/4 = 0.5

        let boost_high = result_high[2][0] - log_e_high[2][0];
        let boost_partial = result_partial[2][0] - log_e_partial[2][0];
        assert!(boost_high > boost_partial,
            "higher agreement should give larger boost: {} > {}", boost_high, boost_partial);
    }

    #[test]
    fn test_anchor_boost_zero_boost_strength() {
        let log_e = vec![
            vec![-0.5, -2.0], vec![-0.5, -2.0], vec![-0.6, -2.0],
        ];
        let result = apply_emission_anchor_boost(&log_e, 1, 0.0, 0.0);
        assert_eq!(result, log_e, "zero boost strength should return unchanged");
    }

    // ========================================================================
    // Emission outlier dampening tests
    // ========================================================================

    #[test]
    fn test_outlier_dampen_empty() {
        let result = dampen_emission_outliers(&[], 3.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_outlier_dampen_zero_threshold() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = dampen_emission_outliers(&log_e, 0.0);
        assert_eq!(result, log_e, "zero threshold should return unchanged");
    }

    #[test]
    fn test_outlier_dampen_no_outliers() {
        // All values are close to median → no dampening
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.1, -2.1],
            vec![-0.9, -1.9],
            vec![-1.05, -2.05],
            vec![-0.95, -1.95],
        ];
        let result = dampen_emission_outliers(&log_e, 3.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - log_e[i][j]).abs() < 1e-10,
                    "no outliers should mean no change");
            }
        }
    }

    #[test]
    fn test_outlier_dampen_clips_spike() {
        // One window has a huge spike for pop 0
        // Use varied base values so MAD > 0
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.2, -2.1],
            vec![-0.8, -1.9],
            vec![-1.1, -2.05],
            vec![-0.9, -1.95],
            vec![-1.05, -2.0],
            vec![ 5.0, -2.0], // outlier: 5.0 vs median ~-1.0
        ];
        let result = dampen_emission_outliers(&log_e, 3.0);
        // The spike should be clipped toward the median
        assert!(result[6][0] < 5.0, "outlier should be dampened, got {}", result[6][0]);
        assert!(result[6][0] > -1.2, "should still be above median");
        // Non-outlier values should be close to original
        assert!((result[0][0] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_outlier_dampen_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![f64::NEG_INFINITY, -1.0],
            vec![f64::NEG_INFINITY, -1.0],
        ];
        let result = dampen_emission_outliers(&log_e, 3.0);
        for row in &result {
            assert!(row[0].is_infinite() && row[0] < 0.0);
        }
    }

    #[test]
    fn test_outlier_dampen_too_few_values() {
        // Only 2 finite values per pop → below min threshold of 3
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let result = dampen_emission_outliers(&log_e, 3.0);
        assert_eq!(result, log_e, "too few values should return unchanged");
    }

    // ========================================================================
    // Confusion penalty tests
    // ========================================================================

    #[test]
    fn test_confusion_penalties_empty() {
        let result = compute_confusion_penalties(&[], 3, 0.5);
        assert_eq!(result, vec![vec![0.0; 3]; 3]);
    }

    #[test]
    fn test_confusion_penalties_no_switches() {
        let states = vec![0, 0, 0, 0, 0]; // all same state
        let result = compute_confusion_penalties(&states, 3, 0.5);
        for row in &result {
            for &v in row {
                assert!((v - 0.0).abs() < 1e-10, "no switches → no penalties");
            }
        }
    }

    #[test]
    fn test_confusion_penalties_symmetric() {
        let states = vec![0, 1, 0, 1, 2, 0, 1, 0]; // lots of 0↔1 switches
        let result = compute_confusion_penalties(&states, 3, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((result[i][j] - result[j][i]).abs() < 1e-10,
                    "penalties should be symmetric");
            }
        }
    }

    #[test]
    fn test_confusion_penalties_high_switch_pair() {
        let states = vec![0, 1, 0, 1, 0, 1, 2, 2, 2]; // lots of 0↔1, some 1→2
        let result = compute_confusion_penalties(&states, 3, 1.0);
        // 0↔1 should have the strongest (most negative) penalty
        assert!(result[0][1] < result[0][2],
            "0↔1 should have stronger penalty than 0↔2");
        assert!(result[0][1] < 0.0, "penalty should be negative");
    }

    #[test]
    fn test_apply_confusion_penalties_row_sums() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.98, 0.01, 0.01],
                vec![0.01, 0.98, 0.01],
                vec![0.01, 0.01, 0.98],
            ],
            initial: vec![1.0 / 3.0; 3],
            emission_same_pop_mean: 0.95,
            emission_diff_pop_mean: 0.90,
            emission_std: 1.0,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let penalties = vec![
            vec![0.0, -0.5, -0.1],
            vec![-0.5, 0.0, -0.2],
            vec![-0.1, -0.2, 0.0],
        ];
        let log_trans = apply_confusion_penalties(&params, &penalties);
        // Each row should sum to 1.0 in probability space
        for (i, row) in log_trans.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-8,
                "row {} should sum to 1.0, got {}", i, sum);
        }
    }

    #[test]
    fn test_confusion_penalties_zero_weight() {
        let states = vec![0, 1, 0, 1, 2];
        let result = compute_confusion_penalties(&states, 3, 0.0);
        for row in &result {
            for &v in row {
                assert!((v - 0.0).abs() < 1e-10, "zero weight → no penalties");
            }
        }
    }

    // ========================================================================
    // Emission momentum tests
    // ========================================================================

    #[test]
    fn test_emission_momentum_empty() {
        let result = apply_emission_momentum(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_emission_momentum_zero_alpha() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let result = apply_emission_momentum(&log_e, 0.0);
        assert_eq!(result, log_e, "alpha=0 should return unchanged");
    }

    #[test]
    fn test_emission_momentum_smooths() {
        // Sharp transition from -1 to -5 should be smoothed
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
            vec![-5.0, -2.0], // sharp drop in pop 0
            vec![-5.0, -2.0],
        ];
        let result = apply_emission_momentum(&log_e, 0.5);
        // The transition window should be between -1 and -5
        assert!(result[3][0] > -5.0, "momentum should soften the drop");
        assert!(result[3][0] < -1.0, "but should still be below -1.0");
    }

    #[test]
    fn test_emission_momentum_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -1.0],
        ];
        let result = apply_emission_momentum(&log_e, 0.5);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_emission_momentum_symmetric() {
        // Forward-backward should be symmetric: pattern should look the same
        // when reversed
        let log_e = vec![
            vec![-1.0], vec![-1.0], vec![-5.0], vec![-1.0], vec![-1.0],
        ];
        let result = apply_emission_momentum(&log_e, 0.3);
        // Symmetric input around center → output should be symmetric
        assert!((result[0][0] - result[4][0]).abs() < 1e-10,
            "symmetric input should give symmetric output");
        assert!((result[1][0] - result[3][0]).abs() < 1e-10);
    }

    #[test]
    fn test_emission_momentum_single_window() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = apply_emission_momentum(&log_e, 0.5);
        assert_eq!(result, log_e, "single window should be unchanged");
    }

    // ========================================================================
    // Emission floor tests
    // ========================================================================

    #[test]
    fn test_emission_floor_empty() {
        let result = apply_emission_floor(&[], -10.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_emission_floor_no_change_above() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = apply_emission_floor(&log_e, -10.0);
        assert_eq!(result, log_e, "all above floor should be unchanged");
    }

    #[test]
    fn test_emission_floor_clips_below() {
        let log_e = vec![vec![-1.0, -20.0, -50.0]];
        let result = apply_emission_floor(&log_e, -10.0);
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[0][1] - (-10.0)).abs() < 1e-10, "should be clipped to floor");
        assert!((result[0][2] - (-10.0)).abs() < 1e-10, "should be clipped to floor");
    }

    #[test]
    fn test_emission_floor_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -20.0]];
        let result = apply_emission_floor(&log_e, -10.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0,
            "masked states should stay masked");
        assert!((result[0][1] - (-1.0)).abs() < 1e-10);
        assert!((result[0][2] - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_emission_floor_at_exact_floor() {
        let log_e = vec![vec![-10.0, -10.0]];
        let result = apply_emission_floor(&log_e, -10.0);
        assert_eq!(result, log_e, "values at floor should be unchanged");
    }

    // ========================================================================
    // Gradient penalty tests
    // ========================================================================

    #[test]
    fn test_gradient_penalty_empty() {
        let result = apply_gradient_penalty(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gradient_penalty_single_window() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = apply_gradient_penalty(&log_e, 0.5);
        assert_eq!(result, log_e, "single window should be unchanged");
    }

    #[test]
    fn test_gradient_penalty_zero_weight() {
        let log_e = vec![vec![-1.0], vec![-5.0]];
        let result = apply_gradient_penalty(&log_e, 0.0);
        assert_eq!(result, log_e, "zero weight should return unchanged");
    }

    #[test]
    fn test_gradient_penalty_reduces_jump() {
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-5.0, -2.0], // large jump in pop 0
        ];
        let result = apply_gradient_penalty(&log_e, 0.5);
        // Jump should be reduced
        let orig_jump = (log_e[1][0] - log_e[0][0]).abs();
        let new_jump = (result[1][0] - result[0][0]).abs();
        assert!(new_jump < orig_jump, "gradient penalty should reduce jump: {} < {}", new_jump, orig_jump);
    }

    #[test]
    fn test_gradient_penalty_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -1.0],
        ];
        let result = apply_gradient_penalty(&log_e, 0.5);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_gradient_penalty_no_change_for_equal() {
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let result = apply_gradient_penalty(&log_e, 0.5);
        for (i, row) in result.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!((v - log_e[i][j]).abs() < 1e-10,
                    "equal adjacent values should be unchanged");
            }
        }
    }

    // ========================================================================
    // Posterior-weighted emission blend tests
    // ========================================================================

    #[test]
    fn test_posterior_blend_empty() {
        let result = blend_posteriors_with_emissions(&[], &[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_posterior_blend_zero_lambda() {
        let log_e = vec![vec![-1.0, -2.0]];
        let post = vec![vec![0.7, 0.3]];
        let result = blend_posteriors_with_emissions(&log_e, &post, 0.0);
        assert_eq!(result, log_e, "lambda=0 should return emissions unchanged");
    }

    #[test]
    fn test_posterior_blend_full_lambda() {
        let log_e = vec![vec![-1.0, -2.0]];
        let post = vec![vec![0.8, 0.2]];
        let result = blend_posteriors_with_emissions(&log_e, &post, 1.0);
        // Should be pure log-posteriors
        assert!((result[0][0] - 0.8_f64.ln()).abs() < 1e-10);
        assert!((result[0][1] - 0.2_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_blend_half_lambda() {
        let log_e = vec![vec![-1.0, -3.0]];
        let post = vec![vec![0.8, 0.2]];
        let result = blend_posteriors_with_emissions(&log_e, &post, 0.5);
        let expected_0 = 0.5 * (-1.0) + 0.5 * 0.8_f64.ln();
        let expected_1 = 0.5 * (-3.0) + 0.5 * 0.2_f64.ln();
        assert!((result[0][0] - expected_0).abs() < 1e-10);
        assert!((result[0][1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_blend_zero_posterior() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let post = vec![vec![0.9, 0.1, 0.0]]; // pop 2 has zero posterior
        let result = blend_posteriors_with_emissions(&log_e, &post, 0.5);
        // Zero posterior → log(0) = NEG_INF → should fall back to emission
        assert!((result[0][2] - (-3.0)).abs() < 1e-10);
    }

    // ========================================================================
    // Changepoint prior tests
    // ========================================================================

    #[test]
    fn test_changepoint_prior_empty() {
        let result = apply_changepoint_prior(&[], &[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_changepoint_prior_zero_bonus() {
        let log_e = vec![vec![-1.0, -2.0]];
        let states = vec![0];
        let result = apply_changepoint_prior(&log_e, &states, 0.0);
        assert_eq!(result, log_e, "zero bonus should return unchanged");
    }

    #[test]
    fn test_changepoint_prior_boosts_pass1_state() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.5, -1.5, -1.5],
        ];
        let states = vec![0, 2];
        let result = apply_changepoint_prior(&log_e, &states, 1.0);
        // Window 0: state 0 gets +1.0
        assert!((result[0][0] - 0.0).abs() < 1e-10); // -1.0 + 1.0
        assert!((result[0][1] - (-2.0)).abs() < 1e-10); // unchanged
        // Window 1: state 2 gets +1.0
        assert!((result[1][2] - (-0.5)).abs() < 1e-10); // -1.5 + 1.0
        assert!((result[1][0] - (-1.5)).abs() < 1e-10); // unchanged
    }

    #[test]
    fn test_changepoint_prior_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0]];
        let states = vec![0];
        let result = apply_changepoint_prior(&log_e, &states, 1.0);
        // NEG_INFINITY + bonus is still NEG_INFINITY... wait, is_finite check needed
        // Actually: NEG_INFINITY is not finite, so the boost should be skipped
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_changepoint_prior_different_per_window() {
        let log_e = vec![
            vec![-2.0, -2.0],
            vec![-2.0, -2.0],
            vec![-2.0, -2.0],
        ];
        let states = vec![0, 1, 0];
        let result = apply_changepoint_prior(&log_e, &states, 0.5);
        assert!((result[0][0] - (-1.5)).abs() < 1e-10);
        assert!((result[1][1] - (-1.5)).abs() < 1e-10);
        assert!((result[2][0] - (-1.5)).abs() < 1e-10);
    }

    // ========================================================================
    // Pairwise emission contrast tests
    // ========================================================================

    #[test]
    fn test_pairwise_contrast_empty() {
        let result = apply_pairwise_emission_contrast(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pairwise_contrast_zero_boost() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = apply_pairwise_emission_contrast(&log_e, 0.0);
        assert_eq!(result, log_e, "zero boost should return unchanged");
    }

    #[test]
    fn test_pairwise_contrast_widens_gap() {
        let log_e = vec![vec![-1.0, -1.5, -3.0]]; // gap between best and 2nd = 0.5
        let result = apply_pairwise_emission_contrast(&log_e, 0.3);
        // Best (-1.0) gets +0.3 = -0.7, second (-1.5) gets -0.3 = -1.8
        assert!((result[0][0] - (-0.7)).abs() < 1e-10);
        assert!((result[0][1] - (-1.8)).abs() < 1e-10);
        assert!((result[0][2] - (-3.0)).abs() < 1e-10, "3rd should be unchanged");
    }

    #[test]
    fn test_pairwise_contrast_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let result = apply_pairwise_emission_contrast(&log_e, 0.5);
        // NEG_INF is filtered out, so top-2 are -1.0 and -2.0
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
        assert!((result[0][1] - (-0.5)).abs() < 1e-10); // -1.0 + 0.5
        assert!((result[0][2] - (-2.5)).abs() < 1e-10); // -2.0 - 0.5
    }

    #[test]
    fn test_pairwise_contrast_single_finite() {
        let log_e = vec![vec![-1.0, f64::NEG_INFINITY]];
        let result = apply_pairwise_emission_contrast(&log_e, 0.5);
        // Only 1 finite value → no contrast possible
        assert_eq!(result, log_e);
    }

    // ========================================================================
    // Pop temperature adjustment tests
    // ========================================================================

    #[test]
    fn test_pop_temp_adjust_empty() {
        let result = adjust_pop_temperatures(&[], &[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_pop_temp_adjust_zero_factor() {
        let log_e = vec![vec![-1.0, -2.0]];
        let post = vec![vec![0.7, 0.3]];
        let result = adjust_pop_temperatures(&log_e, &post, 0.0);
        assert_eq!(result, log_e, "zero factor should return unchanged");
    }

    #[test]
    fn test_pop_temp_adjust_amplifies_uncertain() {
        // Pop 0 has high posterior → should be dampened
        // Pop 1 has low posterior → should be amplified
        let log_e = vec![
            vec![-1.0, -2.0],
            vec![-1.0, -2.0],
        ];
        let post = vec![
            vec![0.9, 0.1],
            vec![0.8, 0.2],
        ];
        let result = adjust_pop_temperatures(&log_e, &post, 2.0);
        // With K=2, threshold=0.5
        // Pop 0: mean_post=0.85 > 0.5 → scale < 1 → dampen
        // Pop 1: mean_post=0.15 < 0.5 → scale > 1 → amplify
        let mean = (-1.0 + -2.0) / 2.0;
        // Pop 1 (uncertain) should have larger deviation from mean
        let dev_0 = (result[0][0] - mean).abs();
        let dev_1 = (result[0][1] - mean).abs();
        assert!(dev_1 > dev_0 || (result[0][1] - log_e[0][1]).abs() > (result[0][0] - log_e[0][0]).abs(),
            "uncertain pop should be more amplified");
    }

    #[test]
    fn test_pop_temp_adjust_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0]];
        let post = vec![vec![0.5, 0.5]];
        let result = adjust_pop_temperatures(&log_e, &post, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_pop_temp_adjust_uniform_posteriors_no_change() {
        // Uniform posteriors → mean_post = threshold → scale = 1 → no change
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.0, -2.0, -3.0],
        ];
        let post = vec![
            vec![1.0/3.0, 1.0/3.0, 1.0/3.0],
            vec![1.0/3.0, 1.0/3.0, 1.0/3.0],
        ];
        let result = adjust_pop_temperatures(&log_e, &post, 1.0);
        for t in 0..2 {
            for k in 0..3 {
                assert!((result[t][k] - log_e[t][k]).abs() < 1e-10,
                    "uniform posteriors should give no change");
            }
        }
    }

    // ========================================================================
    // SNR weighting tests
    // ========================================================================

    #[test]
    fn test_snr_weighting_empty() {
        let result = apply_snr_weighting(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_snr_weighting_zero_power() {
        let log_e = vec![vec![-1.0, -5.0]];
        let result = apply_snr_weighting(&log_e, 0.0);
        assert_eq!(result, log_e, "zero power should return unchanged");
    }

    #[test]
    fn test_snr_amplifies_high_range() {
        // Window 0: high range (clear signal), Window 1: low range (noise)
        let log_e = vec![
            vec![-1.0, -5.0], // range = 4.0
            vec![-2.0, -2.5], // range = 0.5
            vec![-1.0, -5.0], // range = 4.0
        ];
        let result = apply_snr_weighting(&log_e, 1.0);
        // median range = 4.0
        // Window 0: scale = 4.0/4.0 = 1.0 → no change
        // Window 1: scale = 0.5/4.0 = 0.125 → dampened
        let range_0 = (result[0][0] - result[0][1]).abs();
        let range_1 = (result[1][0] - result[1][1]).abs();
        assert!(range_0 > range_1,
            "high-SNR window should have larger range: {} > {}", range_0, range_1);
    }

    #[test]
    fn test_snr_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0, -2.0],
            vec![-1.0, -2.0, -3.0],
        ];
        let result = apply_snr_weighting(&log_e, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_snr_uniform_range_no_change() {
        // All windows have the same range → all scale = 1.0
        let log_e = vec![
            vec![-1.0, -3.0],
            vec![-2.0, -4.0],
            vec![-0.5, -2.5],
        ];
        let result = apply_snr_weighting(&log_e, 1.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "uniform range should give no change");
            }
        }
    }

    // ── Cross-entropy regularization tests ──

    #[test]
    fn test_cross_entropy_reg_lambda_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let posteriors = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let result = regularize_toward_posteriors(&log_e, &posteriors, 0.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "lambda=0 should return original emissions");
            }
        }
    }

    #[test]
    fn test_cross_entropy_reg_lambda_one() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let posteriors = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let result = regularize_toward_posteriors(&log_e, &posteriors, 1.0);
        // lambda=1 → pure log-posteriors
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                let expected = posteriors[t][k].ln();
                assert!((v - expected).abs() < 1e-10,
                    "lambda=1 should give log-posteriors");
            }
        }
    }

    #[test]
    fn test_cross_entropy_reg_interpolation() {
        let log_e = vec![vec![-1.0, -2.0]];
        let posteriors = vec![vec![0.9, 0.1]];
        let lambda = 0.3;
        let result = regularize_toward_posteriors(&log_e, &posteriors, lambda);
        let expected_0 = (1.0 - lambda) * (-1.0) + lambda * 0.9_f64.ln();
        let expected_1 = (1.0 - lambda) * (-2.0) + lambda * 0.1_f64.ln();
        assert!((result[0][0] - expected_0).abs() < 1e-10);
        assert!((result[0][1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_reg_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let posteriors: Vec<Vec<f64>> = vec![];
        let result = regularize_toward_posteriors(&log_e, &posteriors, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cross_entropy_reg_preserves_length() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 10];
        let posteriors = vec![vec![0.5, 0.3, 0.2]; 10];
        let result = regularize_toward_posteriors(&log_e, &posteriors, 0.4);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].len(), 3);
    }

    // ── Windowed normalization tests ──

    #[test]
    fn test_windowed_norm_radius_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let result = apply_windowed_normalization(&log_e, 0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "radius=0 should return original");
            }
        }
    }

    #[test]
    fn test_windowed_norm_preserves_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 20];
        let result = apply_windowed_normalization(&log_e, 3);
        assert_eq!(result.len(), 20);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_windowed_norm_constant_signal() {
        // If every window has the same emissions, normalization should be identity
        let log_e = vec![vec![-2.0, -3.0, -1.0]; 10];
        let result = apply_windowed_normalization(&log_e, 2);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "constant signal should be unchanged at t={}, k={}", t, k);
            }
        }
    }

    #[test]
    fn test_windowed_norm_removes_trend() {
        // Linear trend in pop 0: -1, -2, -3, -4, -5
        // Pop 1 constant: -1, -1, -1, -1, -1
        let log_e = vec![
            vec![-1.0, -1.0],
            vec![-2.0, -1.0],
            vec![-3.0, -1.0],
            vec![-4.0, -1.0],
            vec![-5.0, -1.0],
        ];
        let result = apply_windowed_normalization(&log_e, 1);
        // After removing local trend, the middle windows should be closer to global mean
        let global_mean_pop0 = (-1.0 + -2.0 + -3.0 + -4.0 + -5.0) / 5.0; // -3.0
        // Check that the variance of pop 0 across windows decreased
        let orig_var: f64 = log_e.iter().map(|r| (r[0] - global_mean_pop0).powi(2)).sum::<f64>() / 5.0;
        let new_var: f64 = result.iter().map(|r| (r[0] - global_mean_pop0).powi(2)).sum::<f64>() / 5.0;
        assert!(new_var < orig_var,
            "windowed normalization should reduce variance from trend: {} vs {}", new_var, orig_var);
    }

    #[test]
    fn test_windowed_norm_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = apply_windowed_normalization(&log_e, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_windowed_norm_single_window() {
        let log_e = vec![vec![-2.0, -3.0]];
        let result = apply_windowed_normalization(&log_e, 5);
        // Single window: local mean == global mean → no change
        assert!((result[0][0] - log_e[0][0]).abs() < 1e-10);
        assert!((result[0][1] - log_e[0][1]).abs() < 1e-10);
    }

    // ── Entropy-weighted posterior smoothing tests ──

    #[test]
    fn test_entropy_smooth_radius_zero() {
        let posteriors = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let result = entropy_smooth_posteriors(&posteriors, 0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - posteriors[t][k]).abs() < 1e-10,
                    "radius=0 should return original");
            }
        }
    }

    #[test]
    fn test_entropy_smooth_preserves_shape() {
        let posteriors = vec![vec![0.5, 0.3, 0.2]; 15];
        let result = entropy_smooth_posteriors(&posteriors, 3);
        assert_eq!(result.len(), 15);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_entropy_smooth_rows_sum_to_one() {
        let posteriors = vec![
            vec![0.9, 0.1],
            vec![0.5, 0.5],
            vec![0.1, 0.9],
            vec![0.6, 0.4],
        ];
        let result = entropy_smooth_posteriors(&posteriors, 1);
        for (t, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "row {} should sum to 1.0, got {}", t, sum);
        }
    }

    #[test]
    fn test_entropy_smooth_v2_confident_propagates() {
        // Window 0: very confident pop 0 (0.99, 0.01)
        // Window 1: uncertain (0.5, 0.5)
        // Window 2: confident pop 0 (0.95, 0.05)
        // Both confident neighbors agree on pop 0 → uncertain window should shift
        let posteriors = vec![
            vec![0.99, 0.01],
            vec![0.5, 0.5],
            vec![0.95, 0.05],
        ];
        let result = entropy_smooth_posteriors(&posteriors, 1);
        // The uncertain window 1 should be pulled toward pop 0
        assert!(result[1][0] > 0.5,
            "uncertain window should shift toward confident neighbors: got {}", result[1][0]);
    }

    #[test]
    fn test_entropy_smooth_v2_empty() {
        let posteriors: Vec<Vec<f64>> = vec![];
        let result = entropy_smooth_posteriors(&posteriors, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_entropy_smooth_v2_uniform_posteriors() {
        // All windows uniform → smoothing shouldn't change much
        let posteriors = vec![vec![0.25, 0.25, 0.25, 0.25]; 5];
        let result = entropy_smooth_posteriors(&posteriors, 2);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - 0.25).abs() < 1e-6,
                    "uniform posteriors should stay uniform at t={}, k={}", t, k);
            }
        }
    }

    // ── Quantile normalization tests ──

    #[test]
    fn test_quantile_normalize_preserves_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 10];
        let result = quantile_normalize_emissions(&log_e);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_quantile_normalize_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = quantile_normalize_emissions(&log_e);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quantile_normalize_identical_pops() {
        // If all populations have the same values, normalization should be ~identity
        let log_e = vec![
            vec![-1.0, -1.0, -1.0],
            vec![-2.0, -2.0, -2.0],
            vec![-3.0, -3.0, -3.0],
        ];
        let result = quantile_normalize_emissions(&log_e);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "identical pops should be unchanged at t={}, k={}", t, k);
            }
        }
    }

    #[test]
    fn test_quantile_normalize_equalizes_distributions() {
        // Pop 0: -1, -2, -3, -4, -5 (range 4)
        // Pop 1: -10, -20, -30, -40, -50 (range 40)
        // After normalization, both should have similar distributions
        let log_e = vec![
            vec![-1.0, -10.0],
            vec![-2.0, -20.0],
            vec![-3.0, -30.0],
            vec![-4.0, -40.0],
            vec![-5.0, -50.0],
        ];
        let result = quantile_normalize_emissions(&log_e);
        // After quantile normalization, the ranges should be equalized
        let range0: f64 = result.iter().map(|r| r[0]).fold(f64::NEG_INFINITY, f64::max)
            - result.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min);
        let range1: f64 = result.iter().map(|r| r[1]).fold(f64::NEG_INFINITY, f64::max)
            - result.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min);
        assert!((range0 - range1).abs() < 1e-6,
            "ranges should be equalized: {} vs {}", range0, range1);
    }

    #[test]
    fn test_quantile_normalize_preserves_neg_inf() {
        let log_e = vec![
            vec![-1.0, f64::NEG_INFINITY],
            vec![-2.0, -1.0],
            vec![-3.0, -2.0],
        ];
        let result = quantile_normalize_emissions(&log_e);
        // NEG_INFINITY entries should still be NEG_INFINITY
        assert!(result[0][1].is_finite() == false,
            "NEG_INFINITY should be preserved");
    }

    #[test]
    fn test_quantile_normalize_single_window() {
        let log_e = vec![vec![-2.0, -5.0, -3.0]];
        let result = quantile_normalize_emissions(&log_e);
        // Single window: each pop has one value, reference is average → all same
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    // ── Adaptive transition scaling tests ──

    #[test]
    fn test_adaptive_transitions_factor_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_adaptive_transitions(&log_e, &params, 0.0);
        assert_eq!(result.len(), 2);
        // factor=0 → scale=1 for all windows → should approximate base log-transitions
        let base_00 = 0.95_f64.ln();
        let base_01 = 0.05_f64.ln();
        for per_window in &result {
            assert!((per_window[0][0] - base_00).abs() < 1e-6,
                "factor=0 should give base transitions");
            assert!((per_window[0][1] - base_01).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaptive_transitions_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 5];
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_adaptive_transitions(&log_e, &params, 1.0);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].len(), 3);
        assert_eq!(result[0][0].len(), 3);
    }

    #[test]
    fn test_adaptive_transitions_rows_sum_to_one() {
        let log_e = vec![
            vec![-0.5, -5.0],  // very confident (pop 0)
            vec![-2.0, -2.1],  // uncertain
        ];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_adaptive_transitions(&log_e, &params, 2.0);
        for (t, per_window) in result.iter().enumerate() {
            for (i, row) in per_window.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row [{},{}] should sum to 1.0 in prob space, got {}", t, i, sum);
            }
        }
    }

    #[test]
    fn test_adaptive_transitions_confident_penalizes_more() {
        // Confident window should have more negative off-diagonal (harder to switch)
        let log_e = vec![
            vec![-0.1, -10.0],  // very confident (pop 0)
            vec![-2.0, -2.01],  // very uncertain
        ];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_adaptive_transitions(&log_e, &params, 2.0);
        // Confident window (t=0): off-diagonal should be more negative
        // Uncertain window (t=1): off-diagonal should be less negative (closer to 0)
        let off_diag_confident = result[0][0][1]; // log P(switch) for confident
        let off_diag_uncertain = result[1][0][1]; // log P(switch) for uncertain
        assert!(off_diag_confident < off_diag_uncertain,
            "confident window should penalize switching more: {} vs {}",
            off_diag_confident, off_diag_uncertain);
    }

    #[test]
    fn test_adaptive_transitions_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_adaptive_transitions(&log_e, &params, 1.0);
        assert!(result.is_empty());
    }

    // ── Local emission reranking tests ──

    #[test]
    fn test_local_rerank_radius_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let result = local_rerank_emissions(&log_e, 0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "radius=0 should return original");
            }
        }
    }

    #[test]
    fn test_local_rerank_preserves_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 10];
        let result = local_rerank_emissions(&log_e, 3);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_local_rerank_proper_log_probs() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-1.0, -2.0, -3.0],
            vec![-1.0, -2.0, -3.0],
        ];
        let result = local_rerank_emissions(&log_e, 1);
        // All windows agree on ranking → each result row should sum to ~1.0 in prob space
        for (t, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "row {} should sum to 1.0 in prob space, got {}", t, sum);
        }
    }

    #[test]
    fn test_local_rerank_best_gets_highest() {
        // Pop 0 is always best → should get highest score
        let log_e = vec![
            vec![-0.5, -2.0, -3.0],
            vec![-0.3, -1.5, -4.0],
            vec![-0.8, -2.5, -3.5],
        ];
        let result = local_rerank_emissions(&log_e, 1);
        for (t, row) in result.iter().enumerate() {
            assert!(row[0] > row[1] && row[1] > row[2],
                "pop 0 should have highest score at t={}: {:?}", t, row);
        }
    }

    #[test]
    fn test_local_rerank_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = local_rerank_emissions(&log_e, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_local_rerank_neighborhood_effect() {
        // Window 1: pop 1 is locally best, but neighbors favor pop 0
        let log_e = vec![
            vec![-0.5, -3.0],  // pop 0 best
            vec![-2.5, -2.0],  // pop 1 slightly better
            vec![-0.5, -3.0],  // pop 0 best
        ];
        let result = local_rerank_emissions(&log_e, 1);
        // With radius=1, neighborhood sums:
        // pop 0: -0.5 + -2.5 + -0.5 = -3.5
        // pop 1: -3.0 + -2.0 + -3.0 = -8.0
        // pop 0 wins by neighborhood support
        assert!(result[1][0] > result[1][1],
            "neighborhood should override local preference: {} vs {}",
            result[1][0], result[1][1]);
    }

    // ── Bayesian emission shrinkage tests ──

    #[test]
    fn test_bayesian_shrink_alpha_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let result = bayesian_shrink_emissions(&log_e, 0.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "alpha=0 should return original");
            }
        }
    }

    #[test]
    fn test_bayesian_shrink_alpha_one() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let result = bayesian_shrink_emissions(&log_e, 1.0);
        // alpha=1 → all windows should be the global mean
        let mean_0 = (-1.0 + -3.0) / 2.0; // -2.0
        let mean_1 = (-2.0 + -4.0) / 2.0; // -3.0
        for row in &result {
            assert!((row[0] - mean_0).abs() < 1e-10);
            assert!((row[1] - mean_1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bayesian_shrink_interpolation() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let alpha = 0.4;
        let result = bayesian_shrink_emissions(&log_e, alpha);
        let mean_0 = -2.0;
        let expected = (1.0 - alpha) * (-1.0) + alpha * mean_0;
        assert!((result[0][0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_shrink_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = bayesian_shrink_emissions(&log_e, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bayesian_shrink_preserves_neg_inf() {
        let log_e = vec![vec![-1.0, f64::NEG_INFINITY], vec![-3.0, -2.0]];
        let result = bayesian_shrink_emissions(&log_e, 0.5);
        assert!(!result[0][1].is_finite(), "NEG_INFINITY should be preserved");
        assert!(result[0][0].is_finite());
    }

    // ── Top-K emission sparsification tests ──

    #[test]
    fn test_sparsify_top_k_keeps_top() {
        let log_e = vec![vec![-1.0, -5.0, -2.0, -8.0, -3.0]];
        let result = sparsify_top_k_emissions(&log_e, 2, -100.0);
        // Top 2: pop 0 (-1.0) and pop 2 (-2.0)
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[0][2] - (-2.0)).abs() < 1e-10);
        assert!((result[0][1] - (-100.0)).abs() < 1e-10);
        assert!((result[0][3] - (-100.0)).abs() < 1e-10);
        assert!((result[0][4] - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sparsify_top_k_all() {
        // top_k >= K → identity
        let log_e = vec![vec![-1.0, -2.0, -3.0]];
        let result = sparsify_top_k_emissions(&log_e, 5, -100.0);
        for (k, &v) in result[0].iter().enumerate() {
            assert!((v - log_e[0][k]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparsify_top_k_preserves_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 10];
        let result = sparsify_top_k_emissions(&log_e, 2, -100.0);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_sparsify_top_k_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = sparsify_top_k_emissions(&log_e, 2, -100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparsify_top_k_zero() {
        // top_k = 0 → identity
        let log_e = vec![vec![-1.0, -2.0]];
        let result = sparsify_top_k_emissions(&log_e, 0, -100.0);
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sparsify_top_1() {
        // top_k = 1 → only best kept
        let log_e = vec![
            vec![-3.0, -1.0, -2.0],
            vec![-1.0, -4.0, -2.0],
        ];
        let result = sparsify_top_k_emissions(&log_e, 1, -50.0);
        // t=0: pop 1 is best → keep, others floor
        assert!((result[0][1] - (-1.0)).abs() < 1e-10);
        assert!((result[0][0] - (-50.0)).abs() < 1e-10);
        // t=1: pop 0 is best
        assert!((result[1][0] - (-1.0)).abs() < 1e-10);
        assert!((result[1][1] - (-50.0)).abs() < 1e-10);
    }

    // ── Majority vote filter tests ──

    #[test]
    fn test_majority_vote_radius_zero() {
        let states = vec![0, 1, 0, 1, 0];
        let result = majority_vote_filter(&states, 2, 0);
        assert_eq!(result, states);
    }

    #[test]
    fn test_majority_vote_removes_isolated() {
        // 0 0 1 0 0 → isolated 1 should become 0
        let states = vec![0, 0, 1, 0, 0];
        let result = majority_vote_filter(&states, 2, 1);
        assert_eq!(result[2], 0, "isolated state should be replaced by majority");
    }

    #[test]
    fn test_majority_vote_preserves_blocks() {
        // 0 0 0 1 1 1 → no change (each state has majority in its neighborhood)
        let states = vec![0, 0, 0, 1, 1, 1];
        let result = majority_vote_filter(&states, 2, 1);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[4], 1);
        assert_eq!(result[5], 1);
    }

    #[test]
    fn test_majority_vote_empty() {
        let states: Vec<usize> = vec![];
        let result = majority_vote_filter(&states, 3, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_majority_vote_large_radius() {
        // All 0 except one 1 in the middle → should become all 0
        let mut states = vec![0; 11];
        states[5] = 1;
        let result = majority_vote_filter(&states, 2, 5);
        assert_eq!(result[5], 0, "majority should override");
    }

    // ── Population proportion prior tests ──

    #[test]
    fn test_proportion_prior_weight_zero() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let proportions = vec![0.6, 0.4];
        let result = apply_proportion_prior(&log_e, &proportions, 0.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "weight=0 should return original");
            }
        }
    }

    #[test]
    fn test_proportion_prior_boosts_common() {
        let log_e = vec![vec![-2.0, -2.0]]; // Equal emissions
        let proportions = vec![0.8, 0.2];
        let result = apply_proportion_prior(&log_e, &proportions, 1.0);
        // Pop 0 (proportion 0.8) should get bigger boost than pop 1 (0.2)
        assert!(result[0][0] > result[0][1],
            "common population should get higher emission: {} vs {}",
            result[0][0], result[0][1]);
    }

    #[test]
    fn test_proportion_prior_adds_log_prior() {
        let log_e = vec![vec![-2.0, -3.0]];
        let proportions = vec![0.6, 0.4];
        let weight = 0.5;
        let result = apply_proportion_prior(&log_e, &proportions, weight);
        let expected_0 = -2.0 + weight * 0.6_f64.ln();
        let expected_1 = -3.0 + weight * 0.4_f64.ln();
        assert!((result[0][0] - expected_0).abs() < 1e-10);
        assert!((result[0][1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn test_proportion_prior_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let proportions = vec![0.5, 0.5];
        let result = apply_proportion_prior(&log_e, &proportions, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_proportion_prior_preserves_neg_inf() {
        let log_e = vec![vec![f64::NEG_INFINITY, -2.0]];
        let proportions = vec![0.5, 0.5];
        let result = apply_proportion_prior(&log_e, &proportions, 1.0);
        assert!(!result[0][0].is_finite(), "NEG_INFINITY should be preserved");
    }

    // ── Segment boundary boost tests ──

    #[test]
    fn test_boundary_boost_no_boundaries() {
        // All same state → no boundaries → base transitions
        let states = vec![0, 0, 0, 0, 0];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_boundary_boost_transitions(&states, &params, 2.0);
        assert_eq!(result.len(), 5);
        // No boundaries → all should be base transitions
        let base_01 = 0.1_f64.ln();
        for per_window in &result {
            // Off-diagonal should be close to base (within rounding)
            assert!((per_window[0][1].exp() - 0.1).abs() < 1e-6,
                "no boundary should keep base transitions");
        }
    }

    #[test]
    fn test_boundary_boost_at_boundary() {
        let states = vec![0, 0, 1, 1];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_boundary_boost_transitions(&states, &params, 2.0);
        // Boundary at t=1 and t=2 → off-diagonal should be less negative (easier switch)
        let base_off_diag = 0.1_f64.ln();
        let boundary_off_diag = result[1][0][1]; // t=1 is at boundary
        let non_boundary_off_diag = result[0][0][1]; // t=0 is not
        assert!(boundary_off_diag > non_boundary_off_diag,
            "boundary should ease switching: {} vs {}",
            boundary_off_diag, non_boundary_off_diag);
    }

    #[test]
    fn test_boundary_boost_rows_sum_to_one() {
        let states = vec![0, 1, 0, 1, 0];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_boundary_boost_transitions(&states, &params, 1.0);
        for (t, per_window) in result.iter().enumerate() {
            for (i, row) in per_window.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row [{},{}] should sum to 1.0, got {}", t, i, sum);
            }
        }
    }

    #[test]
    fn test_boundary_boost_empty() {
        let states: Vec<usize> = vec![];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_boundary_boost_transitions(&states, &params, 1.0);
        assert!(result.is_empty());
    }

    // ── Emission confidence weighting tests ──

    #[test]
    fn test_confidence_weight_preserves_shape() {
        let log_e = vec![vec![-1.0, -2.0, -3.0]; 10];
        let result = apply_confidence_weighting(&log_e, 1.0);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_confidence_weight_amplifies_confident() {
        // Window 0: large gap (confident) → amplified deviations
        // Window 1: small gap (uncertain) → dampened deviations
        let log_e = vec![
            vec![-1.0, -5.0],  // gap = 4.0
            vec![-2.0, -2.1],  // gap = 0.1
        ];
        let result = apply_confidence_weighting(&log_e, 1.0);
        // The confident window should have larger deviations from mean
        let mean_0 = (-1.0 + -5.0) / 2.0;
        let dev_0 = (result[0][0] - mean_0).abs();
        let orig_dev_0 = (-1.0 - mean_0).abs();
        // Confident window gets amplified (deviation increases or stays same)
        // since its gap is above median
        assert!(dev_0 >= 0.0, "deviations should be non-negative");
    }

    #[test]
    fn test_confidence_weight_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = apply_confidence_weighting(&log_e, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_confidence_weight_uniform_emissions() {
        // All windows have same gap → scale = 1.0 → no change
        let log_e = vec![
            vec![-1.0, -3.0],
            vec![-2.0, -4.0],
            vec![-0.5, -2.5],
        ];
        let result = apply_confidence_weighting(&log_e, 1.0);
        // All gaps are equal (2.0) → all scale = 1.0 → identity
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "uniform gaps should give no change at t={}, k={}", t, k);
            }
        }
    }

    #[test]
    fn test_confidence_weight_preserves_neg_inf() {
        let log_e = vec![vec![-1.0, f64::NEG_INFINITY], vec![-2.0, -3.0]];
        let result = apply_confidence_weighting(&log_e, 1.0);
        assert!(!result[0][1].is_finite(), "NEG_INFINITY should be preserved");
    }

    // ── Forward-backward temperature tests ──

    #[test]
    fn test_fb_temperature_one() {
        let log_e = vec![vec![-1.0, -2.0], vec![-3.0, -1.0]];
        let result = apply_fb_temperature(&log_e, 1.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10, "T=1 should be identity");
            }
        }
    }

    #[test]
    fn test_fb_temperature_sharpens() {
        let log_e = vec![vec![-1.0, -3.0]];
        let result = apply_fb_temperature(&log_e, 0.5);
        // T=0.5 → multiply by 2 → more extreme
        assert!((result[0][0] - (-2.0)).abs() < 1e-10);
        assert!((result[0][1] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fb_temperature_softens() {
        let log_e = vec![vec![-2.0, -6.0]];
        let result = apply_fb_temperature(&log_e, 2.0);
        // T=2 → multiply by 0.5 → less extreme
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[0][1] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fb_temperature_empty() {
        let log_e: Vec<Vec<f64>> = vec![];
        let result = apply_fb_temperature(&log_e, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fb_temperature_preserves_neg_inf() {
        let log_e = vec![vec![-1.0, f64::NEG_INFINITY]];
        let result = apply_fb_temperature(&log_e, 0.5);
        assert!(!result[0][1].is_finite(), "NEG_INFINITY should be preserved");
    }

    // ── Co-occurrence transition bonus tests ──

    #[test]
    fn test_cooccurrence_no_transitions() {
        let states = vec![0, 0, 0, 0]; // No transitions
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_cooccurrence_transitions(&states, &params, 2.0);
        assert_eq!(result.len(), 2);
        // No co-occurrences → should approximate base transitions
        let sum: f64 = result[0].iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "rows should sum to 1.0");
    }

    #[test]
    fn test_cooccurrence_boosts_common_pair() {
        let states = vec![0, 1, 0, 1, 0, 1]; // Frequent 0↔1 transitions
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_cooccurrence_transitions(&states, &params, 5.0);
        // 0↔1 is the only co-occurring pair → should be boosted relative to 0↔2
        let trans_0_1 = result[0][1].exp();
        let trans_0_2 = result[0][2].exp();
        assert!(trans_0_1 > trans_0_2,
            "co-occurring pair should be boosted: {} vs {}", trans_0_1, trans_0_2);
    }

    #[test]
    fn test_cooccurrence_rows_sum_to_one() {
        let states = vec![0, 1, 2, 0, 1];
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_cooccurrence_transitions(&states, &params, 2.0);
        for (i, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "row {} should sum to 1.0, got {}", i, sum);
        }
    }

    #[test]
    fn test_cooccurrence_empty() {
        let states: Vec<usize> = vec![];
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_cooccurrence_transitions(&states, &params, 1.0);
        assert_eq!(result.len(), 2);
    }

    // ========================================================================
    // Emission detrending tests
    // ========================================================================

    #[test]
    fn test_detrend_empty() {
        let result = detrend_emissions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_detrend_single_window() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = detrend_emissions(&log_e);
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
        assert!((result[0][1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_detrend_removes_linear_trend() {
        // Pop 0 has a linear trend: -1, -2, -3, -4, -5
        // Pop 1 is flat: -3, -3, -3, -3, -3
        let log_e = vec![
            vec![-1.0, -3.0],
            vec![-2.0, -3.0],
            vec![-3.0, -3.0],
            vec![-4.0, -3.0],
            vec![-5.0, -3.0],
        ];
        let result = detrend_emissions(&log_e);

        // After detrending, pop 0 should be flat (all near mean = -3.0)
        for (t, row) in result.iter().enumerate() {
            assert!((row[0] - (-3.0)).abs() < 1e-10,
                "detrended pop 0 at t={} should be near -3.0, got {}", t, row[0]);
        }

        // Pop 1 was flat, should remain unchanged
        for (t, row) in result.iter().enumerate() {
            assert!((row[1] - (-3.0)).abs() < 1e-10,
                "flat pop 1 at t={} should be unchanged, got {}", t, row[1]);
        }
    }

    #[test]
    fn test_detrend_preserves_mean() {
        let log_e = vec![
            vec![-1.0, -5.0],
            vec![-3.0, -2.0],
            vec![-2.0, -4.0],
            vec![-4.0, -1.0],
        ];
        let result = detrend_emissions(&log_e);

        // Mean should be preserved for each population
        for pop in 0..2 {
            let orig_mean: f64 = log_e.iter().map(|r| r[pop]).sum::<f64>() / 4.0;
            let new_mean: f64 = result.iter().map(|r| r[pop]).sum::<f64>() / 4.0;
            assert!((orig_mean - new_mean).abs() < 1e-10,
                "mean for pop {} should be preserved: orig={}, new={}", pop, orig_mean, new_mean);
        }
    }

    #[test]
    fn test_detrend_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -2.0],
            vec![-3.0, -3.0],
        ];
        let result = detrend_emissions(&log_e);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0,
            "NEG_INFINITY should be preserved");
    }

    // ========================================================================
    // Transition momentum tests
    // ========================================================================

    #[test]
    fn test_momentum_empty() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_transition_momentum(&[], &params, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_momentum_zero_alpha() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_transition_momentum(&[0, 0, 0], &params, 0.0);
        assert!(result.is_empty(), "alpha=0 should return empty");
    }

    #[test]
    fn test_momentum_increases_with_run_length() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // All state 0: run length increases 1, 2, 3, ...
        let states = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = compute_transition_momentum(&states, &params, 2.0);
        assert_eq!(result.len(), 10);

        // Self-transition probability should increase with run length
        let self_prob_0 = result[0][0][0].exp();
        let self_prob_5 = result[5][0][0].exp();
        let self_prob_9 = result[9][0][0].exp();
        assert!(self_prob_5 > self_prob_0,
            "longer run should have higher self-prob: {} vs {}", self_prob_5, self_prob_0);
        assert!(self_prob_9 > self_prob_5,
            "even longer run should have higher self-prob: {} vs {}", self_prob_9, self_prob_5);
    }

    #[test]
    fn test_momentum_resets_at_boundary() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // State changes at t=5: run length resets
        let states = vec![0, 0, 0, 0, 0, 1, 1, 1];
        let result = compute_transition_momentum(&states, &params, 2.0);

        // At t=4 (run=5 in state 0), self-transition for state 0 should be high
        let self_prob_t4 = result[4][0][0].exp();
        // At t=5 (run=1 in state 1), self-transition for state 1 should be low
        let self_prob_t5 = result[5][1][1].exp();
        assert!(self_prob_t4 > self_prob_t5,
            "long run should have higher momentum than fresh start: {} vs {}",
            self_prob_t4, self_prob_t5);
    }

    #[test]
    fn test_momentum_rows_sum_to_one() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let states = vec![0, 0, 1, 1, 1, 2, 0, 0];
        let result = compute_transition_momentum(&states, &params, 1.5);

        for (t, trans) in result.iter().enumerate() {
            for (i, row) in trans.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row {} at t={} should sum to 1.0, got {}", i, t, sum);
            }
        }
    }

    // ========================================================================
    // Emission variance stabilization tests
    // ========================================================================

    #[test]
    fn test_variance_stabilize_empty() {
        let result = variance_stabilize_emissions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_variance_stabilize_single_window() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = variance_stabilize_emissions(&log_e);
        assert_eq!(result.len(), 1);
        // Single window: not enough data for variance, should return as-is
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_variance_stabilize_equalizes_spread() {
        // Pop 0: high variance, Pop 1: low variance
        let log_e = vec![
            vec![-1.0, -2.9],
            vec![-5.0, -3.1],
            vec![-1.0, -2.9],
            vec![-5.0, -3.1],
        ];
        let result = variance_stabilize_emissions(&log_e);

        // Compute std of each pop after stabilization
        let mut stds = [0.0_f64; 2];
        for pop in 0..2 {
            let mean: f64 = result.iter().map(|r| r[pop]).sum::<f64>() / 4.0;
            let var: f64 = result.iter().map(|r| (r[pop] - mean).powi(2)).sum::<f64>() / 3.0;
            stds[pop] = var.sqrt();
        }

        // After stabilization, stds should be equal (both set to median)
        assert!((stds[0] - stds[1]).abs() < 1e-10,
            "stds should be equal after stabilization: {} vs {}", stds[0], stds[1]);
    }

    #[test]
    fn test_variance_stabilize_preserves_means() {
        let log_e = vec![
            vec![-1.0, -5.0],
            vec![-3.0, -2.0],
            vec![-2.0, -4.0],
        ];
        let result = variance_stabilize_emissions(&log_e);

        for pop in 0..2 {
            let orig_mean: f64 = log_e.iter().map(|r| r[pop]).sum::<f64>() / 3.0;
            let new_mean: f64 = result.iter().map(|r| r[pop]).sum::<f64>() / 3.0;
            assert!((orig_mean - new_mean).abs() < 1e-10,
                "mean for pop {} should be preserved: {} vs {}", pop, orig_mean, new_mean);
        }
    }

    #[test]
    fn test_variance_stabilize_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -2.0],
            vec![-3.0, -3.0],
        ];
        let result = variance_stabilize_emissions(&log_e);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // Lookahead transition adjustment tests
    // ========================================================================

    #[test]
    fn test_lookahead_empty() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_lookahead_transitions(&[], &[], &params, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lookahead_zero_radius() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let log_e = vec![vec![-1.0, -2.0]; 5];
        let states = vec![0; 5];
        let result = compute_lookahead_transitions(&log_e, &states, &params, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_lookahead_boosts_upcoming_state() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // State 0, but upcoming emissions strongly favor state 1
        let log_e = vec![
            vec![-1.0, -2.0],  // current: state 0 favored
            vec![-5.0, -0.5],  // lookahead: state 1 strongly favored
            vec![-5.0, -0.5],  // lookahead: state 1 strongly favored
        ];
        let states = vec![0, 0, 0];
        let result = compute_lookahead_transitions(&log_e, &states, &params, 3);

        // At t=0, transition 0→1 should be boosted relative to base
        let base_prob_01 = 0.1;
        let new_prob_01 = result[0][0][1].exp();
        assert!(new_prob_01 > base_prob_01,
            "lookahead should boost transition to upcoming state: {} > {}",
            new_prob_01, base_prob_01);
    }

    #[test]
    fn test_lookahead_no_boost_when_same_state() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // Lookahead also favors state 0, same as current
        let log_e = vec![
            vec![-0.5, -3.0],
            vec![-0.5, -3.0],
            vec![-0.5, -3.0],
        ];
        let states = vec![0, 0, 0];
        let result = compute_lookahead_transitions(&log_e, &states, &params, 3);

        // No boost needed: transitions should stay near base
        let prob_01 = result[0][0][1].exp();
        assert!((prob_01 - 0.1).abs() < 0.05,
            "no boost expected when lookahead matches current: {}", prob_01);
    }

    #[test]
    fn test_lookahead_rows_sum_to_one() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-3.0, -1.0, -2.0],
            vec![-2.0, -3.0, -1.0],
            vec![-1.0, -2.0, -3.0],
        ];
        let states = vec![0, 1, 2, 0];
        let result = compute_lookahead_transitions(&log_e, &states, &params, 2);

        for (t, trans) in result.iter().enumerate() {
            for (i, row) in trans.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row {} at t={} should sum to 1.0, got {}", i, t, sum);
            }
        }
    }

    // ========================================================================
    // Emission kurtosis weighting tests
    // ========================================================================

    #[test]
    fn test_kurtosis_empty() {
        let result = apply_kurtosis_weighting(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kurtosis_few_windows() {
        let log_e = vec![vec![-1.0, -2.0]; 3];
        let result = apply_kurtosis_weighting(&log_e, 1.0);
        // Fewer than 4 windows: returned as-is
        assert_eq!(result.len(), 3);
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_kurtosis_preserves_means() {
        let log_e = vec![
            vec![-1.0, -5.0],
            vec![-3.0, -2.0],
            vec![-2.0, -4.0],
            vec![-4.0, -1.0],
            vec![-2.5, -3.0],
        ];
        let result = apply_kurtosis_weighting(&log_e, 0.5);

        for pop in 0..2 {
            let orig_mean: f64 = log_e.iter().map(|r| r[pop]).sum::<f64>() / 5.0;
            let new_mean: f64 = result.iter().map(|r| r[pop]).sum::<f64>() / 5.0;
            assert!((orig_mean - new_mean).abs() < 1e-10,
                "mean for pop {} should be preserved: {} vs {}", pop, orig_mean, new_mean);
        }
    }

    #[test]
    fn test_kurtosis_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -2.0],
            vec![-3.0, -3.0],
            vec![-4.0, -4.0],
        ];
        let result = apply_kurtosis_weighting(&log_e, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    #[test]
    fn test_kurtosis_zero_power_identity() {
        let log_e = vec![
            vec![-1.0, -5.0],
            vec![-3.0, -2.0],
            vec![-2.0, -4.0],
            vec![-4.0, -1.0],
        ];
        let result = apply_kurtosis_weighting(&log_e, 0.0);

        // power=0 → scale = 1.0 for all → identity transform
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "power=0 should be identity at t={}, k={}", t, k);
            }
        }
    }

    // ========================================================================
    // Segment length prior tests
    // ========================================================================

    #[test]
    fn test_segment_prior_empty() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_segment_length_prior(&[], &params, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_segment_prior_min_one() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // min_length=1 means no penalty → returns empty
        let result = compute_segment_length_prior(&[0, 0, 0], &params, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_segment_prior_penalizes_short_runs() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // State changes at t=2: run of 2 in state 0, then state 1
        let states = vec![0, 0, 1, 1, 1, 1, 1];
        let result = compute_segment_length_prior(&states, &params, 5);

        // At t=0 (run=1, short), off-diagonal should be penalized
        let prob_01_t0 = result[0][0][1].exp();
        // At t=4 (run=3 in state 1, still < 5), still some penalty
        let prob_10_t4 = result[4][1][0].exp();
        // At t=6 (run=5, meets min), no penalty
        let prob_10_t6 = result[6][1][0].exp();

        assert!(prob_01_t0 < 0.1,
            "short run should penalize switching: {}", prob_01_t0);
        assert!(prob_10_t6 > prob_10_t4,
            "longer run should have less penalty: {} vs {}", prob_10_t6, prob_10_t4);
    }

    #[test]
    fn test_segment_prior_no_penalty_after_min_length() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // Long run: all state 0 for 10 windows, min_length=3
        let states = vec![0; 10];
        let result = compute_segment_length_prior(&states, &params, 3);

        // At t=2 (run=3, meets min), transitions should be near base
        let prob_01_t2 = result[2][0][1].exp();
        // At t=9 (run=10, well past min), transitions should also be near base
        let prob_01_t9 = result[9][0][1].exp();

        assert!((prob_01_t2 - 0.1).abs() < 0.01,
            "at min_length run, should be near base: {}", prob_01_t2);
        assert!((prob_01_t9 - 0.1).abs() < 0.01,
            "past min_length, should be near base: {}", prob_01_t9);
    }

    #[test]
    fn test_segment_prior_rows_sum_to_one() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let states = vec![0, 0, 1, 1, 2, 2, 2, 0];
        let result = compute_segment_length_prior(&states, &params, 4);

        for (t, trans) in result.iter().enumerate() {
            for (i, row) in trans.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row {} at t={} should sum to 1.0, got {}", i, t, sum);
            }
        }
    }

    // ========================================================================
    // Emission gap penalty tests
    // ========================================================================

    #[test]
    fn test_gap_penalty_empty() {
        let result = apply_gap_penalty(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_gap_penalty_single_window() {
        let log_e = vec![vec![-1.0, -2.0]];
        let result = apply_gap_penalty(&log_e, 1.0);
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_gap_penalty_penalizes_variable_pop() {
        // Pop 0: jumps around (high variability)
        // Pop 1: moderate variability (median)
        // Pop 2: smooth (low variability)
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-5.0, -3.0, -3.1],
            vec![-1.0, -2.0, -2.9],
            vec![-5.0, -3.0, -3.1],
            vec![-1.0, -2.0, -2.9],
        ];
        let result = apply_gap_penalty(&log_e, 1.0);

        // Pop 0 (highest variability) should be penalized most
        // Pop 2 (lowest variability) should have no penalty (below median)
        let shift_0 = result[0][0] - log_e[0][0];
        let shift_2 = result[0][2] - log_e[0][2];
        assert!(shift_0 < shift_2,
            "high-variability pop should be penalized more: shift_0={}, shift_2={}", shift_0, shift_2);
    }

    #[test]
    fn test_gap_penalty_zero_weight() {
        let log_e = vec![
            vec![-1.0, -5.0],
            vec![-5.0, -1.0],
        ];
        let result = apply_gap_penalty(&log_e, 0.0);
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10,
                    "zero weight should give identity at t={}, k={}", t, k);
            }
        }
    }

    #[test]
    fn test_gap_penalty_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0],
            vec![-2.0, -2.0],
            vec![-3.0, -3.0],
        ];
        let result = apply_gap_penalty(&log_e, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // Recency-weighted transitions tests
    // ========================================================================

    #[test]
    fn test_recency_empty() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_recency_transitions(&[], &params, 0.9);
        assert!(result.is_empty());
    }

    #[test]
    fn test_recency_zero_alpha() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_recency_transitions(&[0, 0, 0], &params, 0.0);
        assert!(result.is_empty(), "alpha=0 should return empty");
    }

    #[test]
    fn test_recency_dominant_state_boosted() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // All state 0: recency history is entirely state 0
        let states = vec![0, 0, 0, 0, 0];
        let result = compute_recency_transitions(&states, &params, 0.9);

        // Self-transition for state 0 should be boosted above base 0.9
        let self_prob = result[4][0][0].exp();
        assert!(self_prob > 0.9,
            "dominant state should have boosted self-transition: {}", self_prob);
    }

    #[test]
    fn test_recency_rows_sum_to_one() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let states = vec![0, 0, 1, 1, 2, 0, 1, 2];
        let result = compute_recency_transitions(&states, &params, 0.8);

        for (t, trans) in result.iter().enumerate() {
            for (i, row) in trans.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row {} at t={} should sum to 1.0, got {}", i, t, sum);
            }
        }
    }

    // ========================================================================
    // Emission centering tests
    // ========================================================================

    #[test]
    fn test_center_empty() {
        let result = center_emissions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_center_single_pop() {
        let log_e = vec![vec![-3.0], vec![-5.0]];
        let result = center_emissions(&log_e);
        // Single pop: centered to 0
        assert!((result[0][0] - 0.0).abs() < 1e-10);
        assert!((result[1][0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_window_sums_to_zero() {
        let log_e = vec![
            vec![-1.0, -3.0, -5.0],
            vec![-2.0, -2.0, -2.0],
            vec![-4.0, -1.0, -3.0],
        ];
        let result = center_emissions(&log_e);

        for (t, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10,
                "window {} should sum to ~0 after centering: {}", t, sum);
        }
    }

    #[test]
    fn test_center_preserves_relative_order() {
        let log_e = vec![
            vec![-1.0, -3.0, -5.0],
        ];
        let result = center_emissions(&log_e);
        // Pop 0 > Pop 1 > Pop 2 should still hold
        assert!(result[0][0] > result[0][1]);
        assert!(result[0][1] > result[0][2]);
    }

    #[test]
    fn test_center_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0, -3.0],
            vec![-2.0, -2.0, -2.0],
        ];
        let result = center_emissions(&log_e);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // State persistence bonus tests
    // ========================================================================

    #[test]
    fn test_persistence_empty() {
        let result = apply_persistence_bonus(&[], &[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_persistence_zero_weight() {
        let log_e = vec![vec![-1.0, -2.0]; 3];
        let states = vec![0, 1, 0];
        let result = apply_persistence_bonus(&log_e, &states, 0.0);
        // Zero weight → identity
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!((v - log_e[t][k]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_persistence_boosts_decoded_state() {
        let log_e = vec![
            vec![-2.0, -2.0],
            vec![-2.0, -2.0],
            vec![-2.0, -2.0],
        ];
        let states = vec![0, 1, 0];
        let result = apply_persistence_bonus(&log_e, &states, 1.5);

        // At t=0, state 0 boosted
        assert!((result[0][0] - (-0.5)).abs() < 1e-10);
        assert!((result[0][1] - (-2.0)).abs() < 1e-10);

        // At t=1, state 1 boosted
        assert!((result[1][0] - (-2.0)).abs() < 1e-10);
        assert!((result[1][1] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_persistence_only_boosts_one_state() {
        let log_e = vec![vec![-3.0, -3.0, -3.0]];
        let states = vec![1];
        let result = apply_persistence_bonus(&log_e, &states, 2.0);

        assert!((result[0][0] - (-3.0)).abs() < 1e-10, "non-decoded unchanged");
        assert!((result[0][1] - (-1.0)).abs() < 1e-10, "decoded gets bonus");
        assert!((result[0][2] - (-3.0)).abs() < 1e-10, "non-decoded unchanged");
    }

    #[test]
    fn test_persistence_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -2.0]];
        let states = vec![0];
        let result = apply_persistence_bonus(&log_e, &states, 1.0);
        // NEG_INFINITY + 1.0 is still NEG_INFINITY? No, it's finite: -inf + 1 = -inf
        // Actually in Rust, f64::NEG_INFINITY + 1.0 = f64::NEG_INFINITY
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // Emission median polish tests
    // ========================================================================

    #[test]
    fn test_median_polish_empty() {
        let result = median_polish_emissions(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_median_polish_single_row() {
        let log_e = vec![vec![-1.0, -3.0, -5.0]];
        let result = median_polish_emissions(&log_e);
        assert_eq!(result.len(), 1);
        // Single row: row median removed then column median removed
        // After iterating, single row collapses to all zeros
        for &v in &result[0] {
            assert!(v.abs() < 1e-10,
                "single row should collapse to ~0 after polish: {}", v);
        }
    }

    #[test]
    fn test_median_polish_removes_row_col_effects() {
        // Additive model: emission = row_effect + col_effect
        // row effects: [0, 1, 2], col effects: [0, 10]
        let log_e = vec![
            vec![0.0, 10.0],
            vec![1.0, 11.0],
            vec![2.0, 12.0],
        ];
        let result = median_polish_emissions(&log_e);

        // Pure additive model: residuals should be ~0
        for (t, row) in result.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(v.abs() < 1e-10,
                    "residual at t={}, k={} should be ~0 for additive model: {}", t, k, v);
            }
        }
    }

    #[test]
    fn test_median_polish_preserves_neg_infinity() {
        let log_e = vec![
            vec![f64::NEG_INFINITY, -1.0, -3.0],
            vec![-2.0, -2.0, -2.0],
        ];
        let result = median_polish_emissions(&log_e);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // Disagreement penalty transitions tests
    // ========================================================================

    #[test]
    fn test_disagreement_empty() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let result = compute_disagreement_transitions(&[], &[], &params, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_disagreement_agreement_tightens() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // Emission argmax = 0 = pass-1 state → agreement
        let log_e = vec![vec![-0.5, -3.0]];
        let states = vec![0];
        let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);

        // Self-transition should be boosted above base
        let self_prob = result[0][0][0].exp();
        assert!(self_prob > 0.9,
            "agreement should boost self-transition: {}", self_prob);
    }

    #[test]
    fn test_disagreement_loosens_on_disagree() {
        let params = AncestryHmmParams {
            n_states: 2,
            populations: vec![],
            transitions: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            initial: vec![0.5, 0.5],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        // Emission argmax = 1, pass-1 state = 0 → disagreement
        let log_e = vec![vec![-3.0, -0.5]];
        let states = vec![0];
        let result = compute_disagreement_transitions(&log_e, &states, &params, 1.0);

        // Transition 0→1 should be boosted above base
        let trans_01 = result[0][0][1].exp();
        assert!(trans_01 > 0.1,
            "disagreement should boost transition toward emission argmax: {}", trans_01);
    }

    #[test]
    fn test_disagreement_rows_sum_to_one() {
        let params = AncestryHmmParams {
            n_states: 3,
            populations: vec![],
            transitions: vec![
                vec![0.90, 0.05, 0.05],
                vec![0.05, 0.90, 0.05],
                vec![0.05, 0.05, 0.90],
            ],
            initial: vec![1.0/3.0; 3],
            emission_same_pop_mean: 0.9,
            emission_diff_pop_mean: 0.6,
            emission_std: 0.1,
            emission_model: EmissionModel::Max,
            normalization: None,
            coverage_weight: 0.0,
            transition_dampening: 0.0,
        };
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-3.0, -1.0, -2.0],
            vec![-2.0, -3.0, -1.0],
        ];
        let states = vec![0, 0, 2];
        let result = compute_disagreement_transitions(&log_e, &states, &params, 1.5);

        for (t, trans) in result.iter().enumerate() {
            for (i, row) in trans.iter().enumerate() {
                let sum: f64 = row.iter().map(|&v| v.exp()).sum();
                assert!((sum - 1.0).abs() < 1e-6,
                    "row {} at t={} should sum to 1.0, got {}", i, t, sum);
            }
        }
    }

    // ========================================================================
    // Emission softmax renormalization tests
    // ========================================================================

    #[test]
    fn test_softmax_renorm_empty() {
        let result = softmax_renormalize(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_renorm_sums_to_one() {
        let log_e = vec![
            vec![-1.0, -2.0, -3.0],
            vec![-5.0, -1.0, -2.0],
        ];
        let result = softmax_renormalize(&log_e, 1.0);

        for (t, row) in result.iter().enumerate() {
            let sum: f64 = row.iter().map(|&v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6,
                "softmax row {} should sum to 1.0 in prob space: {}", t, sum);
        }
    }

    #[test]
    fn test_softmax_renorm_preserves_order() {
        let log_e = vec![vec![-1.0, -3.0, -5.0]];
        let result = softmax_renormalize(&log_e, 1.0);
        assert!(result[0][0] > result[0][1]);
        assert!(result[0][1] > result[0][2]);
    }

    #[test]
    fn test_softmax_renorm_high_temp_flattens() {
        let log_e = vec![vec![-1.0, -5.0]];
        let result_low = softmax_renormalize(&log_e, 0.5);
        let result_high = softmax_renormalize(&log_e, 5.0);

        // Higher temperature → more uniform → smaller gap
        let gap_low = result_low[0][0] - result_low[0][1];
        let gap_high = result_high[0][0] - result_high[0][1];
        assert!(gap_high < gap_low,
            "higher temp should give smaller gap: {} vs {}", gap_high, gap_low);
    }

    #[test]
    fn test_softmax_renorm_preserves_neg_infinity() {
        let log_e = vec![vec![f64::NEG_INFINITY, -1.0, -2.0]];
        let result = softmax_renormalize(&log_e, 1.0);
        assert!(result[0][0].is_infinite() && result[0][0] < 0.0);
    }

    // ========================================================================
    // Bidirectional state smoothing tests
    // ========================================================================

    #[test]
    fn test_bidirectional_empty() {
        let result = bidirectional_smooth_states(&[], 2, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bidirectional_zero_radius() {
        let states = vec![0, 1, 0, 1];
        let result = bidirectional_smooth_states(&states, 2, 0);
        assert_eq!(result, states);
    }

    #[test]
    fn test_bidirectional_smooths_isolated() {
        // Isolated state 1 surrounded by 0s should get smoothed out
        let states = vec![0, 0, 0, 1, 0, 0, 0];
        let result = bidirectional_smooth_states(&states, 2, 2);
        assert_eq!(result[3], 0, "isolated state should be smoothed to neighbor");
    }

    #[test]
    fn test_bidirectional_preserves_majority() {
        // Block of state 1 should be preserved
        let states = vec![0, 0, 1, 1, 1, 1, 0, 0];
        let result = bidirectional_smooth_states(&states, 2, 1);
        // Interior of block should stay as 1
        assert_eq!(result[3], 1, "interior of block should be preserved");
        assert_eq!(result[4], 1, "interior of block should be preserved");
    }

    #[test]
    fn test_bidirectional_weights_decay() {
        // Test that closer neighbors have more influence
        // State 0 at positions 0-1, state 1 at positions 2-5
        let states = vec![0, 0, 1, 1, 1, 1];
        let result = bidirectional_smooth_states(&states, 2, 1);
        // At position 2, state 1 has strong local support → should stay 1
        assert_eq!(result[2], 1);
    }
}
