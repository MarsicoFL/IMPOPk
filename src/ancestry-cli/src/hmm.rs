//! HMM for Local Ancestry Inference
//!
//! This module implements a Hidden Markov Model for inferring local ancestry
//! from similarity data against multiple reference populations.

use std::collections::HashMap;

/// Ancestral population definition
#[derive(Debug, Clone)]
pub struct AncestralPopulation {
    /// Population/species name
    pub name: String,
    /// Reference haplotype IDs belonging to this population
    pub haplotypes: Vec<String>,
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
}

impl AncestryHmmParams {
    /// Set the emission temperature (softmax sharpness)
    pub fn set_temperature(&mut self, temp: f64) {
        self.emission_std = temp;
    }

    /// Update switch probability and recalculate transition matrix
    pub fn set_switch_prob(&mut self, switch_prob: f64) {
        let stay_prob = 1.0 - switch_prob;
        let switch_each = switch_prob / (self.n_states - 1) as f64;

        for i in 0..self.n_states {
            for j in 0..self.n_states {
                self.transitions[i][j] = if i == j { stay_prob } else { switch_each };
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

        // Uniform initial distribution
        let initial = vec![1.0 / n_states as f64; n_states];

        // Transition matrix: high self-transition, uniform switch probability
        let stay_prob = 1.0 - switch_prob;
        let switch_each = switch_prob / (n_states - 1) as f64;

        let mut transitions = vec![vec![0.0; n_states]; n_states];
        for i in 0..n_states {
            for j in 0..n_states {
                transitions[i][j] = if i == j { stay_prob } else { switch_each };
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
        }
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
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
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
    pub fn log_emission(&self, obs: &AncestryObservation, state: usize) -> f64 {
        // Get max similarity for each population (None if no data)
        let pop_sims: Vec<Option<f64>> = self.populations.iter()
            .map(|pop| {
                let sims: Vec<f64> = pop.haplotypes.iter()
                    .filter_map(|h| obs.similarities.get(h))
                    .cloned()
                    .collect();
                if sims.is_empty() {
                    None  // No data for this population
                } else {
                    Some(sims.iter().cloned().fold(0.0_f64, f64::max))
                }
            })
            .collect();

        // Check if we have data for the target state
        let target_sim = match pop_sims[state] {
            Some(s) if s > 0.0 => s,
            _ => return f64::NEG_INFINITY,  // No data for target state
        };

        // Get only populations with actual data
        let valid_sims: Vec<f64> = pop_sims.iter()
            .filter_map(|&s| s)
            .filter(|&s| s > 0.0)
            .collect();

        // If only one population has data, it gets probability 1
        if valid_sims.len() <= 1 {
            return 0.0;  // log(1) = 0
        }

        // Use softmax with temperature parameter
        // Lower temperature = more confident (sharper distribution)
        let temperature = self.emission_std;

        // For numerical stability, subtract max before exp
        let max_sim = valid_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Log-softmax only over populations with data
        let log_sum_exp: f64 = valid_sims.iter()
            .map(|&s| ((s - max_sim) / temperature).exp())
            .sum::<f64>()
            .ln();

        (target_sim - max_sim) / temperature - log_sum_exp
    }
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
    for s in 0..k {
        v[0][s] = params.initial[s].ln() + params.log_emission(&observations[0], s);
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
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            v[t][s] = best_prob + emission;
            backptr[t][s] = best_prev;
        }
    }

    // Backtrack
    let mut states = vec![0; n];
    states[n-1] = (0..k)
        .max_by(|&a, &b| v[n-1][a].partial_cmp(&v[n-1][b]).unwrap())
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

    for s in 0..k {
        alpha[0][s] = params.initial[s].ln() + params.log_emission(&observations[0], s);
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

/// Estimate optimal temperature for softmax emissions from observed similarity differences.
/// Uses the median of (max_sim - min_sim) across populations as the temperature.
/// This makes the model adaptive to actual data signal strength.
pub fn estimate_temperature(
    observations: &[AncestryObservation],
    populations: &[AncestralPopulation],
) -> f64 {
    let mut diffs: Vec<f64> = Vec::new();

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
                diffs.push(max - min);
            }
        }
    }

    if diffs.is_empty() {
        return 0.03; // fallback default
    }

    // Use median as temperature
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = diffs[diffs.len() / 2];

    // Clamp to reasonable range
    median.clamp(0.01, 0.15)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_populations() -> Vec<AncestralPopulation> {
        vec![
            AncestralPopulation {
                name: "commissarisi".to_string(),
                haplotypes: vec!["commissarisi#HAP1".to_string(), "commissarisi#HAP2".to_string()],
            },
            AncestralPopulation {
                name: "mutica".to_string(),
                haplotypes: vec!["mutica#A".to_string(), "mutica#B".to_string()],
            },
            AncestralPopulation {
                name: "soricina".to_string(),
                haplotypes: vec!["soricina#HAP1".to_string(), "soricina#HAP2".to_string()],
            },
        ]
    }

    fn make_observation(start: u64, comm: f64, muti: f64, sori: f64) -> AncestryObservation {
        AncestryObservation {
            chrom: "super15".to_string(),
            start,
            end: start + 5000,
            sample: "TBG_5116#1".to_string(),
            similarities: [
                ("commissarisi#HAP1".to_string(), comm),
                ("commissarisi#HAP2".to_string(), comm - 0.01),
                ("mutica#A".to_string(), muti),
                ("mutica#B".to_string(), muti - 0.01),
                ("soricina#HAP1".to_string(), sori),
                ("soricina#HAP2".to_string(), sori - 0.01),
            ].into_iter().collect(),
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

        // Create observations strongly favoring commissarisi
        let obs = vec![make_observation(0, 0.98, 0.85, 0.88)];

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 0); // commissarisi
    }

    #[test]
    fn test_viterbi_with_switch() {
        let pops = make_test_populations();
        let params = AncestryHmmParams::new(pops, 0.01); // Higher switch prob for test

        // Create sequence: commissarisi -> mutica -> mutica
        let obs = vec![
            make_observation(0, 0.95, 0.80, 0.82),      // comm clear winner
            make_observation(5000, 0.95, 0.80, 0.82),   // comm
            make_observation(10000, 0.75, 0.95, 0.80),  // muti clear winner
            make_observation(15000, 0.75, 0.95, 0.80),  // muti
            make_observation(20000, 0.75, 0.95, 0.80),  // muti
        ];

        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 5);
        assert_eq!(states[0], 0); // commissarisi
        assert_eq!(states[1], 0); // commissarisi
        assert_eq!(states[2], 1); // mutica
        assert_eq!(states[3], 1); // mutica
        assert_eq!(states[4], 1); // mutica
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
            make_observation(0, 0.99, 0.50, 0.50),     // comm overwhelmingly wins
            make_observation(5000, 0.50, 0.99, 0.50),  // muti overwhelmingly wins
            make_observation(10000, 0.50, 0.50, 0.99), // sori overwhelmingly wins
        ];

        let states = viterbi(&obs, &params);
        assert_eq!(states[0], 0, "Window 0 should be commissarisi");
        assert_eq!(states[1], 1, "Window 1 should be mutica");
        assert_eq!(states[2], 2, "Window 2 should be soricina");

        let posteriors = forward_backward(&obs, &params);
        // Each window should have high posterior for the winning state
        assert!(posteriors[0][0] > 0.8, "Posterior for comm at t=0: {}", posteriors[0][0]);
        assert!(posteriors[1][1] > 0.8, "Posterior for muti at t=1: {}", posteriors[1][1]);
        assert!(posteriors[2][2] > 0.8, "Posterior for sori at t=2: {}", posteriors[2][2]);
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

        // mutica (state 1) should have highest emission
        let log_em_comm = params.log_emission(&obs, 0);
        let log_em_muti = params.log_emission(&obs, 1);
        let log_em_sori = params.log_emission(&obs, 2);

        assert!(log_em_muti > log_em_sori, "mutica emission should be > soricina");
        assert!(log_em_muti > log_em_comm, "mutica emission should be > commissarisi");
        assert!(log_em_sori > log_em_comm, "soricina emission should be > commissarisi");
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

        // Create observations with very small differences (should clamp to 0.01)
        let obs_small: Vec<AncestryObservation> = (0..10)
            .map(|i| make_observation(i * 5000, 0.90, 0.899, 0.898))
            .collect();

        let temp_small = estimate_temperature(&obs_small, &pops);
        assert!(temp_small >= 0.01, "Temperature should be clamped to >= 0.01, got {}", temp_small);

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
}
