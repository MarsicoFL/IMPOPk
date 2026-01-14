//! Hidden Markov Model for IBD State Inference
//!
//! This module implements a two-state Hidden Markov Model (HMM) for distinguishing
//! IBD (Identity-By-Descent) from non-IBD regions based on sequence identity observations.
//!
//! ## Model Overview
//!
//! The HMM uses two hidden states:
//! - **State 0 (Non-IBD)**: Haplotypes do not share recent common ancestry
//! - **State 1 (IBD)**: Haplotypes share recent common ancestry
//!
//! Observations are sequence identity values in the range [0, 1], where values
//! close to 1 indicate near-identical sequences and values around 0.5 indicate
//! random similarity.
//!
//! ## Algorithm
//!
//! The module implements:
//! 1. **Parameter estimation**: Automatically estimate emission distributions from data
//!    using k-means clustering
//! 2. **Viterbi algorithm**: Find the most likely state sequence given observations
//! 3. **Segment extraction**: Convert state sequences into IBD segment coordinates
//!
//! ## Example
//!
//! ```rust
//! use hprc_ibd::hmm::{HmmParams, viterbi, extract_ibd_segments};
//!
//! // Identity observations from sliding windows
//! let observations = vec![
//!     0.5, 0.6, 0.55,  // Non-IBD region
//!     0.999, 0.998, 0.9995, 0.999,  // IBD region
//!     0.5, 0.4,  // Non-IBD region
//! ];
//!
//! // Create HMM with expected 50-window IBD segments
//! let mut params = HmmParams::from_expected_length(50.0, 0.0001);
//!
//! // Estimate emission distributions from observed data
//! params.estimate_emissions(&observations);
//!
//! // Run Viterbi to get state sequence
//! let states = viterbi(&observations, &params);
//!
//! // Extract IBD segments
//! let segments = extract_ibd_segments(&states);
//! for (start, end, n_windows) in segments {
//!     println!("IBD segment: windows {}-{} ({} windows)", start, end, n_windows);
//! }
//! ```

use crate::stats::{kmeans_1d, GaussianParams};

/// Parameters for the two-state IBD Hidden Markov Model.
///
/// The HMM is parameterized by:
/// - Initial state probabilities
/// - State transition probabilities
/// - Emission distributions (Gaussian) for each state
///
/// ## States
///
/// - State 0: Non-IBD (background/random similarity)
/// - State 1: IBD (shared ancestry)
///
/// ## Transition Matrix Layout
///
/// ```text
/// transition[from][to]:
///   transition[0][0] = P(stay in non-IBD)
///   transition[0][1] = P(enter IBD from non-IBD)
///   transition[1][0] = P(exit IBD to non-IBD)
///   transition[1][1] = P(stay in IBD)
/// ```
///
/// ## Example
///
/// ```rust
/// use hprc_ibd::hmm::HmmParams;
///
/// // Create parameters expecting 50-window IBD segments
/// let params = HmmParams::from_expected_length(50.0, 0.0001);
///
/// // Check transition probabilities
/// assert!(params.transition[1][1] > 0.9); // High probability to stay in IBD
/// ```
#[derive(Debug, Clone)]
pub struct HmmParams {
    /// Initial state probabilities: [P(non-IBD), P(IBD)]
    pub initial: [f64; 2],
    /// Transition matrix: transition[from_state][to_state]
    pub transition: [[f64; 2]; 2],
    /// Emission distributions: [non-IBD Gaussian, IBD Gaussian]
    pub emission: [GaussianParams; 2],
}

impl HmmParams {
    /// Create HMM parameters from expected IBD segment length.
    ///
    /// This constructor derives transition probabilities from the expected
    /// segment length, which determines how "sticky" the IBD state is.
    ///
    /// ## Parameters
    ///
    /// - `expected_ibd_windows`: Expected number of consecutive windows in an IBD segment.
    ///   Higher values make the model expect longer segments.
    /// - `p_enter_ibd`: Probability of transitioning from non-IBD to IBD state.
    ///   Lower values make IBD calls more conservative.
    ///
    /// ## Transition Probability Calculation
    ///
    /// ```text
    /// p_stay_ibd = 1 - 1/expected_ibd_windows
    /// p_exit_ibd = 1 - p_stay_ibd
    /// ```
    ///
    /// The `p_stay_ibd` is clamped to [0.5, 0.9999] for numerical stability.
    ///
    /// ## Default Emission Distributions
    ///
    /// - Non-IBD: Gaussian(mean=0.5, std=0.2) - random similarity
    /// - IBD: Gaussian(mean=0.99, std=0.01) - high identity
    ///
    /// Use [`estimate_emissions`](Self::estimate_emissions) to adapt these to your data.
    ///
    /// ## Panics
    ///
    /// Panics if `p_enter_ibd` is not in the open interval (0, 1).
    ///
    /// ## Example
    ///
    /// ```rust
    /// use hprc_ibd::hmm::HmmParams;
    ///
    /// // Conservative settings: expect long segments, rare IBD transitions
    /// let params = HmmParams::from_expected_length(100.0, 0.00001);
    ///
    /// // Sensitive settings: expect shorter segments, easier IBD transitions
    /// let params = HmmParams::from_expected_length(20.0, 0.001);
    /// ```
    pub fn from_expected_length(expected_ibd_windows: f64, p_enter_ibd: f64) -> Self {
        assert!(
            p_enter_ibd > 0.0 && p_enter_ibd < 1.0,
            "p_enter_ibd must be in range (0, 1), got {}",
            p_enter_ibd
        );

        let p_stay_ibd = 1.0 - 1.0 / expected_ibd_windows;
        let p_stay_ibd = p_stay_ibd.clamp(0.5, 0.9999);
        let p_exit_ibd = 1.0 - p_stay_ibd;

        HmmParams {
            initial: [0.99, 0.01],
            transition: [
                [1.0 - p_enter_ibd, p_enter_ibd],
                [p_exit_ibd, p_stay_ibd],
            ],
            emission: [
                GaussianParams { mean: 0.5, std: 0.2 },
                GaussianParams { mean: 0.99, std: 0.01 },
            ],
        }
    }

    /// Estimate emission distributions from observed data using k-means clustering.
    ///
    /// This method adapts the emission Gaussians to the actual distribution of
    /// identity values in the data, improving HMM accuracy for different datasets.
    ///
    /// ## Algorithm
    ///
    /// 1. Cluster observations into two groups using k-means
    /// 2. Compute mean and standard deviation for each cluster
    /// 3. Assign lower cluster to non-IBD state, higher to IBD state
    ///
    /// If k-means fails (e.g., insufficient variance), falls back to quantile-based
    /// estimation using the 30th and 90th percentiles.
    ///
    /// ## Requirements
    ///
    /// - Requires at least 3 observations
    /// - Data must have non-trivial variance (> 1e-12)
    ///
    /// If these conditions are not met, emissions remain unchanged.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use hprc_ibd::hmm::HmmParams;
    ///
    /// let mut params = HmmParams::from_expected_length(50.0, 0.0001);
    ///
    /// // Observations with clear two-cluster structure
    /// let observations = vec![
    ///     0.5, 0.6, 0.55, 0.45,  // Non-IBD cluster
    ///     0.999, 0.998, 0.9995,  // IBD cluster
    /// ];
    ///
    /// params.estimate_emissions(&observations);
    ///
    /// // Emissions are now adapted to the data
    /// assert!(params.emission[0].mean < 0.7);  // Low cluster
    /// assert!(params.emission[1].mean > 0.99); // High cluster
    /// ```
    pub fn estimate_emissions(&mut self, observations: &[f64]) {
        if observations.len() < 3 {
            return;
        }

        let variance: f64 = {
            let mean = observations.iter().sum::<f64>() / observations.len() as f64;
            observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / observations.len() as f64
        };

        if variance < 1e-12 {
            return;
        }

        match kmeans_1d(observations, 2, 20) {
            Some((centers, assignments)) => {
                let idx_low = if centers[0] < centers[1] { 0 } else { 1 };

                let mut sum_low = 0.0;
                let mut sum_high = 0.0;
                let mut sq_sum_low = 0.0;
                let mut sq_sum_high = 0.0;
                let mut n_low = 0;
                let mut n_high = 0;

                for (obs, &cluster) in observations.iter().zip(assignments.iter()) {
                    if cluster == idx_low {
                        sum_low += obs;
                        sq_sum_low += obs * obs;
                        n_low += 1;
                    } else {
                        sum_high += obs;
                        sq_sum_high += obs * obs;
                        n_high += 1;
                    }
                }

                if n_low > 0 {
                    let mean = sum_low / n_low as f64;
                    let var = (sq_sum_low / n_low as f64) - mean * mean;
                    self.emission[0] = GaussianParams {
                        mean,
                        std: var.sqrt().max(0.01),
                    };
                }

                if n_high > 0 {
                    let mean = sum_high / n_high as f64;
                    let var = (sq_sum_high / n_high as f64) - mean * mean;
                    self.emission[1] = GaussianParams {
                        mean,
                        std: var.sqrt().max(0.001),
                    };
                }
            }
            None => {
                let mut sorted = observations.to_vec();
                // Use total_cmp instead of partial_cmp to handle NaN values safely
                sorted.sort_by(|a, b| a.total_cmp(b));

                let q30_idx = (sorted.len() as f64 * 0.3) as usize;
                let q90_idx = (sorted.len() as f64 * 0.9) as usize;

                self.emission[0].mean = sorted[q30_idx];
                self.emission[1].mean = sorted[q90_idx.min(sorted.len() - 1)];

                let overall_std = variance.sqrt();
                self.emission[0].std = overall_std.max(0.05);
                self.emission[1].std = overall_std.max(0.01);
            }
        }
    }
}

/// Find the most likely state sequence using the Viterbi algorithm.
///
/// The Viterbi algorithm is a dynamic programming algorithm that finds the
/// single best state sequence (global decoding) given a sequence of observations
/// and HMM parameters.
///
/// ## Algorithm
///
/// For each position t, computes:
/// ```text
/// delta[t][s] = max_{prev} { delta[t-1][prev] * P(prev->s) * P(obs[t]|s) }
/// ```
///
/// All computations are performed in log-space for numerical stability.
///
/// ## Arguments
///
/// - `observations`: Sequence of identity values (one per window)
/// - `params`: HMM parameters (transition and emission distributions)
///
/// ## Returns
///
/// Vector of states (0=non-IBD, 1=IBD) with one entry per observation.
/// Returns empty vector if `observations` is empty.
///
/// ## Example
///
/// ```rust
/// use hprc_ibd::hmm::{HmmParams, viterbi};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001);
///
/// // Clear transition: low -> high -> low
/// let observations = vec![0.5, 0.6, 0.999, 0.998, 0.997, 0.5, 0.4];
/// let states = viterbi(&observations, &params);
///
/// // First windows should be non-IBD
/// assert_eq!(states[0], 0);
/// assert_eq!(states[1], 0);
///
/// // Middle windows should be IBD
/// assert_eq!(states[2], 1);
/// assert_eq!(states[3], 1);
/// assert_eq!(states[4], 1);
/// ```
///
/// ## Performance
///
/// Time complexity: O(n * k^2) where n = observations.len() and k = 2 (states)
/// Space complexity: O(n * k) for delta and psi matrices
pub fn viterbi(observations: &[f64], params: &HmmParams) -> Vec<usize> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];
    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut delta: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut psi: Vec<[usize; 2]> = Vec::with_capacity(n);

    delta.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);
    psi.push([0, 0]);

    for t in 1..n {
        let mut dt = [f64::NEG_INFINITY; 2];
        let mut pt = [0usize; 2];

        for s in 0..2 {
            for prev in 0..2 {
                let score = delta[t - 1][prev] + log_trans[prev][s] + log_emit[t][s];
                if score > dt[s] {
                    dt[s] = score;
                    pt[s] = prev;
                }
            }
        }

        delta.push(dt);
        psi.push(pt);
    }

    let mut states = vec![0usize; n];
    states[n - 1] = if delta[n - 1][1] > delta[n - 1][0] { 1 } else { 0 };

    for t in (0..n - 1).rev() {
        states[t] = psi[t + 1][states[t + 1]];
    }

    states
}

/// Extract contiguous IBD segments from a state sequence.
///
/// Scans through the state sequence produced by [`viterbi`] and identifies
/// contiguous runs of IBD state (state = 1).
///
/// ## Arguments
///
/// - `states`: State sequence from Viterbi algorithm (0=non-IBD, 1=IBD)
///
/// ## Returns
///
/// Vector of tuples `(start_idx, end_idx, n_windows)` where:
/// - `start_idx`: First window index of the IBD segment (inclusive)
/// - `end_idx`: Last window index of the IBD segment (inclusive)
/// - `n_windows`: Number of windows in the segment
///
/// ## Example
///
/// ```rust
/// use hprc_ibd::hmm::extract_ibd_segments;
///
/// // State sequence with two IBD regions
/// let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
///
/// let segments = extract_ibd_segments(&states);
///
/// assert_eq!(segments.len(), 2);
/// assert_eq!(segments[0], (2, 4, 3));  // Windows 2-4, 3 windows
/// assert_eq!(segments[1], (7, 8, 2));  // Windows 7-8, 2 windows
/// ```
///
/// ## Notes
///
/// - Returns empty vector if input is empty or contains no IBD windows
/// - Single IBD windows are returned as segments with n_windows = 1
/// - Segments at the end of the sequence are properly handled
pub fn extract_ibd_segments(states: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut segments = Vec::new();
    let n = states.len();

    if n == 0 {
        return segments;
    }

    let mut in_ibd = false;
    let mut start_idx = 0;

    for (i, &state) in states.iter().enumerate() {
        if state == 1 && !in_ibd {
            in_ibd = true;
            start_idx = i;
        } else if state == 0 && in_ibd {
            in_ibd = false;
            let n_windows = i - start_idx;
            segments.push((start_idx, i - 1, n_windows));
        }
    }

    if in_ibd {
        let n_windows = n - start_idx;
        segments.push((start_idx, n - 1, n_windows));
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi_simple() {
        let params = HmmParams::from_expected_length(10.0, 0.001);
        let obs = vec![0.5, 0.6, 0.99, 0.995, 0.998, 0.5, 0.4];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 7);
    }

    #[test]
    fn test_extract_segments() {
        let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], (2, 4, 3));
        assert_eq!(segments[1], (7, 8, 2));
    }

    #[test]
    #[should_panic(expected = "p_enter_ibd must be in range (0, 1)")]
    fn test_p_enter_ibd_zero_panics() {
        // p_enter_ibd = 0 is invalid (must be > 0)
        let _ = HmmParams::from_expected_length(10.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "p_enter_ibd must be in range (0, 1)")]
    fn test_p_enter_ibd_one_panics() {
        // p_enter_ibd = 1 is invalid (must be < 1)
        let _ = HmmParams::from_expected_length(10.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "p_enter_ibd must be in range (0, 1)")]
    fn test_p_enter_ibd_negative_panics() {
        // p_enter_ibd < 0 is invalid
        let _ = HmmParams::from_expected_length(10.0, -0.1);
    }

    #[test]
    fn test_p_enter_ibd_valid_values() {
        // These should all succeed without panicking
        let _ = HmmParams::from_expected_length(10.0, 0.001);
        let _ = HmmParams::from_expected_length(10.0, 0.5);
        let _ = HmmParams::from_expected_length(10.0, 0.999);
    }

    // === Edge case tests for Viterbi algorithm ===

    #[test]
    fn test_viterbi_empty_observations() {
        let params = HmmParams::from_expected_length(10.0, 0.001);
        let obs: Vec<f64> = vec![];
        let states = viterbi(&obs, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_viterbi_single_observation() {
        let params = HmmParams::from_expected_length(10.0, 0.001);

        // Single high identity observation
        let obs_high = vec![0.995];
        let states_high = viterbi(&obs_high, &params);
        assert_eq!(states_high.len(), 1);
        // With default params, very high identity should be classified as IBD (state 1)
        assert_eq!(states_high[0], 1);

        // Single low identity observation
        let obs_low = vec![0.5];
        let states_low = viterbi(&obs_low, &params);
        assert_eq!(states_low.len(), 1);
        // Low identity should be non-IBD (state 0)
        assert_eq!(states_low[0], 0);
    }

    #[test]
    fn test_viterbi_all_high_identity() {
        // All observations indicate IBD
        let params = HmmParams::from_expected_length(10.0, 0.001);
        let obs = vec![0.995, 0.998, 0.999, 0.997, 0.996, 0.998, 0.999, 0.995];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 8);
        // All should be IBD (state 1) due to high identity values
        for (i, &state) in states.iter().enumerate() {
            assert_eq!(state, 1, "Expected IBD at position {}", i);
        }
    }

    #[test]
    fn test_viterbi_all_low_identity() {
        // All observations indicate non-IBD
        let params = HmmParams::from_expected_length(10.0, 0.001);
        let obs = vec![0.3, 0.4, 0.5, 0.45, 0.35, 0.42, 0.38, 0.41];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 8);
        // All should be non-IBD (state 0)
        for (i, &state) in states.iter().enumerate() {
            assert_eq!(state, 0, "Expected non-IBD at position {}", i);
        }
    }

    #[test]
    fn test_viterbi_clear_state_transitions() {
        // Clear transition from non-IBD to IBD and back
        let params = HmmParams::from_expected_length(10.0, 0.001);
        // Low, Low, High, High, High, Low, Low
        let obs = vec![0.4, 0.45, 0.995, 0.998, 0.996, 0.42, 0.38];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 7);

        // First two should be non-IBD
        assert_eq!(states[0], 0);
        assert_eq!(states[1], 0);
        // Middle three should be IBD
        assert_eq!(states[2], 1);
        assert_eq!(states[3], 1);
        assert_eq!(states[4], 1);
        // Last two should be non-IBD
        assert_eq!(states[5], 0);
        assert_eq!(states[6], 0);
    }

    #[test]
    fn test_viterbi_boundary_identity_values() {
        // Test with values near the emission distribution boundaries
        let params = HmmParams::from_expected_length(10.0, 0.001);
        // Values around the decision boundary
        let obs = vec![0.75, 0.80, 0.85, 0.90, 0.95];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 5);
        // All results should be valid states (0 or 1)
        for &state in &states {
            assert!(state == 0 || state == 1);
        }
    }

    // === Edge case tests for extract_ibd_segments ===

    #[test]
    fn test_extract_ibd_segments_empty() {
        let states: Vec<usize> = vec![];
        let segments = extract_ibd_segments(&states);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_extract_ibd_segments_all_non_ibd() {
        let states = vec![0, 0, 0, 0, 0];
        let segments = extract_ibd_segments(&states);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_extract_ibd_segments_all_ibd() {
        let states = vec![1, 1, 1, 1, 1];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (0, 4, 5)); // start_idx, end_idx, n_windows
    }

    #[test]
    fn test_extract_ibd_segments_single_ibd_window() {
        let states = vec![0, 0, 1, 0, 0];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (2, 2, 1));
    }

    #[test]
    fn test_extract_ibd_segments_ibd_at_start() {
        let states = vec![1, 1, 1, 0, 0];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (0, 2, 3));
    }

    #[test]
    fn test_extract_ibd_segments_ibd_at_end() {
        let states = vec![0, 0, 1, 1, 1];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (2, 4, 3));
    }

    #[test]
    fn test_extract_ibd_segments_multiple_segments() {
        let states = vec![1, 1, 0, 0, 1, 1, 1, 0, 1];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], (0, 1, 2)); // First segment
        assert_eq!(segments[1], (4, 6, 3)); // Second segment
        assert_eq!(segments[2], (8, 8, 1)); // Third segment (at end)
    }

    #[test]
    fn test_extract_ibd_segments_alternating() {
        let states = vec![1, 0, 1, 0, 1, 0, 1];
        let segments = extract_ibd_segments(&states);
        assert_eq!(segments.len(), 4);
        // Each IBD segment is a single window
        for (i, seg) in segments.iter().enumerate() {
            let expected_idx = i * 2;
            assert_eq!(seg.0, expected_idx); // start_idx
            assert_eq!(seg.1, expected_idx); // end_idx
            assert_eq!(seg.2, 1);            // n_windows
        }
    }

    // === Edge case tests for estimate_emissions ===

    #[test]
    fn test_estimate_emissions_few_observations() {
        let mut params = HmmParams::from_expected_length(10.0, 0.001);
        let original_emission = params.emission.clone();

        // Less than 3 observations should not change emissions
        params.estimate_emissions(&[0.5, 0.9]);
        assert_eq!(params.emission[0].mean, original_emission[0].mean);
        assert_eq!(params.emission[1].mean, original_emission[1].mean);
    }

    #[test]
    fn test_estimate_emissions_identical_values() {
        let mut params = HmmParams::from_expected_length(10.0, 0.001);
        let original_emission = params.emission.clone();

        // All identical values (zero variance) should not change emissions
        let obs = vec![0.8, 0.8, 0.8, 0.8, 0.8];
        params.estimate_emissions(&obs);
        // Emissions should remain unchanged due to variance < 1e-12
        assert_eq!(params.emission[0].mean, original_emission[0].mean);
        assert_eq!(params.emission[1].mean, original_emission[1].mean);
    }

    #[test]
    fn test_estimate_emissions_two_clusters() {
        let mut params = HmmParams::from_expected_length(10.0, 0.001);

        // Clear two-cluster data
        let obs = vec![0.3, 0.35, 0.32, 0.31, 0.95, 0.96, 0.97, 0.98];
        params.estimate_emissions(&obs);

        // Low cluster should have mean around 0.32
        assert!(params.emission[0].mean < 0.5, "Low cluster mean should be < 0.5");
        // High cluster should have mean around 0.965
        assert!(params.emission[1].mean > 0.9, "High cluster mean should be > 0.9");
    }

    #[test]
    fn test_hmm_params_transition_probabilities() {
        let params = HmmParams::from_expected_length(10.0, 0.001);

        // Check initial probabilities sum to 1
        let init_sum = params.initial[0] + params.initial[1];
        assert!((init_sum - 1.0).abs() < 1e-10);

        // Check transition probabilities sum to 1 for each state
        let trans_from_0_sum = params.transition[0][0] + params.transition[0][1];
        let trans_from_1_sum = params.transition[1][0] + params.transition[1][1];
        assert!((trans_from_0_sum - 1.0).abs() < 1e-10);
        assert!((trans_from_1_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hmm_params_expected_length_clamping() {
        // Very short expected length should be clamped
        let params_short = HmmParams::from_expected_length(1.0, 0.001);
        // p_stay_ibd should be clamped to at least 0.5
        assert!(params_short.transition[1][1] >= 0.5);

        // Very long expected length
        let params_long = HmmParams::from_expected_length(100000.0, 0.001);
        // p_stay_ibd should be clamped to at most 0.9999
        assert!(params_long.transition[1][1] <= 0.9999);
    }
}
