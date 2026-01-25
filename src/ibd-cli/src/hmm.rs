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
//! close to 1 indicate near-identical sequences (both IBD and non-IBD in humans
//! have identity ~0.999 due to low nucleotide diversity).
//!
//! ## Population-Specific Parameters
//!
//! The non-IBD emission distribution depends on population-specific nucleotide
//! diversity (π). For humans:
//! - AFR: π ≈ 0.00125, so E[identity|non-IBD] ≈ 0.99875
//! - EUR: π ≈ 0.00085, so E[identity|non-IBD] ≈ 0.99915
//! - EAS: π ≈ 0.00080, so E[identity|non-IBD] ≈ 0.99920
//!
//! ## Algorithm
//!
//! The module implements:
//! 1. **Parameter estimation**: Automatically estimate emission distributions from data
//!    using k-means clustering, with population-aware priors
//! 2. **Viterbi algorithm**: Find the most likely state sequence given observations
//! 3. **Segment extraction**: Convert state sequences into IBD segment coordinates
//!
//! ## Example
//!
//! ```rust
//! use hprc_ibd::hmm::{HmmParams, Population, viterbi, extract_ibd_segments};
//!
//! // Identity observations from sliding windows
//! let observations = vec![
//!     0.998, 0.997, 0.9985,  // Non-IBD region
//!     0.9998, 0.9999, 0.9997, 0.9998,  // IBD region
//!     0.997, 0.998,  // Non-IBD region
//! ];
//!
//! // Create HMM with population-specific parameters
//! let mut params = HmmParams::from_population(
//!     Population::EUR,
//!     50.0,    // expected IBD segment length in windows
//!     0.0001,  // probability of entering IBD
//! );
//!
//! // Optionally refine emissions from observed data
//! params.estimate_emissions_robust(&observations, Some(Population::EUR));
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

/// Human population for population-specific HMM parameters.
///
/// Nucleotide diversity (π) varies between populations, affecting the expected
/// identity distribution for non-IBD haplotype pairs.
///
/// ## Population Diversity (from 1000 Genomes)
///
/// | Population | π (SNPs/bp) | E[identity] |
/// |------------|-------------|-------------|
/// | AFR | 0.00125 | 0.99875 |
/// | EUR | 0.00085 | 0.99915 |
/// | EAS | 0.00080 | 0.99920 |
/// | CSA | 0.00095 | 0.99905 |
/// | AMR | 0.00100 | 0.99900 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Population {
    /// African populations (highest diversity)
    AFR,
    /// European populations
    EUR,
    /// East Asian populations
    EAS,
    /// Central/South Asian populations
    CSA,
    /// American populations (admixed)
    AMR,
    /// Inter-population comparison (use when comparing across populations)
    InterPop,
    /// Generic/unknown population (uses conservative estimates)
    Generic,
}

impl Population {
    /// Get the nucleotide diversity (π) for this population.
    ///
    /// Values are based on 1000 Genomes Project data.
    pub fn diversity(&self) -> f64 {
        match self {
            Population::AFR => 0.00125,
            Population::EUR => 0.00085,
            Population::EAS => 0.00080,
            Population::CSA => 0.00095,
            Population::AMR => 0.00100,
            Population::InterPop => 0.00110,  // Higher due to Fst
            Population::Generic => 0.00100,   // Conservative middle estimate
        }
    }

    /// Get the expected non-IBD emission parameters (mean, std) for this population.
    ///
    /// The mean is 1 - π (expected identity), and std is derived from
    /// the Poisson variance of SNP counts in a window, with empirical
    /// correction for linkage disequilibrium.
    pub fn non_ibd_emission(&self) -> GaussianParams {
        let pi = self.diversity();
        let mean = 1.0 - pi;

        // Variance: Poisson approximation with LD correction factor (~3x)
        // For 5kb windows: std ≈ sqrt(π / 5000 * 3)
        let window_size = 5000.0;
        let ld_correction = 3.0;
        let std = (pi / window_size * ld_correction).sqrt();

        GaussianParams { mean, std }
    }

    /// Parse population from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "AFR" => Some(Population::AFR),
            "EUR" => Some(Population::EUR),
            "EAS" => Some(Population::EAS),
            "CSA" => Some(Population::CSA),
            "AMR" => Some(Population::AMR),
            "INTERPOP" | "INTER" => Some(Population::InterPop),
            "GENERIC" | "UNKNOWN" => Some(Population::Generic),
            _ => None,
        }
    }
}

/// Default IBD emission parameters.
///
/// IBD segments have very high identity (~0.9997) with low variance,
/// as differences are only due to:
/// - Sequencing/assembly errors (ε ≈ 0.0003-0.0005)
/// - Mutations since MRCA (negligible for recent IBD)
///
/// Based on Browning & Browning (2020), the discordance rate within IBD
/// is ε ≈ 0.0003-0.0005 (UK Biobank estimates). This gives identity ~0.9997.
///
/// The key challenge: non-IBD identity is ~0.999 (1-π), so the separation
/// between states is only ~0.0007. Detection requires accumulating evidence
/// over multiple consecutive windows.
pub const IBD_EMISSION: GaussianParams = GaussianParams {
    mean: 0.9997,
    std: 0.0005,
};

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
        // Use Generic population for backwards compatibility
        Self::from_population(Population::Generic, expected_ibd_windows, p_enter_ibd)
    }

    /// Create HMM parameters with population-specific background.
    ///
    /// This constructor uses biologically correct emission parameters based on
    /// population-specific nucleotide diversity (π).
    ///
    /// ## Parameters
    ///
    /// - `population`: The population for estimating non-IBD background
    /// - `expected_ibd_windows`: Expected number of consecutive windows in an IBD segment
    /// - `p_enter_ibd`: Probability of transitioning from non-IBD to IBD state
    ///
    /// ## Population-Specific Background
    ///
    /// The non-IBD emission mean is set to 1 - π, where π is the nucleotide diversity:
    /// - AFR: 0.99875 (highest diversity, lowest identity)
    /// - EUR: 0.99915
    /// - EAS: 0.99920 (lowest diversity, highest identity)
    ///
    /// ## Example
    ///
    /// ```rust
    /// use hprc_ibd::hmm::{HmmParams, Population};
    ///
    /// // For European samples
    /// let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001);
    /// assert!(params.emission[0].mean > 0.99);  // Biologically correct!
    ///
    /// // For inter-population comparison (AFR vs EAS)
    /// let params = HmmParams::from_population(Population::InterPop, 50.0, 0.00001);
    /// ```
    pub fn from_population(
        population: Population,
        expected_ibd_windows: f64,
        p_enter_ibd: f64,
    ) -> Self {
        assert!(
            p_enter_ibd > 0.0 && p_enter_ibd < 1.0,
            "p_enter_ibd must be in range (0, 1), got {}",
            p_enter_ibd
        );

        let p_stay_ibd = 1.0 - 1.0 / expected_ibd_windows;
        let p_stay_ibd = p_stay_ibd.clamp(0.5, 0.9999);
        let p_exit_ibd = 1.0 - p_stay_ibd;

        // Get population-specific non-IBD emission
        let non_ibd_emission = population.non_ibd_emission();

        HmmParams {
            initial: [1.0 - p_enter_ibd, p_enter_ibd],
            transition: [
                [1.0 - p_enter_ibd, p_enter_ibd],
                [p_exit_ibd, p_stay_ibd],
            ],
            emission: [non_ibd_emission, IBD_EMISSION],
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

    /// Robust emission estimation with population-aware priors.
    ///
    /// This method improves on `estimate_emissions` by:
    /// 1. Using population-specific priors as regularization
    /// 2. Only updating emissions if clear bimodal structure exists
    /// 3. Applying sensible bounds based on biological constraints
    ///
    /// ## Parameters
    ///
    /// - `observations`: Identity values from windowed analysis
    /// - `population_prior`: Optional population for prior parameters
    ///
    /// ## Algorithm
    ///
    /// 1. Compute data statistics
    /// 2. Run k-means to find potential clusters
    /// 3. Check if cluster separation is biologically meaningful (> 0.0005)
    /// 4. If yes, update emissions with data-driven estimates bounded by priors
    /// 5. If no, keep population-based defaults
    ///
    /// ## Example
    ///
    /// ```rust
    /// use hprc_ibd::hmm::{HmmParams, Population};
    ///
    /// let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001);
    ///
    /// // Data with clear IBD signal
    /// let observations = vec![0.998, 0.997, 0.9995, 0.9998, 0.9996, 0.997];
    /// params.estimate_emissions_robust(&observations, Some(Population::EUR));
    /// ```
    pub fn estimate_emissions_robust(
        &mut self,
        observations: &[f64],
        population_prior: Option<Population>,
    ) {
        if observations.len() < 10 {
            // Need sufficient data for robust estimation
            return;
        }

        // Get prior parameters
        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd = prior.non_ibd_emission();

        // Compute data statistics
        let n = observations.len() as f64;
        let mean: f64 = observations.iter().sum::<f64>() / n;
        let variance: f64 = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        // If very low variance, data is likely all one state
        if variance < 1e-8 {
            // Check if data looks like IBD or non-IBD based on mean
            if mean > 0.9993 {
                // Likely all IBD - keep prior for non-IBD
                self.emission[1] = GaussianParams {
                    mean,
                    std: variance.sqrt().max(0.0005),
                };
            } else {
                // Likely all non-IBD - keep prior for IBD
                self.emission[0] = GaussianParams {
                    mean,
                    std: variance.sqrt().max(0.001),
                };
            }
            return;
        }

        // Try k-means clustering
        if let Some((centers, assignments)) = kmeans_1d(observations, 2, 30) {
            let idx_low = if centers[0] < centers[1] { 0 } else { 1 };

            let separation = (centers[0] - centers[1]).abs();

            // Minimum separation for meaningful distinction (0.05%)
            // This is biologically motivated: IBD vs non-IBD differ by ~0.05-0.1%
            const MIN_SEPARATION: f64 = 0.0005;

            if separation > MIN_SEPARATION {
                // Compute cluster statistics
                let mut stats_low = (0.0, 0.0, 0usize);  // (sum, sq_sum, count)
                let mut stats_high = (0.0, 0.0, 0usize);

                for (obs, &cluster) in observations.iter().zip(assignments.iter()) {
                    if cluster == idx_low {
                        stats_low.0 += obs;
                        stats_low.1 += obs * obs;
                        stats_low.2 += 1;
                    } else {
                        stats_high.0 += obs;
                        stats_high.1 += obs * obs;
                        stats_high.2 += 1;
                    }
                }

                // Update non-IBD (low cluster) with bounds
                if stats_low.2 > 2 {
                    let mean_low = stats_low.0 / stats_low.2 as f64;
                    let var_low = (stats_low.1 / stats_low.2 as f64) - mean_low * mean_low;

                    // Bound mean to reasonable non-IBD range
                    let bounded_mean = mean_low.clamp(
                        prior_non_ibd.mean - 0.005,  // Allow 0.5% below prior
                        0.9993,                       // Must be below IBD threshold
                    );

                    self.emission[0] = GaussianParams {
                        mean: bounded_mean,
                        std: var_low.sqrt().clamp(0.0005, 0.005),
                    };
                }

                // Update IBD (high cluster) with bounds
                if stats_high.2 > 2 {
                    let mean_high = stats_high.0 / stats_high.2 as f64;
                    let var_high = (stats_high.1 / stats_high.2 as f64) - mean_high * mean_high;

                    // Bound mean to reasonable IBD range
                    let bounded_mean = mean_high.clamp(
                        0.999,   // Must be very high
                        1.0,     // Can't exceed 1.0
                    );

                    self.emission[1] = GaussianParams {
                        mean: bounded_mean,
                        std: var_high.sqrt().clamp(0.0003, 0.002),
                    };
                }
            }
            // If separation too small, keep population-based defaults
        }
    }

    /// Get a summary of the current HMM parameters.
    pub fn summary(&self) -> String {
        format!(
            "HMM Parameters:\n\
             - Initial: P(non-IBD)={:.4}, P(IBD)={:.4}\n\
             - Transition: P(stay non-IBD)={:.6}, P(enter IBD)={:.6}\n\
             - Transition: P(exit IBD)={:.6}, P(stay IBD)={:.6}\n\
             - Emission non-IBD: mean={:.6}, std={:.6}\n\
             - Emission IBD: mean={:.6}, std={:.6}",
            self.initial[0], self.initial[1],
            self.transition[0][0], self.transition[0][1],
            self.transition[1][0], self.transition[1][1],
            self.emission[0].mean, self.emission[0].std,
            self.emission[1].mean, self.emission[1].std,
        )
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
/// // For demonstration, use balanced priors (p_enter_ibd = 0.5)
/// let params = HmmParams::from_expected_length(10.0, 0.5);
///
/// // Clear low identity observations -> all non-IBD
/// let low_obs = vec![0.5, 0.5, 0.5];
/// let states_low = viterbi(&low_obs, &params);
/// assert_eq!(states_low, vec![0, 0, 0]); // All non-IBD
///
/// // Clear very high identity observations -> all IBD
/// let high_obs = vec![0.9999, 0.9999, 0.9999];
/// let states_high = viterbi(&high_obs, &params);
/// assert_eq!(states_high, vec![1, 1, 1]); // All IBD
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
        // Use higher p_enter_ibd for single observation test to reduce prior effect
        let params = HmmParams::from_expected_length(10.0, 0.5);

        // Single very high identity observation (above IBD mean ~0.9997)
        let obs_high = vec![0.9999];
        let states_high = viterbi(&obs_high, &params);
        assert_eq!(states_high.len(), 1);
        // With balanced prior, very high identity should be classified as IBD
        assert_eq!(states_high[0], 1);

        // Single low identity observation (well below non-IBD mean ~0.999)
        let obs_low = vec![0.5];
        let states_low = viterbi(&obs_low, &params);
        assert_eq!(states_low.len(), 1);
        // Low identity should be non-IBD (state 0)
        assert_eq!(states_low[0], 0);
    }

    #[test]
    fn test_viterbi_all_high_identity() {
        // All observations indicate IBD (very high identity ~0.9997-0.9999)
        // For human data, IBD mean is ~0.9997, so values must be above this
        let params = HmmParams::from_expected_length(10.0, 0.001);
        let obs = vec![0.9998, 0.9999, 0.9999, 0.9998, 0.9997, 0.9999, 0.9999, 0.9998];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 8);
        // All should be IBD (state 1) due to very high identity values
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
        // Use higher p_enter_ibd to allow transitions
        let params = HmmParams::from_expected_length(5.0, 0.1);
        // Low (well below non-IBD), Low, Very High (IBD) x5, Low, Low
        // Need enough IBD observations to overcome transition cost
        let obs = vec![0.5, 0.5, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.5, 0.5];
        let states = viterbi(&obs, &params);
        assert_eq!(states.len(), 9);

        // First two should be non-IBD (clearly below non-IBD mean)
        assert_eq!(states[0], 0);
        assert_eq!(states[1], 0);
        // Middle five should be IBD (above IBD mean with enough evidence)
        assert_eq!(states[2], 1);
        assert_eq!(states[3], 1);
        assert_eq!(states[4], 1);
        assert_eq!(states[5], 1);
        assert_eq!(states[6], 1);
        // Last two should be non-IBD
        assert_eq!(states[7], 0);
        assert_eq!(states[8], 0);
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
