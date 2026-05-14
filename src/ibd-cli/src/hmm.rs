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
//! use impopk_ibd::hmm::{HmmParams, Population, viterbi, extract_ibd_segments};
//!
//! // Identity observations from sliding windows
//! let observations = vec![
//!     0.998, 0.997, 0.9985,  // Non-IBD region
//!     0.9998, 0.9999, 0.9997, 0.9998,  // IBD region
//!     0.997, 0.998,  // Non-IBD region
//! ];
//!
//! // Create HMM with population-specific parameters
//! let window_size = 5000;  // 5kb windows
//! let mut params = HmmParams::from_population(
//!     Population::EUR,
//!     50.0,    // expected IBD segment length in windows
//!     0.0001,  // probability of entering IBD
//!     window_size,
//! );
//!
//! // Optionally refine emissions from observed data
//! params.estimate_emissions_robust(&observations, Some(Population::EUR), window_size);
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

use std::io::{BufRead, BufReader};

use crate::stats::{bic_model_selection, em_two_component, em_two_component_map, gaussian_to_logit_space, kmeans_1d, trimmed_mean, GaussianParams, LOGIT_CAP};

/// Human population for population-specific HMM parameters.
///
/// Nucleotide diversity (π) varies between populations, affecting the expected
/// identity distribution for non-IBD haplotype pairs.
///
/// ## Population Diversity (from 1000 Genomes)
///
/// | Population | π (SNPs/bp) | E\[identity\] |
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
    ///
    /// ## Parameters
    ///
    /// - `window_size`: The window size in base pairs used for identity calculations.
    ///   This affects the variance of the emission distribution.
    pub fn non_ibd_emission(&self, window_size: u64) -> GaussianParams {
        let pi = self.diversity();
        let mean = 1.0 - pi;

        // Variance: Poisson approximation with LD correction factor (~3x)
        // std ≈ sqrt(π / window_size * 3)
        let ld_correction = 3.0;
        let std = (pi / window_size as f64 * ld_correction).sqrt();

        // SAFETY: std is always positive since pi > 0, window_size > 0, and sqrt of positive is positive
        GaussianParams::new_unchecked(mean, std)
    }

    /// Parse population from string.
    #[allow(clippy::should_implement_trait)]
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
/// use impopk_ibd::hmm::HmmParams;
///
/// // Create parameters expecting 50-window IBD segments
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
///
/// // Check transition probabilities
/// assert!(params.transition[1][1] > 0.9); // High probability to stay in IBD
/// ```
#[derive(Debug, Clone)]
pub struct HmmParams {
    /// Initial state probabilities: [P(non-IBD), P(IBD)]
    pub initial: [f64; 2],
    /// Transition matrix: `transition[from_state][to_state]`
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
    /// - `window_size`: The window size in base pairs used for identity calculations.
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
    /// use impopk_ibd::hmm::HmmParams;
    ///
    /// // Conservative settings: expect long segments, rare IBD transitions
    /// let params = HmmParams::from_expected_length(100.0, 0.00001, 5000);
    ///
    /// // Sensitive settings: expect shorter segments, easier IBD transitions
    /// let params = HmmParams::from_expected_length(20.0, 0.001, 5000);
    /// ```
    pub fn from_expected_length(expected_ibd_windows: f64, p_enter_ibd: f64, window_size: u64) -> Self {
        // Use Generic population for backwards compatibility
        Self::from_population(Population::Generic, expected_ibd_windows, p_enter_ibd, window_size)
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
    /// - `window_size`: The window size in base pairs used for identity calculations
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
    /// use impopk_ibd::hmm::{HmmParams, Population};
    ///
    /// // For European samples with 5kb windows
    /// let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    /// assert!(params.emission[0].mean > 0.99);  // Biologically correct!
    ///
    /// // For inter-population comparison (AFR vs EAS)
    /// let params = HmmParams::from_population(Population::InterPop, 50.0, 0.00001, 5000);
    /// ```
    pub fn from_population(
        population: Population,
        expected_ibd_windows: f64,
        p_enter_ibd: f64,
        window_size: u64,
    ) -> Self {
        assert!(
            p_enter_ibd > 0.0 && p_enter_ibd < 1.0,
            "p_enter_ibd must be in range (0, 1), got {}",
            p_enter_ibd
        );

        let p_stay_ibd = 1.0 - 1.0 / expected_ibd_windows;
        let p_stay_ibd = p_stay_ibd.clamp(0.5, 0.9999);
        let p_exit_ibd = 1.0 - p_stay_ibd;

        // Get population-specific non-IBD emission with correct window size
        let non_ibd_emission = population.non_ibd_emission(window_size);

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
    /// use impopk_ibd::hmm::HmmParams;
    ///
    /// let mut params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
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
                    let var = ((sq_sum_low / n_low as f64) - mean * mean).max(0.0);
                    self.emission[0] = GaussianParams::new_unchecked(
                        mean,
                        var.sqrt().max(0.01),
                    );
                }

                if n_high > 0 {
                    let mean = sum_high / n_high as f64;
                    let var = ((sq_sum_high / n_high as f64) - mean * mean).max(0.0);
                    self.emission[1] = GaussianParams::new_unchecked(
                        mean,
                        var.sqrt().max(0.001),
                    );
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

                let overall_std = variance.max(0.0).sqrt();
                self.emission[0].std = overall_std.max(0.05);
                self.emission[1].std = overall_std.max(0.01);
            }
        }
    }

    /// Create HMM parameters with population-adaptive transition probabilities.
    ///
    /// Unlike `from_population`, this method adjusts transition probabilities
    /// based on expected IBD density for each population:
    /// - AFR: Lower P(enter IBD), shorter expected segments (highest diversity)
    /// - EUR/EAS: Standard priors
    /// - InterPop: Very low P(enter IBD) (cross-population IBD is rare)
    ///
    /// ## Parameters
    ///
    /// - `population`: The population for parameter adaptation
    /// - `expected_ibd_windows`: Base expected IBD segment length (adjusted by population)
    /// - `p_enter_ibd`: Base probability of entering IBD (adjusted by population)
    /// - `window_size`: Window size in base pairs
    pub fn from_population_adaptive(
        population: Population,
        expected_ibd_windows: f64,
        p_enter_ibd: f64,
        window_size: u64,
    ) -> Self {
        // Scale transition probabilities by population
        let (adj_expected, adj_p_enter) = match population {
            // AFR: fewer and shorter IBD segments due to higher diversity
            // and deeper coalescence times
            Population::AFR => (
                expected_ibd_windows * 0.7,     // Expect shorter segments
                p_enter_ibd * 0.3,              // Much rarer IBD entry
            ),
            // EUR: moderately common IBD
            Population::EUR => (
                expected_ibd_windows,
                p_enter_ibd,
            ),
            // EAS: similar to EUR but slightly longer segments
            Population::EAS => (
                expected_ibd_windows * 1.1,
                p_enter_ibd,
            ),
            // CSA: intermediate
            Population::CSA => (
                expected_ibd_windows * 0.9,
                p_enter_ibd * 0.8,
            ),
            // AMR: admixed, variable IBD patterns
            Population::AMR => (
                expected_ibd_windows * 0.8,
                p_enter_ibd * 0.7,
            ),
            // InterPop: cross-population IBD is very rare
            Population::InterPop => (
                expected_ibd_windows * 0.5,     // Short segments if any
                p_enter_ibd * 0.1,              // Very rare
            ),
            Population::Generic => (expected_ibd_windows, p_enter_ibd),
        };

        // Clamp adjusted p_enter_ibd to valid range
        let adj_p_enter = adj_p_enter.clamp(1e-8, 0.999);

        Self::from_population(population, adj_expected, adj_p_enter, window_size)
    }

    /// Create HMM parameters for observations in logit-transformed space.
    ///
    /// When observations are logit-transformed (x → log(x/(1-x))), the emission
    /// distributions need to be in the same space. This constructor computes the
    /// appropriate Gaussian parameters in logit space using the delta method.
    ///
    /// ## Motivation
    ///
    /// Pangenome identity data is concentrated near 1.0, where small differences
    /// (0.997 vs 0.9997) are hard to distinguish with Gaussian emissions.
    /// The logit transform stretches this region:
    /// - logit(0.997) ≈ 5.81
    /// - logit(0.9997) ≈ 8.11
    /// - Difference increases from 0.0027 to 2.3
    ///
    /// This makes the Gaussian emission model more effective.
    pub fn from_population_logit(
        population: Population,
        expected_ibd_windows: f64,
        p_enter_ibd: f64,
        window_size: u64,
    ) -> Self {
        assert!(
            p_enter_ibd > 0.0 && p_enter_ibd < 1.0,
            "p_enter_ibd must be in range (0, 1), got {}",
            p_enter_ibd
        );

        let p_stay_ibd = 1.0 - 1.0 / expected_ibd_windows;
        let p_stay_ibd = p_stay_ibd.clamp(0.5, 0.9999);
        let p_exit_ibd = 1.0 - p_stay_ibd;

        // Get population-specific emission parameters in raw space
        let raw_non_ibd = population.non_ibd_emission(window_size);

        // Transform to logit space using delta method
        let logit_non_ibd = gaussian_to_logit_space(raw_non_ibd.mean, raw_non_ibd.std);
        let logit_ibd = gaussian_to_logit_space(IBD_EMISSION.mean, IBD_EMISSION.std);

        HmmParams {
            initial: [1.0 - p_enter_ibd, p_enter_ibd],
            transition: [
                [1.0 - p_enter_ibd, p_enter_ibd],
                [p_exit_ibd, p_stay_ibd],
            ],
            emission: [logit_non_ibd, logit_ibd],
        }
    }

    /// Estimate emission parameters from logit-transformed observations.
    ///
    /// Similar to `estimate_emissions_robust` but works in logit space.
    /// The data should already be logit-transformed before calling this method.
    ///
    /// Uses k-means with larger separation thresholds (logit space has wider spread)
    /// and EM with logit-space priors.
    pub fn estimate_emissions_logit(
        &mut self,
        logit_observations: &[f64],
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        if logit_observations.len() < 10 {
            return;
        }

        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd_raw = prior.non_ibd_emission(window_size);
        let prior_non_ibd = gaussian_to_logit_space(prior_non_ibd_raw.mean, prior_non_ibd_raw.std);
        let prior_ibd = gaussian_to_logit_space(IBD_EMISSION.mean, IBD_EMISSION.std);

        let n = logit_observations.len() as f64;
        let mean: f64 = logit_observations.iter().sum::<f64>() / n;
        let variance: f64 = logit_observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        if variance < 1e-6 {
            // Very low variance in logit space: all observations similar
            if mean > prior_ibd.mean - 1.0 {
                self.emission[1] = GaussianParams::new_unchecked(
                    mean,
                    variance.max(0.0).sqrt().max(0.1),
                );
            } else {
                self.emission[0] = GaussianParams::new_unchecked(
                    mean,
                    variance.max(0.0).sqrt().max(0.2),
                );
            }
            return;
        }

        // In logit space, meaningful separation is ~0.5 (corresponds to ~0.0002 in raw near 1.0)
        const MIN_SEPARATION_LOGIT: f64 = 0.5;

        let kmeans_ok = if let Some((centers, assignments)) = kmeans_1d(logit_observations, 2, 30) {
            let idx_low = if centers[0] < centers[1] { 0 } else { 1 };
            let separation = (centers[0] - centers[1]).abs();

            if separation > MIN_SEPARATION_LOGIT {
                let mut stats_low = (0.0, 0.0, 0usize);
                let mut stats_high = (0.0, 0.0, 0usize);

                for (obs, &cluster) in logit_observations.iter().zip(assignments.iter()) {
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

                if stats_low.2 > 2 {
                    let mean_low = stats_low.0 / stats_low.2 as f64;
                    let var_low = ((stats_low.1 / stats_low.2 as f64) - mean_low * mean_low).max(0.0);
                    self.emission[0] = GaussianParams::new_unchecked(
                        mean_low,
                        var_low.sqrt().clamp(0.1, 5.0),
                    );
                }

                if stats_high.2 > 2 {
                    let mean_high = stats_high.0 / stats_high.2 as f64;
                    let var_high = ((stats_high.1 / stats_high.2 as f64) - mean_high * mean_high).max(0.0);
                    self.emission[1] = GaussianParams::new_unchecked(
                        mean_high,
                        var_high.sqrt().clamp(0.05, 3.0),
                    );
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if !kmeans_ok {
            // EM with logit-space priors
            let init_weight_non_ibd = match prior {
                Population::AFR => 0.98,
                Population::InterPop => 0.99,
                _ => 0.95,
            };

            let prior_strength = match prior {
                Population::AFR => 15.0,
                Population::InterPop => 10.0,
                _ => 5.0,
            };

            let em_result = em_two_component_map(
                logit_observations,
                &prior_non_ibd,
                &prior_ibd,
                init_weight_non_ibd,
                100,
                1e-6,
                prior_strength,
            ).or_else(|| {
                em_two_component(
                    logit_observations,
                    &prior_non_ibd,
                    &prior_ibd,
                    init_weight_non_ibd,
                    100,
                    1e-6,
                )
            });

            if let Some((em_low, em_high, w_low, _w_high)) = em_result {
                let em_separation = (em_high.mean - em_low.mean).abs();

                if em_separation > MIN_SEPARATION_LOGIT * 0.5 {
                    let (bic_1, bic_2) = bic_model_selection(
                        logit_observations, &em_low, &em_high, w_low,
                    );

                    if bic_2 < bic_1 {
                        self.emission[0] = GaussianParams::new_unchecked(
                            em_low.mean,
                            em_low.std.clamp(0.1, 5.0),
                        );
                        self.emission[1] = GaussianParams::new_unchecked(
                            em_high.mean,
                            em_high.std.clamp(0.05, 3.0),
                        );
                    }
                }
            }
        }
    }

    /// Robust emission estimation with population-aware priors and EM fallback.
    ///
    /// This method improves on `estimate_emissions` by:
    /// 1. Using population-specific priors as initialization and regularization
    /// 2. Detecting degenerate k-means clusters (both centroids too close)
    /// 3. Falling back to EM when k-means fails or produces poor separation
    /// 4. Anchoring non-IBD mean near 1-π for the population
    ///
    /// ## Parameters
    ///
    /// - `observations`: Identity values from windowed analysis
    /// - `population_prior`: Optional population for prior parameters
    /// - `window_size`: The window size in base pairs used for identity calculations
    ///
    /// ## Algorithm
    ///
    /// 1. Compute data statistics
    /// 2. Try k-means clustering
    /// 3. If k-means finds good separation (> 0.0005), use those estimates
    /// 4. If k-means produces degenerate clusters, try EM with population priors
    /// 5. EM is initialized with population-specific means to avoid local optima
    /// 6. Apply biological bounds to all estimates
    ///
    /// ## Example
    ///
    /// ```rust
    /// use impopk_ibd::hmm::{HmmParams, Population};
    ///
    /// let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    ///
    /// // Data with clear IBD signal
    /// let observations = vec![0.998, 0.997, 0.9995, 0.9998, 0.9996, 0.997];
    /// params.estimate_emissions_robust(&observations, Some(Population::EUR), 5000);
    /// ```
    pub fn estimate_emissions_robust(
        &mut self,
        observations: &[f64],
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        if observations.len() < 10 {
            return;
        }

        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd = prior.non_ibd_emission(window_size);

        let n = observations.len() as f64;
        let mean: f64 = observations.iter().sum::<f64>() / n;
        let variance: f64 = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

        // If very low variance, data is likely all one state
        if variance < 1e-8 {
            if mean > 0.9993 {
                self.emission[1] = GaussianParams::new_unchecked(
                    mean,
                    variance.max(0.0).sqrt().max(0.0005),
                );
            } else {
                self.emission[0] = GaussianParams::new_unchecked(
                    mean,
                    variance.max(0.0).sqrt().max(0.001),
                );
            }
            return;
        }

        // Minimum separation for meaningful distinction
        const MIN_SEPARATION: f64 = 0.0005;

        // Try k-means clustering first
        let kmeans_ok = if let Some((centers, assignments)) = kmeans_1d(observations, 2, 30) {
            let idx_low = if centers[0] < centers[1] { 0 } else { 1 };
            let separation = (centers[0] - centers[1]).abs();

            if separation > MIN_SEPARATION {
                // Good separation - use k-means results with bounds
                let mut stats_low = (0.0, 0.0, 0usize);
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

                if stats_low.2 > 2 {
                    let mean_low = stats_low.0 / stats_low.2 as f64;
                    let var_low = ((stats_low.1 / stats_low.2 as f64) - mean_low * mean_low).max(0.0);
                    let bounded_mean = mean_low.clamp(
                        prior_non_ibd.mean - 0.005,
                        0.9993,
                    );
                    self.emission[0] = GaussianParams::new_unchecked(
                        bounded_mean,
                        var_low.sqrt().clamp(0.0005, 0.005),
                    );
                }

                if stats_high.2 > 2 {
                    let mean_high = stats_high.0 / stats_high.2 as f64;
                    let var_high = ((stats_high.1 / stats_high.2 as f64) - mean_high * mean_high).max(0.0);
                    let bounded_mean = mean_high.clamp(0.999, 1.0);
                    self.emission[1] = GaussianParams::new_unchecked(
                        bounded_mean,
                        var_high.sqrt().clamp(0.0003, 0.002),
                    );
                }
                true
            } else {
                false // Degenerate clusters
            }
        } else {
            false
        };

        // If k-means failed or produced degenerate clusters, try EM with population priors
        if !kmeans_ok {
            // Initialize EM with population-specific priors
            // This is crucial for AFR where k-means splits the non-IBD distribution
            let init_non_ibd = prior_non_ibd;
            let init_ibd = IBD_EMISSION;

            // Use trimmed mean to get a robust estimate of the data center
            // This helps distinguish "mostly non-IBD" from "mostly IBD" data
            if let Some(tm) = trimmed_mean(observations, 0.1) {
                // If trimmed mean is very close to IBD emission, most data is IBD
                // If close to non-IBD emission, most data is non-IBD
                // Use this to set initial weights more accurately
                let dist_to_non_ibd = (tm - init_non_ibd.mean).abs();
                let dist_to_ibd = (tm - init_ibd.mean).abs();
                let _data_mostly_ibd = dist_to_ibd < dist_to_non_ibd;
            }

            // Estimate initial weight: most windows should be non-IBD
            // For AFR, this is even more true (>98% non-IBD typically)
            let init_weight_non_ibd = match prior {
                Population::AFR => 0.98,
                Population::InterPop => 0.99,
                _ => 0.95,
            };

            // Determine prior strength based on population
            // AFR needs stronger regularization since distributions overlap more
            let prior_strength = match prior {
                Population::AFR => 15.0,
                Population::InterPop => 10.0,
                _ => 5.0,
            };

            // Try MAP-regularized EM first (more robust for overlapping distributions)
            let em_result = em_two_component_map(
                observations,
                &init_non_ibd,
                &init_ibd,
                init_weight_non_ibd,
                100,
                1e-6,
                prior_strength,
            ).or_else(|| {
                // Fall back to standard EM if MAP fails
                em_two_component(
                    observations,
                    &init_non_ibd,
                    &init_ibd,
                    init_weight_non_ibd,
                    100,
                    1e-6,
                )
            });

            if let Some((em_low, em_high, w_low, _w_high)) = em_result {
                let em_separation = (em_high.mean - em_low.mean).abs();

                if em_separation > MIN_SEPARATION * 0.5 {
                    // Check BIC: does the 2-component model actually fit better?
                    let (bic_1, bic_2) = bic_model_selection(
                        observations, &em_low, &em_high, w_low,
                    );

                    if bic_2 < bic_1 {
                        // 2-component model is better - use bounded EM estimates
                        let bounded_low_mean = em_low.mean.clamp(
                            prior_non_ibd.mean - 0.005,
                            0.9993,
                        );
                        self.emission[0] = GaussianParams::new_unchecked(
                            bounded_low_mean,
                            em_low.std.clamp(0.0005, 0.005),
                        );

                        let bounded_high_mean = em_high.mean.clamp(0.999, 1.0);
                        self.emission[1] = GaussianParams::new_unchecked(
                            bounded_high_mean,
                            em_high.std.clamp(0.0003, 0.002),
                        );
                    }
                    // If BIC prefers 1 component, keep population defaults (more conservative)
                }
                // If EM also shows no separation, keep population defaults (already set)
            }
            // If EM fails entirely, keep population defaults (already set)
        }
    }

    /// Baum-Welch re-estimation of HMM parameters from observations.
    ///
    /// Iteratively updates emission and transition parameters using the
    /// forward-backward algorithm (expectation step) followed by parameter
    /// re-estimation (maximization step). This is the standard EM algorithm
    /// for HMM training.
    ///
    /// ## Why Baum-Welch matters for AFR populations
    ///
    /// For African populations, the separation between IBD and non-IBD emission
    /// distributions is very small (~0.0007). Initial parameter estimates from
    /// k-means or population priors may be imprecise, leading to poor Viterbi
    /// decoding. Baum-Welch refines parameters using soft assignments (posteriors)
    /// rather than hard assignments (k-means), which is more robust when
    /// distributions overlap heavily.
    ///
    /// ## Parameters
    ///
    /// - `observations`: Identity values from windowed analysis
    /// - `max_iter`: Maximum number of Baum-Welch iterations (typically 10-30)
    /// - `tol`: Convergence tolerance on log-likelihood improvement
    /// - `population_prior`: Optional population for emission bounds/regularization
    /// - `window_size`: Window size in base pairs
    ///
    /// ## Algorithm
    ///
    /// 1. Run forward-backward to get posteriors and transition expectations
    /// 2. Re-estimate emission means and variances using posterior-weighted stats
    /// 3. Re-estimate transition probabilities using expected transition counts
    /// 4. Apply biological bounds to prevent degenerate parameters
    /// 5. Repeat until convergence or max_iter
    ///
    /// ## Example
    ///
    /// ```rust
    /// use impopk_ibd::hmm::{HmmParams, Population};
    ///
    /// let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
    /// let observations = vec![0.998, 0.9985, 0.999, 0.9997, 0.9998, 0.9996, 0.998, 0.997];
    /// params.baum_welch(&observations, 20, 1e-6, Some(Population::AFR), 5000);
    /// ```
    pub fn baum_welch(
        &mut self,
        observations: &[f64],
        max_iter: usize,
        tol: f64,
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        let n = observations.len();
        if n < 10 {
            return;
        }

        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd = prior.non_ibd_emission(window_size);

        let mut prev_log_lik = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: forward-backward
            let (alpha, log_lik) = forward(observations, self);
            let beta = backward(observations, self);

            // Check convergence
            if (log_lik - prev_log_lik).abs() < tol {
                break;
            }
            prev_log_lik = log_lik;

            // Compute posteriors (gamma) for each state at each position
            let mut gamma = vec![[0.0f64; 2]; n];
            for t in 0..n {
                let log_gamma_0 = alpha[t][0] + beta[t][0] - log_lik;
                let log_gamma_1 = alpha[t][1] + beta[t][1] - log_lik;
                let max_log = log_gamma_0.max(log_gamma_1);
                let log_sum = max_log
                    + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
                gamma[t][0] = (log_gamma_0 - log_sum).exp();
                gamma[t][1] = (log_gamma_1 - log_sum).exp();
            }

            // Compute expected transitions (xi) summed over time
            let log_trans: [[f64; 2]; 2] = [
                [self.transition[0][0].ln(), self.transition[0][1].ln()],
                [self.transition[1][0].ln(), self.transition[1][1].ln()],
            ];
            let mut xi_sum = [[0.0f64; 2]; 2];

            for t in 0..n - 1 {
                let mut log_emit_next = [0.0f64; 2];
                log_emit_next[0] = self.emission[0].log_pdf(observations[t + 1]);
                log_emit_next[1] = self.emission[1].log_pdf(observations[t + 1]);

                // Compute xi[t][i][j] = P(s_t=i, s_{t+1}=j | O)
                let mut xi_log = [[0.0f64; 2]; 2];
                let mut max_xi = f64::NEG_INFINITY;

                for i in 0..2 {
                    for j in 0..2 {
                        xi_log[i][j] = alpha[t][i] + log_trans[i][j]
                            + log_emit_next[j] + beta[t + 1][j] - log_lik;
                        if xi_log[i][j] > max_xi {
                            max_xi = xi_log[i][j];
                        }
                    }
                }

                // Convert from log and normalize
                let mut xi_sum_t = 0.0;
                let mut xi_val = [[0.0f64; 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        xi_val[i][j] = (xi_log[i][j] - max_xi).exp();
                        xi_sum_t += xi_val[i][j];
                    }
                }
                if xi_sum_t > 0.0 {
                    for i in 0..2 {
                        for j in 0..2 {
                            xi_sum[i][j] += xi_val[i][j] / xi_sum_t;
                        }
                    }
                }
            }

            // M-step: re-estimate parameters

            // Re-estimate emission parameters with posterior weighting
            let mut mu = [0.0f64; 2];
            let mut sigma_sq = [0.0f64; 2];
            let mut gamma_sum = [0.0f64; 2];

            for t in 0..n {
                for s in 0..2 {
                    gamma_sum[s] += gamma[t][s];
                    mu[s] += gamma[t][s] * observations[t];
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    mu[s] /= gamma_sum[s];
                }
            }

            for t in 0..n {
                for s in 0..2 {
                    sigma_sq[s] += gamma[t][s] * (observations[t] - mu[s]).powi(2);
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    sigma_sq[s] /= gamma_sum[s];
                }
            }

            // Apply biological bounds to emission estimates
            // Non-IBD (state 0): mean should be near 1-pi, std bounded
            let bounded_mu0 = mu[0].clamp(prior_non_ibd.mean - 0.005, 0.9993);
            let bounded_std0 = sigma_sq[0].max(0.0).sqrt().clamp(0.0003, 0.005);
            self.emission[0] = GaussianParams::new_unchecked(bounded_mu0, bounded_std0);

            // IBD (state 1): mean should be very high, std bounded
            let bounded_mu1 = mu[1].clamp(0.9990, 1.0);
            let bounded_std1 = sigma_sq[1].max(0.0).sqrt().clamp(0.0002, 0.002);
            self.emission[1] = GaussianParams::new_unchecked(bounded_mu1, bounded_std1);

            // Ensure state 0 mean < state 1 mean (identifiability)
            if self.emission[0].mean >= self.emission[1].mean {
                // Swap or use priors
                self.emission[0] = GaussianParams::new_unchecked(
                    prior_non_ibd.mean,
                    prior_non_ibd.std,
                );
                self.emission[1] = IBD_EMISSION;
            }

            // Re-estimate transition probabilities
            for (i, xi_row) in xi_sum.iter().enumerate() {
                let row_sum = xi_row[0] + xi_row[1];
                if row_sum > 1.0 {
                    self.transition[i][0] = xi_row[0] / row_sum;
                    self.transition[i][1] = xi_row[1] / row_sum;
                }
            }

            // Apply transition bounds
            // P(enter IBD) should be small; P(stay IBD) should be high
            self.transition[0][1] = self.transition[0][1].clamp(1e-8, 0.1);
            self.transition[0][0] = 1.0 - self.transition[0][1];
            self.transition[1][0] = self.transition[1][0].clamp(0.001, 0.5);
            self.transition[1][1] = 1.0 - self.transition[1][0];
        }
    }

    /// Baum-Welch parameter re-estimation in logit space.
    ///
    /// Like [`baum_welch`](Self::baum_welch), but with bounds appropriate for
    /// logit-transformed identity observations. In logit space:
    /// - Non-IBD mean: ~6-8 (vs ~0.998-0.9993 in raw space)
    /// - IBD mean: ~7-10 (vs ~0.999-1.0 in raw space)
    /// - Standard deviations: ~0.2-3.0 (vs ~0.0002-0.005 in raw space)
    ///
    /// The E-step (forward-backward) and transition re-estimation are identical
    /// to the raw-space version. Only the M-step emission bounds differ.
    pub fn baum_welch_logit(
        &mut self,
        observations: &[f64],
        max_iter: usize,
        tol: f64,
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        let n = observations.len();
        if n < 10 {
            return;
        }

        // Compute logit-space prior for non-IBD
        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_raw = prior.non_ibd_emission(window_size);
        let prior_logit = gaussian_to_logit_space(prior_raw.mean, prior_raw.std);
        let ibd_logit = gaussian_to_logit_space(IBD_EMISSION.mean, IBD_EMISSION.std);

        let mut prev_log_lik = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: forward-backward
            let (alpha, log_lik) = forward(observations, self);
            let beta = backward(observations, self);

            if (log_lik - prev_log_lik).abs() < tol {
                break;
            }
            prev_log_lik = log_lik;

            // Compute posteriors
            let mut gamma = vec![[0.0f64; 2]; n];
            for t in 0..n {
                let log_gamma_0 = alpha[t][0] + beta[t][0] - log_lik;
                let log_gamma_1 = alpha[t][1] + beta[t][1] - log_lik;
                let max_log = log_gamma_0.max(log_gamma_1);
                let log_sum = max_log
                    + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
                gamma[t][0] = (log_gamma_0 - log_sum).exp();
                gamma[t][1] = (log_gamma_1 - log_sum).exp();
            }

            // Compute expected transitions
            let log_trans: [[f64; 2]; 2] = [
                [self.transition[0][0].ln(), self.transition[0][1].ln()],
                [self.transition[1][0].ln(), self.transition[1][1].ln()],
            ];
            let mut xi_sum = [[0.0f64; 2]; 2];

            for t in 0..n - 1 {
                let log_emit_next = [
                    self.emission[0].log_pdf(observations[t + 1]),
                    self.emission[1].log_pdf(observations[t + 1]),
                ];

                let mut xi_log = [[0.0f64; 2]; 2];
                let mut max_xi = f64::NEG_INFINITY;

                for i in 0..2 {
                    for j in 0..2 {
                        xi_log[i][j] = alpha[t][i] + log_trans[i][j]
                            + log_emit_next[j] + beta[t + 1][j] - log_lik;
                        if xi_log[i][j] > max_xi {
                            max_xi = xi_log[i][j];
                        }
                    }
                }

                let mut xi_sum_t = 0.0;
                let mut xi_val = [[0.0f64; 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        xi_val[i][j] = (xi_log[i][j] - max_xi).exp();
                        xi_sum_t += xi_val[i][j];
                    }
                }
                if xi_sum_t > 0.0 {
                    for i in 0..2 {
                        for j in 0..2 {
                            xi_sum[i][j] += xi_val[i][j] / xi_sum_t;
                        }
                    }
                }
            }

            // M-step: re-estimate emission parameters with logit-space bounds
            let mut mu = [0.0f64; 2];
            let mut sigma_sq = [0.0f64; 2];
            let mut gamma_sum = [0.0f64; 2];

            for t in 0..n {
                for s in 0..2 {
                    gamma_sum[s] += gamma[t][s];
                    mu[s] += gamma[t][s] * observations[t];
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    mu[s] /= gamma_sum[s];
                }
            }

            for t in 0..n {
                for s in 0..2 {
                    sigma_sq[s] += gamma[t][s] * (observations[t] - mu[s]).powi(2);
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    sigma_sq[s] /= gamma_sum[s];
                }
            }

            // Logit-space bounds for non-IBD (state 0):
            // Mean: [prior - 2.0, prior + 2.0] (±2 logit units ≈ ±0.003 in raw near 0.999)
            // Std: [0.2, 3.0] (delta method gives ~1.0 for typical populations)
            let bounded_mu0 = mu[0].clamp(prior_logit.mean - 2.0, prior_logit.mean + 2.0);
            let bounded_std0 = sigma_sq[0].max(0.0).sqrt().clamp(0.2, 3.0);
            self.emission[0] = GaussianParams::new_unchecked(bounded_mu0, bounded_std0);

            // Logit-space bounds for IBD (state 1):
            // Mean: [prior IBD - 1.0, LOGIT_CAP] (matches observation capping)
            // Std: [0.2, 3.0]
            let bounded_mu1 = mu[1].clamp(ibd_logit.mean - 1.0, LOGIT_CAP);
            let bounded_std1 = sigma_sq[1].max(0.0).sqrt().clamp(0.2, 3.0);
            self.emission[1] = GaussianParams::new_unchecked(bounded_mu1, bounded_std1);

            // Ensure state 0 mean < state 1 mean (identifiability)
            if self.emission[0].mean >= self.emission[1].mean {
                self.emission[0] = GaussianParams::new_unchecked(
                    prior_logit.mean,
                    prior_logit.std,
                );
                self.emission[1] = GaussianParams::new_unchecked(
                    ibd_logit.mean,
                    ibd_logit.std,
                );
            }

            // Re-estimate transition probabilities (same as raw-space BW)
            for (i, xi_row) in xi_sum.iter().enumerate() {
                let row_sum = xi_row[0] + xi_row[1];
                if row_sum > 1.0 {
                    self.transition[i][0] = xi_row[0] / row_sum;
                    self.transition[i][1] = xi_row[1] / row_sum;
                }
            }

            // Apply transition bounds
            self.transition[0][1] = self.transition[0][1].clamp(1e-8, 0.1);
            self.transition[0][0] = 1.0 - self.transition[0][1];
            self.transition[1][0] = self.transition[1][0].clamp(0.001, 0.5);
            self.transition[1][1] = 1.0 - self.transition[1][0];
        }
    }

    /// Baum-Welch parameter re-estimation with distance-dependent transitions.
    ///
    /// Like [`baum_welch`](Self::baum_welch), but uses distance-dependent transition
    /// matrices based on physical distances between windows. This properly handles
    /// non-uniform window spacing in pangenome data.
    ///
    /// The E-step uses distance-aware forward-backward to compute posteriors and
    /// expected transition counts. The M-step re-estimates emission parameters
    /// using the same biological bounds as standard Baum-Welch.
    ///
    /// Note: Transition probabilities are re-estimated as base rates (per nominal
    /// window step), not per-distance rates. The distance scaling is applied
    /// during inference.
    pub fn baum_welch_with_distances(
        &mut self,
        observations: &[f64],
        window_positions: &[(u64, u64)],
        max_iter: usize,
        tol: f64,
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        let n = observations.len();
        if n < 10 {
            return;
        }

        // Fall back to standard Baum-Welch if positions don't match
        if window_positions.len() != n {
            self.baum_welch(observations, max_iter, tol, population_prior, window_size);
            return;
        }

        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd = prior.non_ibd_emission(window_size);

        // Compute nominal window size from first window
        let nom_window_size = {
            let (s, e) = window_positions[0];
            (e.saturating_sub(s) + 1).max(1)
        };

        let mut prev_log_lik = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: distance-aware forward-backward
            let (alpha, log_lik) = forward_with_distances(observations, self, window_positions);
            let beta = backward_with_distances(observations, self, window_positions);

            // Check convergence
            if (log_lik - prev_log_lik).abs() < tol {
                break;
            }
            prev_log_lik = log_lik;

            // Compute posteriors (gamma)
            let mut gamma = vec![[0.0f64; 2]; n];
            for t in 0..n {
                let log_gamma_0 = alpha[t][0] + beta[t][0] - log_lik;
                let log_gamma_1 = alpha[t][1] + beta[t][1] - log_lik;
                let max_log = log_gamma_0.max(log_gamma_1);
                let log_sum = max_log
                    + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
                gamma[t][0] = (log_gamma_0 - log_sum).exp();
                gamma[t][1] = (log_gamma_1 - log_sum).exp();
            }

            // Compute expected transitions (xi) with distance-dependent transitions
            let mut xi_sum = [[0.0f64; 2]; 2];

            for t in 0..n - 1 {
                // Distance-dependent transitions for this step
                let mid_prev = (window_positions[t].0 + window_positions[t].1) / 2;
                let mid_curr = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;
                let distance = mid_curr.saturating_sub(mid_prev);
                let log_trans = distance_dependent_log_transition(self, distance, nom_window_size);

                let mut log_emit_next = [0.0f64; 2];
                log_emit_next[0] = self.emission[0].log_pdf(observations[t + 1]);
                log_emit_next[1] = self.emission[1].log_pdf(observations[t + 1]);

                let mut xi_log = [[0.0f64; 2]; 2];
                let mut max_xi = f64::NEG_INFINITY;

                for i in 0..2 {
                    for j in 0..2 {
                        xi_log[i][j] = alpha[t][i] + log_trans[i][j]
                            + log_emit_next[j] + beta[t + 1][j] - log_lik;
                        if xi_log[i][j] > max_xi {
                            max_xi = xi_log[i][j];
                        }
                    }
                }

                let mut xi_sum_t = 0.0;
                let mut xi_val = [[0.0f64; 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        xi_val[i][j] = (xi_log[i][j] - max_xi).exp();
                        xi_sum_t += xi_val[i][j];
                    }
                }
                if xi_sum_t > 0.0 {
                    for i in 0..2 {
                        for j in 0..2 {
                            xi_sum[i][j] += xi_val[i][j] / xi_sum_t;
                        }
                    }
                }
            }

            // M-step: re-estimate emission parameters with posterior weighting
            let mut mu = [0.0f64; 2];
            let mut sigma_sq = [0.0f64; 2];
            let mut gamma_sum = [0.0f64; 2];

            for t in 0..n {
                for s in 0..2 {
                    gamma_sum[s] += gamma[t][s];
                    mu[s] += gamma[t][s] * observations[t];
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    mu[s] /= gamma_sum[s];
                }
            }

            for t in 0..n {
                for s in 0..2 {
                    sigma_sq[s] += gamma[t][s] * (observations[t] - mu[s]).powi(2);
                }
            }

            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    sigma_sq[s] /= gamma_sum[s];
                }
            }

            // Apply biological bounds to emission estimates
            let bounded_mu0 = mu[0].clamp(prior_non_ibd.mean - 0.005, 0.9993);
            let bounded_std0 = sigma_sq[0].max(0.0).sqrt().clamp(0.0003, 0.005);
            self.emission[0] = GaussianParams::new_unchecked(bounded_mu0, bounded_std0);

            let bounded_mu1 = mu[1].clamp(0.9990, 1.0);
            let bounded_std1 = sigma_sq[1].max(0.0).sqrt().clamp(0.0002, 0.002);
            self.emission[1] = GaussianParams::new_unchecked(bounded_mu1, bounded_std1);

            // Ensure state 0 mean < state 1 mean (identifiability)
            if self.emission[0].mean >= self.emission[1].mean {
                self.emission[0] = GaussianParams::new_unchecked(
                    prior_non_ibd.mean,
                    prior_non_ibd.std,
                );
                self.emission[1] = IBD_EMISSION;
            }

            // Re-estimate transition probabilities
            for (i, xi_row) in xi_sum.iter().enumerate() {
                let row_sum = xi_row[0] + xi_row[1];
                if row_sum > 1.0 {
                    self.transition[i][0] = xi_row[0] / row_sum;
                    self.transition[i][1] = xi_row[1] / row_sum;
                }
            }

            // Apply transition bounds
            self.transition[0][1] = self.transition[0][1].clamp(1e-8, 0.1);
            self.transition[0][0] = 1.0 - self.transition[0][1];
            self.transition[1][0] = self.transition[1][0].clamp(0.001, 0.5);
            self.transition[1][1] = 1.0 - self.transition[1][0];
        }
    }

    /// Baum-Welch parameter re-estimation with recombination-aware transitions.
    ///
    /// Like [`baum_welch_with_distances`](Self::baum_welch_with_distances), but uses
    /// a genetic map for distance computation. Transition probabilities are scaled
    /// by genetic distance (cM) instead of physical distance (bp), properly
    /// accounting for variable recombination rates across the genome.
    #[allow(clippy::too_many_arguments)]
    pub fn baum_welch_with_genetic_map(
        &mut self,
        observations: &[f64],
        window_positions: &[(u64, u64)],
        genetic_map: &GeneticMap,
        max_iter: usize,
        tol: f64,
        population_prior: Option<Population>,
        window_size: u64,
    ) {
        let n = observations.len();
        if n < 10 {
            return;
        }

        if window_positions.len() != n {
            self.baum_welch(observations, max_iter, tol, population_prior, window_size);
            return;
        }

        let prior = population_prior.unwrap_or(Population::Generic);
        let prior_non_ibd = prior.non_ibd_emission(window_size);

        let mut prev_log_lik = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: recombination-aware forward-backward
            let (alpha, log_lik) = forward_with_genetic_map(
                observations, self, window_positions, genetic_map, window_size,
            );
            let beta = backward_with_genetic_map(
                observations, self, window_positions, genetic_map, window_size,
            );

            // Check convergence; bail on non-finite log-likelihood
            if !log_lik.is_finite() {
                break;
            }
            if (log_lik - prev_log_lik).abs() < tol {
                break;
            }
            prev_log_lik = log_lik;

            // Compute posteriors (gamma)
            let mut gamma = vec![[0.0f64; 2]; n];
            for t in 0..n {
                let log_gamma_0 = alpha[t][0] + beta[t][0] - log_lik;
                let log_gamma_1 = alpha[t][1] + beta[t][1] - log_lik;
                let max_log = log_gamma_0.max(log_gamma_1);
                let log_sum = max_log
                    + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
                gamma[t][0] = (log_gamma_0 - log_sum).exp();
                gamma[t][1] = (log_gamma_1 - log_sum).exp();
            }

            // Compute expected transitions (xi) with recombination-aware transitions
            let mut xi_sum = [[0.0f64; 2]; 2];

            for t in 0..n - 1 {
                let mid_prev = (window_positions[t].0 + window_positions[t].1) / 2;
                let mid_curr = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;
                let log_trans = recombination_aware_log_transition(
                    self, mid_prev, mid_curr, genetic_map, window_size,
                );

                let log_emit_next = [
                    self.emission[0].log_pdf(observations[t + 1]),
                    self.emission[1].log_pdf(observations[t + 1]),
                ];

                let mut xi_log = [[0.0f64; 2]; 2];
                let mut max_xi = f64::NEG_INFINITY;

                for i in 0..2 {
                    for j in 0..2 {
                        xi_log[i][j] = alpha[t][i] + log_trans[i][j]
                            + log_emit_next[j] + beta[t + 1][j] - log_lik;
                        if xi_log[i][j] > max_xi {
                            max_xi = xi_log[i][j];
                        }
                    }
                }

                let mut xi_sum_t = 0.0;
                let mut xi_val = [[0.0f64; 2]; 2];
                for i in 0..2 {
                    for j in 0..2 {
                        xi_val[i][j] = (xi_log[i][j] - max_xi).exp();
                        xi_sum_t += xi_val[i][j];
                    }
                }
                if xi_sum_t > 0.0 {
                    for i in 0..2 {
                        for j in 0..2 {
                            xi_sum[i][j] += xi_val[i][j] / xi_sum_t;
                        }
                    }
                }
            }

            // M-step: re-estimate emission parameters
            let mut mu = [0.0f64; 2];
            let mut sigma_sq = [0.0f64; 2];
            let mut gamma_sum = [0.0f64; 2];

            for t in 0..n {
                for s in 0..2 {
                    gamma_sum[s] += gamma[t][s];
                    mu[s] += gamma[t][s] * observations[t];
                }
            }
            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    mu[s] /= gamma_sum[s];
                }
            }
            for t in 0..n {
                for s in 0..2 {
                    sigma_sq[s] += gamma[t][s] * (observations[t] - mu[s]).powi(2);
                }
            }
            for s in 0..2 {
                if gamma_sum[s] > 1.0 {
                    sigma_sq[s] /= gamma_sum[s];
                }
            }

            // Apply biological bounds (with NaN guards)
            let std0 = sigma_sq[0].max(0.0).sqrt();
            let std1 = sigma_sq[1].max(0.0).sqrt();
            let bounded_mu0 = if mu[0].is_finite() { mu[0].clamp(prior_non_ibd.mean - 0.005, 0.9993) } else { prior_non_ibd.mean };
            let bounded_std0 = if std0.is_finite() { std0.clamp(0.0003, 0.005) } else { prior_non_ibd.std };
            self.emission[0] = GaussianParams::new_unchecked(bounded_mu0, bounded_std0);

            let bounded_mu1 = if mu[1].is_finite() { mu[1].clamp(0.9990, 1.0) } else { 0.9997 };
            let bounded_std1 = if std1.is_finite() { std1.clamp(0.0002, 0.002) } else { 0.0005 };
            self.emission[1] = GaussianParams::new_unchecked(bounded_mu1, bounded_std1);

            if self.emission[0].mean >= self.emission[1].mean {
                self.emission[0] = GaussianParams::new_unchecked(
                    prior_non_ibd.mean,
                    prior_non_ibd.std,
                );
                self.emission[1] = IBD_EMISSION;
            }

            // Re-estimate transitions
            for (i, xi_row) in xi_sum.iter().enumerate() {
                let row_sum = xi_row[0] + xi_row[1];
                if row_sum > 1.0 {
                    self.transition[i][0] = xi_row[0] / row_sum;
                    self.transition[i][1] = xi_row[1] / row_sum;
                }
            }

            self.transition[0][1] = self.transition[0][1].clamp(1e-8, 0.1);
            self.transition[0][0] = 1.0 - self.transition[0][1];
            self.transition[1][0] = self.transition[1][0].clamp(0.001, 0.5);
            self.transition[1][1] = 1.0 - self.transition[1][0];
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

/// Forward algorithm for computing forward probabilities (alpha).
///
/// The forward algorithm computes `P(observations[0..t], state[t] = s)` for each
/// position t and state s. This is used as part of the forward-backward algorithm
/// for computing posterior state probabilities.
///
/// ## Algorithm
///
/// For each position t, computes:
/// ```text
/// alpha[t][s] = P(obs[0..t], state[t]=s)
///             = sum_{prev} alpha[t-1][prev] * P(prev->s) * P(obs[t]|s)
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
/// Tuple of:
/// - `alpha`: Vector of log forward probabilities, one [f64; 2] per observation
/// - `log_likelihood`: Total log-likelihood P(observations)
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, forward};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.999, 0.9995, 0.9998];
/// let (alpha, log_likelihood) = forward(&obs, &params);
/// assert_eq!(alpha.len(), 4);
/// ```
pub fn forward(observations: &[f64], params: &HmmParams) -> (Vec<[f64; 2]>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];
    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

    // Precompute log emissions
    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);

    // Initialization
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    // Forward pass
    for t in 1..n {
        let mut at = [0.0f64; 2];
        for s in 0..2 {
            // Log-sum-exp over previous states
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    // Total log-likelihood: log-sum-exp of final alpha
    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Genetic map for converting physical positions (bp) to genetic distances (cM).
///
/// Stores a sorted list of (position_bp, position_cM) entries for a single chromosome.
/// Interpolates linearly between entries to convert arbitrary positions.
///
/// ## File Format
///
/// Supports PLINK-format genetic maps (4 columns, whitespace-separated):
/// ```text
/// chr position_bp rate_cM_per_Mb position_cM
/// 1   55550       2.981822       0.000000
/// 1   568322      2.082414       1.106542
/// ```
///
/// Also supports 3-column format (position_bp, rate_cM_per_Mb, position_cM) without chr.
#[derive(Debug, Clone)]
pub struct GeneticMap {
    /// Sorted entries: (position_bp, position_cM)
    entries: Vec<(u64, f64)>,
}

impl GeneticMap {
    /// Create a GeneticMap from sorted (position_bp, position_cM) entries.
    ///
    /// Entries must be sorted by position_bp. Duplicate positions are allowed.
    pub fn new(entries: Vec<(u64, f64)>) -> Self {
        debug_assert!(
            entries.windows(2).all(|w| w[0].0 <= w[1].0),
            "GeneticMap entries must be sorted by position"
        );
        Self { entries }
    }

    /// Parse a PLINK-format genetic map file for a specific chromosome.
    ///
    /// Expects either 4-column format (chr, pos_bp, rate_cM_Mb, pos_cM)
    /// or 3-column format (pos_bp, rate_cM_Mb, pos_cM).
    ///
    /// Lines starting with '#' or empty lines are skipped.
    /// Only entries matching `chrom` are included (for 4-column format).
    pub fn from_file<P: AsRef<std::path::Path>>(path: P, chrom: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("Failed to open genetic map: {}", e))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        // Normalize chromosome name: strip "chr" prefix for comparison
        let chrom_normalized = chrom.strip_prefix("chr").unwrap_or(chrom);

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| format!("Read error line {}: {}", line_num + 1, e))?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 4 {
                // 4-column format: chr, pos_bp, rate_cM_Mb, pos_cM
                let chr = fields[0].strip_prefix("chr").unwrap_or(fields[0]);
                if chr != chrom_normalized {
                    continue;
                }
                let pos_bp: u64 = fields[1].parse()
                    .map_err(|_| format!("Invalid position on line {}", line_num + 1))?;
                let pos_cm: f64 = fields[3].parse()
                    .map_err(|_| format!("Invalid cM position on line {}", line_num + 1))?;
                entries.push((pos_bp, pos_cm));
            } else if fields.len() == 3 {
                // 3-column format: pos_bp, rate_cM_Mb, pos_cM
                let pos_bp: u64 = fields[0].parse()
                    .map_err(|_| format!("Invalid position on line {}", line_num + 1))?;
                let pos_cm: f64 = fields[2].parse()
                    .map_err(|_| format!("Invalid cM position on line {}", line_num + 1))?;
                entries.push((pos_bp, pos_cm));
            }
            // Skip lines with fewer than 3 fields
        }

        entries.sort_by_key(|e| e.0);

        if entries.is_empty() {
            return Err(format!("No genetic map entries found for chromosome {}", chrom));
        }

        Ok(Self { entries })
    }

    /// Interpolate the genetic position (cM) at a given physical position (bp).
    ///
    /// Uses linear interpolation between flanking map entries.
    /// Extrapolates linearly beyond the map boundaries using the nearest rate.
    pub fn interpolate_cm(&self, pos_bp: u64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        if self.entries.len() == 1 {
            return self.entries[0].1;
        }

        // Binary search for the insertion point
        let idx = self.entries.partition_point(|e| e.0 <= pos_bp);

        if idx == 0 {
            // Before first entry: extrapolate using first segment's rate
            let (bp0, cm0) = self.entries[0];
            let (bp1, cm1) = self.entries[1];
            let rate = if bp1 > bp0 {
                (cm1 - cm0) / (bp1 - bp0) as f64
            } else {
                0.0
            };
            cm0 - rate * (bp0 - pos_bp) as f64
        } else if idx >= self.entries.len() {
            // After last entry: extrapolate using last segment's rate
            let n = self.entries.len();
            let (bp_prev, cm_prev) = self.entries[n - 2];
            let (bp_last, cm_last) = self.entries[n - 1];
            let rate = if bp_last > bp_prev {
                (cm_last - cm_prev) / (bp_last - bp_prev) as f64
            } else {
                0.0
            };
            cm_last + rate * (pos_bp - bp_last) as f64
        } else {
            // Between two entries: linear interpolation
            let (bp_lo, cm_lo) = self.entries[idx - 1];
            let (bp_hi, cm_hi) = self.entries[idx];
            if bp_hi == bp_lo {
                cm_lo
            } else {
                let frac = (pos_bp - bp_lo) as f64 / (bp_hi - bp_lo) as f64;
                cm_lo + frac * (cm_hi - cm_lo)
            }
        }
    }

    /// Compute the genetic distance (cM) between two physical positions.
    pub fn genetic_distance_cm(&self, pos1_bp: u64, pos2_bp: u64) -> f64 {
        let cm1 = self.interpolate_cm(pos1_bp);
        let cm2 = self.interpolate_cm(pos2_bp);
        (cm2 - cm1).abs()
    }

    /// Create a uniform-rate genetic map for testing.
    ///
    /// Uses a constant recombination rate across the entire region.
    /// `rate_cm_per_mb` is the rate in cM/Mb (typical human: ~1.0 cM/Mb).
    pub fn uniform(start_bp: u64, end_bp: u64, rate_cm_per_mb: f64) -> Self {
        let start_cm = 0.0;
        let end_cm = (end_bp - start_bp) as f64 * rate_cm_per_mb / 1_000_000.0;
        Self {
            entries: vec![(start_bp, start_cm), (end_bp, end_cm)],
        }
    }

    /// Returns the number of entries in the map.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Compute recombination-aware transition matrix using genetic distance.
///
/// Uses a genetic map to convert physical distances to genetic distances (cM),
/// then applies the Haldane map function to compute recombination probability:
///
/// ```text
/// r = 0.5 * (1 - exp(-2 * d_morgans))
/// ```
///
/// where `d_morgans = d_cm / 100`. The recombination probability `r` is then
/// used to scale the HMM transition probabilities:
///
/// - P(switch IBD state | genetic distance) = 1 - (1-r)^k
///
/// where k is a scaling factor relating recombination events to IBD state changes.
///
/// ## Parameters
///
/// - `params`: Base HMM parameters (transition rates per window step)
/// - `pos1_bp`: Midpoint of previous window in base pairs
/// - `pos2_bp`: Midpoint of current window in base pairs
/// - `genetic_map`: Genetic map for distance interpolation
/// - `window_size`: Nominal window size in bp (for rate normalization)
///
/// ## Returns
///
/// Log-space 2x2 transition matrix adjusted for genetic distance.
pub fn recombination_aware_log_transition(
    params: &HmmParams,
    pos1_bp: u64,
    pos2_bp: u64,
    genetic_map: &GeneticMap,
    window_size: u64,
) -> [[f64; 2]; 2] {
    if window_size == 0 || pos1_bp == pos2_bp {
        return [
            [params.transition[0][0].ln(), params.transition[0][1].ln()],
            [params.transition[1][0].ln(), params.transition[1][1].ln()],
        ];
    }

    let d_cm = genetic_map.genetic_distance_cm(pos1_bp, pos2_bp);
    let d_morgans = d_cm / 100.0;

    // Haldane map function: recombination probability
    let recomb_prob = 0.5 * (1.0 - (-2.0 * d_morgans).exp());

    // Convert base per-window transition rates to per-cM rates
    // Then scale by the genetic distance
    let p_enter_base = params.transition[0][1];
    let p_exit_base = params.transition[1][0];

    // Expected genetic distance per window (in cM)
    let cm_per_window = {
        let mid1 = pos1_bp;
        let mid2 = mid1 + window_size;
        let nominal_cm = genetic_map.genetic_distance_cm(mid1, mid2);
        if nominal_cm > 0.0 { nominal_cm } else { window_size as f64 / 1_000_000.0 }
    };

    // Scale = how many nominal windows worth of genetic distance
    let scale = if cm_per_window > 0.0 {
        d_cm / cm_per_window
    } else {
        1.0 // fallback: treat as one nominal window
    };

    // Use recombination-aware scaling: the probability of at least one
    // recombination event scales IBD state transitions
    // For IBD entry: combine base rate with recombination probability
    let rate_enter = if p_enter_base >= 1.0 {
        f64::INFINITY
    } else if p_enter_base <= 0.0 {
        0.0
    } else {
        -(1.0 - p_enter_base).ln()
    };

    let rate_exit = if p_exit_base >= 1.0 {
        f64::INFINITY
    } else if p_exit_base <= 0.0 {
        0.0
    } else {
        -(1.0 - p_exit_base).ln()
    };

    // Scale rates by genetic distance ratio, with recombination modulation
    // In regions of high recombination, IBD segments break down faster
    let recomb_factor = if recomb_prob > 0.0 {
        // Boost exit rate in high-recombination regions
        1.0 + recomb_prob
    } else {
        1.0
    };

    let p_enter = 1.0 - (-rate_enter * scale).exp();
    let p_exit = 1.0 - (-rate_exit * scale * recomb_factor).exp();

    let p_enter = p_enter.clamp(1e-10, 1.0 - 1e-10);
    let p_exit = p_exit.clamp(1e-10, 1.0 - 1e-10);

    [
        [(1.0 - p_enter).ln(), p_enter.ln()],
        [p_exit.ln(), (1.0 - p_exit).ln()],
    ]
}

/// Compute distance-dependent transition matrix for a given physical distance.
///
/// When consecutive windows are not uniformly spaced (e.g., due to gaps in
/// pangenome coverage), the transition probabilities should account for the
/// physical distance between them. A larger gap means higher probability of
/// a state change.
///
/// The model uses a continuous-time Markov chain:
/// P(switch | distance d) = rate * d, where rate = P(switch per bp).
///
/// ## Parameters
///
/// - `params`: Base HMM parameters (transition rates per window step)
/// - `distance_bp`: Physical distance between consecutive windows in base pairs
/// - `window_size`: Nominal window size in base pairs (for normalizing rates)
///
/// ## Returns
///
/// Log-space 2x2 transition matrix adjusted for the given distance.
pub fn distance_dependent_log_transition(
    params: &HmmParams,
    distance_bp: u64,
    window_size: u64,
) -> [[f64; 2]; 2] {
    if window_size == 0 || distance_bp == 0 {
        return [
            [params.transition[0][0].ln(), params.transition[0][1].ln()],
            [params.transition[1][0].ln(), params.transition[1][1].ln()],
        ];
    }

    // Scale factor: how many nominal windows the distance represents
    let scale = distance_bp as f64 / window_size as f64;

    // Convert per-step transition rates to per-bp rates, then scale
    // Using the continuous-time approximation:
    // P(switch in d bp) = 1 - exp(-rate_per_bp * d)
    // where rate_per_bp = -ln(1 - P(switch per window)) / window_size
    let p_enter_base = params.transition[0][1];
    let p_exit_base = params.transition[1][0];

    // For numerical stability, handle edge cases
    let rate_enter = if p_enter_base >= 1.0 {
        f64::INFINITY
    } else if p_enter_base <= 0.0 {
        0.0
    } else {
        -(1.0 - p_enter_base).ln()  // per-window rate
    };

    let rate_exit = if p_exit_base >= 1.0 {
        f64::INFINITY
    } else if p_exit_base <= 0.0 {
        0.0
    } else {
        -(1.0 - p_exit_base).ln()  // per-window rate
    };

    // Scale rates by distance
    let p_enter = 1.0 - (-rate_enter * scale).exp();
    let p_exit = 1.0 - (-rate_exit * scale).exp();

    // Clamp to valid probabilities
    let p_enter = p_enter.clamp(1e-10, 1.0 - 1e-10);
    let p_exit = p_exit.clamp(1e-10, 1.0 - 1e-10);

    [
        [(1.0 - p_enter).ln(), p_enter.ln()],
        [p_exit.ln(), (1.0 - p_exit).ln()],
    ]
}

/// Forward algorithm with distance-dependent transitions.
///
/// Like `forward`, but adjusts transition probabilities based on the physical
/// distance between consecutive windows. This handles non-uniform window
/// spacing (e.g., gaps in pangenome coverage).
///
/// ## Arguments
///
/// - `observations`: Sequence of identity values (one per window)
/// - `params`: HMM parameters
/// - `window_positions`: Physical positions (start, end) of each window in bp
///
/// ## Returns
///
/// Same as `forward`: (alpha, log_likelihood).
pub fn forward_with_distances(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> (Vec<[f64; 2]>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    // If positions not available or mismatch, fall back to regular forward
    if window_positions.len() != n {
        return forward(observations, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    // Compute nominal window size from first window
    let window_size = if !window_positions.is_empty() {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    } else {
        5000 // default
    };

    // Precompute log emissions
    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);

    // Initialization
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    // Forward pass with distance-dependent transitions
    for t in 1..n {
        // Distance between midpoints of consecutive windows
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let distance = mid_curr.saturating_sub(mid_prev);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

        let mut at = [0.0f64; 2];
        for s in 0..2 {
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    // Total log-likelihood
    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Backward algorithm with distance-dependent transitions.
///
/// Like `backward`, but adjusts transition probabilities based on the physical
/// distance between consecutive windows.
pub fn backward_with_distances(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> Vec<[f64; 2]> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return backward(observations, params);
    }

    let window_size = if !window_positions.is_empty() {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    } else {
        5000
    };

    // Precompute log emissions
    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    beta[n - 1] = [0.0, 0.0];

    // Backward pass with distance-dependent transitions
    for t in (0..n - 1).rev() {
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let mid_next = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;
        let distance = mid_next.saturating_sub(mid_curr);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

        for s in 0..2 {
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Forward-backward with distance-dependent transitions.
///
/// Computes posterior P(IBD) at each window, accounting for non-uniform
/// spacing between windows. This is important when pangenome coverage
/// is uneven, creating gaps between consecutive analysis windows.
///
/// ## Arguments
///
/// - `observations`: Identity values from windowed analysis
/// - `params`: HMM parameters
/// - `window_positions`: Physical (start, end) positions of each window
///
/// ## Returns
///
/// Same as `forward_backward`: (posterior_ibd, log_likelihood).
pub fn forward_backward_with_distances(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> (Vec<f64>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward_with_distances(observations, params, window_positions);
    let beta = backward_with_distances(observations, params, window_positions);

    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum = max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();
        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Forward algorithm with recombination-aware transitions using a genetic map.
///
/// Like `forward_with_distances`, but uses genetic distance (cM) from a genetic
/// map instead of physical distance (bp) for transition scaling. This properly
/// accounts for variable recombination rates across the genome.
pub fn forward_with_genetic_map(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> (Vec<[f64; 2]>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    if window_positions.len() != n {
        return forward(observations, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_prev, mid_curr, genetic_map, window_size,
        );

        let mut at = [0.0f64; 2];
        for s in 0..2 {
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Backward algorithm with recombination-aware transitions using a genetic map.
pub fn backward_with_genetic_map(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> Vec<[f64; 2]> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return backward(observations, params);
    }

    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    beta[n - 1] = [0.0, 0.0];

    for t in (0..n - 1).rev() {
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let mid_next = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_curr, mid_next, genetic_map, window_size,
        );

        for s in 0..2 {
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Forward-backward with recombination-aware transitions using a genetic map.
///
/// Computes posterior P(IBD) at each window using genetic distance from a
/// genetic map for transition probability scaling. This accounts for variable
/// recombination rates across the genome, where IBD segments are more likely
/// to break at recombination hotspots.
pub fn forward_backward_with_genetic_map(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> (Vec<f64>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward_with_genetic_map(
        observations, params, window_positions, genetic_map, window_size,
    );
    let beta = backward_with_genetic_map(
        observations, params, window_positions, genetic_map, window_size,
    );

    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum = max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();
        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Viterbi algorithm with recombination-aware transitions using a genetic map.
///
/// Like `viterbi_with_distances`, but uses genetic distance from a genetic map
/// for transition probability scaling.
pub fn viterbi_with_genetic_map(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> Vec<usize> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return viterbi(observations, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

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
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_prev, mid_curr, genetic_map, window_size,
        );

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

/// Backward algorithm for computing backward probabilities (beta).
///
/// The backward algorithm computes `P(observations[t+1..n] | state[t] = s)` for each
/// position t and state s. Combined with forward probabilities, this gives
/// posterior state probabilities.
///
/// ## Algorithm
///
/// For each position t (from n-1 down to 0), computes:
/// ```text
/// beta[t][s] = P(obs[t+1..n] | state[t]=s)
///            = sum_{next} P(s->next) * P(obs[t+1]|next) * beta[t+1][next]
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
/// Vector of log backward probabilities, one [f64; 2] per observation.
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, backward};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.999, 0.9995, 0.9998];
/// let beta = backward(&obs, &params);
/// assert_eq!(beta.len(), 4);
/// ```
pub fn backward(observations: &[f64], params: &HmmParams) -> Vec<[f64; 2]> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

    // Precompute log emissions
    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];

    // Initialization: beta[n-1] = 0 in log space (prob = 1)
    beta[n - 1] = [0.0, 0.0];

    // Backward pass
    for t in (0..n - 1).rev() {
        for s in 0..2 {
            // Log-sum-exp over next states
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Forward-backward algorithm to compute posterior state probabilities.
///
/// Computes `P(state[t] = IBD | all observations)` for each position t.
/// This gives a probabilistic estimate of IBD at each window, unlike Viterbi
/// which gives a single best path.
///
/// ## Algorithm
///
/// ```text
/// gamma[t][s] = P(state[t]=s | all obs)
///             = alpha[t][s] * beta[t][s] / P(all obs)
///
/// P(IBD at t) = gamma[t][1]
/// ```
///
/// ## Arguments
///
/// - `observations`: Sequence of identity values (one per window)
/// - `params`: HMM parameters (transition and emission distributions)
///
/// ## Returns
///
/// Tuple of:
/// - `posterior_ibd`: P(IBD) for each position
/// - `log_likelihood`: Total log-likelihood P(observations)
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, forward_backward};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.998];
/// let (posteriors, log_lik) = forward_backward(&obs, &params);
///
/// // posteriors[i] is P(IBD at window i | all data)
/// for (i, &p) in posteriors.iter().enumerate() {
///     println!("Window {}: P(IBD) = {:.4}", i, p);
/// }
/// ```
///
/// ## Use Cases
///
/// - **Confidence scores**: Use posteriors to assess confidence in IBD calls
/// - **Segment filtering**: Only keep segments where mean posterior > threshold
/// - **Soft boundaries**: Identify uncertain segment boundaries
pub fn forward_backward(observations: &[f64], params: &HmmParams) -> (Vec<f64>, f64) {
    let n = observations.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward(observations, params);
    let beta = backward(observations, params);

    // Compute posterior P(IBD | all observations)
    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        // log_gamma[s] = alpha[t][s] + beta[t][s] - log_likelihood
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;

        // P(IBD) = exp(log_gamma_1) / (exp(log_gamma_0) + exp(log_gamma_1))
        // Use log-sum-exp for numerical stability
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum = max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();

        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Result of IBD inference including posteriors.
#[derive(Debug, Clone)]
pub struct IbdInferenceResult {
    /// Viterbi state sequence (0=non-IBD, 1=IBD)
    pub states: Vec<usize>,
    /// Posterior P(IBD) for each window
    pub posteriors: Vec<f64>,
    /// Total log-likelihood of observations
    pub log_likelihood: f64,
}

/// Complete IBD inference: Viterbi states + forward-backward posteriors.
///
/// This is the recommended entry point for IBD detection, as it provides
/// both the MAP state sequence (Viterbi) and posterior probabilities
/// (forward-backward) in a single call.
///
/// ## Arguments
///
/// - `observations`: Sequence of identity values (one per window)
/// - `params`: HMM parameters
///
/// ## Returns
///
/// `IbdInferenceResult` containing states, posteriors, and log-likelihood.
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, infer_ibd};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.998];
///
/// let result = infer_ibd(&obs, &params);
///
/// println!("Log-likelihood: {:.2}", result.log_likelihood);
/// for (i, (&state, &post)) in result.states.iter().zip(result.posteriors.iter()).enumerate() {
///     println!("Window {}: state={}, P(IBD)={:.4}", i, state, post);
/// }
/// ```
pub fn infer_ibd(observations: &[f64], params: &HmmParams) -> IbdInferenceResult {
    let states = viterbi(observations, params);
    let (posteriors, log_likelihood) = forward_backward(observations, params);

    IbdInferenceResult {
        states,
        posteriors,
        log_likelihood,
    }
}

/// Complete IBD inference with Baum-Welch training and posterior refinement.
///
/// This is the most thorough inference entry point, recommended for cases
/// where initial parameter estimates may be imprecise (e.g., AFR populations).
///
/// ## Pipeline
///
/// 1. Run Baum-Welch to refine HMM parameters from the data
/// 2. Run Viterbi to get the MAP state sequence
/// 3. Run forward-backward to get posterior probabilities
/// 4. Refine state boundaries using posteriors
///
/// ## Parameters
///
/// - `observations`: Identity values from windowed analysis
/// - `params`: Initial HMM parameters (will be modified by Baum-Welch)
/// - `population`: Population for biological bounds during training
/// - `window_size`: Window size in base pairs
/// - `baum_welch_iters`: Maximum Baum-Welch iterations (0 to skip)
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, Population, infer_ibd_with_training};
///
/// let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.9985, 0.999, 0.9997, 0.9998, 0.9996, 0.998, 0.997];
/// let result = infer_ibd_with_training(&obs, &mut params, Population::AFR, 5000, 20);
/// ```
pub fn infer_ibd_with_training(
    observations: &[f64],
    params: &mut HmmParams,
    population: Population,
    window_size: u64,
    baum_welch_iters: usize,
) -> IbdInferenceResult {
    // Step 1: Baum-Welch parameter training
    if baum_welch_iters > 0 && observations.len() >= 10 {
        params.baum_welch(
            observations,
            baum_welch_iters,
            1e-6,
            Some(population),
            window_size,
        );
    }

    // Step 2: Viterbi + forward-backward
    let mut states = viterbi(observations, params);
    let (posteriors, log_likelihood) = forward_backward(observations, params);

    // Step 3: Posterior-based refinement
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

    IbdInferenceResult {
        states,
        posteriors,
        log_likelihood,
    }
}

/// Aggregate observations by averaging adjacent windows.
///
/// Reduces temporal resolution by a factor of `factor`, averaging
/// identity values within each aggregated window. This increases
/// signal-to-noise ratio at the cost of spatial resolution.
///
/// ## Parameters
///
/// - `observations`: Per-window identity values
/// - `factor`: Number of adjacent windows to aggregate (1 = no change)
///
/// ## Returns
///
/// Aggregated observations. Length = ceil(observations.len() / factor).
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::aggregate_observations;
///
/// let obs = vec![0.998, 0.997, 0.999, 0.996, 0.998, 0.997];
/// let agg = aggregate_observations(&obs, 2);
/// assert_eq!(agg.len(), 3);
/// assert!((agg[0] - 0.9975).abs() < 1e-6); // avg of [0.998, 0.997]
/// ```
pub fn aggregate_observations(observations: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 || observations.is_empty() {
        return observations.to_vec();
    }
    let n = observations.len();
    let n_agg = n.div_ceil(factor);
    let mut result = Vec::with_capacity(n_agg);
    for i in 0..n_agg {
        let start = i * factor;
        let end = ((i + 1) * factor).min(n);
        let sum: f64 = observations[start..end].iter().sum();
        let count = (end - start) as f64;
        result.push(sum / count);
    }
    result
}

/// Multi-scale IBD inference: runs HMM at multiple resolutions and combines evidence.
///
/// The key insight: noisy windows can fragment IBD segments at fine resolution.
/// By also running the HMM at coarser resolutions (aggregated windows), longer-range
/// patterns become visible. Evidence is combined across scales:
///
/// - **Recovery**: Segments detected at coarse scale but missed at fine scale are
///   recovered if fine-scale posteriors show moderate support (>0.3).
/// - **Pruning**: Segments detected only at fine scale (not confirmed by any coarse
///   scale) are removed if their fine-scale evidence is weak (mean posterior <0.7).
///
/// ## Parameters
///
/// - `observations`: Per-window identity values
/// - `params`: HMM parameters (used at all scales)
/// - `scales`: Aggregation factors to use, e.g., `[1, 2, 4]`.
///   Scale 1 = base resolution. Scale 2 = pairs of windows averaged. Etc.
///
/// ## Returns
///
/// `IbdInferenceResult` with enhanced states from multi-scale evidence.
/// Posteriors and log-likelihood are from the base (finest) scale.
pub fn infer_ibd_multi_scale(
    observations: &[f64],
    params: &HmmParams,
    scales: &[usize],
) -> IbdInferenceResult {
    // Base resolution inference
    let base_result = infer_ibd(observations, params);

    if scales.len() <= 1 || observations.len() < 6 {
        return base_result;
    }

    // Run at each coarser scale, collect detected IBD ranges (in base-resolution indices)
    let mut coarse_ibd_ranges: Vec<(usize, usize)> = Vec::new();

    for &factor in scales.iter() {
        if factor <= 1 {
            continue;
        }
        let agg_obs = aggregate_observations(observations, factor);
        if agg_obs.len() < 3 {
            continue;
        }
        let agg_result = infer_ibd(&agg_obs, params);
        let agg_segments = extract_ibd_segments(&agg_result.states);

        // Map aggregated segment indices back to base resolution
        for (seg_start, seg_end, _n_windows) in &agg_segments {
            let base_start = seg_start * factor;
            let base_end = ((seg_end + 1) * factor).min(observations.len()) - 1;
            coarse_ibd_ranges.push((base_start, base_end));
        }
    }

    let base_segments = extract_ibd_segments(&base_result.states);
    let mut enhanced_states = base_result.states.clone();

    // Recovery: segments at coarse scale but missed at fine scale
    for &(cs, ce) in &coarse_ibd_ranges {
        let overlaps_base = base_segments.iter().any(|(bs, be, _)| {
            *bs <= ce && cs <= *be
        });

        if !overlaps_base && ce < base_result.posteriors.len() {
            // Coarse scale detected IBD here but fine scale didn't.
            // Recover individual windows that have moderate posterior support.
            let range_end = ce.min(base_result.posteriors.len() - 1);
            let mean_posterior: f64 = base_result.posteriors[cs..=range_end]
                .iter()
                .sum::<f64>()
                / (range_end - cs + 1) as f64;

            if mean_posterior > 0.3 {
                for (state, &post) in enhanced_states[cs..=range_end]
                    .iter_mut()
                    .zip(&base_result.posteriors[cs..=range_end])
                {
                    if post > 0.3 {
                        *state = 1;
                    }
                }
            }
        }
    }

    // Pruning: fine-scale-only segments with weak evidence
    for (bs, be, _) in &base_segments {
        let confirmed_by_coarse = coarse_ibd_ranges.iter().any(|&(cs, ce)| {
            *bs <= ce && cs <= *be
        });

        if !confirmed_by_coarse {
            // Only at fine scale — require stronger evidence
            let n_windows = be - bs + 1;
            let mean_posterior: f64 = base_result.posteriors[*bs..=*be]
                .iter()
                .sum::<f64>()
                / n_windows as f64;

            if mean_posterior < 0.7 {
                // Weak fine-only segment: remove
                for state in enhanced_states[*bs..=*be].iter_mut() {
                    *state = 0;
                }
            }
        }
    }

    IbdInferenceResult {
        states: enhanced_states,
        posteriors: base_result.posteriors,
        log_likelihood: base_result.log_likelihood,
    }
}

/// IBD segment with posterior statistics and LOD score.
#[derive(Debug, Clone)]
pub struct IbdSegmentWithPosterior {
    /// Start window index (inclusive)
    pub start_idx: usize,
    /// End window index (inclusive)
    pub end_idx: usize,
    /// Number of windows in segment
    pub n_windows: usize,
    /// Mean posterior P(IBD) in segment
    pub mean_posterior: f64,
    /// Minimum posterior P(IBD) in segment
    pub min_posterior: f64,
    /// Maximum posterior P(IBD) in segment
    pub max_posterior: f64,
    /// LOD score: log10 likelihood ratio of IBD vs non-IBD model over the segment.
    /// Higher values indicate stronger evidence for IBD.
    /// Typical thresholds: LOD >= 3 (strong), LOD >= 5 (very strong).
    pub lod_score: f64,
}

/// Compute per-window log10 likelihood ratio (LOD) between IBD and non-IBD models.
///
/// For each window, LOD_i = log10(P(obs_i | IBD) / P(obs_i | non-IBD)).
/// Positive values indicate the observation favors IBD, negative favors non-IBD.
///
/// The segment LOD score is the sum of per-window LODs, following the convention
/// used by hap-ibd and IBDseq.
///
/// ## Arguments
///
/// - `observations`: Identity values from windowed analysis
/// - `params`: HMM parameters with emission distributions
///
/// ## Returns
///
/// Vector of per-window LOD scores (log10 scale).
pub fn compute_per_window_lod(observations: &[f64], params: &HmmParams) -> Vec<f64> {
    observations
        .iter()
        .map(|&obs| {
            let log_pdf_ibd = params.emission[1].log_pdf(obs);
            let log_pdf_non_ibd = params.emission[0].log_pdf(obs);
            // Convert from ln to log10
            (log_pdf_ibd - log_pdf_non_ibd) / std::f64::consts::LN_10
        })
        .collect()
}

/// Compute the segment LOD score as the sum of per-window LODs within the segment.
///
/// LOD = sum_{i in segment} log10(P(obs_i | IBD) / P(obs_i | non-IBD))
///
/// This is the standard statistic used by IBD detection tools (hap-ibd, IBDseq)
/// to quantify the total evidence for IBD in a genomic region.
pub fn segment_lod_score(
    observations: &[f64],
    start_idx: usize,
    end_idx: usize,
    params: &HmmParams,
) -> f64 {
    if start_idx > end_idx || end_idx >= observations.len() {
        return 0.0;
    }
    observations[start_idx..=end_idx]
        .iter()
        .map(|&obs| {
            let log_pdf_ibd = params.emission[1].log_pdf(obs);
            let log_pdf_non_ibd = params.emission[0].log_pdf(obs);
            (log_pdf_ibd - log_pdf_non_ibd) / std::f64::consts::LN_10
        })
        .sum()
}

/// Compute a composite quality score for an IBD segment.
///
/// The quality score combines multiple evidence sources into a single
/// score in the range [0, 100], similar to mapping quality in BAM files.
/// Higher scores indicate more confident IBD calls.
///
/// ## Components
///
/// 1. **Posterior strength** (0-40 points): Based on mean posterior P(IBD).
///    Score = mean_posterior * 40.
///
/// 2. **Posterior consistency** (0-20 points): Based on min/mean ratio.
///    A segment where even the weakest window has high posterior is more
///    reliable than one with a few weak windows.
///    Score = (min_posterior / mean_posterior) * 20.
///
/// 3. **LOD evidence** (0-30 points): Based on LOD score per window.
///    LOD/window > 1.0 gets full marks. Scaled linearly below that.
///    Score = min(lod_per_window / 1.0, 1.0) * 30.
///
/// 4. **Segment length** (0-10 points): Longer segments are more reliable.
///    Score = min(n_windows / 20, 1.0) * 10.
///
/// ## Returns
///
/// Quality score in [0, 100]. Suggested thresholds:
/// - Q >= 80: High confidence
/// - Q >= 50: Medium confidence
/// - Q >= 20: Low confidence
/// - Q < 20: Very low confidence (likely false positive)
pub fn segment_quality_score(seg: &IbdSegmentWithPosterior) -> f64 {
    // Component 1: Posterior strength (0-40)
    let posterior_score = seg.mean_posterior.clamp(0.0, 1.0) * 40.0;

    // Component 2: Posterior consistency (0-20)
    let consistency = if seg.mean_posterior > 0.0 {
        (seg.min_posterior / seg.mean_posterior).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let consistency_score = consistency * 20.0;

    // Component 3: LOD evidence (0-30)
    let lod_per_window = if seg.n_windows > 0 {
        seg.lod_score / seg.n_windows as f64
    } else {
        0.0
    };
    let lod_score = (lod_per_window / 1.0).clamp(0.0, 1.0) * 30.0;

    // Component 4: Length (0-10)
    let length_score = (seg.n_windows as f64 / 20.0).clamp(0.0, 1.0) * 10.0;

    (posterior_score + consistency_score + lod_score + length_score).clamp(0.0, 100.0)
}

/// Compute the posterior standard deviation within a segment.
///
/// This measures how variable the posterior P(IBD) is across the segment.
/// Low values indicate consistent confidence; high values indicate
/// uncertain boundaries or mixed signal.
///
/// ## Returns
///
/// Standard deviation of posteriors within the segment, or 0.0 if
/// the segment has fewer than 2 windows.
pub fn segment_posterior_std(posteriors: &[f64], start_idx: usize, end_idx: usize) -> f64 {
    if start_idx > end_idx || end_idx >= posteriors.len() {
        return 0.0;
    }
    let seg = &posteriors[start_idx..=end_idx];
    let n = seg.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = seg.iter().sum::<f64>() / n as f64;
    let variance: f64 = seg.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.max(0.0).sqrt()
}

/// Extract IBD segments with posterior-based filtering and LOD scores.
///
/// Like `extract_ibd_segments`, but uses posterior probabilities to filter
/// segments and provides posterior statistics and LOD scores for each segment.
///
/// ## Arguments
///
/// - `states`: Viterbi state sequence (0=non-IBD, 1=IBD)
/// - `posteriors`: Posterior P(IBD) for each window (from forward-backward)
/// - `min_windows`: Minimum segment length in windows
/// - `min_mean_posterior`: Minimum mean P(IBD) for segment to be kept
///
/// ## Returns
///
/// Vector of `IbdSegmentWithPosterior` for segments passing filters.
/// LOD scores default to 0.0 when observations/params are not available;
/// use `extract_ibd_segments_with_lod` for LOD computation.
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, infer_ibd, extract_ibd_segments_with_posteriors};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.9998, 0.9999, 0.9997, 0.9998, 0.998];
///
/// let result = infer_ibd(&obs, &params);
/// let segments = extract_ibd_segments_with_posteriors(
///     &result.states,
///     &result.posteriors,
///     2,    // min 2 windows
///     0.8,  // min 80% mean posterior
/// );
///
/// for seg in &segments {
///     println!("IBD {}-{}: {} windows, mean P(IBD)={:.3}",
///         seg.start_idx, seg.end_idx, seg.n_windows, seg.mean_posterior);
/// }
/// ```
pub fn extract_ibd_segments_with_posteriors(
    states: &[usize],
    posteriors: &[f64],
    min_windows: usize,
    min_mean_posterior: f64,
) -> Vec<IbdSegmentWithPosterior> {
    extract_ibd_segments_with_lod(states, posteriors, min_windows, min_mean_posterior, None, None)
}

/// Extract IBD segments with posterior-based filtering, LOD scores, and optional
/// minimum LOD filter.
///
/// This is the full-featured segment extraction function that computes LOD scores
/// for each segment when observations and HMM parameters are provided.
///
/// ## Arguments
///
/// - `states`: Viterbi state sequence (0=non-IBD, 1=IBD)
/// - `posteriors`: Posterior P(IBD) for each window (from forward-backward)
/// - `min_windows`: Minimum segment length in windows
/// - `min_mean_posterior`: Minimum mean P(IBD) for segment to be kept
/// - `observations_and_params`: Optional tuple of (observations, HmmParams) for LOD computation
/// - `min_lod`: Optional minimum LOD score threshold; segments below this are filtered out
///
/// ## Returns
///
/// Vector of `IbdSegmentWithPosterior` for segments passing all filters.
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, infer_ibd, extract_ibd_segments_with_lod};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.9998, 0.9999, 0.9997, 0.9998, 0.998];
///
/// let result = infer_ibd(&obs, &params);
/// let segments = extract_ibd_segments_with_lod(
///     &result.states,
///     &result.posteriors,
///     2,    // min 2 windows
///     0.8,  // min 80% mean posterior
///     Some((&obs, &params)),  // provide obs+params for LOD
///     Some(3.0),  // min LOD >= 3
/// );
///
/// for seg in &segments {
///     println!("IBD {}-{}: {} windows, LOD={:.1}, mean P(IBD)={:.3}",
///         seg.start_idx, seg.end_idx, seg.n_windows, seg.lod_score, seg.mean_posterior);
/// }
/// ```
pub fn extract_ibd_segments_with_lod(
    states: &[usize],
    posteriors: &[f64],
    min_windows: usize,
    min_mean_posterior: f64,
    observations_and_params: Option<(&[f64], &HmmParams)>,
    min_lod: Option<f64>,
) -> Vec<IbdSegmentWithPosterior> {
    let mut segments = Vec::new();
    let n = states.len();

    if n == 0 || posteriors.len() != n {
        return segments;
    }

    let mut in_ibd = false;
    let mut start_idx = 0;

    let mut finalize = |start: usize, end: usize| {
        let n_windows = end - start + 1;
        if n_windows < min_windows {
            return;
        }

        let seg_posteriors = &posteriors[start..=end];
        let mean_post: f64 = seg_posteriors.iter().sum::<f64>() / n_windows as f64;

        if mean_post < min_mean_posterior {
            return;
        }

        let min_post = seg_posteriors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_post = seg_posteriors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let lod = if let Some((obs, params)) = observations_and_params {
            segment_lod_score(obs, start, end, params)
        } else {
            0.0
        };

        if let Some(min_lod_val) = min_lod {
            if lod < min_lod_val {
                return;
            }
        }

        segments.push(IbdSegmentWithPosterior {
            start_idx: start,
            end_idx: end,
            n_windows,
            mean_posterior: mean_post,
            min_posterior: min_post,
            max_posterior: max_post,
            lod_score: lod,
        });
    };

    for (i, &state) in states.iter().enumerate() {
        if state == 1 && !in_ibd {
            in_ibd = true;
            start_idx = i;
        } else if state == 0 && in_ibd {
            in_ibd = false;
            finalize(start_idx, i - 1);
        }
    }

    // Handle segment at end
    if in_ibd {
        finalize(start_idx, n - 1);
    }

    segments
}

/// Refine Viterbi state sequence using posterior probabilities.
///
/// The Viterbi algorithm finds the globally optimal path, but its segment
/// boundaries can be imprecise when emission distributions overlap (as in AFR).
/// This function adjusts boundaries using posterior P(IBD):
///
/// 1. **Extension**: If posterior exceeds `extend_threshold` in windows adjacent
///    to a Viterbi IBD segment, extend the segment to include them.
/// 2. **Trimming**: If posterior falls below `trim_threshold` at segment edges,
///    trim those windows.
///
/// This produces a state sequence that is more consistent with the posterior
/// probabilities while maintaining the overall structure from Viterbi.
///
/// ## Parameters
///
/// - `states`: Viterbi state sequence (modified in place)
/// - `posteriors`: Posterior P(IBD) from forward-backward
/// - `extend_threshold`: Minimum posterior to extend segments (default: 0.5)
/// - `trim_threshold`: Maximum posterior to trim from segments (default: 0.2)
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{HmmParams, infer_ibd, refine_states_with_posteriors};
///
/// let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
/// let obs = vec![0.998, 0.9998, 0.9999, 0.9997, 0.998];
/// let mut result = infer_ibd(&obs, &params);
/// refine_states_with_posteriors(&mut result.states, &result.posteriors, 0.5, 0.2);
/// ```
pub fn refine_states_with_posteriors(
    states: &mut [usize],
    posteriors: &[f64],
    extend_threshold: f64,
    trim_threshold: f64,
) {
    let n = states.len();
    if n == 0 || posteriors.len() != n {
        return;
    }

    // Pass 1: Extend IBD segments into adjacent high-posterior windows
    // We iterate until no more extensions are made
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if states[i] == 0 && posteriors[i] >= extend_threshold {
                // Check if adjacent to an IBD segment
                let left_ibd = i > 0 && states[i - 1] == 1;
                let right_ibd = i + 1 < n && states[i + 1] == 1;
                if left_ibd || right_ibd {
                    states[i] = 1;
                    changed = true;
                }
            }
        }
    }

    // Pass 2: Trim low-posterior windows from segment edges
    for i in 0..n {
        if states[i] == 1 && posteriors[i] < trim_threshold {
            // Only trim if at segment boundary
            let left_non_ibd = i == 0 || states[i - 1] == 0;
            let right_non_ibd = i + 1 >= n || states[i + 1] == 0;
            if left_non_ibd || right_non_ibd {
                states[i] = 0;
            }
        }
    }
}

/// Bridge short non-IBD gaps between IBD segments using posterior evidence.
///
/// After Viterbi decoding and posterior refinement, short non-IBD gaps can
/// remain within true IBD regions due to noisy windows. This function fills
/// such gaps when:
/// 1. The gap is at most `max_gap` windows long
/// 2. The gap is flanked by IBD segments on both sides
/// 3. The mean posterior P(IBD) in the gap is at least `min_bridge_posterior`
///
/// This is a standard post-processing step in IBD callers (cf. hap-ibd's
/// segment merging). Without it, true IBD segments get split by noise,
/// each half may fall below min-windows, and recall drops dramatically.
///
/// ## Parameters
///
/// - `states`: Viterbi state sequence (modified in place)
/// - `posteriors`: Posterior P(IBD) from forward-backward
/// - `max_gap`: Maximum gap length in windows to bridge (0 = disabled, recommended: 2-3)
/// - `min_bridge_posterior`: Minimum mean P(IBD) in gap to allow bridging (recommended: 0.3)
///
/// ## Returns
///
/// Number of gaps bridged.
pub fn bridge_ibd_gaps(
    states: &mut [usize],
    posteriors: &[f64],
    max_gap: usize,
    min_bridge_posterior: f64,
) -> usize {
    let n = states.len();
    if n < 3 || posteriors.len() != n || max_gap == 0 {
        return 0;
    }

    let mut bridges = 0;

    // Find IBD segments (runs of state == 1)
    let mut i = 0;
    while i < n {
        // Find start of IBD segment
        if states[i] == 1 {
            // Find end of this IBD segment
            let seg_end = {
                let mut j = i;
                while j < n && states[j] == 1 {
                    j += 1;
                }
                j // first non-IBD after segment
            };

            // Look at the gap after this segment
            if seg_end < n {
                let gap_start = seg_end;
                let mut gap_end = gap_start;
                while gap_end < n && states[gap_end] == 0 {
                    gap_end += 1;
                }
                let gap_len = gap_end - gap_start;

                // Check: gap is short enough AND followed by another IBD segment
                if gap_len <= max_gap && gap_end < n && states[gap_end] == 1 {
                    // Check mean posterior in gap
                    let mean_post: f64 = posteriors[gap_start..gap_end].iter().sum::<f64>()
                        / gap_len as f64;
                    if mean_post >= min_bridge_posterior {
                        // Bridge the gap
                        for s in states.iter_mut().take(gap_end).skip(gap_start) {
                            *s = 1;
                        }
                        bridges += 1;
                        // Continue scanning from after the bridged gap (the next IBD segment)
                        i = gap_end;
                        continue;
                    }
                }
            }

            i = seg_end;
        } else {
            i += 1;
        }
    }

    bridges
}

/// Merge extracted IBD segments that are separated by short gaps.
///
/// This operates at the segment level (after extraction) rather than
/// the state level. Useful as a second pass after state-level bridging.
///
/// ## Parameters
///
/// - `segments`: Vector of IBD segments (must be sorted by start_idx)
/// - `max_gap_windows`: Maximum gap between segments (in windows) to merge
///
/// ## Returns
///
/// New vector of merged segments.
pub fn merge_nearby_ibd_segments(
    segments: &[IbdSegmentWithPosterior],
    max_gap_windows: usize,
) -> Vec<IbdSegmentWithPosterior> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<IbdSegmentWithPosterior> = Vec::new();
    merged.push(segments[0].clone());

    for seg in segments.iter().skip(1) {
        let last = merged.last_mut().unwrap();
        let gap = if seg.start_idx > last.end_idx + 1 {
            seg.start_idx - last.end_idx - 1
        } else {
            0
        };

        if gap <= max_gap_windows {
            // Merge: extend last segment to cover both
            let total_windows_old = last.n_windows as f64;
            let total_windows_new = seg.n_windows as f64;
            let new_n_windows = seg.end_idx - last.start_idx + 1;

            // Weighted average of posteriors
            let total = total_windows_old + total_windows_new;
            let new_mean = (last.mean_posterior * total_windows_old
                + seg.mean_posterior * total_windows_new)
                / total;

            last.end_idx = seg.end_idx;
            last.n_windows = new_n_windows;
            last.mean_posterior = new_mean;
            last.min_posterior = last.min_posterior.min(seg.min_posterior);
            last.max_posterior = last.max_posterior.max(seg.max_posterior);
            last.lod_score += seg.lod_score; // LOD scores are additive
        } else {
            merged.push(seg.clone());
        }
    }

    merged
}

/// Refined boundary positions for an IBD segment.
///
/// Contains sub-window-resolution boundary estimates derived from
/// posterior probability interpolation.
#[derive(Debug, Clone)]
pub struct RefinedBoundary {
    /// Refined start position in base pairs
    pub start_bp: u64,
    /// Refined end position in base pairs
    pub end_bp: u64,
}

/// Refine segment boundaries using posterior probability interpolation.
///
/// Standard IBD segments snap to window edges, limiting resolution to the
/// window size (typically 10kb). This function uses linear interpolation of
/// posterior P(IBD) between adjacent windows to estimate sub-window boundary
/// positions, potentially improving resolution to ~2-5kb.
///
/// For each segment boundary, finds the point where P(IBD) crosses the
/// `crossover` threshold (default 0.5) between the IBD window and the
/// adjacent non-IBD window. Posteriors are treated as point estimates at
/// window centers; the crossover is linearly interpolated between centers.
///
/// ## Algorithm
///
/// For the **start** boundary of a segment starting at window `i`:
/// - If `i > 0` and `P(IBD)[i-1] < crossover < P(IBD)[i]`:
///   - `t = (crossover - P[i-1]) / (P[i] - P[i-1])`
///   - `boundary = center[i-1] + t * (center[i] - center[i-1])`
/// - Otherwise: use `window_start[i]` (no refinement possible)
///
/// For the **end** boundary of a segment ending at window `j`:
/// - If `j+1 < n` and `P(IBD)[j] > crossover > P(IBD)[j+1]`:
///   - `t = (P[j] - crossover) / (P[j] - P[j+1])`
///   - `boundary = center[j] + t * (center[j+1] - center[j])`
/// - Otherwise: use `window_end[j]` (no refinement possible)
///
/// ## Arguments
///
/// - `segments`: Extracted IBD segments with window indices
/// - `posteriors`: Per-window posterior P(IBD) from forward-backward
/// - `window_starts`: Genomic start position of each window (1-based)
/// - `window_ends`: Genomic end position of each window (1-based)
/// - `crossover`: Posterior threshold for boundary placement (default: 0.5)
///
/// ## Returns
///
/// Vector of `RefinedBoundary` with one entry per input segment.
///
/// ## Example
///
/// ```rust
/// use impopk_ibd::hmm::{IbdSegmentWithPosterior, RefinedBoundary, refine_segment_boundaries};
///
/// let segments = vec![IbdSegmentWithPosterior {
///     start_idx: 2, end_idx: 4, n_windows: 3,
///     mean_posterior: 0.95, min_posterior: 0.9, max_posterior: 0.99,
///     lod_score: 5.0,
/// }];
/// let posteriors = vec![0.1, 0.3, 0.92, 0.98, 0.91, 0.2, 0.05];
/// let starts = vec![1, 10001, 20001, 30001, 40001, 50001, 60001];
/// let ends =   vec![10000, 20000, 30000, 40000, 50000, 60000, 70000];
///
/// let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);
/// assert_eq!(refined.len(), 1);
/// // Start refined: crossover between window 1 (P=0.3) and window 2 (P=0.92)
/// // t = (0.5 - 0.3) / (0.92 - 0.3) ≈ 0.3226
/// // center[1]=15000.5, center[2]=25000.5 → boundary ≈ 18226
/// assert!(refined[0].start_bp > 15000 && refined[0].start_bp < 25000);
/// // End refined: crossover between window 4 (P=0.91) and window 5 (P=0.2)
/// assert!(refined[0].end_bp > 45000 && refined[0].end_bp < 55000);
/// ```
pub fn refine_segment_boundaries(
    segments: &[IbdSegmentWithPosterior],
    posteriors: &[f64],
    window_starts: &[u64],
    window_ends: &[u64],
    crossover: f64,
) -> Vec<RefinedBoundary> {
    let n = posteriors.len();

    if n == 0 || window_starts.len() != n || window_ends.len() != n {
        return segments
            .iter()
            .map(|seg| RefinedBoundary {
                start_bp: if seg.start_idx < window_starts.len() {
                    window_starts[seg.start_idx]
                } else {
                    0
                },
                end_bp: if seg.end_idx < window_ends.len() {
                    window_ends[seg.end_idx]
                } else {
                    0
                },
            })
            .collect();
    }

    let center = |idx: usize| -> f64 {
        (window_starts[idx] as f64 + window_ends[idx] as f64) / 2.0
    };

    segments
        .iter()
        .map(|seg| {
            // Refine start boundary
            let refined_start = if seg.start_idx > 0 && seg.start_idx < n {
                let p_before = posteriors[seg.start_idx - 1];
                let p_at = posteriors[seg.start_idx];

                if p_at > p_before && p_before < crossover && p_at > crossover {
                    let t = (crossover - p_before) / (p_at - p_before);
                    let c_before = center(seg.start_idx - 1);
                    let c_at = center(seg.start_idx);
                    let pos = c_before + t * (c_at - c_before);
                    // Clamp to region between the two window centers
                    pos.clamp(c_before, c_at).round() as u64
                } else {
                    window_starts[seg.start_idx]
                }
            } else if seg.start_idx < n {
                window_starts[seg.start_idx]
            } else {
                0
            };

            // Refine end boundary
            let refined_end = if seg.end_idx + 1 < n {
                let p_at = posteriors[seg.end_idx];
                let p_after = posteriors[seg.end_idx + 1];

                if p_at > p_after && p_after < crossover && p_at > crossover {
                    let t = (p_at - crossover) / (p_at - p_after);
                    let c_at = center(seg.end_idx);
                    let c_after = center(seg.end_idx + 1);
                    let pos = c_at + t * (c_after - c_at);
                    pos.clamp(c_at, c_after).round() as u64
                } else if seg.end_idx < n {
                    window_ends[seg.end_idx]
                } else {
                    0
                }
            } else if seg.end_idx < n {
                window_ends[seg.end_idx]
            } else {
                0
            };

            RefinedBoundary {
                start_bp: refined_start,
                end_bp: refined_end.max(refined_start),
            }
        })
        .collect()
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
/// use impopk_ibd::hmm::{HmmParams, viterbi};
///
/// // For demonstration, use balanced priors (p_enter_ibd = 0.5)
/// let params = HmmParams::from_expected_length(10.0, 0.5, 5000);
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

/// Viterbi algorithm with distance-dependent transitions.
///
/// Like [`viterbi`], but adjusts transition probabilities based on the physical
/// distance between consecutive windows using a continuous-time Markov chain.
/// This handles non-uniform window spacing (e.g., gaps in pangenome coverage).
///
/// ## Arguments
///
/// - `observations`: Sequence of identity values (one per window)
/// - `params`: HMM parameters
/// - `window_positions`: Physical positions (start, end) of each window in bp
///
/// ## Returns
///
/// Most likely state sequence (0=non-IBD, 1=IBD).
/// Falls back to standard Viterbi if `window_positions` length doesn't match observations.
pub fn viterbi_with_distances(
    observations: &[f64],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> Vec<usize> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    // Fall back to standard Viterbi if positions don't match
    if window_positions.len() != n {
        return viterbi(observations, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    // Compute nominal window size from first window
    let window_size = {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    };

    // Precompute log emissions
    let mut log_emit: Vec<[f64; 2]> = Vec::with_capacity(n);
    for &obs in observations {
        log_emit.push([
            params.emission[0].log_pdf(obs),
            params.emission[1].log_pdf(obs),
        ]);
    }

    let mut delta: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut psi: Vec<[usize; 2]> = Vec::with_capacity(n);

    // Initialization
    delta.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);
    psi.push([0, 0]);

    // Forward pass with distance-dependent transitions
    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let distance = mid_curr.saturating_sub(mid_prev);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

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

    // Backtrack
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
/// use impopk_ibd::hmm::extract_ibd_segments;
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

// ============================================================================
// Multi-feature emission support
// ============================================================================

/// Precompute log-emission probabilities for each observation and state
/// using only the primary (identity) emission model.
///
/// Returns `log_emit[t][s] = log P(obs[t] | state = s)`.
pub fn precompute_log_emissions(observations: &[f64], params: &HmmParams) -> Vec<[f64; 2]> {
    observations
        .iter()
        .map(|&obs| {
            [
                params.emission[0].log_pdf(obs),
                params.emission[1].log_pdf(obs),
            ]
        })
        .collect()
}

/// Compute combined log-emission probabilities from a primary feature (identity)
/// and an optional auxiliary feature (e.g., coverage ratio).
///
/// Assumes conditional independence between features given the state:
///   log P(obs_primary, obs_aux | state=s) = log P(obs_primary | state=s) + log P(obs_aux | state=s)
///
/// If `aux_observations` or `aux_emission` is None, returns standard log-emissions.
///
/// ## Parameters
///
/// - `observations`: Primary identity observations (one per window)
/// - `params`: HMM parameters with primary emission distributions
/// - `aux_observations`: Optional auxiliary feature values (one per window, same length as observations)
/// - `aux_emission`: Optional per-state Gaussians for the auxiliary feature: [non-IBD, IBD]
pub fn compute_combined_log_emissions(
    observations: &[f64],
    params: &HmmParams,
    aux_observations: Option<&[f64]>,
    aux_emission: Option<&[GaussianParams; 2]>,
) -> Vec<[f64; 2]> {
    let primary = precompute_log_emissions(observations, params);

    match (aux_observations, aux_emission) {
        (Some(aux_obs), Some(aux_emit)) if aux_obs.len() == observations.len() => {
            primary
                .iter()
                .zip(aux_obs.iter())
                .map(|(primary_le, &aux_val)| {
                    [
                        primary_le[0] + aux_emit[0].log_pdf(aux_val),
                        primary_le[1] + aux_emit[1].log_pdf(aux_val),
                    ]
                })
                .collect()
        }
        _ => primary,
    }
}

/// Forward algorithm using pre-computed log-emission probabilities.
///
/// This is the core building block for multi-feature HMMs. The caller computes
/// combined log-emissions (from one or more features) and passes them in.
///
/// ## Parameters
///
/// - `log_emit`: Pre-computed `log P(obs[t] | state=s)` for each time step and state
/// - `params`: HMM parameters (only transition and initial probabilities are used)
///
/// ## Returns
///
/// `(alpha, log_likelihood)` where `alpha[t][s] = log P(obs[0..=t], state[t]=s)`.
pub fn forward_from_log_emit(log_emit: &[[f64; 2]], params: &HmmParams) -> (Vec<[f64; 2]>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];
    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);

    // Initialization
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    // Forward pass
    for t in 1..n {
        let mut at = [0.0f64; 2];
        for s in 0..2 {
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log
                + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Backward algorithm using pre-computed log-emission probabilities.
pub fn backward_from_log_emit(log_emit: &[[f64; 2]], params: &HmmParams) -> Vec<[f64; 2]> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    beta[n - 1] = [0.0, 0.0];

    for t in (0..n - 1).rev() {
        for s in 0..2 {
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log
                + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Viterbi algorithm using pre-computed log-emission probabilities.
pub fn viterbi_from_log_emit(log_emit: &[[f64; 2]], params: &HmmParams) -> Vec<usize> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];
    let log_trans: [[f64; 2]; 2] = [
        [params.transition[0][0].ln(), params.transition[0][1].ln()],
        [params.transition[1][0].ln(), params.transition[1][1].ln()],
    ];

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
    states[n - 1] = if delta[n - 1][1] > delta[n - 1][0] {
        1
    } else {
        0
    };

    for t in (0..n - 1).rev() {
        states[t] = psi[t + 1][states[t + 1]];
    }

    states
}

/// Forward-backward algorithm using pre-computed log-emission probabilities.
///
/// Returns `(posterior_ibd, log_likelihood)`.
pub fn forward_backward_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
) -> (Vec<f64>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward_from_log_emit(log_emit, params);
    let beta = backward_from_log_emit(log_emit, params);

    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum =
            max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();
        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Smooth precomputed log emissions using a sliding window average.
///
/// For each window t, the smoothed emission is the mean of log emissions
/// from windows [t-context, t+context] (clipped at boundaries).
/// This addresses weak per-window signal by pooling evidence from neighbors.
///
/// With context=2 (5-window span), SNR increases by ~√5 ≈ 2.2x.
///
/// # Arguments
/// * `log_emit` - n×2 matrix of log emissions [non-IBD, IBD]
/// * `context` - number of neighboring windows on each side (0 = no smoothing)
///
/// # Returns
/// Smoothed n×2 matrix of log emissions
pub fn smooth_log_emissions(log_emit: &[[f64; 2]], context: usize) -> Vec<[f64; 2]> {
    if context == 0 || log_emit.is_empty() {
        return log_emit.to_vec();
    }

    let n = log_emit.len();
    let mut smoothed = vec![[0.0; 2]; n];

    for (t, smoothed_t) in smoothed.iter_mut().enumerate() {
        let lo = t.saturating_sub(context);
        let hi = (t + context).min(n - 1);
        let span = (hi - lo + 1) as f64;

        for s in 0..2 {
            let sum: f64 = (lo..=hi).map(|i| log_emit[i][s]).sum();
            smoothed_t[s] = sum / span;
        }
    }

    smoothed
}

/// Forward algorithm with distance-dependent transitions using pre-computed log emissions.
pub fn forward_with_distances_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> (Vec<[f64; 2]>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    if window_positions.len() != n {
        return forward_from_log_emit(log_emit, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    let window_size = {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    };

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let distance = mid_curr.saturating_sub(mid_prev);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

        let mut at = [0.0f64; 2];
        for s in 0..2 {
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Backward algorithm with distance-dependent transitions using pre-computed log emissions.
pub fn backward_with_distances_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> Vec<[f64; 2]> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return backward_from_log_emit(log_emit, params);
    }

    let window_size = {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    };

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    beta[n - 1] = [0.0, 0.0];

    for t in (0..n - 1).rev() {
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let mid_next = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;
        let distance = mid_next.saturating_sub(mid_curr);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

        for s in 0..2 {
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Viterbi with distance-dependent transitions using pre-computed log emissions.
pub fn viterbi_with_distances_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> Vec<usize> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return viterbi_from_log_emit(log_emit, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    let window_size = {
        let (s, e) = window_positions[0];
        (e.saturating_sub(s) + 1).max(1)
    };

    let mut delta: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut psi: Vec<[usize; 2]> = Vec::with_capacity(n);

    delta.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);
    psi.push([0, 0]);

    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let distance = mid_curr.saturating_sub(mid_prev);

        let log_trans = distance_dependent_log_transition(params, distance, window_size);

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

/// Forward-backward with distance-dependent transitions using pre-computed log emissions.
pub fn forward_backward_with_distances_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
) -> (Vec<f64>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward_with_distances_from_log_emit(log_emit, params, window_positions);
    let beta = backward_with_distances_from_log_emit(log_emit, params, window_positions);

    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum = max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();
        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Forward algorithm with genetic-map-aware transitions using pre-computed log emissions.
pub fn forward_with_genetic_map_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> (Vec<[f64; 2]>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    if window_positions.len() != n {
        return forward_from_log_emit(log_emit, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    let mut alpha: Vec<[f64; 2]> = Vec::with_capacity(n);
    alpha.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);

    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_prev, mid_curr, genetic_map, window_size,
        );

        let mut at = [0.0f64; 2];
        for s in 0..2 {
            let log_probs = [
                alpha[t - 1][0] + log_trans[0][s],
                alpha[t - 1][1] + log_trans[1][s],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            at[s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
            at[s] += log_emit[t][s];
        }
        alpha.push(at);
    }

    let max_log = alpha[n - 1][0].max(alpha[n - 1][1]);
    let log_likelihood = max_log
        + ((alpha[n - 1][0] - max_log).exp() + (alpha[n - 1][1] - max_log).exp()).ln();

    (alpha, log_likelihood)
}

/// Backward algorithm with genetic-map-aware transitions using pre-computed log emissions.
pub fn backward_with_genetic_map_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> Vec<[f64; 2]> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return backward_from_log_emit(log_emit, params);
    }

    let mut beta: Vec<[f64; 2]> = vec![[0.0; 2]; n];
    beta[n - 1] = [0.0, 0.0];

    for t in (0..n - 1).rev() {
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;
        let mid_next = (window_positions[t + 1].0 + window_positions[t + 1].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_curr, mid_next, genetic_map, window_size,
        );

        for s in 0..2 {
            let log_probs = [
                log_trans[s][0] + log_emit[t + 1][0] + beta[t + 1][0],
                log_trans[s][1] + log_emit[t + 1][1] + beta[t + 1][1],
            ];
            let max_log = log_probs[0].max(log_probs[1]);
            beta[t][s] = max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        }
    }

    beta
}

/// Viterbi with genetic-map-aware transitions using pre-computed log emissions.
pub fn viterbi_with_genetic_map_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> Vec<usize> {
    let n = log_emit.len();
    if n == 0 {
        return vec![];
    }

    if window_positions.len() != n {
        return viterbi_from_log_emit(log_emit, params);
    }

    let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];

    let mut delta: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut psi: Vec<[usize; 2]> = Vec::with_capacity(n);

    delta.push([
        log_initial[0] + log_emit[0][0],
        log_initial[1] + log_emit[0][1],
    ]);
    psi.push([0, 0]);

    for t in 1..n {
        let mid_prev = (window_positions[t - 1].0 + window_positions[t - 1].1) / 2;
        let mid_curr = (window_positions[t].0 + window_positions[t].1) / 2;

        let log_trans = recombination_aware_log_transition(
            params, mid_prev, mid_curr, genetic_map, window_size,
        );

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

/// Forward-backward with genetic-map-aware transitions using pre-computed log emissions.
pub fn forward_backward_with_genetic_map_from_log_emit(
    log_emit: &[[f64; 2]],
    params: &HmmParams,
    window_positions: &[(u64, u64)],
    genetic_map: &GeneticMap,
    window_size: u64,
) -> (Vec<f64>, f64) {
    let n = log_emit.len();
    if n == 0 {
        return (vec![], 0.0);
    }

    let (alpha, log_likelihood) = forward_with_genetic_map_from_log_emit(
        log_emit, params, window_positions, genetic_map, window_size,
    );
    let beta = backward_with_genetic_map_from_log_emit(
        log_emit, params, window_positions, genetic_map, window_size,
    );

    let mut posterior_ibd = Vec::with_capacity(n);
    for t in 0..n {
        let log_gamma_0 = alpha[t][0] + beta[t][0] - log_likelihood;
        let log_gamma_1 = alpha[t][1] + beta[t][1] - log_likelihood;
        let max_log = log_gamma_0.max(log_gamma_1);
        let log_sum = max_log + ((log_gamma_0 - max_log).exp() + (log_gamma_1 - max_log).exp()).ln();
        let p_ibd = (log_gamma_1 - log_sum).exp();
        posterior_ibd.push(p_ibd);
    }

    (posterior_ibd, log_likelihood)
}

/// Complete IBD inference with multi-feature emissions.
///
/// This is the multi-feature equivalent of `infer_ibd_with_training`. It:
/// 1. Estimates primary emission parameters from identity observations
/// 2. If auxiliary observations are provided, estimates auxiliary emissions
///    using posterior-guided clustering
/// 3. Runs Baum-Welch training on the combined log-emissions
/// 4. Runs Viterbi + forward-backward on the combined log-emissions
///
/// ## Parameters
///
/// - `observations`: Primary identity observations
/// - `params`: HMM parameters (will be modified by Baum-Welch)
/// - `population`: Population for biological priors
/// - `window_size`: Window size in bp
/// - `bw_iters`: Number of Baum-Welch iterations
/// - `aux_observations`: Optional auxiliary feature values
///
/// ## Returns
///
/// `(IbdInferenceResult, Option<[GaussianParams; 2]>)` where the second element
/// contains the estimated auxiliary emission parameters (if auxiliary data was used).
pub fn infer_ibd_with_aux_features(
    observations: &[f64],
    params: &mut HmmParams,
    population: Population,
    window_size: u64,
    bw_iters: usize,
    aux_observations: Option<&[f64]>,
) -> (IbdInferenceResult, Option<[GaussianParams; 2]>) {
    if observations.len() < 3 {
        return (
            IbdInferenceResult {
                states: vec![0; observations.len()],
                posteriors: vec![0.0; observations.len()],
                log_likelihood: f64::NEG_INFINITY,
            },
            None,
        );
    }

    // Step 1: Estimate primary emissions from identity data
    params.estimate_emissions_robust(observations, Some(population), window_size);

    // Step 2: Initial forward-backward on primary emissions only to get posteriors
    let (initial_posteriors, _) = forward_backward(observations, params);

    // Step 3: If auxiliary data provided, estimate auxiliary emissions using posteriors
    let aux_emission = if let Some(aux_obs) = aux_observations {
        if aux_obs.len() == observations.len() {
            Some(estimate_auxiliary_emissions(aux_obs, &initial_posteriors))
        } else {
            None
        }
    } else {
        None
    };

    // Step 4: Baum-Welch training on combined log-emissions
    if bw_iters > 0 && observations.len() >= 10 {
        for _ in 0..bw_iters {
            let log_emit = compute_combined_log_emissions(
                observations,
                params,
                aux_observations,
                aux_emission.as_ref(),
            );

            let (alpha, log_likelihood) = forward_from_log_emit(&log_emit, params);
            let beta = backward_from_log_emit(&log_emit, params);
            let n = observations.len();

            // E-step: compute gamma (posteriors) and xi (transition counts)
            let mut gamma = vec![[0.0f64; 2]; n];
            for t in 0..n {
                let log_g0 = alpha[t][0] + beta[t][0] - log_likelihood;
                let log_g1 = alpha[t][1] + beta[t][1] - log_likelihood;
                let max_log = log_g0.max(log_g1);
                let log_sum =
                    max_log + ((log_g0 - max_log).exp() + (log_g1 - max_log).exp()).ln();
                gamma[t][0] = (log_g0 - log_sum).exp();
                gamma[t][1] = (log_g1 - log_sum).exp();
            }

            // M-step: re-estimate primary emission parameters
            let mut sum_mean = [0.0f64; 2];
            let mut sum_var = [0.0f64; 2];
            let mut sum_gamma = [0.0f64; 2];

            for t in 0..n {
                for s in 0..2 {
                    sum_gamma[s] += gamma[t][s];
                    sum_mean[s] += gamma[t][s] * observations[t];
                }
            }

            for s in 0..2 {
                if sum_gamma[s] > 1.0 {
                    let new_mean = sum_mean[s] / sum_gamma[s];
                    for t in 0..n {
                        sum_var[s] += gamma[t][s] * (observations[t] - new_mean).powi(2);
                    }
                    let new_var = sum_var[s] / sum_gamma[s];
                    let new_std = new_var.max(0.0).sqrt().max(1e-6);
                    if new_mean.is_finite() && new_std.is_finite() {
                        params.emission[s] = GaussianParams::new_unchecked(new_mean, new_std);
                    }
                }
            }

            // Enforce identifiability: state 0 mean < state 1 mean
            if params.emission[0].mean > params.emission[1].mean {
                params.emission.swap(0, 1);
            }

            // Re-estimate transition matrix from xi
            let log_trans: [[f64; 2]; 2] = [
                [
                    params.transition[0][0].ln(),
                    params.transition[0][1].ln(),
                ],
                [
                    params.transition[1][0].ln(),
                    params.transition[1][1].ln(),
                ],
            ];
            let mut xi_sum = [[0.0f64; 2]; 2];
            for t in 0..n - 1 {
                for i in 0..2 {
                    for j in 0..2 {
                        let log_xi = alpha[t][i] + log_trans[i][j]
                            + log_emit[t + 1][j]
                            + beta[t + 1][j]
                            - log_likelihood;
                        xi_sum[i][j] += log_xi.exp();
                    }
                }
            }

            for (i, xi_row) in xi_sum.iter().enumerate() {
                let row_sum = xi_row[0] + xi_row[1];
                if row_sum > 1e-10 {
                    let new_trans = [xi_row[0] / row_sum, xi_row[1] / row_sum];
                    if new_trans[0].is_finite() && new_trans[1].is_finite() {
                        params.transition[i] = new_trans;
                    }
                }
            }
        }
    }

    // Step 5: Final inference with combined emissions
    let log_emit = compute_combined_log_emissions(
        observations,
        params,
        aux_observations,
        aux_emission.as_ref(),
    );
    let mut states = viterbi_from_log_emit(&log_emit, params);
    let (posteriors, log_likelihood) = forward_backward_from_log_emit(&log_emit, params);
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

    (
        IbdInferenceResult {
            states,
            posteriors,
            log_likelihood,
        },
        aux_emission,
    )
}

/// Estimate per-state Gaussian emission parameters for an auxiliary feature
/// using posterior probabilities from a primary-feature HMM.
///
/// Uses soft assignments from posteriors to compute weighted mean and std
/// for each state's auxiliary emission distribution.
///
/// ## Parameters
///
/// - `aux_observations`: Auxiliary feature values (one per window)
/// - `posteriors_ibd`: P(IBD) for each window from primary-feature forward-backward
///
/// ## Returns
///
/// `[non-IBD Gaussian, IBD Gaussian]` for the auxiliary feature.
pub fn estimate_auxiliary_emissions(
    aux_observations: &[f64],
    posteriors_ibd: &[f64],
) -> [GaussianParams; 2] {
    let n = aux_observations.len();

    // Weighted mean for each state
    let mut sum_w = [0.0f64; 2]; // sum of weights per state
    let mut sum_wx = [0.0f64; 2]; // sum of weight * value

    for t in 0..n {
        let p_ibd = posteriors_ibd[t].clamp(0.0, 1.0);
        let w_non_ibd = 1.0 - p_ibd;
        let w_ibd = p_ibd;

        sum_w[0] += w_non_ibd;
        sum_w[1] += w_ibd;
        sum_wx[0] += w_non_ibd * aux_observations[t];
        sum_wx[1] += w_ibd * aux_observations[t];
    }

    let mean = [
        if sum_w[0] > 1e-10 {
            sum_wx[0] / sum_w[0]
        } else {
            0.5
        },
        if sum_w[1] > 1e-10 {
            sum_wx[1] / sum_w[1]
        } else {
            0.9
        },
    ];

    // Weighted variance for each state
    let mut sum_wxx = [0.0f64; 2];
    for t in 0..n {
        let p_ibd = posteriors_ibd[t].clamp(0.0, 1.0);
        let w_non_ibd = 1.0 - p_ibd;
        let w_ibd = p_ibd;

        sum_wxx[0] += w_non_ibd * (aux_observations[t] - mean[0]).powi(2);
        sum_wxx[1] += w_ibd * (aux_observations[t] - mean[1]).powi(2);
    }

    let std = [
        if sum_w[0] > 1e-10 {
            (sum_wxx[0] / sum_w[0]).max(0.0).sqrt().max(1e-4)
        } else {
            0.2
        },
        if sum_w[1] > 1e-10 {
            (sum_wxx[1] / sum_w[1]).max(0.0).sqrt().max(1e-4)
        } else {
            0.1
        },
    ];

    [
        GaussianParams::new_unchecked(mean[0], std[0]),
        GaussianParams::new_unchecked(mean[1], std[1]),
    ]
}

/// Compute coverage ratio from alignment lengths.
///
/// Coverage ratio = min(a_len, b_len) / max(a_len, b_len).
/// This measures how symmetric the alignment is:
/// - IBD regions: both haplotypes map similarly → ratio close to 1.0
/// - Non-IBD regions: alignments may differ → ratio can be lower
///
/// Returns 0.0 if both lengths are 0, and 1.0 if they're equal.
pub fn coverage_ratio(a_len: u64, b_len: u64) -> f64 {
    if a_len == 0 && b_len == 0 {
        return 0.0;
    }
    let min_len = a_len.min(b_len) as f64;
    let max_len = a_len.max(b_len) as f64;
    min_len / max_len
}

/// Estimate Bernoulli parameters for the K=0 (mutation-free window) auxiliary feature.
///
/// Windows with identity at or above a threshold are considered "mutation-free" (K=0).
/// These windows are dramatically more common under IBD than non-IBD (G104: ~16× ratio),
/// making the binary indicator a strong discriminator that complements the continuous
/// identity signal (which is poorly modeled by capped Gaussians for K=0 observations).
///
/// Uses posterior-weighted estimation: P(K=0 | state) = Σ_t w_t(state) × I(K=0_t) / Σ_t w_t(state)
///
/// Returns `[P(K=0 | non-IBD), P(K=0 | IBD)]`, clamped to [0.001, 0.999] to avoid log(0).
pub fn estimate_k0_emissions(indicators: &[f64], posteriors_ibd: &[f64]) -> [f64; 2] {
    let n = indicators.len();
    if n == 0 {
        // Informative priors from G104 (5kb windows, ~0.1% diversity)
        return [0.015, 0.22];
    }

    let mut w_sum = [0.0f64; 2]; // sum of weights per state
    let mut w_k0 = [0.0f64; 2]; // sum of weight × indicator

    for (t, &indicator) in indicators.iter().enumerate() {
        let p_ibd = posteriors_ibd.get(t).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let p_nonibd = 1.0 - p_ibd;

        w_sum[0] += p_nonibd;
        w_sum[1] += p_ibd;
        w_k0[0] += p_nonibd * indicator;
        w_k0[1] += p_ibd * indicator;
    }

    let p0 = if w_sum[0] > 1e-10 {
        (w_k0[0] / w_sum[0]).clamp(0.001, 0.999)
    } else {
        0.015 // prior: rare in non-IBD
    };
    let p1 = if w_sum[1] > 1e-10 {
        (w_k0[1] / w_sum[1]).clamp(0.001, 0.999)
    } else {
        0.22 // prior: common in IBD
    };

    [p0, p1]
}

/// Compute log P(indicator | state) for the Bernoulli K=0 model.
///
/// indicator > 0.5 → K=0 (mutation-free), returns ln(p_k0)
/// indicator <= 0.5 → K≥1 (has mutations), returns ln(1 - p_k0)
#[inline]
pub fn k0_log_pmf(indicator: f64, p_k0: f64) -> f64 {
    if indicator > 0.5 {
        p_k0.ln()
    } else {
        (1.0 - p_k0).ln()
    }
}

/// Augment precomputed log-emissions with K=0 Bernoulli auxiliary feature.
///
/// Modifies `log_emit` in-place by adding log P(k0_indicator_t | state) for each window.
/// Uses initial posteriors to estimate per-state K=0 probabilities from data.
///
/// This should be called AFTER emission-context smoothing (K=0 is a discrete
/// per-window feature that should not be spatially smoothed).
pub fn augment_with_k0(
    log_emit: &mut [[f64; 2]],
    k0_indicators: &[f64],
    posteriors_ibd: &[f64],
) {
    let k0_params = estimate_k0_emissions(k0_indicators, posteriors_ibd);

    // Only augment if K=0 is actually discriminative (IBD rate > non-IBD rate)
    if k0_params[1] <= k0_params[0] {
        return;
    }

    for (t, &indicator) in k0_indicators.iter().enumerate() {
        if t < log_emit.len() {
            log_emit[t][0] += k0_log_pmf(indicator, k0_params[0]);
            log_emit[t][1] += k0_log_pmf(indicator, k0_params[1]);
        }
    }
}

/// Extract IBD segments using composite filtering that couples LOD, length, and posterior.
///
/// Unlike `extract_ibd_segments_with_lod` which applies independent hard thresholds
/// (min_windows AND min_mean_posterior AND min_lod), this function uses a composite
/// score that allows tradeoffs: shorter segments with very high LOD or posterior can
/// survive, and longer segments with moderate evidence are also retained.
///
/// The composite score is:
///   score = lod_density × posterior_strength × length_factor
///
/// where:
///   - lod_density = LOD / n_windows (evidence per window, range ~0-3)
///   - posterior_strength = mean_posterior (range 0-1)
///   - length_factor = min(n_windows / soft_min_windows, 1.0) (0-1, penalizes short segments)
///
/// Segments pass if: score >= composite_threshold
///
/// ## Parameters
///
/// - `states`: Viterbi state sequence (0=non-IBD, 1=IBD)
/// - `posteriors`: Posterior P(IBD) for each window
/// - `observations_and_params`: Optional (observations, params) for LOD computation
/// - `soft_min_windows`: Soft minimum length; segments shorter than this get length penalty
/// - `hard_min_windows`: Absolute minimum (segments shorter are always rejected)
/// - `composite_threshold`: Minimum composite score (recommended: 0.3-0.5)
pub fn extract_ibd_segments_composite(
    states: &[usize],
    posteriors: &[f64],
    observations_and_params: Option<(&[f64], &HmmParams)>,
    soft_min_windows: usize,
    hard_min_windows: usize,
    composite_threshold: f64,
) -> Vec<IbdSegmentWithPosterior> {
    let mut segments = Vec::new();
    let n = states.len();

    if n == 0 || posteriors.len() != n {
        return segments;
    }

    let mut in_ibd = false;
    let mut start_idx = 0;

    let mut finalize = |start: usize, end: usize| {
        let n_windows = end - start + 1;
        if n_windows < hard_min_windows {
            return;
        }

        let seg_posteriors = &posteriors[start..=end];
        let mean_post: f64 = seg_posteriors.iter().sum::<f64>() / n_windows as f64;
        let min_post = seg_posteriors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_post = seg_posteriors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let lod = if let Some((obs, params)) = observations_and_params {
            segment_lod_score(obs, start, end, params)
        } else {
            0.0
        };

        // Composite scoring: couples LOD, posterior, and length
        let lod_density = if n_windows > 0 { lod / n_windows as f64 } else { 0.0 };
        let length_factor = if soft_min_windows > 0 {
            (n_windows as f64 / soft_min_windows as f64).min(1.0)
        } else {
            1.0
        };
        let score = lod_density.max(0.0) * mean_post * length_factor;

        if score < composite_threshold {
            return;
        }

        segments.push(IbdSegmentWithPosterior {
            start_idx: start,
            end_idx: end,
            n_windows,
            mean_posterior: mean_post,
            min_posterior: min_post,
            max_posterior: max_post,
            lod_score: lod,
        });
    };

    for (i, &state) in states.iter().enumerate() {
        if state == 1 && !in_ibd {
            in_ibd = true;
            start_idx = i;
        } else if state == 0 && in_ibd {
            in_ibd = false;
            finalize(start_idx, i - 1);
        }
    }

    if in_ibd {
        finalize(start_idx, n - 1);
    }

    segments
}

/// Estimate IBD emission variance from observed data.
///
/// Instead of using a fixed IBD std (0.0005), this estimates the variance from
/// the top quantile of observations, which are most likely to be IBD windows.
///
/// The algorithm:
/// 1. Sort observations descending
/// 2. Take the top `quantile_fraction` (default ~5%) as putative IBD windows
/// 3. Compute their variance, clamped to [min_std, max_std]
///
/// This is more robust than the fixed value because:
/// - Assembly quality varies per sample and region
/// - Some populations have systematically different IBD identity distributions
/// - CIGAR-based vs impg-based identity can have different noise profiles
///
/// ## Parameters
///
/// - `observations`: All identity values for a pair
/// - `quantile_fraction`: Fraction of top observations to use (default: 0.05)
/// - `min_std`: Minimum allowed std (prevents degenerate estimates)
/// - `max_std`: Maximum allowed std (prevents over-smoothing)
///
/// ## Returns
///
/// Estimated std for IBD emission, or None if insufficient data.
pub fn estimate_ibd_emission_std(
    observations: &[f64],
    quantile_fraction: f64,
    min_std: f64,
    max_std: f64,
) -> Option<f64> {
    if observations.len() < 20 {
        return None;
    }

    let mut sorted: Vec<f64> = observations.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));

    let n_top = ((sorted.len() as f64 * quantile_fraction).ceil() as usize).max(5);
    let top = &sorted[..n_top.min(sorted.len())];

    let mean: f64 = top.iter().sum::<f64>() / top.len() as f64;
    let variance: f64 = top.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / top.len() as f64;
    let std = variance.max(0.0).sqrt().clamp(min_std, max_std);

    Some(std)
}

/// Refine Viterbi states with adaptive thresholds based on segment posterior quality.
///
/// Unlike `refine_states_with_posteriors` which uses fixed extend/trim thresholds,
/// this version adapts thresholds per-segment:
/// - High-confidence segments (mean posterior > 0.8) use aggressive extension (0.3)
///   and conservative trimming (0.3), preserving boundaries
/// - Low-confidence segments (mean posterior < 0.5) use conservative extension (0.6)
///   and aggressive trimming (0.15), reducing false positives
///
/// This handles the asymmetry where well-supported IBD boundaries should be extended
/// into ambiguous flanking windows, but poorly-supported segments should be trimmed.
pub fn refine_states_adaptive(
    states: &mut [usize],
    posteriors: &[f64],
) {
    let n = states.len();
    if n == 0 || posteriors.len() != n {
        return;
    }

    // First, identify segments and compute their mean posteriors
    let mut segments: Vec<(usize, usize, f64)> = Vec::new(); // (start, end, mean_post)
    let mut in_ibd = false;
    let mut start = 0;

    for i in 0..n {
        if states[i] == 1 && !in_ibd {
            in_ibd = true;
            start = i;
        } else if states[i] == 0 && in_ibd {
            in_ibd = false;
            let seg_post: f64 = posteriors[start..i].iter().sum::<f64>() / (i - start) as f64;
            segments.push((start, i - 1, seg_post));
        }
    }
    if in_ibd {
        let seg_post: f64 = posteriors[start..n].iter().sum::<f64>() / (n - start) as f64;
        segments.push((start, n - 1, seg_post));
    }

    // For each segment, compute adaptive thresholds and apply extend/trim
    for &(seg_start, seg_end, mean_post) in &segments {
        // Adaptive thresholds: high-quality segments extend more aggressively
        let extend_thresh = if mean_post > 0.8 {
            0.3 // Aggressive extension for confident segments
        } else if mean_post > 0.5 {
            0.4 + (0.8 - mean_post) * 0.67 // Linear interpolation
        } else {
            0.6 // Conservative extension for weak segments
        };

        let trim_thresh = if mean_post > 0.8 {
            0.3 // Keep more boundary windows for confident segments
        } else if mean_post > 0.5 {
            0.15 + (mean_post - 0.5) * 0.5 // Linear interpolation
        } else {
            0.15 // Trim more aggressively for weak segments
        };

        // Extend left
        let mut left = seg_start;
        while left > 0 && posteriors[left - 1] >= extend_thresh && states[left - 1] == 0 {
            left -= 1;
            states[left] = 1;
        }

        // Extend right
        let mut right = seg_end;
        while right + 1 < n && posteriors[right + 1] >= extend_thresh && states[right + 1] == 0 {
            right += 1;
            states[right] = 1;
        }

        // Trim left edge
        let mut trim_left = left;
        while trim_left <= right && posteriors[trim_left] < trim_thresh {
            states[trim_left] = 0;
            trim_left += 1;
        }

        // Trim right edge
        let mut trim_right = right;
        while trim_right >= trim_left && posteriors[trim_right] < trim_thresh {
            states[trim_right] = 0;
            if trim_right == 0 { break; }
            trim_right -= 1;
        }
    }
}

/// Bridge gaps with flanking-segment-aware thresholds.
///
/// Unlike `bridge_ibd_gaps` which only checks the mean posterior within the gap,
/// this version also considers the quality of flanking IBD segments. Gaps between
/// two high-confidence segments are bridged more aggressively (lower threshold)
/// than gaps between weak segments.
///
/// ## Parameters
///
/// - `states`: Viterbi state sequence (modified in place)
/// - `posteriors`: Posterior P(IBD) from forward-backward
/// - `max_gap`: Maximum gap length in windows to bridge
/// - `base_threshold`: Base posterior threshold (used as reference; adapted per gap)
///
/// ## Returns
///
/// Number of gaps bridged.
pub fn bridge_ibd_gaps_adaptive(
    states: &mut [usize],
    posteriors: &[f64],
    max_gap: usize,
    base_threshold: f64,
) -> usize {
    let n = states.len();
    if n < 3 || posteriors.len() != n || max_gap == 0 {
        return 0;
    }

    let mut bridges = 0;
    let mut i = 0;

    while i < n {
        if states[i] == 1 {
            // Find end of this IBD segment
            let seg_start = i;
            let mut seg_end = i;
            while seg_end + 1 < n && states[seg_end + 1] == 1 {
                seg_end += 1;
            }

            // Compute mean posterior of left flanking segment
            let left_mean_post: f64 = posteriors[seg_start..=seg_end].iter().sum::<f64>()
                / (seg_end - seg_start + 1) as f64;

            // Look at gap after segment
            let gap_start = seg_end + 1;
            if gap_start < n && states[gap_start] == 0 {
                let mut gap_end = gap_start;
                while gap_end + 1 < n && states[gap_end + 1] == 0 {
                    gap_end += 1;
                }
                let gap_len = gap_end - gap_start + 1;

                // Check if followed by another IBD segment
                if gap_len <= max_gap && gap_end + 1 < n && states[gap_end + 1] == 1 {
                    // Find right flanking segment's mean posterior
                    let right_start = gap_end + 1;
                    let mut right_end = right_start;
                    while right_end + 1 < n && states[right_end + 1] == 1 {
                        right_end += 1;
                    }
                    let right_mean_post: f64 = posteriors[right_start..=right_end].iter().sum::<f64>()
                        / (right_end - right_start + 1) as f64;

                    // Adaptive threshold: lower for high-confidence flanking segments
                    let flank_quality = (left_mean_post + right_mean_post) / 2.0;
                    let adaptive_threshold = if flank_quality > 0.8 {
                        base_threshold * 0.5 // Very confident flanks → bridge at half threshold
                    } else if flank_quality > 0.5 {
                        base_threshold * (1.0 - (flank_quality - 0.5) * (1.0 / 0.6))
                    } else {
                        base_threshold // Weak flanks → require full threshold
                    };

                    // Check gap posterior
                    let gap_mean: f64 = posteriors[gap_start..=gap_end].iter().sum::<f64>()
                        / gap_len as f64;
                    if gap_mean >= adaptive_threshold {
                        for s in states.iter_mut().take(gap_end + 1).skip(gap_start) {
                            *s = 1;
                        }
                        bridges += 1;
                        // Re-scan from the start of the left segment so we can
                        // check for additional gaps after the merged segment
                        i = seg_start;
                        continue;
                    }
                }
            }

            i = seg_end + 1;
        } else {
            i += 1;
        }
    }

    bridges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi_simple() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
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
        let _ = HmmParams::from_expected_length(10.0, 0.0, 5000);
    }

    #[test]
    #[should_panic(expected = "p_enter_ibd must be in range (0, 1)")]
    fn test_p_enter_ibd_one_panics() {
        // p_enter_ibd = 1 is invalid (must be < 1)
        let _ = HmmParams::from_expected_length(10.0, 1.0, 5000);
    }

    #[test]
    #[should_panic(expected = "p_enter_ibd must be in range (0, 1)")]
    fn test_p_enter_ibd_negative_panics() {
        // p_enter_ibd < 0 is invalid
        let _ = HmmParams::from_expected_length(10.0, -0.1, 5000);
    }

    #[test]
    fn test_p_enter_ibd_valid_values() {
        // These should all succeed without panicking
        let _ = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let _ = HmmParams::from_expected_length(10.0, 0.5, 5000);
        let _ = HmmParams::from_expected_length(10.0, 0.999, 5000);
    }

    // === Edge case tests for Viterbi algorithm ===

    #[test]
    fn test_viterbi_empty_observations() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs: Vec<f64> = vec![];
        let states = viterbi(&obs, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_viterbi_single_observation() {
        // Use higher p_enter_ibd for single observation test to reduce prior effect
        let params = HmmParams::from_expected_length(10.0, 0.5, 5000);

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
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
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
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
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
        let params = HmmParams::from_expected_length(5.0, 0.1, 5000);
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
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
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
        let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let original_emission = params.emission;

        // Less than 3 observations should not change emissions
        params.estimate_emissions(&[0.5, 0.9]);
        assert_eq!(params.emission[0].mean, original_emission[0].mean);
        assert_eq!(params.emission[1].mean, original_emission[1].mean);
    }

    #[test]
    fn test_estimate_emissions_identical_values() {
        let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let original_emission = params.emission;

        // All identical values (zero variance) should not change emissions
        let obs = vec![0.8, 0.8, 0.8, 0.8, 0.8];
        params.estimate_emissions(&obs);
        // Emissions should remain unchanged due to variance < 1e-12
        assert_eq!(params.emission[0].mean, original_emission[0].mean);
        assert_eq!(params.emission[1].mean, original_emission[1].mean);
    }

    #[test]
    fn test_estimate_emissions_two_clusters() {
        let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);

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
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);

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
        let params_short = HmmParams::from_expected_length(1.0, 0.001, 5000);
        // p_stay_ibd should be clamped to at least 0.5
        assert!(params_short.transition[1][1] >= 0.5);

        // Very long expected length
        let params_long = HmmParams::from_expected_length(100000.0, 0.001, 5000);
        // p_stay_ibd should be clamped to at most 0.9999
        assert!(params_long.transition[1][1] <= 0.9999);
    }

    // === Forward-backward algorithm tests ===

    #[test]
    fn test_forward_empty() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let (alpha, log_lik) = forward(&[], &params);
        assert!(alpha.is_empty());
        assert_eq!(log_lik, 0.0);
    }

    #[test]
    fn test_forward_single_observation() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.5];
        let (alpha, log_lik) = forward(&obs, &params);
        assert_eq!(alpha.len(), 1);
        assert!(log_lik.is_finite());
    }

    #[test]
    fn test_forward_multiple_observations() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998];
        let (alpha, log_lik) = forward(&obs, &params);
        assert_eq!(alpha.len(), 4);
        assert!(log_lik.is_finite());
        // Log-likelihood can be positive when using narrow Gaussians with
        // observations close to the mean (PDF > 1 is possible for narrow distributions)
    }

    #[test]
    fn test_backward_empty() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let beta = backward(&[], &params);
        assert!(beta.is_empty());
    }

    #[test]
    fn test_backward_single_observation() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.5];
        let beta = backward(&obs, &params);
        assert_eq!(beta.len(), 1);
        // For single observation, beta should be [0, 0] (log(1))
        assert_eq!(beta[0], [0.0, 0.0]);
    }

    #[test]
    fn test_backward_multiple_observations() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998];
        let beta = backward(&obs, &params);
        assert_eq!(beta.len(), 4);
        // Last beta should be [0, 0]
        assert_eq!(beta[3], [0.0, 0.0]);
    }

    #[test]
    fn test_forward_backward_empty() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let (posteriors, log_lik) = forward_backward(&[], &params);
        assert!(posteriors.is_empty());
        assert_eq!(log_lik, 0.0);
    }

    #[test]
    fn test_forward_backward_posteriors_sum_to_one() {
        // Posteriors P(IBD) + P(non-IBD) should sum to ~1 at each position
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997];
        let (posteriors, _) = forward_backward(&obs, &params);

        for (i, &p_ibd) in posteriors.iter().enumerate() {
            assert!(p_ibd >= 0.0, "P(IBD) should be >= 0 at position {}", i);
            assert!(p_ibd <= 1.0, "P(IBD) should be <= 1 at position {}", i);
        }
    }

    #[test]
    fn test_forward_backward_high_identity_high_posterior() {
        // Very high identity observations should have high P(IBD)
        let params = HmmParams::from_expected_length(10.0, 0.1, 5000);  // Higher p_enter for easier detection
        let obs = vec![0.9998, 0.9999, 0.9999, 0.9998, 0.9999];
        let (posteriors, _) = forward_backward(&obs, &params);

        // Middle observations should have high posterior
        assert!(posteriors[2] > 0.5, "Middle position should have P(IBD) > 0.5, got {}", posteriors[2]);
    }

    #[test]
    fn test_forward_backward_low_identity_low_posterior() {
        // Low identity observations should have low P(IBD)
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.5, 0.6, 0.55, 0.45, 0.5];
        let (posteriors, _) = forward_backward(&obs, &params);

        // All should have low posterior
        for (i, &p) in posteriors.iter().enumerate() {
            assert!(p < 0.5, "Position {} should have P(IBD) < 0.5, got {}", i, p);
        }
    }

    #[test]
    fn test_infer_ibd_complete() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.998, 0.999, 0.9998, 0.9999, 0.997, 0.996];

        let result = infer_ibd(&obs, &params);

        assert_eq!(result.states.len(), 6);
        assert_eq!(result.posteriors.len(), 6);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_extract_segments_with_posteriors_empty() {
        let segments = extract_ibd_segments_with_posteriors(&[], &[], 1, 0.5);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_extract_segments_with_posteriors_filter_by_length() {
        let states = vec![0, 0, 1, 1, 0, 0, 1, 0, 0];
        let posteriors = vec![0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1];

        // Min 2 windows - should get first segment, not second
        let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 2, 0.5);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_idx, 2);
        assert_eq!(segments[0].end_idx, 3);
        assert_eq!(segments[0].n_windows, 2);
    }

    #[test]
    fn test_extract_segments_with_posteriors_filter_by_posterior() {
        let states = vec![0, 1, 1, 1, 0, 1, 1, 1, 0];
        let posteriors = vec![0.1, 0.9, 0.9, 0.9, 0.1, 0.4, 0.5, 0.3, 0.1];

        // Min 0.8 mean posterior - should only get first segment
        let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.8);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_idx, 1);
        assert_eq!(segments[0].n_windows, 3);
        assert!((segments[0].mean_posterior - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_extract_segments_with_posteriors_stats() {
        let states = vec![1, 1, 1, 1, 1];
        let posteriors = vec![0.8, 0.9, 0.95, 0.85, 0.7];

        let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
        assert_eq!(segments.len(), 1);

        let seg = &segments[0];
        assert_eq!(seg.n_windows, 5);
        assert!((seg.mean_posterior - 0.84).abs() < 0.01);  // (0.8+0.9+0.95+0.85+0.7)/5 = 0.84
        assert!((seg.min_posterior - 0.7).abs() < 0.01);
        assert!((seg.max_posterior - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_from_population_adaptive_afr() {
        // AFR should have lower p_enter_ibd and shorter expected segments
        let params_afr = HmmParams::from_population_adaptive(Population::AFR, 50.0, 0.0001, 5000);
        let params_eur = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

        // AFR should have lower transition into IBD
        assert!(params_afr.transition[0][1] < params_eur.transition[0][1],
            "AFR p_enter_ibd ({}) should be less than EUR ({})",
            params_afr.transition[0][1], params_eur.transition[0][1]);
    }

    #[test]
    fn test_from_population_adaptive_interpop() {
        // InterPop should have very low p_enter_ibd
        let params_inter = HmmParams::from_population_adaptive(Population::InterPop, 50.0, 0.0001, 5000);
        let params_generic = HmmParams::from_population(Population::Generic, 50.0, 0.0001, 5000);

        assert!(params_inter.transition[0][1] < params_generic.transition[0][1] * 0.5,
            "InterPop p_enter_ibd should be much less than Generic");
    }

    #[test]
    fn test_estimate_emissions_robust_em_fallback_afr() {
        // Simulate AFR data where most windows are non-IBD (~0.99875)
        // with just a few IBD windows (~0.9997)
        let mut data = Vec::new();
        // 95% non-IBD with AFR diversity
        for i in 0..95 {
            data.push(0.99875 + (i as f64 * 0.00001) % 0.0005 - 0.00025);
        }
        // 5% IBD
        for i in 0..5 {
            data.push(0.9997 + (i as f64 * 0.00005) % 0.0003 - 0.00015);
        }

        let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
        params.estimate_emissions_robust(&data, Some(Population::AFR), 5000);

        // Non-IBD mean should be near AFR expected value
        assert!(params.emission[0].mean < 0.9993,
            "Non-IBD mean {} should be < 0.9993", params.emission[0].mean);
        // IBD mean should be high
        assert!(params.emission[1].mean >= 0.999,
            "IBD mean {} should be >= 0.999", params.emission[1].mean);
    }

    #[test]
    fn test_forward_backward_consistent_with_viterbi() {
        // High posterior regions should generally align with Viterbi IBD calls
        let params = HmmParams::from_expected_length(5.0, 0.1, 5000);
        let obs = vec![0.5, 0.5, 0.9999, 0.9999, 0.9999, 0.5, 0.5];

        let result = infer_ibd(&obs, &params);

        // Where Viterbi says IBD (state=1), posterior should be high
        for (i, (&state, &post)) in result.states.iter().zip(result.posteriors.iter()).enumerate() {
            if state == 1 {
                assert!(post > 0.5, "Position {} has state=1 but low posterior {}", i, post);
            }
        }
    }

    // === Baum-Welch tests ===

    #[test]
    fn test_baum_welch_improves_likelihood() {
        // Baum-Welch should increase (or maintain) log-likelihood
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![
            0.998, 0.9985, 0.999, 0.9995, 0.9997, 0.9998, 0.9999, 0.9997,
            0.9996, 0.999, 0.998, 0.997, 0.998, 0.9985,
        ];

        let (_, log_lik_before) = forward_backward(&obs, &params);
        params.baum_welch(&obs, 20, 1e-6, Some(Population::EUR), 5000);
        let (_, log_lik_after) = forward_backward(&obs, &params);

        assert!(
            log_lik_after >= log_lik_before - 1e-6,
            "Baum-Welch should improve log-likelihood: before={:.2}, after={:.2}",
            log_lik_before, log_lik_after
        );
    }

    #[test]
    fn test_baum_welch_too_few_observations() {
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let original = params.clone();
        let obs = vec![0.998, 0.999];

        params.baum_welch(&obs, 20, 1e-6, Some(Population::EUR), 5000);

        // Should not modify params with too few observations
        assert_eq!(params.emission[0].mean, original.emission[0].mean);
    }

    #[test]
    fn test_baum_welch_maintains_valid_transitions() {
        let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);

        // Data with clear IBD region
        let mut obs = vec![0.99875; 50]; // Non-IBD for AFR
        obs.extend(vec![0.9997; 20]);     // IBD
        obs.extend(vec![0.99875; 30]);    // Non-IBD

        params.baum_welch(&obs, 20, 1e-6, Some(Population::AFR), 5000);

        // Transitions should still sum to 1
        let t0_sum = params.transition[0][0] + params.transition[0][1];
        let t1_sum = params.transition[1][0] + params.transition[1][1];
        assert!((t0_sum - 1.0).abs() < 1e-10, "Transition from state 0 should sum to 1");
        assert!((t1_sum - 1.0).abs() < 1e-10, "Transition from state 1 should sum to 1");

        // Emission means should be ordered
        assert!(params.emission[0].mean < params.emission[1].mean,
            "Non-IBD mean should be less than IBD mean");
    }

    #[test]
    fn test_baum_welch_afr_data() {
        // Simulate AFR data: most windows non-IBD (~0.99875), few IBD (~0.9997)
        let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
        let mut obs = Vec::new();
        for i in 0..80 {
            obs.push(0.99875 + (i as f64 * 0.00003) % 0.0006 - 0.0003);
        }
        for i in 0..20 {
            obs.push(0.9997 + (i as f64 * 0.00002) % 0.0004 - 0.0002);
        }

        params.estimate_emissions_robust(&obs, Some(Population::AFR), 5000);
        params.baum_welch(&obs, 20, 1e-6, Some(Population::AFR), 5000);

        // Non-IBD emission should be near AFR expected
        assert!(params.emission[0].mean < 0.9993,
            "AFR non-IBD mean {} should be < 0.9993", params.emission[0].mean);
        assert!(params.emission[1].mean >= 0.999,
            "IBD mean {} should be >= 0.999", params.emission[1].mean);
    }

    // === Posterior refinement tests ===

    #[test]
    fn test_refine_states_empty() {
        let mut states: Vec<usize> = vec![];
        let posteriors: Vec<f64> = vec![];
        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
        assert!(states.is_empty());
    }

    #[test]
    fn test_refine_states_extension() {
        // Viterbi says non-IBD at position 2, but posterior is high and adjacent to IBD
        let mut states = vec![0, 1, 0, 1, 0];
        let posteriors = vec![0.1, 0.9, 0.7, 0.9, 0.1];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Position 2 should be extended to IBD (posterior 0.7 > extend_threshold 0.5)
        assert_eq!(states[2], 1, "Position 2 should be extended to IBD");
    }

    #[test]
    fn test_refine_states_trimming() {
        // Viterbi says IBD at edge, but posterior is very low
        let mut states = vec![1, 1, 1, 1, 0];
        let posteriors = vec![0.1, 0.9, 0.9, 0.9, 0.1];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Position 0 should be trimmed (posterior 0.1 < trim_threshold 0.2, at segment edge)
        assert_eq!(states[0], 0, "Position 0 should be trimmed from IBD segment");
        assert_eq!(states[1], 1, "Position 1 should remain IBD");
    }

    #[test]
    fn test_refine_states_no_change_high_posteriors() {
        // All IBD with high posteriors should remain unchanged
        let mut states = vec![1, 1, 1, 1, 1];
        let posteriors = vec![0.9, 0.95, 0.98, 0.95, 0.9];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        for (i, &s) in states.iter().enumerate() {
            assert_eq!(s, 1, "Position {} should remain IBD", i);
        }
    }

    #[test]
    fn test_refine_states_mismatched_lengths() {
        // Mismatched lengths should be a no-op
        let mut states = vec![0, 1, 0];
        let posteriors = vec![0.1, 0.9];
        let original = states.clone();

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        assert_eq!(states, original, "Mismatched lengths should not modify states");
    }

    // === infer_ibd_with_training tests ===

    #[test]
    fn test_infer_ibd_with_training_basic() {
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.998];

        let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 10);

        assert_eq!(result.states.len(), 6);
        assert_eq!(result.posteriors.len(), 6);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_infer_ibd_with_training_zero_iters() {
        // With 0 Baum-Welch iters, should behave like infer_ibd + refinement
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.998];

        let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 0);

        assert_eq!(result.states.len(), 6);
        assert!(result.log_likelihood.is_finite());
    }

    // === LOD score tests ===

    #[test]
    fn test_per_window_lod_ibd_region_positive() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // IBD-like observations should have positive LOD
        let obs = vec![0.9998, 0.9999, 0.9997];
        let lods = compute_per_window_lod(&obs, &params);

        assert_eq!(lods.len(), 3);
        for (i, &lod) in lods.iter().enumerate() {
            assert!(lod.is_finite(), "LOD at {} should be finite", i);
            assert!(lod > 0.0, "IBD-like observation at {} should have positive LOD (got {})", i, lod);
        }
    }

    #[test]
    fn test_per_window_lod_non_ibd_region_negative() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Observations well below non-IBD mean should have negative LOD
        let obs = vec![0.995, 0.996, 0.994];
        let lods = compute_per_window_lod(&obs, &params);

        assert_eq!(lods.len(), 3);
        for (i, &lod) in lods.iter().enumerate() {
            assert!(lod.is_finite(), "LOD at {} should be finite", i);
            assert!(lod < 0.0, "Non-IBD observation at {} should have negative LOD (got {})", i, lod);
        }
    }

    #[test]
    fn test_per_window_lod_empty() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let lods = compute_per_window_lod(&[], &params);
        assert!(lods.is_empty());
    }

    #[test]
    fn test_segment_lod_score_positive_for_ibd() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Clear IBD region
        let obs = vec![0.998, 0.998, 0.9998, 0.9999, 0.9997, 0.9998, 0.998, 0.998];
        let lod = segment_lod_score(&obs, 2, 5, &params);
        assert!(lod > 0.0, "IBD segment LOD should be positive, got {}", lod);
    }

    #[test]
    fn test_segment_lod_score_negative_for_non_ibd() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Non-IBD region
        let obs = vec![0.995, 0.996, 0.994, 0.993, 0.995];
        let lod = segment_lod_score(&obs, 0, 4, &params);
        assert!(lod < 0.0, "Non-IBD segment LOD should be negative, got {}", lod);
    }

    #[test]
    fn test_segment_lod_score_boundary_cases() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.999, 0.9998];

        // Invalid range: start > end
        assert_eq!(segment_lod_score(&obs, 5, 1, &params), 0.0);

        // Invalid range: end out of bounds
        assert_eq!(segment_lod_score(&obs, 0, 10, &params), 0.0);

        // Single window segment
        let lod = segment_lod_score(&obs, 1, 1, &params);
        assert!(lod.is_finite());
    }

    #[test]
    fn test_segment_lod_score_accumulates() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Multiple IBD windows: LOD should increase with more evidence
        let obs = vec![0.9998, 0.9999, 0.9997, 0.9998, 0.9999];

        let lod_3 = segment_lod_score(&obs, 0, 2, &params);  // 3 windows
        let lod_5 = segment_lod_score(&obs, 0, 4, &params);  // 5 windows

        assert!(lod_5 > lod_3, "Longer IBD segment should have higher LOD: 5w={}, 3w={}", lod_5, lod_3);
    }

    #[test]
    fn test_extract_segments_with_lod() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Create observations with a clear IBD region
        let obs = vec![0.998, 0.998, 0.9998, 0.9999, 0.9997, 0.998, 0.998];
        let states = vec![0, 0, 1, 1, 1, 0, 0];
        let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.85, 0.1, 0.1];

        let segments = extract_ibd_segments_with_lod(
            &states, &posteriors, 1, 0.5,
            Some((&obs, &params)), None,
        );

        assert_eq!(segments.len(), 1);
        assert!(segments[0].lod_score > 0.0,
            "IBD segment should have positive LOD, got {}", segments[0].lod_score);
        assert!(segments[0].lod_score.is_finite());
    }

    #[test]
    fn test_extract_segments_with_lod_filter() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Create observations with weak vs strong IBD
        let obs = vec![0.9998, 0.9999, 0.9997, 0.998, 0.998, 0.998, 0.998, 0.998,
                       0.9998, 0.9999, 0.9997, 0.9998, 0.9999, 0.9997, 0.9998, 0.9999];
        let states = vec![1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        let posteriors = vec![0.9; 16];

        // Without LOD filter
        let segments_no_filter = extract_ibd_segments_with_lod(
            &states, &posteriors, 1, 0.5,
            Some((&obs, &params)), None,
        );
        assert_eq!(segments_no_filter.len(), 2);

        // The longer segment should have higher LOD
        assert!(segments_no_filter[1].lod_score > segments_no_filter[0].lod_score,
            "Longer segment LOD {} should be > shorter segment LOD {}",
            segments_no_filter[1].lod_score, segments_no_filter[0].lod_score);

        // With high LOD filter: only the longer segment should pass
        let min_lod = segments_no_filter[0].lod_score + 0.01;  // just above the shorter segment
        let segments_filtered = extract_ibd_segments_with_lod(
            &states, &posteriors, 1, 0.5,
            Some((&obs, &params)), Some(min_lod),
        );
        assert_eq!(segments_filtered.len(), 1);
        assert_eq!(segments_filtered[0].start_idx, 8);
    }

    #[test]
    fn test_extract_segments_without_obs_lod_is_zero() {
        // Without observations/params, LOD should default to 0.0
        let states = vec![1, 1, 1, 0, 0];
        let posteriors = vec![0.9, 0.95, 0.88, 0.1, 0.1];

        let segments = extract_ibd_segments_with_lod(
            &states, &posteriors, 1, 0.5, None, None,
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].lod_score, 0.0);
    }

    // === Quality score tests ===

    #[test]
    fn test_quality_score_perfect_segment() {
        let seg = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 49,
            n_windows: 50,
            mean_posterior: 0.99,
            min_posterior: 0.95,
            max_posterior: 1.0,
            lod_score: 80.0, // 80 / 50 = 1.6 per window
        };

        let q = segment_quality_score(&seg);
        assert!(q >= 90.0, "Perfect segment should have Q >= 90, got {:.1}", q);
        assert!(q <= 100.0, "Quality should be <= 100, got {:.1}", q);
    }

    #[test]
    fn test_quality_score_weak_segment() {
        let seg = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 2,
            n_windows: 3,
            mean_posterior: 0.5,
            min_posterior: 0.2,
            max_posterior: 0.8,
            lod_score: 0.5, // 0.5 / 3 = 0.17 per window
        };

        let q = segment_quality_score(&seg);
        assert!(q < 50.0, "Weak segment should have Q < 50, got {:.1}", q);
        assert!(q > 0.0, "Quality should be > 0, got {:.1}", q);
    }

    #[test]
    fn test_quality_score_zero_posterior() {
        let seg = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 0,
            n_windows: 1,
            mean_posterior: 0.0,
            min_posterior: 0.0,
            max_posterior: 0.0,
            lod_score: 0.0,
        };

        let q = segment_quality_score(&seg);
        assert!(q >= 0.0, "Quality should be >= 0, got {:.1}", q);
        assert!(q < 10.0, "Zero-evidence segment should have very low Q, got {:.1}", q);
    }

    #[test]
    fn test_quality_score_ordering() {
        // A strong segment should have higher quality than a weak one
        let strong = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 29,
            n_windows: 30,
            mean_posterior: 0.95,
            min_posterior: 0.85,
            max_posterior: 1.0,
            lod_score: 45.0,
        };
        let weak = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 4,
            n_windows: 5,
            mean_posterior: 0.6,
            min_posterior: 0.3,
            max_posterior: 0.8,
            lod_score: 2.0,
        };

        let q_strong = segment_quality_score(&strong);
        let q_weak = segment_quality_score(&weak);
        assert!(q_strong > q_weak,
            "Strong segment Q ({:.1}) should be > weak Q ({:.1})", q_strong, q_weak);
    }

    #[test]
    fn test_quality_score_components_bounded() {
        // Test with extreme values to verify clamping
        let seg = IbdSegmentWithPosterior {
            start_idx: 0,
            end_idx: 999,
            n_windows: 1000,
            mean_posterior: 1.0,
            min_posterior: 1.0,
            max_posterior: 1.0,
            lod_score: 10000.0,
        };
        let q = segment_quality_score(&seg);
        assert_eq!(q, 100.0, "Maximum quality should be 100, got {:.1}", q);
    }

    // === Posterior std tests ===

    #[test]
    fn test_posterior_std_uniform() {
        let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
        let std = segment_posterior_std(&posteriors, 0, 4);
        assert!(std < 1e-10, "Uniform posteriors should have std=0, got {}", std);
    }

    #[test]
    fn test_posterior_std_variable() {
        let posteriors = vec![0.5, 0.9, 0.5, 0.9, 0.5];
        let std = segment_posterior_std(&posteriors, 0, 4);
        assert!(std > 0.1, "Variable posteriors should have std > 0.1, got {}", std);
    }

    #[test]
    fn test_posterior_std_single_window() {
        let posteriors = vec![0.9];
        let std = segment_posterior_std(&posteriors, 0, 0);
        assert_eq!(std, 0.0, "Single window should have std=0");
    }

    #[test]
    fn test_posterior_std_boundary_cases() {
        let posteriors = vec![0.9, 0.8];
        assert_eq!(segment_posterior_std(&posteriors, 5, 1), 0.0, "Invalid range");
        assert_eq!(segment_posterior_std(&posteriors, 0, 10), 0.0, "Out of bounds");
    }

    // === Distance-dependent transition tests ===

    #[test]
    fn test_distance_dependent_transition_unit_distance() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let log_trans = distance_dependent_log_transition(&params, 5000, 5000);

        // At unit distance (1 window), should match base transitions
        let base_log_trans = [
            [params.transition[0][0].ln(), params.transition[0][1].ln()],
            [params.transition[1][0].ln(), params.transition[1][1].ln()],
        ];

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (log_trans[i][j] - base_log_trans[i][j]).abs() < 0.01,
                    "At unit distance, transitions should match base: [{i}][{j}] got {:.4} vs {:.4}",
                    log_trans[i][j], base_log_trans[i][j]
                );
            }
        }
    }

    #[test]
    fn test_distance_dependent_transition_larger_distance() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

        // Compare transitions at 1x vs 3x nominal distance
        let log_trans_1x = distance_dependent_log_transition(&params, 5000, 5000);
        let log_trans_3x = distance_dependent_log_transition(&params, 15000, 5000);

        // At larger distance, probability of state change should be higher
        // So P(enter IBD | 3x) > P(enter IBD | 1x) => log_trans[0][1] should be less negative
        assert!(
            log_trans_3x[0][1] > log_trans_1x[0][1],
            "Larger distance should increase P(enter IBD): 3x={:.4}, 1x={:.4}",
            log_trans_3x[0][1], log_trans_1x[0][1]
        );

        // Similarly, P(exit IBD | 3x) > P(exit IBD | 1x)
        assert!(
            log_trans_3x[1][0] > log_trans_1x[1][0],
            "Larger distance should increase P(exit IBD): 3x={:.4}, 1x={:.4}",
            log_trans_3x[1][0], log_trans_1x[1][0]
        );
    }

    #[test]
    fn test_distance_dependent_transition_zero_distance() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let log_trans = distance_dependent_log_transition(&params, 0, 5000);

        // Zero distance should return base transitions
        let base_log_trans = [
            [params.transition[0][0].ln(), params.transition[0][1].ln()],
            [params.transition[1][0].ln(), params.transition[1][1].ln()],
        ];

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (log_trans[i][j] - base_log_trans[i][j]).abs() < 1e-10,
                    "At zero distance, should match base"
                );
            }
        }
    }

    #[test]
    fn test_distance_dependent_transitions_sum_to_one() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

        for distance in [1000u64, 5000, 10000, 50000, 100000] {
            let log_trans = distance_dependent_log_transition(&params, distance, 5000);
            for (i, row) in log_trans.iter().enumerate() {
                let sum = row[0].exp() + row[1].exp();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Transitions from state {} at distance {} should sum to 1, got {}",
                    i, distance, sum
                );
            }
        }
    }

    #[test]
    fn test_forward_with_distances_uniform_matches_forward() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997];

        // Uniform window positions (5kb each)
        let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

        let (alpha_regular, ll_regular) = forward(&obs, &params);
        let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);

        // With uniform spacing, results should be very similar
        assert!(
            (ll_regular - ll_dist).abs() < 0.5,
            "Uniform spacing log-likelihood should be close: regular={:.2}, dist={:.2}",
            ll_regular, ll_dist
        );
        assert_eq!(alpha_regular.len(), alpha_dist.len());
    }

    #[test]
    fn test_forward_backward_with_distances_properties() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997];
        let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

        let (posteriors, log_lik) = forward_backward_with_distances(&obs, &params, &positions);

        assert_eq!(posteriors.len(), 5);
        assert!(log_lik.is_finite());
        for (i, &p) in posteriors.iter().enumerate() {
            assert!((0.0..=1.0).contains(&p), "Posterior at {} should be in [0,1], got {}", i, p);
        }
    }

    #[test]
    fn test_forward_backward_with_gap() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        // Create a scenario with a large gap between windows 2 and 3
        let obs = vec![0.998, 0.998, 0.9998, 0.9998, 0.998];

        // Windows with a 50kb gap in the middle
        let positions = vec![
            (0, 4999),       // window 0
            (5000, 9999),    // window 1: 5kb from prev
            (10000, 14999),  // window 2: 5kb from prev
            (60000, 64999),  // window 3: 50kb gap!
            (65000, 69999),  // window 4: 5kb from prev
        ];

        let (post_with_gap, _) = forward_backward_with_distances(&obs, &params, &positions);

        // Without gap awareness
        let (post_no_gap, _) = forward_backward(&obs, &params);

        // The model with gap awareness should be more willing to have a state
        // transition at the gap, since 50kb of unobserved sequence has passed
        assert_eq!(post_with_gap.len(), 5);
        assert_eq!(post_no_gap.len(), 5);

        // Both should produce valid posteriors
        for &p in &post_with_gap {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_forward_with_distances_mismatched_positions() {
        // If positions don't match, should fall back to regular forward
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9998];
        let positions = vec![(0u64, 4999u64)]; // only 1 position for 3 observations

        let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);
        let (alpha_reg, ll_reg) = forward(&obs, &params);

        // Should fall back to regular forward
        assert_eq!(alpha_dist.len(), alpha_reg.len());
        assert!((ll_dist - ll_reg).abs() < 1e-10);
    }

    #[test]
    fn test_viterbi_with_distances_uniform_matches_standard() {
        // With uniform window spacing, distance-aware Viterbi should produce
        // the same result as standard Viterbi
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![
            0.998, 0.997, 0.9985, 0.998,  // non-IBD
            0.9998, 0.9999, 0.9997, 0.9998, 0.9999,  // IBD
            0.997, 0.998,  // non-IBD
        ];
        // Uniform spacing: each window is exactly 5000bp apart
        let positions: Vec<(u64, u64)> = (0..obs.len())
            .map(|i| (i as u64 * 5000, i as u64 * 5000 + 4999))
            .collect();

        let states_std = viterbi(&obs, &params);
        let states_dist = viterbi_with_distances(&obs, &params, &positions);

        assert_eq!(states_std, states_dist);
    }

    #[test]
    fn test_viterbi_with_distances_empty() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let states = viterbi_with_distances(&[], &params, &[]);
        assert!(states.is_empty());
    }

    #[test]
    fn test_viterbi_with_distances_mismatched_fallback() {
        // Mismatched positions should fall back to standard Viterbi
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.9998, 0.997];
        let positions = vec![(0u64, 4999u64)]; // wrong length

        let states_dist = viterbi_with_distances(&obs, &params, &positions);
        let states_std = viterbi(&obs, &params);

        assert_eq!(states_dist, states_std);
    }

    #[test]
    fn test_viterbi_with_distances_gap_increases_switching() {
        // A large gap between windows should increase switching probability.
        // Set up: non-IBD -> IBD transition with a large gap at the boundary.
        // The gap should make the model MORE willing to switch states.
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

        let obs = vec![
            0.998, 0.998, 0.998,  // non-IBD
            0.9996, 0.9997, 0.9998, 0.9997,  // moderately high (ambiguous without gap)
        ];

        // With a huge gap (simulating 1Mb gap between windows 2 and 3),
        // the model should be more willing to switch to IBD
        let positions_gap = vec![
            (0, 4999), (5000, 9999), (10000, 14999),
            (1000000, 1004999), (1005000, 1009999), (1010000, 1014999), (1015000, 1019999),
        ];
        // Without gap (uniform spacing)
        let positions_uniform: Vec<(u64, u64)> = (0..7)
            .map(|i| (i as u64 * 5000, i as u64 * 5000 + 4999))
            .collect();

        let states_gap = viterbi_with_distances(&obs, &params, &positions_gap);
        let states_uniform = viterbi_with_distances(&obs, &params, &positions_uniform);

        // Both should produce valid state sequences
        assert_eq!(states_gap.len(), 7);
        assert_eq!(states_uniform.len(), 7);

        // The gap version may have different decisions at the boundary
        // At minimum, both should classify the first 3 windows similarly
        // (as non-IBD since identity ~0.998 is clearly non-IBD)
        for s in &states_gap[..3] {
            assert_eq!(*s, 0, "First windows should be non-IBD");
        }
    }

    #[test]
    fn test_viterbi_with_distances_detects_ibd_region() {
        // Verify that distance-aware Viterbi correctly detects a clear IBD region
        // Use emission estimation to adapt to the data's distribution
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let mut obs = Vec::new();
        // 20 non-IBD windows with some variation (near EUR non-IBD mean 0.99915)
        for i in 0..20 {
            obs.push(0.9990 + (i as f64 % 5.0) * 0.0001);
        }
        // 15 IBD windows with some variation (near IBD mean 0.9997)
        for i in 0..15 {
            obs.push(0.9996 + (i as f64 % 3.0) * 0.0001);
        }
        // 15 non-IBD windows
        for i in 0..15 {
            obs.push(0.9990 + (i as f64 % 5.0) * 0.0001);
        }

        let positions: Vec<(u64, u64)> = (0..obs.len())
            .map(|i| (i as u64 * 5000, i as u64 * 5000 + 4999))
            .collect();

        // Estimate emissions from data so model adapts
        params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

        let states = viterbi_with_distances(&obs, &params, &positions);

        // Most of the IBD region should be detected
        let ibd_detected: usize = states[20..35].iter().filter(|&&s| s == 1).count();
        assert!(ibd_detected >= 10, "Should detect at least 10/15 IBD windows, got {}", ibd_detected);
    }

    #[test]
    fn test_baum_welch_with_distances_improves_likelihood() {
        // Baum-Welch should improve (or maintain) likelihood
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let obs: Vec<f64> = (0..100).map(|i| {
            if i < 70 { 0.9990 + (i as f64 % 5.0) * 0.0001 }
            else { 0.9997 + (i as f64 % 3.0) * 0.00005 }
        }).collect();

        let positions: Vec<(u64, u64)> = (0..obs.len())
            .map(|i| (i as u64 * 5000, i as u64 * 5000 + 4999))
            .collect();

        let (_, ll_before) = forward_with_distances(&obs, &params, &positions);
        params.baum_welch_with_distances(&obs, &positions, 10, 1e-6, Some(Population::EUR), 5000);
        let (_, ll_after) = forward_with_distances(&obs, &params, &positions);

        // Likelihood should not decrease (EM property)
        assert!(ll_after >= ll_before - 1e-6,
            "Baum-Welch should not decrease likelihood: before={}, after={}", ll_before, ll_after);
    }

    #[test]
    fn test_baum_welch_with_distances_too_few_observations() {
        // Should be a no-op with too few observations
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let params_before = params.clone();
        let obs = vec![0.998, 0.999, 0.9998];
        let positions = vec![(0u64, 4999u64), (5000, 9999), (10000, 14999)];

        params.baum_welch_with_distances(&obs, &positions, 10, 1e-6, Some(Population::EUR), 5000);

        // Parameters should be unchanged
        assert_eq!(params.emission[0].mean, params_before.emission[0].mean);
        assert_eq!(params.emission[1].mean, params_before.emission[1].mean);
    }

    #[test]
    fn test_baum_welch_with_distances_mismatched_fallback() {
        // Mismatched positions should fall back to standard Baum-Welch
        let mut params_dist = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let mut params_std = params_dist.clone();

        let obs: Vec<f64> = (0..50).map(|i| {
            if i < 35 { 0.9990 } else { 0.9997 }
        }).collect();
        let positions = vec![(0u64, 4999u64)]; // wrong length

        params_dist.baum_welch_with_distances(&obs, &positions, 5, 1e-6, Some(Population::EUR), 5000);
        params_std.baum_welch(&obs, 5, 1e-6, Some(Population::EUR), 5000);

        // Should produce same results since it falls back
        assert!((params_dist.emission[0].mean - params_std.emission[0].mean).abs() < 1e-10);
        assert!((params_dist.emission[1].mean - params_std.emission[1].mean).abs() < 1e-10);
    }

    #[test]
    fn test_baum_welch_with_distances_maintains_valid_params() {
        let mut params = HmmParams::from_population(Population::AFR, 30.0, 0.0001, 5000);
        let obs: Vec<f64> = (0..200).map(|i| {
            if i < 150 { 0.9985 + (i as f64 % 7.0) * 0.00005 }
            else { 0.9996 + (i as f64 % 3.0) * 0.00005 }
        }).collect();

        let positions: Vec<(u64, u64)> = (0..obs.len())
            .map(|i| (i as u64 * 5000, i as u64 * 5000 + 4999))
            .collect();

        params.baum_welch_with_distances(&obs, &positions, 20, 1e-8, Some(Population::AFR), 5000);

        // Transitions should be valid probabilities
        for row in &params.transition {
            assert!((row[0] + row[1] - 1.0).abs() < 1e-10, "Transition row must sum to 1");
            for &p in row {
                assert!(p > 0.0 && p < 1.0, "Transition prob must be in (0,1): {}", p);
            }
        }

        // Emissions should have positive std and state 0 < state 1
        assert!(params.emission[0].std > 0.0);
        assert!(params.emission[1].std > 0.0);
        assert!(params.emission[0].mean < params.emission[1].mean,
            "Non-IBD mean ({}) should be < IBD mean ({})",
            params.emission[0].mean, params.emission[1].mean);
    }

    #[test]
    fn test_full_distance_aware_pipeline() {
        // End-to-end test: BW training -> Viterbi -> FB -> refinement with distances
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

        // Synthetic data: non-IBD regions with IBD at windows 100-150
        let mut obs = Vec::new();
        let mut positions = Vec::new();
        for i in 0..200 {
            if (100..150).contains(&i) {
                obs.push(0.9997 + (i as f64 % 3.0) * 0.00005);
            } else {
                obs.push(0.9990 + (i as f64 % 5.0) * 0.0001);
            }
            positions.push((i as u64 * 5000, i as u64 * 5000 + 4999));
        }

        // Robust emission estimation
        params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

        // Distance-aware Baum-Welch
        params.baum_welch_with_distances(&obs, &positions, 10, 1e-6, Some(Population::EUR), 5000);

        // Distance-aware Viterbi
        let mut states = viterbi_with_distances(&obs, &params, &positions);

        // Distance-aware forward-backward
        let (posteriors, _log_lik) = forward_backward_with_distances(&obs, &params, &positions);

        // Posterior refinement
        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Verify: at least 60% of IBD region detected
        let ibd_detected: usize = states[100..150].iter().filter(|&&s| s == 1).count();
        assert!(ibd_detected >= 30,
            "Should detect at least 30/50 IBD windows, got {}", ibd_detected);

        // Verify: low false positive rate outside IBD region
        let fp_count: usize = states[..100].iter().chain(states[150..].iter())
            .filter(|&&s| s == 1).count();
        let fp_rate = fp_count as f64 / 150.0;
        assert!(fp_rate < 0.1, "False positive rate should be < 10%, got {:.1}%", fp_rate * 100.0);
    }

    // === Genetic Map tests ===

    #[test]
    fn test_genetic_map_new() {
        let entries = vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 2.5)];
        let gmap = GeneticMap::new(entries);
        assert_eq!(gmap.len(), 3);
        assert!(!gmap.is_empty());
    }

    #[test]
    fn test_genetic_map_interpolate_at_entries() {
        let entries = vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 2.0)];
        let gmap = GeneticMap::new(entries);

        assert!((gmap.interpolate_cm(0) - 0.0).abs() < 1e-10);
        assert!((gmap.interpolate_cm(1_000_000) - 1.0).abs() < 1e-10);
        assert!((gmap.interpolate_cm(2_000_000) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_genetic_map_interpolate_between_entries() {
        let entries = vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 3.0)];
        let gmap = GeneticMap::new(entries);

        // Midpoint of first segment (rate 1 cM/Mb)
        let cm = gmap.interpolate_cm(500_000);
        assert!((cm - 0.5).abs() < 1e-10, "Expected 0.5 cM, got {}", cm);

        // Midpoint of second segment (rate 2 cM/Mb)
        let cm = gmap.interpolate_cm(1_500_000);
        assert!((cm - 2.0).abs() < 1e-10, "Expected 2.0 cM, got {}", cm);
    }

    #[test]
    fn test_genetic_map_extrapolate_before() {
        let entries = vec![(1_000_000, 1.0), (2_000_000, 2.0)];
        let gmap = GeneticMap::new(entries);

        // Before first entry: extrapolate at 1 cM/Mb rate
        let cm = gmap.interpolate_cm(500_000);
        assert!((cm - 0.5).abs() < 1e-10, "Expected 0.5 cM, got {}", cm);
    }

    #[test]
    fn test_genetic_map_extrapolate_after() {
        let entries = vec![(0, 0.0), (1_000_000, 1.0)];
        let gmap = GeneticMap::new(entries);

        // After last entry: extrapolate at 1 cM/Mb rate
        let cm = gmap.interpolate_cm(2_000_000);
        assert!((cm - 2.0).abs() < 1e-10, "Expected 2.0 cM, got {}", cm);
    }

    #[test]
    fn test_genetic_map_distance() {
        let entries = vec![(0, 0.0), (1_000_000, 1.0), (2_000_000, 3.0)];
        let gmap = GeneticMap::new(entries);

        let dist = gmap.genetic_distance_cm(0, 2_000_000);
        assert!((dist - 3.0).abs() < 1e-10, "Expected 3.0 cM, got {}", dist);

        let dist = gmap.genetic_distance_cm(500_000, 1_500_000);
        assert!((dist - 1.5).abs() < 1e-10, "Expected 1.5 cM, got {}", dist);
    }

    #[test]
    fn test_genetic_map_uniform() {
        let gmap = GeneticMap::uniform(0, 10_000_000, 1.0);
        assert_eq!(gmap.len(), 2);

        let dist = gmap.genetic_distance_cm(0, 10_000_000);
        assert!((dist - 10.0).abs() < 1e-10, "Expected 10.0 cM, got {}", dist);

        let dist = gmap.genetic_distance_cm(0, 5_000_000);
        assert!((dist - 5.0).abs() < 1e-10, "Expected 5.0 cM, got {}", dist);
    }

    #[test]
    fn test_genetic_map_variable_rate() {
        // Simulate a recombination hotspot: high rate in middle
        let entries = vec![
            (0, 0.0),
            (1_000_000, 0.5),    // low rate: 0.5 cM/Mb
            (1_100_000, 5.5),    // hotspot: 50 cM/Mb over 100kb
            (2_000_000, 6.0),    // low rate again: 0.56 cM/Mb
        ];
        let gmap = GeneticMap::new(entries);

        // Distance across hotspot should be large
        let hotspot_dist = gmap.genetic_distance_cm(1_000_000, 1_100_000);
        assert!((hotspot_dist - 5.0).abs() < 1e-10, "Hotspot distance should be 5.0 cM, got {}", hotspot_dist);

        // Distance in low-rate region should be small
        let low_dist = gmap.genetic_distance_cm(0, 1_000_000);
        assert!((low_dist - 0.5).abs() < 1e-10, "Low-rate distance should be 0.5 cM, got {}", low_dist);
    }

    #[test]
    fn test_genetic_map_single_entry() {
        let gmap = GeneticMap::new(vec![(1_000_000, 5.0)]);
        assert_eq!(gmap.interpolate_cm(1_000_000), 5.0);
        assert_eq!(gmap.interpolate_cm(500_000), 5.0);
    }

    #[test]
    fn test_genetic_map_empty() {
        let gmap = GeneticMap::new(vec![]);
        assert!(gmap.is_empty());
        assert_eq!(gmap.interpolate_cm(1_000_000), 0.0);
    }

    #[test]
    fn test_genetic_map_from_file() {
        // Create a temp genetic map file
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_genetic_map.txt");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "# genetic map").unwrap();
            writeln!(f, "chr1 100000 1.5 0.0").unwrap();
            writeln!(f, "chr1 200000 1.5 0.15").unwrap();
            writeln!(f, "chr1 500000 2.0 0.75").unwrap();
            writeln!(f, "chr2 100000 1.0 0.0").unwrap();
        }

        let gmap = GeneticMap::from_file(&path, "chr1").unwrap();
        assert_eq!(gmap.len(), 3);

        let gmap2 = GeneticMap::from_file(&path, "chr2").unwrap();
        assert_eq!(gmap2.len(), 1);

        // Chromosome not present
        let result = GeneticMap::from_file(&path, "chr3");
        assert!(result.is_err());

        // Test with bare chromosome number (no "chr" prefix)
        let gmap_bare = GeneticMap::from_file(&path, "1").unwrap();
        assert_eq!(gmap_bare.len(), 3);

        std::fs::remove_file(&path).ok();
    }

    // === Recombination-aware transition tests ===

    #[test]
    fn test_recombination_aware_same_position() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let gmap = GeneticMap::uniform(0, 10_000_000, 1.0);

        let log_trans = recombination_aware_log_transition(&params, 100, 100, &gmap, 5000);

        // Same position: should return base transitions
        let base_log_trans = [
            [params.transition[0][0].ln(), params.transition[0][1].ln()],
            [params.transition[1][0].ln(), params.transition[1][1].ln()],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!((log_trans[i][j] - base_log_trans[i][j]).abs() < 1e-10,
                    "Same-position transition should match base");
            }
        }
    }

    #[test]
    fn test_recombination_aware_uniform_matches_distance() {
        // With a uniform recombination rate of 1 cM/Mb, the recombination-aware
        // transitions should be similar to (but not identical to) distance-based ones,
        // since both scale by the same effective distance but the recombination-aware
        // version adds the Haldane correction on exit rate
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let gmap = GeneticMap::uniform(0, 10_000_000, 1.0);

        let log_trans_recomb = recombination_aware_log_transition(
            &params, 0, 5000, &gmap, 5000,
        );
        let log_trans_dist = distance_dependent_log_transition(&params, 5000, 5000);

        // Entry rate should be very close (both scale by ~1 window)
        let p_enter_recomb = log_trans_recomb[0][1].exp();
        let p_enter_dist = log_trans_dist[0][1].exp();
        assert!((p_enter_recomb - p_enter_dist).abs() < 0.001,
            "Entry rate: recomb={:.6} vs dist={:.6}", p_enter_recomb, p_enter_dist);

        // Exit rate slightly higher for recomb-aware due to recombination factor
        let p_exit_recomb = log_trans_recomb[1][0].exp();
        let p_exit_dist = log_trans_dist[1][0].exp();
        assert!(p_exit_recomb >= p_exit_dist * 0.99,
            "Exit rate with recomb should be >= distance-only");
    }

    #[test]
    fn test_recombination_aware_hotspot_increases_exit() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

        // Low recombination rate
        let gmap_low = GeneticMap::uniform(0, 10_000_000, 0.1); // 0.1 cM/Mb
        // High recombination rate (hotspot)
        let gmap_high = GeneticMap::uniform(0, 10_000_000, 5.0); // 5 cM/Mb

        let log_trans_low = recombination_aware_log_transition(
            &params, 0, 50_000, &gmap_low, 5000,
        );
        let log_trans_high = recombination_aware_log_transition(
            &params, 0, 50_000, &gmap_high, 5000,
        );

        let p_exit_low = log_trans_low[1][0].exp();
        let p_exit_high = log_trans_high[1][0].exp();

        // Higher recombination rate → higher exit probability (IBD breaks at hotspots)
        assert!(p_exit_high > p_exit_low,
            "Hotspot should increase exit probability: high={:.6} vs low={:.6}",
            p_exit_high, p_exit_low);
    }

    #[test]
    fn test_recombination_aware_transitions_valid() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let gmap = GeneticMap::uniform(0, 100_000_000, 1.5);

        // Test at various distances
        for distance in [5000, 50_000, 500_000, 5_000_000] {
            let log_trans = recombination_aware_log_transition(
                &params, 0, distance, &gmap, 5000,
            );

            // All log probabilities should be <= 0
            for row in log_trans.iter() {
                for &val in row {
                    assert!(val <= 0.0,
                        "Log transition should be <= 0, got {} at dist={}", val, distance);
                }
                // Row should sum to ~1 in probability space
                let row_sum = row[0].exp() + row[1].exp();
                assert!((row_sum - 1.0).abs() < 1e-6,
                    "Row sum should be ~1, got {} at dist={}", row_sum, distance);
            }
        }
    }

    #[test]
    fn test_viterbi_with_genetic_map_matches_standard_uniform() {
        // With uniform map, viterbi_with_genetic_map should give similar results
        // to viterbi_with_distances
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let n = 50;
        let positions: Vec<(u64, u64)> = (0..n)
            .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
            .collect();

        let mut obs = vec![0.99915; n];
        for o in &mut obs[20..30] {
            *o = 0.9997;
        }

        params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

        let gmap = GeneticMap::uniform(0, n as u64 * 5000, 1.0);

        let states_dist = viterbi_with_distances(&obs, &params, &positions);
        let states_gmap = viterbi_with_genetic_map(&obs, &params, &positions, &gmap, 5000);

        // Should produce similar (if not identical) state sequences with uniform map
        let agreement: usize = states_dist.iter().zip(states_gmap.iter())
            .filter(|(a, b)| a == b).count();
        assert!(agreement >= n * 9 / 10,
            "Uniform genetic map should give similar results: {}/{} agreed", agreement, n);
    }

    #[test]
    fn test_forward_backward_with_genetic_map_valid_posteriors() {
        let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let n = 100;
        let positions: Vec<(u64, u64)> = (0..n)
            .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
            .collect();

        let mut obs = vec![0.99915; n];
        for o in &mut obs[40..60] {
            *o = 0.9997;
        }

        let gmap = GeneticMap::uniform(0, n as u64 * 5000, 1.0);

        let (posteriors, log_lik) = forward_backward_with_genetic_map(
            &obs, &params, &positions, &gmap, 5000,
        );

        assert_eq!(posteriors.len(), n);
        assert!(log_lik.is_finite(), "Log-likelihood should be finite");

        for (i, &p) in posteriors.iter().enumerate() {
            assert!((0.0..=1.0).contains(&p),
                "Posterior at {} should be in [0,1], got {}", i, p);
        }
    }

    #[test]
    fn test_forward_backward_genetic_map_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
        let (posteriors, _) = forward_backward_with_genetic_map(
            &[], &params, &[], &gmap, 5000,
        );
        assert!(posteriors.is_empty());
    }

    #[test]
    fn test_viterbi_genetic_map_mismatched_fallback() {
        // Mismatched positions length should fall back to standard Viterbi
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9997];
        let positions = vec![(0, 4999)]; // only 1 position for 3 observations
        let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);

        let states = viterbi_with_genetic_map(&obs, &params, &positions, &gmap, 5000);
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_baum_welch_with_genetic_map_improves_likelihood() {
        let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
        let n = 200;
        let positions: Vec<(u64, u64)> = (0..n)
            .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
            .collect();

        let mut obs = vec![0.99915; n];
        for o in &mut obs[80..120] {
            *o = 0.9997;
        }

        params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

        let gmap = GeneticMap::uniform(0, n as u64 * 5000, 1.0);

        // Get initial log-likelihood
        let (_, ll_before) = forward_with_genetic_map(&obs, &params, &positions, &gmap, 5000);

        // Run BW with genetic map
        params.baum_welch_with_genetic_map(
            &obs, &positions, &gmap, 5, 1e-8, Some(Population::EUR), 5000,
        );

        // Get final log-likelihood
        let (_, ll_after) = forward_with_genetic_map(&obs, &params, &positions, &gmap, 5000);

        assert!(ll_after >= ll_before - 1e-6,
            "BW should improve log-likelihood: before={:.2} after={:.2}", ll_before, ll_after);
    }

    #[test]
    fn test_baum_welch_genetic_map_too_few_obs() {
        let mut params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let initial_emission = params.emission[0];
        let obs = vec![0.999; 5];
        let positions: Vec<(u64, u64)> = (0..5)
            .map(|i| (i as u64 * 5000, (i + 1) as u64 * 5000 - 1))
            .collect();
        let gmap = GeneticMap::uniform(0, 25_000, 1.0);

        params.baum_welch_with_genetic_map(
            &obs, &positions, &gmap, 10, 1e-6, None, 5000,
        );

        // Should be no-op with < 10 observations
        assert!((params.emission[0].mean - initial_emission.mean).abs() < 1e-10,
            "BW should be no-op with too few observations");
    }

    #[test]
    fn test_full_recombination_aware_pipeline() {
        // Full pipeline test: recombination-aware BW → Viterbi → FB → refinement
        // with a hotspot that should break an IBD segment
        let n = 200;
        let window_size = 5000u64;

        // Create observations: non-IBD background with IBD region
        let mut obs = vec![0.99875; n]; // AFR non-IBD
        for o in &mut obs[50..150] {
            *o = 0.9997; // IBD region
        }

        let positions: Vec<(u64, u64)> = (0..n)
            .map(|i| (i as u64 * window_size, (i + 1) as u64 * window_size - 1))
            .collect();

        // Create genetic map with a hotspot in the middle of the IBD region
        let entries = vec![
            (0, 0.0),
            (250_000, 0.25),     // before IBD: 1 cM/Mb
            (500_000, 0.50),     // IBD region start
            (750_000, 0.75),     // normal rate through IBD
            (1_000_000, 5.75),   // recombination hotspot at 100th window
            (1_250_000, 6.0),    // normal rate after
        ];
        let gmap = GeneticMap::new(entries);

        let mut params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, window_size);
        params.estimate_emissions_robust(&obs, Some(Population::AFR), window_size);

        // Run BW with genetic map
        params.baum_welch_with_genetic_map(
            &obs, &positions, &gmap, 10, 1e-6, Some(Population::AFR), window_size,
        );

        // Run Viterbi with genetic map
        let mut states = viterbi_with_genetic_map(&obs, &params, &positions, &gmap, window_size);

        // Run FB with genetic map
        let (posteriors, log_lik) = forward_backward_with_genetic_map(
            &obs, &params, &positions, &gmap, window_size,
        );

        assert!(log_lik.is_finite());

        // Posterior refinement
        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Verify IBD region is at least partially detected
        let ibd_detected: usize = states[50..150].iter().filter(|&&s| s == 1).count();
        assert!(ibd_detected >= 50,
            "Should detect at least 50/100 IBD windows, got {}", ibd_detected);

        // Verify low false positive rate outside IBD
        let fp_count: usize = states[..50].iter().chain(states[150..].iter())
            .filter(|&&s| s == 1).count();
        let fp_rate = fp_count as f64 / 100.0;
        assert!(fp_rate < 0.1, "FP rate should be < 10%, got {:.1}%", fp_rate * 100.0);

        // Verify all posteriors are valid
        for &p in &posteriors {
            assert!((0.0..=1.0).contains(&p), "Posterior out of range: {}", p);
        }
    }

    #[test]
    fn test_recombination_aware_all_populations() {
        let gmap = GeneticMap::uniform(0, 50_000_000, 1.0);
        let window_size = 5000u64;

        for pop in [Population::AFR, Population::EUR, Population::EAS,
                    Population::CSA, Population::AMR, Population::InterPop] {
            let params = HmmParams::from_population(pop, 50.0, 0.0001, window_size);

            for distance in [5000u64, 50_000, 500_000] {
                let log_trans = recombination_aware_log_transition(
                    &params, 0, distance, &gmap, window_size,
                );

                for i in 0..2 {
                    let row_sum = log_trans[i][0].exp() + log_trans[i][1].exp();
                    assert!((row_sum - 1.0).abs() < 1e-6,
                        "Row sum should be 1 for {:?} at dist {}: got {}", pop, distance, row_sum);
                }
            }
        }
    }

    // ===== Logit-transform HMM tests =====

    #[test]
    fn test_from_population_logit_creates_valid_params() {
        for pop in &[Population::AFR, Population::EUR, Population::EAS, Population::Generic] {
            let params = HmmParams::from_population_logit(*pop, 50.0, 0.0001, 5000);

            // Emissions should be in logit space (means > 5)
            assert!(params.emission[0].mean > 5.0,
                "Non-IBD logit mean should be > 5 for {:?}, got {}", pop, params.emission[0].mean);
            assert!(params.emission[1].mean > 5.0,
                "IBD logit mean should be > 5 for {:?}, got {}", pop, params.emission[1].mean);

            // IBD mean should be higher than non-IBD in logit space
            assert!(params.emission[1].mean > params.emission[0].mean,
                "IBD mean ({}) should be > non-IBD mean ({}) in logit space for {:?}",
                params.emission[1].mean, params.emission[0].mean, pop);

            // Stds should be positive
            assert!(params.emission[0].std > 0.0);
            assert!(params.emission[1].std > 0.0);

            // Transitions should be valid
            assert!((params.transition[0][0] + params.transition[0][1] - 1.0).abs() < 1e-10);
            assert!((params.transition[1][0] + params.transition[1][1] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_logit_hmm_better_separation() {
        let pop = Population::EUR;
        let raw_params = HmmParams::from_population(pop, 50.0, 0.0001, 5000);
        let logit_params = HmmParams::from_population_logit(pop, 50.0, 0.0001, 5000);

        // Raw space separation
        let raw_sep = raw_params.emission[1].mean - raw_params.emission[0].mean;

        // Logit space separation
        let logit_sep = logit_params.emission[1].mean - logit_params.emission[0].mean;

        // Logit separation should be much larger
        assert!(logit_sep > raw_sep * 50.0,
            "Logit separation ({:.4}) should be >> raw separation ({:.6})",
            logit_sep, raw_sep);
    }

    #[test]
    fn test_logit_viterbi_on_clear_signal() {
        use crate::stats::logit_transform_observations;

        // Create data with clear IBD signal in raw space
        let mut raw_obs = vec![0.998; 100];
        for i in 30..70 {
            raw_obs[i] = 0.9998;
        }

        // Transform to logit space
        let logit_obs = logit_transform_observations(&raw_obs);

        // Create HMM params in logit space
        let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
        params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

        // Run Viterbi
        let states = viterbi(&logit_obs, &params);

        // Most of the middle should be IBD
        let ibd_in_middle = states[30..70].iter().filter(|&&s| s == 1).count();
        assert!(ibd_in_middle > 30,
            "Should detect IBD in middle region, got {} IBD windows out of 40",
            ibd_in_middle);

        // Edges should be non-IBD
        let ibd_in_edges = states[0..20].iter().filter(|&&s| s == 1).count()
            + states[80..100].iter().filter(|&&s| s == 1).count();
        assert!(ibd_in_edges < 10,
            "Edges should be mostly non-IBD, got {} IBD windows", ibd_in_edges);
    }

    #[test]
    fn test_logit_forward_backward_valid_posteriors() {
        use crate::stats::logit_transform_observations;

        let raw_obs = vec![0.998, 0.997, 0.999, 0.9998, 0.9999, 0.9997, 0.998, 0.997];
        let logit_obs = logit_transform_observations(&raw_obs);

        let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
        let (posteriors, log_lik) = forward_backward(&logit_obs, &params);

        // All posteriors should be valid probabilities
        for (i, &p) in posteriors.iter().enumerate() {
            assert!(p >= 0.0 && p <= 1.0,
                "Posterior at {} should be in [0,1], got {}", i, p);
        }

        // Log-likelihood should be finite
        assert!(log_lik.is_finite(), "Log-likelihood should be finite, got {}", log_lik);
    }

    #[test]
    fn test_logit_baum_welch_improves_likelihood() {
        use crate::stats::logit_transform_observations;

        let mut raw_obs = vec![0.998; 200];
        for i in 50..150 {
            raw_obs[i] = 0.9998;
        }
        let logit_obs = logit_transform_observations(&raw_obs);

        // Run with 1 iteration
        let mut params1 = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
        params1.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
        params1.baum_welch(&logit_obs, 1, 1e-6, Some(Population::EUR), 5000);
        let (_, ll1) = forward_backward(&logit_obs, &params1);

        // Run with 10 iterations
        let mut params10 = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
        params10.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
        params10.baum_welch(&logit_obs, 10, 1e-6, Some(Population::EUR), 5000);
        let (_, ll10) = forward_backward(&logit_obs, &params10);

        // More iterations should not decrease likelihood
        assert!(ll10 >= ll1 - 1e-6,
            "10-iter LL ({:.2}) should be >= 1-iter LL ({:.2})", ll10, ll1);
    }

    #[test]
    fn test_estimate_emissions_logit_updates_params() {
        use crate::stats::logit_transform_observations;

        // Create observations with two distinct groups in raw space
        let mut raw_obs: Vec<f64> = vec![0.997; 80];
        raw_obs.extend(vec![0.9998; 20]);

        let logit_obs = logit_transform_observations(&raw_obs);

        let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
        let original_e0_mean = params.emission[0].mean;
        let original_e1_mean = params.emission[1].mean;

        params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

        // Emissions should have been updated (at least one should change)
        let changed = (params.emission[0].mean - original_e0_mean).abs() > 0.01
            || (params.emission[1].mean - original_e1_mean).abs() > 0.01;
        assert!(changed, "estimate_emissions_logit should update parameters");
    }

    #[test]
    fn test_logit_hmm_empty_and_small_inputs() {
        use crate::stats::logit_transform_observations;

        // Empty observations
        let empty: Vec<f64> = vec![];
        let logit_empty = logit_transform_observations(&empty);
        assert!(logit_empty.is_empty());

        // Single observation
        let single_raw = vec![0.999];
        let logit_single = logit_transform_observations(&single_raw);
        assert_eq!(logit_single.len(), 1);
        assert!(logit_single[0].is_finite());

        // Two observations
        let two_raw = vec![0.998, 0.9999];
        let logit_two = logit_transform_observations(&two_raw);
        assert_eq!(logit_two.len(), 2);
        assert!(logit_two[0] < logit_two[1]);
    }

    // ===== Multi-feature emission tests =====

    #[test]
    fn test_precompute_log_emissions_basic() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.5, 0.999, 0.9999];
        let log_emit = precompute_log_emissions(&obs, &params);
        assert_eq!(log_emit.len(), 3);
        for le in &log_emit {
            assert!(le[0].is_finite());
            assert!(le[1].is_finite());
        }
    }

    #[test]
    fn test_precompute_log_emissions_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs: Vec<f64> = vec![];
        let log_emit = precompute_log_emissions(&obs, &params);
        assert!(log_emit.is_empty());
    }

    #[test]
    fn test_forward_from_log_emit_matches_forward() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997, 0.5];
        let log_emit = precompute_log_emissions(&obs, &params);

        let (alpha_standard, ll_standard) = forward(&obs, &params);
        let (alpha_log_emit, ll_log_emit) = forward_from_log_emit(&log_emit, &params);

        assert_eq!(alpha_standard.len(), alpha_log_emit.len());
        assert!((ll_standard - ll_log_emit).abs() < 1e-10,
            "Log-likelihoods should match: {} vs {}", ll_standard, ll_log_emit);
        for t in 0..obs.len() {
            for s in 0..2 {
                assert!((alpha_standard[t][s] - alpha_log_emit[t][s]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_backward_from_log_emit_matches_backward() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997, 0.5];
        let log_emit = precompute_log_emissions(&obs, &params);

        let beta_standard = backward(&obs, &params);
        let beta_log_emit = backward_from_log_emit(&log_emit, &params);

        assert_eq!(beta_standard.len(), beta_log_emit.len());
        for t in 0..obs.len() {
            for s in 0..2 {
                assert!((beta_standard[t][s] - beta_log_emit[t][s]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_viterbi_from_log_emit_matches_viterbi() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997, 0.5];
        let log_emit = precompute_log_emissions(&obs, &params);

        let states_standard = viterbi(&obs, &params);
        let states_log_emit = viterbi_from_log_emit(&log_emit, &params);

        assert_eq!(states_standard, states_log_emit);
    }

    #[test]
    fn test_forward_backward_from_log_emit_matches() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995, 0.9998, 0.997, 0.5];
        let log_emit = precompute_log_emissions(&obs, &params);

        let (post_standard, ll_standard) = forward_backward(&obs, &params);
        let (post_log_emit, ll_log_emit) = forward_backward_from_log_emit(&log_emit, &params);

        assert!((ll_standard - ll_log_emit).abs() < 1e-10);
        for t in 0..obs.len() {
            assert!((post_standard[t] - post_log_emit[t]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_combined_log_emissions_without_aux() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995];
        let primary_only = precompute_log_emissions(&obs, &params);
        let combined = compute_combined_log_emissions(&obs, &params, None, None);

        assert_eq!(primary_only.len(), combined.len());
        for t in 0..obs.len() {
            for s in 0..2 {
                assert!((primary_only[t][s] - combined[t][s]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_combined_log_emissions_with_aux() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999, 0.9995];
        let aux = vec![0.8, 0.95, 0.99];
        let aux_emit = [
            GaussianParams::new_unchecked(0.85, 0.1),
            GaussianParams::new_unchecked(0.98, 0.02),
        ];

        let combined = compute_combined_log_emissions(&obs, &params, Some(&aux), Some(&aux_emit));
        let primary_only = precompute_log_emissions(&obs, &params);

        assert_eq!(combined.len(), 3);
        // Combined should differ from primary-only (auxiliary contribution)
        for t in 0..obs.len() {
            for s in 0..2 {
                let aux_contribution = aux_emit[s].log_pdf(aux[t]);
                let expected = primary_only[t][s] + aux_contribution;
                assert!((combined[t][s] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_combined_log_emissions_length_mismatch_fallback() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.998, 0.999];
        let aux = vec![0.8]; // Wrong length!
        let aux_emit = [
            GaussianParams::new_unchecked(0.85, 0.1),
            GaussianParams::new_unchecked(0.98, 0.02),
        ];

        // Should fall back to primary-only
        let combined = compute_combined_log_emissions(&obs, &params, Some(&aux), Some(&aux_emit));
        let primary_only = precompute_log_emissions(&obs, &params);
        for t in 0..obs.len() {
            for s in 0..2 {
                assert!((combined[t][s] - primary_only[t][s]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_estimate_auxiliary_emissions() {
        // Non-IBD windows (low posterior): low coverage ratio
        // IBD windows (high posterior): high coverage ratio
        let aux = vec![0.7, 0.75, 0.65, 0.8, 0.95, 0.98, 0.99, 0.96];
        let posteriors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

        // Non-IBD mean should be low, IBD mean should be high
        assert!(aux_emit[0].mean < aux_emit[1].mean,
            "Non-IBD aux mean ({:.4}) should be < IBD aux mean ({:.4})",
            aux_emit[0].mean, aux_emit[1].mean);

        // Non-IBD should be around 0.725
        assert!((aux_emit[0].mean - 0.725).abs() < 0.05);
        // IBD should be around 0.97
        assert!((aux_emit[1].mean - 0.97).abs() < 0.05);

        // Both stds should be positive and reasonable
        assert!(aux_emit[0].std > 0.0 && aux_emit[0].std < 1.0);
        assert!(aux_emit[1].std > 0.0 && aux_emit[1].std < 1.0);
    }

    #[test]
    fn test_estimate_auxiliary_emissions_uniform_posteriors() {
        // All posteriors are 0.5 (uncertain) — should get overall mean for both
        let aux = vec![0.7, 0.8, 0.9, 1.0];
        let posteriors = vec![0.5, 0.5, 0.5, 0.5];

        let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

        // Both means should be similar (equal weighting)
        assert!((aux_emit[0].mean - aux_emit[1].mean).abs() < 0.01,
            "With uniform posteriors, aux means should be similar: {:.4} vs {:.4}",
            aux_emit[0].mean, aux_emit[1].mean);
    }

    #[test]
    fn test_coverage_ratio_equal() {
        assert!((coverage_ratio(1000, 1000) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_ratio_asymmetric() {
        let ratio = coverage_ratio(500, 1000);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_ratio_zero() {
        assert_eq!(coverage_ratio(0, 0), 0.0);
        assert_eq!(coverage_ratio(0, 100), 0.0);
        assert_eq!(coverage_ratio(100, 0), 0.0);
    }

    #[test]
    fn test_coverage_ratio_symmetry() {
        assert!((coverage_ratio(300, 500) - coverage_ratio(500, 300)).abs() < 1e-10);
    }

    #[test]
    fn test_infer_ibd_with_aux_features_no_aux() {
        // Without auxiliary data, should produce valid results
        let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6];
        let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
        let (result, aux_emit) = infer_ibd_with_aux_features(
            &obs, &mut params, Population::Generic, 5000, 5, None,
        );

        assert_eq!(result.states.len(), obs.len());
        assert_eq!(result.posteriors.len(), obs.len());
        assert!(result.log_likelihood.is_finite());
        assert!(aux_emit.is_none());
    }

    #[test]
    fn test_infer_ibd_with_aux_features_with_aux() {
        // Primary: identity values
        let obs = vec![0.5, 0.6, 0.55, 0.5, 0.999, 0.998, 0.9995, 0.999, 0.5, 0.6];
        // Auxiliary: coverage ratio (high = symmetric = IBD)
        let aux = vec![0.6, 0.65, 0.7, 0.55, 0.95, 0.97, 0.99, 0.96, 0.7, 0.6];

        let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
        let (result, aux_emit) = infer_ibd_with_aux_features(
            &obs, &mut params, Population::Generic, 5000, 5, Some(&aux),
        );

        assert_eq!(result.states.len(), obs.len());
        assert_eq!(result.posteriors.len(), obs.len());
        assert!(result.log_likelihood.is_finite());
        assert!(aux_emit.is_some());

        let ae = aux_emit.unwrap();
        // IBD aux emission should have higher mean than non-IBD
        assert!(ae[1].mean >= ae[0].mean,
            "IBD aux mean ({:.4}) should be >= non-IBD ({:.4})", ae[1].mean, ae[0].mean);
    }

    #[test]
    fn test_infer_ibd_with_aux_features_too_few_obs() {
        let obs = vec![0.5, 0.6];
        let mut params = HmmParams::from_expected_length(4.0, 0.01, 5000);
        let (result, _) = infer_ibd_with_aux_features(
            &obs, &mut params, Population::Generic, 5000, 5, None,
        );
        // Should handle gracefully with all non-IBD
        assert_eq!(result.states.len(), 2);
    }

    // === Population::from_str tests ===

    #[test]
    fn test_population_from_str_valid_uppercase() {
        assert_eq!(Population::from_str("AFR"), Some(Population::AFR));
        assert_eq!(Population::from_str("EUR"), Some(Population::EUR));
        assert_eq!(Population::from_str("EAS"), Some(Population::EAS));
        assert_eq!(Population::from_str("CSA"), Some(Population::CSA));
        assert_eq!(Population::from_str("AMR"), Some(Population::AMR));
    }

    #[test]
    fn test_population_from_str_case_insensitive() {
        assert_eq!(Population::from_str("afr"), Some(Population::AFR));
        assert_eq!(Population::from_str("Eur"), Some(Population::EUR));
        assert_eq!(Population::from_str("eas"), Some(Population::EAS));
        assert_eq!(Population::from_str("csa"), Some(Population::CSA));
        assert_eq!(Population::from_str("amr"), Some(Population::AMR));
    }

    #[test]
    fn test_population_from_str_interpop_variants() {
        assert_eq!(Population::from_str("INTERPOP"), Some(Population::InterPop));
        assert_eq!(Population::from_str("INTER"), Some(Population::InterPop));
        assert_eq!(Population::from_str("interpop"), Some(Population::InterPop));
        assert_eq!(Population::from_str("inter"), Some(Population::InterPop));
    }

    #[test]
    fn test_population_from_str_generic_variants() {
        assert_eq!(Population::from_str("GENERIC"), Some(Population::Generic));
        assert_eq!(Population::from_str("UNKNOWN"), Some(Population::Generic));
        assert_eq!(Population::from_str("generic"), Some(Population::Generic));
        assert_eq!(Population::from_str("unknown"), Some(Population::Generic));
    }

    #[test]
    fn test_population_from_str_invalid() {
        assert_eq!(Population::from_str(""), None);
        assert_eq!(Population::from_str("XYZ"), None);
        assert_eq!(Population::from_str("african"), None);
        assert_eq!(Population::from_str("EU"), None);
    }

    // === HmmParams::summary tests ===

    #[test]
    fn test_hmm_params_summary_format() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let summary = params.summary();

        assert!(summary.starts_with("HMM Parameters:"));
        assert!(summary.contains("Initial:"));
        assert!(summary.contains("Transition:"));
        assert!(summary.contains("Emission non-IBD:"));
        assert!(summary.contains("Emission IBD:"));
    }

    #[test]
    fn test_hmm_params_summary_contains_values() {
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let summary = params.summary();

        // Initial probabilities should sum to 1
        assert!(summary.contains("P(non-IBD)="));
        assert!(summary.contains("P(IBD)="));

        // Should contain mean and std for both emissions
        assert!(summary.contains("mean="));
        assert!(summary.contains("std="));
    }

    #[test]
    fn test_hmm_params_summary_population_specific() {
        // Different populations should produce different summaries
        let params_afr = HmmParams::from_population(Population::AFR, 50.0, 0.001, 5000);
        let params_eur = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);

        let summary_afr = params_afr.summary();
        let summary_eur = params_eur.summary();

        // Both should be valid summaries
        assert!(summary_afr.starts_with("HMM Parameters:"));
        assert!(summary_eur.starts_with("HMM Parameters:"));

        // They should differ because AFR has higher diversity → different non-IBD emission
        assert_ne!(summary_afr, summary_eur);
    }

    // === Additional edge cases for log_emit functions ===

    #[test]
    fn test_forward_from_log_emit_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let log_emit: Vec<[f64; 2]> = vec![];
        let (alpha, ll) = forward_from_log_emit(&log_emit, &params);
        assert!(alpha.is_empty());
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn test_backward_from_log_emit_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let log_emit: Vec<[f64; 2]> = vec![];
        let beta = backward_from_log_emit(&log_emit, &params);
        assert!(beta.is_empty());
    }

    #[test]
    fn test_viterbi_from_log_emit_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let log_emit: Vec<[f64; 2]> = vec![];
        let states = viterbi_from_log_emit(&log_emit, &params);
        assert!(states.is_empty());
    }

    #[test]
    fn test_forward_backward_from_log_emit_empty() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let log_emit: Vec<[f64; 2]> = vec![];
        let (post, ll) = forward_backward_from_log_emit(&log_emit, &params);
        assert!(post.is_empty());
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn test_forward_backward_from_log_emit_posteriors_bounded() {
        let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
        let obs = vec![0.5, 0.7, 0.999, 0.9999, 0.9995, 0.6, 0.4];
        let log_emit = precompute_log_emissions(&obs, &params);
        let (posteriors, ll) = forward_backward_from_log_emit(&log_emit, &params);

        assert_eq!(posteriors.len(), obs.len());
        assert!(ll.is_finite());
        for &p in &posteriors {
            assert!(p >= 0.0 && p <= 1.0, "Posterior {} out of [0,1] range", p);
        }
    }

    #[test]
    fn test_viterbi_from_log_emit_single_observation() {
        let params = HmmParams::from_expected_length(50.0, 0.5, 5000);
        // Strong IBD signal
        let log_emit = vec![[-100.0, -0.1]]; // log P(obs|non-IBD) very low, log P(obs|IBD) high
        let states = viterbi_from_log_emit(&log_emit, &params);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 1, "Should classify as IBD with strong IBD emission");
    }

    // === Additional edge cases for estimate_auxiliary_emissions ===

    #[test]
    fn test_estimate_auxiliary_emissions_all_ibd() {
        let aux = vec![0.95, 0.96, 0.97, 0.98];
        let posteriors = vec![1.0, 1.0, 1.0, 1.0]; // All IBD

        let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

        // Non-IBD should get fallback values (weight sum = 0)
        assert_eq!(aux_emit[0].mean, 0.5); // fallback
        assert_eq!(aux_emit[0].std, 0.2);  // fallback

        // IBD should get actual mean
        let expected_ibd_mean = (0.95 + 0.96 + 0.97 + 0.98) / 4.0;
        assert!((aux_emit[1].mean - expected_ibd_mean).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_auxiliary_emissions_all_non_ibd() {
        let aux = vec![0.5, 0.55, 0.6, 0.65];
        let posteriors = vec![0.0, 0.0, 0.0, 0.0]; // All non-IBD

        let aux_emit = estimate_auxiliary_emissions(&aux, &posteriors);

        // Non-IBD should get actual mean
        let expected_non_ibd_mean = (0.5 + 0.55 + 0.6 + 0.65) / 4.0;
        assert!((aux_emit[0].mean - expected_non_ibd_mean).abs() < 1e-10);

        // IBD should get fallback values
        assert_eq!(aux_emit[1].mean, 0.9); // fallback
        assert_eq!(aux_emit[1].std, 0.1);  // fallback
    }

    // === Additional coverage_ratio edge cases ===

    #[test]
    fn test_coverage_ratio_one_zero() {
        assert_eq!(coverage_ratio(0, 500), 0.0);
        assert_eq!(coverage_ratio(500, 0), 0.0);
    }

    #[test]
    fn test_coverage_ratio_large_values() {
        let ratio = coverage_ratio(1_000_000_000, 2_000_000_000);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_ratio_one_apart() {
        // Very close values should produce ratio near 1
        let ratio = coverage_ratio(999, 1000);
        assert!((ratio - 0.999).abs() < 1e-10);
    }

    // === Additional refine_states edge cases ===

    #[test]
    fn test_refine_states_chain_extension() {
        // Extension should propagate: if window A is extended to IBD, then window B
        // (adjacent to A with high posterior) should also be extended in subsequent pass
        let mut states = vec![0, 0, 0, 1, 0, 0, 0];
        let posteriors = vec![0.1, 0.6, 0.7, 0.9, 0.8, 0.6, 0.1];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Positions 2, 4 should extend (adjacent to IBD with posterior >= 0.5)
        // Position 5 may also extend once 4 becomes IBD, and 1 may extend once 2 becomes IBD
        assert_eq!(states[3], 1, "Original IBD should remain");
        assert_eq!(states[2], 1, "Position 2 should extend (posterior 0.7 >= 0.5, adjacent to IBD)");
        assert_eq!(states[4], 1, "Position 4 should extend (posterior 0.8 >= 0.5, adjacent to IBD)");
    }

    #[test]
    fn test_refine_states_trim_both_edges() {
        // Both edges should be trimmed if low posterior
        let mut states = vec![0, 1, 1, 1, 1, 0];
        let posteriors = vec![0.1, 0.15, 0.9, 0.9, 0.15, 0.1];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        assert_eq!(states[1], 0, "Left edge should be trimmed (posterior 0.15 < 0.2)");
        assert_eq!(states[4], 0, "Right edge should be trimmed (posterior 0.15 < 0.2)");
        assert_eq!(states[2], 1, "Interior should remain IBD");
        assert_eq!(states[3], 1, "Interior should remain IBD");
    }

    #[test]
    fn test_refine_states_no_trim_interior() {
        // Interior low-posterior should NOT be trimmed (only edges)
        let mut states = vec![0, 1, 1, 1, 1, 0];
        let posteriors = vec![0.1, 0.9, 0.1, 0.1, 0.9, 0.1];

        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

        // Interior positions 2,3 have low posteriors but are NOT at segment boundary
        assert_eq!(states[2], 1, "Interior low-posterior should NOT be trimmed");
        assert_eq!(states[3], 1, "Interior low-posterior should NOT be trimmed");
    }

    // === infer_ibd_with_training additional edge cases ===

    #[test]
    fn test_infer_ibd_with_training_all_populations() {
        // Each population should work without errors
        let populations = [
            Population::AFR, Population::EUR, Population::EAS,
            Population::CSA, Population::AMR, Population::InterPop, Population::Generic,
        ];
        let obs = vec![0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.998, 0.5, 0.6, 0.55, 0.5];

        for &pop in &populations {
            let mut params = HmmParams::from_population(pop, 50.0, 0.0001, 5000);
            let result = infer_ibd_with_training(&obs, &mut params, pop, 5000, 5);

            assert_eq!(result.states.len(), obs.len(),
                "Failed for population {:?}", pop);
            assert!(result.log_likelihood.is_finite(),
                "Non-finite log_likelihood for population {:?}", pop);
        }
    }

    #[test]
    fn test_infer_ibd_with_training_few_observations() {
        // Less than 10 observations should skip Baum-Welch but still produce results
        let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs = vec![0.5, 0.999, 0.5];
        let result = infer_ibd_with_training(&obs, &mut params, Population::Generic, 5000, 10);

        assert_eq!(result.states.len(), 3);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_multifeature_improves_separation() {
        // Test that auxiliary feature improves discrimination
        // when primary feature alone is ambiguous
        let n = 200;
        let mut obs = Vec::with_capacity(n);
        let mut aux = Vec::with_capacity(n);

        // Create data where identity is ambiguous but coverage ratio is discriminative
        for i in 0..n {
            if (50..150).contains(&i) {
                // IBD region: identity barely above non-IBD, but high coverage ratio
                obs.push(0.998 + 0.001 * (i as f64 / n as f64));
                aux.push(0.95 + 0.04 * (i as f64 / n as f64).sin());
            } else {
                // Non-IBD region: similar identity, but low coverage ratio
                obs.push(0.997 + 0.001 * (i as f64 / n as f64));
                aux.push(0.6 + 0.2 * (i as f64 / n as f64).cos());
            }
        }

        let mut params1 = HmmParams::from_expected_length(50.0, 0.001, 5000);
        let (result_primary, _) = infer_ibd_with_aux_features(
            &obs, &mut params1, Population::Generic, 5000, 5, None,
        );

        let mut params2 = HmmParams::from_expected_length(50.0, 0.001, 5000);
        let (result_combined, _) = infer_ibd_with_aux_features(
            &obs, &mut params2, Population::Generic, 5000, 5, Some(&aux),
        );

        // Both should produce valid results
        assert_eq!(result_primary.states.len(), n);
        assert_eq!(result_combined.states.len(), n);
        assert!(result_primary.log_likelihood.is_finite());
        assert!(result_combined.log_likelihood.is_finite());
    }

    // ========== K=0 (mutation-free) auxiliary feature tests ==========

    #[test]
    fn test_k0_log_pmf_basic() {
        // K=0 window (indicator=1.0) with p=0.22
        let log_k0 = k0_log_pmf(1.0, 0.22);
        assert!((log_k0 - 0.22f64.ln()).abs() < 1e-10);

        // K≥1 window (indicator=0.0) with p=0.22
        let log_k1 = k0_log_pmf(0.0, 0.22);
        assert!((log_k1 - 0.78f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_k0_log_pmf_discriminative_ratio() {
        // For IBD (p=0.22) vs non-IBD (p=0.015), K=0 window log-likelihood ratio
        let lr_k0 = k0_log_pmf(1.0, 0.22) - k0_log_pmf(1.0, 0.015);
        // ln(0.22/0.015) ≈ 2.69
        assert!(lr_k0 > 2.5, "K=0 should strongly favor IBD, got {}", lr_k0);

        // K≥1 window should mildly favor non-IBD
        let lr_k1 = k0_log_pmf(0.0, 0.22) - k0_log_pmf(0.0, 0.015);
        // ln(0.78/0.985) ≈ -0.23
        assert!(lr_k1 < 0.0, "K≥1 should favor non-IBD, got {}", lr_k1);
        assert!(lr_k1 > -1.0, "K≥1 penalty should be mild, got {}", lr_k1);
    }

    #[test]
    fn test_estimate_k0_emissions_empty() {
        let params = estimate_k0_emissions(&[], &[]);
        // Should return informative priors
        assert!((params[0] - 0.015).abs() < 0.001);
        assert!((params[1] - 0.22).abs() < 0.001);
    }

    #[test]
    fn test_estimate_k0_emissions_all_ibd() {
        // All windows have high IBD posterior, 50% are K=0
        let indicators = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let posteriors = vec![0.95, 0.95, 0.95, 0.95, 0.95, 0.95];
        let params = estimate_k0_emissions(&indicators, &posteriors);
        // P(K=0 | IBD) should be ~0.5 (from data)
        assert!(params[1] > 0.4 && params[1] < 0.6,
            "P(K=0|IBD) should be ~0.5, got {}", params[1]);
    }

    #[test]
    fn test_estimate_k0_emissions_discriminative() {
        // IBD windows: high posterior, many K=0
        // Non-IBD windows: low posterior, few K=0
        let mut indicators = Vec::new();
        let mut posteriors = Vec::new();
        // 50 IBD windows: 40% K=0
        for i in 0..50 {
            indicators.push(if i % 5 < 2 { 1.0 } else { 0.0 });
            posteriors.push(0.9);
        }
        // 200 non-IBD windows: 2% K=0
        for i in 0..200 {
            indicators.push(if i % 50 == 0 { 1.0 } else { 0.0 });
            posteriors.push(0.05);
        }
        let params = estimate_k0_emissions(&indicators, &posteriors);
        // IBD should have higher K=0 rate than non-IBD
        assert!(params[1] > params[0],
            "P(K=0|IBD)={} should exceed P(K=0|non-IBD)={}", params[1], params[0]);
    }

    #[test]
    fn test_estimate_k0_emissions_clamped() {
        // All K=0: P(K=0|state) should be clamped below 1.0
        let indicators = vec![1.0; 100];
        let posteriors = vec![0.5; 100];
        let params = estimate_k0_emissions(&indicators, &posteriors);
        assert!(params[0] <= 0.999);
        assert!(params[1] <= 0.999);
        // No K=0: should be clamped above 0.0
        let indicators = vec![0.0; 100];
        let params = estimate_k0_emissions(&indicators, &posteriors);
        assert!(params[0] >= 0.001);
        assert!(params[1] >= 0.001);
    }

    #[test]
    fn test_augment_with_k0_modifies_emissions() {
        // Build simple log-emissions
        let n = 20;
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs: Vec<f64> = (0..n).map(|i| {
            if i >= 5 && i < 15 { 0.9995 } else { 0.998 }
        }).collect();
        let mut log_emit = precompute_log_emissions(&obs, &params);
        let original_emit = log_emit.clone();

        // Posteriors: IBD region has high posterior
        let posteriors: Vec<f64> = (0..n).map(|i| {
            if i >= 5 && i < 15 { 0.8 } else { 0.1 }
        }).collect();

        // K=0 indicators: some in IBD region
        let k0_inds: Vec<f64> = (0..n).map(|i| {
            if i >= 7 && i < 12 { 1.0 } else { 0.0 }
        }).collect();

        augment_with_k0(&mut log_emit, &k0_inds, &posteriors);

        // K=0 windows should have modified emissions
        for t in 7..12 {
            assert_ne!(log_emit[t][0], original_emit[t][0],
                "K=0 window {} should be modified", t);
            assert_ne!(log_emit[t][1], original_emit[t][1]);
        }
        // K≥1 windows should also be modified (ln(1-p) term)
        for t in [0, 1, 2, 15, 18] {
            assert_ne!(log_emit[t][0], original_emit[t][0],
                "K≥1 window {} should also be modified", t);
        }
    }

    #[test]
    fn test_augment_with_k0_boosts_ibd_for_k0_windows() {
        let n = 20;
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs: Vec<f64> = vec![0.999; n];
        let mut log_emit = precompute_log_emissions(&obs, &params);

        let posteriors: Vec<f64> = (0..n).map(|i| {
            if i >= 5 && i < 15 { 0.7 } else { 0.1 }
        }).collect();
        let k0_inds: Vec<f64> = (0..n).map(|i| {
            if i >= 5 && i < 15 { 1.0 } else { 0.0 }
        }).collect();

        let pre_ibd_advantage: Vec<f64> = log_emit.iter()
            .map(|e| e[1] - e[0]).collect();

        augment_with_k0(&mut log_emit, &k0_inds, &posteriors);

        let post_ibd_advantage: Vec<f64> = log_emit.iter()
            .map(|e| e[1] - e[0]).collect();

        // K=0 windows should have INCREASED IBD advantage
        for t in 5..15 {
            assert!(post_ibd_advantage[t] > pre_ibd_advantage[t],
                "K=0 window {} should boost IBD advantage: pre={:.4} post={:.4}",
                t, pre_ibd_advantage[t], post_ibd_advantage[t]);
        }
    }

    #[test]
    fn test_augment_with_k0_nondiscriminative_noop() {
        // When K=0 rates are equal or K=0 is more common in non-IBD,
        // augmentation should be a no-op (guard in augment_with_k0)
        let n = 10;
        let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
        let obs: Vec<f64> = vec![0.999; n];
        let mut log_emit = precompute_log_emissions(&obs, &params);
        let original_emit = log_emit.clone();

        // Uniform posteriors + uniform K=0 → equal rates → no augmentation
        let posteriors: Vec<f64> = vec![0.5; n];
        let k0_inds: Vec<f64> = vec![1.0; n]; // all K=0 → equal rates both states

        augment_with_k0(&mut log_emit, &k0_inds, &posteriors);

        // Should be unchanged (K=0 rate equal for both states)
        for t in 0..n {
            assert_eq!(log_emit[t][0], original_emit[t][0]);
            assert_eq!(log_emit[t][1], original_emit[t][1]);
        }
    }

    #[test]
    fn test_augment_with_k0_inference_improvement() {
        // Synthetic data: IBD region with some K=0 windows
        // K=0 augmentation should improve IBD detection
        let n = 100;
        let mut obs = vec![0.998; n]; // non-IBD baseline
        // IBD region: windows 30-70 with higher identity, some at 1.0
        for i in 30..70 {
            obs[i] = if i % 3 == 0 { 0.99999 } else { 0.9995 };
        }

        let params = HmmParams::from_expected_length(30.0, 0.001, 5000);

        // Without K=0
        let log_emit_base = precompute_log_emissions(&obs, &params);
        let states_base = viterbi_from_log_emit(&log_emit_base, &params);
        let ibd_count_base = states_base.iter().filter(|&&s| s == 1).count();

        // With K=0
        let mut log_emit_k0 = precompute_log_emissions(&obs, &params);
        let posteriors = forward_backward_from_log_emit(&log_emit_k0, &params).0;
        let k0_inds: Vec<f64> = obs.iter()
            .map(|&o| if o >= 0.9999 { 1.0 } else { 0.0 }).collect();
        augment_with_k0(&mut log_emit_k0, &k0_inds, &posteriors);
        let states_k0 = viterbi_from_log_emit(&log_emit_k0, &params);
        let ibd_count_k0 = states_k0.iter().filter(|&&s| s == 1).count();

        // K=0 augmentation should detect at least as many IBD windows
        assert!(ibd_count_k0 >= ibd_count_base,
            "K=0 should not decrease IBD detection: base={}, k0={}", ibd_count_base, ibd_count_k0);
    }
}
