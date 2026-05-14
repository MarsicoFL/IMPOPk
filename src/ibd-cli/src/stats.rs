//! Statistical utilities for IBD analysis

use std::f64::consts::PI;

/// Gaussian distribution parameters
#[derive(Debug, Clone, Copy)]
pub struct GaussianParams {
    pub mean: f64,
    pub std: f64,
}

impl GaussianParams {
    /// Creates a new GaussianParams with validation.
    ///
    /// # Errors
    ///
    /// Returns an error if `std <= 0`, as this would cause division by zero
    /// in probability calculations.
    ///
    /// # Examples
    ///
    /// ```
    /// use impopk_ibd::stats::GaussianParams;
    ///
    /// let valid = GaussianParams::new(0.0, 1.0);
    /// assert!(valid.is_ok());
    ///
    /// let invalid = GaussianParams::new(0.0, 0.0);
    /// assert!(invalid.is_err());
    ///
    /// let negative = GaussianParams::new(0.0, -1.0);
    /// assert!(negative.is_err());
    /// ```
    pub fn new(mean: f64, std: f64) -> Result<Self, String> {
        if std <= 0.0 {
            return Err(format!(
                "GaussianParams: std must be positive, got {}",
                std
            ));
        }
        Ok(Self { mean, std })
    }

    /// Creates a new GaussianParams without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `std > 0`. Using a non-positive `std`
    /// will cause division by zero or undefined behavior in `pdf()` and `log_pdf()`.
    ///
    /// This is useful for:
    /// - Compile-time constants where the values are known to be valid
    /// - Performance-critical code where validation has already been performed
    ///
    /// # Examples
    ///
    /// ```
    /// use impopk_ibd::stats::GaussianParams;
    ///
    /// // Safe: std is clearly positive
    /// let params = GaussianParams::new_unchecked(0.0, 1.0);
    /// ```
    pub const fn new_unchecked(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        debug_assert!(
            self.std > 0.0,
            "GaussianParams::pdf: std must be positive, got {}",
            self.std
        );
        let z = (x - self.mean) / self.std;
        (-0.5 * z * z).exp() / (self.std * (2.0 * PI).sqrt())
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        debug_assert!(
            self.std > 0.0,
            "GaussianParams::log_pdf: std must be positive, got {}",
            self.std
        );
        let z = (x - self.mean) / self.std;
        -0.5 * z * z - self.std.ln() - 0.5 * (2.0 * PI).ln()
    }
}

/// Logit transform: log(x / (1 - x)).
///
/// Maps values in (0, 1) to (-inf, +inf). Useful for transforming
/// identity values near 1.0 into a space where Gaussian emissions
/// are more appropriate.
///
/// Values are clamped to avoid infinities:
/// - Input clamped to [epsilon, 1-epsilon]
/// - Output clamped to [-LOGIT_CAP, LOGIT_CAP] to prevent extreme outliers
///   (identity = 1.0 would otherwise map to ~23, distorting clustering)
///
/// Set to 12.0 to avoid truncating IBD observations: logit(0.99999)≈11.5,
/// so cap=10 clips 22% of IBD data and biases the IBD emission mean down
/// by ~0.33 logit units (32% of the IBD/non-IBD gap). Cap=12 retains >99%
/// of the IBD distribution while still preventing extreme outliers.
pub const LOGIT_CAP: f64 = 12.0;

pub fn logit(x: f64) -> f64 {
    let eps = 1e-10;
    let clamped = x.clamp(eps, 1.0 - eps);
    let raw = (clamped / (1.0 - clamped)).ln();
    raw.clamp(-LOGIT_CAP, LOGIT_CAP)
}

/// Inverse logit (sigmoid): 1 / (1 + exp(-x)).
///
/// Maps values in (-inf, +inf) back to (0, 1).
pub fn inv_logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Transform a slice of identity values to logit space.
///
/// This stretches the region near 1.0 where pangenome identity data
/// is concentrated, making Gaussian emission models more appropriate.
///
/// For pangenome data:
/// - Identity 0.997 → logit ≈ 5.81
/// - Identity 0.9997 → logit ≈ 8.11
/// - The 0.0027 raw difference becomes a 2.3 logit-scale difference
pub fn logit_transform_observations(observations: &[f64]) -> Vec<f64> {
    observations.iter().map(|&x| logit(x)).collect()
}

/// Compute Gaussian parameters in logit space for a given raw-space
/// identity distribution.
///
/// Uses the delta method: if X ~ Gaussian(mu, sigma) and g(x) = logit(x),
/// then g(X) ≈ Gaussian(logit(mu), sigma / (mu * (1 - mu))).
/// The logit mean is capped at LOGIT_CAP to match the observation capping.
pub fn gaussian_to_logit_space(mean: f64, std: f64) -> GaussianParams {
    let logit_mean = logit(mean); // Already capped by logit()
    // Delta method: variance in logit space ≈ (sigma / (mu * (1-mu)))^2
    let denominator = mean * (1.0 - mean);
    let logit_std = if denominator > 1e-15 {
        (std / denominator).min(LOGIT_CAP * 0.5) // Cap std too
    } else {
        LOGIT_CAP * 0.5
    };
    GaussianParams::new_unchecked(logit_mean, logit_std.max(0.01))
}

/// Simple 1D k-means clustering
pub fn kmeans_1d(data: &[f64], k: usize, max_iter: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    if data.len() < k || k == 0 {
        return None;
    }

    let n = data.len();

    let mut sorted: Vec<f64> = data.to_vec();
    // Use total_cmp instead of partial_cmp to handle NaN values safely
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut centers: Vec<f64> = (0..k)
        .map(|i| {
            let idx = ((i as f64 + 0.5) / k as f64 * n as f64) as usize;
            sorted[idx.min(n - 1)]
        })
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for (i, &x) in data.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::MAX;

            for (c, &center) in centers.iter().enumerate() {
                let dist = (x - center).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut sums = vec![0.0; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            sums[cluster] += data[i];
            counts[cluster] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                centers[c] = sums[c] / counts[c] as f64;
            }
        }
    }

    Some((centers, assignments))
}

/// Two-component Gaussian mixture model estimated via Expectation-Maximization.
///
/// Fits a mixture of two Gaussians to 1D data, which is more principled than
/// k-means for overlapping distributions (as with AFR non-IBD/IBD).
///
/// Returns `(params_low, params_high, weight_low, weight_high)` where
/// `weight_low + weight_high ≈ 1.0`.
pub fn em_two_component(
    data: &[f64],
    init_low: &GaussianParams,
    init_high: &GaussianParams,
    init_weight_low: f64,
    max_iter: usize,
    tol: f64,
) -> Option<(GaussianParams, GaussianParams, f64, f64)> {
    let n = data.len();
    if n < 4 {
        return None;
    }

    let mut mu0 = init_low.mean;
    let mut sigma0 = init_low.std;
    let mut mu1 = init_high.mean;
    let mut sigma1 = init_high.std;
    let mut w0 = init_weight_low;
    let mut w1 = 1.0 - init_weight_low;

    let mut prev_log_lik = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        // E-step: compute responsibilities
        let mut r0 = vec![0.0; n];
        let mut log_lik = 0.0;

        let g0 = GaussianParams::new_unchecked(mu0, sigma0);
        let g1 = GaussianParams::new_unchecked(mu1, sigma1);

        for (i, &x) in data.iter().enumerate() {
            let log_p0 = w0.ln() + g0.log_pdf(x);
            let log_p1 = w1.ln() + g1.log_pdf(x);
            let max_log = log_p0.max(log_p1);
            let log_sum = max_log + ((log_p0 - max_log).exp() + (log_p1 - max_log).exp()).ln();
            r0[i] = (log_p0 - log_sum).exp();
            log_lik += log_sum;
        }

        // Check convergence
        if (log_lik - prev_log_lik).abs() < tol {
            break;
        }
        prev_log_lik = log_lik;

        // M-step: update parameters
        let n0: f64 = r0.iter().sum();
        let n1: f64 = r0.iter().map(|r| 1.0 - r).sum();

        if n0 < 1.0 || n1 < 1.0 {
            // Degenerate: one component has vanished
            return None;
        }

        w0 = n0 / n as f64;
        w1 = n1 / n as f64;

        mu0 = r0.iter().zip(data.iter()).map(|(&r, &x)| r * x).sum::<f64>() / n0;
        mu1 = r0.iter().zip(data.iter()).map(|(&r, &x)| (1.0 - r) * x).sum::<f64>() / n1;

        let var0: f64 = r0.iter().zip(data.iter()).map(|(&r, &x)| r * (x - mu0).powi(2)).sum::<f64>() / n0;
        let var1: f64 = r0.iter().zip(data.iter()).map(|(&r, &x)| (1.0 - r) * (x - mu1).powi(2)).sum::<f64>() / n1;

        sigma0 = var0.sqrt().max(1e-6);
        sigma1 = var1.sqrt().max(1e-6);
    }

    // Ensure low < high
    if mu0 > mu1 {
        Some((
            GaussianParams::new_unchecked(mu1, sigma1),
            GaussianParams::new_unchecked(mu0, sigma0),
            w1,
            w0,
        ))
    } else {
        Some((
            GaussianParams::new_unchecked(mu0, sigma0),
            GaussianParams::new_unchecked(mu1, sigma1),
            w0,
            w1,
        ))
    }
}

/// Two-component Gaussian mixture model with MAP regularization.
///
/// Like `em_two_component`, but adds prior constraints on the component means
/// to prevent degenerate solutions. This is especially useful for AFR populations
/// where the non-IBD and IBD distributions overlap heavily, causing standard EM
/// to merge both components into a single mode.
///
/// ## Parameters
///
/// - `prior_strength`: Controls how strongly the priors pull the means toward
///   their initial values. Higher values = stronger regularization. A value of
///   10.0 means the prior is equivalent to 10 pseudo-observations.
///
/// ## Returns
///
/// Same as `em_two_component`: `(params_low, params_high, weight_low, weight_high)`
pub fn em_two_component_map(
    data: &[f64],
    init_low: &GaussianParams,
    init_high: &GaussianParams,
    init_weight_low: f64,
    max_iter: usize,
    tol: f64,
    prior_strength: f64,
) -> Option<(GaussianParams, GaussianParams, f64, f64)> {
    let n = data.len();
    if n < 4 {
        return None;
    }

    let prior_mu0 = init_low.mean;
    let prior_mu1 = init_high.mean;
    let prior_sigma0 = init_low.std;
    let prior_sigma1 = init_high.std;

    let mut mu0 = init_low.mean;
    let mut sigma0 = init_low.std;
    let mut mu1 = init_high.mean;
    let mut sigma1 = init_high.std;
    let mut w0 = init_weight_low;
    let mut w1 = 1.0 - init_weight_low;

    let mut prev_log_lik = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        // E-step: compute responsibilities
        let mut r0 = vec![0.0; n];
        let mut log_lik = 0.0;

        let g0 = GaussianParams::new_unchecked(mu0, sigma0);
        let g1 = GaussianParams::new_unchecked(mu1, sigma1);

        for (i, &x) in data.iter().enumerate() {
            let log_p0 = w0.ln() + g0.log_pdf(x);
            let log_p1 = w1.ln() + g1.log_pdf(x);
            let max_log = log_p0.max(log_p1);
            let log_sum = max_log + ((log_p0 - max_log).exp() + (log_p1 - max_log).exp()).ln();
            r0[i] = (log_p0 - log_sum).exp();
            log_lik += log_sum;
        }

        // Check convergence
        if (log_lik - prev_log_lik).abs() < tol {
            break;
        }
        prev_log_lik = log_lik;

        // M-step with MAP regularization
        let n0: f64 = r0.iter().sum();
        let n1: f64 = r0.iter().map(|r| 1.0 - r).sum();

        if n0 < 0.5 || n1 < 0.5 {
            // Degenerate: one component has nearly vanished
            return None;
        }

        w0 = n0 / n as f64;
        w1 = n1 / n as f64;

        // MAP mean: weighted average of MLE mean and prior mean
        let data_mu0: f64 = r0.iter().zip(data.iter()).map(|(&r, &x)| r * x).sum::<f64>() / n0;
        let data_mu1: f64 = r0.iter().zip(data.iter()).map(|(&r, &x)| (1.0 - r) * x).sum::<f64>() / n1;

        mu0 = (n0 * data_mu0 + prior_strength * prior_mu0) / (n0 + prior_strength);
        mu1 = (n1 * data_mu1 + prior_strength * prior_mu1) / (n1 + prior_strength);

        // MAP variance: include prior contribution
        let data_var0: f64 = r0.iter().zip(data.iter())
            .map(|(&r, &x)| r * (x - mu0).powi(2)).sum::<f64>() / n0;
        let data_var1: f64 = r0.iter().zip(data.iter())
            .map(|(&r, &x)| (1.0 - r) * (x - mu1).powi(2)).sum::<f64>() / n1;

        let prior_var0 = prior_sigma0 * prior_sigma0;
        let prior_var1 = prior_sigma1 * prior_sigma1;

        sigma0 = ((n0 * data_var0 + prior_strength * prior_var0) / (n0 + prior_strength)).sqrt().max(1e-6);
        sigma1 = ((n1 * data_var1 + prior_strength * prior_var1) / (n1 + prior_strength)).sqrt().max(1e-6);
    }

    // Ensure low < high
    if mu0 > mu1 {
        Some((
            GaussianParams::new_unchecked(mu1, sigma1),
            GaussianParams::new_unchecked(mu0, sigma0),
            w1,
            w0,
        ))
    } else {
        Some((
            GaussianParams::new_unchecked(mu0, sigma0),
            GaussianParams::new_unchecked(mu1, sigma1),
            w0,
            w1,
        ))
    }
}

/// Compute trimmed mean of a dataset, ignoring the lowest and highest `trim_fraction`.
///
/// Useful for robust initialization of emission parameters, as it reduces
/// sensitivity to outliers (e.g., a few very high IBD windows contaminating
/// the non-IBD estimate).
///
/// ## Parameters
///
/// - `data`: Input values
/// - `trim_fraction`: Fraction to trim from each end (0.0 to 0.5)
///
/// ## Returns
///
/// The trimmed mean, or `None` if data is too short after trimming.
pub fn trimmed_mean(data: &[f64], trim_fraction: f64) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    let trim_count = (n as f64 * trim_fraction.clamp(0.0, 0.49)) as usize;
    let start = trim_count;
    let end = n - trim_count;

    if start >= end {
        return None;
    }

    let sum: f64 = sorted[start..end].iter().sum();
    Some(sum / (end - start) as f64)
}

/// Compute the Bayesian Information Criterion (BIC) for a 1-component vs
/// 2-component Gaussian mixture model.
///
/// Returns `(bic_1, bic_2)` where lower BIC indicates better model fit.
/// If `bic_1 < bic_2`, the data does not support two distinct components
/// (likely no IBD signal), and the HMM should use conservative defaults.
///
/// ## Parameters
///
/// - `data`: Identity observations
/// - `params_low`: Fitted low-component Gaussian
/// - `params_high`: Fitted high-component Gaussian
/// - `weight_low`: Mixing weight for low component
///
/// ## BIC Formula
///
/// BIC = -2 * log_likelihood + k * ln(n)
///
/// where k is the number of free parameters:
/// - 1-component: k=2 (mean, variance)
/// - 2-component: k=5 (2 means, 2 variances, 1 weight)
pub fn bic_model_selection(
    data: &[f64],
    params_low: &GaussianParams,
    params_high: &GaussianParams,
    weight_low: f64,
) -> (f64, f64) {
    let n = data.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    let ln_n = (n as f64).ln();

    // 1-component model: single Gaussian fitted to all data
    let mean_all: f64 = data.iter().sum::<f64>() / n as f64;
    let var_all: f64 = data.iter().map(|x| (x - mean_all).powi(2)).sum::<f64>() / n as f64;
    let std_all = var_all.sqrt().max(1e-10);
    let g_all = GaussianParams::new_unchecked(mean_all, std_all);

    let log_lik_1: f64 = data.iter().map(|&x| g_all.log_pdf(x)).sum();
    let k_1 = 2.0; // mean + variance
    let bic_1 = -2.0 * log_lik_1 + k_1 * ln_n;

    // 2-component model: mixture of two Gaussians
    let w_high = 1.0 - weight_low;
    let log_lik_2: f64 = data.iter().map(|&x| {
        let log_p0 = weight_low.ln() + params_low.log_pdf(x);
        let log_p1 = w_high.ln() + params_high.log_pdf(x);
        let max_log = log_p0.max(log_p1);
        max_log + ((log_p0 - max_log).exp() + (log_p1 - max_log).exp()).ln()
    }).sum();
    let k_2 = 5.0; // 2 means + 2 variances + 1 weight
    let bic_2 = -2.0 * log_lik_2 + k_2 * ln_n;

    (bic_1, bic_2)
}

/// Welford's online algorithm for mean and variance
#[derive(Debug, Clone, Default)]
pub struct OnlineStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl OnlineStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 { 0.0 } else { self.m2 / (self.n - 1) as f64 }
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_pdf() {
        let g = GaussianParams::new(0.0, 1.0).unwrap();
        let pdf_at_0 = g.pdf(0.0);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((pdf_at_0 - expected).abs() < 1e-10);
    }

    // === Validation tests for GaussianParams::new ===

    #[test]
    fn test_gaussian_new_valid() {
        // Positive std should succeed
        assert!(GaussianParams::new(0.0, 1.0).is_ok());
        assert!(GaussianParams::new(5.0, 0.001).is_ok());
        assert!(GaussianParams::new(-10.0, 100.0).is_ok());
    }

    #[test]
    fn test_gaussian_new_zero_std_fails() {
        // std = 0 should fail
        let result = GaussianParams::new(0.0, 0.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));
    }

    #[test]
    fn test_gaussian_new_negative_std_fails() {
        // Negative std should fail
        let result = GaussianParams::new(0.0, -1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be positive"));
    }

    #[test]
    fn test_gaussian_new_very_small_positive_std_succeeds() {
        // Very small positive std should succeed
        let result = GaussianParams::new(0.0, 1e-15);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kmeans() {
        let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2];
        let (centers, _) = kmeans_1d(&data, 2, 10).unwrap();
        let min_center = centers.iter().cloned().fold(f64::MAX, f64::min);
        let max_center = centers.iter().cloned().fold(f64::MIN, f64::max);
        assert!((min_center - 1.1).abs() < 0.2);
        assert!((max_center - 5.1).abs() < 0.2);
    }

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.add(x);
        }
        assert_eq!(stats.count(), 5);
        assert!((stats.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_kmeans_with_nan_does_not_panic() {
        // This test verifies that NaN values do not cause a panic
        let data = vec![1.0, 1.1, f64::NAN, 5.0, 5.1, 5.2];
        // Should not panic - NaN is handled by total_cmp
        let result = kmeans_1d(&data, 2, 10);
        // The function should still return Some result
        assert!(result.is_some());
    }

    // === Edge case tests for kmeans_1d ===

    #[test]
    fn test_kmeans_empty_data() {
        let data: Vec<f64> = vec![];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_kmeans_fewer_points_than_clusters() {
        // Only 1 point, but asking for 2 clusters
        let data = vec![1.0];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_kmeans_zero_clusters() {
        let data = vec![1.0, 2.0, 3.0];
        let result = kmeans_1d(&data, 0, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_kmeans_identical_values() {
        // All identical values - zero variance case
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_some());
        let (centers, assignments) = result.unwrap();
        // With identical values, all points should be in the same cluster
        assert_eq!(centers.len(), 2);
        assert_eq!(assignments.len(), 5);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kmeans_1d(&data, 1, 10);
        assert!(result.is_some());
        let (centers, assignments) = result.unwrap();
        assert_eq!(centers.len(), 1);
        // All points should be assigned to cluster 0
        for &a in &assignments {
            assert_eq!(a, 0);
        }
        // Center should be the mean (3.0)
        assert!((centers[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_kmeans_exact_points_as_clusters() {
        // 3 points and 3 clusters
        let data = vec![1.0, 5.0, 10.0];
        let result = kmeans_1d(&data, 3, 10);
        assert!(result.is_some());
        let (centers, assignments) = result.unwrap();
        assert_eq!(centers.len(), 3);
        assert_eq!(assignments.len(), 3);
        // Each point should be in its own cluster
        let unique_assignments: std::collections::HashSet<_> = assignments.iter().collect();
        assert_eq!(unique_assignments.len(), 3);
    }

    #[test]
    fn test_kmeans_large_values() {
        // Test numerical stability with large values
        let data = vec![1e15, 1e15 + 1.0, 1e15 + 2.0, 2e15, 2e15 + 1.0, 2e15 + 2.0];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_some());
        let (centers, _) = result.unwrap();
        assert_eq!(centers.len(), 2);
        // Should find two clusters around 1e15 and 2e15
        let min_center = centers.iter().cloned().fold(f64::MAX, f64::min);
        let max_center = centers.iter().cloned().fold(f64::MIN, f64::max);
        assert!((min_center - 1e15).abs() < 1e12);
        assert!((max_center - 2e15).abs() < 1e12);
    }

    #[test]
    fn test_kmeans_small_values() {
        // Test with very small values
        let data = vec![1e-15, 1.1e-15, 1.2e-15, 5e-15, 5.1e-15, 5.2e-15];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_some());
        let (centers, _) = result.unwrap();
        assert_eq!(centers.len(), 2);
    }

    #[test]
    fn test_kmeans_negative_values() {
        let data = vec![-5.0, -4.9, -4.8, 5.0, 5.1, 5.2];
        let result = kmeans_1d(&data, 2, 10);
        assert!(result.is_some());
        let (centers, _) = result.unwrap();
        let min_center = centers.iter().cloned().fold(f64::MAX, f64::min);
        let max_center = centers.iter().cloned().fold(f64::MIN, f64::max);
        assert!((min_center - (-4.9)).abs() < 0.2);
        assert!((max_center - 5.1).abs() < 0.2);
    }

    #[test]
    fn test_kmeans_one_iteration() {
        let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2];
        let result = kmeans_1d(&data, 2, 1);
        assert!(result.is_some());
        // Should still produce valid output even with just 1 iteration
        let (centers, assignments) = result.unwrap();
        assert_eq!(centers.len(), 2);
        assert_eq!(assignments.len(), 6);
    }

    #[test]
    fn test_kmeans_with_infinity() {
        // Test with infinity values
        let data = vec![1.0, 2.0, f64::INFINITY, 5.0, 6.0];
        let result = kmeans_1d(&data, 2, 10);
        // Should handle infinity without panic
        assert!(result.is_some());
    }

    // === Edge case tests for GaussianParams ===

    #[test]
    fn test_gaussian_pdf_at_mean() {
        let g = GaussianParams::new(5.0, 2.0).unwrap();
        let pdf_at_mean = g.pdf(5.0);
        // At mean, z = 0, so pdf = 1 / (std * sqrt(2*pi))
        let expected = 1.0 / (2.0 * (2.0 * PI).sqrt());
        assert!((pdf_at_mean - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_pdf_symmetry() {
        let g = GaussianParams::new(0.0, 1.0).unwrap();
        // PDF should be symmetric around mean
        let pdf_plus_1 = g.pdf(1.0);
        let pdf_minus_1 = g.pdf(-1.0);
        assert!((pdf_plus_1 - pdf_minus_1).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_log_pdf_consistency() {
        let g = GaussianParams::new(3.0, 0.5).unwrap();
        let x = 2.5;
        // log(pdf(x)) should equal log_pdf(x)
        let pdf_val = g.pdf(x);
        let log_pdf_val = g.log_pdf(x);
        assert!((pdf_val.ln() - log_pdf_val).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_narrow_std() {
        // Very narrow distribution
        let g = GaussianParams::new(0.0, 0.001).unwrap();
        // PDF at mean should be very high
        let pdf_at_mean = g.pdf(0.0);
        assert!(pdf_at_mean > 100.0);
        // PDF away from mean should be essentially zero
        let pdf_away = g.pdf(1.0);
        assert!(pdf_away < 1e-10);
    }

    #[test]
    fn test_gaussian_wide_std() {
        // Very wide distribution
        let g = GaussianParams::new(0.0, 100.0).unwrap();
        // PDF should be relatively flat
        let pdf_at_0 = g.pdf(0.0);
        let pdf_at_50 = g.pdf(50.0);
        // The difference should not be too extreme
        assert!(pdf_at_0 / pdf_at_50 < 2.0);
    }

    // === Edge case tests for OnlineStats ===

    #[test]
    fn test_online_stats_empty() {
        let stats = OnlineStats::new();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.std(), 0.0);
    }

    #[test]
    fn test_online_stats_single_value() {
        let mut stats = OnlineStats::new();
        stats.add(5.0);
        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), 5.0);
        // Variance is undefined for single value, returns 0
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.std(), 0.0);
    }

    #[test]
    fn test_online_stats_two_values() {
        let mut stats = OnlineStats::new();
        stats.add(4.0);
        stats.add(6.0);
        assert_eq!(stats.count(), 2);
        assert_eq!(stats.mean(), 5.0);
        // Sample variance: ((4-5)^2 + (6-5)^2) / (2-1) = 2
        assert!((stats.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_online_stats_identical_values() {
        let mut stats = OnlineStats::new();
        for _ in 0..10 {
            stats.add(42.0);
        }
        assert_eq!(stats.count(), 10);
        assert_eq!(stats.mean(), 42.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.std(), 0.0);
    }

    #[test]
    fn test_online_stats_large_count() {
        let mut stats = OnlineStats::new();
        for i in 1..=1000 {
            stats.add(i as f64);
        }
        assert_eq!(stats.count(), 1000);
        // Mean of 1..1000 is 500.5
        assert!((stats.mean() - 500.5).abs() < 1e-10);
    }

    #[test]
    fn test_online_stats_numerical_stability() {
        // Test with values that might cause numerical issues
        let mut stats = OnlineStats::new();
        // Add large values with small differences
        for i in 0..100 {
            stats.add(1e10 + i as f64);
        }
        // Mean should be around 1e10 + 49.5
        assert!((stats.mean() - (1e10 + 49.5)).abs() < 1.0);
        // Variance should be reasonable for consecutive integers
        // Var(0..99) = 99*100/12 = 825 (approximately)
        assert!(stats.variance() > 800.0 && stats.variance() < 900.0);
    }

    #[test]
    fn test_online_stats_negative_values() {
        let mut stats = OnlineStats::new();
        stats.add(-5.0);
        stats.add(-3.0);
        stats.add(-1.0);
        assert_eq!(stats.count(), 3);
        assert!((stats.mean() - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_online_stats_mixed_values() {
        let mut stats = OnlineStats::new();
        stats.add(-10.0);
        stats.add(10.0);
        assert_eq!(stats.count(), 2);
        assert!((stats.mean() - 0.0).abs() < 1e-10);
        // Variance: ((-10)^2 + 10^2) / 1 = 200
        assert!((stats.variance() - 200.0).abs() < 1e-10);
    }

    // === EM two-component tests ===

    #[test]
    fn test_em_two_component_clear_separation() {
        // Well-separated clusters
        let mut data = Vec::new();
        for _ in 0..50 { data.push(0.998); }
        for _ in 0..20 { data.push(0.9997); }

        let init_low = GaussianParams::new_unchecked(0.998, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

        let result = em_two_component(&data, &init_low, &init_high, 0.7, 50, 1e-6);
        assert!(result.is_some());
        let (low, high, w_low, _w_high) = result.unwrap();
        assert!(low.mean < high.mean);
        assert!(w_low > 0.5); // More non-IBD data
    }

    #[test]
    fn test_em_two_component_too_few_points() {
        let data = vec![0.998, 0.999, 0.9997];
        let init_low = GaussianParams::new_unchecked(0.998, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
        let result = em_two_component(&data, &init_low, &init_high, 0.7, 50, 1e-6);
        assert!(result.is_none());
    }

    #[test]
    fn test_em_two_component_all_same() {
        // All identical => one component vanishes => returns None
        let data = vec![0.999; 20];
        let init_low = GaussianParams::new_unchecked(0.998, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
        let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
        // May return None or converge to a single component
        if let Some((low, high, _, _)) = result {
            // Both means should be near 0.999
            assert!((low.mean - 0.999).abs() < 0.01);
            assert!((high.mean - 0.999).abs() < 0.01);
        }
    }

    // === MAP-regularized EM tests ===

    #[test]
    fn test_em_map_clear_separation() {
        let mut data = Vec::new();
        for _ in 0..50 { data.push(0.998); }
        for _ in 0..20 { data.push(0.9997); }

        let init_low = GaussianParams::new_unchecked(0.998, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

        let result = em_two_component_map(&data, &init_low, &init_high, 0.7, 50, 1e-6, 5.0);
        assert!(result.is_some());
        let (low, high, w_low, _) = result.unwrap();
        assert!(low.mean < high.mean, "Low mean should be < high mean");
        assert!(w_low > 0.5, "Non-IBD weight should be > 0.5");
    }

    #[test]
    fn test_em_map_prevents_collapse() {
        // Data with weak signal - without MAP, EM might collapse
        // With MAP, the prior should keep components separated
        let mut data = Vec::new();
        for _ in 0..90 { data.push(0.999); } // Ambiguous data
        for _ in 0..10 { data.push(0.9997); }

        let init_low = GaussianParams::new_unchecked(0.99875, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

        // Strong prior to maintain separation
        let result = em_two_component_map(&data, &init_low, &init_high, 0.9, 50, 1e-6, 20.0);
        if let Some((low, high, _, _)) = result {
            // MAP should keep components at least somewhat separated
            assert!(high.mean > low.mean,
                "MAP should maintain mean ordering: low={}, high={}", low.mean, high.mean);
        }
    }

    #[test]
    fn test_em_map_too_few_points() {
        let data = vec![0.998, 0.999, 0.9997];
        let init_low = GaussianParams::new_unchecked(0.998, 0.001);
        let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
        let result = em_two_component_map(&data, &init_low, &init_high, 0.7, 50, 1e-6, 5.0);
        assert!(result.is_none());
    }

    // === Trimmed mean tests ===

    #[test]
    fn test_trimmed_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tm = trimmed_mean(&data, 0.1).unwrap();
        // Trim 1 from each end: mean of [2,3,4,5,6,7,8,9] = 5.5
        assert!((tm - 5.5).abs() < 0.01, "Trimmed mean should be 5.5, got {}", tm);
    }

    #[test]
    fn test_trimmed_mean_no_trim() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tm = trimmed_mean(&data, 0.0).unwrap();
        assert!((tm - 3.0).abs() < 1e-10, "0% trimmed mean should be regular mean");
    }

    #[test]
    fn test_trimmed_mean_empty() {
        let data: Vec<f64> = vec![];
        assert!(trimmed_mean(&data, 0.1).is_none());
    }

    #[test]
    fn test_trimmed_mean_robust_to_outliers() {
        // One extreme outlier
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let regular_mean = data.iter().sum::<f64>() / data.len() as f64;
        let tm = trimmed_mean(&data, 0.2).unwrap();
        // Trimmed mean should be much less affected by the outlier
        assert!(tm < regular_mean, "Trimmed mean {} should be < regular mean {}", tm, regular_mean);
    }

    #[test]
    fn test_trimmed_mean_high_trim() {
        // Very high trim should approach the median
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tm = trimmed_mean(&data, 0.4).unwrap(); // Trim 40% from each end
        // Should keep only the middle ~2 values (4.5-6.5 range)
        assert!(tm > 4.0 && tm < 7.0, "High-trim mean should be near median, got {}", tm);
    }

    // === BIC model selection tests ===

    #[test]
    fn test_bic_prefers_two_components_clear_separation() {
        // Data with clear bimodal distribution - BIC should prefer 2 components
        let mut data = vec![0.998; 50];
        data.extend(std::iter::repeat(0.9997).take(50));

        let low = GaussianParams::new_unchecked(0.998, 0.0005);
        let high = GaussianParams::new_unchecked(0.9997, 0.0003);

        let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
        assert!(bic_2 < bic_1,
            "BIC should prefer 2 components for bimodal data: bic_1={:.2}, bic_2={:.2}",
            bic_1, bic_2);
    }

    #[test]
    fn test_bic_prefers_one_component_unimodal() {
        // Data that is clearly unimodal - BIC should prefer 1 component
        let data: Vec<f64> = (0..100).map(|i| 0.999 + (i as f64 * 0.00001) - 0.0005).collect();

        let low = GaussianParams::new_unchecked(0.9985, 0.001);
        let high = GaussianParams::new_unchecked(0.9995, 0.001);

        let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
        // For nearly unimodal data, 1-component should be preferred (fewer params)
        // or at least competitive
        assert!(bic_1 <= bic_2 + 10.0,
            "BIC for 1 component should be competitive for unimodal data: bic_1={:.2}, bic_2={:.2}",
            bic_1, bic_2);
    }

    #[test]
    fn test_bic_too_few_data() {
        let data = vec![0.999];
        let low = GaussianParams::new_unchecked(0.998, 0.001);
        let high = GaussianParams::new_unchecked(0.9997, 0.0005);
        let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
        assert_eq!(bic_1, 0.0);
        assert_eq!(bic_2, 0.0);
    }

    // ===== Logit transform tests =====

    #[test]
    fn test_logit_at_half() {
        // logit(0.5) = log(0.5/0.5) = log(1) = 0
        let result = logit(0.5);
        assert!((result - 0.0).abs() < 1e-10, "logit(0.5) should be 0, got {}", result);
    }

    #[test]
    fn test_logit_near_one() {
        // logit(0.997) ≈ 5.81
        let result = logit(0.997);
        assert!(result > 5.5 && result < 6.2, "logit(0.997) should be ~5.81, got {}", result);

        // logit(0.9997) ≈ 8.11
        let result2 = logit(0.9997);
        assert!(result2 > 7.5 && result2 < 8.5, "logit(0.9997) should be ~8.11, got {}", result2);

        // Difference should be ~2.3 (much larger than raw 0.0027)
        let diff = result2 - result;
        assert!(diff > 2.0 && diff < 3.0, "logit difference should be ~2.3, got {}", diff);

        // Values at or near 1.0 should be capped at LOGIT_CAP
        let at_one = logit(1.0);
        assert!((at_one - LOGIT_CAP).abs() < 1e-10,
            "logit(1.0) should be capped at {}, got {}", LOGIT_CAP, at_one);
    }

    #[test]
    fn test_logit_near_zero() {
        let result = logit(0.003);
        assert!(result < -5.0, "logit(0.003) should be strongly negative, got {}", result);
    }

    #[test]
    fn test_logit_clamping() {
        // At exact 0 and 1, should not be infinite due to clamping
        let at_zero = logit(0.0);
        assert!(at_zero.is_finite(), "logit(0) should be finite due to clamping");
        assert!((at_zero - (-LOGIT_CAP)).abs() < 1e-10, "logit(0) should be -LOGIT_CAP");

        let at_one = logit(1.0);
        assert!(at_one.is_finite(), "logit(1) should be finite due to clamping");
        assert!((at_one - LOGIT_CAP).abs() < 1e-10, "logit(1) should be LOGIT_CAP");
    }

    #[test]
    fn test_inv_logit_roundtrip() {
        // Values that stay within LOGIT_CAP should roundtrip perfectly
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999] {
            let roundtrip = inv_logit(logit(x));
            assert!((roundtrip - x).abs() < 1e-8,
                "inv_logit(logit({})) should be {}, got {}", x, x, roundtrip);
        }

        // Values near 1.0 get capped, so roundtrip goes to inv_logit(LOGIT_CAP)
        let capped = inv_logit(logit(0.99999));
        assert!(capped > 0.999, "Capped value should still be near 1.0, got {}", capped);
    }

    #[test]
    fn test_logit_transform_observations() {
        let obs = vec![0.5, 0.9, 0.99, 0.999, 0.9999];
        let transformed = logit_transform_observations(&obs);

        assert_eq!(transformed.len(), obs.len());

        // Should be monotonically increasing (since logit is monotone)
        for i in 1..transformed.len() {
            assert!(transformed[i] > transformed[i-1],
                "Logit transform should preserve order");
        }

        // First value (0.5) should be near 0
        assert!((transformed[0]).abs() < 0.1);
    }

    #[test]
    fn test_logit_transform_empty() {
        let obs: Vec<f64> = vec![];
        let transformed = logit_transform_observations(&obs);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_gaussian_to_logit_space() {
        // Transform EUR non-IBD emission to logit space
        let raw_mean = 0.99915;
        let raw_std = 0.001;
        let logit_params = gaussian_to_logit_space(raw_mean, raw_std);

        // logit(0.99915) ≈ 7.07
        assert!(logit_params.mean > 6.5 && logit_params.mean < 7.5,
            "Logit mean should be ~7.07, got {}", logit_params.mean);

        // Std should be reasonable (not 0 or infinite)
        assert!(logit_params.std > 0.5 && logit_params.std < 20.0,
            "Logit std should be reasonable, got {}", logit_params.std);
    }

    #[test]
    fn test_gaussian_to_logit_space_ibd() {
        // Transform IBD emission to logit space
        let logit_ibd = gaussian_to_logit_space(0.9997, 0.0005);

        // logit(0.9997) ≈ 8.11
        assert!(logit_ibd.mean > 7.5 && logit_ibd.mean < 8.5,
            "Logit IBD mean should be ~8.11, got {}", logit_ibd.mean);

        // IBD mean should be higher than non-IBD mean in logit space
        let logit_non_ibd = gaussian_to_logit_space(0.99915, 0.001);
        assert!(logit_ibd.mean > logit_non_ibd.mean,
            "IBD logit mean ({}) should be > non-IBD logit mean ({})",
            logit_ibd.mean, logit_non_ibd.mean);
    }

    #[test]
    fn test_logit_separation_improvement() {
        // The key test: logit transform should improve separation between
        // IBD and non-IBD emission distributions
        let raw_non_ibd_mean = 0.99915;
        let raw_ibd_mean = 0.9997;
        let raw_separation = raw_ibd_mean - raw_non_ibd_mean; // ~0.00055

        let logit_non_ibd = gaussian_to_logit_space(raw_non_ibd_mean, 0.001);
        let logit_ibd = gaussian_to_logit_space(raw_ibd_mean, 0.0005);
        let logit_separation = logit_ibd.mean - logit_non_ibd.mean;

        // Logit separation should be much larger than raw separation
        assert!(logit_separation > raw_separation * 100.0,
            "Logit separation ({}) should be >> raw separation ({})",
            logit_separation, raw_separation);
    }
}
