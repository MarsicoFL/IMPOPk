//! Statistical utilities for IBD analysis

use std::f64::consts::PI;

/// Gaussian distribution parameters
#[derive(Debug, Clone, Copy)]
pub struct GaussianParams {
    pub mean: f64,
    pub std: f64,
}

impl GaussianParams {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        (-0.5 * z * z).exp() / (self.std * (2.0 * PI).sqrt())
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        -0.5 * z * z - self.std.ln() - 0.5 * (2.0 * PI).ln()
    }
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
        let g = GaussianParams::new(0.0, 1.0);
        let pdf_at_0 = g.pdf(0.0);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((pdf_at_0 - expected).abs() < 1e-10);
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
        let g = GaussianParams::new(5.0, 2.0);
        let pdf_at_mean = g.pdf(5.0);
        // At mean, z = 0, so pdf = 1 / (std * sqrt(2*pi))
        let expected = 1.0 / (2.0 * (2.0 * PI).sqrt());
        assert!((pdf_at_mean - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_pdf_symmetry() {
        let g = GaussianParams::new(0.0, 1.0);
        // PDF should be symmetric around mean
        let pdf_plus_1 = g.pdf(1.0);
        let pdf_minus_1 = g.pdf(-1.0);
        assert!((pdf_plus_1 - pdf_minus_1).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_log_pdf_consistency() {
        let g = GaussianParams::new(3.0, 0.5);
        let x = 2.5;
        // log(pdf(x)) should equal log_pdf(x)
        let pdf_val = g.pdf(x);
        let log_pdf_val = g.log_pdf(x);
        assert!((pdf_val.ln() - log_pdf_val).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_narrow_std() {
        // Very narrow distribution
        let g = GaussianParams::new(0.0, 0.001);
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
        let g = GaussianParams::new(0.0, 100.0);
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
}
