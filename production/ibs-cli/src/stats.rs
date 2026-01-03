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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
}
