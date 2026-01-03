//! Hidden Markov Model for IBD state inference
//!
//! This module implements a two-state HMM for distinguishing
//! IBD (Identity-By-Descent) from non-IBD regions based on
//! sequence identity observations.

use crate::stats::{kmeans_1d, GaussianParams};

/// HMM parameters for IBD calling
#[derive(Debug, Clone)]
pub struct HmmParams {
    pub initial: [f64; 2],
    pub transition: [[f64; 2]; 2],
    pub emission: [GaussianParams; 2],
}

impl HmmParams {
    pub fn from_expected_length(expected_ibd_windows: f64, p_enter_ibd: f64) -> Self {
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
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

/// Viterbi algorithm for most likely state sequence
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

/// Extract IBD segments from state sequence
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
}
