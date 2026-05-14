//! Tests for logit-space emissions, standalone forward/backward algorithms,
//! and segment histogram edge cases.

use impopk_ibd::hmm::{
    backward, backward_with_distances, backward_with_genetic_map, forward, forward_with_distances,
    forward_with_genetic_map, GeneticMap, HmmParams,
};
use impopk_ibd::segment::{segment_length_histogram, Segment};
// ── from_population_logit ────────────────────────────────────────────────

#[test]
fn from_population_logit_creates_valid_params() {
    use impopk_ibd::hmm::Population;
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    // Transition rows sum to 1
    for row in &params.transition {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Transition row sums to {}", sum);
    }
    // Initial probs sum to 1
    let init_sum: f64 = params.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-10);
    // Emission means: IBD state (logit space) should be higher than non-IBD
    assert!(
        params.emission[1].mean > params.emission[0].mean,
        "IBD emission mean ({}) should exceed non-IBD ({})",
        params.emission[1].mean,
        params.emission[0].mean
    );
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn from_population_logit_panics_on_zero_p_enter() {
    use impopk_ibd::hmm::Population;
    let _ = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0, 5000);
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn from_population_logit_panics_on_one_p_enter() {
    use impopk_ibd::hmm::Population;
    let _ = HmmParams::from_population_logit(Population::EUR, 50.0, 1.0, 5000);
}

#[test]
fn from_population_logit_clamps_expected_windows() {
    use impopk_ibd::hmm::Population;
    // Very small expected_ibd_windows → p_stay_ibd clamped to 0.5
    let params = HmmParams::from_population_logit(Population::AFR, 1.0, 0.01, 5000);
    // p_stay should be clamped to minimum 0.5
    assert!(params.transition[1][1] >= 0.5);
}

// ── estimate_emissions_logit ─────────────────────────────────────────────

#[test]
fn estimate_emissions_logit_low_variance_ibd_branch() {
    use impopk_ibd::hmm::Population;
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let ibd_prior_mean = params.emission[1].mean;

    // All observations near the IBD prior mean (high mean, low variance)
    let obs: Vec<f64> = (0..20).map(|i| ibd_prior_mean + 0.0001 * i as f64).collect();
    let old_emission0 = params.emission[0].clone();

    params.estimate_emissions_logit(&obs, Some(Population::EUR), 5000);

    // Since mean > prior_ibd.mean - 1.0, emission[1] (IBD state) should be updated
    // emission[0] should remain mostly the same
    assert!(
        (params.emission[0].mean - old_emission0.mean).abs() < 1e-3,
        "non-IBD emission should not change much"
    );
}

#[test]
fn estimate_emissions_logit_low_variance_non_ibd_branch() {
    use impopk_ibd::hmm::Population;
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let ibd_prior_mean = params.emission[1].mean;

    // All observations well below the IBD prior mean (low mean, low variance)
    let low_val = ibd_prior_mean - 5.0;
    let obs: Vec<f64> = (0..20).map(|i| low_val + 0.00001 * i as f64).collect();
    let old_emission1 = params.emission[1].clone();

    params.estimate_emissions_logit(&obs, Some(Population::EUR), 5000);

    // Since mean < prior_ibd.mean - 1.0, emission[0] (non-IBD) should be updated
    // emission[1] should remain mostly the same
    assert!(
        (params.emission[1].mean - old_emission1.mean).abs() < 1e-3,
        "IBD emission should not change much"
    );
    // non-IBD mean should be close to the observation mean
    assert!(
        (params.emission[0].mean - low_val).abs() < 0.1,
        "non-IBD emission mean should be near {}",
        low_val
    );
}

#[test]
fn estimate_emissions_logit_too_few_observations() {
    use impopk_ibd::hmm::Population;
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let original = params.clone();
    // Only 5 observations (<10), should return early without modifying
    params.estimate_emissions_logit(&[1.0, 2.0, 3.0, 4.0, 5.0], Some(Population::EUR), 5000);
    assert_eq!(params.emission[0].mean, original.emission[0].mean);
    assert_eq!(params.emission[1].mean, original.emission[1].mean);
}

// ── forward standalone ──────────────────────────────────────────────────

#[test]
fn forward_empty_returns_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let (alpha, ll) = forward(&[], &params);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn forward_single_observation() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let (alpha, ll) = forward(&[0.998], &params);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
    // Log-sum-exp of the two states should equal ll
    let max_log = alpha[0][0].max(alpha[0][1]);
    let computed_ll =
        max_log + ((alpha[0][0] - max_log).exp() + (alpha[0][1] - max_log).exp()).ln();
    assert!((ll - computed_ll).abs() < 1e-10);
}

#[test]
fn forward_increasing_observations_shifts_ibd_posterior() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    // Low identity
    let (alpha_low, _) = forward(&[0.990; 5], &params);
    // High identity
    let (alpha_high, _) = forward(&[0.9999; 5], &params);

    // For high identity, the IBD state should have higher probability
    let last_low = alpha_low.last().unwrap();
    let last_high = alpha_high.last().unwrap();
    // Difference (state1 - state0) should be larger for high identity
    let diff_low = last_low[1] - last_low[0];
    let diff_high = last_high[1] - last_high[0];
    assert!(
        diff_high > diff_low,
        "Higher identity should shift towards IBD state"
    );
}

// ── backward standalone ─────────────────────────────────────────────────

#[test]
fn backward_empty_returns_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let beta = backward(&[], &params);
    assert!(beta.is_empty());
}

#[test]
fn backward_single_observation() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let beta = backward(&[0.998], &params);
    assert_eq!(beta.len(), 1);
    // Last element should be [0.0, 0.0] in log space
    assert_eq!(beta[0][0], 0.0);
    assert_eq!(beta[0][1], 0.0);
}

#[test]
fn forward_backward_consistency_via_standalone() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9995, 0.9998, 0.999, 0.9997];
    let (alpha, ll_forward) = forward(&obs, &params);
    let beta = backward(&obs, &params);

    // At every time t, the log-sum-exp of alpha[t] + beta[t] should equal ll_forward
    for t in 0..obs.len() {
        let log_probs = [alpha[t][0] + beta[t][0], alpha[t][1] + beta[t][1]];
        let max_log = log_probs[0].max(log_probs[1]);
        let ll_t =
            max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        assert!(
            (ll_t - ll_forward).abs() < 1e-6,
            "LL at t={} ({}) should match forward LL ({})",
            t,
            ll_t,
            ll_forward
        );
    }
}

// ── forward_with_distances standalone ───────────────────────────────────

#[test]
fn forward_with_distances_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let (alpha, ll) = forward_with_distances(&[], &params, &[]);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn forward_with_distances_mismatched_falls_back() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = [0.998, 0.999, 0.9995];
    // Positions length doesn't match observations — should fall back to forward()
    let positions = [(0, 5000), (5001, 10000)];
    let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);
    let (alpha_plain, ll_plain) = forward(&obs, &params);
    assert_eq!(alpha_dist.len(), alpha_plain.len());
    assert!((ll_dist - ll_plain).abs() < 1e-10);
}

#[test]
fn forward_with_distances_single_obs() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let (alpha, ll) = forward_with_distances(&[0.999], &params, &[(0, 5000)]);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

// ── backward_with_distances standalone ──────────────────────────────────

#[test]
fn backward_with_distances_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let beta = backward_with_distances(&[], &params, &[]);
    assert!(beta.is_empty());
}

#[test]
fn backward_with_distances_mismatched_falls_back() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = [0.998, 0.999];
    let positions = [(0, 5000)]; // mismatch
    let beta_dist = backward_with_distances(&obs, &params, &positions);
    let beta_plain = backward(&obs, &params);
    assert_eq!(beta_dist.len(), beta_plain.len());
    for t in 0..obs.len() {
        assert!((beta_dist[t][0] - beta_plain[t][0]).abs() < 1e-10);
        assert!((beta_dist[t][1] - beta_plain[t][1]).abs() < 1e-10);
    }
}

// ── forward_with_genetic_map standalone ─────────────────────────────────

#[test]
fn forward_with_genetic_map_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![]);
    let (alpha, ll) = forward_with_genetic_map(&[], &params, &[], &gm, 5000);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn forward_with_genetic_map_mismatched_falls_back() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    let obs = [0.998, 0.999, 0.9995];
    let positions = [(0, 5000)]; // mismatch
    let (alpha_gm, ll_gm) = forward_with_genetic_map(&obs, &params, &positions, &gm, 5000);
    let (alpha_plain, ll_plain) = forward(&obs, &params);
    assert_eq!(alpha_gm.len(), alpha_plain.len());
    assert!((ll_gm - ll_plain).abs() < 1e-10);
}

#[test]
fn forward_with_genetic_map_single_obs() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    let (alpha, ll) = forward_with_genetic_map(&[0.999], &params, &[(0, 5000)], &gm, 5000);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

// ── backward_with_genetic_map standalone ────────────────────────────────

#[test]
fn backward_with_genetic_map_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![]);
    let beta = backward_with_genetic_map(&[], &params, &[], &gm, 5000);
    assert!(beta.is_empty());
}

#[test]
fn backward_with_genetic_map_mismatched_falls_back() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![(0, 0.0), (1_000_000, 1.0)]);
    let obs = [0.998, 0.999];
    let positions = [(0, 5000)]; // mismatch
    let beta_gm = backward_with_genetic_map(&obs, &params, &positions, &gm, 5000);
    let beta_plain = backward(&obs, &params);
    assert_eq!(beta_gm.len(), beta_plain.len());
    for t in 0..obs.len() {
        assert!((beta_gm[t][0] - beta_plain[t][0]).abs() < 1e-10);
        assert!((beta_gm[t][1] - beta_plain[t][1]).abs() < 1e-10);
    }
}

#[test]
fn forward_backward_with_distances_consistency() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9995, 0.9998, 0.999];
    let positions: Vec<(u64, u64)> = (0..4).map(|i| (i * 5000, i * 5000 + 4999)).collect();

    let (alpha, ll_fwd) = forward_with_distances(&obs, &params, &positions);
    let beta = backward_with_distances(&obs, &params, &positions);

    // At every time t, alpha[t] + beta[t] should give log-likelihood
    for t in 0..obs.len() {
        let log_probs = [alpha[t][0] + beta[t][0], alpha[t][1] + beta[t][1]];
        let max_log = log_probs[0].max(log_probs[1]);
        let ll_t =
            max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        assert!(
            (ll_t - ll_fwd).abs() < 1e-6,
            "LL at t={}: {} vs forward {}",
            t,
            ll_t,
            ll_fwd
        );
    }
}

#[test]
fn forward_backward_with_genetic_map_consistency() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::new(vec![(0, 0.0), (100_000, 0.1)]);
    let obs = vec![0.998, 0.9995, 0.9998, 0.999];
    let positions: Vec<(u64, u64)> = (0..4).map(|i| (i * 5000, i * 5000 + 4999)).collect();

    let (alpha, ll_fwd) = forward_with_genetic_map(&obs, &params, &positions, &gm, 5000);
    let beta = backward_with_genetic_map(&obs, &params, &positions, &gm, 5000);

    for t in 0..obs.len() {
        let log_probs = [alpha[t][0] + beta[t][0], alpha[t][1] + beta[t][1]];
        let max_log = log_probs[0].max(log_probs[1]);
        let ll_t =
            max_log + ((log_probs[0] - max_log).exp() + (log_probs[1] - max_log).exp()).ln();
        assert!(
            (ll_t - ll_fwd).abs() < 1e-6,
            "LL at t={}: {} vs forward {}",
            t,
            ll_t,
            ll_fwd
        );
    }
}

// ── segment_length_histogram edge cases ─────────────────────────────────

fn make_segment(start: u64, end: u64) -> Segment {
    Segment {
        chrom: "chr1".to_string(),
        start,
        end,
        hap_a: "A#1".to_string(),
        hap_b: "B#1".to_string(),
        mean_identity: 0.999,
        min_identity: 0.998,
        identity_sum: 9.99,
        n_windows: 10,
        n_called: 10,
        start_idx: 0,
        end_idx: 9,
    }
}

#[test]
fn histogram_all_segments_shorter_than_bin() {
    // All segment lengths < bin_size → all go into bin 0
    let segments = vec![
        make_segment(1000, 2000), // 1000 bp
        make_segment(3000, 3500), // 500 bp
        make_segment(5000, 5800), // 800 bp
    ];
    let hist = segment_length_histogram(&segments, 10000);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 3)); // All in first bin
}

#[test]
fn histogram_single_segment_exact_bin_boundary() {
    // Length exactly at bin_size boundary
    let segments = vec![make_segment(0, 10000)]; // 10000 bp
    let hist = segment_length_histogram(&segments, 10000);
    // 10000 / 10000 = 1, so bin 1
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (10000, 1));
}

#[test]
fn histogram_mixed_bins() {
    let segments = vec![
        make_segment(0, 500),    // 500 bp → bin 0
        make_segment(0, 1500),   // 1500 bp → bin 1
        make_segment(0, 2500),   // 2500 bp → bin 2
        make_segment(0, 1200),   // 1200 bp → bin 1
    ];
    let hist = segment_length_histogram(&segments, 1000);
    // Expected: (0, 1), (1000, 2), (2000, 1)
    assert_eq!(hist.len(), 3);
    assert_eq!(hist[0], (0, 1));
    assert_eq!(hist[1], (1000, 2));
    assert_eq!(hist[2], (2000, 1));
}

// ── GeneticMap empty interpolation ──────────────────────────────────────

#[test]
fn genetic_map_empty_interpolation() {
    let gm = GeneticMap::new(vec![]);
    // Empty map: interpolate returns 0.0
    assert_eq!(gm.interpolate_cm(1000), 0.0);
    assert_eq!(gm.genetic_distance_cm(0, 1_000_000), 0.0);
}

#[test]
fn genetic_map_single_entry_interpolation() {
    let gm = GeneticMap::new(vec![(50000, 0.5)]);
    // Single entry: always returns that cM value
    assert_eq!(gm.interpolate_cm(50000), 0.5);
    assert_eq!(gm.interpolate_cm(0), 0.5);
    assert_eq!(gm.interpolate_cm(100000), 0.5);
    // Distance is 0 since interpolation always returns 0.5
    assert_eq!(gm.genetic_distance_cm(0, 100000), 0.0);
}

// ── forward_with_distances large gap ────────────────────────────────────

#[test]
fn forward_with_distances_large_gap_between_windows() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    // Windows with a huge gap between them — should still produce finite results
    let obs = [0.999, 0.999];
    let positions = [(0, 5000), (10_000_000, 10_005_000)];
    let (alpha, ll) = forward_with_distances(&obs, &params, &positions);
    assert_eq!(alpha.len(), 2);
    assert!(ll.is_finite(), "LL should be finite even with large gap");
}

// ── HmmParams with logit: Viterbi produces valid states ─────────────────

#[test]
fn logit_params_viterbi_produces_valid_states() {
    use impopk_ibd::hmm::{viterbi, Population};

    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    // Generate observations in logit space (high values = high identity)
    let obs: Vec<f64> = (0..20)
        .map(|i| {
            if i < 10 {
                params.emission[0].mean
            } else {
                params.emission[1].mean
            }
        })
        .collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 20);
    // All states are 0 or 1
    assert!(states.iter().all(|&s| s == 0 || s == 1));
}
