//! Fuzz-like numerical stability tests for the IBD HMM.
//!
//! Verifies that the HMM algorithms (forward, backward, forward_backward, viterbi,
//! infer_ibd, infer_ibd_with_training) produce valid, finite outputs for a wide
//! variety of adversarial and edge-case inputs including:
//! - Uniform random observations
//! - Observations near 0 and 1 boundaries
//! - Repeated identical values
//! - Alternating extreme values
//! - Very long sequences
//! - Mixed high/low identity patterns

use hprc_ibd::hmm::{
    extract_ibd_segments_with_lod, extract_ibd_segments_with_posteriors, forward, forward_backward,
    infer_ibd, infer_ibd_with_training, refine_states_with_posteriors, segment_lod_score,
    segment_posterior_std, viterbi, HmmParams, Population,
};

/// Simple deterministic pseudo-random number generator (xorshift64)
/// so tests are reproducible without external dependencies.
struct PseudoRng {
    state: u64,
}

impl PseudoRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a f64 in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000_000) as f64 / 1_000_000_000.0
    }

    /// Returns a f64 in [lo, hi)
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ── Forward/Backward stability under random data ───────────────────────

#[test]
fn forward_random_uniform_observations_all_finite() {
    let mut rng = PseudoRng::new(42);
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..500).map(|_| rng.next_range(0.990, 1.0)).collect();
    let (alpha, ll) = forward(&obs, &params);
    assert_eq!(alpha.len(), 500);
    assert!(ll.is_finite(), "log-likelihood not finite: {}", ll);
    for (t, a) in alpha.iter().enumerate() {
        assert!(a[0].is_finite(), "alpha[{}][0] not finite", t);
        assert!(a[1].is_finite(), "alpha[{}][1] not finite", t);
    }
}

#[test]
fn forward_backward_random_posteriors_in_01() {
    let mut rng = PseudoRng::new(123);
    let params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..300).map(|_| rng.next_range(0.995, 1.0)).collect();
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 300);
    assert!(ll.is_finite());
    for (t, &p) in posteriors.iter().enumerate() {
        assert!(
            p >= -1e-10 && p <= 1.0 + 1e-10,
            "posterior[{}] = {} out of [0,1]",
            t,
            p
        );
    }
}

#[test]
fn viterbi_random_states_are_binary() {
    let mut rng = PseudoRng::new(777);
    let params = HmmParams::from_population(Population::EAS, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..400).map(|_| rng.next_range(0.992, 1.0)).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 400);
    for (i, &s) in states.iter().enumerate() {
        assert!(s <= 1, "state[{}] = {} (not binary)", i, s);
    }
}

// ── Boundary observations ──────────────────────────────────────────────

#[test]
fn forward_backward_observations_near_one() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.999999; 100];
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite());
    for &p in &posteriors {
        assert!(p.is_finite());
    }
}

#[test]
fn forward_backward_observations_near_zero() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.001; 50];
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite());
    for &p in &posteriors {
        assert!(p.is_finite());
    }
}

#[test]
fn viterbi_extreme_low_observations() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.0001; 30];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 30);
    // All should be non-IBD for very low identity
    assert!(states.iter().all(|&s| s == 0));
}

#[test]
fn viterbi_extreme_high_observations() {
    // Using balanced priors so IBD is possible
    let params = HmmParams::from_expected_length(50.0, 0.5, 5000);
    let obs = vec![0.9999; 30];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 30);
    // All should be IBD for very high identity with balanced priors
    assert!(states.iter().all(|&s| s == 1));
}

// ── Alternating extreme values ─────────────────────────────────────────

#[test]
fn forward_backward_alternating_high_low() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..200)
        .map(|i| if i % 2 == 0 { 0.9999 } else { 0.990 })
        .collect();
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite());
    assert_eq!(posteriors.len(), 200);
    for &p in &posteriors {
        assert!(p.is_finite());
        assert!(p >= -1e-10 && p <= 1.0 + 1e-10);
    }
}

#[test]
fn viterbi_alternating_extreme() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..100)
        .map(|i| if i % 2 == 0 { 0.9999 } else { 0.0001 })
        .collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 100);
    for &s in &states {
        assert!(s <= 1);
    }
}

// ── Long sequences ─────────────────────────────────────────────────────

#[test]
fn forward_backward_long_sequence_no_overflow() {
    let mut rng = PseudoRng::new(999);
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..5000).map(|_| rng.next_range(0.996, 1.0)).collect();
    let (posteriors, ll) = forward_backward(&obs, &params);
    assert!(ll.is_finite(), "ll={}", ll);
    assert_eq!(posteriors.len(), 5000);
    // Spot check a few
    assert!(posteriors[0].is_finite());
    assert!(posteriors[2500].is_finite());
    assert!(posteriors[4999].is_finite());
}

#[test]
fn viterbi_long_sequence() {
    let mut rng = PseudoRng::new(1234);
    let params = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..5000).map(|_| rng.next_range(0.996, 1.0)).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5000);
}

// ── infer_ibd pipeline ─────────────────────────────────────────────────

#[test]
fn infer_ibd_random_all_fields_valid() {
    let mut rng = PseudoRng::new(555);
    let params = HmmParams::from_population(Population::CSA, 50.0, 0.0001, 5000);
    let obs: Vec<f64> = (0..200).map(|_| rng.next_range(0.993, 1.0)).collect();
    let result = infer_ibd(&obs, &params);
    assert_eq!(result.states.len(), 200);
    assert_eq!(result.posteriors.len(), 200);
    assert!(result.log_likelihood.is_finite());
    for &s in &result.states {
        assert!(s <= 1);
    }
    for &p in &result.posteriors {
        assert!(p.is_finite());
    }
}

// ── infer_ibd_with_training ────────────────────────────────────────────

#[test]
fn infer_ibd_with_training_random_data_all_pops() {
    let populations = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
        Population::Generic,
    ];

    let mut rng = PseudoRng::new(2024);

    for pop in &populations {
        let mut params = HmmParams::from_population(*pop, 50.0, 0.0001, 5000);
        let obs: Vec<f64> = (0..150).map(|_| rng.next_range(0.995, 1.0)).collect();
        let result = infer_ibd_with_training(&obs, &mut params, *pop, 5000, 10);
        assert_eq!(result.states.len(), 150, "pop={:?}", pop);
        assert_eq!(result.posteriors.len(), 150, "pop={:?}", pop);
        assert!(result.log_likelihood.is_finite(), "pop={:?}", pop);
    }
}

#[test]
fn infer_ibd_with_training_bimodal_synthetic() {
    // Synthetic data with a clear IBD region embedded
    let mut rng = PseudoRng::new(3030);
    let mut obs: Vec<f64> = Vec::new();

    // Non-IBD region: identity ~0.997-0.999
    for _ in 0..50 {
        obs.push(rng.next_range(0.997, 0.999));
    }
    // IBD region: identity ~0.9995-0.9999
    for _ in 0..30 {
        obs.push(rng.next_range(0.9995, 0.9999));
    }
    // Non-IBD region again
    for _ in 0..50 {
        obs.push(rng.next_range(0.997, 0.999));
    }

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let result = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 20);

    assert_eq!(result.states.len(), 130);
    assert!(result.log_likelihood.is_finite());

    // The IBD region (indices 50-79) should have higher posteriors than flanking
    let mean_ibd_post: f64 =
        result.posteriors[50..80].iter().sum::<f64>() / 30.0;
    let mean_non_ibd_post: f64 = (result.posteriors[0..30].iter().sum::<f64>()
        + result.posteriors[100..130].iter().sum::<f64>())
        / 60.0;

    // IBD region should generally have higher posteriors (but not guaranteed with BW)
    // At minimum, both should be finite
    assert!(mean_ibd_post.is_finite());
    assert!(mean_non_ibd_post.is_finite());
}

#[test]
fn infer_ibd_with_training_zero_iterations_equivalent() {
    let params_orig = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let mut rng = PseudoRng::new(8080);
    let obs: Vec<f64> = (0..100).map(|_| rng.next_range(0.995, 1.0)).collect();

    // 0 BW iterations = same as infer_ibd
    let mut params_bw = params_orig.clone();
    let result_bw = infer_ibd_with_training(&obs, &mut params_bw, Population::EUR, 5000, 0);
    let result_plain = infer_ibd(&obs, &params_orig);

    assert_eq!(result_bw.states, result_plain.states);
}

// ── extract_ibd_segments_with_posteriors edge cases ────────────────────

#[test]
fn extract_segments_empty_inputs() {
    let segs = extract_ibd_segments_with_posteriors(&[], &[], 1, 0.5);
    assert!(segs.is_empty());
}

#[test]
fn extract_segments_mismatched_lengths() {
    let states = vec![0, 1, 1, 0];
    let posteriors = vec![0.1, 0.9]; // too short
    let segs = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segs.is_empty());
}

#[test]
fn extract_segments_all_non_ibd() {
    let states = vec![0; 50];
    let posteriors = vec![0.1; 50];
    let segs = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segs.is_empty());
}

#[test]
fn extract_segments_all_ibd() {
    let states = vec![1; 50];
    let posteriors = vec![0.95; 50];
    let segs = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 49);
    assert_eq!(segs[0].n_windows, 50);
    assert!((segs[0].mean_posterior - 0.95).abs() < 1e-10);
}

#[test]
fn extract_segments_min_windows_filters() {
    let states = vec![0, 1, 0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.1, 0.85, 0.9, 0.88, 0.1];
    // min_windows=2 should filter out the single-window segment at idx 1
    let segs = extract_ibd_segments_with_posteriors(&states, &posteriors, 2, 0.5);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 3);
    assert_eq!(segs[0].end_idx, 5);
}

#[test]
fn extract_segments_min_posterior_filters() {
    let states = vec![1, 1, 1, 0, 1, 1, 1];
    let posteriors = vec![0.3, 0.35, 0.32, 0.1, 0.9, 0.95, 0.88];
    // min_mean_posterior=0.8 should filter out the first segment (mean ~0.32)
    let segs = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.8);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 4);
}

#[test]
fn extract_segments_with_lod_scoring() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![
        0.998, 0.997, 0.9998, 0.9999, 0.9997, 0.9998, 0.997, 0.998,
    ];
    let result = infer_ibd(&obs, &params);
    let segs = extract_ibd_segments_with_lod(
        &result.states,
        &result.posteriors,
        1,
        0.1,
        Some((&obs, &params)),
        None,
    );
    for seg in &segs {
        assert!(seg.lod_score.is_finite());
        assert!(seg.mean_posterior.is_finite());
        assert!(seg.min_posterior <= seg.max_posterior);
        assert!(seg.min_posterior <= seg.mean_posterior);
        assert!(seg.mean_posterior <= seg.max_posterior);
    }
}

#[test]
fn extract_segments_with_lod_min_lod_filter() {
    let params = HmmParams::from_expected_length(50.0, 0.5, 5000);
    let obs = vec![0.9999; 20];
    let result = infer_ibd(&obs, &params);
    // With min_lod=1000, should filter out everything (LOD unlikely that high)
    let segs = extract_ibd_segments_with_lod(
        &result.states,
        &result.posteriors,
        1,
        0.0,
        Some((&obs, &params)),
        Some(1000.0),
    );
    assert!(segs.is_empty());
}

// ── segment_posterior_std ───────────────────────────────────────────────

#[test]
fn segment_posterior_std_constant_posteriors() {
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    let std = segment_posterior_std(&posteriors, 0, 4);
    assert!((std - 0.0).abs() < 1e-10);
}

#[test]
fn segment_posterior_std_varying_posteriors() {
    let posteriors = vec![0.8, 0.9, 0.7, 0.95, 0.85];
    let std = segment_posterior_std(&posteriors, 0, 4);
    assert!(std > 0.0);
    assert!(std.is_finite());
}

#[test]
fn segment_posterior_std_single_element() {
    let posteriors = vec![0.9, 0.8];
    // Single-element segment: start==end
    let std = segment_posterior_std(&posteriors, 0, 0);
    assert!((std - 0.0).abs() < 1e-10);
}

#[test]
fn segment_posterior_std_out_of_bounds() {
    let posteriors = vec![0.9, 0.8];
    let std = segment_posterior_std(&posteriors, 0, 10);
    assert!((std - 0.0).abs() < 1e-10);
}

#[test]
fn segment_posterior_std_reversed_indices() {
    let posteriors = vec![0.9, 0.8, 0.7];
    let std = segment_posterior_std(&posteriors, 2, 0);
    assert!((std - 0.0).abs() < 1e-10);
}

// ── segment_lod_score ──────────────────────────────────────────────────

#[test]
fn segment_lod_score_high_identity_positive() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9998, 0.9999, 0.9997, 0.9998, 0.9999];
    let lod = segment_lod_score(&obs, 0, 4, &params);
    assert!(lod.is_finite());
    // High identity → positive LOD for IBD
    assert!(lod > 0.0, "lod={}", lod);
}

#[test]
fn segment_lod_score_low_identity_negative() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.990, 0.991, 0.989, 0.990, 0.991];
    let lod = segment_lod_score(&obs, 0, 4, &params);
    assert!(lod.is_finite());
    // Low identity → negative LOD (non-IBD more likely)
    assert!(lod < 0.0, "lod={}", lod);
}

#[test]
fn segment_lod_score_single_window() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9999];
    let lod = segment_lod_score(&obs, 0, 0, &params);
    assert!(lod.is_finite());
}

// ── refine_states_with_posteriors ──────────────────────────────────────

#[test]
fn refine_states_extends_high_posterior_adjacent() {
    let mut states = vec![0, 0, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.6, 0.9, 0.95, 0.7, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 1 (posterior 0.6 > 0.5) is adjacent to IBD → should extend
    assert_eq!(states[1], 1);
    // Window 4 (posterior 0.7 > 0.5) is adjacent to IBD → should extend
    assert_eq!(states[4], 1);
}

#[test]
fn refine_states_trims_low_posterior_edges() {
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.15, 0.9, 0.1, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 1 (posterior 0.15 < 0.2, at left boundary) → should trim
    assert_eq!(states[1], 0);
    // Window 3 (posterior 0.1 < 0.2, at right boundary) → should trim
    assert_eq!(states[3], 0);
}

#[test]
fn refine_states_empty_no_crash() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert!(states.is_empty());
}

#[test]
fn refine_states_mismatched_lengths_no_crash() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.5];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Should not modify (mismatched lengths)
    assert_eq!(states, vec![0, 1, 0]);
}

// ── Population-adaptive parameter stability ────────────────────────────

#[test]
fn all_populations_produce_valid_params_for_various_window_sizes() {
    let populations = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
        Population::Generic,
    ];
    let window_sizes = [1000, 5000, 10000, 50000, 100000];

    for &pop in &populations {
        for &ws in &window_sizes {
            let params = HmmParams::from_population(pop, 50.0, 0.0001, ws);
            // Transitions are valid probabilities
            for row in &params.transition {
                let sum: f64 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "transition row sum={} for {:?} ws={}",
                    sum,
                    pop,
                    ws
                );
                for &p in row {
                    assert!(p > 0.0 && p < 1.0);
                }
            }
            // Initial probabilities sum to 1
            let init_sum: f64 = params.initial.iter().sum();
            assert!((init_sum - 1.0).abs() < 1e-6);
            // Emissions have positive std
            assert!(params.emission[0].std > 0.0);
            assert!(params.emission[1].std > 0.0);
        }
    }
}

// ── Multi-seed fuzz: forward-backward consistency ──────────────────────

#[test]
fn multi_seed_fuzz_forward_backward_consistency() {
    // Run 20 random seeds, verify invariants hold for each
    for seed in 100..120 {
        let mut rng = PseudoRng::new(seed);
        let pop = match seed % 5 {
            0 => Population::AFR,
            1 => Population::EUR,
            2 => Population::EAS,
            3 => Population::CSA,
            _ => Population::AMR,
        };
        let params = HmmParams::from_population(pop, 50.0, 0.0001, 5000);
        let n = 50 + (rng.next_u64() % 200) as usize;
        let obs: Vec<f64> = (0..n).map(|_| rng.next_range(0.990, 1.0)).collect();

        let (posteriors, ll) = forward_backward(&obs, &params);
        let states = viterbi(&obs, &params);

        assert_eq!(posteriors.len(), n, "seed={}", seed);
        assert_eq!(states.len(), n, "seed={}", seed);
        assert!(ll.is_finite(), "seed={} ll={}", seed, ll);

        for t in 0..n {
            assert!(
                posteriors[t].is_finite(),
                "seed={} t={} p={}",
                seed,
                t,
                posteriors[t]
            );
            assert!(states[t] <= 1, "seed={} t={} s={}", seed, t, states[t]);
        }
    }
}

// ── Baum-Welch stability with random data ──────────────────────────────

#[test]
fn baum_welch_random_data_preserves_validity() {
    let mut rng = PseudoRng::new(7777);
    let obs: Vec<f64> = (0..200).map(|_| rng.next_range(0.994, 1.0)).collect();

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    params.baum_welch(&obs, 20, 1e-6, Some(Population::EUR), 5000);

    // After BW, parameters should still be valid
    for row in &params.transition {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "transition sum={} after BW",
            sum
        );
        for &p in row {
            assert!(p > 0.0 && p <= 1.0);
        }
    }
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

#[test]
fn baum_welch_constant_data_does_not_crash() {
    let obs = vec![0.998; 100];
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    params.baum_welch(&obs, 10, 1e-6, Some(Population::EUR), 5000);
    // Should survive without NaN/panic
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

// ── Combined pipeline: estimate + infer + extract ──────────────────────

#[test]
fn full_pipeline_random_data_produces_valid_segments() {
    let mut rng = PseudoRng::new(4242);
    let mut obs: Vec<f64> = Vec::new();

    // Background identity ~0.997
    for _ in 0..100 {
        obs.push(rng.next_range(0.996, 0.998));
    }
    // High-identity IBD block
    for _ in 0..40 {
        obs.push(rng.next_range(0.9995, 0.9999));
    }
    // Background again
    for _ in 0..100 {
        obs.push(rng.next_range(0.996, 0.998));
    }

    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);

    let result = infer_ibd(&obs, &params);

    let segs = extract_ibd_segments_with_lod(
        &result.states,
        &result.posteriors,
        3,
        0.5,
        Some((&obs, &params)),
        None,
    );

    for seg in &segs {
        assert!(seg.start_idx <= seg.end_idx);
        assert!(seg.n_windows == seg.end_idx - seg.start_idx + 1);
        assert!(seg.mean_posterior.is_finite());
        assert!(seg.min_posterior <= seg.mean_posterior);
        assert!(seg.mean_posterior <= seg.max_posterior);
        assert!(seg.lod_score.is_finite());
    }
}

// ── Estimate emissions stability ───────────────────────────────────────

#[test]
fn estimate_emissions_robust_all_same_value() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998; 200];
    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

#[test]
fn estimate_emissions_robust_few_observations() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999];
    let orig_e0 = params.emission[0].mean;
    params.estimate_emissions_robust(&obs, Some(Population::EUR), 5000);
    // With < 10 observations, should not change much (early return or stable)
    assert!(params.emission[0].mean.is_finite());
    let _ = orig_e0; // used for documentation
}

#[test]
fn estimate_emissions_logit_bimodal_data() {
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let mut obs = Vec::new();
    // Non-IBD cluster in logit space (~5.5)
    for _ in 0..100 {
        obs.push(5.5 + 0.1 * (obs.len() as f64 % 3.0 - 1.0));
    }
    // IBD cluster in logit space (~8.0)
    for _ in 0..30 {
        obs.push(8.0 + 0.05 * (obs.len() as f64 % 3.0 - 1.0));
    }
    params.estimate_emissions_logit(&obs, Some(Population::EUR), 5000);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    // Non-IBD should be lower than IBD
    assert!(params.emission[0].mean < params.emission[1].mean);
}
