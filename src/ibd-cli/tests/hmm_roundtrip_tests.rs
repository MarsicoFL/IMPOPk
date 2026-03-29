//! End-to-end round-trip tests for IBD HMM inference.
//!
//! Strategy: construct known parameters → generate synthetic observations that
//! match the emission distributions → run Viterbi/forward-backward → verify
//! the recovered state sequence matches the ground truth.
//!
//! These tests verify that the full pipeline (estimate → infer → extract)
//! correctly recovers known IBD segments from synthetic data.

use hprc_ibd::hmm::*;
use hprc_ibd::stats::GaussianParams;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate synthetic observations with known state sequence.
/// Non-IBD windows get identity values near `non_ibd_mean`,
/// IBD windows get identity values near `ibd_mean`.
fn generate_observations(
    state_sequence: &[usize],
    non_ibd_mean: f64,
    ibd_mean: f64,
    seed: u64,
) -> Vec<f64> {
    // Simple deterministic pseudo-random variation
    state_sequence
        .iter()
        .enumerate()
        .map(|(i, &state)| {
            let base = if state == 0 { non_ibd_mean } else { ibd_mean };
            // Add small deterministic variation to avoid perfectly uniform data
            let variation = ((i as f64 * 0.618 + seed as f64 * 0.317).sin() * 0.0003).abs();
            let sign = if (i + seed as usize) % 2 == 0 { 1.0 } else { -1.0 };
            (base + sign * variation).clamp(0.0, 1.0)
        })
        .collect()
}

/// Create params with well-separated emission distributions
fn clear_signal_params() -> HmmParams {
    HmmParams {
        initial: [0.9, 0.1],
        transition: [[0.99, 0.01], [0.02, 0.98]],
        emission: [
            GaussianParams { mean: 0.998, std: 0.001 },
            GaussianParams { mean: 0.9999, std: 0.0003 },
        ],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: all non-IBD
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_all_non_ibd() {
    let params = clear_signal_params();
    let ground_truth = vec![0_usize; 100];
    let obs = generate_observations(&ground_truth, 0.998, 0.9999, 42);

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 100);

    // All should be non-IBD
    let ibd_count = states.iter().filter(|&&s| s == 1).count();
    assert_eq!(ibd_count, 0, "all non-IBD data should produce no IBD states");
}

#[test]
fn roundtrip_all_ibd() {
    let params = clear_signal_params();
    let ground_truth = vec![1_usize; 100];
    let obs = generate_observations(&ground_truth, 0.998, 0.9999, 42);

    let states = viterbi(&obs, &params);

    // Most should be IBD (allowing some boundary effects)
    let ibd_count = states.iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count >= 90,
        "all IBD data should produce mostly IBD states, got {}/100",
        ibd_count
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: single IBD segment in the middle
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_single_ibd_segment_center() {
    let params = clear_signal_params();

    // 20 non-IBD, 60 IBD, 20 non-IBD
    let mut ground_truth = vec![0_usize; 20];
    ground_truth.extend(vec![1_usize; 60]);
    ground_truth.extend(vec![0_usize; 20]);

    let obs = generate_observations(&ground_truth, 0.998, 0.9999, 17);
    let states = viterbi(&obs, &params);

    // Count IBD windows in the middle region (indices 25-75)
    let core_ibd: usize = states[25..75].iter().filter(|&&s| s == 1).count();
    assert!(
        core_ibd >= 45,
        "core IBD region should be mostly recovered, got {}/50",
        core_ibd
    );

    // Count non-IBD in flanking regions
    let left_non_ibd: usize = states[0..15].iter().filter(|&&s| s == 0).count();
    let right_non_ibd: usize = states[85..100].iter().filter(|&&s| s == 0).count();
    assert!(left_non_ibd >= 12, "left flank should be mostly non-IBD: {}/15", left_non_ibd);
    assert!(right_non_ibd >= 12, "right flank should be mostly non-IBD: {}/15", right_non_ibd);
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: two separate IBD segments
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_two_ibd_segments() {
    let params = clear_signal_params();

    // Pattern: 20 non-IBD, 30 IBD, 20 non-IBD, 30 IBD, 20 non-IBD
    let mut gt = Vec::new();
    gt.extend(vec![0_usize; 20]);
    gt.extend(vec![1_usize; 30]);
    gt.extend(vec![0_usize; 20]);
    gt.extend(vec![1_usize; 30]);
    gt.extend(vec![0_usize; 20]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 7);
    let states = viterbi(&obs, &params);

    let segments = extract_ibd_segments(&states);

    // Should detect at least 2 segments (could be more if noise causes splits)
    assert!(
        segments.len() >= 2,
        "should detect at least 2 IBD segments, got {}",
        segments.len()
    );

    // Segments should be non-overlapping and in order
    for pair in segments.windows(2) {
        assert!(pair[0].1 < pair[1].0, "segments should be non-overlapping");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: infer_ibd pipeline consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_infer_ibd_pipeline() {
    let params = clear_signal_params();

    let mut gt = vec![0_usize; 30];
    gt.extend(vec![1_usize; 40]);
    gt.extend(vec![0_usize; 30]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 33);
    let result = infer_ibd(&obs, &params);

    // States and posteriors must have same length as observations
    assert_eq!(result.states.len(), obs.len());
    assert_eq!(result.posteriors.len(), obs.len());

    // Log-likelihood should be finite
    assert!(result.log_likelihood.is_finite(), "log-likelihood must be finite");

    // Posteriors should be in [0, 1]
    for &p in &result.posteriors {
        assert!((0.0..=1.0).contains(&p), "posterior {} not in [0,1]", p);
    }

    // Extract segments with posteriors
    let segments = extract_ibd_segments_with_posteriors(
        &result.states,
        &result.posteriors,
        3,   // min 3 windows
        0.5, // min 50% posterior
    );

    // Should find at least one IBD segment if any windows were called IBD
    let has_ibd = result.states.iter().any(|&s| s == 1);
    if has_ibd {
        assert!(
            !segments.is_empty(),
            "pipeline should produce segments when IBD windows exist"
        );

        for seg in &segments {
            assert!(seg.n_windows >= 3, "segment must have >= 3 windows");
            assert!(seg.mean_posterior >= 0.5, "segment posterior must be >= 0.5");
            assert!(seg.start_idx <= seg.end_idx, "start <= end");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: forward-backward posteriors match Viterbi direction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_posteriors_agree_with_viterbi() {
    let params = clear_signal_params();

    let mut gt = vec![0_usize; 30];
    gt.extend(vec![1_usize; 40]);
    gt.extend(vec![0_usize; 30]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 99);
    let states = viterbi(&obs, &params);
    let (posteriors, _ll) = forward_backward(&obs, &params);

    // Where Viterbi says IBD (state=1), posterior should be high
    // Where Viterbi says non-IBD (state=0), posterior should be low
    // Check the "core" regions (away from boundaries)
    for i in 0..obs.len() {
        if states[i] == 1 && i >= 35 && i <= 65 {
            assert!(
                posteriors[i] >= 0.5,
                "core IBD posterior at {} should be >= 0.5, got {}",
                i,
                posteriors[i]
            );
        }
        if states[i] == 0 && (i <= 20 || i >= 80) {
            assert!(
                posteriors[i] <= 0.5,
                "core non-IBD posterior at {} should be <= 0.5, got {}",
                i,
                posteriors[i]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: from_population constructor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_population_params_all_populations() {
    for pop in [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::Generic,
    ] {
        let params = HmmParams::from_population(pop, 50.0, 0.0001, 5000);

        // Generate clear IBD signal
        let mut gt = vec![0_usize; 20];
        gt.extend(vec![1_usize; 60]);
        gt.extend(vec![0_usize; 20]);

        let obs = generate_observations(&gt, params.emission[0].mean, params.emission[1].mean, 42);
        let states = viterbi(&obs, &params);

        // Core IBD region should be recovered
        let core_ibd: usize = states[30..70].iter().filter(|&&s| s == 1).count();
        assert!(
            core_ibd >= 30,
            "population {:?} should recover core IBD, got {}/40",
            pop,
            core_ibd
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: infer_ibd_with_training doesn't degrade results
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_training_preserves_or_improves_quality() {
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);

    let mut gt = vec![0_usize; 30];
    gt.extend(vec![1_usize; 40]);
    gt.extend(vec![0_usize; 30]);

    let obs = generate_observations(&gt, params.emission[0].mean, params.emission[1].mean, 55);

    // Inference without training
    let result_no_train = infer_ibd(&obs, &params);

    // Inference with training
    let result_with_train = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 10);

    // Both should produce valid output
    assert_eq!(result_no_train.states.len(), obs.len());
    assert_eq!(result_with_train.states.len(), obs.len());
    assert!(result_no_train.log_likelihood.is_finite());
    assert!(result_with_train.log_likelihood.is_finite());

    // Training should not make log-likelihood worse (or at most slightly)
    // (Baum-Welch is guaranteed to be non-decreasing in likelihood)
    // We allow a small tolerance for numerical precision
    assert!(
        result_with_train.log_likelihood >= result_no_train.log_likelihood - 1.0,
        "training should not significantly degrade log-likelihood: {} vs {}",
        result_with_train.log_likelihood,
        result_no_train.log_likelihood
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: extract_ibd_segments_with_lod
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_lod_scores_positive_for_true_ibd() {
    let params = clear_signal_params();

    let mut gt = vec![0_usize; 20];
    gt.extend(vec![1_usize; 60]);
    gt.extend(vec![0_usize; 20]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 88);
    let result = infer_ibd(&obs, &params);

    let segments = extract_ibd_segments_with_lod(
        &result.states,
        &result.posteriors,
        5,    // min 5 windows
        0.5,  // min 50% posterior
        Some((&obs, &params)),
        None, // no LOD threshold
    );

    // For clear IBD signal, LOD should be positive
    for seg in &segments {
        assert!(
            seg.lod_score > 0.0,
            "LOD score should be positive for true IBD, got {}",
            seg.lod_score
        );
    }
}

#[test]
fn roundtrip_lod_filtering_removes_weak_segments() {
    let params = clear_signal_params();

    let mut gt = vec![0_usize; 20];
    gt.extend(vec![1_usize; 60]);
    gt.extend(vec![0_usize; 20]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 77);
    let result = infer_ibd(&obs, &params);

    // Without LOD filter
    let segs_no_filter = extract_ibd_segments_with_lod(
        &result.states, &result.posteriors, 2, 0.3, Some((&obs, &params)), None,
    );

    // With high LOD filter
    let segs_high_lod = extract_ibd_segments_with_lod(
        &result.states, &result.posteriors, 2, 0.3, Some((&obs, &params)), Some(100.0),
    );

    // High LOD filter should produce <= segments than no filter
    assert!(
        segs_high_lod.len() <= segs_no_filter.len(),
        "LOD filtering should reduce or maintain segment count"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: short IBD segment (boundary of detectability)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_short_ibd_segment() {
    let params = clear_signal_params();

    // Very short IBD segment: 5 windows
    let mut gt = vec![0_usize; 50];
    gt.extend(vec![1_usize; 5]);
    gt.extend(vec![0_usize; 45]);

    let obs = generate_observations(&gt, 0.998, 0.9999, 11);
    let states = viterbi(&obs, &params);

    // Short segments are harder to detect; just verify no crash and valid output
    assert_eq!(states.len(), 100);
    for &s in &states {
        assert!(s == 0 || s == 1, "states must be 0 or 1");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: identical observations → deterministic output
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_deterministic_same_input() {
    let params = clear_signal_params();

    let obs = generate_observations(&vec![0; 50], 0.998, 0.9999, 42);

    let states1 = viterbi(&obs, &params);
    let states2 = viterbi(&obs, &params);
    assert_eq!(states1, states2, "Viterbi must be deterministic for same input");

    let (post1, ll1) = forward_backward(&obs, &params);
    let (post2, ll2) = forward_backward(&obs, &params);
    assert_eq!(ll1, ll2, "log-likelihood must be deterministic");
    assert_eq!(post1, post2, "posteriors must be deterministic");
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: extract_ibd_segments from state sequence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_segments_matches_known_pattern() {
    // Direct test: known state sequence → known segments
    let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
    let segments = extract_ibd_segments(&states);

    assert_eq!(segments.len(), 2);
    // First segment: indices 2-4 (3 windows)
    assert_eq!(segments[0], (2, 4, 3));
    // Second segment: indices 7-8 (2 windows)
    assert_eq!(segments[1], (7, 8, 2));
}

#[test]
fn extract_segments_single_window_ibd() {
    let states = vec![0, 0, 1, 0, 0];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (2, 2, 1));
}

#[test]
fn extract_segments_no_ibd() {
    let states = vec![0; 100];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 0);
}

#[test]
fn extract_segments_all_ibd() {
    let states = vec![1; 50];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (0, 49, 50));
}

#[test]
fn extract_segments_alternating() {
    let states: Vec<usize> = (0..10).map(|i| i % 2).collect();
    let segments = extract_ibd_segments(&states);
    // 1 at positions 1, 3, 5, 7, 9 → 5 single-window segments
    assert_eq!(segments.len(), 5);
    for seg in &segments {
        assert_eq!(seg.2, 1, "each alternating segment should be 1 window");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: empty and single-element inputs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_empty_observations() {
    let params = clear_signal_params();
    let obs: Vec<f64> = Vec::new();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 0);

    let (posteriors, _) = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 0);
}

#[test]
fn roundtrip_single_observation() {
    let params = clear_signal_params();
    let obs = vec![0.9999];

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] == 0 || states[0] == 1);

    let (posteriors, ll) = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 1);
    assert!(ll.is_finite());
    assert!((0.0..=1.0).contains(&posteriors[0]));
}

// ═══════════════════════════════════════════════════════════════════════════
// Round-trip: strong signal => high concordance with ground truth
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_very_strong_signal_high_concordance() {
    // Very well-separated emissions → near-perfect recovery
    let params = HmmParams {
        initial: [0.9, 0.1],
        transition: [[0.99, 0.01], [0.02, 0.98]],
        emission: [
            GaussianParams { mean: 0.990, std: 0.001 },
            GaussianParams { mean: 0.9999, std: 0.0001 },
        ],
    };

    let mut gt = vec![0_usize; 30];
    gt.extend(vec![1_usize; 40]);
    gt.extend(vec![0_usize; 30]);

    let obs = generate_observations(&gt, 0.990, 0.9999, 42);
    let states = viterbi(&obs, &params);

    // Count concordance
    let concordant: usize = states.iter().zip(gt.iter()).filter(|(a, b)| a == b).count();
    let concordance = concordant as f64 / gt.len() as f64;
    assert!(
        concordance >= 0.90,
        "strong signal should give >= 90% concordance, got {:.1}%",
        concordance * 100.0
    );
}
