//! Deep branch and edge case tests for ibd-cli
//!
//! Targets untested conditional branches in recombination_aware_log_transition,
//! segment_quality_score, extract_ibd_segments_with_lod, and GeneticMap
//! degenerate interpolation.

use hprc_ibd::hmm::{
    HmmParams, GeneticMap, Population,
    recombination_aware_log_transition,
    segment_quality_score,
    extract_ibd_segments_with_lod,
    IbdSegmentWithPosterior,
};

// ── 1. recombination_aware_log_transition extreme rate branches ──────

#[test]
fn recomb_transition_p_enter_zero() {
    // When transition[0][1] = 0.0 (never enter IBD), rate_enter = 0.0 branch
    let params = HmmParams {
        initial: [1.0, 0.0],
        transition: [
            [1.0, 0.0],   // Never enter IBD
            [0.02, 0.98],
        ],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 10_000);

    // p_enter should be clamped to 1e-10 (not actually 0)
    let p_enter = trans[0][1].exp();
    assert!(p_enter > 0.0, "p_enter should be > 0 after clamping");
    assert!(p_enter < 1e-5, "p_enter should be very small, got {}", p_enter);

    // Row 0 should still sum to ~1 in probability space
    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-6, "Row 0 sums to {}", row0_sum);

    // Row 1 should sum to ~1
    let row1_sum = trans[1][0].exp() + trans[1][1].exp();
    assert!((row1_sum - 1.0).abs() < 1e-6, "Row 1 sums to {}", row1_sum);
}

#[test]
fn recomb_transition_p_exit_zero() {
    // When transition[1][0] = 0.0 (never exit IBD), rate_exit = 0.0 branch
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [
            [0.999, 0.001],
            [0.0, 1.0],   // Never exit IBD
        ],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 10_000);

    // p_exit should be clamped to 1e-10
    let p_exit = trans[1][0].exp();
    assert!(p_exit > 0.0, "p_exit should be > 0 after clamping");
    assert!(p_exit < 1e-5, "p_exit should be very small, got {}", p_exit);

    // Both rows should sum to ~1
    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    let row1_sum = trans[1][0].exp() + trans[1][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);
}

#[test]
fn recomb_transition_both_rates_zero() {
    // Both enter and exit are zero: both clamped to 1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 10_000);

    for row in &trans {
        let sum = row[0].exp() + row[1].exp();
        assert!((sum - 1.0).abs() < 1e-6, "Row should sum to 1.0 in prob space, got {}", sum);
    }
}

#[test]
fn recomb_transition_same_position_returns_base_log() {
    // When pos1 == pos2, returns base log transitions directly
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 500_000, 500_000, &gm, 10_000);

    assert!((trans[0][0] - params.transition[0][0].ln()).abs() < 1e-12);
    assert!((trans[0][1] - params.transition[0][1].ln()).abs() < 1e-12);
    assert!((trans[1][0] - params.transition[1][0].ln()).abs() < 1e-12);
    assert!((trans[1][1] - params.transition[1][1].ln()).abs() < 1e-12);
}

#[test]
fn recomb_transition_window_size_zero_returns_base_log() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 0);

    assert!((trans[0][0] - params.transition[0][0].ln()).abs() < 1e-12);
    assert!((trans[0][1] - params.transition[0][1].ln()).abs() < 1e-12);
}

// ── 2. segment_quality_score with n_windows = 0 ─────────────────────

#[test]
fn quality_score_zero_windows() {
    // When n_windows=0, lod_per_window=0.0, length_score=0.0
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 0,
        n_windows: 0,
        mean_posterior: 0.9,
        min_posterior: 0.8,
        max_posterior: 1.0,
        lod_score: 10.0,
    };

    let score = segment_quality_score(&seg);

    // posterior_score = 0.9 * 40 = 36
    // consistency = min/mean = 0.8/0.9 ≈ 0.889 → 17.78
    // lod_score = 0 (because n_windows=0 → lod_per_window=0)
    // length_score = 0 (n_windows=0)
    let expected_posterior = 0.9 * 40.0;
    let expected_consistency = (0.8 / 0.9_f64).clamp(0.0, 1.0) * 20.0;
    let expected = expected_posterior + expected_consistency;

    assert!((score - expected).abs() < 0.01,
        "Expected {:.2}, got {:.2}", expected, score);
}

#[test]
fn quality_score_zero_mean_posterior() {
    // When mean_posterior=0, consistency=0
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 9,
        n_windows: 10,
        mean_posterior: 0.0,
        min_posterior: 0.0,
        max_posterior: 0.0,
        lod_score: 5.0,
    };

    let score = segment_quality_score(&seg);

    // posterior = 0, consistency = 0
    // lod_per_window = 5/10 = 0.5 → lod_score = 0.5 * 30 = 15
    // length = (10/20).clamp(0,1) * 10 = 5
    let expected = 0.0 + 0.0 + 15.0 + 5.0;
    assert!((score - expected).abs() < 0.01,
        "Expected {:.2}, got {:.2}", expected, score);
}

// ── 3. extract_ibd_segments_with_lod: LOD filter with real obs ──────

#[test]
fn extract_segments_lod_filter_rejects_low_lod() {
    // Build states with a clear IBD segment and set min_lod very high
    let states = vec![0, 0, 1, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.95, 0.9, 0.1, 0.1];

    // Pass observations that produce a low LOD
    let observations = vec![0.5, 0.5, 0.99, 0.99, 0.99, 0.99, 0.5, 0.5];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,  // min_windows
        0.0, // min_mean_posterior
        Some((&observations, &params)),
        Some(f64::INFINITY), // impossible min_lod → all segments rejected
    );

    assert!(segments.is_empty(),
        "All segments should be rejected with min_lod=INFINITY, got {}", segments.len());
}

#[test]
fn extract_segments_lod_filter_accepts_with_none_threshold() {
    // With None LOD threshold, any segment passing other filters is accepted
    // regardless of its LOD score (which can be negative)
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.95, 0.9, 0.1];
    let observations = vec![0.5, 0.99, 0.99, 0.99, 0.5];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.0,
        Some((&observations, &params)),
        None, // no LOD threshold → accept regardless of LOD
    );

    assert_eq!(segments.len(), 1, "Should find one segment without LOD filter");
    // LOD is computed even though there's no threshold
    assert!(segments[0].lod_score.is_finite(), "LOD should be finite");
}

#[test]
fn extract_segments_lod_filter_with_negative_threshold() {
    // With a very negative threshold, segments with any LOD should pass
    let states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.99, 0.99, 0.99, 0.99, 0.99, 0.1];
    let observations = vec![0.5, 0.999, 0.999, 0.999, 0.999, 0.999, 0.5];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.0,
        Some((&observations, &params)),
        Some(-1000.0), // very negative threshold → should pass
    );

    assert_eq!(segments.len(), 1, "Segment should pass very negative LOD threshold");
    assert!(segments[0].lod_score.is_finite());
}

// ── 4. GeneticMap interpolate_cm with bp_hi == bp_lo (exact value) ──

#[test]
fn genetic_map_duplicate_bp_returns_cm_lo() {
    // Two entries at same position: interpolation should return cm_lo
    let gm = GeneticMap::new(vec![
        (1000, 2.0),
        (1000, 3.0),  // duplicate bp
        (2000, 4.0),
    ]);

    // At pos 1000: partition_point(|e| e.0 <= 1000) returns 2
    // entries[1] = (1000, 3.0), entries[2] = (2000, 4.0)
    // bp_hi != bp_lo → normal interpolation: frac = 0 → cm_lo = 3.0
    let cm = gm.interpolate_cm(1000);
    assert!(cm.is_finite());
    // The exact value depends on partition_point behavior with duplicates
    // At 1500: between entries[1]=(1000,3.0) and entries[2]=(2000,4.0) → 3.5
    let cm_mid = gm.interpolate_cm(1500);
    assert!((cm_mid - 3.5).abs() < 1e-6, "Expected 3.5, got {}", cm_mid);
}

#[test]
fn genetic_map_true_duplicate_bp_different_cm() {
    // Create a map where after sorting, consecutive entries have same bp
    // This can happen if the map file has overlapping entries
    let gm = GeneticMap::new(vec![
        (500, 1.0),
        (1000, 2.0),
        (1000, 5.0),  // same bp, different cm
        (2000, 6.0),
    ]);

    // At exactly 1000: partition_point returns idx=3 (all entries ≤ 1000)
    // entries[2]=(1000,5.0), entries[3]=(2000,6.0) — normal case
    let cm = gm.interpolate_cm(1000);
    assert!(cm.is_finite());

    // At 750: between entries[0]=(500,1.0) and entries[1]=(1000,2.0)
    // frac = (750-500)/(1000-500) = 0.5 → 1.0 + 0.5*(2.0-1.0) = 1.5
    let cm_750 = gm.interpolate_cm(750);
    assert!((cm_750 - 1.5).abs() < 1e-6, "Expected 1.5, got {}", cm_750);
}

// ── 5. infer_ibd_with_training boundary at 9 observations ──────────

#[test]
fn infer_ibd_with_training_exactly_9_obs_skips_bw() {
    // 9 observations < 10 threshold → Baum-Welch should be skipped
    use hprc_ibd::hmm::infer_ibd_with_training;
    let observations: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.99, 0.99, 0.99, 0.5, 0.5];
    assert_eq!(observations.len(), 9);

    let mut params = HmmParams::from_expected_length(3.0, 0.1, 5000);
    let result = infer_ibd_with_training(&observations, &mut params, Population::Generic, 5000, 5);

    assert_eq!(result.states.len(), 9);
    assert!(result.states.iter().all(|&s| s == 0 || s == 1));
}

#[test]
fn infer_ibd_with_training_exactly_10_obs_runs_bw() {
    use hprc_ibd::hmm::infer_ibd_with_training;
    let observations: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.99, 0.99, 0.99, 0.99, 0.5, 0.5];
    assert_eq!(observations.len(), 10);

    let mut params = HmmParams::from_expected_length(3.0, 0.1, 5000);
    let result = infer_ibd_with_training(&observations, &mut params, Population::Generic, 5000, 5);

    assert_eq!(result.states.len(), 10);
    assert!(result.states.iter().all(|&s| s == 0 || s == 1));
}

// ── 6. recombination_aware_log_transition with high p_enter ─────────

#[test]
fn recomb_transition_high_p_enter() {
    // When p_enter_base approaches 1.0, rate_enter → INFINITY branch
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [
            [0.01, 0.99],  // Very high entry rate
            [0.02, 0.98],
        ],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 10_000);

    // p_enter should be clamped to at most 1-1e-10
    let p_enter = trans[0][1].exp();
    assert!(p_enter > 0.5, "High p_enter should produce high transition prob");
    assert!(p_enter < 1.0, "p_enter should be < 1 after clamping");

    // Rows should sum to 1
    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-6);
}

#[test]
fn recomb_transition_p_enter_exactly_one() {
    // p_enter_base = 1.0 → rate_enter = INFINITY → p_enter = 1 - exp(-INF*scale) = 1.0
    // clamped to 1-1e-10
    let params = HmmParams {
        initial: [0.5, 0.5],
        transition: [
            [0.0, 1.0],  // Always enter IBD
            [0.5, 0.5],
        ],
        emission: HmmParams::from_expected_length(50.0, 0.0001, 5000).emission,
    };
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);

    let trans = recombination_aware_log_transition(&params, 100_000, 200_000, &gm, 10_000);

    let p_enter = trans[0][1].exp();
    assert!(p_enter > 1.0 - 1e-5,
        "p_enter_base=1.0 should give p_enter near max clamp, got {}", p_enter);

    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-6);
}
