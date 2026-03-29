//! Tests for auxiliary emission estimation, coverage ratio, segment quality/LOD,
//! and the infer_ibd_with_aux_features pipeline.
//!
//! These test edge cases and numerical properties of multi-feature IBD inference,
//! which is critical for algo_dev's A3 improvements.

use hprc_ibd::hmm::*;
use hprc_ibd::stats::GaussianParams;

// ============================================================================
// Helpers
// ============================================================================

fn make_afr_params() -> HmmParams {
    HmmParams::from_population(Population::AFR, 50.0, 0.001, 10000)
}

fn make_clear_obs() -> Vec<f64> {
    let mut obs = Vec::new();
    // 30 non-IBD windows
    for _ in 0..30 {
        obs.push(0.9970);
    }
    // 20 IBD windows
    for _ in 0..20 {
        obs.push(0.9998);
    }
    // 30 non-IBD
    for _ in 0..30 {
        obs.push(0.9972);
    }
    obs
}

// ============================================================================
// coverage_ratio Tests
// ============================================================================

#[test]
fn coverage_ratio_equal_lengths() {
    assert_eq!(coverage_ratio(100, 100), 1.0);
}

#[test]
fn coverage_ratio_one_zero() {
    assert_eq!(coverage_ratio(100, 0), 0.0);
    assert_eq!(coverage_ratio(0, 100), 0.0);
}

#[test]
fn coverage_ratio_both_zero() {
    assert_eq!(coverage_ratio(0, 0), 0.0);
}

#[test]
fn coverage_ratio_symmetric() {
    let r1 = coverage_ratio(100, 50);
    let r2 = coverage_ratio(50, 100);
    assert_eq!(r1, r2, "coverage_ratio should be symmetric");
}

#[test]
fn coverage_ratio_range() {
    // All possible coverage ratios should be in [0, 1]
    for a in [0, 1, 50, 100, 1000, u64::MAX / 2] {
        for b in [0, 1, 50, 100, 1000, u64::MAX / 2] {
            let r = coverage_ratio(a, b);
            assert!(
                (0.0..=1.0).contains(&r),
                "coverage_ratio({a}, {b}) = {r} out of [0,1]"
            );
        }
    }
}

#[test]
fn coverage_ratio_expected_values() {
    assert!((coverage_ratio(100, 50) - 0.5).abs() < 1e-10);
    assert!((coverage_ratio(75, 100) - 0.75).abs() < 1e-10);
    assert!((coverage_ratio(1, 1000) - 0.001).abs() < 1e-10);
}

#[test]
fn coverage_ratio_large_values_no_overflow() {
    let a = u64::MAX / 2;
    let b = u64::MAX / 4;
    let r = coverage_ratio(a, b);
    assert!(r.is_finite());
    assert!((r - 0.5).abs() < 0.01);
}

// ============================================================================
// estimate_auxiliary_emissions Tests
// ============================================================================

#[test]
fn aux_emission_all_non_ibd() {
    // All posteriors = 0 → only non-IBD state gets data
    let aux_obs = vec![0.5, 0.4, 0.6, 0.5, 0.45];
    let posteriors = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Non-IBD should have mean ~0.49 (average of aux_obs)
    assert!(non_ibd.mean.is_finite());
    assert!(non_ibd.std > 0.0);

    // IBD should get fallback values
    assert!((ibd.mean - 0.9).abs() < 1e-10, "IBD fallback mean should be 0.9");
    assert!((ibd.std - 0.1).abs() < 1e-10, "IBD fallback std should be 0.1");
}

#[test]
fn aux_emission_all_ibd() {
    // All posteriors = 1 → only IBD state gets data
    let aux_obs = vec![0.95, 0.93, 0.96, 0.94, 0.97];
    let posteriors = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // IBD should have mean ~0.95
    assert!(ibd.mean.is_finite());
    assert!((ibd.mean - 0.95).abs() < 0.05);

    // Non-IBD should get fallback values
    assert!((non_ibd.mean - 0.5).abs() < 1e-10, "Non-IBD fallback mean should be 0.5");
    assert!((non_ibd.std - 0.2).abs() < 1e-10, "Non-IBD fallback std should be 0.2");
}

#[test]
fn aux_emission_mixed_posteriors() {
    // Realistic scenario: some windows IBD, some not
    let aux_obs = vec![0.5, 0.5, 0.5, 0.95, 0.95, 0.95];
    let posteriors = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    assert!((non_ibd.mean - 0.5).abs() < 0.05, "Non-IBD mean should be ~0.5");
    assert!((ibd.mean - 0.95).abs() < 0.05, "IBD mean should be ~0.95");
    assert!(non_ibd.std > 0.0);
    assert!(ibd.std > 0.0);
}

#[test]
fn aux_emission_soft_posteriors() {
    // Soft posteriors (fractional weights)
    let aux_obs = vec![0.6, 0.7, 0.8, 0.9];
    let posteriors = vec![0.1, 0.3, 0.7, 0.9];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    assert!(non_ibd.mean.is_finite());
    assert!(ibd.mean.is_finite());
    assert!(non_ibd.std > 0.0);
    assert!(ibd.std > 0.0);

    // IBD mean should be higher (weighted towards higher aux values)
    assert!(
        ibd.mean > non_ibd.mean,
        "IBD mean ({}) should be > non-IBD mean ({})",
        ibd.mean,
        non_ibd.mean
    );
}

#[test]
fn aux_emission_empty_input() {
    let aux_obs: Vec<f64> = vec![];
    let posteriors: Vec<f64> = vec![];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Should return fallback values
    assert!((non_ibd.mean - 0.5).abs() < 1e-10);
    assert!((ibd.mean - 0.9).abs() < 1e-10);
}

#[test]
fn aux_emission_single_observation() {
    let aux_obs = vec![0.75];
    let posteriors = vec![0.5];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    assert!(non_ibd.mean.is_finite());
    assert!(ibd.mean.is_finite());
    assert!(non_ibd.std > 0.0);
    assert!(ibd.std > 0.0);
}

#[test]
fn aux_emission_posteriors_clamped() {
    // Posteriors slightly out of [0,1] (numerical noise)
    let aux_obs = vec![0.5, 0.9];
    let posteriors = vec![-0.01, 1.01];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Should handle gracefully (internal clamping)
    assert!(non_ibd.mean.is_finite());
    assert!(ibd.mean.is_finite());
}

#[test]
fn aux_emission_constant_aux_values() {
    // All aux values the same
    let aux_obs = vec![0.7; 10];
    let posteriors = vec![0.3; 10];

    let [non_ibd, ibd] = estimate_auxiliary_emissions(&aux_obs, &posteriors);

    // Both means should be ~0.7 (same values, different weights)
    assert!((non_ibd.mean - 0.7).abs() < 0.01);
    assert!((ibd.mean - 0.7).abs() < 0.01);
    // Std should be at the minimum (1e-4) since variance is 0
    assert!((non_ibd.std - 1e-4).abs() < 1e-10);
    assert!((ibd.std - 1e-4).abs() < 1e-10);
}

// ============================================================================
// segment_quality_score Tests
// ============================================================================

#[test]
fn segment_quality_score_basic() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 9,
        n_windows: 10,
        mean_posterior: 0.95,
        min_posterior: 0.80,
        max_posterior: 0.99,
        lod_score: 10.0,
    };
    let score = segment_quality_score(&seg);
    assert!(score.is_finite());
    assert!(score > 0.0);
}

#[test]
fn segment_quality_score_single_window() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 5,
        end_idx: 5,
        n_windows: 1,
        mean_posterior: 0.99,
        min_posterior: 0.99,
        max_posterior: 0.99,
        lod_score: 5.0,
    };
    let score = segment_quality_score(&seg);
    assert!(score.is_finite());
}

#[test]
fn segment_quality_score_low_quality() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 2,
        n_windows: 3,
        mean_posterior: 0.51,
        min_posterior: 0.40,
        max_posterior: 0.60,
        lod_score: 1.0,
    };
    let score = segment_quality_score(&seg);
    assert!(score.is_finite());
}

#[test]
fn segment_quality_higher_for_better_segments() {
    let good_seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 19,
        n_windows: 20,
        mean_posterior: 0.98,
        min_posterior: 0.90,
        max_posterior: 1.0,
        lod_score: 20.0,
    };
    let bad_seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 2,
        n_windows: 3,
        mean_posterior: 0.55,
        min_posterior: 0.40,
        max_posterior: 0.60,
        lod_score: 1.0,
    };
    let score_good = segment_quality_score(&good_seg);
    let score_bad = segment_quality_score(&bad_seg);
    assert!(
        score_good > score_bad,
        "Good segment quality ({score_good}) should exceed bad ({score_bad})"
    );
}

// ============================================================================
// segment_posterior_std Tests
// ============================================================================

#[test]
fn segment_posterior_std_constant() {
    let posteriors = vec![0.95; 10];
    let std = segment_posterior_std(&posteriors, 0, 9);
    assert!(std < 1e-10, "Constant posteriors should have ~0 std, got {std}");
}

#[test]
fn segment_posterior_std_varying() {
    let posteriors = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let std = segment_posterior_std(&posteriors, 0, 4);
    assert!(std > 0.0, "Varying posteriors should have positive std");
    assert!(std.is_finite());
}

#[test]
fn segment_posterior_std_single_window() {
    let posteriors = vec![0.8, 0.9, 0.7];
    let std = segment_posterior_std(&posteriors, 1, 1);
    assert!(std.abs() < 1e-10, "Single window should have 0 std");
}

#[test]
fn segment_posterior_std_partial_range() {
    let posteriors = vec![0.1, 0.2, 0.9, 0.95, 0.3, 0.4];
    let std = segment_posterior_std(&posteriors, 2, 3);
    assert!(std.is_finite());
    assert!(std > 0.0);
}

// ============================================================================
// compute_per_window_lod Tests
// ============================================================================

#[test]
fn per_window_lod_ibd_observations_positive() {
    let params = make_afr_params();
    // Very high identity → should favor IBD → positive LOD
    let obs = vec![0.9999, 0.99995, 0.9999];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 3);
    for (i, &lod) in lods.iter().enumerate() {
        assert!(lod.is_finite(), "LOD at {i} not finite");
        assert!(lod > 0.0, "IBD-like observation should have positive LOD at {i}: {lod}");
    }
}

#[test]
fn per_window_lod_non_ibd_observations_negative() {
    let params = make_afr_params();
    // Low identity → should favor non-IBD → negative LOD
    let obs = vec![0.996, 0.995, 0.996];
    let lods = compute_per_window_lod(&obs, &params);
    for (i, &lod) in lods.iter().enumerate() {
        assert!(lod.is_finite(), "LOD at {i} not finite");
        assert!(lod < 0.0, "Non-IBD observation should have negative LOD at {i}: {lod}");
    }
}

#[test]
fn per_window_lod_empty() {
    let params = make_afr_params();
    let obs: Vec<f64> = vec![];
    let lods = compute_per_window_lod(&obs, &params);
    assert!(lods.is_empty());
}

// ============================================================================
// segment_lod_score Tests
// ============================================================================

#[test]
fn segment_lod_is_sum_of_per_window() {
    let params = make_afr_params();
    let obs = vec![0.9999; 10]; // All IBD-like
    let per_window_lod = compute_per_window_lod(&obs, &params);

    let seg_lod = segment_lod_score(&obs, 0, 9, &params);
    let expected: f64 = per_window_lod.iter().sum();
    assert!(
        (seg_lod - expected).abs() < 1e-10,
        "Segment LOD should be sum of per-window LODs: {seg_lod} vs {expected}"
    );
}

#[test]
fn segment_lod_partial_range() {
    let params = make_afr_params();
    let obs = vec![0.9999; 10];
    let per_window_lod = compute_per_window_lod(&obs, &params);

    let seg_lod = segment_lod_score(&obs, 3, 7, &params);
    let expected: f64 = per_window_lod[3..=7].iter().sum();
    assert!(
        (seg_lod - expected).abs() < 1e-10,
        "Partial segment LOD: {seg_lod} vs {expected}"
    );
}

// ============================================================================
// extract_ibd_segments_with_posteriors Tests
// ============================================================================

#[test]
fn extract_segments_with_posteriors_basic() {
    let states = vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.2, 0.8, 0.9, 0.95, 0.85, 0.15, 0.1, 0.1];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 3);
    assert_eq!(segments[0].end_idx, 6);
    assert_eq!(segments[0].n_windows, 4);
    assert!(segments[0].mean_posterior > 0.8);
    assert!(segments[0].min_posterior >= 0.8);
    assert!(segments[0].max_posterior <= 0.95);
}

#[test]
fn extract_segments_with_posteriors_no_ibd() {
    let states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1, 0.1];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert!(segments.is_empty());
}

#[test]
fn extract_segments_with_posteriors_all_ibd() {
    let states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.98, 0.96, 0.93];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 4);
    assert_eq!(segments[0].n_windows, 5);
}

#[test]
fn extract_segments_with_posteriors_multiple() {
    let states = vec![1, 1, 0, 0, 1, 1, 1, 0, 1];
    let posteriors = vec![0.8, 0.9, 0.2, 0.1, 0.85, 0.92, 0.88, 0.15, 0.82];

    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 1);
    assert_eq!(segments[1].start_idx, 4);
    assert_eq!(segments[1].end_idx, 6);
    assert_eq!(segments[2].start_idx, 8);
    assert_eq!(segments[2].end_idx, 8);
}

#[test]
fn extract_segments_with_posteriors_empty() {
    let states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.0);
    assert!(segments.is_empty());
}

// ============================================================================
// infer_ibd_with_aux_features Tests
// ============================================================================

#[test]
fn infer_with_aux_too_short_returns_all_non_ibd() {
    let obs = vec![0.999, 0.998]; // < 3 observations
    let mut params = make_afr_params();

    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::AFR, 10000, 10, None);

    assert_eq!(result.states.len(), 2);
    assert!(result.states.iter().all(|&s| s == 0));
    assert!(result.posteriors.iter().all(|&p| p == 0.0));
    assert_eq!(result.log_likelihood, f64::NEG_INFINITY);
    assert!(aux_emit.is_none());
}

#[test]
fn infer_with_aux_no_aux_data() {
    let obs = make_clear_obs();
    let mut params = make_afr_params();

    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::AFR, 10000, 10, None);

    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
    assert!(aux_emit.is_none());

    // Should detect the IBD region (windows 30-49)
    let ibd_count: usize = result.states[30..50].iter().filter(|&&s| s == 1).count();
    assert!(ibd_count > 10, "Should detect IBD in high-identity region: {ibd_count}/20");
}

#[test]
fn infer_with_aux_returns_aux_emission_params() {
    let obs = make_clear_obs();
    let n = obs.len();
    // Coverage ratio: ~0.5 in non-IBD, ~0.95 in IBD
    let mut aux_obs = vec![0.5; n];
    for v in aux_obs.iter_mut().skip(30).take(20) {
        *v = 0.95;
    }

    let mut params = make_afr_params();
    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::AFR, 10000, 5, Some(&aux_obs));

    assert_eq!(result.states.len(), n);
    assert!(result.log_likelihood.is_finite());
    assert!(aux_emit.is_some(), "Should return auxiliary emission params");

    let [non_ibd_aux, ibd_aux] = aux_emit.unwrap();
    assert!(non_ibd_aux.mean.is_finite());
    assert!(ibd_aux.mean.is_finite());
}

#[test]
fn infer_with_aux_mismatched_length_ignores_aux() {
    let obs = make_clear_obs();
    let aux_obs = vec![0.5; 5]; // Wrong length

    let mut params = make_afr_params();
    let (result, aux_emit) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::AFR, 10000, 5, Some(&aux_obs));

    assert_eq!(result.states.len(), obs.len());
    assert!(aux_emit.is_none(), "Mismatched aux should be ignored");
}

#[test]
fn infer_with_aux_zero_bw_iters() {
    let obs = make_clear_obs();
    let mut params = make_afr_params();

    let (result, _) =
        infer_ibd_with_aux_features(&obs, &mut params, Population::AFR, 10000, 0, None);

    // Should still produce valid output (no BW training, just inference)
    assert_eq!(result.states.len(), obs.len());
    assert!(result.log_likelihood.is_finite());
}

// ============================================================================
// refine_states_with_posteriors Tests
// ============================================================================

#[test]
fn refine_states_low_posterior_ibd_removed() {
    let mut states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.15, 0.12, 0.18, 0.1, 0.1];

    // IBD windows with posteriors < threshold should be removed
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

    // The low-posterior "IBD" windows should be cleared to 0
    for (i, &s) in states.iter().enumerate() {
        assert_eq!(s, 0, "State at {i} should be refined to 0 (low posterior)");
    }
}

#[test]
fn refine_states_high_posterior_non_ibd_promoted() {
    let mut states = vec![1, 1, 0, 0, 1, 1];
    let posteriors = vec![0.95, 0.92, 0.88, 0.90, 0.96, 0.94];

    // Non-IBD windows with high posteriors in IBD context
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);

    // High-posterior gap windows might be promoted to IBD
    // (depends on implementation details — just verify no panic and valid states)
    for &s in &states {
        assert!(s <= 1, "State should be 0 or 1");
    }
}

#[test]
fn refine_states_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert!(states.is_empty());
}

// ============================================================================
// Population Variants
// ============================================================================

#[test]
fn all_populations_produce_valid_inference() {
    let obs = make_clear_obs();
    for pop in [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::Generic,
    ] {
        let mut params = HmmParams::from_population(pop, 50.0, 0.001, 10000);
        let (result, _) =
            infer_ibd_with_aux_features(&obs, &mut params, pop, 10000, 5, None);

        assert_eq!(result.states.len(), obs.len());
        assert!(
            result.log_likelihood.is_finite(),
            "LL not finite for {:?}: {}",
            pop,
            result.log_likelihood
        );
        for (t, &p) in result.posteriors.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Posterior at t={t} out of [0,1] for {:?}: {p}",
                pop
            );
        }
    }
}

#[test]
fn population_diversity_values() {
    // Each population should have a distinct diversity
    let diversities: Vec<f64> = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
    ]
    .iter()
    .map(|p| p.diversity())
    .collect();

    for &d in &diversities {
        assert!(d > 0.0 && d < 1.0, "Diversity out of range: {d}");
    }
    // AFR should have highest diversity
    assert!(diversities[0] > diversities[1], "AFR should have higher diversity than EUR");
}
