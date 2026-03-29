//! Tests for raw observation smoothing, SNR-weighted emission smoothing, and contrast normalization.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    smooth_log_emissions, smooth_log_emissions_weighted, contrast_normalize_emissions,
    smooth_observations, AncestryObservation,
};

fn make_obs(start: u64, sims: Vec<(&str, f64)>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr12".to_string(),
        start,
        end: start + 10000,
        sample: "test#1".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// === smooth_observations tests (pre-emission similarity smoothing) ===

#[test]
fn test_smooth_observations_empty() {
    let result = smooth_observations(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn test_smooth_observations_context_zero() {
    let obs = vec![
        make_obs(0, vec![("A", 0.999), ("B", 0.998)]),
        make_obs(10000, vec![("A", 0.997), ("B", 0.996)]),
    ];
    let result = smooth_observations(&obs, 0);
    assert_eq!(result.len(), 2);
    assert!((result[0].similarities["A"] - 0.999).abs() < 1e-10);
    assert!((result[1].similarities["B"] - 0.996).abs() < 1e-10);
}

#[test]
fn test_smooth_observations_basic_averaging() {
    let obs = vec![
        make_obs(0, vec![("A", 0.990), ("B", 0.980)]),
        make_obs(10000, vec![("A", 0.996), ("B", 0.994)]),
        make_obs(20000, vec![("A", 0.992), ("B", 0.988)]),
    ];
    // context=1: window 1 averages windows 0,1,2
    let result = smooth_observations(&obs, 1);
    let expected_a = (0.990 + 0.996 + 0.992) / 3.0;
    let expected_b = (0.980 + 0.994 + 0.988) / 3.0;
    assert!((result[1].similarities["A"] - expected_a).abs() < 1e-10);
    assert!((result[1].similarities["B"] - expected_b).abs() < 1e-10);
}

#[test]
fn test_smooth_observations_boundary_clamping() {
    let obs = vec![
        make_obs(0, vec![("A", 1.0)]),
        make_obs(10000, vec![("A", 0.5)]),
        make_obs(20000, vec![("A", 0.0)]),
    ];
    // context=1: window 0 averages only windows 0,1 (clamped at left)
    let result = smooth_observations(&obs, 1);
    assert!((result[0].similarities["A"] - 0.75).abs() < 1e-10);
    // window 2 averages only windows 1,2 (clamped at right)
    assert!((result[2].similarities["A"] - 0.25).abs() < 1e-10);
}

#[test]
fn test_smooth_observations_preserves_metadata() {
    let obs = vec![
        make_obs(0, vec![("A", 0.999)]),
        make_obs(10000, vec![("A", 0.998)]),
    ];
    let result = smooth_observations(&obs, 1);
    assert_eq!(result[0].chrom, "chr12");
    assert_eq!(result[0].start, 0);
    assert_eq!(result[0].end, 10000);
    assert_eq!(result[0].sample, "test#1");
    assert_eq!(result[1].start, 10000);
}

#[test]
fn test_smooth_observations_single_window() {
    let obs = vec![make_obs(0, vec![("A", 0.999), ("B", 0.998)])];
    let result = smooth_observations(&obs, 5);
    assert_eq!(result.len(), 1);
    assert!((result[0].similarities["A"] - 0.999).abs() < 1e-10);
}

#[test]
fn test_smooth_observations_large_context() {
    let obs = vec![
        make_obs(0, vec![("A", 0.990)]),
        make_obs(10000, vec![("A", 0.992)]),
        make_obs(20000, vec![("A", 0.994)]),
    ];
    // context=100 >> n: all windows use full range
    let result = smooth_observations(&obs, 100);
    let expected = (0.990 + 0.992 + 0.994) / 3.0;
    for r in &result {
        assert!((r.similarities["A"] - expected).abs() < 1e-10);
    }
}

#[test]
fn test_smooth_observations_handles_missing_haplotypes() {
    // Haplotype B only appears in window 1
    let obs = vec![
        make_obs(0, vec![("A", 0.999)]),
        make_obs(10000, vec![("A", 0.998), ("B", 0.997)]),
        make_obs(20000, vec![("A", 0.996)]),
    ];
    let result = smooth_observations(&obs, 1);
    // Window 1 should have B averaged from just itself
    assert!(result[1].similarities.contains_key("B"));
    // Window 0 with context=1 includes windows 0,1: B only in window 1
    assert!(result[0].similarities.contains_key("B"));
    assert!((result[0].similarities["B"] - 0.997).abs() < 1e-10);
}

#[test]
fn test_smooth_observations_with_coverage_ratios() {
    let mut obs0 = make_obs(0, vec![("A", 0.999)]);
    obs0.coverage_ratios = Some([("A".to_string(), 0.8)].into_iter().collect());
    let mut obs1 = make_obs(10000, vec![("A", 0.998)]);
    obs1.coverage_ratios = Some([("A".to_string(), 0.9)].into_iter().collect());

    let result = smooth_observations(&[obs0, obs1], 1);
    assert!(result[0].coverage_ratios.is_some());
    let cov = result[0].coverage_ratios.as_ref().unwrap();
    assert!((cov["A"] - 0.85).abs() < 1e-10);
}

// === smooth_log_emissions_weighted tests ===

#[test]
fn test_weighted_smoothing_empty() {
    let result = smooth_log_emissions_weighted(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn test_weighted_smoothing_context_zero() {
    let emissions = vec![vec![-1.0, -2.0], vec![-1.5, -0.5]];
    let result = smooth_log_emissions_weighted(&emissions, 0);
    assert_eq!(result, emissions, "context=0 should be identity");
}

#[test]
fn test_weighted_smoothing_single_window() {
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let result = smooth_log_emissions_weighted(&emissions, 3);
    assert_eq!(result.len(), 1);
    // Single window: weight doesn't matter, result should equal input
    for (a, b) in result[0].iter().zip(&emissions[0]) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_weighted_smoothing_gives_more_weight_to_discriminative() {
    // Window 0: ambiguous (both states have similar emissions)
    // Window 1: very discriminative (large gap between states)
    // Window 2: ambiguous again
    let emissions = vec![
        vec![-1.0, -1.1],   // gap = 0.1 (ambiguous)
        vec![-0.5, -3.0],   // gap = 2.5 (very discriminative)
        vec![-1.0, -1.1],   // gap = 0.1 (ambiguous)
    ];

    let weighted = smooth_log_emissions_weighted(&emissions, 1);
    let uniform = smooth_log_emissions(&emissions, 1);

    // For the center window (t=1), the discriminative window dominates in weighted
    // but in uniform, all three windows contribute equally.
    // The weighted smoothing at t=0 should be pulled more toward window 1's values
    // (since window 1 has higher weight)
    let uniform_diff_t0 = (uniform[0][0] - uniform[0][1]).abs();
    let weighted_diff_t0 = (weighted[0][0] - weighted[0][1]).abs();

    // Weighted smoothing should show more discrimination at window 0
    // because it borrows more from the discriminative window 1
    assert!(weighted_diff_t0 > uniform_diff_t0,
        "weighted discrimination ({:.4}) should exceed uniform ({:.4})",
        weighted_diff_t0, uniform_diff_t0);
}

#[test]
fn test_weighted_smoothing_uniform_when_all_same_gap() {
    // All windows have the same gap → weights are equal → same as uniform
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
        vec![-1.0, -2.0],
    ];

    let weighted = smooth_log_emissions_weighted(&emissions, 1);
    let uniform = smooth_log_emissions(&emissions, 1);

    for t in 0..3 {
        for s in 0..2 {
            assert!((weighted[t][s] - uniform[t][s]).abs() < 1e-10,
                "equal gaps should give same result as uniform at [{t}][{s}]");
        }
    }
}

#[test]
fn test_weighted_smoothing_preserves_dimensions() {
    let emissions = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-1.5, -0.5, -2.0],
        vec![-2.0, -1.0, -1.5],
        vec![-0.5, -3.0, -1.0],
    ];
    let result = smooth_log_emissions_weighted(&emissions, 2);
    assert_eq!(result.len(), 4);
    for row in &result {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn test_weighted_smoothing_handles_neg_inf() {
    // One state has -inf in some windows (no data)
    let emissions = vec![
        vec![-1.0, f64::NEG_INFINITY],
        vec![-1.0, -2.0],
        vec![-1.0, f64::NEG_INFINITY],
    ];

    let result = smooth_log_emissions_weighted(&emissions, 1);
    assert_eq!(result.len(), 3);
    // State 0 should have finite values everywhere
    for row in &result {
        assert!(row[0].is_finite(), "state 0 should remain finite");
    }
}

#[test]
fn test_weighted_smoothing_large_context() {
    // Context larger than array length → uses full range
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-0.5, -2.5],
    ];
    let result = smooth_log_emissions_weighted(&emissions, 100);
    assert_eq!(result.len(), 2);
    // Both windows should use the same range (full array)
    // so both should have the same smoothed values
    for s in 0..2 {
        assert!((result[0][s] - result[1][s]).abs() < 1e-10,
            "with full-range context, all windows should have same smoothed value");
    }
}

// === contrast_normalize_emissions tests ===

#[test]
fn test_contrast_normalize_empty() {
    let result = contrast_normalize_emissions(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_contrast_normalize_centers_to_zero_mean() {
    let emissions = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-0.5, -1.5, -0.5],
    ];
    let normalized = contrast_normalize_emissions(&emissions);

    for (t, row) in normalized.iter().enumerate() {
        let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
        assert!(mean.abs() < 1e-10,
            "window {} mean should be ~0, got {:.6}", t, mean);
    }
}

#[test]
fn test_contrast_normalize_preserves_relative_order() {
    let emissions = vec![
        vec![-0.5, -1.0, -2.0], // state 0 is best
        vec![-2.0, -0.3, -1.0], // state 1 is best
    ];
    let normalized = contrast_normalize_emissions(&emissions);

    // Window 0: state 0 should still be best
    assert!(normalized[0][0] > normalized[0][1]);
    assert!(normalized[0][0] > normalized[0][2]);

    // Window 1: state 1 should still be best
    assert!(normalized[1][1] > normalized[1][0]);
    assert!(normalized[1][1] > normalized[1][2]);
}

#[test]
fn test_contrast_normalize_removes_global_offset() {
    // Two windows with same relative pattern but different global levels
    let emissions = vec![
        vec![-1.0, -2.0, -3.0], // level = -2.0
        vec![-5.0, -6.0, -7.0], // level = -6.0 (same pattern, shifted)
    ];
    let normalized = contrast_normalize_emissions(&emissions);

    // After normalization, both windows should have the same values
    for s in 0..3 {
        assert!((normalized[0][s] - normalized[1][s]).abs() < 1e-10,
            "same relative pattern should give same normalized values at state {}", s);
    }
}

#[test]
fn test_contrast_normalize_handles_neg_inf() {
    let emissions = vec![
        vec![-1.0, f64::NEG_INFINITY, -2.0],
    ];
    let normalized = contrast_normalize_emissions(&emissions);

    // -inf should remain -inf
    assert!(normalized[0][1] == f64::NEG_INFINITY);
    // Finite values should be centered (mean of -1.0 and -2.0 = -1.5)
    assert!((normalized[0][0] - 0.5).abs() < 1e-10);
    assert!((normalized[0][2] - (-0.5)).abs() < 1e-10);
}

#[test]
fn test_contrast_normalize_all_neg_inf() {
    let emissions = vec![
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
    ];
    let normalized = contrast_normalize_emissions(&emissions);
    // All -inf: no change
    assert!(normalized[0][0] == f64::NEG_INFINITY);
    assert!(normalized[0][1] == f64::NEG_INFINITY);
}

#[test]
fn test_contrast_normalize_single_state() {
    let emissions = vec![
        vec![-2.0],
        vec![-3.0],
    ];
    let normalized = contrast_normalize_emissions(&emissions);
    // Single state: mean = value, so normalized = 0
    assert!((normalized[0][0]).abs() < 1e-10);
    assert!((normalized[1][0]).abs() < 1e-10);
}

// === Combined: contrast + weighted smoothing ===

#[test]
fn test_contrast_then_weighted_pipeline() {
    let emissions = vec![
        vec![-0.5, -1.0, -1.5],  // pop 0 best
        vec![-0.8, -0.3, -1.2],  // pop 1 best
        vec![-0.4, -1.1, -1.3],  // pop 0 best
        vec![-1.0, -0.5, -0.9],  // pop 1 best
        vec![-0.3, -1.0, -1.5],  // pop 0 best (strong)
    ];

    // Apply contrast normalization first
    let contrasted = contrast_normalize_emissions(&emissions);
    // Then apply weighted smoothing
    let result = smooth_log_emissions_weighted(&contrasted, 1);

    assert_eq!(result.len(), 5);
    for row in &result {
        assert_eq!(row.len(), 3);
        for &v in row {
            assert!(v.is_finite(), "all values should be finite");
        }
    }
}

#[test]
fn test_weighted_vs_uniform_on_strong_signal() {
    // Strong signal: all windows clearly favor state 0
    let emissions = vec![
        vec![-0.1, -5.0],
        vec![-0.1, -5.0],
        vec![-0.1, -5.0],
    ];

    let weighted = smooth_log_emissions_weighted(&emissions, 1);
    let uniform = smooth_log_emissions(&emissions, 1);

    // With uniform strong signal, both methods should give similar results
    for t in 0..3 {
        for s in 0..2 {
            assert!((weighted[t][s] - uniform[t][s]).abs() < 0.5,
                "strong uniform signal should give similar results");
        }
    }
}
