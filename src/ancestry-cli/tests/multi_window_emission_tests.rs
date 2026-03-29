/// Tests for multi-window emission context (precompute, smooth, from_log_emissions)
/// and temperature auto-scaling functions.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    precompute_log_emissions, smooth_log_emissions,
    viterbi_from_log_emissions, forward_backward_from_log_emissions,
    scale_temperature_for_panel,
};

fn make_obs(sims: Vec<(&str, f64)>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "query".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_at(start: u64, sims: Vec<(&str, f64)>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_populations() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["hapA1".to_string(), "hapA2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["hapB1".to_string(), "hapB2".to_string()],
        },
    ]
}

fn make_params() -> AncestryHmmParams {
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);
    params
}

// =========================================================================
// precompute_log_emissions tests
// =========================================================================

#[test]
fn test_precompute_log_emissions_empty() {
    let params = make_params();
    let result = precompute_log_emissions(&[], &params);
    assert!(result.is_empty());
}

#[test]
fn test_precompute_log_emissions_single_obs() {
    let params = make_params();
    let obs = vec![make_obs(vec![("hapA1", 0.999), ("hapA2", 0.998), ("hapB1", 0.995), ("hapB2", 0.994)])];
    let result = precompute_log_emissions(&obs, &params);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2); // 2 states
    // POP_A has higher similarity → should have higher log emission
    assert!(result[0][0] > result[0][1]);
}

#[test]
fn test_precompute_log_emissions_matches_direct() {
    let params = make_params();
    let observations = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.995)]),
        make_obs(vec![("hapA1", 0.997), ("hapB1", 0.998)]),
    ];
    let precomputed = precompute_log_emissions(&observations, &params);

    // Should match direct log_emission calls
    for (t, obs) in observations.iter().enumerate() {
        for s in 0..2 {
            let direct = params.log_emission(obs, s);
            assert!((precomputed[t][s] - direct).abs() < 1e-12,
                "Mismatch at t={}, s={}: {} vs {}", t, s, precomputed[t][s], direct);
        }
    }
}

#[test]
fn test_precompute_log_emissions_n_states() {
    let params = make_params();
    let obs = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.995)]),
        make_obs(vec![("hapA1", 0.996), ("hapB1", 0.997)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.994)]),
    ];
    let result = precompute_log_emissions(&obs, &params);
    assert_eq!(result.len(), 3);
    for row in &result {
        assert_eq!(row.len(), 2);
    }
}

// =========================================================================
// smooth_log_emissions tests
// =========================================================================

#[test]
fn test_smooth_log_emissions_context_zero() {
    let emissions = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
    let smoothed = smooth_log_emissions(&emissions, 0);
    assert_eq!(smoothed, emissions);
}

#[test]
fn test_smooth_log_emissions_empty() {
    let smoothed = smooth_log_emissions(&[], 2);
    assert!(smoothed.is_empty());
}

#[test]
fn test_smooth_log_emissions_single_element() {
    let emissions = vec![vec![-1.0, -2.0]];
    let smoothed = smooth_log_emissions(&emissions, 3);
    // With a single element, no neighbors to average with
    assert_eq!(smoothed[0][0], -1.0);
    assert_eq!(smoothed[0][1], -2.0);
}

#[test]
fn test_smooth_log_emissions_context_one_middle() {
    // 3 windows, context=1: each window averages with ±1 neighbor
    let emissions = vec![
        vec![-1.0, -4.0],
        vec![-2.0, -5.0],
        vec![-3.0, -6.0],
    ];
    let smoothed = smooth_log_emissions(&emissions, 1);

    // Middle window: average of all 3 = (-1 + -2 + -3)/3 = -2.0
    assert!((smoothed[1][0] - (-2.0)).abs() < 1e-12);
    assert!((smoothed[1][1] - (-5.0)).abs() < 1e-12);

    // First window: average of [0,1] = (-1 + -2)/2 = -1.5
    assert!((smoothed[0][0] - (-1.5)).abs() < 1e-12);

    // Last window: average of [1,2] = (-2 + -3)/2 = -2.5
    assert!((smoothed[2][0] - (-2.5)).abs() < 1e-12);
}

#[test]
fn test_smooth_log_emissions_large_context() {
    // Context larger than sequence → all windows average everything
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-3.0, -4.0],
        vec![-5.0, -6.0],
    ];
    let smoothed = smooth_log_emissions(&emissions, 100);

    let expected_s0 = (-1.0 + -3.0 + -5.0) / 3.0;
    let expected_s1 = (-2.0 + -4.0 + -6.0) / 3.0;

    for row in &smoothed {
        assert!((row[0] - expected_s0).abs() < 1e-12);
        assert!((row[1] - expected_s1).abs() < 1e-12);
    }
}

#[test]
fn test_smooth_log_emissions_preserves_relative_order() {
    // If state 0 consistently has higher emission, smoothing should preserve that
    let emissions = vec![
        vec![-0.5, -2.0],
        vec![-0.3, -1.5],
        vec![-0.7, -2.5],
        vec![-0.4, -1.8],
    ];
    let smoothed = smooth_log_emissions(&emissions, 1);

    for t in 0..4 {
        assert!(smoothed[t][0] > smoothed[t][1],
            "State 0 should still dominate at t={}", t);
    }
}

// =========================================================================
// viterbi_from_log_emissions tests
// =========================================================================

#[test]
fn test_viterbi_from_log_emissions_empty() {
    let params = make_params();
    let result = viterbi_from_log_emissions(&[], &params);
    assert!(result.is_empty());
}

#[test]
fn test_viterbi_from_log_emissions_single() {
    let params = make_params();
    let emissions = vec![vec![0.0, -5.0]]; // state 0 much more likely
    let result = viterbi_from_log_emissions(&emissions, &params);
    assert_eq!(result, vec![0]);
}

#[test]
fn test_viterbi_from_log_emissions_consistent_state() {
    let params = make_params();
    // All windows strongly favor state 0
    let emissions = vec![
        vec![0.0, -10.0],
        vec![0.0, -10.0],
        vec![0.0, -10.0],
    ];
    let result = viterbi_from_log_emissions(&emissions, &params);
    assert_eq!(result, vec![0, 0, 0]);
}

#[test]
fn test_viterbi_from_log_emissions_matches_standard() {
    let params = make_params();
    let observations = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.995)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.994)]),
        make_obs(vec![("hapA1", 0.997), ("hapB1", 0.996)]),
    ];

    let standard = hprc_ancestry_cli::viterbi(&observations, &params);
    let precomputed = precompute_log_emissions(&observations, &params);
    let from_emissions = viterbi_from_log_emissions(&precomputed, &params);

    assert_eq!(standard, from_emissions);
}

// =========================================================================
// forward_backward_from_log_emissions tests
// =========================================================================

#[test]
fn test_fb_from_log_emissions_empty() {
    let params = make_params();
    let result = forward_backward_from_log_emissions(&[], &params);
    assert!(result.is_empty());
}

#[test]
fn test_fb_from_log_emissions_posteriors_sum_to_one() {
    let params = make_params();
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-0.5, -1.5],
        vec![-2.0, -0.5],
    ];
    let posteriors = forward_backward_from_log_emissions(&emissions, &params);

    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "Posteriors at t={} sum to {} (expected 1.0)", t, sum);
    }
}

#[test]
fn test_fb_from_log_emissions_matches_standard() {
    let params = make_params();
    let observations = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.995)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.994)]),
        make_obs(vec![("hapA1", 0.997), ("hapB1", 0.996)]),
    ];

    let standard = hprc_ancestry_cli::forward_backward(&observations, &params);
    let precomputed = precompute_log_emissions(&observations, &params);
    let from_emissions = forward_backward_from_log_emissions(&precomputed, &params);

    assert_eq!(standard.len(), from_emissions.len());
    for t in 0..standard.len() {
        for s in 0..2 {
            assert!((standard[t][s] - from_emissions[t][s]).abs() < 1e-10,
                "Posterior mismatch at t={}, s={}: {} vs {}", t, s, standard[t][s], from_emissions[t][s]);
        }
    }
}

// =========================================================================
// Integration: smoothed emissions improve signal
// =========================================================================

#[test]
fn test_smoothed_emissions_produce_valid_states() {
    let params = make_params();
    let observations = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.995)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.994)]),
        make_obs(vec![("hapA1", 0.997), ("hapB1", 0.996)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.995)]),
    ];

    let raw = precompute_log_emissions(&observations, &params);
    let smoothed = smooth_log_emissions(&raw, 1);
    let states = viterbi_from_log_emissions(&smoothed, &params);

    assert_eq!(states.len(), 4);
    for &s in &states {
        assert!(s < 2, "State should be 0 or 1");
    }
}

#[test]
fn test_smoothed_emissions_reduce_noise() {
    let params = make_params();
    // One noisy window (t=2) disagrees with the surrounding pattern
    let observations = vec![
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.994)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.993)]),
        make_obs(vec![("hapA1", 0.993), ("hapB1", 0.997)]), // noisy: B wins
        make_obs(vec![("hapA1", 0.999), ("hapB1", 0.994)]),
        make_obs(vec![("hapA1", 0.998), ("hapB1", 0.993)]),
    ];

    let raw = precompute_log_emissions(&observations, &params);
    let smoothed = smooth_log_emissions(&raw, 1);

    // After smoothing, the noisy window should be pulled toward state 0
    // by its neighbors (which all favor state 0)
    let states_raw = viterbi_from_log_emissions(&raw, &params);
    let states_smooth = viterbi_from_log_emissions(&smoothed, &params);

    // Smoothed should assign all state 0 (noise corrected)
    // Raw might assign state 1 at t=2
    assert_eq!(states_smooth, vec![0, 0, 0, 0, 0],
        "Smoothing should correct the noisy window");

    // The raw Viterbi might or might not get t=2 wrong depending on
    // transition penalty, but the smoothed version should be unanimous
    let _ = states_raw; // just verify it runs
}

// =========================================================================
// scale_temperature_for_panel tests
// =========================================================================

#[test]
fn test_scale_temperature_for_panel_reference_size() {
    // At reference panel size (10 haps/pop), correction should be ~1.0
    let base = 0.03;
    let scaled = scale_temperature_for_panel(base, 10.0);
    assert!((scaled - base).abs() < 0.001,
        "At reference size, scaling should be near identity: {} vs {}", scaled, base);
}

#[test]
fn test_scale_temperature_decreases_with_larger_panel() {
    let base = 0.03;
    let small = scale_temperature_for_panel(base, 5.0);
    let medium = scale_temperature_for_panel(base, 10.0);
    let large = scale_temperature_for_panel(base, 50.0);

    assert!(small > medium, "Smaller panel → higher temperature");
    assert!(medium > large, "Larger panel → lower temperature");
}

#[test]
fn test_scale_temperature_single_haplotype() {
    let base = 0.03;
    let scaled = scale_temperature_for_panel(base, 1.0);
    assert_eq!(scaled, base, "Single haplotype → no scaling");
}

#[test]
fn test_scale_temperature_clamped() {
    // Very large panel should still produce reasonable temperature
    let base = 0.03;
    let scaled = scale_temperature_for_panel(base, 10000.0);
    assert!(scaled >= 0.001, "Should be clamped above 0.001");
    assert!(scaled <= 1.0, "Should be clamped below 1.0");
}

#[test]
fn test_scale_temperature_zero_haplotypes() {
    // Edge case: 0 haplotypes
    let base = 0.03;
    let scaled = scale_temperature_for_panel(base, 0.0);
    assert_eq!(scaled, base, "Zero haplotypes → no scaling");
}
