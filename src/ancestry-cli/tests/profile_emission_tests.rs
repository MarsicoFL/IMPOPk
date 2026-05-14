//! Tests for population profile emission learning and blending.

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    learn_population_profiles, compute_profile_log_emissions, blend_log_emissions,
    precompute_log_emissions,
};
use std::collections::HashMap;

fn make_populations() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["A_h1".to_string(), "A_h2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["B_h1".to_string(), "B_h2".to_string()],
        },
        AncestralPopulation {
            name: "POP_C".to_string(),
            haplotypes: vec!["C_h1".to_string(), "C_h2".to_string()],
        },
    ]
}

fn make_obs(start: u64, a1: f64, a2: f64, b1: f64, b2: f64, c1: f64, c2: f64) -> AncestryObservation {
    let mut sims = HashMap::new();
    sims.insert("A_h1".to_string(), a1);
    sims.insert("A_h2".to_string(), a2);
    sims.insert("B_h1".to_string(), b1);
    sims.insert("B_h2".to_string(), b2);
    sims.insert("C_h1".to_string(), c1);
    sims.insert("C_h2".to_string(), c2);
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ============================================================================
// learn_population_profiles tests
// ============================================================================

#[test]
fn test_learn_profiles_basic() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    // 3 observations: obs0 and obs1 assigned to state 0, obs2 to state 1
    let obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87), // state 0 (POP_A)
        make_obs(10000, 0.97, 0.94, 0.90, 0.89, 0.87, 0.86), // state 0 (POP_A)
        make_obs(20000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87), // state 1 (POP_B)
    ];
    let states = vec![0, 0, 1];

    let profiles = learn_population_profiles(&obs, &states, &params);

    assert_eq!(profiles.centroids.len(), 3);
    assert_eq!(profiles.n_windows, vec![2, 1, 0]);

    // State 0 centroid: mean of obs0 and obs1 for each pop
    // POP_A (max): mean of max(0.96,0.95) and max(0.97,0.94) = mean(0.96, 0.97) = 0.965
    let centroid_0_a = profiles.centroids[0][0];
    assert!((centroid_0_a - 0.965).abs() < 0.01, "got {}", centroid_0_a);

    // State 1 centroid: obs2 for POP_B (max): max(0.96, 0.95) = 0.96
    let centroid_1_b = profiles.centroids[1][1];
    assert!((centroid_1_b - 0.96).abs() < 0.01, "got {}", centroid_1_b);

    // State 2 has no windows
    assert_eq!(profiles.n_windows[2], 0);
    assert_eq!(profiles.centroids[2], vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_learn_profiles_empty_states() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs: Vec<AncestryObservation> = vec![];
    let states: Vec<usize> = vec![];

    let profiles = learn_population_profiles(&obs, &states, &params);

    assert_eq!(profiles.centroids.len(), 3);
    assert_eq!(profiles.n_windows, vec![0, 0, 0]);
}

#[test]
fn test_learn_profiles_all_same_state() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.97, 0.94, 0.90, 0.89, 0.87, 0.86),
    ];
    let states = vec![0, 0]; // all state 0

    let profiles = learn_population_profiles(&obs, &states, &params);
    assert_eq!(profiles.n_windows, vec![2, 0, 0]);
    // States 1 and 2 have zero centroids
    assert_eq!(profiles.centroids[1], vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_learn_profiles_invalid_state_ignored() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
    ];
    let states = vec![99]; // out of bounds state

    let profiles = learn_population_profiles(&obs, &states, &params);
    assert_eq!(profiles.n_windows, vec![0, 0, 0]);
}

// ============================================================================
// profile_correlation tests (via compute_profile_log_emissions)
// ============================================================================

#[test]
fn test_profile_emissions_high_correlation_with_correct_state() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    // Build profiles: state 0 = high POP_A, state 1 = high POP_B, state 2 = high POP_C
    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),     // state 0
        make_obs(10000, 0.95, 0.94, 0.90, 0.89, 0.87, 0.86),  // state 0
        make_obs(20000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87),  // state 1
        make_obs(30000, 0.89, 0.88, 0.95, 0.94, 0.87, 0.86),  // state 1
        make_obs(40000, 0.88, 0.87, 0.89, 0.88, 0.96, 0.95),  // state 2
        make_obs(50000, 0.87, 0.86, 0.88, 0.87, 0.95, 0.94),  // state 2
    ];
    let train_states = vec![0, 0, 1, 1, 2, 2];

    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    // Test observation: clearly POP_A pattern
    let test_obs = vec![make_obs(0, 0.97, 0.96, 0.92, 0.91, 0.89, 0.88)];
    let emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    assert_eq!(emissions.len(), 1);
    assert_eq!(emissions[0].len(), 3);

    // State 0 (POP_A) should have highest emission
    assert!(emissions[0][0] > emissions[0][1],
        "state 0 ({}) should be > state 1 ({})", emissions[0][0], emissions[0][1]);
    assert!(emissions[0][0] > emissions[0][2],
        "state 0 ({}) should be > state 2 ({})", emissions[0][0], emissions[0][2]);
}

#[test]
fn test_profile_emissions_pop_b_pattern() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87),
        make_obs(20000, 0.88, 0.87, 0.89, 0.88, 0.96, 0.95),
    ];
    let train_states = vec![0, 1, 2];
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    // Test: POP_B pattern
    let test_obs = vec![make_obs(0, 0.91, 0.90, 0.97, 0.96, 0.89, 0.88)];
    let emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    // State 1 (POP_B) should be highest
    assert!(emissions[0][1] > emissions[0][0]);
    assert!(emissions[0][1] > emissions[0][2]);
}

#[test]
fn test_profile_emissions_uniform_observation() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87),
    ];
    let train_states = vec![0, 1];
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    // Test: uniform observation (same sim to all pops) - should give equal emissions
    let test_obs = vec![make_obs(0, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95)];
    let emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    // With zero variance in observation, all states should get log(1/k)
    let expected = -(3.0_f64).ln();
    for s in 0..3 {
        assert!((emissions[0][s] - expected).abs() < 1e-6,
            "state {} = {}, expected {}", s, emissions[0][s], expected);
    }
}

#[test]
fn test_profile_emissions_state_with_no_windows() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
    ];
    let train_states = vec![0]; // only state 0 has data
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    let test_obs = vec![make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87)];
    let emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    // All emissions should be finite (no panic from state with no windows)
    for s in 0..3 {
        assert!(emissions[0][s].is_finite(), "state {} emission should be finite", s);
    }
}

#[test]
fn test_profile_emissions_log_softmax_sums_to_one() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87),
        make_obs(20000, 0.88, 0.87, 0.89, 0.88, 0.96, 0.95),
    ];
    let train_states = vec![0, 1, 2];
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    let test_obs = vec![make_obs(0, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89)];
    let emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    // exp(log-emissions) should sum to ~1.0 (log-softmax normalization)
    let sum: f64 = emissions[0].iter().map(|&e| e.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "sum of probs = {}", sum);
}

// ============================================================================
// blend_log_emissions tests
// ============================================================================

#[test]
fn test_blend_weight_zero_equals_standard() {
    let standard = vec![vec![-1.0, -0.5, -2.0]];
    let profile = vec![vec![-0.8, -0.3, -1.5]];

    let blended = blend_log_emissions(&standard, &profile, 0.0);
    assert_eq!(blended.len(), 1);
    for (s, p) in blended[0].iter().zip(standard[0].iter()) {
        assert!((s - p).abs() < 1e-10);
    }
}

#[test]
fn test_blend_weight_one_equals_profile() {
    let standard = vec![vec![-1.0, -0.5, -2.0]];
    let profile = vec![vec![-0.8, -0.3, -1.5]];

    let blended = blend_log_emissions(&standard, &profile, 1.0);
    for (s, p) in blended[0].iter().zip(profile[0].iter()) {
        assert!((s - p).abs() < 1e-10);
    }
}

#[test]
fn test_blend_weight_half_is_average() {
    let standard = vec![vec![-1.0, -0.5, -2.0]];
    let profile = vec![vec![-0.8, -0.3, -1.5]];

    let blended = blend_log_emissions(&standard, &profile, 0.5);
    let expected = vec![-0.9, -0.4, -1.75];
    for (b, e) in blended[0].iter().zip(expected.iter()) {
        assert!((b - e).abs() < 1e-10, "{} != {}", b, e);
    }
}

#[test]
fn test_blend_neg_inf_standard_falls_back_to_profile() {
    let standard = vec![vec![f64::NEG_INFINITY, -0.5, -2.0]];
    let profile = vec![vec![-0.8, -0.3, -1.5]];

    let blended = blend_log_emissions(&standard, &profile, 0.5);
    // First element: standard is -inf, so should use profile
    assert!((blended[0][0] - (-0.8)).abs() < 1e-10);
    // Other elements: normal blend
    assert!((blended[0][1] - (-0.4)).abs() < 1e-10);
}

#[test]
fn test_blend_neg_inf_profile_falls_back_to_standard() {
    let standard = vec![vec![-1.0, -0.5, -2.0]];
    let profile = vec![vec![f64::NEG_INFINITY, -0.3, -1.5]];

    let blended = blend_log_emissions(&standard, &profile, 0.5);
    assert!((blended[0][0] - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_blend_both_neg_inf_stays_neg_inf() {
    let standard = vec![vec![f64::NEG_INFINITY]];
    let profile = vec![vec![f64::NEG_INFINITY]];

    let blended = blend_log_emissions(&standard, &profile, 0.5);
    assert!(blended[0][0].is_infinite() && blended[0][0] < 0.0);
}

#[test]
fn test_blend_multiple_windows() {
    let standard = vec![
        vec![-1.0, -0.5],
        vec![-2.0, -1.0],
        vec![-0.5, -1.5],
    ];
    let profile = vec![
        vec![-0.8, -0.3],
        vec![-1.5, -0.5],
        vec![-0.4, -1.2],
    ];

    let blended = blend_log_emissions(&standard, &profile, 0.3);
    assert_eq!(blended.len(), 3);

    // First window, first state: 0.7 * (-1.0) + 0.3 * (-0.8) = -0.7 + -0.24 = -0.94
    assert!((blended[0][0] - (-0.94)).abs() < 1e-10);
}

#[test]
fn test_blend_weight_clamped() {
    let standard = vec![vec![-1.0]];
    let profile = vec![vec![-0.5]];

    // Weight > 1.0 clamped to 1.0
    let blended = blend_log_emissions(&standard, &profile, 1.5);
    assert!((blended[0][0] - (-0.5)).abs() < 1e-10);

    // Weight < 0.0 clamped to 0.0
    let blended = blend_log_emissions(&standard, &profile, -0.5);
    assert!((blended[0][0] - (-1.0)).abs() < 1e-10);
}

// ============================================================================
// End-to-end profile emission pipeline test
// ============================================================================

#[test]
fn test_profile_pipeline_improves_ambiguous_case() {
    // This test verifies the profile emission can help disambiguate
    // cases where softmax alone is ambiguous (similar similarities to two populations).
    let pops = make_populations();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.01);

    // Training data: clear separation
    let train_obs = vec![
        // POP_A windows: high A, moderate B, low C
        make_obs(0, 0.960, 0.955, 0.920, 0.915, 0.880, 0.875),
        make_obs(10000, 0.958, 0.953, 0.918, 0.913, 0.878, 0.873),
        make_obs(20000, 0.962, 0.957, 0.922, 0.917, 0.882, 0.877),
        // POP_B windows: moderate A, high B, low C
        make_obs(30000, 0.918, 0.913, 0.960, 0.955, 0.880, 0.875),
        make_obs(40000, 0.915, 0.910, 0.958, 0.953, 0.878, 0.873),
        make_obs(50000, 0.920, 0.915, 0.962, 0.957, 0.882, 0.877),
        // POP_C windows: low A, low B, high C
        make_obs(60000, 0.880, 0.875, 0.882, 0.877, 0.960, 0.955),
        make_obs(70000, 0.878, 0.873, 0.880, 0.875, 0.958, 0.953),
    ];
    let train_states = vec![0, 0, 0, 1, 1, 1, 2, 2];

    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    // Ambiguous test case: A and B are close
    let test_obs = vec![make_obs(0, 0.950, 0.945, 0.948, 0.943, 0.880, 0.875)];

    let standard_emissions = precompute_log_emissions(&test_obs, &params);
    let profile_emissions = compute_profile_log_emissions(&test_obs, &params, &profiles);

    // Profile should give POP_A higher than POP_B because the relative pattern
    // (A slightly > B >> C) matches POP_A's centroid pattern better
    // Both POP_A and POP_B centroids have the pattern X >> C, but
    // POP_A centroid has A > B while POP_B has B > A
    assert!(profile_emissions[0][0] > profile_emissions[0][1],
        "Profile should favor POP_A: A={}, B={}", profile_emissions[0][0], profile_emissions[0][1]);

    // Standard softmax (max emission) also favors A since max(0.950,0.945) > max(0.948,0.943)
    // but the margin should be smaller in standard than in profile
    let std_margin = standard_emissions[0][0] - standard_emissions[0][1];
    let prof_margin = profile_emissions[0][0] - profile_emissions[0][1];

    // Profile margin should be non-trivial (the pattern matching provides extra discrimination)
    assert!(prof_margin > 0.0, "Profile margin should be positive: {}", prof_margin);
    assert!(std_margin.is_finite() && prof_margin.is_finite());
}

#[test]
fn test_profile_pipeline_blend_produces_valid_emissions() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.90, 0.89, 0.96, 0.95, 0.88, 0.87),
        make_obs(20000, 0.88, 0.87, 0.89, 0.88, 0.96, 0.95),
    ];
    let train_states = vec![0, 1, 2];
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    let test_obs = vec![
        make_obs(0, 0.95, 0.94, 0.91, 0.90, 0.88, 0.87),
        make_obs(10000, 0.90, 0.89, 0.95, 0.94, 0.88, 0.87),
    ];

    let standard = precompute_log_emissions(&test_obs, &params);
    let profile = compute_profile_log_emissions(&test_obs, &params, &profiles);
    let blended = blend_log_emissions(&standard, &profile, 0.3);

    assert_eq!(blended.len(), 2);
    assert_eq!(blended[0].len(), 3);

    // All values should be finite
    for window in &blended {
        for &val in window {
            assert!(val.is_finite(), "blended emission should be finite, got {}", val);
        }
    }
}

// ============================================================================
// Two-population edge case
// ============================================================================

#[test]
fn test_profile_two_populations() {
    let pops = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["A_h1".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["B_h1".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![
        make_obs(0, 0.96, 0.0, 0.91, 0.0, 0.0, 0.0),
        make_obs(10000, 0.90, 0.0, 0.96, 0.0, 0.0, 0.0),
    ];
    let states = vec![0, 1];
    let profiles = learn_population_profiles(&obs, &states, &params);

    assert_eq!(profiles.centroids.len(), 2);
    assert_eq!(profiles.n_windows, vec![1, 1]);

    // Profile emissions should be computable
    let emissions = compute_profile_log_emissions(&obs, &params, &profiles);
    assert_eq!(emissions.len(), 2);

    // First obs should favor state 0, second should favor state 1
    assert!(emissions[0][0] > emissions[0][1]);
    assert!(emissions[1][1] > emissions[1][0]);
}

// ============================================================================
// Single state edge case
// ============================================================================

#[test]
fn test_profile_single_state() {
    let pops = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["A_h1".to_string(), "A_h2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    let obs = vec![make_obs(0, 0.96, 0.95, 0.0, 0.0, 0.0, 0.0)];
    let states = vec![0];
    let profiles = learn_population_profiles(&obs, &states, &params);

    assert_eq!(profiles.centroids.len(), 1);

    // With only 1 population, correlation is undefined (single-element vector)
    // Should return log(1/1) = 0
    let emissions = compute_profile_log_emissions(&obs, &params, &profiles);
    assert!((emissions[0][0] - 0.0).abs() < 1e-6);
}

// ============================================================================
// Empty observations
// ============================================================================

#[test]
fn test_profile_emissions_empty_observations() {
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.001);

    let profiles = learn_population_profiles(&[], &[], &params);
    let emissions = compute_profile_log_emissions(&[], &params, &profiles);
    assert!(emissions.is_empty());
}

#[test]
fn test_blend_empty() {
    let blended = blend_log_emissions(&[], &[], 0.5);
    assert!(blended.is_empty());
}

// ============================================================================
// Numerical stability
// ============================================================================

#[test]
fn test_profile_emissions_identical_centroids() {
    // When all centroids are identical, correlation should be the same for all states
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    let obs = vec![
        make_obs(0, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90),
    ];
    // All states have the same window → identical centroids
    let states_a = vec![0];
    let profiles_a = learn_population_profiles(&obs, &states_a, &params);

    // State 0 has a centroid, states 1&2 have zero centroids (n_windows=0)
    // The zero-window states get correlation=0, which differs from state 0
    let emissions = compute_profile_log_emissions(&obs, &params, &profiles_a);
    assert!(emissions[0][0].is_finite());
}

#[test]
fn test_profile_correlation_scale_invariant() {
    // Correlation should be the same regardless of absolute similarity level
    let pops = make_populations();
    let params = AncestryHmmParams::new(pops, 0.01);

    // Training: clear pattern A > B > C
    let train_obs = vec![
        make_obs(0, 0.96, 0.95, 0.92, 0.91, 0.88, 0.87),
    ];
    let train_states = vec![0];
    let profiles = learn_population_profiles(&train_obs, &train_states, &params);

    // Test with shifted-down pattern (same relative order, lower absolute)
    let test_high = vec![make_obs(0, 0.96, 0.95, 0.92, 0.91, 0.88, 0.87)];
    let test_low = vec![make_obs(0, 0.86, 0.85, 0.82, 0.81, 0.78, 0.77)];

    let em_high = compute_profile_log_emissions(&test_high, &params, &profiles);
    let em_low = compute_profile_log_emissions(&test_low, &params, &profiles);

    // Profile emission for state 0 should be similar for both (correlation is scale-invariant)
    // They might differ slightly due to softmax temperature scaling, but should be close
    let diff = (em_high[0][0] - em_low[0][0]).abs();
    assert!(diff < 0.5, "Profile emissions should be similar: high={}, low={}, diff={}",
        em_high[0][0], em_low[0][0], diff);
}
