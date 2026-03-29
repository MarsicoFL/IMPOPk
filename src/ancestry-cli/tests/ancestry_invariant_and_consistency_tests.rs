//! Ancestry HMM invariant and cross-function consistency tests.
//!
//! Cycle 48: Verify:
//! - Transition matrix rows sum to 1
//! - Viterbi path agrees with posteriors
//! - forward_backward produces valid posteriors (per-state probs sum to 1)
//! - Genetic map variants produce valid results
//! - EmissionModel aggregate properties
//! - BW convergence properties
//! - AncestryGeneticMap modulated_switch_prob range

use hprc_ancestry_cli::hmm::*;
use std::collections::HashMap;

// ============================================================================
// Helpers
// ============================================================================

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
        AncestralPopulation {
            name: "POP_C".to_string(),
            haplotypes: vec!["hapC1".to_string(), "hapC2".to_string()],
        },
    ]
}

fn make_params() -> AncestryHmmParams {
    AncestryHmmParams::new(make_populations(), 0.01)
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_pos(sims: &[(&str, f64)], start: u64, end: u64) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_cov(
    sims: &[(&str, f64)],
    covs: &[(&str, f64)],
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: Some(covs.iter().map(|(k, v)| (k.to_string(), *v)).collect()),
            haplotype_consistency_bonus: None,
    }
}

fn make_pop_a_dominant_obs() -> Vec<AncestryObservation> {
    (0..20)
        .map(|i| {
            let base = 0.98 + (i as f64) * 0.0001;
            let start = (i as u64) * 5000;
            AncestryObservation {
                chrom: "chr1".to_string(),
                start,
                end: start + 4999,
                sample: "query".to_string(),
                similarities: [
                    ("hapA1".to_string(), base + 0.01),
                    ("hapA2".to_string(), base + 0.008),
                    ("hapB1".to_string(), base - 0.01),
                    ("hapB2".to_string(), base - 0.012),
                    ("hapC1".to_string(), base - 0.015),
                    ("hapC2".to_string(), base - 0.018),
                ].into_iter().collect(),
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            }
        })
        .collect()
}

// ============================================================================
// Transition matrix properties
// ============================================================================

#[test]
fn transition_rows_sum_to_one() {
    let params = make_params();
    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "Transition row {} sums to {}", i, sum);
    }
}

#[test]
fn initial_probs_sum_to_one() {
    let params = make_params();
    let sum: f64 = params.initial.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "Initial probs sum to {}", sum);
}

#[test]
fn transition_matrix_is_correct_size() {
    let params = make_params();
    assert_eq!(params.transitions.len(), 3);
    for row in &params.transitions {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn set_switch_prob_preserves_row_sums() {
    let mut params = make_params();
    params.set_switch_prob(0.05);

    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "After set_switch_prob, row {} sums to {}", i, sum);
    }
}

// ============================================================================
// Viterbi properties
// ============================================================================

#[test]
fn viterbi_returns_valid_states() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let states = viterbi(&obs, &params);

    assert_eq!(states.len(), obs.len());
    for &s in &states {
        assert!(s < 3, "State {} exceeds n_states=3", s);
    }
}

#[test]
fn viterbi_empty_observations() {
    let params = make_params();
    let states = viterbi(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn viterbi_single_observation() {
    let params = make_params();
    let obs = vec![make_obs(&[
        ("hapA1", 0.99), ("hapA2", 0.98),
        ("hapB1", 0.95), ("hapB2", 0.94),
        ("hapC1", 0.93), ("hapC2", 0.92),
    ])];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0, "Should assign to POP_A");
}

#[test]
fn viterbi_detects_ancestry_switch() {
    let params = make_params();
    let mut obs: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_with_pos(&[
            ("hapA1", 0.999), ("hapA2", 0.998),
            ("hapB1", 0.980), ("hapB2", 0.979),
            ("hapC1", 0.975), ("hapC2", 0.974),
        ], (i as u64) * 5000, (i as u64) * 5000 + 4999))
        .collect();
    obs.extend((10..20).map(|i| make_obs_with_pos(&[
        ("hapA1", 0.980), ("hapA2", 0.979),
        ("hapB1", 0.999), ("hapB2", 0.998),
        ("hapC1", 0.975), ("hapC2", 0.974),
    ], (i as u64) * 5000, (i as u64) * 5000 + 4999)));

    let states = viterbi(&obs, &params);

    let first_half_a = states[..10].iter().filter(|&&s| s == 0).count();
    let second_half_b = states[10..].iter().filter(|&&s| s == 1).count();
    assert!(first_half_a >= 8, "First half should be mostly POP_A, got {}/10", first_half_a);
    assert!(second_half_b >= 8, "Second half should be mostly POP_B, got {}/10", second_half_b);
}

// ============================================================================
// Forward-backward properties
// ============================================================================

#[test]
fn forward_backward_posteriors_sum_to_one() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let posteriors = forward_backward(&obs, &params);

    assert_eq!(posteriors.len(), obs.len());
    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 3);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors at t={} sum to {}", t, sum);
        for &p in post {
            assert!(p >= -1e-10 && p <= 1.0 + 1e-10, "Posterior {} out of [0,1] at t={}", p, t);
        }
    }
}

#[test]
fn forward_backward_empty_observations() {
    let params = make_params();
    let posteriors = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
}

#[test]
fn posterior_decode_agrees_with_viterbi_in_clear_cases() {
    let params = make_params();
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs_with_pos(&[
            ("hapA1", 0.9999), ("hapA2", 0.9998),
            ("hapB1", 0.950), ("hapB2", 0.949),
            ("hapC1", 0.940), ("hapC2", 0.939),
        ], (i as u64) * 5000, (i as u64) * 5000 + 4999))
        .collect();

    let vit_states = viterbi(&obs, &params);
    let post_states = posterior_decode(&obs, &params);

    assert_eq!(vit_states, post_states, "Viterbi and posterior decode should agree when unambiguous");
}

// ============================================================================
// Genetic map variants
// ============================================================================

#[test]
fn viterbi_with_genetic_map_produces_valid_states() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let n = obs.len();
    let gmap = AncestryGeneticMap::uniform(0, (n as u64) * 5000, 1.0);

    let states = viterbi_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), n);
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn forward_backward_with_genetic_map_posteriors_valid() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let n = obs.len();
    let gmap = AncestryGeneticMap::uniform(0, (n as u64) * 5000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(posteriors.len(), n);
    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Genetic map posteriors at t={} sum to {}", t, sum);
    }
}

#[test]
fn posterior_decode_with_genetic_map_valid() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let n = obs.len();
    let gmap = AncestryGeneticMap::uniform(0, (n as u64) * 5000, 1.0);

    let states = posterior_decode_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), n);
    for &s in &states {
        assert!(s < 3);
    }
}

// ============================================================================
// AncestryGeneticMap properties
// ============================================================================

#[test]
fn ancestry_genetic_map_uniform_interpolation_monotone() {
    let gmap = AncestryGeneticMap::uniform(0, 2_000_000, 1.5);
    let mut prev = f64::NEG_INFINITY;
    for pos in (0..2_000_001).step_by(100_000) {
        let cm = gmap.interpolate_cm(pos);
        assert!(cm >= prev, "Non-monotone at {}: {} < {}", pos, cm, prev);
        prev = cm;
    }
}

#[test]
fn ancestry_genetic_map_distance_nonnegative() {
    let gmap = AncestryGeneticMap::uniform(0, 2_000_000, 1.5);
    let dist = gmap.genetic_distance_cm(100_000, 500_000);
    assert!(dist >= 0.0);
}

#[test]
fn modulated_switch_prob_between_zero_and_one() {
    let gmap = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let positions: Vec<(u64, u64)> = vec![(0, 0), (0, 100), (0, 5000), (0, 50000), (0, 500000)];

    for (pos1, pos2) in positions {
        let sp = gmap.modulated_switch_prob(0.01, pos1, pos2, 5000);
        assert!(sp > 0.0 && sp < 1.0, "Switch prob {} out of (0,1) for ({}, {})", sp, pos1, pos2);
    }
}

#[test]
fn modulated_switch_prob_increases_with_distance() {
    let gmap = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let sp_close = gmap.modulated_switch_prob(0.01, 0, 1000, 5000);
    let sp_far = gmap.modulated_switch_prob(0.01, 0, 100000, 5000);
    assert!(sp_far >= sp_close, "Switch prob should increase with distance: close={} far={}", sp_close, sp_far);
}

// ============================================================================
// Emission model properties
// ============================================================================

#[test]
fn log_emission_is_nonpositive() {
    let params = make_params();
    let obs = make_obs(&[
        ("hapA1", 0.99), ("hapA2", 0.98),
        ("hapB1", 0.95), ("hapB2", 0.94),
        ("hapC1", 0.93), ("hapC2", 0.92),
    ]);

    for state in 0..3 {
        let le = params.log_emission(&obs, state);
        assert!(le <= 0.0 + 1e-10, "Log-emission for state {} is {}, should be <= 0", state, le);
    }
}

#[test]
fn log_emission_softmax_sums_correctly() {
    let params = make_params();
    let obs = make_obs(&[
        ("hapA1", 0.99), ("hapA2", 0.98),
        ("hapB1", 0.95), ("hapB2", 0.94),
        ("hapC1", 0.93), ("hapC2", 0.92),
    ]);

    let mut sum = 0.0;
    for state in 0..3 {
        sum += params.log_emission(&obs, state).exp();
    }
    assert!((sum - 1.0).abs() < 1e-6, "Softmax exp-sum should be ~1.0, got {}", sum);
}

#[test]
fn log_emission_highest_for_most_similar_pop() {
    let params = make_params();
    let obs = make_obs(&[
        ("hapA1", 0.999), ("hapA2", 0.998),
        ("hapB1", 0.980), ("hapB2", 0.979),
        ("hapC1", 0.970), ("hapC2", 0.969),
    ]);

    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);
    let le_c = params.log_emission(&obs, 2);

    assert!(le_a > le_b && le_a > le_c, "POP_A should have highest: A={} B={} C={}", le_a, le_b, le_c);
}

#[test]
fn log_emission_with_coverage_adds_information() {
    let mut params = make_params();
    params.set_coverage_weight(0.5);

    let obs = make_obs_with_cov(
        &[
            ("hapA1", 0.99), ("hapA2", 0.98),
            ("hapB1", 0.95), ("hapB2", 0.94),
            ("hapC1", 0.93), ("hapC2", 0.92),
        ],
        &[
            ("hapA1", 0.95), ("hapA2", 0.93),
            ("hapB1", 0.50), ("hapB2", 0.48),
            ("hapC1", 0.40), ("hapC2", 0.38),
        ],
    );

    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);
    assert!(le_a > le_b, "Coverage should reinforce POP_A");
}

#[test]
fn log_emission_missing_haplotype_data() {
    let params = make_params();
    let obs = make_obs(&[("hapA1", 0.99), ("hapA2", 0.98)]);

    let le_a = params.log_emission(&obs, 0);
    let le_b = params.log_emission(&obs, 1);
    let le_c = params.log_emission(&obs, 2);

    assert!(le_a.is_finite(), "POP_A emission should be finite");
    assert_eq!(le_b, f64::NEG_INFINITY, "POP_B should be -inf (no data)");
    assert_eq!(le_c, f64::NEG_INFINITY, "POP_C should be -inf (no data)");
}

// ============================================================================
// Emission model parsing
// ============================================================================

#[test]
fn emission_model_from_str_roundtrip() {
    let models = ["max", "mean", "median", "top3", "top5", "top10"];
    for &m in &models {
        let parsed: EmissionModel = m.parse().unwrap();
        let displayed = format!("{}", parsed);
        assert_eq!(displayed, m, "Roundtrip failed: '{}' -> '{}'", m, displayed);
    }
}

#[test]
fn emission_model_from_str_invalid() {
    assert!("invalid".parse::<EmissionModel>().is_err());
    assert!("topXYZ".parse::<EmissionModel>().is_err());
}

#[test]
fn emission_model_from_str_case_insensitive() {
    let cases = [("MAX", EmissionModel::Max), ("Mean", EmissionModel::Mean), ("MEDIAN", EmissionModel::Median), ("Top5", EmissionModel::TopK(5))];
    for (s, expected) in &cases {
        let parsed: EmissionModel = s.parse().unwrap();
        assert_eq!(parsed, *expected);
    }
}

// ============================================================================
// Normalization
// ============================================================================

#[test]
fn learn_normalization_produces_valid_stats() {
    let mut params = make_params();
    let obs = make_pop_a_dominant_obs();
    params.learn_normalization(&obs);

    let norm = params.normalization.as_ref().expect("Normalization should be set");
    assert_eq!(norm.means.len(), 3);
    assert_eq!(norm.stds.len(), 3);
    for &m in &norm.means {
        assert!(m.is_finite());
    }
    for &s in &norm.stds {
        assert!(s > 0.0 && s.is_finite());
    }
}

#[test]
fn learn_normalization_empty_observations() {
    let mut params = make_params();
    params.learn_normalization(&[]);

    let norm = params.normalization.as_ref().expect("Normalization should be set");
    for &m in &norm.means {
        assert!((m - 0.0).abs() < 1e-10);
    }
}

// ============================================================================
// Baum-Welch convergence
// ============================================================================

#[test]
fn baum_welch_preserves_transition_validity() {
    let mut params = make_params();
    let obs = make_pop_a_dominant_obs();
    let obs_slice: &[AncestryObservation] = &obs;

    params.baum_welch(&[obs_slice], 5, 1e-6);

    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "After BW, row {} sums to {}", i, sum);
        for &t in row {
            assert!(t > 0.0);
        }
    }
}

#[test]
fn baum_welch_zero_iterations_is_noop() {
    let mut params = make_params();
    let orig_trans = params.transitions.clone();
    let obs = make_pop_a_dominant_obs();
    let obs_slice: &[AncestryObservation] = &obs;

    params.baum_welch(&[obs_slice], 0, 1e-6);

    for (i, (orig, cur)) in orig_trans.iter().zip(params.transitions.iter()).enumerate() {
        for j in 0..3 {
            assert!((orig[j] - cur[j]).abs() < 1e-12, "BW(0) changed transitions[{}][{}]", i, j);
        }
    }
}

// ============================================================================
// Temperature estimation
// ============================================================================

#[test]
fn estimate_temperature_returns_positive() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let temp = estimate_temperature(&obs, &params.populations);
    assert!(temp > 0.0, "Temperature should be positive, got {}", temp);
    assert!(temp.is_finite());
}

#[test]
fn estimate_temperature_normalized_returns_positive() {
    let mut params = make_params();
    let obs = make_pop_a_dominant_obs();
    params.learn_normalization(&obs);
    let temp = estimate_temperature_normalized(&obs, &params);
    assert!(temp > 0.0, "Normalized temperature should be positive, got {}", temp);
    assert!(temp.is_finite());
}

#[test]
fn estimate_switch_prob_in_range() {
    let params = make_params();
    let obs = make_pop_a_dominant_obs();
    let sp = estimate_switch_prob(&obs, &params.populations, params.emission_std);
    assert!(sp > 0.0 && sp < 1.0, "Switch prob {} out of (0,1)", sp);
}

// ============================================================================
// Edge: two populations only
// ============================================================================

#[test]
fn two_population_model_works() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["h1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["h2".to_string()] },
    ];
    let params = AncestryHmmParams::new(pops, 0.01);
    assert_eq!(params.n_states, 2);

    let obs = vec![
        make_obs(&[("h1", 0.99), ("h2", 0.95)]),
        make_obs(&[("h1", 0.98), ("h2", 0.96)]),
    ];

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 2);
    assert_eq!(states[0], 0);
    assert_eq!(states[1], 0);
}

// ============================================================================
// Edge: many populations
// ============================================================================

#[test]
fn many_population_model_works() {
    let pops: Vec<AncestralPopulation> = (0..10)
        .map(|i| AncestralPopulation {
            name: format!("POP_{}", i),
            haplotypes: vec![format!("hap_{}", i)],
        })
        .collect();
    let params = AncestryHmmParams::new(pops, 0.01);
    assert_eq!(params.n_states, 10);

    let obs: Vec<AncestryObservation> = (0..5)
        .map(|_| {
            let mut sim_map = HashMap::new();
            for j in 0..10 {
                let val = if j == 5 { 0.999 } else { 0.980 };
                sim_map.insert(format!("hap_{}", j), val);
            }
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: 0, end: 5000,
                sample: "q".to_string(),
                similarities: sim_map,
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            }
        })
        .collect();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 5);
    for &s in &states {
        assert_eq!(s, 5, "Should assign to POP_5, got {}", s);
    }
}
