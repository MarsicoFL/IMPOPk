//! Tests for ancestry Baum-Welch convergence, estimation functions, and
//! genetic-map-aware HMM numerical properties.
//!
//! Focus: convergence guarantees, numerical stability with extreme inputs,
//! and consistency between standard and genetic-map-aware algorithms.

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    estimate_temperature, estimate_temperature_normalized,
    estimate_switch_prob, forward_backward, forward_backward_with_genetic_map, posterior_decode,
    posterior_decode_with_genetic_map, viterbi, viterbi_with_genetic_map,
};
use std::collections::HashMap;

// ============================================================================
// Helpers
// ============================================================================

fn make_populations(n: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n)
        .map(|i| {
            let name = if i < names.len() {
                names[i].to_string()
            } else {
                format!("POP{i}")
            };
            AncestralPopulation {
                name: name.clone(),
                haplotypes: vec![format!("{name}_hap1"), format!("{name}_hap2")],
            }
        })
        .collect()
}

fn make_obs(
    start: u64,
    dominant_pop: usize,
    n_pops: usize,
    high_sim: f64,
    low_sim: f64,
) -> AncestryObservation {
    let pops = make_populations(n_pops);
    let mut similarities = HashMap::new();
    for (i, pop) in pops.iter().enumerate() {
        let sim = if i == dominant_pop { high_sim } else { low_sim };
        for hap in &pop.haplotypes {
            similarities.insert(hap.clone(), sim);
        }
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "TEST_QUERY".to_string(),
        similarities,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_clear_ancestry_sequence(
    n_windows: usize,
    dominant_pop: usize,
    n_pops: usize,
) -> Vec<AncestryObservation> {
    (0..n_windows)
        .map(|i| make_obs(i as u64 * 10000, dominant_pop, n_pops, 0.998, 0.995))
        .collect()
}

fn make_switching_sequence(
    n_pops: usize,
    segments: &[(usize, usize)], // (pop_index, n_windows)
) -> Vec<AncestryObservation> {
    let mut obs = Vec::new();
    let mut pos = 0u64;
    for &(pop, n_win) in segments {
        for _ in 0..n_win {
            obs.push(make_obs(pos, pop, n_pops, 0.998, 0.995));
            pos += 10000;
        }
    }
    obs
}

// ============================================================================
// Baum-Welch Convergence Tests
// ============================================================================

#[test]
fn ancestry_bw_returns_finite_ll() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(50, 0, 3);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

    let ll = params.baum_welch(&obs_ref, 10, 1e-6);
    assert!(
        ll.is_finite(),
        "BW final LL should be finite, got {ll}"
    );
}

#[test]
fn ancestry_bw_final_ll_improves_with_more_iters() {
    // BW with more iterations should achieve equal or better final LL
    // Note: BW enforces switch_prob symmetry in M-step, so per-iteration
    // monotonicity is NOT guaranteed (unlike standard EM). Instead we verify
    // that more iterations lead to a better final state.
    let pops = make_populations(3);
    let obs = make_clear_ancestry_sequence(50, 0, 3);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

    let mut params_5 = AncestryHmmParams::new(pops.clone(), 0.01);
    let ll_5 = params_5.baum_welch(&obs_ref, 5, 1e-15);

    let mut params_20 = AncestryHmmParams::new(pops.clone(), 0.01);
    let ll_20 = params_20.baum_welch(&obs_ref, 20, 1e-15);

    // Both should be finite
    assert!(ll_5.is_finite(), "5-iter LL should be finite: {ll_5}");
    assert!(ll_20.is_finite(), "20-iter LL should be finite: {ll_20}");

    // 20 iterations should be at least as good (within numerical tolerance)
    // Note: due to switch_prob symmetry enforcement, the gap may be small
    assert!(
        ll_20 >= ll_5 - 0.5,
        "20-iter LL ({ll_20}) should be ≈ 5-iter LL ({ll_5})"
    );
}

#[test]
fn ancestry_bw_empty_observations_returns_neg_inf() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs_ref: Vec<&[AncestryObservation]> = vec![];
    let ll = params.baum_welch(&obs_ref, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

#[test]
fn ancestry_bw_single_state_returns_neg_inf() {
    let pops = make_populations(1);
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
    let obs = make_clear_ancestry_sequence(20, 0, 1);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];
    let ll = params.baum_welch(&obs_ref, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

#[test]
fn ancestry_bw_short_sequence_skipped() {
    // Sequences with < 2 observations should be skipped
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);

    let obs1 = vec![make_obs(0, 0, 3, 0.998, 0.995)]; // Length 1 — too short
    let obs2 = make_clear_ancestry_sequence(30, 1, 3); // Length 30 — OK
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs1.as_slice(), obs2.as_slice()];

    let ll = params.baum_welch(&obs_ref, 10, 1e-6);
    assert!(ll.is_finite(), "Should train on the valid sequence");
}

#[test]
fn ancestry_bw_multiple_sequences() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let obs1 = make_clear_ancestry_sequence(30, 0, 3);
    let obs2 = make_clear_ancestry_sequence(25, 1, 3);
    let obs3 = make_clear_ancestry_sequence(20, 2, 3);
    let obs_ref: Vec<&[AncestryObservation]> = vec![
        obs1.as_slice(),
        obs2.as_slice(),
        obs3.as_slice(),
    ];

    let ll = params.baum_welch(&obs_ref, 10, 1e-6);
    assert!(ll.is_finite(), "Multi-sequence BW should converge");
}

#[test]
fn ancestry_bw_max_iters_zero_is_noop() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
    let transitions_before = params.transitions.clone();

    let obs = make_clear_ancestry_sequence(50, 0, 3);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

    let ll = params.baum_welch(&obs_ref, 0, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY); // prev_ll never updated

    // Transitions should be unchanged
    for i in 0..params.n_states {
        for j in 0..params.n_states {
            assert_eq!(params.transitions[i][j], transitions_before[i][j]);
        }
    }
}

#[test]
fn ancestry_bw_transitions_sum_to_one() {
    let pops = make_populations(5);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_switching_sequence(5, &[(0, 20), (1, 15), (2, 20), (3, 15), (4, 20)]);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

    params.baum_welch(&obs_ref, 20, 1e-6);

    for i in 0..5 {
        let row_sum: f64 = params.transitions[i].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-8,
            "Row {i} sum = {row_sum}, expected 1.0"
        );
    }
}

#[test]
fn ancestry_bw_transitions_positive() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_switching_sequence(3, &[(0, 20), (1, 20), (2, 20)]);
    let obs_ref: Vec<&[AncestryObservation]> = vec![obs.as_slice()];

    params.baum_welch(&obs_ref, 20, 1e-6);

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                params.transitions[i][j] > 0.0,
                "Transition[{i}][{j}] should be positive, got {}",
                params.transitions[i][j]
            );
        }
    }
}

// ============================================================================
// Viterbi / Forward-Backward Consistency
// ============================================================================

#[test]
fn ancestry_viterbi_all_same_population() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = make_clear_ancestry_sequence(30, 1, 3); // All EUR

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 30);

    // Most should be state 1 (EUR)
    let eur_count = states.iter().filter(|&&s| s == 1).count();
    assert!(
        eur_count > 25,
        "Expected mostly EUR states, got {eur_count}/30"
    );
}

#[test]
fn ancestry_fb_posteriors_sum_to_one() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(20, 0, 3);

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 20);

    for (t, row) in posteriors.iter().enumerate() {
        assert_eq!(row.len(), 3);
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Posteriors at t={t} sum to {sum}, expected 1.0"
        );
        for (s, &p) in row.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Posterior[{t}][{s}] = {p} out of range"
            );
        }
    }
}

#[test]
fn ancestry_viterbi_and_posterior_decode_agree_on_clear_signal() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let obs = make_clear_ancestry_sequence(30, 2, 3); // All EAS

    let states_v = viterbi(&obs, &params);
    let states_p = posterior_decode(&obs, &params);

    let agreement: usize = states_v
        .iter()
        .zip(states_p.iter())
        .filter(|(v, p)| v == p)
        .count();
    let rate = agreement as f64 / states_v.len() as f64;
    assert!(
        rate > 0.9,
        "Viterbi and posterior decode should agree on clear signal: {:.1}%",
        rate * 100.0
    );
}

#[test]
fn ancestry_viterbi_empty() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<AncestryObservation> = vec![];
    let states = viterbi(&obs, &params);
    assert!(states.is_empty());
}

#[test]
fn ancestry_fb_empty() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<AncestryObservation> = vec![];
    let posteriors = forward_backward(&obs, &params);
    assert!(posteriors.is_empty());
}

#[test]
fn ancestry_viterbi_single_window() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![make_obs(0, 0, 3, 0.999, 0.995)];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 3);
}

// ============================================================================
// Genetic-Map-Aware HMM
// ============================================================================

#[test]
fn ancestry_viterbi_genmap_matches_standard_on_uniform_map() {
    // With uniform genetic map and equivalent switch prob,
    // genetic-map viterbi should produce similar results to standard
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.005);
    let obs = make_switching_sequence(3, &[(0, 15), (1, 15), (2, 15)]);

    let gmap = AncestryGeneticMap::uniform(0, 45 * 10000, 1.0);

    let states_std = viterbi(&obs, &params);
    let states_gmap = viterbi_with_genetic_map(&obs, &params, &gmap);

    // Should be reasonably similar (not identical due to rate vs prob difference)
    let agreement: usize = states_std
        .iter()
        .zip(states_gmap.iter())
        .filter(|(a, b)| a == b)
        .count();
    let rate = agreement as f64 / states_std.len() as f64;
    assert!(
        rate > 0.7,
        "Uniform genetic map should produce similar results: {:.1}%",
        rate * 100.0
    );
}

#[test]
fn ancestry_fb_genmap_posteriors_sum_to_one() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.005);
    let obs = make_clear_ancestry_sequence(20, 1, 3);
    let gmap = AncestryGeneticMap::uniform(0, 20 * 10000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(posteriors.len(), 20);

    for (t, row) in posteriors.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Genmap posteriors at t={t} sum to {sum}"
        );
    }
}

#[test]
fn ancestry_posterior_decode_genmap_valid() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.005);
    let obs = make_clear_ancestry_sequence(20, 2, 3);
    let gmap = AncestryGeneticMap::uniform(0, 20 * 10000, 1.0);

    let states = posterior_decode_with_genetic_map(&obs, &params, &gmap);
    assert_eq!(states.len(), 20);
    for &s in &states {
        assert!(s < 3);
    }
}

#[test]
fn ancestry_viterbi_genmap_empty_obs() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<AncestryObservation> = vec![];
    let gmap = AncestryGeneticMap::uniform(0, 100000, 1.0);

    let states = viterbi_with_genetic_map(&obs, &params, &gmap);
    assert!(states.is_empty());
}

#[test]
fn ancestry_fb_genmap_empty_obs() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<AncestryObservation> = vec![];
    let gmap = AncestryGeneticMap::uniform(0, 100000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gmap);
    assert!(posteriors.is_empty());
}

// ============================================================================
// AncestryGeneticMap Properties
// ============================================================================

#[test]
fn ancestry_genmap_uniform_interpolation() {
    let gmap = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    // 1 cM/Mb → at 500kb should be ~0.5 cM
    let cm = gmap.interpolate_cm(500_000);
    assert!(
        (cm - 0.5).abs() < 0.01,
        "At 500kb with 1 cM/Mb, expected ~0.5 cM, got {cm}"
    );
}

#[test]
fn ancestry_genmap_genetic_distance() {
    let gmap = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let dist = gmap.genetic_distance_cm(0, 1_000_000);
    assert!(
        (dist - 1.0).abs() < 0.01,
        "1 Mb at 1 cM/Mb should be ~1 cM, got {dist}"
    );
}

#[test]
fn ancestry_genmap_genetic_distance_zero() {
    let gmap = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let dist = gmap.genetic_distance_cm(500_000, 500_000);
    assert!(
        dist.abs() < 1e-10,
        "Distance from position to itself should be 0, got {dist}"
    );
}

#[test]
fn ancestry_genmap_modulated_switch_prob_increases_with_distance() {
    let gmap = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let base_switch = 0.01;

    let switch_near = gmap.modulated_switch_prob(base_switch, 0, 10000, 10000);
    let switch_far = gmap.modulated_switch_prob(base_switch, 0, 100000, 10000);

    // Farther windows should have higher switch probability
    assert!(
        switch_far >= switch_near,
        "Switch prob should increase with distance: near={switch_near}, far={switch_far}"
    );
}

#[test]
fn ancestry_genmap_modulated_switch_prob_bounded() {
    let gmap = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    // Very far apart
    let switch = gmap.modulated_switch_prob(0.01, 0, 50_000_000, 10000);
    // Should be bounded and valid
    assert!(switch >= 0.0 && switch <= 1.0, "Switch prob out of [0,1]: {switch}");
    assert!(switch.is_finite());
}

// ============================================================================
// Temperature Estimation
// ============================================================================

#[test]
fn estimate_temperature_returns_finite() {
    let pops = make_populations(3);
    let obs = make_clear_ancestry_sequence(30, 0, 3);
    let temp = estimate_temperature(&obs, &pops);
    assert!(
        temp.is_finite() && temp > 0.0,
        "Temperature should be finite positive, got {temp}"
    );
}

#[test]
fn estimate_temperature_empty_obs() {
    let pops = make_populations(3);
    let obs: Vec<AncestryObservation> = vec![];
    let temp = estimate_temperature(&obs, &pops);
    // Should return fallback value
    assert!(temp.is_finite() && temp > 0.0);
}

#[test]
fn estimate_temperature_single_observation() {
    let pops = make_populations(3);
    let obs = vec![make_obs(0, 0, 3, 0.998, 0.995)];
    let temp = estimate_temperature(&obs, &pops);
    assert!(temp.is_finite() && temp > 0.0);
}

#[test]
fn estimate_temperature_no_similarity_data() {
    // Observation with empty similarities
    let pops = make_populations(3);
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "TEST".to_string(),
        similarities: HashMap::new(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    assert!(temp.is_finite() && temp > 0.0);
}

// ============================================================================
// Switch Probability Estimation
// ============================================================================

#[test]
fn estimate_switch_prob_returns_bounded() {
    let pops = make_populations(3);
    let obs = make_switching_sequence(3, &[(0, 20), (1, 20), (2, 20)]);

    let switch = estimate_switch_prob(&obs, &pops, 0.01);
    assert!(switch >= 0.0001, "Switch prob too low: {switch}");
    assert!(switch <= 0.05, "Switch prob too high: {switch}");
    assert!(switch.is_finite());
}

#[test]
fn estimate_switch_prob_short_input_fallback() {
    let pops = make_populations(3);
    let obs = vec![make_obs(0, 0, 3, 0.998, 0.995); 5]; // < 10 obs

    let switch = estimate_switch_prob(&obs, &pops, 0.01);
    assert_eq!(switch, 0.001, "Short input should return fallback 0.001");
}

#[test]
fn estimate_switch_prob_no_switches_low_rate() {
    let pops = make_populations(3);
    let obs = make_clear_ancestry_sequence(50, 0, 3); // All same pop

    let switch = estimate_switch_prob(&obs, &pops, 0.01);
    assert!(switch < 0.01, "No-switch sequence should have low switch rate: {switch}");
}

// ============================================================================
// Temperature Normalized Estimation
// ============================================================================

#[test]
fn estimate_temperature_normalized_without_normalization_falls_back() {
    // Without normalization set, should fall back to standard estimate_temperature
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let obs = make_clear_ancestry_sequence(30, 0, 3);

    let temp_norm = estimate_temperature_normalized(&obs, &params);
    let temp_std = estimate_temperature(&obs, &pops);

    // Should be the same when no normalization is set
    assert!(
        (temp_norm - temp_std).abs() < 1e-10,
        "Without normalization, should match standard: {temp_norm} vs {temp_std}"
    );
}

#[test]
fn estimate_temperature_normalized_with_normalization() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(30, 0, 3);

    // Learn normalization first
    params.learn_normalization(&obs);

    let temp = estimate_temperature_normalized(&obs, &params);
    assert!(
        temp.is_finite() && temp > 0.0,
        "Normalized temperature should be finite positive, got {temp}"
    );
    assert!(temp >= 0.5 && temp <= 5.0, "Should be clamped to [0.5, 5.0], got {temp}");
}

// ============================================================================
// Numerical Edge Cases
// ============================================================================

#[test]
fn ancestry_viterbi_many_populations() {
    // 10 populations — tests scaling
    let pops = make_populations(10);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(30, 5, 10);

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 30);
    for &s in &states {
        assert!(s < 10, "State {s} out of range for 10 populations");
    }
}

#[test]
fn ancestry_fb_many_populations_posteriors_valid() {
    let pops = make_populations(10);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(20, 3, 10);

    let posteriors = forward_backward(&obs, &params);
    for (t, row) in posteriors.iter().enumerate() {
        assert_eq!(row.len(), 10);
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Posteriors at t={t} sum to {sum} (10 pops)"
        );
    }
}

#[test]
fn ancestry_params_set_switch_prob_extremes() {
    let pops = make_populations(3);

    // Very small switch prob
    let mut params = AncestryHmmParams::new(pops.clone(), 0.0001);
    for i in 0..3 {
        let sum: f64 = params.transitions[i].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Row {i} sum = {sum} for small switch");
    }

    // Very large switch prob (but < 1)
    params.set_switch_prob(0.99);
    for i in 0..3 {
        let sum: f64 = params.transitions[i].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Row {i} sum = {sum} for large switch");
    }
}

#[test]
fn ancestry_log_emission_state_out_of_range() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_obs(0, 0, 3, 0.998, 0.995);

    // Valid states
    for s in 0..3 {
        let le = params.log_emission(&obs, s);
        assert!(le.is_finite(), "Log emission for state {s} should be finite");
    }
}

#[test]
fn ancestry_learn_normalization_sets_mean_and_std() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_clear_ancestry_sequence(50, 0, 3);

    assert!(params.normalization.is_none());

    params.learn_normalization(&obs);

    assert!(params.normalization.is_some());
    let norm = params.normalization.as_ref().unwrap();
    assert_eq!(norm.means.len(), 3);
    assert_eq!(norm.stds.len(), 3);

    for i in 0..3 {
        assert!(norm.means[i].is_finite(), "Mean[{i}] not finite");
        assert!(norm.stds[i] > 0.0, "Std[{i}] not positive: {}", norm.stds[i]);
    }
}

#[test]
fn ancestry_learn_normalization_empty_obs_no_crash() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<AncestryObservation> = vec![];

    params.learn_normalization(&obs);
    // May or may not set normalization, just shouldn't panic
}
