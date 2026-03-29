//! Deep edge case tests for ancestry-cli
//!
//! Targets untested conditional branches, degenerate parameter states, and
//! algorithm boundary conditions discovered via systematic code analysis.

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    AncestryGeneticMap, PopulationNormalization,
    estimate_temperature_normalized, viterbi_with_genetic_map,
    forward_backward_with_genetic_map,
};

// ── Helper factories ──────────────────────────────────────────────────

fn make_pop(name: &str, haplotypes: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haplotypes.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs(start: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".into(),
        start,
        end: start + 10_000,
        sample: "QUERY".into(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ── 1. set_switch_prob with n_states = 1 ─────────────────────────────

#[test]
fn set_switch_prob_single_state_zero_switch() {
    // With 1 state, set_switch_prob divides by (1-1)=0.
    // When switch_prob=0, stay_prob=1.0, the loop only hits i==j,
    // so switch_each=NaN is never stored.
    let mut params = AncestryHmmParams::new(
        vec![make_pop("EUR", &["h1", "h2"])],
        0.0,
    );
    params.set_switch_prob(0.0);
    // The transition matrix should have stay_prob=1.0
    assert_eq!(params.transitions[0][0], 1.0);
}

#[test]
fn set_switch_prob_single_state_nonzero_switch() {
    // Single state: P(stay) must be 1.0 regardless of switch_prob
    let mut params = AncestryHmmParams::new(
        vec![make_pop("EUR", &["h1"])],
        0.0,
    );
    params.set_switch_prob(0.01);
    assert!((params.transitions[0][0] - 1.0).abs() < 1e-12);
}

#[test]
fn new_single_state_transitions_correct() {
    // Single state: P(stay)=1.0, P(init)=1.0
    let params = AncestryHmmParams::new(
        vec![make_pop("POP", &["h1", "h2"])],
        0.01,
    );
    assert_eq!(params.n_states, 1);
    assert!((params.transitions[0][0] - 1.0).abs() < 1e-12);
    assert!((params.initial[0] - 1.0).abs() < 1e-12);
}

// ── 2. log_emission_similarity_only with target_sim = 0.0 ───────────

#[test]
fn log_emission_target_sim_zero_returns_neg_infinity() {
    // When a population's aggregated similarity is exactly 0.0,
    // the Some(s) if s > 0.0 guard fails → NEG_INFINITY
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    // ha1 gets 0.0 similarity, hb1 gets 0.9
    let obs = make_obs(0, &[("ha1", 0.0), ("hb1", 0.9)]);

    // State 0 (pop A) should get NEG_INFINITY emission
    let log_em_a = params.log_emission(&obs, 0);
    assert!(log_em_a.is_infinite() && log_em_a < 0.0,
        "Expected NEG_INFINITY for zero-similarity population, got {}", log_em_a);

    // State 1 (pop B) should get log(1) = 0 since it's the only valid pop
    let log_em_b = params.log_emission(&obs, 1);
    assert!((log_em_b - 0.0).abs() < 1e-10,
        "Expected 0.0 (sole valid population), got {}", log_em_b);
}

#[test]
fn viterbi_never_assigns_zero_similarity_state() {
    // If one population has 0.0 similarity throughout, Viterbi should
    // never assign that state (NEG_INFINITY emission blocks it)
    let pops = vec![
        make_pop("DEAD", &["hd1"]),
        make_pop("ALIVE", &["ha1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    let observations: Vec<_> = (0..10).map(|i| {
        make_obs(i * 10_000, &[("hd1", 0.0), ("ha1", 0.95)])
    }).collect();

    let states = hprc_ancestry_cli::viterbi(&observations, &params);
    assert!(states.iter().all(|&s| s == 1),
        "Viterbi should never assign the zero-similarity state");
}

// ── 3. estimate_temperature_normalized with all-equal z-scores ───────

#[test]
fn estimate_temperature_normalized_all_equal_zscore_returns_fallback() {
    // When all populations give identical z-scores, diffs is empty → fallback 1.0
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);

    // Set normalization with same mean/std for both populations
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.9],
        stds: vec![0.05, 0.05],
    });

    // All observations have identical similarities for both pops
    // z-scores will be identical → max_z == min_z → no diffs pushed
    let observations: Vec<_> = (0..20).map(|i| {
        make_obs(i * 10_000, &[("ha1", 0.9), ("hb1", 0.9)])
    }).collect();

    let temp = estimate_temperature_normalized(&observations, &params);
    assert!((temp - 1.0).abs() < 1e-12,
        "Expected fallback 1.0 when all z-scores equal, got {}", temp);
}

#[test]
fn estimate_temperature_normalized_no_normalization_delegates() {
    // When normalization is None, should delegate to estimate_temperature
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);
    assert!(params.normalization.is_none());

    let observations: Vec<_> = (0..20).map(|i| {
        make_obs(i * 10_000, &[("ha1", 0.95), ("hb1", 0.85)])
    }).collect();

    let temp = estimate_temperature_normalized(&observations, &params);
    // Should return a clamped value from estimate_temperature (between 0.01 and 0.15)
    assert!(temp >= 0.01 && temp <= 0.15,
        "Temperature {} out of expected range [0.01, 0.15]", temp);
}

#[test]
fn estimate_temperature_normalized_single_population_observed() {
    // When only one population has data per window, z_scores.len() < 2 → no diffs
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.8],
        stds: vec![0.05, 0.05],
    });

    // Only ha1 has data, hb1 missing
    let observations: Vec<_> = (0..20).map(|i| {
        make_obs(i * 10_000, &[("ha1", 0.95)])
    }).collect();

    let temp = estimate_temperature_normalized(&observations, &params);
    assert!((temp - 1.0).abs() < 1e-12,
        "Expected fallback 1.0 when only one pop has data, got {}", temp);
}

// ── 4. genetic_map_transition_log with k = 1 (single state) ─────────

#[test]
fn viterbi_genetic_map_single_state() {
    // With k=1, the .max(1) guard in genetic_map_transition_log prevents /0
    let pops = vec![make_pop("SOLO", &["h1"])];
    let params = AncestryHmmParams::new(pops, 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);

    let observations: Vec<_> = (0..5).map(|i| {
        make_obs(i * 10_000, &[("h1", 0.95)])
    }).collect();

    let states = viterbi_with_genetic_map(&observations, &params, &gm);
    assert_eq!(states.len(), 5);
    assert!(states.iter().all(|&s| s == 0),
        "Single-state Viterbi should always assign state 0");
}

#[test]
fn fb_genetic_map_single_state_posteriors_sum_to_one() {
    let pops = vec![make_pop("SOLO", &["h1"])];
    let params = AncestryHmmParams::new(pops, 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);

    let observations: Vec<_> = (0..3).map(|i| {
        make_obs(i * 10_000, &[("h1", 0.9)])
    }).collect();

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    assert_eq!(posteriors.len(), 3);
    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 1);
        assert!((post[0] - 1.0).abs() < 1e-6,
            "Posterior at t={} should be 1.0 for single state, got {}", t, post[0]);
    }
}

// ── 5. AncestryGeneticMap interpolate_cm with bp_hi == bp_lo ─────────

#[test]
fn ancestry_genetic_map_duplicate_position_interpolation() {
    // Manually construct a map with duplicate positions by writing to temp file
    let dir = std::env::temp_dir().join("ancestry_genmap_dup_test");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("dup_map.txt");
    // Two entries at position 1000 (same bp), then one at 2000
    std::fs::write(&path, "chr1 1000 0.0 2.0\nchr1 1000 0.0 3.0\nchr1 2000 0.0 4.0\n").unwrap();

    let gm = AncestryGeneticMap::from_file(&path, "chr1").unwrap();

    // At position 1000, both entries exist; partition_point will put idx
    // past both duplicate entries (idx=2), so entries[1]=(1000,3.0) and entries[2]=(2000,4.0)
    // → normal interpolation: frac=0, result=3.0
    let cm = gm.interpolate_cm(1000);
    assert!(cm.is_finite(), "Should not panic on duplicate positions");
    // Value should be deterministic (exact depends on sort stability)
    assert!((cm - 3.0).abs() < 1e-6 || (cm - 2.0).abs() < 1e-6,
        "Expected cm near 2.0 or 3.0 at duplicate position, got {}", cm);

    std::fs::remove_dir_all(&dir).ok();
}

// ── 6. fb_genetic_map_single_observation_posterior_sums ──────────────

#[test]
fn fb_genetic_map_single_obs_posterior_sums_to_one() {
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);

    let observations = vec![make_obs(500_000, &[("ha1", 0.95), ("hb1", 0.85)])];

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    assert_eq!(posteriors.len(), 1);
    let sum: f64 = posteriors[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6,
        "Single-observation posterior should sum to 1.0, got {}", sum);
    // State 0 (A) should have higher posterior since ha1 has higher similarity
    assert!(posteriors[0][0] > posteriors[0][1],
        "State with higher similarity should have higher posterior");
}

// ── 7. log_emission with all populations missing data ────────────────

#[test]
fn log_emission_all_populations_missing() {
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    // No similarities match any reference haplotype
    let obs = make_obs(0, &[("unknown_hap", 0.95)]);

    let log_em_a = params.log_emission(&obs, 0);
    let log_em_b = params.log_emission(&obs, 1);
    assert!(log_em_a.is_infinite() && log_em_a < 0.0);
    assert!(log_em_b.is_infinite() && log_em_b < 0.0);
}

// ── 8. set_switch_prob with 3 states verifies uniform off-diagonal ───

#[test]
fn set_switch_prob_three_states_uniform_off_diagonal() {
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
        make_pop("C", &["hc1"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_switch_prob(0.06); // switch_each = 0.06/2 = 0.03

    for i in 0..3 {
        let mut row_sum: f64 = 0.0;
        for j in 0..3 {
            if i == j {
                assert!((params.transitions[i][j] - 0.94).abs() < 1e-12);
            } else {
                assert!((params.transitions[i][j] - 0.03).abs() < 1e-12);
            }
            row_sum += params.transitions[i][j];
        }
        assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
    }
}

// ── 9. estimate_temperature_normalized with distinct z-scores ────────

#[test]
fn estimate_temperature_normalized_distinct_zscores_clamped() {
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.normalization = Some(PopulationNormalization {
        means: vec![0.9, 0.7],
        stds: vec![0.05, 0.05],
    });

    // Create observations with very different z-scores (large diff → clamped to 5.0)
    let observations: Vec<_> = (0..20).map(|i| {
        make_obs(i * 10_000, &[("ha1", 0.99), ("hb1", 0.5)])
    }).collect();

    let temp = estimate_temperature_normalized(&observations, &params);
    // z_a = (0.99-0.9)/0.05 = 1.8, z_b = (0.5-0.7)/0.05 = -4.0
    // diff = 1.8 - (-4.0) = 5.8, clamped to 5.0
    assert!((temp - 5.0).abs() < 1e-6,
        "Expected clamped temperature 5.0, got {}", temp);
}

// ── 10. learn_normalization with no observations ─────────────────────

#[test]
fn learn_normalization_empty_observations_safe() {
    let pops = vec![
        make_pop("A", &["ha1"]),
        make_pop("B", &["hb1"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let obs: Vec<AncestryObservation> = vec![];

    params.learn_normalization(&obs);
    // Should have set normalization (with zero means, small stds)
    assert!(params.normalization.is_some());
    let norm = params.normalization.as_ref().unwrap();
    assert_eq!(norm.means.len(), 2);
    assert_eq!(norm.stds.len(), 2);
    // With no data, means should be 0 and stds should be 1e-6
    for &m in &norm.means { assert!((m - 0.0).abs() < 1e-10); }
    for &s in &norm.stds { assert!((s - 1e-6).abs() < 1e-10); }
}

// ── 11. viterbi with empty observations ──────────────────────────────

#[test]
fn viterbi_genetic_map_empty_returns_empty() {
    let pops = vec![make_pop("A", &["ha1"])];
    let params = AncestryHmmParams::new(pops, 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let observations: Vec<AncestryObservation> = vec![];

    let states = viterbi_with_genetic_map(&observations, &params, &gm);
    assert!(states.is_empty());
}

#[test]
fn fb_genetic_map_empty_returns_empty() {
    let pops = vec![make_pop("A", &["ha1"])];
    let params = AncestryHmmParams::new(pops, 0.001);
    let gm = AncestryGeneticMap::uniform(0, 1_000_000, 1.0);
    let observations: Vec<AncestryObservation> = vec![];

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    assert!(posteriors.is_empty());
}
