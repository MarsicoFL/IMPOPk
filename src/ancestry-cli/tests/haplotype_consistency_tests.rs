use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    apply_haplotype_consistency, forward_backward,
    precompute_log_emissions, viterbi,
};

fn make_pops() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR#1".to_string(), "EUR#2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR#1".to_string(), "AFR#2".to_string()],
        },
    ]
}

fn make_obs(start: u64, eur1: f64, eur2: f64, afr1: f64, afr2: f64) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: [
            ("EUR#1".to_string(), eur1),
            ("EUR#2".to_string(), eur2),
            ("AFR#1".to_string(), afr1),
            ("AFR#2".to_string(), afr2),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

// --- apply_haplotype_consistency basic tests ---

#[test]
fn test_consistency_disabled_when_context_zero() {
    let pops = make_pops();
    let params = AncestryHmmParams::new(pops, 0.01);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
    ];
    apply_haplotype_consistency(&mut obs, &params, 0, 0.5);
    // No bonus should be applied
    assert!(obs[0].haplotype_consistency_bonus.is_none());
    assert!(obs[1].haplotype_consistency_bonus.is_none());
}

#[test]
fn test_consistency_empty_observations() {
    let pops = make_pops();
    let params = AncestryHmmParams::new(pops, 0.01);
    let mut obs: Vec<AncestryObservation> = vec![];
    apply_haplotype_consistency(&mut obs, &params, 3, 0.5);
    // Should not panic
}

#[test]
fn test_consistency_single_observation() {
    let pops = make_pops();
    let params = AncestryHmmParams::new(pops, 0.01);
    let mut obs = vec![make_obs(0, 0.999, 0.998, 0.997, 0.996)];
    apply_haplotype_consistency(&mut obs, &params, 3, 0.5);
    // Single obs has no neighbors, so bonus should be 0 for all pops
    let bonus = obs[0].haplotype_consistency_bonus.as_ref().unwrap();
    assert_eq!(bonus.len(), 2);
    assert_eq!(bonus[0], 0.0);
    assert_eq!(bonus[1], 0.0);
}

#[test]
fn test_consistency_perfect_consistency() {
    // All windows have the same best haplotype per population
    // EUR#1 is always best for EUR, AFR#1 is always best for AFR
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
        make_obs(20000, 0.999, 0.998, 0.997, 0.996),
        make_obs(30000, 0.999, 0.998, 0.997, 0.996),
        make_obs(40000, 0.999, 0.998, 0.997, 0.996),
    ];
    apply_haplotype_consistency(&mut obs, &params, 2, 1.0);

    // Middle window (index 2) has 4 neighbors in ±2 range, all consistent
    let bonus = obs[2].haplotype_consistency_bonus.as_ref().unwrap();
    assert_eq!(bonus.len(), 2);
    // Perfect consistency → bonus = weight * 1.0 * temperature = 1.0 * 0.003
    let expected_bonus = 1.0 * 1.0 * 0.003;
    assert!((bonus[0] - expected_bonus).abs() < 1e-10,
        "EUR bonus {} should be {}", bonus[0], expected_bonus);
    assert!((bonus[1] - expected_bonus).abs() < 1e-10,
        "AFR bonus {} should be {}", bonus[1], expected_bonus);
}

#[test]
fn test_consistency_no_consistency() {
    // Each window has a DIFFERENT best haplotype for EUR
    // Window 0: EUR#1 best, Window 1: EUR#2 best, Window 2: EUR#1 best, etc.
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),      // EUR#1 best
        make_obs(10000, 0.998, 0.999, 0.997, 0.996),  // EUR#2 best
        make_obs(20000, 0.999, 0.998, 0.997, 0.996),  // EUR#1 best
        make_obs(30000, 0.998, 0.999, 0.997, 0.996),  // EUR#2 best
        make_obs(40000, 0.999, 0.998, 0.997, 0.996),  // EUR#1 best
    ];
    apply_haplotype_consistency(&mut obs, &params, 1, 1.0);

    // For EUR: middle window (index 2) has EUR#1 best. Neighbors index 1 has EUR#2, index 3 has EUR#2.
    // Consistency = 0/2 = 0.0
    let bonus = obs[2].haplotype_consistency_bonus.as_ref().unwrap();
    assert!((bonus[0] - 0.0).abs() < 1e-10,
        "EUR bonus should be 0 for alternating best haps, got {}", bonus[0]);

    // AFR is always consistent (AFR#1 best always)
    let afr_expected = 1.0 * 1.0 * 0.003;
    assert!((bonus[1] - afr_expected).abs() < 1e-10,
        "AFR bonus should be {} for consistent AFR#1, got {}", afr_expected, bonus[1]);
}

#[test]
fn test_consistency_partial_matching() {
    // 3 windows, context=1. Middle window's EUR#1 matches left but not right.
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),      // EUR#1 best
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),  // EUR#1 best
        make_obs(20000, 0.998, 0.999, 0.997, 0.996),  // EUR#2 best
    ];
    apply_haplotype_consistency(&mut obs, &params, 1, 1.0);

    // Middle window (index 1): EUR#1. Neighbors: index 0 has EUR#1 (match), index 2 has EUR#2 (no match)
    let bonus = obs[1].haplotype_consistency_bonus.as_ref().unwrap();
    let expected = 1.0 * 0.5 * 0.003; // consistency = 1/2 = 0.5
    assert!((bonus[0] - expected).abs() < 1e-10,
        "EUR bonus should be {} for 50% consistency, got {}", expected, bonus[0]);
}

#[test]
fn test_consistency_weight_scaling() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs1 = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
        make_obs(20000, 0.999, 0.998, 0.997, 0.996),
    ];
    let mut obs2 = obs1.clone();

    apply_haplotype_consistency(&mut obs1, &params, 1, 0.5);
    apply_haplotype_consistency(&mut obs2, &params, 1, 1.0);

    let bonus1 = obs1[1].haplotype_consistency_bonus.as_ref().unwrap();
    let bonus2 = obs2[1].haplotype_consistency_bonus.as_ref().unwrap();

    // weight=1.0 should give 2x the bonus of weight=0.5
    assert!((bonus2[0] - 2.0 * bonus1[0]).abs() < 1e-10,
        "Double weight should give double bonus: {} vs {}", bonus2[0], bonus1[0]);
}

#[test]
fn test_consistency_temperature_scaling() {
    let pops = make_pops();
    let mut params1 = AncestryHmmParams::new(pops.clone(), 0.01);
    params1.set_temperature(0.003);
    let mut params2 = AncestryHmmParams::new(pops, 0.01);
    params2.set_temperature(0.006);

    let mut obs1 = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
        make_obs(20000, 0.999, 0.998, 0.997, 0.996),
    ];
    let mut obs2 = obs1.clone();

    apply_haplotype_consistency(&mut obs1, &params1, 1, 1.0);
    apply_haplotype_consistency(&mut obs2, &params2, 1, 1.0);

    let bonus1 = obs1[1].haplotype_consistency_bonus.as_ref().unwrap();
    let bonus2 = obs2[1].haplotype_consistency_bonus.as_ref().unwrap();

    // 2x temperature should give 2x the raw bonus (so softmax effect stays constant)
    assert!((bonus2[0] - 2.0 * bonus1[0]).abs() < 1e-10,
        "2x temperature should give 2x bonus: {} vs {}", bonus2[0], bonus1[0]);
}

// --- Effect on emissions ---

#[test]
fn test_consistency_boosts_emission_for_consistent_population() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    // Create ambiguous observations where EUR and AFR are very close
    let mut obs_no_bonus = vec![
        make_obs(0, 0.9993, 0.9990, 0.9992, 0.9989),
        make_obs(10000, 0.9993, 0.9990, 0.9992, 0.9989),
        make_obs(20000, 0.9993, 0.9990, 0.9992, 0.9989),
    ];
    // Get emissions without consistency
    let emissions_no_bonus = precompute_log_emissions(&obs_no_bonus, &params);

    // Apply consistency (EUR#1 is consistently best → EUR gets a boost)
    apply_haplotype_consistency(&mut obs_no_bonus, &params, 1, 0.5);
    let emissions_with_bonus = precompute_log_emissions(&obs_no_bonus, &params);

    // EUR emission should increase (or stay same) relative to AFR
    let eur_diff_no = emissions_no_bonus[1][0] - emissions_no_bonus[1][1]; // EUR - AFR
    let eur_diff_with = emissions_with_bonus[1][0] - emissions_with_bonus[1][1];

    // With both populations equally consistent, the relative difference should stay similar
    // (both get equal boost). The absolute emissions change but log-softmax normalizes.
    // Both EUR and AFR have stable best haplotypes, so both get equal bonus.
    // The relative ordering should be preserved.
    assert!((eur_diff_with - eur_diff_no).abs() < 0.1,
        "Equal consistency shouldn't change relative log-emissions much: diff_no={}, diff_with={}",
        eur_diff_no, eur_diff_with);
}

#[test]
fn test_consistency_favors_consistent_over_inconsistent() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    // EUR is consistent (same best hap), AFR alternates best haplotype
    let mut obs = vec![
        make_obs(0, 0.9993, 0.9990, 0.9992, 0.9989),      // EUR#1, AFR#1 best
        make_obs(10000, 0.9993, 0.9990, 0.9989, 0.9992),  // EUR#1, AFR#2 best
        make_obs(20000, 0.9993, 0.9990, 0.9992, 0.9989),  // EUR#1, AFR#1 best
    ];

    // Without consistency, EUR has slight edge (0.9993 vs 0.9992)
    let emissions_before = precompute_log_emissions(&obs, &params);
    let eur_advantage_before = emissions_before[1][0] - emissions_before[1][1];

    // Apply consistency: EUR is consistent (all EUR#1), AFR alternates
    apply_haplotype_consistency(&mut obs, &params, 1, 0.5);
    let emissions_after = precompute_log_emissions(&obs, &params);
    let eur_advantage_after = emissions_after[1][0] - emissions_after[1][1];

    // EUR advantage should INCREASE because EUR is more consistent
    assert!(eur_advantage_after > eur_advantage_before,
        "Consistent EUR should have bigger advantage: before={:.4}, after={:.4}",
        eur_advantage_before, eur_advantage_after);
}

// --- Effect on HMM decoding ---

#[test]
fn test_consistency_improves_viterbi_on_ambiguous_data() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    // Create data where EUR is the true ancestry but AFR is very close
    // EUR has consistent best haplotype (EUR#1 always best)
    // AFR alternates (sometimes AFR#1, sometimes AFR#2)
    let mut obs: Vec<AncestryObservation> = (0..10).map(|i| {
        let afr1 = if i % 2 == 0 { 0.9992 } else { 0.9989 };
        let afr2 = if i % 2 == 0 { 0.9989 } else { 0.9992 };
        make_obs(i as u64 * 10000, 0.9993, 0.9990, afr1, afr2)
    }).collect();

    // Without consistency
    let states_before = viterbi(&obs, &params);

    // With consistency
    apply_haplotype_consistency(&mut obs, &params, 3, 0.5);
    let states_after = viterbi(&obs, &params);

    // Count EUR assignments (state 0) — should increase or stay same
    let eur_before: usize = states_before.iter().filter(|&&s| s == 0).count();
    let eur_after: usize = states_after.iter().filter(|&&s| s == 0).count();
    assert!(eur_after >= eur_before,
        "Consistency should favor EUR (consistent best hap): before={}, after={}",
        eur_before, eur_after);
}

#[test]
fn test_consistency_preserves_clear_signal() {
    // When signal is already clear, consistency shouldn't change the result
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    let mut obs: Vec<AncestryObservation> = (0..10).map(|i| {
        // Clear EUR ancestry (big gap)
        make_obs(i as u64 * 10000, 0.999, 0.998, 0.995, 0.994)
    }).collect();

    let states_before = viterbi(&obs, &params);
    apply_haplotype_consistency(&mut obs, &params, 3, 0.5);
    let states_after = viterbi(&obs, &params);

    assert_eq!(states_before, states_after,
        "Clear signal should not be affected by consistency bonus");
}

// --- Edge cases ---

#[test]
fn test_consistency_context_larger_than_sequence() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
    ];
    // Context=10 but only 2 observations
    apply_haplotype_consistency(&mut obs, &params, 10, 0.5);
    // Should not panic, and both should have bonuses
    assert!(obs[0].haplotype_consistency_bonus.is_some());
    assert!(obs[1].haplotype_consistency_bonus.is_some());
}

#[test]
fn test_consistency_missing_haplotypes() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    // Some windows missing haplotypes
    let mut obs = vec![
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: 0,
            end: 10000,
            sample: "query#1".to_string(),
            similarities: [("EUR#1".to_string(), 0.999)].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        },
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: 10000,
            end: 20000,
            sample: "query#1".to_string(),
            similarities: [("AFR#1".to_string(), 0.997)].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        },
    ];
    apply_haplotype_consistency(&mut obs, &params, 1, 0.5);
    // Should not panic
    let bonus0 = obs[0].haplotype_consistency_bonus.as_ref().unwrap();
    let bonus1 = obs[1].haplotype_consistency_bonus.as_ref().unwrap();
    assert_eq!(bonus0.len(), 2);
    assert_eq!(bonus1.len(), 2);
}

#[test]
fn test_consistency_zero_weight() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);
    let mut obs = vec![
        make_obs(0, 0.999, 0.998, 0.997, 0.996),
        make_obs(10000, 0.999, 0.998, 0.997, 0.996),
        make_obs(20000, 0.999, 0.998, 0.997, 0.996),
    ];
    apply_haplotype_consistency(&mut obs, &params, 1, 0.0);
    // All bonuses should be 0 regardless of consistency
    let bonus = obs[1].haplotype_consistency_bonus.as_ref().unwrap();
    assert_eq!(bonus[0], 0.0);
    assert_eq!(bonus[1], 0.0);
}

#[test]
fn test_consistency_bonus_stored_per_population() {
    let pops = vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR#1".to_string(), "EUR#2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR#1".to_string(), "AFR#2".to_string()],
        },
        AncestralPopulation {
            name: "EAS".to_string(),
            haplotypes: vec!["EAS#1".to_string(), "EAS#2".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    let make_3pop = |start: u64| -> AncestryObservation {
        AncestryObservation {
            chrom: "chr1".to_string(),
            start,
            end: start + 10000,
            sample: "query#1".to_string(),
            similarities: [
                ("EUR#1".to_string(), 0.999),
                ("EUR#2".to_string(), 0.998),
                ("AFR#1".to_string(), 0.997),
                ("AFR#2".to_string(), 0.996),
                ("EAS#1".to_string(), 0.995),
                ("EAS#2".to_string(), 0.994),
            ].into_iter().collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    };

    let mut obs: Vec<AncestryObservation> = (0..5).map(|i| make_3pop(i as u64 * 10000)).collect();
    apply_haplotype_consistency(&mut obs, &params, 2, 0.5);

    let bonus = obs[2].haplotype_consistency_bonus.as_ref().unwrap();
    assert_eq!(bonus.len(), 3, "Should have one bonus per population");
    // All three populations are perfectly consistent
    let expected = 0.5 * 1.0 * 0.003;
    for (i, &b) in bonus.iter().enumerate() {
        assert!((b - expected).abs() < 1e-10,
            "Pop {} bonus {} should be {}", i, b, expected);
    }
}

#[test]
fn test_consistency_edge_windows_have_fewer_neighbors() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    let mut obs: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs(i as u64 * 10000, 0.999, 0.998, 0.997, 0.996))
        .collect();

    apply_haplotype_consistency(&mut obs, &params, 2, 1.0);

    // Window 0: only has 2 neighbors (1, 2) → still gets consistency from them
    let bonus_edge = obs[0].haplotype_consistency_bonus.as_ref().unwrap()[0];
    // Window 2: has 4 neighbors (0, 1, 3, 4)
    let bonus_middle = obs[2].haplotype_consistency_bonus.as_ref().unwrap()[0];

    // Both should be the same consistency (1.0) since all windows are identical
    // but the raw bonus is the same: weight * 1.0 * temperature
    let expected = 1.0 * 1.0 * 0.003;
    assert!((bonus_edge - expected).abs() < 1e-10);
    assert!((bonus_middle - expected).abs() < 1e-10);
}

#[test]
fn test_consistency_interacts_with_forward_backward() {
    let pops = make_pops();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.003);

    let mut obs: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs(i as u64 * 10000, 0.9993, 0.9990, 0.9992, 0.9989))
        .collect();

    // Forward-backward should work with consistency bonus applied
    apply_haplotype_consistency(&mut obs, &params, 3, 0.5);
    let posteriors = forward_backward(&obs, &params);

    assert_eq!(posteriors.len(), 10);
    for post in &posteriors {
        assert_eq!(post.len(), 2);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors should sum to 1.0, got {}", sum);
    }
}
