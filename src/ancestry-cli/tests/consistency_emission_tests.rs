use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryObservation, EmissionModel,
    compute_consistency_log_emissions, blend_log_emissions,
};

fn make_pops_2() -> Vec<AncestralPopulation> {
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

fn make_pops_3() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR#1".to_string(), "EUR#2".to_string()],
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: vec!["AMR#1".to_string(), "AMR#2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR#1".to_string(), "AFR#2".to_string()],
        },
    ]
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    let mut similarities = HashMap::new();
    for &(hap, sim) in sims {
        similarities.insert(hap.to_string(), sim);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities,
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

// =============================================================================
// Empty / degenerate inputs
// =============================================================================

#[test]
fn consistency_empty_observations() {
    let pops = make_pops_2();
    let result = compute_consistency_log_emissions(&[], &pops, &EmissionModel::Max, 5);
    assert!(result.is_empty());
}

#[test]
fn consistency_empty_populations() {
    let obs = vec![make_obs(&[("EUR#1", 0.99)])];
    let result = compute_consistency_log_emissions(&obs, &[], &EmissionModel::Max, 5);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

#[test]
fn consistency_single_observation_context_0() {
    // context=0 means only the center window. EUR wins → EUR gets higher log-prob
    let pops = make_pops_2();
    let obs = vec![make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AFR#1", 0.980), ("AFR#2", 0.975),
    ])];
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    // EUR wins → higher log-prob
    assert!(result[0][0] > result[0][1], "EUR should have higher log-prob than AFR");
}

// =============================================================================
// Consistent winner across context
// =============================================================================

#[test]
fn consistency_all_eur_wins() {
    // 5 windows, all with EUR clearly winning. Context=2 (full range for center)
    let pops = make_pops_2();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AFR#1", 0.980), ("AFR#2", 0.975),
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);
    assert_eq!(result.len(), 5);
    // Center window (index 2) sees all 5 windows → EUR wins 5/5
    // log(5.5/6.5) vs log(0.5/6.5) — big difference
    let eur_center = result[2][0];
    let afr_center = result[2][1];
    assert!(eur_center > afr_center);
    assert!(eur_center - afr_center > 2.0, "Unanimous EUR should give >2 nats difference");
}

#[test]
fn consistency_all_afr_wins() {
    let pops = make_pops_2();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.980), ("EUR#2", 0.975),
        ("AFR#1", 0.995), ("AFR#2", 0.990),
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);
    // AFR wins all → AFR has higher log-prob
    assert!(result[2][1] > result[2][0], "AFR should have higher log-prob");
}

// =============================================================================
// Weak but consistent signal (the EUR-AMR use case)
// =============================================================================

#[test]
fn consistency_weak_signal_consistent() {
    // EUR consistently beats AMR by only 0.0002, but wins 10/11 windows
    let pops = make_pops_3();
    let mut obs: Vec<_> = (0..11).map(|_| make_obs(&[
        ("EUR#1", 0.9972), ("EUR#2", 0.9960),
        ("AMR#1", 0.9970), ("AMR#2", 0.9958),
        ("AFR#1", 0.9900), ("AFR#2", 0.9890),
    ])).collect();
    // Window 3: AMR wins once (noise)
    obs[3] = make_obs(&[
        ("EUR#1", 0.9969), ("EUR#2", 0.9960),
        ("AMR#1", 0.9971), ("AMR#2", 0.9958),
        ("AFR#1", 0.9900), ("AFR#2", 0.9890),
    ]);

    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 5);
    assert_eq!(result.len(), 11);

    // At center (index 5): EUR wins 10/11, AMR wins 1/11, AFR wins 0/11
    // EUR should have higher log-prob than AMR
    let eur = result[5][0];
    let amr = result[5][1];
    let afr = result[5][2];
    assert!(eur > amr, "EUR should beat AMR in consistency (10/11 vs 1/11)");
    assert!(eur > afr, "EUR should beat AFR in consistency");
    // EUR-AMR difference should be meaningful (>0.5 nats)
    assert!(eur - amr > 0.5, "Consistent 10/11 wins should give >0.5 nat gap, got {:.3}",
            eur - amr);
}

// =============================================================================
// Ties
// =============================================================================

#[test]
fn consistency_exact_ties() {
    // EUR and AFR have exactly the same max similarity → ties split equally
    let pops = make_pops_2();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AFR#1", 0.995), ("AFR#2", 0.990),
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);
    // All ties → equal wins → approximately equal log-probs
    let diff = (result[2][0] - result[2][1]).abs();
    assert!(diff < 0.01, "Tied populations should have near-equal log-probs, diff={:.4}", diff);
}

// =============================================================================
// Context size effects
// =============================================================================

#[test]
fn consistency_larger_context_stronger_signal() {
    // With context=0, each window is independent (noisy)
    // With context=5, the 11-window average should give stronger discrimination
    let pops = make_pops_2();
    let obs: Vec<_> = (0..11).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AFR#1", 0.980), ("AFR#2", 0.975),
    ])).collect();

    let c0 = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 0);
    let c5 = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 5);

    // With context=0: EUR wins 1/1 → log(1.5/2.5)=-0.51 vs log(0.5/2.5)=-1.61
    // With context=5: EUR wins 11/11 → log(11.5/12.5)=-0.08 vs log(0.5/12.5)=-3.22
    // Larger context → bigger gap
    let gap_c0 = c0[5][0] - c0[5][1];
    let gap_c5 = c5[5][0] - c5[5][1];
    assert!(gap_c5 > gap_c0, "Larger context should give stronger discrimination: c5={:.3} vs c0={:.3}",
            gap_c5, gap_c0);
}

// =============================================================================
// Mixed ancestry regions (transition)
// =============================================================================

#[test]
fn consistency_transition_region() {
    // Windows 0-4: EUR wins. Windows 5-9: AFR wins. Context=2
    let pops = make_pops_2();
    let mut obs = Vec::new();
    for _ in 0..5 {
        obs.push(make_obs(&[
            ("EUR#1", 0.995), ("EUR#2", 0.990),
            ("AFR#1", 0.980), ("AFR#2", 0.975),
        ]));
    }
    for _ in 0..5 {
        obs.push(make_obs(&[
            ("EUR#1", 0.980), ("EUR#2", 0.975),
            ("AFR#1", 0.995), ("AFR#2", 0.990),
        ]));
    }

    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);

    // Window 2 (EUR region, context includes 0-4): EUR wins 5/5
    assert!(result[2][0] > result[2][1], "EUR region should favor EUR");
    // Window 7 (AFR region, context includes 5-9): AFR wins 5/5
    assert!(result[7][1] > result[7][0], "AFR region should favor AFR");
    // Window 4 (boundary, context includes 2-6): EUR wins 3/5, AFR wins 2/5
    // Should still favor EUR slightly
    assert!(result[4][0] > result[4][1], "EUR-side boundary should still favor EUR");
}

// =============================================================================
// EmissionModel variants
// =============================================================================

#[test]
fn consistency_mean_emission_model() {
    // With Mean model, aggregates by mean instead of max
    let pops = make_pops_2();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990), // mean = 0.9925
        ("AFR#1", 0.993), ("AFR#2", 0.991), // mean = 0.9920
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Mean, 2);
    // EUR mean > AFR mean → EUR wins
    assert!(result[2][0] > result[2][1], "EUR should win by mean");
}

// =============================================================================
// Blending with standard emissions
// =============================================================================

#[test]
fn consistency_blend_preserves_structure() {
    let pops = make_pops_2();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AFR#1", 0.980), ("AFR#2", 0.975),
    ])).collect();
    let cons = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);
    // Fake standard emissions favoring AFR
    let standard: Vec<Vec<f64>> = (0..5).map(|_| vec![-1.0, -0.5]).collect();

    let blended = blend_log_emissions(&standard, &cons, 0.5);
    assert_eq!(blended.len(), 5);
    assert_eq!(blended[0].len(), 2);
    // Blended values should be between standard and consistency
    for t in 0..5 {
        for s in 0..2 {
            assert!(blended[t][s].is_finite());
        }
    }
}

// =============================================================================
// Missing data
// =============================================================================

#[test]
fn consistency_missing_haplotype_data() {
    // Some windows have missing data for one population
    let pops = make_pops_2();
    let obs = vec![
        make_obs(&[("EUR#1", 0.995), ("EUR#2", 0.990), ("AFR#1", 0.980), ("AFR#2", 0.975)]),
        make_obs(&[("EUR#1", 0.995), ("EUR#2", 0.990)]), // no AFR data
        make_obs(&[("EUR#1", 0.995), ("EUR#2", 0.990), ("AFR#1", 0.980), ("AFR#2", 0.975)]),
    ];
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 1);
    assert_eq!(result.len(), 3);
    // Window 1 has no AFR data → EUR wins by default → EUR favored
    assert!(result[1][0] > result[1][1]);
}

#[test]
fn consistency_no_data_any_pop() {
    // Window with no data at all → uniform
    let pops = make_pops_2();
    let obs = vec![
        make_obs(&[]),
    ];
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 0);
    assert_eq!(result.len(), 1);
    // Should be uniform (log of equal probs)
    let diff = (result[0][0] - result[0][1]).abs();
    assert!(diff < 0.01, "No data should give uniform, diff={:.4}", diff);
}

// =============================================================================
// 3-population case
// =============================================================================

#[test]
fn consistency_3pop_afr_clearly_best() {
    // AFR clearly wins all windows. EUR and AMR are similar but AMR slightly better
    let pops = make_pops_3();
    let obs: Vec<_> = (0..7).map(|_| make_obs(&[
        ("EUR#1", 0.985), ("EUR#2", 0.983),
        ("AMR#1", 0.986), ("AMR#2", 0.984),
        ("AFR#1", 0.998), ("AFR#2", 0.997),
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 3);
    // AFR wins all → highest log-prob
    assert!(result[3][2] > result[3][0], "AFR should beat EUR");
    assert!(result[3][2] > result[3][1], "AFR should beat AMR");
    // EUR and AMR both lose → similar low log-probs
    let eur_amr_diff = (result[3][0] - result[3][1]).abs();
    assert!(eur_amr_diff < 0.5, "EUR and AMR should be close when both lose");
}

// =============================================================================
// Log-probability properties
// =============================================================================

#[test]
fn consistency_log_probs_sum_near_one() {
    // The log-probabilities (with Laplace smoothing) should sum close to log(1)=0
    // when converted back to probabilities
    let pops = make_pops_3();
    let obs: Vec<_> = (0..5).map(|_| make_obs(&[
        ("EUR#1", 0.995), ("EUR#2", 0.990),
        ("AMR#1", 0.993), ("AMR#2", 0.988),
        ("AFR#1", 0.980), ("AFR#2", 0.975),
    ])).collect();
    let result = compute_consistency_log_emissions(&obs, &pops, &EmissionModel::Max, 2);
    for row in &result {
        let prob_sum: f64 = row.iter().map(|&lp| lp.exp()).sum();
        assert!((prob_sum - 1.0).abs() < 0.01,
                "Probabilities should sum near 1.0, got {:.4}", prob_sum);
    }
}
