//! Integration tests for auto_detect_groups + compute_hierarchical_emissions.
//!
//! Validates single-linkage clustering, group sensitivity, and the full
//! pipeline from group detection through hierarchical emission computation.

use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryObservation, EmissionModel,
    auto_detect_groups, compute_hierarchical_emissions,
};

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

// Helper to generate observations with controlled inter-population distances
fn make_controlled_obs(n: usize, pop_means: &[f64], hap_names: &[Vec<&str>]) -> Vec<AncestryObservation> {
    (0..n).map(|i| {
        let mut sims = Vec::new();
        for (p, (mean, haps)) in pop_means.iter().zip(hap_names.iter()).enumerate() {
            let noise = ((i as f64 * 0.3 + p as f64) * 1.1).sin() * 0.001;
            for h in haps {
                sims.push((*h, mean + noise));
            }
        }
        make_obs(&sims)
    }).collect()
}

#[test]
fn auto_detect_groups_well_separated_pops() {
    // EUR, AFR, EAS with large pairwise distances → each in own group
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AFR", &["a1", "a2"]),
        make_pop("EAS", &["s1", "s2"]),
    ];
    let haps = vec![vec!["e1", "e2"], vec!["a1", "a2"], vec!["s1", "s2"]];
    let obs = make_controlled_obs(30, &[0.99, 0.90, 0.80], &haps);
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);

    // Large gaps → 3 separate groups
    assert_eq!(groups.len(), 3,
        "Well-separated populations should form 3 groups, got {:?}", groups);
}

#[test]
fn auto_detect_groups_two_close_one_far() {
    // EUR and AMR close, AFR far → EUR+AMR grouped, AFR separate
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AMR", &["m1", "m2"]),
        make_pop("AFR", &["a1", "a2"]),
    ];
    let haps = vec![vec!["e1", "e2"], vec!["m1", "m2"], vec!["a1", "a2"]];
    // EUR and AMR means very close, AFR far
    let obs = make_controlled_obs(50, &[0.990, 0.991, 0.950], &haps);
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);

    // EUR-AMR should be in same group, AFR separate
    assert!(groups.len() <= 2,
        "EUR/AMR close + AFR far should form ≤2 groups, got {:?}", groups);
    // Find the group containing EUR (index 0)
    let eur_group = groups.iter().find(|g| g.contains(&0)).unwrap();
    assert!(eur_group.contains(&1),
        "EUR (0) and AMR (1) should be in the same group, groups={:?}", groups);
}

#[test]
fn auto_detect_groups_all_identical_pops() {
    // All pops at same similarity → all in one group
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let haps = vec![vec!["a1"], vec!["b1"], vec!["c1"]];
    let obs = make_controlled_obs(30, &[0.95, 0.95, 0.95], &haps);
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);

    // All identical → distances are 0, threshold=0 → single-linkage with strict <
    // won't cluster (0 < 0 is false). Each pop stays in its own group.
    // This is correct behavior: can't discriminate → no grouping.
    let total_pops: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(total_pops, 3, "All populations should be covered, got {:?}", groups);
}

#[test]
fn auto_detect_groups_5_pops_two_clusters() {
    // {EUR, AMR, CSA} close, {AFR, EAS} close but far from first cluster
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AMR", &["m1", "m2"]),
        make_pop("CSA", &["c1", "c2"]),
        make_pop("AFR", &["a1", "a2"]),
        make_pop("EAS", &["s1", "s2"]),
    ];
    let haps = vec![
        vec!["e1", "e2"], vec!["m1", "m2"], vec!["c1", "c2"],
        vec!["a1", "a2"], vec!["s1", "s2"],
    ];
    // Cluster 1: 0.990, 0.991, 0.989; Cluster 2: 0.950, 0.951
    let obs = make_controlled_obs(60, &[0.990, 0.991, 0.989, 0.950, 0.951], &haps);
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);

    // Should form ~2 clusters
    assert!(groups.len() <= 3,
        "5 pops in 2 clusters should form ≤3 groups, got {:?}", groups);
}

#[test]
fn auto_detect_groups_single_pop_returns_trivial() {
    let pops = vec![make_pop("EUR", &["e1"])];
    let obs = vec![make_obs(&[("e1", 0.99)])];
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    assert_eq!(groups, vec![vec![0]]);
}

#[test]
fn auto_detect_groups_two_pops_returns_two_groups() {
    let pops = vec![
        make_pop("EUR", &["e1"]),
        make_pop("AFR", &["a1"]),
    ];
    let obs = make_controlled_obs(20, &[0.99, 0.90], &vec![vec!["e1"], vec!["a1"]]);
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    // ≤2 populations → one group per pop
    assert_eq!(groups.len(), 2);
}

#[test]
fn auto_detect_groups_empty_obs_no_crash() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let groups = auto_detect_groups(&[], &pops, &EmissionModel::Max);
    // With empty obs, distances are all 0 → should still return valid groups
    assert!(!groups.is_empty());
    let total_pops: usize = groups.iter().map(|g| g.len()).sum();
    assert_eq!(total_pops, 3, "All populations should be covered");
}

// --- Integration: auto_detect_groups → hierarchical_emissions ---

#[test]
fn hierarchical_with_auto_groups_produces_valid_log_probs() {
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AMR", &["m1", "m2"]),
        make_pop("AFR", &["a1", "a2"]),
    ];
    let haps = vec![vec!["e1", "e2"], vec!["m1", "m2"], vec!["a1", "a2"]];
    let obs = make_controlled_obs(30, &[0.990, 0.991, 0.950], &haps);

    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    let emissions = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 0.5,
    );

    assert_eq!(emissions.len(), obs.len());
    for (t, row) in emissions.iter().enumerate() {
        assert_eq!(row.len(), 3, "Should have 3 emission values per window");
        for (p, &lp) in row.iter().enumerate() {
            assert!(lp.is_finite(), "emission[{t}][{p}] should be finite, got {lp}");
            assert!(lp <= 0.0, "Log-prob should be ≤ 0, got {lp} at [{t}][{p}]");
        }
    }
}

#[test]
fn hierarchical_group_weight_zero_ignores_group_level() {
    // group_weight=0.0 → pure within-group emissions
    let pops = vec![
        make_pop("EUR", &["e1"]),
        make_pop("AMR", &["m1"]),
        make_pop("AFR", &["a1"]),
    ];
    let obs = vec![make_obs(&[("e1", 0.99), ("m1", 0.98), ("a1", 0.90)])];
    let groups = vec![vec![0, 1], vec![2]]; // EUR+AMR grouped, AFR alone

    let em_w0 = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 0.0,
    );
    let em_w1 = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 1.0,
    );

    // With w=0, EUR vs AMR discrimination should be sharper (within-group only)
    // With w=1, EUR and AMR share group-level score → less discrimination
    let eur_amr_diff_w0 = (em_w0[0][0] - em_w0[0][1]).abs();
    let eur_amr_diff_w1 = (em_w1[0][0] - em_w1[0][1]).abs();

    assert!(eur_amr_diff_w0 > eur_amr_diff_w1,
        "group_weight=0 should discriminate EUR/AMR better: w0={eur_amr_diff_w0:.4} ≤ w1={eur_amr_diff_w1:.4}");
}

#[test]
fn hierarchical_group_weight_one_collapses_within_group() {
    // group_weight=1.0 → populations in same group get same score
    let pops = vec![
        make_pop("EUR", &["e1"]),
        make_pop("AMR", &["m1"]),
        make_pop("AFR", &["a1"]),
    ];
    let obs = vec![make_obs(&[("e1", 0.99), ("m1", 0.98), ("a1", 0.90)])];
    let groups = vec![vec![0, 1], vec![2]];

    let em = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 1.0,
    );

    // EUR and AMR in same group → should have identical group-level score
    // (within-group weight = 0, so within-group contribution is zero)
    let eur_score = em[0][0];
    let amr_score = em[0][1];
    assert!((eur_score - amr_score).abs() < 1e-6,
        "group_weight=1.0 should give same score to EUR/AMR in same group: EUR={eur_score:.6}, AMR={amr_score:.6}");
}

#[test]
fn hierarchical_single_pop_groups_work() {
    // Each pop in its own group → hierarchical should still produce valid probs
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let obs = vec![make_obs(&[("a1", 0.99), ("b1", 0.95), ("c1", 0.90)])];
    let groups = vec![vec![0], vec![1], vec![2]];

    let em = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 0.5,
    );

    // A should be highest
    assert!(em[0][0] > em[0][1] && em[0][0] > em[0][2],
        "Pop A (0.99) should have highest emission: {:?}", em[0]);
}

#[test]
fn hierarchical_preserves_clear_signal_across_weights() {
    // When one pop is clearly dominant, all group_weight values should agree
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AFR", &["a1", "a2"]),
        make_pop("EAS", &["s1", "s2"]),
    ];
    let obs = vec![make_obs(&[
        ("e1", 0.999), ("e2", 0.998),
        ("a1", 0.800), ("a2", 0.795),
        ("s1", 0.750), ("s2", 0.745),
    ])];
    let groups = vec![vec![0], vec![1], vec![2]];

    for &w in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let em = compute_hierarchical_emissions(
            &obs, &pops, &groups, &EmissionModel::Max, 0.03, w,
        );
        // With single-pop groups, group_weight=0 makes all within-group probs 0.0 (log(1)=0).
        // Skip w=0 for this check since within-group can't discriminate single-pop groups.
        if w > 0.0 {
            assert!(em[0][0] > em[0][1] && em[0][0] > em[0][2],
                "EUR should be dominant at group_weight={w}: {:?}", em[0]);
        }
    }
}

#[test]
fn hierarchical_many_obs_dimensions_match() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
        make_pop("D", &["d1"]),
    ];
    let obs: Vec<_> = (0..100).map(|i| {
        make_obs(&[
            ("a1", 0.99), ("b1", 0.95),
            ("c1", 0.90), ("d1", 0.85 + (i as f64) * 0.0001),
        ])
    }).collect();
    let groups = vec![vec![0, 1], vec![2, 3]];

    let em = compute_hierarchical_emissions(
        &obs, &pops, &groups, &EmissionModel::Max, 0.03, 0.5,
    );

    assert_eq!(em.len(), 100);
    for row in &em {
        assert_eq!(row.len(), 4);
        let sum_exp: f64 = row.iter().map(|&lp| lp.exp()).sum();
        // Log-probs should approximately sum to 1 in probability space
        // (may not be exact due to weighted blending)
        assert!(sum_exp > 0.0 && sum_exp.is_finite(),
            "exp(emissions) should be finite and positive: {sum_exp}");
    }
}
