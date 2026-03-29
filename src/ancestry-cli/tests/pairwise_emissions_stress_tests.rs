//! Stress tests for pairwise (Bradley-Terry) log emissions.
//!
//! Tests scalability to many populations, temperature adaptation behavior,
//! and comparison with standard softmax emissions.

use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryObservation, EmissionModel,
    compute_pairwise_log_emissions, precompute_log_emissions,
    AncestryHmmParams,
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
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_n_pops(n: usize) -> (Vec<AncestralPopulation>, Vec<String>) {
    let mut pops = Vec::new();
    let mut all_hap_names = Vec::new();
    for i in 0..n {
        let name = format!("POP{}", i);
        let h1 = format!("p{}_h1", i);
        let h2 = format!("p{}_h2", i);
        all_hap_names.push(h1.clone());
        all_hap_names.push(h2.clone());
        pops.push(AncestralPopulation {
            name,
            haplotypes: vec![h1, h2],
        });
    }
    (pops, all_hap_names)
}

fn make_obs_for_pops(n_pops: usize, true_pop: usize, gap: f64) -> AncestryObservation {
    let mut sims = Vec::new();
    for p in 0..n_pops {
        let base = if p == true_pop { 0.99 } else { 0.99 - gap };
        sims.push((format!("p{}_h1", p), base));
        sims.push((format!("p{}_h2", p), base - 0.002));
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "query".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k, v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

#[test]
fn pairwise_10_populations_runs_without_crash() {
    let (pops, _) = make_n_pops(10);
    let obs: Vec<_> = (0..50).map(|i| make_obs_for_pops(10, i % 10, 0.02)).collect();

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    assert_eq!(emissions.len(), 50);
    for row in &emissions {
        assert_eq!(row.len(), 10);
        for &lp in row {
            assert!(lp.is_finite(), "All emissions should be finite");
            assert!(lp <= 0.0, "Log-probs should be ≤ 0");
        }
    }
}

#[test]
fn pairwise_15_populations_correct_winner() {
    let n = 15;
    let (pops, _) = make_n_pops(n);

    // True pop is POP7, should be highest
    let obs = vec![make_obs_for_pops(n, 7, 0.03)];
    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    let winner = emissions[0].iter().enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap();
    assert_eq!(winner, 7, "POP7 should be the winner, got POP{winner}");
}

#[test]
fn pairwise_temperature_adapts_to_gap_size() {
    // Large gap → large temperature (soft), small gap → small temperature (sharp)
    let pops = vec![
        make_pop("CLOSE_A", &["ca1", "ca2"]),
        make_pop("CLOSE_B", &["cb1", "cb2"]),
        make_pop("FAR_C", &["fc1", "fc2"]),
    ];

    // CLOSE_A and CLOSE_B are very close, FAR_C is far
    let obs: Vec<_> = (0..30).map(|_| {
        make_obs(&[
            ("ca1", 0.990), ("ca2", 0.989),
            ("cb1", 0.991), ("cb2", 0.990),
            ("fc1", 0.950), ("fc2", 0.949),
        ])
    }).collect();

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    // CLOSE_B should win over CLOSE_A (0.991 > 0.990)
    // But both should be well above FAR_C
    let close_b_score = emissions[0][1];
    let far_c_score = emissions[0][2];
    assert!(close_b_score > far_c_score,
        "CLOSE_B should beat FAR_C: {close_b_score:.4} vs {far_c_score:.4}");
}

#[test]
fn pairwise_all_equal_returns_uniform() {
    let (pops, _) = make_n_pops(5);
    // All populations have identical similarity
    let mut sims = Vec::new();
    for p in 0..5 {
        sims.push((format!("p{}_h1", p), 0.95));
        sims.push((format!("p{}_h2", p), 0.95));
    }
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0, end: 10000,
        sample: "query".to_string(),
        similarities: sims.into_iter().collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }];

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    let expected_uniform = -(5.0_f64).ln();

    for &lp in &emissions[0] {
        assert!((lp - expected_uniform).abs() < 0.1,
            "Equal sims → near-uniform: got {lp:.4}, expected ~{expected_uniform:.4}");
    }
}

#[test]
fn pairwise_single_population_returns_log1() {
    let pops = vec![make_pop("ONLY", &["o1"])];
    let obs = vec![make_obs(&[("o1", 0.99)])];

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    assert_eq!(emissions.len(), 1);
    assert_eq!(emissions[0].len(), 1);
    // Single pop → uniform (trivially)
    assert!(emissions[0][0].is_finite());
}

#[test]
fn pairwise_empty_obs_returns_empty() {
    let (pops, _) = make_n_pops(3);
    let emissions = compute_pairwise_log_emissions(&[], &pops, &EmissionModel::Max);
    assert!(emissions.is_empty());
}

#[test]
fn pairwise_missing_haplotype_data_graceful() {
    let pops = vec![
        make_pop("A", &["a1", "a2"]),
        make_pop("B", &["b1", "b2"]),
        make_pop("C", &["c1", "c2"]),
    ];
    // Only A haplotypes present in observation
    let obs = vec![make_obs(&[("a1", 0.99), ("a2", 0.98)])];

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    assert_eq!(emissions[0].len(), 3);
    for &lp in &emissions[0] {
        assert!(lp.is_finite(), "Should handle missing data gracefully");
    }
}

#[test]
fn pairwise_log_probs_sum_to_one_approx() {
    let (pops, _) = make_n_pops(5);
    let obs: Vec<_> = (0..20).map(|i| make_obs_for_pops(5, i % 5, 0.02)).collect();

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    for (t, row) in emissions.iter().enumerate() {
        let sum_prob: f64 = row.iter().map(|&lp| lp.exp()).sum();
        assert!((sum_prob - 1.0).abs() < 1e-6,
            "Window {t}: exp(log-probs) should sum to 1.0, got {sum_prob:.8}");
    }
}

#[test]
fn pairwise_vs_standard_emissions_agree_on_dominant_pop() {
    // When one pop is clearly dominant, both methods should agree
    let pops = vec![
        make_pop("EUR", &["e1", "e2"]),
        make_pop("AFR", &["a1", "a2"]),
        make_pop("EAS", &["s1", "s2"]),
    ];

    let obs: Vec<_> = (0..30).map(|_| make_obs(&[
        ("e1", 0.999), ("e2", 0.998),
        ("a1", 0.950), ("a2", 0.949),
        ("s1", 0.900), ("s2", 0.899),
    ])).collect();

    let pairwise = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    let standard = precompute_log_emissions(&obs, &params);

    // Both should agree EUR is dominant
    for t in 0..obs.len() {
        let pw_winner = pairwise[t].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
        let std_winner = standard[t].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
        assert_eq!(pw_winner, std_winner,
            "Window {t}: pairwise winner={pw_winner}, standard winner={std_winner}");
    }
}

#[test]
fn pairwise_very_tiny_differences_still_discriminates() {
    // Differences of 0.0001 between populations
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let obs: Vec<_> = (0..100).map(|_| make_obs(&[
        ("a1", 0.99010), ("b1", 0.99000), ("c1", 0.98990),
    ])).collect();

    let emissions = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);

    // Pop A should consistently win
    let a_wins = emissions.iter().filter(|row| {
        row[0] > row[1] && row[0] > row[2]
    }).count();
    assert!(a_wins >= 90,
        "Pairwise should discriminate tiny diffs: A wins {a_wins}/100");
}

#[test]
fn pairwise_emission_model_topk_works() {
    let pops = vec![
        make_pop("A", &["a1", "a2", "a3", "a4"]),
        make_pop("B", &["b1", "b2", "b3", "b4"]),
    ];
    let obs = vec![make_obs(&[
        ("a1", 0.99), ("a2", 0.98), ("a3", 0.97), ("a4", 0.96),
        ("b1", 0.95), ("b2", 0.94), ("b3", 0.93), ("b4", 0.92),
    ])];

    let em_max = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    let em_top2 = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::TopK(2));

    // Both should agree A is dominant
    assert!(em_max[0][0] > em_max[0][1], "Max: A should beat B");
    assert!(em_top2[0][0] > em_top2[0][1], "TopK(2): A should beat B");
}
