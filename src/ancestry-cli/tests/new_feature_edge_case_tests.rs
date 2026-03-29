// Edge case tests for algo_dev A3 cycle 15-21 features:
// - estimate_identity_floor
// - compute_pairwise_log_emissions
// - compute_rank_log_emissions
// - compute_hierarchical_emissions
// - auto_detect_groups
// - ensemble_decode
// - estimate_emission_context
// - dampen_low_confidence_emissions
// - compute_distance_transitions



use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    auto_detect_groups, compute_distance_transitions, compute_hierarchical_emissions,
    compute_pairwise_log_emissions, compute_rank_log_emissions, dampen_low_confidence_emissions,
    ensemble_decode, estimate_emission_context, estimate_identity_floor,
};

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs_at(pos: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: pos,
        end: pos + 10000,
        sample: "test".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    make_obs_at(0, sims)
}

// =====================================================================
// estimate_identity_floor
// =====================================================================

#[test]
fn identity_floor_few_observations_returns_fallback() {
    let obs: Vec<_> = (0..19)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.95)]))
        .collect();
    assert_eq!(estimate_identity_floor(&obs), 0.995);
}

#[test]
fn identity_floor_empty_returns_fallback() {
    assert_eq!(estimate_identity_floor(&[]), 0.995);
}

#[test]
fn identity_floor_tight_distribution() {
    // All values very close together — should get a floor near data
    let obs: Vec<_> = (0..100)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.998 + (i as f64) * 0.00001)]))
        .collect();
    let floor = estimate_identity_floor(&obs);
    assert!(floor >= 0.9, "Floor should be at least 0.9: {}", floor);
    assert!(floor <= 0.999, "Floor should be reasonable: {}", floor);
}

#[test]
fn identity_floor_wide_distribution() {
    // Wide spread — floor should be more aggressive
    let mut obs: Vec<_> = (0..80)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.995 + (i as f64) * 0.00005)]))
        .collect();
    // Add some low-identity outliers
    for i in 80..100 {
        obs.push(make_obs_at(i * 10000, &[("a1", 0.90 + (i - 80) as f64 * 0.001)]));
    }
    let floor = estimate_identity_floor(&obs);
    assert!(floor >= 0.9, "Floor at least 0.9: {}", floor);
}

#[test]
fn identity_floor_all_identical() {
    let obs: Vec<_> = (0..50)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999)]))
        .collect();
    let floor = estimate_identity_floor(&obs);
    // Tight IQR → floor near Q1
    assert!(floor >= 0.9 && floor <= 0.999, "Floor: {}", floor);
}

#[test]
fn identity_floor_all_zero_similarity() {
    let obs: Vec<_> = (0..50)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.0)]))
        .collect();
    let floor = estimate_identity_floor(&obs);
    // Should clamp to 0.9 minimum
    assert_eq!(floor, 0.9, "Floor clamped to minimum: {}", floor);
}

#[test]
fn identity_floor_bimodal_distribution() {
    // Clear bimodal gap in bottom portion
    let mut obs: Vec<_> = (0..80)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.998 + (i as f64) * 0.00001)]))
        .collect();
    // 20 low outliers well separated from main cluster
    for i in 0..20 {
        obs.push(make_obs_at((80 + i) * 10000, &[("a1", 0.91 + (i as f64) * 0.0001)]));
    }
    let floor = estimate_identity_floor(&obs);
    assert!(floor >= 0.9, "Floor: {}", floor);
}

// =====================================================================
// compute_pairwise_log_emissions
// =====================================================================

#[test]
fn pairwise_emissions_empty_observations() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let result = compute_pairwise_log_emissions(&[], &pops, &EmissionModel::Max);
    assert!(result.is_empty());
}

#[test]
fn pairwise_emissions_single_population() {
    let pops = vec![make_pop("A", &["a1"])];
    let obs = vec![make_obs(&[("a1", 0.99)])];
    let result = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
}

#[test]
fn pairwise_emissions_two_populations_clear_signal() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs: Vec<_> = (0..20)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999), ("b1", 0.990)]))
        .collect();
    let result = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result.len(), 20);
    for row in &result {
        assert_eq!(row.len(), 2);
        // Population A should have higher emission (higher similarity)
        assert!(row[0] > row[1], "A should beat B: {:?}", row);
    }
}

#[test]
fn pairwise_emissions_three_populations() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let obs: Vec<_> = (0..20)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999), ("b1", 0.990), ("c1", 0.985)]))
        .collect();
    let result = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result.len(), 20);
    for row in &result {
        assert_eq!(row.len(), 3);
        // All values should be finite
        for v in row {
            assert!(v.is_finite(), "Non-finite emission: {}", v);
        }
    }
}

#[test]
fn pairwise_emissions_identical_similarities() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs: Vec<_> = (0..20)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.995), ("b1", 0.995)]))
        .collect();
    let result = compute_pairwise_log_emissions(&obs, &pops, &EmissionModel::Max);
    // Equal similarities → emissions should be roughly equal (log-softmax uniform)
    for row in &result {
        let diff = (row[0] - row[1]).abs();
        assert!(diff < 1.0, "Too much difference for equal sims: {:?}", row);
    }
}

// =====================================================================
// compute_rank_log_emissions
// =====================================================================

#[test]
fn rank_emissions_empty() {
    let pops = vec![make_pop("A", &["a1"])];
    let result = compute_rank_log_emissions(&[], &pops, 5);
    assert!(result.is_empty());
}

#[test]
fn rank_emissions_single_pop() {
    let pops = vec![make_pop("A", &["a1", "a2"])];
    let obs = vec![make_obs(&[("a1", 0.99), ("a2", 0.98)])];
    let result = compute_rank_log_emissions(&obs, &pops, 2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
    assert!(result[0][0].is_finite());
}

#[test]
fn rank_emissions_clear_winner() {
    let pops = vec![
        make_pop("A", &["a1", "a2", "a3"]),
        make_pop("B", &["b1", "b2", "b3"]),
    ];
    // Pop A haplotypes are always top-3
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[
            ("a1", 0.999), ("a2", 0.998), ("a3", 0.997),
            ("b1", 0.990), ("b2", 0.989), ("b3", 0.988),
        ]))
        .collect();
    let result = compute_rank_log_emissions(&obs, &pops, 3);
    for row in &result {
        assert!(row[0] > row[1], "A should dominate rank: {:?}", row);
    }
}

#[test]
fn rank_emissions_topk_zero_auto() {
    // top_k = 0 means auto-compute
    let pops = vec![
        make_pop("A", &["a1", "a2"]),
        make_pop("B", &["b1", "b2"]),
    ];
    let obs = vec![make_obs(&[("a1", 0.99), ("a2", 0.98), ("b1", 0.97), ("b2", 0.96)])];
    let result = compute_rank_log_emissions(&obs, &pops, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    for v in &result[0] {
        assert!(v.is_finite());
    }
}

#[test]
fn rank_emissions_topk_exceeds_total() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs = vec![make_obs(&[("a1", 0.99), ("b1", 0.98)])];
    let result = compute_rank_log_emissions(&obs, &pops, 100);
    assert_eq!(result.len(), 1);
    for v in &result[0] {
        assert!(v.is_finite());
    }
}

#[test]
fn rank_emissions_missing_haplotype_data() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    // Only pop A has data
    let obs = vec![make_obs(&[("a1", 0.99)])];
    let result = compute_rank_log_emissions(&obs, &pops, 1);
    assert_eq!(result.len(), 1);
    for v in &result[0] {
        assert!(v.is_finite());
    }
}

// =====================================================================
// compute_hierarchical_emissions
// =====================================================================

#[test]
fn hierarchical_emissions_empty_groups() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs = vec![make_obs(&[("a1", 0.99), ("b1", 0.98)])];
    let result = compute_hierarchical_emissions(&obs, &pops, &[], &EmissionModel::Max, 0.01, 0.5);
    // Empty groups → fallback (uniform)
    assert_eq!(result.len(), 1);
}

#[test]
fn hierarchical_emissions_single_group_all_pops() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let groups = vec![vec![0, 1]]; // one group with both
    let obs = vec![make_obs(&[("a1", 0.999), ("b1", 0.990)])];
    let result = compute_hierarchical_emissions(&obs, &pops, &groups, &EmissionModel::Max, 0.01, 0.5);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
}

#[test]
fn hierarchical_emissions_weight_zero_pure_within() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let groups = vec![vec![0], vec![1]];
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999), ("b1", 0.990)]))
        .collect();
    let result = compute_hierarchical_emissions(&obs, &pops, &groups, &EmissionModel::Max, 0.01, 0.0);
    for row in &result {
        for v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn hierarchical_emissions_weight_one_pure_group() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let groups = vec![vec![0], vec![1]];
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999), ("b1", 0.990)]))
        .collect();
    let result = compute_hierarchical_emissions(&obs, &pops, &groups, &EmissionModel::Max, 0.01, 1.0);
    for row in &result {
        for v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn hierarchical_emissions_three_pops_two_groups() {
    let pops = vec![
        make_pop("EUR", &["e1"]),
        make_pop("AMR", &["m1"]),
        make_pop("AFR", &["f1"]),
    ];
    let groups = vec![vec![0, 1], vec![2]]; // EUR+AMR vs AFR
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("e1", 0.998), ("m1", 0.997), ("f1", 0.990)]))
        .collect();
    let result = compute_hierarchical_emissions(&obs, &pops, &groups, &EmissionModel::Max, 0.01, 0.5);
    assert_eq!(result.len(), 10);
    for row in &result {
        assert_eq!(row.len(), 3);
        for v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn hierarchical_emissions_empty_obs() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let groups = vec![vec![0], vec![1]];
    let result = compute_hierarchical_emissions(&[], &pops, &groups, &EmissionModel::Max, 0.01, 0.5);
    assert!(result.is_empty());
}

// =====================================================================
// auto_detect_groups
// =====================================================================

#[test]
fn auto_groups_single_pop() {
    let pops = vec![make_pop("A", &["a1"])];
    let obs = vec![make_obs(&[("a1", 0.99)])];
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec![0]);
}

#[test]
fn auto_groups_two_pops() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs = vec![make_obs(&[("a1", 0.99), ("b1", 0.98)])];
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    // 2 or fewer → one group per pop
    assert_eq!(groups.len(), 2);
}

#[test]
fn auto_groups_three_pops_two_similar() {
    let pops = vec![
        make_pop("EUR", &["e1"]),
        make_pop("AMR", &["m1"]),
        make_pop("AFR", &["f1"]),
    ];
    // EUR and AMR very close, AFR distant
    let obs: Vec<_> = (0..50)
        .map(|i| make_obs_at(i * 10000, &[("e1", 0.998), ("m1", 0.9978), ("f1", 0.990)]))
        .collect();
    let groups = auto_detect_groups(&obs, &pops, &EmissionModel::Max);
    // Should find some grouping — all populations should be present
    let all_pops: Vec<usize> = groups.iter().flatten().cloned().collect();
    assert!(all_pops.contains(&0));
    assert!(all_pops.contains(&1));
    assert!(all_pops.contains(&2));
}

#[test]
fn auto_groups_empty_obs() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let groups = auto_detect_groups(&[], &pops, &EmissionModel::Max);
    // Empty obs → one group per pop fallback
    let all_pops: Vec<usize> = groups.iter().flatten().cloned().collect();
    assert_eq!(all_pops.len(), 3);
}

// =====================================================================
// ensemble_decode
// =====================================================================

#[test]
fn ensemble_decode_empty_emissions() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let (posteriors, states) = ensemble_decode(&[], &params, 5, 2.0);
    assert!(posteriors.is_empty());
    assert!(states.is_empty());
}

#[test]
fn ensemble_decode_zero_ensemble() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let emissions = vec![vec![-0.5, -1.0]; 10];
    let (posteriors, states) = ensemble_decode(&emissions, &params, 0, 2.0);
    assert!(posteriors.is_empty());
    assert!(states.is_empty());
}

#[test]
fn ensemble_decode_single_member() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let emissions = vec![vec![-0.5, -1.5]; 10];
    let (posteriors, states) = ensemble_decode(&emissions, &params, 1, 2.0);
    assert_eq!(posteriors.len(), 10);
    assert_eq!(states.len(), 10);
    for row in &posteriors {
        assert_eq!(row.len(), 2);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors should sum to 1: {}", sum);
    }
}

#[test]
fn ensemble_decode_multiple_members() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let emissions = vec![vec![-0.3, -1.5]; 20];
    let (posteriors, states) = ensemble_decode(&emissions, &params, 5, 2.0);
    assert_eq!(posteriors.len(), 20);
    assert_eq!(states.len(), 20);
    for row in &posteriors {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum: {}", sum);
    }
    // Clear signal → all states should be 0 (population A)
    for s in &states {
        assert_eq!(*s, 0);
    }
}

#[test]
fn ensemble_decode_scale_factor_one() {
    // scale_factor = 1.0 → no perturbation, all members identical
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let emissions = vec![vec![-0.5, -1.0]; 10];
    let (posteriors, _states) = ensemble_decode(&emissions, &params, 3, 1.0);
    assert_eq!(posteriors.len(), 10);
    for row in &posteriors {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// =====================================================================
// estimate_emission_context
// =====================================================================

#[test]
fn emission_context_empty_obs() {
    let pops = vec![make_pop("A", &["a1"])];
    let result = estimate_emission_context(&[], &pops, 5, 1, 15);
    assert_eq!(result, 5); // returns base_context
}

#[test]
fn emission_context_empty_pops() {
    let obs = vec![make_obs(&[("a1", 0.99)])];
    let result = estimate_emission_context(&obs, &[], 5, 1, 15);
    assert_eq!(result, 5);
}

#[test]
fn emission_context_base_zero() {
    let pops = vec![make_pop("A", &["a1"])];
    let obs = vec![make_obs(&[("a1", 0.99)])];
    let result = estimate_emission_context(&obs, &pops, 0, 1, 15);
    assert_eq!(result, 0);
}

#[test]
fn emission_context_min_equals_max() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs: Vec<_> = (0..30)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.98)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 5, 7, 7);
    // min == max → clamped to 7
    assert_eq!(result, 7);
}

#[test]
fn emission_context_result_within_bounds() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let obs: Vec<_> = (0..100)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.999), ("b1", 0.990)]))
        .collect();
    let result = estimate_emission_context(&obs, &pops, 5, 1, 15);
    assert!(result >= 1 && result <= 15, "Context: {}", result);
}

// =====================================================================
// dampen_low_confidence_emissions
// =====================================================================

#[test]
fn dampen_empty_emissions() {
    let result = dampen_low_confidence_emissions(&[], 1.5);
    assert!(result.is_empty());
}

#[test]
fn dampen_single_state() {
    let emissions = vec![vec![-0.5]; 5];
    let result = dampen_low_confidence_emissions(&emissions, 1.5);
    assert_eq!(result.len(), 5);
    // Single state → no change
    assert_eq!(result, emissions);
}

#[test]
fn dampen_uniform_emissions_unchanged() {
    // All windows have same emissions → discriminability = 0 for all → max dampening
    let emissions = vec![vec![-0.693, -0.693]; 10]; // log(0.5)
    let result = dampen_low_confidence_emissions(&emissions, 1.5);
    assert_eq!(result.len(), 10);
    for row in &result {
        assert_eq!(row.len(), 2);
        for v in row {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn dampen_preserves_length_and_width() {
    let emissions = vec![vec![-0.3, -1.5, -2.0]; 20];
    let result = dampen_low_confidence_emissions(&emissions, 1.5);
    assert_eq!(result.len(), 20);
    for row in &result {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn dampen_high_confidence_less_affected() {
    // Create emissions with varying discriminability
    let mut emissions = Vec::new();
    // High discriminability windows
    for _ in 0..10 {
        emissions.push(vec![-0.1, -5.0]); // very peaked
    }
    // Low discriminability windows
    for _ in 0..10 {
        emissions.push(vec![-0.69, -0.70]); // nearly uniform
    }
    let result = dampen_low_confidence_emissions(&emissions, 1.5);
    assert_eq!(result.len(), 20);
    // High-disc windows should have larger gap than low-disc after dampening
    let gap_high = (result[0][0] - result[0][1]).abs();
    let gap_low = (result[10][0] - result[10][1]).abs();
    assert!(gap_high >= gap_low, "High-disc gap {} should be >= low-disc gap {}", gap_high, gap_low);
}

#[test]
fn dampen_all_finite_values() {
    let emissions = vec![vec![-0.1, -3.0, -5.0]; 15];
    let result = dampen_low_confidence_emissions(&emissions, 0.5);
    for (i, row) in result.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            assert!(v.is_finite(), "Non-finite at [{},{}]: {}", i, j, v);
        }
    }
}

// =====================================================================
// compute_distance_transitions
// =====================================================================

#[test]
fn distance_transitions_empty_obs() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let result = compute_distance_transitions(&[], &params, 10000);
    assert!(result.is_empty());
}

#[test]
fn distance_transitions_single_obs() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![make_obs_at(0, &[("a1", 0.99), ("b1", 0.98)])];
    let result = compute_distance_transitions(&obs, &params, 10000);
    assert!(result.is_empty());
}

#[test]
fn distance_transitions_zero_window_size() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.98)]))
        .collect();
    let result = compute_distance_transitions(&obs, &params, 0);
    assert!(result.is_empty());
}

#[test]
fn distance_transitions_uniform_spacing() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[("a1", 0.99), ("b1", 0.98)]))
        .collect();
    let result = compute_distance_transitions(&obs, &params, 10000);
    // n-1 transition matrices
    assert_eq!(result.len(), 9);
    for tm in &result {
        assert_eq!(tm.len(), 2);
        for row in tm {
            assert_eq!(row.len(), 2);
            // Values are in log-space; exp-sum should be ~1.0
            let exp_sum: f64 = row.iter().map(|v| v.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-6, "Row exp-sum should be ~1: {}", exp_sum);
        }
    }
}

#[test]
fn distance_transitions_large_gap() {
    let pops = vec![make_pop("A", &["a1"]), make_pop("B", &["b1"])];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![
        make_obs_at(0, &[("a1", 0.99), ("b1", 0.98)]),
        make_obs_at(1_000_000, &[("a1", 0.99), ("b1", 0.98)]), // 1 Mb gap
    ];
    let result = compute_distance_transitions(&obs, &params, 10000);
    assert_eq!(result.len(), 1);
    // Large gap → higher switch probability (values are log-space)
    let log_switch = result[0][0][1];
    let base_log_switch = params.transitions[0][1].ln();
    assert!(log_switch > base_log_switch,
            "Large gap should increase switch prob: exp({}) > exp({})",
            log_switch, base_log_switch);
}

#[test]
fn distance_transitions_row_sums() {
    let pops = vec![
        make_pop("A", &["a1"]),
        make_pop("B", &["b1"]),
        make_pop("C", &["c1"]),
    ];
    let params = AncestryHmmParams::new(pops, 0.01);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 15000, &[("a1", 0.99), ("b1", 0.98), ("c1", 0.97)]))
        .collect();
    let result = compute_distance_transitions(&obs, &params, 10000);
    for (t, tm) in result.iter().enumerate() {
        for (i, row) in tm.iter().enumerate() {
            // Values are in log-space; exp-sum should be ~1.0
            let exp_sum: f64 = row.iter().map(|v| v.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-6, "Row [{},{}] exp-sums to {}", t, i, exp_sum);
            for v in row {
                assert!(*v <= 0.0, "Log-prob should be <= 0: {}", v);
            }
        }
    }
}
