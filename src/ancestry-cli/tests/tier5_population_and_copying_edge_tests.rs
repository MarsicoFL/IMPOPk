// Tier 5 edge case tests: population variance/distance/purity, within-pop variance,
// LOO robust emissions, adaptive transitions, forward-backward/Viterbi with transitions,
// population-aware transitions, purity-weighted observations, distance-weighted transitions,
// and infer_ancestry_copying_em.
//
// Covers the final 12 previously-untested public functions in ancestry-cli/src/hmm.rs:
//   - compute_population_variances
//   - compute_population_aware_transitions
//   - forward_backward_from_log_emissions_with_transitions
//   - viterbi_from_log_emissions_with_transitions
//   - compute_population_distances
//   - set_distance_weighted_transitions
//   - compute_adaptive_transitions
//   - compute_reference_purity
//   - apply_purity_weighted_observations
//   - compute_within_pop_variance
//   - compute_loo_robust_emissions
//   - infer_ancestry_copying_em

use std::collections::HashMap;
use hprc_ancestry_cli::hmm::{
    apply_purity_weighted_observations,
    compute_adaptive_transitions,
    compute_loo_robust_emissions,
    compute_population_aware_transitions,
    compute_population_distances,
    compute_population_variances,
    compute_reference_purity,
    compute_within_pop_variance,
    forward_backward_from_log_emissions_with_transitions,
    infer_ancestry_copying_em,
    set_distance_weighted_transitions,
    viterbi_from_log_emissions_with_transitions,
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
};

// ─── Helpers ────────────────────────────────────────────────────────────

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|h| h.to_string()).collect(),
    }
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    let mut map = HashMap::new();
    for &(h, v) in sims {
        map.insert(h.to_string(), v);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 1000,
        sample: "query".to_string(),
        similarities: map,
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_params(n_pops: usize, switch_prob: f64) -> AncestryHmmParams {
    let pops: Vec<AncestralPopulation> = (0..n_pops)
        .map(|i| AncestralPopulation {
            name: format!("pop_{}", i),
            haplotypes: vec![format!("hap_{}_1", i), format!("hap_{}_2", i)],
        })
        .collect();
    AncestryHmmParams::new(pops, switch_prob)
}

// ===========================================================================
// compute_population_variances
// ===========================================================================

#[test]
fn cpv_empty_observations_returns_zeros() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let result = compute_population_variances(&[], &pops, &EmissionModel::Max);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn cpv_empty_populations_returns_empty() {
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_population_variances(&obs, &[], &EmissionModel::Max);
    assert!(result.is_empty());
}

#[test]
fn cpv_single_observation_returns_zeros() {
    // With only 1 window, counts[p] < 2, so variance = 0.0
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_population_variances(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result, vec![0.0]);
}

#[test]
fn cpv_constant_values_zero_variance() {
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![
        make_obs(&[("h1", 0.8)]),
        make_obs(&[("h1", 0.8)]),
        make_obs(&[("h1", 0.8)]),
    ];
    let result = compute_population_variances(&obs, &pops, &EmissionModel::Max);
    assert!(result[0].abs() < 1e-10);
}

#[test]
fn cpv_varying_values_positive_variance() {
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![
        make_obs(&[("h1", 0.5)]),
        make_obs(&[("h1", 0.9)]),
        make_obs(&[("h1", 0.7)]),
    ];
    let result = compute_population_variances(&obs, &pops, &EmissionModel::Max);
    assert!(result[0] > 0.0, "variance should be positive for varying data");
}

#[test]
fn cpv_zero_similarities_excluded() {
    // Similarities <= 0.0 are filtered out; only 1 valid → insufficient → 0.0
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![
        make_obs(&[("h1", 0.0)]),
        make_obs(&[("h1", 0.8)]),
    ];
    let result = compute_population_variances(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result[0], 0.0);
}

#[test]
fn cpv_multiple_populations_independent() {
    let pops = vec![
        make_pop("A", &["h1"]),
        make_pop("B", &["h2"]),
    ];
    // A: constant 0.8 → variance 0
    // B: varies 0.5/0.9 → positive variance
    let obs = vec![
        make_obs(&[("h1", 0.8), ("h2", 0.5)]),
        make_obs(&[("h1", 0.8), ("h2", 0.9)]),
    ];
    let result = compute_population_variances(&obs, &pops, &EmissionModel::Max);
    assert!(result[0].abs() < 1e-10, "A should have ~0 variance");
    assert!(result[1] > 0.0, "B should have positive variance");
}

// ===========================================================================
// compute_population_distances
// ===========================================================================

#[test]
fn cpd_empty_observations_returns_zeros() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let result = compute_population_distances(&[], &pops, &EmissionModel::Max);
    assert_eq!(result, vec![vec![0.0; 2]; 2]);
}

#[test]
fn cpd_single_population_returns_zero_matrix() {
    // Need >=2 valid pops per window; with 1 pop, all skipped
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_population_distances(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result, vec![vec![0.0]]);
}

#[test]
fn cpd_identical_populations_zero_distance() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![
        make_obs(&[("h1", 0.8), ("h2", 0.8)]),
        make_obs(&[("h1", 0.7), ("h2", 0.7)]),
    ];
    let result = compute_population_distances(&obs, &pops, &EmissionModel::Max);
    assert!(result[0][1].abs() < 1e-10, "identical pops → 0 distance");
    assert!(result[1][0].abs() < 1e-10, "symmetric");
}

#[test]
fn cpd_different_populations_positive_distance() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.3)]),
    ];
    let result = compute_population_distances(&obs, &pops, &EmissionModel::Max);
    let expected_d = (0.9 - 0.3_f64).abs();
    assert!((result[0][1] - expected_d).abs() < 1e-10);
}

#[test]
fn cpd_symmetric_matrix() {
    let pops = vec![
        make_pop("A", &["h1"]),
        make_pop("B", &["h2"]),
        make_pop("C", &["h3"]),
    ];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.5), ("h3", 0.7)]),
        make_obs(&[("h1", 0.8), ("h2", 0.6), ("h3", 0.4)]),
    ];
    let result = compute_population_distances(&obs, &pops, &EmissionModel::Max);
    for i in 0..3 {
        for j in 0..3 {
            assert!((result[i][j] - result[j][i]).abs() < 1e-10,
                "D[{i}][{j}] != D[{j}][{i}]");
        }
        assert!(result[i][i].abs() < 1e-10, "diagonal should be zero");
    }
}

#[test]
fn cpd_zero_similarity_windows_skipped() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    // Both sims are 0 or negative → skipped
    let obs = vec![make_obs(&[("h1", 0.0), ("h2", -0.1)])];
    let result = compute_population_distances(&obs, &pops, &EmissionModel::Max);
    assert_eq!(result[0][1], 0.0);
}

// ===========================================================================
// compute_reference_purity
// ===========================================================================

#[test]
fn crp_empty_observations_returns_empty() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let result = compute_reference_purity(&[], &pops);
    assert!(result.is_empty());
}

#[test]
fn crp_single_population_all_purity_one() {
    // No out-of-population references → mean_other = 0 → purity = 1.0
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_reference_purity(&obs, &pops);
    assert_eq!(result.get("h1"), Some(&1.0));
}

#[test]
fn crp_strong_population_signal_high_purity() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    // h1 has much higher sim than h2 → high purity for h1
    let obs = vec![
        make_obs(&[("h1", 0.95), ("h2", 0.3)]),
        make_obs(&[("h1", 0.90), ("h2", 0.2)]),
    ];
    let result = compute_reference_purity(&obs, &pops);
    let p1 = result["h1"];
    assert!(p1 > 1.0, "h1 should have purity > 1.0 (strong indicator), got {p1}");
}

#[test]
fn crp_uninformative_haplotype_purity_near_one() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    // h1 and h2 have similar similarities → purity ≈ 1
    let obs = vec![
        make_obs(&[("h1", 0.8), ("h2", 0.79)]),
        make_obs(&[("h1", 0.7), ("h2", 0.71)]),
    ];
    let result = compute_reference_purity(&obs, &pops);
    let p1 = result["h1"];
    assert!((p1 - 1.0).abs() < 0.2, "near-equal sims → purity ≈ 1, got {p1}");
}

#[test]
fn crp_zero_similarity_skipped() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![make_obs(&[("h1", 0.0), ("h2", 0.5)])];
    let result = compute_reference_purity(&obs, &pops);
    assert!(!result.contains_key("h1"), "h1 with sim=0 should be skipped");
}

#[test]
fn crp_multiple_haplotypes_per_pop() {
    let pops = vec![
        make_pop("A", &["h1a", "h1b"]),
        make_pop("B", &["h2a"]),
    ];
    let obs = vec![
        make_obs(&[("h1a", 0.9), ("h1b", 0.8), ("h2a", 0.3)]),
    ];
    let result = compute_reference_purity(&obs, &pops);
    // h1a: mean_sim=0.9, max_other=0.3 → purity=0.9/0.3=3.0
    assert!((result["h1a"] - 3.0).abs() < 1e-10);
}

// ===========================================================================
// apply_purity_weighted_observations
// ===========================================================================

#[test]
fn apwo_empty_observations_returns_empty() {
    let purity: HashMap<String, f64> = HashMap::new();
    let result = apply_purity_weighted_observations(&[], &purity, 1.0);
    assert!(result.is_empty());
}

#[test]
fn apwo_gamma_zero_no_weighting() {
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let mut purity = HashMap::new();
    purity.insert("h1".to_string(), 2.0);
    let result = apply_purity_weighted_observations(&obs, &purity, 0.0);
    // gamma=0 → weight=purity^0=1.0 → unchanged
    assert!((result[0].similarities["h1"] - 0.9).abs() < 1e-10);
}

#[test]
fn apwo_gamma_one_full_weighting() {
    let obs = vec![make_obs(&[("h1", 0.8)])];
    let mut purity = HashMap::new();
    purity.insert("h1".to_string(), 2.0);
    let result = apply_purity_weighted_observations(&obs, &purity, 1.0);
    // weight=2.0^1=2.0 → sim=0.8*2.0=1.6
    assert!((result[0].similarities["h1"] - 1.6).abs() < 1e-10);
}

#[test]
fn apwo_missing_haplotype_defaults_to_one() {
    let obs = vec![make_obs(&[("h1", 0.8)])];
    let purity: HashMap<String, f64> = HashMap::new(); // empty: h1 not in scores
    let result = apply_purity_weighted_observations(&obs, &purity, 1.0);
    // purity default=1.0, weight=1.0^1=1.0 → unchanged
    assert!((result[0].similarities["h1"] - 0.8).abs() < 1e-10);
}

#[test]
fn apwo_fractional_gamma() {
    let obs = vec![make_obs(&[("h1", 1.0)])];
    let mut purity = HashMap::new();
    purity.insert("h1".to_string(), 4.0);
    let result = apply_purity_weighted_observations(&obs, &purity, 0.5);
    // weight=4.0^0.5=2.0 → sim=1.0*2.0=2.0
    assert!((result[0].similarities["h1"] - 2.0).abs() < 1e-10);
}

#[test]
fn apwo_preserves_window_metadata() {
    let obs = vec![make_obs(&[("h1", 0.5)])];
    let purity: HashMap<String, f64> = HashMap::new();
    let result = apply_purity_weighted_observations(&obs, &purity, 1.0);
    assert_eq!(result[0].chrom, "chr1");
    assert_eq!(result[0].start, 0);
    assert_eq!(result[0].end, 1000);
    assert_eq!(result[0].sample, "query");
}

// ===========================================================================
// compute_within_pop_variance
// ===========================================================================

#[test]
fn cwpv_empty_observations_returns_empty() {
    let pops = vec![make_pop("A", &["h1", "h2"])];
    let result = compute_within_pop_variance(&[], &pops);
    assert!(result.is_empty());
}

#[test]
fn cwpv_single_haplotype_returns_zero() {
    // < 2 haplotypes → variance = 0.0
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_within_pop_variance(&obs, &pops);
    assert_eq!(result[0][0], 0.0);
}

#[test]
fn cwpv_identical_haplotypes_zero_variance() {
    let pops = vec![make_pop("A", &["h1", "h2"])];
    let obs = vec![make_obs(&[("h1", 0.8), ("h2", 0.8)])];
    let result = compute_within_pop_variance(&obs, &pops);
    assert!(result[0][0].abs() < 1e-10);
}

#[test]
fn cwpv_different_haplotypes_positive_variance() {
    let pops = vec![make_pop("A", &["h1", "h2"])];
    let obs = vec![make_obs(&[("h1", 0.3), ("h2", 0.9)])];
    let result = compute_within_pop_variance(&obs, &pops);
    // Sample variance: mean=0.6, var=((0.3-0.6)^2+(0.9-0.6)^2)/(2-1)=0.18
    assert!((result[0][0] - 0.18).abs() < 1e-10);
}

#[test]
fn cwpv_multiple_windows_independent() {
    let pops = vec![make_pop("A", &["h1", "h2"])];
    let obs = vec![
        make_obs(&[("h1", 0.8), ("h2", 0.8)]), // zero variance
        make_obs(&[("h1", 0.3), ("h2", 0.9)]), // positive variance
    ];
    let result = compute_within_pop_variance(&obs, &pops);
    assert_eq!(result.len(), 2);
    assert!(result[0][0].abs() < 1e-10);
    assert!(result[1][0] > 0.0);
}

#[test]
fn cwpv_multiple_populations() {
    let pops = vec![
        make_pop("A", &["h1", "h2"]),
        make_pop("B", &["h3", "h4"]),
    ];
    let obs = vec![make_obs(&[("h1", 0.5), ("h2", 0.5), ("h3", 0.2), ("h4", 0.8)])];
    let result = compute_within_pop_variance(&obs, &pops);
    assert!(result[0][0].abs() < 1e-10, "A: equal sims → 0 variance");
    assert!(result[0][1] > 0.0, "B: different sims → positive variance");
}

// ===========================================================================
// compute_loo_robust_emissions
// ===========================================================================

#[test]
fn clre_empty_observations_returns_empty() {
    let pops = vec![make_pop("A", &["h1"])];
    let result = compute_loo_robust_emissions(&[], &pops, &EmissionModel::Max, 1.0);
    assert!(result.is_empty());
}

#[test]
fn clre_zero_populations_returns_empty_rows() {
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_loo_robust_emissions(&obs, &[], &EmissionModel::Max, 1.0);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_empty());
}

#[test]
fn clre_single_haplotype_falls_back() {
    // 1 haplotype: can't do LOO, returns that haplotype's similarity
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![make_obs(&[("h1", 0.9), ("h2", 0.3)])];
    let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, 1.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    // Log-probabilities should sum to ~0 in exp-space (normalized)
    let sum: f64 = result[0].iter().map(|&v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-6, "posteriors should sum to 1, got {sum}");
}

#[test]
fn clre_negative_temperature_uses_fallback() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![make_obs(&[("h1", 0.9), ("h2", 0.3)])];
    // Negative temp → clamped to 0.01 (should not panic)
    let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, -1.0);
    assert_eq!(result.len(), 1);
    assert!(!result[0].iter().any(|v| v.is_nan()));
}

#[test]
fn clre_two_haplotype_loo_pessimistic() {
    // With 2 haps in pop, LOO removes each and takes min aggregation
    let pops = vec![
        make_pop("A", &["h1", "h2"]),
        make_pop("B", &["h3"]),
    ];
    let obs = vec![make_obs(&[("h1", 0.95), ("h2", 0.3), ("h3", 0.5)])];
    let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, 1.0);
    // Pop A LOO: remove h1 → 0.3, remove h2 → 0.95 → min=0.3
    // Pop B (single hap): 0.5
    // With softmax, pop B should dominate since 0.5 > 0.3
    let prob_a: f64 = result[0][0].exp();
    let prob_b: f64 = result[0][1].exp();
    assert!(prob_b > prob_a, "LOO pessimistic: pop B (0.5) > pop A (0.3)");
}

#[test]
fn clre_output_valid_log_probabilities() {
    let pops = vec![make_pop("A", &["h1", "h2"]), make_pop("B", &["h3", "h4"])];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.7), ("h3", 0.6), ("h4", 0.5)]),
        make_obs(&[("h1", 0.3), ("h2", 0.4), ("h3", 0.8), ("h4", 0.7)]),
    ];
    let result = compute_loo_robust_emissions(&obs, &pops, &EmissionModel::Max, 1.0);
    for (t, row) in result.iter().enumerate() {
        assert!(row.iter().all(|v| v.is_finite()), "window {t}: non-finite values");
        assert!(row.iter().all(|&v| v <= 0.0), "window {t}: log-probs should be <= 0");
        let sum: f64 = row.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "window {t}: exp-sum = {sum}, want 1.0");
    }
}

// ===========================================================================
// compute_population_aware_transitions
// ===========================================================================

#[test]
fn cpat_single_population_returns_stay() {
    let pops = vec![make_pop("A", &["h1"])];
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let result = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
    // k < 2 → returns stay probability
    let stay = (1.0 - 0.01_f64).ln();
    assert!((result[0][0] - stay).abs() < 1e-10);
}

#[test]
fn cpat_empty_observations_fallback_uniform() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let result = compute_population_aware_transitions(&[], &pops, &EmissionModel::Max, 0.01);
    // No observations → all pair_diffs empty → median=0.01 fallback
    assert_eq!(result.len(), 2);
    // Diagonal: stay prob, off-diagonal: switch prob
    let stay = (1.0 - 0.01_f64).ln();
    assert!((result[0][0] - stay).abs() < 1e-10);
}

#[test]
fn cpat_close_populations_easier_transitions() {
    let pops = vec![
        make_pop("A", &["h1"]),
        make_pop("B", &["h2"]),
        make_pop("C", &["h3"]),
    ];
    // A and B close (similar values), C distant
    let obs = vec![
        make_obs(&[("h1", 0.90), ("h2", 0.89), ("h3", 0.50)]),
        make_obs(&[("h1", 0.85), ("h2", 0.84), ("h3", 0.45)]),
    ];
    let result = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.1);
    // P(A→B) should be > P(A→C) since A and B are closer
    let p_ab = result[0][1]; // log
    let p_ac = result[0][2]; // log
    assert!(p_ab > p_ac, "close pops should have higher transition prob: A→B={p_ab} > A→C={p_ac}");
}

#[test]
fn cpat_output_is_log_space() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![make_obs(&[("h1", 0.9), ("h2", 0.5)])];
    let result = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.01);
    // All values should be negative (log of probability < 1)
    for row in &result {
        for &v in row {
            assert!(v < 0.0, "log-transition should be negative, got {v}");
        }
    }
}

#[test]
fn cpat_rows_approximately_sum_to_one() {
    let pops = vec![
        make_pop("A", &["h1"]),
        make_pop("B", &["h2"]),
        make_pop("C", &["h3"]),
    ];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.5), ("h3", 0.7)]),
    ];
    let result = compute_population_aware_transitions(&obs, &pops, &EmissionModel::Max, 0.05);
    for (i, row) in result.iter().enumerate() {
        let sum: f64 = row.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 0.01, "row {i} exp-sum = {sum}, want ~1.0");
    }
}

// ===========================================================================
// forward_backward_from_log_emissions_with_transitions
// ===========================================================================

#[test]
fn fbwt_empty_emissions_returns_empty() {
    let params = make_params(2, 0.01);
    let result = forward_backward_from_log_emissions_with_transitions(
        &[], &params, &[],
    );
    assert!(result.is_empty());
}

#[test]
fn fbwt_single_window_returns_valid_posteriors() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    // Use finite uniform log-emissions: log(1/k) for each state
    let log_uniform = -(k as f64).ln();
    let log_emissions = vec![vec![log_uniform; k]];
    let result = forward_backward_from_log_emissions_with_transitions(
        &log_emissions, &params, &[],
    );
    assert_eq!(result.len(), 1);
    // With 1 window, no transitions needed. Posteriors should be well-defined.
    let sum: f64 = result[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "posteriors should sum to 1, got {sum}");
}

#[test]
fn fbwt_posteriors_sum_to_one() {
    let params = make_params(3, 0.01);
    let k = params.n_states;
    // Create 5 windows with varying log-emissions
    let log_emissions: Vec<Vec<f64>> = (0..5).map(|t| {
        (0..k).map(|s| {
            if s == t % k { -0.1 } else { -2.0 }
        }).collect()
    }).collect();
    // Build per-window transitions (need n-1 matrices)
    let log_trans: Vec<Vec<Vec<f64>>> = (0..4).map(|_| {
        params.transitions.iter().map(|row| {
            row.iter().map(|&p| p.max(1e-20).ln()).collect()
        }).collect()
    }).collect();

    let result = forward_backward_from_log_emissions_with_transitions(
        &log_emissions, &params, &log_trans,
    );
    assert_eq!(result.len(), 5);
    for (t, row) in result.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "window {t}: sum={sum}, want 1.0");
        assert!(row.iter().all(|&v| v >= 0.0), "window {t}: negative posterior");
    }
}

#[test]
fn fbwt_insufficient_transitions_falls_back() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    let log_emissions: Vec<Vec<f64>> = (0..3).map(|_| vec![-1.0; k]).collect();
    // Only provide 1 transition matrix instead of 2 → should fall back to standard fb
    let log_trans: Vec<Vec<Vec<f64>>> = vec![
        params.transitions.iter().map(|row| {
            row.iter().map(|&p| p.max(1e-20).ln()).collect()
        }).collect(),
    ];
    let result = forward_backward_from_log_emissions_with_transitions(
        &log_emissions, &params, &log_trans,
    );
    assert_eq!(result.len(), 3);
    for row in &result {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

// ===========================================================================
// viterbi_from_log_emissions_with_transitions
// ===========================================================================

#[test]
fn vwt_empty_emissions_returns_empty() {
    let params = make_params(2, 0.01);
    let result = viterbi_from_log_emissions_with_transitions(&[], &params, &[]);
    assert!(result.is_empty());
}

#[test]
fn vwt_single_window_returns_argmax() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    // Strong signal for state 0
    let mut log_e = vec![-5.0; k];
    log_e[0] = -0.01;
    let log_emissions = vec![log_e];
    let result = viterbi_from_log_emissions_with_transitions(
        &log_emissions, &params, &[],
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 0);
}

#[test]
fn vwt_strong_signal_decoded_correctly() {
    let params = make_params(3, 0.01);
    let k = params.n_states;
    // 5 windows, state pattern: 0,0,1,1,2
    let expected = vec![0, 0, 1, 1, 2];
    let log_emissions: Vec<Vec<f64>> = expected.iter().map(|&s| {
        (0..k).map(|i| if i == s { -0.01 } else { -10.0 }).collect()
    }).collect();
    let log_trans: Vec<Vec<Vec<f64>>> = (0..4).map(|_| {
        params.transitions.iter().map(|row| {
            row.iter().map(|&p| p.max(1e-20).ln()).collect()
        }).collect()
    }).collect();
    let result = viterbi_from_log_emissions_with_transitions(
        &log_emissions, &params, &log_trans,
    );
    assert_eq!(result, expected);
}

#[test]
fn vwt_insufficient_transitions_falls_back() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    let log_emissions: Vec<Vec<f64>> = (0..3).map(|_| vec![-1.0; k]).collect();
    // 0 transition matrices for 3 windows → falls back
    let result = viterbi_from_log_emissions_with_transitions(
        &log_emissions, &params, &[],
    );
    assert_eq!(result.len(), 3);
    assert!(result.iter().all(|&s| s < k));
}

#[test]
fn vwt_output_states_in_valid_range() {
    let params = make_params(4, 0.05);
    let k = params.n_states;
    let log_emissions: Vec<Vec<f64>> = (0..10).map(|t| {
        (0..k).map(|s| if s == t % k { -0.5 } else { -3.0 }).collect()
    }).collect();
    let log_trans: Vec<Vec<Vec<f64>>> = (0..9).map(|_| {
        params.transitions.iter().map(|row| {
            row.iter().map(|&p| p.max(1e-20).ln()).collect()
        }).collect()
    }).collect();
    let result = viterbi_from_log_emissions_with_transitions(
        &log_emissions, &params, &log_trans,
    );
    assert_eq!(result.len(), 10);
    assert!(result.iter().all(|&s| s < k), "all states in [0, k)");
}

// ===========================================================================
// set_distance_weighted_transitions
// ===========================================================================

#[test]
fn sdwt_mismatched_dimensions_noop() {
    let mut params = make_params(2, 0.01);
    let orig_trans = params.transitions.clone();
    // Wrong dimensions → no-op
    set_distance_weighted_transitions(
        &mut params,
        &vec![vec![0.0; 3]; 3], // 3x3 for 2-state params
        &vec![0.5; 3],
        &vec![0.01; 3],
    );
    assert_eq!(params.transitions, orig_trans);
}

#[test]
fn sdwt_zero_distances_uniform_off_diagonal() {
    let mut params = make_params(3, 0.01);
    let k = params.n_states;
    let distances = vec![vec![0.0; k]; k];
    let proportions = vec![1.0 / k as f64; k];
    let switch_rates = vec![0.1; k];

    set_distance_weighted_transitions(&mut params, &distances, &proportions, &switch_rates);

    // All distances are 0 → all_dists empty → scale=1.0, closeness=exp(0)=1 everywhere
    for i in 0..k {
        let row_sum: f64 = params.transitions[i].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "row {i} should sum to 1");
        // Diagonal should be 1 - switch_rate
        assert!((params.transitions[i][i] - 0.9).abs() < 1e-6);
    }
}

#[test]
fn sdwt_large_distance_suppresses_transition() {
    let mut params = make_params(3, 0.01);
    let k = params.n_states;
    // Pop 0 and 1 close, pop 2 far
    let mut distances = vec![vec![0.0; k]; k];
    distances[0][1] = 0.01; distances[1][0] = 0.01;
    distances[0][2] = 10.0; distances[2][0] = 10.0;
    distances[1][2] = 10.0; distances[2][1] = 10.0;

    let proportions = vec![1.0 / k as f64; k];
    let switch_rates = vec![0.1; k];

    set_distance_weighted_transitions(&mut params, &distances, &proportions, &switch_rates);

    // P(0→1) should be much larger than P(0→2) since d(0,1) << d(0,2)
    assert!(params.transitions[0][1] > params.transitions[0][2],
        "close pop should have higher transition: 0→1={} > 0→2={}",
        params.transitions[0][1], params.transitions[0][2]);
}

#[test]
fn sdwt_switch_rate_clamped() {
    let mut params = make_params(2, 0.01);
    let k = params.n_states;
    let distances = vec![vec![0.0; k]; k];
    let proportions = vec![0.5; k];
    // Switch rates out of [0,1] → clamped
    let switch_rates = vec![1.5, -0.3];

    set_distance_weighted_transitions(&mut params, &distances, &proportions, &switch_rates);

    // Row sums should still be ~1.0
    for i in 0..k {
        let row_sum: f64 = params.transitions[i].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "row {i} sum={row_sum}");
    }
}

#[test]
fn sdwt_rows_sum_to_one() {
    let mut params = make_params(4, 0.01);
    let k = params.n_states;
    let mut distances = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in (i+1)..k {
            let d = (i as f64 - j as f64).abs() * 0.1;
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    let proportions = vec![0.25; k];
    let switch_rates = vec![0.05; k];

    set_distance_weighted_transitions(&mut params, &distances, &proportions, &switch_rates);

    for i in 0..k {
        let row_sum: f64 = params.transitions[i].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "row {i} sum={row_sum}");
        assert!(params.transitions[i].iter().all(|&v| v >= 0.0));
    }
}

// ===========================================================================
// compute_adaptive_transitions
// ===========================================================================

#[test]
fn cat_empty_emissions_returns_empty() {
    let params = make_params(2, 0.01);
    let result = compute_adaptive_transitions(&[], &params, 1.0);
    assert!(result.is_empty());
}

#[test]
fn cat_factor_zero_returns_base_transitions() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    let log_emissions = vec![vec![-1.0; k]; 3];
    let result = compute_adaptive_transitions(&log_emissions, &params, 0.0);
    assert_eq!(result.len(), 3);
    // factor=0 → scale=1.0 for all windows → base transitions (just normalized)
    for mat in &result {
        for (i, row) in mat.iter().enumerate() {
            let sum: f64 = row.iter().map(|v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {i} exp-sum={sum}");
        }
    }
}

#[test]
fn cat_confident_emissions_harder_transitions() {
    let params = make_params(3, 0.1);
    let k = params.n_states;
    // Window 0: very confident (one state dominates)
    let mut confident = vec![-20.0; k];
    confident[0] = -0.01;
    // Window 1: uncertain (uniform)
    let uncertain = vec![-1.0; k];

    let result = compute_adaptive_transitions(
        &[confident, uncertain], &params, 2.0,
    );
    assert_eq!(result.len(), 2);

    // Confident window should have harder off-diagonal (more negative in log)
    let off_diag_confident = result[0][0][1]; // log P(0→1) for confident window
    let off_diag_uncertain = result[1][0][1]; // log P(0→1) for uncertain window
    assert!(off_diag_confident < off_diag_uncertain,
        "confident should penalize switching more: {off_diag_confident} < {off_diag_uncertain}");
}

#[test]
fn cat_all_neg_infinity_emissions_fully_uncertain() {
    let params = make_params(2, 0.1);
    let k = params.n_states;
    let log_emissions = vec![vec![f64::NEG_INFINITY; k]];
    let result = compute_adaptive_transitions(&log_emissions, &params, 1.0);
    assert_eq!(result.len(), 1);
    // All NEG_INFINITY → entropy=1.0 → scale = 1 + factor*(1-1) = 1.0
    for row in &result[0] {
        let sum: f64 = row.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

#[test]
fn cat_rows_normalized_in_log_space() {
    let params = make_params(4, 0.05);
    let k = params.n_states;
    let log_emissions: Vec<Vec<f64>> = (0..5).map(|t| {
        (0..k).map(|s| if s == t % k { -0.5 } else { -3.0 }).collect()
    }).collect();
    let result = compute_adaptive_transitions(&log_emissions, &params, 1.5);
    for (t, mat) in result.iter().enumerate() {
        for (i, row) in mat.iter().enumerate() {
            let sum: f64 = row.iter().map(|v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 0.01,
                "t={t}, row {i}: exp-sum={sum}, want ~1.0");
        }
    }
}

#[test]
fn cat_output_count_matches_input() {
    let params = make_params(2, 0.01);
    let k = params.n_states;
    let log_emissions = vec![vec![-1.0; k]; 7];
    let result = compute_adaptive_transitions(&log_emissions, &params, 1.0);
    assert_eq!(result.len(), 7, "one matrix per window");
    for mat in &result {
        assert_eq!(mat.len(), k);
        for row in mat {
            assert_eq!(row.len(), k);
        }
    }
}

// ===========================================================================
// infer_ancestry_copying_em
// ===========================================================================

#[test]
fn iace_empty_observations_falls_back() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let (states, posteriors) = infer_ancestry_copying_em(
        &[], &pops, 0.01, 0.1, 1.0, 0.5, 3,
    );
    assert!(states.is_empty());
    assert!(posteriors.is_empty());
}

#[test]
fn iace_zero_em_iterations_falls_back() {
    let pops = vec![make_pop("A", &["h1"]), make_pop("B", &["h2"])];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.3)]),
    ];
    let (states, posteriors) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 0,
    );
    assert_eq!(states.len(), 1);
    assert_eq!(posteriors.len(), 1);
}

#[test]
fn iace_empty_populations_falls_back() {
    let obs = vec![make_obs(&[("h1", 0.9)])];
    let (states, posteriors) = infer_ancestry_copying_em(
        &obs, &[], 0.01, 0.1, 1.0, 0.5, 3,
    );
    // n_pops == 0 → falls back to infer_ancestry_copying which returns empty
    assert!(states.is_empty());
    assert!(posteriors.is_empty());
}

#[test]
fn iace_single_iteration_produces_valid_output() {
    let pops = vec![
        make_pop("A", &["h1", "h2"]),
        make_pop("B", &["h3", "h4"]),
    ];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.85), ("h3", 0.3), ("h4", 0.25)]),
        make_obs(&[("h1", 0.88), ("h2", 0.82), ("h3", 0.35), ("h4", 0.3)]),
        make_obs(&[("h1", 0.4), ("h2", 0.35), ("h3", 0.85), ("h4", 0.8)]),
    ];
    let (states, posteriors) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 1,
    );
    assert_eq!(states.len(), 3);
    assert_eq!(posteriors.len(), 3);
    for (t, row) in posteriors.iter().enumerate() {
        assert_eq!(row.len(), 2);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 0.05, "window {t}: posterior sum={sum}");
    }
}

#[test]
fn iace_multiple_iterations_converge() {
    let pops = vec![
        make_pop("A", &["h1", "h2"]),
        make_pop("B", &["h3", "h4"]),
    ];
    // Strong signal: first 3 windows A, last 3 windows B
    let obs: Vec<AncestryObservation> = (0..6).map(|t| {
        if t < 3 {
            make_obs(&[("h1", 0.95), ("h2", 0.90), ("h3", 0.2), ("h4", 0.15)])
        } else {
            make_obs(&[("h1", 0.15), ("h2", 0.2), ("h3", 0.9), ("h4", 0.95)])
        }
    }).collect();

    let (states_1, _) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 1,
    );
    let (states_5, _) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 5,
    );
    // Both should give reasonable decodings (states should be valid)
    assert_eq!(states_1.len(), 6);
    assert_eq!(states_5.len(), 6);
    assert!(states_1.iter().all(|&s| s < 2));
    assert!(states_5.iter().all(|&s| s < 2));
}

#[test]
fn iace_states_in_valid_range() {
    let pops = vec![
        make_pop("A", &["h1"]),
        make_pop("B", &["h2"]),
        make_pop("C", &["h3"]),
    ];
    let obs = vec![
        make_obs(&[("h1", 0.9), ("h2", 0.5), ("h3", 0.3)]),
        make_obs(&[("h1", 0.3), ("h2", 0.9), ("h3", 0.5)]),
    ];
    let (states, posteriors) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 2,
    );
    assert!(states.iter().all(|&s| s < 3), "states should be in [0, 3)");
    for row in &posteriors {
        assert_eq!(row.len(), 3);
    }
}

#[test]
fn iace_empty_haplotypes_uniform_posteriors() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec![] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec![] },
    ];
    let obs = vec![make_obs(&[])];
    let (states, posteriors) = infer_ancestry_copying_em(
        &obs, &pops, 0.01, 0.1, 1.0, 0.5, 3,
    );
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0);
    assert_eq!(posteriors.len(), 1);
    // Uniform posteriors for empty haplotypes
    for &p in &posteriors[0] {
        assert!((p - 0.5).abs() < 1e-6, "uniform: expected 0.5, got {p}");
    }
}
