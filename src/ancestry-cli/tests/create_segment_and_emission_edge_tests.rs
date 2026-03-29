//! Tests for create_segment discriminability with single-population observations,
//! EmissionModel::aggregate NaN/edge cases, log_emission_similarity_only normalization paths,
//! log_emission_with_coverage cascading returns, learn_normalization with varied observations,
//! estimate_emissions classification logic, Baum-Welch M-step convergence guard,
//! and modulated_switch_prob Haldane scaling.

use std::collections::HashMap;

use hprc_ancestry_cli::ancestry::{
    extract_ancestry_segments, parse_similarity_data, AncestrySegment,
};
use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|h| h.to_string()).collect(),
    }
}

fn make_obs(
    sims: &[(&str, f64)],
    chrom: &str,
    start: u64,
    end: u64,
    sample: &str,
) -> AncestryObservation {
    AncestryObservation {
        chrom: chrom.to_string(),
        start,
        end,
        sample: sample.to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_with_coverage(
    sims: &[(&str, f64)],
    covs: &[(&str, f64)],
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 1000,
        end: 2000,
        sample: "sample1".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: Some(covs.iter().map(|(k, v)| (k.to_string(), *v)).collect()),
            haplotype_consistency_bonus: None,
    }
}

fn make_3way_params() -> AncestryHmmParams {
    AncestryHmmParams::new(
        vec![
            make_pop("EUR", &["EUR#1", "EUR#2"]),
            make_pop("AFR", &["AFR#1", "AFR#2"]),
            make_pop("AMR", &["AMR#1", "AMR#2"]),
        ],
        0.001,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// create_segment (via extract_ancestry_segments): discriminability edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// When only one population has data (non-zero similarity), discriminability = 0.0
#[test]
fn test_segment_single_population_data_zero_discriminability() {
    let params = make_3way_params();
    // Only EUR haplotypes have similarity; AFR and AMR have no matching haps
    let obs = vec![
        make_obs(&[("EUR#1", 0.95), ("EUR#2", 0.93)], "chr1", 1000, 2000, "s1"),
        make_obs(&[("EUR#1", 0.94), ("EUR#2", 0.92)], "chr1", 2000, 3000, "s1"),
    ];
    let states = vec![0, 0]; // state 0 = EUR

    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    // Only EUR pop has data → pop_sims.len() == 1 → discriminability = 0.0
    assert!((segments[0].discriminability - 0.0).abs() < 1e-10,
        "Expected 0.0 discriminability, got {}", segments[0].discriminability);
    assert!(segments[0].mean_similarity > 0.0);
}

/// When two populations have data, discriminability is max_pop - min_pop
#[test]
fn test_segment_two_populations_nonzero_discriminability() {
    let params = make_3way_params();
    let obs = vec![
        make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 1000, 2000, "s1"),
    ];
    let states = vec![0]; // EUR

    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    // EUR max = 0.95, AFR max = 0.85, discriminability = 0.95 - 0.85 = 0.10
    assert!((segments[0].discriminability - 0.10).abs() < 1e-10,
        "Expected 0.10 discriminability, got {}", segments[0].discriminability);
}

/// Three populations with data → discriminability is difference between highest and lowest
#[test]
fn test_segment_three_populations_discriminability() {
    let params = make_3way_params();
    let obs = vec![
        make_obs(
            &[("EUR#1", 0.95), ("AFR#1", 0.85), ("AMR#1", 0.90)],
            "chr1", 1000, 2000, "s1",
        ),
    ];
    let states = vec![0]; // EUR

    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    // max_pop = 0.95, min_pop = 0.85, disc = 0.10
    assert!((segments[0].discriminability - 0.10).abs() < 1e-10);
}

/// Zero similarity for all haplotypes → mean_similarity = 0, discriminability = 0
#[test]
fn test_segment_all_zero_similarities() {
    let params = make_3way_params();
    let obs = vec![
        make_obs(&[("EUR#1", 0.0), ("AFR#1", 0.0), ("AMR#1", 0.0)],
            "chr1", 1000, 2000, "s1"),
    ];
    let states = vec![0];
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].mean_similarity - 0.0).abs() < 1e-10);
    assert!((segments[0].discriminability - 0.0).abs() < 1e-10);
}

/// Multiple windows with varying discriminability → averaged
#[test]
fn test_segment_averaged_discriminability() {
    let params = make_3way_params();
    let obs = vec![
        make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 1000, 2000, "s1"), // disc = 0.10
        make_obs(&[("EUR#1", 0.90), ("AFR#1", 0.90)], "chr1", 2000, 3000, "s1"), // disc = 0.00
    ];
    let states = vec![0, 0];
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].discriminability - 0.05).abs() < 1e-10,
        "Expected 0.05, got {}", segments[0].discriminability);
}

/// Segment with posteriors captures mean_posterior correctly
#[test]
fn test_segment_with_posteriors() {
    let params = make_3way_params();
    let obs = vec![
        make_obs(&[("EUR#1", 0.95)], "chr1", 1000, 2000, "s1"),
        make_obs(&[("EUR#1", 0.93)], "chr1", 2000, 3000, "s1"),
    ];
    let states = vec![0, 0];
    let posteriors = vec![
        vec![0.9, 0.05, 0.05],
        vec![0.7, 0.15, 0.15],
    ];
    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
    assert_eq!(segments.len(), 1);
    // mean_posterior for state 0 = (0.9 + 0.7) / 2 = 0.8
    assert!((segments[0].mean_posterior.unwrap() - 0.8).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// log_emission: normalization branch coverage
// ─────────────────────────────────────────────────────────────────────────────

/// Without normalization, raw similarity goes to softmax
#[test]
fn test_log_emission_no_normalization() {
    let params = make_3way_params();
    let obs = make_obs(
        &[("EUR#1", 0.95), ("AFR#1", 0.85), ("AMR#1", 0.90)],
        "chr1", 1000, 2000, "s1",
    );
    let log_e = params.log_emission(&obs, 0); // EUR state
    assert!(log_e.is_finite());
    assert!(log_e <= 0.0, "Log probability should be <= 0");
}

/// With normalization, z-scores modify the softmax inputs
#[test]
fn test_log_emission_with_normalization() {
    let mut params = make_3way_params();
    // Create observations to learn normalization from
    let train_obs: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(
            &[("EUR#1", 0.95 + (i as f64) * 0.001),
              ("AFR#1", 0.85 + (i as f64) * 0.001),
              ("AMR#1", 0.90 + (i as f64) * 0.001)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    params.learn_normalization(&train_obs);
    assert!(params.normalization.is_some());

    let obs = make_obs(
        &[("EUR#1", 0.96), ("AFR#1", 0.86), ("AMR#1", 0.91)],
        "chr1", 1000, 2000, "s1",
    );
    let log_e = params.log_emission(&obs, 0);
    assert!(log_e.is_finite());
    assert!(log_e <= 0.0);
}

/// learn_normalization with all-same similarities → std = 1e-6 (floor)
#[test]
fn test_learn_normalization_constant_sims() {
    let mut params = make_3way_params();
    let obs: Vec<AncestryObservation> = (0..10).map(|i| {
        make_obs(
            &[("EUR#1", 0.95), ("AFR#1", 0.95), ("AMR#1", 0.95)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    // All populations have same similarity → std very small → floored to 1e-6
    for &std in &norm.stds {
        assert!((std - 1e-6).abs() < 1e-8, "Expected std floor 1e-6, got {std}");
    }
}

/// learn_normalization with no observations → means = 0, stds = 1e-6
#[test]
fn test_learn_normalization_no_observations() {
    let mut params = make_3way_params();
    params.learn_normalization(&[]);
    let norm = params.normalization.as_ref().unwrap();
    for (&mean, &std) in norm.means.iter().zip(norm.stds.iter()) {
        assert!((mean - 0.0).abs() < 1e-10);
        assert!((std - 1e-6).abs() < 1e-8);
    }
}

/// learn_normalization with one population having no data
#[test]
fn test_learn_normalization_missing_population() {
    let mut params = make_3way_params();
    // Only EUR and AFR have data, AMR has nothing
    let obs: Vec<AncestryObservation> = (0..10).map(|i| {
        make_obs(
            &[("EUR#1", 0.95), ("AFR#1", 0.85)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();
    // AMR (index 2) should have mean=0, std=1e-6
    assert!((norm.means[2] - 0.0).abs() < 1e-10);
    assert!((norm.stds[2] - 1e-6).abs() < 1e-8);
    // EUR and AFR should have non-zero means
    assert!(norm.means[0] > 0.0);
    assert!(norm.means[1] > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// log_emission_with_coverage: cascading return paths
// ─────────────────────────────────────────────────────────────────────────────

/// coverage_weight > 0 but no coverage data → returns sim-only emission
#[test]
fn test_log_emission_coverage_weight_no_data() {
    let mut params = make_3way_params();
    params.set_coverage_weight(1.0);
    let obs = make_obs(
        &[("EUR#1", 0.95), ("AFR#1", 0.85)],
        "chr1", 1000, 2000, "s1",
    ); // No coverage_ratios
    let log_e = params.log_emission(&obs, 0);
    assert!(log_e.is_finite());
}

/// coverage_weight > 0 with coverage data → combined emission
#[test]
fn test_log_emission_coverage_combined() {
    let mut params = make_3way_params();
    params.set_coverage_weight(0.5);
    let obs = make_obs_with_coverage(
        &[("EUR#1", 0.95), ("AFR#1", 0.85)],
        &[("EUR#1", 0.9), ("AFR#1", 0.7)],
    );
    let log_e_combined = params.log_emission(&obs, 0);
    params.set_coverage_weight(0.0);
    let log_e_sim_only = params.log_emission(&obs, 0);
    // Combined should differ from sim-only
    assert!((log_e_combined - log_e_sim_only).abs() > 1e-6,
        "Combined should differ: {log_e_combined} vs {log_e_sim_only}");
}

/// coverage_weight > 0 but target coverage is zero → returns sim-only
#[test]
fn test_log_emission_coverage_zero_target() {
    let mut params = make_3way_params();
    params.set_coverage_weight(1.0);
    let obs = make_obs_with_coverage(
        &[("EUR#1", 0.95), ("AFR#1", 0.85)],
        &[("EUR#1", 0.0), ("AFR#1", 0.7)], // EUR coverage = 0
    );
    // State 0 = EUR, target coverage = 0 → returns sim-only
    let log_e = params.log_emission(&obs, 0);
    assert!(log_e.is_finite());
}

/// Only one population has coverage → returns sim-only (valid_covs.len() <= 1)
#[test]
fn test_log_emission_coverage_single_population() {
    let mut params = make_3way_params();
    params.set_coverage_weight(1.0);
    let obs = make_obs_with_coverage(
        &[("EUR#1", 0.95), ("AFR#1", 0.85)],
        &[("EUR#1", 0.9)], // Only EUR has coverage
    );
    let log_e = params.log_emission(&obs, 0);
    assert!(log_e.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// estimate_emissions: supervised classification heuristic
// ─────────────────────────────────────────────────────────────────────────────

/// Observations where EUR is always best → same_pop_mean set from EUR
#[test]
fn test_estimate_emissions_single_dominant_population() {
    let mut params = make_3way_params();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(
            &[("EUR#1", 0.98), ("AFR#1", 0.80), ("AMR#1", 0.85)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    params.estimate_emissions(&obs);
    // EUR is best every time → same_pop_mean ~0.98
    assert!(params.emission_same_pop_mean > 0.95);
    // AFR and AMR are diff-pop → diff_pop_mean in range [0.8, 0.85]
    assert!(params.emission_diff_pop_mean > 0.75 && params.emission_diff_pop_mean < 0.90);
}

/// Mixed observations with different "best" populations
#[test]
fn test_estimate_emissions_mixed_best_populations() {
    let mut params = make_3way_params();
    let mut obs = Vec::new();
    // 10 obs where EUR is best
    for i in 0..10 {
        obs.push(make_obs(
            &[("EUR#1", 0.98), ("AFR#1", 0.80)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        ));
    }
    // 10 obs where AFR is best
    for i in 10..20 {
        obs.push(make_obs(
            &[("EUR#1", 0.80), ("AFR#1", 0.98)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        ));
    }
    params.estimate_emissions(&obs);
    // Both EUR and AFR contribute to same_pop and diff_pop
    assert!(params.emission_same_pop_mean > 0.95); // 0.98 from best
    assert!(params.emission_diff_pop_mean < 0.85); // 0.80 from non-best
}

/// Empty observations → emissions unchanged from defaults
#[test]
fn test_estimate_emissions_empty_observations() {
    let mut params = make_3way_params();
    let original_same = params.emission_same_pop_mean;
    let original_diff = params.emission_diff_pop_mean;
    params.estimate_emissions(&[]);
    assert!((params.emission_same_pop_mean - original_same).abs() < 1e-10);
    assert!((params.emission_diff_pop_mean - original_diff).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Baum-Welch: convergence and M-step edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// Single-iteration BW with clear signal → transitions updated
#[test]
fn test_bw_single_iteration_updates_transitions() {
    let mut params = make_3way_params();
    let original_switch = 1.0 - params.transitions[0][0];

    let obs = vec![
        make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 0, 1000, "s1"),
        make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 1000, 2000, "s1"),
        make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 2000, 3000, "s1"),
    ];
    let obs_slices: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_slices, 1, 1e-6);
    let new_switch = 1.0 - params.transitions[0][0];

    // Transitions should change after BW
    assert!(ll.is_finite() || ll == f64::NEG_INFINITY);
    // With only 3 observations of EUR, switch prob should decrease (more sure about EUR)
    assert!(new_switch != original_switch || ll == f64::NEG_INFINITY,
        "BW should modify transitions unless degenerate");
}

/// BW with k=1 returns NEG_INFINITY
#[test]
fn test_bw_single_state_returns_neg_inf() {
    let mut params = AncestryHmmParams::new(
        vec![make_pop("EUR", &["EUR#1"])],
        0.0, // single state, no switch
    );
    let obs = vec![
        make_obs(&[("EUR#1", 0.95)], "chr1", 0, 1000, "s1"),
    ];
    let obs_slices: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_slices, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

/// BW with empty observations returns NEG_INFINITY
#[test]
fn test_bw_empty_observations_returns_neg_inf() {
    let mut params = make_3way_params();
    let obs_slices: Vec<&[AncestryObservation]> = vec![];
    let ll = params.baum_welch(&obs_slices, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

/// BW skips sequences with < 2 observations
#[test]
fn test_bw_short_sequences_skipped() {
    let mut params = make_3way_params();
    let obs1 = vec![
        make_obs(&[("EUR#1", 0.95)], "chr1", 0, 1000, "s1"),
    ]; // length 1 → skipped
    let obs_slices: Vec<&[AncestryObservation]> = vec![&obs1];
    let ll = params.baum_welch(&obs_slices, 10, 1e-6);
    // All sequences skipped → LL stays at NEG_INFINITY
    assert!(ll == f64::NEG_INFINITY || ll == 0.0,
        "Expected degenerate result, got {ll}");
}

// ─────────────────────────────────────────────────────────────────────────────
// EmissionModel: edge cases via log_emission (indirect aggregate testing)
// ─────────────────────────────────────────────────────────────────────────────

/// TopK(0) → aggregate returns None → log_emission returns NEG_INFINITY
#[test]
fn test_emission_model_topk_zero() {
    let mut params = make_3way_params();
    params.set_emission_model(EmissionModel::TopK(0));
    let obs = make_obs(&[("EUR#1", 0.95), ("AFR#1", 0.85)], "chr1", 0, 1000, "s1");
    let log_e = params.log_emission(&obs, 0);
    assert_eq!(log_e, f64::NEG_INFINITY, "TopK(0) should yield NEG_INFINITY");
}

/// Median with even count → averages two middle values
#[test]
fn test_emission_model_median_even_count() {
    let mut params = AncestryHmmParams::new(
        vec![
            make_pop("POP_A", &["A#1", "A#2", "A#3", "A#4"]),
            make_pop("POP_B", &["B#1", "B#2"]),
        ],
        0.001,
    );
    params.set_emission_model(EmissionModel::Median);
    // 4 haplotypes in POP_A → even count median
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 1000,
        sample: "s1".to_string(),
        similarities: [
            ("A#1".to_string(), 0.90),
            ("A#2".to_string(), 0.92),
            ("A#3".to_string(), 0.94),
            ("A#4".to_string(), 0.96),
            ("B#1".to_string(), 0.80),
            ("B#2".to_string(), 0.82),
        ].into_iter().collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };
    let log_e = params.log_emission(&obs, 0);
    assert!(log_e.is_finite());
    // Median of [0.90, 0.92, 0.94, 0.96] = (0.92 + 0.94) / 2 = 0.93
    // The emission should prefer state 0 (POP_A) since median is higher
    let log_e_b = params.log_emission(&obs, 1);
    assert!(log_e > log_e_b, "POP_A should have higher emission");
}

/// Mean with single haplotype → same as Max
#[test]
fn test_emission_model_mean_single_haplotype() {
    let mut params = AncestryHmmParams::new(
        vec![
            make_pop("POP_A", &["A#1"]),
            make_pop("POP_B", &["B#1"]),
        ],
        0.001,
    );
    params.set_emission_model(EmissionModel::Mean);
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 1000,
        sample: "s1".to_string(),
        similarities: [
            ("A#1".to_string(), 0.95),
            ("B#1".to_string(), 0.85),
        ].into_iter().collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };
    let log_e_mean = params.log_emission(&obs, 0);
    params.set_emission_model(EmissionModel::Max);
    let log_e_max = params.log_emission(&obs, 0);
    assert!((log_e_mean - log_e_max).abs() < 1e-10,
        "Single haplotype: Mean should equal Max");
}

/// TopK(k) where k > number of haplotypes → same as Mean
#[test]
fn test_emission_model_topk_exceeds_haps() {
    let mut params = AncestryHmmParams::new(
        vec![
            make_pop("POP_A", &["A#1", "A#2"]),
            make_pop("POP_B", &["B#1"]),
        ],
        0.001,
    );
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 1000,
        sample: "s1".to_string(),
        similarities: [
            ("A#1".to_string(), 0.95),
            ("A#2".to_string(), 0.93),
            ("B#1".to_string(), 0.85),
        ].into_iter().collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };
    params.set_emission_model(EmissionModel::TopK(100)); // k >> haps
    let log_e_topk = params.log_emission(&obs, 0);
    params.set_emission_model(EmissionModel::Mean);
    let log_e_mean = params.log_emission(&obs, 0);
    assert!((log_e_topk - log_e_mean).abs() < 1e-10,
        "TopK with k > haps should equal Mean");
}

// ─────────────────────────────────────────────────────────────────────────────
// set_temperature and set_switch_prob: parameter update effects
// ─────────────────────────────────────────────────────────────────────────────

/// Very low temperature → emissions become more extreme
#[test]
fn test_low_temperature_sharpens_emissions() {
    let mut params_cold = make_3way_params();
    let mut params_warm = make_3way_params();
    params_cold.set_temperature(0.01);
    params_warm.set_temperature(0.5);

    let obs = make_obs(
        &[("EUR#1", 0.95), ("AFR#1", 0.85), ("AMR#1", 0.90)],
        "chr1", 1000, 2000, "s1",
    );

    let cold_best = params_cold.log_emission(&obs, 0); // EUR (best)
    let cold_worst = params_cold.log_emission(&obs, 1); // AFR (worst)
    let warm_best = params_warm.log_emission(&obs, 0);
    let warm_worst = params_warm.log_emission(&obs, 1);

    // Cold: bigger gap between best and worst
    let cold_gap = cold_best - cold_worst;
    let warm_gap = warm_best - warm_worst;
    assert!(cold_gap > warm_gap,
        "Cold temp should give bigger gap: {cold_gap} vs {warm_gap}");
}

/// Very high switch prob → transitions mostly off-diagonal
#[test]
fn test_high_switch_prob_transitions() {
    let mut params = make_3way_params();
    params.set_switch_prob(0.9);
    // Stay = 0.1, switch = 0.9/2 = 0.45 each
    assert!((params.transitions[0][0] - 0.1).abs() < 1e-10);
    assert!((params.transitions[0][1] - 0.45).abs() < 1e-10);
    assert!((params.transitions[0][2] - 0.45).abs() < 1e-10);
}

/// Very low switch prob → transitions mostly diagonal
#[test]
fn test_low_switch_prob_transitions() {
    let mut params = make_3way_params();
    params.set_switch_prob(0.0001);
    assert!((params.transitions[0][0] - 0.9999).abs() < 1e-10);
    assert!((params.transitions[0][1] - 0.00005).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Viterbi and forward_backward: property tests
// ─────────────────────────────────────────────────────────────────────────────

/// Viterbi assigns highest-similarity population as state
#[test]
fn test_viterbi_assigns_most_similar_population() {
    let params = make_3way_params();
    // All windows clearly favor EUR
    let obs: Vec<AncestryObservation> = (0..10).map(|i| {
        make_obs(
            &[("EUR#1", 0.98), ("AFR#1", 0.80), ("AMR#1", 0.85)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    let states = hprc_ancestry_cli::hmm::viterbi(&obs, &params);
    assert!(states.iter().all(|&s| s == 0), "All states should be EUR (0)");
}

/// Forward-backward posteriors sum to ~1.0 at each position
#[test]
fn test_fb_posteriors_sum_to_one() {
    let params = make_3way_params();
    let obs: Vec<AncestryObservation> = (0..5).map(|i| {
        make_obs(
            &[("EUR#1", 0.95), ("AFR#1", 0.85), ("AMR#1", 0.90)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        )
    }).collect();
    let posteriors = hprc_ancestry_cli::hmm::forward_backward(&obs, &params);
    for (t, probs) in posteriors.iter().enumerate() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Posteriors at t={t} sum to {sum}, not 1.0");
    }
}

/// Switching signal: different populations in different regions
#[test]
fn test_viterbi_detects_ancestry_switch() {
    let params = make_3way_params();
    let mut obs = Vec::new();
    // First 10 windows: EUR dominant
    for i in 0..10 {
        obs.push(make_obs(
            &[("EUR#1", 0.98), ("AFR#1", 0.80)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        ));
    }
    // Next 10 windows: AFR dominant
    for i in 10..20 {
        obs.push(make_obs(
            &[("EUR#1", 0.80), ("AFR#1", 0.98)],
            "chr1", (i * 1000) as u64, ((i + 1) * 1000) as u64, "s1",
        ));
    }
    let states = hprc_ancestry_cli::hmm::viterbi(&obs, &params);
    // Should have at least one switch
    let switches: usize = states.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(switches >= 1, "Should detect at least one ancestry switch");
    // First half should mostly be EUR (0), second half AFR (1)
    let first_half_eur = states[0..5].iter().filter(|&&s| s == 0).count();
    let second_half_afr = states[15..20].iter().filter(|&&s| s == 1).count();
    assert!(first_half_eur >= 3, "First half should be mostly EUR");
    assert!(second_half_afr >= 3, "Second half should be mostly AFR");
}
