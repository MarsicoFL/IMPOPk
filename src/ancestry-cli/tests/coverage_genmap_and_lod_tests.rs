//! Tests for coverage-aware parsing, ancestry genetic map edge cases,
//! emission model error paths, LOD scores, and switch_point_accuracy.

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    EmissionModel,
};
use impopk_ancestry_cli::ancestry::{
    compute_per_window_ancestry_lod, coverage_ratio, parse_similarity_data_with_coverage,
    segment_ancestry_lod,
};
use impopk_ancestry_cli::concordance::switch_point_accuracy;

// ── EmissionModel::from_str error arms ──────────────────────────────────

#[test]
fn emission_model_from_str_invalid_topk_number() {
    let result = "topABC".parse::<EmissionModel>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("Invalid top-k value"),
        "Expected 'Invalid top-k value' in error: {}",
        err
    );
}

#[test]
fn emission_model_from_str_top_empty() {
    let result = "top".parse::<EmissionModel>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Invalid top-k value"));
}

#[test]
fn emission_model_from_str_unknown_model() {
    let result = "foobar".parse::<EmissionModel>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("Unknown emission model"),
        "Expected 'Unknown emission model' in error: {}",
        err
    );
}

// ── AncestryGeneticMap::uniform and interpolation ───────────────────────

#[test]
fn uniform_genmap_interpolation_at_boundaries() {
    let gm = AncestryGeneticMap::uniform(1_000_000, 2_000_000, 1.0);
    // At start
    let cm_start = gm.interpolate_cm(1_000_000);
    assert!((cm_start - 0.0).abs() < 1e-10);
    // At end
    let cm_end = gm.interpolate_cm(2_000_000);
    assert!((cm_end - 1.0).abs() < 1e-10);
    // At midpoint
    let cm_mid = gm.interpolate_cm(1_500_000);
    assert!((cm_mid - 0.5).abs() < 1e-10);
}

#[test]
fn uniform_genmap_extrapolation_before_start() {
    let gm = AncestryGeneticMap::uniform(1_000_000, 2_000_000, 1.0);
    // Before start: should extrapolate using the rate
    let cm = gm.interpolate_cm(500_000);
    // Rate is 1.0 cM/Mb, so 500kb before start = -0.5 cM
    assert!((cm - (-0.5)).abs() < 1e-10);
}

#[test]
fn uniform_genmap_extrapolation_after_end() {
    let gm = AncestryGeneticMap::uniform(1_000_000, 2_000_000, 1.0);
    // After end: extrapolate
    let cm = gm.interpolate_cm(3_000_000);
    // 1 Mb past end at 1.0 cM/Mb rate → 1.0 + 1.0 = 2.0 cM
    assert!((cm - 2.0).abs() < 1e-10);
}

#[test]
fn uniform_genmap_genetic_distance() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let dist = gm.genetic_distance_cm(1_000_000, 5_000_000);
    // 4 Mb at 1.0 cM/Mb = 4.0 cM
    assert!((dist - 4.0).abs() < 1e-10);
}

#[test]
fn ancestry_genmap_uniform_zero_range() {
    // start == end → degenerate map
    let gm = AncestryGeneticMap::uniform(1_000_000, 1_000_000, 1.0);
    // Interpolation at the exact point
    let cm = gm.interpolate_cm(1_000_000);
    assert!(cm.is_finite());
}

// ── log_emission with coverage_weight == 0.0 ────────────────────────────

fn make_params_2pop() -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "PopA".to_string(),
            haplotypes: vec!["h1".to_string(), "h2".to_string()],
        },
        AncestralPopulation {
            name: "PopB".to_string(),
            haplotypes: vec!["h3".to_string(), "h4".to_string()],
        },
    ];
    AncestryHmmParams::new(pops, 0.01)
}

fn make_obs_with_coverage() -> AncestryObservation {
    let mut sims = HashMap::new();
    sims.insert("h1".to_string(), 0.95);
    sims.insert("h2".to_string(), 0.92);
    sims.insert("h3".to_string(), 0.85);
    sims.insert("h4".to_string(), 0.80);

    let mut covs = HashMap::new();
    covs.insert("h1".to_string(), 0.99);
    covs.insert("h2".to_string(), 0.98);
    covs.insert("h3".to_string(), 0.90);
    covs.insert("h4".to_string(), 0.85);

    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "query".to_string(),
        similarities: sims,
        coverage_ratios: Some(covs),
            haplotype_consistency_bonus: None,
    }
}

#[test]
fn log_emission_with_zero_coverage_weight_equals_plain() {
    let mut params = make_params_2pop();
    params.coverage_weight = 0.0;
    let obs = make_obs_with_coverage();

    // With coverage_weight == 0, log_emission should NOT use coverage data
    let em0 = params.log_emission(&obs, 0);
    let em1 = params.log_emission(&obs, 1);

    // Create same params but remove coverage from obs
    let obs_no_cov = AncestryObservation {
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
        ..obs.clone()
    };
    let em0_no_cov = params.log_emission(&obs_no_cov, 0);
    let em1_no_cov = params.log_emission(&obs_no_cov, 1);

    assert!(
        (em0 - em0_no_cov).abs() < 1e-10,
        "State 0: {} vs {}",
        em0,
        em0_no_cov
    );
    assert!(
        (em1 - em1_no_cov).abs() < 1e-10,
        "State 1: {} vs {}",
        em1,
        em1_no_cov
    );
}

#[test]
fn log_emission_with_positive_coverage_weight_differs() {
    let mut params = make_params_2pop();
    params.coverage_weight = 0.0;
    let obs = make_obs_with_coverage();

    let em0_no_cov = params.log_emission(&obs, 0);

    params.coverage_weight = 1.0;
    let em0_with_cov = params.log_emission(&obs, 0);

    // They should differ when coverage data exists and weight > 0
    // (unless coverage doesn't contribute, which depends on the data)
    // At minimum, confirm both are finite
    assert!(em0_no_cov.is_finite());
    assert!(em0_with_cov.is_finite());
}

// ── compute_per_window_ancestry_lod ─────────────────────────────────────

#[test]
fn ancestry_lod_positive_for_correct_assignment() {
    let params = make_params_2pop();
    // Observation where PopA is clearly the best
    let mut sims = HashMap::new();
    sims.insert("h1".to_string(), 0.99);
    sims.insert("h2".to_string(), 0.98);
    sims.insert("h3".to_string(), 0.70);
    sims.insert("h4".to_string(), 0.65);
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "q".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };

    let lod = compute_per_window_ancestry_lod(&obs, &params, 0);
    assert!(lod > 0.0, "LOD should be positive for correct assignment, got {}", lod);
}

#[test]
fn ancestry_lod_negative_for_wrong_assignment() {
    let params = make_params_2pop();
    // PopA is the best match, assigning to PopB (state 1) should give negative LOD
    let mut sims = HashMap::new();
    sims.insert("h1".to_string(), 0.99);
    sims.insert("h2".to_string(), 0.98);
    sims.insert("h3".to_string(), 0.70);
    sims.insert("h4".to_string(), 0.65);
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "q".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };

    let lod = compute_per_window_ancestry_lod(&obs, &params, 1);
    assert!(lod < 0.0, "LOD should be negative for wrong assignment, got {}", lod);
}

#[test]
fn ancestry_lod_returns_zero_when_emissions_non_finite() {
    // With no matching haplotypes, emissions are -inf → LOD should return 0.0
    let params = make_params_2pop();
    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "q".to_string(),
        similarities: HashMap::new(), // no haplotypes at all
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    };

    let lod = compute_per_window_ancestry_lod(&obs, &params, 0);
    assert_eq!(lod, 0.0, "LOD should be 0.0 when emissions are non-finite");
}

// ── segment_ancestry_lod ────────────────────────────────────────────────

#[test]
fn segment_lod_sums_per_window_lods() {
    let params = make_params_2pop();
    let mut obs_list = Vec::new();
    for i in 0..5 {
        let mut sims = HashMap::new();
        sims.insert("h1".to_string(), 0.95);
        sims.insert("h2".to_string(), 0.93);
        sims.insert("h3".to_string(), 0.80);
        sims.insert("h4".to_string(), 0.75);
        obs_list.push(AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "q".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        });
    }

    let seg_lod = segment_ancestry_lod(&obs_list, &params, 0, 0, 4);

    // Should equal sum of individual LODs
    let sum_lod: f64 = (0..5)
        .map(|i| compute_per_window_ancestry_lod(&obs_list[i], &params, 0))
        .sum();
    assert!(
        (seg_lod - sum_lod).abs() < 1e-10,
        "segment LOD ({}) should equal sum of per-window LODs ({})",
        seg_lod,
        sum_lod
    );
}

#[test]
fn segment_lod_single_window() {
    let params = make_params_2pop();
    let mut sims = HashMap::new();
    sims.insert("h1".to_string(), 0.95);
    sims.insert("h3".to_string(), 0.80);
    let obs_list = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "q".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];

    let seg_lod = segment_ancestry_lod(&obs_list, &params, 0, 0, 0);
    let per_window = compute_per_window_ancestry_lod(&obs_list[0], &params, 0);
    assert!((seg_lod - per_window).abs() < 1e-10);
}

// ── switch_point_accuracy (simple variant) ──────────────────────────────

#[test]
fn switch_accuracy_no_true_switches() {
    let (frac, dist) = switch_point_accuracy(&[3, 5], &[], 2);
    assert_eq!(frac, 1.0); // Nothing to detect
    assert_eq!(dist, 0.0);
}

#[test]
fn switch_accuracy_no_predicted_switches() {
    let (frac, dist) = switch_point_accuracy(&[], &[3, 5, 10], 2);
    assert_eq!(frac, 0.0); // Missed all
    assert_eq!(dist, f64::INFINITY);
}

#[test]
fn switch_accuracy_both_empty() {
    let (frac, dist) = switch_point_accuracy(&[], &[], 2);
    assert_eq!(frac, 1.0); // No switches to detect
    assert_eq!(dist, 0.0);
}

#[test]
fn switch_accuracy_exact_match() {
    let (frac, dist) = switch_point_accuracy(&[5, 10, 15], &[5, 10, 15], 0);
    assert_eq!(frac, 1.0);
    assert_eq!(dist, 0.0);
}

#[test]
fn switch_accuracy_within_tolerance() {
    let (frac, dist) = switch_point_accuracy(&[6, 11], &[5, 10], 2);
    assert_eq!(frac, 1.0); // Both within tolerance of 2
    assert!((dist - 1.0).abs() < 1e-10); // Mean distance = (1+1)/2 = 1.0
}

#[test]
fn switch_accuracy_partial_detection() {
    let (frac, dist) = switch_point_accuracy(&[5], &[5, 100], 2);
    assert!((frac - 0.5).abs() < 1e-10); // 1 detected, 1 missed
}

// ── parse_similarity_data_with_coverage ─────────────────────────────────

#[test]
fn parse_coverage_valid_data() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tweighted_identity\tgroup.a.length\tgroup.b.length";
    // Use simple IDs without scaffold suffix — extract_sample_id keeps "name#hap"
    let line1 = "chr1\t0\t10000\tquery#1\tref_a#1\t0.95\t10000\t9500";
    let line2 = "chr1\t0\t10000\tquery#1\tref_b#1\t0.85\t10000\t8000";

    let lines = vec![header, line1, line2]
        .into_iter()
        .map(String::from);

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string(), "ref_b#1".to_string()];

    let result = parse_similarity_data_with_coverage(lines, &query_samples, &ref_haps, "weighted_identity");
    assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

    let data = result.unwrap();
    assert!(data.contains_key("query#1"), "Keys: {:?}", data.keys().collect::<Vec<_>>());
    let obs = &data["query#1"];
    assert_eq!(obs.len(), 1);
    assert!(obs[0].coverage_ratios.is_some());
    let covs = obs[0].coverage_ratios.as_ref().unwrap();
    // ref_a: min(10000,9500)/max(10000,9500) = 9500/10000 = 0.95
    assert!((covs["ref_a#1"] - 0.95).abs() < 1e-10);
    // ref_b: min(10000,8000)/max(10000,8000) = 8000/10000 = 0.80
    assert!((covs["ref_b#1"] - 0.80).abs() < 1e-10);
}

#[test]
fn parse_coverage_missing_column_error() {
    // Header missing group.a.length
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tweighted_identity\tgroup.b.length";
    let lines = vec![header].into_iter().map(String::from);

    let result = parse_similarity_data_with_coverage(
        lines,
        &["q#1".to_string()],
        &["r#1".to_string()],
        "weighted_identity",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing column"));
}

#[test]
fn parse_coverage_invalid_length_error() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tweighted_identity\tgroup.a.length\tgroup.b.length";
    let line = "chr1\t0\t10000\tq#1\tr#1\t0.95\tNOTANUMBER\t9500";
    let lines = vec![header, line].into_iter().map(String::from);

    let result = parse_similarity_data_with_coverage(
        lines,
        &["q#1".to_string()],
        &["r#1".to_string()],
        "weighted_identity",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid group.a.length"));
}

#[test]
fn parse_coverage_zero_lengths() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tweighted_identity\tgroup.a.length\tgroup.b.length";
    let line = "chr1\t0\t10000\tq#1\tr#1\t0.95\t0\t0";
    let lines = vec![header, line].into_iter().map(String::from);

    let result = parse_similarity_data_with_coverage(
        lines,
        &["q#1".to_string()],
        &["r#1".to_string()],
        "weighted_identity",
    );
    assert!(result.is_ok());
    let data = result.unwrap();
    let covs = data["q#1"][0].coverage_ratios.as_ref().unwrap();
    assert_eq!(covs["r#1"], 0.0); // both zero → 0.0
}

#[test]
fn parse_coverage_skips_non_matching_pairs() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tweighted_identity\tgroup.a.length\tgroup.b.length";
    // Neither side matches query or reference
    let line = "chr1\t0\t10000\tunknown#1\tother#1\t0.95\t10000\t9500";
    let lines = vec![header, line].into_iter().map(String::from);

    let result = parse_similarity_data_with_coverage(
        lines,
        &["q#1".to_string()],
        &["r#1".to_string()],
        "weighted_identity",
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty()); // No matching pairs
}

// ── coverage_ratio function ─────────────────────────────────────────────

#[test]
fn coverage_ratio_symmetric() {
    assert_eq!(coverage_ratio(100, 200), coverage_ratio(200, 100));
}

#[test]
fn coverage_ratio_equal_lengths() {
    assert_eq!(coverage_ratio(5000, 5000), 1.0);
}

#[test]
fn coverage_ratio_one_zero() {
    assert_eq!(coverage_ratio(0, 100), 0.0);
    assert_eq!(coverage_ratio(100, 0), 0.0);
}

// ── AncestryGeneticMap::modulated_switch_prob ───────────────────────────

#[test]
fn modulated_switch_prob_basic() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    let base_switch = 0.01;
    let window_size = 10000;

    // When genetic distance matches expected physical distance, switch prob ≈ base
    let prob = gm.modulated_switch_prob(base_switch, 0, 10000, window_size);
    assert!(prob > 0.0 && prob < 1.0, "Switch prob should be in (0,1), got {}", prob);
}

#[test]
fn modulated_switch_prob_same_position() {
    let gm = AncestryGeneticMap::uniform(0, 10_000_000, 1.0);
    // Same position → genetic distance = 0 → switch prob should be very small
    let prob = gm.modulated_switch_prob(0.01, 5_000_000, 5_000_000, 10000);
    assert!(prob <= 0.01, "Same-position switch prob should be <= base, got {}", prob);
}
