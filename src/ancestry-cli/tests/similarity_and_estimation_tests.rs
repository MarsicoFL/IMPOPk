//! Tests for parse_similarity_data_column(), estimate_temperature(),
//! estimate_temperature_normalized(), estimate_switch_prob(), and EmissionModel::TopK.

use hprc_ancestry_cli::ancestry::parse_similarity_data_column;
use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    estimate_switch_prob, estimate_temperature, estimate_temperature_normalized,
};
use std::collections::HashMap;

// ===========================================================================
// Helpers
// ===========================================================================

fn two_pop() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR#hap1".to_string(), "EUR#hap2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR#hap1".to_string(), "AFR#hap2".to_string()],
        },
    ]
}

fn three_pop() -> Vec<AncestralPopulation> {
    let mut pops = two_pop();
    pops.push(AncestralPopulation {
        name: "EAS".to_string(),
        haplotypes: vec!["EAS#hap1".to_string(), "EAS#hap2".to_string()],
    });
    pops
}

/// Build a TSV dataset as lines for parse_similarity_data_column.
/// The header is the first line; subsequent lines are data rows.
fn make_tsv_lines(header: &str, rows: &[&str]) -> Vec<String> {
    let mut lines = vec![header.to_string()];
    lines.extend(rows.iter().map(|r| r.to_string()));
    lines
}

/// Create observations with explicit per-haplotype similarities for two pops.
fn make_obs(
    n_windows: usize,
    eur_sim: f64,
    afr_sim: f64,
) -> Vec<AncestryObservation> {
    let pops = two_pop();
    (0..n_windows)
        .map(|i| {
            let mut sims = HashMap::new();
            for pop in &pops {
                for hap in &pop.haplotypes {
                    let val = if pop.name == "EUR" { eur_sim } else { afr_sim };
                    sims.insert(hap.clone(), val);
                }
            }
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: i as u64 * 10000,
                end: (i as u64 + 1) * 10000,
                sample: "query".to_string(),
                similarities: sims,
                coverage_ratios: None,
            haplotype_consistency_bonus: None,
            }
        })
        .collect()
}

// ===========================================================================
// parse_similarity_data_column tests
// ===========================================================================

#[test]
fn test_parse_sim_column_default_identity() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        "chr1\t0\t10000\tQUERY#1#scaffold:0-10000\tEUR#hap1#scaffold:0-10000\t0.95",
        "chr1\t0\t10000\tQUERY#1#scaffold:0-10000\tAFR#hap1#scaffold:0-10000\t0.85",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string(), "AFR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    );
    assert!(result.is_ok());
    let data = result.unwrap();
    assert_eq!(data.len(), 1); // 1 query sample
    let obs = &data["QUERY#1"];
    assert_eq!(obs.len(), 1); // 1 window
    assert!((obs[0].similarities["EUR#hap1"] - 0.95).abs() < 1e-10);
    assert!((obs[0].similarities["AFR#hap1"] - 0.85).abs() < 1e-10);
}

#[test]
fn test_parse_sim_column_jaccard() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tjaccard.similarity";
    let rows = &[
        "chr1\t0\t10000\tQUERY#1#s:0-10000\tEUR#hap1#s:0-10000\t0.95\t0.80",
        "chr1\t0\t10000\tQUERY#1#s:0-10000\tAFR#hap1#s:0-10000\t0.85\t0.70",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string(), "AFR#hap1".to_string()];

    // Using jaccard.similarity instead of estimated.identity
    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "jaccard.similarity",
    );
    assert!(result.is_ok());
    let data = result.unwrap();
    let obs = &data["QUERY#1"];
    // Should pick jaccard values, not identity
    assert!((obs[0].similarities["EUR#hap1"] - 0.80).abs() < 1e-10);
    assert!((obs[0].similarities["AFR#hap1"] - 0.70).abs() < 1e-10);
}

#[test]
fn test_parse_sim_column_missing_column_error() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        "chr1\t0\t10000\tQUERY#1#s:0-10000\tEUR#hap1#s:0-10000\t0.95",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string()];

    // Request a column that doesn't exist
    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "nonexistent.column",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing column"));
}

#[test]
fn test_parse_sim_column_empty_data() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let lines = make_tsv_lines(header, &[]);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    );
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_parse_sim_column_skips_non_query_ref_pairs() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        // Query vs reference → should be included
        "chr1\t0\t10000\tQUERY#1#s:0-10000\tEUR#hap1#s:0-10000\t0.95",
        // Ref vs ref → should be skipped
        "chr1\t0\t10000\tEUR#hap1#s:0-10000\tAFR#hap1#s:0-10000\t0.80",
        // Unknown vs ref → should be skipped
        "chr1\t0\t10000\tUNKNOWN#1#s:0-10000\tEUR#hap1#s:0-10000\t0.90",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string(), "AFR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    )
    .unwrap();

    // Only QUERY#1 should appear
    assert_eq!(result.len(), 1);
    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 1);
    // Only EUR#hap1 should be in similarities (query vs ref pair)
    assert!(obs[0].similarities.contains_key("EUR#hap1"));
    assert!(!obs[0].similarities.contains_key("AFR#hap1"));
}

#[test]
fn test_parse_sim_column_multiple_windows_sorted() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        // Windows out of order
        "chr1\t20000\t30000\tQUERY#1#s:20000-30000\tEUR#hap1#s:20000-30000\t0.90",
        "chr1\t0\t10000\tQUERY#1#s:0-10000\tEUR#hap1#s:0-10000\t0.95",
        "chr1\t10000\t20000\tQUERY#1#s:10000-20000\tEUR#hap1#s:10000-20000\t0.85",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    )
    .unwrap();

    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 3);
    // Should be sorted by position
    assert_eq!(obs[0].start, 0);
    assert_eq!(obs[1].start, 10000);
    assert_eq!(obs[2].start, 20000);
}

#[test]
fn test_parse_sim_column_reversed_group_order() {
    // group.b is query, group.a is reference → should still work
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        "chr1\t0\t10000\tEUR#hap1#s:0-10000\tQUERY#1#s:0-10000\t0.95",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    let obs = &result["QUERY#1"];
    assert!((obs[0].similarities["EUR#hap1"] - 0.95).abs() < 1e-10);
}

#[test]
fn test_parse_sim_column_max_of_multiple_alignments() {
    // Two alignments for same window and pair → keep max
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let rows = &[
        "chr1\t0\t10000\tQUERY#1#s1:0-10000\tEUR#hap1#s1:0-10000\t0.85",
        "chr1\t0\t10000\tQUERY#1#s2:0-10000\tEUR#hap1#s2:0-10000\t0.95",
    ];
    let lines = make_tsv_lines(header, rows);

    let query_samples = vec!["QUERY#1".to_string()];
    let ref_haps = vec!["EUR#hap1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(),
        &query_samples,
        &ref_haps,
        "estimated.identity",
    )
    .unwrap();

    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 1);
    // Should keep max of 0.85 and 0.95
    assert!((obs[0].similarities["EUR#hap1"] - 0.95).abs() < 1e-10);
}

// ===========================================================================
// estimate_switch_prob tests
// ===========================================================================

#[test]
fn test_estimate_switch_prob_small_data_fallback() {
    let pops = two_pop();
    let obs = make_obs(5, 0.95, 0.85); // < 10 windows → fallback
    let result = estimate_switch_prob(&obs, &pops, 0.03);
    assert!((result - 0.001).abs() < 1e-10, "Should return fallback 0.001 for < 10 obs");
}

#[test]
fn test_estimate_switch_prob_homogeneous_signal() {
    // All windows strongly favor EUR → no switches → low switch prob
    let pops = two_pop();
    let obs = make_obs(100, 0.99, 0.80);
    let result = estimate_switch_prob(&obs, &pops, 0.03);
    // With homogeneous signal, observed rate should be near 0 → regularized towards 0.001
    // Expected: alpha*prior + (1-alpha)*0 ≈ 0.3*0.001 = 0.0003, clamped to 0.0001
    assert!(result >= 0.0001, "Must be >= lower clamp: {}", result);
    assert!(result <= 0.005, "Should be very low for no-switch signal: {}", result);
}

#[test]
fn test_estimate_switch_prob_alternating_signal() {
    // Alternating signal: windows alternate between EUR-favoring and AFR-favoring
    let pops = two_pop();
    let mut obs = Vec::new();
    for i in 0..50 {
        let mut sims = HashMap::new();
        if i % 2 == 0 {
            for hap in &pops[0].haplotypes { sims.insert(hap.clone(), 0.99); }
            for hap in &pops[1].haplotypes { sims.insert(hap.clone(), 0.80); }
        } else {
            for hap in &pops[0].haplotypes { sims.insert(hap.clone(), 0.80); }
            for hap in &pops[1].haplotypes { sims.insert(hap.clone(), 0.99); }
        }
        obs.push(AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        });
    }
    let result = estimate_switch_prob(&obs, &pops, 0.03);
    // Many switches → higher estimated rate
    assert!(result > 0.001, "Should detect high switch rate: {}", result);
    assert!(result <= 0.05, "Must be <= upper clamp: {}", result);
}

#[test]
fn test_estimate_switch_prob_within_clamp_range() {
    let pops = two_pop();
    let obs = make_obs(50, 0.95, 0.85);
    let result = estimate_switch_prob(&obs, &pops, 0.03);
    assert!(result >= 0.0001, "Below lower clamp");
    assert!(result <= 0.05, "Above upper clamp");
}

// ===========================================================================
// estimate_temperature_normalized tests
// ===========================================================================

#[test]
fn test_estimate_temp_normalized_without_normalization_fallback() {
    // No normalization set → should fall back to standard estimate_temperature
    let pops = two_pop();
    let params = AncestryHmmParams::new(pops, 0.001);
    // params.normalization is None
    let obs = make_obs(20, 0.95, 0.85);
    let result = estimate_temperature_normalized(&obs, &params);
    // Standard temperature estimation: median of (max - min) per window
    // Here each window has max=0.95, min=0.85, diff=0.10
    // Median = 0.10, clamped to [0.01, 0.15]
    assert!(result >= 0.01, "Below lower clamp: {}", result);
    assert!(result <= 0.15, "Above upper clamp: {}", result);
}

#[test]
fn test_estimate_temp_normalized_with_normalization() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let obs = make_obs(30, 0.95, 0.85);
    params.learn_normalization(&obs);
    assert!(params.normalization.is_some());

    let result = estimate_temperature_normalized(&obs, &params);
    // In z-score space, clamp range is [0.5, 5.0]
    assert!(result >= 0.5, "Below lower clamp for normalized: {}", result);
    assert!(result <= 5.0, "Above upper clamp for normalized: {}", result);
}

#[test]
fn test_estimate_temp_normalized_empty_obs() {
    let pops = two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.normalization = Some(hprc_ancestry_cli::hmm::PopulationNormalization {
        means: vec![0.9, 0.8],
        stds: vec![0.01, 0.01],
    });
    let obs: Vec<AncestryObservation> = vec![];
    let result = estimate_temperature_normalized(&obs, &params);
    // Empty → fallback 1.0
    assert!((result - 1.0).abs() < 1e-10, "Empty obs should return 1.0: {}", result);
}

// ===========================================================================
// EmissionModel::TopK / FromStr tests
// ===========================================================================

#[test]
fn test_emission_model_from_str_topk() {
    let model: EmissionModel = "top5".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(5));

    let model: EmissionModel = "top1".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(1));

    let model: EmissionModel = "top10".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(10));
}

#[test]
fn test_emission_model_from_str_standard() {
    let model: EmissionModel = "max".parse().unwrap();
    assert_eq!(model, EmissionModel::Max);

    let model: EmissionModel = "mean".parse().unwrap();
    assert_eq!(model, EmissionModel::Mean);

    let model: EmissionModel = "median".parse().unwrap();
    assert_eq!(model, EmissionModel::Median);
}

#[test]
fn test_emission_model_from_str_case_insensitive() {
    let model: EmissionModel = "MAX".parse().unwrap();
    assert_eq!(model, EmissionModel::Max);

    let model: EmissionModel = "Top3".parse().unwrap();
    assert_eq!(model, EmissionModel::TopK(3));
}

#[test]
fn test_emission_model_from_str_invalid() {
    assert!("unknown".parse::<EmissionModel>().is_err());
    assert!("topX".parse::<EmissionModel>().is_err());
    assert!("top".parse::<EmissionModel>().is_err());
    assert!("".parse::<EmissionModel>().is_err());
}

#[test]
fn test_topk_via_hmm_emission() {
    // Test TopK indirectly: with TopK(1) emission model, HMM should behave like Max
    let pops = two_pop();
    let mut params_max = AncestryHmmParams::new(pops.clone(), 0.001);
    params_max.set_emission_model(EmissionModel::Max);

    let mut params_top1 = AncestryHmmParams::new(pops, 0.001);
    params_top1.set_emission_model(EmissionModel::TopK(1));

    let obs = make_obs(20, 0.95, 0.85);

    // Viterbi with Max and TopK(1) should produce identical state sequences
    use hprc_ancestry_cli::hmm::viterbi;
    let states_max = viterbi(&obs, &params_max);
    let states_top1 = viterbi(&obs, &params_top1);
    assert_eq!(states_max, states_top1, "TopK(1) should produce same states as Max");
}

// ===========================================================================
// Normalization + estimation integration
// ===========================================================================

#[test]
fn test_normalization_then_temperature_estimation_consistent() {
    let pops = three_pop();
    let mut params = AncestryHmmParams::new(pops.clone(), 0.001);

    // Create observations with different signals per population
    let mut obs = Vec::new();
    for i in 0..50 {
        let mut sims = HashMap::new();
        for (pi, pop) in pops.iter().enumerate() {
            for hap in &pop.haplotypes {
                // Pop 0 has highest sim, pop 2 lowest
                sims.insert(hap.clone(), 0.95 - pi as f64 * 0.05 + (i as f64 * 0.001));
            }
        }
        obs.push(AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        });
    }

    params.learn_normalization(&obs);
    let norm = params.normalization.as_ref().unwrap();

    // Normalization means should reflect the per-pop averages
    assert!(norm.means[0] > norm.means[1], "Pop 0 should have higher mean");
    assert!(norm.means[1] > norm.means[2], "Pop 1 should have higher mean than pop 2");
    assert!(norm.stds.iter().all(|&s| s > 0.0), "All stds should be positive");

    // Temperature in normalized space should be in valid range
    let temp = estimate_temperature_normalized(&obs, &params);
    assert!(temp >= 0.5 && temp <= 5.0, "Normalized temp out of range: {}", temp);
}

// ===========================================================================
// estimate_temperature (raw / non-normalized) tests
// ===========================================================================

#[test]
fn test_estimate_temperature_empty_observations_returns_fallback() {
    let pops = two_pop();
    let result = estimate_temperature(&[], &pops);
    assert!((result - 0.03).abs() < 1e-10, "Empty obs should return fallback 0.03, got {}", result);
}

#[test]
fn test_estimate_temperature_identical_pop_sims_returns_fallback() {
    // When all populations have equal similarity, max-min = 0, so no diffs → fallback
    let pops = two_pop();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        let mut sims = HashMap::new();
        for pop in &pops {
            for hap in &pop.haplotypes {
                sims.insert(hap.clone(), 0.5); // same similarity for all pops
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    assert!((result - 0.03).abs() < 1e-10, "Identical sims should return fallback 0.03, got {}", result);
}

#[test]
fn test_estimate_temperature_clamped_low() {
    // When the median diff is very small (< 0.01), should be clamped to 0.01
    let pops = two_pop();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        let mut sims = HashMap::new();
        for pop in &pops {
            for hap in &pop.haplotypes {
                // Tiny diff: EUR=0.500, AFR=0.501 → diff=0.001
                let val = if pop.name == "EUR" { 0.500 } else { 0.501 };
                sims.insert(hap.clone(), val);
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    // All diffs = 0.001, upper-Q mean = 0.001 (above 0.0005 clamp)
    assert!((result - 0.001).abs() < 1e-10, "Upper-Q mean of constant diffs should be 0.001, got {}", result);
}

#[test]
fn test_estimate_temperature_clamped_high() {
    // When the median diff is very large (> 0.15), should be clamped to 0.15
    let pops = two_pop();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        let mut sims = HashMap::new();
        for pop in &pops {
            for hap in &pop.haplotypes {
                // Large diff: EUR=0.9, AFR=0.3 → diff=0.6
                let val = if pop.name == "EUR" { 0.9 } else { 0.3 };
                sims.insert(hap.clone(), val);
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    assert!((result - 0.15).abs() < 1e-10, "Large diffs should clamp to 0.15, got {}", result);
}

#[test]
fn test_estimate_temperature_within_range() {
    // Normal case: moderate differences
    let pops = two_pop();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        let mut sims = HashMap::new();
        for pop in &pops {
            for hap in &pop.haplotypes {
                // EUR=0.9, AFR=0.85 → diff=0.05 (within [0.01, 0.15])
                let val = if pop.name == "EUR" { 0.9 } else { 0.85 };
                sims.insert(hap.clone(), val);
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    assert!(result >= 0.01 && result <= 0.15, "Temperature {} out of clamp range", result);
    // For uniform diff of 0.05, median should be 0.05
    assert!((result - 0.05).abs() < 1e-10, "Expected 0.05, got {}", result);
}

#[test]
fn test_estimate_temperature_three_pops() {
    // Three populations: max-min picks the largest spread
    let pops = three_pop();
    let obs: Vec<AncestryObservation> = (0..20).map(|i| {
        let mut sims = HashMap::new();
        for pop in &pops {
            for hap in &pop.haplotypes {
                let val = match pop.name.as_str() {
                    "EUR" => 0.95,
                    "AFR" => 0.85,
                    "EAS" => 0.80,
                    _ => 0.5,
                };
                sims.insert(hap.clone(), val);
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    // max(0.95) - min(0.80) = 0.15 → clamped to 0.15
    assert!((result - 0.15).abs() < 1e-10, "Expected 0.15 for 3-pop spread, got {}", result);
}

#[test]
fn test_estimate_temperature_single_observation() {
    let pops = two_pop();
    let mut sims = HashMap::new();
    for pop in &pops {
        for hap in &pop.haplotypes {
            let val = if pop.name == "EUR" { 0.9 } else { 0.82 };
            sims.insert(hap.clone(), val);
        }
    }
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "query".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];

    let result = estimate_temperature(&obs, &pops);
    // diff = 0.08, median of one value = 0.08
    assert!((result - 0.08).abs() < 1e-10, "Single obs temp should be 0.08, got {}", result);
}

#[test]
fn test_estimate_temperature_missing_haplotype_data() {
    // Observations that only have data for one population → pop_sims.len() < 2 → skipped
    let pops = two_pop();
    let obs: Vec<AncestryObservation> = (0..10).map(|i| {
        let mut sims = HashMap::new();
        // Only EUR haplotypes, no AFR data
        sims.insert("EUR#hap1".to_string(), 0.9);
        sims.insert("EUR#hap2".to_string(), 0.88);
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 10000,
            end: (i + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let result = estimate_temperature(&obs, &pops);
    // No observations with ≥2 pop sims → empty diffs → fallback
    assert!((result - 0.03).abs() < 1e-10, "Missing pop data should return fallback, got {}", result);
}
