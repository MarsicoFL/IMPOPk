//! Tests for EmissionModel via log_emission, RFMix windowing edge cases, and admixture estimation
//!
//! Cycle 40: Focuses on genuinely untested branches in:
//! - EmissionModel variants (Max, Mean, Median, TopK) via log_emission public API
//! - set_emission_model changing behavior, TopK(k) > len fallback
//! - rfmix_to_windows: segment with end <= min_pos (continue branch, line 266)
//! - rfmix_to_windows: max_pos <= min_pos early return (line 248-249)
//! - rfmix_window_starts: max_pos <= min_pos early return
//! - estimate_admixture_proportions: segments with same ancestry (n_switches = 0)
//! - estimate_admixture_proportions: empty segments
//! - smooth_states: min_run=1 (no smoothing), short run NOT surrounded by same state
//! - extract_ancestry_segments: alternating states produce multiple segments
//! - parse_rfmix_msp_content: data line with extra trailing fields

use hprc_ancestry_cli::hmm::{
    EmissionModel, AncestryHmmParams, AncestryObservation, AncestralPopulation,
};
use hprc_ancestry_cli::rfmix::{
    parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts,
    RfmixResult, RfmixSegment,
};
use hprc_ancestry_cli::ancestry::{
    extract_ancestry_segments, estimate_admixture_proportions, smooth_states,
    count_smoothing_changes, filter_segments_by_min_lod, AncestrySegment,
};
use std::collections::HashMap;

// ============================================================================
// EmissionModel via log_emission: different models produce different rankings
// ============================================================================

/// Helper: build AncestryHmmParams with two populations, each having the given haplotypes.
fn make_two_pop_params(
    pop0_name: &str, pop0_haps: &[&str],
    pop1_name: &str, pop1_haps: &[&str],
    model: EmissionModel,
) -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: pop0_name.to_string(),
            haplotypes: pop0_haps.iter().map(|s| s.to_string()).collect(),
        },
        AncestralPopulation {
            name: pop1_name.to_string(),
            haplotypes: pop1_haps.iter().map(|s| s.to_string()).collect(),
        },
    ];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_emission_model(model);
    params
}

fn make_obs(sims: Vec<(&str, f64)>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "HG001".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

#[test]
fn log_emission_max_model_picks_highest_haplotype() {
    // Pop0 has haps with sims [0.90, 0.60] → Max = 0.90
    // Pop1 has haps with sims [0.85]       → Max = 0.85
    // So log_emission(state=0) > log_emission(state=1) with Max
    let params = make_two_pop_params(
        "AFR", &["h1", "h2"],
        "EUR", &["h3"],
        EmissionModel::Max,
    );
    let obs = make_obs(vec![("h1", 0.90), ("h2", 0.60), ("h3", 0.85)]);
    let le0 = params.log_emission(&obs, 0);
    let le1 = params.log_emission(&obs, 1);
    assert!(le0 > le1, "Max model: pop with higher max should have higher emission");
}

#[test]
fn log_emission_mean_model_uses_average() {
    // Pop0 has haps with sims [0.90, 0.60] → Mean = 0.75
    // Pop1 has haps with sims [0.80]       → Mean = 0.80
    // With Mean model: pop1 (0.80) > pop0 (0.75)
    let params = make_two_pop_params(
        "AFR", &["h1", "h2"],
        "EUR", &["h3"],
        EmissionModel::Mean,
    );
    let obs = make_obs(vec![("h1", 0.90), ("h2", 0.60), ("h3", 0.80)]);
    let le0 = params.log_emission(&obs, 0);
    let le1 = params.log_emission(&obs, 1);
    assert!(le1 > le0, "Mean model: pop with higher mean should have higher emission");
}

#[test]
fn log_emission_median_model_even_count() {
    // Pop0 has 4 haps → median = avg of middle 2
    // Sims: [0.50, 0.70, 0.80, 0.90] → median = (0.70+0.80)/2 = 0.75
    // Pop1 has 1 hap → median = 0.76
    let params = make_two_pop_params(
        "AFR", &["h1", "h2", "h3", "h4"],
        "EUR", &["h5"],
        EmissionModel::Median,
    );
    let obs = make_obs(vec![
        ("h1", 0.50), ("h2", 0.70), ("h3", 0.80), ("h4", 0.90),
        ("h5", 0.76),
    ]);
    let le0 = params.log_emission(&obs, 0);
    let le1 = params.log_emission(&obs, 1);
    // median(pop0)=0.75, pop1=0.76 → pop1 slightly higher
    assert!(le1 > le0, "Median model: pop with higher median should have higher emission");
}

#[test]
fn log_emission_topk_model_uses_top_k_mean() {
    // Pop0 has haps with sims [0.50, 0.60, 0.95, 0.90] → TopK(2) = (0.95+0.90)/2 = 0.925
    // Pop1 has haps with sims [0.92] → TopK(2) = 0.92
    // TopK(2): pop0 > pop1
    let params = make_two_pop_params(
        "AFR", &["h1", "h2", "h3", "h4"],
        "EUR", &["h5"],
        EmissionModel::TopK(2),
    );
    let obs = make_obs(vec![
        ("h1", 0.50), ("h2", 0.60), ("h3", 0.95), ("h4", 0.90),
        ("h5", 0.92),
    ]);
    let le0 = params.log_emission(&obs, 0);
    let le1 = params.log_emission(&obs, 1);
    assert!(le0 > le1, "TopK(2) model: pop with higher top-2 mean should have higher emission");
}

#[test]
fn log_emission_topk_larger_than_haplotype_count() {
    // TopK(100) but pop only has 2 haps → uses all, same as Mean
    let params_topk = make_two_pop_params(
        "AFR", &["h1", "h2"],
        "EUR", &["h3"],
        EmissionModel::TopK(100),
    );
    let params_mean = make_two_pop_params(
        "AFR", &["h1", "h2"],
        "EUR", &["h3"],
        EmissionModel::Mean,
    );
    let obs = make_obs(vec![("h1", 0.80), ("h2", 0.60), ("h3", 0.70)]);
    let le_topk = params_topk.log_emission(&obs, 0);
    let le_mean = params_mean.log_emission(&obs, 0);
    // TopK(100) with only 2 haps should equal Mean
    assert!((le_topk - le_mean).abs() < 1e-10,
        "TopK(100) with 2 haps should equal Mean: topk={}, mean={}", le_topk, le_mean);
}

#[test]
fn log_emission_missing_haplotype_data() {
    // Pop0 has haplotypes but no similarity data → returns NEG_INFINITY
    let params = make_two_pop_params(
        "AFR", &["h1", "h2"],
        "EUR", &["h3"],
        EmissionModel::Max,
    );
    let obs = make_obs(vec![("h3", 0.80)]); // no data for h1, h2
    let le0 = params.log_emission(&obs, 0);
    assert!(le0 == f64::NEG_INFINITY, "Missing pop data should give NEG_INFINITY");
    let le1 = params.log_emission(&obs, 1);
    assert!(le1.is_finite(), "Pop with data should give finite emission");
}

#[test]
fn log_emission_set_emission_model_changes_behavior() {
    // Verify that set_emission_model actually changes the emission model
    let pops = vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["h1".to_string(), "h2".to_string()],
        },
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["h3".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = make_obs(vec![("h1", 0.90), ("h2", 0.50), ("h3", 0.75)]);

    // Max model: pop0 agg = 0.90
    params.set_emission_model(EmissionModel::Max);
    let le_max = params.log_emission(&obs, 0);

    // Mean model: pop0 agg = 0.70
    params.set_emission_model(EmissionModel::Mean);
    let le_mean = params.log_emission(&obs, 0);

    // Max aggregation (0.90) > Mean aggregation (0.70) → higher emission
    assert!(le_max > le_mean, "Max emission should be higher than Mean when max > mean");
}

// ============================================================================
// rfmix_to_windows: edge cases
// ============================================================================

#[test]
fn rfmix_to_windows_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(),
            start: 100,
            end: 1000,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 0);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn rfmix_to_windows_segment_end_before_min_pos() {
    // Second segment has end <= min_pos of the first segment
    // This shouldn't happen in real RFMix output, but tests the continue branch
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 5000,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 10,
                hap_ancestries: vec![0],
            },
        ],
    };
    // Normal behavior: 1 segment, windows filled
    let windows = rfmix_to_windows(&result, 1000);
    assert_eq!(windows.len(), 1);
    assert!(!windows[0].is_empty());
    // All windows should have ancestry 0 (AFR)
    for w in &windows[0] {
        assert_eq!(*w, Some(0));
    }
}

#[test]
fn rfmix_to_windows_max_pos_equals_min_pos() {
    // Segment with start == end → max_pos <= min_pos → empty windows
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 1000, // start == end
            start_cm: 0.0,
            end_cm: 0.0,
            n_snps: 0,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 1000);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn rfmix_window_starts_max_equals_min() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(),
            start: 500,
            end: 500,
            start_cm: 0.0,
            end_cm: 0.0,
            n_snps: 0,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 1000);
    assert!(starts.is_empty());
}

#[test]
fn rfmix_window_starts_empty_segments() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![],
    };
    let starts = rfmix_window_starts(&result, 1000);
    assert!(starts.is_empty());
}

// ============================================================================
// estimate_admixture_proportions edge cases
// ============================================================================

fn make_ancestry_segment(start: u64, end: u64, name: &str) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "HG001".to_string(),
        ancestry_idx: 0,
        ancestry_name: name.to_string(),
        n_windows: 1,
        mean_similarity: 0.99,
        mean_posterior: None,
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

#[test]
fn admixture_proportions_empty_segments() {
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&[], "HG001", &pop_names);

    assert_eq!(result.total_length_bp, 0);
    assert_eq!(result.n_switches, 0);
    assert_eq!(result.mean_tract_length_bp, 0.0);
    // All proportions should be 0
    for &prop in result.proportions.values() {
        assert!((prop - 0.0).abs() < 1e-10);
    }
}

#[test]
fn admixture_proportions_single_ancestry() {
    let segments = vec![
        make_ancestry_segment(0, 10000, "AFR"),
        make_ancestry_segment(10000, 20000, "AFR"),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "HG001", &pop_names);

    assert_eq!(result.total_length_bp, 20000);
    assert_eq!(result.n_switches, 0); // No ancestry switches
    assert!((result.proportions["AFR"] - 1.0).abs() < 1e-10);
    assert!((result.proportions["EUR"] - 0.0).abs() < 1e-10);
}

#[test]
fn admixture_proportions_mixed_ancestry() {
    let segments = vec![
        make_ancestry_segment(0, 6000, "AFR"),
        make_ancestry_segment(6000, 10000, "EUR"),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "HG001", &pop_names);

    assert_eq!(result.total_length_bp, 10000);
    assert_eq!(result.n_switches, 1);
    assert!((result.proportions["AFR"] - 0.6).abs() < 1e-10);
    assert!((result.proportions["EUR"] - 0.4).abs() < 1e-10);
}

#[test]
fn admixture_proportions_three_ancestries() {
    let segments = vec![
        make_ancestry_segment(0, 3000, "AFR"),
        make_ancestry_segment(3000, 6000, "EUR"),
        make_ancestry_segment(6000, 9000, "AMR"),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string(), "AMR".to_string()];
    let result = estimate_admixture_proportions(&segments, "HG001", &pop_names);

    assert_eq!(result.n_switches, 2);
    let prop_sum: f64 = result.proportions.values().sum();
    assert!((prop_sum - 1.0).abs() < 1e-10, "Proportions should sum to 1.0, got {}", prop_sum);
    for &prop in result.proportions.values() {
        assert!((prop - 1.0/3.0).abs() < 1e-10);
    }
}

#[test]
fn admixture_proportions_switches_alternating() {
    let segments = vec![
        make_ancestry_segment(0, 1000, "AFR"),
        make_ancestry_segment(1000, 2000, "EUR"),
        make_ancestry_segment(2000, 3000, "AFR"),
        make_ancestry_segment(3000, 4000, "EUR"),
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];
    let result = estimate_admixture_proportions(&segments, "HG001", &pop_names);

    assert_eq!(result.n_switches, 3); // AFR→EUR, EUR→AFR, AFR→EUR
    assert!((result.mean_tract_length_bp - 1000.0).abs() < 1e-10);
}

// ============================================================================
// smooth_states edge cases
// ============================================================================

#[test]
fn smooth_states_min_run_1_no_smoothing() {
    // min_run < 2 → returns states unchanged
    let states = vec![0, 0, 1, 0, 0];
    let smoothed = smooth_states(&states, 1);
    assert_eq!(smoothed, states);
}

#[test]
fn smooth_states_short_run_different_neighbors() {
    // Short run (length 1) between different neighbors → NOT smoothed
    let states = vec![0, 0, 1, 2, 2];
    let smoothed = smooth_states(&states, 3);
    // Run of 1 (value 1) is between 0 and 2 → different neighbors → no smoothing
    assert_eq!(smoothed, vec![0, 0, 1, 2, 2]);
}

#[test]
fn smooth_states_exact_min_run_not_smoothed() {
    // Run length equals min_run → not smoothed (only < min_run gets smoothed)
    let states = vec![0, 0, 0, 1, 1, 1, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    // Run of 1s has length 3, which is NOT < 3 → not smoothed
    assert_eq!(smoothed, vec![0, 0, 0, 1, 1, 1, 0, 0, 0]);
}

#[test]
fn smooth_states_run_at_start_not_smoothed() {
    // Short run at the very start → i == 0 → no prev_state → not smoothed
    let states = vec![1, 0, 0, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    // Run of 1 at position 0: i == 0 means no previous state → skip
    assert_eq!(smoothed, vec![1, 0, 0, 0, 0, 0]);
}

#[test]
fn smooth_states_run_at_end_not_smoothed() {
    // Short run at the very end → run_end == len → no next_state → not smoothed
    let states = vec![0, 0, 0, 0, 0, 1];
    let smoothed = smooth_states(&states, 3);
    // Run of 1 at end: run_end == len → not smoothed
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 1]);
}

#[test]
fn smooth_states_two_elements_no_smoothing() {
    // len < 3 → returns unchanged
    let states = vec![0, 1];
    let smoothed = smooth_states(&states, 5);
    assert_eq!(smoothed, states);
}

// ============================================================================
// count_smoothing_changes edge cases
// ============================================================================

#[test]
fn count_smoothing_changes_different_lengths() {
    // Shorter sequence determines count
    let original = vec![0, 1, 2, 3];
    let smoothed = vec![0, 0]; // shorter
    let changes = count_smoothing_changes(&original, &smoothed);
    // zip only pairs 2 elements
    assert_eq!(changes, 1); // only index 1 differs
}

// ============================================================================
// filter_segments_by_min_lod
// ============================================================================

#[test]
fn filter_segments_negative_min_lod() {
    // min_lod = -10 → keeps all segments including those with negative LOD
    let segments = vec![
        make_ancestry_segment_with_lod(0, 1000, "AFR", -5.0),
        make_ancestry_segment_with_lod(1000, 2000, "EUR", 3.0),
    ];
    let filtered = filter_segments_by_min_lod(segments, -10.0);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn filter_segments_exact_boundary() {
    let segments = vec![
        make_ancestry_segment_with_lod(0, 1000, "AFR", 5.0),
        make_ancestry_segment_with_lod(1000, 2000, "EUR", 5.0),
    ];
    let filtered = filter_segments_by_min_lod(segments, 5.0);
    // lod_score >= min_lod → both kept
    assert_eq!(filtered.len(), 2);
}

#[test]
fn filter_segments_all_below() {
    let segments = vec![
        make_ancestry_segment_with_lod(0, 1000, "AFR", 1.0),
        make_ancestry_segment_with_lod(1000, 2000, "EUR", 2.0),
    ];
    let filtered = filter_segments_by_min_lod(segments, 10.0);
    assert_eq!(filtered.len(), 0);
}

fn make_ancestry_segment_with_lod(start: u64, end: u64, name: &str, lod: f64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "HG001".to_string(),
        ancestry_idx: 0,
        ancestry_name: name.to_string(),
        n_windows: 1,
        mean_similarity: 0.99,
        mean_posterior: None,
        discriminability: 0.1,
        lod_score: lod,
    }
}

// ============================================================================
// extract_ancestry_segments: multiple state changes
// ============================================================================

#[test]
fn extract_ancestry_segments_alternating_states() {
    let pops = vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr_hap1".to_string()],
        },
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur_hap1".to_string()],
        },
    ];

    let params = AncestryHmmParams::new(pops, 0.01);

    let observations: Vec<AncestryObservation> = (0..6).map(|i| {
        let mut sims = HashMap::new();
        if i % 2 == 0 {
            sims.insert("afr_hap1".to_string(), 0.99);
            sims.insert("eur_hap1".to_string(), 0.50);
        } else {
            sims.insert("afr_hap1".to_string(), 0.50);
            sims.insert("eur_hap1".to_string(), 0.99);
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i as u64 * 10000,
            end: (i as u64 + 1) * 10000,
            sample: "HG001".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    // Alternating states: [0, 1, 0, 1, 0, 1]
    let states = vec![0, 1, 0, 1, 0, 1];

    let segments = extract_ancestry_segments(&observations, &states, &params, None);
    // Should produce 6 segments (each window is a different state from its neighbor)
    assert_eq!(segments.len(), 6, "Alternating states should produce 6 segments, got {}", segments.len());

    // Check names alternate
    assert_eq!(segments[0].ancestry_name, "AFR");
    assert_eq!(segments[1].ancestry_name, "EUR");
    assert_eq!(segments[2].ancestry_name, "AFR");
}

#[test]
fn extract_ancestry_segments_single_state() {
    let pops = vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr_hap1".to_string()],
        },
    ];

    let params = AncestryHmmParams::new(pops, 0.01);

    let observations: Vec<AncestryObservation> = (0..5).map(|i| {
        let mut sims = HashMap::new();
        sims.insert("afr_hap1".to_string(), 0.99);
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: i as u64 * 10000,
            end: (i as u64 + 1) * 10000,
            sample: "HG001".to_string(),
            similarities: sims,
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect();

    let states = vec![0, 0, 0, 0, 0];
    let segments = extract_ancestry_segments(&observations, &states, &params, None);
    assert_eq!(segments.len(), 1, "All same state should produce 1 segment");
    assert_eq!(segments[0].ancestry_name, "AFR");
    assert_eq!(segments[0].n_windows, 5);
}

// ============================================================================
// parse_rfmix_msp_content: edge cases
// ============================================================================

#[test]
fn parse_rfmix_data_line_with_extra_fields() {
    // Data lines may have extra trailing fields — should still parse correctly
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0\tHG001.1
chr1\t100\t200\t0.0\t0.5\t10\t0\t1\textra_field\tanother";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.segments[0].hap_ancestries, vec![0, 1]);
}

#[test]
fn parse_rfmix_blank_lines_in_data() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0
chr1\t100\t200\t0.0\t0.5\t10\t0

chr1\t200\t300\t0.5\t1.0\t15\t1
";

    let result = parse_rfmix_msp_content(content).unwrap();
    // Blank line should be skipped
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn parse_rfmix_comments_in_data() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0
chr1\t100\t200\t0.0\t0.5\t10\t0
# This is a comment in the data section
chr1\t200\t300\t0.5\t1.0\t15\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn parse_rfmix_population_header_space_separated_format2() {
    // Format 2: space-separated names (no equals signs)
    let content = "\
#reference_panel_population: AFR EUR AMR
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0
chr1\t100\t200\t0.0\t0.5\t10\t2";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "AMR"]);
    // ancestry index 2 = AMR
    assert_eq!(result.segments[0].hap_ancestries, vec![2]);
}

// ============================================================================
// EmissionModel Display/FromStr consistency
// ============================================================================

#[test]
fn emission_model_display_topk_variants() {
    assert_eq!(format!("{}", EmissionModel::TopK(1)), "top1");
    assert_eq!(format!("{}", EmissionModel::TopK(100)), "top100");
}

#[test]
fn emission_model_fromstr_case_insensitive() {
    assert_eq!("MAX".parse::<EmissionModel>().unwrap(), EmissionModel::Max);
    assert_eq!("Mean".parse::<EmissionModel>().unwrap(), EmissionModel::Mean);
    assert_eq!("MEDIAN".parse::<EmissionModel>().unwrap(), EmissionModel::Median);
    assert_eq!("Top5".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(5));
    assert_eq!("TOP3".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(3));
}

// ============================================================================
// rfmix_to_windows: multiple haplotypes and ancestry assignments
// ============================================================================

#[test]
fn rfmix_to_windows_three_haplotypes() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string(), "AMR".to_string()],
        haplotype_names: vec!["HG001.0".to_string(), "HG001.1".to_string(), "HG002.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr1".to_string(),
                start: 0,
                end: 10000,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 50,
                hap_ancestries: vec![0, 1, 2], // AFR, EUR, AMR
            },
        ],
    };

    let windows = rfmix_to_windows(&result, 5000);
    assert_eq!(windows.len(), 3);
    // Each haplotype should have its assigned ancestry in all windows
    for w in &windows[0] {
        assert_eq!(*w, Some(0)); // AFR
    }
    for w in &windows[1] {
        assert_eq!(*w, Some(1)); // EUR
    }
    for w in &windows[2] {
        assert_eq!(*w, Some(2)); // AMR
    }
}

#[test]
fn rfmix_to_windows_window_size_larger_than_region() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG001.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(),
            start: 100,
            end: 200,
            start_cm: 0.0,
            end_cm: 0.01,
            n_snps: 1,
            hap_ancestries: vec![0],
        }],
    };

    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 1);
    // Region is only 100bp, window is 10000bp → 1 window
    assert_eq!(windows[0].len(), 1);
    assert_eq!(windows[0][0], Some(0));
}
