//! Edge case tests for ancestry LOD scores and admixture proportion estimation.
//!
//! These tests complement the unit tests in ancestry.rs by covering
//! additional edge cases and integration scenarios.

use hprc_ancestry_cli::ancestry::{
    compute_per_window_ancestry_lod, count_smoothing_changes, estimate_admixture_proportions,
    extract_ancestry_segments, filter_segments_by_min_lod, segment_ancestry_lod, smooth_states,
    AncestrySegment,
};
use hprc_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: create a simple 2-population HMM params
// ---------------------------------------------------------------------------

fn two_pop_params(temperature: f64, switch_prob: f64) -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, switch_prob);
    params.set_temperature(temperature);
    params
}

fn three_pop_params(temperature: f64, switch_prob: f64) -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
        AncestralPopulation {
            name: "pop_c".to_string(),
            haplotypes: vec!["hap_c1".to_string(), "hap_c2".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, switch_prob);
    params.set_temperature(temperature);
    params
}

fn make_obs(sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "test_sample".to_string(),
        similarities: sims
            .iter()
            .map(|(name, val)| (name.to_string(), *val))
            .collect::<HashMap<String, f64>>(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ---------------------------------------------------------------------------
// 1. LOD score edge cases
// ---------------------------------------------------------------------------

/// LOD with nearly identical similarities across populations should be near zero.
#[test]
fn test_lod_near_zero_for_similar_populations() {
    let params = two_pop_params(0.03, 0.01);

    // Both populations have nearly equal similarity
    let obs = make_obs(&[
        ("hap_a1", 0.995),
        ("hap_a2", 0.995),
        ("hap_b1", 0.995),
        ("hap_b2", 0.995),
    ]);

    let lod_0 = compute_per_window_ancestry_lod(&obs, &params, 0);
    let lod_1 = compute_per_window_ancestry_lod(&obs, &params, 1);

    // With identical similarities, LOD should be approximately 0
    assert!(
        lod_0.abs() < 0.1,
        "LOD for pop_a should be near 0 with identical sims, got {}",
        lod_0
    );
    assert!(
        lod_1.abs() < 0.1,
        "LOD for pop_b should be near 0 with identical sims, got {}",
        lod_1
    );
}

/// LOD should be strongly positive when assigned state is clearly correct.
#[test]
fn test_lod_strongly_positive_for_clear_assignment() {
    let params = two_pop_params(0.03, 0.01);

    let obs = make_obs(&[
        ("hap_a1", 0.999),
        ("hap_a2", 0.998),
        ("hap_b1", 0.500),
        ("hap_b2", 0.510),
    ]);

    let lod_a = compute_per_window_ancestry_lod(&obs, &params, 0);
    let lod_b = compute_per_window_ancestry_lod(&obs, &params, 1);

    assert!(
        lod_a > 1.0,
        "LOD for correct assignment (pop_a) should be strongly positive, got {}",
        lod_a
    );
    assert!(
        lod_b < -1.0,
        "LOD for wrong assignment (pop_b) should be strongly negative, got {}",
        lod_b
    );
}

/// LOD should be symmetric: LOD(correct) = -LOD(wrong) for 2 populations.
#[test]
fn test_lod_symmetry_two_populations() {
    let params = two_pop_params(0.03, 0.01);

    let obs = make_obs(&[
        ("hap_a1", 0.990),
        ("hap_a2", 0.985),
        ("hap_b1", 0.950),
        ("hap_b2", 0.940),
    ]);

    let lod_a = compute_per_window_ancestry_lod(&obs, &params, 0);
    let lod_b = compute_per_window_ancestry_lod(&obs, &params, 1);

    // With 2 states, LOD(0) + LOD(1) should be approximately 0
    // because log10(P(a)/P(b)) + log10(P(b)/P(a)) = 0
    assert!(
        (lod_a + lod_b).abs() < 1e-6,
        "LOD(a) + LOD(b) should be ~0, got {} + {} = {}",
        lod_a,
        lod_b,
        lod_a + lod_b
    );
}

/// Segment LOD should accumulate correctly across a multi-window segment.
#[test]
fn test_segment_lod_accumulation_multi_window() {
    let params = three_pop_params(0.03, 0.01);

    let observations: Vec<AncestryObservation> = (0..10)
        .map(|i| {
            let boost = if i < 5 { 0.05 } else { 0.02 };
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: i as u64 * 5000,
                end: (i as u64 + 1) * 5000,
                sample: "test".to_string(),
                similarities: [
                    ("hap_a1".to_string(), 0.95 + boost),
                    ("hap_a2".to_string(), 0.94 + boost),
                    ("hap_b1".to_string(), 0.90),
                    ("hap_b2".to_string(), 0.89),
                    ("hap_c1".to_string(), 0.85),
                    ("hap_c2".to_string(), 0.84),
                ].into_iter().collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
            }
        })
        .collect();

    let lod_full = segment_ancestry_lod(&observations, &params, 0, 0, 9);
    let lod_first_half = segment_ancestry_lod(&observations, &params, 0, 0, 4);
    let lod_second_half = segment_ancestry_lod(&observations, &params, 0, 5, 9);

    // Segment LOD should be additive
    assert!(
        (lod_full - (lod_first_half + lod_second_half)).abs() < 1e-10,
        "Segment LOD should be additive: {} != {} + {}",
        lod_full,
        lod_first_half,
        lod_second_half
    );

    // First half should have higher per-window LOD (larger boost)
    let avg_first = lod_first_half / 5.0;
    let avg_second = lod_second_half / 5.0;
    assert!(
        avg_first > avg_second,
        "First half avg LOD ({}) should exceed second half ({})",
        avg_first,
        avg_second
    );
}

/// LOD with a single-window segment should match per-window LOD.
#[test]
fn test_segment_lod_single_window() {
    let params = two_pop_params(0.03, 0.01);

    let obs = vec![make_obs(&[
        ("hap_a1", 0.990),
        ("hap_a2", 0.985),
        ("hap_b1", 0.950),
        ("hap_b2", 0.940),
    ])];

    let per_window = compute_per_window_ancestry_lod(&obs[0], &params, 0);
    let segment = segment_ancestry_lod(&obs, &params, 0, 0, 0);

    assert!(
        (per_window - segment).abs() < 1e-10,
        "Single-window segment LOD ({}) should match per-window LOD ({})",
        segment,
        per_window
    );
}

// ---------------------------------------------------------------------------
// 2. Admixture proportion edge cases
// ---------------------------------------------------------------------------

fn make_segment(
    chrom: &str,
    start: u64,
    end: u64,
    ancestry_idx: usize,
    ancestry_name: &str,
) -> AncestrySegment {
    AncestrySegment {
        chrom: chrom.to_string(),
        start,
        end,
        sample: "test_sample".to_string(),
        ancestry_idx,
        ancestry_name: ancestry_name.to_string(),
        n_windows: ((end - start) / 5000) as usize,
        mean_similarity: 0.99,
        mean_posterior: Some(0.95),
        discriminability: 0.05,
        lod_score: 5.0,
    }
}

/// Adjacent segments with the same ancestry should NOT increment n_switches.
#[test]
fn test_admixture_no_switch_same_ancestry() {
    let segments = vec![
        make_segment("chr1", 0, 50000, 0, "pop_a"),
        make_segment("chr1", 50000, 100000, 0, "pop_a"),
        make_segment("chr1", 100000, 150000, 0, "pop_a"),
    ];

    let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    assert_eq!(props.n_switches, 0, "Same-ancestry adjacent segments should have 0 switches");
    assert!(
        (props.proportions["pop_a"] - 1.0).abs() < 1e-10,
        "All pop_a segments should give 100% proportion"
    );
    assert!(
        (props.proportions["pop_b"] - 0.0).abs() < 1e-10,
        "pop_b should be 0%"
    );
}

/// Proportions should sum to exactly 1.0 (within floating point tolerance).
#[test]
fn test_admixture_proportions_sum_to_one() {
    let segments = vec![
        make_segment("chr1", 0, 30000, 0, "pop_a"),
        make_segment("chr1", 30000, 70000, 1, "pop_b"),
        make_segment("chr1", 70000, 100000, 2, "pop_c"),
    ];

    let pop_names = vec![
        "pop_a".to_string(),
        "pop_b".to_string(),
        "pop_c".to_string(),
    ];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    let sum: f64 = props.proportions.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Proportions should sum to 1.0, got {}",
        sum
    );
}

/// Multiple switches in quick succession should be counted correctly.
#[test]
fn test_admixture_rapid_switches() {
    let segments = vec![
        make_segment("chr1", 0, 10000, 0, "pop_a"),
        make_segment("chr1", 10000, 20000, 1, "pop_b"),
        make_segment("chr1", 20000, 30000, 0, "pop_a"),
        make_segment("chr1", 30000, 40000, 1, "pop_b"),
        make_segment("chr1", 40000, 50000, 0, "pop_a"),
    ];

    let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    assert_eq!(props.n_switches, 4, "5 alternating segments should have 4 switches");
    assert!(
        (props.proportions["pop_a"] - 0.6).abs() < 1e-10,
        "pop_a should be 60%, got {}",
        props.proportions["pop_a"]
    );
    assert!(
        (props.proportions["pop_b"] - 0.4).abs() < 1e-10,
        "pop_b should be 40%, got {}",
        props.proportions["pop_b"]
    );
    assert_eq!(
        props.mean_tract_length_bp, 10000.0,
        "Mean tract length should be 10kb"
    );
}

/// Admixture with a population that appears in pop_names but not in segments
/// should get 0 proportion.
#[test]
fn test_admixture_absent_population() {
    let segments = vec![
        make_segment("chr1", 0, 50000, 0, "pop_a"),
        make_segment("chr1", 50000, 100000, 1, "pop_b"),
    ];

    let pop_names = vec![
        "pop_a".to_string(),
        "pop_b".to_string(),
        "pop_c".to_string(), // not in segments
    ];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    assert!(
        (props.proportions["pop_c"] - 0.0).abs() < 1e-10,
        "Absent population should have 0 proportion"
    );
    // Sum should still be 1.0
    let sum: f64 = props.proportions.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Proportions should sum to 1.0 even with absent population"
    );
}

/// Total length should be the sum of all segment lengths.
#[test]
fn test_admixture_total_length() {
    let segments = vec![
        make_segment("chr1", 100, 50100, 0, "pop_a"),
        make_segment("chr1", 50100, 70100, 1, "pop_b"),
        make_segment("chr2", 0, 30000, 0, "pop_a"),
    ];

    let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    let expected_total = 50000u64 + 20000 + 30000;
    assert_eq!(
        props.total_length_bp, expected_total,
        "Total length should be sum of segment lengths"
    );
    assert_eq!(
        props.tract_lengths_bp["pop_a"],
        50000 + 30000,
        "pop_a tract length should be 80000"
    );
    assert_eq!(
        props.tract_lengths_bp["pop_b"], 20000,
        "pop_b tract length should be 20000"
    );
}

/// Admixture with a single very long tract should have mean_tract_length = total_length.
#[test]
fn test_admixture_single_long_tract() {
    let segments = vec![make_segment("chr1", 0, 10_000_000, 0, "pop_a")];

    let pop_names = vec!["pop_a".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    assert_eq!(props.n_switches, 0);
    assert_eq!(props.total_length_bp, 10_000_000);
    assert_eq!(props.mean_tract_length_bp, 10_000_000.0);
    assert!(
        (props.proportions["pop_a"] - 1.0).abs() < 1e-10,
        "Single tract should be 100%"
    );
}

// ---------------------------------------------------------------------------
// 3. LOD + admixture integration
// ---------------------------------------------------------------------------

/// Segments with high LOD should correspond to confident admixture proportions.
#[test]
fn test_high_lod_segments_produce_valid_admixture() {
    let segments = vec![
        AncestrySegment {
            chrom: "chr1".to_string(),
            start: 0,
            end: 100000,
            sample: "test".to_string(),
            ancestry_idx: 0,
            ancestry_name: "pop_a".to_string(),
            n_windows: 20,
            mean_similarity: 0.99,
            mean_posterior: Some(0.99),
            discriminability: 0.10,
            lod_score: 25.0,
        },
        AncestrySegment {
            chrom: "chr1".to_string(),
            start: 100000,
            end: 200000,
            sample: "test".to_string(),
            ancestry_idx: 1,
            ancestry_name: "pop_b".to_string(),
            n_windows: 20,
            mean_similarity: 0.98,
            mean_posterior: Some(0.95),
            discriminability: 0.08,
            lod_score: 18.0,
        },
    ];

    let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    assert_eq!(props.n_switches, 1);
    assert!(
        (props.proportions["pop_a"] - 0.5).abs() < 1e-10,
        "Should be 50% pop_a"
    );
    assert!(
        (props.proportions["pop_b"] - 0.5).abs() < 1e-10,
        "Should be 50% pop_b"
    );
}

// ---------------------------------------------------------------------------
// 4. smooth_states edge cases
// ---------------------------------------------------------------------------

/// All elements the same state — smoothing should be a no-op.
#[test]
fn test_smooth_states_all_same_state() {
    let states = vec![2, 2, 2, 2, 2, 2, 2, 2];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states, "All-same input should be unchanged");
}

/// Three states with alternating short runs between the same state.
#[test]
fn test_smooth_states_three_states_sandwich() {
    // State 0 sandwiched between state 1s, but state 2 at the end
    let states = vec![1, 1, 1, 0, 1, 1, 1, 2, 2, 2];
    let smoothed = smooth_states(&states, 3);
    // The single 0 between 1s should be smoothed to 1
    assert_eq!(smoothed[3], 1, "Short run of 0 between 1s should become 1");
    // The 2s at the end should be unchanged
    assert_eq!(smoothed[7], 2);
    assert_eq!(smoothed[8], 2);
    assert_eq!(smoothed[9], 2);
}

/// Short run at the start should NOT be smoothed (no left neighbor).
#[test]
fn test_smooth_states_short_run_at_start() {
    let states = vec![1, 0, 0, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    // Single 1 at start has no left neighbor, so it shouldn't be smoothed
    assert_eq!(smoothed[0], 1, "Short run at start should be preserved");
}

/// Short run at the end should NOT be smoothed (no right neighbor).
#[test]
fn test_smooth_states_short_run_at_end() {
    let states = vec![0, 0, 0, 0, 0, 1];
    let smoothed = smooth_states(&states, 3);
    // Single 1 at end has no right neighbor, so it shouldn't be smoothed
    assert_eq!(smoothed[5], 1, "Short run at end should be preserved");
}

/// count_smoothing_changes with mismatched lengths uses zip (shorter length).
#[test]
fn test_count_smoothing_changes_different_lengths() {
    let original = vec![0, 1, 2, 3];
    let smoothed = vec![0, 1]; // shorter
    // zip truncates, so only 2 elements compared, both same => 0 changes
    assert_eq!(count_smoothing_changes(&original, &smoothed), 0);
}

/// count_smoothing_changes with all different.
#[test]
fn test_count_smoothing_changes_all_different() {
    let original = vec![0, 1, 2, 0, 1];
    let smoothed = vec![1, 0, 1, 2, 0];
    assert_eq!(count_smoothing_changes(&original, &smoothed), 5);
}

// ---------------------------------------------------------------------------
// 5. extract_ancestry_segments with 3 populations
// ---------------------------------------------------------------------------

/// extract_ancestry_segments should produce correct segments with 3 populations.
#[test]
fn test_extract_segments_three_populations() {
    let pops = vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string(), "eur2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string(), "afr2".to_string()],
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: vec!["amr1".to_string(), "amr2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.01);

    let obs: Vec<AncestryObservation> = (0..6)
        .map(|i| AncestryObservation {
            chrom: "chr10".to_string(),
            start: i as u64 * 5000,
            end: (i as u64 + 1) * 5000,
            sample: "HG00733".to_string(),
            similarities: [
                ("eur1".to_string(), if i < 2 { 0.99 } else { 0.80 }),
                ("eur2".to_string(), if i < 2 { 0.98 } else { 0.79 }),
                ("afr1".to_string(), if i >= 2 && i < 4 { 0.99 } else { 0.80 }),
                ("afr2".to_string(), if i >= 2 && i < 4 { 0.98 } else { 0.79 }),
                ("amr1".to_string(), if i >= 4 { 0.99 } else { 0.80 }),
                ("amr2".to_string(), if i >= 4 { 0.98 } else { 0.79 }),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    // States: EUR EUR AFR AFR AMR AMR
    let states = vec![0, 0, 1, 1, 2, 2];
    let segments = extract_ancestry_segments(&obs, &states, &params, None);

    assert_eq!(segments.len(), 3, "Should have 3 segments for 3 runs");
    assert_eq!(segments[0].ancestry_name, "EUR");
    assert_eq!(segments[0].n_windows, 2);
    assert_eq!(segments[1].ancestry_name, "AFR");
    assert_eq!(segments[1].n_windows, 2);
    assert_eq!(segments[2].ancestry_name, "AMR");
    assert_eq!(segments[2].n_windows, 2);
}

// ---------------------------------------------------------------------------
// 6. filter_segments_by_min_lod edge cases
// ---------------------------------------------------------------------------

/// filter_segments_by_min_lod with negative LOD threshold keeps negative segments.
#[test]
fn test_filter_segments_negative_threshold() {
    let segments = vec![
        make_segment("chr1", 0, 50000, 0, "pop_a"),
        AncestrySegment {
            lod_score: -1.0,
            ..make_segment("chr1", 50000, 100000, 1, "pop_b")
        },
    ];

    let filtered = filter_segments_by_min_lod(segments, -2.0);
    assert_eq!(filtered.len(), 2, "Both segments should pass threshold of -2.0");
}

/// filter_segments_by_min_lod with exact threshold boundary.
#[test]
fn test_filter_segments_exact_boundary() {
    let segments = vec![AncestrySegment {
        lod_score: 5.0,
        ..make_segment("chr1", 0, 50000, 0, "pop_a")
    }];

    // Exact match: lod_score >= min_lod
    let filtered = filter_segments_by_min_lod(segments, 5.0);
    assert_eq!(filtered.len(), 1, "Segment with LOD exactly at threshold should be kept");
}

// ---------------------------------------------------------------------------
// 7. extract_ancestry_segments edge cases
// ---------------------------------------------------------------------------

/// extract_ancestry_segments with empty observations should return empty.
#[test]
fn test_extract_segments_empty_observations() {
    let params = two_pop_params(0.03, 0.01);
    let segments = extract_ancestry_segments(&[], &[], &params, None);
    assert!(segments.is_empty(), "Empty input should give empty output");
}

/// extract_ancestry_segments with all same state should return 1 segment.
#[test]
fn test_extract_segments_all_same_state() {
    let params = two_pop_params(0.03, 0.01);

    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i as u64 * 5000,
            end: (i as u64 + 1) * 5000,
            sample: "test_sample".to_string(),
            similarities: [
                ("hap_a1".to_string(), 0.99),
                ("hap_a2".to_string(), 0.98),
                ("hap_b1".to_string(), 0.80),
                ("hap_b2".to_string(), 0.79),
            ]
            .into_iter()
            .collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();

    let states = vec![0, 0, 0, 0, 0]; // All state 0
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1, "All-same-state should produce 1 segment");
    assert_eq!(segments[0].ancestry_name, "pop_a");
    assert_eq!(segments[0].n_windows, 5);
    assert_eq!(segments[0].start, 0);
    assert_eq!(segments[0].end, 25000);
}

/// extract_ancestry_segments with single observation.
#[test]
fn test_extract_segments_single_observation() {
    let params = two_pop_params(0.03, 0.01);

    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "test_sample".to_string(),
        similarities: [
            ("hap_a1".to_string(), 0.99),
            ("hap_a2".to_string(), 0.98),
            ("hap_b1".to_string(), 0.80),
            ("hap_b2".to_string(), 0.79),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];

    let states = vec![1];
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].ancestry_name, "pop_b");
    assert_eq!(segments[0].n_windows, 1);
}

// ---------------------------------------------------------------------------
// 8. estimate_admixture_proportions edge cases
// ---------------------------------------------------------------------------

/// Empty segments should return 0 proportions and 0 total length.
#[test]
fn test_admixture_empty_segments() {
    let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
    let props = estimate_admixture_proportions(&[], "test", &pop_names);

    assert_eq!(props.total_length_bp, 0);
    assert_eq!(props.n_switches, 0);
    assert_eq!(props.mean_tract_length_bp, 0.0);
    // Proportions should exist but be 0
    assert!(
        (props.proportions["pop_a"] - 0.0).abs() < 1e-10,
        "Empty segments should give 0 proportion"
    );
    assert!(
        (props.proportions["pop_b"] - 0.0).abs() < 1e-10,
        "Empty segments should give 0 proportion"
    );
}

/// Single segment with start == end should have 0 length.
#[test]
fn test_admixture_zero_length_segment() {
    let segments = vec![make_segment("chr1", 5000, 5000, 0, "pop_a")];
    let pop_names = vec!["pop_a".to_string()];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);

    // start == end → saturating_sub gives 0
    assert_eq!(props.total_length_bp, 0);
    assert_eq!(props.mean_tract_length_bp, 0.0);
}

// ---------------------------------------------------------------------------
// 9. smooth_states additional edge cases
// ---------------------------------------------------------------------------

/// smooth_states with min_run larger than input length.
#[test]
fn test_smooth_states_min_run_larger_than_length() {
    let states = vec![0, 1, 0];
    let smoothed = smooth_states(&states, 100);
    // The middle run (length 1) is shorter than min_run (100),
    // and it's between two 0s → should be smoothed to 0
    assert_eq!(smoothed, vec![0, 0, 0]);
}

/// smooth_states with exactly min_run=2 (minimum effective value).
#[test]
fn test_smooth_states_min_run_two() {
    let states = vec![0, 0, 1, 0, 0];
    let smoothed = smooth_states(&states, 2);
    // Single 1 between 0s: run length 1 < min_run 2, and left==right → smooth
    assert_eq!(smoothed[2], 0, "Single-window run should be smoothed with min_run=2");
}

/// smooth_states with min_run=1 is a no-op.
#[test]
fn test_smooth_states_min_run_one_is_noop() {
    let states = vec![0, 1, 0, 1, 0];
    let smoothed = smooth_states(&states, 1);
    assert_eq!(smoothed, states, "min_run=1 should be a no-op");
}

/// smooth_states with length exactly 2 should be a no-op (length < 3 guard).
#[test]
fn test_smooth_states_length_two_noop() {
    let states = vec![0, 1];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states, "Length < 3 should be a no-op");
}

/// smooth_states with different surrounding states should NOT smooth.
#[test]
fn test_smooth_states_different_neighbors_no_smoothing() {
    // Short run of 1 between 0 and 2 → neighbors differ → no smoothing
    let states = vec![0, 0, 0, 1, 2, 2, 2];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed[3], 1, "Short run between different states should be preserved");
}

// ---------------------------------------------------------------------------
// 10. CrossValidationResult methods
// ---------------------------------------------------------------------------

use hprc_ancestry_cli::CrossValidationResult;

#[test]
fn test_cross_validation_result_confusion_matrix_tsv_basic() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("EUR".to_string(), 0.9);
    accuracy_per_pop.insert("AFR".to_string(), 0.8);

    let mut n_windows_per_pop = HashMap::new();
    n_windows_per_pop.insert("EUR".to_string(), 100);
    n_windows_per_pop.insert("AFR".to_string(), 50);

    let mut confusion = HashMap::new();
    confusion.insert(("EUR".to_string(), "EUR".to_string()), 90);
    confusion.insert(("EUR".to_string(), "AFR".to_string()), 10);
    confusion.insert(("AFR".to_string(), "AFR".to_string()), 40);
    confusion.insert(("AFR".to_string(), "EUR".to_string()), 10);

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.867,
        n_windows_per_pop,
        confusion,
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    let tsv = result.confusion_matrix_tsv();

    // Should contain header and data rows
    assert!(tsv.contains("true_pop\tpred_pop\tcount"), "TSV should have header");
    assert!(tsv.contains("EUR\tEUR\t90"), "TSV should have EUR→EUR=90");
    assert!(tsv.contains("EUR\tAFR\t10"), "TSV should have EUR→AFR=10");
    assert!(tsv.contains("AFR\tAFR\t40"), "TSV should have AFR→AFR=40");
    assert!(tsv.contains("AFR\tEUR\t10"), "TSV should have AFR→EUR=10");
}

#[test]
fn test_cross_validation_result_confusion_matrix_tsv_empty() {
    let result = CrossValidationResult {
        accuracy_per_pop: HashMap::new(),
        overall_accuracy: 0.0,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    let tsv = result.confusion_matrix_tsv();
    // Should only have the header
    assert!(tsv.contains("true_pop\tpred_pop\tcount"));
    // No data rows (other than the newline after header)
    let lines: Vec<&str> = tsv.lines().collect();
    assert_eq!(lines.len(), 1, "Empty confusion matrix should only have header");
}

#[test]
fn test_cross_validation_result_has_bias_no_bias() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("EUR".to_string(), 0.95);
    accuracy_per_pop.insert("AFR".to_string(), 0.80);

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.9,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(!result.has_bias(), "All accuracies >= 50% → no bias");
}

#[test]
fn test_cross_validation_result_has_bias_with_bias() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("EUR".to_string(), 0.95);
    accuracy_per_pop.insert("AFR".to_string(), 0.40); // below 50%

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.7,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(result.has_bias(), "AFR accuracy < 50% → bias detected");
}

#[test]
fn test_cross_validation_result_has_bias_boundary() {
    let mut accuracy_per_pop = HashMap::new();
    accuracy_per_pop.insert("EUR".to_string(), 0.50); // exactly 50%

    let result = CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy: 0.5,
        n_windows_per_pop: HashMap::new(),
        confusion: HashMap::new(),
        precision_per_pop: HashMap::new(),
        recall_per_pop: HashMap::new(),
        f1_per_pop: HashMap::new(),
    };

    assert!(!result.has_bias(), "Exactly 50% should not count as bias (< 0.5 required)");
}

// ---------------------------------------------------------------------------
// 11. AncestryHmmParams setter methods
// ---------------------------------------------------------------------------

#[test]
fn test_set_temperature() {
    let mut params = two_pop_params(0.03, 0.01);
    assert_eq!(params.emission_std, 0.03);

    params.set_temperature(0.05);
    assert_eq!(params.emission_std, 0.05);

    params.set_temperature(0.001);
    assert_eq!(params.emission_std, 0.001);
}

#[test]
fn test_set_switch_prob() {
    let mut params = two_pop_params(0.03, 0.01);

    params.set_switch_prob(0.02);

    // Check transition matrix
    let stay = 1.0 - 0.02;
    let switch_each = 0.02 / (params.n_states - 1) as f64;

    for i in 0..params.n_states {
        for j in 0..params.n_states {
            if i == j {
                assert!((params.transitions[i][j] - stay).abs() < 1e-10,
                    "Stay prob should be {}, got {}", stay, params.transitions[i][j]);
            } else {
                assert!((params.transitions[i][j] - switch_each).abs() < 1e-10,
                    "Switch prob should be {}, got {}", switch_each, params.transitions[i][j]);
            }
        }
    }
}

#[test]
fn test_set_switch_prob_three_states() {
    let mut params = three_pop_params(0.03, 0.01);

    params.set_switch_prob(0.06);

    let stay = 1.0 - 0.06;
    let switch_each = 0.06 / 2.0; // 3 states, switch divided among 2 others

    for i in 0..3 {
        let row_sum: f64 = params.transitions[i].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10, "Row {} should sum to 1.0, got {}", i, row_sum);

        assert!((params.transitions[i][i] - stay).abs() < 1e-10);
        for j in 0..3 {
            if i != j {
                assert!((params.transitions[i][j] - switch_each).abs() < 1e-10);
            }
        }
    }
}
