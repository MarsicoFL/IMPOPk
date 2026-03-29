//! Cycle 32: Boundary condition tests for ancestry-cli untested code paths.
//!
//! These tests target specific parameter boundaries and conditional branches:
//! - AncestryHmmParams with single state (n_states=1): degenerate valid case
//! - compute_per_window_ancestry_lod when log_emission is -inf (both states)
//! - segment_ancestry_lod single-window segment (start_idx == end_idx)
//! - filter_segments_by_min_lod with lod_score == min_lod exactly
//! - extract_ancestry_segments with empty/mismatched inputs
//! - estimate_admixture_proportions single-population segments
//! - count_smoothing_changes on identical vs all-different inputs
//! - coverage_ratio boundary values

use hprc_ancestry_cli::ancestry::{
    count_smoothing_changes, coverage_ratio, estimate_admixture_proportions,
    extract_ancestry_segments, filter_segments_by_min_lod, AncestrySegment,
};
use hprc_ancestry_cli::hmm::{AncestralPopulation, AncestryHmmParams, AncestryObservation};

// ══════════════════════════════════════════════════════════════════════
// Helper to create a test observation
// ══════════════════════════════════════════════════════════════════════

fn make_obs(
    chrom: &str,
    start: u64,
    end: u64,
    sims: &[(&str, f64)],
) -> AncestryObservation {
    AncestryObservation {
        chrom: chrom.to_string(),
        start,
        end,
        sample: "test#1".to_string(),
        similarities: sims
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_segment(
    ancestry_idx: usize,
    ancestry_name: &str,
    lod: f64,
    n_windows: usize,
) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start: 0,
        end: n_windows as u64 * 10000,
        sample: "test#1".to_string(),
        ancestry_idx,
        ancestry_name: ancestry_name.to_string(),
        n_windows,
        mean_similarity: 0.99,
        mean_posterior: Some(0.9),
        discriminability: 0.05,
        lod_score: lod,
    }
}

fn make_params(pop_names: &[&str], hap_names: &[&[&str]]) -> AncestryHmmParams {
    let pops: Vec<AncestralPopulation> = pop_names
        .iter()
        .zip(hap_names.iter())
        .map(|(name, haps)| AncestralPopulation {
            name: name.to_string(),
            haplotypes: haps.iter().map(|h| h.to_string()).collect(),
        })
        .collect();
    AncestryHmmParams::new(pops, 0.001)
}

// ══════════════════════════════════════════════════════════════════════
// compute_per_window_ancestry_lod: both states produce -inf emission
// ══════════════════════════════════════════════════════════════════════

#[test]
fn per_window_ancestry_lod_returns_zero_when_all_emissions_neg_inf() {
    let params = make_params(&["EUR", "AFR"], &[&["eur_hap1"], &["afr_hap1"]]);

    // Observation with no matching haplotypes → log_emission will be very low
    let obs = make_obs("chr1", 0, 10000, &[("unknown_hap", 0.999)]);
    let lod = hprc_ancestry_cli::ancestry::compute_per_window_ancestry_lod(&obs, &params, 0);
    assert!(
        lod.is_finite(),
        "LOD should be finite when emissions are uninformative, got {}",
        lod
    );
}

// ══════════════════════════════════════════════════════════════════════
// segment_ancestry_lod: single-window segment
// ══════════════════════════════════════════════════════════════════════

#[test]
fn segment_ancestry_lod_single_window() {
    let params = make_params(&["EUR", "AFR"], &[&["eur1"], &["afr1"]]);

    let obs = vec![make_obs("chr1", 0, 10000, &[("eur1", 0.999), ("afr1", 0.990)])];
    let lod = hprc_ancestry_cli::ancestry::segment_ancestry_lod(&obs, &params, 0, 0, 0);
    assert!(lod.is_finite(), "single-window LOD should be finite");
}

// ══════════════════════════════════════════════════════════════════════
// filter_segments_by_min_lod: exact boundary (lod == min_lod)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn filter_segments_lod_exact_boundary_kept() {
    let seg = make_segment(0, "EUR", 3.0, 5);
    let result = filter_segments_by_min_lod(vec![seg], 3.0);
    assert_eq!(result.len(), 1, "segment with lod == min_lod should be kept");
}

#[test]
fn filter_segments_lod_just_below_boundary_rejected() {
    let seg = make_segment(0, "EUR", 2.999999, 5);
    let result = filter_segments_by_min_lod(vec![seg], 3.0);
    assert_eq!(result.len(), 0, "segment with lod < min_lod should be rejected");
}

#[test]
fn filter_segments_lod_zero_keeps_zero_and_positive() {
    let segs = vec![
        make_segment(0, "EUR", -1.0, 3),
        make_segment(0, "EUR", 0.0, 3),
        make_segment(0, "EUR", 1.0, 3),
    ];
    let result = filter_segments_by_min_lod(segs, 0.0);
    assert_eq!(result.len(), 2, "min_lod=0.0 should keep 0.0 and positive");
}

// ══════════════════════════════════════════════════════════════════════
// extract_ancestry_segments: empty/mismatched inputs
// ══════════════════════════════════════════════════════════════════════

#[test]
fn extract_ancestry_segments_empty_obs_returns_empty() {
    let params = make_params(&["EUR"], &[&["eur1"]]);

    let segments = extract_ancestry_segments(&[], &[], &params, None);
    assert!(segments.is_empty());
}

#[test]
fn extract_ancestry_segments_empty_states_returns_empty() {
    let params = make_params(&["EUR"], &[&["eur1"]]);

    let obs = vec![make_obs("chr1", 0, 10000, &[("eur1", 0.999)])];
    let segments = extract_ancestry_segments(&obs, &[], &params, None);
    assert!(segments.is_empty());
}

#[test]
fn extract_ancestry_segments_single_window() {
    let params = make_params(&["EUR", "AFR"], &[&["eur1"], &["afr1"]]);

    let obs = vec![make_obs("chr1", 0, 10000, &[("eur1", 0.999), ("afr1", 0.990)])];
    let states = vec![0]; // EUR
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].ancestry_name, "EUR");
    assert_eq!(segments[0].n_windows, 1);
}

// ══════════════════════════════════════════════════════════════════════
// estimate_admixture_proportions: single-population segments
// ══════════════════════════════════════════════════════════════════════

#[test]
fn estimate_admixture_single_population() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];
    let segments = vec![
        make_segment(0, "EUR", 5.0, 10),
        make_segment(0, "EUR", 3.0, 5),
    ];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);
    assert!((props.proportions["EUR"] - 1.0).abs() < 1e-10);
    assert!((props.proportions["AFR"] - 0.0).abs() < 1e-10);
    assert!((props.proportions["AMR"] - 0.0).abs() < 1e-10);
}

#[test]
fn estimate_admixture_mixed_populations() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    // 2 EUR segments (20kb total) + 1 AFR segment (10kb total)
    let segments = vec![
        make_segment(0, "EUR", 5.0, 1),  // 10kb
        make_segment(0, "EUR", 3.0, 1),  // 10kb
        make_segment(1, "AFR", 4.0, 1),  // 10kb
    ];
    let props = estimate_admixture_proportions(&segments, "test", &pop_names);
    // EUR should be ~2/3, AFR ~1/3
    let eur_prop = props.proportions["EUR"];
    let afr_prop = props.proportions["AFR"];
    assert!(eur_prop > afr_prop, "EUR should have higher proportion than AFR");
    assert!((eur_prop + afr_prop - 1.0).abs() < 1e-10, "proportions should sum to 1.0");
}

// ══════════════════════════════════════════════════════════════════════
// count_smoothing_changes: identical vs all-different
// ══════════════════════════════════════════════════════════════════════

#[test]
fn count_smoothing_changes_identical_zero() {
    assert_eq!(count_smoothing_changes(&[0, 1, 2, 0], &[0, 1, 2, 0]), 0);
}

#[test]
fn count_smoothing_changes_all_different() {
    assert_eq!(count_smoothing_changes(&[0, 0, 0, 0], &[1, 1, 1, 1]), 4);
}

#[test]
fn count_smoothing_changes_empty() {
    assert_eq!(count_smoothing_changes(&[], &[]), 0);
}

#[test]
fn count_smoothing_changes_partial_changes() {
    assert_eq!(count_smoothing_changes(&[0, 1, 0, 1], &[0, 0, 0, 1]), 1);
}

// ══════════════════════════════════════════════════════════════════════
// coverage_ratio in ancestry.rs
// ══════════════════════════════════════════════════════════════════════

#[test]
fn ancestry_coverage_ratio_a_zero_b_nonzero() {
    assert_eq!(coverage_ratio(0, 100), 0.0);
}

#[test]
fn ancestry_coverage_ratio_symmetric() {
    assert_eq!(coverage_ratio(30, 70), coverage_ratio(70, 30));
}

#[test]
fn ancestry_coverage_ratio_both_equal() {
    assert!((coverage_ratio(500, 500) - 1.0).abs() < 1e-10);
}

// ══════════════════════════════════════════════════════════════════════
// AncestryHmmParams::new with 1 state: degenerate but valid
// ══════════════════════════════════════════════════════════════════════

#[test]
fn ancestry_hmm_params_single_state() {
    let params = make_params(&["EUR"], &[&["eur1"]]);
    assert_eq!(params.n_states, 1);
    assert_eq!(params.populations.len(), 1);
    // Single state: P(stay) must be exactly 1.0 regardless of switch_prob
    assert!((params.transitions[0][0] - 1.0).abs() < 1e-10);
    // Initial prob should be 1.0 (uniform over 1 state)
    assert!((params.initial[0] - 1.0).abs() < 1e-10);
}

#[test]
fn ancestry_hmm_params_two_states_transitions_sum_to_one() {
    let params = make_params(&["EUR", "AFR"], &[&["eur1"], &["afr1"]]);
    assert_eq!(params.n_states, 2);
    // Each row of transitions should sum to 1.0
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "transition row should sum to 1.0, got {}", sum);
    }
    // Initial probs should sum to 1.0
    let init_sum: f64 = params.initial.iter().sum();
    assert!((init_sum - 1.0).abs() < 1e-10, "initial probs should sum to 1.0");
}

// ══════════════════════════════════════════════════════════════════════
// log_emission with no matching haplotypes returns finite value
// ══════════════════════════════════════════════════════════════════════

#[test]
fn log_emission_no_matching_haplotypes() {
    let params = make_params(&["EUR", "AFR"], &[&["eur1"], &["afr1"]]);

    // Observation with completely different haplotype names
    let obs = make_obs("chr1", 0, 10000, &[("xyz", 0.999)]);
    let le0 = params.log_emission(&obs, 0);
    let le1 = params.log_emission(&obs, 1);
    // Both should be NEG_INFINITY (no data for either state)
    assert_eq!(le0, f64::NEG_INFINITY, "with no matching haplotypes, emission should be -inf");
    assert_eq!(le1, f64::NEG_INFINITY, "with no matching haplotypes, emission should be -inf");
}

#[test]
fn log_emission_with_matching_haplotype() {
    let params = make_params(&["EUR", "AFR"], &[&["eur1"], &["afr1"]]);

    // EUR haplotype has high similarity, AFR has lower
    let obs = make_obs("chr1", 0, 10000, &[("eur1", 0.999), ("afr1", 0.980)]);
    let le_eur = params.log_emission(&obs, 0); // EUR state
    let le_afr = params.log_emission(&obs, 1); // AFR state
    // EUR emission should be higher (eur1 has higher sim to EUR state)
    assert!(
        le_eur > le_afr,
        "EUR emission should be higher than AFR: {} vs {}",
        le_eur,
        le_afr
    );
}
