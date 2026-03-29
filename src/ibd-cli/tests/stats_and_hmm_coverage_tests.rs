//! Tests for previously uncovered ibd-cli functions:
//! - logit_transform_observations (stats.rs)
//! - kmeans_1d (stats.rs)
//! - involves_pair (hapibd.rs)
//! - extract_ibd_segments (basic version, hmm.rs)
//! - refine_states_with_posteriors (hmm.rs)
//! - baum_welch_with_distances smoke test (hmm.rs)
//! - baum_welch_with_genetic_map smoke test (hmm.rs)

use hprc_ibd::hmm::{
    extract_ibd_segments, refine_states_with_posteriors, GeneticMap, HmmParams,
};
use hprc_ibd::stats::{kmeans_1d, logit, logit_transform_observations, LOGIT_CAP};

// ============================================================================
// logit_transform_observations tests
// ============================================================================

#[test]
fn logit_transform_empty_slice() {
    let result = logit_transform_observations(&[]);
    assert!(result.is_empty());
}

#[test]
fn logit_transform_single_value() {
    let result = logit_transform_observations(&[0.5]);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.0).abs() < 1e-6, "logit(0.5) should be ~0.0");
}

#[test]
fn logit_transform_preserves_order() {
    // Higher identity → higher logit
    let identities = vec![0.5, 0.9, 0.99, 0.999, 0.9999];
    let transformed = logit_transform_observations(&identities);
    assert_eq!(transformed.len(), 5);
    for i in 1..transformed.len() {
        assert!(
            transformed[i] > transformed[i - 1],
            "logit should preserve ordering: {} not > {}",
            transformed[i],
            transformed[i - 1]
        );
    }
}

#[test]
fn logit_transform_matches_individual_logit() {
    let identities = vec![0.1, 0.5, 0.9];
    let transformed = logit_transform_observations(&identities);
    for (i, &val) in identities.iter().enumerate() {
        let expected = logit(val);
        assert!(
            (transformed[i] - expected).abs() < 1e-12,
            "logit_transform_observations should match logit() for each element"
        );
    }
}

#[test]
fn logit_transform_extreme_values_capped() {
    // Values very close to 0 and 1 should be capped at ±LOGIT_CAP
    let extremes = vec![0.0, 1.0, 1e-15, 1.0 - 1e-15];
    let transformed = logit_transform_observations(&extremes);
    for &val in &transformed {
        assert!(val >= -LOGIT_CAP && val <= LOGIT_CAP, "logit should be capped to [-{}, {}], got {}", LOGIT_CAP, LOGIT_CAP, val);
    }
}

#[test]
fn logit_transform_pangenome_typical_range() {
    // Typical pangenome identity values: 0.997 to 0.9999
    let identities = vec![0.997, 0.998, 0.999, 0.9995, 0.9999];
    let transformed = logit_transform_observations(&identities);
    // All should be positive (identity > 0.5) and spread out
    for &val in &transformed {
        assert!(val > 0.0, "logit of >0.5 should be positive");
    }
    // Check that the spread is magnified
    let raw_spread = identities.last().unwrap() - identities.first().unwrap();
    let logit_spread = transformed.last().unwrap() - transformed.first().unwrap();
    assert!(
        logit_spread > raw_spread * 100.0,
        "logit should magnify spread near 1.0: raw={}, logit={}",
        raw_spread,
        logit_spread
    );
}

// ============================================================================
// kmeans_1d tests
// ============================================================================

#[test]
fn kmeans_1d_returns_none_for_empty_data() {
    assert!(kmeans_1d(&[], 2, 100).is_none());
}

#[test]
fn kmeans_1d_returns_none_for_k_zero() {
    assert!(kmeans_1d(&[1.0, 2.0], 0, 100).is_none());
}

#[test]
fn kmeans_1d_returns_none_when_k_exceeds_data() {
    assert!(kmeans_1d(&[1.0], 2, 100).is_none());
}

#[test]
fn kmeans_1d_single_cluster() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = kmeans_1d(&data, 1, 100).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 1);
    assert!((centers[0] - 3.0).abs() < 1e-6, "single cluster center should be mean");
    assert!(assignments.iter().all(|&a| a == 0));
}

#[test]
fn kmeans_1d_two_well_separated_clusters() {
    // Two clearly separated groups
    let data = vec![1.0, 1.1, 1.2, 10.0, 10.1, 10.2];
    let result = kmeans_1d(&data, 2, 100).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 2);

    // Centers should be near 1.1 and 10.1 (in some order)
    let mut sorted_centers = centers.clone();
    sorted_centers.sort_by(|a, b| a.total_cmp(b));
    assert!((sorted_centers[0] - 1.1).abs() < 0.2);
    assert!((sorted_centers[1] - 10.1).abs() < 0.2);

    // First 3 should be in one cluster, last 3 in another
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[1], assignments[2]);
    assert_eq!(assignments[3], assignments[4]);
    assert_eq!(assignments[4], assignments[5]);
    assert_ne!(assignments[0], assignments[3]);
}

#[test]
fn kmeans_1d_identical_values() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let result = kmeans_1d(&data, 2, 100).unwrap();
    let (centers, _assignments) = result;
    assert_eq!(centers.len(), 2);
}

#[test]
fn kmeans_1d_k_equals_n() {
    let data = vec![1.0, 5.0, 10.0];
    let result = kmeans_1d(&data, 3, 100).unwrap();
    let (_centers, assignments) = result;
    assert_eq!(assignments.len(), 3);
}

#[test]
fn kmeans_1d_converges_with_single_iteration() {
    let data = vec![0.0, 0.0, 100.0, 100.0];
    let result = kmeans_1d(&data, 2, 1).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 2);
    assert_eq!(assignments.len(), 4);
}

#[test]
fn kmeans_1d_assignments_length_matches_data() {
    let data = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
    let result = kmeans_1d(&data, 3, 100).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 3);
    assert_eq!(assignments.len(), data.len());
    // All assignments should be valid cluster indices
    for &a in &assignments {
        assert!(a < 3, "assignment {} out of range", a);
    }
}

// ============================================================================
// involves_pair tests (HapIbdSegment)
// ============================================================================

use hprc_ibd::hapibd::parse_hapibd_content;

fn make_segment_content(s1: &str, s2: &str) -> String {
    format!("{}\t1\t{}\t1\tchr1\t1000\t2000\t5.0", s1, s2)
}

#[test]
fn involves_pair_exact_match() {
    let content = make_segment_content("SAMPLE_A", "SAMPLE_B");
    let segments = parse_hapibd_content(&content);
    assert!(segments[0].involves_pair("SAMPLE_A", "SAMPLE_B"));
}

#[test]
fn involves_pair_reversed_order() {
    let content = make_segment_content("SAMPLE_A", "SAMPLE_B");
    let segments = parse_hapibd_content(&content);
    assert!(segments[0].involves_pair("SAMPLE_B", "SAMPLE_A"));
}

#[test]
fn involves_pair_wrong_sample() {
    let content = make_segment_content("SAMPLE_A", "SAMPLE_B");
    let segments = parse_hapibd_content(&content);
    assert!(!segments[0].involves_pair("SAMPLE_A", "SAMPLE_C"));
    assert!(!segments[0].involves_pair("SAMPLE_C", "SAMPLE_D"));
}

#[test]
fn involves_pair_same_sample_twice() {
    let content = make_segment_content("SAMPLE_A", "SAMPLE_A");
    let segments = parse_hapibd_content(&content);
    assert!(segments[0].involves_pair("SAMPLE_A", "SAMPLE_A"));
}

// ============================================================================
// extract_ibd_segments (basic) tests
// ============================================================================

#[test]
fn extract_ibd_segments_empty_input() {
    let segments = extract_ibd_segments(&[]);
    assert!(segments.is_empty());
}

#[test]
fn extract_ibd_segments_all_non_ibd() {
    let states = vec![0, 0, 0, 0, 0];
    let segments = extract_ibd_segments(&states);
    assert!(segments.is_empty());
}

#[test]
fn extract_ibd_segments_all_ibd() {
    let states = vec![1, 1, 1, 1, 1];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (0, 4, 5));
}

#[test]
fn extract_ibd_segments_single_window_ibd() {
    let states = vec![0, 0, 1, 0, 0];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (2, 2, 1));
}

#[test]
fn extract_ibd_segments_two_segments() {
    let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0], (2, 4, 3));
    assert_eq!(segments[1], (7, 8, 2));
}

#[test]
fn extract_ibd_segments_starts_with_ibd() {
    let states = vec![1, 1, 0, 0, 0];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (0, 1, 2));
}

#[test]
fn extract_ibd_segments_ends_with_ibd() {
    let states = vec![0, 0, 0, 1, 1];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (3, 4, 2));
}

#[test]
fn extract_ibd_segments_alternating() {
    let states = vec![1, 0, 1, 0, 1];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0], (0, 0, 1));
    assert_eq!(segments[1], (2, 2, 1));
    assert_eq!(segments[2], (4, 4, 1));
}

#[test]
fn extract_ibd_segments_single_non_ibd() {
    let states = vec![0];
    let segments = extract_ibd_segments(&states);
    assert!(segments.is_empty());
}

#[test]
fn extract_ibd_segments_single_ibd() {
    let states = vec![1];
    let segments = extract_ibd_segments(&states);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], (0, 0, 1));
}

// ============================================================================
// refine_states_with_posteriors tests
// ============================================================================

#[test]
fn refine_posteriors_empty_input() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert!(states.is_empty());
}

#[test]
fn refine_posteriors_mismatched_lengths() {
    // Should return without changes when lengths don't match
    let mut states = vec![0, 0, 1, 1, 0];
    let posteriors = vec![0.1, 0.2, 0.8]; // shorter
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original);
}

#[test]
fn refine_posteriors_extend_high_posterior_adjacent() {
    // Window 2 is non-IBD but has high posterior and is adjacent to IBD segment
    let mut states = vec![0, 0, 0, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.7, 0.9, 0.9, 0.1, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 2 should be extended to IBD (posterior 0.7 >= extend_threshold 0.5, adjacent to IBD)
    assert_eq!(states[2], 1, "high-posterior window adjacent to IBD should be extended");
}

#[test]
fn refine_posteriors_no_extend_without_adjacency() {
    // Window 2 has high posterior but is NOT adjacent to any IBD segment
    let mut states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.1, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // No extension should happen since there's no IBD to extend from
    assert_eq!(states[2], 0, "high-posterior window without adjacent IBD should not extend");
}

#[test]
fn refine_posteriors_trim_low_posterior_edge() {
    // Edge of IBD segment has low posterior
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.1, 0.9, 0.1, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Windows 1 and 3 are at edges with posteriors below trim_threshold (0.2)
    assert_eq!(states[1], 0, "low-posterior edge should be trimmed");
    assert_eq!(states[3], 0, "low-posterior edge should be trimmed");
    // Window 2 should remain IBD (not at edge after trimming neighbors)
    // But after trimming both edges, window 2 becomes an edge itself
}

#[test]
fn refine_posteriors_no_trim_interior() {
    // Interior IBD windows are not trimmed even with low posterior
    let mut states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.3, 0.3, 0.1, 0.3, 0.3, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 3 has low posterior (0.1) but is interior (both neighbors are IBD)
    // It should NOT be trimmed
    assert_eq!(states[3], 1, "interior window should not be trimmed");
}

#[test]
fn refine_posteriors_cascade_extension() {
    // Extension can cascade: first window 2 extends from 3, then window 1 extends from 2
    let mut states = vec![0, 0, 0, 1, 1, 0];
    let posteriors = vec![0.1, 0.6, 0.6, 0.9, 0.9, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states[2], 1, "cascade extension: window 2 should extend");
    assert_eq!(states[1], 1, "cascade extension: window 1 should extend after window 2");
}

#[test]
fn refine_posteriors_all_ibd_high_posterior() {
    // All IBD, all high posterior — no changes
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.99, 0.95, 0.9];
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original);
}

// ============================================================================
// baum_welch_with_distances smoke test
// ============================================================================

#[test]
fn baum_welch_with_distances_too_few_observations() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission = params.emission.clone();
    let obs = vec![0.998; 5]; // < 10 observations
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 10000, (i + 1) * 10000)).collect();
    params.baum_welch_with_distances(&obs, &positions, 10, 1e-4, None, 10000);
    // Should return without changing params
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
    assert_eq!(params.emission[1].mean, original_emission[1].mean);
}

#[test]
fn baum_welch_with_distances_mismatched_positions() {
    // When positions length != observations length, should fall back to standard baum_welch
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998; 20];
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 10000, (i + 1) * 10000)).collect();
    // This should not panic — it falls back to standard baum_welch
    params.baum_welch_with_distances(&obs, &positions, 5, 1e-4, None, 10000);
}

#[test]
fn baum_welch_with_distances_smoke_converges() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    // Simulated data: mostly non-IBD with a clear IBD region
    let mut obs = vec![0.998; 50];
    for item in obs.iter_mut().take(40).skip(20) {
        *item = 0.9999;
    }
    let positions: Vec<(u64, u64)> = (0..50)
        .map(|i| (i * 10000, (i + 1) * 10000 - 1))
        .collect();
    params.baum_welch_with_distances(&obs, &positions, 10, 1e-6, None, 10000);
    // After training, state 1 (IBD) should have higher mean than state 0
    assert!(
        params.emission[1].mean > params.emission[0].mean,
        "After training, IBD emission mean ({}) should > non-IBD mean ({})",
        params.emission[1].mean,
        params.emission[0].mean
    );
}

// ============================================================================
// baum_welch_with_genetic_map smoke test
// ============================================================================

#[test]
fn baum_welch_with_genetic_map_too_few_observations() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let original_emission = params.emission.clone();
    let obs = vec![0.998; 5];
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 10000, (i + 1) * 10000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (100000, 0.1)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 10, 1e-4, None, 10000);
    assert_eq!(params.emission[0].mean, original_emission[0].mean);
}

#[test]
fn baum_welch_with_genetic_map_mismatched_positions() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.998; 20];
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 10000, (i + 1) * 10000)).collect();
    let gmap = GeneticMap::new(vec![(0, 0.0), (200000, 0.2)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 5, 1e-4, None, 10000);
    // Should not panic
}

#[test]
fn baum_welch_with_genetic_map_smoke_converges() {
    let mut params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let mut obs = vec![0.998; 50];
    for item in obs.iter_mut().take(40).skip(20) {
        *item = 0.9999;
    }
    let positions: Vec<(u64, u64)> = (0..50)
        .map(|i| (i * 10000, (i + 1) * 10000 - 1))
        .collect();
    // Genetic map covering the region
    let gmap = GeneticMap::new(vec![(0, 0.0), (250000, 0.25), (500000, 0.5)]);
    params.baum_welch_with_genetic_map(&obs, &positions, &gmap, 10, 1e-6, None, 10000);
    assert!(
        params.emission[1].mean > params.emission[0].mean,
        "After training with genetic map, IBD mean ({}) should > non-IBD mean ({})",
        params.emission[1].mean,
        params.emission[0].mean
    );
}
