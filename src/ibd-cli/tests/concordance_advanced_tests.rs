//! Advanced concordance edge case tests.
//!
//! Covers untested paths:
//! - length_correlation: constant lengths (zero variance), perfect positive, negative
//! - boundary_accuracy: exact matches, threshold filtering, even/odd match count
//! - per_window_concordance: both empty, zero-length region
//! - f1_score: boundary values
//! - matched_segments: zero-length segments, exact threshold boundary
//! - haplotype_level_concordance: empty data, one-sided data
//! - extract_haplotype_index / extract_sample_id: edge cases

use hprc_ibd::concordance::*;

// =============================================
// length_correlation edge cases
// =============================================

#[test]
fn test_length_correlation_constant_lengths() {
    // All segments identical length → zero variance → r = 0.0
    let matches: Vec<MatchedInterval> = vec![
        ((100, 200), (300, 400)),
        ((500, 600), (700, 800)),
        ((900, 1000), (1100, 1200)),
    ];
    let r = length_correlation(&matches);
    assert!((r - 0.0).abs() < 1e-10, "Constant lengths → r=0, got {}", r);
}

#[test]
fn test_length_correlation_perfect_positive() {
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (0, 200)),
        ((0, 200), (0, 400)),
        ((0, 300), (0, 600)),
    ];
    let r = length_correlation(&matches);
    assert!((r - 1.0).abs() < 1e-6, "Proportional → r=1, got {}", r);
}

#[test]
fn test_length_correlation_perfect_negative() {
    // Inverse correlation
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (0, 300)),
        ((0, 200), (0, 200)),
        ((0, 300), (0, 100)),
    ];
    let r = length_correlation(&matches);
    assert!((r - (-1.0)).abs() < 1e-6, "Inverse → r=-1, got {}", r);
}

#[test]
fn test_length_correlation_single_match() {
    let matches: Vec<MatchedInterval> = vec![((0, 100), (0, 200))];
    let r = length_correlation(&matches);
    assert!((r - 0.0).abs() < 1e-10);
}

#[test]
fn test_length_correlation_empty() {
    let matches: Vec<MatchedInterval> = vec![];
    let r = length_correlation(&matches);
    assert!((r - 0.0).abs() < 1e-10);
}

// =============================================
// boundary_accuracy edge cases
// =============================================

#[test]
fn test_boundary_accuracy_exact_matches() {
    let matches: Vec<MatchedInterval> = vec![
        ((100, 200), (100, 200)),
        ((300, 500), (300, 500)),
    ];
    let ba = boundary_accuracy(&matches, 10).unwrap();
    assert_eq!(ba.n_matched, 2);
    assert!((ba.mean_start_distance_bp - 0.0).abs() < 1e-10);
    assert!((ba.mean_end_distance_bp - 0.0).abs() < 1e-10);
    assert_eq!(ba.max_start_distance_bp, 0);
    assert_eq!(ba.max_end_distance_bp, 0);
    assert!((ba.frac_start_within_threshold - 1.0).abs() < 1e-10);
    assert!((ba.frac_end_within_threshold - 1.0).abs() < 1e-10);
}

#[test]
fn test_boundary_accuracy_empty() {
    let matches: Vec<MatchedInterval> = vec![];
    assert!(boundary_accuracy(&matches, 10).is_none());
}

#[test]
fn test_boundary_accuracy_threshold_filtering() {
    let matches: Vec<MatchedInterval> = vec![
        ((100, 200), (105, 198)),  // start: 5bp, end: 2bp
        ((300, 500), (350, 510)),  // start: 50bp, end: 10bp
    ];
    let ba = boundary_accuracy(&matches, 10).unwrap();
    assert_eq!(ba.n_matched, 2);
    // start within 10bp: first only → 0.5
    assert!((ba.frac_start_within_threshold - 0.5).abs() < 1e-10);
    // end within 10bp: both → 1.0
    assert!((ba.frac_end_within_threshold - 1.0).abs() < 1e-10);
}

#[test]
fn test_boundary_accuracy_single_match() {
    let matches: Vec<MatchedInterval> = vec![((100, 200), (120, 210))];
    let ba = boundary_accuracy(&matches, 100).unwrap();
    assert_eq!(ba.n_matched, 1);
    // Odd count → median is middle element
    assert!((ba.median_start_distance_bp - 20.0).abs() < 1e-10);
    assert!((ba.median_end_distance_bp - 10.0).abs() < 1e-10);
}

#[test]
fn test_boundary_accuracy_even_count_median() {
    // Even number of matches → median is average of two middle values
    let matches: Vec<MatchedInterval> = vec![
        ((100, 200), (110, 210)),  // start: 10, end: 10
        ((300, 400), (330, 420)),  // start: 30, end: 20
    ];
    let ba = boundary_accuracy(&matches, 100).unwrap();
    assert_eq!(ba.n_matched, 2);
    // Sorted start distances: [10, 30], median = (10+30)/2 = 20
    assert!((ba.median_start_distance_bp - 20.0).abs() < 1e-10);
    // Sorted end distances: [10, 20], median = (10+20)/2 = 15
    assert!((ba.median_end_distance_bp - 15.0).abs() < 1e-10);
}

// =============================================
// per_window_concordance edge cases
// =============================================

#[test]
fn test_per_window_concordance_both_empty() {
    let empty: Vec<(u64, u64)> = vec![];
    let c = per_window_concordance(&empty, &empty, (0, 1000), 100);
    assert!((c - 1.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_region_zero_length() {
    let segs = vec![(100, 200)];
    let c = per_window_concordance(&segs, &segs, (500, 500), 10);
    assert!((c - 0.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_window_size_zero() {
    let segs = vec![(100, 200)];
    let c = per_window_concordance(&segs, &segs, (0, 1000), 0);
    assert!((c - 0.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_perfect_agreement() {
    let segs = vec![(100, 500)];
    let c = per_window_concordance(&segs, &segs, (0, 1000), 100);
    assert!((c - 1.0).abs() < 1e-10, "Identical segments → perfect concordance, got {}", c);
}

#[test]
fn test_per_window_concordance_complete_disagreement() {
    // Our segments cover first half, theirs cover second half — no overlap
    let ours = vec![(0, 500)];
    let theirs = vec![(500, 1000)];
    let c = per_window_concordance(&ours, &theirs, (0, 1000), 100);
    // Each window is called by exactly one side → 0% concordance
    assert!((c - 0.0).abs() < 1e-10, "Complete disagreement should be 0, got {}", c);
}

// =============================================
// f1_score edge cases
// =============================================

#[test]
fn test_f1_score_boundary_values() {
    assert!((f1_score(0.0, 0.0) - 0.0).abs() < 1e-10);
    assert!((f1_score(1.0, 1.0) - 1.0).abs() < 1e-10);
    assert!((f1_score(1.0, 0.0) - 0.0).abs() < 1e-10);
    assert!((f1_score(0.0, 1.0) - 0.0).abs() < 1e-10);
    assert!((f1_score(0.5, 0.5) - 0.5).abs() < 1e-10);
}

#[test]
fn test_f1_score_asymmetric() {
    // F1(0.8, 0.6) = 2*0.8*0.6/(0.8+0.6) = 0.96/1.4 ≈ 0.6857
    let f1 = f1_score(0.8, 0.6);
    let expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
    assert!((f1 - expected).abs() < 1e-10);
}

// =============================================
// extract_haplotype_index / extract_sample_id edge cases
// =============================================

#[test]
fn test_extract_haplotype_index_valid() {
    assert_eq!(extract_haplotype_index("HG00280#2#JBHDWB010000002.1"), Some(2));
    assert_eq!(extract_haplotype_index("HG00280#1#CONTIG:100-200"), Some(1));
}

#[test]
fn test_extract_haplotype_index_no_hash() {
    assert_eq!(extract_haplotype_index("HG00280"), None);
}

#[test]
fn test_extract_haplotype_index_non_numeric() {
    assert_eq!(extract_haplotype_index("sample#HAP1#contig"), None);
}

#[test]
fn test_extract_haplotype_index_zero() {
    assert_eq!(extract_haplotype_index("sample#0#contig"), Some(0));
}

#[test]
fn test_extract_sample_id_concordance() {
    assert_eq!(extract_sample_id("HG00280#2#CONTIG"), "HG00280");
    assert_eq!(extract_sample_id("HG00280"), "HG00280");
    assert_eq!(extract_sample_id(""), "");
}

#[test]
fn test_extract_sample_id_hash_at_start() {
    // Edge: '#' at start
    assert_eq!(extract_sample_id("#2#contig"), "");
}

// =============================================
// matched_segments edge cases
// =============================================

#[test]
fn test_matched_segments_no_overlap() {
    let ours = vec![(100, 200)];
    let theirs = vec![(300, 400)];
    let matches = matched_segments(&ours, &theirs, 0.5);
    assert!(matches.is_empty());
}

#[test]
fn test_matched_segments_zero_length_ignored() {
    let ours = vec![(100, 100)];
    let theirs = vec![(100, 200)];
    let matches = matched_segments(&ours, &theirs, 0.5);
    assert!(matches.is_empty());
}

#[test]
fn test_matched_segments_perfect_match() {
    let segs = vec![(100, 200), (300, 400)];
    let matches = matched_segments(&segs, &segs, 0.5);
    assert_eq!(matches.len(), 2);
}

#[test]
fn test_matched_segments_threshold_boundary() {
    let ours = vec![(100, 200)]; // len=100
    let theirs = vec![(150, 300)]; // len=150, overlap=50, shorter=100, frac=0.5
    let matches = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(matches.len(), 1);
    let matches = matched_segments(&ours, &theirs, 0.51);
    assert!(matches.is_empty());
}

#[test]
fn test_matched_segments_many_to_many() {
    // Multiple ours segments overlap with multiple theirs segments
    let ours = vec![(100, 400), (300, 600)]; // overlapping
    let theirs = vec![(150, 350), (350, 550)];
    let matches = matched_segments(&ours, &theirs, 0.3);
    // Both ours overlap with both theirs
    assert!(matches.len() >= 2, "Should have multiple matches, got {}", matches.len());
}

// =============================================
// haplotype_level_concordance edge cases
// =============================================

#[test]
fn test_haplotype_level_concordance_no_data() {
    let our_segs: Vec<(String, String, u64, u64)> = vec![];
    let hapibd_segs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 1000000)
    );
    assert!(result.is_none());
}

#[test]
fn test_haplotype_level_concordance_only_our_data() {
    let our_segs = vec![
        ("HG001#1#contig".to_string(), "HG002#2#contig".to_string(), 100, 200),
    ];
    let hapibd_segs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 1000)
    );
    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 1);
    assert_eq!(r.n_hapibd_hap_combos, 0);
}

#[test]
fn test_haplotype_level_concordance_only_hapibd_data() {
    let our_segs: Vec<(String, String, u64, u64)> = vec![];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1, "HG002".to_string(), 2, 100u64, 200u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 1000)
    );
    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 0);
    assert_eq!(r.n_hapibd_hap_combos, 1);
}

#[test]
fn test_haplotype_level_concordance_reversed_sample_order() {
    // hap-ibd segments have samples in reverse order vs our data
    let our_segs = vec![
        ("HG001#1#contig".to_string(), "HG002#2#contig".to_string(), 100, 200),
    ];
    let hapibd_segs = vec![
        ("HG002".to_string(), 2, "HG001".to_string(), 1, 100u64, 200u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 1000)
    );
    assert!(result.is_some());
    let r = result.unwrap();
    assert!(r.best_jaccard > 0.0, "Should find overlap despite reversed sample order");
}

#[test]
fn test_haplotype_level_concordance_wrong_pair() {
    // Segments for different pairs should not match
    let our_segs = vec![
        ("HG001#1#contig".to_string(), "HG003#2#contig".to_string(), 100, 200),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1, "HG002".to_string(), 2, 100u64, 200u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 1000)
    );
    // Our data is for HG001-HG003, querying HG001-HG002 → only hapibd has data
    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 0);
    assert_eq!(r.n_hapibd_hap_combos, 1);
}

// =============================================
// segments_jaccard: additional edge cases
// =============================================

#[test]
fn test_jaccard_zero_length_region() {
    let segs = vec![(100, 200)];
    let j = segments_jaccard(&segs, &segs, (500, 500));
    assert!((j - 0.0).abs() < 1e-10);
}

#[test]
fn test_jaccard_segments_outside_region() {
    let ours = vec![(100, 200)];
    let theirs = vec![(300, 400)];
    // Region doesn't contain either segment
    let j = segments_jaccard(&ours, &theirs, (500, 600));
    assert!((j - 0.0).abs() < 1e-10);
}

#[test]
fn test_jaccard_symmetry() {
    let ours = vec![(100, 300)];
    let theirs = vec![(200, 400)];
    let j1 = segments_jaccard(&ours, &theirs, (0, 1000));
    let j2 = segments_jaccard(&theirs, &ours, (0, 1000));
    assert!((j1 - j2).abs() < 1e-10, "Jaccard should be symmetric");
}

#[test]
fn test_precision_recall_one_empty() {
    let segs = vec![(100, 200)];
    let empty: Vec<(u64, u64)> = vec![];
    // ours = segs, theirs (ground truth) = empty
    let (p, r) = segments_precision_recall(&segs, &empty, (0, 1000));
    // precision: intersection=0, covered_ours=100 → 0/100 = 0
    assert!((p - 0.0).abs() < 1e-10);
    // recall: covered_theirs=0 → 0.0
    assert!((r - 0.0).abs() < 1e-10);
}

#[test]
fn test_precision_recall_superset() {
    // Our segments are a superset of theirs
    let ours = vec![(50, 300)]; // 250bp
    let theirs = vec![(100, 200)]; // 100bp, fully contained
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // precision: intersection=100, covered_ours=250 → 0.4
    assert!((p - 100.0 / 250.0).abs() < 1e-6);
    // recall: intersection=100, covered_theirs=100 → 1.0
    assert!((r - 1.0).abs() < 1e-6);
}
