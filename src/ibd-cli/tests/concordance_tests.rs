//! Edge case tests for the IBD concordance metrics module.
//!
//! Tests cover:
//! - segment_overlap_bp: boundary conditions, zero-length, reversed, adjacent
//! - segments_jaccard: symmetry, edge cases, overlapping inputs
//! - segments_precision_recall: asymmetric cases, one-empty, superset/subset
//! - per_window_concordance: window size edge cases, partial coverage
//! - matched_segments: threshold boundaries, many-to-one, empty
//! - length_correlation: constant lengths, zero variance, single match
//! - f1_score: edge values

use hprc_ibd::concordance::*;

// =============================================
// segment_overlap_bp edge cases
// =============================================

#[test]
fn test_overlap_identical() {
    assert_eq!(segment_overlap_bp((100, 200), (100, 200)), 100);
}

#[test]
fn test_overlap_no_overlap_before() {
    assert_eq!(segment_overlap_bp((100, 200), (50, 80)), 0);
}

#[test]
fn test_overlap_no_overlap_after() {
    assert_eq!(segment_overlap_bp((100, 200), (250, 350)), 0);
}

#[test]
fn test_overlap_adjacent_right() {
    // [100,200) and [200,300) → half-open, no overlap
    assert_eq!(segment_overlap_bp((100, 200), (200, 300)), 0);
}

#[test]
fn test_overlap_one_bp_overlap() {
    // [100,201) and [200,300) → overlap is [200,201) = 1bp
    assert_eq!(segment_overlap_bp((100, 201), (200, 300)), 1);
}

#[test]
fn test_overlap_contained() {
    // B inside A
    assert_eq!(segment_overlap_bp((100, 500), (200, 300)), 100);
}

#[test]
fn test_overlap_reversed_containment() {
    // A inside B
    assert_eq!(segment_overlap_bp((200, 300), (100, 500)), 100);
}

#[test]
fn test_overlap_zero_length_intervals() {
    // Zero-length interval [100,100) → no overlap with anything
    assert_eq!(segment_overlap_bp((100, 100), (100, 200)), 0);
    assert_eq!(segment_overlap_bp((100, 200), (150, 150)), 0);
    assert_eq!(segment_overlap_bp((100, 100), (100, 100)), 0);
}

#[test]
fn test_overlap_very_large_intervals() {
    let a = (0u64, 248_956_422); // chr1 length
    let b = (100_000_000u64, 200_000_000);
    assert_eq!(segment_overlap_bp(a, b), 100_000_000);
}

#[test]
fn test_overlap_symmetry() {
    let a = (100, 300);
    let b = (200, 400);
    assert_eq!(segment_overlap_bp(a, b), segment_overlap_bp(b, a));
}

// =============================================
// segments_jaccard edge cases
// =============================================

#[test]
fn test_jaccard_empty_region() {
    let segs = vec![(100, 200)];
    let j = segments_jaccard(&segs, &segs, (0, 0));
    assert!((j - 0.0).abs() < 1e-9);
}

#[test]
fn test_jaccard_segments_outside_region() {
    let ours = vec![(50, 80)];
    let theirs = vec![(50, 80)];
    let j = segments_jaccard(&ours, &theirs, (100, 200));
    assert!((j - 0.0).abs() < 1e-9, "Segments outside region → Jaccard=0, got {}", j);
}

#[test]
fn test_jaccard_symmetry() {
    let ours = vec![(100, 300)];
    let theirs = vec![(200, 400)];
    let j1 = segments_jaccard(&ours, &theirs, (0, 1000));
    let j2 = segments_jaccard(&theirs, &ours, (0, 1000));
    assert!((j1 - j2).abs() < 1e-9, "Jaccard should be symmetric: {} vs {}", j1, j2);
}

#[test]
fn test_jaccard_overlapping_own_segments() {
    // Our segments overlap each other
    let ours = vec![(100, 300), (200, 400)];
    let theirs = vec![(100, 400)];
    // After merging ours: 100-400 = 300bp, theirs: 100-400 = 300bp
    // Intersection: 300bp, union: 300bp → Jaccard = 1.0
    let j = segments_jaccard(&ours, &theirs, (0, 1000));
    assert!((j - 1.0).abs() < 1e-9, "Expected Jaccard=1.0, got {}", j);
}

#[test]
fn test_jaccard_partially_clipped_by_region() {
    // Segment extends beyond region
    let ours = vec![(50, 150)];
    let theirs = vec![(50, 150)];
    let j = segments_jaccard(&ours, &theirs, (100, 200));
    // Both clipped to [100,150), so 50bp each → Jaccard = 1.0
    assert!((j - 1.0).abs() < 1e-9, "Expected Jaccard=1.0 after clipping, got {}", j);
}

#[test]
fn test_jaccard_known_value() {
    // ours: [100,300) = 200bp, theirs: [200,500) = 300bp
    // Intersection: [200,300) = 100bp, Union: [100,500) = 400bp
    // Jaccard = 100/400 = 0.25
    let ours = vec![(100, 300)];
    let theirs = vec![(200, 500)];
    let j = segments_jaccard(&ours, &theirs, (0, 1000));
    assert!((j - 0.25).abs() < 1e-9, "Expected Jaccard=0.25, got {}", j);
}

#[test]
fn test_jaccard_many_small_segments() {
    // 10 non-overlapping 10bp segments vs one big segment
    let ours: Vec<(u64, u64)> = (0..10).map(|i| (i * 20, i * 20 + 10)).collect();
    let theirs = vec![(0, 200)];
    // ours covers 100bp, theirs covers 200bp, intersection = 100bp, union = 200bp
    let j = segments_jaccard(&ours, &theirs, (0, 200));
    assert!((j - 0.5).abs() < 1e-9, "Expected Jaccard=0.5, got {}", j);
}

// =============================================
// segments_precision_recall edge cases
// =============================================

#[test]
fn test_precision_recall_both_empty() {
    let empty: Vec<(u64, u64)> = vec![];
    let (p, r) = segments_precision_recall(&empty, &empty, (0, 1000));
    assert!((p - 0.0).abs() < 1e-9);
    assert!((r - 0.0).abs() < 1e-9);
}

#[test]
fn test_precision_recall_ours_empty() {
    let empty: Vec<(u64, u64)> = vec![];
    let theirs = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&empty, &theirs, (0, 1000));
    assert!((p - 0.0).abs() < 1e-9);
    assert!((r - 0.0).abs() < 1e-9);
}

#[test]
fn test_precision_recall_theirs_empty() {
    let ours = vec![(100, 200)];
    let empty: Vec<(u64, u64)> = vec![];
    let (p, r) = segments_precision_recall(&ours, &empty, (0, 1000));
    // Precision: 0/100 intersection → 0.0 (no TP)
    // Actually precision = intersection/covered_ours = 0/100 = 0.0
    // Recall = intersection/covered_theirs = 0/0 → 0.0
    assert!((p - 0.0).abs() < 1e-9);
    assert!((r - 0.0).abs() < 1e-9);
}

#[test]
fn test_precision_recall_superset() {
    // We call more than truth → precision < 1, recall = 1
    let ours = vec![(50, 250)];
    let theirs = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // Our coverage: 200bp, their coverage: 100bp, intersection: 100bp
    // Precision = 100/200 = 0.5, Recall = 100/100 = 1.0
    assert!((p - 0.5).abs() < 1e-9, "Expected precision=0.5, got {}", p);
    assert!((r - 1.0).abs() < 1e-9, "Expected recall=1.0, got {}", r);
}

#[test]
fn test_precision_recall_subset() {
    // We call less than truth → precision = 1, recall < 1
    let ours = vec![(150, 200)];
    let theirs = vec![(100, 300)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // Our coverage: 50bp, their coverage: 200bp, intersection: 50bp
    // Precision = 50/50 = 1.0, Recall = 50/200 = 0.25
    assert!((p - 1.0).abs() < 1e-9, "Expected precision=1.0, got {}", p);
    assert!((r - 0.25).abs() < 1e-9, "Expected recall=0.25, got {}", r);
}

#[test]
fn test_precision_recall_two_segments_one_match() {
    // Truth: 2 segments, we call 1 correctly
    let ours = vec![(100, 200)];
    let theirs = vec![(100, 200), (500, 600)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // Our coverage: 100bp, their coverage: 200bp, intersection: 100bp
    // Precision = 100/100 = 1.0, Recall = 100/200 = 0.5
    assert!((p - 1.0).abs() < 1e-9, "Expected precision=1.0, got {}", p);
    assert!((r - 0.5).abs() < 1e-9, "Expected recall=0.5, got {}", r);
}

#[test]
fn test_precision_recall_three_calls_one_match() {
    // We call 3, only 1 overlaps truth
    let ours = vec![(100, 200), (400, 500), (700, 800)];
    let theirs = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // Our coverage: 300bp, their coverage: 100bp, intersection: 100bp
    // Precision = 100/300 ≈ 0.333, Recall = 100/100 = 1.0
    assert!((p - 1.0 / 3.0).abs() < 1e-9, "Expected precision≈0.333, got {}", p);
    assert!((r - 1.0).abs() < 1e-9, "Expected recall=1.0, got {}", r);
}

// =============================================
// per_window_concordance edge cases
// =============================================

#[test]
fn test_concordance_zero_window_size() {
    let segs = vec![(0, 100)];
    let c = per_window_concordance(&segs, &segs, (0, 100), 0);
    assert!((c - 0.0).abs() < 1e-9, "Zero window size → 0.0");
}

#[test]
fn test_concordance_empty_region() {
    let segs = vec![(0, 100)];
    let c = per_window_concordance(&segs, &segs, (50, 50), 10);
    assert!((c - 0.0).abs() < 1e-9, "Empty region → 0.0");
}

#[test]
fn test_concordance_window_larger_than_region() {
    let ours = vec![(0, 100)];
    let theirs = vec![(0, 100)];
    // Region = 100bp, window = 200bp → 1 window
    let c = per_window_concordance(&ours, &theirs, (0, 100), 200);
    assert!((c - 1.0).abs() < 1e-9, "Single large window, both agree → 1.0");
}

#[test]
fn test_concordance_one_has_ibd_other_doesnt() {
    let ours = vec![(0, 100)];
    let empty: Vec<(u64, u64)> = vec![];
    let c = per_window_concordance(&ours, &empty, (0, 100), 10);
    assert!((c - 0.0).abs() < 1e-9, "All disagree → 0.0, got {}", c);
}

#[test]
fn test_concordance_partial_window_coverage() {
    // A segment covers 60% of the first window (>50% → call IBD)
    // but only 40% of the second window (<50% → no IBD)
    let ours = vec![(0, 16)]; // covers [0,16) in 10bp windows: win0=10/10, win1=6/10
    let theirs = vec![(0, 16)];
    // win0: both call IBD (10 > 5), win1: both call IBD (6 > 5), win2-9: both no IBD
    // Actually region needs to be wider
    let c = per_window_concordance(&ours, &theirs, (0, 30), 10);
    // 3 windows: [0,10)→both IBD, [10,20)→both IBD (6>5), [20,30)→both no IBD → 3/3=1.0
    assert!((c - 1.0).abs() < 1e-9, "Expected 1.0, got {}", c);
}

#[test]
fn test_concordance_exact_half_no_call() {
    // Segment covers exactly 50% of window → >50% threshold not met → no IBD
    let ours = vec![(0, 5)]; // covers exactly 5/10 bp
    let theirs = vec![(0, 10)]; // covers all 10bp
    let c = per_window_concordance(&ours, &theirs, (0, 10), 10);
    // ours: 5bp, half=5, 5 > 5 is false → no IBD. theirs: 10bp > 5 → IBD.
    // Disagree → 0.0
    assert!((c - 0.0).abs() < 1e-9, "Expected 0.0, got {}", c);
}

// =============================================
// matched_segments edge cases
// =============================================

#[test]
fn test_matched_empty_inputs() {
    let empty: Vec<(u64, u64)> = vec![];
    let segs = vec![(100, 200)];
    assert!(matched_segments(&empty, &segs, 0.5).is_empty());
    assert!(matched_segments(&segs, &empty, 0.5).is_empty());
    assert!(matched_segments(&empty, &empty, 0.5).is_empty());
}

#[test]
fn test_matched_threshold_boundary() {
    // 50bp overlap out of 100bp shorter → exactly 50%
    let ours = vec![(100, 200)];
    let theirs = vec![(150, 250)];

    // At 0.5 threshold → matches (50 >= 50)
    let m1 = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m1.len(), 1);

    // At 0.51 threshold → no match
    let m2 = matched_segments(&ours, &theirs, 0.51);
    assert!(m2.is_empty());
}

#[test]
fn test_matched_one_to_many() {
    // One of ours matches multiple of theirs
    let ours = vec![(100, 500)];
    let theirs = vec![(100, 200), (300, 500)];
    // Overlap with first: 100bp, shorter=100 → frac=1.0
    // Overlap with second: 200bp, shorter=200 → frac=1.0
    let m = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m.len(), 2);
    assert!(m.contains(&(0, 0)));
    assert!(m.contains(&(0, 1)));
}

#[test]
fn test_matched_zero_threshold() {
    // With threshold 0.0, any overlap counts
    let ours = vec![(100, 200)];
    let theirs = vec![(199, 300)]; // 1bp overlap
    let m = matched_segments(&ours, &theirs, 0.0);
    assert_eq!(m.len(), 1);
}

// =============================================
// length_correlation edge cases
// =============================================

#[test]
fn test_length_correlation_empty() {
    let matches: Vec<MatchedInterval> = vec![];
    assert!((length_correlation(&matches) - 0.0).abs() < 1e-9);
}

#[test]
fn test_length_correlation_single_pair() {
    let matches = vec![((0u64, 100u64), (0u64, 200u64))];
    // Need >= 2 for correlation
    assert!((length_correlation(&matches) - 0.0).abs() < 1e-9);
}

#[test]
fn test_length_correlation_constant_lengths() {
    // All same length → zero variance → r = 0.0
    let matches = vec![
        ((0u64, 100u64), (0u64, 200u64)),
        ((0, 100), (0, 200)),
        ((0, 100), (0, 200)),
    ];
    let r = length_correlation(&matches);
    assert!((r - 0.0).abs() < 1e-9, "Constant lengths → r=0, got {}", r);
}

#[test]
fn test_length_correlation_perfect_linear() {
    // y = 2x → perfect correlation
    let matches = vec![
        ((0u64, 100u64), (0u64, 200u64)),
        ((0, 200), (0, 400)),
        ((0, 300), (0, 600)),
    ];
    let r = length_correlation(&matches);
    assert!((r - 1.0).abs() < 1e-9, "Expected r=1.0, got {}", r);
}

#[test]
fn test_length_correlation_negative_linear() {
    // y = -x + C → perfect negative correlation
    let matches = vec![
        ((0u64, 100u64), (0u64, 300u64)), // 100 vs 300
        ((0, 200), (0, 200)),               // 200 vs 200
        ((0, 300), (0, 100)),               // 300 vs 100
    ];
    let r = length_correlation(&matches);
    assert!((r - (-1.0)).abs() < 1e-9, "Expected r=-1.0, got {}", r);
}

#[test]
fn test_length_correlation_bounded() {
    // Verify r is always in [-1, 1]
    let matches = vec![
        ((0u64, 50u64), (0u64, 1000u64)),
        ((0, 1000), (0, 50)),
        ((0, 500), (0, 500)),
        ((0, 200), (0, 800)),
    ];
    let r = length_correlation(&matches);
    assert!((-1.0..=1.0).contains(&r), "r should be in [-1,1], got {}", r);
}

// =============================================
// f1_score edge cases
// =============================================

#[test]
fn test_f1_both_zero() {
    assert!((f1_score(0.0, 0.0) - 0.0).abs() < 1e-9);
}

#[test]
fn test_f1_both_one() {
    assert!((f1_score(1.0, 1.0) - 1.0).abs() < 1e-9);
}

#[test]
fn test_f1_one_zero() {
    // If either is 0, F1 = 0
    assert!((f1_score(1.0, 0.0) - 0.0).abs() < 1e-9);
    assert!((f1_score(0.0, 1.0) - 0.0).abs() < 1e-9);
}

#[test]
fn test_f1_known_value() {
    // precision=0.8, recall=0.6 → F1 = 2*0.48/1.4 ≈ 0.6857
    let f = f1_score(0.8, 0.6);
    let expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
    assert!((f - expected).abs() < 1e-9, "Expected F1={}, got {}", expected, f);
}

#[test]
fn test_f1_symmetry() {
    // F1(p, r) == F1(r, p)
    assert!((f1_score(0.3, 0.7) - f1_score(0.7, 0.3)).abs() < 1e-9);
}

// =============================================
// Integration: precision_recall + f1 pipeline
// =============================================

#[test]
fn test_precision_recall_f1_pipeline() {
    // Realistic scenario: 2 truth segments, we find 1 perfectly + 1 extra FP
    let ours = vec![(100, 200), (500, 600)];
    let theirs = vec![(100, 200), (300, 400)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    let f = f1_score(p, r);

    // Our coverage: 200bp, their coverage: 200bp, intersection: 100bp
    // Precision = 100/200 = 0.5, Recall = 100/200 = 0.5
    assert!((p - 0.5).abs() < 1e-9, "Expected precision=0.5, got {}", p);
    assert!((r - 0.5).abs() < 1e-9, "Expected recall=0.5, got {}", r);
    assert!((f - 0.5).abs() < 1e-9, "Expected F1=0.5, got {}", f);
}

// =============================================
// Additional edge cases for region clipping and merging
// =============================================

#[test]
fn test_jaccard_region_clips_both_sides() {
    // Segments extend past both sides of the region
    let ours = vec![(50, 350)]; // extends before and after region
    let theirs = vec![(150, 250)]; // fully within region
    let region = (100, 300);
    // ours clipped: [100, 300) = 200bp. theirs clipped: [150, 250) = 100bp.
    // intersection: [150, 250) = 100bp. union: 200+100-100 = 200bp. J = 0.5
    let j = segments_jaccard(&ours, &theirs, region);
    assert!((j - 0.5).abs() < 1e-9, "Expected Jaccard=0.5, got {}", j);
}

#[test]
fn test_precision_recall_with_overlapping_own_segments() {
    // Our segments overlap each other; verify merge handles correctly
    let ours = vec![(100, 250), (200, 350)]; // overlapping → merged [100, 350) = 250bp
    let theirs = vec![(100, 300)]; // 200bp
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 500));
    // intersection: [100, 300) ∩ [100, 350) after merge
    // merged ours: [100, 350), merged theirs: [100, 300)
    // intersection: [100, 300) = 200bp
    // precision = 200/250 = 0.8, recall = 200/200 = 1.0
    assert!((p - 0.8).abs() < 1e-9, "Expected precision=0.8, got {}", p);
    assert!((r - 1.0).abs() < 1e-9, "Expected recall=1.0, got {}", r);
}

#[test]
fn test_concordance_odd_region_size() {
    // Region not evenly divisible by window_size → last window is partial
    // Region = [0, 25), window = 10bp → windows: [0,10), [10,20), [20,25)
    let ours = vec![(0, 25)]; // covers everything
    let theirs = vec![(0, 25)]; // covers everything
    let c = per_window_concordance(&ours, &theirs, (0, 25), 10);
    // 3 windows, all agree → 1.0
    assert!((c - 1.0).abs() < 1e-9, "Expected 1.0, got {}", c);
}

#[test]
fn test_concordance_partial_final_window() {
    // Final window is only 5bp, segment covers the first 2 full windows only
    // Region [0, 25), window 10bp → 3 windows: [0,10), [10,20), [20,25)
    let ours = vec![(0, 20)]; // covers first 2 windows only
    let empty: Vec<(u64, u64)> = vec![];
    let c = per_window_concordance(&ours, &empty, (0, 25), 10);
    // Win 0: ours=10bp > half(5) → IBD, theirs=0 → no IBD → disagree
    // Win 1: ours=10bp > half(5) → IBD, theirs=0 → disagree
    // Win 2: ours=0bp, theirs=0bp → both no IBD → agree
    // 1 agree / 3 total = 1/3
    assert!((c - 1.0 / 3.0).abs() < 1e-9, "Expected 1/3, got {}", c);
}

#[test]
fn test_matched_segments_zero_length_segment() {
    // Zero-length segment [100, 100) → shorter=0 → skip
    let ours = vec![(100, 100)];
    let theirs = vec![(100, 200)];
    let m = matched_segments(&ours, &theirs, 0.0);
    assert!(m.is_empty(), "Zero-length segment should not match");
}

#[test]
fn test_matched_segments_contained_different_fracs() {
    // Small segment completely inside large one
    let ours = vec![(150, 170)]; // 20bp
    let theirs = vec![(100, 300)]; // 200bp
    // overlap=20bp, shorter=20bp → frac=1.0 → always matches
    let m1 = matched_segments(&ours, &theirs, 0.9);
    assert_eq!(m1.len(), 1, "Small inside large should match at 0.9");

    // But reversed: ours is large, theirs is small
    // overlap=20bp, shorter=20bp → frac=1.0
    let m2 = matched_segments(&theirs, &ours, 0.9);
    assert_eq!(m2.len(), 1, "Large containing small should also match at 0.9");
}

// =============================================
// Integration: matched_segments → length_correlation pipeline
// =============================================

#[test]
fn test_matched_segments_to_length_correlation() {
    // Realistic validation: 3 our segments, 3 truth segments
    // Two match, one each is unmatched
    let ours = vec![
        (1_000_000, 3_000_000),   // 2Mb — matches truth[0]
        (5_000_000, 8_000_000),   // 3Mb — matches truth[1]
        (15_000_000, 16_000_000), // 1Mb — no match in truth
    ];
    let theirs = vec![
        (1_500_000, 3_500_000),   // 2Mb — matches ours[0]
        (5_000_000, 9_000_000),   // 4Mb — matches ours[1]
        (30_000_000, 32_000_000), // 2Mb — no match in ours
    ];

    // Find matches with 0.3 reciprocal overlap threshold
    let matches = matched_segments(&ours, &theirs, 0.3);

    // ours[0]=(1M,3M) vs theirs[0]=(1.5M,3.5M): overlap=1.5M, shorter=2M, frac=0.75 → match
    // ours[1]=(5M,8M) vs theirs[1]=(5M,9M): overlap=3M, shorter=3M, frac=1.0 → match
    // ours[2] vs theirs[2]: no overlap
    assert_eq!(matches.len(), 2);
    assert!(matches.contains(&(0, 0)));
    assert!(matches.contains(&(1, 1)));

    // Build matched intervals for length correlation
    let matched_pairs: Vec<MatchedInterval> = matches
        .iter()
        .map(|&(i, j)| (ours[i], theirs[j]))
        .collect();

    let r = length_correlation(&matched_pairs);
    // Lengths: ours=[2M, 3M], theirs=[2M, 4M]
    // These are positively correlated (larger ours → larger theirs)
    assert!(r > 0.0, "Expected positive correlation, got {}", r);
    assert!(r <= 1.0, "Correlation should be <= 1.0, got {}", r);
}

#[test]
fn test_full_validation_pipeline() {
    // End-to-end: segments → jaccard + precision/recall + concordance + matched + correlation
    let ours = vec![(100, 300), (500, 700)]; // 2 segments, 400bp total
    let theirs = vec![(200, 400), (500, 800)]; // 2 segments, 500bp total
    let region = (0, 1000);
    let window_size = 50;

    // Jaccard
    let j = segments_jaccard(&ours, &theirs, region);
    // ours merged: [100,300) + [500,700) = 400bp
    // theirs merged: [200,400) + [500,800) = 500bp
    // intersection: [200,300)=100 + [500,700)=200 = 300bp
    // union: 400+500-300 = 600bp
    // Jaccard = 300/600 = 0.5
    assert!((j - 0.5).abs() < 1e-9, "Expected Jaccard=0.5, got {}", j);

    // Precision & recall
    let (p, r) = segments_precision_recall(&ours, &theirs, region);
    // precision = 300/400 = 0.75
    assert!((p - 0.75).abs() < 1e-9, "Expected precision=0.75, got {}", p);
    // recall = 300/500 = 0.6
    assert!((r - 0.6).abs() < 1e-9, "Expected recall=0.6, got {}", r);

    // F1
    let f = f1_score(p, r);
    let expected_f1 = 2.0 * 0.75 * 0.6 / (0.75 + 0.6);
    assert!((f - expected_f1).abs() < 1e-9, "Expected F1={}, got {}", expected_f1, f);

    // Window concordance
    let c = per_window_concordance(&ours, &theirs, region, window_size);
    assert!(c > 0.5, "Expected >50% concordance, got {}", c);
    assert!(c < 1.0, "Expected <100% concordance, got {}", c);

    // Matched segments
    let matches = matched_segments(&ours, &theirs, 0.3);
    assert!(!matches.is_empty(), "Should have at least one match");
}

// =============================================
// boundary_accuracy edge cases
// =============================================

#[test]
fn test_boundary_accuracy_empty() {
    let result = boundary_accuracy(&[], 50_000);
    assert!(result.is_none(), "Empty matches → None");
}

#[test]
fn test_boundary_accuracy_perfect_match() {
    let matches: Vec<MatchedInterval> = vec![
        ((1000, 5000), (1000, 5000)),
        ((10000, 20000), (10000, 20000)),
    ];
    let ba = boundary_accuracy(&matches, 100).expect("Should return stats");
    assert_eq!(ba.n_matched, 2);
    assert!((ba.mean_start_distance_bp - 0.0).abs() < 1e-9, "Perfect → 0 distance");
    assert!((ba.mean_end_distance_bp - 0.0).abs() < 1e-9);
    assert!((ba.median_start_distance_bp - 0.0).abs() < 1e-9);
    assert!((ba.median_end_distance_bp - 0.0).abs() < 1e-9);
    assert_eq!(ba.max_start_distance_bp, 0);
    assert_eq!(ba.max_end_distance_bp, 0);
    assert!((ba.frac_start_within_threshold - 1.0).abs() < 1e-9);
    assert!((ba.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_known_distances() {
    // Ours: [100, 500), Theirs: [200, 600) → start_dist=100, end_dist=100
    // Ours: [1000, 3000), Theirs: [1000, 4000) → start_dist=0, end_dist=1000
    let matches: Vec<MatchedInterval> = vec![
        ((100, 500), (200, 600)),
        ((1000, 3000), (1000, 4000)),
    ];
    let ba = boundary_accuracy(&matches, 500).expect("Should return stats");
    assert_eq!(ba.n_matched, 2);
    // mean start = (100 + 0) / 2 = 50
    assert!((ba.mean_start_distance_bp - 50.0).abs() < 1e-9);
    // mean end = (100 + 1000) / 2 = 550
    assert!((ba.mean_end_distance_bp - 550.0).abs() < 1e-9);
    assert_eq!(ba.max_start_distance_bp, 100);
    assert_eq!(ba.max_end_distance_bp, 1000);
    // threshold=500: starts [100,0] → both within → 1.0
    assert!((ba.frac_start_within_threshold - 1.0).abs() < 1e-9);
    // threshold=500: ends [100,1000] → only first within → 0.5
    assert!((ba.frac_end_within_threshold - 0.5).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_single_match() {
    let matches: Vec<MatchedInterval> = vec![((0, 1000), (500, 2000))];
    let ba = boundary_accuracy(&matches, 1000).expect("Should return stats");
    assert_eq!(ba.n_matched, 1);
    assert!((ba.mean_start_distance_bp - 500.0).abs() < 1e-9);
    assert!((ba.mean_end_distance_bp - 1000.0).abs() < 1e-9);
    // median = mean for n=1
    assert!((ba.median_start_distance_bp - 500.0).abs() < 1e-9);
    assert!((ba.median_end_distance_bp - 1000.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_zero_threshold() {
    let matches: Vec<MatchedInterval> = vec![
        ((100, 500), (200, 600)),
    ];
    let ba = boundary_accuracy(&matches, 0).expect("Should return stats");
    // Non-zero distances → frac within threshold=0 → 0.0
    assert!((ba.frac_start_within_threshold - 0.0).abs() < 1e-9);
    assert!((ba.frac_end_within_threshold - 0.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_median_odd_count() {
    // 3 matches: median = middle value
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (10, 100)),     // start_dist=10
        ((200, 400), (250, 400)),  // start_dist=50
        ((500, 700), (600, 700)),  // start_dist=100
    ];
    let ba = boundary_accuracy(&matches, 1000).expect("Should return stats");
    assert_eq!(ba.n_matched, 3);
    // sorted start_distances: [10, 50, 100], median = 50
    assert!((ba.median_start_distance_bp - 50.0).abs() < 1e-9);
}

// =============================================
// extract_haplotype_index edge cases
// =============================================

#[test]
fn test_extract_haplotype_standard() {
    assert_eq!(extract_haplotype_index("HG00280#2#JBHDWB010000002.1"), Some(2));
    assert_eq!(extract_haplotype_index("HG00280#1#JBHDWB010000002.1"), Some(1));
}

#[test]
fn test_extract_haplotype_with_range() {
    assert_eq!(
        extract_haplotype_index("HG00280#1#JBHDWB010000002.1:130787850-130792849"),
        Some(1)
    );
}

#[test]
fn test_extract_haplotype_no_hash() {
    assert_eq!(extract_haplotype_index("HG00280"), None);
}

#[test]
fn test_extract_haplotype_non_numeric() {
    // If haplotype index isn't a valid u8
    assert_eq!(extract_haplotype_index("HG00280#abc#CONTIG"), None);
}

#[test]
fn test_extract_haplotype_large_index() {
    // u8 max is 255
    assert_eq!(extract_haplotype_index("HG00280#255#CONTIG"), Some(255));
    assert_eq!(extract_haplotype_index("HG00280#256#CONTIG"), None); // overflow
}

#[test]
fn test_extract_haplotype_zero_index() {
    assert_eq!(extract_haplotype_index("HG00280#0#CONTIG"), Some(0));
}

#[test]
fn test_extract_haplotype_empty_string() {
    assert_eq!(extract_haplotype_index(""), None);
}

// =============================================
// extract_sample_id edge cases
// =============================================

#[test]
fn test_extract_sample_id_standard() {
    assert_eq!(extract_sample_id("HG00280#2#JBHDWB010000002.1"), "HG00280");
}

#[test]
fn test_extract_sample_id_no_hash() {
    assert_eq!(extract_sample_id("HG00280"), "HG00280");
}

#[test]
fn test_extract_sample_id_empty() {
    assert_eq!(extract_sample_id(""), "");
}

#[test]
fn test_extract_sample_id_multiple_hashes() {
    // Only first part before # is the sample
    assert_eq!(extract_sample_id("NA20503#1#CONTIG#extra"), "NA20503");
}

// =============================================
// haplotype_level_concordance edge cases
// =============================================

#[test]
fn test_haplotype_concordance_empty_both() {
    let ours: Vec<(String, String, u64, u64)> = vec![];
    let theirs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 1000000));
    assert!(result.is_none(), "Empty data → None");
}

#[test]
fn test_haplotype_concordance_ours_only() {
    // We have segments but hap-ibd has none for this pair
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#2#C".to_string(), 1000, 5000),
    ];
    let theirs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    assert!(result.is_some(), "Has our data → Some");
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 1);
    assert_eq!(r.n_hapibd_hap_combos, 0);
    assert!((r.best_jaccard - 0.0).abs() < 1e-9, "No theirs → Jaccard 0");
}

#[test]
fn test_haplotype_concordance_theirs_only() {
    let ours: Vec<(String, String, u64, u64)> = vec![];
    let theirs = vec![
        ("HG00280".to_string(), 1_u8, "HG00323".to_string(), 2_u8, 1000_u64, 5000_u64),
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    assert!(result.is_some());
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 0);
    assert_eq!(r.n_hapibd_hap_combos, 1);
}

#[test]
fn test_haplotype_concordance_perfect_same_haplotypes() {
    // Same segments on same haplotype combination
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#2#C".to_string(), 1000, 5000),
    ];
    let theirs = vec![
        ("HG00280".to_string(), 1_u8, "HG00323".to_string(), 2_u8, 1000_u64, 5000_u64),
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    let r = result.unwrap();
    assert!((r.best_jaccard - 1.0).abs() < 1e-6, "Perfect match → Jaccard 1.0, got {}", r.best_jaccard);
    assert!((r.best_f1 - 1.0).abs() < 1e-6);
    assert!((r.sample_level_jaccard - 1.0).abs() < 1e-6);
}

#[test]
fn test_haplotype_concordance_misphased() {
    // Same region but different haplotype combination → hap-level Jaccard 0, sample-level 1
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#2#C".to_string(), 1000, 5000),
    ];
    let theirs = vec![
        // Different haplotype combo (2,1 instead of 1,2)
        ("HG00280".to_string(), 2_u8, "HG00323".to_string(), 1_u8, 1000_u64, 5000_u64),
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    let r = result.unwrap();
    // Haplotype-level: different combos → Jaccard 0 for each combo
    assert!((r.best_jaccard - 0.0).abs() < 1e-9, "Misphased → hap Jaccard 0");
    // Sample-level: same region → Jaccard 1
    assert!((r.sample_level_jaccard - 1.0).abs() < 1e-6, "Same region → sample Jaccard 1");
}

#[test]
fn test_haplotype_concordance_reversed_pair_order() {
    // hap-ibd has (sample2, sample1) order — should still match
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#2#C".to_string(), 1000, 5000),
    ];
    let theirs = vec![
        // Reversed sample order: HG00323 first, HG00280 second
        ("HG00323".to_string(), 2_u8, "HG00280".to_string(), 1_u8, 1000_u64, 5000_u64),
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    let r = result.unwrap();
    // Reversed order should normalize to same combo
    assert!((r.best_jaccard - 1.0).abs() < 1e-6, "Reversed order → still matches");
}

#[test]
fn test_haplotype_concordance_unrelated_pair_filtered() {
    // Segments from a different pair should not affect result
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#2#C".to_string(), 1000, 5000),
        ("HG99999#1#C".to_string(), "HG88888#2#C".to_string(), 1000, 5000), // unrelated
    ];
    let theirs = vec![
        ("HG00280".to_string(), 1_u8, "HG00323".to_string(), 2_u8, 1000_u64, 5000_u64),
        ("HG99999".to_string(), 1_u8, "HG88888".to_string(), 2_u8, 1000_u64, 5000_u64), // unrelated
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    let r = result.unwrap();
    // Should only see the HG00280/HG00323 pair
    assert_eq!(r.n_our_hap_combos, 1);
    assert_eq!(r.n_hapibd_hap_combos, 1);
}

#[test]
fn test_haplotype_concordance_multiple_hap_combos() {
    // Multiple haplotype combinations from same pair
    let ours = vec![
        ("HG00280#1#C".to_string(), "HG00323#1#C".to_string(), 1000, 5000),
        ("HG00280#2#C".to_string(), "HG00323#2#C".to_string(), 6000, 9000),
    ];
    let theirs = vec![
        ("HG00280".to_string(), 1_u8, "HG00323".to_string(), 1_u8, 1000_u64, 5000_u64),
        ("HG00280".to_string(), 2_u8, "HG00323".to_string(), 2_u8, 6000_u64, 9000_u64),
    ];
    let result = haplotype_level_concordance(&ours, &theirs, "HG00280", "HG00323", (0, 10000));
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 2);
    assert_eq!(r.n_hapibd_hap_combos, 2);
    assert_eq!(r.per_hap_combo.len(), 2);
    // Both combos are perfect matches
    assert!((r.best_jaccard - 1.0).abs() < 1e-6);
    assert!((r.best_f1 - 1.0).abs() < 1e-6);
}
