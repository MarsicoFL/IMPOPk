//! Tests for IBD concordance boundary accuracy edge cases,
//! merge_intervals robustness, haplotype concordance advanced scenarios,
//! and matched_segments many-to-one patterns.

use impopk_ibd::concordance::{
    boundary_accuracy, f1_score, haplotype_level_concordance, length_correlation,
    matched_segments, per_window_concordance, segment_overlap_bp, segments_jaccard,
    segments_precision_recall, MatchedInterval,
};

// ============================================================================
// boundary_accuracy: even-count median validation
// ============================================================================

#[test]
fn boundary_accuracy_even_count_4_median() {
    // 4 matches → even count, median = average of [1] and [2]
    let matches: Vec<MatchedInterval> = vec![
        ((1000, 5000), (1100, 5100)),   // start diff=100, end diff=100
        ((2000, 6000), (2300, 6500)),   // start diff=300, end diff=500
        ((3000, 7000), (3500, 7200)),   // start diff=500, end diff=200
        ((4000, 8000), (4700, 8900)),   // start diff=700, end diff=900
    ];
    let acc = boundary_accuracy(&matches, 1000).unwrap();
    assert_eq!(acc.n_matched, 4);
    // Sorted start distances: [100, 300, 500, 700] → median = (300+500)/2 = 400
    assert!(
        (acc.median_start_distance_bp - 400.0).abs() < 1e-9,
        "Expected median_start=400, got {}",
        acc.median_start_distance_bp
    );
    // Sorted end distances: [100, 200, 500, 900] → median = (200+500)/2 = 350
    assert!(
        (acc.median_end_distance_bp - 350.0).abs() < 1e-9,
        "Expected median_end=350, got {}",
        acc.median_end_distance_bp
    );
    // Mean start: (100+300+500+700)/4 = 400
    assert!((acc.mean_start_distance_bp - 400.0).abs() < 1e-9);
    // Mean end: (100+200+500+900)/4 = 425
    assert!((acc.mean_end_distance_bp - 425.0).abs() < 1e-9);
    assert_eq!(acc.max_start_distance_bp, 700);
    assert_eq!(acc.max_end_distance_bp, 900);
}

#[test]
fn boundary_accuracy_even_count_6_median() {
    // 6 matches for another even-count validation
    let matches: Vec<MatchedInterval> = vec![
        ((100, 200), (100, 200)),   // 0, 0
        ((100, 200), (110, 220)),   // 10, 20
        ((100, 200), (130, 250)),   // 30, 50
        ((100, 200), (170, 290)),   // 70, 90
        ((100, 200), (200, 380)),   // 100, 180
        ((100, 200), (250, 500)),   // 150, 300
    ];
    let acc = boundary_accuracy(&matches, 1000).unwrap();
    assert_eq!(acc.n_matched, 6);
    // Sorted start: [0, 10, 30, 70, 100, 150] → median = (30+70)/2 = 50
    assert!(
        (acc.median_start_distance_bp - 50.0).abs() < 1e-9,
        "Expected median_start=50, got {}",
        acc.median_start_distance_bp
    );
    // Sorted end: [0, 20, 50, 90, 180, 300] → median = (50+90)/2 = 70
    assert!(
        (acc.median_end_distance_bp - 70.0).abs() < 1e-9,
        "Expected median_end=70, got {}",
        acc.median_end_distance_bp
    );
}

#[test]
fn boundary_accuracy_all_within_threshold() {
    let matches: Vec<MatchedInterval> = vec![
        ((1000, 5000), (1001, 5001)),
        ((2000, 6000), (2002, 6002)),
        ((3000, 7000), (3003, 7003)),
    ];
    let acc = boundary_accuracy(&matches, 5).unwrap();
    assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
    assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

#[test]
fn boundary_accuracy_none_within_threshold() {
    let matches: Vec<MatchedInterval> = vec![
        ((1000, 5000), (2000, 6000)), // diff=1000
        ((3000, 7000), (5000, 9000)), // diff=2000
    ];
    let acc = boundary_accuracy(&matches, 500).unwrap();
    assert!((acc.frac_start_within_threshold - 0.0).abs() < 1e-9);
    assert!((acc.frac_end_within_threshold - 0.0).abs() < 1e-9);
}

#[test]
fn boundary_accuracy_threshold_exact_equal() {
    // Distance exactly equals threshold — should count as within
    let matches: Vec<MatchedInterval> = vec![((1000, 5000), (1100, 5100))];
    let acc = boundary_accuracy(&matches, 100).unwrap();
    assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
    assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

#[test]
fn boundary_accuracy_large_distances() {
    // Very large genomic coordinates (chromosome-scale)
    let matches: Vec<MatchedInterval> = vec![
        ((10_000_000, 50_000_000), (10_500_000, 50_100_000)),
        ((100_000_000, 200_000_000), (101_000_000, 199_000_000)),
    ];
    let acc = boundary_accuracy(&matches, 2_000_000).unwrap();
    assert_eq!(acc.n_matched, 2);
    // Start distances: 500_000, 1_000_000
    assert!((acc.mean_start_distance_bp - 750_000.0).abs() < 1e-9);
    // End distances: 100_000, 1_000_000
    assert!((acc.mean_end_distance_bp - 550_000.0).abs() < 1e-9);
    // All within 2M threshold
    assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
    assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

#[test]
fn boundary_accuracy_reversed_direction() {
    // Our segments start/end BEFORE ground truth (negative direction)
    let matches: Vec<MatchedInterval> = vec![
        ((900, 4800), (1000, 5000)),   // ours before theirs
        ((1800, 5700), (2000, 6000)),
    ];
    let acc = boundary_accuracy(&matches, 500).unwrap();
    // abs_diff handles direction correctly
    assert!((acc.mean_start_distance_bp - 150.0).abs() < 1e-9); // (100+200)/2
    assert!((acc.mean_end_distance_bp - 250.0).abs() < 1e-9);   // (200+300)/2
}

// ============================================================================
// matched_segments: many-to-one and threshold edge cases
// ============================================================================

#[test]
fn matched_segments_many_to_one() {
    // Multiple of our segments match one of theirs
    let ours = vec![(100, 200), (150, 250), (300, 400)];
    let theirs = vec![(100, 250)]; // 150bp, overlaps with ours[0] (100bp overlap) and ours[1] (100bp overlap)
    let m = matched_segments(&ours, &theirs, 0.5);
    // ours[0]: 100bp overlap / min(100, 150)=100 → 1.0 ≥ 0.5 ✓
    // ours[1]: 100bp overlap / min(100, 150)=100 → 1.0 ≥ 0.5 ✓
    // ours[2]: 0bp overlap → no match
    assert_eq!(m.len(), 2);
    assert!(m.contains(&(0, 0)));
    assert!(m.contains(&(1, 0)));
}

#[test]
fn matched_segments_one_to_many() {
    // One of our large segments matches multiple of theirs
    let ours = vec![(100, 500)]; // 400bp
    let theirs = vec![(100, 200), (300, 400), (600, 700)];
    let m = matched_segments(&ours, &theirs, 0.5);
    // vs theirs[0]: 100bp overlap / min(400,100)=100 → 1.0 ✓
    // vs theirs[1]: 100bp overlap / min(400,100)=100 → 1.0 ✓
    // vs theirs[2]: 0bp overlap → no
    assert_eq!(m.len(), 2);
    assert!(m.contains(&(0, 0)));
    assert!(m.contains(&(0, 1)));
}

#[test]
fn matched_segments_barely_meets_threshold() {
    // Exactly at threshold boundary
    let ours = vec![(100, 200)]; // 100bp
    let theirs = vec![(150, 350)]; // 200bp, overlap=50bp
    // frac = 50 / min(100, 200) = 50/100 = 0.5
    let m = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m.len(), 1);
    // Just above
    let m2 = matched_segments(&ours, &theirs, 0.50001);
    assert!(m2.is_empty());
}

#[test]
fn matched_segments_empty_both() {
    let m = matched_segments(&[], &[], 0.5);
    assert!(m.is_empty());
}

#[test]
fn matched_segments_empty_ours() {
    let m = matched_segments(&[], &[(100, 200)], 0.5);
    assert!(m.is_empty());
}

#[test]
fn matched_segments_empty_theirs() {
    let m = matched_segments(&[(100, 200)], &[], 0.5);
    assert!(m.is_empty());
}

// ============================================================================
// length_correlation: uncorrelated, many pairs, constant one-side
// ============================================================================

#[test]
fn length_correlation_uncorrelated() {
    // Lengths designed to have zero correlation
    // x = [100, 200, 300], y = [200, 100, 200]
    // x_mean=200, y_mean=166.67
    // This won't be exactly zero but should be low
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (0, 200)),
        ((0, 200), (0, 100)),
        ((0, 300), (0, 200)),
    ];
    let r = length_correlation(&matches);
    // Not asserting exact 0, just that it's between -1 and 1
    assert!(r >= -1.0 && r <= 1.0);
}

#[test]
fn length_correlation_many_pairs() {
    // Many pairs with strong positive correlation
    let matches: Vec<MatchedInterval> = (1..=20)
        .map(|i| {
            let len = i as u64 * 1000;
            ((0u64, len), (0u64, len + 50)) // slightly longer
        })
        .collect();
    let r = length_correlation(&matches);
    assert!(r > 0.999, "Expected near-perfect correlation, got {}", r);
}

#[test]
fn length_correlation_constant_one_side() {
    // All ours have same length, theirs vary → r = 0
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (0, 50)),
        ((0, 100), (0, 100)),
        ((0, 100), (0, 150)),
    ];
    let r = length_correlation(&matches);
    assert!(
        (r - 0.0).abs() < 1e-9,
        "Constant x → r=0, got {}",
        r
    );
}

// ============================================================================
// segments_jaccard: complex overlapping patterns
// ============================================================================

#[test]
fn jaccard_overlapping_ours_disjoint_theirs() {
    // Our segments overlap with each other
    let ours = vec![(100, 300), (200, 400)]; // merged: [100, 400) = 300bp
    let theirs = vec![(100, 400)]; // 300bp
    let j = segments_jaccard(&ours, &theirs, (0, 1000));
    // Union=300, intersection=300 → J=1.0
    assert!(
        (j - 1.0).abs() < 1e-9,
        "Overlapping ours matching theirs should give J=1.0, got {}",
        j
    );
}

#[test]
fn jaccard_region_clips_segments() {
    // Segments extend beyond region
    let ours = vec![(0, 1000)];
    let theirs = vec![(0, 1000)];
    let j = segments_jaccard(&ours, &theirs, (200, 800));
    // Both clipped to [200, 800) = 600bp each, intersection=600, union=600
    assert!((j - 1.0).abs() < 1e-9);
}

#[test]
fn jaccard_asymmetric_region_clipping() {
    // Different segments extending beyond region in different directions
    let ours = vec![(0, 500)]; // clipped to [200, 500) = 300bp
    let theirs = vec![(300, 900)]; // clipped to [300, 800) = 500bp
    let j = segments_jaccard(&ours, &theirs, (200, 800));
    // Intersection: [300, 500) = 200bp
    // Union: ours_covered=300 + theirs_covered=500 - 200 = 600bp
    let expected = 200.0 / 600.0;
    assert!(
        (j - expected).abs() < 1e-9,
        "Expected Jaccard={:.4}, got {:.4}",
        expected,
        j
    );
}

#[test]
fn jaccard_many_small_segments() {
    // Many small segments scattered
    let ours: Vec<(u64, u64)> = (0..10).map(|i| (i * 100, i * 100 + 50)).collect();
    let theirs: Vec<(u64, u64)> = (0..10).map(|i| (i * 100 + 25, i * 100 + 75)).collect();
    let j = segments_jaccard(&ours, &theirs, (0, 1000));
    // Each pair: overlap=25bp, union=75bp. 10 pairs. Total intersection=250, total union=750
    let expected = 250.0 / 750.0;
    assert!(
        (j - expected).abs() < 1e-9,
        "Expected Jaccard={:.4}, got {:.4}",
        expected,
        j
    );
}

// ============================================================================
// segments_precision_recall: asymmetric scenarios
// ============================================================================

#[test]
fn precision_recall_overcalling() {
    // We call IBD everywhere, truth has small segments
    let ours = vec![(0, 10000)];
    let theirs = vec![(1000, 2000)]; // 1000bp
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 10000));
    // Precision = 1000/10000 = 0.1, Recall = 1000/1000 = 1.0
    assert!((p - 0.1).abs() < 1e-9, "Expected precision=0.1, got {}", p);
    assert!((r - 1.0).abs() < 1e-9, "Expected recall=1.0, got {}", r);
}

#[test]
fn precision_recall_undercalling() {
    // We call small segment, truth has large
    let ours = vec![(4000, 5000)]; // 1000bp
    let theirs = vec![(0, 10000)]; // 10000bp
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 10000));
    // Precision = 1000/1000 = 1.0, Recall = 1000/10000 = 0.1
    assert!((p - 1.0).abs() < 1e-9);
    assert!((r - 0.1).abs() < 1e-9, "Expected recall=0.1, got {}", r);
}

// ============================================================================
// per_window_concordance: fine-grained window edge cases
// ============================================================================

#[test]
fn per_window_concordance_single_bp_windows() {
    // Window size = 1bp
    let ours = vec![(0, 5)];
    let theirs = vec![(3, 8)];
    let c = per_window_concordance(&ours, &theirs, (0, 10), 1);
    // Windows 0-4: ours covers, 5-9: not
    // Windows 3-7: theirs covers, 0-2, 8-9: not
    // Agreement: windows 3,4 (both yes), 8,9 (both no) = 4 out of 10? No...
    // Window [i, i+1): ours_cov=1 or 0, half=0 (1/2=0), so ours_call = cov > 0
    // Actually half = (1-0)/2 = 0. So cov > 0 means covered.
    // ours covers 0-4 (inclusive): windows [0,1),[1,2),[2,3),[3,4),[4,5) → yes
    // theirs covers 3-7: windows [3,4),[4,5),[5,6),[6,7),[7,8) → yes
    // Concordance: both yes: 3,4 → 2; both no: 5(no ours, no theirs wait...
    // ours: yes=[0..5), theirs: yes=[3..8)
    // Window 0: ours=yes, theirs=no → disagree
    // Window 1: ours=yes, theirs=no → disagree
    // Window 2: ours=yes, theirs=no → disagree
    // Window 3: both yes → agree
    // Window 4: both yes → agree
    // Window 5: ours=no, theirs=yes → disagree
    // Window 6: ours=no, theirs=yes → disagree
    // Window 7: ours=no, theirs=yes → disagree
    // Window 8: both no → agree
    // Window 9: both no → agree
    // 4/10 = 0.4
    assert!(
        (c - 0.4).abs() < 1e-9,
        "Expected concordance=0.4, got {}",
        c
    );
}

#[test]
fn per_window_concordance_uneven_last_window() {
    // Region not evenly divisible by window size
    // Region [0, 25), window_size=10 → windows [0,10), [10,20), [20,25)
    let ours = vec![(0, 25)];
    let theirs = vec![(0, 25)];
    let c = per_window_concordance(&ours, &theirs, (0, 25), 10);
    assert!((c - 1.0).abs() < 1e-9);
}

#[test]
fn per_window_concordance_overlapping_segments() {
    // Our segments overlap — should be merged internally
    let ours = vec![(0, 60), (40, 100)]; // merged: [0, 100)
    let theirs = vec![(0, 100)];
    let c = per_window_concordance(&ours, &theirs, (0, 100), 10);
    assert!(
        (c - 1.0).abs() < 1e-9,
        "Overlapping ours should merge and agree, got {}",
        c
    );
}

// ============================================================================
// segment_overlap_bp: additional edge cases
// ============================================================================

#[test]
fn segment_overlap_bp_one_contains_other() {
    // a fully contains b
    assert_eq!(segment_overlap_bp((0, 1000), (100, 200)), 100);
    // b fully contains a
    assert_eq!(segment_overlap_bp((100, 200), (0, 1000)), 100);
}

#[test]
fn segment_overlap_bp_same_start() {
    assert_eq!(segment_overlap_bp((100, 200), (100, 300)), 100);
}

#[test]
fn segment_overlap_bp_same_end() {
    assert_eq!(segment_overlap_bp((100, 300), (200, 300)), 100);
}

#[test]
fn segment_overlap_bp_single_bp() {
    // 1bp segments
    assert_eq!(segment_overlap_bp((100, 101), (100, 101)), 1);
    assert_eq!(segment_overlap_bp((100, 101), (101, 102)), 0);
}

// ============================================================================
// f1_score: property tests
// ============================================================================

#[test]
fn f1_score_symmetry() {
    // F1(p, r) == F1(r, p)
    let f1 = f1_score(0.7, 0.3);
    let f1_rev = f1_score(0.3, 0.7);
    assert!((f1 - f1_rev).abs() < 1e-9);
}

#[test]
fn f1_score_bounded_by_min() {
    // F1 ≤ min(precision, recall) is NOT true; F1 ≤ max is true
    // Actually F1 ≤ arithmetic mean and F1 ≥ 0
    let f = f1_score(0.9, 0.1);
    assert!(f >= 0.0 && f <= 1.0);
    // F1 = 2*0.9*0.1/(0.9+0.1) = 0.18
    assert!((f - 0.18).abs() < 1e-9);
}

#[test]
fn f1_score_one_is_one() {
    // If either is 0, F1 = 0
    assert!((f1_score(1.0, 0.0) - 0.0).abs() < 1e-9);
    assert!((f1_score(0.0, 1.0) - 0.0).abs() < 1e-9);
}

// ============================================================================
// haplotype_level_concordance: advanced scenarios
// ============================================================================

#[test]
fn haplotype_concordance_only_our_data() {
    // We have segments but hap-ibd doesn't
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 1000u64, 5000u64),
    ];
    let hapibd_segs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    )
    .unwrap();
    assert_eq!(result.n_our_hap_combos, 1);
    assert_eq!(result.n_hapibd_hap_combos, 0);
    assert!((result.best_jaccard - 0.0).abs() < 1e-9);
    assert!((result.sample_level_jaccard - 0.0).abs() < 1e-9);
}

#[test]
fn haplotype_concordance_only_hapibd_data() {
    // hap-ibd has segments but we don't
    let our_segs: Vec<(String, String, u64, u64)> = vec![];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 1000u64, 5000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    )
    .unwrap();
    assert_eq!(result.n_our_hap_combos, 0);
    assert_eq!(result.n_hapibd_hap_combos, 1);
    assert!((result.best_jaccard - 0.0).abs() < 1e-9);
    assert!((result.best_f1 - 0.0).abs() < 1e-9);
}

#[test]
fn haplotype_concordance_unrelated_pair_returns_none() {
    // Data exists for other pairs, but not for the queried pair
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG003#2#C".to_string(), 1000u64, 5000u64),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG003".to_string(), 2u8, 1000u64, 5000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    );
    assert!(result.is_none());
}

#[test]
fn haplotype_concordance_all_four_combos() {
    // Both haplotypes of both samples → 4 combinations (1,1), (1,2), (2,1), (2,2)
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 1000u64, 3000u64),
        ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 5000u64, 7000u64),
        ("HG001#2#C".to_string(), "HG002#1#C".to_string(), 10000u64, 12000u64),
        ("HG001#2#C".to_string(), "HG002#2#C".to_string(), 15000u64, 17000u64),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 1000u64, 3000u64),
        ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 5000u64, 7000u64),
        ("HG001".to_string(), 2u8, "HG002".to_string(), 1u8, 10000u64, 12000u64),
        ("HG001".to_string(), 2u8, "HG002".to_string(), 2u8, 15000u64, 17000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 20000),
    )
    .unwrap();
    assert_eq!(result.n_our_hap_combos, 4);
    assert_eq!(result.n_hapibd_hap_combos, 4);
    assert!((result.best_jaccard - 1.0).abs() < 1e-9);
    assert!((result.best_f1 - 1.0).abs() < 1e-9);
    // All per-combo should have Jaccard > 0
    for combo in &result.per_hap_combo {
        assert!(combo.jaccard > 0.0, "Combo ({},{}) should have J>0", combo.hap1, combo.hap2);
    }
}

#[test]
fn haplotype_concordance_multiple_segments_per_combo() {
    // Multiple segments for the same haplotype combination
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 1000u64, 3000u64),
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 5000u64, 7000u64),
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 9000u64, 11000u64),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 1000u64, 3000u64),
        ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 5000u64, 7000u64),
        ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 9000u64, 11000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 15000),
    )
    .unwrap();
    assert_eq!(result.per_hap_combo.len(), 1); // all same combo
    assert!((result.best_jaccard - 1.0).abs() < 1e-9);
    assert_eq!(result.per_hap_combo[0].n_ours, 3);
    assert_eq!(result.per_hap_combo[0].n_theirs, 3);
}

#[test]
fn haplotype_concordance_sample_level_vs_haplotype_level() {
    // Sample-level should differ from haplotype-level when haplotypes swap
    // Our: hap(1,1) has segment at [1000,5000)
    // hap-ibd: hap(1,2) has segment at [1000,5000)
    // Haplotype-level: (1,1) J=0, (1,2) J=0 → best_jaccard=0
    // Wait, no: our (1,1) and theirs (1,2) are different combos
    // At sample level: both cover [1000,5000) → J=1.0
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 1000u64, 5000u64),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 1000u64, 5000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    )
    .unwrap();
    // Haplotype-level should be 0 (wrong haplotype)
    assert!(
        (result.best_jaccard - 0.0).abs() < 1e-9,
        "Haplotype-level Jaccard should be 0 for swapped haplotypes"
    );
    // Sample-level should be 1.0
    assert!(
        (result.sample_level_jaccard - 1.0).abs() < 1e-9,
        "Sample-level Jaccard should be 1.0"
    );
}

#[test]
fn haplotype_concordance_contig_region_suffix_ignored() {
    // Region suffix in contig names should be handled by extract_sample_id
    let our_segs = vec![
        (
            "HG00280#2#JBHDWB010000002.1:130787850-130792849".to_string(),
            "HG00323#1#JBHDWB010000015.1:5000-10000".to_string(),
            1000u64,
            5000u64,
        ),
    ];
    let hapibd_segs = vec![
        ("HG00280".to_string(), 2u8, "HG00323".to_string(), 1u8, 1000u64, 5000u64),
    ];
    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG00280", "HG00323", (0, 10000),
    )
    .unwrap();
    assert!((result.best_jaccard - 1.0).abs() < 1e-9);
}
