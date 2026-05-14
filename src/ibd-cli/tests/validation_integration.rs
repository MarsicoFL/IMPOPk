//! Integration tests for the IBD validation pipeline.
//!
//! Tests the full pipeline: parse hap-ibd output → parse our output → compute concordance metrics.
//! Uses synthetic but realistic data to verify the pipeline works end-to-end.

use impopk_ibd::concordance::*;
use impopk_ibd::hapibd::*;

/// Simulate a scenario where we have 3 hap-ibd segments and our tool finds 2 matching + 1 different.
#[test]
fn test_validation_pipeline_partial_match() {
    // Synthetic hap-ibd output: 3 IBD segments
    let hapibd_content = "\
# hap-ibd v1.0 output
HG00733\t1\tNA12878\t1\tchr20\t1000000\t5000000\t12.5
HG00733\t1\tNA12878\t1\tchr20\t8000000\t12000000\t8.3
HG00733\t1\tNA12878\t1\tchr20\t15000000\t20000000\t15.7
";

    // Parse hap-ibd output
    let hapibd_segments = parse_hapibd_content(hapibd_content);
    assert_eq!(hapibd_segments.len(), 3);

    // Our tool's segments: 2 matching + 1 extra (false positive) + 1 missing (false negative)
    let our_segments: Vec<(u64, u64)> = vec![
        (1_000_000, 5_000_000),   // Matches hapibd segment 1 exactly
        (8_500_000, 12_000_000),  // Partially matches hapibd segment 2 (shifted start)
        (22_000_000, 25_000_000), // False positive (no hapibd segment here)
        // Missing: 15M-20M (false negative)
    ];

    let hapibd_intervals: Vec<(u64, u64)> = hapibd_segments
        .iter()
        .map(|s| s.as_interval())
        .collect();

    let region = (0u64, 30_000_000);

    // --- Jaccard ---
    let jaccard = segments_jaccard(&our_segments, &hapibd_intervals, region);
    // Our coverage: 4M + 3.5M + 3M = 10.5M
    // Their coverage: 4M + 4M + 5M = 13M
    // Intersection: 4M (exact) + 3.5M (partial overlap 8.5M-12M) = 7.5M
    // Union: our(10.5M) + theirs(13M) - intersection(7.5M) = 16M
    // Jaccard = 7.5/16 = 0.46875
    assert!(jaccard > 0.4 && jaccard < 0.6, "Expected Jaccard ~0.47, got {}", jaccard);

    // --- Precision & Recall ---
    let (precision, recall) = segments_precision_recall(&our_segments, &hapibd_intervals, region);
    // Precision = intersection / our_coverage = 7.5M / 10.5M ≈ 0.714
    assert!(precision > 0.6 && precision < 0.8, "Expected precision ~0.71, got {}", precision);
    // Recall = intersection / their_coverage = 7.5M / 13M ≈ 0.577
    assert!(recall > 0.5 && recall < 0.7, "Expected recall ~0.58, got {}", recall);

    // --- F1 ---
    let f1 = f1_score(precision, recall);
    assert!(f1 > 0.5 && f1 < 0.8, "Expected F1 ~0.64, got {}", f1);

    // --- Per-window concordance ---
    let concordance = per_window_concordance(&our_segments, &hapibd_intervals, region, 100_000);
    // Both agree on most non-IBD regions + agree on segment 1
    // Disagree on the FP and FN regions
    assert!(concordance > 0.7, "Expected high concordance, got {}", concordance);

    // --- Matched segments ---
    let matches = matched_segments(&our_segments, &hapibd_intervals, 0.5);
    assert!(!matches.is_empty(), "Should match at least 1 segment");
    assert!(matches.len() <= 3, "Should not match more than 3");

    // --- Length correlation ---
    if matches.len() >= 2 {
        let matched_pairs: Vec<MatchedInterval> = matches
            .iter()
            .map(|&(i, j)| (our_segments[i], hapibd_intervals[j]))
            .collect();
        let r = length_correlation(&matched_pairs);
        // Matched segments should have correlated lengths
        assert!(r > -1.0 && r <= 1.0, "r should be in [-1,1], got {}", r);
    }
}

/// Test with perfect agreement between our tool and hap-ibd.
#[test]
fn test_validation_pipeline_perfect_agreement() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t5000000\t10000000\t20.0
HG001\t1\tHG002\t2\tchr20\t15000000\t22000000\t15.5
";

    let hapibd_segments = parse_hapibd_content(hapibd_content);
    let hapibd_intervals: Vec<(u64, u64)> = hapibd_segments
        .iter()
        .map(|s| s.as_interval())
        .collect();

    // Our segments match exactly
    let our_segments = hapibd_intervals.clone();
    let region = (0u64, 30_000_000);

    let jaccard = segments_jaccard(&our_segments, &hapibd_intervals, region);
    assert!((jaccard - 1.0).abs() < 1e-9, "Perfect match → Jaccard=1.0, got {}", jaccard);

    let (p, r) = segments_precision_recall(&our_segments, &hapibd_intervals, region);
    assert!((p - 1.0).abs() < 1e-9);
    assert!((r - 1.0).abs() < 1e-9);

    let concordance = per_window_concordance(&our_segments, &hapibd_intervals, region, 100_000);
    assert!((concordance - 1.0).abs() < 1e-9);
}

/// Test with no overlap at all.
#[test]
fn test_validation_pipeline_no_overlap() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t5000000\t10000000\t20.0
";
    let hapibd_segments = parse_hapibd_content(hapibd_content);
    let hapibd_intervals: Vec<(u64, u64)> = hapibd_segments
        .iter()
        .map(|s| s.as_interval())
        .collect();

    // Our segments are in a completely different region
    let our_segments: Vec<(u64, u64)> = vec![(20_000_000, 25_000_000)];
    let region = (0u64, 30_000_000);

    let jaccard = segments_jaccard(&our_segments, &hapibd_intervals, region);
    assert!((jaccard - 0.0).abs() < 1e-9, "No overlap → Jaccard=0.0, got {}", jaccard);

    let (p, r) = segments_precision_recall(&our_segments, &hapibd_intervals, region);
    assert!((p - 0.0).abs() < 1e-9);
    assert!((r - 0.0).abs() < 1e-9);
}

/// Test filtering by sample pair before concordance.
#[test]
fn test_validation_pipeline_with_pair_filter() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t5000000\t10000000\t20.0
HG001\t1\tHG003\t1\tchr20\t5000000\t10000000\t15.0
HG003\t2\tHG004\t1\tchr20\t1000000\t3000000\t8.0
";
    let all_segments = parse_hapibd_content(hapibd_content);
    assert_eq!(all_segments.len(), 3);

    // Filter for HG001-HG002 pair only
    let pair_segments = hapibd_segments_for_pair(&all_segments, "HG001", "HG002");
    assert_eq!(pair_segments.len(), 1);

    // Filter for HG001-HG003 pair
    let pair_segments_2 = hapibd_segments_for_pair(&all_segments, "HG001", "HG003");
    assert_eq!(pair_segments_2.len(), 1);

    // Get unique pairs
    let pairs = unique_pairs(&all_segments);
    assert_eq!(pairs.len(), 3);

    // Compute concordance for each pair separately
    let region = (0u64, 15_000_000);
    for (s1, s2) in &pairs {
        let pair_segs = hapibd_segments_for_pair(&all_segments, s1, s2);
        let intervals: Vec<(u64, u64)> = pair_segs.iter().map(|s| s.as_interval()).collect();
        // Just verify no panics with any pair
        let _j = segments_jaccard(&intervals, &intervals, region);
    }
}

/// Test LOD filtering before concordance.
#[test]
fn test_validation_pipeline_with_lod_filter() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t1000000\t5000000\t25.0
HG001\t1\tHG002\t2\tchr20\t6000000\t8000000\t3.5
HG001\t1\tHG002\t2\tchr20\t10000000\t15000000\t12.0
";
    let all_segments = parse_hapibd_content(hapibd_content);
    assert_eq!(all_segments.len(), 3);

    // Filter at LOD >= 10
    let high_lod = hapibd_segments_above_lod(&all_segments, 10.0);
    assert_eq!(high_lod.len(), 2); // 25.0 and 12.0

    // Concordance with high-LOD only should differ from all segments
    let all_intervals: Vec<(u64, u64)> = all_segments.iter().map(|s| s.as_interval()).collect();
    let high_intervals: Vec<(u64, u64)> = high_lod.iter().map(|s| s.as_interval()).collect();
    let region = (0u64, 20_000_000);

    let j_all = segments_jaccard(&all_intervals, &all_intervals, region);
    let j_high = segments_jaccard(&high_intervals, &all_intervals, region);

    assert!((j_all - 1.0).abs() < 1e-9, "Self-Jaccard should be 1.0");
    assert!(j_high < 1.0, "LOD-filtered should have lower Jaccard with full set");
}

/// Test chromosome filtering before concordance.
#[test]
fn test_validation_pipeline_with_chr_filter() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t1000000\t5000000\t10.0
HG001\t1\tHG002\t2\tchr15\t2000000\t6000000\t12.0
HG001\t1\tHG002\t2\tchr20\t8000000\t12000000\t8.0
";
    let all_segments = parse_hapibd_content(hapibd_content);
    let chr20 = hapibd_segments_for_chr(&all_segments, "chr20");
    let chr15 = hapibd_segments_for_chr(&all_segments, "chr15");

    assert_eq!(chr20.len(), 2);
    assert_eq!(chr15.len(), 1);

    // Per-chromosome concordance
    let region_20 = (0u64, 64_444_167); // chr20 length
    let intervals_20: Vec<(u64, u64)> = chr20.iter().map(|s| s.as_interval()).collect();
    let j20 = segments_jaccard(&intervals_20, &intervals_20, region_20);
    assert!((j20 - 1.0).abs() < 1e-9);
}

/// Integration test: parse hap-ibd → match segments → compute boundary accuracy.
/// Tests the full pipeline from raw hap-ibd text through to boundary statistics.
#[test]
fn test_validation_pipeline_boundary_accuracy() {
    // Simulate hap-ibd ground truth with 3 segments
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t1000000\t5000000\t20.0
HG001\t1\tHG002\t2\tchr20\t8000000\t12000000\t15.0
HG001\t1\tHG002\t2\tchr20\t18000000\t25000000\t10.0
";
    let hapibd_segments = parse_hapibd_content(hapibd_content);
    let hapibd_intervals: Vec<(u64, u64)> = hapibd_segments
        .iter()
        .map(|s| s.as_interval())
        .collect();

    // Our segments: shifted boundaries (realistic detection error)
    let our_segments: Vec<(u64, u64)> = vec![
        (1_050_000, 4_900_000),   // ~50kb start shift, ~100kb end shift
        (8_200_000, 12_100_000),  // ~200kb start shift, ~100kb end shift
        (17_800_000, 25_300_000), // ~200kb start shift, ~300kb end shift
    ];

    // Match segments (50% reciprocal overlap threshold)
    let matches = matched_segments(&our_segments, &hapibd_intervals, 0.5);
    assert_eq!(matches.len(), 3, "All 3 segments should match");

    // Build matched intervals for boundary accuracy
    let matched_pairs: Vec<MatchedInterval> = matches
        .iter()
        .map(|&(i, j)| (our_segments[i], hapibd_intervals[j]))
        .collect();

    // Compute boundary accuracy with 500kb threshold
    let ba = boundary_accuracy(&matched_pairs, 500_000).expect("Should return stats");
    assert_eq!(ba.n_matched, 3);

    // Mean start distance: (50k + 200k + 200k) / 3 = 150kb
    assert!(
        (ba.mean_start_distance_bp - 150_000.0).abs() < 1.0,
        "Expected mean start ~150kb, got {}",
        ba.mean_start_distance_bp
    );

    // Mean end distance: (100k + 100k + 300k) / 3 ≈ 166.7kb
    assert!(
        (ba.mean_end_distance_bp - 166_666.67).abs() < 1.0,
        "Expected mean end ~167kb, got {}",
        ba.mean_end_distance_bp
    );

    // All boundaries within 500kb threshold
    assert!(
        (ba.frac_start_within_threshold - 1.0).abs() < 1e-9,
        "All starts should be within 500kb"
    );
    assert!(
        (ba.frac_end_within_threshold - 1.0).abs() < 1e-9,
        "All ends should be within 500kb"
    );

    // With tighter 150kb threshold, not all should pass
    let ba_tight = boundary_accuracy(&matched_pairs, 150_000).expect("Should return stats");
    assert!(
        ba_tight.frac_start_within_threshold < 1.0 || ba_tight.frac_end_within_threshold < 1.0,
        "Tight threshold should exclude some boundaries"
    );
}

/// Integration test: boundary accuracy with LOD filtering.
/// Higher-LOD segments should have better boundary accuracy.
#[test]
fn test_validation_pipeline_boundary_accuracy_lod_stratified() {
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t1000000\t5000000\t25.0
HG001\t1\tHG002\t2\tchr20\t8000000\t12000000\t5.0
HG001\t1\tHG002\t2\tchr20\t18000000\t25000000\t20.0
";
    let all_segments = parse_hapibd_content(hapibd_content);

    // High-LOD segments have tighter boundaries in our detection
    let our_segments: Vec<(u64, u64)> = vec![
        (1_010_000, 4_990_000),   // LOD 25: very close (10kb, 10kb)
        (8_500_000, 11_500_000),  // LOD 5: loose (500kb, 500kb)
        (18_020_000, 25_050_000), // LOD 20: close (20kb, 50kb)
    ];

    // Filter hap-ibd at LOD >= 10
    let high_lod = hapibd_segments_above_lod(&all_segments, 10.0);
    assert_eq!(high_lod.len(), 2);
    let high_intervals: Vec<(u64, u64)> = high_lod.iter().map(|s| s.as_interval()).collect();

    // Match our high-LOD segments against high-LOD ground truth
    // Only segments 0 and 2 are high-LOD
    let our_high_lod: Vec<(u64, u64)> = vec![our_segments[0], our_segments[2]];
    let matches = matched_segments(&our_high_lod, &high_intervals, 0.5);
    assert_eq!(matches.len(), 2);

    let matched_pairs: Vec<MatchedInterval> = matches
        .iter()
        .map(|&(i, j)| (our_high_lod[i], high_intervals[j]))
        .collect();

    let ba = boundary_accuracy(&matched_pairs, 100_000).expect("Should return stats");
    // High-LOD boundaries: starts (10k, 20k), ends (10k, 50k)
    assert!(
        ba.mean_start_distance_bp < 50_000.0,
        "High-LOD mean start should be < 50kb, got {}",
        ba.mean_start_distance_bp
    );
    assert!(
        ba.frac_start_within_threshold > 0.99,
        "High-LOD starts should all be within 100kb"
    );
}

/// Integration test: haplotype-level concordance through the full pipeline.
/// Parses hap-ibd, constructs our segments with haplotype IDs, computes concordance.
#[test]
fn test_validation_pipeline_haplotype_concordance() {
    // hap-ibd output with haplotype-specific segments
    let hapibd_content = "\
HG00733\t1\tNA12878\t2\tchr20\t5000000\t10000000\t15.0
HG00733\t2\tNA12878\t1\tchr20\t15000000\t20000000\t12.0
";
    let hapibd_segments = parse_hapibd_content(hapibd_content);

    // Convert to the tuple format haplotype_level_concordance expects
    let hapibd_tuples: Vec<(String, u8, String, u8, u64, u64)> = hapibd_segments
        .iter()
        .map(|s| {
            (
                s.sample1.clone(),
                s.hap1,
                s.sample2.clone(),
                s.hap2,
                s.start,
                s.end,
            )
        })
        .collect();

    // Our segments with full pangenome haplotype IDs (matching haplotypes)
    let our_segments: Vec<(String, String, u64, u64)> = vec![
        (
            "HG00733#1#JAHEPP020000002.1".to_string(),
            "NA12878#2#CM089972.1".to_string(),
            5_000_000,
            10_000_000,
        ),
        (
            "HG00733#2#JAHEPQ020000001.1".to_string(),
            "NA12878#1#CM089973.1".to_string(),
            15_000_000,
            20_000_000,
        ),
    ];

    let region = (0u64, 30_000_000);
    let result = haplotype_level_concordance(
        &our_segments,
        &hapibd_tuples,
        "HG00733",
        "NA12878",
        region,
    )
    .expect("Should return concordance");

    assert_eq!(result.sample1, "HG00733");
    assert_eq!(result.sample2, "NA12878");
    assert_eq!(result.n_our_hap_combos, 2);
    assert_eq!(result.n_hapibd_hap_combos, 2);

    // Perfect haplotype-level match → Jaccard and F1 should be 1.0
    assert!(
        (result.best_jaccard - 1.0).abs() < 1e-9,
        "Perfect match → best Jaccard=1.0, got {}",
        result.best_jaccard
    );
    assert!(
        (result.best_f1 - 1.0).abs() < 1e-9,
        "Perfect match → best F1=1.0, got {}",
        result.best_f1
    );

    // Sample-level should also be 1.0 (no overlap between combos)
    assert!(
        (result.sample_level_jaccard - 1.0).abs() < 1e-9,
        "Sample-level Jaccard should be 1.0"
    );
}

/// Integration test: haplotype concordance when our segments have wrong haplotype assignment.
/// Demonstrates that haplotype-level metrics distinguish correct vs wrong phasing.
#[test]
fn test_validation_pipeline_haplotype_concordance_misphased() {
    // hap-ibd says: HG001 hap1 shares IBD with HG002 hap2
    let hapibd_content = "\
HG001\t1\tHG002\t2\tchr20\t5000000\t10000000\t15.0
";
    let hapibd_segments = parse_hapibd_content(hapibd_content);
    let hapibd_tuples: Vec<(String, u8, String, u8, u64, u64)> = hapibd_segments
        .iter()
        .map(|s| {
            (
                s.sample1.clone(),
                s.hap1,
                s.sample2.clone(),
                s.hap2,
                s.start,
                s.end,
            )
        })
        .collect();

    // Our detection assigns WRONG haplotype (hap2 instead of hap1 for HG001)
    let our_misphased: Vec<(String, String, u64, u64)> = vec![(
        "HG001#2#CONTIG.1".to_string(),
        "HG002#2#CONTIG.2".to_string(),
        5_000_000,
        10_000_000,
    )];

    // Correctly phased version
    let our_correct: Vec<(String, String, u64, u64)> = vec![(
        "HG001#1#CONTIG.1".to_string(),
        "HG002#2#CONTIG.2".to_string(),
        5_000_000,
        10_000_000,
    )];

    let region = (0u64, 30_000_000);

    let misphased = haplotype_level_concordance(
        &our_misphased,
        &hapibd_tuples,
        "HG001",
        "HG002",
        region,
    )
    .expect("Should return result");

    let correct = haplotype_level_concordance(
        &our_correct,
        &hapibd_tuples,
        "HG001",
        "HG002",
        region,
    )
    .expect("Should return result");

    // Correct phasing → perfect haplotype match
    assert!(
        (correct.best_jaccard - 1.0).abs() < 1e-9,
        "Correct phasing → Jaccard=1.0"
    );

    // Misphased → haplotype-level Jaccard = 0 (wrong combo), but sample-level = 1.0
    assert!(
        misphased.best_jaccard < 1e-9,
        "Misphased → haplotype Jaccard=0, got {}",
        misphased.best_jaccard
    );
    assert!(
        (misphased.sample_level_jaccard - 1.0).abs() < 1e-9,
        "Sample-level should still be 1.0 despite misphasing"
    );
}
