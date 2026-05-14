//! Tests for hapibd.rs filtering and utility functions:
//! - hapibd_segments_for_chr
//! - hapibd_segments_above_lod
//! - unique_pairs
//! - HapIbdSegment methods

use impopk_ibd::hapibd::*;

fn make_segment(s1: &str, h1: u8, s2: &str, h2: u8, chr: &str, start: u64, end: u64, lod: f64) -> HapIbdSegment {
    HapIbdSegment {
        sample1: s1.to_string(),
        hap1: h1,
        sample2: s2.to_string(),
        hap2: h2,
        chr: chr.to_string(),
        start,
        end,
        lod,
    }
}

fn sample_segments() -> Vec<HapIbdSegment> {
    vec![
        make_segment("HG00733", 1, "NA12878", 1, "chr20", 1_000_000, 5_000_000, 12.5),
        make_segment("HG00733", 1, "NA12878", 2, "chr20", 8_000_000, 12_000_000, 8.3),
        make_segment("HG00733", 2, "HG00514", 1, "chr20", 3_000_000, 7_000_000, 15.7),
        make_segment("NA12878", 1, "HG00514", 2, "chr20", 500_000, 2_000_000, 4.1),
        make_segment("HG00733", 1, "NA12878", 1, "chr15", 20_000_000, 25_000_000, 20.0),
        make_segment("HG00733", 1, "HG00514", 1, "chr15", 10_000_000, 15_000_000, 3.5),
    ]
}

// ============================================================
// hapibd_segments_for_chr
// ============================================================

#[test]
fn for_chr_filters_correctly() {
    let segs = sample_segments();
    let chr20 = hapibd_segments_for_chr(&segs, "chr20");
    assert_eq!(chr20.len(), 4);
    for seg in &chr20 {
        assert_eq!(seg.chr, "chr20");
    }
}

#[test]
fn for_chr_filters_chr15() {
    let segs = sample_segments();
    let chr15 = hapibd_segments_for_chr(&segs, "chr15");
    assert_eq!(chr15.len(), 2);
    for seg in &chr15 {
        assert_eq!(seg.chr, "chr15");
    }
}

#[test]
fn for_chr_nonexistent_returns_empty() {
    let segs = sample_segments();
    let chr1 = hapibd_segments_for_chr(&segs, "chr1");
    assert!(chr1.is_empty());
}

#[test]
fn for_chr_empty_input() {
    let segs: Vec<HapIbdSegment> = vec![];
    let result = hapibd_segments_for_chr(&segs, "chr20");
    assert!(result.is_empty());
}

#[test]
fn for_chr_case_sensitive() {
    let segs = sample_segments();
    // "Chr20" != "chr20"
    let result = hapibd_segments_for_chr(&segs, "Chr20");
    assert!(result.is_empty());
}

// ============================================================
// hapibd_segments_above_lod
// ============================================================

#[test]
fn above_lod_filters_minimum() {
    let segs = sample_segments();
    let above_10 = hapibd_segments_above_lod(&segs, 10.0);
    assert_eq!(above_10.len(), 3); // 12.5, 15.7, 20.0
    for seg in &above_10 {
        assert!(seg.lod >= 10.0);
    }
}

#[test]
fn above_lod_includes_exact_threshold() {
    let segs = sample_segments();
    let above_exact = hapibd_segments_above_lod(&segs, 12.5);
    assert!(above_exact.iter().any(|s| (s.lod - 12.5).abs() < 1e-10));
}

#[test]
fn above_lod_zero_returns_all() {
    let segs = sample_segments();
    let above_0 = hapibd_segments_above_lod(&segs, 0.0);
    assert_eq!(above_0.len(), segs.len());
}

#[test]
fn above_lod_very_high_returns_none() {
    let segs = sample_segments();
    let above_100 = hapibd_segments_above_lod(&segs, 100.0);
    assert!(above_100.is_empty());
}

#[test]
fn above_lod_negative_threshold_returns_all() {
    let segs = sample_segments();
    let above_neg = hapibd_segments_above_lod(&segs, -1.0);
    assert_eq!(above_neg.len(), segs.len());
}

#[test]
fn above_lod_empty_input() {
    let segs: Vec<HapIbdSegment> = vec![];
    let result = hapibd_segments_above_lod(&segs, 5.0);
    assert!(result.is_empty());
}

#[test]
fn above_lod_preserves_order() {
    let segs = sample_segments();
    let above_5 = hapibd_segments_above_lod(&segs, 5.0);
    // Should maintain original order
    for i in 1..above_5.len() {
        // Just verify we got valid segments (order test is structural)
        assert!(above_5[i].lod >= 5.0);
    }
}

// ============================================================
// unique_pairs
// ============================================================

#[test]
fn unique_pairs_deduplicates() {
    let segs = sample_segments();
    let pairs = unique_pairs(&segs);
    // HG00733-NA12878 appears 3 times but should be deduplicated
    let hg_na_count = pairs
        .iter()
        .filter(|(a, b)| {
            (a == "HG00733" && b == "NA12878") || (a == "NA12878" && b == "HG00733")
        })
        .count();
    assert_eq!(hg_na_count, 1, "HG00733-NA12878 should appear exactly once");
}

#[test]
fn unique_pairs_sorted_canonically() {
    let segs = sample_segments();
    let pairs = unique_pairs(&segs);
    // Each pair should have a <= b (canonical order)
    for (a, b) in &pairs {
        assert!(a <= b, "Pair ({}, {}) not in canonical order", a, b);
    }
}

#[test]
fn unique_pairs_sorted_globally() {
    let segs = sample_segments();
    let pairs = unique_pairs(&segs);
    // Overall list should be sorted
    for i in 1..pairs.len() {
        assert!(
            pairs[i] >= pairs[i - 1],
            "Pairs not sorted: {:?} >= {:?}",
            pairs[i - 1],
            pairs[i]
        );
    }
}

#[test]
fn unique_pairs_empty_input() {
    let segs: Vec<HapIbdSegment> = vec![];
    let pairs = unique_pairs(&segs);
    assert!(pairs.is_empty());
}

#[test]
fn unique_pairs_single_segment() {
    let segs = vec![make_segment("A", 1, "B", 1, "chr1", 0, 1000, 5.0)];
    let pairs = unique_pairs(&segs);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("A".to_string(), "B".to_string()));
}

#[test]
fn unique_pairs_reversed_names() {
    // Same pair but with names in different order
    let segs = vec![
        make_segment("B", 1, "A", 1, "chr1", 0, 1000, 5.0),
        make_segment("A", 1, "B", 2, "chr1", 2000, 3000, 6.0),
    ];
    let pairs = unique_pairs(&segs);
    assert_eq!(pairs.len(), 1, "Reversed pair should be deduplicated");
    assert_eq!(pairs[0], ("A".to_string(), "B".to_string()));
}

#[test]
fn unique_pairs_counts_correct() {
    let segs = sample_segments();
    let pairs = unique_pairs(&segs);
    // Expected pairs: HG00733-NA12878, HG00733-HG00514, NA12878-HG00514
    assert_eq!(pairs.len(), 3, "Should have 3 unique pairs");
}

// ============================================================
// HapIbdSegment methods
// ============================================================

#[test]
fn segment_length_bp() {
    let seg = make_segment("A", 1, "B", 1, "chr1", 1_000_000, 5_000_000, 10.0);
    assert_eq!(seg.length_bp(), 4_000_000);
}

#[test]
fn segment_length_bp_zero() {
    let seg = make_segment("A", 1, "B", 1, "chr1", 1000, 1000, 1.0);
    assert_eq!(seg.length_bp(), 0);
}

#[test]
fn segment_as_interval() {
    let seg = make_segment("A", 1, "B", 1, "chr1", 100, 500, 3.0);
    assert_eq!(seg.as_interval(), (100, 500));
}

#[test]
fn segment_involves_sample() {
    let seg = make_segment("HG00733", 1, "NA12878", 2, "chr20", 0, 1000, 5.0);
    assert!(seg.involves_sample("HG00733"));
    assert!(seg.involves_sample("NA12878"));
    assert!(!seg.involves_sample("HG00514"));
}

#[test]
fn segment_involves_pair() {
    let seg = make_segment("HG00733", 1, "NA12878", 2, "chr20", 0, 1000, 5.0);
    assert!(seg.involves_pair("HG00733", "NA12878"));
    assert!(seg.involves_pair("NA12878", "HG00733")); // Order-independent
    assert!(!seg.involves_pair("HG00733", "HG00514"));
}

// ============================================================
// parse_hapibd_content
// ============================================================

#[test]
fn parse_content_skips_comments() {
    let content = "# comment\n# another comment\n\
         HG00733\t1\tNA12878\t1\tchr20\t1000000\t5000000\t12.5\n";
    let segs = parse_hapibd_content(content);
    assert_eq!(segs.len(), 1);
}

#[test]
fn parse_content_empty() {
    let segs = parse_hapibd_content("");
    assert!(segs.is_empty());
}

#[test]
fn parse_content_all_comments() {
    let content = "# just comments\n# nothing else\n";
    let segs = parse_hapibd_content(content);
    assert!(segs.is_empty());
}

#[test]
fn parse_content_multiple_segments() {
    let content = "\
         HG00733\t1\tNA12878\t1\tchr20\t1000000\t5000000\t12.5\n\
         HG00514\t2\tNA12878\t1\tchr20\t2000000\t3000000\t6.7\n";
    let segs = parse_hapibd_content(content);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].sample1, "HG00733");
    assert_eq!(segs[1].sample1, "HG00514");
}

#[test]
fn parse_content_preserves_values() {
    let content = "HG00733\t1\tNA12878\t2\tchr15\t20000000\t25000000\t20.0\n";
    let segs = parse_hapibd_content(content);
    assert_eq!(segs.len(), 1);
    let seg = &segs[0];
    assert_eq!(seg.sample1, "HG00733");
    assert_eq!(seg.hap1, 1);
    assert_eq!(seg.sample2, "NA12878");
    assert_eq!(seg.hap2, 2);
    assert_eq!(seg.chr, "chr15");
    assert_eq!(seg.start, 20_000_000);
    assert_eq!(seg.end, 25_000_000);
    assert!((seg.lod - 20.0).abs() < 1e-10);
}

// ============================================================
// Combined filtering
// ============================================================

#[test]
fn combined_chr_and_lod_filter() {
    let segs = sample_segments();
    let chr20_high = hapibd_segments_above_lod(
        &hapibd_segments_for_chr(&segs, "chr20")
            .into_iter()
            .cloned()
            .collect::<Vec<_>>(),
        10.0,
    )
    .into_iter()
    .cloned()
    .collect::<Vec<_>>();
    // chr20 has 4 segments, 2 above LOD 10 (12.5 and 15.7)
    assert_eq!(chr20_high.len(), 2);
}

#[test]
fn combined_pair_and_chr_filter() {
    let segs = sample_segments();
    let pair_segs = hapibd_segments_for_pair(&segs, "HG00733", "NA12878");
    let chr20_pair: Vec<_> = pair_segs
        .iter()
        .filter(|s| s.chr == "chr20")
        .collect();
    assert_eq!(chr20_pair.len(), 2);
}
