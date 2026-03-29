//! Edge case tests for the hap-ibd parser module.
//!
//! These tests cover edge cases not covered by the inline tests in hapibd.rs:
//! - Empty file → empty vec
//! - Header-only file → empty vec
//! - Malformed lines (missing fields, bad types) → skipped
//! - Duplicate segments → all parsed
//! - Very large coordinates → no overflow
//! - Negative LOD scores → parsed correctly
//! - Multiple chromosomes → filter by chr works
//! - Sample pair filtering → correct subset

use hprc_ibd::hapibd::*;

// --- Empty and header-only inputs ---

#[test]
fn test_parse_empty_string() {
    let segments = parse_hapibd_content("");
    assert!(segments.is_empty());
}

#[test]
fn test_parse_whitespace_only() {
    let segments = parse_hapibd_content("   \n  \n\t\n");
    assert!(segments.is_empty());
}

#[test]
fn test_parse_only_header_lines() {
    let content = "# hap-ibd output v2.0\n# sample1\thap1\tsample2\thap2\tchr\tstart\tend\tLOD\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_parse_only_comments_and_blanks() {
    let content = "# comment 1\n\n# comment 2\n  \n# comment 3\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

// --- Malformed line handling ---

#[test]
fn test_malformed_too_few_fields() {
    // Only 5 fields instead of 8
    let content = "HG001\t1\tHG002\t2\tchr1\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_malformed_non_numeric_hap() {
    // hap1 = "X" instead of integer
    let content = "HG001\tX\tHG002\t2\tchr1\t100\t200\t5.0\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_malformed_non_numeric_start() {
    // start = "abc"
    let content = "HG001\t1\tHG002\t2\tchr1\tabc\t200\t5.0\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_malformed_non_numeric_end() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\txyz\t5.0\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_malformed_non_numeric_lod() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\tNaN_bad\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty());
}

#[test]
fn test_malformed_mixed_with_valid() {
    // 3 lines: valid, malformed, valid → should get 2
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0
bad_line
HG003\t1\tHG004\t2\tchr1\t300\t400\t6.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].sample1, "HG001");
    assert_eq!(segments[1].sample1, "HG003");
}

#[test]
fn test_malformed_extra_fields_still_parses() {
    // 9 fields instead of 8 — should still parse (extra ignored)
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\textra_field\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].sample1, "HG001");
}

// --- Duplicate segments ---

#[test]
fn test_duplicate_segments_all_parsed() {
    let line = "HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n";
    let content = format!("{}{}{}", line, line, line);
    let segments = parse_hapibd_content(&content);
    assert_eq!(segments.len(), 3);
    // All should have same values
    for seg in &segments {
        assert_eq!(seg.sample1, "HG001");
        assert_eq!(seg.start, 100);
        assert_eq!(seg.end, 200);
    }
}

// --- Very large coordinates ---

#[test]
fn test_large_coordinates_chr1() {
    // chr1 is ~248,956,422 bp; test with coordinates near that
    let content = "HG001\t1\tHG002\t2\tchr1\t240000000\t248956422\t10.5\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 240_000_000);
    assert_eq!(segments[0].end, 248_956_422);
    assert_eq!(segments[0].length_bp(), 8_956_422);
}

#[test]
fn test_very_large_coordinates_no_overflow() {
    // Test with coordinates > 4 billion (u32 max) to ensure u64 works
    let content = "HG001\t1\tHG002\t2\tchr1\t5000000000\t6000000000\t10.0\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 5_000_000_000);
    assert_eq!(segments[0].end, 6_000_000_000);
    assert_eq!(segments[0].length_bp(), 1_000_000_000);
}

#[test]
fn test_zero_length_segment() {
    // start == end
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t100\t3.0\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].length_bp(), 0);
}

// --- Negative and zero LOD scores ---

#[test]
fn test_negative_lod_score() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t-3.5\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod - (-3.5)).abs() < 1e-9);
}

#[test]
fn test_zero_lod_score() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t0.0\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod - 0.0).abs() < 1e-9);
}

#[test]
fn test_very_high_lod_score() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t999.99\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod - 999.99).abs() < 1e-9);
}

#[test]
fn test_lod_filtering_with_negative() {
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\t-3.5
HG001\t1\tHG002\t2\tchr1\t300\t400\t0.0
HG001\t1\tHG002\t2\tchr1\t500\t600\t5.0
HG001\t1\tHG002\t2\tchr1\t700\t800\t15.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 4);

    // LOD >= 0
    let above_zero = hapibd_segments_above_lod(&segments, 0.0);
    assert_eq!(above_zero.len(), 3); // 0.0, 5.0, 15.0

    // LOD >= 5
    let above_five = hapibd_segments_above_lod(&segments, 5.0);
    assert_eq!(above_five.len(), 2); // 5.0, 15.0

    // LOD >= -10 (should get all)
    let above_neg10 = hapibd_segments_above_lod(&segments, -10.0);
    assert_eq!(above_neg10.len(), 4);
}

// --- Multiple chromosomes ---

#[test]
fn test_multiple_chromosomes_filter() {
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0
HG001\t1\tHG002\t2\tchr20\t300\t400\t6.0
HG001\t1\tHG002\t2\tchr1\t500\t600\t7.0
HG001\t1\tHG002\t2\tchrX\t700\t800\t8.0
HG001\t1\tHG002\t2\tchr20\t900\t1000\t9.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 5);

    let chr1 = hapibd_segments_for_chr(&segments, "chr1");
    assert_eq!(chr1.len(), 2);

    let chr20 = hapibd_segments_for_chr(&segments, "chr20");
    assert_eq!(chr20.len(), 2);

    let chrx = hapibd_segments_for_chr(&segments, "chrX");
    assert_eq!(chrx.len(), 1);

    let chr2 = hapibd_segments_for_chr(&segments, "chr2");
    assert!(chr2.is_empty());
}

// --- Sample pair filtering ---

#[test]
fn test_pair_filtering_order_independent() {
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0
HG002\t1\tHG001\t2\tchr1\t300\t400\t6.0
HG001\t1\tHG003\t2\tchr1\t500\t600\t7.0
";
    let segments = parse_hapibd_content(content);

    // HG001-HG002 in both orders
    let pair_ab = hapibd_segments_for_pair(&segments, "HG001", "HG002");
    assert_eq!(pair_ab.len(), 2);

    let pair_ba = hapibd_segments_for_pair(&segments, "HG002", "HG001");
    assert_eq!(pair_ba.len(), 2);

    // HG001-HG003
    let pair_ac = hapibd_segments_for_pair(&segments, "HG001", "HG003");
    assert_eq!(pair_ac.len(), 1);
}

#[test]
fn test_pair_filtering_no_self_pairs() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n";
    let segments = parse_hapibd_content(content);

    // Searching for same sample as both sides → empty
    let self_pair = hapibd_segments_for_pair(&segments, "HG001", "HG001");
    assert!(self_pair.is_empty());
}

// --- Unique pairs ---

#[test]
fn test_unique_pairs_canonical_order() {
    let content = "\
HG002\t1\tHG001\t2\tchr1\t100\t200\t5.0
HG001\t1\tHG002\t2\tchr1\t300\t400\t6.0
HG003\t1\tHG001\t2\tchr1\t500\t600\t7.0
";
    let segments = parse_hapibd_content(content);
    let pairs = unique_pairs(&segments);

    assert_eq!(pairs.len(), 2);
    // All pairs should be in canonical (sorted) order
    for (a, b) in &pairs {
        assert!(a <= b, "Pair should be in canonical order: ({}, {})", a, b);
    }
    assert!(pairs.contains(&("HG001".to_string(), "HG002".to_string())));
    assert!(pairs.contains(&("HG001".to_string(), "HG003".to_string())));
}

#[test]
fn test_unique_pairs_empty_input() {
    let segments: Vec<HapIbdSegment> = vec![];
    let pairs = unique_pairs(&segments);
    assert!(pairs.is_empty());
}

// --- Haplotype indices ---

#[test]
fn test_haplotype_indices_1_and_2() {
    let content = "\
HG001\t1\tHG002\t1\tchr1\t100\t200\t5.0
HG001\t1\tHG002\t2\tchr1\t300\t400\t6.0
HG001\t2\tHG002\t1\tchr1\t500\t600\t7.0
HG001\t2\tHG002\t2\tchr1\t700\t800\t8.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 4);
    assert_eq!(segments[0].hap1, 1);
    assert_eq!(segments[0].hap2, 1);
    assert_eq!(segments[1].hap2, 2);
    assert_eq!(segments[2].hap1, 2);
    assert_eq!(segments[3].hap1, 2);
    assert_eq!(segments[3].hap2, 2);
}

// --- as_interval and involves_sample ---

#[test]
fn test_as_interval_consistency() {
    let seg = HapIbdSegment {
        sample1: "A".into(),
        hap1: 1,
        sample2: "B".into(),
        hap2: 1,
        chr: "chr1".into(),
        start: 1_000_000,
        end: 5_000_000,
        lod: 10.0,
    };
    let (s, e) = seg.as_interval();
    assert_eq!(s, seg.start);
    assert_eq!(e, seg.end);
    assert_eq!(e - s, seg.length_bp());
}

#[test]
fn test_involves_sample_neither() {
    let seg = HapIbdSegment {
        sample1: "HG001".into(),
        hap1: 1,
        sample2: "HG002".into(),
        hap2: 1,
        chr: "chr1".into(),
        start: 100,
        end: 200,
        lod: 5.0,
    };
    assert!(!seg.involves_sample("HG003"));
    assert!(!seg.involves_sample(""));
}

// --- Tab vs space separation ---

#[test]
fn test_space_separated_not_parsed() {
    // hap-ibd uses tabs; spaces should cause parse failure
    let content = "HG001 1 HG002 2 chr1 100 200 5.0\n";
    let segments = parse_hapibd_content(content);
    assert!(segments.is_empty(), "Space-separated lines should not parse as tab-separated");
}

// --- File-based parsing errors ---

#[test]
fn test_parse_nonexistent_file() {
    let result = parse_hapibd_file("/nonexistent/path/to/file.ibd");
    assert!(result.is_err());
}

// --- Stress test: many segments ---

#[test]
fn test_many_segments() {
    let mut content = String::new();
    for i in 0..1000 {
        content.push_str(&format!(
            "HG{:04}\t1\tHG{:04}\t2\tchr1\t{}\t{}\t{:.1}\n",
            i / 50,
            i % 50,
            i * 10000,
            (i + 1) * 10000,
            i as f64 * 0.1
        ));
    }
    let segments = parse_hapibd_content(&content);
    assert_eq!(segments.len(), 1000);
}

// --- Scientific notation LOD ---

#[test]
fn test_scientific_notation_lod() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t1.5e2\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod - 150.0).abs() < 1e-9);
}

#[test]
fn test_scientific_notation_small_lod() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t3.7e-4\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod - 0.00037).abs() < 1e-9);
}

// --- Special float values for LOD ---

#[test]
fn test_inf_lod_parsed() {
    // Rust's f64::parse accepts "inf"
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\tinf\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].lod.is_infinite());
    assert!(segments[0].lod > 0.0);
}

#[test]
fn test_neg_inf_lod_parsed() {
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t-inf\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].lod.is_infinite());
    assert!(segments[0].lod < 0.0);
}

#[test]
fn test_nan_lod_parsed() {
    // Rust's f64::parse accepts "NaN"
    let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\tNaN\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].lod.is_nan());
}

// --- Start > End (degenerate segment) ---

#[test]
fn test_start_greater_than_end() {
    // Parser accepts it; length_bp uses saturating_sub so returns 0
    let content = "HG001\t1\tHG002\t2\tchr1\t500\t100\t5.0\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 500);
    assert_eq!(segments[0].end, 100);
    assert_eq!(segments[0].length_bp(), 0); // saturating_sub handles this
}

// --- Combined chr + pair filtering ---

#[test]
fn test_combined_chr_and_pair_filter() {
    let content = "\
HG001\t1\tHG002\t2\tchr10\t100\t200\t5.0
HG001\t1\tHG002\t2\tchr11\t300\t400\t6.0
HG001\t1\tHG003\t2\tchr10\t500\t600\t7.0
HG002\t1\tHG003\t2\tchr10\t700\t800\t8.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 4);

    // Filter by pair first, then by chr
    let pair_segs = hapibd_segments_for_pair(&segments, "HG001", "HG002");
    assert_eq!(pair_segs.len(), 2);

    // Manually filter pair_segs by chr10
    let pair_chr10: Vec<_> = pair_segs.iter().filter(|s| s.chr == "chr10").collect();
    assert_eq!(pair_chr10.len(), 1);
    assert_eq!(pair_chr10[0].start, 100);
}

// --- LOD filtering edge: exact threshold match ---

#[test]
fn test_lod_filter_exact_threshold() {
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\t3.0
HG001\t1\tHG002\t2\tchr1\t300\t400\t3.0
HG001\t1\tHG002\t2\tchr1\t500\t600\t2.99
";
    let segments = parse_hapibd_content(content);
    let above = hapibd_segments_above_lod(&segments, 3.0);
    // 3.0 >= 3.0 is true, 2.99 >= 3.0 is false
    assert_eq!(above.len(), 2);
}

// --- NaN LOD and filtering behavior ---

#[test]
fn test_nan_lod_filtered_out_by_threshold() {
    let content = "\
HG001\t1\tHG002\t2\tchr1\t100\t200\tNaN
HG001\t1\tHG002\t2\tchr1\t300\t400\t5.0
";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 2);

    // NaN >= 3.0 is false in IEEE 754
    let above = hapibd_segments_above_lod(&segments, 3.0);
    assert_eq!(above.len(), 1);
    assert_eq!(above[0].start, 300);
}

// --- Unique pairs with many duplicates ---

#[test]
fn test_unique_pairs_many_duplicates() {
    let mut content = String::new();
    for _ in 0..100 {
        content.push_str("HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n");
    }
    for _ in 0..50 {
        content.push_str("HG002\t1\tHG001\t2\tchr1\t300\t400\t6.0\n");
    }
    let segments = parse_hapibd_content(&content);
    assert_eq!(segments.len(), 150);

    let pairs = unique_pairs(&segments);
    // Only one unique pair: (HG001, HG002)
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("HG001".to_string(), "HG002".to_string()));
}

// --- Haplotype index 0 ---

#[test]
fn test_haplotype_index_zero() {
    // hap-ibd typically uses 1/2, but 0 is valid for u8 parse
    let content = "HG001\t0\tHG002\t0\tchr1\t100\t200\t5.0\n";
    let segments = parse_hapibd_content(content);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].hap1, 0);
    assert_eq!(segments[0].hap2, 0);
}

// --- Segment equality ---

#[test]
fn test_segment_equality() {
    let seg1 = HapIbdSegment {
        sample1: "A".into(),
        hap1: 1,
        sample2: "B".into(),
        hap2: 2,
        chr: "chr1".into(),
        start: 100,
        end: 200,
        lod: 5.0,
    };
    let seg2 = seg1.clone();
    assert_eq!(seg1, seg2);

    let seg3 = HapIbdSegment {
        sample1: "A".into(),
        hap1: 1,
        sample2: "B".into(),
        hap2: 2,
        chr: "chr1".into(),
        start: 100,
        end: 200,
        lod: 5.1, // different LOD
    };
    assert_ne!(seg1, seg3);
}
