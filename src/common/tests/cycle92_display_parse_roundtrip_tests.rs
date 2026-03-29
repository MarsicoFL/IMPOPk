//! Cycle 92 edge-case tests for Region::fmt (Display, 2 test-file refs before this).

use hprc_common::Region;

// ===================== Region Display (fmt) =====================

#[test]
fn region_display_basic() {
    let r = Region {
        chrom: "chr1".to_string(),
        start: 1000,
        end: 2000,
    };
    assert_eq!(format!("{}", r), "chr1:1000-2000");
}

#[test]
fn region_display_large_coords() {
    let r = Region {
        chrom: "chr1".to_string(),
        start: 1,
        end: 248_956_422,
    };
    assert_eq!(format!("{}", r), "chr1:1-248956422");
}

#[test]
fn region_display_zero_start() {
    let r = Region {
        chrom: "chr20".to_string(),
        start: 0,
        end: 100,
    };
    assert_eq!(format!("{}", r), "chr20:0-100");
}

#[test]
fn region_display_same_start_end() {
    let r = Region {
        chrom: "chrX".to_string(),
        start: 500,
        end: 500,
    };
    assert_eq!(format!("{}", r), "chrX:500-500");
}

#[test]
fn region_display_chrm() {
    let r = Region {
        chrom: "chrM".to_string(),
        start: 1,
        end: 16569,
    };
    assert_eq!(format!("{}", r), "chrM:1-16569");
}

#[test]
fn region_display_no_chr_prefix() {
    let r = Region {
        chrom: "20".to_string(),
        start: 100,
        end: 200,
    };
    assert_eq!(format!("{}", r), "20:100-200");
}

#[test]
fn region_display_max_u64() {
    let r = Region {
        chrom: "chr1".to_string(),
        start: u64::MAX - 1,
        end: u64::MAX,
    };
    let s = format!("{}", r);
    assert!(s.starts_with("chr1:"));
    assert!(s.contains("-"));
}

// ===================== Display → parse roundtrip =====================

#[test]
fn region_display_parse_roundtrip() {
    let original = Region {
        chrom: "chr12".to_string(),
        start: 50_000_000,
        end: 133_275_309,
    };
    let displayed = format!("{}", original);
    let parsed = Region::parse(&displayed, None).unwrap();
    assert_eq!(parsed.chrom, original.chrom);
    assert_eq!(parsed.start, original.start);
    assert_eq!(parsed.end, original.end);
}

#[test]
fn region_display_parse_roundtrip_small() {
    let original = Region {
        chrom: "chrY".to_string(),
        start: 1,
        end: 57_227_415,
    };
    let displayed = format!("{}", original);
    let parsed = Region::parse(&displayed, None).unwrap();
    assert_eq!(parsed.chrom, "chrY");
    assert_eq!(parsed.start, 1);
    assert_eq!(parsed.end, 57_227_415);
}

#[test]
fn region_to_string_matches_display() {
    let r = Region {
        chrom: "chr1".to_string(),
        start: 100,
        end: 200,
    };
    assert_eq!(r.to_string(), format!("{}", r));
}
