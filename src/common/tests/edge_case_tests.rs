//! Edge case tests for hprc-common: ColumnIndices missing column variants,
//! Region::parse edge cases, HprcError variant Display, and WindowIterator
//! with non-unit start offsets.

use hprc_common::{ColumnIndices, HprcError, Region, Window, WindowIterator};

// ═══════════════════════════════════════════════════════════════════════════
// ColumnIndices — missing column error variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn column_indices_missing_start_column() {
    let header = "chrom\tend\tgroup.a\tgroup.b\testimated.identity";
    let result = ColumnIndices::from_header(header);
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "start"),
        other => panic!("Expected MissingColumn('start'), got {:?}", other),
    }
}

#[test]
fn column_indices_missing_end_column() {
    let header = "chrom\tstart\tgroup.a\tgroup.b\testimated.identity";
    let result = ColumnIndices::from_header(header);
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "end"),
        other => panic!("Expected MissingColumn('end'), got {:?}", other),
    }
}

#[test]
fn column_indices_missing_group_a_column() {
    let header = "chrom\tstart\tend\tgroup.b\testimated.identity";
    let result = ColumnIndices::from_header(header);
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "group.a"),
        other => panic!("Expected MissingColumn('group.a'), got {:?}", other),
    }
}

#[test]
fn column_indices_missing_group_b_column() {
    let header = "chrom\tstart\tend\tgroup.a\testimated.identity";
    let result = ColumnIndices::from_header(header);
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "group.b"),
        other => panic!("Expected MissingColumn('group.b'), got {:?}", other),
    }
}

#[test]
fn column_indices_missing_multiple_returns_first_missing() {
    // Only "chrom" present — first missing in order is "estimated.identity"
    let header = "chrom";
    let result = ColumnIndices::from_header(header);
    // from_header tries estimated.identity first, so that's the first error
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "estimated.identity"),
        other => panic!("Expected MissingColumn('estimated.identity'), got {:?}", other),
    }
}

#[test]
fn column_indices_whitespace_not_tab_treated_as_single_column() {
    // Spaces are not tabs — entire string is one "column"
    let header = "chrom start end group.a group.b estimated.identity";
    let result = ColumnIndices::from_header(header);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Region::parse — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_parse_empty_string_no_length() {
    // Empty string with no colon → chrom-only path → requires region_length
    let result = Region::parse("", None);
    match result {
        Err(HprcError::InvalidParameter(msg)) => {
            assert!(msg.contains("requires --region-length"), "msg: {}", msg);
        }
        other => panic!("Expected InvalidParameter, got {:?}", other),
    }
}

#[test]
fn region_parse_empty_string_with_length() {
    // Empty string with length → chrom="" start=1 end=length
    let region = Region::parse("", Some(1000)).unwrap();
    assert_eq!(region.chrom, "");
    assert_eq!(region.start, 1);
    assert_eq!(region.end, 1000);
}

#[test]
fn region_parse_empty_end_after_dash() {
    // "chr1:100-" — empty end string fails u64 parse
    let result = Region::parse("chr1:100-", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => assert_eq!(s, "chr1:100-"),
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

#[test]
fn region_parse_empty_start_before_dash() {
    // "chr1:-100" — empty start string fails u64 parse
    let result = Region::parse("chr1:-100", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => assert_eq!(s, "chr1:-100"),
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

#[test]
fn region_parse_colon_only() {
    // ":" — chrom is empty, rest is empty, no dash → InvalidRegion
    let result = Region::parse(":", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => assert_eq!(s, ":"),
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

#[test]
fn region_parse_colon_dash_only() {
    // ":-" — chrom is empty, start is empty, end is empty → InvalidRegion
    let result = Region::parse(":-", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => assert_eq!(s, ":-"),
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

#[test]
fn region_parse_overflow_start() {
    // Start exceeds u64::MAX
    let result = Region::parse("chr1:99999999999999999999-200", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => {
            assert!(s.contains("99999999999999999999"));
        }
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

#[test]
fn region_parse_overflow_end() {
    let result = Region::parse("chr1:1-99999999999999999999", None);
    match result {
        Err(HprcError::InvalidRegion(s)) => {
            assert!(s.contains("99999999999999999999"));
        }
        other => panic!("Expected InvalidRegion, got {:?}", other),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Region::Display and roundtrip edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_display_empty_chrom_roundtrip() {
    let region = Region {
        chrom: "".to_string(),
        start: 1000,
        end: 2000,
    };
    let display = format!("{}", region);
    assert_eq!(display, ":1000-2000");
    // Roundtrip: parsing ":1000-2000" should give empty chrom
    let parsed = Region::parse(&display, None).unwrap();
    assert_eq!(parsed.chrom, "");
    assert_eq!(parsed.start, 1000);
    assert_eq!(parsed.end, 2000);
}

#[test]
fn region_display_single_position() {
    let region = Region {
        chrom: "chrX".to_string(),
        start: 42,
        end: 42,
    };
    assert_eq!(format!("{}", region), "chrX:42-42");
}

// ═══════════════════════════════════════════════════════════════════════════
// Region::to_impg_ref edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_to_impg_ref_empty_ref_name() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 100,
        end: 200,
    };
    let impg = region.to_impg_ref("");
    assert_eq!(impg, "#0#chr1:100-200");
}

#[test]
fn region_to_impg_ref_with_hash_in_ref_name() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 100,
        end: 200,
    };
    let impg = region.to_impg_ref("GRCh38#v1");
    assert_eq!(impg, "GRCh38#v1#0#chr1:100-200");
}

// ═══════════════════════════════════════════════════════════════════════════
// HprcError — Display trait for all variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hprc_error_display_parse() {
    let err = HprcError::Parse("bad data".to_string());
    let msg = format!("{}", err);
    assert_eq!(msg, "Parse error: bad data");
}

#[test]
fn hprc_error_display_external_tool() {
    let err = HprcError::ExternalTool("impg crashed".to_string());
    let msg = format!("{}", err);
    assert_eq!(msg, "External tool error: impg crashed");
}

#[test]
fn hprc_error_display_invalid_region() {
    let err = HprcError::InvalidRegion("chr1:abc".to_string());
    let msg = format!("{}", err);
    assert_eq!(msg, "Invalid region format: chr1:abc");
}

#[test]
fn hprc_error_display_missing_column() {
    let err = HprcError::MissingColumn("chrom".to_string());
    let msg = format!("{}", err);
    assert_eq!(msg, "Missing column: chrom");
}

#[test]
fn hprc_error_display_invalid_parameter() {
    let err = HprcError::InvalidParameter("bad param".to_string());
    let msg = format!("{}", err);
    assert_eq!(msg, "Invalid parameter: bad param");
}

#[test]
fn hprc_error_display_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err = HprcError::Io(io_err);
    let msg = format!("{}", err);
    assert!(msg.contains("IO error"), "msg: {}", msg);
    assert!(msg.contains("file not found"), "msg: {}", msg);
}

#[test]
fn hprc_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no access");
    let err: HprcError = io_err.into();
    match err {
        HprcError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied),
        other => panic!("Expected HprcError::Io, got {:?}", other),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ColumnIndices::max_index — edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn column_indices_max_index_identity_is_max() {
    // estimated.identity at position 10, others at 0-4
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tex1\tex2\tex3\tex4\tex5\testimated.identity";
    let cols = ColumnIndices::from_header(header).unwrap();
    assert_eq!(cols.max_index(), 10);
}

#[test]
fn column_indices_max_index_group_b_is_max() {
    let header = "estimated.identity\tchrom\tstart\tend\tgroup.a\tex\tgroup.b";
    let cols = ColumnIndices::from_header(header).unwrap();
    assert_eq!(cols.max_index(), 6);
}

// ═══════════════════════════════════════════════════════════════════════════
// WindowIterator — non-unit start offset
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn window_iterator_mid_chromosome_start() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 50001,
        end: 100000,
    };
    let windows: Vec<_> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 5);
    assert_eq!(windows[0].start, 50001);
    assert_eq!(windows[0].end, 60000);
    assert_eq!(windows[4].start, 90001);
    assert_eq!(windows[4].end, 100000);
}

#[test]
fn window_iterator_large_window_size_with_offset_start() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 999999,
        end: 1000001,
    };
    let windows: Vec<_> = WindowIterator::new(&region, 100000).collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].start, 999999);
    assert_eq!(windows[0].end, 1000001);
    assert_eq!(windows[0].length(), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Window — edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn window_length_large_region() {
    let w = Window::new(1, 1_000_000_000);
    assert_eq!(w.length(), 1_000_000_000);
}

#[test]
fn window_debug_format() {
    let w = Window::new(10, 20);
    let debug = format!("{:?}", w);
    assert!(debug.contains("10"), "debug: {}", debug);
    assert!(debug.contains("20"), "debug: {}", debug);
}

#[test]
fn window_copy_semantics() {
    let w1 = Window::new(1, 100);
    let w2 = w1; // Copy
    assert_eq!(w1.start, w2.start);
    assert_eq!(w1.end, w2.end);
}
