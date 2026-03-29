//! Property-based and round-trip tests for hprc-common types.
//!
//! Tests invariants that must hold for ALL inputs, not just specific examples:
//! - WindowIterator: contiguous coverage, monotonic, no gaps, total length
//! - Region: Display↔parse round-trip, to_impg_ref format
//! - ColumnIndices: duplicate columns, max_index properties
//! - Window: arithmetic edge cases

use hprc_common::{ColumnIndices, HprcError, Region, Window, WindowIterator};

// ═══════════════════════════════════════════════════════════════════════════
// WindowIterator — property-based invariants
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: test WindowIterator invariants for a given region and window_size
fn assert_window_invariants(start: u64, end: u64, window_size: u64) {
    let region = Region {
        chrom: "chr1".to_string(),
        start,
        end,
    };
    let windows: Vec<Window> = WindowIterator::new(&region, window_size).collect();

    if start > end {
        assert!(windows.is_empty(), "start > end should produce no windows");
        return;
    }

    // Must produce at least one window
    assert!(!windows.is_empty(), "start <= end should produce at least one window");

    // First window starts at region start
    assert_eq!(windows[0].start, start, "first window must start at region.start");

    // Last window ends at region end
    assert_eq!(
        windows.last().unwrap().end,
        end,
        "last window must end at region.end"
    );

    // Windows are contiguous (no gaps, no overlaps)
    for pair in windows.windows(2) {
        assert_eq!(
            pair[1].start,
            pair[0].end + 1,
            "windows must be contiguous: w[i].end+1 == w[i+1].start"
        );
    }

    // All windows except possibly the last have exact window_size
    for w in &windows[..windows.len() - 1] {
        assert_eq!(
            w.length(),
            window_size,
            "non-final windows must have exact window_size"
        );
    }

    // Last window is <= window_size
    assert!(
        windows.last().unwrap().length() <= window_size,
        "final window must be <= window_size"
    );

    // Total coverage equals region length
    let total: u64 = windows.iter().map(|w| w.length()).sum();
    assert_eq!(
        total,
        end - start + 1,
        "total window coverage must equal region length"
    );

    // Windows are monotonically increasing
    for pair in windows.windows(2) {
        assert!(pair[0].start < pair[1].start, "window starts must be strictly increasing");
        assert!(pair[0].end < pair[1].end, "window ends must be strictly increasing");
    }
}

#[test]
fn window_invariants_exact_multiple() {
    assert_window_invariants(1, 100, 10);
}

#[test]
fn window_invariants_partial_last() {
    assert_window_invariants(1, 103, 10);
}

#[test]
fn window_invariants_single_bp_region() {
    assert_window_invariants(500, 500, 10);
}

#[test]
fn window_invariants_window_equals_region() {
    assert_window_invariants(1, 10, 10);
}

#[test]
fn window_invariants_window_larger_than_region() {
    assert_window_invariants(1, 5, 100);
}

#[test]
fn window_invariants_window_size_one() {
    assert_window_invariants(1, 20, 1);
}

#[test]
fn window_invariants_large_region_large_window() {
    assert_window_invariants(1, 248_956_422, 10_000);
}

#[test]
fn window_invariants_non_unit_start() {
    assert_window_invariants(100_000, 200_000, 5_000);
}

#[test]
fn window_invariants_start_equals_end() {
    assert_window_invariants(42, 42, 1);
}

#[test]
fn window_invariants_start_greater_than_end() {
    assert_window_invariants(100, 50, 10);
}

#[test]
fn window_invariants_sweep_sizes() {
    // Test a range of window sizes for the same region
    for ws in [1, 2, 3, 7, 10, 50, 100, 500, 1000] {
        assert_window_invariants(1, 999, ws);
    }
}

#[test]
fn window_invariants_sweep_regions() {
    // Test various region sizes with fixed window
    for end in [1, 2, 9, 10, 11, 99, 100, 101, 999, 1000, 1001] {
        assert_window_invariants(1, end, 10);
    }
}

// Exact window count formula
#[test]
fn window_count_matches_ceiling_division() {
    for (start, end, ws) in [
        (1u64, 100u64, 10u64),
        (1, 101, 10),
        (1, 1, 1),
        (1, 1, 100),
        (50, 149, 25),
        (1, 248_956_422, 10_000),
    ] {
        let region = Region {
            chrom: "chr1".to_string(),
            start,
            end,
        };
        let windows: Vec<Window> = WindowIterator::new(&region, ws).collect();
        let region_len = end - start + 1;
        let expected_count = (region_len + ws - 1) / ws;
        assert_eq!(
            windows.len() as u64,
            expected_count,
            "count mismatch for region {}:{}-{} ws={}",
            "chr1",
            start,
            end,
            ws
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Region — Display ↔ parse round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_display_parse_roundtrip_basic() {
    let original = Region {
        chrom: "chr20".to_string(),
        start: 1_000_000,
        end: 64_444_167,
    };
    let display_str = format!("{}", original);
    let reparsed = Region::parse(&display_str, None).unwrap();
    assert_eq!(reparsed.chrom, original.chrom);
    assert_eq!(reparsed.start, original.start);
    assert_eq!(reparsed.end, original.end);
}

#[test]
fn region_display_parse_roundtrip_small() {
    let original = Region {
        chrom: "chrX".to_string(),
        start: 1,
        end: 100,
    };
    let display_str = format!("{}", original);
    let reparsed = Region::parse(&display_str, None).unwrap();
    assert_eq!(reparsed.chrom, original.chrom);
    assert_eq!(reparsed.start, original.start);
    assert_eq!(reparsed.end, original.end);
}

#[test]
fn region_display_parse_roundtrip_same_start_end() {
    let original = Region {
        chrom: "chr1".to_string(),
        start: 42,
        end: 42,
    };
    let display_str = format!("{}", original);
    let reparsed = Region::parse(&display_str, None).unwrap();
    assert_eq!(reparsed.start, 42);
    assert_eq!(reparsed.end, 42);
}

// ═══════════════════════════════════════════════════════════════════════════
// Region — to_impg_ref format
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_impg_ref_format_contains_all_components() {
    let region = Region {
        chrom: "chr12".to_string(),
        start: 5000,
        end: 15000,
    };
    let impg = region.to_impg_ref("CHM13");
    assert!(impg.starts_with("CHM13#0#"));
    assert!(impg.contains("chr12:"));
    assert!(impg.contains("5000-15000"));
    assert_eq!(impg, "CHM13#0#chr12:5000-15000");
}

#[test]
fn region_impg_ref_with_grch38() {
    let region = Region::parse("chr1:1-248956422", None).unwrap();
    let impg = region.to_impg_ref("GRCh38");
    assert_eq!(impg, "GRCh38#0#chr1:1-248956422");
}

// ═══════════════════════════════════════════════════════════════════════════
// Region::parse — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_parse_chrom_only_with_length() {
    let region = Region::parse("chrY", Some(57227415)).unwrap();
    assert_eq!(region.chrom, "chrY");
    assert_eq!(region.start, 1);
    assert_eq!(region.end, 57227415);
}

#[test]
fn region_parse_chrom_only_without_length_is_error() {
    let result = Region::parse("chrM", None);
    assert!(result.is_err());
    match result {
        Err(HprcError::InvalidParameter(msg)) => {
            assert!(msg.contains("requires --region-length"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

#[test]
fn region_parse_multiple_dashes_in_range() {
    // "chr1:100-200-300" — should fail because "200-300" isn't a valid u64
    let result = Region::parse("chr1:100-200-300", None);
    assert!(result.is_err());
}

#[test]
fn region_parse_colon_but_no_dash_is_error() {
    let result = Region::parse("chr1:12345", None);
    assert!(result.is_err());
}

#[test]
fn region_parse_just_colon() {
    let result = Region::parse(":", None);
    assert!(result.is_err()); // empty rest, no dash
}

#[test]
fn region_parse_colon_at_end() {
    // "chr1:" — after colon is "", no dash → error
    let result = Region::parse("chr1:", None);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// ColumnIndices — duplicate column names
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn column_indices_duplicate_column_takes_first() {
    // If "chrom" appears twice, from_header uses the first occurrence
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tchrom";
    let cols = ColumnIndices::from_header(header).unwrap();
    assert_eq!(cols.chrom, 0, "should use first 'chrom' at index 0, not the duplicate at 6");
}

#[test]
fn column_indices_all_required_columns_must_be_present() {
    // Missing group.a
    let header = "chrom\tstart\tend\tgroup.b\testimated.identity";
    let result = ColumnIndices::from_header(header);
    assert!(result.is_err());
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "group.a"),
        _ => panic!("Expected MissingColumn('group.a')"),
    }
}

#[test]
fn column_indices_missing_group_b() {
    let header = "chrom\tstart\tend\tgroup.a\testimated.identity";
    let result = ColumnIndices::from_header(header);
    assert!(result.is_err());
    match result {
        Err(HprcError::MissingColumn(col)) => assert_eq!(col, "group.b"),
        _ => panic!("Expected MissingColumn('group.b')"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ColumnIndices — max_index property
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn column_indices_max_index_is_largest_index() {
    let header = "extra\tchrom\tstart\tend\textra2\tgroup.a\tgroup.b\textra3\textra4\testimated.identity";
    let cols = ColumnIndices::from_header(header).unwrap();
    // estimated.identity is at index 9 (the largest)
    assert_eq!(cols.max_index(), 9);
}

#[test]
fn column_indices_max_index_all_at_front() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\textra1\textra2";
    let cols = ColumnIndices::from_header(header).unwrap();
    assert_eq!(cols.max_index(), 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Window — arithmetic edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn window_length_single_bp() {
    let w = Window::new(100, 100);
    assert_eq!(w.length(), 1);
}

#[test]
fn window_length_two_bp() {
    let w = Window::new(100, 101);
    assert_eq!(w.length(), 2);
}

#[test]
fn window_length_large() {
    let w = Window::new(1, 248_956_422);
    assert_eq!(w.length(), 248_956_422);
}

#[test]
fn window_start_zero() {
    // Window with start=0 (0-based coordinates)
    let w = Window::new(0, 9999);
    assert_eq!(w.length(), 10000);
}

// ═══════════════════════════════════════════════════════════════════════════
// HprcError — Display implementations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hprc_error_display_all_variants() {
    let errors = vec![
        (HprcError::Parse("bad input".into()), "Parse error: bad input"),
        (HprcError::InvalidRegion("chr1:abc".into()), "Invalid region format: chr1:abc"),
        (HprcError::MissingColumn("chrom".into()), "Missing column: chrom"),
        (HprcError::ExternalTool("impg failed".into()), "External tool error: impg failed"),
        (HprcError::InvalidParameter("negative size".into()), "Invalid parameter: negative size"),
    ];

    for (error, expected) in errors {
        assert_eq!(format!("{}", error), expected);
    }
}

#[test]
fn hprc_error_io_variant_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let hprc_err = HprcError::from(io_err);
    let display = format!("{}", hprc_err);
    assert!(display.starts_with("IO error:"));
    assert!(display.contains("file missing"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Region Clone — verify deep copy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn region_clone_is_independent() {
    let original = Region {
        chrom: "chr1".to_string(),
        start: 100,
        end: 200,
    };
    let cloned = original.clone();
    assert_eq!(cloned.chrom, "chr1");
    assert_eq!(cloned.start, 100);
    assert_eq!(cloned.end, 200);
    // Mutating original shouldn't affect clone
    // (Region fields are owned, so this is guaranteed by Rust's ownership model)
}

// ═══════════════════════════════════════════════════════════════════════════
// WindowIterator — fused iterator behavior
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn window_iterator_returns_none_after_exhaustion() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 1,
        end: 10,
    };
    let mut iter = WindowIterator::new(&region, 10);
    assert!(iter.next().is_some());
    assert!(iter.next().is_none());
    // Calling next() again should still return None
    assert!(iter.next().is_none());
}

#[test]
fn window_iterator_empty_region_immediate_none() {
    let region = Region {
        chrom: "chr1".to_string(),
        start: 100,
        end: 50,
    };
    let mut iter = WindowIterator::new(&region, 10);
    assert!(iter.next().is_none());
}
