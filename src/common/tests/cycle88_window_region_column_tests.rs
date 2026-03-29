// Cycle 88: Deep edge-case tests for common crate functions with minimal coverage.
//
// Targets:
//   - Window::length() — zero direct external tests
//   - Region::to_impg_ref() — 1-2 tests only
//   - ColumnIndices::max_index() — 1-2 tests only
//   - WindowIterator boundary conditions

use hprc_common::{ColumnIndices, Region, Window, WindowIterator};

// ===========================================================================
// Window::length
// ===========================================================================

#[test]
fn window_length_single_bp() {
    let w = Window::new(100, 100);
    assert_eq!(w.length(), 1, "single bp window: end==start → length 1");
}

#[test]
fn window_length_standard() {
    let w = Window::new(1000, 5999);
    assert_eq!(w.length(), 5000);
}

#[test]
fn window_length_one_based_inclusive() {
    // 1-based inclusive: 1 to 100 = 100 bases
    let w = Window::new(1, 100);
    assert_eq!(w.length(), 100);
}

#[test]
fn window_length_large_coordinates() {
    let w = Window::new(200_000_000, 248_956_422);
    assert_eq!(w.length(), 48_956_423);
}

#[test]
fn window_length_max_u64_range() {
    // Near u64 max — should not overflow
    let w = Window::new(u64::MAX - 10, u64::MAX);
    assert_eq!(w.length(), 11);
}

#[test]
fn window_length_zero_start() {
    let w = Window::new(0, 9);
    assert_eq!(w.length(), 10);
}

#[test]
fn window_length_consistent_with_iterator() {
    // Windows from iterator should have length matching window_size (or remainder)
    let region = Region { chrom: "chr1".to_string(), start: 1, end: 25000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows[0].length(), 10000);
    assert_eq!(windows[1].length(), 10000);
    assert_eq!(windows[2].length(), 5000);
}

#[test]
fn window_length_copy_semantics() {
    let w1 = Window::new(10, 20);
    let w2 = w1;
    assert_eq!(w1.length(), w2.length());
    assert_eq!(w1.length(), 11);
}

// ===========================================================================
// Region::to_impg_ref
// ===========================================================================

#[test]
fn to_impg_ref_standard() {
    let r = Region::parse("chr20:1-1000000", None).unwrap();
    assert_eq!(r.to_impg_ref("CHM13"), "CHM13#0#chr20:1-1000000");
}

#[test]
fn to_impg_ref_grch38() {
    let r = Region { chrom: "chr1".to_string(), start: 100, end: 200 };
    assert_eq!(r.to_impg_ref("GRCh38"), "GRCh38#0#chr1:100-200");
}

#[test]
fn to_impg_ref_empty_ref_name() {
    let r = Region { chrom: "chr1".to_string(), start: 1, end: 10 };
    assert_eq!(r.to_impg_ref(""), "#0#chr1:1-10");
}

#[test]
fn to_impg_ref_ref_with_hash() {
    // If ref_name contains '#', it should still format correctly (no escaping)
    let r = Region { chrom: "chr1".to_string(), start: 1, end: 10 };
    let result = r.to_impg_ref("ref#1");
    assert_eq!(result, "ref#1#0#chr1:1-10");
}

#[test]
fn to_impg_ref_chrx() {
    let r = Region { chrom: "chrX".to_string(), start: 50000, end: 100000 };
    assert_eq!(r.to_impg_ref("CHM13"), "CHM13#0#chrX:50000-100000");
}

#[test]
fn to_impg_ref_large_coordinates() {
    let r = Region { chrom: "chr1".to_string(), start: 1, end: 248_956_422 };
    assert_eq!(r.to_impg_ref("CHM13"), "CHM13#0#chr1:1-248956422");
}

#[test]
fn to_impg_ref_single_bp_region() {
    let r = Region { chrom: "chr22".to_string(), start: 42, end: 42 };
    assert_eq!(r.to_impg_ref("CHM13"), "CHM13#0#chr22:42-42");
}

#[test]
fn to_impg_ref_roundtrip_with_display() {
    // to_impg_ref should contain the Display representation embedded
    let r = Region::parse("chr12:5000-10000", None).unwrap();
    let impg = r.to_impg_ref("REF");
    let display = format!("{}", r);
    assert!(impg.contains(&display), "impg ref should contain display: {} not in {}", display, impg);
}

// ===========================================================================
// ColumnIndices::max_index
// ===========================================================================

#[test]
fn max_index_standard_order() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let cols = ColumnIndices::from_header(header).unwrap();
    // Indices: chrom=0, start=1, end=2, group.a=3, group.b=4, estimated.identity=5
    assert_eq!(cols.max_index(), 5);
}

#[test]
fn max_index_reversed_order() {
    let header = "estimated.identity\tgroup.b\tgroup.a\tend\tstart\tchrom";
    let cols = ColumnIndices::from_header(header).unwrap();
    // estimated.identity=0, group.b=1, group.a=2, end=3, start=4, chrom=5
    assert_eq!(cols.max_index(), 5);
}

#[test]
fn max_index_with_extra_columns() {
    let header = "extra1\tchrom\textra2\tstart\tend\textra3\tgroup.a\tgroup.b\textra4\testimated.identity";
    let cols = ColumnIndices::from_header(header).unwrap();
    // chrom=1, start=3, end=4, group.a=6, group.b=7, estimated.identity=9
    assert_eq!(cols.max_index(), 9);
}

#[test]
fn max_index_identity_at_zero() {
    let header = "estimated.identity\tchrom\tstart\tend\tgroup.a\tgroup.b";
    let cols = ColumnIndices::from_header(header).unwrap();
    // estimated.identity=0, others > 0
    assert_eq!(cols.max_index(), 5);
}

#[test]
fn max_index_all_adjacent() {
    // All required columns are first 6 → max_index = 5
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\textra";
    let cols = ColumnIndices::from_header(header).unwrap();
    assert_eq!(cols.max_index(), 5);
}

#[test]
fn max_index_sparse_columns() {
    // Required columns scattered among many extras
    let parts: Vec<&str> = (0..20).map(|_| "filler").collect();
    let mut header_parts = parts.clone();
    header_parts[3] = "chrom";
    header_parts[7] = "start";
    header_parts[11] = "end";
    header_parts[13] = "group.a";
    header_parts[17] = "group.b";
    header_parts[19] = "estimated.identity";
    let header = header_parts.join("\t");
    let cols = ColumnIndices::from_header(&header).unwrap();
    assert_eq!(cols.max_index(), 19);
}

#[test]
fn max_index_validates_row_length() {
    // max_index can be used to check if a data row has enough columns
    let header = "extra\tchrom\tstart\tend\tgroup.a\tgroup.b\textra2\testimated.identity";
    let cols = ColumnIndices::from_header(header).unwrap();
    let max = cols.max_index();

    let valid_row: Vec<&str> = (0..=max).map(|_| "x").collect();
    assert!(valid_row.len() > max, "row with max+1 columns is sufficient");

    let short_row: Vec<&str> = (0..max).map(|_| "x").collect();
    assert!(short_row.len() <= max, "row shorter than max_index is insufficient");
}

// ===========================================================================
// WindowIterator additional boundary tests
// ===========================================================================

#[test]
fn window_iter_single_window_exact_fit() {
    let region = Region { chrom: "chr1".to_string(), start: 1, end: 10000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].start, 1);
    assert_eq!(windows[0].end, 10000);
    assert_eq!(windows[0].length(), 10000);
}

#[test]
fn window_iter_region_smaller_than_window() {
    let region = Region { chrom: "chr1".to_string(), start: 1, end: 5000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].length(), 5000);
}

#[test]
fn window_iter_exact_multiple() {
    let region = Region { chrom: "chr1".to_string(), start: 1, end: 30000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 3);
    for w in &windows {
        assert_eq!(w.length(), 10000);
    }
}

#[test]
fn window_iter_remainder_window() {
    let region = Region { chrom: "chr1".to_string(), start: 1, end: 25000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 3);
    assert_eq!(windows[2].length(), 5000);
}

#[test]
fn window_iter_windows_contiguous() {
    let region = Region { chrom: "chr1".to_string(), start: 100, end: 50000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 7000).collect();
    // Check no gaps or overlaps
    for i in 1..windows.len() {
        assert_eq!(windows[i].start, windows[i-1].end + 1,
            "windows should be contiguous at index {}: {} vs {}",
            i, windows[i].start, windows[i-1].end);
    }
}

#[test]
fn window_iter_covers_entire_region() {
    let region = Region { chrom: "chr1".to_string(), start: 500, end: 15700 };
    let windows: Vec<Window> = WindowIterator::new(&region, 3000).collect();
    assert_eq!(windows.first().unwrap().start, 500);
    assert_eq!(windows.last().unwrap().end, 15700);
}

#[test]
fn window_iter_single_bp_region() {
    let region = Region { chrom: "chr1".to_string(), start: 42, end: 42 };
    let windows: Vec<Window> = WindowIterator::new(&region, 10000).collect();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].length(), 1);
}

#[test]
fn window_iter_total_length_equals_region() {
    let region = Region { chrom: "chr1".to_string(), start: 1000, end: 50000 };
    let windows: Vec<Window> = WindowIterator::new(&region, 7500).collect();
    let total: u64 = windows.iter().map(|w| w.length()).sum();
    assert_eq!(total, region.end - region.start + 1);
}
