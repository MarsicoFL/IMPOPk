//! # impopk-common: Shared Types for impopk IBD/IBS Analysis
//!
//! This library provides common types and utilities shared between
//! `impopk-ibd` and `impopk-ibs` crates.
//!
//! ## Overview
//!
//! - **Region Management**: Parse and iterate over genomic regions
//! - **Window Iteration**: Generate non-overlapping windows for analysis
//! - **Error Types**: Common error handling
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use impopk_common::{Region, WindowIterator};
//!
//! // Parse a genomic region
//! let region = Region::parse("chr20:1-1000000", None).unwrap();
//!
//! // Iterate over 5kb windows
//! for window in WindowIterator::new(&region, 5000) {
//!     println!("Processing window {}-{}", window.start, window.end);
//! }
//! ```

use thiserror::Error;

/// Custom error types for the impopk IBD/IBS libraries.
///
/// This enum covers all error conditions that can occur during IBS/IBD analysis,
/// including IO errors, parsing failures, invalid parameters, and external tool errors.
///
/// # Example
///
/// ```rust
/// use impopk_common::{ImpopkError, Region};
///
/// let result = Region::parse("invalid", None);
/// match result {
///     Err(ImpopkError::InvalidParameter(msg)) => {
///         println!("Parameter error: {}", msg);
///     }
///     _ => {}
/// }
/// ```
#[derive(Error, Debug)]
pub enum ImpopkError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Invalid region format: {0}")]
    InvalidRegion(String),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("External tool error: {0}")]
    ExternalTool(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Convenience type alias for Results with [`ImpopkError`].
pub type Result<T> = std::result::Result<T, ImpopkError>;

/// Genomic region specification for IBS/IBD analysis.
///
/// Represents a contiguous region of a chromosome with start and end coordinates.
/// Used to define the scope of sliding window analysis.
///
/// # Fields
///
/// * `chrom` - Chromosome name (e.g., "chr1", "chr20")
/// * `start` - Start position in base pairs (1-based, inclusive)
/// * `end` - End position in base pairs (inclusive)
///
/// # Example
///
/// ```rust
/// use impopk_common::Region;
///
/// // Parse from string format
/// let region = Region::parse("chr20:1000000-2000000", None).unwrap();
/// assert_eq!(region.chrom, "chr20");
/// assert_eq!(region.start, 1000000);
/// assert_eq!(region.end, 2000000);
///
/// // Create directly
/// let region = Region {
///     chrom: "chr1".to_string(),
///     start: 1,
///     end: 248956422,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Region {
    /// Chromosome name (e.g., "chr1", "chrX")
    pub chrom: String,
    /// Start position in base pairs (1-based, inclusive)
    pub start: u64,
    /// End position in base pairs (inclusive)
    pub end: u64,
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}-{}", self.chrom, self.start, self.end)
    }
}

impl Region {
    /// Parse a region string in either "chr:start-end" or "chr" format.
    ///
    /// # Arguments
    ///
    /// * `region_str` - Region specification string
    /// * `region_length` - Required if `region_str` contains only chromosome name
    ///
    /// # Returns
    ///
    /// A `Result<Region>` containing the parsed region or an error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use impopk_common::Region;
    ///
    /// // With explicit coordinates
    /// let region = Region::parse("chr1:1000-2000", None).unwrap();
    ///
    /// // Chromosome-only (requires length)
    /// let region = Region::parse("chr1", Some(248956422)).unwrap();
    /// assert_eq!(region.start, 1);
    /// assert_eq!(region.end, 248956422);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ImpopkError::InvalidRegion`] if the format is malformed.
    /// Returns [`ImpopkError::InvalidParameter`] if chromosome-only format is used without length.
    pub fn parse(region_str: &str, region_length: Option<u64>) -> Result<Self> {
        if let Some(colon_pos) = region_str.find(':') {
            let chrom = region_str[..colon_pos].to_string();
            let rest = &region_str[colon_pos + 1..];

            let dash_pos = rest
                .find('-')
                .ok_or_else(|| ImpopkError::InvalidRegion(region_str.to_string()))?;

            let start: u64 = rest[..dash_pos]
                .parse()
                .map_err(|_| ImpopkError::InvalidRegion(region_str.to_string()))?;
            let end: u64 = rest[dash_pos + 1..]
                .parse()
                .map_err(|_| ImpopkError::InvalidRegion(region_str.to_string()))?;

            if start == 0 {
                return Err(ImpopkError::InvalidRegion(format!(
                    "{} (coordinates are 1-based; start must be >= 1)",
                    region_str
                )));
            }
            if start > end {
                return Err(ImpopkError::InvalidRegion(format!(
                    "{} (start {} > end {})",
                    region_str, start, end
                )));
            }

            Ok(Region { chrom, start, end })
        } else {
            let end = region_length.ok_or_else(|| {
                ImpopkError::InvalidParameter(format!(
                    "Region '{}' requires --region-length",
                    region_str
                ))
            })?;

            Ok(Region {
                chrom: region_str.to_string(),
                start: 1,
                end,
            })
        }
    }

    /// Format region for impg reference specification.
    ///
    /// Creates a string in the format expected by `impg` for specifying
    /// reference regions: `REF#0#chrom:start-end`
    ///
    /// # Arguments
    ///
    /// * `ref_name` - Reference genome name (e.g., "CHM13", "GRCh38")
    ///
    /// # Example
    ///
    /// ```rust
    /// use impopk_common::Region;
    ///
    /// let region = Region::parse("chr20:1-1000000", None).unwrap();
    /// let impg_ref = region.to_impg_ref("CHM13");
    /// assert_eq!(impg_ref, "CHM13#0#chr20:1-1000000");
    /// ```
    pub fn to_impg_ref(&self, ref_name: &str) -> String {
        format!("{}#0#{}:{}-{}", ref_name, self.chrom, self.start, self.end)
    }
}

/// Column indices for parsing impg similarity output.
///
/// This structure maps column names to their indices in the TSV output
/// from `impg similarity`. It allows flexible parsing regardless of column order.
///
/// # Example
///
/// ```rust,ignore
/// use impopk_common::ColumnIndices;
///
/// let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
/// let cols = ColumnIndices::from_header(header).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ColumnIndices {
    /// Index of the "estimated.identity" column
    pub estimated_identity: usize,
    /// Index of the "chrom" column
    pub chrom: usize,
    /// Index of the "start" column
    pub start: usize,
    /// Index of the "end" column
    pub end: usize,
    /// Index of the "group.a" column
    pub group_a: usize,
    /// Index of the "group.b" column
    pub group_b: usize,
}

impl ColumnIndices {
    /// Parse column indices from a TSV header line.
    ///
    /// # Arguments
    ///
    /// * `header` - Tab-separated header line from impg similarity output
    ///
    /// # Returns
    ///
    /// A `Result<ColumnIndices>` containing the parsed indices or an error
    /// if any required column is missing.
    ///
    /// # Errors
    ///
    /// Returns [`ImpopkError::MissingColumn`] if a required column is not found.
    pub fn from_header(header: &str) -> Result<Self> {
        let columns: Vec<&str> = header.split('\t').collect();

        let find_col = |name: &str| -> Result<usize> {
            columns
                .iter()
                .position(|&c| c == name)
                .ok_or_else(|| ImpopkError::MissingColumn(name.to_string()))
        };

        Ok(ColumnIndices {
            estimated_identity: find_col("estimated.identity")?,
            chrom: find_col("chrom")?,
            start: find_col("start")?,
            end: find_col("end")?,
            group_a: find_col("group.a")?,
            group_b: find_col("group.b")?,
        })
    }

    /// Returns the maximum column index needed to parse a row.
    ///
    /// Useful for validating that a data row has enough columns.
    pub fn max_index(&self) -> usize {
        self.estimated_identity
            .max(self.chrom)
            .max(self.start)
            .max(self.end)
            .max(self.group_a)
            .max(self.group_b)
    }
}

/// A single analysis window within a genomic region.
///
/// Windows are the fundamental unit of IBS/IBD analysis. Each window
/// represents a contiguous stretch of the genome that is analyzed independently
/// for sequence identity between haplotypes.
///
/// # Fields
///
/// * `start` - Start position of the window (1-based, inclusive)
/// * `end` - End position of the window (inclusive)
///
/// # Example
///
/// ```rust
/// use impopk_common::Window;
///
/// let window = Window::new(1000, 5999);
/// assert_eq!(window.length(), 5000);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Window {
    /// Start position in base pairs (inclusive)
    pub start: u64,
    /// End position in base pairs (inclusive)
    pub end: u64,
}

impl Window {
    /// Create a new window with the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (inclusive)
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Calculate the length of the window in base pairs.
    ///
    /// Returns `end - start + 1` to account for inclusive coordinates.
    ///
    /// # Example
    ///
    /// ```rust
    /// use impopk_common::Window;
    ///
    /// let window = Window::new(1, 100);
    /// assert_eq!(window.length(), 100);
    /// ```
    pub fn length(&self) -> u64 {
        self.end.saturating_sub(self.start) + 1
    }
}

/// Iterator that generates non-overlapping windows across a genomic region.
///
/// This iterator divides a [`Region`] into fixed-size windows for sliding window
/// analysis. The final window may be smaller than `window_size` if the region
/// does not divide evenly.
///
/// # Example
///
/// ```rust
/// use impopk_common::{Region, WindowIterator};
///
/// let region = Region {
///     chrom: "chr1".to_string(),
///     start: 1,
///     end: 25000,
/// };
///
/// // Create 10kb windows
/// let windows: Vec<_> = WindowIterator::new(&region, 10000).collect();
///
/// assert_eq!(windows.len(), 3);
/// assert_eq!(windows[0].start, 1);
/// assert_eq!(windows[0].end, 10000);
/// assert_eq!(windows[1].start, 10001);
/// assert_eq!(windows[1].end, 20000);
/// assert_eq!(windows[2].start, 20001);
/// assert_eq!(windows[2].end, 25000);  // Partial final window
/// ```
pub struct WindowIterator {
    current: u64,
    end: u64,
    window_size: u64,
}

impl WindowIterator {
    /// Create a new window iterator over a genomic region.
    ///
    /// # Arguments
    ///
    /// * `region` - The genomic region to iterate over
    /// * `window_size` - Size of each window in base pairs
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0, which would cause an infinite loop.
    ///
    /// # Example
    ///
    /// ```rust
    /// use impopk_common::{Region, WindowIterator};
    ///
    /// let region = Region::parse("chr1:1-100000", None).unwrap();
    /// let mut iter = WindowIterator::new(&region, 5000);
    ///
    /// let first = iter.next().unwrap();
    /// assert_eq!(first.start, 1);
    /// assert_eq!(first.end, 5000);
    /// ```
    pub fn new(region: &Region, window_size: u64) -> Self {
        assert!(
            window_size > 0,
            "window_size must be greater than 0 to avoid infinite loop"
        );
        Self {
            current: region.start,
            end: region.end,
            window_size,
        }
    }
}

impl Iterator for WindowIterator {
    type Item = Window;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.end {
            return None;
        }

        let window_end = (self.current + self.window_size - 1).min(self.end);
        let window = Window::new(self.current, window_end);
        self.current = window_end + 1;
        Some(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_parse_with_coords() {
        let region = Region::parse("chr1:1000-2000", None).unwrap();
        assert_eq!(region.chrom, "chr1");
        assert_eq!(region.start, 1000);
        assert_eq!(region.end, 2000);
    }

    #[test]
    fn test_region_parse_without_coords() {
        let region = Region::parse("chr1", Some(248956422)).unwrap();
        assert_eq!(region.chrom, "chr1");
        assert_eq!(region.start, 1);
        assert_eq!(region.end, 248956422);
    }

    #[test]
    fn test_window_iterator() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 10,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 3).collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0].start, 1);
        assert_eq!(windows[0].end, 3);
        assert_eq!(windows[3].start, 10);
        assert_eq!(windows[3].end, 10);
    }

    #[test]
    #[should_panic(expected = "window_size must be greater than 0")]
    fn test_window_iterator_zero_size_panics() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 10,
        };
        let _ = WindowIterator::new(&region, 0);
    }

    #[test]
    fn test_region_parse_missing_dash() {
        let result = Region::parse("chr1:1000", None);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::InvalidRegion(s)) => assert_eq!(s, "chr1:1000"),
            _ => panic!("Expected InvalidRegion error"),
        }
    }

    #[test]
    fn test_region_parse_non_numeric_start() {
        let result = Region::parse("chr1:abc-2000", None);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::InvalidRegion(s)) => assert_eq!(s, "chr1:abc-2000"),
            _ => panic!("Expected InvalidRegion error"),
        }
    }

    #[test]
    fn test_region_parse_non_numeric_end() {
        let result = Region::parse("chr1:1000-xyz", None);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::InvalidRegion(s)) => assert_eq!(s, "chr1:1000-xyz"),
            _ => panic!("Expected InvalidRegion error"),
        }
    }

    #[test]
    fn test_region_parse_chrom_only_without_length() {
        let result = Region::parse("chr1", None);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::InvalidParameter(s)) => {
                assert!(s.contains("requires --region-length"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_region_parse_empty_chrom() {
        let result = Region::parse(":1000-2000", None);
        assert!(result.is_ok());
        let region = result.unwrap();
        assert_eq!(region.chrom, "");
        assert_eq!(region.start, 1000);
        assert_eq!(region.end, 2000);
    }

    #[test]
    fn test_region_parse_zero_start_rejected() {
        let result = Region::parse("chr1:0-100", None);
        assert!(matches!(result, Err(ImpopkError::InvalidRegion(_))));
    }

    #[test]
    fn test_region_parse_zero_zero_rejected() {
        let result = Region::parse("chr1:0-0", None);
        assert!(matches!(result, Err(ImpopkError::InvalidRegion(_))));
    }

    #[test]
    fn test_region_parse_start_greater_than_end_rejected() {
        let result = Region::parse("chr1:2000-1000", None);
        assert!(matches!(result, Err(ImpopkError::InvalidRegion(_))));
    }

    #[test]
    fn test_region_parse_start_equals_end_ok() {
        let region = Region::parse("chr1:100-100", None).unwrap();
        assert_eq!(region.start, 100);
        assert_eq!(region.end, 100);
    }

    #[test]
    fn test_region_parse_large_coordinates() {
        let result = Region::parse("chr1:1-18446744073709551615", None);
        assert!(result.is_ok());
        let region = result.unwrap();
        assert_eq!(region.start, 1);
        assert_eq!(region.end, u64::MAX);
    }

    #[test]
    fn test_region_impg_ref_format() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 2000,
        };
        let impg_ref = region.to_impg_ref("GRCh38");
        assert_eq!(impg_ref, "GRCh38#0#chr1:1000-2000");
    }

    #[test]
    fn test_window_iterator_single_window_exact_fit() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 10,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 10).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start, 1);
        assert_eq!(windows[0].end, 10);
        assert_eq!(windows[0].length(), 10);
    }

    #[test]
    fn test_window_iterator_single_position_region() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 100,
            end: 100,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 10).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start, 100);
        assert_eq!(windows[0].end, 100);
        assert_eq!(windows[0].length(), 1);
    }

    #[test]
    fn test_window_iterator_partial_last_window() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 15,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 10).collect();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start, 1);
        assert_eq!(windows[0].end, 10);
        assert_eq!(windows[0].length(), 10);
        assert_eq!(windows[1].start, 11);
        assert_eq!(windows[1].end, 15);
        assert_eq!(windows[1].length(), 5);
    }

    #[test]
    fn test_window_iterator_window_larger_than_region() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 5,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 100).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start, 1);
        assert_eq!(windows[0].end, 5);
        assert_eq!(windows[0].length(), 5);
    }

    #[test]
    fn test_window_iterator_size_one() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 1,
            end: 5,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 1).collect();
        assert_eq!(windows.len(), 5);
        for (i, w) in windows.iter().enumerate() {
            let expected_pos = (i + 1) as u64;
            assert_eq!(w.start, expected_pos);
            assert_eq!(w.end, expected_pos);
            assert_eq!(w.length(), 1);
        }
    }

    #[test]
    fn test_window_iterator_empty_region() {
        let region = Region {
            chrom: "chr1".to_string(),
            start: 100,
            end: 50,
        };
        let windows: Vec<_> = WindowIterator::new(&region, 10).collect();
        assert_eq!(windows.len(), 0);
    }

    #[test]
    fn test_window_length() {
        let w = Window::new(100, 199);
        assert_eq!(w.length(), 100);

        let w_single = Window::new(100, 100);
        assert_eq!(w_single.length(), 1);
    }

    // === ColumnIndices tests ===

    #[test]
    fn test_column_indices_standard_header() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let cols = ColumnIndices::from_header(header).unwrap();
        assert_eq!(cols.chrom, 0);
        assert_eq!(cols.start, 1);
        assert_eq!(cols.end, 2);
        assert_eq!(cols.group_a, 3);
        assert_eq!(cols.group_b, 4);
        assert_eq!(cols.estimated_identity, 5);
    }

    #[test]
    fn test_column_indices_reordered_header() {
        let header = "estimated.identity\tgroup.b\tgroup.a\tend\tstart\tchrom";
        let cols = ColumnIndices::from_header(header).unwrap();
        assert_eq!(cols.estimated_identity, 0);
        assert_eq!(cols.group_b, 1);
        assert_eq!(cols.group_a, 2);
        assert_eq!(cols.end, 3);
        assert_eq!(cols.start, 4);
        assert_eq!(cols.chrom, 5);
    }

    #[test]
    fn test_column_indices_extra_columns() {
        let header = "extra1\tchrom\textra2\tstart\tend\tgroup.a\tgroup.b\textra3\testimated.identity";
        let cols = ColumnIndices::from_header(header).unwrap();
        assert_eq!(cols.chrom, 1);
        assert_eq!(cols.start, 3);
        assert_eq!(cols.end, 4);
        assert_eq!(cols.group_a, 5);
        assert_eq!(cols.group_b, 6);
        assert_eq!(cols.estimated_identity, 8);
    }

    #[test]
    fn test_column_indices_missing_chrom() {
        let header = "start\tend\tgroup.a\tgroup.b\testimated.identity";
        let result = ColumnIndices::from_header(header);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::MissingColumn(col)) => assert_eq!(col, "chrom"),
            _ => panic!("Expected MissingColumn error for 'chrom'"),
        }
    }

    #[test]
    fn test_column_indices_missing_identity() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b";
        let result = ColumnIndices::from_header(header);
        assert!(result.is_err());
        match result {
            Err(ImpopkError::MissingColumn(col)) => assert_eq!(col, "estimated.identity"),
            _ => panic!("Expected MissingColumn error for 'estimated.identity'"),
        }
    }

    #[test]
    fn test_column_indices_empty_header() {
        let header = "";
        let result = ColumnIndices::from_header(header);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_indices_max_index() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let cols = ColumnIndices::from_header(header).unwrap();
        assert_eq!(cols.max_index(), 5);
    }

    #[test]
    fn test_column_indices_max_index_reordered() {
        let header = "estimated.identity\tchrom\tstart\tend\tgroup.a\tgroup.b\textra";
        let cols = ColumnIndices::from_header(header).unwrap();
        // group.b is at index 5, which is the max
        assert_eq!(cols.max_index(), 5);
    }

    // === Region Display tests ===

    #[test]
    fn test_region_display() {
        let region = Region {
            chrom: "chr20".to_string(),
            start: 1000000,
            end: 2000000,
        };
        assert_eq!(format!("{}", region), "chr20:1000000-2000000");
    }

    #[test]
    fn test_region_display_roundtrip() {
        let original = Region::parse("chr10:500-1500", None).unwrap();
        let display_str = format!("{}", original);
        let reparsed = Region::parse(&display_str, None).unwrap();
        assert_eq!(reparsed.chrom, original.chrom);
        assert_eq!(reparsed.start, original.start);
        assert_eq!(reparsed.end, original.end);
    }
}
