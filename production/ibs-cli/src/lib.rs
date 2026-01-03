//! HPRC-IBD: Identity-By-State and Identity-By-Descent detection
//!
//! This library provides tools for detecting IBS and IBD segments
//! in pangenome data from the Human Pangenome Reference Consortium.
//!
//! # Modules
//!
//! - `hmm`: Hidden Markov Model for IBD state inference
//! - `stats`: Statistical utilities for identity analysis
//! - `segment`: Segment detection and merging algorithms

pub mod hmm;
pub mod segment;
pub mod stats;

use thiserror::Error;

/// Custom error types for the library
#[derive(Error, Debug)]
pub enum IbdError {
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

pub type Result<T> = std::result::Result<T, IbdError>;

/// Genomic region specification
#[derive(Debug, Clone)]
pub struct Region {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
}

impl Region {
    /// Parse a region string like "chr1:1-1000000" or "chr1"
    pub fn parse(region_str: &str, region_length: Option<u64>) -> Result<Self> {
        if let Some(colon_pos) = region_str.find(':') {
            let chrom = region_str[..colon_pos].to_string();
            let rest = &region_str[colon_pos + 1..];

            let dash_pos = rest
                .find('-')
                .ok_or_else(|| IbdError::InvalidRegion(region_str.to_string()))?;

            let start: u64 = rest[..dash_pos]
                .parse()
                .map_err(|_| IbdError::InvalidRegion(region_str.to_string()))?;
            let end: u64 = rest[dash_pos + 1..]
                .parse()
                .map_err(|_| IbdError::InvalidRegion(region_str.to_string()))?;

            Ok(Region { chrom, start, end })
        } else {
            let end = region_length.ok_or_else(|| {
                IbdError::InvalidParameter(format!(
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

    /// Format region for impg reference specification
    pub fn to_impg_ref(&self, ref_name: &str) -> String {
        format!("{}#0#{}:{}-{}", ref_name, self.chrom, self.start, self.end)
    }
}

/// Window specification for sliding window analysis
#[derive(Debug, Clone, Copy)]
pub struct Window {
    pub start: u64,
    pub end: u64,
}

impl Window {
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    pub fn length(&self) -> u64 {
        self.end - self.start + 1
    }
}

/// Iterator over windows in a region
pub struct WindowIterator {
    current: u64,
    end: u64,
    window_size: u64,
}

impl WindowIterator {
    pub fn new(region: &Region, window_size: u64) -> Self {
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
}
