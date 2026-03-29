//! # HPRC-IBS: Identity-By-State Detection
//!
//! This library provides tools for detecting IBS (Identity-By-State) windows
//! in pangenome data from the Human Pangenome Reference Consortium (HPRC).
//!
//! ## Overview
//!
//! The library provides region management and window iteration for IBS analysis:
//!
//! 1. **Region Management**: Parse and iterate over genomic regions using sliding windows
//! 2. **Window Iteration**: Generate non-overlapping windows for analysis
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hprc_ibs::{Region, WindowIterator};
//!
//! // Parse a genomic region
//! let region = Region::parse("chr20:1-1000000", None).unwrap();
//!
//! // Iterate over 5kb windows
//! for window in WindowIterator::new(&region, 5000) {
//!     println!("Processing window {}-{}", window.start, window.end);
//! }
//! ```
//!
//! ## Key Concepts
//!
//! ### IBS (Identity-By-State)
//!
//! Observable sequence similarity between haplotypes. Each window is analyzed
//! independently using `impg similarity` to compute pairwise identity scores.

pub mod paf;

// TPA format support
pub mod agc_access;
pub mod cigar_reconstruct;
pub mod spatial_index;

// Re-export common types from hprc-common for backwards compatibility
pub use hprc_common::{HprcError as IbsError, Region, Result, Window, WindowIterator};
