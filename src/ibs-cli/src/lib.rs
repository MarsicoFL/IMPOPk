//! # impopk-ibs: Identity-By-State detection for pangenome data
//!
//! This crate wraps `impg similarity` and produces a table of pairwise
//! sequence identities per genomic window, which is the input to the
//! downstream IBD and ancestry HMMs in impopk.
//!
//! ## Pipeline
//!
//! 1. The `ibs` binary drives `impg similarity` over a genomic region,
//!    one window at a time.
//! 2. The output TSV lists, for each window, the pairwise identity
//!    between every pair of haplotypes in the subset list.
//! 3. This TSV feeds directly into `ibd` and `ancestry`.

// Re-export common types for convenience
pub use hprc_common::{HprcError as IbsError, Region, Result, Window, WindowIterator};
