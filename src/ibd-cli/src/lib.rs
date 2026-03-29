//! # HPRC-IBD: Identity-By-State and Identity-By-Descent Detection
//!
//! This library provides tools for detecting IBS (Identity-By-State) and IBD
//! (Identity-By-Descent) segments in pangenome data from the Human Pangenome
//! Reference Consortium (HPRC).
//!
//! ## Overview
//!
//! The library implements a complete pipeline for haplotype identity analysis:
//!
//! 1. **Region Management**: Parse and iterate over genomic regions using sliding windows
//! 2. **Statistical Analysis**: Compute identity distributions and cluster observations
//! 3. **HMM-based IBD Calling**: Use a two-state Hidden Markov Model to distinguish
//!    true IBD from sporadic IBS
//! 4. **Segment Detection**: Identify and merge contiguous IBD segments
//!
//! ## Modules
//!
//! - [`hmm`]: Hidden Markov Model for IBD state inference with Viterbi and forward-backward
//! - [`stats`]: Statistical utilities including Gaussian distributions, k-means
//!   clustering, and online statistics computation
//! - [`segment`]: Segment detection using run-length encoding and segment merging
//! - [`concordance`]: Segment concordance metrics (Jaccard, precision/recall, boundary
//!   accuracy) for validation against external tools like hap-ibd
//! - [`hapibd`]: Parser for hap-ibd `.ibd` output format and segment filtering utilities
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hprc_ibd::{Region, WindowIterator};
//! use hprc_ibd::hmm::{HmmParams, viterbi, extract_ibd_segments};
//!
//! // Parse a genomic region
//! let region = Region::parse("chr20:1-1000000", None).unwrap();
//!
//! // Iterate over 5kb windows
//! for window in WindowIterator::new(&region, 5000) {
//!     println!("Processing window {}-{}", window.start, window.end);
//! }
//!
//! // Run IBD detection on identity observations
//! let observations = vec![0.5, 0.6, 0.999, 0.998, 0.997, 0.5];
//! let mut params = HmmParams::from_expected_length(50.0, 0.0001);
//! params.estimate_emissions(&observations);
//! let states = viterbi(&observations, &params);
//! let segments = extract_ibd_segments(&states);
//! ```
//!
//! ## Key Concepts
//!
//! ### IBS vs IBD
//!
//! - **IBS (Identity-By-State)**: Observable sequence similarity between haplotypes
//! - **IBD (Identity-By-Descent)**: Inferred shared ancestry based on sustained IBS patterns
//!
//! The HMM distinguishes true IBD (long stretches of high identity) from sporadic IBS
//! (isolated windows of similarity that may be due to chance).
//!
//! ### Binary IBD Model
//!
//! Unlike traditional diploid genotype-based IBD (IBD0/IBD1/IBD2), pangenome assemblies
//! enable direct haplotype comparison with a simpler binary model:
//! - **IBD = 0**: Haplotypes do not share recent common ancestry
//! - **IBD = 1**: Haplotypes do share recent common ancestry
//!
//! See the conceptual framework documentation for more details on this distinction.

pub mod concordance;
pub mod hapibd;
pub mod hmm;
pub mod segment;
pub mod stats;

// Re-export common types from hprc-common for backwards compatibility
pub use hprc_common::{HprcError as IbdError, Region, Result, Window, WindowIterator};
