//! Experimental: ancestral recombination graph inference from pangenome subgraphs.
//!
//! v0.1: parse GFA, enumerate bubbles, classify by mechanism, emit per-path
//! genotype matrix + reference-projected positions ready for tsinfer.

pub mod gfa;
pub mod bubble;
pub mod classify;
pub mod site;

pub use gfa::{Graph, NodeId, Path};
pub use bubble::{Bubble, enumerate_bubbles};
pub use classify::{BubbleType, classify};
pub use site::{build_site, Panel, Site, MISSING_GENOTYPE};
