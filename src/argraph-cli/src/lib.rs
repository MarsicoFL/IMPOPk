//! Experimental: ancestral recombination graph inference from pangenome subgraphs.
//!
//! v0.1: parse GFA, enumerate bubbles, classify by mechanism.
//! Downstream (tsinfer wiring, dating) is out of scope for v0.1 — the
//! classifier output is the bridge contract.

pub mod gfa;
pub mod bubble;
pub mod classify;

pub use gfa::{Graph, NodeId, Path};
pub use bubble::{Bubble, enumerate_bubbles};
pub use classify::{BubbleType, classify};
