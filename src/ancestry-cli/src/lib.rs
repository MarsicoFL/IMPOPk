//! Local Ancestry Inference CLI
//!
//! This crate provides HMM-based local ancestry inference from pangenome
//! alignment data. Given a set of query samples and reference populations
//! (species or ancestral groups), it determines which ancestral population
//! each genomic segment most likely derives from.
//!
//! ## Model
//!
//! The HMM has N states, one per ancestral population. For each genomic window:
//! - **Observations**: Similarity values of query sample vs each reference haplotype
//! - **Emissions**: P(similarities | ancestry) - higher similarity expected for matching ancestry
//! - **Transitions**: Matrix modeling ancestral switches (e.g., ancient recombination)
//!
//! ## Inputs
//!
//! - `--similarity-file`: per-window pairwise identity TSV produced by `ibs`
//! - `--populations`: TSV mapping ancestral population labels to reference
//!   haplotype IDs (one row per haplotype)
//! - `--query-samples`: list of query haplotype IDs to paint
//!
//! For each query haplotype window, similarities to every reference haplotype
//! are aggregated per population (see `EmissionModel`) and fed to the HMM,
//! which decodes the most likely ancestry assignment per window.

pub mod hmm;
pub mod ancestry;
pub mod demography;
pub mod egrm;
pub mod params;
mod validation;
pub mod rfmix;
pub mod concordance;

pub use hmm::*;
pub use ancestry::*;
pub use demography::{
    DemographyParams, DemographicResult, PulseEstimate, SampleDemographicResult,
    extract_tract_lengths, infer_demography, infer_all_demography,
    infer_per_sample_demography, ks_test_exponential,
    format_demography_report, write_demography_tsv,
};
pub use egrm::{parse_similarity_for_egrm, write_gcta_grm, write_diploid_gcta_grm};
pub use params::LearnedParams;
pub use validation::*;
