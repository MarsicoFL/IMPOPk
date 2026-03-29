//! TPA record → CIGAR reconstruction → mismatch positions.
//!
//! Reads TPA records, reconstructs CIGAR strings from tracepoints + AGC
//! sequences via WFA2, then extracts mismatch positions for pairwise identity.
//!
//! The heavy work (AGC fetch + WFA2 alignment) is parallelized with rayon.
//! Each thread uses its own AGC decompressor pool (built into AgcIndex).

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use tpa::{AlignmentRecord, TpaReader};
use tracepoints::ComplexityMetric;

use crate::agc_access::AgcIndex;
use crate::paf::{
    extract_hap_id, extract_sample_from_hap, extract_target_chrom, parse_cigar_mismatches,
    PafAlignment,
};

/// Reverse complement a DNA sequence in-place.
fn reverse_complement(seq: &mut [u8]) {
    seq.reverse();
    for b in seq.iter_mut() {
        *b = match *b {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            b'a' => b't',
            b't' => b'a',
            b'c' => b'g',
            b'g' => b'c',
            other => other,
        };
    }
}

/// Metadata needed for CIGAR reconstruction, extracted from TPA header.
#[derive(Clone)]
struct ReconstructParams {
    complexity_metric: ComplexityMetric,
    max_complexity: u32,
    distance: lib_wfa2::affine_wavefront::Distance,
}

/// A filtered record ready for parallel CIGAR reconstruction.
struct FilteredRecord {
    hap_id: String,
    query_name: String,
    query_start: u64,
    query_end: u64,
    target_name: String,
    target_start: u64,
    target_end: u64,
    strand: char,
    residue_matches: u64,
    alignment_block_len: u64,
    tracepoints: tracepoints::TracepointData,
}

/// Convert TPA records to PafAlignments with mismatch positions.
///
/// Two-phase approach for performance:
/// 1. Sequential: read TPA records + filter (fast, I/O bound)
/// 2. Parallel: AGC fetch + WFA2 reconstruction + mismatch parsing (CPU bound)
///
/// AGC decompression is I/O-bound with internal locks, so we limit parallelism
/// for the reconstruction phase to avoid mutex contention. The pairwise
/// computation phase (downstream) benefits from full thread count.
pub fn records_to_paf_alignments(
    reader: &mut TpaReader,
    agc: &AgcIndex,
    record_ids: &[u64],
    target_chrom: &str,
    subset: Option<&HashSet<String>>,
    min_aligned_length: u64,
) -> anyhow::Result<Vec<PafAlignment>> {
    let params = ReconstructParams {
        complexity_metric: reader.header().complexity_metric(),
        max_complexity: reader.header().max_complexity(),
        distance: reader.header().distance(),
    };

    // Phase 1: Sequential TPA read + filtering
    let filtered = read_and_filter(reader, record_ids, target_chrom, subset, min_aligned_length)?;
    eprintln!(
        "  {} records passed filters (of {} fetched)",
        filtered.len(),
        record_ids.len()
    );

    if filtered.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 2: Parallel CIGAR reconstruction with limited concurrency.
    // AGC decompression has internal mutex contention — more than ~4 threads
    // causes thrashing on the 3.1GB AGC file. Use a dedicated small pool.
    let n_reconstruct_threads = 4.min(rayon::current_num_threads());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_reconstruct_threads)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    let errors = AtomicU64::new(0);
    let mut alignments: Vec<PafAlignment> = pool.install(|| {
        filtered
            .par_iter()
            .filter_map(|rec| match reconstruct_one(rec, agc, &params) {
                Some(aln) => Some(aln),
                None => {
                    errors.fetch_add(1, Ordering::Relaxed);
                    None
                }
            })
            .collect()
    });

    let n_errors = errors.load(Ordering::Relaxed);
    if n_errors > 0 {
        eprintln!(
            "  WARNING: {} records failed CIGAR reconstruction",
            n_errors
        );
    }

    // Sort by target_start for window processing
    alignments.sort_by_key(|a| a.target_start);

    Ok(alignments)
}

/// Phase 1: Read records from TPA and apply filters.
fn read_and_filter(
    reader: &mut TpaReader,
    record_ids: &[u64],
    target_chrom: &str,
    subset: Option<&HashSet<String>>,
    min_aligned_length: u64,
) -> anyhow::Result<Vec<FilteredRecord>> {
    let mut filtered = Vec::new();

    for &id in record_ids {
        let record: AlignmentRecord = match reader.get_record(id) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Chromosome filter
        if extract_target_chrom(&record.target_name) != target_chrom {
            continue;
        }

        // Length filter
        if record.alignment_block_len < min_aligned_length {
            continue;
        }

        // Reference filter
        let hap_id = extract_hap_id(&record.query_name);
        let sample = extract_sample_from_hap(hap_id);
        if sample == "CHM13" || sample == "GRCh38" {
            continue;
        }

        // Subset filter
        if let Some(ss) = subset {
            if !ss.contains(hap_id) && !ss.contains(sample) {
                continue;
            }
        }

        filtered.push(FilteredRecord {
            hap_id: hap_id.to_string(),
            query_name: record.query_name,
            query_start: record.query_start,
            query_end: record.query_end,
            target_name: record.target_name,
            target_start: record.target_start,
            target_end: record.target_end,
            strand: record.strand,
            residue_matches: record.residue_matches,
            alignment_block_len: record.alignment_block_len,
            tracepoints: record.tracepoints,
        });
    }

    Ok(filtered)
}

/// Phase 2: Reconstruct CIGAR for one record (called in parallel).
fn reconstruct_one(
    rec: &FilteredRecord,
    agc: &AgcIndex,
    params: &ReconstructParams,
) -> Option<PafAlignment> {
    // Fetch sequences from AGC (thread-safe via per-thread decompressors)
    let mut query_seq = agc
        .fetch_sequence(
            &rec.query_name,
            rec.query_start as usize,
            rec.query_end as usize,
        )
        .ok()?;

    let target_seq = agc
        .fetch_sequence(
            &rec.target_name,
            rec.target_start as usize,
            rec.target_end as usize,
        )
        .ok()?;

    // Reverse complement for minus strand
    if rec.strand == '-' {
        reverse_complement(&mut query_seq);
    }

    // Reconstruct CIGAR from tracepoints via WFA2
    let cigar = tpa::reconstruct_cigar_with_heuristic(
        &rec.tracepoints,
        &query_seq,
        &target_seq,
        0,
        0,
        &params.distance,
        params.complexity_metric,
        0,     // spacing (non-FastGA)
        false, // complement already handled
        params.max_complexity,
    );

    // Parse mismatch positions from reconstructed CIGAR
    let (mismatch_positions, aligned_bases) = parse_cigar_mismatches(&cigar, rec.target_start);

    let gap_identity = if rec.alignment_block_len > 0 {
        rec.residue_matches as f64 / rec.alignment_block_len as f64
    } else {
        0.0
    };

    Some(PafAlignment {
        hap_id: rec.hap_id.clone(),
        target_start: rec.target_start,
        target_end: rec.target_end,
        gap_identity,
        mismatch_positions,
        aligned_bases,
    })
}
