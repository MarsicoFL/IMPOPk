use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use hprc_common::Region;
use rayon::prelude::*;
use tpa::TpaReader;
use hprc_ibs::agc_access::AgcIndex;
use hprc_ibs::cigar_reconstruct::records_to_paf_alignments;
use hprc_ibs::paf;
use hprc_ibs::spatial_index::SpatialIndex;

#[derive(Parser)]
#[command(name = "ibs-from-tpa", about = "Compute IBS identity from TPA alignments")]
struct Args {
    /// TPA alignment file
    #[arg(long)]
    tpa: String,

    /// Spatial index file (.sidx)
    #[arg(long = "spatial-index")]
    spatial_index: String,

    /// AGC file for sequence access
    #[arg(long)]
    agc: String,

    /// Reference name in PAF target (e.g., CHM13)
    #[arg(long = "ref-name", default_value = "CHM13")]
    ref_name: String,

    /// Region to process (e.g., chr12:1000000-2000000)
    #[arg(long)]
    region: String,

    /// Window size in bp
    #[arg(long)]
    size: u64,

    /// Output TSV file
    #[arg(long)]
    output: String,

    /// Minimum identity cutoff
    #[arg(short = 'c', default_value = "0.0")]
    cutoff: f64,

    /// Minimum alignment block length
    #[arg(long = "min-aligned-length", default_value = "5000")]
    min_aligned_length: u64,

    /// Haplotype subset list file
    #[arg(long = "subset-sequence-list")]
    subset_list: Option<String>,

    /// Query samples file
    #[arg(long = "query-samples")]
    query_samples: Option<String>,

    /// Reference samples file
    #[arg(long = "ref-samples")]
    ref_samples: Option<String>,

    /// Region length (if region is chrom-only)
    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Number of threads
    #[arg(short = 't', long, default_value = "0")]
    threads: usize,
}

fn load_sample_set(path: &str) -> Result<HashSet<String>> {
    let file = File::open(path).context(format!("Failed to open: {}", path))?;
    let reader = BufReader::new(file);
    let mut samples = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            samples.insert(trimmed.to_string());
        }
    }
    Ok(samples)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Thread pool
    let n_threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();
    eprintln!("Using {} threads", n_threads);

    let total_start = Instant::now();

    // Load spatial index
    eprintln!("Loading spatial index from {}...", args.spatial_index);
    let sidx = SpatialIndex::load(&args.spatial_index)?;
    eprintln!(
        "  {} records, {} chroms, bin_size={}",
        sidx.total_records, sidx.num_chroms, sidx.bin_size
    );

    // Open TPA
    eprintln!("Opening TPA file {}...", args.tpa);
    let mut reader = TpaReader::new(&args.tpa)?;
    reader.load_string_table()?;
    eprintln!("  {} records", reader.num_records());

    // Open AGC
    eprintln!("Opening AGC file {}...", args.agc);
    let agc = AgcIndex::build(std::slice::from_ref(&args.agc)).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("  {} contigs indexed", agc.num_contigs());

    // Parse region
    let region = Region::parse(&args.region, args.region_length)?;
    eprintln!(
        "Processing {}:{}-{} (window_size={})",
        region.chrom, region.start, region.end, args.size
    );

    // Load subset/query/ref filters
    let subset = args
        .subset_list
        .as_ref()
        .map(|p| load_sample_set(p))
        .transpose()?;
    let query_set = args
        .query_samples
        .as_ref()
        .map(|p| load_sample_set(p))
        .transpose()?;
    let ref_set = args
        .ref_samples
        .as_ref()
        .map(|p| load_sample_set(p))
        .transpose()?;

    // Query spatial index for overlapping records
    let record_ids = sidx.query(&region.chrom, region.start, region.end);
    eprintln!(
        "  Spatial index: {} record IDs for region",
        record_ids.len()
    );

    // Convert TPA records → PafAlignments (CIGAR reconstruction)
    let reconstruct_start = Instant::now();
    let alignments = records_to_paf_alignments(
        &mut reader,
        &agc,
        &record_ids,
        &region.chrom,
        subset.as_ref(),
        args.min_aligned_length,
    )?;
    eprintln!(
        "  {} alignments reconstructed in {:.1}s",
        alignments.len(),
        reconstruct_start.elapsed().as_secs_f64()
    );

    // Count unique haplotypes
    let hap_count: HashSet<&str> = alignments.iter().map(|a| a.hap_id.as_str()).collect();
    eprintln!("  {} unique haplotypes", hap_count.len());

    // Generate windows
    let windows: Vec<(u64, u64)> = {
        let mut ws = Vec::new();
        let mut start = region.start;
        while start <= region.end {
            let end = (start + args.size - 1).min(region.end);
            ws.push((start, end));
            start = end + 1;
        }
        ws
    };

    // Compute pairwise identity per window (parallel)
    let compute_start = Instant::now();
    let window_results: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(window_start, window_end)| {
            let w_start_0 = window_start - 1;
            let w_end_0 = window_end;
            paf::compute_window_pairwise(
                &alignments,
                w_start_0,
                w_end_0,
                &args.ref_name,
                query_set.as_ref(),
                ref_set.as_ref(),
                args.cutoff,
            )
        })
        .collect();

    // Write output
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);
    writeln!(
        output,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length"
    )?;

    let mut n_pairs = 0u64;
    for (i, pairs) in window_results.iter().enumerate() {
        let (window_start, window_end) = windows[i];
        for pair in pairs {
            writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
                region.chrom,
                window_start,
                window_end,
                pair.group_a,
                pair.group_b,
                pair.identity,
                pair.a_length,
                pair.b_length
            )?;
        }
        n_pairs += pairs.len() as u64;
    }
    output.flush()?;

    eprintln!(
        "  {} windows, {} pairs in {:.1}s",
        windows.len(),
        n_pairs,
        compute_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "Total time: {:.1}s. Output: {}",
        total_start.elapsed().as_secs_f64(),
        args.output
    );

    Ok(())
}
