//! ibs-from-paf: Compute IBS identity directly from PAF alignments.
//!
//! This is a >100x faster alternative to the standard `ibs` command,
//! which calls `impg similarity` per window. Instead, this reads the
//! PAF alignment file once and computes pairwise identity from CIGAR strings.
//!
//! The output format is identical to `ibs` and can be used directly
//! with `ancestry-cli` and `ibd-cli`.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use hprc_common::Region;
use hprc_ibs::paf;
use rayon::prelude::*;

fn validate_cutoff(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("cutoff must be in [0.0, 1.0], got {}", v))
    }
}

fn validate_positive_u64(val: &str) -> Result<u64, String> {
    let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0 {
        Ok(v)
    } else {
        Err("value must be > 0".to_string())
    }
}

/// ibs-from-paf: compute IBS identity from PAF alignments directly.
///
/// Reads pangenome alignments (PAF format, optionally gzipped) and computes
/// pairwise identity between haplotypes for sliding windows on a reference
/// chromosome. Much faster than `ibs` because it reads pre-computed
/// alignments instead of calling `impg similarity` per window.
///
/// Output format is identical to `ibs` (tab-separated: chrom, start, end,
/// group.a, group.b, estimated.identity).
#[derive(Parser, Debug)]
#[command(name = "ibs-from-paf", version, about)]
struct Args {
    /// PAF alignment file (.paf or .paf.gz)
    #[arg(short = 'a', long = "alignment", required = true)]
    alignment: String,

    /// Reference name used in PAF target (e.g., CHM13)
    #[arg(short = 'r', long = "ref-name", default_value = "CHM13")]
    ref_name: String,

    /// Region to process, e.g., chr12:1-133324548 or chr12
    #[arg(long = "region", required_unless_present = "bed")]
    region: Option<String>,

    /// BED file with regions to process
    #[arg(long = "bed", conflicts_with = "region")]
    bed: Option<String>,

    /// Window size in bp
    #[arg(long = "size", required = true, value_parser = validate_positive_u64)]
    window_size: u64,

    /// Output file
    #[arg(long = "output", required = true)]
    output: String,

    /// Minimum identity cutoff for output pairs [0.0, 1.0]
    #[arg(short = 'c', default_value = "0.0", value_parser = validate_cutoff)]
    cutoff: f64,

    /// Minimum alignment block length to include from PAF
    #[arg(long = "min-aligned-length", default_value = "5000")]
    min_aligned_length: u64,

    /// Haplotype subset list (one per line). Only include these haplotypes.
    /// Entries can be sample names (e.g., HG00097) or haplotype IDs (e.g., HG00097#1).
    #[arg(long = "subset-sequence-list")]
    subset_list: Option<String>,

    /// File with query sample names (for query-vs-ref mode)
    #[arg(long = "query-samples")]
    query_samples: Option<String>,

    /// File with reference sample names (for query-vs-ref mode)
    #[arg(long = "ref-samples")]
    ref_samples: Option<String>,

    /// Total length of region if --region is chrom-only (no coordinates)
    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Number of threads for parallel window processing (default: available cores)
    #[arg(short = 't', long = "threads", default_value = "0")]
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

fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
    let file = File::open(bed_path).context(format!("Failed to open BED: {}", bed_path))?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();
    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            bail!("BED line {} has fewer than 3 fields", line_num + 1);
        }
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse().context("Invalid start")?;
        let end: u64 = fields[2].parse().context("Invalid end")?;
        regions.push(Region {
            chrom,
            start: start + 1, // BED 0-based → 1-based
            end,
        });
    }
    Ok(regions)
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Validate files
    if !Path::new(&args.alignment).exists() {
        bail!("Alignment file not found: {}", args.alignment);
    }

    // Configure rayon thread pool
    let n_threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // Ignore error if pool already initialized
    eprintln!("Using {} threads for parallel window processing", n_threads);

    // Load subset list
    let subset = if let Some(ref path) = args.subset_list {
        let set = load_sample_set(path)?;
        eprintln!("Loaded {} haplotype/sample IDs from subset list", set.len());
        Some(set)
    } else {
        None
    };

    // Load query/ref filters
    let query_set = if let Some(ref path) = args.query_samples {
        let set = load_sample_set(path)?;
        eprintln!("Loaded {} query samples", set.len());
        Some(set)
    } else {
        None
    };
    let ref_set = if let Some(ref path) = args.ref_samples {
        let set = load_sample_set(path)?;
        eprintln!("Loaded {} reference samples", set.len());
        Some(set)
    } else {
        None
    };

    // Collect regions
    let regions = if let Some(ref bed_path) = args.bed {
        parse_bed_regions(bed_path)?
    } else if let Some(ref region_str) = args.region {
        vec![Region::parse(region_str, args.region_length)?]
    } else {
        bail!("Either --region or --bed must be specified");
    };

    // Get unique chromosomes from regions
    let chroms: Vec<String> = {
        let mut c: Vec<String> = regions.iter().map(|r| r.chrom.clone()).collect();
        c.sort();
        c.dedup();
        c
    };

    // Open output
    let output_file =
        File::create(&args.output).context(format!("Failed to create output: {}", args.output))?;
    let mut output = BufWriter::new(output_file);
    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length")?;

    let total_start = Instant::now();

    // Read PAF file: use multi-chrom read if >1 chromosome to avoid re-reading
    let multi_chrom_data = if chroms.len() > 1 {
        let chrom_set: HashSet<String> = chroms.iter().cloned().collect();
        eprintln!("Reading PAF alignments for {} chromosomes in single pass...", chroms.len());
        let data = paf::read_paf_alignments_multi(
            &args.alignment,
            &chrom_set,
            subset.as_ref(),
            args.min_aligned_length,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        Some(data)
    } else {
        None
    };

    // Process each chromosome
    for chrom in &chroms {
        let chrom_start = Instant::now();

        let alignments = if let Some(ref multi) = multi_chrom_data {
            // Already loaded in the multi-chromosome pass
            match multi.get(chrom) {
                Some(alns) => {
                    eprintln!("Using {} pre-loaded alignments for {}", alns.len(), chrom);
                    std::borrow::Cow::Borrowed(alns.as_slice())
                }
                None => {
                    eprintln!("  WARNING: No alignments found for {}", chrom);
                    continue;
                }
            }
        } else {
            // Single chromosome: read directly
            eprintln!("Reading PAF alignments for {}...", chrom);
            let alns = paf::read_paf_alignments(
                &args.alignment,
                chrom,
                subset.as_ref(),
                args.min_aligned_length,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
            eprintln!(
                "  {} alignments loaded in {:.1}s",
                alns.len(),
                chrom_start.elapsed().as_secs_f64()
            );
            std::borrow::Cow::Owned(alns)
        };

        if alignments.is_empty() {
            eprintln!("  WARNING: No alignments found for {}", chrom);
            continue;
        }

        // Count unique haplotypes
        let hap_count: HashSet<&str> = alignments.iter().map(|a| a.hap_id.as_str()).collect();
        eprintln!("  {} unique haplotypes", hap_count.len());

        // Process regions for this chromosome
        let chrom_regions: Vec<&Region> = regions.iter().filter(|r| r.chrom == *chrom).collect();

        for region in &chrom_regions {
            eprintln!(
                "  Processing {}:{}-{} (window_size={}, threads={})",
                region.chrom, region.start, region.end, args.window_size, n_threads
            );

            // Generate all windows for this region
            let windows: Vec<(u64, u64)> = {
                let mut ws = Vec::new();
                let mut start = region.start;
                while start <= region.end {
                    let end = (start + args.window_size - 1).min(region.end);
                    ws.push((start, end));
                    start = end + 1;
                }
                ws
            };

            let compute_start = Instant::now();

            // Process windows in parallel using rayon
            // Each window's compute_window_pairwise is read-only on alignments
            let window_results: Vec<Vec<paf::PairwiseIdentity>> = windows
                .par_iter()
                .map(|&(window_start, window_end)| {
                    // Convert to 0-based for PAF (PAF uses 0-based coordinates)
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

            // Write results in order (preserves deterministic output)
            let mut n_pairs = 0u64;
            for (i, pairs) in window_results.iter().enumerate() {
                let (window_start, window_end) = windows[i];
                for pair in pairs {
                    writeln!(
                        output,
                        "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
                        chrom, window_start, window_end,
                        pair.group_a, pair.group_b, pair.identity,
                        pair.a_length, pair.b_length
                    )?;
                }
                n_pairs += pairs.len() as u64;
            }

            eprintln!(
                "  {} windows, {} pairs in {:.1}s ({:.0} windows/s)",
                windows.len(), n_pairs,
                compute_start.elapsed().as_secs_f64(),
                windows.len() as f64 / compute_start.elapsed().as_secs_f64().max(0.001)
            );
        }

        eprintln!(
            "  {} completed in {:.1}s",
            chrom,
            chrom_start.elapsed().as_secs_f64()
        );
    }

    output.flush()?;
    eprintln!(
        "Total time: {:.1}s. Output: {}",
        total_start.elapsed().as_secs_f64(),
        args.output
    );

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
