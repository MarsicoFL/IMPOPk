use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;
use hprc_common::{ColumnIndices, Region};
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
        Err("value must be > 0 (window_size=0 causes infinite loop)".to_string())
    }
}

/// ibs: wrapper around `impg similarity` to obtain IBS segments.
///
/// Pipeline:
///   1. Slide a window across a reference chromosome.
///   2. Run `impg similarity` in each window.
///   3. For each window, immediately:
///        - filter rows by estimated.identity >= cutoff
///        - drop self-self and ref-involving comparisons
///        - drop duplicated A–B / B–A (keep canonical order)
///        - reduce to: chrom, start, end, group.a, group.b, estimated.identity
///        - append to output (streaming)
#[derive(Parser, Debug)]
#[command(name = "ibs", version, about)]
struct Args {
    /// Sequence file(s) for impg (e.g. .agc)
    #[arg(long = "sequence-files", required = true)]
    sequence_files: String,

    /// Alignment file (.paf/.paf.gz/.1aln) [passed to impg as -p]
    #[arg(short = 'a', required = true)]
    align: String,

    /// Reference name (e.g. CHM13)
    #[arg(short = 'r', required = true)]
    ref_name: String,

    /// Region, e.g. chr1:1-248956422 or chr1
    #[arg(long = "region", required = true)]
    region: String,

    /// Window size in bp (must be > 0; window_size=0 causes infinite loop)
    #[arg(long = "size", required = true, value_parser = validate_positive_u64)]
    window_size: u64,

    /// Haplotypes to compare (e.g. ibs_example.txt)
    #[arg(long = "subset-sequence-list", required = true)]
    subset_list: String,

    /// Output file
    #[arg(long = "output", required = true)]
    output: String,

    /// Cutoff on estimated.identity (must be in [0.0, 1.0], default: 0.999 to account for sequencing errors)
    #[arg(short = 'c', default_value = "0.999", value_parser = validate_cutoff)]
    cutoff: f64,

    /// Metric (only informational for now)
    #[arg(short = 'm', default_value = "cosin")]
    metric: String,

    /// Total length of REGION if you use -region chr1 (without coordinates)
    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Number of threads for parallel processing (default: auto-detect)
    #[arg(short = 't', long = "threads")]
    threads: Option<usize>,
}

/// Filtered row from impg output
#[derive(Clone)]
struct FilteredRow {
    chrom: String,
    start: String,
    end: String,
    group_a: String,
    group_b: String,
    identity: f64,
}

/// Process a single window and return filtered results
fn process_window(
    args: &Args,
    region: &Region,
    window_start: u64,
    window_end: u64,
) -> Result<Vec<FilteredRow>> {
    let ref_region = format!(
        "{}#0#{}:{}-{}",
        args.ref_name, region.chrom, window_start, window_end
    );

    eprintln!("Processing window {}", ref_region);

    // Run impg similarity
    let mut child = Command::new("impg")
        .arg("similarity")
        .arg("--sequence-files")
        .arg(&args.sequence_files)
        .arg("-a")
        .arg(&args.align)
        .arg("-r")
        .arg(&ref_region)
        .arg("--subset-sequence-list")
        .arg(&args.subset_list)
        .arg("--force-large-region")
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to spawn impg process. Is 'impg' in PATH?")?;

    let stdout = child.stdout.take()
        .context("Failed to capture stdout from impg")?;
    let reader = BufReader::new(stdout);

    let mut lines = reader.lines();

    // Parse header
    let header_line = lines.next()
        .context("impg produced no output")?
        .context("Failed to read header line")?;

    let cols = ColumnIndices::from_header(&header_line)
        .context("Failed to parse impg header")?;

    let ref_prefix = format!("{}#", args.ref_name);
    let mut results = Vec::new();

    // Process data rows
    for line_result in lines {
        let line = line_result.context("Failed to read line from impg output")?;
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() <= cols.max_index() {
            continue; // Skip malformed lines
        }

        // Parse estimated identity and apply cutoff
        let identity: f64 = match fields[cols.estimated_identity].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        if identity < args.cutoff {
            continue;
        }

        let group_a = fields[cols.group_a];
        let group_b = fields[cols.group_b];

        // Skip self-self comparisons
        if group_a == group_b {
            continue;
        }

        // Skip comparisons involving the reference
        if group_a.starts_with(&ref_prefix) || group_b.starts_with(&ref_prefix) {
            continue;
        }

        // Keep only canonical order (A < B lexicographically)
        if group_a > group_b {
            continue;
        }

        // Collect the filtered row
        results.push(FilteredRow {
            chrom: fields[cols.chrom].to_string(),
            start: fields[cols.start].to_string(),
            end: fields[cols.end].to_string(),
            group_a: group_a.to_string(),
            group_b: group_b.to_string(),
            identity,
        });
    }

    // Wait for child process to complete
    let status = child.wait().context("Failed to wait for impg process")?;
    if !status.success() {
        bail!("impg process exited with status: {}", status);
    }

    Ok(results)
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Validate input files exist
    if !Path::new(&args.sequence_files).exists() {
        bail!("sequence-files does not exist: {}", args.sequence_files);
    }
    if !Path::new(&args.align).exists() {
        bail!("alignment file does not exist: {}", args.align);
    }
    if !Path::new(&args.subset_list).exists() {
        bail!("subset-sequence-list does not exist: {}", args.subset_list);
    }

    // Configure thread pool if --threads is specified
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .context("Failed to configure thread pool")?;
        eprintln!("Using {} threads for parallel processing", num_threads);
    }

    // Check that impg is available
    if Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }

    // Parse region
    let region = Region::parse(&args.region, args.region_length)?;

    // Collect all windows into a Vec for parallel processing
    let mut windows = Vec::new();
    let mut start_pos = region.start;
    while start_pos <= region.end {
        let end_pos = (start_pos + args.window_size - 1).min(region.end);
        windows.push((start_pos, end_pos));
        start_pos = end_pos + 1;
    }

    eprintln!("Processing {} windows in parallel...", windows.len());

    // Process windows in parallel using rayon
    let results: Result<Vec<Vec<FilteredRow>>> = windows
        .par_iter()
        .map(|(start, end)| process_window(&args, &region, *start, *end))
        .collect();

    let all_results = results?;

    // Open output file and write results
    let output_file = File::create(&args.output)
        .context(format!("Failed to create output file: {}", args.output))?;
    let mut output = BufWriter::new(output_file);

    // Write header
    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;

    // Write all results (flattened from all windows)
    for window_results in all_results {
        for row in window_results {
            writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}",
                row.chrom, row.start, row.end, row.group_a, row.group_b, row.identity
            )?;
        }
    }

    output.flush()?;
    eprintln!("IBS written to: {}", args.output);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
