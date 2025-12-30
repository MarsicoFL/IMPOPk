use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;

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

    /// Window size in bp
    #[arg(long = "size", required = true)]
    window_size: u64,

    /// Haplotypes to compare (e.g. ibs_example.txt)
    #[arg(long = "subset-sequence-list", required = true)]
    subset_list: String,

    /// Output file
    #[arg(long = "output", required = true)]
    output: String,

    /// Cutoff on estimated.identity (default: 1.0)
    #[arg(short = 'c', default_value = "1.0")]
    cutoff: f64,

    /// Metric (only informational for now)
    #[arg(short = 'm', default_value = "cosin")]
    metric: String,

    /// Total length of REGION if you use -region chr1 (without coordinates)
    #[arg(long = "region-length")]
    region_length: Option<u64>,
}

/// Parsed region information
struct Region {
    chrom: String,
    start: u64,
    end: u64,
}

impl Region {
    fn parse(region_str: &str, region_length: Option<u64>) -> Result<Self> {
        if let Some(colon_pos) = region_str.find(':') {
            // Format: chr1:1-248956422
            let chrom = region_str[..colon_pos].to_string();
            let rest = &region_str[colon_pos + 1..];
            
            let dash_pos = rest.find('-')
                .context("Invalid region format: expected 'chrom:start-end'")?;
            
            let start: u64 = rest[..dash_pos].parse()
                .context("Invalid start position in region")?;
            let end: u64 = rest[dash_pos + 1..].parse()
                .context("Invalid end position in region")?;
            
            Ok(Region { chrom, start, end })
        } else {
            // Format: chr1 (requires --region-length)
            let end = region_length
                .context(format!("-region '{}' needs --region-length", region_str))?;
            
            Ok(Region {
                chrom: region_str.to_string(),
                start: 1,
                end,
            })
        }
    }
}

/// Column indices from the similarity output header
struct ColumnIndices {
    estimated_identity: usize,
    chrom: usize,
    start: usize,
    end: usize,
    group_a: usize,
    group_b: usize,
}

impl ColumnIndices {
    fn from_header(header: &str) -> Result<Self> {
        let columns: Vec<&str> = header.split('\t').collect();
        
        let find_col = |name: &str| -> Result<usize> {
            columns.iter()
                .position(|&c| c == name)
                .context(format!("Missing required column: {}", name))
        };
        
        Ok(ColumnIndices {
            estimated_identity: find_col("estimated.identity")?,
            chrom: find_col("chrom")?,
            start: find_col("start")?,
            end: find_col("end")?,
            group_a: find_col("group.a")?,
            group_b: find_col("group.b")?,
        })
    }
}

/// Process a single window and write filtered results to output
fn process_window(
    args: &Args,
    region: &Region,
    window_start: u64,
    window_end: u64,
    output: &mut BufWriter<File>,
    write_header: bool,
) -> Result<()> {
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
        .arg("-p")
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
    
    let cols = ColumnIndices::from_header(&header_line)?;
    
    // Write output header if this is the first window
    if write_header {
        writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;
    }
    
    let ref_prefix = format!("{}#", args.ref_name);
    
    // Process data rows
    for line_result in lines {
        let line = line_result.context("Failed to read line from impg output")?;
        let fields: Vec<&str> = line.split('\t').collect();
        
        if fields.len() <= cols.estimated_identity.max(cols.group_a).max(cols.group_b) {
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
        
        // Write the filtered row
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}",
            fields[cols.chrom],
            fields[cols.start],
            fields[cols.end],
            group_a,
            group_b,
            identity
        )?;
    }
    
    // Wait for child process to complete
    let status = child.wait().context("Failed to wait for impg process")?;
    if !status.success() {
        bail!("impg process exited with status: {}", status);
    }
    
    Ok(())
}

fn run() -> Result<()> {
    let args = Args::parse();
    
    // Check that impg is available
    if Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }
    
    // Parse region
    let region = Region::parse(&args.region, args.region_length)?;
    
    // Open output file
    let output_file = File::create(&args.output)
        .context(format!("Failed to create output file: {}", args.output))?;
    let mut output = BufWriter::new(output_file);
    
    // Process windows
    let mut start_pos = region.start;
    let mut first_window = true;
    
    while start_pos <= region.end {
        let end_pos = (start_pos + args.window_size - 1).min(region.end);
        
        process_window(
            &args,
            &region,
            start_pos,
            end_pos,
            &mut output,
            first_window,
        )?;
        
        first_window = false;
        start_pos = end_pos + 1;
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
