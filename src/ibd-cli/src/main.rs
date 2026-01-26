//! IBD segment detection using HMM

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;

use hprc_ibd::hmm::{extract_ibd_segments, viterbi, HmmParams, Population};
use hprc_ibd::{Region, WindowIterator};

#[derive(Parser, Debug)]
#[command(name = "ibd", version, about = "IBD segment detection using HMM")]
struct Args {
    #[arg(long = "sequence-files", required = true)]
    sequence_files: String,

    #[arg(short = 'a', required = true)]
    align: String,

    #[arg(short = 'r', required = true)]
    ref_name: String,

    #[arg(long = "region", required = true)]
    region: String,

    #[arg(long = "size", required = true)]
    window_size: u64,

    #[arg(long = "subset-sequence-list", required = true)]
    subset_list: String,

    #[arg(long = "output", required = true)]
    output: String,

    #[arg(long = "ibs-output")]
    ibs_output: Option<String>,

    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Minimum IBD segment length in base pairs (default: 2 Mb for reliable detection)
    #[arg(long = "min-len-bp", default_value = "2000000")]
    min_len_bp: u64,

    /// Minimum number of consecutive windows for IBD segment (default: 400 = 2 Mb with 5kb windows)
    #[arg(long = "min-windows", default_value = "400")]
    min_windows: usize,

    #[arg(long = "expected-seg-windows", default_value = "50")]
    expected_seg_windows: f64,

    #[arg(long = "p-enter-ibd", default_value = "0.0001")]
    p_enter_ibd: f64,

    /// Population for HMM parameters (AFR, EUR, EAS, CSA, AMR, InterPop, Generic)
    #[arg(long = "population", default_value = "Generic")]
    population: String,

    /// Number of threads for parallel HMM processing (default: auto-detect)
    #[arg(short = 't', long = "threads")]
    threads: Option<usize>,
}

#[derive(Debug, Clone)]
struct WindowRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
}

fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) }
}

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
            columns.iter().position(|&c| c == name).context(format!("Missing: {}", name))
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

fn collect_identities(
    args: &Args,
    region: &Region,
    mut ibs_output: Option<&mut BufWriter<File>>,
) -> Result<HashMap<(String, String), Vec<WindowRecord>>> {
    let ref_prefix = format!("{}#", args.ref_name);
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let mut first_window = true;

    for window in WindowIterator::new(region, args.window_size) {
        let ref_region = format!("{}#0#{}:{}-{}", args.ref_name, region.chrom, window.start, window.end);
        eprintln!("Collecting identities for {}", ref_region);

        let mut cmd = Command::new("impg");
        cmd.arg("similarity")
            .arg("--sequence-files").arg(&args.sequence_files)
            .arg("-a").arg(&args.align)
            .arg("-r").arg(&ref_region)
            .arg("--subset-sequence-list").arg(&args.subset_list)
            .arg("--force-large-region")
            .stdout(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn impg")?;
        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        let header = lines.next().context("No output")?.context("Failed to read header")?;
        let cols = ColumnIndices::from_header(&header)?;

        if let Some(ref mut out) = ibs_output {
            if first_window {
                writeln!(out, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;
            }
        }

        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() <= cols.estimated_identity.max(cols.group_a).max(cols.group_b) {
                continue;
            }

            let identity: f64 = match fields[cols.estimated_identity].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let group_a = fields[cols.group_a];
            let group_b = fields[cols.group_b];

            if group_a == group_b { continue; }
            if group_a.starts_with(&ref_prefix) || group_b.starts_with(&ref_prefix) { continue; }
            if group_a > group_b { continue; }

            let chrom = fields[cols.chrom].to_string();
            let start: u64 = match fields[cols.start].parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("WARNING: invalid start coordinate '{}', skipping line", fields[cols.start]);
                    continue;
                }
            };
            let end: u64 = match fields[cols.end].parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("WARNING: invalid end coordinate '{}', skipping line", fields[cols.end]);
                    continue;
                }
            };

            if let Some(ref mut out) = ibs_output {
                writeln!(out, "{}\t{}\t{}\t{}\t{}\t{}", chrom, start, end, group_a, group_b, identity)?;
            }

            let key = pair_key(group_a, group_b);
            pair_data.entry(key).or_default().push(WindowRecord { chrom, start, end, identity });
        }

        let status = child.wait()?;
        if !status.success() { bail!("impg failed"); }
        first_window = false;
    }

    Ok(pair_data)
}

/// IBD segment result from HMM processing
#[derive(Debug, Clone)]
struct IbdSegment {
    chrom: String,
    start_bp: u64,
    end_bp: u64,
    hap_a: String,
    hap_b: String,
    n_windows: usize,
    mean_identity: f64,
}

/// Process a single haplotype pair and return IBD segments
fn process_pair(
    hap_a: String,
    hap_b: String,
    mut records: Vec<WindowRecord>,
    expected_seg_windows: f64,
    p_enter_ibd: f64,
    min_windows: usize,
    min_len_bp: u64,
    population: Population,
) -> Vec<IbdSegment> {
    if records.len() < 3 {
        return Vec::new();
    }

    records.sort_by_key(|r| r.start);
    let observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

    // Use population-specific parameters for biologically correct HMM
    let mut params = HmmParams::from_population(population, expected_seg_windows, p_enter_ibd);
    // Use robust estimation with population priors
    params.estimate_emissions_robust(&observations, Some(population));

    let states = viterbi(&observations, &params);
    let segments = extract_ibd_segments(&states);

    let mut results = Vec::new();
    for (start_idx, end_idx, n_windows) in segments {
        if n_windows < min_windows {
            continue;
        }

        let start_bp = records[start_idx].start;
        let end_bp = records[end_idx].end;
        let length_bp = end_bp.saturating_sub(start_bp);

        if length_bp < min_len_bp {
            continue;
        }

        let mean_identity: f64 =
            observations[start_idx..=end_idx].iter().sum::<f64>() / n_windows as f64;

        results.push(IbdSegment {
            chrom: records[start_idx].chrom.clone(),
            start_bp,
            end_bp,
            hap_a: hap_a.clone(),
            hap_b: hap_b.clone(),
            n_windows,
            mean_identity,
        });
    }

    results
}

/// Process all haplotype pairs in parallel and return IBD segments
fn call_ibd_segments(
    pair_data: HashMap<(String, String), Vec<WindowRecord>>,
    args: &Args,
    population: Population,
) -> Vec<IbdSegment> {
    eprintln!("Running HMM on {} pairs in parallel with population {:?}...", pair_data.len(), population);

    // Convert HashMap to Vec for parallel iteration
    let pairs: Vec<_> = pair_data.into_iter().collect();

    // Process pairs in parallel
    pairs
        .into_par_iter()
        .flat_map(|((hap_a, hap_b), records)| {
            process_pair(
                hap_a,
                hap_b,
                records,
                args.expected_seg_windows,
                args.p_enter_ibd,
                args.min_windows,
                args.min_len_bp,
                population,
            )
        })
        .collect()
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Parse population
    let population = Population::from_str(&args.population)
        .ok_or_else(|| anyhow::anyhow!("Invalid population '{}'. Valid options: AFR, EUR, EAS, CSA, AMR, InterPop, Generic", args.population))?;

    // Configure thread pool if --threads is specified
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .context("Failed to configure thread pool")?;
        eprintln!("Using {} threads for parallel processing", num_threads);
    }

    if Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }

    let region = Region::parse(&args.region, args.region_length)?;
    eprintln!("Processing region {}:{}-{}", region.chrom, region.start, region.end);
    eprintln!("Using population-specific parameters for {:?} (π = {:.5})", population, population.diversity());

    let mut ibs_output = match &args.ibs_output {
        Some(path) => Some(BufWriter::new(File::create(path)?)),
        None => None,
    };

    let pair_data = collect_identities(&args, &region, ibs_output.as_mut())?;
    if let Some(ref mut out) = ibs_output {
        out.flush()?;
    }

    eprintln!("Collected data for {} pairs", pair_data.len());

    // Process pairs in parallel using rayon with population-specific HMM
    let segments = call_ibd_segments(pair_data, &args, population);

    // Write results to output file
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);

    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity")?;
    for seg in &segments {
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            seg.chrom, seg.start_bp, seg.end_bp, seg.hap_a, seg.hap_b, seg.n_windows, seg.mean_identity
        )?;
    }

    output.flush()?;
    eprintln!("IBD complete: {} segments written to {}", segments.len(), args.output);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
