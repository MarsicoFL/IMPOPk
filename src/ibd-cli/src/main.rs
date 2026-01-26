//! IBD segment detection using HMM

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;

fn validate_probability(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(format!("probability must be in (0.0, 1.0), got {}", v))
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

fn validate_positive_usize(val: &str) -> Result<usize, String> {
    let v: usize = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0 {
        Ok(v)
    } else {
        Err("value must be > 0".to_string())
    }
}
use rayon::prelude::*;

use hprc_common::ColumnIndices;
use hprc_ibd::hmm::{infer_ibd, extract_ibd_segments_with_posteriors, HmmParams, Population};
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

    /// Window size in bp (must be > 0)
    #[arg(long = "size", required = true, value_parser = validate_positive_u64)]
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
    #[arg(long = "min-windows", default_value = "400", value_parser = validate_positive_usize)]
    min_windows: usize,

    #[arg(long = "expected-seg-windows", default_value = "50")]
    expected_seg_windows: f64,

    /// Probability of entering IBD state (must be in (0.0, 1.0))
    #[arg(long = "p-enter-ibd", default_value = "0.0001", value_parser = validate_probability)]
    p_enter_ibd: f64,

    /// Population for HMM parameters (AFR, EUR, EAS, CSA, AMR, InterPop, Generic)
    #[arg(long = "population", default_value = "Generic")]
    population: String,

    /// Number of threads for parallel HMM processing (default: auto-detect)
    #[arg(short = 't', long = "threads")]
    threads: Option<usize>,

    /// Minimum mean posterior P(IBD) for segment to be reported (0.0-1.0)
    /// Uses forward-backward algorithm for posterior computation
    #[arg(long = "posterior-threshold", default_value = "0.0")]
    posterior_threshold: f64,

    /// Output file for per-window posteriors (optional)
    /// Format: chrom, start, end, group.a, group.b, identity, posterior
    #[arg(long = "output-posteriors")]
    output_posteriors: Option<String>,
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
        let cols = ColumnIndices::from_header(&header)
            .context("Failed to parse impg header")?;

        if let Some(ref mut out) = ibs_output {
            if first_window {
                writeln!(out, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;
            }
        }

        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() <= cols.max_index() {
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
    mean_posterior: f64,
    min_posterior: f64,
    max_posterior: f64,
}

/// Per-window posterior for optional output
#[derive(Debug, Clone)]
struct WindowPosterior {
    chrom: String,
    start: u64,
    end: u64,
    hap_a: String,
    hap_b: String,
    identity: f64,
    posterior: f64,
}

/// Result from processing a haplotype pair
struct PairResult {
    segments: Vec<IbdSegment>,
    posteriors: Vec<WindowPosterior>,
}

/// Process a single haplotype pair and return IBD segments with posteriors
fn process_pair(
    hap_a: String,
    hap_b: String,
    mut records: Vec<WindowRecord>,
    expected_seg_windows: f64,
    p_enter_ibd: f64,
    min_windows: usize,
    min_len_bp: u64,
    posterior_threshold: f64,
    population: Population,
    window_size: u64,
    collect_posteriors: bool,
) -> PairResult {
    if records.len() < 3 {
        return PairResult {
            segments: Vec::new(),
            posteriors: Vec::new(),
        };
    }

    records.sort_by_key(|r| r.start);
    let observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

    // Use population-specific parameters for biologically correct HMM
    let mut params = HmmParams::from_population(population, expected_seg_windows, p_enter_ibd, window_size);
    // Use robust estimation with population priors
    params.estimate_emissions_robust(&observations, Some(population), window_size);

    // Run complete inference: Viterbi + forward-backward
    let inference = infer_ibd(&observations, &params);

    // Extract segments with posterior filtering
    let hmm_segments = extract_ibd_segments_with_posteriors(
        &inference.states,
        &inference.posteriors,
        min_windows,
        posterior_threshold,
    );

    let mut segments = Vec::new();
    for seg in hmm_segments {
        let start_bp = records[seg.start_idx].start;
        let end_bp = records[seg.end_idx].end;
        let length_bp = end_bp.saturating_sub(start_bp);

        if length_bp < min_len_bp {
            continue;
        }

        let mean_identity: f64 =
            observations[seg.start_idx..=seg.end_idx].iter().sum::<f64>() / seg.n_windows as f64;

        segments.push(IbdSegment {
            chrom: records[seg.start_idx].chrom.clone(),
            start_bp,
            end_bp,
            hap_a: hap_a.clone(),
            hap_b: hap_b.clone(),
            n_windows: seg.n_windows,
            mean_identity,
            mean_posterior: seg.mean_posterior,
            min_posterior: seg.min_posterior,
            max_posterior: seg.max_posterior,
        });
    }

    // Collect per-window posteriors if requested
    let posteriors = if collect_posteriors {
        records
            .iter()
            .zip(inference.posteriors.iter())
            .zip(observations.iter())
            .map(|((rec, &post), &ident)| WindowPosterior {
                chrom: rec.chrom.clone(),
                start: rec.start,
                end: rec.end,
                hap_a: hap_a.clone(),
                hap_b: hap_b.clone(),
                identity: ident,
                posterior: post,
            })
            .collect()
    } else {
        Vec::new()
    };

    PairResult { segments, posteriors }
}

/// Result from processing all pairs
struct AllPairsResult {
    segments: Vec<IbdSegment>,
    posteriors: Vec<WindowPosterior>,
}

/// Process all haplotype pairs in parallel and return IBD segments
fn call_ibd_segments(
    pair_data: HashMap<(String, String), Vec<WindowRecord>>,
    args: &Args,
    population: Population,
    collect_posteriors: bool,
) -> AllPairsResult {
    eprintln!("Running HMM on {} pairs in parallel with population {:?}...", pair_data.len(), population);
    if args.posterior_threshold > 0.0 {
        eprintln!("Filtering segments with mean P(IBD) >= {:.2}", args.posterior_threshold);
    }

    // Convert HashMap to Vec for parallel iteration
    let pairs: Vec<_> = pair_data.into_iter().collect();

    // Process pairs in parallel
    let results: Vec<PairResult> = pairs
        .into_par_iter()
        .map(|((hap_a, hap_b), records)| {
            process_pair(
                hap_a,
                hap_b,
                records,
                args.expected_seg_windows,
                args.p_enter_ibd,
                args.min_windows,
                args.min_len_bp,
                args.posterior_threshold,
                population,
                args.window_size,
                collect_posteriors,
            )
        })
        .collect();

    // Flatten results
    let mut all_segments = Vec::new();
    let mut all_posteriors = Vec::new();
    for result in results {
        all_segments.extend(result.segments);
        all_posteriors.extend(result.posteriors);
    }

    AllPairsResult {
        segments: all_segments,
        posteriors: all_posteriors,
    }
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
    let collect_posteriors = args.output_posteriors.is_some();
    let result = call_ibd_segments(pair_data, &args, population, collect_posteriors);

    // Write IBD segments to output file
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);

    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity\tmean_posterior\tmin_posterior\tmax_posterior")?;
    for seg in &result.segments {
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}",
            seg.chrom, seg.start_bp, seg.end_bp, seg.hap_a, seg.hap_b,
            seg.n_windows, seg.mean_identity, seg.mean_posterior, seg.min_posterior, seg.max_posterior
        )?;
    }
    output.flush()?;
    eprintln!("IBD complete: {} segments written to {}", result.segments.len(), args.output);

    // Write per-window posteriors if requested
    if let Some(ref posteriors_path) = args.output_posteriors {
        let posteriors_file = File::create(posteriors_path)?;
        let mut posteriors_out = BufWriter::new(posteriors_file);

        writeln!(posteriors_out, "chrom\tstart\tend\tgroup.a\tgroup.b\tidentity\tposterior")?;
        for wp in &result.posteriors {
            writeln!(
                posteriors_out,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}",
                wp.chrom, wp.start, wp.end, wp.hap_a, wp.hap_b, wp.identity, wp.posterior
            )?;
        }
        posteriors_out.flush()?;
        eprintln!("Per-window posteriors written to {}", posteriors_path);
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
