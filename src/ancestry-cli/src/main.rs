//! Local Ancestry Inference CLI
//!
//! Infers local ancestry from pangenome similarity data using an HMM.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    extract_ancestry_segments, forward_backward, glossophaga_populations,
    parse_similarity_data, viterbi,
    // NEW:
    estimate_temperature, estimate_switch_prob,
    smooth_states, count_smoothing_changes,
    cross_validate,
};

#[derive(Parser, Debug)]
#[command(name = "ancestry", version, about = "Local ancestry inference from pangenome data")]
struct Args {
    /// AGC file with assemblies
    #[arg(long = "sequence-files", required = true)]
    sequence_files: PathBuf,

    /// Alignment file (PAF)
    #[arg(short = 'a', long = "alignment", required = true)]
    alignment: PathBuf,

    /// Reference name for coordinate system (e.g., "soricina#HAP1")
    #[arg(short = 'r', long = "reference", required = true)]
    reference: String,

    /// Region to analyze (e.g., "super15" or "super15:1-1000000")
    #[arg(long = "region", required = true)]
    region: String,

    /// Window size in bp
    #[arg(long = "window-size", default_value = "5000")]
    window_size: u64,

    /// Query samples file (one sample#haplotype per line)
    #[arg(long = "query-samples", required = true)]
    query_samples: PathBuf,

    /// Population definition file (TSV: pop_name, haplotype_id)
    /// If not provided, uses default Glossophaga populations
    #[arg(long = "populations")]
    populations: Option<PathBuf>,

    /// Output file for ancestry segments
    #[arg(short = 'o', long = "output", required = true)]
    output: PathBuf,

    /// Output file for per-window posteriors (optional)
    #[arg(long = "posteriors-output")]
    posteriors_output: Option<PathBuf>,

    /// Ancestry switch probability per window
    #[arg(long = "switch-prob", default_value = "0.001")]
    switch_prob: f64,

    /// Minimum segment length in bp
    #[arg(long = "min-len-bp", default_value = "10000")]
    min_len_bp: u64,

    /// Minimum windows per segment
    #[arg(long = "min-windows", default_value = "3")]
    min_windows: usize,

    /// Region length (required if region is just chromosome name)
    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Number of threads
    #[arg(short = 't', long = "threads", default_value = "4")]
    threads: usize,

    /// Pre-computed similarity file (skip impg if provided)
    #[arg(long = "similarity-file")]
    similarity_file: Option<PathBuf>,

    /// Minimum mean posterior probability to keep a segment
    #[arg(long = "min-posterior", default_value = "0.0")]
    min_posterior: f64,

    /// Minimum consecutive windows to keep a state assignment (for smoothing)
    /// Set to 0 to disable smoothing
    #[arg(long = "smooth-min-windows", default_value = "0")]
    smooth_min_windows: usize,

    /// Run leave-one-out cross-validation on reference haplotypes
    #[arg(long = "cross-validate")]
    cross_validate: bool,

    /// Automatically estimate HMM parameters (temperature and switch-prob) from data
    /// When enabled, --switch-prob becomes initial value for regularization
    #[arg(long = "estimate-params")]
    estimate_params: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // Load populations
    let populations = if let Some(pop_file) = &args.populations {
        load_populations(pop_file)?
    } else {
        eprintln!("Using default Glossophaga populations");
        glossophaga_populations()
    };

    eprintln!("Populations: {:?}", populations.iter().map(|p| &p.name).collect::<Vec<_>>());

    // Load query samples
    let query_samples = load_sample_list(&args.query_samples)?;
    eprintln!("Query samples: {} haplotypes", query_samples.len());

    // Get reference haplotypes from populations
    let reference_haplotypes: Vec<String> = populations.iter()
        .flat_map(|p| p.haplotypes.clone())
        .collect();
    eprintln!("Reference haplotypes: {:?}", reference_haplotypes);

    // Parse region
    let (chrom, start, end) = parse_region(&args.region, &args.reference, args.region_length)?;
    eprintln!("Region: {}:{}-{} ({:.2} Mb)", chrom, start, end, (end - start) as f64 / 1_000_000.0);

    // Get similarity data
    let similarity_data = if let Some(sim_file) = &args.similarity_file {
        eprintln!("Reading pre-computed similarities from {:?}", sim_file);
        load_similarity_file(sim_file, &query_samples, &reference_haplotypes)?
    } else {
        eprintln!("Computing similarities with impg...");
        compute_similarities(
            &args.sequence_files,
            &args.alignment,
            &args.reference,
            &chrom,
            start,
            end,
            args.window_size,
            &query_samples,
            &reference_haplotypes,
        )?
    };

    eprintln!("Loaded observations for {} samples", similarity_data.len());

    // Estimate parameters if requested
    let (temperature, switch_prob) = if args.estimate_params {
        eprintln!("Estimating HMM parameters from data...");

        // Collect all observations for estimation
        let all_obs: Vec<&AncestryObservation> = similarity_data.values()
            .flat_map(|v| v.iter())
            .collect();

        // Convert to owned for estimation functions
        let obs_slice: Vec<AncestryObservation> = all_obs.iter().map(|o| (*o).clone()).collect();

        let temp = estimate_temperature(&obs_slice, &populations);
        let switch = estimate_switch_prob(&obs_slice, &populations, temp);

        eprintln!("  Estimated temperature: {:.4}", temp);
        eprintln!("  Estimated switch probability: {:.6}", switch);

        (temp, switch)
    } else {
        (0.03, args.switch_prob) // defaults
    };

    // Create HMM parameters with estimated or default values
    let mut params = AncestryHmmParams::new(populations.clone(), switch_prob);
    params.set_temperature(temperature);

    // Cross-validation if requested
    if args.cross_validate {
        eprintln!("\nRunning cross-validation...");
        let cv_result = cross_validate(&similarity_data, &populations, &params);
        cv_result.print_summary();

        if cv_result.has_bias() {
            eprintln!("\nWARNING: Cross-validation detected potential population bias!");
            eprintln!("Some populations have <50% accuracy. Results may be unreliable.");
        }
        eprintln!();
    }

    // Process each sample
    let results: Vec<_> = similarity_data.par_iter()
        .map(|(sample, observations)| {
            if observations.len() < 3 {
                eprintln!("  {} - too few observations ({})", sample, observations.len());
                return (sample.clone(), Vec::new(), Vec::new());
            }

            // Run Viterbi
            let states = viterbi(observations, &params);

            // Apply smoothing if requested
            let (states, n_smoothed) = if args.smooth_min_windows > 0 {
                let smoothed = smooth_states(&states, args.smooth_min_windows);
                let n_changed = count_smoothing_changes(&states, &smoothed);
                (smoothed, n_changed)
            } else {
                (states, 0)
            };

            // Run forward-backward for posteriors
            let posteriors = forward_backward(observations, &params);

            // Extract segments
            let segments = extract_ancestry_segments(
                observations,
                &states,
                &params,
                Some(&posteriors),
            );

            // Filter by length and posterior
            let filtered: Vec<_> = segments.into_iter()
                .filter(|s| {
                    s.end - s.start >= args.min_len_bp
                    && s.n_windows >= args.min_windows
                    && s.mean_posterior.unwrap_or(1.0) >= args.min_posterior
                })
                .collect();

            eprintln!("  {} - {} windows -> {} segments (smoothed {} windows)",
                sample, observations.len(), filtered.len(), n_smoothed);

            (sample.clone(), filtered, posteriors)
        })
        .collect();

    // Write output
    let output_file = File::create(&args.output)?;
    let mut out = BufWriter::new(output_file);

    writeln!(out, "chrom\tstart\tend\tsample\tancestry\tn_windows\tmean_similarity\tmean_posterior\tdiscriminability")?;

    for (_, segments, _) in &results {
        for seg in segments {
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}",
                seg.chrom,
                seg.start,
                seg.end,
                seg.sample,
                seg.ancestry_name,
                seg.n_windows,
                seg.mean_similarity,
                seg.mean_posterior.unwrap_or(0.0),
                seg.discriminability,
            )?;
        }
    }
    out.flush()?;

    eprintln!("Wrote ancestry segments to {:?}", args.output);

    // Write posteriors if requested
    if let Some(post_path) = &args.posteriors_output {
        let post_file = File::create(post_path)?;
        let mut post_out = BufWriter::new(post_file);

        // Header with population names
        write!(post_out, "chrom\tstart\tend\tsample")?;
        for pop in &populations {
            write!(post_out, "\tP({})", pop.name)?;
        }
        writeln!(post_out)?;

        for (sample, _, posteriors) in &results {
            if let Some(observations) = similarity_data.get(sample) {
                for (i, obs) in observations.iter().enumerate() {
                    if i < posteriors.len() {
                        write!(post_out, "{}\t{}\t{}\t{}", obs.chrom, obs.start, obs.end, sample)?;
                        for p in &posteriors[i] {
                            write!(post_out, "\t{:.6}", p)?;
                        }
                        writeln!(post_out)?;
                    }
                }
            }
        }
        post_out.flush()?;
        eprintln!("Wrote posteriors to {:?}", post_path);
    }

    // Print diagnostics
    print_diagnostics(&results, &params, &populations);

    Ok(())
}

fn print_diagnostics(
    results: &[(String, Vec<hprc_ancestry_cli::AncestrySegment>, Vec<Vec<f64>>)],
    params: &AncestryHmmParams,
    populations: &[AncestralPopulation],
) {
    eprintln!("\n=== Diagnostics ===");

    // Collect all max posteriors
    let all_max_posts: Vec<f64> = results.iter()
        .flat_map(|(_, _, posts)| {
            posts.iter().map(|p| p.iter().cloned().fold(0.0_f64, f64::max))
        })
        .collect();

    if all_max_posts.is_empty() {
        eprintln!("No posterior data available");
        return;
    }

    let confident = all_max_posts.iter().filter(|&&p| p > 0.8).count();
    let uncertain = all_max_posts.iter().filter(|&&p| p >= 0.5 && p <= 0.8).count();
    let ambiguous = all_max_posts.iter().filter(|&&p| p < 0.5).count();
    let total = all_max_posts.len();

    eprintln!("Posterior distribution ({} windows):", total);
    eprintln!("  Confident (>0.8):    {:>6} ({:>5.1}%)", confident, 100.0 * confident as f64 / total as f64);
    eprintln!("  Uncertain (0.5-0.8): {:>6} ({:>5.1}%)", uncertain, 100.0 * uncertain as f64 / total as f64);
    eprintln!("  Ambiguous (<0.5):    {:>6} ({:>5.1}%)", ambiguous, 100.0 * ambiguous as f64 / total as f64);

    eprintln!("\nHMM parameters:");
    eprintln!("  Temperature: {:.4}", params.emission_std);
    eprintln!("  Switch prob: {:.6}", params.transitions[0][1]);

    // Count segments per population
    eprintln!("\nSegments by ancestry:");
    for pop in populations {
        let count: usize = results.iter()
            .flat_map(|(_, segs, _)| segs.iter())
            .filter(|s| s.ancestry_name == pop.name)
            .count();
        eprintln!("  {}: {}", pop.name, count);
    }

    // Warnings
    if ambiguous as f64 / total as f64 > 0.1 {
        eprintln!("\nWARNING: >10% of windows are ambiguous (posterior <0.5)");
        eprintln!("Consider increasing temperature or checking data quality.");
    }

    if uncertain as f64 / total as f64 > 0.3 {
        eprintln!("\nWARNING: >30% of windows are uncertain (posterior 0.5-0.8)");
        eprintln!("Results should be interpreted with caution.");
    }
}

fn load_populations(path: &PathBuf) -> Result<Vec<AncestralPopulation>> {
    let file = File::open(path).context("Failed to open populations file")?;
    let reader = BufReader::new(file);

    let mut pop_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let pop_name = parts[0].to_string();
            let haplotype = parts[1].to_string();
            pop_map.entry(pop_name).or_default().push(haplotype);
        }
    }

    let populations: Vec<AncestralPopulation> = pop_map.into_iter()
        .map(|(name, haplotypes)| AncestralPopulation { name, haplotypes })
        .collect();

    Ok(populations)
}

fn load_sample_list(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path).context("Failed to open sample list")?;
    let reader = BufReader::new(file);

    let samples: Vec<String> = reader.lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .collect();

    Ok(samples)
}

fn parse_region(region: &str, reference: &str, region_length: Option<u64>) -> Result<(String, u64, u64)> {
    if region.contains(':') {
        // Format: chrom:start-end
        let parts: Vec<&str> = region.split(':').collect();
        let chrom = format!("{}#{}", reference, parts[0]);
        let coords: Vec<&str> = parts[1].split('-').collect();
        let start: u64 = coords[0].parse()?;
        let end: u64 = coords[1].parse()?;
        Ok((chrom, start, end))
    } else {
        // Just chromosome name, need region_length
        let end = region_length.context("--region-length required when region is just chromosome name")?;
        let chrom = format!("{}#{}", reference, region);
        Ok((chrom, 1, end))
    }
}

fn load_similarity_file(
    path: &PathBuf,
    query_samples: &[String],
    reference_haplotypes: &[String],
) -> Result<std::collections::HashMap<String, Vec<AncestryObservation>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines = reader.lines().filter_map(|l| l.ok());
    parse_similarity_data(lines, query_samples, reference_haplotypes)
        .map_err(|e| anyhow::anyhow!("Failed to parse similarity file: {}", e))
}

fn compute_similarities(
    sequence_files: &PathBuf,
    alignment: &PathBuf,
    _reference: &str,
    chrom: &str,
    start: u64,
    end: u64,
    window_size: u64,
    query_samples: &[String],
    reference_haplotypes: &[String],
) -> Result<std::collections::HashMap<String, Vec<AncestryObservation>>> {
    // Create combined sample list (queries + references)
    let _all_samples: HashSet<String> = query_samples.iter()
        .chain(reference_haplotypes.iter())
        .cloned()
        .collect();

    let mut all_lines = Vec::new();
    let mut pos = start;
    let mut window_count = 0;

    while pos < end {
        let window_end = (pos + window_size - 1).min(end);
        let region = format!("{}:{}-{}", chrom, pos, window_end);

        if window_count % 100 == 0 {
            eprintln!("  Window {}: {}", window_count, region);
        }

        // Run impg similarity
        let output = Command::new("impg")
            .arg("similarity")
            .arg("--sequence-files")
            .arg(sequence_files)
            .arg("-a")
            .arg(alignment)
            .arg("-r")
            .arg(&region)
            .arg("--force-large-region")
            .arg("-v")
            .arg("0")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .context("Failed to run impg")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            all_lines.push(line.to_string());
        }

        pos = window_end + 1;
        window_count += 1;
    }

    eprintln!("Collected {} similarity lines from {} windows", all_lines.len(), window_count);

    // Remove duplicate headers, keep first
    let mut seen_header = false;
    let filtered_lines: Vec<String> = all_lines.into_iter()
        .filter(|line| {
            if line.starts_with("chrom\t") {
                if seen_header {
                    false
                } else {
                    seen_header = true;
                    true
                }
            } else {
                true
            }
        })
        .collect();

    parse_similarity_data(
        filtered_lines.into_iter(),
        query_samples,
        reference_haplotypes,
    ).map_err(|e| anyhow::anyhow!("Failed to parse similarity data: {}", e))
}
