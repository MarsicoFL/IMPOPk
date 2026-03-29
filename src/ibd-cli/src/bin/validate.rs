//! IBD validation binary - reads IBS from file and runs HMM
//!
//! This binary is used for validation experiments where we have
//! pre-computed IBS data (either synthetic or from previous runs).
//! It supports population-specific HMM parameters, Baum-Welch training,
//! and posterior-based segment calling.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use hprc_ibd::concordance;
use hprc_ibd::hapibd;
use hprc_ibd::hmm::{
    coverage_ratio, extract_ibd_segments_with_lod, infer_ibd_with_aux_features,
    infer_ibd_with_training, forward_backward_with_distances,
    forward_backward_with_genetic_map, refine_states_with_posteriors,
    viterbi_with_distances, viterbi_with_genetic_map, GeneticMap, HmmParams,
    IbdInferenceResult, Population,
};
use hprc_ibd::stats::logit_transform_observations;

fn validate_probability(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(format!("probability must be in (0.0, 1.0), got {}", v))
    }
}

#[derive(Parser, Debug)]
#[command(name = "ibd-validate", version, about = "IBD validation from pre-computed IBS data")]
struct Args {
    /// Input IBS file (TSV with columns: chrom, start, end, group.a, group.b, estimated.identity)
    #[arg(short = 'i', long = "input", required = true)]
    input: PathBuf,

    /// Output IBD segments file
    #[arg(short = 'o', long = "output", required = true)]
    output: PathBuf,

    /// Output per-window states file (for detailed validation)
    #[arg(long = "states-output")]
    states_output: Option<PathBuf>,

    /// Minimum segment length in bp (default: 5kb for validation; use 2Mb for production)
    #[arg(long = "min-len-bp", default_value = "5000")]
    min_len_bp: u64,

    /// Minimum windows per segment (default: 3 for validation; use 400 for production with 5kb windows)
    #[arg(long = "min-windows", default_value = "3")]
    min_windows: usize,

    /// Expected IBD segment length in windows
    #[arg(long = "expected-seg-windows", default_value = "50")]
    expected_seg_windows: f64,

    /// Probability of entering IBD state
    #[arg(long = "p-enter-ibd", default_value = "0.0001", value_parser = validate_probability)]
    p_enter_ibd: f64,

    /// Window size in bp (for coordinate calculations)
    #[arg(long = "window-size", default_value = "5000")]
    window_size: u64,

    /// Population for HMM parameters (AFR, EUR, EAS, CSA, AMR, InterPop, Generic)
    #[arg(long = "population", default_value = "Generic")]
    population: String,

    /// Minimum mean posterior P(IBD) for segment to be reported (0.0-1.0)
    #[arg(long = "posterior-threshold", default_value = "0.0")]
    posterior_threshold: f64,

    /// Number of Baum-Welch iterations for parameter training (0 to disable)
    #[arg(long = "baum-welch-iters", default_value = "10")]
    baum_welch_iters: usize,

    /// Use population-adaptive transition probabilities
    #[arg(long = "adaptive-transitions", default_value = "false")]
    adaptive_transitions: bool,

    /// Use distance-aware HMM transitions.
    /// Scales transition probabilities by physical distance between windows.
    #[arg(long = "distance-aware", default_value = "false")]
    distance_aware: bool,

    /// Genetic map file (PLINK format) for recombination-aware transitions.
    /// Format: chr pos_bp rate_cM_Mb pos_cM (whitespace-separated)
    #[arg(long = "genetic-map")]
    genetic_map: Option<PathBuf>,

    /// Minimum identity threshold for including a window in HMM inference.
    /// Windows with identity below this threshold are discarded as alignment gaps.
    /// Useful for pangenome data where unaligned regions have very low identity.
    /// Default 0.0 (no filtering). Recommended: 0.9 for pangenome data.
    #[arg(long = "identity-floor", default_value = "0.0")]
    identity_floor: f64,

    /// Apply logit transform to identity observations before HMM inference.
    /// Maps identity values via log(x/(1-x)), stretching the near-1.0 region
    /// where pangenome data is concentrated. This improves Gaussian emission
    /// model fit for pangenome data where IBD/non-IBD identity differences
    /// are very small (0.997 vs 0.9997).
    #[arg(long = "logit-transform", default_value = "false")]
    logit_transform: bool,

    /// Enable region-specific background filtering for pangenome artifacts.
    /// Computes per-window population background: counts how many pairs show
    /// high identity at each genomic position. Windows where too many pairs
    /// are high-identity are masked as systematic artifacts (not IBD signal).
    #[arg(long = "background-filter", default_value = "false")]
    background_filter: bool,

    /// Identity threshold for counting a pair as "high identity" at a window
    /// position in background filtering (default: 0.999).
    #[arg(long = "bg-identity-threshold", default_value = "0.999")]
    bg_identity_threshold: f64,

    /// Maximum background ratio: mask windows where more than this fraction
    /// of pairs show high identity (default: 0.5 = mask if >50% of pairs).
    #[arg(long = "bg-ratio-threshold", default_value = "0.5")]
    bg_ratio_threshold: f64,

    /// Normalize identity values by subtracting the per-window population mean.
    /// Transforms identity from absolute to "excess identity" relative to background.
    /// This removes systematic regional biases in pangenome identity.
    #[arg(long = "bg-normalize", default_value = "false")]
    bg_normalize: bool,

    /// Use coverage ratio (min(a_len, b_len)/max(a_len, b_len)) as an auxiliary
    /// emission feature in the HMM. IBD regions have near-symmetric coverage (~1.0),
    /// while non-IBD regions may have asymmetric alignments (~0.85).
    /// Requires group.a.length and group.b.length columns in the IBS input.
    #[arg(long = "coverage-feature", default_value = "false")]
    coverage_feature: bool,

    /// BED file of regions to exclude from IBD output.
    /// Segments overlapping any excluded region are discarded post-HMM.
    /// Format: tab-separated chrom/start/end (0-based half-open).
    /// Useful for masking known artifact regions (e.g., high-identity hotspots).
    #[arg(long = "exclude-regions")]
    exclude_regions: Option<PathBuf>,

    /// hap-ibd .ibd file to validate against (computes concordance metrics)
    #[arg(long = "validate-against")]
    validate_against: Option<PathBuf>,

    /// Minimum LOD for hap-ibd segments in validation (default: 3.0)
    #[arg(long = "validate-min-lod", default_value = "3.0")]
    validate_min_lod: f64,

    /// Window size for per-window concordance in validation (default: 5000)
    #[arg(long = "validate-window-size", default_value = "5000")]
    validate_window_size: u64,

    /// Output file for validation concordance metrics
    #[arg(long = "validate-output")]
    validate_output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct IbsRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
    /// Alignment length of group A (0 if column not present)
    a_length: u64,
    /// Alignment length of group B (0 if column not present)
    b_length: u64,
}

/// Extract base haplotype ID from full ID (removes coordinate suffix if present)
/// e.g., "HG00280#2#JBHDWB010000002.1:130787850-130792849" -> "HG00280#2#JBHDWB010000002.1"
fn extract_haplotype_id(full_id: &str) -> String {
    // Check if there's a coordinate suffix (format: ...:#####-#####)
    if let Some(colon_pos) = full_id.rfind(':') {
        let after_colon = &full_id[colon_pos + 1..];
        // Check if what follows looks like coordinates (digits and hyphen)
        if after_colon.contains('-') && after_colon.chars().all(|c| c.is_ascii_digit() || c == '-') {
            return full_id[..colon_pos].to_string();
        }
    }
    full_id.to_string()
}

/// Extract sample ID from a haplotype ID
/// e.g., "HG00280#2#JBHDWB010000002.1" -> "HG00280"
/// e.g., "HG00280#2#JBHDWB010000002.1:130787850-130792849" -> "HG00280"
/// e.g., "HG00280" -> "HG00280"
fn extract_sample_id(hap_id: &str) -> String {
    hap_id.split('#').next().unwrap_or(hap_id).to_string()
}

fn read_ibs_file(path: &PathBuf) -> Result<HashMap<(String, String), Vec<IbsRecord>>> {
    let file = File::open(path).context("Failed to open IBS file")?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header
    let header = lines.next().context("Empty file")?.context("Failed to read header")?;
    let columns: Vec<&str> = header.split('\t').collect();

    let find_col = |name: &str| -> Result<usize> {
        columns.iter().position(|&c| c == name)
            .context(format!("Missing column: {}", name))
    };

    let col_chrom = find_col("chrom")?;
    let col_start = find_col("start")?;
    let col_end = find_col("end")?;
    let col_group_a = find_col("group.a")?;
    let col_group_b = find_col("group.b")?;
    let col_identity = find_col("estimated.identity")?;

    // Optional alignment length columns (for coverage ratio feature)
    let col_a_length = columns.iter().position(|&c| c == "group.a.length");
    let col_b_length = columns.iter().position(|&c| c == "group.b.length");

    let mut pair_data: HashMap<(String, String), Vec<IbsRecord>> = HashMap::new();

    for line_result in lines {
        let line = line_result?;
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() <= col_identity.max(col_group_a).max(col_group_b) {
            continue;
        }

        // Extract base haplotype IDs (without coordinate suffix)
        let group_a = extract_haplotype_id(fields[col_group_a]);
        let group_b = extract_haplotype_id(fields[col_group_b]);

        // Skip self-comparisons
        if group_a == group_b {
            continue;
        }

        // Ensure consistent ordering
        let key = if group_a <= group_b {
            (group_a, group_b)
        } else {
            (group_b, group_a)
        };

        let start: u64 = match fields[col_start].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("WARNING: invalid start '{}', skipping", fields[col_start]);
                continue;
            }
        };
        let end: u64 = match fields[col_end].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("WARNING: invalid end '{}', skipping", fields[col_end]);
                continue;
            }
        };
        let identity: f64 = match fields[col_identity].parse() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("WARNING: invalid identity '{}', skipping", fields[col_identity]);
                continue;
            }
        };

        // Parse alignment lengths (optional columns)
        let a_length: u64 = col_a_length
            .and_then(|col| fields.get(col))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let b_length: u64 = col_b_length
            .and_then(|col| fields.get(col))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let record = IbsRecord {
            chrom: fields[col_chrom].to_string(),
            start,
            end,
            identity,
            a_length,
            b_length,
        };

        pair_data.entry(key).or_default().push(record);
    }

    Ok(pair_data)
}

#[derive(Debug)]
struct IbdSegment {
    chrom: String,
    start: u64,
    end: u64,
    hap_a: String,
    hap_b: String,
    n_windows: usize,
    mean_identity: f64,
    mean_posterior: f64,
    min_posterior: f64,
    max_posterior: f64,
    lod_score: f64,
}

#[allow(clippy::type_complexity)]
fn process_pair(
    hap_a: &str,
    hap_b: &str,
    mut records: Vec<IbsRecord>,
    args: &Args,
    population: Population,
    genetic_map: Option<&GeneticMap>,
) -> (Vec<IbdSegment>, Vec<(u64, u64, usize, f64)>) {
    let mut segments = Vec::new();
    let mut window_states: Vec<(u64, u64, usize, f64)> = Vec::new();

    // Filter out alignment-gap windows (identity below floor)
    if args.identity_floor > 0.0 {
        records.retain(|r| r.identity >= args.identity_floor);
    }

    if records.len() < 3 {
        return (segments, window_states);
    }

    // Sort by start position
    records.sort_by_key(|r| r.start);

    let raw_observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

    // Compute coverage ratio auxiliary feature if requested
    let coverage_ratios: Option<Vec<f64>> = if args.coverage_feature {
        let ratios: Vec<f64> = records
            .iter()
            .map(|r| coverage_ratio(r.a_length, r.b_length))
            .collect();
        // Only use if we have non-trivial data (not all zeros from missing columns)
        if ratios.iter().any(|&r| r > 0.0) {
            Some(ratios)
        } else {
            eprintln!("WARNING: coverage-feature requested but alignment length columns missing/zero");
            None
        }
    } else {
        None
    };

    // Optionally transform observations to logit space
    let (observations, mut params) = if args.logit_transform {
        let logit_obs = logit_transform_observations(&raw_observations);
        let mut p = if args.adaptive_transitions {
            let mut p = HmmParams::from_population_logit(
                population, args.expected_seg_windows, args.p_enter_ibd, args.window_size,
            );
            // Apply adaptive transition scaling
            let base = HmmParams::from_population_adaptive(
                population, args.expected_seg_windows, args.p_enter_ibd, args.window_size,
            );
            p.transition = base.transition;
            p.initial = base.initial;
            p
        } else {
            HmmParams::from_population_logit(
                population, args.expected_seg_windows, args.p_enter_ibd, args.window_size,
            )
        };
        p.estimate_emissions_logit(&logit_obs, Some(population), args.window_size);
        (logit_obs, p)
    } else {
        let mut p = if args.adaptive_transitions {
            HmmParams::from_population_adaptive(
                population, args.expected_seg_windows, args.p_enter_ibd, args.window_size,
            )
        } else {
            HmmParams::from_population(
                population, args.expected_seg_windows, args.p_enter_ibd, args.window_size,
            )
        };
        p.estimate_emissions_robust(&raw_observations, Some(population), args.window_size);
        (raw_observations, p)
    };

    // Run inference: choose pipeline based on flags
    let result = if coverage_ratios.is_some() && !args.distance_aware {
        // Multi-feature pipeline with coverage ratio
        let (res, _aux_emit) = infer_ibd_with_aux_features(
            &observations,
            &mut params,
            population,
            args.window_size,
            args.baum_welch_iters,
            coverage_ratios.as_deref(),
        );
        res
    } else if args.distance_aware {
        let window_positions: Vec<(u64, u64)> = records.iter()
            .map(|r| (r.start, r.end))
            .collect();

        if let Some(gmap) = genetic_map {
            // Recombination-aware pipeline
            if args.baum_welch_iters > 0 && observations.len() >= 10 {
                params.baum_welch_with_genetic_map(
                    &observations, &window_positions, gmap, args.baum_welch_iters,
                    1e-6, Some(population), args.window_size,
                );
            }
            let mut states = viterbi_with_genetic_map(
                &observations, &params, &window_positions, gmap, args.window_size,
            );
            let (posteriors, log_likelihood) = forward_backward_with_genetic_map(
                &observations, &params, &window_positions, gmap, args.window_size,
            );
            refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
            IbdInferenceResult { states, posteriors, log_likelihood }
        } else {
            // Physical distance-aware pipeline
            if args.baum_welch_iters > 0 && observations.len() >= 10 {
                params.baum_welch_with_distances(
                    &observations, &window_positions, args.baum_welch_iters,
                    1e-6, Some(population), args.window_size,
                );
            }
            let mut states = viterbi_with_distances(&observations, &params, &window_positions);
            let (posteriors, log_likelihood) = forward_backward_with_distances(
                &observations, &params, &window_positions,
            );
            refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
            IbdInferenceResult { states, posteriors, log_likelihood }
        }
    } else {
        infer_ibd_with_training(
            &observations,
            &mut params,
            population,
            args.window_size,
            args.baum_welch_iters,
        )
    };

    // Store window states with posteriors
    for (i, (&state, &posterior)) in result.states.iter().zip(result.posteriors.iter()).enumerate() {
        window_states.push((records[i].start, records[i].end, state, posterior));
    }

    // Extract segments with posterior filtering and LOD scores
    let hmm_segments = extract_ibd_segments_with_lod(
        &result.states,
        &result.posteriors,
        args.min_windows,
        args.posterior_threshold,
        Some((&observations, &params)),
        None,
    );

    for seg in hmm_segments {
        let start_bp = records[seg.start_idx].start;
        let end_bp = records[seg.end_idx].end;
        let length_bp = end_bp.saturating_sub(start_bp);

        if length_bp < args.min_len_bp {
            continue;
        }

        // Always compute mean_identity from raw values (not logit-transformed)
        let mean_identity: f64 = records[seg.start_idx..=seg.end_idx]
            .iter()
            .map(|r| r.identity)
            .sum::<f64>() / seg.n_windows as f64;

        segments.push(IbdSegment {
            chrom: records[seg.start_idx].chrom.clone(),
            start: start_bp,
            end: end_bp,
            hap_a: hap_a.to_string(),
            hap_b: hap_b.to_string(),
            n_windows: seg.n_windows,
            mean_identity,
            mean_posterior: seg.mean_posterior,
            min_posterior: seg.min_posterior,
            max_posterior: seg.max_posterior,
            lod_score: seg.lod_score,
        });
    }

    (segments, window_states)
}

/// A genomic region to exclude from IBD output.
#[derive(Debug, Clone)]
struct ExcludeRegion {
    chrom: String,
    start: u64,
    end: u64,
}

/// Parse a BED file of exclusion regions.
/// Format: tab-separated chrom/start/end (0-based half-open).
fn parse_exclude_regions(path: &PathBuf) -> Result<Vec<ExcludeRegion>> {
    let file = File::open(path).context("Failed to open exclude regions BED file")?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }
        let start: u64 = fields[1].parse().context("Invalid start coordinate")?;
        let end: u64 = fields[2].parse().context("Invalid end coordinate")?;
        regions.push(ExcludeRegion {
            chrom: fields[0].to_string(),
            start,
            end,
        });
    }

    Ok(regions)
}

/// Check if a segment overlaps any exclusion region on the same chromosome.
fn overlaps_excluded(seg_chrom: &str, seg_start: u64, seg_end: u64, regions: &[ExcludeRegion]) -> bool {
    for r in regions {
        if r.chrom == seg_chrom && seg_start < r.end && seg_end > r.start {
            return true;
        }
    }
    false
}

/// Per-window population background statistics.
struct WindowBackgroundStats {
    /// Map from (start, end) → (n_high_identity, n_total_pairs)
    high_identity_counts: HashMap<(u64, u64), (usize, usize)>,
    /// Map from (start, end) → (sum_identity, count) for mean computation
    identity_sums: HashMap<(u64, u64), (f64, usize)>,
}

/// Compute per-window background statistics: for each genomic window position,
/// track identity distribution across all pairs.
fn compute_window_background(
    pair_data: &HashMap<(String, String), Vec<IbsRecord>>,
    identity_threshold: f64,
    identity_floor: f64,
) -> WindowBackgroundStats {
    let mut high_counts: HashMap<(u64, u64), (usize, usize)> = HashMap::new();
    let mut id_sums: HashMap<(u64, u64), (f64, usize)> = HashMap::new();

    for records in pair_data.values() {
        for r in records {
            if identity_floor > 0.0 && r.identity < identity_floor {
                continue;
            }
            let key = (r.start, r.end);
            let hc = high_counts.entry(key).or_insert((0, 0));
            hc.1 += 1;
            if r.identity >= identity_threshold {
                hc.0 += 1;
            }
            let is = id_sums.entry(key).or_insert((0.0, 0));
            is.0 += r.identity;
            is.1 += 1;
        }
    }

    WindowBackgroundStats {
        high_identity_counts: high_counts,
        identity_sums: id_sums,
    }
}

/// Remove windows where the population-level background ratio exceeds the threshold.
#[allow(clippy::ptr_arg)] // retain() requires &mut Vec, not &mut [T]
fn apply_background_filter(
    records: &mut Vec<IbsRecord>,
    background: &WindowBackgroundStats,
    ratio_threshold: f64,
) {
    records.retain(|r| {
        let key = (r.start, r.end);
        match background.high_identity_counts.get(&key) {
            Some(&(n_high, n_total)) if n_total > 0 => {
                let ratio = n_high as f64 / n_total as f64;
                ratio <= ratio_threshold
            }
            _ => true,
        }
    });
}

/// Per-window population background with mean and variance.
struct WindowMeanVar {
    mean: f64,
    std: f64,
}

/// Compute per-window mean and std of identity across all pairs.
fn compute_window_mean_var(
    background: &WindowBackgroundStats,
    pair_data: &HashMap<(String, String), Vec<IbsRecord>>,
    identity_floor: f64,
) -> HashMap<(u64, u64), WindowMeanVar> {
    // First pass: compute mean (already have sum/count)
    let means: HashMap<(u64, u64), f64> = background.identity_sums.iter()
        .filter(|(_, &(_, count))| count > 1)
        .map(|(&key, &(sum, count))| (key, sum / count as f64))
        .collect();

    // Second pass: compute variance
    let mut var_accum: HashMap<(u64, u64), (f64, usize)> = HashMap::new();
    for records in pair_data.values() {
        for r in records {
            if identity_floor > 0.0 && r.identity < identity_floor {
                continue;
            }
            let key = (r.start, r.end);
            if let Some(&mean) = means.get(&key) {
                let entry = var_accum.entry(key).or_insert((0.0, 0));
                entry.0 += (r.identity - mean).powi(2);
                entry.1 += 1;
            }
        }
    }

    var_accum.into_iter().filter_map(|(key, (sum_sq, count))| {
        if count > 1 {
            let mean = *means.get(&key)?;
            let std = (sum_sq / count as f64).sqrt().max(1e-8);
            Some((key, WindowMeanVar { mean, std }))
        } else {
            None
        }
    }).collect()
}

/// Normalize identity values using z-score per window position.
/// z = (identity - pop_mean) / pop_std
/// Then rescale to [0, 1]: normalized = sigmoid(z) = 1/(1+exp(-z))
/// This preserves the ordering but maps population-average identity to 0.5.
fn normalize_identity_by_background(
    records: &mut [IbsRecord],
    window_stats: &HashMap<(u64, u64), WindowMeanVar>,
) {
    for r in records.iter_mut() {
        let key = (r.start, r.end);
        if let Some(stats) = window_stats.get(&key) {
            let z = (r.identity - stats.mean) / stats.std;
            // Sigmoid maps z-score to (0, 1), centered at 0.5
            r.identity = 1.0 / (1.0 + (-z).exp());
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Parse population
    let population = Population::from_str(&args.population)
        .ok_or_else(|| anyhow::anyhow!(
            "Invalid population '{}'. Valid options: AFR, EUR, EAS, CSA, AMR, InterPop, Generic",
            args.population
        ))?;

    eprintln!("Reading IBS data from {:?}", args.input);
    eprintln!("Population: {:?} (pi = {:.5})", population, population.diversity());
    if args.baum_welch_iters > 0 {
        eprintln!("Baum-Welch training: {} iterations", args.baum_welch_iters);
    }
    if args.posterior_threshold > 0.0 {
        eprintln!("Posterior threshold: {:.2}", args.posterior_threshold);
    }
    if args.identity_floor > 0.0 {
        eprintln!("Identity floor: {:.2} (filtering alignment gaps)", args.identity_floor);
    }
    if args.logit_transform {
        eprintln!("Logit transform: enabled (observations mapped via log(x/(1-x)))");
    }
    if args.distance_aware {
        eprintln!("Distance-aware transitions: enabled");
    }
    if args.background_filter {
        eprintln!("Background filter: enabled (threshold={:.3}, ratio={:.2})",
            args.bg_identity_threshold, args.bg_ratio_threshold);
    }
    if args.coverage_feature {
        eprintln!("Coverage ratio feature: enabled (auxiliary emission in HMM)");
    }

    // Load exclusion regions if specified
    let exclude_regions = if let Some(ref bed_path) = args.exclude_regions {
        let regions = parse_exclude_regions(bed_path)?;
        eprintln!("Exclude regions: {} regions loaded from {:?}", regions.len(), bed_path);
        for r in &regions {
            eprintln!("  Excluding: {}-{} ({:.1} Mb)", r.start, r.end,
                (r.end - r.start) as f64 / 1e6);
        }
        Some(regions)
    } else {
        None
    };

    let pair_data = read_ibs_file(&args.input)?;
    eprintln!("Found {} haplotype pairs", pair_data.len());

    // Compute background statistics if background filter or normalization is enabled
    let background = if args.background_filter || args.bg_normalize {
        let bg = compute_window_background(&pair_data, args.bg_identity_threshold, args.identity_floor);
        let n_windows = bg.high_identity_counts.len();
        if args.background_filter {
            let n_masked = bg.high_identity_counts.values().filter(|&&(n_high, n_total)| {
                n_total > 0 && (n_high as f64 / n_total as f64) > args.bg_ratio_threshold
            }).count();
            eprintln!("Background filter: {} window positions, {} masked ({:.1}%)",
                n_windows, n_masked, 100.0 * n_masked as f64 / n_windows.max(1) as f64);
        }
        if args.bg_normalize {
            eprintln!("Background normalization: enabled ({} window positions)", n_windows);
        }
        Some(bg)
    } else {
        None
    };

    // Compute per-window mean/var for z-score normalization
    let window_stats = if args.bg_normalize {
        if let Some(ref bg) = background {
            let stats = compute_window_mean_var(bg, &pair_data, args.identity_floor);
            eprintln!("Window z-score stats: {} windows with mean/std computed", stats.len());
            Some(stats)
        } else {
            None
        }
    } else {
        None
    };

    // Load genetic map if specified, using chromosome from the data
    let genetic_map = if let Some(ref gmap_path) = args.genetic_map {
        // Extract chromosome from first record
        let chrom = pair_data.values().next()
            .and_then(|records| records.first())
            .map(|r| {
                // Extract chromosome name (e.g. "chr20" from "CHM13#0#chr20")
                r.chrom.rsplit('#').next().unwrap_or(&r.chrom).to_string()
            })
            .unwrap_or_else(|| "20".to_string());
        eprintln!("Loading genetic map for chromosome '{}'", chrom);
        match GeneticMap::from_file(gmap_path, &chrom) {
            Ok(gmap) => {
                eprintln!("Loaded genetic map: {} entries", gmap.len());
                Some(gmap)
            }
            Err(e) => {
                eprintln!("WARNING: Failed to load genetic map: {}. Falling back to physical distance.", e);
                None
            }
        }
    } else {
        None
    };

    let mut all_segments = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut all_window_states: Vec<(String, String, Vec<(u64, u64, usize, f64)>)> = Vec::new();

    for ((hap_a, hap_b), mut records) in pair_data {
        // Apply background filter if enabled
        if args.background_filter {
            if let Some(ref bg) = background {
                apply_background_filter(&mut records, bg, args.bg_ratio_threshold);
            }
        }
        // Apply z-score normalization if enabled
        if args.bg_normalize {
            if let Some(ref ws) = window_stats {
                normalize_identity_by_background(&mut records, ws);
            }
        }
        let n_windows = records.len();
        let (segments, window_states) = process_pair(
            &hap_a, &hap_b, records, &args, population, genetic_map.as_ref(),
        );

        eprintln!(
            "  Pair {}-{}: {} windows -> {} IBD segments",
            hap_a, hap_b, n_windows, segments.len()
        );

        all_segments.extend(segments);
        all_window_states.push((hap_a, hap_b, window_states));
    }

    // Apply exclusion region filter if specified
    if let Some(ref regions) = exclude_regions {
        let before = all_segments.len();
        all_segments.retain(|seg| !overlaps_excluded(&seg.chrom, seg.start, seg.end, regions));
        let removed = before - all_segments.len();
        if removed > 0 {
            eprintln!("Exclude regions: removed {} of {} segments ({:.1}%)",
                removed, before, 100.0 * removed as f64 / before.max(1) as f64);
        }
    }

    // Write IBD segments
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);

    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity\tmean_posterior\tmin_posterior\tmax_posterior\tlod_score")?;
    for seg in &all_segments {
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.2}",
            seg.chrom, seg.start, seg.end, seg.hap_a, seg.hap_b,
            seg.n_windows, seg.mean_identity, seg.mean_posterior, seg.min_posterior, seg.max_posterior,
            seg.lod_score
        )?;
    }
    output.flush()?;

    eprintln!("Wrote {} IBD segments to {:?}", all_segments.len(), args.output);

    // Validate against hap-ibd if requested
    if let Some(hapibd_path) = &args.validate_against {
        eprintln!("\n=== Validation against hap-ibd ===");
        let hapibd_segments = hapibd::parse_hapibd_file(hapibd_path)?;
        let hapibd_above_lod = hapibd::hapibd_segments_above_lod(&hapibd_segments, args.validate_min_lod);
        eprintln!("  hap-ibd segments (LOD >= {:.1}): {}", args.validate_min_lod, hapibd_above_lod.len());

        // Group our segments by sample-level pair (not haplotype-level).
        // hap-ibd uses sample IDs (e.g., "HG00280") while our data uses haplotype IDs
        // (e.g., "HG00280#2#CONTIG"). We merge all haplotype-pair segments into
        // sample-level pairs so they can be compared against hap-ibd output.
        let mut our_by_sample_pair: HashMap<(String, String), Vec<(u64, u64)>> = HashMap::new();
        for seg in &all_segments {
            let sa = extract_sample_id(&seg.hap_a);
            let sb = extract_sample_id(&seg.hap_b);
            // Skip if same sample (self-comparison at sample level)
            if sa == sb {
                continue;
            }
            let key = if sa <= sb {
                (sa, sb)
            } else {
                (sb, sa)
            };
            our_by_sample_pair.entry(key).or_default().push((seg.start, seg.end));
        }
        eprintln!("  Our sample-level pairs with IBD: {}", our_by_sample_pair.len());

        // Build haplotype-level segment lists for haplotype-aware concordance
        let our_hap_segs: Vec<(String, String, u64, u64)> = all_segments.iter()
            .map(|seg| (seg.hap_a.clone(), seg.hap_b.clone(), seg.start, seg.end))
            .collect();
        let hapibd_hap_segs: Vec<(String, u8, String, u8, u64, u64)> = hapibd_above_lod.iter()
            .map(|seg| (seg.sample1.clone(), seg.hap1, seg.sample2.clone(), seg.hap2, seg.start, seg.end))
            .collect();

        // Get unique pairs from hap-ibd (above LOD threshold)
        let hapibd_pairs: Vec<(String, String)> = {
            let mut pairs: Vec<(String, String)> = hapibd_above_lod
                .iter()
                .map(|seg| {
                    let (a, b) = if seg.sample1 <= seg.sample2 {
                        (seg.sample1.clone(), seg.sample2.clone())
                    } else {
                        (seg.sample2.clone(), seg.sample1.clone())
                    };
                    (a, b)
                })
                .collect();
            pairs.sort();
            pairs.dedup();
            pairs
        };
        eprintln!("  hap-ibd sample pairs (LOD >= {:.1}): {}", args.validate_min_lod, hapibd_pairs.len());

        // Get region bounds from all data
        let region_start = all_segments.iter().map(|s| s.start).min().unwrap_or(0);
        let region_end = all_segments.iter().map(|s| s.end).max()
            .unwrap_or(0)
            .max(hapibd_above_lod.iter().map(|s| s.end).max().unwrap_or(0));
        let region = (region_start, region_end);
        eprintln!("  Comparison region: {}-{} ({:.1} Mb)", region_start, region_end,
            (region_end - region_start) as f64 / 1e6);

        // Open validation output
        let mut val_out: Box<dyn Write> = if let Some(ref val_path) = args.validate_output {
            Box::new(BufWriter::new(File::create(val_path)?))
        } else {
            Box::new(std::io::stderr())
        };

        writeln!(val_out, "pair\tn_ours\tn_hapibd\tjaccard\tprecision\trecall\tf1\tconcordance\tlength_r\thap_best_jaccard\thap_best_f1\tboundary_mean_start_bp\tboundary_mean_end_bp")?;

        let mut total_jaccard = 0.0;
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;
        let mut total_concordance = 0.0;
        let mut total_hap_best_jaccard = 0.0;
        let mut total_hap_best_f1 = 0.0;
        let mut n_compared = 0u64;
        let mut n_hap_compared = 0u64;
        let mut all_boundary_matches: Vec<concordance::MatchedInterval> = Vec::new();

        // Collect all sample-level pair keys from both datasets
        let all_pair_keys: Vec<(String, String)> = {
            let mut keys: Vec<_> = our_by_sample_pair.keys().cloned().collect();
            for (s1, s2) in &hapibd_pairs {
                let key = if s1 <= s2 { (s1.clone(), s2.clone()) } else { (s2.clone(), s1.clone()) };
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            keys.sort();
            keys
        };

        for (s1, s2) in &all_pair_keys {
            let our_segs = our_by_sample_pair.get(&(s1.clone(), s2.clone()))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let hapibd_for_pair: Vec<(u64, u64)> = hapibd_above_lod.iter()
                .filter(|seg| seg.involves_pair(s1, s2))
                .map(|seg| (seg.start, seg.end))
                .collect();

            if our_segs.is_empty() && hapibd_for_pair.is_empty() {
                continue;
            }

            let jaccard = concordance::segments_jaccard(our_segs, &hapibd_for_pair, region);
            let (precision, recall) = concordance::segments_precision_recall(our_segs, &hapibd_for_pair, region);
            let f1 = concordance::f1_score(precision, recall);
            let conc = concordance::per_window_concordance(our_segs, &hapibd_for_pair, region, args.validate_window_size);

            let matched = concordance::matched_segments(our_segs, &hapibd_for_pair, 0.5);
            let match_pairs: Vec<((u64, u64), (u64, u64))> = matched.iter()
                .map(|&(i, j)| (our_segs[i], hapibd_for_pair[j]))
                .collect();
            let length_r = concordance::length_correlation(&match_pairs);

            // Boundary accuracy for matched segments
            let boundary_acc = concordance::boundary_accuracy(&match_pairs, 10000);
            all_boundary_matches.extend_from_slice(&match_pairs);

            let (boundary_start_str, boundary_end_str) = match &boundary_acc {
                Some(acc) => (format!("{:.0}", acc.mean_start_distance_bp),
                              format!("{:.0}", acc.mean_end_distance_bp)),
                None => ("NA".to_string(), "NA".to_string()),
            };

            // Haplotype-level concordance
            let hap_conc = concordance::haplotype_level_concordance(
                &our_hap_segs, &hapibd_hap_segs, s1, s2, region,
            );
            let (hap_best_j, hap_best_f) = match &hap_conc {
                Some(hc) => {
                    total_hap_best_jaccard += hc.best_jaccard;
                    total_hap_best_f1 += hc.best_f1;
                    n_hap_compared += 1;
                    (format!("{:.4}", hc.best_jaccard), format!("{:.4}", hc.best_f1))
                }
                None => ("NA".to_string(), "NA".to_string()),
            };

            writeln!(val_out, "{}_{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}",
                s1, s2, our_segs.len(), hapibd_for_pair.len(),
                jaccard, precision, recall, f1, conc, length_r,
                hap_best_j, hap_best_f, boundary_start_str, boundary_end_str)?;

            total_jaccard += jaccard;
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            total_concordance += conc;
            n_compared += 1;
        }

        if n_compared > 0 {
            eprintln!("\n  === Sample-level metrics ({} pairs) ===", n_compared);
            eprintln!("  Mean Jaccard:      {:.4}", total_jaccard / n_compared as f64);
            eprintln!("  Mean Precision:    {:.4}", total_precision / n_compared as f64);
            eprintln!("  Mean Recall:       {:.4}", total_recall / n_compared as f64);
            eprintln!("  Mean F1:           {:.4}", total_f1 / n_compared as f64);
            eprintln!("  Mean Concordance:  {:.4}", total_concordance / n_compared as f64);
        } else {
            eprintln!("  No common pairs found for comparison");
        }

        if n_hap_compared > 0 {
            eprintln!("\n  === Haplotype-level metrics ({} pairs) ===", n_hap_compared);
            eprintln!("  Mean best Jaccard: {:.4}", total_hap_best_jaccard / n_hap_compared as f64);
            eprintln!("  Mean best F1:      {:.4}", total_hap_best_f1 / n_hap_compared as f64);
        }

        if !all_boundary_matches.is_empty() {
            if let Some(acc) = concordance::boundary_accuracy(&all_boundary_matches, 10000) {
                eprintln!("\n  === Segment boundary accuracy ({} matched) ===", acc.n_matched);
                eprintln!("  Mean start distance: {:.0} bp ({:.1} kb)", acc.mean_start_distance_bp, acc.mean_start_distance_bp / 1000.0);
                eprintln!("  Mean end distance:   {:.0} bp ({:.1} kb)", acc.mean_end_distance_bp, acc.mean_end_distance_bp / 1000.0);
                eprintln!("  Median start dist:   {:.0} bp ({:.1} kb)", acc.median_start_distance_bp, acc.median_start_distance_bp / 1000.0);
                eprintln!("  Median end dist:     {:.0} bp ({:.1} kb)", acc.median_end_distance_bp, acc.median_end_distance_bp / 1000.0);
                eprintln!("  Max start distance:  {} bp ({:.1} kb)", acc.max_start_distance_bp, acc.max_start_distance_bp as f64 / 1000.0);
                eprintln!("  Max end distance:    {} bp ({:.1} kb)", acc.max_end_distance_bp, acc.max_end_distance_bp as f64 / 1000.0);
                eprintln!("  Starts within 10kb:  {:.1}%", acc.frac_start_within_threshold * 100.0);
                eprintln!("  Ends within 10kb:    {:.1}%", acc.frac_end_within_threshold * 100.0);
            }
        }

        if let Some(ref val_path) = args.validate_output {
            eprintln!("  Validation metrics written to {:?}", val_path);
        }
    }

    // Write per-window states if requested
    if let Some(states_path) = &args.states_output {
        let states_file = File::create(states_path)?;
        let mut states_out = BufWriter::new(states_file);

        writeln!(states_out, "group.a\tgroup.b\tstart\tend\tpredicted_state\tposterior")?;

        for (hap_a, hap_b, window_states) in &all_window_states {
            for (start, end, state, posterior) in window_states {
                writeln!(states_out, "{}\t{}\t{}\t{}\t{}\t{:.4}", hap_a, hap_b, start, end, state, posterior)?;
            }
        }

        states_out.flush()?;
        eprintln!("Wrote per-window states to {:?}", states_path);
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
