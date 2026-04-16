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

fn validate_unit_interval(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("value must be in [0.0, 1.0], got {}", v))
    }
}

fn validate_positive_f64(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(format!("value must be > 0.0, got {}", v))
    }
}
use rayon::prelude::*;

use hprc_common::ColumnIndices;
use hprc_ibd::hmm::{
    infer_ibd, infer_ibd_with_training, infer_ibd_multi_scale,
    infer_ibd_with_aux_features, coverage_ratio, estimate_auxiliary_emissions, augment_with_k0,
    extract_ibd_segments_with_lod,
    extract_ibd_segments_composite, estimate_ibd_emission_std,
    forward_backward_with_distances, viterbi_with_distances,
    forward_backward_with_genetic_map, viterbi_with_genetic_map,
    forward_backward_with_distances_from_log_emit, viterbi_with_distances_from_log_emit,
    forward_backward_with_genetic_map_from_log_emit, viterbi_with_genetic_map_from_log_emit,
    forward_backward_from_log_emit, viterbi_from_log_emit,
    precompute_log_emissions, smooth_log_emissions,
    refine_states_with_posteriors, bridge_ibd_gaps, merge_nearby_ibd_segments,
    refine_states_adaptive, bridge_ibd_gaps_adaptive,
    refine_segment_boundaries,
    GeneticMap, HmmParams, Population,
};
use hprc_ibd::stats::GaussianParams;
use hprc_ibd::{Region, WindowIterator};

type RegionResult = (HashMap<(String, String), Vec<WindowRecord>>, Option<GeneticMap>);

#[derive(Parser, Debug)]
#[command(name = "ibd", version, about = "IBD segment detection using HMM",
    after_help = "\
RECOMMENDED PARAMETER PRESETS:

  Production (pangenome IBD):
    --identity-floor 0.9 --min-len-bp 2000000 --baum-welch-iters 20 --bridge-gaps 2

  High precision (fewer false positives):
    --identity-floor 0.95 --min-len-bp 3000000 --baum-welch-iters 20 --min-lod 5

  High recall (more detections, more false positives):
    --identity-floor 0.9 --min-len-bp 1000000 --min-windows 100 --baum-welch-iters 20
    --bridge-gaps 3 --bridge-min-posterior 0.2

  Key findings from validation:
    - Identity floor 0.9: filters alignment gaps that confuse emission estimation
    - Baum-Welch 20 iters: convergence analysis shows 10 iters leave ~35% residual error
    - Bridge-gaps 2-3: prevents segment splitting from noisy windows within true IBD
    - LOD filtering: use --min-lod 3-5 for high-confidence segments
    - Distance-aware and genetic-map transitions have no effect (emission model is bottleneck)
")]
struct Args {
    #[arg(long = "sequence-files", required_unless_present = "similarity_file")]
    sequence_files: Option<String>,

    #[arg(short = 'a', required_unless_present = "similarity_file")]
    align: Option<String>,

    #[arg(short = 'r', required_unless_present = "similarity_file")]
    ref_name: Option<String>,

    /// Region, e.g. chr1:1-248956422 or chr1
    #[arg(long = "region")]
    region: Option<String>,

    /// BED file with regions to process (chrom, start, end, tab-separated)
    /// Enables batch processing of multiple chromosomes/regions in a single run
    #[arg(long = "bed", conflicts_with = "region")]
    bed: Option<String>,

    /// Window size in bp (must be > 0)
    #[arg(long = "size", default_value = "10000", value_parser = validate_positive_u64)]
    window_size: u64,

    #[arg(long = "subset-sequence-list", required_unless_present = "similarity_file")]
    subset_list: Option<String>,

    /// Pre-computed similarity file (TSV with chrom, start, end, group.a, group.b, estimated.identity).
    /// When provided, skips impg and reads pairwise identities from this file.
    #[arg(long = "similarity-file")]
    similarity_file: Option<String>,

    #[arg(long = "output", required = true)]
    output: String,

    #[arg(long = "ibs-output")]
    ibs_output: Option<String>,

    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Minimum IBD segment length in base pairs (default: 2 Mb for reliable detection)
    #[arg(long = "min-len-bp", default_value = "2000000")]
    min_len_bp: u64,

    /// Minimum number of consecutive windows for IBD segment.
    /// If not specified, automatically computed as min-len-bp / window-size.
    /// With --size 10000 and --min-len-bp 2000000, this defaults to 200.
    #[arg(long = "min-windows")]
    min_windows: Option<usize>,

    #[arg(long = "expected-seg-windows", default_value = "50", value_parser = validate_positive_f64)]
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
    #[arg(long = "posterior-threshold", default_value = "0.0", value_parser = validate_unit_interval)]
    posterior_threshold: f64,

    /// Minimum LOD score for segment to be reported.
    /// LOD = log10(P(data|IBD)/P(data|non-IBD)), summed across segment windows.
    /// Typical thresholds: 3 (suggestive), 5 (strong), 10 (very strong).
    /// When combined with --min-len-bp, segments must pass BOTH filters.
    /// Set to 0 to disable LOD filtering.
    #[arg(long = "min-lod", default_value = "3.0")]
    min_lod: Option<f64>,

    /// Output file for per-window posteriors (optional)
    /// Format: chrom, start, end, group.a, group.b, identity, posterior
    #[arg(long = "output-posteriors")]
    output_posteriors: Option<String>,

    /// Number of Baum-Welch iterations for parameter training (0 to disable)
    /// 20 iterations recommended: convergence analysis shows 10 iterations leave ~35% residual
    /// error while 20-30 iterations achieve >96% convergence
    #[arg(long = "baum-welch-iters", default_value = "20")]
    baum_welch_iters: usize,

    /// Use population-adaptive transition probabilities
    /// Adjusts P(enter IBD) and expected segment length based on population
    #[arg(long = "adaptive-transitions", default_value = "false")]
    adaptive_transitions: bool,

    /// Use distance-aware HMM transitions
    /// Adjusts transition probabilities based on physical distance between windows,
    /// accounting for gaps in pangenome coverage
    #[arg(long = "distance-aware", default_value = "false")]
    distance_aware: bool,

    /// Genetic map file (PLINK format) for recombination-aware transitions.
    /// When provided with --distance-aware, uses genetic distance (cM) instead
    /// of physical distance (bp) for transition probability scaling.
    /// Format: chr pos_bp rate_cM_Mb pos_cM (whitespace-separated)
    #[arg(long = "genetic-map")]
    genetic_map: Option<String>,

    /// Output IBD segments in BED format (compatible with bedtools).
    /// Format: chrom start end name score strand
    /// Name = "hapA_hapB", score = LOD * 100 (capped at 1000)
    #[arg(long = "output-bed")]
    output_bed: Option<String>,

    /// Minimum identity threshold for including a window in HMM inference.
    /// Windows with identity below this threshold are discarded as alignment gaps.
    /// Pangenome data often has windows near 0.0 from unaligned regions; filtering
    /// these prevents alignment gaps from corrupting emission estimation.
    /// Set to 0.0 to disable. Values 0.9-0.95 recommended for pangenome data.
    #[arg(long = "identity-floor", default_value = "0.9", value_parser = validate_unit_interval)]
    identity_floor: f64,

    /// Number of neighboring windows on each side for emission smoothing.
    /// Pools evidence from [t-context, t+context] windows to strengthen weak signals.
    /// With context=2 (5-window span), per-window SNR increases by ~2.2x.
    /// Set to 0 to disable (default). Recommended: 1-2 for IBD.
    #[arg(long = "emission-context", default_value = "0")]
    emission_context: usize,

    /// BED file of regions to mask (exclude from HMM inference).
    /// Windows overlapping masked regions are removed before inference.
    /// Use for segmental duplications, centromeric repeats, etc.
    #[arg(long = "mask-bed")]
    mask_bed: Option<String>,

    /// Maximum gap (in windows) between IBD segments to bridge.
    /// Short non-IBD gaps within true IBD regions are filled when the gap
    /// is ≤ this many windows and posterior evidence supports IBD.
    /// Set to 0 to disable (default). Recommended: 2-3 for pangenome data.
    #[arg(long = "bridge-gaps", default_value = "0")]
    bridge_gaps: usize,

    /// Minimum mean posterior P(IBD) in a gap for it to be bridged.
    /// Only used when --bridge-gaps > 0. Lower values bridge more aggressively.
    #[arg(long = "bridge-min-posterior", default_value = "0.3", value_parser = validate_unit_interval)]
    bridge_min_posterior: f64,

    /// Use composite segment filtering instead of independent hard thresholds.
    /// Composite scoring couples LOD, posterior, and length: shorter segments
    /// can survive if they have high LOD and posterior evidence.
    /// When enabled, --min-windows becomes a soft minimum (penalty, not cutoff).
    #[arg(long = "composite-filter")]
    composite_filter: bool,

    /// Composite score threshold (0.0-1.0). Only used with --composite-filter.
    /// Higher values are more stringent. Recommended: 0.3-0.5.
    #[arg(long = "composite-threshold", default_value = "0.3", value_parser = validate_unit_interval)]
    composite_threshold: f64,

    /// Use adaptive gap bridging that considers flanking segment quality.
    /// Gaps between high-confidence IBD segments are bridged more aggressively.
    #[arg(long = "adaptive-bridge")]
    adaptive_bridge: bool,

    /// Use adaptive state refinement with per-segment thresholds.
    /// High-confidence segments extend more aggressively; weak segments
    /// are trimmed more aggressively. Replaces fixed 0.5/0.2 thresholds.
    #[arg(long = "adaptive-refinement")]
    adaptive_refinement: bool,

    /// Estimate IBD emission variance from data instead of using fixed std=0.0005.
    /// Uses top quantile of observations as putative IBD windows to estimate noise.
    #[arg(long = "adaptive-ibd-emission")]
    adaptive_ibd_emission: bool,

    /// Refine segment boundaries using posterior probability interpolation.
    /// Estimates sub-window boundary positions by finding where P(IBD) crosses 0.5
    /// between adjacent windows. Improves boundary resolution from ~10kb to ~2-5kb.
    #[arg(long = "refine-boundaries")]
    refine_boundaries: bool,

    /// Enable multi-scale IBD inference.
    /// Runs HMM at base resolution AND coarser scales (2x, 4x aggregated windows).
    /// Segments confirmed at multiple scales are kept; fine-only weak segments are
    /// pruned; coarse-only segments with moderate posterior support are recovered.
    /// This improves detection robustness by leveraging long-range patterns.
    #[arg(long = "multi-scale")]
    multi_scale: bool,

    /// Normalize identity values by per-haplotype baselines before HMM inference.
    /// For each window, computes each haplotype's mean identity with all partners,
    /// then adjusts pair identity by removing per-haplotype deviations from the
    /// global mean. This removes systematic biases from assembly quality,
    /// population-specific diversity, and region-specific effects.
    /// The adjusted values preserve the original identity scale, so existing
    /// emission priors and Baum-Welch training work without modification.
    #[arg(long = "contrast-normalize")]
    contrast_normalize: bool,

    /// Run HMM inference in logit space instead of raw identity space.
    /// Transforms identity observations via logit(x) = ln(x/(1-x)) before
    /// emission estimation and HMM inference. In logit space, the separation
    /// between IBD (~0.9997) and non-IBD (~0.999) means increases from ~0.0007
    /// to ~1.2, dramatically improving signal-to-noise ratio. The Gaussian
    /// emission model is also more appropriate in logit space since it removes
    /// the [0,1] boundary effect. States and posteriors are unaffected since
    /// the logit transform is monotonic.
    #[arg(long = "logit-emissions")]
    logit_emissions: bool,

    /// Use coverage ratio as an auxiliary feature for IBD detection.
    /// When enabled, computes min(a_len, b_len) / max(a_len, b_len)
    /// from per-haplotype alignment lengths (group.a.length, group.b.length
    /// columns). IBD regions have symmetric coverage (ratio ~1.0) while
    /// non-IBD regions may have asymmetric alignments (ratio < 1.0).
    /// Requires input data with group.a.length and group.b.length columns
    /// (produced by the `ibs` wrapper around impg similarity).
    #[arg(long = "coverage-feature")]
    coverage_feature: bool,

    /// Use mutation-free window indicator (K=0) as an auxiliary feature for
    /// IBD detection. Windows with identity >= threshold are classified as
    /// mutation-free. These are dramatically more common in IBD (~22%) than
    /// non-IBD (~1.5%), providing strong per-window discriminative signal
    /// that complements the continuous identity value (G104). Bernoulli
    /// emission parameters are estimated from data using initial posteriors.
    #[arg(long = "k0-feature")]
    k0_feature: bool,

    /// Identity threshold for the K=0 (mutation-free) indicator.
    /// Windows with raw identity >= this value are classified as K=0.
    /// Default 0.9999 corresponds to ≤1 mismatch per 10kb.
    #[arg(long = "k0-threshold", default_value_t = 0.9999)]
    k0_threshold: f64,
}

#[derive(Debug, Clone)]
struct WindowRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
    /// Alignment length of group A (0 if column not present)
    a_length: u64,
    /// Alignment length of group B (0 if column not present)
    b_length: u64,
}

fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) }
}

/// Per-haplotype, per-window baseline statistics for contrast normalization.
///
/// For each (haplotype, window_start), stores the sum of identities and count
/// of pairs involving that haplotype at that window. The global mean per window
/// is also tracked. This enables computing:
///
///   adjusted(A,B,w) = identity(A,B,w) - (baseline_A(w) - μ(w)) - (baseline_B(w) - μ(w))
///
/// which removes systematic per-haplotype biases while preserving the identity scale.
#[derive(Debug, Clone)]
struct ContrastBaselines {
    /// (haplotype, window_start) → (sum_identity, count)
    hap_window: HashMap<(String, u64), (f64, usize)>,
    /// window_start → (sum_identity, count)
    global_window: HashMap<u64, (f64, usize)>,
}

impl ContrastBaselines {
    /// Build baselines from all pair data.
    fn from_pair_data(pair_data: &HashMap<(String, String), Vec<WindowRecord>>) -> Self {
        let mut hap_window: HashMap<(String, u64), (f64, usize)> = HashMap::new();
        let mut global_window: HashMap<u64, (f64, usize)> = HashMap::new();

        for ((hap_a, hap_b), records) in pair_data {
            for rec in records {
                // Accumulate for haplotype A
                let entry_a = hap_window
                    .entry((hap_a.clone(), rec.start))
                    .or_insert((0.0, 0));
                entry_a.0 += rec.identity;
                entry_a.1 += 1;

                // Accumulate for haplotype B
                let entry_b = hap_window
                    .entry((hap_b.clone(), rec.start))
                    .or_insert((0.0, 0));
                entry_b.0 += rec.identity;
                entry_b.1 += 1;

                // Accumulate global
                let entry_g = global_window
                    .entry(rec.start)
                    .or_insert((0.0, 0));
                entry_g.0 += rec.identity;
                entry_g.1 += 1;
            }
        }

        ContrastBaselines { hap_window, global_window }
    }

    /// Get the baseline (mean identity) for a haplotype at a window position.
    fn hap_baseline(&self, hap: &str, window_start: u64) -> Option<f64> {
        self.hap_window
            .get(&(hap.to_string(), window_start))
            .map(|(sum, count)| sum / *count as f64)
    }

    /// Get the global mean identity at a window position.
    fn global_mean(&self, window_start: u64) -> Option<f64> {
        self.global_window
            .get(&window_start)
            .map(|(sum, count)| sum / *count as f64)
    }

    /// Adjust identity for a pair at a window, removing per-haplotype bias.
    ///
    /// adjusted = identity - (baseline_A - μ) - (baseline_B - μ)
    ///          = identity - baseline_A - baseline_B + 2μ
    ///
    /// Falls back to raw identity if baselines are unavailable.
    fn adjust(&self, hap_a: &str, hap_b: &str, window_start: u64, identity: f64) -> f64 {
        let mu = match self.global_mean(window_start) {
            Some(m) => m,
            None => return identity,
        };
        let base_a = self.hap_baseline(hap_a, window_start).unwrap_or(mu);
        let base_b = self.hap_baseline(hap_b, window_start).unwrap_or(mu);

        // Remove per-haplotype deviations from global mean
        let adjusted = identity - (base_a - mu) - (base_b - mu);

        // Clamp to [0, 1] since these are identity values
        adjusted.clamp(0.0, 1.0)
    }
}

/// Parse regions from a BED file (chrom, start, end, tab-separated).
fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
    let file = File::open(bed_path)
        .context(format!("Failed to open BED file: {}", bed_path))?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.context("Failed to read BED file")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            bail!("BED line {} has fewer than 3 fields: {}", line_num + 1, line);
        }
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse()
            .context(format!("Invalid start coordinate on BED line {}", line_num + 1))?;
        let end: u64 = fields[2].parse()
            .context(format!("Invalid end coordinate on BED line {}", line_num + 1))?;

        // BED is 0-based half-open; Region is 1-based inclusive
        regions.push(Region { chrom, start: start + 1, end });
    }

    Ok(regions)
}

fn collect_identities(
    args: &Args,
    region: &Region,
    mut ibs_output: Option<&mut BufWriter<File>>,
) -> Result<HashMap<(String, String), Vec<WindowRecord>>> {
    let ref_name = args.ref_name.as_deref()
        .ok_or_else(|| anyhow::anyhow!("ref_name required for impg mode"))?;
    let ref_prefix = format!("{}#", ref_name);
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let mut first_window = true;

    let sequence_files = args.sequence_files.as_deref()
        .ok_or_else(|| anyhow::anyhow!("sequence-files required for impg mode"))?;
    let align = args.align.as_deref()
        .ok_or_else(|| anyhow::anyhow!("alignment required for impg mode"))?;
    let subset_list = args.subset_list.as_deref()
        .ok_or_else(|| anyhow::anyhow!("subset-sequence-list required for impg mode"))?;

    for window in WindowIterator::new(region, args.window_size) {
        let ref_region = format!("{}#0#{}:{}-{}", ref_name, region.chrom, window.start, window.end);
        eprintln!("Collecting identities for {}", ref_region);

        let mut cmd = Command::new("impg");
        cmd.arg("similarity")
            .arg("--sequence-files").arg(sequence_files)
            .arg("-a").arg(align)
            .arg("-r").arg(&ref_region)
            .arg("--subset-sequence-list").arg(subset_list)
            .arg("--force-large-region")
            .stdout(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn impg")?;
        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        let header = lines.next().context("No output")?.context("Failed to read header")?;
        let cols = ColumnIndices::from_header(&header)
            .context("Failed to parse impg header")?;

        // Optional alignment length columns (for coverage ratio feature)
        let header_fields: Vec<&str> = header.split('\t').collect();
        let col_a_length = header_fields.iter().position(|&c| c == "group.a.length");
        let col_b_length = header_fields.iter().position(|&c| c == "group.b.length");

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

            let a_length: u64 = col_a_length
                .and_then(|col| fields.get(col))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let b_length: u64 = col_b_length
                .and_then(|col| fields.get(col))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            if let Some(ref mut out) = ibs_output {
                writeln!(out, "{}\t{}\t{}\t{}\t{}\t{}", chrom, start, end, group_a, group_b, identity)?;
            }

            let key = pair_key(group_a, group_b);
            pair_data.entry(key).or_default().push(WindowRecord { chrom, start, end, identity, a_length, b_length });
        }

        let status = child.wait()?;
        if !status.success() { bail!("impg failed"); }
        first_window = false;
    }

    Ok(pair_data)
}

/// Read pre-computed similarity data from a TSV file.
/// Format: chrom, start, end, group.a, group.b, ..., estimated.identity
/// Extracts haplotype IDs from group.a/group.b (e.g., "SIM_IND_001#1#chr12" → "SIM_IND_001#1")
fn collect_identities_from_file(
    sim_file: &str,
) -> Result<HashMap<(String, String), Vec<WindowRecord>>> {
    let file = File::open(sim_file)
        .context(format!("Failed to open similarity file: {}", sim_file))?;
    let reader = BufReader::new(file);
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let mut lines = reader.lines();

    let header = lines.next()
        .context("Empty similarity file")?
        .context("Failed to read header")?;
    let cols = ColumnIndices::from_header(&header)
        .context("Failed to parse similarity file header")?;

    // Optional alignment length columns (for coverage ratio feature)
    let header_fields: Vec<&str> = header.split('\t').collect();
    let col_a_length = header_fields.iter().position(|&c| c == "group.a.length");
    let col_b_length = header_fields.iter().position(|&c| c == "group.b.length");

    let mut line_count = 0usize;
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

        let group_a_full = fields[cols.group_a];
        let group_b_full = fields[cols.group_b];

        // Extract haplotype ID (first two # components: SAMPLE#HAP)
        let hap_a = extract_hap_id(group_a_full);
        let hap_b = extract_hap_id(group_b_full);

        if hap_a == hap_b { continue; }
        // Skip self-alignments
        if group_a_full == group_b_full { continue; }

        // Use chrom from data (strip ref prefix if present)
        let raw_chrom = fields[cols.chrom];
        let chrom = if let Some(pos) = raw_chrom.rfind('#') {
            // "CHM13#0#chr12" → "chr12"
            let suffix = &raw_chrom[pos+1..];
            if suffix.starts_with("chr") { suffix.to_string() } else { raw_chrom.to_string() }
        } else {
            raw_chrom.to_string()
        };

        let start: u64 = match fields[cols.start].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end: u64 = match fields[cols.end].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let a_length: u64 = col_a_length
            .and_then(|col| fields.get(col))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let b_length: u64 = col_b_length
            .and_then(|col| fields.get(col))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let key = pair_key(&hap_a, &hap_b);
        pair_data.entry(key).or_default().push(WindowRecord { chrom, start, end, identity, a_length, b_length });
        line_count += 1;
    }

    eprintln!("Read {} identity records from {}", line_count, sim_file);
    eprintln!("  {} unique haplotype pairs", pair_data.len());

    Ok(pair_data)
}

/// Extract haplotype ID from a full contig name.
/// "SIM_IND_001#1#chr12" → "SIM_IND_001#1"
/// "HG00097#1#CM099663.1:1234-5678" → "HG00097#1"
fn extract_hap_id(full_name: &str) -> String {
    let parts: Vec<&str> = full_name.splitn(3, '#').collect();
    if parts.len() >= 2 {
        format!("{}#{}", parts[0], parts[1])
    } else {
        full_name.to_string()
    }
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
    lod_score: f64,
    /// Fraction of windows in this segment with identity >= K=0 threshold.
    /// Provides an independent coalescent age proxy: >50% → very recent IBD,
    /// <5% → old IBD. Only set when --k0-feature is active.
    k0_fraction: Option<f64>,
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
#[allow(clippy::too_many_arguments)]
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
    baum_welch_iters: usize,
    adaptive_transitions: bool,
    distance_aware: bool,
    genetic_map: Option<&GeneticMap>,
    identity_floor: f64,
    emission_context: usize,
    mask_regions: &[(String, u64, u64)],
    bridge_gaps: usize,
    bridge_min_posterior: f64,
    min_lod: Option<f64>,
    composite_filter: bool,
    composite_threshold: f64,
    adaptive_bridge: bool,
    adaptive_refinement: bool,
    adaptive_ibd_emission: bool,
    refine_boundaries: bool,
    multi_scale: bool,
    baselines: Option<&ContrastBaselines>,
    logit_emissions: bool,
    coverage_feature: bool,
    k0_feature: bool,
    k0_threshold: f64,
) -> PairResult {
    // Filter out alignment-gap windows (identity below floor)
    if identity_floor > 0.0 {
        records.retain(|r| r.identity >= identity_floor);
    }

    // Filter out windows overlapping masked regions (seg-dups, centromeres, etc.)
    if !mask_regions.is_empty() {
        records.retain(|r| {
            !mask_regions.iter().any(|(chrom, mstart, mend)| {
                r.chrom == *chrom && r.start < *mend && r.end > *mstart
            })
        });
    }

    // Capture pre-normalization identities for K=0 feature before contrast
    // normalization shifts values. K=0 should reflect the original mutation
    // count, not the bias-adjusted value (T87 Theorem 4: additive bias
    // removal before nonlinear features; K=0 is inherently nonlinear).
    let pre_norm_k0: Option<Vec<(u64, f64)>> = if k0_feature {
        Some(records.iter().map(|r| (r.start, r.identity)).collect())
    } else {
        None
    };

    // Apply per-haplotype contrast normalization if baselines provided.
    // This removes systematic biases from assembly quality and population diversity,
    // improving HMM emission signal-to-noise ratio.
    if let Some(bl) = baselines {
        for rec in &mut records {
            rec.identity = bl.adjust(&hap_a, &hap_b, rec.start, rec.identity);
        }
    }

    if records.len() < 3 {
        return PairResult {
            segments: Vec::new(),
            posteriors: Vec::new(),
        };
    }

    records.sort_by_key(|r| r.start);
    let raw_observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

    // Compute K=0 (mutation-free) indicators from pre-normalization identities.
    // Uses a HashMap lookup to handle the sort order change.
    let k0_indicators: Option<Vec<f64>> = if let Some(ref pre) = pre_norm_k0 {
        let pre_map: std::collections::HashMap<u64, f64> = pre.iter().cloned().collect();
        let inds: Vec<f64> = records.iter().map(|r| {
            let orig_id = pre_map.get(&r.start).copied().unwrap_or(r.identity);
            if orig_id >= k0_threshold { 1.0 } else { 0.0 }
        }).collect();
        // Only use if there's at least one K=0 window (otherwise no signal)
        if inds.iter().any(|&v| v > 0.5) { Some(inds) } else { None }
    } else {
        None
    };

    // Compute coverage ratios as auxiliary feature when requested
    let coverage_ratios: Option<Vec<f64>> = if coverage_feature {
        let ratios: Vec<f64> = records.iter().map(|r| coverage_ratio(r.a_length, r.b_length)).collect();
        let has_data = ratios.iter().any(|&r| r > 0.0 && r < 1.0);
        if has_data { Some(ratios) } else { None }
    } else {
        None
    };

    // Optionally transform to logit space for better IBD/non-IBD separation.
    // In logit space, the ~0.0007 raw separation becomes ~1.2, dramatically
    // improving discrimination for the Gaussian emission model.
    let observations: Vec<f64> = if logit_emissions {
        hprc_ibd::stats::logit_transform_observations(&raw_observations)
    } else {
        raw_observations.clone()
    };

    // Use population-specific parameters for biologically correct HMM
    let mut params = if logit_emissions {
        // Logit-space params: means and stds in logit coordinates
        HmmParams::from_population_logit(population, expected_seg_windows, p_enter_ibd, window_size)
    } else if adaptive_transitions {
        HmmParams::from_population_adaptive(population, expected_seg_windows, p_enter_ibd, window_size)
    } else {
        HmmParams::from_population(population, expected_seg_windows, p_enter_ibd, window_size)
    };

    // Estimate emissions from data
    if logit_emissions {
        params.estimate_emissions_logit(&observations, Some(population), window_size);
    } else {
        params.estimate_emissions_robust(&observations, Some(population), window_size);
    }

    // Optionally estimate IBD emission variance from data (raw space only)
    if adaptive_ibd_emission && !logit_emissions {
        if let Some(estimated_std) = estimate_ibd_emission_std(
            &observations, 0.05, 0.0003, 0.002,
        ) {
            params.emission[1] = GaussianParams::new_unchecked(
                params.emission[1].mean,
                estimated_std,
            );
        }
    }

    // Run inference: choose between distance-aware and standard pipeline.
    // When logit_emissions is active, use baum_welch_logit (logit-space bounds)
    // instead of standard baum_welch (raw-space bounds 0.999-1.0).
    let bw_ok = !logit_emissions; // Standard BW only valid in raw space

    // Logit-space BW training: runs before the inference dispatch since it
    // applies to all pipeline variants when logit_emissions + BW are both active.
    if logit_emissions && baum_welch_iters > 0 && observations.len() >= 10 {
        params.baum_welch_logit(
            &observations, baum_welch_iters, 1e-6,
            Some(population), window_size,
        );
    }
    let inference = if k0_indicators.is_some() {
        // K=0 auxiliary feature active: unified precomputed-emission path.
        // Handles all feature combinations (coverage, emission_context,
        // distance_aware, genetic_map) via precomputed log-emissions with
        // K=0 Bernoulli augmentation added AFTER smoothing.

        // Standard (non-logit) BW training when applicable
        let window_positions: Vec<(u64, u64)> = if distance_aware {
            records.iter().map(|r| (r.start, r.end)).collect()
        } else {
            Vec::new()
        };
        if bw_ok && baum_welch_iters > 0 && observations.len() >= 10 {
            if distance_aware {
                if let Some(gmap) = genetic_map {
                    params.baum_welch_with_genetic_map(
                        &observations, &window_positions, gmap, baum_welch_iters,
                        1e-6, Some(population), window_size,
                    );
                } else {
                    params.baum_welch_with_distances(
                        &observations, &window_positions, baum_welch_iters,
                        1e-6, Some(population), window_size,
                    );
                }
            } else {
                params.baum_welch(
                    &observations, baum_welch_iters, 1e-6,
                    Some(population), window_size,
                );
            }
        }

        // Build combined log-emissions
        let mut log_emit = precompute_log_emissions(&observations, &params);

        // Add coverage ratio emissions if present
        if let Some(ref cov_ratios) = coverage_ratios {
            let initial_post = if distance_aware {
                if let Some(gmap) = genetic_map {
                    forward_backward_with_genetic_map_from_log_emit(
                        &log_emit, &params, &window_positions, gmap, window_size,
                    ).0
                } else {
                    forward_backward_with_distances_from_log_emit(
                        &log_emit, &params, &window_positions,
                    ).0
                }
            } else {
                forward_backward_from_log_emit(&log_emit, &params).0
            };
            let aux_emit = estimate_auxiliary_emissions(cov_ratios, &initial_post);
            for (i, &aux_val) in cov_ratios.iter().enumerate() {
                log_emit[i][0] += aux_emit[0].log_pdf(aux_val);
                log_emit[i][1] += aux_emit[1].log_pdf(aux_val);
            }
        }

        // Apply emission context smoothing (smooths identity + coverage)
        let mut log_emit = if emission_context > 0 {
            smooth_log_emissions(&log_emit, emission_context)
        } else {
            log_emit
        };

        // K=0 augmentation: AFTER smoothing (discrete, should not be smoothed).
        {
            let k0_inds = k0_indicators.as_deref().unwrap();
            let initial_post = if distance_aware {
                if let Some(gmap) = genetic_map {
                    forward_backward_with_genetic_map_from_log_emit(
                        &log_emit, &params, &window_positions, gmap, window_size,
                    ).0
                } else {
                    forward_backward_with_distances_from_log_emit(
                        &log_emit, &params, &window_positions,
                    ).0
                }
            } else {
                forward_backward_from_log_emit(&log_emit, &params).0
            };
            augment_with_k0(&mut log_emit, k0_inds, &initial_post);
        }

        // Dispatch to appropriate inference variant
        if distance_aware {
            if let Some(gmap) = genetic_map {
                let mut states = viterbi_with_genetic_map_from_log_emit(
                    &log_emit, &params, &window_positions, gmap, window_size,
                );
                let (posteriors, log_likelihood) = forward_backward_with_genetic_map_from_log_emit(
                    &log_emit, &params, &window_positions, gmap, window_size,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            } else {
                let mut states = viterbi_with_distances_from_log_emit(
                    &log_emit, &params, &window_positions,
                );
                let (posteriors, log_likelihood) = forward_backward_with_distances_from_log_emit(
                    &log_emit, &params, &window_positions,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            }
        } else {
            let mut states = viterbi_from_log_emit(&log_emit, &params);
            let (posteriors, log_likelihood) = forward_backward_from_log_emit(&log_emit, &params);
            refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
            hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
        }
    } else if coverage_ratios.is_some() && !distance_aware {
        // Multi-feature pipeline with coverage ratio auxiliary feature
        let (res, _aux_emit) = infer_ibd_with_aux_features(
            &observations,
            &mut params,
            population,
            window_size,
            if bw_ok { baum_welch_iters } else { 0 },
            coverage_ratios.as_deref(),
        );
        res
    } else if distance_aware {
        // Distance-aware: use per-window positions for transition scaling
        let window_positions: Vec<(u64, u64)> = records.iter()
            .map(|r| (r.start, r.end))
            .collect();

        // Estimate auxiliary emissions from coverage ratios if available.
        // Uses an initial distance-aware FB pass on primary emissions to get
        // posteriors, then fits per-state Gaussians on coverage ratios.
        let aux_emission: Option<[GaussianParams; 2]> = if let Some(ref cov_ratios) = coverage_ratios {
            let primary_emit = precompute_log_emissions(&observations, &params);
            let initial_posteriors = if let Some(gmap) = genetic_map {
                let (p, _) = forward_backward_with_genetic_map_from_log_emit(
                    &primary_emit, &params, &window_positions, gmap, window_size,
                );
                p
            } else {
                let (p, _) = forward_backward_with_distances_from_log_emit(
                    &primary_emit, &params, &window_positions,
                );
                p
            };
            Some(estimate_auxiliary_emissions(cov_ratios, &initial_posteriors))
        } else {
            None
        };

        if let Some(gmap) = genetic_map {
            // Recombination-aware pipeline using genetic map
            if bw_ok && baum_welch_iters > 0 && observations.len() >= 10 {
                params.baum_welch_with_genetic_map(
                    &observations, &window_positions, gmap, baum_welch_iters,
                    1e-6, Some(population), window_size,
                );
            }
            // Use pre-computed log emissions when coverage or smoothing is needed
            let use_log_emit = emission_context > 0 || coverage_ratios.is_some();
            if use_log_emit {
                let mut raw_emit = precompute_log_emissions(&observations, &params);
                // Add auxiliary coverage emissions
                if let (Some(ref cov_ratios), Some(ref aux_emit)) = (&coverage_ratios, &aux_emission) {
                    for (i, &aux_val) in cov_ratios.iter().enumerate() {
                        raw_emit[i][0] += aux_emit[0].log_pdf(aux_val);
                        raw_emit[i][1] += aux_emit[1].log_pdf(aux_val);
                    }
                }
                let log_emit = if emission_context > 0 {
                    smooth_log_emissions(&raw_emit, emission_context)
                } else {
                    raw_emit
                };
                let mut states = viterbi_with_genetic_map_from_log_emit(
                    &log_emit, &params, &window_positions, gmap, window_size,
                );
                let (posteriors, log_likelihood) = forward_backward_with_genetic_map_from_log_emit(
                    &log_emit, &params, &window_positions, gmap, window_size,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            } else {
                let mut states = viterbi_with_genetic_map(
                    &observations, &params, &window_positions, gmap, window_size,
                );
                let (posteriors, log_likelihood) = forward_backward_with_genetic_map(
                    &observations, &params, &window_positions, gmap, window_size,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            }
        } else {
            // Physical distance-aware pipeline (no genetic map)
            if bw_ok && baum_welch_iters > 0 && observations.len() >= 10 {
                params.baum_welch_with_distances(
                    &observations, &window_positions, baum_welch_iters,
                    1e-6, Some(population), window_size,
                );
            }
            let use_log_emit = emission_context > 0 || coverage_ratios.is_some();
            if use_log_emit {
                let mut raw_emit = precompute_log_emissions(&observations, &params);
                if let (Some(ref cov_ratios), Some(ref aux_emit)) = (&coverage_ratios, &aux_emission) {
                    for (i, &aux_val) in cov_ratios.iter().enumerate() {
                        raw_emit[i][0] += aux_emit[0].log_pdf(aux_val);
                        raw_emit[i][1] += aux_emit[1].log_pdf(aux_val);
                    }
                }
                let log_emit = if emission_context > 0 {
                    smooth_log_emissions(&raw_emit, emission_context)
                } else {
                    raw_emit
                };
                let mut states = viterbi_with_distances_from_log_emit(
                    &log_emit, &params, &window_positions,
                );
                let (posteriors, log_likelihood) = forward_backward_with_distances_from_log_emit(
                    &log_emit, &params, &window_positions,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            } else {
                let mut states = viterbi_with_distances(&observations, &params, &window_positions);
                let (posteriors, log_likelihood) = forward_backward_with_distances(
                    &observations, &params, &window_positions,
                );
                refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
                hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
            }
        }
    } else if emission_context > 0 {
        // Standard pipeline with emission smoothing
        if bw_ok && baum_welch_iters > 0 && observations.len() >= 10 {
            params.baum_welch(
                &observations, baum_welch_iters, 1e-6,
                Some(population), window_size,
            );
        }
        let raw_emit = precompute_log_emissions(&observations, &params);
        let log_emit = smooth_log_emissions(&raw_emit, emission_context);
        let mut states = viterbi_from_log_emit(&log_emit, &params);
        let (posteriors, log_likelihood) = forward_backward_from_log_emit(&log_emit, &params);
        refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
        hprc_ibd::hmm::IbdInferenceResult { states, posteriors, log_likelihood }
    } else if bw_ok && baum_welch_iters > 0 {
        if multi_scale {
            // BW training at base resolution, then multi-scale inference
            if observations.len() >= 10 {
                params.baum_welch(
                    &observations, baum_welch_iters, 1e-6,
                    Some(population), window_size,
                );
            }
            infer_ibd_multi_scale(&observations, &params, &[1, 2, 4])
        } else {
            infer_ibd_with_training(&observations, &mut params, population, window_size, baum_welch_iters)
        }
    } else if multi_scale {
        infer_ibd_multi_scale(&observations, &params, &[1, 2, 4])
    } else {
        infer_ibd(&observations, &params)
    };

    // Destructure inference result
    let hprc_ibd::hmm::IbdInferenceResult {
        mut states,
        posteriors: inf_posteriors,
        log_likelihood: _,
    } = inference;

    // Apply adaptive refinement if requested (overrides fixed 0.5/0.2 thresholds)
    if adaptive_refinement {
        refine_states_adaptive(&mut states, &inf_posteriors);
    }

    // Bridge short non-IBD gaps between IBD segments using posterior evidence
    if bridge_gaps > 0 {
        if adaptive_bridge {
            bridge_ibd_gaps_adaptive(&mut states, &inf_posteriors, bridge_gaps, bridge_min_posterior);
        } else {
            bridge_ibd_gaps(&mut states, &inf_posteriors, bridge_gaps, bridge_min_posterior);
        }

        // Post-bridge re-refinement: re-apply state refinement after gap bridging
        // to ensure extended segments have consistent boundaries
        if adaptive_refinement {
            refine_states_adaptive(&mut states, &inf_posteriors);
        } else {
            refine_states_with_posteriors(&mut states, &inf_posteriors, 0.5, 0.2);
        }
    }

    // Extract segments: composite or traditional filtering
    let hmm_segments = if composite_filter {
        // Composite scoring: couples LOD, posterior, and length
        // hard_min_windows = 10% of soft min, absolute floor
        let hard_min = (min_windows / 10).max(5);
        extract_ibd_segments_composite(
            &states,
            &inf_posteriors,
            Some((&observations, &params)),
            min_windows,
            hard_min,
            composite_threshold,
        )
    } else {
        extract_ibd_segments_with_lod(
            &states,
            &inf_posteriors,
            min_windows,
            posterior_threshold,
            Some((&observations, &params)),
            min_lod,
        )
    };

    // Merge nearby segments that survived extraction but have small gaps
    let hmm_segments = if bridge_gaps > 0 {
        merge_nearby_ibd_segments(&hmm_segments, bridge_gaps)
    } else {
        hmm_segments
    };

    // Optionally refine segment boundaries using posterior interpolation
    let refined_boundaries = if refine_boundaries {
        let window_starts: Vec<u64> = records.iter().map(|r| r.start).collect();
        let window_ends: Vec<u64> = records.iter().map(|r| r.end).collect();
        Some(refine_segment_boundaries(
            &hmm_segments,
            &inf_posteriors,
            &window_starts,
            &window_ends,
            0.5,
        ))
    } else {
        None
    };

    let mut segments = Vec::new();
    for (i, seg) in hmm_segments.iter().enumerate() {
        let (start_bp, end_bp) = if let Some(ref refined) = refined_boundaries {
            (refined[i].start_bp, refined[i].end_bp)
        } else {
            (records[seg.start_idx].start, records[seg.end_idx].end)
        };
        let length_bp = end_bp.saturating_sub(start_bp);

        if length_bp < min_len_bp {
            continue;
        }

        // Always report mean identity in raw space (not logit-transformed)
        let mean_identity: f64 =
            raw_observations[seg.start_idx..=seg.end_idx].iter().sum::<f64>() / seg.n_windows as f64;

        // Compute K=0 fraction within segment (coalescent age diagnostic)
        let k0_frac = k0_indicators.as_ref().map(|inds| {
            let seg_inds = &inds[seg.start_idx..=seg.end_idx];
            let k0_count = seg_inds.iter().filter(|&&v| v > 0.5).count();
            k0_count as f64 / seg.n_windows as f64
        });

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
            lod_score: seg.lod_score,
            k0_fraction: k0_frac,
        });
    }

    // Collect per-window posteriors if requested (always report raw identity)
    let posteriors = if collect_posteriors {
        records
            .iter()
            .zip(inf_posteriors.iter())
            .zip(raw_observations.iter())
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

/// Parse mask BED file into (chrom, start, end) tuples.
/// BED is 0-based half-open; we keep it in 1-based inclusive for comparison with WindowRecords.
fn parse_mask_bed(path: &str) -> Result<Vec<(String, u64, u64)>> {
    let file = File::open(path)
        .context(format!("Failed to open mask BED file: {}", path))?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for line_result in reader.lines() {
        let line = line_result.context("Failed to read mask BED file")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse().unwrap_or(0);
        let end: u64 = fields[2].parse().unwrap_or(0);
        // BED 0-based half-open → 1-based start for comparison
        regions.push((chrom, start + 1, end));
    }

    Ok(regions)
}

/// Process all haplotype pairs in parallel and return IBD segments
fn call_ibd_segments(
    pair_data: HashMap<(String, String), Vec<WindowRecord>>,
    args: &Args,
    population: Population,
    collect_posteriors: bool,
    genetic_map: Option<&GeneticMap>,
    mask_regions: &[(String, u64, u64)],
    min_windows: usize,
) -> AllPairsResult {
    eprintln!("Running HMM on {} pairs in parallel with population {:?}...", pair_data.len(), population);
    if args.posterior_threshold > 0.0 {
        eprintln!("Filtering segments with mean P(IBD) >= {:.2}", args.posterior_threshold);
    }
    if genetic_map.is_some() {
        eprintln!("Using recombination-aware transitions from genetic map");
    }

    // Compute per-haplotype contrast baselines if requested
    let baselines = if args.contrast_normalize {
        let bl = ContrastBaselines::from_pair_data(&pair_data);
        let n_haps = bl.hap_window.keys()
            .map(|(h, _)| h.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let n_windows = bl.global_window.len();
        eprintln!("Contrast normalization: {} haplotypes × {} window positions", n_haps, n_windows);
        Some(bl)
    } else {
        None
    };

    // Convert HashMap to Vec for parallel iteration, sorted for deterministic output
    let mut pairs: Vec<_> = pair_data.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

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
                min_windows,
                args.min_len_bp,
                args.posterior_threshold,
                population,
                args.window_size,
                collect_posteriors,
                args.baum_welch_iters,
                args.adaptive_transitions,
                args.distance_aware,
                genetic_map,
                args.identity_floor,
                args.emission_context,
                mask_regions,
                args.bridge_gaps,
                args.bridge_min_posterior,
                args.min_lod,
                args.composite_filter,
                args.composite_threshold,
                args.adaptive_bridge,
                args.adaptive_refinement,
                args.adaptive_ibd_emission,
                args.refine_boundaries,
                args.multi_scale,
                baselines.as_ref(),
                args.logit_emissions,
                args.coverage_feature,
                args.k0_feature,
                args.k0_threshold,
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
    let use_similarity_file = args.similarity_file.is_some();
    if !use_similarity_file {
        let seq_files = args.sequence_files.as_deref().unwrap();
        let align = args.align.as_deref().unwrap();
        let subset = args.subset_list.as_deref().unwrap();
        if !Path::new(seq_files).exists() {
            bail!("sequence-files does not exist: {}", seq_files);
        }
        if !Path::new(align).exists() {
            bail!("alignment file does not exist: {}", align);
        }
        if !Path::new(subset).exists() {
            bail!("subset-sequence-list does not exist: {}", subset);
        }
    } else {
        let sim_file = args.similarity_file.as_deref().unwrap();
        if !Path::new(sim_file).exists() {
            bail!("similarity-file does not exist: {}", sim_file);
        }
    }

    // Warn about unusual parameter choices
    if args.baum_welch_iters > 100 {
        eprintln!("Warning: --baum-welch-iters={} is unusually high (recommended: 10-30). \
                   Baum-Welch typically converges within 20-30 iterations.", args.baum_welch_iters);
    }
    if args.identity_floor > 0.999 {
        eprintln!("Warning: --identity-floor={:.4} is very high and may discard most windows. \
                   Recommended: 0.9 for pangenome data.", args.identity_floor);
    }
    if args.min_len_bp > 0 && args.min_len_bp < args.window_size {
        eprintln!("Warning: --min-len-bp={} is less than window size ({}). \
                   Segments shorter than one window cannot be detected.", args.min_len_bp, args.window_size);
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

    if !use_similarity_file && Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }

    // Collect regions: either from --region or --bed (not needed for similarity-file mode)
    let regions = if use_similarity_file {
        Vec::new() // regions not needed when reading from file
    } else if let Some(ref bed_path) = args.bed {
        if !Path::new(bed_path).exists() {
            bail!("BED file does not exist: {}", bed_path);
        }
        parse_bed_regions(bed_path)?
    } else if let Some(ref region_str) = args.region {
        vec![Region::parse(region_str, args.region_length)?]
    } else {
        bail!("Either --region, --bed, or --similarity-file must be specified");
    };

    eprintln!("Processing {} region(s) with population {:?} (π = {:.5})",
        regions.len(), population, population.diversity());
    if args.identity_floor > 0.0 {
        eprintln!("Identity floor: {:.2} (filtering alignment gaps)", args.identity_floor);
    }

    // Open output files
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);
    if args.k0_feature {
        writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity\tmean_posterior\tmin_posterior\tmax_posterior\tlod_score\tk0_fraction")?;
    } else {
        writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity\tmean_posterior\tmin_posterior\tmax_posterior\tlod_score")?;
    }

    let mut posteriors_out = match &args.output_posteriors {
        Some(path) => {
            let f = File::create(path)?;
            let mut out = BufWriter::new(f);
            writeln!(out, "chrom\tstart\tend\tgroup.a\tgroup.b\tidentity\tposterior")?;
            Some(out)
        }
        None => None,
    };

    let mut bed_out = match &args.output_bed {
        Some(path) => {
            let f = File::create(path)?;
            Some(BufWriter::new(f))
        }
        None => None,
    };

    let collect_posteriors = args.output_posteriors.is_some();
    let mut total_segments = 0usize;

    // Validate genetic map flag
    if args.genetic_map.is_some() && !args.distance_aware {
        eprintln!("Warning: --genetic-map requires --distance-aware to have effect; enabling distance-aware mode");
    }

    // Parse mask regions
    let mask_regions = if let Some(ref mask_path) = args.mask_bed {
        let regions = parse_mask_bed(mask_path)?;
        eprintln!("Loaded {} mask regions from {}", regions.len(), mask_path);
        regions
    } else {
        Vec::new()
    };

    // Compute effective min_windows from min_len_bp / window_size if not explicitly set
    let effective_min_windows = args.min_windows.unwrap_or_else(|| {
        let computed = (args.min_len_bp / args.window_size).max(1) as usize;
        eprintln!("Auto min-windows: {} (from min-len-bp={} / window-size={})",
            computed, args.min_len_bp, args.window_size);
        computed
    });

    if args.emission_context > 0 {
        eprintln!("Emission smoothing: context={} ({}-window span)",
            args.emission_context, 2 * args.emission_context + 1);
    }

    // Collect pair data: either from similarity file or impg
    let region_results: Vec<RegionResult> = if use_similarity_file {
        let sim_file = args.similarity_file.as_deref().unwrap();
        eprintln!("Reading pre-computed similarities from {}", sim_file);
        let pair_data = collect_identities_from_file(sim_file)?;
        vec![(pair_data, None)]
    } else {
        let mut results = Vec::new();
        for (i, region) in regions.iter().enumerate() {
            eprintln!("[{}/{}] Processing {}:{}-{}",
                i + 1, regions.len(), region.chrom, region.start, region.end);

            let mut ibs_output = match &args.ibs_output {
                Some(path) => {
                    let file = if i == 0 {
                        File::create(path)?
                    } else {
                        std::fs::OpenOptions::new().append(true).open(path)?
                    };
                    Some(BufWriter::new(file))
                }
                None => None,
            };

            let pair_data = collect_identities(&args, region, ibs_output.as_mut())?;
            if let Some(ref mut out) = ibs_output {
                out.flush()?;
            }

            let genetic_map = if let Some(ref gmap_path) = args.genetic_map {
                match GeneticMap::from_file(gmap_path, &region.chrom) {
                    Ok(gmap) => {
                        eprintln!("  Loaded genetic map: {} entries for {}", gmap.len(), region.chrom);
                        Some(gmap)
                    }
                    Err(e) => {
                        eprintln!("  Warning: could not load genetic map for {}: {}", region.chrom, e);
                        None
                    }
                }
            } else {
                None
            };

            results.push((pair_data, genetic_map));
        }
        results
    };

    // Process each batch of pair data
    for (pair_data, genetic_map) in &region_results {
        eprintln!("  Processing {} pairs", pair_data.len());

        let result = call_ibd_segments(pair_data.clone(), &args, population, collect_posteriors, genetic_map.as_ref(), &mask_regions, effective_min_windows);

        for seg in &result.segments {
            if let Some(k0f) = seg.k0_fraction {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}",
                    seg.chrom, seg.start_bp, seg.end_bp, seg.hap_a, seg.hap_b,
                    seg.n_windows, seg.mean_identity, seg.mean_posterior, seg.min_posterior, seg.max_posterior,
                    seg.lod_score, k0f
                )?;
            } else {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{:.2}",
                    seg.chrom, seg.start_bp, seg.end_bp, seg.hap_a, seg.hap_b,
                    seg.n_windows, seg.mean_identity, seg.mean_posterior, seg.min_posterior, seg.max_posterior,
                    seg.lod_score
                )?;
            }
        }
        total_segments += result.segments.len();

        if let Some(ref mut bout) = bed_out {
            for seg in &result.segments {
                // BED format: 0-based half-open coordinates
                // Score: LOD * 100 capped at 1000 (BED score range 0-1000)
                let bed_start = seg.start_bp.saturating_sub(1); // convert 1-based to 0-based
                let score = ((seg.lod_score * 100.0).round() as u64).min(1000);
                let name = format!("{}_{}", seg.hap_a, seg.hap_b);
                writeln!(
                    bout,
                    "{}\t{}\t{}\t{}\t{}\t.",
                    seg.chrom, bed_start, seg.end_bp, name, score
                )?;
            }
        }

        if let Some(ref mut pout) = posteriors_out {
            for wp in &result.posteriors {
                writeln!(
                    pout,
                    "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.4}",
                    wp.chrom, wp.start, wp.end, wp.hap_a, wp.hap_b, wp.identity, wp.posterior
                )?;
            }
        }

        eprintln!("  {} segments found in this region", result.segments.len());
    }

    output.flush()?;
    eprintln!("IBD complete: {} total segments written to {}", total_segments, args.output);

    if let Some(ref mut bout) = bed_out {
        bout.flush()?;
        eprintln!("BED output written to {}", args.output_bed.as_ref().unwrap());
    }

    if let Some(ref mut pout) = posteriors_out {
        pout.flush()?;
        eprintln!("Per-window posteriors written to {}", args.output_posteriors.as_ref().unwrap());
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
