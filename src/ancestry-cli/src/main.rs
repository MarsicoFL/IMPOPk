//! Local Ancestry Inference CLI
//!
//! Infers local ancestry from pangenome similarity data using an HMM.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;

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

fn validate_non_negative_f64(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("value must be >= 0.0, got {}", v))
    }
}

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation, EmissionModel,
    AncestryGeneticMap, DecodingMethod, LearnedParams,
    extract_ancestry_segments, forward_backward,
    forward_backward_with_genetic_map, viterbi_with_genetic_map,
    posterior_decode_with_genetic_map,
    precompute_log_emissions, smooth_log_emissions_weighted, smooth_observations,
    apply_haplotype_consistency,
    contrast_normalize_emissions, dampen_low_confidence_emissions,
    viterbi_from_log_emissions, forward_backward_from_log_emissions,
    viterbi_from_log_emissions_with_genetic_map,
    forward_backward_from_log_emissions_with_genetic_map,
    parse_similarity_data, parse_similarity_data_column, parse_similarity_data_with_coverage,
    viterbi, posterior_decode, mpel_decode_from_posteriors,
    estimate_temperature_with_spread, estimate_temperature_normalized, estimate_switch_prob,
    scale_temperature_for_panel, scale_temperature_for_populations, estimate_emission_context,
    auto_configure_pairwise_params, deconvolve_admixed_populations,
    estimate_proportions_from_states, estimate_per_state_switch_rates,
    learn_population_profiles, compute_profile_log_emissions, blend_log_emissions,
    blend_log_emissions_adaptive, blend_log_emissions_agreement, blend_log_emissions_hybrid,
    PerPopAgreementScales, compute_per_pop_agreement_scales,
    blend_log_emissions_per_pop_agreement, blend_log_emissions_per_pop_hybrid,
    apply_posterior_feedback, apply_focused_masking, apply_label_smoothing,
    apply_margin_persistence,
    compute_reference_purity, apply_purity_weighted_observations,
    compute_within_pop_variance, apply_variance_penalty,
    apply_flank_informed_bonus,
    compute_loo_robust_emissions, sharpen_posteriors,
    correct_short_segments, whiten_log_emissions,
    compute_window_quality, iterative_refine,
    compute_calibration_boosts, apply_calibration_boosts,
    apply_diversity_scaling, amplify_emission_residuals,
    rank_transform_emissions, apply_emission_anchor_boost,
    dampen_emission_outliers, compute_confusion_penalties, apply_confusion_penalties,
    apply_emission_momentum, apply_emission_floor,
    apply_gradient_penalty, blend_posteriors_with_emissions,
    apply_changepoint_prior, apply_pairwise_emission_contrast,
    adjust_pop_temperatures, apply_snr_weighting,
    regularize_toward_posteriors, apply_windowed_normalization,
    entropy_smooth_posteriors, quantile_normalize_emissions,
    compute_adaptive_transitions, local_rerank_emissions,
    bayesian_shrink_emissions, sparsify_top_k_emissions,
    majority_vote_filter, apply_proportion_prior,
    compute_boundary_boost_transitions, apply_confidence_weighting,
    apply_fb_temperature, compute_cooccurrence_transitions,
    detrend_emissions, compute_transition_momentum,
    variance_stabilize_emissions, compute_lookahead_transitions,
    apply_kurtosis_weighting, compute_segment_length_prior,
    apply_gap_penalty, compute_recency_transitions,
    center_emissions, apply_persistence_bonus,
    median_polish_emissions, compute_disagreement_transitions,
    softmax_renormalize, bidirectional_smooth_states,
    entropy_weighted_smooth_posteriors,
    compute_population_variances, compute_heteroscedastic_temperatures,
    precompute_heteroscedastic_log_emissions,
    compute_population_distances, set_distance_weighted_transitions,
    compute_rank_log_emissions, compute_consistency_log_emissions, compute_pairwise_log_emissions,
    compute_hierarchical_emissions, parse_population_groups, auto_detect_groups,
    compute_distance_transitions, compute_population_aware_transitions,
    forward_backward_from_log_emissions_with_transitions,
    viterbi_from_log_emissions_with_transitions, ensemble_decode,
    smooth_states, posterior_smooth_states, count_smoothing_changes,
    cross_validate,
    infer_ancestry_copying, infer_ancestry_copying_em, estimate_copying_params,
    scale_temperature_for_copying,
    estimate_admixture_proportions, estimate_identity_floor,
    refine_ancestry_boundaries,
    parse_similarity_for_egrm, write_gcta_grm, write_diploid_gcta_grm,
    DemographyParams, infer_all_demography, infer_per_sample_demography,
    format_demography_report, write_demography_tsv,
    rfmix, concordance,
};

#[derive(Parser, Debug)]
#[command(name = "ancestry", version, about = "Local ancestry inference from pangenome data",
    after_help = "\
RECOMMENDED PARAMETER PRESETS:

  Production (best validated concordance=93.4% vs RFMix on chr20):
    --emission-model max --estimate-params

  Explicit best params (single sample, HG00733):
    --emission-model max --temperature 0.03 --switch-prob 0.01

  Minority ancestry detection (higher AFR F1, lower concordance):
    --emission-model max --temperature 0.03 --switch-prob 0.02

  Key findings from validation against RFMix (chr20, 15 AMR samples):
    - Emission model 'max' is critical: 93.2% vs mean 88.7% vs median 86.4%
    - Auto-estimation (--estimate-params) achieves near-optimal 92.9% concordance
    - Baum-Welch training is counterproductive (degrades concordance by 2-3%)
    - Posterior decoding = Viterbi (signal too weak for disagreement)
    - Normalize-emissions is catastrophic (destroys the signal)
    - Coverage-ratio feature provides marginal improvement at best
    - Pangenome identity SNR ~0.07-0.26: insufficient for reliable minority ancestry
")]
struct Args {
    /// AGC file with assemblies
    #[arg(long = "sequence-files", required_unless_present = "similarity_file")]
    sequence_files: Option<PathBuf>,

    /// Alignment file (PAF)
    #[arg(short = 'a', long = "alignment", required_unless_present = "similarity_file")]
    alignment: Option<PathBuf>,

    /// Reference name for coordinate system (e.g., "CHM13")
    #[arg(short = 'r', long = "reference", required_unless_present = "similarity_file")]
    reference: Option<String>,

    /// Region to analyze (e.g., "chr12" or "chr12:1-133000000")
    /// Mutually exclusive with --bed
    #[arg(long = "region")]
    region: Option<String>,

    /// BED file with multiple regions to process (tab-separated: chrom, start, end)
    /// Mutually exclusive with --region
    #[arg(long = "bed", conflicts_with = "region")]
    bed: Option<PathBuf>,

    /// Window size in bp
    #[arg(long = "window-size", default_value = "5000")]
    window_size: u64,

    /// Query samples file (one sample#haplotype per line)
    #[arg(long = "query-samples", required = true)]
    query_samples: PathBuf,

    /// Population definition file (TSV: pop_name, haplotype_id)
    #[arg(long = "populations")]
    populations: Option<PathBuf>,

    /// Output file for ancestry segments
    #[arg(short = 'o', long = "output", required = true)]
    output: PathBuf,

    /// Output file for per-window posteriors (optional)
    #[arg(long = "posteriors-output")]
    posteriors_output: Option<PathBuf>,

    /// Output ancestry segments in BED format (chrom, start, end, name, score)
    /// Compatible with bedtools and genome browsers
    #[arg(long = "output-bed")]
    output_bed: Option<PathBuf>,

    /// Ancestry switch probability per window
    #[arg(long = "switch-prob", default_value = "0.001", value_parser = validate_unit_interval)]
    switch_prob: f64,

    /// Softmax temperature for emission model (controls sharpness of similarity → probability).
    /// Lower values = sharper (more confident). If --estimate-params is used, this is ignored.
    #[arg(long = "temperature", value_parser = validate_positive_f64)]
    temperature: Option<f64>,

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

    /// Which column to use as similarity metric from the similarity file.
    /// Options: estimated.identity (default), jaccard.similarity, cosine.similarity, dice.similarity
    #[arg(long = "similarity-column", default_value = "estimated.identity")]
    similarity_column: String,

    /// Minimum mean posterior probability to keep a segment
    #[arg(long = "min-posterior", default_value = "0.0", value_parser = validate_unit_interval)]
    min_posterior: f64,

    /// Minimum LOD score to keep an ancestry segment (0.0 = no filtering)
    /// Segments with LOD < min_lod are removed. Higher values keep only confident calls.
    #[arg(long = "min-lod", default_value = "0.0", value_parser = validate_non_negative_f64)]
    min_lod: f64,

    /// Minimum consecutive windows to keep a state assignment (for smoothing)
    /// Set to 0 to disable smoothing
    #[arg(long = "smooth-min-windows", default_value = "0")]
    smooth_min_windows: usize,

    /// Enable posterior-aware smoothing. Uses posterior probabilities to
    /// decide when to flip uncertain windows, rather than just run length.
    /// Requires --smooth-min-windows > 0 to activate.
    #[arg(long = "posterior-smooth")]
    posterior_smooth: bool,

    /// Confidence threshold for posterior smoothing. Windows with posterior
    /// below this for their assigned state may be flipped. Default: 0.6.
    #[arg(long = "posterior-smooth-threshold", default_value = "0.6",
          value_parser = validate_unit_interval)]
    posterior_smooth_threshold: f64,

    /// Number of neighboring windows on each side for emission smoothing.
    /// Pools evidence from [t-context, t+context] windows to strengthen weak signals.
    /// With context=2 (5-window span), per-window SNR increases by ~2.2x.
    /// Set to 0 to disable (default). Recommended: 2-3 for ancestry.
    #[arg(long = "emission-context", default_value = "0")]
    emission_context: usize,

    /// Automatically adapt emission context per sample based on signal quality.
    /// Weak-signal samples get wider context (more averaging) while strong-signal
    /// samples keep narrow context (better boundary precision). Uses the base
    /// --emission-context value as starting point, scales between min_ec=1 and max_ec=15.
    #[arg(long = "adaptive-context")]
    adaptive_context: bool,

    /// Run leave-one-out cross-validation on reference haplotypes
    #[arg(long = "cross-validate")]
    cross_validate: bool,

    /// Number of folds for k-fold cross-validation (0 = leave-one-out)
    #[arg(long = "cv-folds", default_value = "0")]
    cv_folds: usize,

    /// Output file for confusion matrix TSV (requires --cross-validate)
    #[arg(long = "confusion-output")]
    confusion_output: Option<PathBuf>,

    /// Automatically estimate HMM parameters (temperature and switch-prob) from data
    /// When enabled, --switch-prob becomes initial value for regularization
    #[arg(long = "estimate-params")]
    estimate_params: bool,

    /// Auto-scale temperature based on panel size (haplotypes per population).
    /// Uses extreme value theory: with more haplotypes, Max aggregation produces
    /// larger values, requiring lower temperature. Applied after estimation.
    #[arg(long = "auto-temperature-scaling")]
    auto_temperature_scaling: bool,

    /// Number of Baum-Welch iterations for HMM parameter training (0 = disabled)
    /// Re-estimates switch probability from data using EM
    #[arg(long = "baum-welch-iters", default_value = "0")]
    baum_welch_iters: usize,

    /// Use full Baum-Welch (re-estimate initial probs + optionally temperature).
    /// Standard BW only re-estimates transitions. Full BW also updates initial state
    /// distribution from the data, which helps with unbalanced ancestry proportions.
    /// Requires --baum-welch-iters > 0.
    #[arg(long = "baum-welch-full")]
    baum_welch_full: bool,

    /// Re-estimate temperature during full Baum-Welch via grid search.
    /// After each BW iteration, searches for the temperature that maximizes
    /// log-likelihood. Only effective with --baum-welch-full.
    #[arg(long = "baum-welch-temperature")]
    baum_welch_temperature: bool,

    /// Dampening factor for Baum-Welch transition learning (T52).
    /// Interpolates between MLE transitions and prior: 0.0 = full MLE (default),
    /// 1.0 = don't learn transitions. When --auto-configure is active with
    /// --baum-welch, defaults to 0.5 to prevent double-correction with pairwise.
    #[arg(long = "bw-dampening", default_value = "0.0")]
    bw_dampening: f64,

    /// Emission model for aggregating per-haplotype similarities: max, mean, median
    #[arg(long = "emission-model", default_value = "max")]
    emission_model: String,

    /// Exponential decay factor for TopK weighted aggregation.
    /// When set (0 < value < 1), TopK uses rank-decaying weights: top-1 gets weight 1,
    /// top-2 gets decay^1, top-3 gets decay^2, etc. Emphasizes the most similar
    /// haplotype while still using others for noise reduction. Only applies when
    /// emission-model is top3/top5/etc. Default 1.0 = uniform weights (standard TopK).
    #[arg(long = "topk-decay", default_value_t = 1.0)]
    topk_decay: f64,

    /// Enable heteroscedastic emission temperatures.
    /// Computes per-population identity variance and adjusts temperature accordingly:
    /// high-variance populations (e.g., admixed AMR) get softer emissions,
    /// low-variance populations (e.g., well-defined AFR) get sharper emissions.
    /// The gamma parameter controls adjustment strength (0.0=off, 0.5=moderate, 1.0=full).
    #[arg(long = "heteroscedastic-emissions", default_value_t = 0.0)]
    heteroscedastic_gamma: f64,

    /// Focused population masking threshold for two-pass inference.
    /// In pass-2, populations with pass-1 posterior below this threshold are
    /// hard-masked (set to -infinity). This concentrates discriminative power
    /// on locally competitive populations. Minimum 2 populations always remain.
    /// 0.0 = off, 0.05 = moderate (recommended), 0.1 = aggressive.
    /// Requires --two-pass. Complements --posterior-feedback (hard vs soft gating).
    #[arg(long = "focused-pass2", default_value_t = 0.0)]
    focused_pass2_threshold: f64,

    /// Entropy-weighted posterior smoothing radius.
    /// After HMM inference, smooth posteriors using a confidence-weighted kernel:
    /// confident windows (low entropy) propagate signal to uncertain neighbors.
    /// 0 = off, 3 = moderate (recommended), 5 = aggressive.
    /// Applied as post-processing after any decoding method.
    #[arg(long = "entropy-smooth", default_value_t = 0)]
    entropy_smooth_radius: usize,

    /// Beta parameter for entropy-weighted smoothing selectivity.
    /// Higher beta = more weight on confident windows.
    /// 2.0 = ~7x weight ratio between most/least confident.
    #[arg(long = "entropy-smooth-beta", default_value_t = 2.0)]
    entropy_smooth_beta: f64,

    /// Use population-distance weighted transitions.
    /// Computes pairwise population distances from identity profiles and sets
    /// transition rates proportional to closeness: EUR↔AMR (close) gets higher
    /// inter-transition rate than EUR↔EAS (distant). Requires --two-pass.
    #[arg(long = "pop-distance-transitions")]
    pop_distance_transitions: bool,

    /// Emission label smoothing factor.
    /// Interpolates log-emissions toward uniform: log_e' = (1-α)×log_e + α×log(1/K).
    /// Prevents overconfident emissions that trap the HMM in one state.
    /// 0.0 = off, 0.05 = mild (recommended), 0.1 = moderate. Range [0, 1].
    #[arg(long = "label-smooth", default_value_t = 0.0)]
    label_smooth_alpha: f64,

    /// Margin-gated state persistence bonus for two-pass inference.
    /// In pass-2, adds emission bonus to pass-1 argmax state when pass-1 margin
    /// exceeds a threshold, locking in high-confidence calls. Bonus scales linearly
    /// from 0 at margin=threshold to full value at margin=1.0.
    /// 0.0 = off, 1.0 = moderate (recommended), 2.0 = strong.
    /// Requires --two-pass.
    #[arg(long = "margin-persistence", default_value_t = 0.0)]
    margin_persistence_bonus: f64,

    /// Margin threshold for state persistence. Only windows with pass-1 margin
    /// above this threshold receive the persistence bonus.
    /// 0.5 = moderate (recommended for K=5), 0.3 = permissive, 0.7 = strict.
    #[arg(long = "margin-persistence-threshold", default_value_t = 0.5)]
    margin_persistence_threshold: f64,

    /// Use per-window adaptive pairwise weight scaling.
    /// Windows with small emission gaps (ambiguous) get higher pairwise weight;
    /// windows with large gaps (clear) get lower weight. Replaces the global
    /// --pairwise-weight with a locally adaptive version.
    #[arg(long = "adaptive-pairwise-per-window")]
    adaptive_pairwise_per_window: bool,

    /// Reference purity weighting gamma.
    /// Computes per-haplotype purity scores and weights similarities accordingly.
    /// High-purity references (strong population indicators) get more weight;
    /// low-purity (admixed) references get less weight. Weight = purity^gamma.
    /// 0.0 = off, 0.5 = moderate (recommended), 1.0 = full weighting.
    #[arg(long = "purity-weight", default_value_t = 0.0)]
    purity_weight_gamma: f64,

    /// Within-population variance penalty weight.
    /// Penalizes populations where the aggregated similarity is driven by one outlier
    /// haplotype rather than consistent similarity across all references.
    /// 0.0 = off, 1.0 = moderate, 5.0 = strong.
    #[arg(long = "variance-penalty", default_value_t = 0.0)]
    variance_penalty: f64,

    /// Flank-informed emission bonus radius.
    /// In pass-2, examines decoded states from pass-1 in flanking windows.
    /// When both left and right flanks agree on the same state, adds a bonus
    /// to that state's emission. Leverages spatial context from neighbors.
    /// 0 = off, 3-5 recommended.
    #[arg(long = "flank-inform", default_value_t = 0)]
    flank_inform_radius: usize,

    /// Flank-informed emission bonus strength (nats).
    /// Applied when both flanks agree. Default 0.5 nats ≈ 1.65× likelihood boost.
    #[arg(long = "flank-inform-bonus", default_value_t = 0.5)]
    flank_inform_bonus: f64,

    /// Use leave-one-out robust emissions.
    /// For each population, computes the minimum aggregated similarity after
    /// removing each haplotype in turn. Detects when a score is driven by
    /// one outlier haplotype. Can be blended with standard emissions.
    #[arg(long = "loo-robust")]
    loo_robust: bool,

    /// Weight for blending LOO robust emissions with standard emissions.
    /// 0.0 = all standard, 1.0 = all LOO robust. Default 0.3.
    #[arg(long = "loo-robust-weight", default_value_t = 0.3)]
    loo_robust_weight: f64,

    /// Posterior sharpening temperature.
    /// Applied after forward-backward to sharpen/soften posteriors before
    /// MPEL decode or entropy smoothing. T < 1 sharpens, T > 1 softens.
    /// 0.0 = off, 0.5 = moderate sharpening. Default 0.0 (off).
    #[arg(long = "sharpen-posteriors", default_value_t = 0.0)]
    sharpen_posteriors_temp: f64,

    /// Minimum ancestry segment length in windows.
    /// Post-decoding: segments shorter than this are merged with the
    /// best-supported neighbor (by emission evidence). Prevents isolated
    /// 1-2 window misassignments. 0 = off, 3-5 recommended.
    #[arg(long = "min-ancestry-windows", default_value_t = 0)]
    min_ancestry_windows: usize,

    /// Apply ZCA whitening to log-emissions before HMM inference.
    /// Decorrelates population signals to improve softmax discrimination
    /// for highly correlated pairs (EUR↔AMR). Uses eigendecomposition
    /// with regularization.
    #[arg(long = "whiten-emissions")]
    whiten_emissions: bool,

    /// Number of iterative refinement passes.
    /// Generalizes two-pass to N-pass. Each pass uses the previous pass's
    /// posteriors to refine emissions via posterior feedback. Lambda increases
    /// progressively. 0 = off, 3-5 recommended.
    #[arg(long = "iterative-refine", default_value_t = 0)]
    iterative_refine_passes: usize,

    /// Output per-window quality scores to stderr.
    /// Composite metric combining posterior margin, emission discriminability,
    /// and neighbor agreement. Useful for identifying low-confidence regions.
    #[arg(long = "output-quality")]
    output_quality: bool,

    /// Population-specific emission calibration scale factor.
    /// Corrects systematic under/over-calling by boosting emissions for
    /// under-represented populations in pass-2. Based on pass-1 observed
    /// vs expected proportions. 0.0 = off, 0.3 = moderate, 0.5 = aggressive.
    #[arg(long = "calibrate-scale", default_value_t = 0.0)]
    calibrate_scale: f64,

    /// Entropy-based diversity scaling. Amplifies emissions in low-entropy (confident)
    /// windows and dampens in high-entropy (ambiguous) windows. 0.0 = off.
    /// Amplify factor for confident windows (e.g. 1.2).
    #[arg(long = "diversity-amplify", default_value_t = 0.0)]
    diversity_amplify: f64,

    /// Dampen factor for ambiguous windows (e.g. 0.5). Only used when diversity-amplify > 0.
    #[arg(long = "diversity-dampen", default_value_t = 0.5)]
    diversity_dampen: f64,

    /// Amplify cross-population emission residuals (deviation from per-window mean).
    /// factor > 1.0 amplifies population-specific signal. 0.0 = off, 1.0 = centering only,
    /// 2.0 = 2x amplified residuals.
    #[arg(long = "residual-amplify", default_value_t = 0.0)]
    residual_amplify: f64,

    /// Convert emissions to rank-based scores. Eliminates absolute scale
    /// differences between populations, forces HMM to rely on relative ordering.
    /// Different from --rank-emissions (which counts haplotypes in top-K).
    #[arg(long = "rank-transform")]
    rank_transform: bool,

    /// Emission anchor boosting: boosts emission for argmax state when
    /// neighborhood argmax agrees. Radius = number of windows on each side.
    /// 0 = off.
    #[arg(long = "anchor-boost-radius", default_value_t = 0)]
    anchor_boost_radius: usize,

    /// Anchor boost minimum agreement fraction to trigger boost (0.0 to 1.0).
    #[arg(long = "anchor-boost-threshold", default_value_t = 0.6,
          value_parser = validate_unit_interval)]
    anchor_boost_threshold: f64,

    /// Anchor boost maximum log-space bonus.
    #[arg(long = "anchor-boost-strength", default_value_t = 0.5)]
    anchor_boost_strength: f64,

    /// Dampen outlier emissions that deviate beyond this many MADs from the
    /// per-population median. Prevents isolated high-emission spikes from
    /// misleading the HMM. 0.0 = off, 3.0 = moderate, 2.0 = aggressive.
    #[arg(long = "outlier-dampen", default_value_t = 0.0)]
    outlier_dampen: f64,

    /// Apply confusion penalty to transitions for frequently confused population
    /// pairs from pass-1. Makes it harder to switch between EUR↔AMR when they
    /// have high mutual switch rates. 0.0 = off, 0.5 = moderate, 1.0 = strong.
    #[arg(long = "confusion-penalty", default_value_t = 0.0)]
    confusion_penalty: f64,

    /// Emission momentum: forward-backward exponential moving average.
    /// Smooths emissions temporally, reducing single-window noise.
    /// 0.0 = off, 0.3 = light, 0.7 = heavy smoothing.
    #[arg(long = "emission-momentum", default_value_t = 0.0)]
    emission_momentum: f64,

    /// Minimum log-emission per population per window. Prevents any population
    /// from being completely ruled out. Unlike label smoothing (which moves
    /// toward uniform), this sets a hard floor. 0.0 = off, -10.0 = moderate.
    #[arg(long = "emission-floor", default_value_t = 0.0)]
    emission_floor: f64,

    /// Penalize large emission changes between adjacent windows.
    /// Encourages smoother emission landscapes. 0.0 = off, 0.3 = moderate.
    #[arg(long = "gradient-penalty", default_value_t = 0.0,
          value_parser = validate_unit_interval)]
    gradient_penalty: f64,

    /// Posterior-weighted Viterbi: blends log-posteriors with log-emissions
    /// before a final Viterbi decode. Combines posterior evidence with path
    /// smoothness. 0.0 = off (standard decode), 0.5 = balanced.
    #[arg(long = "posterior-viterbi-lambda", default_value_t = 0.0)]
    posterior_viterbi_lambda: f64,

    /// Changepoint prior: adds a persistence bonus to pass-1 decoded states.
    /// Creates inertia that resists state changes. 0.0 = off, 0.5 = moderate.
    #[arg(long = "changepoint-prior", default_value_t = 0.0)]
    changepoint_prior: f64,

    /// Pairwise emission contrast: widens the gap between the top-2 populations
    /// per window. Directly targets EUR/AMR confusion. 0.0 = off, 0.3 = moderate.
    #[arg(long = "pairwise-contrast", default_value_t = 0.0)]
    pairwise_contrast: f64,

    /// Per-population temperature adjustment based on pass-1 posterior confidence.
    /// Low-confidence pops get sharper (cooled), high-confidence get softer (warmed).
    /// 0.0 = off, 1.0 = moderate, 2.0 = strong.
    #[arg(long = "pop-temp-adjust", default_value_t = 0.0)]
    pop_temp_adjust: f64,

    /// SNR-weighted emissions: scale emission deviations by per-window
    /// signal-to-noise ratio. High-SNR windows get amplified, low-SNR dampened.
    /// Power exponent: 0.0 = off, 0.5 = sqrt, 1.0 = linear.
    #[arg(long = "snr-weight", default_value_t = 0.0)]
    snr_weight: f64,

    /// Cross-entropy regularization: blend emissions toward pass-1 posteriors.
    /// Prevents pass-2 from deviating too far from pass-1 beliefs.
    /// 0.0 = off, 0.2 = light, 0.5 = moderate.
    #[arg(long = "cross-entropy-reg", default_value_t = 0.0)]
    cross_entropy_reg: f64,

    /// Windowed normalization: removes slowly-varying emission trends by
    /// normalizing relative to a local window mean. Radius in windows.
    /// 0 = off, 10 = moderate, 50 = aggressive.
    #[arg(long = "window-normalize", default_value_t = 0)]
    window_normalize: usize,

    /// (Removed: consolidated into entropy_smooth_radius above)

    /// Quantile normalize emissions: ensures all populations have the same
    /// marginal emission distribution across windows, removing pop-specific biases.
    #[arg(long = "quantile-normalize", default_value_t = false)]
    quantile_normalize: bool,

    /// Adaptive transition scaling: scales transition penalties per window based
    /// on emission entropy. Confident windows resist switching, uncertain allow it.
    /// 0.0 = off, 1.0 = moderate, 2.0 = aggressive.
    #[arg(long = "adaptive-transition-scale", default_value_t = 0.0)]
    adaptive_transition_scale: f64,

    /// Local emission reranking: re-ranks populations by cumulative emission
    /// support in a [t-R, t+R] neighborhood. Radius in windows. 0 = off.
    #[arg(long = "local-rerank", default_value_t = 0)]
    local_rerank: usize,

    /// Bayesian emission shrinkage: pulls per-window emissions toward global
    /// population mean. Reduces noise. 0.0 = off, 0.3 = moderate, 1.0 = full.
    #[arg(long = "bayesian-shrink", default_value_t = 0.0)]
    bayesian_shrink: f64,

    /// Top-K emission sparsification: keep only top-K populations per window,
    /// set rest to floor. 0 = off (keep all).
    #[arg(long = "sparsify-top", default_value_t = 0)]
    sparsify_top: usize,

    /// Sliding window majority vote: post-decoding filter that replaces each
    /// state with the most common state in neighborhood. Radius in windows. 0 = off.
    #[arg(long = "majority-vote", default_value_t = 0)]
    majority_vote: usize,

    /// Population proportion prior: adds log-proportion bonus to emissions
    /// based on pass-1 estimated population frequencies. 0.0 = off, 1.0 = standard.
    #[arg(long = "proportion-prior", default_value_t = 0.0)]
    proportion_prior: f64,

    /// Segment boundary boost: eases transitions at pass-1 decoded boundaries.
    /// Makes ancestry switches more likely where pass-1 detected them.
    /// 0.0 = off, 1.0 = moderate, 3.0 = aggressive.
    #[arg(long = "boundary-boost", default_value_t = 0.0)]
    boundary_boost: f64,

    /// Emission confidence weighting: amplifies emissions at confident windows
    /// (large gap between top-2 populations). Power exponent. 0.0 = off.
    #[arg(long = "confidence-weight", default_value_t = 0.0)]
    confidence_weight: f64,

    /// Forward-backward temperature: scales log-emissions by 1/T before inference.
    /// T<1 sharpens (more confident), T>1 softens (more uniform). 1.0 = off.
    #[arg(long = "fb-temperature", default_value_t = 1.0)]
    fb_temperature: f64,

    /// Population co-occurrence bonus: boosts transitions between populations
    /// that frequently co-occur at pass-1 boundaries. 0.0 = off.
    #[arg(long = "cooccurrence-bonus", default_value_t = 0.0)]
    cooccurrence_bonus: f64,

    /// Emission detrending: removes linear trends per population across genomic
    /// position. Prevents systematic drift from alignment quality gradients.
    #[arg(long = "detrend-emissions", default_value_t = false)]
    detrend_emissions: bool,

    /// Transition momentum: per-window transitions where longer runs in a state
    /// increase self-transition probability. Makes it harder to switch out of
    /// well-established ancestry blocks. 0.0 = off.
    #[arg(long = "transition-momentum", default_value_t = 0.0)]
    transition_momentum: f64,

    /// Emission variance stabilization: rescales per-population emissions to have
    /// comparable spread (median std across populations). Prevents high-variance
    /// populations from dominating.
    #[arg(long = "variance-stabilize", default_value_t = false)]
    variance_stabilize: bool,

    /// Lookahead transition adjustment: uses emission evidence from R future windows
    /// to ease transitions toward upcoming dominant populations. 0 = off.
    #[arg(long = "lookahead-transitions", default_value_t = 0)]
    lookahead_transitions: usize,

    /// Emission kurtosis weighting: dampens heavy-tailed populations, boosts
    /// light-tailed ones. Controls outlier sensitivity. 0.0 = off.
    #[arg(long = "kurtosis-weight", default_value_t = 0.0)]
    kurtosis_weight: f64,

    /// Segment length prior: penalizes transitions creating segments shorter
    /// than L windows. Encourages longer, more stable ancestry blocks. 0 = off.
    #[arg(long = "segment-length-prior", default_value_t = 0)]
    segment_length_prior: usize,

    /// Emission gap penalty: penalizes populations with high temporal variability
    /// (large emission jumps between adjacent windows). 0.0 = off.
    #[arg(long = "gap-penalty", default_value_t = 0.0)]
    gap_penalty: f64,

    /// Recency-weighted transitions: recent decoded state history influences
    /// per-window transitions via exponential decay. Alpha=0.9 typical. 0.0 = off.
    #[arg(long = "recency-transitions", default_value_t = 0.0)]
    recency_transitions: f64,

    /// Emission centering: per-window mean subtraction. Transforms emissions
    /// to relative advantage space. false = off.
    #[arg(long = "center-emissions", default_value_t = false)]
    center_emissions: bool,

    /// State persistence bonus: adds emission bonus to pass-1 decoded state.
    /// Acts as soft sticky prior on emissions. 0.0 = off.
    #[arg(long = "persistence-bonus", default_value_t = 0.0)]
    persistence_bonus: f64,

    /// Emission median polish: Tukey's median polish removing row/column effects.
    /// Isolates window×population interaction residuals.
    #[arg(long = "median-polish", default_value_t = false)]
    median_polish: bool,

    /// Disagreement penalty transitions: boosts self-transition when emission
    /// argmax agrees with pass-1 state, eases switching when they disagree. 0.0 = off.
    #[arg(long = "disagreement-penalty", default_value_t = 0.0)]
    disagreement_penalty: f64,

    /// Emission softmax renormalization: converts log-emissions to softmax probabilities
    /// at temperature T, then back to log space. Standardizes emission scale. 0.0 = off.
    #[arg(long = "softmax-renorm", default_value_t = 0.0)]
    softmax_renorm: f64,

    /// Bidirectional state smoothing: post-decoding exponentially-weighted vote
    /// from [t-R, t+R] neighborhood. Closer neighbors have stronger influence. 0 = off.
    #[arg(long = "bidirectional-smooth", default_value_t = 0)]
    bidirectional_smooth: usize,

    /// Decoding method: viterbi, posterior, or mpel
    /// 'viterbi' finds the most likely state *sequence* (MAP path), favoring contiguous blocks.
    /// 'posterior' chooses the most likely state at each window independently from
    /// forward-backward posteriors. Better for detecting minority ancestry tracts.
    /// 'mpel' (Maximum Posterior Expected Loss): runs forward-backward then Viterbi on
    /// log-posteriors. Combines full-sequence evidence with path smoothness — principled
    /// replacement for heuristic smoothing.
    #[arg(long = "decoding", default_value = "viterbi")]
    decoding: String,

    /// Genetic map file (PLINK format) for recombination-rate-aware transitions.
    /// Regions with high recombination rate get higher switch probability.
    /// Format: chr pos_bp rate_cM_Mb pos_cM (4-column) or pos_bp rate pos_cM (3-column).
    #[arg(long = "genetic-map")]
    genetic_map: Option<PathBuf>,

    /// Normalize per-population emission scores before softmax.
    /// Removes systematic bias where some populations have inherently higher
    /// similarity (e.g., EUR > AFR in pangenome data). Learns per-population
    /// mean and std from the data, then z-score normalizes before softmax.
    /// This improves detection of minority ancestry tracts.
    #[arg(long = "normalize-emissions")]
    normalize_emissions: bool,

    /// Use alignment coverage ratio as auxiliary emission feature.
    /// Coverage ratio = min(a_len, b_len) / max(a_len, b_len) measures
    /// alignment symmetry. Requires group.a.length and group.b.length
    /// columns in the similarity file.
    #[arg(long = "coverage-feature")]
    coverage_feature: bool,

    /// Weight for coverage-ratio auxiliary emission (default: 1.0).
    /// Combined emission = log P(sim|state) + weight * log P(cov|state).
    /// Only used when --coverage-feature is enabled.
    #[arg(long = "coverage-weight", default_value = "1.0", value_parser = validate_non_negative_f64)]
    coverage_weight: f64,

    /// Filter out windows where max similarity across all reference haplotypes
    /// is below this threshold. Removes alignment-gap windows that produce
    /// near-zero similarity values and confuse emission estimation.
    /// Recommended: 0.9 for pangenome data. Default: 0.0 (no filtering).
    /// Use --auto-identity-floor to estimate from data.
    #[arg(long = "identity-floor", default_value = "0.0", value_parser = validate_unit_interval)]
    identity_floor: f64,

    /// Automatically estimate the identity floor from data.
    /// Detects alignment-gap windows by finding gaps in the distribution of
    /// per-window maximum similarity. Overrides --identity-floor if set.
    #[arg(long = "auto-identity-floor")]
    auto_identity_floor: bool,

    /// Validate against RFMix .msp.tsv file. Computes concordance metrics
    /// comparing our ancestry calls with RFMix ground truth.
    #[arg(long = "validate-against")]
    validate_against: Option<PathBuf>,

    /// Output file for validation metrics TSV (requires --validate-against)
    #[arg(long = "validate-output")]
    validate_output: Option<PathBuf>,

    /// BED file of regions to mask (exclude from HMM inference).
    /// Windows overlapping masked regions are removed before inference.
    /// Use for segmental duplications, centromeric repeats, etc.
    #[arg(long = "mask-bed")]
    mask_bed: Option<PathBuf>,

    /// Enable two-pass inference with adaptive priors.
    /// Pass 1: standard inference with uniform priors to estimate ancestry proportions.
    /// Pass 2: re-run with proportion-based priors and asymmetric transitions.
    /// This helps when ancestry is unbalanced (e.g., 90% EUR + 10% AFR), as the
    /// uniform prior (1/k) biases toward equal ancestry. Improves boundary precision
    /// and reduces spurious minority ancestry calls.
    #[arg(long = "two-pass")]
    two_pass: bool,

    /// Posterior feedback strength for two-pass inference.
    /// Uses pass-1 posteriors to focus pass-2 emissions on locally relevant populations.
    /// Higher values = stronger focusing. 0.0 = off, 0.5 = moderate, 1.0 = strong.
    /// Requires --two-pass.
    #[arg(long = "posterior-feedback", default_value_t = 0.0)]
    posterior_feedback: f64,

    /// Scale temperature based on number of populations.
    /// With more populations competing in softmax, temperature should decrease
    /// to maintain discriminability. Uses k=3 as reference.
    /// Applied after panel-size scaling.
    #[arg(long = "population-temperature-scaling")]
    population_temperature_scaling: bool,

    /// Use SNR-weighted emission context smoothing instead of uniform averaging.
    /// Weights neighboring windows by their discrimination gap (top1 - top2 log emission).
    /// More informative windows get more influence. Only effective when --emission-context > 0.
    #[arg(long = "weighted-context")]
    weighted_context: bool,

    /// Apply per-window contrast normalization to emissions.
    /// Centers each window's log emissions to zero mean across states, removing
    /// global identity level effects. Focuses on RELATIVE similarity ranking.
    /// Helps when absolute identity varies across windows (repetitive regions, etc.).
    #[arg(long = "contrast-normalize")]
    contrast_normalize: bool,

    /// Dampen emissions for low-discriminability windows. Windows where the gap
    /// between best and second-best state emission is small get emissions scaled
    /// toward uniform, letting the HMM transition model dominate for uncertain
    /// windows. Addresses weak-signal errors by preventing noisy emissions from
    /// causing spurious state switches.
    #[arg(long = "dampen-emissions")]
    dampen_emissions: bool,

    /// Scale factor for emission dampening. Controls how aggressively to dampen.
    /// 1.0 = dampen windows below median discriminability (aggressive).
    /// 2.0 = only dampen well below median (gentle). Default: 1.5.
    #[arg(long = "dampen-scale", default_value = "1.5",
          value_parser = validate_positive_f64)]
    dampen_scale: f64,

    /// Blend rank-based emissions with standard softmax emissions.
    /// Ranks all haplotypes by similarity per window and counts how many from
    /// each population appear in the top K. Robust to weak absolute signal —
    /// when similarity differences are tiny (e.g., 0.0002), rank ordering is
    /// more stable than value aggregation.
    #[arg(long = "rank-emissions")]
    rank_emissions: bool,

    /// Number of top-ranked haplotypes for rank-based emissions. 0 = auto
    /// (total haplotypes / number of populations). Recommended: 3-10.
    #[arg(long = "rank-topk", default_value = "0")]
    rank_topk: usize,

    /// Weight for rank-based emissions when --rank-emissions is active.
    /// Blends standard softmax (1-w) with rank-based (w).
    /// 0.0 = pure softmax, 1.0 = pure rank. Default: 0.3.
    #[arg(long = "rank-weight", default_value = "0.3",
          value_parser = validate_unit_interval)]
    rank_weight: f64,

    /// Use hierarchical emission computation to address softmax IIA limitation.
    /// Decomposes emissions into group-level (broad ancestry) and within-group
    /// (fine ancestry) components. Helps distinguish related populations (e.g.,
    /// EUR vs AMR) that standard softmax conflates.
    #[arg(long = "hierarchical-emissions")]
    hierarchical_emissions: bool,

    /// Population group specification for hierarchical emissions.
    /// Format: "EUR,AMR;AFR;EAS,CSA" — semicolons separate groups, commas
    /// separate populations within a group. If not specified, groups are
    /// auto-detected from inter-population similarity.
    #[arg(long = "population-groups")]
    population_groups: Option<String>,

    /// Weight for the group-level component in hierarchical emissions.
    /// 0.0 = pure within-group (no hierarchy), 1.0 = pure group-level.
    /// Default: 0.5 (equal blend of broad and fine discrimination).
    #[arg(long = "group-weight", default_value = "0.5",
          value_parser = validate_unit_interval)]
    group_weight: f64,

    /// Blend consistency-based emissions with standard emissions. Counts how
    /// many neighboring windows each population "wins" (highest similarity),
    /// converting tiny absolute differences into robust frequency statistics.
    /// Effective for discriminating populations with very similar identity
    /// profiles (e.g., EUR vs AMR where per-window differences are ~0.0002).
    /// Computed on raw (unsmoothed) observations for independence from softmax.
    #[arg(long = "consistency-emissions")]
    consistency_emissions: bool,

    /// Number of neighboring windows on each side for consistency counting.
    /// Default: same as --emission-context (or 10 if emission-context is 0).
    /// Larger values give more robust counts but reduce boundary resolution.
    #[arg(long = "consistency-context", default_value = "0")]
    consistency_context: usize,

    /// Weight for consistency-based emissions when --consistency-emissions is active.
    /// Blends standard (1-w) with consistency (w). Default: 0.3.
    #[arg(long = "consistency-emission-weight", default_value = "0.3",
          value_parser = validate_unit_interval)]
    consistency_emission_weight: f64,

    /// Use population-similarity-aware transitions. Makes transitions between
    /// similar populations more likely than between distant ones:
    /// P(i→j) ∝ exp(-gap(i,j)/scale). For EUR/AMR/AFR, EUR↔AMR transitions
    /// are more likely than EUR↔AFR, reflecting biological admixture patterns.
    #[arg(long = "population-transitions")]
    population_transitions: bool,

    /// Use pairwise contrast emissions with per-pair adaptive temperature.
    /// Each pair of populations gets its own temperature calibrated to the
    /// typical similarity gap between them. This avoids the IIA problem of
    /// standard softmax: EUR-vs-AMR uses a tiny temperature (matching their
    /// ~0.0002 gaps) while EUR-vs-AFR uses a larger one. Scores are computed
    /// Bradley-Terry style and normalized via log-softmax.
    #[arg(long = "pairwise-emissions")]
    pairwise_emissions: bool,

    /// Weight for pairwise contrast emissions when --pairwise-emissions is active.
    /// Blends standard softmax (1-w) with pairwise (w). Default: 0.3.
    #[arg(long = "pairwise-weight", default_value = "0.3",
          value_parser = validate_unit_interval)]
    pairwise_weight: f64,

    /// Adaptive per-window pairwise weight. Scales the pairwise weight at each
    /// window by the pairwise emission confidence (margin between best and
    /// second-best population). Strong pairwise signal → higher weight; weak
    /// signal → falls back to standard emission. Requires --pairwise-emissions.
    #[arg(long = "adaptive-pairwise")]
    adaptive_pairwise: bool,

    /// Agreement-based per-window pairwise weight (T76). Scales pairwise weight
    /// based on whether the standard and pairwise models agree on argmax population.
    /// Agreement → up-weight (likely correct); disagreement → down-weight (likely
    /// biased). Avoids the Stein paradox where high-margin biased windows get
    /// amplified. Requires --pairwise-emissions. Mutually exclusive with
    /// --adaptive-pairwise.
    #[arg(long = "agreement-pairwise")]
    agreement_pairwise: bool,

    /// Scale factor for pairwise weight when BT and standard models agree.
    /// Used with --agreement-pairwise. Default 1.5 (50% boost on agreement).
    #[arg(long = "agreement-agree-scale", default_value_t = 1.5)]
    agreement_agree_scale: f64,

    /// Scale factor for pairwise weight when BT and standard models disagree.
    /// Used with --agreement-pairwise. Default 0.2 (80% reduction on disagreement).
    #[arg(long = "agreement-disagree-scale", default_value_t = 0.2)]
    agreement_disagree_scale: f64,

    /// Hybrid agreement-margin pairwise blending (T78 §4.2). Combines agreement
    /// gating with margin scaling: on agree windows, weight is modulated by BT
    /// margin ratio (strong signal → more pairwise). On disagree windows, flat
    /// down-weight (no margin). Recovers ~0.3 bits in well-specified regime.
    /// Requires --pairwise-emissions. Mutually exclusive with --agreement-pairwise
    /// and --adaptive-pairwise.
    #[arg(long = "hybrid-pairwise")]
    hybrid_pairwise: bool,

    /// Lower clamp for margin ratio in hybrid mode. Default 0.2.
    #[arg(long = "hybrid-margin-clamp-lo", default_value_t = 0.2)]
    hybrid_margin_clamp_lo: f64,

    /// Upper clamp for margin ratio in hybrid mode. Default 3.0.
    #[arg(long = "hybrid-margin-clamp-hi", default_value_t = 3.0)]
    hybrid_margin_clamp_hi: f64,

    /// Per-population agreement scaling (T79). Uses D_min heuristic to compute
    /// population-specific agree_scale (well-separated populations like AFR get
    /// higher boost, poorly separated like AMR get near-neutral). Disagreement
    /// uses per-pair scaling: close pairs (EUR↔AMR) get moderate suppression,
    /// distant pairs (AFR↔EUR) get stronger suppression. Combines with
    /// --agreement-pairwise or --hybrid-pairwise. Requires --pairwise-emissions.
    #[arg(long = "per-population-agreement")]
    per_population_agreement: bool,

    /// Auto-configure pairwise_weight and emission_context from data statistics.
    /// Computes Cohen's d for each population pair and uses the T53 formula to
    /// determine optimal parameters. Eliminates cross-application penalty between
    /// simulation and real data. Implies --pairwise-emissions and --estimate-params.
    #[arg(long = "auto-configure")]
    auto_configure: bool,

    /// Deconvolve admixed reference populations into sub-components.
    /// When enabled (or auto-triggered by --auto-configure when D_min < 0.01),
    /// populations with high within-haplotype variance and low inter-population
    /// Cohen's d are split via k-means clustering. Sub-populations are merged
    /// back in the output (e.g., AMR_0 + AMR_1 → AMR).
    #[arg(long = "deconvolve")]
    deconvolve: bool,

    /// Cohen's d threshold for triggering deconvolution.
    /// Population pairs with d below this value are candidates for splitting.
    #[arg(long = "deconvolve-threshold", default_value = "0.01")]
    deconvolve_threshold: f64,

    /// Scale transition probabilities based on physical distance between
    /// consecutive windows. When windows are farther apart (filtered windows,
    /// missing data), the switch probability increases proportionally. This
    /// provides a simple approximation to genetic-map-aware transitions without
    /// requiring a genetic map file.
    #[arg(long = "distance-transitions")]
    distance_transitions: bool,

    /// Refine segment boundaries using posterior probability interpolation.
    /// Estimates sub-window boundary positions by finding where the assigned
    /// state's posterior crosses 0.5 between adjacent windows.
    /// Improves boundary resolution from ~10kb to ~2-5kb.
    #[arg(long = "refine-boundaries")]
    refine_boundaries: bool,

    /// Number of ensemble members for bootstrap-aggregated decoding.
    /// Runs forward-backward with multiple parameter perturbations and averages
    /// posteriors for more robust state assignments. 0 or 1 = disabled (single run).
    /// Recommended: 5-11 (odd numbers for tie-breaking). Default: 0.
    #[arg(long = "ensemble-size", default_value = "0")]
    ensemble_size: usize,

    /// Scale factor for ensemble parameter perturbation.
    /// Temperature and switch probability are varied from 1/factor to factor.
    /// Larger values = wider exploration. Default: 2.0.
    #[arg(long = "ensemble-scale", default_value = "2.0",
          value_parser = validate_positive_f64)]
    ensemble_scale: f64,

    /// Learn emission profiles from pass-1 assignments for cross-population
    /// pattern matching. Uses Pearson correlation between the observation's
    /// full similarity vector and each state's learned centroid profile.
    /// Addresses the softmax IIA limitation by capturing which population
    /// similarities co-occur. Requires --two-pass. Combined with --emission-context
    /// for best results.
    #[arg(long = "learn-profiles")]
    learn_profiles: bool,

    /// Weight for profile-based emissions when --learn-profiles is active.
    /// Blends standard softmax (1-w) with profile correlation (w).
    /// 0.0 = pure softmax, 1.0 = pure profile. Default: 0.3.
    #[arg(long = "profile-weight", default_value = "0.3",
          value_parser = validate_unit_interval)]
    profile_weight: f64,

    /// Context window for haplotype consistency tracking.
    /// For each window and population, checks if the best-matching reference
    /// haplotype is consistent across ±N neighboring windows. Consistent matches
    /// indicate genuine haplotype sharing (LD-like signal), boosting that
    /// population's emission probability. 0 = disabled (default). Recommended: 3-5.
    #[arg(long = "haplotype-consistency", default_value = "0")]
    haplotype_consistency: usize,

    /// Weight for haplotype consistency bonus in softmax-score space.
    /// A weight of 1.0 means a fully consistent population gets exp(1.0) ≈ 2.7x
    /// probability boost. Default: 0.5.
    #[arg(long = "consistency-weight", default_value = "0.5",
          value_parser = validate_positive_f64)]
    consistency_weight: f64,

    /// Use haplotype copying model (Li & Stephens inspired) instead of
    /// population-aggregate emission model. Models the query as a mosaic of
    /// reference haplotypes, capturing haplotype continuity information that
    /// population-level Max/Mean/TopK aggregation discards.
    #[arg(long = "copying-model")]
    copying_model: bool,

    /// Haplotype switch rate for copying model: per-window probability of
    /// switching the copied reference haplotype. Lower values favor longer
    /// copying tracts. Default: auto-estimated from data.
    #[arg(long = "copying-switch-rate",
          value_parser = validate_unit_interval)]
    copying_switch_rate: Option<f64>,

    /// Ancestry switch fraction for copying model: fraction of haplotype
    /// switches that also change the population assignment. Lower values
    /// mean most recombination is within-population. Default: 0.1.
    #[arg(long = "copying-ancestry-frac", default_value = "0.1",
          value_parser = validate_unit_interval)]
    copying_ancestry_frac: f64,

    /// Number of EM iterations for the copying model. Re-estimates haplotype
    /// switch rate and ancestry fraction from expected transition counts
    /// (Baum-Welch style). 0 = single pass with initial estimates.
    #[arg(long = "copying-em-iters", default_value = "0")]
    copying_em_iters: usize,

    /// Output eGRM (expected Genetic Relationship Matrix) in GCTA binary format.
    /// Parses the similarity file for ALL pairwise identities (not just query-vs-ref)
    /// and outputs the averaged identity matrix as .grm.bin, .grm.N.bin, .grm.id files.
    /// Compatible with GCTA, BOLT-LMM, SAIGE for GWAS, PCA, heritability estimation.
    /// Requires --similarity-file (pre-computed similarities).
    #[arg(long = "output-egrm")]
    output_egrm: Option<String>,

    /// Center the eGRM by subtracting the grand mean identity.
    /// Standard for GRM: diagonal values represent excess relatedness above
    /// population average. Only effective with --output-egrm.
    #[arg(long = "center-egrm")]
    center_egrm: bool,

    /// Output a diploid GRM by averaging 4 haplotype comparisons per individual pair.
    /// G_dip[i,j] = 1/4 × Σ I(h_iα, h_jβ). Self-identity imputed as 1.0 (T75).
    /// Only effective with --output-egrm.
    #[arg(long = "diploid-egrm")]
    diploid_egrm: bool,

    /// Apply Gower double-centering: G̃ = G - row_means - col_means + grand_mean.
    /// Removes population structure bias; required for REML heritability (T75).
    /// Only effective with --output-egrm --diploid-egrm.
    #[arg(long = "double-center-egrm")]
    double_center_egrm: bool,

    /// Save learned HMM parameters to a JSON file after estimation.
    /// Enables cross-chromosome transfer: learn parameters on one chromosome
    /// (e.g., chr12 with lots of data) and apply to others without re-estimation.
    /// Saves temperature, switch_prob, pairwise_weight, emission_context,
    /// identity_floor, transition matrix (if BW-trained), and initial probs.
    #[arg(long = "save-params")]
    save_params: Option<PathBuf>,

    /// Load pre-learned HMM parameters from a JSON file.
    /// Skips parameter estimation (--estimate-params, --auto-configure) and uses
    /// the loaded values directly. Population names must match. Validated by T68:
    /// cross-chromosome transfer incurs <1pp accuracy loss.
    #[arg(long = "load-params")]
    load_params: Option<PathBuf>,

    /// Run multi-pulse demographic inference from ancestry tract lengths (T80).
    /// Extends established tract-length demographic inference (MultiWaver: Ni et al.
    /// 2018, Heredity; tracts: Gravel 2012) to pangenome-derived ancestry tracts.
    /// Fits shifted exponential mixture via EM with BIC model selection for 1-M
    /// admixture pulses. KS test decides if single-pulse is sufficient.
    /// Reports admixture times in generations.
    #[arg(long = "demographic-inference")]
    demographic_inference: bool,

    /// Recombination rate per bp per generation for demographic inference.
    /// Default: 1e-8 (human genome average). Only used with --demographic-inference.
    #[arg(long = "recomb-rate", default_value = "1e-8", value_parser = validate_positive_f64)]
    recomb_rate: f64,

    /// Maximum number of admixture pulses to test (default: 3).
    /// BIC selects the best-fitting model from 1..max-pulses.
    /// Only used with --demographic-inference.
    #[arg(long = "max-pulses", default_value = "3")]
    max_pulses: usize,

    /// Output file for demographic inference results (TSV).
    /// Writes per-sample and pooled demographic inference to a tab-separated file.
    /// Columns: sample, population, n_tracts, n_pulses, pulse_idx, generations,
    /// proportion, rate, bic, log_likelihood, ks_statistic, ks_rejected.
    /// Implies --demographic-inference.
    #[arg(long = "demographic-output")]
    demographic_output: Option<PathBuf>,

    /// Constrain demographic EM using Baum-Welch transition estimate (T80 §5.2).
    /// Projects mixture rates so Σ π_m λ_m matches the BW-inferred switch rate.
    /// Reduces parameter space by 1 dimension and improves convergence.
    /// Only effective with --demographic-inference.
    #[arg(long = "constrain-demography")]
    constrain_demography: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Post-parse parameter warnings
    if args.baum_welch_iters > 100 {
        eprintln!("Warning: --baum-welch-iters={} is very high. \
                   Values > 100 rarely improve results and slow execution.", args.baum_welch_iters);
    }
    if let Some(t) = args.temperature {
        if t > 10.0 {
            eprintln!("Warning: --temperature={:.2} is very high. \
                       This produces near-uniform posteriors. Typical range: 0.01-1.0.", t);
        }
        if t < 0.001 {
            eprintln!("Warning: --temperature={:.6} is very low. \
                       This produces extremely sharp posteriors. Typical range: 0.01-1.0.", t);
        }
    }
    if args.switch_prob > 0.1 {
        eprintln!("Warning: --switch-prob={:.4} is very high. \
                   This allows frequent ancestry switches. Typical range: 0.0001-0.05.", args.switch_prob);
    }
    if args.min_len_bp > 0 && args.min_len_bp < args.window_size {
        eprintln!("Warning: --min-len-bp={} is less than window size ({}). \
                   Segments shorter than one window cannot be detected.", args.min_len_bp, args.window_size);
    }
    if args.identity_floor > 0.999 {
        eprintln!("Warning: --identity-floor={:.4} is very high and may discard most windows. \
                   Recommended: 0.9 for pangenome data.", args.identity_floor);
    }

    // Set thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // Load populations
    let pop_file = args.populations.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--populations is required (TSV: pop_name, haplotype_id)")
    })?;
    let populations = load_populations(pop_file)?;

    eprintln!("Populations: {:?}", populations.iter().map(|p| &p.name).collect::<Vec<_>>());

    // Load query samples
    let query_samples = load_sample_list(&args.query_samples)?;
    eprintln!("Query samples: {} haplotypes", query_samples.len());

    // Get reference haplotypes from populations
    let reference_haplotypes: Vec<String> = populations.iter()
        .flat_map(|p| p.haplotypes.clone())
        .collect();
    eprintln!("Reference haplotypes: {:?}", reference_haplotypes);

    // Get similarity data: either from pre-computed file or compute via impg
    let mut similarity_data: std::collections::HashMap<String, Vec<AncestryObservation>> = std::collections::HashMap::new();

    if let Some(sim_file) = &args.similarity_file {
        eprintln!("Reading pre-computed similarities from {:?}", sim_file);
        let region_data = if args.coverage_feature {
            load_similarity_file_with_coverage(sim_file, &query_samples, &reference_haplotypes, &args.similarity_column)?
        } else {
            load_similarity_file(sim_file, &query_samples, &reference_haplotypes, &args.similarity_column)?
        };
        for (sample, obs) in region_data {
            similarity_data.entry(sample).or_default().extend(obs);
        }
    } else {
        let ref_name = args.reference.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--reference required without --similarity-file"))?;
        // Parse regions: either single --region or multi-region --bed
        let regions: Vec<(String, u64, u64)> = if let Some(bed_path) = &args.bed {
            parse_bed_regions(bed_path, ref_name)?
        } else {
            let region_str = args.region.as_ref()
                .ok_or_else(|| anyhow::anyhow!("--region or --bed required"))?;
            vec![parse_region(region_str, ref_name, args.region_length)?]
        };

        eprintln!("Processing {} region(s)", regions.len());

        for (region_idx, (chrom, start, end)) in regions.iter().enumerate() {
            eprintln!("Region {}/{}: {}:{}-{} ({:.2} Mb)",
                region_idx + 1, regions.len(), chrom, start, end,
                (*end - *start) as f64 / 1_000_000.0);

            let region_data = compute_similarities(
                args.sequence_files.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--sequence-files required without --similarity-file"))?,
                args.alignment.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("--alignment required without --similarity-file"))?,
                ref_name,
                chrom,
                *start,
                *end,
                args.window_size,
                &query_samples,
                &reference_haplotypes,
            )?;

            for (sample, obs) in region_data {
                similarity_data.entry(sample).or_default().extend(obs);
            }
        }
    }

    eprintln!("Loaded observations for {} samples", similarity_data.len());

    // Output eGRM if requested
    if let Some(egrm_prefix) = &args.output_egrm {
        let sim_file = args.similarity_file.as_ref()
            .context("--output-egrm requires --similarity-file (pre-computed similarities)")?;
        eprintln!("Computing eGRM from all pairwise identities in {:?}", sim_file);
        let acc = parse_similarity_for_egrm(sim_file, &args.similarity_column)?;
        if args.diploid_egrm {
            write_diploid_gcta_grm(&acc, egrm_prefix, args.double_center_egrm)?;
        } else {
            write_gcta_grm(&acc, egrm_prefix, args.center_egrm)?;
        }
    }

    // Load pre-learned parameters if requested
    let loaded_params = if let Some(ref load_path) = args.load_params {
        eprintln!("Loading pre-learned parameters from {:?}...", load_path);
        let mut lp = LearnedParams::load(load_path)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let pop_names: Vec<String> = populations.iter().map(|p| p.name.clone()).collect();
        lp.validate_and_reorder(&pop_names)
            .map_err(|e| anyhow::anyhow!("population mismatch with loaded params: {}", e))?;
        eprintln!("  temperature={:.4}, switch_prob={:.6}, pairwise_weight={:.3}, ec={}, floor={:.2}",
            lp.temperature, lp.switch_prob, lp.pairwise_weight, lp.emission_context, lp.identity_floor);
        if let Some(ref src) = lp.source {
            eprintln!("  source: {}", src);
        }
        Some(lp)
    } else {
        None
    };

    // Auto-estimate identity floor if requested
    let effective_floor = if let Some(ref lp) = loaded_params {
        // Use loaded identity floor (unless CLI explicitly overrides)
        if args.identity_floor > 0.0 {
            args.identity_floor
        } else {
            lp.identity_floor
        }
    } else if args.auto_identity_floor {
        // Pool all observations to estimate the floor
        let all_obs: Vec<&AncestryObservation> = similarity_data.values()
            .flat_map(|v| v.iter())
            .collect();
        let obs_refs: Vec<AncestryObservation> = all_obs.iter().map(|o| (*o).clone()).collect();
        let auto_floor = estimate_identity_floor(&obs_refs);
        eprintln!("Auto-estimated identity floor: {:.6}", auto_floor);
        if args.identity_floor > 0.0 {
            let floor = auto_floor.max(args.identity_floor);
            eprintln!("Using max(auto={:.6}, manual={:.6}) = {:.6}",
                auto_floor, args.identity_floor, floor);
            floor
        } else {
            auto_floor
        }
    } else if args.auto_configure && args.identity_floor == 0.0 {
        // B65 validated: floor=0.9 is the universal optimal for pangenome data.
        // When auto-configure is active and no explicit floor was set, use 0.9.
        eprintln!("Auto-configure: using identity floor 0.9 (B65 validated optimal)");
        0.9
    } else {
        args.identity_floor
    };

    // Apply identity floor filter: remove windows where max similarity is below threshold
    if effective_floor > 0.0 {
        let mut total_before = 0usize;
        let mut total_after = 0usize;
        for obs_vec in similarity_data.values_mut() {
            total_before += obs_vec.len();
            obs_vec.retain(|obs| {
                let max_sim = obs.similarities.values()
                    .cloned()
                    .fold(0.0_f64, f64::max);
                max_sim >= effective_floor
            });
            total_after += obs_vec.len();
        }
        let pct_removed = if total_before > 0 {
            100.0 * (total_before - total_after) as f64 / total_before as f64
        } else {
            0.0
        };
        eprintln!("Identity floor {:.4}: {}/{} windows retained ({:.1}% removed)",
            effective_floor, total_after, total_before, pct_removed);
    }

    // Apply mask-bed filter: remove windows overlapping masked regions
    if let Some(ref mask_path) = args.mask_bed {
        let mask_regions = parse_mask_bed_ancestry(mask_path)?;
        eprintln!("Loaded {} mask regions from {:?}", mask_regions.len(), mask_path);
        let mut total_before = 0usize;
        let mut total_after = 0usize;
        for obs_vec in similarity_data.values_mut() {
            total_before += obs_vec.len();
            obs_vec.retain(|obs| {
                !mask_regions.iter().any(|(chrom, mstart, mend)| {
                    obs.chrom == *chrom && obs.start < *mend && obs.end > *mstart
                })
            });
            total_after += obs_vec.len();
        }
        eprintln!("Mask filter: {}/{} windows retained", total_after, total_before);
    }

    // Parse emission model and decoding method (needed for both load and estimate paths)
    let mut emission_model: EmissionModel = args.emission_model.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;

    // Convert TopK to TopKWeighted if decay is specified
    if args.topk_decay > 0.0 && args.topk_decay < 1.0 {
        if let EmissionModel::TopK(k) = emission_model {
            emission_model = EmissionModel::TopKWeighted(k, args.topk_decay);
        }
    }
    eprintln!("Emission model: {}", emission_model);

    let decoding_method: DecodingMethod = args.decoding.parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;
    eprintln!("Decoding method: {}", decoding_method);

    // Parameter estimation: either load from file or estimate from data
    let (populations, deconvolution_map, auto_pairwise_weight, auto_emission_context, mut params) =
    if let Some(ref lp) = loaded_params {
        // ── Load path: use pre-learned parameters, skip estimation ──
        eprintln!("Using loaded parameters (skipping estimation/auto-configure/BW)");

        let deconvolution_map: Option<std::collections::HashMap<String, String>> = None;
        let auto_pw = Some(lp.pairwise_weight);
        let auto_ec = Some(lp.emission_context);

        let mut params = AncestryHmmParams::new(populations.clone(), lp.switch_prob);
        params.set_temperature(lp.temperature);
        params.set_emission_model(emission_model);
        params.transition_dampening = lp.transition_dampening;

        // Apply loaded transition matrix if available (BW-trained)
        if let Some(ref trans) = lp.transitions {
            if trans.len() == params.n_states {
                params.transitions = trans.clone();
                eprintln!("  Loaded BW-trained transition matrix");
            } else {
                eprintln!("  Warning: loaded transition matrix size {} != {} states, using switch_prob",
                    trans.len(), params.n_states);
            }
        }

        // Apply loaded initial probs if available (BW-full trained)
        if let Some(ref init) = lp.initial_probs {
            if init.len() == params.n_states {
                params.initial = init.clone();
                eprintln!("  Loaded BW-trained initial probs: {:?}",
                    init.iter().map(|p| format!("{:.4}", p)).collect::<Vec<_>>());
            }
        }

        // Enable coverage-ratio auxiliary emission if requested
        if args.coverage_feature {
            params.set_coverage_weight(args.coverage_weight);
        }

        (populations, deconvolution_map, auto_pw, auto_ec, params)
    } else {
        // ── Normal path: estimate parameters from data ──

        // Deconvolve admixed populations if requested or auto-triggered
        let do_deconvolve = args.deconvolve || args.auto_configure;
        let (populations, deconvolution_map) = if do_deconvolve {
            eprintln!("Attempting deconvolution of admixed reference populations (threshold d={:.3})...",
                args.deconvolve_threshold);
            let all_obs: Vec<AncestryObservation> = similarity_data
                .values()
                .flat_map(|v| v.iter().cloned())
                .collect();
            let (new_pops, parent_map) = deconvolve_admixed_populations(
                &all_obs, &populations, args.deconvolve_threshold,
            );
            if new_pops.len() > populations.len() {
                eprintln!("Deconvolution: {} → {} populations", populations.len(), new_pops.len());
                for pop in &new_pops {
                    eprintln!("  {} ({} haplotypes) → parent: {}",
                        pop.name, pop.haplotypes.len(), parent_map[&pop.name]);
                }
            } else {
                eprintln!("Deconvolution: no admixed populations detected");
            }
            (new_pops, Some(parent_map))
        } else {
            (populations, None)
        };

        // Auto-configure: compute optimal pairwise_weight and emission_context from data
        let (auto_pairwise_weight, auto_emission_context) = if args.auto_configure {
            eprintln!("Auto-configuring pairwise parameters from data statistics...");
            let raw_obs: Vec<AncestryObservation> = similarity_data
                .values()
                .flat_map(|v| v.iter().cloned())
                .collect();
            let (pw, ec) = auto_configure_pairwise_params(&raw_obs, &populations);
            (Some(pw), Some(ec))
        } else {
            (None, None)
        };

        // Estimate parameters if requested (auto-configure implies estimate-params)
        let do_estimate = args.estimate_params || args.auto_configure;
        let (temperature, switch_prob) = if do_estimate {
            eprintln!("Estimating HMM parameters from data...");

            // Temperature estimation strategy depends on population count:
            // - 2-pop: smoothed observations match the HMM's input scale, giving
            //   more accurate temperature (no multi-pop boost needed).
            // - 3+ pop: raw observations preserve inter-population distance structure
            //   needed for the multi-pop spread ratio boost. Smoothing compresses
            //   pairwise distances and underestimates the boost for close pairs.
            let n_pops = populations.len();
            let effective_ec = auto_emission_context.unwrap_or(args.emission_context);
            let obs_slice: Vec<AncestryObservation> = if effective_ec > 0 && n_pops < 3 {
                similarity_data.values()
                    .flat_map(|v| smooth_observations(v, effective_ec))
                    .collect()
            } else {
                similarity_data.values()
                    .flat_map(|v| v.iter().cloned())
                    .collect()
            };

            let temp = estimate_temperature_with_spread(&obs_slice, &populations, None);
            let switch = estimate_switch_prob(&obs_slice, &populations, temp);

            eprintln!("  Estimated temperature: {:.4}", temp);
            eprintln!("  Estimated switch probability: {:.6}", switch);

            (temp, switch)
        } else {
            (args.temperature.unwrap_or(0.03), args.switch_prob)
        };

        // Create HMM parameters with estimated or default values
        let mut params = AncestryHmmParams::new(populations.clone(), switch_prob);

        // Apply panel-size temperature scaling if requested
        let mut temperature = if args.auto_temperature_scaling {
            let total_haps: usize = populations.iter().map(|p| p.haplotypes.len()).sum();
            let n_pops = populations.len().max(1);
            let avg_haps = total_haps as f64 / n_pops as f64;
            let scaled = scale_temperature_for_panel(temperature, avg_haps);
            eprintln!("Auto-scaled temperature: {:.4} → {:.4} (avg {:.0} haps/pop)", temperature, scaled, avg_haps);
            scaled
        } else {
            temperature
        };

        // Apply population-count temperature scaling if requested
        if args.population_temperature_scaling {
            let n_pops = populations.len();
            let scaled = scale_temperature_for_populations(temperature, n_pops);
            eprintln!("Population-count temperature scaling: {:.4} → {:.4} ({} populations)", temperature, scaled, n_pops);
            temperature = scaled;
        }

        params.set_temperature(temperature);
        params.set_emission_model(emission_model);

        if args.haplotype_consistency > 0 {
            eprintln!("Haplotype consistency: context=±{}, weight={:.2}", args.haplotype_consistency, args.consistency_weight);
        }

        // Enable coverage-ratio auxiliary emission if requested
        if args.coverage_feature {
            params.set_coverage_weight(args.coverage_weight);
            eprintln!("Coverage-ratio auxiliary emission: weight={:.2}", args.coverage_weight);
        }

        // Learn emission normalization if requested
        if args.normalize_emissions {
            eprintln!("Learning per-population emission normalization...");
            let all_obs: Vec<AncestryObservation> = similarity_data.values()
                .flat_map(|v| v.iter())
                .cloned()
                .collect();
            params.learn_normalization(&all_obs);
            if let Some(ref norm) = params.normalization {
                for (i, pop) in populations.iter().enumerate() {
                    eprintln!("  {}: mean={:.6}, std={:.6}", pop.name, norm.means[i], norm.stds[i]);
                }
            }

            // Always re-estimate temperature in z-score space when normalization is active,
            // since raw temperature values are calibrated for similarity scores (0.85-1.0),
            // not z-scores (centered near 0, spread ~1).
            {
                let all_obs: Vec<AncestryObservation> = similarity_data.values()
                    .flat_map(|v| v.iter())
                    .cloned()
                    .collect();
                let norm_temp = estimate_temperature_normalized(&all_obs, &params);
                eprintln!("  Normalized temperature: {:.4} (adjusted for z-score scale)", norm_temp);
                params.set_temperature(norm_temp);
            }
        }

        // Auto-configure: enable BW with dampening (3 iterations) if not explicitly set
        let effective_bw_iters = if args.baum_welch_iters > 0 {
            args.baum_welch_iters
        } else if args.auto_configure {
            3 // Conservative BW with dampening for auto-configure
        } else {
            0
        };

        // Baum-Welch parameter training if requested
        if effective_bw_iters > 0 {
            // T52: set transition dampening (auto-default when pairwise + BW)
            let effective_dampening = if args.bw_dampening > 0.0 {
                args.bw_dampening
            } else if args.auto_configure || args.pairwise_emissions {
                // Auto-dampening when pairwise emissions are active to prevent
                // double-correction of weakly-discriminable population pairs
                0.5
            } else {
                0.0
            };
            params.transition_dampening = effective_dampening;
            if effective_dampening > 0.0 {
                eprintln!("BW transition dampening: {:.2} (T52 anti-double-correction)", effective_dampening);
            }

            let obs_seqs: Vec<Vec<AncestryObservation>> = similarity_data.values().cloned().collect();
            let obs_refs: Vec<&[AncestryObservation]> = obs_seqs.iter().map(|v| v.as_slice()).collect();

            let ll = if args.baum_welch_full || args.auto_configure {
                eprintln!("Running full Baum-Welch ({} iters, initial probs{})...",
                    effective_bw_iters,
                    if args.baum_welch_temperature { " + temperature" } else { "" });
                params.baum_welch_full(
                    &obs_refs,
                    effective_bw_iters,
                    1e-4,
                    args.baum_welch_temperature,
                )
            } else {
                eprintln!("Running Baum-Welch parameter training ({} iterations)...", effective_bw_iters);
                params.baum_welch(&obs_refs, effective_bw_iters, 1e-4)
            };

            // Report trained parameters
            if params.n_states > 2 {
                eprintln!("  Trained transition matrix (asymmetric):");
                for (i, row) in params.transitions.iter().enumerate() {
                    let probs: Vec<String> = row.iter().enumerate().map(|(j, &p)| {
                        format!("{}→{}: {:.6}", populations[i].name, populations[j].name, p)
                    }).collect();
                    eprintln!("    {}", probs.join("  "));
                }
            } else {
                let trained_switch = 1.0 - params.transitions[0][0];
                eprintln!("  Trained switch probability: {:.6}", trained_switch);
            }
            if args.baum_welch_full || args.auto_configure {
                eprintln!("  Trained initial probs: {:?}", params.initial.iter()
                    .map(|p| format!("{:.4}", p)).collect::<Vec<_>>());
                if args.baum_welch_temperature {
                    eprintln!("  Trained temperature: {:.6}", params.emission_std);
                }
            }
            eprintln!("  Final log-likelihood: {:.2}", ll);
        }

        (populations, deconvolution_map, auto_pairwise_weight, auto_emission_context, params)
    };

    // Save learned parameters if requested
    if let Some(ref save_path) = args.save_params {
        let pop_names: Vec<String> = populations.iter().map(|p| p.name.clone()).collect();
        let effective_ec = auto_emission_context.unwrap_or(args.emission_context);
        let effective_pw = auto_pairwise_weight.unwrap_or(args.pairwise_weight);

        let mut learned = LearnedParams::new(
            pop_names,
            params.emission_std,  // final temperature
            1.0 - params.transitions[0][0],  // switch_prob from diagonal
            effective_pw,
            effective_ec,
            effective_floor,
            params.transition_dampening,
        );

        // Save BW-trained transitions if they differ from symmetric
        let is_symmetric = params.n_states > 2 && params.transitions.iter().enumerate().any(|(i, row)| {
            row.iter().enumerate().any(|(j, &p)| {
                i != j && (p - params.transitions[j][i]).abs() > 1e-10
            })
        });
        if is_symmetric {
            learned.transitions = Some(params.transitions.clone());
        }

        // Save BW-trained initial probs if non-uniform
        let uniform = 1.0 / params.n_states as f64;
        let non_uniform = params.initial.iter().any(|&p| (p - uniform).abs() > 1e-6);
        if non_uniform {
            learned.initial_probs = Some(params.initial.clone());
        }

        // Set source region info
        let first_chrom = similarity_data.values()
            .flat_map(|v| v.first())
            .next()
            .map(|obs| obs.chrom.clone());
        learned.source = first_chrom;

        learned.save(std::path::Path::new(save_path))
            .map_err(|e| anyhow::anyhow!("failed to save params to {:?}: {}", save_path, e))?;
        eprintln!("Saved learned parameters to {:?}", save_path);
    }

    // Load genetic map if provided
    let genetic_map: Option<AncestryGeneticMap> = if let Some(gmap_path) = &args.genetic_map {
        // Extract chromosome from first observation's window
        let first_chrom = similarity_data.values()
            .flat_map(|v| v.first())
            .next()
            .map(|obs| obs.chrom.clone())
            .unwrap_or_default();
        let chrom = &first_chrom;
        // Strip reference prefix to get just the chromosome name
        let chrom_name = chrom.split('#').next_back().unwrap_or(chrom);
        eprintln!("Loading genetic map from {:?} for {}", gmap_path, chrom_name);
        match AncestryGeneticMap::from_file(gmap_path, chrom_name) {
            Ok(gmap) => {
                eprintln!("  Loaded genetic map");
                Some(gmap)
            }
            Err(e) => {
                eprintln!("  WARNING: Failed to load genetic map: {}. Using fixed transitions.", e);
                None
            }
        }
    } else {
        None
    };

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
    let two_pass = args.two_pass;
    let emission_context = auto_emission_context.unwrap_or(args.emission_context);
    let weighted_context = args.weighted_context;
    let contrast_normalize = args.contrast_normalize;
    let dampen_emissions = args.dampen_emissions;
    let dampen_scale = args.dampen_scale;
    let learn_profiles = args.learn_profiles && two_pass; // profiles require two-pass
    let profile_weight = args.profile_weight;
    let haplotype_consistency = args.haplotype_consistency;
    let consistency_weight = args.consistency_weight;
    let use_copying_model = args.copying_model;
    let copying_switch_rate = args.copying_switch_rate;
    let copying_ancestry_frac = args.copying_ancestry_frac;
    let copying_em_iters = args.copying_em_iters;
    let adaptive_context = args.adaptive_context;
    let distance_transitions = args.distance_transitions;
    let ensemble_size = args.ensemble_size;
    let ensemble_scale = args.ensemble_scale;
    let rank_emissions = args.rank_emissions;
    let rank_topk = args.rank_topk;
    let rank_weight = args.rank_weight;
    let hierarchical_emissions = args.hierarchical_emissions;
    let group_weight = args.group_weight;
    let consistency_emissions = args.consistency_emissions;
    let consistency_context = if args.consistency_context > 0 {
        args.consistency_context
    } else if emission_context > 0 {
        emission_context
    } else {
        10
    };
    let consistency_emission_weight = args.consistency_emission_weight;
    let pairwise_emissions = args.pairwise_emissions || args.auto_configure;
    let pairwise_weight = auto_pairwise_weight.unwrap_or(args.pairwise_weight);
    let adaptive_pairwise = args.adaptive_pairwise;
    // Auto-configure: enable agreement-pairwise + per-population-agreement for K>=3
    // G86: pairwise hurts admixed panels without agreement gating; T79: per-pop scaling
    // mitigates by giving near-neutral weight to poorly separated populations
    let agreement_pairwise = args.agreement_pairwise
        || (args.auto_configure && populations.len() >= 3
            && !args.hybrid_pairwise && !args.adaptive_pairwise);
    let hybrid_pairwise = args.hybrid_pairwise;
    let agreement_agree_scale = args.agreement_agree_scale;
    let agreement_disagree_scale = args.agreement_disagree_scale;
    let hybrid_margin_clamp_lo = args.hybrid_margin_clamp_lo;
    let hybrid_margin_clamp_hi = args.hybrid_margin_clamp_hi;
    let per_population_agreement = args.per_population_agreement
        || (args.auto_configure && populations.len() >= 3);
    if args.auto_configure && populations.len() >= 3
        && !args.agreement_pairwise && !args.hybrid_pairwise && !args.adaptive_pairwise
    {
        eprintln!("  Auto-configure: K={} ≥ 3 → enabling agreement-pairwise + per-population-agreement (T79/G86)",
            populations.len());
    }
    let population_transitions = args.population_transitions;

    // Parse or auto-detect population groups for hierarchical emissions
    let population_groups: Option<Vec<Vec<usize>>> = if hierarchical_emissions {
        if let Some(ref spec) = args.population_groups {
            match parse_population_groups(spec, &populations) {
                Some(groups) => {
                    eprintln!("Hierarchical emissions with {} groups:", groups.len());
                    for group in &groups {
                        let names: Vec<_> = group.iter()
                            .map(|&i| populations[i].name.as_str())
                            .collect();
                        eprintln!("  [{}]", names.join(", "));
                    }
                    Some(groups)
                }
                None => {
                    eprintln!("Warning: --population-groups contains unknown population names. \
                               Auto-detecting groups.");
                    None
                }
            }
        } else {
            None // Will auto-detect per sample
        }
    } else {
        None
    };
    if args.learn_profiles && !args.two_pass {
        eprintln!("Warning: --learn-profiles requires --two-pass. Profiles will not be used.");
    }
    if use_copying_model {
        eprintln!("Using haplotype copying model (Li & Stephens inspired)");
        if two_pass {
            eprintln!("Warning: --two-pass is not used with --copying-model. \
                       The copying model handles haplotype continuity natively.");
        }
    }
    // Apply population-similarity-aware transitions if requested
    if population_transitions && params.n_states >= 2 {
        let all_obs: Vec<&AncestryObservation> = similarity_data.values()
            .flat_map(|v| v.iter()).collect();
        let obs_vec: Vec<AncestryObservation> = all_obs.iter().map(|o| (*o).clone()).collect();
        let switch_prob = 1.0 - params.transitions[0][0];
        let pop_trans = compute_population_aware_transitions(
            &obs_vec, &populations, &params.emission_model, switch_prob);
        // Convert from log to probability for params.transitions
        for (i, row) in pop_trans.iter().enumerate() {
            for (j, &lp) in row.iter().enumerate() {
                params.transitions[i][j] = lp.exp();
            }
        }
        eprintln!("Population-similarity-aware transitions:");
        for (i, row) in params.transitions.iter().enumerate() {
            let probs: Vec<String> = row.iter().enumerate().map(|(j, &p)| {
                format!("  {}→{}: {:.6}", populations[i].name, populations[j].name, p)
            }).collect();
            eprintln!("{}", probs.join(""));
        }
    }

    // Compute per-population agreement scales if requested (T79)
    let per_pop_scales: Option<PerPopAgreementScales> = if per_population_agreement
        && pairwise_emissions && (agreement_pairwise || hybrid_pairwise)
    {
        let raw_obs: Vec<AncestryObservation> = similarity_data
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect();
        Some(compute_per_pop_agreement_scales(
            &raw_obs, &populations, agreement_agree_scale, agreement_disagree_scale,
        ))
    } else {
        None
    };

    // Compute heteroscedastic per-population temperatures if requested
    let hetero_temperatures: Option<Vec<f64>> = if args.heteroscedastic_gamma > 0.0 {
        let raw_obs: Vec<AncestryObservation> = similarity_data
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect();
        let variances = compute_population_variances(&raw_obs, &populations, &emission_model);
        let base_temp = params.emission_std;
        let temps = compute_heteroscedastic_temperatures(
            &variances, base_temp, args.heteroscedastic_gamma);
        eprintln!("Heteroscedastic temperatures (gamma={:.2}):", args.heteroscedastic_gamma);
        for (i, (pop, temp)) in populations.iter().zip(temps.iter()).enumerate() {
            eprintln!("  {}: T={:.6} (var={:.8}, base={:.6})", pop.name, temp, variances[i], base_temp);
        }
        Some(temps)
    } else {
        None
    };

    // Compute reference purity scores if requested
    let purity_scores = if args.purity_weight_gamma > 0.0 {
        let all_obs: Vec<AncestryObservation> = similarity_data.values()
            .flat_map(|v| v.iter().cloned()).collect();
        let scores = compute_reference_purity(&all_obs, &populations);
        eprintln!("Reference purity scores (gamma={:.2}):", args.purity_weight_gamma);
        for pop in &populations {
            let avg_purity: f64 = pop.haplotypes.iter()
                .filter_map(|h| scores.get(h))
                .sum::<f64>() / pop.haplotypes.len().max(1) as f64;
            eprintln!("  {}: avg purity = {:.4}", pop.name, avg_purity);
        }
        Some(scores)
    } else {
        None
    };

    // Apply purity weighting if active
    let purity_weighted_data = if let Some(ref scores) = purity_scores {
        let gamma = args.purity_weight_gamma;
        similarity_data.iter()
            .map(|(sample, obs)| {
                (sample.clone(), apply_purity_weighted_observations(obs, scores, gamma))
            })
            .collect::<HashMap<_, _>>()
    } else {
        HashMap::new()
    };
    let effective_data = if purity_scores.is_some() { &purity_weighted_data } else { &similarity_data };

    let results: Vec<_> = effective_data.par_iter()
        .map(|(sample, observations)| {
            if observations.len() < 3 {
                eprintln!("  {} - too few observations ({})", sample, observations.len());
                return (sample.clone(), Vec::new(), Vec::new());
            }

            // Compute per-sample emission context (adaptive if requested)
            let effective_ec = if adaptive_context && emission_context > 0 {
                let ec = estimate_emission_context(
                    observations, &populations, emission_context, 1, 15);
                if ec != emission_context {
                    eprintln!("  {} — adaptive ec: {} (base: {})", sample, ec, emission_context);
                }
                ec
            } else {
                emission_context
            };

            // Helper: run single-pass HMM inference with given params
            let run_inference = |obs: &[AncestryObservation], p: &AncestryHmmParams| -> (Vec<Vec<f64>>, Vec<usize>) {
                // Step 1: Smooth raw similarities if emission_context > 0
                // This smooths per-haplotype similarity values BEFORE softmax,
                // which is mathematically superior to smoothing log-emissions.
                let mut processed_obs;
                let effective_obs = if effective_ec > 0 || haplotype_consistency > 0 {
                    processed_obs = if effective_ec > 0 {
                        smooth_observations(obs, effective_ec)
                    } else {
                        obs.to_vec()
                    };
                    // Step 1b: Apply haplotype consistency bonus
                    if haplotype_consistency > 0 {
                        apply_haplotype_consistency(
                            &mut processed_obs, p, haplotype_consistency, consistency_weight);
                    }
                    &processed_obs[..]
                } else {
                    obs
                };

                // Step 2: Optionally precompute and further process log emissions
                if contrast_normalize || weighted_context || dampen_emissions || rank_emissions
                    || hierarchical_emissions || consistency_emissions || pairwise_emissions
                {
                    // For contrast normalization, weighted context, dampening, rank, or
                    // hierarchical blending, we need the log-emission matrix for post-processing
                    let raw_emissions = if hierarchical_emissions {
                        // Auto-detect or use provided groups
                        let groups = population_groups.clone().unwrap_or_else(|| {
                            auto_detect_groups(effective_obs, &p.populations, &p.emission_model)
                        });
                        compute_hierarchical_emissions(
                            effective_obs, &p.populations, &groups,
                            &p.emission_model, p.emission_std, group_weight)
                    } else if let Some(ref ht) = hetero_temperatures {
                        precompute_heteroscedastic_log_emissions(effective_obs, p, ht)
                    } else {
                        precompute_log_emissions(effective_obs, p)
                    };

                    let emissions = if contrast_normalize {
                        contrast_normalize_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };

                    // Apply emission dampening for low-discriminability windows
                    let emissions = if dampen_emissions {
                        dampen_low_confidence_emissions(&emissions, dampen_scale)
                    } else {
                        emissions
                    };

                    // Blend with rank-based emissions
                    let emissions = if rank_emissions {
                        let rank_emis = compute_rank_log_emissions(
                            effective_obs, &p.populations, rank_topk);
                        blend_log_emissions(&emissions, &rank_emis, rank_weight)
                    } else {
                        emissions
                    };

                    // Blend with consistency-based emissions (computed on RAW observations
                    // for independence from the smoothed softmax emissions above)
                    let emissions = if consistency_emissions {
                        let cons_emis = compute_consistency_log_emissions(
                            obs, &p.populations, &p.emission_model, consistency_context);
                        blend_log_emissions(&emissions, &cons_emis, consistency_emission_weight)
                    } else {
                        emissions
                    };

                    // Blend with pairwise contrast emissions (per-pair adaptive temperature)
                    let emissions = if pairwise_emissions {
                        let pw_emis = compute_pairwise_log_emissions(
                            effective_obs, &p.populations, &p.emission_model);
                        if hybrid_pairwise {
                            if let Some(ref scales) = per_pop_scales {
                                blend_log_emissions_per_pop_hybrid(
                                    &emissions, &pw_emis, pairwise_weight,
                                    scales, hybrid_margin_clamp_lo, hybrid_margin_clamp_hi)
                            } else {
                                blend_log_emissions_hybrid(
                                    &emissions, &pw_emis, pairwise_weight,
                                    agreement_agree_scale, agreement_disagree_scale,
                                    hybrid_margin_clamp_lo, hybrid_margin_clamp_hi)
                            }
                        } else if agreement_pairwise {
                            if let Some(ref scales) = per_pop_scales {
                                blend_log_emissions_per_pop_agreement(
                                    &emissions, &pw_emis, pairwise_weight, scales)
                            } else {
                                blend_log_emissions_agreement(
                                    &emissions, &pw_emis, pairwise_weight,
                                    agreement_agree_scale, agreement_disagree_scale)
                            }
                        } else if adaptive_pairwise {
                            blend_log_emissions_adaptive(&emissions, &pw_emis, pairwise_weight)
                        } else {
                            blend_log_emissions(&emissions, &pw_emis, pairwise_weight)
                        }
                    } else {
                        emissions
                    };

                    // Apply weighted log-emission smoothing as additional refinement
                    let final_emissions = if weighted_context && effective_ec > 0 {
                        smooth_log_emissions_weighted(&emissions, effective_ec)
                    } else {
                        emissions
                    };

                    // Ensemble decoding (if enabled, overrides standard FB+decode)
                    if ensemble_size > 1 {
                        let (posteriors, states) = ensemble_decode(
                            &final_emissions, p, ensemble_size, ensemble_scale);
                        return (posteriors, states);
                    }

                    // Compute per-window transitions if distance-aware
                    let dist_trans = if distance_transitions && genetic_map.is_none() {
                        Some(compute_distance_transitions(
                            effective_obs, p, args.window_size))
                    } else {
                        None
                    };

                    let posteriors = if let Some(ref gmap) = genetic_map {
                        forward_backward_from_log_emissions_with_genetic_map(
                            effective_obs, &final_emissions, p, gmap)
                    } else if let Some(ref dt) = dist_trans {
                        forward_backward_from_log_emissions_with_transitions(
                            &final_emissions, p, dt)
                    } else {
                        forward_backward_from_log_emissions(&final_emissions, p)
                    };

                    let states = match (&genetic_map, decoding_method) {
                        (Some(ref gmap), DecodingMethod::Viterbi) =>
                            viterbi_from_log_emissions_with_genetic_map(
                                effective_obs, &final_emissions, p, gmap),
                        (_, DecodingMethod::Posterior) => {
                            posteriors.iter()
                                .map(|probs| probs.iter().enumerate()
                                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                    .map(|(idx, _)| idx).unwrap_or(0))
                                .collect()
                        }
                        (_, DecodingMethod::Mpel) =>
                            mpel_decode_from_posteriors(&posteriors, p),
                        (None, DecodingMethod::Viterbi) if dist_trans.is_some() =>
                            viterbi_from_log_emissions_with_transitions(
                                &final_emissions, p, dist_trans.as_ref().unwrap()),
                        (None, DecodingMethod::Viterbi) =>
                            viterbi_from_log_emissions(&final_emissions, p),
                    };

                    (posteriors, states)
                } else {
                    // No contrast normalization or weighted context — run directly
                    // on smoothed observations (or original if context=0)
                    let posteriors = if let Some(ref gmap) = genetic_map {
                        forward_backward_with_genetic_map(effective_obs, p, gmap)
                    } else {
                        forward_backward(effective_obs, p)
                    };

                    let states = match (&genetic_map, decoding_method) {
                        (Some(ref gmap), DecodingMethod::Viterbi) =>
                            viterbi_with_genetic_map(effective_obs, p, gmap),
                        (Some(ref gmap), DecodingMethod::Posterior) =>
                            posterior_decode_with_genetic_map(effective_obs, p, gmap),
                        (_, DecodingMethod::Mpel) =>
                            mpel_decode_from_posteriors(&posteriors, p),
                        (None, DecodingMethod::Viterbi) =>
                            viterbi(effective_obs, p),
                        (None, DecodingMethod::Posterior) =>
                            posterior_decode(effective_obs, p),
                    };

                    (posteriors, states)
                }
            };

            // Run inference — optionally two-pass
            let posterior_feedback_lambda = args.posterior_feedback;
            let (posteriors, states) = if two_pass {
                // Pass 1: standard inference with uniform priors
                let (pass1_posteriors, pass1_states) = run_inference(observations, &params);

                // Estimate per-sample ancestry proportions from pass 1 assignments
                let proportions = estimate_proportions_from_states(&pass1_states, params.n_states);
                let switch_rates = estimate_per_state_switch_rates(&pass1_states, params.n_states);

                // Pass 2: refined params with proportion-based priors and transitions
                let mut pass2_params = params.clone();
                pass2_params.set_initial_probs(&proportions);
                if args.pop_distance_transitions {
                    let pop_dists = compute_population_distances(
                        observations, &params.populations, &params.emission_model);
                    set_distance_weighted_transitions(
                        &mut pass2_params, &pop_dists, &proportions, &switch_rates);
                } else {
                    pass2_params.set_proportional_transitions(&proportions, &switch_rates);
                }

                if learn_profiles {
                    // Smooth observations for pass 2 if requested
                    let mut processed_obs2;
                    let effective_obs2 = if effective_ec > 0 || haplotype_consistency > 0 {
                        processed_obs2 = if effective_ec > 0 {
                            smooth_observations(observations, effective_ec)
                        } else {
                            observations.to_vec()
                        };
                        if haplotype_consistency > 0 {
                            apply_haplotype_consistency(
                                &mut processed_obs2, &pass2_params, haplotype_consistency, consistency_weight);
                        }
                        &processed_obs2[..]
                    } else {
                        observations
                    };

                    // Learn emission profiles from pass 1 assignments
                    let profiles = learn_population_profiles(observations, &pass1_states, &params);

                    // Pass 2 with profile-blended emissions (always precompute)
                    let raw_emissions = if hierarchical_emissions {
                        let groups = population_groups.clone().unwrap_or_else(|| {
                            auto_detect_groups(effective_obs2, &pass2_params.populations,
                                &pass2_params.emission_model)
                        });
                        compute_hierarchical_emissions(
                            effective_obs2, &pass2_params.populations, &groups,
                            &pass2_params.emission_model, pass2_params.emission_std, group_weight)
                    } else if let Some(ref ht) = hetero_temperatures {
                        precompute_heteroscedastic_log_emissions(effective_obs2, &pass2_params, ht)
                    } else {
                        precompute_log_emissions(effective_obs2, &pass2_params)
                    };
                    // Apply emission whitening if requested
                    let raw_emissions = if args.whiten_emissions {
                        whiten_log_emissions(&raw_emissions, 1e-6)
                    } else {
                        raw_emissions
                    };
                    // Apply rank transform if requested
                    let raw_emissions = if args.rank_transform {
                        rank_transform_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply outlier dampening if requested
                    let raw_emissions = if args.outlier_dampen > 0.0 {
                        dampen_emission_outliers(&raw_emissions, args.outlier_dampen)
                    } else {
                        raw_emissions
                    };
                    // Apply emission momentum if requested
                    let raw_emissions = if args.emission_momentum > 0.0 {
                        apply_emission_momentum(&raw_emissions, args.emission_momentum)
                    } else {
                        raw_emissions
                    };
                    // Apply pop temperature adjustment if requested
                    let raw_emissions = if args.pop_temp_adjust > 0.0 {
                        adjust_pop_temperatures(&raw_emissions, &pass1_posteriors,
                            args.pop_temp_adjust)
                    } else {
                        raw_emissions
                    };
                    // Apply SNR weighting if requested
                    let raw_emissions = if args.snr_weight > 0.0 {
                        apply_snr_weighting(&raw_emissions, args.snr_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply emission detrending if requested
                    let raw_emissions = if args.detrend_emissions {
                        detrend_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply emission variance stabilization if requested
                    let raw_emissions = if args.variance_stabilize {
                        variance_stabilize_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply kurtosis weighting if requested
                    let raw_emissions = if args.kurtosis_weight > 0.0 {
                        apply_kurtosis_weighting(&raw_emissions, args.kurtosis_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply emission gap penalty if requested
                    let raw_emissions = if args.gap_penalty > 0.0 {
                        apply_gap_penalty(&raw_emissions, args.gap_penalty)
                    } else {
                        raw_emissions
                    };
                    // Apply emission centering if requested
                    let raw_emissions = if args.center_emissions {
                        center_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply emission median polish if requested
                    let raw_emissions = if args.median_polish {
                        median_polish_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply softmax renormalization if requested
                    let raw_emissions = if args.softmax_renorm > 0.0 {
                        softmax_renormalize(&raw_emissions, args.softmax_renorm)
                    } else {
                        raw_emissions
                    };
                    // Apply windowed normalization if requested
                    let raw_emissions = if args.window_normalize > 0 {
                        apply_windowed_normalization(&raw_emissions, args.window_normalize)
                    } else {
                        raw_emissions
                    };
                    // Apply quantile normalization if requested
                    let raw_emissions = if args.quantile_normalize {
                        quantile_normalize_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply local emission reranking if requested
                    let raw_emissions = if args.local_rerank > 0 {
                        local_rerank_emissions(&raw_emissions, args.local_rerank)
                    } else {
                        raw_emissions
                    };
                    // Apply Bayesian emission shrinkage if requested
                    let raw_emissions = if args.bayesian_shrink > 0.0 {
                        bayesian_shrink_emissions(&raw_emissions, args.bayesian_shrink)
                    } else {
                        raw_emissions
                    };
                    // Apply top-K emission sparsification if requested
                    let raw_emissions = if args.sparsify_top > 0 {
                        sparsify_top_k_emissions(&raw_emissions, args.sparsify_top, -100.0)
                    } else {
                        raw_emissions
                    };
                    // Apply population proportion prior if requested
                    let raw_emissions = if args.proportion_prior > 0.0 {
                        let proportions = estimate_proportions_from_states(
                            &pass1_states, pass2_params.n_states);
                        apply_proportion_prior(&raw_emissions, &proportions,
                            args.proportion_prior)
                    } else {
                        raw_emissions
                    };
                    // Apply emission confidence weighting if requested
                    let raw_emissions = if args.confidence_weight > 0.0 {
                        apply_confidence_weighting(&raw_emissions, args.confidence_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply forward-backward temperature if requested
                    let raw_emissions = if (args.fb_temperature - 1.0).abs() > 1e-6 {
                        apply_fb_temperature(&raw_emissions, args.fb_temperature)
                    } else {
                        raw_emissions
                    };
                    // Apply state persistence bonus if requested
                    let raw_emissions = if args.persistence_bonus > 0.0 {
                        apply_persistence_bonus(&raw_emissions, &pass1_states,
                            args.persistence_bonus)
                    } else {
                        raw_emissions
                    };

                    let profile_emissions = compute_profile_log_emissions(
                        effective_obs2, &pass2_params, &profiles);

                    // Blend standard + profile emissions
                    let blended = blend_log_emissions(&raw_emissions, &profile_emissions, profile_weight);

                    // Apply contrast normalization if requested
                    let emissions = if contrast_normalize {
                        contrast_normalize_emissions(&blended)
                    } else {
                        blended
                    };

                    // Apply emission dampening if requested
                    let emissions = if dampen_emissions {
                        dampen_low_confidence_emissions(&emissions, dampen_scale)
                    } else {
                        emissions
                    };

                    // Blend with rank-based emissions if requested
                    let emissions = if rank_emissions {
                        let rank_emis = compute_rank_log_emissions(
                            effective_obs2, &pass2_params.populations, rank_topk);
                        blend_log_emissions(&emissions, &rank_emis, rank_weight)
                    } else {
                        emissions
                    };

                    // Blend with consistency-based emissions (on raw observations)
                    let emissions = if consistency_emissions {
                        let cons_emis = compute_consistency_log_emissions(
                            observations, &pass2_params.populations,
                            &pass2_params.emission_model, consistency_context);
                        blend_log_emissions(&emissions, &cons_emis, consistency_emission_weight)
                    } else {
                        emissions
                    };

                    // Blend with pairwise contrast emissions
                    let emissions = if pairwise_emissions {
                        let pw_emis = compute_pairwise_log_emissions(
                            effective_obs2, &pass2_params.populations, &pass2_params.emission_model);
                        if hybrid_pairwise {
                            if let Some(ref scales) = per_pop_scales {
                                blend_log_emissions_per_pop_hybrid(
                                    &emissions, &pw_emis, pairwise_weight,
                                    scales, hybrid_margin_clamp_lo, hybrid_margin_clamp_hi)
                            } else {
                                blend_log_emissions_hybrid(
                                    &emissions, &pw_emis, pairwise_weight,
                                    agreement_agree_scale, agreement_disagree_scale,
                                    hybrid_margin_clamp_lo, hybrid_margin_clamp_hi)
                            }
                        } else if agreement_pairwise {
                            if let Some(ref scales) = per_pop_scales {
                                blend_log_emissions_per_pop_agreement(
                                    &emissions, &pw_emis, pairwise_weight, scales)
                            } else {
                                blend_log_emissions_agreement(
                                    &emissions, &pw_emis, pairwise_weight,
                                    agreement_agree_scale, agreement_disagree_scale)
                            }
                        } else if adaptive_pairwise {
                            blend_log_emissions_adaptive(&emissions, &pw_emis, pairwise_weight)
                        } else {
                            blend_log_emissions(&emissions, &pw_emis, pairwise_weight)
                        }
                    } else {
                        emissions
                    };

                    // Apply posterior feedback if requested
                    let emissions = if posterior_feedback_lambda > 0.0 {
                        apply_posterior_feedback(&emissions, &pass1_posteriors, posterior_feedback_lambda)
                    } else {
                        emissions
                    };

                    // Apply focused population masking if requested
                    let emissions = if args.focused_pass2_threshold > 0.0 {
                        apply_focused_masking(&emissions, &pass1_posteriors,
                            args.focused_pass2_threshold, 2)
                    } else {
                        emissions
                    };

                    // Apply label smoothing if requested
                    let emissions = if args.label_smooth_alpha > 0.0 {
                        apply_label_smoothing(&emissions, args.label_smooth_alpha)
                    } else {
                        emissions
                    };
                    // Apply cross-entropy regularization if requested
                    let emissions = if args.cross_entropy_reg > 0.0 {
                        regularize_toward_posteriors(&emissions, &pass1_posteriors,
                            args.cross_entropy_reg)
                    } else {
                        emissions
                    };
                    // Apply emission floor if requested
                    let emissions = if args.emission_floor < 0.0 {
                        apply_emission_floor(&emissions, args.emission_floor)
                    } else {
                        emissions
                    };

                    // Apply diversity scaling if requested
                    let emissions = if args.diversity_amplify > 0.0 {
                        apply_diversity_scaling(&emissions,
                            args.diversity_amplify, args.diversity_dampen)
                    } else {
                        emissions
                    };
                    // Apply residual amplification if requested
                    let emissions = if args.residual_amplify > 0.0 {
                        amplify_emission_residuals(&emissions, args.residual_amplify)
                    } else {
                        emissions
                    };
                    // Apply emission anchor boost if requested
                    let emissions = if args.anchor_boost_radius > 0 {
                        apply_emission_anchor_boost(&emissions,
                            args.anchor_boost_radius, args.anchor_boost_threshold,
                            args.anchor_boost_strength)
                    } else {
                        emissions
                    };
                    // Apply gradient penalty if requested
                    let emissions = if args.gradient_penalty > 0.0 {
                        apply_gradient_penalty(&emissions, args.gradient_penalty)
                    } else {
                        emissions
                    };
                    // Apply changepoint prior if requested
                    let emissions = if args.changepoint_prior > 0.0 {
                        apply_changepoint_prior(&emissions, &pass1_states,
                            args.changepoint_prior)
                    } else {
                        emissions
                    };
                    // Apply pairwise emission contrast if requested
                    let emissions = if args.pairwise_contrast > 0.0 {
                        apply_pairwise_emission_contrast(&emissions,
                            args.pairwise_contrast)
                    } else {
                        emissions
                    };

                    // Apply margin-gated persistence if requested
                    let emissions = if args.margin_persistence_bonus > 0.0 {
                        apply_margin_persistence(&emissions, &pass1_posteriors,
                            args.margin_persistence_threshold, args.margin_persistence_bonus)
                    } else {
                        emissions
                    };

                    // Apply within-population variance penalty if requested
                    let emissions = if args.variance_penalty > 0.0 {
                        let variances = compute_within_pop_variance(
                            effective_obs2, &pass2_params.populations);
                        apply_variance_penalty(&emissions, &variances, args.variance_penalty)
                    } else {
                        emissions
                    };

                    // Apply flank-informed emission bonus if requested
                    let emissions = if args.flank_inform_radius > 0 {
                        apply_flank_informed_bonus(&emissions, &pass1_states,
                            args.flank_inform_radius, args.flank_inform_bonus,
                            pass2_params.n_states)
                    } else {
                        emissions
                    };

                    // Run HMM on blended emissions
                    let posteriors = if args.confusion_penalty > 0.0 {
                        let penalties = compute_confusion_penalties(
                            &pass1_states, pass2_params.n_states, args.confusion_penalty);
                        let log_trans = apply_confusion_penalties(&pass2_params, &penalties);
                        let n_windows = emissions.len();
                        let per_window_trans: Vec<Vec<Vec<f64>>> = vec![log_trans; n_windows];
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.adaptive_transition_scale > 0.0 {
                        let per_window_trans = compute_adaptive_transitions(
                            &emissions, &pass2_params, args.adaptive_transition_scale);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.boundary_boost > 0.0 {
                        let per_window_trans = compute_boundary_boost_transitions(
                            &pass1_states, &pass2_params, args.boundary_boost);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.cooccurrence_bonus > 0.0 {
                        let log_trans = compute_cooccurrence_transitions(
                            &pass1_states, &pass2_params, args.cooccurrence_bonus);
                        let n_windows = emissions.len();
                        let per_window_trans: Vec<Vec<Vec<f64>>> = vec![log_trans; n_windows];
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.transition_momentum > 0.0 {
                        let per_window_trans = compute_transition_momentum(
                            &pass1_states, &pass2_params, args.transition_momentum);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.lookahead_transitions > 0 {
                        let per_window_trans = compute_lookahead_transitions(
                            &emissions, &pass1_states, &pass2_params,
                            args.lookahead_transitions);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.segment_length_prior > 1 {
                        let per_window_trans = compute_segment_length_prior(
                            &pass1_states, &pass2_params, args.segment_length_prior);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.recency_transitions > 0.0 {
                        let per_window_trans = compute_recency_transitions(
                            &pass1_states, &pass2_params, args.recency_transitions);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if args.disagreement_penalty > 0.0 {
                        let per_window_trans = compute_disagreement_transitions(
                            &emissions, &pass1_states, &pass2_params,
                            args.disagreement_penalty);
                        forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans)
                    } else if let Some(ref gmap) = genetic_map {
                        forward_backward_from_log_emissions_with_genetic_map(
                            effective_obs2, &emissions, &pass2_params, gmap)
                    } else {
                        forward_backward_from_log_emissions(&emissions, &pass2_params)
                    };

                    // Apply posterior sharpening if requested
                    let posteriors = if args.sharpen_posteriors_temp > 0.0 {
                        sharpen_posteriors(&posteriors, args.sharpen_posteriors_temp)
                    } else {
                        posteriors
                    };

                    // Apply entropy-weighted posterior smoothing if requested
                    let posteriors = if args.entropy_smooth_radius > 0 {
                        entropy_smooth_posteriors(&posteriors, args.entropy_smooth_radius)
                    } else {
                        posteriors
                    };

                    let states = match (&genetic_map, decoding_method) {
                        (Some(ref gmap), DecodingMethod::Viterbi) =>
                            viterbi_from_log_emissions_with_genetic_map(
                                effective_obs2, &emissions, &pass2_params, gmap),
                        (_, DecodingMethod::Posterior) => {
                            posteriors.iter()
                                .map(|probs| probs.iter().enumerate()
                                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                    .map(|(idx, _)| idx).unwrap_or(0))
                                .collect()
                        }
                        (_, DecodingMethod::Mpel) =>
                            mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        (None, DecodingMethod::Viterbi) =>
                            viterbi_from_log_emissions(&emissions, &pass2_params),
                    };

                    // Apply posterior-weighted Viterbi if requested
                    let states = if args.posterior_viterbi_lambda > 0.0 {
                        let blended = blend_posteriors_with_emissions(
                            &emissions, &posteriors, args.posterior_viterbi_lambda);
                        viterbi_from_log_emissions(&blended, &pass2_params)
                    } else {
                        states
                    };

                    // Apply short segment correction if requested
                    let states = if args.min_ancestry_windows > 1 {
                        correct_short_segments(&states, &emissions, args.min_ancestry_windows)
                    } else {
                        states
                    };

                    // Apply majority vote filter if requested
                    let states = if args.majority_vote > 0 {
                        majority_vote_filter(&states, pass2_params.n_states, args.majority_vote)
                    } else {
                        states
                    };
                    // Apply bidirectional smoothing if requested
                    let states = if args.bidirectional_smooth > 0 {
                        bidirectional_smooth_states(&states, pass2_params.n_states,
                            args.bidirectional_smooth)
                    } else {
                        states
                    };

                    (posteriors, states)
                } else if posterior_feedback_lambda > 0.0 || args.focused_pass2_threshold > 0.0
                          || args.label_smooth_alpha > 0.0 || args.margin_persistence_bonus > 0.0
                          || args.variance_penalty > 0.0 || args.flank_inform_radius > 0
                          || args.loo_robust || args.sharpen_posteriors_temp > 0.0
                          || args.whiten_emissions || args.min_ancestry_windows > 1
                          || args.iterative_refine_passes > 1 || args.calibrate_scale > 0.0
                          || args.diversity_amplify > 0.0 || args.residual_amplify > 0.0
                          || args.rank_transform || args.anchor_boost_radius > 0
                          || args.outlier_dampen > 0.0 || args.confusion_penalty > 0.0
                          || args.emission_momentum > 0.0 || args.emission_floor < 0.0
                          || args.gradient_penalty > 0.0 || args.posterior_viterbi_lambda > 0.0
                          || args.changepoint_prior > 0.0 || args.pairwise_contrast > 0.0
                          || args.pop_temp_adjust > 0.0 || args.snr_weight > 0.0
                          || args.cross_entropy_reg > 0.0 || args.window_normalize > 0
                          || args.entropy_smooth_radius > 0 || args.quantile_normalize
                          || args.adaptive_transition_scale > 0.0 || args.local_rerank > 0
                          || args.bayesian_shrink > 0.0 || args.sparsify_top > 0
                          || args.majority_vote > 0 || args.proportion_prior > 0.0
                          || args.boundary_boost > 0.0 || args.confidence_weight > 0.0
                          || (args.fb_temperature - 1.0).abs() > 1e-6 || args.cooccurrence_bonus > 0.0
                          || args.detrend_emissions || args.transition_momentum > 0.0
                          || args.variance_stabilize || args.lookahead_transitions > 0
                          || args.kurtosis_weight > 0.0 || args.segment_length_prior > 1
                          || args.gap_penalty > 0.0 || args.recency_transitions > 0.0
                          || args.center_emissions || args.persistence_bonus > 0.0
                          || args.median_polish || args.disagreement_penalty > 0.0
                          || args.softmax_renorm > 0.0 || args.bidirectional_smooth > 0 {
                    // Two-pass without profiles but with emission modifications:
                    // must precompute emissions to inject feedback, masking, or smoothing
                    let raw_emissions = if let Some(ref ht) = hetero_temperatures {
                        precompute_heteroscedastic_log_emissions(observations, &pass2_params, ht)
                    } else {
                        precompute_log_emissions(observations, &pass2_params)
                    };
                    // Blend LOO robust emissions if requested
                    let raw_emissions = if args.loo_robust {
                        let loo = compute_loo_robust_emissions(
                            observations, &pass2_params.populations,
                            &pass2_params.emission_model, pass2_params.emission_std);
                        let w = args.loo_robust_weight.clamp(0.0, 1.0);
                        raw_emissions.iter().zip(loo.iter()).map(|(std_row, loo_row)| {
                            std_row.iter().zip(loo_row.iter()).map(|(&s, &l)| {
                                if s.is_finite() && l.is_finite() {
                                    (1.0 - w) * s + w * l
                                } else if s.is_finite() {
                                    s
                                } else {
                                    l
                                }
                            }).collect()
                        }).collect()
                    } else {
                        raw_emissions
                    };
                    // Apply emission whitening if requested
                    let raw_emissions = if args.whiten_emissions {
                        whiten_log_emissions(&raw_emissions, 1e-6)
                    } else {
                        raw_emissions
                    };
                    // Apply rank transform if requested
                    let raw_emissions = if args.rank_transform {
                        rank_transform_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply outlier dampening if requested
                    let raw_emissions = if args.outlier_dampen > 0.0 {
                        dampen_emission_outliers(&raw_emissions, args.outlier_dampen)
                    } else {
                        raw_emissions
                    };
                    // Apply emission momentum if requested
                    let raw_emissions = if args.emission_momentum > 0.0 {
                        apply_emission_momentum(&raw_emissions, args.emission_momentum)
                    } else {
                        raw_emissions
                    };
                    // Apply pop temperature adjustment if requested
                    let raw_emissions = if args.pop_temp_adjust > 0.0 {
                        adjust_pop_temperatures(&raw_emissions, &pass1_posteriors,
                            args.pop_temp_adjust)
                    } else {
                        raw_emissions
                    };
                    // Apply SNR weighting if requested
                    let raw_emissions = if args.snr_weight > 0.0 {
                        apply_snr_weighting(&raw_emissions, args.snr_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply emission detrending if requested
                    let raw_emissions = if args.detrend_emissions {
                        detrend_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply emission variance stabilization if requested
                    let raw_emissions = if args.variance_stabilize {
                        variance_stabilize_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply kurtosis weighting if requested
                    let raw_emissions = if args.kurtosis_weight > 0.0 {
                        apply_kurtosis_weighting(&raw_emissions, args.kurtosis_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply emission gap penalty if requested
                    let raw_emissions = if args.gap_penalty > 0.0 {
                        apply_gap_penalty(&raw_emissions, args.gap_penalty)
                    } else {
                        raw_emissions
                    };
                    // Apply emission centering if requested
                    let raw_emissions = if args.center_emissions {
                        center_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply emission median polish if requested
                    let raw_emissions = if args.median_polish {
                        median_polish_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply softmax renormalization if requested
                    let raw_emissions = if args.softmax_renorm > 0.0 {
                        softmax_renormalize(&raw_emissions, args.softmax_renorm)
                    } else {
                        raw_emissions
                    };
                    // Apply windowed normalization if requested
                    let raw_emissions = if args.window_normalize > 0 {
                        apply_windowed_normalization(&raw_emissions, args.window_normalize)
                    } else {
                        raw_emissions
                    };
                    // Apply quantile normalization if requested
                    let raw_emissions = if args.quantile_normalize {
                        quantile_normalize_emissions(&raw_emissions)
                    } else {
                        raw_emissions
                    };
                    // Apply local emission reranking if requested
                    let raw_emissions = if args.local_rerank > 0 {
                        local_rerank_emissions(&raw_emissions, args.local_rerank)
                    } else {
                        raw_emissions
                    };
                    // Apply Bayesian emission shrinkage if requested
                    let raw_emissions = if args.bayesian_shrink > 0.0 {
                        bayesian_shrink_emissions(&raw_emissions, args.bayesian_shrink)
                    } else {
                        raw_emissions
                    };
                    // Apply top-K emission sparsification if requested
                    let raw_emissions = if args.sparsify_top > 0 {
                        sparsify_top_k_emissions(&raw_emissions, args.sparsify_top, -100.0)
                    } else {
                        raw_emissions
                    };
                    // Apply population proportion prior if requested
                    let raw_emissions = if args.proportion_prior > 0.0 {
                        let proportions = estimate_proportions_from_states(
                            &pass1_states, pass2_params.n_states);
                        apply_proportion_prior(&raw_emissions, &proportions,
                            args.proportion_prior)
                    } else {
                        raw_emissions
                    };
                    // Apply emission confidence weighting if requested
                    let raw_emissions = if args.confidence_weight > 0.0 {
                        apply_confidence_weighting(&raw_emissions, args.confidence_weight)
                    } else {
                        raw_emissions
                    };
                    // Apply forward-backward temperature if requested
                    let raw_emissions = if (args.fb_temperature - 1.0).abs() > 1e-6 {
                        apply_fb_temperature(&raw_emissions, args.fb_temperature)
                    } else {
                        raw_emissions
                    };
                    // Apply state persistence bonus if requested
                    let raw_emissions = if args.persistence_bonus > 0.0 {
                        apply_persistence_bonus(&raw_emissions, &pass1_states,
                            args.persistence_bonus)
                    } else {
                        raw_emissions
                    };
                    let emissions = if posterior_feedback_lambda > 0.0 {
                        apply_posterior_feedback(
                            &raw_emissions, &pass1_posteriors, posterior_feedback_lambda)
                    } else {
                        raw_emissions
                    };
                    let emissions = if args.focused_pass2_threshold > 0.0 {
                        apply_focused_masking(&emissions, &pass1_posteriors,
                            args.focused_pass2_threshold, 2)
                    } else {
                        emissions
                    };
                    let emissions = if args.label_smooth_alpha > 0.0 {
                        apply_label_smoothing(&emissions, args.label_smooth_alpha)
                    } else {
                        emissions
                    };
                    // Apply cross-entropy regularization if requested
                    let emissions = if args.cross_entropy_reg > 0.0 {
                        regularize_toward_posteriors(&emissions, &pass1_posteriors,
                            args.cross_entropy_reg)
                    } else {
                        emissions
                    };
                    // Apply emission floor if requested
                    let emissions = if args.emission_floor < 0.0 {
                        apply_emission_floor(&emissions, args.emission_floor)
                    } else {
                        emissions
                    };
                    // Apply population calibration if requested
                    let emissions = if args.calibrate_scale > 0.0 {
                        let proportions = estimate_proportions_from_states(
                            &pass1_states, pass2_params.n_states);
                        let boosts = compute_calibration_boosts(
                            &pass1_states, &proportions, pass2_params.n_states,
                            args.calibrate_scale);
                        apply_calibration_boosts(&emissions, &boosts)
                    } else {
                        emissions
                    };
                    // Apply diversity scaling if requested
                    let emissions = if args.diversity_amplify > 0.0 {
                        apply_diversity_scaling(&emissions,
                            args.diversity_amplify, args.diversity_dampen)
                    } else {
                        emissions
                    };
                    // Apply residual amplification if requested
                    let emissions = if args.residual_amplify > 0.0 {
                        amplify_emission_residuals(&emissions, args.residual_amplify)
                    } else {
                        emissions
                    };
                    // Apply emission anchor boost if requested
                    let emissions = if args.anchor_boost_radius > 0 {
                        apply_emission_anchor_boost(&emissions,
                            args.anchor_boost_radius, args.anchor_boost_threshold,
                            args.anchor_boost_strength)
                    } else {
                        emissions
                    };
                    // Apply gradient penalty if requested
                    let emissions = if args.gradient_penalty > 0.0 {
                        apply_gradient_penalty(&emissions, args.gradient_penalty)
                    } else {
                        emissions
                    };
                    // Apply changepoint prior if requested
                    let emissions = if args.changepoint_prior > 0.0 {
                        apply_changepoint_prior(&emissions, &pass1_states,
                            args.changepoint_prior)
                    } else {
                        emissions
                    };
                    // Apply pairwise emission contrast if requested
                    let emissions = if args.pairwise_contrast > 0.0 {
                        apply_pairwise_emission_contrast(&emissions,
                            args.pairwise_contrast)
                    } else {
                        emissions
                    };
                    let emissions = if args.margin_persistence_bonus > 0.0 {
                        apply_margin_persistence(&emissions, &pass1_posteriors,
                            args.margin_persistence_threshold, args.margin_persistence_bonus)
                    } else {
                        emissions
                    };

                    // Apply within-population variance penalty if requested
                    let emissions = if args.variance_penalty > 0.0 {
                        let variances = compute_within_pop_variance(
                            observations, &pass2_params.populations);
                        apply_variance_penalty(&emissions, &variances, args.variance_penalty)
                    } else {
                        emissions
                    };

                    // Apply flank-informed emission bonus if requested
                    let emissions = if args.flank_inform_radius > 0 {
                        apply_flank_informed_bonus(&emissions, &pass1_states,
                            args.flank_inform_radius, args.flank_inform_bonus,
                            pass2_params.n_states)
                    } else {
                        emissions
                    };

                    // Use iterative refinement if requested, otherwise standard inference
                    let (posteriors, states) = if args.iterative_refine_passes > 1 {
                        let lambda = if posterior_feedback_lambda > 0.0 {
                            posterior_feedback_lambda
                        } else {
                            0.5 // default feedback lambda for iterative mode
                        };
                        iterative_refine(&emissions, &pass2_params,
                            args.iterative_refine_passes, lambda)
                    } else if args.confusion_penalty > 0.0 {
                        // Confusion penalty: modify transitions based on pass-1 switch patterns
                        let penalties = compute_confusion_penalties(
                            &pass1_states, pass2_params.n_states, args.confusion_penalty);
                        let log_trans = apply_confusion_penalties(&pass2_params, &penalties);
                        let n_windows = emissions.len();
                        let per_window_trans: Vec<Vec<Vec<f64>>> = vec![log_trans; n_windows];
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.adaptive_transition_scale > 0.0 {
                        // Adaptive transition scaling: per-window transitions based on emission entropy
                        let per_window_trans = compute_adaptive_transitions(
                            &emissions, &pass2_params, args.adaptive_transition_scale);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.boundary_boost > 0.0 {
                        // Boundary boost: ease transitions at pass-1 detected boundaries
                        let per_window_trans = compute_boundary_boost_transitions(
                            &pass1_states, &pass2_params, args.boundary_boost);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.cooccurrence_bonus > 0.0 {
                        // Co-occurrence bonus: boost transitions for commonly co-occurring pairs
                        let log_trans = compute_cooccurrence_transitions(
                            &pass1_states, &pass2_params, args.cooccurrence_bonus);
                        let n_windows = emissions.len();
                        let per_window_trans: Vec<Vec<Vec<f64>>> = vec![log_trans; n_windows];
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.transition_momentum > 0.0 {
                        let per_window_trans = compute_transition_momentum(
                            &pass1_states, &pass2_params, args.transition_momentum);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.lookahead_transitions > 0 {
                        let per_window_trans = compute_lookahead_transitions(
                            &emissions, &pass1_states, &pass2_params,
                            args.lookahead_transitions);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.segment_length_prior > 1 {
                        let per_window_trans = compute_segment_length_prior(
                            &pass1_states, &pass2_params, args.segment_length_prior);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.recency_transitions > 0.0 {
                        let per_window_trans = compute_recency_transitions(
                            &pass1_states, &pass2_params, args.recency_transitions);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else if args.disagreement_penalty > 0.0 {
                        let per_window_trans = compute_disagreement_transitions(
                            &emissions, &pass1_states, &pass2_params,
                            args.disagreement_penalty);
                        let posteriors = forward_backward_from_log_emissions_with_transitions(
                            &emissions, &pass2_params, &per_window_trans);
                        let states = match decoding_method {
                            DecodingMethod::Viterbi =>
                                viterbi_from_log_emissions_with_transitions(
                                    &emissions, &pass2_params, &per_window_trans),
                            DecodingMethod::Posterior =>
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect(),
                            DecodingMethod::Mpel =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                        };
                        (posteriors, states)
                    } else {
                        let posteriors = if let Some(ref gmap) = genetic_map {
                            forward_backward_from_log_emissions_with_genetic_map(
                                observations, &emissions, &pass2_params, gmap)
                        } else {
                            forward_backward_from_log_emissions(&emissions, &pass2_params)
                        };

                        let states = match (&genetic_map, decoding_method) {
                            (Some(ref gmap), DecodingMethod::Viterbi) =>
                                viterbi_from_log_emissions_with_genetic_map(
                                    observations, &emissions, &pass2_params, gmap),
                            (_, DecodingMethod::Posterior) => {
                                posteriors.iter()
                                    .map(|probs| probs.iter().enumerate()
                                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                        .map(|(idx, _)| idx).unwrap_or(0))
                                    .collect()
                            }
                            (_, DecodingMethod::Mpel) =>
                                mpel_decode_from_posteriors(&posteriors, &pass2_params),
                            (None, DecodingMethod::Viterbi) =>
                                viterbi_from_log_emissions(&emissions, &pass2_params),
                        };
                        (posteriors, states)
                    };

                    // Apply posterior sharpening if requested
                    let posteriors = if args.sharpen_posteriors_temp > 0.0 {
                        sharpen_posteriors(&posteriors, args.sharpen_posteriors_temp)
                    } else {
                        posteriors
                    };

                    // Re-decode if sharpened
                    let states = if args.sharpen_posteriors_temp > 0.0 {
                        posteriors.iter()
                            .map(|probs| probs.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(idx, _)| idx).unwrap_or(0))
                            .collect()
                    } else {
                        states
                    };

                    // Apply entropy-weighted posterior smoothing if requested
                    let posteriors = if args.entropy_smooth_radius > 0 {
                        entropy_smooth_posteriors(&posteriors, args.entropy_smooth_radius)
                    } else {
                        posteriors
                    };
                    // Re-decode if entropy-smoothed
                    let states = if args.entropy_smooth_radius > 0 {
                        posteriors.iter()
                            .map(|probs| probs.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                                .map(|(idx, _)| idx).unwrap_or(0))
                            .collect()
                    } else {
                        states
                    };

                    // Apply posterior-weighted Viterbi if requested
                    let states = if args.posterior_viterbi_lambda > 0.0 {
                        let blended = blend_posteriors_with_emissions(
                            &emissions, &posteriors, args.posterior_viterbi_lambda);
                        viterbi_from_log_emissions(&blended, &pass2_params)
                    } else {
                        states
                    };

                    // Apply short segment correction if requested
                    let states = if args.min_ancestry_windows > 1 {
                        correct_short_segments(&states, &emissions, args.min_ancestry_windows)
                    } else {
                        states
                    };

                    // Apply majority vote filter if requested
                    let states = if args.majority_vote > 0 {
                        majority_vote_filter(&states, pass2_params.n_states, args.majority_vote)
                    } else {
                        states
                    };
                    // Apply bidirectional smoothing if requested
                    let states = if args.bidirectional_smooth > 0 {
                        bidirectional_smooth_states(&states, pass2_params.n_states,
                            args.bidirectional_smooth)
                    } else {
                        states
                    };

                    (posteriors, states)
                } else {
                    run_inference(observations, &pass2_params)
                }
            } else if use_copying_model {
                // Haplotype copying model: Li & Stephens inspired
                // Apply smoothing to observations if requested
                let processed;
                let obs_for_copying = if effective_ec > 0 {
                    processed = smooth_observations(observations, effective_ec);
                    &processed[..]
                } else {
                    observations
                };

                // Estimate or use provided parameters
                let n_total_haps: usize = populations.iter().map(|p| p.haplotypes.len()).sum();
                let pop_temperature = params.emission_std;
                let (copy_temp, copy_switch, copy_default) = if let Some(rate) = copying_switch_rate {
                    // Manual switch rate: temperature was estimated at population level,
                    // so scale it for the haplotype-level softmax
                    let scaled_temp = scale_temperature_for_copying(
                        pop_temperature, populations.len(), n_total_haps);
                    eprintln!("  Copying temp: {:.6} (scaled from pop-level {:.6})",
                        scaled_temp, pop_temperature);
                    (scaled_temp, rate, 0.99)
                } else {
                    let (est_temp, est_switch, est_default) =
                        estimate_copying_params(observations, &populations);
                    if args.estimate_params || args.temperature.is_some() || loaded_params.is_some() {
                        // Temperature from population-level estimation → scale for haplotypes
                        let scaled_temp = scale_temperature_for_copying(
                            pop_temperature, populations.len(), n_total_haps);
                        eprintln!("  Copying temp: {:.6} (scaled from pop-level {:.6})",
                            scaled_temp, pop_temperature);
                        (scaled_temp, est_switch, est_default)
                    } else {
                        // Temperature from estimate_copying_params (haplotype-level) → use as-is
                        (est_temp, est_switch, est_default)
                    }
                };

                let (states, posteriors) = if copying_em_iters > 0 {
                    infer_ancestry_copying_em(
                        obs_for_copying,
                        &populations,
                        copy_switch,
                        copying_ancestry_frac,
                        copy_temp,
                        copy_default,
                        copying_em_iters,
                    )
                } else {
                    infer_ancestry_copying(
                        obs_for_copying,
                        &populations,
                        copy_switch,
                        copying_ancestry_frac,
                        copy_temp,
                        copy_default,
                    )
                };

                (posteriors, states)
            } else {
                run_inference(observations, &params)
            };

            // Apply entropy-weighted posterior smoothing if requested
            let (posteriors, states) = if args.entropy_smooth_radius > 0 {
                let smoothed_post = entropy_weighted_smooth_posteriors(
                    &posteriors, args.entropy_smooth_radius, args.entropy_smooth_beta);
                // Re-decode states from smoothed posteriors
                let new_states: Vec<usize> = smoothed_post.iter()
                    .map(|probs| probs.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(idx, _)| idx).unwrap_or(0))
                    .collect();
                (smoothed_post, new_states)
            } else {
                (posteriors, states)
            };

            // Apply smoothing if requested
            let (states, n_smoothed) = if args.smooth_min_windows > 0 {
                let smoothed = if args.posterior_smooth {
                    posterior_smooth_states(
                        &states,
                        &posteriors,
                        args.smooth_min_windows,
                        args.posterior_smooth_threshold,
                        0.1, // min posterior to flip during flank-consistent pass
                    )
                } else {
                    smooth_states(&states, args.smooth_min_windows)
                };
                let n_changed = count_smoothing_changes(&states, &smoothed);
                (smoothed, n_changed)
            } else {
                (states, 0)
            };

            // Output per-window quality scores if requested
            if args.output_quality {
                // Use posteriors and states for quality; approximate emissions from posteriors
                let approx_emissions: Vec<Vec<f64>> = posteriors.iter()
                    .map(|row| row.iter().map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }).collect())
                    .collect();
                let quality = compute_window_quality(
                    &posteriors, &approx_emissions, &states, 3);
                let mean_q: f64 = quality.iter().sum::<f64>()
                    / quality.len().max(1) as f64;
                let low_q = quality.iter().filter(|&&q| q < 0.3).count();
                eprintln!("Quality scores: mean={:.3}, low_quality_windows={}/{} ({:.1}%)",
                    mean_q, low_q, quality.len(),
                    100.0 * low_q as f64 / quality.len().max(1) as f64);
            }

            // Extract segments
            let segments = extract_ancestry_segments(
                observations,
                &states,
                &params,
                Some(&posteriors),
            );

            // Optionally refine segment boundaries using posterior interpolation
            // Auto-configure enables this by default (T54: +0.15pp)
            let segments = if args.refine_boundaries || args.auto_configure {
                let refined = refine_ancestry_boundaries(
                    &segments,
                    &posteriors,
                    observations,
                    0.5,
                );
                segments.into_iter().zip(refined).map(|(mut seg, rb)| {
                    seg.start = rb.start_bp;
                    seg.end = rb.end_bp;
                    seg
                }).collect()
            } else {
                segments
            };

            // Filter by length, posterior, and LOD score
            let filtered: Vec<_> = segments.into_iter()
                .filter(|s| {
                    s.end - s.start >= args.min_len_bp
                    && s.n_windows >= args.min_windows
                    && s.mean_posterior.unwrap_or(1.0) >= args.min_posterior
                    && s.lod_score >= args.min_lod
                })
                .collect();

            eprintln!("  {} - {} windows -> {} segments (smoothed {} windows)",
                sample, observations.len(), filtered.len(), n_smoothed);

            (sample.clone(), filtered, posteriors)
        })
        .collect();

    // Post-process deconvolution: merge sub-populations back to parents
    let (results, populations) = if let Some(ref parent_map) = deconvolution_map {
        // Check if any population was actually split
        let any_split = parent_map.iter().any(|(child, parent)| child != parent);
        if any_split {
            // Build parent population list (unique parent names, preserving order)
            let mut parent_pops: Vec<AncestralPopulation> = Vec::new();
            let mut seen_parents: Vec<String> = Vec::new();
            for pop in &populations {
                let parent_name = &parent_map[&pop.name];
                if !seen_parents.contains(parent_name) {
                    seen_parents.push(parent_name.clone());
                    // Collect all haplotypes from all child populations of this parent
                    let all_haps: Vec<String> = populations.iter()
                        .filter(|p| &parent_map[&p.name] == parent_name)
                        .flat_map(|p| p.haplotypes.clone())
                        .collect();
                    parent_pops.push(AncestralPopulation {
                        name: parent_name.clone(),
                        haplotypes: all_haps,
                    });
                }
            }

            // Build index mapping: deconvolved pop idx → parent pop idx
            let child_to_parent_idx: Vec<usize> = populations.iter()
                .map(|p| {
                    let parent_name = &parent_map[&p.name];
                    parent_pops.iter().position(|pp| &pp.name == parent_name)
                        .unwrap_or_else(|| {
                            eprintln!("WARNING: parent population '{}' not found in population groups, defaulting to index 0", parent_name);
                            0
                        })
                })
                .collect();
            let n_parent_pops = parent_pops.len();

            // Remap results: rename ancestry, merge adjacent same-parent segments,
            // and merge posteriors
            let remapped: Vec<_> = results.into_iter().map(|(sample, segments, posteriors)| {
                // Remap segment names
                let mut remapped_segs: Vec<_> = segments.into_iter().map(|mut seg| {
                    let parent_idx = child_to_parent_idx[seg.ancestry_idx];
                    seg.ancestry_idx = parent_idx;
                    seg.ancestry_name = parent_pops[parent_idx].name.clone();
                    seg
                }).collect();

                // Merge adjacent segments with same ancestry (from split populations)
                let mut merged_segs: Vec<hprc_ancestry_cli::AncestrySegment> = Vec::new();
                for seg in remapped_segs.drain(..) {
                    if let Some(last) = merged_segs.last_mut() {
                        if last.ancestry_idx == seg.ancestry_idx
                            && last.chrom == seg.chrom
                            && last.sample == seg.sample
                        {
                            // Merge: extend end, combine stats
                            let total_windows = last.n_windows + seg.n_windows;
                            last.mean_similarity = (last.mean_similarity * last.n_windows as f64
                                + seg.mean_similarity * seg.n_windows as f64)
                                / total_windows as f64;
                            if let (Some(lp), Some(sp)) = (last.mean_posterior, seg.mean_posterior) {
                                last.mean_posterior = Some(
                                    (lp * last.n_windows as f64 + sp * seg.n_windows as f64)
                                        / total_windows as f64,
                                );
                            }
                            last.end = seg.end;
                            last.n_windows = total_windows;
                            last.lod_score = last.lod_score.min(seg.lod_score);
                            continue;
                        }
                    }
                    merged_segs.push(seg);
                }

                // Merge posteriors: sum sub-population posteriors into parent
                let merged_posteriors: Vec<Vec<f64>> = posteriors.iter().map(|window_post| {
                    let mut parent_post = vec![0.0; n_parent_pops];
                    for (child_idx, &prob) in window_post.iter().enumerate() {
                        if child_idx < child_to_parent_idx.len() {
                            parent_post[child_to_parent_idx[child_idx]] += prob;
                        }
                    }
                    parent_post
                }).collect();

                (sample, merged_segs, merged_posteriors)
            }).collect();

            eprintln!("Deconvolution post-processing: merged sub-populations back to {} parent populations",
                n_parent_pops);

            (remapped, parent_pops)
        } else {
            (results, populations)
        }
    } else {
        (results, populations)
    };

    // Write output
    let output_file = File::create(&args.output)?;
    let mut out = BufWriter::new(output_file);

    writeln!(out, "chrom\tstart\tend\tsample\tancestry\tn_windows\tmean_similarity\tmean_posterior\tdiscriminability\tlod_score")?;

    for (_, segments, _) in &results {
        for seg in segments {
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                seg.chrom,
                seg.start,
                seg.end,
                seg.sample,
                seg.ancestry_name,
                seg.n_windows,
                seg.mean_similarity,
                seg.mean_posterior.unwrap_or(0.0),
                seg.discriminability,
                seg.lod_score,
            )?;
        }
    }
    out.flush()?;

    eprintln!("Wrote ancestry segments to {:?}", args.output);

    // Write posteriors if requested
    if let Some(post_path) = &args.posteriors_output {
        let post_file = File::create(post_path)?;
        let mut post_out = BufWriter::new(post_file);

        // Header with population names + diagnostic columns
        write!(post_out, "chrom\tstart\tend\tsample")?;
        for pop in &populations {
            write!(post_out, "\tP({})", pop.name)?;
        }
        write!(post_out, "\tmargin\tentropy")?;
        writeln!(post_out)?;

        for (sample, _, posteriors) in &results {
            if let Some(observations) = similarity_data.get(sample) {
                for (i, obs) in observations.iter().enumerate() {
                    if i < posteriors.len() {
                        write!(post_out, "{}\t{}\t{}\t{}", obs.chrom, obs.start, obs.end, sample)?;
                        for p in &posteriors[i] {
                            write!(post_out, "\t{:.6}", p)?;
                        }
                        // Posterior margin: max - 2nd max (quality/confidence score)
                        let (mut best, mut second) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                        for &p in &posteriors[i] {
                            if p > best { second = best; best = p; }
                            else if p > second { second = p; }
                        }
                        let margin = if best.is_finite() && second.is_finite() {
                            best - second
                        } else {
                            0.0
                        };
                        // Shannon entropy: -Σ p·ln(p)
                        let entropy: f64 = posteriors[i].iter()
                            .filter(|&&p| p > 0.0 && p.is_finite())
                            .map(|&p| -p * p.ln())
                            .sum();
                        write!(post_out, "\t{:.6}\t{:.6}", margin, entropy)?;
                        writeln!(post_out)?;
                    }
                }
            }
        }
        post_out.flush()?;
        eprintln!("Wrote posteriors to {:?}", post_path);
    }

    // Write BED output if requested
    if let Some(bed_path) = &args.output_bed {
        let bed_file = File::create(bed_path)?;
        let mut bed_out = BufWriter::new(bed_file);

        for (_, segments, _) in &results {
            for seg in segments {
                // BED format: chrom, start (0-based), end, name, score (0-1000)
                // Score is posterior * 1000, clamped to [0, 1000]
                let score = (seg.mean_posterior.unwrap_or(0.0) * 1000.0).round().clamp(0.0, 1000.0) as u32;
                writeln!(
                    bed_out,
                    "{}\t{}\t{}\t{}:{}\t{}",
                    seg.chrom,
                    seg.start.saturating_sub(1), // Convert 1-based to BED 0-based
                    seg.end,
                    seg.sample,
                    seg.ancestry_name,
                    score,
                )?;
            }
        }
        bed_out.flush()?;
        eprintln!("Wrote BED output to {:?}", bed_path);
    }

    // Print admixture proportions per sample
    let pop_names: Vec<String> = populations.iter().map(|p| p.name.clone()).collect();
    eprintln!("\n=== Admixture Proportions ===");
    for (sample, segments, _) in &results {
        let admix = estimate_admixture_proportions(segments, sample, &pop_names);
        eprint!("  {}: ", sample);
        let mut sorted_props: Vec<_> = admix.proportions.iter().collect();
        sorted_props.sort_by_key(|(name, _)| (*name).clone());
        for (name, prop) in &sorted_props {
            eprint!("{}={:.1}% ", name, *prop * 100.0);
        }
        eprintln!("({}bp, {} switches, mean tract {:.0}bp)",
            admix.total_length_bp, admix.n_switches, admix.mean_tract_length_bp);
    }

    // Multi-pulse demographic inference (T80)
    // --demographic-output implies --demographic-inference
    if args.demographic_inference || args.demographic_output.is_some() {
        let all_segments: Vec<&hprc_ancestry_cli::AncestrySegment> = results.iter()
            .flat_map(|(_, segs, _)| segs.iter())
            .collect();

        // BW constraint (T80 §5.2): λ_bw = p_switch / W_bp
        let bw_constraint = if args.constrain_demography {
            let p_switch = 1.0 - params.transitions[0][0];
            let w_bp = args.window_size as f64;
            let lambda_bw = p_switch / w_bp;
            eprintln!("  BW constraint: p_switch={:.6}, W={}bp → λ_bw={:.2e}", p_switch, args.window_size, lambda_bw);
            Some(lambda_bw)
        } else {
            None
        };

        let demog_params = DemographyParams {
            recomb_rate: args.recomb_rate,
            l_min_bp: (args.window_size * 2) as f64, // 2 windows = effective HMM detection limit
            max_pulses: args.max_pulses,
            bw_constraint,
            ..Default::default()
        };

        // Pooled inference (all samples together)
        let demog_results = infer_all_demography(&all_segments, &pop_names, &demog_params);
        let report = format_demography_report(&demog_results);
        eprint!("{}", report);

        // Per-sample inference + TSV output
        if let Some(ref output_path) = args.demographic_output {
            let per_sample = infer_per_sample_demography(&all_segments, &pop_names, &demog_params);
            eprintln!("\n  Per-sample demographic inference: {} samples", per_sample.len());
            write_demography_tsv(output_path, &demog_results, &per_sample)?;
            eprintln!("  Written to: {}", output_path.display());
        }
    }

    // Validate against RFMix if requested
    if let Some(rfmix_path) = &args.validate_against {
        validate_against_rfmix(rfmix_path, &results, &similarity_data, &pop_names, &args.validate_output)?;
    }

    // Print diagnostics
    print_diagnostics(&results, &params, &populations);

    Ok(())
}

#[allow(clippy::type_complexity)]
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
    let uncertain = all_max_posts.iter().filter(|&&p| (0.5..=0.8).contains(&p)).count();
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

    // Admixture time estimation (T77)
    // From switch probability: ĝ = p_s / (r × W) where r ≈ 1e-8 bp/gen, W = window size
    // From tract lengths: ĝ_corrected = -1/W × ln(1 - W/L̄)
    let window_size_bp = 10_000.0_f64; // default 10kb
    let recomb_rate = 1e-8_f64; // per-bp per-generation

    // Estimate from transition matrix (average off-diagonal switch prob)
    let n_states = params.n_states;
    if n_states >= 2 {
        let total_switch: f64 = params.transitions.iter()
            .enumerate()
            .map(|(i, row)| row.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &p)| p)
                .sum::<f64>())
            .sum::<f64>();
        let avg_switch = total_switch / n_states as f64;
        let g_switch = avg_switch / (recomb_rate * window_size_bp);

        // Estimate from mean tract length (discretization-corrected, T77 §1.3)
        let all_segments: Vec<_> = results.iter()
            .flat_map(|(_, segs, _)| segs.iter())
            .collect();
        if !all_segments.is_empty() {
            let tract_lengths: Vec<f64> = all_segments.iter()
                .map(|s| (s.end - s.start) as f64)
                .collect();
            let mean_tract = tract_lengths.iter().sum::<f64>() / tract_lengths.len() as f64;

            let g_tract = if mean_tract > window_size_bp {
                // Corrected estimator: -1/W × ln(1 - W/L̄)
                let lambda = -(1.0 / window_size_bp) * (1.0 - window_size_bp / mean_tract).ln();
                lambda / recomb_rate
            } else {
                1.0 / (recomb_rate * mean_tract)
            };

            eprintln!("\nAdmixture time estimates (T77):");
            eprintln!("  From switch prob (p_s={:.6}): ĝ ≈ {:.0} generations", avg_switch, g_switch);
            eprintln!("  From mean tract length ({:.0} kb): ĝ ≈ {:.0} generations",
                mean_tract / 1000.0, g_tract);
            eprintln!("  (assuming r=1e-8/bp/gen, W=10kb)");

            // Per-population tract lengths
            eprintln!("  Per-population mean tract length:");
            for pop in populations {
                let pop_tracts: Vec<f64> = all_segments.iter()
                    .filter(|s| s.ancestry_name == pop.name)
                    .map(|s| (s.end - s.start) as f64)
                    .collect();
                if !pop_tracts.is_empty() {
                    let mean_pop = pop_tracts.iter().sum::<f64>() / pop_tracts.len() as f64;
                    let g_pop = if mean_pop > window_size_bp {
                        let lambda = -(1.0 / window_size_bp) * (1.0 - window_size_bp / mean_pop).ln();
                        lambda / recomb_rate
                    } else {
                        1.0 / (recomb_rate * mean_pop)
                    };
                    eprintln!("    {}: {:.0} kb (n={}, ĝ≈{:.0})", pop.name,
                        mean_pop / 1000.0, pop_tracts.len(), g_pop);
                }
            }
        }
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

    let mut pop_map: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();

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
        .map_while(Result::ok)
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

/// Parse a BED file into regions (chrom, start, end).
/// BED format: tab-separated, 0-based half-open coordinates.
/// Converts to 1-based inclusive for internal use.
fn parse_bed_regions(path: &PathBuf, reference: &str) -> Result<Vec<(String, u64, u64)>> {
    let file = File::open(path).context("Failed to open BED file")?;
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
            anyhow::bail!("BED line must have at least 3 columns: {}", line);
        }

        let chrom_raw = fields[0];
        // Add reference prefix if not already present
        let chrom = if chrom_raw.contains('#') {
            chrom_raw.to_string()
        } else {
            format!("{}#{}", reference, chrom_raw)
        };

        let start: u64 = fields[1].parse().context("Invalid BED start")?;
        let end: u64 = fields[2].parse().context("Invalid BED end")?;

        // BED is 0-based half-open, convert to 1-based inclusive
        regions.push((chrom, start + 1, end));
    }

    if regions.is_empty() {
        anyhow::bail!("BED file is empty or has no valid regions");
    }

    Ok(regions)
}

/// Parse mask BED file into (chrom, start, end) tuples.
/// Uses raw chromosome names (no reference prefix).
fn parse_mask_bed_ancestry(path: &PathBuf) -> Result<Vec<(String, u64, u64)>> {
    let file = File::open(path).context("Failed to open mask BED file")?;
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
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse().unwrap_or(0);
        let end: u64 = fields[2].parse().unwrap_or(0);
        // BED 0-based half-open → 1-based start for comparison
        regions.push((chrom, start + 1, end));
    }

    Ok(regions)
}

fn load_similarity_file(
    path: &PathBuf,
    query_samples: &[String],
    reference_haplotypes: &[String],
    similarity_column: &str,
) -> Result<std::collections::HashMap<String, Vec<AncestryObservation>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines = reader.lines().map_while(Result::ok);
    parse_similarity_data_column(lines, query_samples, reference_haplotypes, similarity_column)
        .map_err(|e| anyhow::anyhow!("Failed to parse similarity file: {}", e))
}

fn load_similarity_file_with_coverage(
    path: &PathBuf,
    query_samples: &[String],
    reference_haplotypes: &[String],
    similarity_column: &str,
) -> Result<std::collections::HashMap<String, Vec<AncestryObservation>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines = reader.lines().map_while(Result::ok);
    parse_similarity_data_with_coverage(lines, query_samples, reference_haplotypes, similarity_column)
        .map_err(|e| anyhow::anyhow!("Failed to parse similarity file with coverage: {}", e))
}

#[allow(clippy::too_many_arguments)]
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

/// Validate our ancestry calls against RFMix ground truth.
///
/// Compares per-window ancestry calls from our posteriors against RFMix's
/// .msp.tsv segments. Outputs per-haplotype concordance metrics.
#[allow(clippy::type_complexity)]
fn validate_against_rfmix(
    rfmix_path: &std::path::Path,
    results: &[(String, Vec<hprc_ancestry_cli::AncestrySegment>, Vec<Vec<f64>>)],
    similarity_data: &std::collections::HashMap<String, Vec<AncestryObservation>>,
    our_pop_names: &[String],
    validate_output: &Option<PathBuf>,
) -> Result<()> {
    eprintln!("\n=== Validation against RFMix ===");

    // Parse RFMix file
    let rfmix_result = rfmix::parse_rfmix_msp(rfmix_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse RFMix file: {}", e))?;

    eprintln!("RFMix populations: {:?}", rfmix_result.population_names);
    eprintln!("RFMix haplotypes: {:?}", rfmix_result.haplotype_names);
    eprintln!("RFMix segments: {}", rfmix_result.segments.len());

    // Build population index mapping: RFMix index -> our index
    let mut rfmix_to_ours: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (rfmix_idx, rfmix_name) in rfmix_result.population_names.iter().enumerate() {
        if let Some(our_idx) = our_pop_names.iter().position(|n| n == rfmix_name) {
            rfmix_to_ours.insert(rfmix_idx, our_idx);
        }
    }
    eprintln!("Population mapping (RFMix->Ours): {:?}", rfmix_to_ours);

    // Convert RFMix to windows (10kb)
    let window_size = 10_000u64;
    let rfmix_windows = rfmix::rfmix_to_windows(&rfmix_result, window_size);
    let rfmix_starts = rfmix::rfmix_window_starts(&rfmix_result, window_size);

    eprintln!("RFMix windows: {} per haplotype ({} bp)", rfmix_starts.len(), window_size);

    // Build position-to-rfmix-ancestry map for each haplotype
    let mut output_lines: Vec<String> = Vec::new();
    output_lines.push(format!(
        "sample\trfmix_hap\tconcordance\tn_windows\t{}_precision\t{}_recall\t{}_f1\tswitch_detection\tmean_switch_dist",
        our_pop_names.join("_precision\t") + "_precision",
        our_pop_names.join("_recall\t") + "_recall",
        our_pop_names.join("_f1\t") + "_f1",
    ));

    // Simplified header
    let header = format!(
        "sample\trfmix_hap\tconcordance\tn_windows\tswitch_detection\tmean_switch_dist\tswitch_precision\tn_spurious_switches{}{}",
        our_pop_names.iter().map(|n| format!("\t{}_prec\t{}_recall\t{}_f1", n, n, n)).collect::<String>(),
        our_pop_names.iter().map(|n| format!("\t{}_seg_jaccard\t{}_seg_prec\t{}_seg_recall\t{}_seg_f1", n, n, n, n)).collect::<String>(),
    );
    output_lines.clear();
    output_lines.push(header);

    for (sample, _segments, posteriors) in results {
        // Map our sample name to RFMix haplotype name
        // Convention: our "HG00733#1" → RFMix "HG00733.0"
        let sample_base = sample.split('#').next().unwrap_or(sample);
        let hap_num: usize = sample.split('#').nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let rfmix_hap_name = format!("{}.{}", sample_base, hap_num - 1);

        let rfmix_hap_idx = rfmix_result.haplotype_names.iter()
            .position(|n| n == &rfmix_hap_name);

        if rfmix_hap_idx.is_none() {
            eprintln!("  {}: no matching RFMix haplotype ({}), skipping", sample, rfmix_hap_name);
            continue;
        }
        let rfmix_hap_idx = rfmix_hap_idx.unwrap();

        // Get our per-window ancestry calls from posteriors
        let observations = match similarity_data.get(sample.as_str()) {
            Some(obs) => obs,
            None => continue,
        };

        // Build position-indexed maps for both tools
        let mut our_calls: Vec<(u64, usize)> = Vec::new();
        for (i, obs) in observations.iter().enumerate() {
            if i < posteriors.len() {
                let ancestry_idx = posteriors[i].iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                our_calls.push((obs.start, ancestry_idx));
            }
        }

        // Build RFMix position-indexed calls (mapped to our population indices)
        let rfmix_hap_windows = &rfmix_windows[rfmix_hap_idx];
        let mut rfmix_calls: Vec<(u64, usize)> = Vec::new();
        for (w_idx, &maybe_anc) in rfmix_hap_windows.iter().enumerate() {
            if let Some(anc) = maybe_anc {
                if let Some(&our_idx) = rfmix_to_ours.get(&anc) {
                    if w_idx < rfmix_starts.len() {
                        rfmix_calls.push((rfmix_starts[w_idx], our_idx));
                    }
                }
            }
        }

        // Match windows by position (within half-window tolerance)
        let mut our_matched: Vec<usize> = Vec::new();
        let mut rfmix_matched: Vec<usize> = Vec::new();
        let tolerance = window_size / 2;

        for &(our_pos, our_anc) in &our_calls {
            // Find closest RFMix window
            if let Some(&(_, rfmix_anc)) = rfmix_calls.iter()
                .min_by_key(|&&(rpos, _)| (rpos as i64 - our_pos as i64).unsigned_abs())
            {
                let closest_rfmix_pos = rfmix_calls.iter()
                    .min_by_key(|&&(rpos, _)| (rpos as i64 - our_pos as i64).unsigned_abs())
                    .map(|&(p, _)| p)
                    .unwrap_or(0);

                if (closest_rfmix_pos as i64 - our_pos as i64).unsigned_abs() <= tolerance {
                    our_matched.push(our_anc);
                    rfmix_matched.push(rfmix_anc);
                }
            }
        }

        let n_matched = our_matched.len();
        if n_matched == 0 {
            eprintln!("  {}: no matching windows found", sample);
            continue;
        }

        // Compute concordance metrics using our concordance module
        let report = concordance::compute_concordance_report(
            &our_matched,
            &rfmix_matched,
            our_pop_names,
            5, // 5-window tolerance for switch detection
        );

        eprintln!("\n  {} vs RFMix {}:", sample, rfmix_hap_name);
        eprintln!("    Concordance: {:.4} ({} windows)", report.overall_concordance, report.n_windows);
        eprintln!("    Switches: {} true (RFMix), {} predicted (ours), {} detected, {} spurious",
            report.n_true_switches, report.n_predicted_switches,
            report.n_switches_detected, report.n_spurious_switches);
        eprintln!("    Switch recall: {:.4}, precision: {:.4}",
            report.switch_detection_rate, report.switch_precision);
        eprintln!("    Mean switch distance: {:.1} windows", report.mean_switch_distance);

        for name in our_pop_names {
            if let Some(&(prec, rec, f1)) = report.per_population.get(name) {
                eprintln!("    {}: precision={:.4}, recall={:.4}, F1={:.4}", name, prec, rec, f1);
            }
        }

        // Compute segment-level concordance (bp-level Jaccard and precision/recall)
        let seg_report = concordance::compute_segment_concordance(
            &our_matched,
            &rfmix_matched,
            our_pop_names,
            0, // window_start (relative; positions will be in window-size units)
            window_size,
        );
        eprintln!("    Segment-level (bp):");
        for (i, name) in our_pop_names.iter().enumerate() {
            let (sprec, srec, sf1) = seg_report.precision_recall_per_pop[i];
            eprintln!("      {}: Jaccard={:.4}, precision={:.4}, recall={:.4}, F1={:.4}",
                name, seg_report.jaccard_per_pop[i], sprec, srec, sf1);
        }

        // Format output line
        let mut line = format!(
            "{}\t{}\t{:.6}\t{}\t{:.6}\t{:.2}\t{:.6}\t{}",
            sample, rfmix_hap_name, report.overall_concordance, report.n_windows,
            report.switch_detection_rate, report.mean_switch_distance,
            report.switch_precision, report.n_spurious_switches,
        );
        for name in our_pop_names {
            if let Some(&(prec, rec, f1)) = report.per_population.get(name) {
                line.push_str(&format!("\t{:.6}\t{:.6}\t{:.6}", prec, rec, f1));
            } else {
                line.push_str("\t0\t0\t0");
            }
        }
        // Append segment-level metrics
        for i in 0..our_pop_names.len() {
            let (sprec, srec, sf1) = seg_report.precision_recall_per_pop[i];
            line.push_str(&format!("\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                seg_report.jaccard_per_pop[i], sprec, srec, sf1));
        }
        output_lines.push(line);
    }

    // Write validation output TSV if requested
    if let Some(out_path) = validate_output {
        let file = File::create(out_path)?;
        let mut out = BufWriter::new(file);
        for line in &output_lines {
            writeln!(out, "{}", line)?;
        }
        out.flush()?;
        eprintln!("\nWrote validation metrics to {:?}", out_path);
    }

    Ok(())
}
