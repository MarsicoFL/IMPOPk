//! Ancestry segment extraction and data processing

use std::collections::HashMap;
use std::path::Path;
use std::io::{BufRead, BufReader};
use crate::hmm::{AncestryHmmParams, AncestryObservation, AncestralPopulation};

/// A contiguous segment with assigned ancestry
#[derive(Debug, Clone)]
pub struct AncestrySegment {
    /// Chromosome/scaffold name
    pub chrom: String,
    /// Segment start position
    pub start: u64,
    /// Segment end position
    pub end: u64,
    /// Sample ID
    pub sample: String,
    /// Assigned ancestral population index
    pub ancestry_idx: usize,
    /// Ancestral population name
    pub ancestry_name: String,
    /// Number of windows in segment
    pub n_windows: usize,
    /// Mean similarity to assigned ancestry
    pub mean_similarity: f64,
    /// Mean posterior probability (if using forward-backward)
    pub mean_posterior: Option<f64>,
    /// Discriminability: max_sim - min_sim across populations
    /// Low values (<0.05) indicate regions where ancestry is hard to determine
    pub discriminability: f64,
    /// LOD score: sum of per-window log10(P(obs|assigned) / P(obs|second-best))
    /// Positive values support the assigned ancestry; higher = more confident
    pub lod_score: f64,
}

/// Extract ancestry segments from state sequence using run-length encoding
pub fn extract_ancestry_segments(
    observations: &[AncestryObservation],
    states: &[usize],
    params: &AncestryHmmParams,
    posteriors: Option<&[Vec<f64>]>,
) -> Vec<AncestrySegment> {
    if observations.is_empty() || states.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut seg_start_idx = 0;
    let mut current_state = states[0];

    for (i, &state) in states.iter().enumerate().skip(1) {
        if state != current_state {
            // End current segment
            segments.push(create_segment(
                observations,
                states,
                params,
                posteriors,
                seg_start_idx,
                i - 1,
                current_state,
            ));

            // Start new segment
            seg_start_idx = i;
            current_state = state;
        }
    }

    // Don't forget last segment
    segments.push(create_segment(
        observations,
        states,
        params,
        posteriors,
        seg_start_idx,
        states.len() - 1,
        current_state,
    ));

    segments
}

/// Compute per-window LOD score for ancestry assignment.
///
/// LOD = log10(P(obs | assigned_pop) / P(obs | second_best_pop))
///
/// Uses the HMM emission model to compute log-likelihoods for each population
/// at each window. Positive LOD supports the assigned ancestry.
pub fn compute_per_window_ancestry_lod(
    obs: &AncestryObservation,
    params: &AncestryHmmParams,
    assigned_state: usize,
) -> f64 {
    let log_em_assigned = params.log_emission(obs, assigned_state);

    // Find second-best emission
    let log_em_second_best = (0..params.n_states)
        .filter(|&s| s != assigned_state)
        .map(|s| params.log_emission(obs, s))
        .fold(f64::NEG_INFINITY, f64::max);

    if !log_em_assigned.is_finite() || !log_em_second_best.is_finite() {
        return 0.0;
    }

    // Convert from natural log to log10
    (log_em_assigned - log_em_second_best) / std::f64::consts::LN_10
}

/// Compute segment-level LOD score as sum of per-window LODs.
pub fn segment_ancestry_lod(
    observations: &[AncestryObservation],
    params: &AncestryHmmParams,
    assigned_state: usize,
    start_idx: usize,
    end_idx: usize,
) -> f64 {
    observations[start_idx..=end_idx]
        .iter()
        .map(|obs| compute_per_window_ancestry_lod(obs, params, assigned_state))
        .sum()
}

fn create_segment(
    observations: &[AncestryObservation],
    _states: &[usize],
    params: &AncestryHmmParams,
    posteriors: Option<&[Vec<f64>]>,
    start_idx: usize,
    end_idx: usize,
    state: usize,
) -> AncestrySegment {
    let pop = &params.populations[state];
    let n_windows = end_idx - start_idx + 1;

    // Calculate mean similarity to assigned ancestry and discriminability
    let mut total_sim = 0.0;
    let mut total_discriminability = 0.0;
    let mut count = 0;

    for obs in &observations[start_idx..=end_idx] {
        // Get max similarity to assigned population
        let max_sim = pop.haplotypes.iter()
            .filter_map(|h| obs.similarities.get(h))
            .cloned()
            .fold(0.0_f64, f64::max);

        if max_sim > 0.0 {
            total_sim += max_sim;

            // Calculate discriminability: max - min across all populations
            let pop_sims: Vec<f64> = params.populations.iter()
                .map(|p| {
                    p.haplotypes.iter()
                        .filter_map(|h| obs.similarities.get(h))
                        .cloned()
                        .fold(0.0_f64, f64::max)
                })
                .filter(|&s| s > 0.0)
                .collect();

            if pop_sims.len() >= 2 {
                let max_pop = pop_sims.iter().cloned().fold(0.0_f64, f64::max);
                let min_pop = pop_sims.iter().cloned().fold(f64::INFINITY, f64::min);
                total_discriminability += max_pop - min_pop;
            }
            count += 1;
        }
    }
    let mean_similarity = if count > 0 { total_sim / count as f64 } else { 0.0 };
    let discriminability = if count > 0 { total_discriminability / count as f64 } else { 0.0 };

    // Calculate mean posterior if available
    let mean_posterior = posteriors.map(|p| {
        let sum: f64 = p[start_idx..=end_idx].iter().map(|probs| probs[state]).sum();
        sum / n_windows as f64
    });

    // Compute segment LOD score
    let lod_score = segment_ancestry_lod(observations, params, state, start_idx, end_idx);

    AncestrySegment {
        chrom: observations[start_idx].chrom.clone(),
        start: observations[start_idx].start,
        end: observations[end_idx].end,
        sample: observations[start_idx].sample.clone(),
        ancestry_idx: state,
        ancestry_name: pop.name.clone(),
        n_windows,
        mean_similarity,
        mean_posterior,
        discriminability,
        lod_score,
    }
}

/// Parse similarity data from TSV into AncestryObservations
///
/// Expected format (from impg similarity):
/// chrom, start, end, group.a, group.b, ..., estimated.identity
///
/// Groups sample vs reference observations.
/// `similarity_column` specifies which column to use as the similarity metric
/// (e.g., "estimated.identity", "jaccard.similarity", "cosine.similarity").
pub fn parse_similarity_data(
    lines: impl Iterator<Item = String>,
    query_samples: &[String],
    reference_haplotypes: &[String],
) -> Result<HashMap<String, Vec<AncestryObservation>>, String> {
    parse_similarity_data_column(lines, query_samples, reference_haplotypes, "estimated.identity")
}

/// Parse similarity data using a specific similarity column.
pub fn parse_similarity_data_column(
    lines: impl Iterator<Item = String>,
    query_samples: &[String],
    reference_haplotypes: &[String],
    similarity_column: &str,
) -> Result<HashMap<String, Vec<AncestryObservation>>, String> {
    let mut header_indices: Option<HeaderIndices> = None;
    #[allow(clippy::type_complexity)]
    let mut sample_observations: HashMap<String, HashMap<(String, u64, u64), HashMap<String, f64>>> = HashMap::new();

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();

        if header_indices.is_none() {
            // Parse header
            header_indices = Some(parse_header(&fields, similarity_column)?);
            continue;
        }

        let idx = header_indices.as_ref().unwrap();

        let chrom = fields.get(idx.chrom).ok_or("Missing chrom")?.to_string();
        let start: u64 = fields.get(idx.start).ok_or("Missing start")?
            .parse().map_err(|_| "Invalid start")?;
        let end: u64 = fields.get(idx.end).ok_or("Missing end")?
            .parse().map_err(|_| "Invalid end")?;

        let group_a = fields.get(idx.group_a).ok_or("Missing group.a")?;
        let group_b = fields.get(idx.group_b).ok_or("Missing group.b")?;
        let identity: f64 = fields.get(idx.identity).ok_or("Missing identity")?
            .parse().map_err(|_| "Invalid identity")?;

        // Extract sample and haplotype IDs (remove scaffold:coords suffix)
        let id_a = extract_sample_id(group_a);
        let id_b = extract_sample_id(group_b);

        // Check if this is a query vs reference comparison
        let (query, reference) = if query_samples.contains(&id_a) && reference_haplotypes.contains(&id_b) {
            (id_a, id_b)
        } else if query_samples.contains(&id_b) && reference_haplotypes.contains(&id_a) {
            (id_b, id_a)
        } else {
            continue; // Skip non-query-vs-reference comparisons
        };

        // Store observation - use maximum similarity if multiple alignments exist
        sample_observations
            .entry(query)
            .or_default()
            .entry((chrom, start, end))
            .or_default()
            .entry(reference)
            .and_modify(|existing| {
                if identity > *existing {
                    *existing = identity;
                }
            })
            .or_insert(identity);
    }

    // Convert to AncestryObservations
    let mut result: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    for (sample, windows) in sample_observations {
        let mut obs_list: Vec<AncestryObservation> = windows
            .into_iter()
            .map(|((chrom, start, end), sims)| AncestryObservation {
                chrom,
                start,
                end,
                sample: sample.clone(),
                similarities: sims,
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            })
            .collect();

        // Sort by position
        obs_list.sort_by_key(|o| (o.chrom.clone(), o.start));

        result.insert(sample, obs_list);
    }

    Ok(result)
}

/// Parse similarity data with coverage-ratio auxiliary feature.
///
/// Extracts both the similarity metric and the coverage ratio (min(a_len, b_len) / max(a_len, b_len))
/// from `group.a.length` and `group.b.length` columns. The coverage ratio measures alignment
/// symmetry, which may provide independent ancestry-discriminative information.
pub fn parse_similarity_data_with_coverage(
    lines: impl Iterator<Item = String>,
    query_samples: &[String],
    reference_haplotypes: &[String],
    similarity_column: &str,
) -> Result<HashMap<String, Vec<AncestryObservation>>, String> {
    let mut header_indices: Option<HeaderIndicesWithCoverage> = None;
    #[allow(clippy::type_complexity)]
    let mut sample_observations: HashMap<String, HashMap<(String, u64, u64), (HashMap<String, f64>, HashMap<String, f64>)>> = HashMap::new();

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();

        if header_indices.is_none() {
            header_indices = Some(parse_header_with_coverage(&fields, similarity_column)?);
            continue;
        }

        let idx = header_indices.as_ref().unwrap();

        let chrom = fields.get(idx.chrom).ok_or("Missing chrom")?.to_string();
        let start: u64 = fields.get(idx.start).ok_or("Missing start")?
            .parse().map_err(|_| "Invalid start")?;
        let end: u64 = fields.get(idx.end).ok_or("Missing end")?
            .parse().map_err(|_| "Invalid end")?;

        let group_a = fields.get(idx.group_a).ok_or("Missing group.a")?;
        let group_b = fields.get(idx.group_b).ok_or("Missing group.b")?;
        let identity: f64 = fields.get(idx.identity).ok_or("Missing identity")?
            .parse().map_err(|_| "Invalid identity")?;

        // Parse alignment lengths for coverage ratio
        let a_len: u64 = fields.get(idx.a_length).ok_or("Missing group.a.length")?
            .parse().map_err(|_| "Invalid group.a.length")?;
        let b_len: u64 = fields.get(idx.b_length).ok_or("Missing group.b.length")?
            .parse().map_err(|_| "Invalid group.b.length")?;

        let cov_ratio = if a_len == 0 && b_len == 0 {
            0.0
        } else {
            a_len.min(b_len) as f64 / a_len.max(b_len) as f64
        };

        let id_a = extract_sample_id(group_a);
        let id_b = extract_sample_id(group_b);

        let (query, reference) = if query_samples.contains(&id_a) && reference_haplotypes.contains(&id_b) {
            (id_a, id_b)
        } else if query_samples.contains(&id_b) && reference_haplotypes.contains(&id_a) {
            (id_b, id_a)
        } else {
            continue;
        };

        let (sims, covs) = sample_observations
            .entry(query)
            .or_default()
            .entry((chrom, start, end))
            .or_default();

        // Use max similarity and max coverage ratio if multiple alignments exist
        sims.entry(reference.clone())
            .and_modify(|existing| { if identity > *existing { *existing = identity; } })
            .or_insert(identity);
        covs.entry(reference)
            .and_modify(|existing| { if cov_ratio > *existing { *existing = cov_ratio; } })
            .or_insert(cov_ratio);
    }

    let mut result: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    for (sample, windows) in sample_observations {
        let mut obs_list: Vec<AncestryObservation> = windows
            .into_iter()
            .map(|((chrom, start, end), (sims, covs))| AncestryObservation {
                chrom,
                start,
                end,
                sample: sample.clone(),
                similarities: sims,
                coverage_ratios: Some(covs),
                haplotype_consistency_bonus: None,
            })
            .collect();

        obs_list.sort_by_key(|o| (o.chrom.clone(), o.start));
        result.insert(sample, obs_list);
    }

    Ok(result)
}

/// Compute coverage ratio from alignment lengths: min(a, b) / max(a, b)
pub fn coverage_ratio(a_len: u64, b_len: u64) -> f64 {
    if a_len == 0 && b_len == 0 {
        return 0.0;
    }
    a_len.min(b_len) as f64 / a_len.max(b_len) as f64
}

struct HeaderIndices {
    chrom: usize,
    start: usize,
    end: usize,
    group_a: usize,
    group_b: usize,
    identity: usize,
}

struct HeaderIndicesWithCoverage {
    chrom: usize,
    start: usize,
    end: usize,
    group_a: usize,
    group_b: usize,
    identity: usize,
    a_length: usize,
    b_length: usize,
}

fn parse_header(fields: &[&str], similarity_column: &str) -> Result<HeaderIndices, String> {
    let find = |name: &str| -> Result<usize, String> {
        fields.iter().position(|&f| f == name)
            .ok_or_else(|| format!("Missing column: {}", name))
    };

    Ok(HeaderIndices {
        chrom: find("chrom")?,
        start: find("start")?,
        end: find("end")?,
        group_a: find("group.a")?,
        group_b: find("group.b")?,
        identity: find(similarity_column)?,
    })
}

fn parse_header_with_coverage(fields: &[&str], similarity_column: &str) -> Result<HeaderIndicesWithCoverage, String> {
    let find = |name: &str| -> Result<usize, String> {
        fields.iter().position(|&f| f == name)
            .ok_or_else(|| format!("Missing column: {}", name))
    };

    Ok(HeaderIndicesWithCoverage {
        chrom: find("chrom")?,
        start: find("start")?,
        end: find("end")?,
        group_a: find("group.a")?,
        group_b: find("group.b")?,
        identity: find(similarity_column)?,
        a_length: find("group.a.length")?,
        b_length: find("group.b.length")?,
    })
}

/// Extract sample#haplotype ID from full ID with optional scaffold:coords suffix
fn extract_sample_id(full_id: &str) -> String {
    let parts: Vec<&str> = full_id.split('#').collect();
    if parts.len() >= 2 {
        format!("{}#{}", parts[0], parts[1])
    } else {
        full_id.to_string()
    }
}

/// Smooth state assignments by replacing short runs between longer runs of the same state.
/// This reduces noise from spurious short assignments.
///
/// # Arguments
/// * `states` - Original state sequence from Viterbi
/// * `min_run` - Minimum run length to keep; shorter runs between same-state neighbors are merged
///
/// # Example
/// With min_run=3: [0,0,0,0,1,1,0,0,0,0] -> [0,0,0,0,0,0,0,0,0,0]
/// The short run of 1s (length 2) is replaced because it's between 0s
pub fn smooth_states(states: &[usize], min_run: usize) -> Vec<usize> {
    if states.len() < 3 || min_run < 2 {
        return states.to_vec();
    }

    let mut smoothed = states.to_vec();

    // Find runs and smooth short ones
    let mut i = 0;
    while i < smoothed.len() {
        let current = smoothed[i];

        // Find end of current run
        let mut run_end = i + 1;
        while run_end < smoothed.len() && smoothed[run_end] == current {
            run_end += 1;
        }
        let run_len = run_end - i;

        // If run is short and surrounded by same state, merge it
        if run_len < min_run && i > 0 && run_end < smoothed.len() {
            let prev_state = smoothed[i - 1];
            let next_state = smoothed[run_end];

            if prev_state == next_state {
                // Replace short run with the surrounding state
                for item in smoothed.iter_mut().take(run_end).skip(i) {
                    *item = prev_state;
                }
            }
        }

        i = run_end;
    }

    smoothed
}

/// Posterior-aware state smoothing.
///
/// More sophisticated than `smooth_states`: uses posterior probabilities to
/// decide when to flip uncertain windows. Three smoothing passes:
///
/// 1. **Flank-consistent smoothing**: Short runs (< min_run) flanked by the
///    same state are merged if the window's posterior for the flanking state
///    exceeds `min_posterior_to_flip` (e.g., 0.2). This prevents merging
///    when the window truly has no evidence for the flanking state.
///
/// 2. **Low-confidence smoothing**: Windows with posterior < `confidence_threshold`
///    for their assigned state are flipped to the state with highest posterior,
///    but only if that state matches at least one neighbor (prevents isolated
///    assignments from noise).
///
/// 3. **Segment consolidation**: After the above passes, re-apply run-length
///    smoothing to merge any remaining short fragments.
pub fn posterior_smooth_states(
    states: &[usize],
    posteriors: &[Vec<f64>],
    min_run: usize,
    confidence_threshold: f64,
    min_posterior_to_flip: f64,
) -> Vec<usize> {
    if states.is_empty() || posteriors.is_empty() || states.len() != posteriors.len() {
        return states.to_vec();
    }
    if min_run < 2 {
        return states.to_vec();
    }

    let mut smoothed = states.to_vec();
    let n = smoothed.len();

    // Pass 1: Flank-consistent smoothing with posterior check
    let mut i = 0;
    while i < n {
        let current = smoothed[i];
        let mut run_end = i + 1;
        while run_end < n && smoothed[run_end] == current {
            run_end += 1;
        }
        let run_len = run_end - i;

        if run_len < min_run && i > 0 && run_end < n {
            let prev_state = smoothed[i - 1];
            let next_state = smoothed[run_end];

            if prev_state == next_state {
                // Check that at least some windows have non-trivial posterior
                // for the flanking state (prevents merging when flanking state
                // has zero evidence)
                let has_evidence = (i..run_end).any(|w| {
                    posteriors[w].len() > prev_state
                        && posteriors[w][prev_state] >= min_posterior_to_flip
                });

                if has_evidence {
                    for item in smoothed.iter_mut().take(run_end).skip(i) {
                        *item = prev_state;
                    }
                }
            }
        }

        i = run_end;
    }

    // Pass 2: Low-confidence windows — flip to neighbor-consistent best posterior
    for i in 0..n {
        if posteriors[i].len() <= smoothed[i] {
            continue;
        }
        let current_posterior = posteriors[i][smoothed[i]];
        if current_posterior < confidence_threshold {
            // Find the state with highest posterior
            let best_state = posteriors[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(idx, _)| idx)
                .unwrap_or(smoothed[i]);

            // Only flip if the best state matches at least one neighbor
            let matches_neighbor = (i > 0 && smoothed[i - 1] == best_state)
                || (i + 1 < n && smoothed[i + 1] == best_state);

            if matches_neighbor && best_state != smoothed[i] {
                smoothed[i] = best_state;
            }
        }
    }

    // Pass 3: Final run-length consolidation
    smooth_states(&smoothed, min_run)
}

/// Count how many state assignments changed during smoothing
pub fn count_smoothing_changes(original: &[usize], smoothed: &[usize]) -> usize {
    original.iter()
        .zip(smoothed.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Refined boundary position for an ancestry segment.
#[derive(Debug, Clone)]
pub struct RefinedAncestryBoundary {
    /// Refined start position in base pairs
    pub start_bp: u64,
    /// Refined end position in base pairs
    pub end_bp: u64,
}

/// Refine ancestry segment boundaries using posterior probability interpolation.
///
/// For each segment boundary, interpolates the sub-window position where the
/// posterior probability of the assigned ancestry state crosses the `crossover`
/// threshold (default 0.5) between adjacent windows. This improves boundary
/// resolution from ~10kb (window size) to ~2-5kb.
///
/// ## Algorithm
///
/// At the start of a segment assigned to state `s`:
/// - Look at `posteriors[i-1][s]` (low, previous state) and `posteriors[i][s]` (high, current state)
/// - If they straddle `crossover`, interpolate linearly between window centers
///
/// At the end of a segment:
/// - Look at `posteriors[j][s]` (high) and `posteriors[j+1][s]` (low, next state)
/// - Same interpolation logic
///
/// ## Arguments
///
/// - `segments`: Ancestry segments with start/end positions
/// - `posteriors`: Per-window posterior probabilities (shape: [n_windows][n_states])
/// - `observations`: AncestryObservation with window coordinates
/// - `crossover`: Posterior threshold for boundary placement (default: 0.5)
///
/// ## Returns
///
/// Vector of `RefinedAncestryBoundary` with one entry per input segment.
pub fn refine_ancestry_boundaries(
    segments: &[AncestrySegment],
    posteriors: &[Vec<f64>],
    observations: &[AncestryObservation],
    crossover: f64,
) -> Vec<RefinedAncestryBoundary> {
    let n = posteriors.len();

    if n == 0 || observations.len() != n {
        return segments
            .iter()
            .map(|seg| RefinedAncestryBoundary {
                start_bp: seg.start,
                end_bp: seg.end,
            })
            .collect();
    }

    let center = |idx: usize| -> f64 {
        (observations[idx].start as f64 + observations[idx].end as f64) / 2.0
    };

    // Build a mapping from (chrom, start, end) to window index for segment lookup
    let mut window_idx: HashMap<(u64, u64), usize> = HashMap::new();
    for (i, obs) in observations.iter().enumerate() {
        window_idx.insert((obs.start, obs.end), i);
    }

    segments
        .iter()
        .map(|seg| {
            let state = seg.ancestry_idx;

            // Find the window index for this segment's start and end
            let start_i = window_idx
                .get(&(seg.start, observations.first().map_or(0, |o| o.end)))
                .copied();
            // More robust: find by matching start position
            let start_i = start_i.or_else(|| {
                observations.iter().position(|o| o.start == seg.start)
            });
            let end_i = observations.iter().rposition(|o| o.end == seg.end);

            let (start_i, end_i) = match (start_i, end_i) {
                (Some(s), Some(e)) => (s, e),
                _ => {
                    return RefinedAncestryBoundary {
                        start_bp: seg.start,
                        end_bp: seg.end,
                    };
                }
            };

            // Refine start boundary
            let refined_start = if start_i > 0 && state < posteriors[start_i].len() {
                let p_before = posteriors[start_i - 1].get(state).copied().unwrap_or(0.0);
                let p_at = posteriors[start_i].get(state).copied().unwrap_or(1.0);

                if p_at > p_before && p_before < crossover && p_at > crossover {
                    let t = (crossover - p_before) / (p_at - p_before);
                    let c_before = center(start_i - 1);
                    let c_at = center(start_i);
                    let pos = c_before + t * (c_at - c_before);
                    pos.clamp(c_before, c_at).round() as u64
                } else {
                    seg.start
                }
            } else {
                seg.start
            };

            // Refine end boundary
            let refined_end = if end_i + 1 < n && state < posteriors[end_i].len() {
                let p_at = posteriors[end_i].get(state).copied().unwrap_or(1.0);
                let p_after = posteriors[end_i + 1].get(state).copied().unwrap_or(0.0);

                if p_at > p_after && p_after < crossover && p_at > crossover {
                    let t = (p_at - crossover) / (p_at - p_after);
                    let c_at = center(end_i);
                    let c_after = center(end_i + 1);
                    let pos = c_at + t * (c_after - c_at);
                    pos.clamp(c_at, c_after).round() as u64
                } else {
                    seg.end
                }
            } else {
                seg.end
            };

            RefinedAncestryBoundary {
                start_bp: refined_start,
                end_bp: refined_end.max(refined_start),
            }
        })
        .collect()
}

/// Load population sample IDs from a text file (one ID per line).
///
/// These files are used for human population sample lists (e.g., AFR.txt, EUR.txt)
/// where each line contains a sample ID like "HG01884".
pub fn load_population_samples(path: &Path) -> Result<Vec<String>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open population file {:?}: {}", path, e))?;
    let reader = BufReader::new(file);

    let samples: Vec<String> = reader.lines()
        .map_while(Result::ok)
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    Ok(samples)
}

/// Build ancestral populations from a directory of population sample files.
///
/// Each .txt file in the directory becomes a population, with the filename
/// (without extension) as the population name and its contents as haplotype IDs.
pub fn load_populations_from_dir(dir: &Path) -> Result<Vec<AncestralPopulation>, String> {
    let mut populations = Vec::new();

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {:?}: {}", dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();

        if path.extension().is_some_and(|ext| ext == "txt") {
            let pop_name = path.file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| format!("Invalid filename: {:?}", path))?
                .to_string();

            let haplotypes = load_population_samples(&path)?;
            if !haplotypes.is_empty() {
                populations.push(AncestralPopulation {
                    name: pop_name,
                    haplotypes,
                });
            }
        }
    }

    // Sort for deterministic ordering
    populations.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(populations)
}

/// Admixture proportion estimates for a single sample
#[derive(Debug, Clone)]
pub struct AdmixtureProportions {
    /// Sample ID
    pub sample: String,
    /// Per-population proportion of the genome (sums to 1.0)
    pub proportions: HashMap<String, f64>,
    /// Total genomic length analyzed (bp)
    pub total_length_bp: u64,
    /// Per-population total tract length (bp)
    pub tract_lengths_bp: HashMap<String, u64>,
    /// Number of ancestry switches (indicates admixture complexity)
    pub n_switches: usize,
    /// Mean tract length (bp) — longer tracts suggest more ancient admixture
    pub mean_tract_length_bp: f64,
}

/// Estimate admixture proportions from ancestry segments.
///
/// Computes the fraction of the genome assigned to each ancestral population
/// based on segment lengths. Also computes tract length statistics useful for
/// inferring admixture timing (longer tracts = more recent admixture).
///
/// # Arguments
/// * `segments` - Ancestry segments from HMM inference
/// * `sample` - Sample ID
/// * `pop_names` - List of all population names (for completeness even if 0%)
pub fn estimate_admixture_proportions(
    segments: &[AncestrySegment],
    sample: &str,
    pop_names: &[String],
) -> AdmixtureProportions {
    let mut tract_lengths_bp: HashMap<String, u64> = HashMap::new();
    let mut total_length_bp: u64 = 0;

    // Initialize all populations to 0
    for name in pop_names {
        tract_lengths_bp.insert(name.clone(), 0);
    }

    for seg in segments {
        let seg_len = seg.end.saturating_sub(seg.start);
        *tract_lengths_bp.entry(seg.ancestry_name.clone()).or_insert(0) += seg_len;
        total_length_bp += seg_len;
    }

    let proportions: HashMap<String, f64> = tract_lengths_bp.iter()
        .map(|(name, &len)| {
            let prop = if total_length_bp > 0 {
                len as f64 / total_length_bp as f64
            } else {
                0.0
            };
            (name.clone(), prop)
        })
        .collect();

    let n_switches = if segments.len() > 1 {
        segments.windows(2)
            .filter(|w| w[0].ancestry_name != w[1].ancestry_name)
            .count()
    } else {
        0
    };

    let mean_tract_length_bp = if !segments.is_empty() {
        total_length_bp as f64 / segments.len() as f64
    } else {
        0.0
    };

    AdmixtureProportions {
        sample: sample.to_string(),
        proportions,
        total_length_bp,
        tract_lengths_bp,
        n_switches,
        mean_tract_length_bp,
    }
}

/// Filter ancestry segments by minimum LOD score.
///
/// Removes segments whose LOD score is below the threshold.
/// A min_lod of 0.0 keeps all segments (no filtering).
pub fn filter_segments_by_min_lod(segments: Vec<AncestrySegment>, min_lod: f64) -> Vec<AncestrySegment> {
    segments.into_iter()
        .filter(|s| s.lod_score >= min_lod)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_sample_id() {
        assert_eq!(
            extract_sample_id("HG00097#1#h1tg000001l:0-5000"),
            "HG00097#1"
        );
        assert_eq!(
            extract_sample_id("popA#HAP1#scaffold73:14346-25666"),
            "popA#HAP1"
        );
        assert_eq!(
            extract_sample_id("popB#A"),
            "popB#A"
        );
    }

    #[test]
    fn test_smooth_states_basic() {
        // Short run of 1s between 0s should be smoothed
        let states = vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0];
        let smoothed = smooth_states(&states, 3);
        assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_smooth_states_keeps_long_runs() {
        // Long run should not be smoothed even if surrounded
        let states = vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0];
        let smoothed = smooth_states(&states, 3);
        assert_eq!(smoothed, vec![0, 0, 0, 1, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_smooth_states_different_neighbors() {
        // Short run between different states should not be smoothed
        let states = vec![0, 0, 1, 1, 2, 2];
        let smoothed = smooth_states(&states, 3);
        assert_eq!(smoothed, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_count_smoothing_changes() {
        let original = vec![0, 0, 1, 1, 0, 0];
        let smoothed = vec![0, 0, 0, 0, 0, 0];
        assert_eq!(count_smoothing_changes(&original, &smoothed), 2);
    }

    #[test]
    fn test_per_window_ancestry_lod_positive_for_correct() {
        // LOD should be positive when the assigned population has highest emission
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};

        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = AncestryObservation {
            chrom: "chr1".to_string(),
            start: 0,
            end: 5000,
            sample: "test".to_string(),
            similarities: [
                ("pop_a#1".to_string(), 0.95),
                ("pop_b#1".to_string(), 0.80),
            ].into(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        };

        // Assigned to pop_a (state 0) which has higher similarity
        let lod = compute_per_window_ancestry_lod(&obs, &params, 0);
        assert!(lod > 0.0, "LOD should be positive for correct assignment, got {}", lod);

        // Assigned to pop_b (state 1) which has lower similarity
        let lod_wrong = compute_per_window_ancestry_lod(&obs, &params, 1);
        assert!(lod_wrong < 0.0, "LOD should be negative for incorrect assignment, got {}", lod_wrong);
    }

    #[test]
    fn test_per_window_ancestry_lod_equal_sims() {
        // LOD should be ~0 when similarities are equal
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};

        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = AncestryObservation {
            chrom: "chr1".to_string(),
            start: 0,
            end: 5000,
            sample: "test".to_string(),
            similarities: [
                ("pop_a#1".to_string(), 0.90),
                ("pop_b#1".to_string(), 0.90),
            ].into(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        };

        let lod = compute_per_window_ancestry_lod(&obs, &params, 0);
        assert!(lod.abs() < 1e-6, "LOD should be ~0 for equal similarities, got {}", lod);
    }

    #[test]
    fn test_segment_ancestry_lod_sums_windows() {
        // Segment LOD should equal sum of per-window LODs
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};

        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops, 0.001);

        let obs = vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "test".to_string(),
                similarities: [("pop_a#1".to_string(), 0.95), ("pop_b#1".to_string(), 0.80)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
            AncestryObservation {
                chrom: "chr1".to_string(), start: 5000, end: 10000,
                sample: "test".to_string(),
                similarities: [("pop_a#1".to_string(), 0.92), ("pop_b#1".to_string(), 0.85)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
            AncestryObservation {
                chrom: "chr1".to_string(), start: 10000, end: 15000,
                sample: "test".to_string(),
                similarities: [("pop_a#1".to_string(), 0.88), ("pop_b#1".to_string(), 0.82)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];

        let seg_lod = segment_ancestry_lod(&obs, &params, 0, 0, 2);
        let sum_lod: f64 = obs.iter().map(|o| compute_per_window_ancestry_lod(o, &params, 0)).sum();

        assert!((seg_lod - sum_lod).abs() < 1e-10,
            "Segment LOD ({}) should equal sum of window LODs ({})", seg_lod, sum_lod);
        assert!(seg_lod > 0.0, "LOD for correct assignment should be positive");
    }

    #[test]
    fn test_admixture_proportions_single_ancestry() {
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 100000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 20, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 10.0,
            },
        ];
        let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];

        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);

        assert!((admix.proportions["pop_a"] - 1.0).abs() < 1e-10);
        assert!((admix.proportions["pop_b"] - 0.0).abs() < 1e-10);
        assert_eq!(admix.total_length_bp, 100000);
        assert_eq!(admix.n_switches, 0);
    }

    #[test]
    fn test_admixture_proportions_mixed() {
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 60000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 12, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 8.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 60000, end: 100000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 8, mean_similarity: 0.92,
                mean_posterior: Some(0.95), discriminability: 0.12,
                lod_score: 5.0,
            },
        ];
        let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];

        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);

        assert!((admix.proportions["pop_a"] - 0.6).abs() < 1e-10);
        assert!((admix.proportions["pop_b"] - 0.4).abs() < 1e-10);
        assert_eq!(admix.total_length_bp, 100000);
        assert_eq!(admix.n_switches, 1);
        assert!((admix.mean_tract_length_bp - 50000.0).abs() < 1e-10);
    }

    #[test]
    fn test_admixture_proportions_empty() {
        let segments: Vec<AncestrySegment> = vec![];
        let pop_names = vec!["pop_a".to_string()];

        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);

        assert!((admix.proportions["pop_a"] - 0.0).abs() < 1e-10);
        assert_eq!(admix.total_length_bp, 0);
        assert_eq!(admix.n_switches, 0);
        assert_eq!(admix.mean_tract_length_bp, 0.0);
    }

    #[test]
    fn test_admixture_proportions_three_way() {
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 30000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 6, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 5.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 30000, end: 60000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 6, mean_similarity: 0.93,
                mean_posterior: Some(0.96), discriminability: 0.13,
                lod_score: 4.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 60000, end: 90000,
                sample: "test".to_string(), ancestry_idx: 2,
                ancestry_name: "pop_c".to_string(),
                n_windows: 6, mean_similarity: 0.91,
                mean_posterior: Some(0.94), discriminability: 0.11,
                lod_score: 3.0,
            },
        ];
        let pop_names = vec!["pop_a".to_string(), "pop_b".to_string(), "pop_c".to_string()];

        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);

        // Each population gets exactly 1/3
        for name in &pop_names {
            assert!((admix.proportions[name] - 1.0 / 3.0).abs() < 1e-10,
                "{} proportion should be 1/3, got {}", name, admix.proportions[name]);
        }
        assert_eq!(admix.n_switches, 2);
        assert_eq!(admix.total_length_bp, 90000);
    }

    #[test]
    fn test_extract_segments_include_lod() {
        // Verify that extract_ancestry_segments populates LOD scores
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation, viterbi};

        let pops = vec![
            AncestralPopulation {
                name: "pop_a".to_string(),
                haplotypes: vec!["pop_a#1".to_string()],
            },
            AncestralPopulation {
                name: "pop_b".to_string(),
                haplotypes: vec!["pop_b#1".to_string()],
            },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);

        let obs = vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 5000,
                sample: "test".to_string(),
                similarities: [("pop_a#1".to_string(), 0.95), ("pop_b#1".to_string(), 0.80)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
            AncestryObservation {
                chrom: "chr1".to_string(), start: 5000, end: 10000,
                sample: "test".to_string(),
                similarities: [("pop_a#1".to_string(), 0.93), ("pop_b#1".to_string(), 0.82)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];

        let states = viterbi(&obs, &params);
        let segments = extract_ancestry_segments(&obs, &states, &params, None);

        assert!(!segments.is_empty());
        // LOD should be non-zero for segments with clear signal
        assert!(segments[0].lod_score.abs() > 0.0,
            "LOD score should be non-zero, got {}", segments[0].lod_score);
    }

    #[test]
    fn test_load_population_samples() {
        // Step 8: Test parsing population sample files
        let sample_path = std::path::Path::new("data/samples/AFR.txt");
        if sample_path.exists() {
            let samples = load_population_samples(sample_path).unwrap();
            assert!(!samples.is_empty(), "AFR.txt should contain samples");
            assert!(samples.len() > 10, "AFR.txt should have >10 samples, got {}", samples.len());
            // Verify sample IDs look reasonable (start with HG or NA typically)
            for s in &samples {
                assert!(!s.is_empty(), "Sample ID should not be empty");
                assert!(!s.contains('\t'), "Sample ID should not contain tabs");
            }
        }
    }

    #[test]
    fn test_load_populations_from_dir() {
        // Step 8: Test loading all population files from directory
        let samples_dir = std::path::Path::new("data/samples");
        if samples_dir.exists() {
            let pops = load_populations_from_dir(samples_dir).unwrap();
            assert!(pops.len() >= 3, "Should have at least 3 populations, got {}", pops.len());

            let pop_names: Vec<&str> = pops.iter().map(|p| p.name.as_str()).collect();
            assert!(pop_names.contains(&"AFR"), "Should contain AFR population");
            assert!(pop_names.contains(&"EUR"), "Should contain EUR population");

            for pop in &pops {
                assert!(!pop.haplotypes.is_empty(),
                    "Population {} should have haplotypes", pop.name);
            }
        }
    }

    #[test]
    fn test_load_population_samples_synthetic() {
        // Test with a synthetic temp file (doesn't depend on filesystem)
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_pop.txt");

        std::fs::write(&file_path, "sample1\nsample2\n\n# comment\nsample3\n").unwrap();

        let samples = load_population_samples(&file_path).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], "sample1");
        assert_eq!(samples[1], "sample2");
        assert_eq!(samples[2], "sample3");
    }

    #[test]
    fn test_min_lod_zero_keeps_all() {
        // min_lod=0.0 should keep all segments (default = no filtering)
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 50000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 0.5,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 50000, end: 100000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 10, mean_similarity: 0.90,
                mean_posterior: Some(0.85), discriminability: 0.10,
                lod_score: 1.2,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 100000, end: 150000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.92,
                mean_posterior: Some(0.90), discriminability: 0.12,
                lod_score: 3.0,
            },
        ];

        let filtered = filter_segments_by_min_lod(segments, 0.0);
        assert_eq!(filtered.len(), 3, "min_lod=0.0 should keep all segments");
    }

    #[test]
    fn test_min_lod_filters_weak_segments() {
        // min_lod=2.0 should remove segments with LOD < 2.0
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 50000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 5.0, // above threshold
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 50000, end: 100000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 10, mean_similarity: 0.90,
                mean_posterior: Some(0.85), discriminability: 0.10,
                lod_score: 0.5, // below threshold
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 100000, end: 150000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.92,
                mean_posterior: Some(0.90), discriminability: 0.12,
                lod_score: 3.0, // above threshold
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 150000, end: 200000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 10, mean_similarity: 0.88,
                mean_posterior: Some(0.70), discriminability: 0.08,
                lod_score: 1.9, // just below threshold
            },
        ];

        let filtered = filter_segments_by_min_lod(segments, 2.0);
        assert_eq!(filtered.len(), 2, "min_lod=2.0 should keep only 2 segments with LOD >= 2.0");
        assert_eq!(filtered[0].ancestry_name, "pop_a");
        assert!((filtered[0].lod_score - 5.0).abs() < 1e-10);
        assert_eq!(filtered[1].ancestry_name, "pop_a");
        assert!((filtered[1].lod_score - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_lod_very_high_removes_all() {
        // min_lod=100.0 should remove everything (no segment has LOD that high)
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 50000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.95,
                mean_posterior: Some(0.98), discriminability: 0.15,
                lod_score: 10.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 50000, end: 100000,
                sample: "test".to_string(), ancestry_idx: 1,
                ancestry_name: "pop_b".to_string(),
                n_windows: 10, mean_similarity: 0.90,
                mean_posterior: Some(0.85), discriminability: 0.10,
                lod_score: 25.0,
            },
        ];

        let filtered = filter_segments_by_min_lod(segments, 100.0);
        assert_eq!(filtered.len(), 0, "min_lod=100.0 should remove all segments");
    }

    // --- Coverage ratio tests ---

    #[test]
    fn test_coverage_ratio_equal_lengths() {
        assert!((coverage_ratio(1000, 1000) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_ratio_asymmetric() {
        let ratio = coverage_ratio(500, 1000);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_ratio_zero() {
        assert_eq!(coverage_ratio(0, 0), 0.0);
        assert_eq!(coverage_ratio(0, 100), 0.0);
        assert_eq!(coverage_ratio(100, 0), 0.0);
    }

    #[test]
    fn test_coverage_ratio_symmetry() {
        assert!((coverage_ratio(300, 500) - coverage_ratio(500, 300)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_similarity_data_with_coverage_basic() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity";
        let line1 = "chr20\t1\t10000\tQUERY#1#scaffold:1-10000\tREF_A#1#scaffold:1-10000\t9000\t8000\t7500\t0.80\t0.90\t0.85\t0.95";
        let line2 = "chr20\t1\t10000\tQUERY#1#scaffold:1-10000\tREF_B#1#scaffold:1-10000\t9000\t6000\t5500\t0.70\t0.80\t0.75\t0.88";

        let lines = vec![
            header.to_string(),
            line1.to_string(),
            line2.to_string(),
        ];

        let query = vec!["QUERY#1".to_string()];
        let refs = vec!["REF_A#1".to_string(), "REF_B#1".to_string()];

        let result = parse_similarity_data_with_coverage(
            lines.into_iter(), &query, &refs, "estimated.identity"
        ).unwrap();

        assert_eq!(result.len(), 1, "Should have 1 sample");
        let obs = &result["QUERY#1"];
        assert_eq!(obs.len(), 1, "Should have 1 window");

        // Check similarities
        assert!((obs[0].similarities["REF_A#1"] - 0.95).abs() < 1e-6);
        assert!((obs[0].similarities["REF_B#1"] - 0.88).abs() < 1e-6);

        // Check coverage ratios
        let covs = obs[0].coverage_ratios.as_ref().unwrap();
        // REF_A: min(9000, 8000) / max(9000, 8000) = 8000/9000 = 0.8889
        assert!((covs["REF_A#1"] - 8000.0 / 9000.0).abs() < 1e-4,
            "REF_A coverage ratio: {}", covs["REF_A#1"]);
        // REF_B: min(9000, 6000) / max(9000, 6000) = 6000/9000 = 0.6667
        assert!((covs["REF_B#1"] - 6000.0 / 9000.0).abs() < 1e-4,
            "REF_B coverage ratio: {}", covs["REF_B#1"]);
    }

    #[test]
    fn test_parse_with_coverage_missing_columns() {
        // Header without group.a.length/group.b.length should error
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let lines = vec![header.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data_with_coverage(
            lines.into_iter(), &query, &refs, "estimated.identity"
        );
        assert!(result.is_err(), "Should error on missing coverage columns");
    }

    // --- parse_similarity_data (without coverage) tests ---

    #[test]
    fn test_parse_similarity_data_basic() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let line1 = "chr1\t0\t10000\tQUERY#1#scaffold:0-10000\tREF_A#1#scaffold:0-10000\t0.95";
        let line2 = "chr1\t0\t10000\tQUERY#1#scaffold:0-10000\tREF_B#1#scaffold:0-10000\t0.88";

        let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
        let query = vec!["QUERY#1".to_string()];
        let refs = vec!["REF_A#1".to_string(), "REF_B#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();

        assert_eq!(result.len(), 1);
        let obs = &result["QUERY#1"];
        assert_eq!(obs.len(), 1);
        assert!((obs[0].similarities["REF_A#1"] - 0.95).abs() < 1e-6);
        assert!((obs[0].similarities["REF_B#1"] - 0.88).abs() < 1e-6);
        assert!(obs[0].coverage_ratios.is_none());
    }

    #[test]
    fn test_parse_similarity_data_multiple_windows() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let line1 = "chr1\t0\t10000\tQ#1#s:0-10000\tR#1#s:0-10000\t0.95";
        let line2 = "chr1\t10000\t20000\tQ#1#s:10000-20000\tR#1#s:10000-20000\t0.90";

        let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
        let obs = &result["Q#1"];
        assert_eq!(obs.len(), 2);
        // Should be sorted by position
        assert_eq!(obs[0].start, 0);
        assert_eq!(obs[1].start, 10000);
    }

    #[test]
    fn test_parse_similarity_data_reversed_pair() {
        // group.b is the query, group.a is the reference
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let line1 = "chr1\t0\t10000\tREF#1#s:0-10000\tQUERY#1#s:0-10000\t0.92";

        let lines = vec![header.to_string(), line1.to_string()];
        let query = vec!["QUERY#1".to_string()];
        let refs = vec!["REF#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result["QUERY#1"][0].similarities["REF#1"] - 0.92).abs() < 1e-6);
    }

    #[test]
    fn test_parse_similarity_data_skips_non_query_ref_pairs() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        // Both are references — should be skipped
        let line1 = "chr1\t0\t10000\tREF_A#1#s:0-10000\tREF_B#1#s:0-10000\t0.85";
        // Both are queries — should be skipped
        let line2 = "chr1\t0\t10000\tQ1#1#s:0-10000\tQ2#1#s:0-10000\t0.90";
        // Correct pair
        let line3 = "chr1\t0\t10000\tQ1#1#s:0-10000\tREF_A#1#s:0-10000\t0.95";

        let lines = vec![header.to_string(), line1.to_string(), line2.to_string(), line3.to_string()];
        let query = vec!["Q1#1".to_string(), "Q2#1".to_string()];
        let refs = vec!["REF_A#1".to_string(), "REF_B#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("Q1#1"));
        // Q2 should not appear because it was only paired with Q1 (another query)
        assert!(!result.contains_key("Q2#1"));
    }

    #[test]
    fn test_parse_similarity_data_keeps_max_of_duplicates() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let line1 = "chr1\t0\t10000\tQ#1#s:0-10000\tR#1#s:0-10000\t0.90";
        let line2 = "chr1\t0\t10000\tQ#1#s:0-10000\tR#1#s:0-10000\t0.95"; // higher

        let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
        assert!((result["Q#1"][0].similarities["R#1"] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_parse_similarity_data_empty_lines() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
        let lines = vec![header.to_string()]; // header only, no data
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_similarity_data_missing_column() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b"; // missing identity
        let lines = vec![header.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data(lines.into_iter(), &query, &refs);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_similarity_data_custom_column() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tjaccard.similarity\testimated.identity";
        let line1 = "chr1\t0\t10000\tQ#1#s:0-10000\tR#1#s:0-10000\t0.80\t0.95";

        let lines = vec![header.to_string(), line1.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        // Use jaccard.similarity instead of estimated.identity
        let result = parse_similarity_data_column(
            lines.into_iter(), &query, &refs, "jaccard.similarity"
        ).unwrap();
        assert!((result["Q#1"][0].similarities["R#1"] - 0.80).abs() < 1e-6);
    }

    // --- extract_ancestry_segments edge cases ---

    #[test]
    fn test_extract_segments_empty_observations() {
        use crate::hmm::{AncestryHmmParams, AncestralPopulation};
        let pops = vec![
            AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);
        let result = extract_ancestry_segments(&[], &[], &params, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_segments_single_window() {
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};
        let pops = vec![
            AncestralPopulation { name: "pop_a".to_string(), haplotypes: vec!["pop_a#1".to_string()] },
            AncestralPopulation { name: "pop_b".to_string(), haplotypes: vec!["pop_b#1".to_string()] },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);
        let obs = vec![AncestryObservation {
            chrom: "chr1".to_string(), start: 100, end: 200,
            sample: "test".to_string(),
            similarities: [("pop_a#1".to_string(), 0.95), ("pop_b#1".to_string(), 0.80)].into(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }];
        let states = vec![0];
        let segments = extract_ancestry_segments(&obs, &states, &params, None);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 100);
        assert_eq!(segments[0].end, 200);
        assert_eq!(segments[0].ancestry_idx, 0);
        assert_eq!(segments[0].n_windows, 1);
    }

    #[test]
    fn test_extract_segments_alternating_states() {
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};
        let pops = vec![
            AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
            AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);

        let obs: Vec<AncestryObservation> = (0..5).map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: i * 1000,
            end: (i + 1) * 1000,
            sample: "test".to_string(),
            similarities: [("A#1".to_string(), 0.9), ("B#1".to_string(), 0.8)].into(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }).collect();

        // Alternating: A, B, A, B, A → 5 segments
        let states = vec![0, 1, 0, 1, 0];
        let segments = extract_ancestry_segments(&obs, &states, &params, None);
        assert_eq!(segments.len(), 5);
        for (i, seg) in segments.iter().enumerate() {
            assert_eq!(seg.n_windows, 1);
            assert_eq!(seg.ancestry_idx, i % 2);
        }
    }

    #[test]
    fn test_extract_segments_with_posteriors() {
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};
        let pops = vec![
            AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
            AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);

        let obs = vec![
            AncestryObservation {
                chrom: "chr1".to_string(), start: 0, end: 1000,
                sample: "test".to_string(),
                similarities: [("A#1".to_string(), 0.9), ("B#1".to_string(), 0.8)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
            AncestryObservation {
                chrom: "chr1".to_string(), start: 1000, end: 2000,
                sample: "test".to_string(),
                similarities: [("A#1".to_string(), 0.9), ("B#1".to_string(), 0.8)].into(),
                coverage_ratios: None,
                haplotype_consistency_bonus: None,
            },
        ];
        let states = vec![0, 0];
        let posteriors = vec![vec![0.9, 0.1], vec![0.8, 0.2]];
        let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
        assert_eq!(segments.len(), 1);
        let mean_post = segments[0].mean_posterior.unwrap();
        // Mean of posteriors for state 0: (0.9 + 0.8) / 2 = 0.85
        assert!((mean_post - 0.85).abs() < 1e-10, "Expected 0.85, got {}", mean_post);
    }

    // --- smooth_states edge cases ---

    #[test]
    fn test_smooth_states_empty() {
        assert_eq!(smooth_states(&[], 3), Vec::<usize>::new());
    }

    #[test]
    fn test_smooth_states_single_element() {
        assert_eq!(smooth_states(&[0], 3), vec![0]);
    }

    #[test]
    fn test_smooth_states_two_elements() {
        assert_eq!(smooth_states(&[0, 1], 3), vec![0, 1]);
    }

    #[test]
    fn test_smooth_states_min_run_1_no_smoothing() {
        // min_run < 2 means no smoothing
        let states = vec![0, 0, 1, 0, 0];
        assert_eq!(smooth_states(&states, 1), states);
    }

    #[test]
    fn test_smooth_states_at_boundary() {
        // Run at start or end shouldn't be smoothed (no neighbor on one side)
        let states = vec![1, 0, 0, 0, 0, 0, 1];
        let smoothed = smooth_states(&states, 3);
        // Single 1 at start: prev doesn't exist, so no smoothing
        // Single 1 at end: next doesn't exist, so no smoothing
        assert_eq!(smoothed, vec![1, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_smooth_states_multiple_short_runs() {
        // Multiple short runs between same state
        let states = vec![0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
        let smoothed = smooth_states(&states, 3);
        // Each single 1 is between 0s → should be smoothed to 0
        assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // --- count_smoothing_changes edge cases ---

    #[test]
    fn test_count_smoothing_changes_identical() {
        let v = vec![0, 1, 2, 1, 0];
        assert_eq!(count_smoothing_changes(&v, &v), 0);
    }

    #[test]
    fn test_count_smoothing_changes_empty() {
        assert_eq!(count_smoothing_changes(&[], &[]), 0);
    }

    // --- filter_segments_by_min_lod edge cases ---

    #[test]
    fn test_filter_segments_negative_lod() {
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 50000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 10, mean_similarity: 0.95,
                mean_posterior: None, discriminability: 0.15,
                lod_score: -2.0, // negative LOD
            },
        ];
        let filtered = filter_segments_by_min_lod(segments, 0.0);
        assert_eq!(filtered.len(), 0, "Negative LOD should be filtered at threshold 0.0");
    }

    #[test]
    fn test_filter_segments_empty_input() {
        let segments: Vec<AncestrySegment> = vec![];
        let filtered = filter_segments_by_min_lod(segments, 5.0);
        assert!(filtered.is_empty());
    }

    // --- LOD with missing similarities ---

    #[test]
    fn test_per_window_lod_single_population() {
        use crate::hmm::{AncestryHmmParams, AncestralPopulation, AncestryObservation};
        let pops = vec![
            AncestralPopulation { name: "only".to_string(), haplotypes: vec!["only#1".to_string()] },
        ];
        let params = AncestryHmmParams::new(pops, 0.01);
        let obs = AncestryObservation {
            chrom: "chr1".to_string(), start: 0, end: 5000,
            sample: "test".to_string(),
            similarities: [("only#1".to_string(), 0.95)].into(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        };
        // With single population, no second-best exists → LOD should handle gracefully
        let lod = compute_per_window_ancestry_lod(&obs, &params, 0);
        // NEG_INFINITY for second_best → result should be 0.0 (non-finite guard)
        assert!(lod.is_finite(), "LOD should be finite with single population");
    }

    // --- admixture proportions edge cases ---

    #[test]
    fn test_admixture_proportions_zero_length_segment() {
        // Segment where start == end (degenerate)
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 5000, end: 5000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "pop_a".to_string(),
                n_windows: 0, mean_similarity: 0.0,
                mean_posterior: None, discriminability: 0.0,
                lod_score: 0.0,
            },
        ];
        let pop_names = vec!["pop_a".to_string()];
        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);
        assert_eq!(admix.total_length_bp, 0);
        assert_eq!(admix.n_switches, 0);
    }

    #[test]
    fn test_admixture_proportions_unlisted_ancestry() {
        // Segment with ancestry_name not in pop_names (should still work)
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 50000,
                sample: "test".to_string(), ancestry_idx: 0,
                ancestry_name: "unknown_pop".to_string(),
                n_windows: 10, mean_similarity: 0.9,
                mean_posterior: None, discriminability: 0.1,
                lod_score: 2.0,
            },
        ];
        let pop_names = vec!["pop_a".to_string(), "pop_b".to_string()];
        let admix = estimate_admixture_proportions(&segments, "test", &pop_names);
        // The unknown_pop should be tracked via or_insert
        assert_eq!(admix.total_length_bp, 50000);
        assert!(admix.tract_lengths_bp.contains_key("unknown_pop"));
    }

    // --- extract_sample_id edge cases ---

    #[test]
    fn test_extract_sample_id_no_hash() {
        assert_eq!(extract_sample_id("simple_name"), "simple_name");
    }

    #[test]
    fn test_extract_sample_id_empty() {
        assert_eq!(extract_sample_id(""), "");
    }

    #[test]
    fn test_extract_sample_id_hash_only() {
        // "#1" → parts[0]="" parts[1]="1" → "#1"
        let result = extract_sample_id("#1");
        assert_eq!(result, "#1");
    }

    #[test]
    fn test_parse_with_coverage_max_of_duplicates() {
        let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity";
        // Two alignments for same pair, different coverages
        let line1 = "chr20\t1\t10000\tQ#1#s:1-10000\tR#1#s:1-10000\t8000\t7000\t6000\t0.7\t0.8\t0.75\t0.90";
        let line2 = "chr20\t1\t10000\tQ#1#s:1-10000\tR#1#s:1-10000\t9000\t8500\t8000\t0.8\t0.9\t0.85\t0.95";

        let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
        let query = vec!["Q#1".to_string()];
        let refs = vec!["R#1".to_string()];

        let result = parse_similarity_data_with_coverage(
            lines.into_iter(), &query, &refs, "estimated.identity"
        ).unwrap();

        let obs = &result["Q#1"][0];
        // Should keep max similarity (0.95)
        assert!((obs.similarities["R#1"] - 0.95).abs() < 1e-6);
        // Should keep max coverage ratio: max(7000/8000=0.875, 8500/9000=0.944) = 0.944
        let covs = obs.coverage_ratios.as_ref().unwrap();
        assert!((covs["R#1"] - 8500.0 / 9000.0).abs() < 1e-3,
            "Should keep max coverage ratio: {}", covs["R#1"]);
    }
}
