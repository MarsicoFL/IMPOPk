//! Ancestry segment extraction and data processing

use std::collections::HashMap;
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
    }
}

/// Parse similarity data from TSV into AncestryObservations
///
/// Expected format (from impg similarity):
/// chrom, start, end, group.a, group.b, ..., estimated.identity
///
/// Groups sample vs reference observations
pub fn parse_similarity_data(
    lines: impl Iterator<Item = String>,
    query_samples: &[String],
    reference_haplotypes: &[String],
) -> Result<HashMap<String, Vec<AncestryObservation>>, String> {
    let mut header_indices: Option<HeaderIndices> = None;
    let mut sample_observations: HashMap<String, HashMap<(String, u64, u64), HashMap<String, f64>>> = HashMap::new();

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();

        if header_indices.is_none() {
            // Parse header
            header_indices = Some(parse_header(&fields)?);
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
            })
            .collect();

        // Sort by position
        obs_list.sort_by_key(|o| (o.chrom.clone(), o.start));

        result.insert(sample, obs_list);
    }

    Ok(result)
}

struct HeaderIndices {
    chrom: usize,
    start: usize,
    end: usize,
    group_a: usize,
    group_b: usize,
    identity: usize,
}

fn parse_header(fields: &[&str]) -> Result<HeaderIndices, String> {
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
        identity: find("estimated.identity")?,
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
                for j in i..run_end {
                    smoothed[j] = prev_state;
                }
            }
        }

        i = run_end;
    }

    smoothed
}

/// Count how many state assignments changed during smoothing
pub fn count_smoothing_changes(original: &[usize], smoothed: &[usize]) -> usize {
    original.iter()
        .zip(smoothed.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Define bat populations for Glossophaga case study
pub fn glossophaga_populations() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "commissarisi".to_string(),
            haplotypes: vec![
                "commissarisi#HAP1".to_string(),
                "commissarisi#HAP2".to_string(),
            ],
        },
        AncestralPopulation {
            name: "mutica".to_string(),
            haplotypes: vec![
                "mutica#A".to_string(),
                "mutica#B".to_string(),
            ],
        },
        AncestralPopulation {
            name: "soricina".to_string(),
            haplotypes: vec![
                "soricina#HAP1".to_string(),
                "soricina#HAP2".to_string(),
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_sample_id() {
        assert_eq!(
            extract_sample_id("TBG_5116#1#h1tg000001l:0-5000"),
            "TBG_5116#1"
        );
        assert_eq!(
            extract_sample_id("commissarisi#HAP1#scaffold73:14346-25666"),
            "commissarisi#HAP1"
        );
        assert_eq!(
            extract_sample_id("mutica#A"),
            "mutica#A"
        );
    }

    #[test]
    fn test_glossophaga_populations() {
        let pops = glossophaga_populations();
        assert_eq!(pops.len(), 3);
        assert_eq!(pops[0].name, "commissarisi");
        assert_eq!(pops[0].haplotypes.len(), 2);
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
}
