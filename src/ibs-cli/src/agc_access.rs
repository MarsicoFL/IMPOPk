//! AGC (Assembled Genomes Compressor) sequence access.
//!
//! Adapted from cigzip's AgcIndex. Provides per-thread decompressor pools
//! for concurrent sequence fetching from AGC archives.

use ragc_core::{Decompressor, DecompressorConfig};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct AgcIndex {
    agc_paths: Vec<String>,
    /// (contig_name) or (contig@sample) -> agc file index
    sample_contig_to_agc: HashMap<String, usize>,
    /// contig_name -> (sample, full_contig_name, agc_idx)
    contig_to_sample_info: HashMap<String, (String, String, usize)>,
    /// Per-thread decompressor pools, lazily initialized.
    thread_decompressors: Vec<Mutex<Option<Vec<Decompressor>>>>,
}

impl AgcIndex {
    pub fn build(agc_files: &[String]) -> Result<Self, String> {
        if agc_files.is_empty() {
            return Err("No AGC files provided".to_string());
        }

        let mut index = AgcIndex {
            agc_paths: Vec::new(),
            sample_contig_to_agc: HashMap::new(),
            contig_to_sample_info: HashMap::new(),
            thread_decompressors: Vec::new(),
        };

        // Extract metadata from each AGC file in parallel
        let metadata_results: Vec<_> = agc_files
            .par_iter()
            .enumerate()
            .map(|(agc_idx, agc_path)| {
                let config = DecompressorConfig { verbosity: 0 };
                let mut decompressor = Decompressor::open(agc_path, config)
                    .map_err(|e| format!("Failed to open AGC '{}': {}", agc_path, e))?;

                let samples = decompressor.list_samples();
                let sample_contigs: Vec<_> = samples
                    .into_iter()
                    .map(|sample| {
                        let contigs = decompressor.list_contigs(&sample).unwrap_or_default();
                        (sample, contigs)
                    })
                    .collect();

                Ok((agc_idx, agc_path.clone(), sample_contigs))
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Assemble index sequentially
        for (agc_idx, agc_path, sample_contigs) in metadata_results {
            index.agc_paths.push(agc_path);

            for (sample, contigs) in sample_contigs {
                for contig in contigs {
                    let key = format!("{}@{}", contig, sample);
                    index.sample_contig_to_agc.insert(key, agc_idx);

                    index
                        .sample_contig_to_agc
                        .entry(contig.clone())
                        .or_insert(agc_idx);

                    let sample_info = (sample.clone(), contig.clone(), agc_idx);
                    index
                        .contig_to_sample_info
                        .entry(contig.clone())
                        .or_insert(sample_info.clone());

                    // Short contig name (before whitespace)
                    let short = contig.split_whitespace().next().unwrap_or(&contig);
                    if short != contig {
                        let short_key = format!("{}@{}", short, sample);
                        index
                            .sample_contig_to_agc
                            .entry(short_key)
                            .or_insert(agc_idx);
                        index
                            .sample_contig_to_agc
                            .entry(short.to_string())
                            .or_insert(agc_idx);
                        index
                            .contig_to_sample_info
                            .entry(short.to_string())
                            .or_insert(sample_info);
                    }
                }
            }
        }

        // Initialize per-thread slots (+1 for main thread fallback)
        let num_slots = rayon::current_num_threads() + 1;
        index.thread_decompressors = (0..num_slots).map(|_| Mutex::new(None)).collect();

        index.sample_contig_to_agc.shrink_to_fit();
        index.contig_to_sample_info.shrink_to_fit();

        Ok(index)
    }

    /// Number of unique contigs indexed.
    pub fn num_contigs(&self) -> usize {
        self.contig_to_sample_info.len()
    }

    fn parse_query(&self, seq_name: &str) -> (String, String, Option<usize>) {
        if let Some((contig, sample)) = seq_name.split_once('@') {
            let agc_idx = self.sample_contig_to_agc.get(seq_name).copied();
            (sample.to_string(), contig.to_string(), agc_idx)
        } else if let Some((sample, full_contig, agc_idx)) =
            self.contig_to_sample_info.get(seq_name)
        {
            (sample.clone(), full_contig.clone(), Some(*agc_idx))
        } else {
            (String::new(), seq_name.to_string(), None)
        }
    }

    /// Fetch a subsequence from AGC. Thread-safe via per-thread decompressors.
    pub fn fetch_sequence(
        &self,
        seq_name: &str,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>, String> {
        if start >= end {
            return Ok(Vec::new());
        }

        let (sample, contig, agc_idx) = self.parse_query(seq_name);
        let agc_idx =
            agc_idx.ok_or_else(|| format!("Sequence '{}' not found in AGC", seq_name))?;

        let thread_idx =
            rayon::current_thread_index().unwrap_or(self.thread_decompressors.len() - 1);
        let mut slot = self.thread_decompressors[thread_idx].lock().unwrap();
        let decomps = slot.get_or_insert_with(|| {
            self.agc_paths
                .iter()
                .map(|path| {
                    Decompressor::open(path, DecompressorConfig { verbosity: 0 })
                        .unwrap_or_else(|e| panic!("Failed to open AGC '{}': {}", path, e))
                })
                .collect()
        });

        let sequence = decomps[agc_idx]
            .get_contig_range(&sample, &contig, start, end)
            .map_err(|e| {
                format!(
                    "Failed to fetch '{}@{}:{}:{}': {}",
                    contig, sample, start, end, e
                )
            })?;

        // Convert numeric encoding (0-3) to ASCII
        Ok(sequence
            .into_iter()
            .map(|b| match b {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => b'N',
            })
            .collect())
    }
}
