//! Spatial index over TPA records for regional queries.
//!
//! Maps (chromosome, genomic_bin) → Vec<record_id> for fast lookups.
//! Built by streaming TPA metadata (skips tracepoints for speed).

use crate::paf::extract_target_chrom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tpa::TpaReader;

/// Spatial index: maps (chrom, bin_index) → record IDs.
#[derive(Serialize, Deserialize)]
pub struct SpatialIndex {
    pub bin_size: u64,
    /// (chrom_string, bin_index) → Vec<record_id as u64>
    bins: HashMap<(String, u64), Vec<u64>>,
    /// Total records indexed
    pub total_records: u64,
    /// Number of unique chromosomes
    pub num_chroms: usize,
}

impl SpatialIndex {
    /// Build spatial index by streaming TPA metadata.
    pub fn build(tpa_path: &str, bin_size: u64) -> anyhow::Result<Self> {
        let mut reader = TpaReader::new(tpa_path)?;
        reader.load_string_table()?;
        let string_table = reader.string_table_ref().clone();
        let num_records = reader.num_records();

        let mut bins: HashMap<(String, u64), Vec<u64>> = HashMap::new();
        let mut chroms = std::collections::HashSet::new();

        // Stream through all records — reading only metadata (coordinates)
        // We need compact records to get target_name_id + coordinates
        for record_id in 0..num_records as u64 {
            let record = reader.get_compact_record(record_id)?;

            // Resolve target name
            let target_name = match string_table.get(record.target_name_id) {
                Some(name) => name,
                None => continue,
            };

            let chrom = extract_target_chrom(target_name).to_string();
            chroms.insert(chrom.clone());

            let start_bin = record.target_start / bin_size;
            let end_bin = record.target_end.saturating_sub(1) / bin_size;

            for bin in start_bin..=end_bin {
                bins.entry((chrom.clone(), bin))
                    .or_default()
                    .push(record_id);
            }

            if record_id > 0 && record_id % 100_000 == 0 {
                eprintln!("  Indexed {} / {} records...", record_id, num_records);
            }
        }

        eprintln!(
            "Spatial index: {} records, {} chromosomes, {} bins",
            num_records,
            chroms.len(),
            bins.len()
        );

        Ok(SpatialIndex {
            bin_size,
            bins,
            total_records: num_records as u64,
            num_chroms: chroms.len(),
        })
    }

    /// Query: get all record IDs overlapping a region, deduplicated.
    pub fn query(&self, chrom: &str, start: u64, end: u64) -> Vec<u64> {
        let start_bin = start / self.bin_size;
        let end_bin = end.saturating_sub(1) / self.bin_size;

        let mut ids: Vec<u64> = Vec::new();
        for bin in start_bin..=end_bin {
            if let Some(bin_ids) = self.bins.get(&(chrom.to_string(), bin)) {
                ids.extend_from_slice(bin_ids);
            }
        }

        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Save to disk with bincode.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load from disk.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        if !Path::new(path).exists() {
            anyhow::bail!("Spatial index not found: {}", path);
        }
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let index: SpatialIndex = bincode::deserialize_from(reader)?;
        Ok(index)
    }
}
