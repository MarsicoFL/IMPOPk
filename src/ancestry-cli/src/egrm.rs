//! eGRM (expected Genetic Relationship Matrix) output from pairwise identity.
//!
//! The per-window identity matrix is an affine transformation of the Branch GRM:
//! eGRM_impopk = (1 - I) / (2μ), where I is average identity and μ is mutation rate.
//! This module outputs the identity matrix in GCTA binary format for use with
//! GWAS tools (GCTA, BOLT-LMM, SAIGE), PCA, and heritability estimation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// Pairwise identity accumulator for building the GRM.
/// For each ordered pair (i, j) where i >= j, stores sum of identities and count.
#[derive(Default)]
pub struct PairwiseAccumulator {
    /// Map from sample name to index
    sample_to_idx: HashMap<String, usize>,
    /// Ordered sample names
    sample_names: Vec<String>,
    /// Lower triangle: identity_sum[i*(i+1)/2 + j] for i >= j
    identity_sum: Vec<f64>,
    /// Lower triangle: count[i*(i+1)/2 + j] for i >= j
    count: Vec<u32>,
}

impl PairwiseAccumulator {
    fn new() -> Self {
        Self::default()
    }

    fn get_or_insert_idx(&mut self, sample: &str) -> usize {
        if let Some(&idx) = self.sample_to_idx.get(sample) {
            return idx;
        }
        let idx = self.sample_names.len();
        self.sample_to_idx.insert(sample.to_string(), idx);
        self.sample_names.push(sample.to_string());

        // Expand storage for new sample: need idx+1 diagonal entries
        let new_size = (idx + 1) * (idx + 2) / 2;
        self.identity_sum.resize(new_size, 0.0);
        self.count.resize(new_size, 0);

        idx
    }

    fn add(&mut self, sample_a: &str, sample_b: &str, identity: f64) {
        let idx_a = self.get_or_insert_idx(sample_a);
        let idx_b = self.get_or_insert_idx(sample_b);

        // Store in lower triangle: ensure i >= j
        let (i, j) = if idx_a >= idx_b {
            (idx_a, idx_b)
        } else {
            (idx_b, idx_a)
        };

        let pos = i * (i + 1) / 2 + j;
        self.identity_sum[pos] += identity;
        self.count[pos] += 1;
    }

    fn n_samples(&self) -> usize {
        self.sample_names.len()
    }

    /// Get average identity for pair (i, j). Returns None if no data.
    fn get_average(&self, i: usize, j: usize) -> Option<f64> {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        let pos = ii * (ii + 1) / 2 + jj;
        if self.count[pos] > 0 {
            Some(self.identity_sum[pos] / self.count[pos] as f64)
        } else {
            None
        }
    }

    /// Get window count for pair (i, j).
    fn get_count(&self, i: usize, j: usize) -> u32 {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        let pos = ii * (ii + 1) / 2 + jj;
        self.count[pos]
    }
}

/// Extract sample ID from a full group identifier (e.g., "HG00733#1#chr12:1000-2000" → "HG00733#1").
fn extract_sample_id(full_id: &str) -> String {
    let parts: Vec<&str> = full_id.split('#').collect();
    if parts.len() >= 2 {
        format!("{}#{}", parts[0], parts[1])
    } else {
        full_id.to_string()
    }
}

/// Parse a similarity file and accumulate pairwise identity for ALL sample pairs.
///
/// Unlike `parse_similarity_data` which filters to query-vs-reference pairs only,
/// this function collects identity between all pairs of samples for eGRM output.
pub fn parse_similarity_for_egrm(
    sim_file: &Path,
    similarity_column: &str,
) -> Result<PairwiseAccumulator> {
    let file = File::open(sim_file)
        .with_context(|| format!("Failed to open similarity file: {:?}", sim_file))?;
    let reader = BufReader::new(file);

    let mut acc = PairwiseAccumulator::new();
    let mut col_group_a = None;
    let mut col_group_b = None;
    let mut col_identity = None;
    let mut header_parsed = false;

    for line in reader.lines() {
        let line = line.context("Failed to read line from similarity file")?;
        let fields: Vec<&str> = line.split('\t').collect();

        if !header_parsed {
            // Parse header
            for (i, &field) in fields.iter().enumerate() {
                match field {
                    "group.a" => col_group_a = Some(i),
                    "group.b" => col_group_b = Some(i),
                    f if f == similarity_column => col_identity = Some(i),
                    _ => {}
                }
            }
            if col_group_a.is_none() || col_group_b.is_none() || col_identity.is_none() {
                anyhow::bail!(
                    "Similarity file missing required columns. Need group.a, group.b, {}",
                    similarity_column
                );
            }
            header_parsed = true;
            continue;
        }

        let group_a = fields.get(col_group_a.unwrap()).context("Missing group.a")?;
        let group_b = fields.get(col_group_b.unwrap()).context("Missing group.b")?;
        let identity: f64 = fields
            .get(col_identity.unwrap())
            .context("Missing identity")?
            .parse()
            .context("Invalid identity value")?;

        let id_a = extract_sample_id(group_a);
        let id_b = extract_sample_id(group_b);

        // Skip self-comparisons within same haplotype
        if id_a == id_b {
            continue;
        }

        acc.add(&id_a, &id_b, identity);
    }

    Ok(acc)
}

/// Write the pairwise identity matrix in GCTA binary GRM format.
///
/// Produces three files:
/// - `{prefix}.grm.bin` — Lower triangle of N×N matrix, float32 (identity values)
/// - `{prefix}.grm.N.bin` — Number of windows per pair, float32
/// - `{prefix}.grm.id` — Sample identifiers (FID\tIID)
///
/// If `center` is true, subtracts the grand mean from all entries so diagonal
/// values represent excess relatedness above population average.
pub fn write_gcta_grm(
    acc: &PairwiseAccumulator,
    output_prefix: &str,
    center: bool,
) -> Result<()> {
    let n = acc.n_samples();
    if n == 0 {
        anyhow::bail!("No samples found for eGRM output");
    }

    // Compute grand mean if centering
    let grand_mean = if center {
        let mut sum = 0.0;
        let mut count = 0u64;
        for i in 0..n {
            for j in 0..=i {
                if let Some(val) = acc.get_average(i, j) {
                    sum += val;
                    count += 1;
                }
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    } else {
        0.0
    };

    // Write .grm.bin — lower triangle, float32
    let bin_path = format!("{}.grm.bin", output_prefix);
    let mut bin_file = BufWriter::new(
        File::create(&bin_path).with_context(|| format!("Failed to create {}", bin_path))?,
    );

    for i in 0..n {
        for j in 0..=i {
            let val = acc.get_average(i, j).unwrap_or(0.0) - grand_mean;
            bin_file.write_all(&(val as f32).to_le_bytes())?;
        }
    }
    bin_file.flush()?;

    // Write .grm.N.bin — window counts, float32
    let n_path = format!("{}.grm.N.bin", output_prefix);
    let mut n_file = BufWriter::new(
        File::create(&n_path).with_context(|| format!("Failed to create {}", n_path))?,
    );

    for i in 0..n {
        for j in 0..=i {
            let count = acc.get_count(i, j) as f32;
            n_file.write_all(&count.to_le_bytes())?;
        }
    }
    n_file.flush()?;

    // Write .grm.id — sample identifiers (FID\tIID)
    let id_path = format!("{}.grm.id", output_prefix);
    let mut id_file = BufWriter::new(
        File::create(&id_path).with_context(|| format!("Failed to create {}", id_path))?,
    );

    for name in &acc.sample_names {
        // Split sample#hap into FID (sample) and IID (sample#hap)
        let fid = name.split('#').next().unwrap_or(name);
        writeln!(id_file, "{}\t{}", fid, name)?;
    }
    id_file.flush()?;

    eprintln!(
        "eGRM written: {} samples, {} values, prefix={}{}",
        n,
        n * (n + 1) / 2,
        output_prefix,
        if center {
            format!(" (centered, grand_mean={:.6})", grand_mean)
        } else {
            String::new()
        }
    );

    Ok(())
}

/// Extract base individual ID from a haplotype ID (e.g., "HG00733#1" → "HG00733").
fn extract_individual_id(hap_id: &str) -> &str {
    hap_id.split('#').next().unwrap_or(hap_id)
}

/// Write a diploid GRM by averaging the 4 haplotype comparisons per individual pair.
///
/// For diploid individual i with haplotypes (h_i1, h_i2) and individual j with
/// haplotypes (h_j1, h_j2):
///   G_dip[i,j] = 1/4 × Σ_αβ I(h_iα, h_jβ)
///
/// Self-identity I(h,h) = 1.0 is imputed for diagonal computation:
///   G_dip[i,i] = 1/2 × (1 + I(h_i1, h_i2))
///
/// If `double_center` is true, applies Gower double-centering:
///   G̃[i,j] = G[i,j] - G[i,·] - G[·,j] + G[·,·]
/// This removes population structure bias and is required for REML heritability.
pub fn write_diploid_gcta_grm(
    acc: &PairwiseAccumulator,
    output_prefix: &str,
    double_center: bool,
) -> Result<()> {
    let n_haps = acc.n_samples();
    if n_haps == 0 {
        anyhow::bail!("No samples found for diploid eGRM output");
    }

    // Group haplotypes by individual
    let mut indiv_haps: Vec<(String, Vec<usize>)> = Vec::new();
    let mut indiv_map: HashMap<String, usize> = HashMap::new();

    for (idx, name) in acc.sample_names.iter().enumerate() {
        let indiv = extract_individual_id(name).to_string();
        if let Some(&indiv_idx) = indiv_map.get(&indiv) {
            indiv_haps[indiv_idx].1.push(idx);
        } else {
            let indiv_idx = indiv_haps.len();
            indiv_map.insert(indiv.clone(), indiv_idx);
            indiv_haps.push((indiv, vec![idx]));
        }
    }

    let n_dip = indiv_haps.len();
    if n_dip == 0 {
        anyhow::bail!("No diploid individuals found");
    }

    // Compute diploid GRM (n_dip × n_dip, stored as lower triangle)
    let grm_size = n_dip * (n_dip + 1) / 2;
    let mut grm = vec![0.0_f64; grm_size];
    let mut grm_counts = vec![0.0_f32; grm_size];

    for (i, (_, haps_i_ref)) in indiv_haps.iter().enumerate() {
        for (j, (_, haps_j_ref)) in indiv_haps.iter().enumerate().take(i + 1) {
            let haps_j = haps_j_ref;
            let haps_i = haps_i_ref;
            let pos = i * (i + 1) / 2 + j;

            let mut sum = 0.0;
            let mut count = 0u32;

            for &hi in haps_i {
                for &hj in haps_j {
                    if hi == hj {
                        // Self-identity: I(h,h) = 1.0
                        sum += 1.0;
                        count += 1;
                    } else if let Some(val) = acc.get_average(hi, hj) {
                        sum += val;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                grm[pos] = sum / count as f64;
                grm_counts[pos] = acc.get_count(
                    *haps_i.first().unwrap_or(&0),
                    *haps_j.first().unwrap_or(&0),
                ) as f32;
            }
        }
    }

    // Double centering: G̃[i,j] = G[i,j] - G[i,·] - G[·,j] + G[·,·]
    if double_center {
        // Row means
        let mut row_means = vec![0.0_f64; n_dip];
        for (i, mean) in row_means.iter_mut().enumerate() {
            let mut sum = 0.0;
            for j in 0..n_dip {
                let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
                sum += grm[ii * (ii + 1) / 2 + jj];
            }
            *mean = sum / n_dip as f64;
        }
        // Grand mean
        let grand_mean = row_means.iter().sum::<f64>() / n_dip as f64;

        // Apply double centering
        for i in 0..n_dip {
            for j in 0..=i {
                let pos = i * (i + 1) / 2 + j;
                grm[pos] = grm[pos] - row_means[i] - row_means[j] + grand_mean;
            }
        }
    }

    // Write .grm.bin
    let bin_path = format!("{}.grm.bin", output_prefix);
    let mut bin_file = BufWriter::new(
        File::create(&bin_path).with_context(|| format!("Failed to create {}", bin_path))?,
    );
    for &val in &grm {
        bin_file.write_all(&(val as f32).to_le_bytes())?;
    }
    bin_file.flush()?;

    // Write .grm.N.bin
    let n_path = format!("{}.grm.N.bin", output_prefix);
    let mut n_file = BufWriter::new(
        File::create(&n_path).with_context(|| format!("Failed to create {}", n_path))?,
    );
    for &c in &grm_counts {
        n_file.write_all(&c.to_le_bytes())?;
    }
    n_file.flush()?;

    // Write .grm.id (diploid individuals)
    let id_path = format!("{}.grm.id", output_prefix);
    let mut id_file = BufWriter::new(
        File::create(&id_path).with_context(|| format!("Failed to create {}", id_path))?,
    );
    for (name, haps) in &indiv_haps {
        let hap_names: Vec<&str> = haps.iter()
            .map(|&idx| acc.sample_names[idx].as_str())
            .collect();
        writeln!(id_file, "{}\t{}", name, hap_names.join(","))?;
    }
    id_file.flush()?;

    eprintln!(
        "Diploid eGRM written: {} individuals (from {} haplotypes), prefix={}{}",
        n_dip,
        n_haps,
        output_prefix,
        if double_center { " (double-centered)" } else { "" }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;
    use tempfile::NamedTempFile;

    fn create_sim_file(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_pairwise_accumulator_basic() {
        let mut acc = PairwiseAccumulator::new();
        acc.add("A#1", "B#1", 0.995);
        acc.add("A#1", "B#1", 0.997);

        assert_eq!(acc.n_samples(), 2);
        let avg = acc.get_average(0, 1).unwrap();
        assert!((avg - 0.996).abs() < 1e-10);
        assert_eq!(acc.get_count(0, 1), 2);
    }

    #[test]
    fn test_pairwise_accumulator_symmetric() {
        let mut acc = PairwiseAccumulator::new();
        acc.add("A#1", "B#1", 0.99);
        acc.add("B#1", "A#1", 0.98);

        // Both directions go to same cell
        let avg = acc.get_average(0, 1).unwrap();
        assert!((avg - 0.985).abs() < 1e-10);
        assert_eq!(acc.get_count(0, 1), 2);
    }

    #[test]
    fn test_parse_similarity_for_egrm() {
        let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                        chr1\t0\t10000\tA#1#chr1:0-10000\tB#1#chr1:0-10000\t0.995\n\
                        chr1\t0\t10000\tA#1#chr1:0-10000\tC#1#chr1:0-10000\t0.990\n\
                        chr1\t0\t10000\tB#1#chr1:0-10000\tC#1#chr1:0-10000\t0.992\n\
                        chr1\t10000\t20000\tA#1#chr1:10000-20000\tB#1#chr1:10000-20000\t0.996\n\
                        chr1\t10000\t20000\tA#1#chr1:10000-20000\tC#1#chr1:10000-20000\t0.991\n\
                        chr1\t10000\t20000\tB#1#chr1:10000-20000\tC#1#chr1:10000-20000\t0.993\n";

        let f = create_sim_file(content);
        let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

        assert_eq!(acc.n_samples(), 3);

        // A#1 vs B#1: (0.995 + 0.996) / 2 = 0.9955
        let ab = acc.get_average(0, 1).unwrap();
        assert!((ab - 0.9955).abs() < 1e-10);

        // A#1 vs C#1: (0.990 + 0.991) / 2 = 0.9905
        let ac = acc.get_average(0, 2).unwrap();
        assert!((ac - 0.9905).abs() < 1e-10);
    }

    #[test]
    fn test_write_gcta_grm_uncentered() {
        let mut acc = PairwiseAccumulator::new();
        // 2 samples, 1 window each
        acc.add("A#1", "B#1", 0.995);
        // Self-identity for diagonal (these would come from self-comparisons,
        // but we skip those in parsing. Add explicitly for test)
        acc.add("A#1", "A#2", 0.999); // proxy for self

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("test_grm");
        let prefix_str = prefix.to_str().unwrap();

        write_gcta_grm(&acc, prefix_str, false).unwrap();

        // Verify files exist
        assert!(Path::new(&format!("{}.grm.bin", prefix_str)).exists());
        assert!(Path::new(&format!("{}.grm.N.bin", prefix_str)).exists());
        assert!(Path::new(&format!("{}.grm.id", prefix_str)).exists());

        // Read .grm.id and verify
        let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
        let lines: Vec<&str> = id_content.trim().lines().collect();
        assert_eq!(lines.len(), 3); // A#1, B#1, A#2

        // Read .grm.bin and check values
        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        // 3 samples → 6 values (lower triangle), each 4 bytes
        assert_eq!(bin_data.len(), 6 * 4);
    }

    #[test]
    fn test_write_gcta_grm_centered() {
        let mut acc = PairwiseAccumulator::new();
        acc.add("A#1", "B#1", 0.99);
        acc.add("A#1", "C#1", 0.98);
        acc.add("B#1", "C#1", 0.97);

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("test_centered");
        let prefix_str = prefix.to_str().unwrap();

        write_gcta_grm(&acc, prefix_str, true).unwrap();

        // Read .grm.bin
        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        // 3 samples → 6 values. Some have data, some don't (diagonal has no data → 0.0 - grand_mean)
        assert_eq!(bin_data.len(), 6 * 4);

        // Grand mean of existing values: (0.99 + 0.98 + 0.97) / 3 = 0.98
        // Check A#1-B#1 entry: 0.99 - 0.98 = 0.01
        // Position: sample indices A#1=0, B#1=1. Lower triangle pos(1,0) = 1*(1+1)/2 + 0 = 1
        let val_bytes: [u8; 4] = bin_data[4..8].try_into().unwrap();
        let val = f32::from_le_bytes(val_bytes);
        assert!((val - 0.01).abs() < 1e-5, "Expected ~0.01, got {}", val);
    }

    #[test]
    fn test_skip_self_comparisons() {
        let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                        chr1\t0\t10000\tA#1#chr1:0-10000\tA#1#chr1:0-10000\t1.000\n\
                        chr1\t0\t10000\tA#1#chr1:0-10000\tB#1#chr1:0-10000\t0.995\n";

        let f = create_sim_file(content);
        let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

        assert_eq!(acc.n_samples(), 2);
        // Self-comparison should be skipped
        assert!(acc.get_average(0, 0).is_none());
        // A#1 vs B#1 should exist
        assert!(acc.get_average(0, 1).is_some());
    }

    #[test]
    fn test_empty_file() {
        let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n";
        let f = create_sim_file(content);
        let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();
        assert_eq!(acc.n_samples(), 0);
    }

    // ---- Diploid eGRM tests ----

    fn make_diploid_acc() -> PairwiseAccumulator {
        // Two diploid individuals: A (haps A#1, A#2) and B (haps B#1, B#2)
        let mut acc = PairwiseAccumulator::new();
        // Cross-individual comparisons (4 pairs)
        acc.add("A#1", "B#1", 0.98);
        acc.add("A#1", "B#2", 0.96);
        acc.add("A#2", "B#1", 0.97);
        acc.add("A#2", "B#2", 0.95);
        // Within-individual (A's haplotypes)
        acc.add("A#1", "A#2", 0.992);
        // Within-individual (B's haplotypes)
        acc.add("B#1", "B#2", 0.988);
        acc
    }

    #[test]
    fn test_diploid_grm_basic() {
        let acc = make_diploid_acc();
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("dip_grm");
        let prefix_str = prefix.to_str().unwrap();

        write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

        // Check files exist
        assert!(Path::new(&format!("{}.grm.bin", prefix_str)).exists());
        assert!(Path::new(&format!("{}.grm.N.bin", prefix_str)).exists());
        assert!(Path::new(&format!("{}.grm.id", prefix_str)).exists());

        // Read .grm.id — should have 2 individuals
        let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
        let lines: Vec<&str> = id_content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("A\t"));
        assert!(lines[1].starts_with("B\t"));

        // Read .grm.bin — 2 individuals → 3 values (lower triangle)
        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        assert_eq!(bin_data.len(), 3 * 4);

        // Parse values
        let val = |idx: usize| -> f32 {
            f32::from_le_bytes(bin_data[idx * 4..(idx + 1) * 4].try_into().unwrap())
        };

        // G[A,A] = (1.0 + 0.992 + 0.992 + 1.0) / 4 = 0.996
        // (h1-h1=1.0, h1-h2=0.992, h2-h1=0.992, h2-h2=1.0)
        let g_aa = val(0);
        assert!((g_aa - 0.996).abs() < 1e-5, "G[A,A] = {}, expected 0.996", g_aa);

        // G[B,A] = (0.98 + 0.96 + 0.97 + 0.95) / 4 = 0.965
        let g_ba = val(1);
        assert!((g_ba - 0.965).abs() < 1e-5, "G[B,A] = {}, expected 0.965", g_ba);

        // G[B,B] = (1.0 + 0.988 + 0.988 + 1.0) / 4 = 0.994
        let g_bb = val(2);
        assert!((g_bb - 0.994).abs() < 1e-5, "G[B,B] = {}, expected 0.994", g_bb);
    }

    #[test]
    fn test_diploid_grm_double_centering() {
        let acc = make_diploid_acc();
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("dip_centered");
        let prefix_str = prefix.to_str().unwrap();

        write_diploid_gcta_grm(&acc, prefix_str, true).unwrap();

        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        assert_eq!(bin_data.len(), 3 * 4);

        let val = |idx: usize| -> f64 {
            f32::from_le_bytes(bin_data[idx * 4..(idx + 1) * 4].try_into().unwrap()) as f64
        };

        // Uncentered: G[A,A]=0.996, G[B,A]=0.965, G[B,B]=0.994
        // Row means: r_A = (0.996 + 0.965)/2 = 0.9805, r_B = (0.965 + 0.994)/2 = 0.9795
        // Grand mean: (0.9805 + 0.9795)/2 = 0.9800
        // Centered:
        //   G̃[A,A] = 0.996 - 0.9805 - 0.9805 + 0.9800 = 0.0150
        //   G̃[B,A] = 0.965 - 0.9805 - 0.9795 + 0.9800 = -0.0150
        //   G̃[B,B] = 0.994 - 0.9795 - 0.9795 + 0.9800 = 0.0150

        assert!((val(0) - 0.015).abs() < 1e-4, "G̃[A,A] = {:.6}, expected 0.015", val(0));
        assert!((val(1) - (-0.015)).abs() < 1e-4, "G̃[B,A] = {:.6}, expected -0.015", val(1));
        assert!((val(2) - 0.015).abs() < 1e-4, "G̃[B,B] = {:.6}, expected 0.015", val(2));
    }

    #[test]
    fn test_diploid_grm_double_centering_sums_to_zero() {
        // With 3 individuals, verify row sums are ~0 after double centering
        let mut acc = PairwiseAccumulator::new();
        // Individual A: A#1, A#2
        // Individual B: B#1, B#2
        // Individual C: C#1, C#2
        acc.add("A#1", "A#2", 0.99);
        acc.add("B#1", "B#2", 0.98);
        acc.add("C#1", "C#2", 0.97);
        acc.add("A#1", "B#1", 0.95);
        acc.add("A#1", "B#2", 0.94);
        acc.add("A#2", "B#1", 0.93);
        acc.add("A#2", "B#2", 0.92);
        acc.add("A#1", "C#1", 0.90);
        acc.add("A#1", "C#2", 0.89);
        acc.add("A#2", "C#1", 0.88);
        acc.add("A#2", "C#2", 0.87);
        acc.add("B#1", "C#1", 0.91);
        acc.add("B#1", "C#2", 0.90);
        acc.add("B#2", "C#1", 0.89);
        acc.add("B#2", "C#2", 0.88);

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("dip3_centered");
        let prefix_str = prefix.to_str().unwrap();

        write_diploid_gcta_grm(&acc, prefix_str, true).unwrap();

        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        // 3 individuals → 6 values
        assert_eq!(bin_data.len(), 6 * 4);

        let val = |idx: usize| -> f64 {
            f32::from_le_bytes(bin_data[idx * 4..(idx + 1) * 4].try_into().unwrap()) as f64
        };

        // Reconstruct full matrix from lower triangle
        // pos(0,0)=0, pos(1,0)=1, pos(1,1)=2, pos(2,0)=3, pos(2,1)=4, pos(2,2)=5
        let g = |i: usize, j: usize| -> f64 {
            let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
            val(ii * (ii + 1) / 2 + jj)
        };

        // Row sums should be ~0
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| g(i, j)).sum();
            assert!(row_sum.abs() < 1e-3, "Row {} sum = {:.6}, expected ~0", i, row_sum);
        }
    }

    #[test]
    fn test_diploid_grm_single_individual() {
        let mut acc = PairwiseAccumulator::new();
        acc.add("A#1", "A#2", 0.995);

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("single_dip");
        let prefix_str = prefix.to_str().unwrap();

        write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        // 1 individual → 1 value
        assert_eq!(bin_data.len(), 4);

        let val = f32::from_le_bytes(bin_data[0..4].try_into().unwrap());
        // G[A,A] = (1.0 + 0.995 + 0.995 + 1.0) / 4 = 0.9975
        assert!((val - 0.9975).abs() < 1e-5, "G[A,A] = {}, expected 0.9975", val);
    }

    #[test]
    fn test_diploid_grm_haploid_individual() {
        // Individual with only 1 haplotype
        let mut acc = PairwiseAccumulator::new();
        acc.add("A#1", "B#1", 0.98);

        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("haploid");
        let prefix_str = prefix.to_str().unwrap();

        write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

        let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
        let lines: Vec<&str> = id_content.trim().lines().collect();
        assert_eq!(lines.len(), 2);

        let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
        assert_eq!(bin_data.len(), 3 * 4);

        let val = |idx: usize| -> f32 {
            f32::from_le_bytes(bin_data[idx * 4..(idx + 1) * 4].try_into().unwrap())
        };

        // G[A,A] = 1.0 (only self-comparison h1-h1)
        assert!((val(0) - 1.0).abs() < 1e-5);
        // G[B,A] = 0.98
        assert!((val(1) - 0.98).abs() < 1e-5);
        // G[B,B] = 1.0
        assert!((val(2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_diploid_grm_empty_panics() {
        let acc = PairwiseAccumulator::new();
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("empty_dip");
        let prefix_str = prefix.to_str().unwrap();

        let result = write_diploid_gcta_grm(&acc, prefix_str, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_individual_id() {
        assert_eq!(extract_individual_id("HG00733#1"), "HG00733");
        assert_eq!(extract_individual_id("NA12878#2"), "NA12878");
        assert_eq!(extract_individual_id("SAMPLE"), "SAMPLE");
    }
}
