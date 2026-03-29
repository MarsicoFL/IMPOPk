//! IBD validation binary - reads IBS from file and runs HMM
//!
//! This binary is used for validation experiments where we have
//! pre-computed IBS data (either synthetic or from previous runs).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use hprc_ibd::hmm::{extract_ibd_segments, viterbi, HmmParams};

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
    #[arg(long = "p-enter-ibd", default_value = "0.0001")]
    p_enter_ibd: f64,

    /// Window size in bp (for coordinate calculations)
    #[arg(long = "window-size", default_value = "5000")]
    window_size: u64,
}

#[derive(Debug, Clone)]
struct IbsRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
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

        let record = IbsRecord {
            chrom: fields[col_chrom].to_string(),
            start,
            end,
            identity,
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
}

fn process_pair(
    hap_a: &str,
    hap_b: &str,
    mut records: Vec<IbsRecord>,
    args: &Args,
) -> (Vec<IbdSegment>, Vec<(u64, u64, usize)>) {
    let mut segments = Vec::new();
    let mut window_states: Vec<(u64, u64, usize)> = Vec::new();

    if records.len() < 3 {
        return (segments, window_states);
    }

    // Sort by start position
    records.sort_by_key(|r| r.start);

    let observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

    // Create HMM parameters
    let mut params = HmmParams::from_expected_length(args.expected_seg_windows, args.p_enter_ibd, args.window_size);
    params.estimate_emissions(&observations);

    // Run Viterbi
    let states = viterbi(&observations, &params);

    // Store window states
    for (i, &state) in states.iter().enumerate() {
        window_states.push((records[i].start, records[i].end, state));
    }

    // Extract segments
    let raw_segments = extract_ibd_segments(&states);

    for (start_idx, end_idx, n_windows) in raw_segments {
        if n_windows < args.min_windows {
            continue;
        }

        let start_bp = records[start_idx].start;
        let end_bp = records[end_idx].end;
        let length_bp = end_bp.saturating_sub(start_bp);

        if length_bp < args.min_len_bp {
            continue;
        }

        let mean_identity: f64 = observations[start_idx..=end_idx].iter().sum::<f64>() / n_windows as f64;

        segments.push(IbdSegment {
            chrom: records[start_idx].chrom.clone(),
            start: start_bp,
            end: end_bp,
            hap_a: hap_a.to_string(),
            hap_b: hap_b.to_string(),
            n_windows,
            mean_identity,
        });
    }

    (segments, window_states)
}

fn run() -> Result<()> {
    let args = Args::parse();

    eprintln!("Reading IBS data from {:?}", args.input);
    let pair_data = read_ibs_file(&args.input)?;
    eprintln!("Found {} haplotype pairs", pair_data.len());

    let mut all_segments = Vec::new();
    let mut all_window_states: Vec<(String, String, Vec<(u64, u64, usize)>)> = Vec::new();

    for ((hap_a, hap_b), records) in pair_data {
        let n_windows = records.len();
        let (segments, window_states) = process_pair(&hap_a, &hap_b, records, &args);

        eprintln!(
            "  Pair {}-{}: {} windows -> {} IBD segments",
            hap_a, hap_b, n_windows, segments.len()
        );

        all_segments.extend(segments);
        all_window_states.push((hap_a, hap_b, window_states));
    }

    // Write IBD segments
    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);

    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity")?;
    for seg in &all_segments {
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            seg.chrom, seg.start, seg.end, seg.hap_a, seg.hap_b, seg.n_windows, seg.mean_identity
        )?;
    }
    output.flush()?;

    eprintln!("Wrote {} IBD segments to {:?}", all_segments.len(), args.output);

    // Write per-window states if requested
    if let Some(states_path) = &args.states_output {
        let states_file = File::create(states_path)?;
        let mut states_out = BufWriter::new(states_file);

        writeln!(states_out, "group.a\tgroup.b\tstart\tend\tpredicted_state")?;

        for (hap_a, hap_b, window_states) in &all_window_states {
            for (start, end, state) in window_states {
                writeln!(states_out, "{}\t{}\t{}\t{}\t{}", hap_a, hap_b, start, end, state)?;
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
