//! IBD segment detection using HMM

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;

use hprc_ibd::hmm::{extract_ibd_segments, viterbi, HmmParams};
use hprc_ibd::{Region, WindowIterator};

#[derive(Parser, Debug)]
#[command(name = "ibd", version, about = "IBD segment detection using HMM")]
struct Args {
    #[arg(long = "sequence-files", required = true)]
    sequence_files: String,

    #[arg(short = 'a', required = true)]
    align: String,

    #[arg(short = 'r', required = true)]
    ref_name: String,

    #[arg(long = "region", required = true)]
    region: String,

    #[arg(long = "size", required = true)]
    window_size: u64,

    #[arg(long = "subset-sequence-list", required = true)]
    subset_list: String,

    #[arg(long = "output", required = true)]
    output: String,

    #[arg(long = "ibs-output")]
    ibs_output: Option<String>,

    #[arg(long = "region-length")]
    region_length: Option<u64>,

    #[arg(long = "min-len-bp", default_value = "5000")]
    min_len_bp: u64,

    #[arg(long = "min-windows", default_value = "3")]
    min_windows: usize,

    #[arg(long = "expected-seg-windows", default_value = "50")]
    expected_seg_windows: f64,

    #[arg(long = "p-enter-ibd", default_value = "0.0001")]
    p_enter_ibd: f64,
}

#[derive(Debug, Clone)]
struct WindowRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
}

fn pair_key(a: &str, b: &str) -> (String, String) {
    if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) }
}

struct ColumnIndices {
    estimated_identity: usize,
    chrom: usize,
    start: usize,
    end: usize,
    group_a: usize,
    group_b: usize,
}

impl ColumnIndices {
    fn from_header(header: &str) -> Result<Self> {
        let columns: Vec<&str> = header.split('\t').collect();
        let find_col = |name: &str| -> Result<usize> {
            columns.iter().position(|&c| c == name).context(format!("Missing: {}", name))
        };
        Ok(ColumnIndices {
            estimated_identity: find_col("estimated.identity")?,
            chrom: find_col("chrom")?,
            start: find_col("start")?,
            end: find_col("end")?,
            group_a: find_col("group.a")?,
            group_b: find_col("group.b")?,
        })
    }
}

fn collect_identities(
    args: &Args,
    region: &Region,
    mut ibs_output: Option<&mut BufWriter<File>>,
) -> Result<HashMap<(String, String), Vec<WindowRecord>>> {
    let ref_prefix = format!("{}#", args.ref_name);
    let mut pair_data: HashMap<(String, String), Vec<WindowRecord>> = HashMap::new();
    let mut first_window = true;

    for window in WindowIterator::new(region, args.window_size) {
        let ref_region = format!("{}#0#{}:{}-{}", args.ref_name, region.chrom, window.start, window.end);
        eprintln!("Collecting identities for {}", ref_region);

        let mut cmd = Command::new("impg");
        cmd.arg("similarity")
            .arg("--sequence-files").arg(&args.sequence_files)
            .arg("-p").arg(&args.align)
            .arg("-r").arg(&ref_region)
            .arg("--subset-sequence-list").arg(&args.subset_list)
            .arg("--force-large-region")
            .stdout(Stdio::piped());

        let mut child = cmd.spawn().context("Failed to spawn impg")?;
        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        let header = lines.next().context("No output")?.context("Failed to read header")?;
        let cols = ColumnIndices::from_header(&header)?;

        if let Some(ref mut out) = ibs_output {
            if first_window {
                writeln!(out, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;
            }
        }

        for line_result in lines {
            let line = line_result?;
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() <= cols.estimated_identity.max(cols.group_a).max(cols.group_b) {
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
            let start: u64 = fields[cols.start].parse().unwrap_or(0);
            let end: u64 = fields[cols.end].parse().unwrap_or(0);

            if let Some(ref mut out) = ibs_output {
                writeln!(out, "{}\t{}\t{}\t{}\t{}\t{}", chrom, start, end, group_a, group_b, identity)?;
            }

            let key = pair_key(group_a, group_b);
            pair_data.entry(key).or_default().push(WindowRecord { chrom, start, end, identity });
        }

        let status = child.wait()?;
        if !status.success() { bail!("impg failed"); }
        first_window = false;
    }

    Ok(pair_data)
}

fn call_ibd_segments(
    pair_data: HashMap<(String, String), Vec<WindowRecord>>,
    args: &Args,
    output: &mut BufWriter<File>,
) -> Result<usize> {
    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\tn_windows\tmean_identity")?;
    let mut total_segments = 0;

    for ((hap_a, hap_b), mut records) in pair_data {
        if records.len() < 3 { continue; }

        records.sort_by_key(|r| r.start);
        let observations: Vec<f64> = records.iter().map(|r| r.identity).collect();

        let mut params = HmmParams::from_expected_length(args.expected_seg_windows, args.p_enter_ibd);
        params.estimate_emissions(&observations);

        let states = viterbi(&observations, &params);
        let segments = extract_ibd_segments(&states);

        for (start_idx, end_idx, n_windows) in segments {
            if n_windows < args.min_windows { continue; }

            let start_bp = records[start_idx].start;
            let end_bp = records[end_idx].end;
            let length_bp = end_bp.saturating_sub(start_bp);

            if length_bp < args.min_len_bp { continue; }

            let mean_identity: f64 = observations[start_idx..=end_idx].iter().sum::<f64>() / n_windows as f64;

            writeln!(output, "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
                records[start_idx].chrom, start_bp, end_bp, hap_a, hap_b, n_windows, mean_identity)?;
            total_segments += 1;
        }
    }

    Ok(total_segments)
}

fn run() -> Result<()> {
    let args = Args::parse();

    if Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }

    let region = Region::parse(&args.region, args.region_length)?;
    eprintln!("Processing region {}:{}-{}", region.chrom, region.start, region.end);

    let mut ibs_output = match &args.ibs_output {
        Some(path) => Some(BufWriter::new(File::create(path)?)),
        None => None,
    };

    let pair_data = collect_identities(&args, &region, ibs_output.as_mut())?;
    if let Some(ref mut out) = ibs_output { out.flush()?; }

    eprintln!("Collected data for {} pairs", pair_data.len());

    let output_file = File::create(&args.output)?;
    let mut output = BufWriter::new(output_file);
    let n_segments = call_ibd_segments(pair_data, &args, &mut output)?;

    output.flush()?;
    eprintln!("IBD complete: {} segments written to {}", n_segments, args.output);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}
