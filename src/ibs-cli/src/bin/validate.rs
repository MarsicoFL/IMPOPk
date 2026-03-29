use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(name = "tpa-validate", about = "Compare ibs-from-paf vs ibs-from-tpa outputs")]
struct Args {
    /// Reference output (from ibs-from-paf)
    #[arg(long)]
    reference: String,

    /// Test output (from ibs-from-tpa)
    #[arg(long)]
    test: String,
}

type RowKey = (String, String, String, String, String); // (chrom, start, end, group_a, group_b)

fn load_tsv(path: &str) -> Result<HashMap<RowKey, f64>> {
    let file = File::open(path).context(format!("Failed to open: {}", path))?;
    let reader = BufReader::new(file);
    let mut map = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 || line.starts_with('#') {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 6 {
            continue;
        }

        let key = (
            fields[0].to_string(),
            fields[1].to_string(),
            fields[2].to_string(),
            fields[3].to_string(),
            fields[4].to_string(),
        );
        let identity: f64 = fields[5].parse().unwrap_or(0.0);
        map.insert(key, identity);
    }

    Ok(map)
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading reference: {}", args.reference);
    let reference = load_tsv(&args.reference)?;
    eprintln!("  {} rows", reference.len());

    eprintln!("Loading test: {}", args.test);
    let test = load_tsv(&args.test)?;
    eprintln!("  {} rows", test.len());

    // Match rows
    let mut matched = 0u64;
    let mut ref_only = 0u64;
    let mut test_only = 0u64;
    let mut diffs: Vec<f64> = Vec::new();

    for (key, ref_val) in &reference {
        if let Some(test_val) = test.get(key) {
            matched += 1;
            diffs.push((ref_val - test_val).abs());
        } else {
            ref_only += 1;
        }
    }

    for key in test.keys() {
        if !reference.contains_key(key) {
            test_only += 1;
        }
    }

    // Statistics
    println!("=== Validation Report ===");
    println!("Reference rows: {}", reference.len());
    println!("Test rows:      {}", test.len());
    println!("Matched:        {}", matched);
    println!("Reference only: {}", ref_only);
    println!("Test only:      {}", test_only);

    let pass = if !diffs.is_empty() {
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let median = diffs[diffs.len() / 2];
        let max = diffs.last().copied().unwrap_or(0.0);
        let p99 = diffs[((diffs.len() as f64 * 0.99) as usize).min(diffs.len() - 1)];

        println!();
        println!("Identity difference (|ref - test|):");
        println!("  Mean:   {:.6}", mean);
        println!("  Median: {:.6}", median);
        println!("  P99:    {:.6}", p99);
        println!("  Max:    {:.6}", max);
        println!();

        let ok = mean < 0.001 && max < 0.01;
        if ok {
            println!("VERDICT: PASS (mean < 0.001, max < 0.01)");
        } else {
            println!(
                "VERDICT: FAIL (mean={:.6} [threshold 0.001], max={:.6} [threshold 0.01])",
                mean, max
            );
        }
        ok
    } else if matched == 0 {
        println!();
        println!("VERDICT: FAIL (no matching rows)");
        false
    } else {
        true
    };

    if !pass {
        std::process::exit(1);
    }

    Ok(())
}
