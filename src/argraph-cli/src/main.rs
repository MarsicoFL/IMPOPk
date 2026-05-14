//! argraph CLI — experimental.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use impopk_argraph::{classify, enumerate_bubbles, Graph};

#[derive(Parser, Debug)]
#[command(
    name = "argraph",
    version,
    about = "Experimental ARG-from-pangenome inference (v0.1: classifier only)",
    after_help = "v0.1 scope: parse a GFA, enumerate top-level bubbles, classify each \
                  by mechanism. Downstream wiring to tsinfer is a separate Python helper \
                  (not yet shipped)."
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Enumerate bubbles in a GFA and emit one row per bubble.
    Classify {
        /// Input GFA (from `impg query --output-format gfa`).
        #[arg(long)]
        gfa: PathBuf,
        /// Output TSV path (`-` for stdout).
        #[arg(long, default_value = "-")]
        output: String,
        /// Max BFS depth when searching for a bubble sink (branch length cap).
        #[arg(long, default_value = "200")]
        max_depth: usize,
    },
    /// Just print summary counts.
    Stats {
        #[arg(long)]
        gfa: PathBuf,
        #[arg(long, default_value = "200")]
        max_depth: usize,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Classify { gfa, output, max_depth } => classify_cmd(&gfa, &output, max_depth),
        Cmd::Stats { gfa, max_depth } => stats_cmd(&gfa, max_depth),
    }
}

fn classify_cmd(gfa: &std::path::Path, output: &str, max_depth: usize) -> Result<()> {
    let graph = Graph::parse(gfa).context("parsing GFA")?;
    eprintln!(
        "loaded {} segments, {} paths",
        graph.seq.len(),
        graph.paths.len()
    );

    let bubbles = enumerate_bubbles(&graph, max_depth);
    eprintln!("found {} bubbles", bubbles.len());

    let writer: Box<dyn Write> = if output == "-" {
        Box::new(BufWriter::new(std::io::stdout()))
    } else {
        Box::new(BufWriter::new(File::create(output).context("creating output")?))
    };
    let mut w = writer;
    writeln!(w, "bubble_id\tsource\tsink\tn_branches\ttype\tmu\tbranch_lens")?;
    for (i, b) in bubbles.iter().enumerate() {
        let t = classify(b, &graph);
        let lens: Vec<String> =
            b.branches.iter().map(|br| br.len().to_string()).collect();
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.2e}\t{}",
            i,
            b.source,
            b.sink,
            b.n_branches(),
            t.as_str(),
            t.mu_event(),
            lens.join(",")
        )?;
    }
    Ok(())
}

fn stats_cmd(gfa: &std::path::Path, max_depth: usize) -> Result<()> {
    let graph = Graph::parse(gfa).context("parsing GFA")?;
    let bubbles = enumerate_bubbles(&graph, max_depth);

    let mut counts = std::collections::BTreeMap::new();
    for b in &bubbles {
        let t = classify(b, &graph);
        *counts.entry(t.as_str()).or_insert(0u64) += 1;
    }

    println!("segments\t{}", graph.seq.len());
    println!("paths\t{}", graph.paths.len());
    println!("bubbles\t{}", bubbles.len());
    for (k, v) in &counts {
        println!("{}\t{}", k, v);
    }
    Ok(())
}
