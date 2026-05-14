//! argraph CLI — experimental.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use impopk_argraph::{build_site, classify, enumerate_bubbles, Graph, Panel, MISSING_GENOTYPE};

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
    /// Emit a tsinfer-ready sites TSV plus the panel order. One row per
    /// bubble; genotypes column is comma-separated per-haplotype branch
    /// indices in panel order, with -1 for haplotypes that do not pass
    /// through the bubble source.
    EmitSites {
        /// Input GFA.
        #[arg(long)]
        gfa: PathBuf,
        /// Output sites TSV path (`-` for stdout).
        #[arg(long, default_value = "-")]
        output: String,
        /// Output panel.txt with one panel haplotype name per line, in the
        /// same order as the `genotypes` column. Defaults to <output>.panel
        /// when `--output` is a file; required when `--output -`.
        #[arg(long)]
        panel_out: Option<PathBuf>,
        /// Path-name prefix that marks the reference. Default: CHM13.
        #[arg(long, default_value = "CHM13")]
        reference: String,
        /// Max BFS depth.
        #[arg(long, default_value = "200")]
        max_depth: usize,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Classify { gfa, output, max_depth } => classify_cmd(&gfa, &output, max_depth),
        Cmd::Stats { gfa, max_depth } => stats_cmd(&gfa, max_depth),
        Cmd::EmitSites { gfa, output, panel_out, reference, max_depth } => {
            emit_sites_cmd(&gfa, &output, panel_out.as_deref(), &reference, max_depth)
        }
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
    writeln!(w, "bubble_id\tsource\tsink\tn_branches\ttype\tmu\tbranch_lens\tbfs_closed")?;
    for (i, b) in bubbles.iter().enumerate() {
        let t = classify(b, &graph);
        // True branch count from the graph (number of successors of source).
        // bubble.branches may be empty if BFS didn't converge; the real count
        // is always graph.successors(source).len() for multi-out sources.
        let n_branches = graph.successors(b.source).len();
        let bfs_closed = !b.branches.is_empty() || b.source != b.sink;
        let sink_str = if bfs_closed {
            b.sink.to_string()
        } else {
            "NA".to_string()
        };
        let lens_str = if b.branches.is_empty() {
            "NA".to_string()
        } else {
            b.branches
                .iter()
                .map(|br| br.len().to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.2e}\t{}\t{}",
            i,
            b.source,
            sink_str,
            n_branches,
            t.as_str(),
            t.mu_event(),
            lens_str,
            bfs_closed
        )?;
    }
    Ok(())
}

fn emit_sites_cmd(
    gfa: &std::path::Path,
    output: &str,
    panel_out: Option<&std::path::Path>,
    reference: &str,
    max_depth: usize,
) -> Result<()> {
    let graph = Graph::parse(gfa).context("parsing GFA")?;
    let panel = Panel::from_graph(&graph, reference);
    eprintln!(
        "loaded {} segments, {} paths ({} panel + {} reference)",
        graph.seq.len(),
        graph.paths.len(),
        panel.names.len(),
        if panel.reference.is_some() { 1 } else { 0 },
    );
    if panel.reference.is_none() {
        eprintln!(
            "warning: no path matching prefix '{}' — ref_pos and ancestral_branch will be NA",
            reference
        );
    }

    let bubbles = enumerate_bubbles(&graph, max_depth);
    eprintln!("found {} bubbles", bubbles.len());

    // Resolve panel.txt path.
    let panel_path: Option<PathBuf> = match (panel_out, output) {
        (Some(p), _) => Some(p.to_path_buf()),
        (None, "-") => None, // stdout sites + stdout panel doesn't work; only emit sites.
        (None, path) => Some(PathBuf::from(format!("{}.panel", path))),
    };
    if let Some(p) = &panel_path {
        let mut pw = BufWriter::new(File::create(p).context("creating panel file")?);
        for name in &panel.names {
            writeln!(pw, "{}", name)?;
        }
        pw.flush()?;
        eprintln!("wrote panel: {}", p.display());
    } else {
        eprintln!("warning: --output is stdout and --panel-out not given; panel order discarded");
    }

    // Sites TSV.
    let writer: Box<dyn Write> = if output == "-" {
        Box::new(BufWriter::new(std::io::stdout()))
    } else {
        Box::new(BufWriter::new(File::create(output).context("creating sites file")?))
    };
    let mut w = writer;
    writeln!(
        w,
        "bubble_id\tsource\tsink\tref_pos\tn_branches\ttype\tmu\talleles\tancestral\tgenotypes\tbfs_closed"
    )?;
    for (i, b) in bubbles.into_iter().enumerate() {
        let site = build_site(b, &graph, &panel);
        let sink_str = if site.bfs_closed {
            site.sink.to_string()
        } else {
            "NA".to_string()
        };
        let ref_pos_str = site
            .ref_pos
            .map(|p| p.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let ancestral_str = site
            .ancestral_branch
            .map(|a| a.to_string())
            .unwrap_or_else(|| "NA".to_string());
        let alleles_str = site.alleles.join(",");
        let genotypes_str = site
            .genotypes
            .iter()
            .map(|&g| {
                if g == MISSING_GENOTYPE {
                    "-1".to_string()
                } else {
                    g.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.2e}\t{}\t{}\t{}\t{}",
            i,
            site.source,
            sink_str,
            ref_pos_str,
            site.alleles.len(),
            site.bubble_type.as_str(),
            site.mu,
            alleles_str,
            ancestral_str,
            genotypes_str,
            site.bfs_closed
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
