use anyhow::Result;
use clap::Parser;
use hprc_ibs::spatial_index::SpatialIndex;

#[derive(Parser)]
#[command(name = "tpa-spatial-index", about = "Build spatial index over TPA file")]
struct Args {
    /// TPA file path
    #[arg(long)]
    tpa: String,

    /// Output spatial index file
    #[arg(long)]
    output: String,

    /// Bin size in bp (default: 1Mb)
    #[arg(long, default_value = "1000000")]
    bin_size: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Building spatial index from {}...", args.tpa);
    let start = std::time::Instant::now();

    let index = SpatialIndex::build(&args.tpa, args.bin_size)?;

    eprintln!(
        "  {} records, {} chromosomes, bin_size={}",
        index.total_records, index.num_chroms, index.bin_size
    );

    index.save(&args.output)?;
    eprintln!(
        "  Saved to {} in {:.1}s",
        args.output,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}
