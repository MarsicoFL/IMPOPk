use std::collections::{btree_map::Entry, BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "jacquard", about = "Compute Jacquard delta coefficients from IBS windows")]
struct Args {
    /// IBS windows file (TSV with chrom/start/end/group.a/group.b)
    #[arg(long = "ibs")]
    ibs: PathBuf,

    /// First haplotype of individual A (format: sample#haplotype, e.g. HG00096#1)
    #[arg(long = "hap-a1")]
    hap_a1: String,
    /// Second haplotype of individual A (format: sample#haplotype, e.g. HG00096#2)
    #[arg(long = "hap-a2")]
    hap_a2: String,
    /// First haplotype of individual B (format: sample#haplotype, e.g. HG00097#1)
    #[arg(long = "hap-b1")]
    hap_b1: String,
    /// Second haplotype of individual B (format: sample#haplotype, e.g. HG00097#2)
    #[arg(long = "hap-b2")]
    hap_b2: String,
}

#[derive(Clone, Debug)]
struct Record {
    chrom: String,
    start: i64,
    end: i64,
    hap_a: String,
    hap_b: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Locus {
    chrom: String,
    start: i64,
    end: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Pair {
    left: String,
    right: String,
}

impl Pair {
    fn new(a: String, b: String) -> Self {
        Self { left: a, right: b }
    }
}

#[derive(Default)]
struct LocusData {
    pairs: BTreeSet<Pair>,
}

struct HaplotypeSet {
    a: [String; 2],
    b: [String; 2],
    all: HashSet<String>,
}

impl HaplotypeSet {
    fn new(a1: String, a2: String, b1: String, b2: String) -> Self {
        let mut all = HashSet::new();
        all.insert(a1.clone());
        all.insert(a2.clone());
        all.insert(b1.clone());
        all.insert(b2.clone());
        Self {
            a: [a1, a2],
            b: [b1, b2],
            all,
        }
    }

    fn contains(&self, hap: &str) -> bool {
        self.all.contains(hap)
    }

    fn is_a(&self, hap: &str) -> bool {
        self.a.iter().any(|h| h == hap)
    }

    fn is_b(&self, hap: &str) -> bool {
        self.b.iter().any(|h| h == hap)
    }

    fn nodes(&self) -> [&str; 4] {
        [
            self.a[0].as_str(),
            self.a[1].as_str(),
            self.b[0].as_str(),
            self.b[1].as_str(),
        ]
    }
}

#[derive(Clone, Copy)]
struct BlockStat {
    size: usize,
    count_a: usize,
    count_b: usize,
}

struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new(nodes: &[&str]) -> Self {
        let mut parent = HashMap::new();
        for n in nodes {
            parent.insert((*n).to_string(), (*n).to_string());
        }
        Self { parent }
    }

    fn find(&mut self, node: &str) -> String {
        let parent = self
            .parent
            .get(node)
            .cloned()
            .unwrap_or_else(|| node.to_string());
        if parent == node {
            return parent;
        }
        let root = self.find(&parent);
        self.parent.insert(node.to_string(), root.clone());
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a != root_b {
            self.parent.insert(root_b, root_a);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    // Validate that all 4 haplotypes are distinct between groups A and B
    let haps_a: HashSet<&str> = [args.hap_a1.as_str(), args.hap_a2.as_str()].into_iter().collect();
    let haps_b: HashSet<&str> = [args.hap_b1.as_str(), args.hap_b2.as_str()].into_iter().collect();

    // Check for duplicates within group A
    if haps_a.len() != 2 {
        bail!("hap-a1 and hap-a2 must be distinct (got '{}' and '{}')", args.hap_a1, args.hap_a2);
    }
    // Check for duplicates within group B
    if haps_b.len() != 2 {
        bail!("hap-b1 and hap-b2 must be distinct (got '{}' and '{}')", args.hap_b1, args.hap_b2);
    }
    // Check for overlap between groups A and B
    let overlap: Vec<_> = haps_a.intersection(&haps_b).collect();
    if !overlap.is_empty() {
        bail!(
            "haplotypes must be distinct between groups A and B; overlapping: {:?}",
            overlap
        );
    }

    let haps = HaplotypeSet::new(args.hap_a1, args.hap_a2, args.hap_b1, args.hap_b2);
    let records = load_records(&args.ibs)?;
    if records.is_empty() {
        bail!("no data rows in IBS file: {}", args.ibs.display());
    }

    let mut chrom0: Option<String> = None;
    let mut min_start: Option<i64> = None;
    let mut max_end: Option<i64> = None;
    let mut win_size: Option<i64> = None;
    let mut loci: BTreeMap<Locus, LocusData> = BTreeMap::new();
    let mut locus_order: Vec<Locus> = Vec::new();

    for rec in &records {
        let window_len = rec.end - rec.start + 1;
        if window_len <= 0 {
            bail!("invalid window coordinates: {}:{}-{}", rec.chrom, rec.start, rec.end);
        }

        if chrom0.is_none() {
            chrom0 = Some(rec.chrom.clone());
            min_start = Some(rec.start);
            max_end = Some(rec.end);
            win_size = Some(window_len);
        } else {
            min_start = Some(min_start.unwrap().min(rec.start));
            max_end = Some(max_end.unwrap().max(rec.end));
        }

        let key = Locus {
            chrom: rec.chrom.clone(),
            start: rec.start,
            end: rec.end,
        };

        let hap1 = hap_key(&rec.hap_a);
        let hap2 = hap_key(&rec.hap_b);

        if !haps.contains(&hap1) && !haps.contains(&hap2) {
            continue;
        }
        if !(haps.contains(&hap1) && haps.contains(&hap2)) {
            continue;
        }
        if hap1 == hap2 {
            continue;
        }

        let (left, right) = if hap1 <= hap2 { (hap1, hap2) } else { (hap2, hap1) };

        match loci.entry(key.clone()) {
            Entry::Vacant(slot) => {
                let mut data = LocusData::default();
                data.pairs.insert(Pair::new(left, right));
                slot.insert(data);
                locus_order.push(key);
            }
            Entry::Occupied(mut slot) => {
                slot.get_mut().pairs.insert(Pair::new(left, right));
            }
        }
    }

    let chrom = chrom0.context("no chrom column detected in IBS file")?;
    let min_start = min_start.context("unable to infer region start")?;
    let max_end = max_end.context("unable to infer region end")?;
    let win_size = win_size.context("unable to infer window size")?;

    let mut counts = [0_u64; 10];
    let mut n_unclassified = 0_u64;

    for locus in &locus_order {
        if let Some(data) = loci.get(locus) {
            match classify_locus(data, &haps) {
                Some(delta) => counts[delta as usize] += 1,
                None => n_unclassified += 1,
            }
        }
    }

    let n_loci = locus_order.len() as i64;
    let span = max_end - min_start + 1;
    if span % win_size != 0 {
        eprintln!(
            "WARNING: (max_end - min_start + 1) not divisible by win_size. span={} win_size={}",
            span, win_size
        );
    }
    if win_size == 0 {
        bail!("invalid window size inferred from IBS file");
    }
    let total_windows = (span / win_size).max(0);
    let missing = (total_windows - n_loci).max(0) as u64;
    counts[9] += missing;

    let total: u64 = counts.iter().skip(1).sum();
    if total == 0 {
        bail!("no loci classified into Jacquard states");
    }

    eprintln!(
        "# chrom\t{}\tmin_start\t{}\tmax_end\t{}\twin_size\t{}",
        chrom, min_start, max_end, win_size
    );
    eprintln!(
        "# total_windows\t{}\tloci_with_IBS_fourhaps\t{}\tmissing_windows_as_Delta9\t{}\tunclassified\t{}",
        total_windows, n_loci, missing, n_unclassified
    );

    for delta in 1..=9 {
        let frac = counts[delta] as f64 / total as f64;
        println!("Delta{}\t{:.8}\t(count={})", delta, frac, counts[delta]);
    }

    Ok(())
}

fn classify_locus(data: &LocusData, haps: &HaplotypeSet) -> Option<u8> {
    let nodes = haps.nodes();
    let mut uf = UnionFind::new(&nodes);
    for pair in &data.pairs {
        uf.union(&pair.left, &pair.right);
    }

    let mut block_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for node in nodes {
        let root = uf.find(node);
        block_map.entry(root).or_default().push(node.to_string());
    }

    let mut stats: Vec<BlockStat> = Vec::new();
    for members in block_map.values() {
        let mut count_a = 0;
        let mut count_b = 0;
        for m in members {
            if haps.is_a(m) {
                count_a += 1;
            } else if haps.is_b(m) {
                count_b += 1;
            }
        }
        stats.push(BlockStat {
            size: members.len(),
            count_a,
            count_b,
        });
    }

    classify_state(&stats)
}

fn classify_state(blocks: &[BlockStat]) -> Option<u8> {
    let nb = blocks.len();
    if nb == 0 {
        return None;
    }

    if nb == 1 {
        if blocks[0].size == 4 {
            return Some(1);
        }
        return None;
    }

    if nb == 4 {
        if blocks.iter().all(|b| b.size == 1) {
            return Some(9);
        }
        return None;
    }

    if nb == 2 {
        let b1 = &blocks[0];
        let b2 = &blocks[1];
        if b1.size == 2 && b2.size == 2 {
            let cond_a = b1.count_a == 2 && b1.count_b == 0 && b2.count_a == 0 && b2.count_b == 2;
            let cond_b = b2.count_a == 2 && b2.count_b == 0 && b1.count_a == 0 && b1.count_b == 2;
            if cond_a || cond_b {
                return Some(2);
            }
            if b1.count_a == 1 && b1.count_b == 1 && b2.count_a == 1 && b2.count_b == 1 {
                return Some(7);
            }
            return None;
        }

        if (b1.size == 3 && b2.size == 1) || (b1.size == 1 && b2.size == 3) {
            let trip = if b1.size == 3 { b1 } else { b2 };
            if trip.count_a == 2 && trip.count_b == 1 {
                return Some(3);
            }
            if trip.count_a == 1 && trip.count_b == 2 {
                return Some(5);
            }
            return None;
        }
        return None;
    }

    if nb == 3 {
        let mut pair_idx: Option<&BlockStat> = None;
        for block in blocks {
            match block.size {
                2 => pair_idx = Some(block),
                1 => continue,
                _ => return None,
            }
        }
        let pair = pair_idx?;
        if pair.count_a == 2 && pair.count_b == 0 {
            return Some(4);
        }
        if pair.count_a == 0 && pair.count_b == 2 {
            return Some(6);
        }
        if pair.count_a == 1 && pair.count_b == 1 {
            return Some(8);
        }
        return None;
    }

    None
}

fn load_records(path: &PathBuf) -> Result<Vec<Record>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line_index = 0_usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if line_index == 0 {
            let first = line.split('\t').next().unwrap_or("");
            if first.eq_ignore_ascii_case("chrom") {
                line_index += 1;
                continue;
            }
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            bail!("incomplete IBS row at line {}", line_index + 1);
        }
        let start: i64 = fields[1]
            .parse()
            .with_context(|| format!("invalid start position on line {}", line_index + 1))?;
        let end: i64 = fields[2]
            .parse()
            .with_context(|| format!("invalid end position on line {}", line_index + 1))?;
        records.push(Record {
            chrom: fields[0].to_string(),
            start,
            end,
            hap_a: fields[3].to_string(),
            hap_b: fields[4].to_string(),
        });
        line_index += 1;
    }

    records.sort_by(|a, b| {
        a.chrom
            .cmp(&b.chrom)
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.end.cmp(&b.end))
    });
    Ok(records)
}

fn hap_key(raw: &str) -> String {
    let mut parts = raw.split('#');
    match (parts.next(), parts.next()) {
        (Some(sample), Some(hap)) => format!("{}#{}", sample, hap),
        _ => raw.to_string(),
    }
}
