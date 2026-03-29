use std::collections::{btree_map::Entry, BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "jacquard", version, about = "Compute Jacquard delta coefficients from IBS windows")]
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

    for (delta, &count) in counts.iter().enumerate().skip(1) {
        let frac = if total > 0 { count as f64 / total as f64 } else { 0.0 };
        println!("Delta{}\t{:.8}\t(count={})", delta, frac, count);
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── hap_key tests ─────────────────────────────────────────────────

    #[test]
    fn hap_key_sample_hash_hap() {
        assert_eq!(hap_key("HG00096#1"), "HG00096#1");
    }

    #[test]
    fn hap_key_three_parts_keeps_first_two() {
        // "HG00096#1#scaffold:0-5000" → "HG00096#1"
        assert_eq!(hap_key("HG00096#1#scaffold:0-5000"), "HG00096#1");
    }

    #[test]
    fn hap_key_no_hash_returns_raw() {
        assert_eq!(hap_key("nohash"), "nohash");
    }

    #[test]
    fn hap_key_empty_string() {
        assert_eq!(hap_key(""), "");
    }

    #[test]
    fn hap_key_single_hash_empty_hap() {
        // "sample#" → "sample#"
        assert_eq!(hap_key("sample#"), "sample#");
    }

    #[test]
    fn hap_key_hash_at_start() {
        // "#1" → "#1"
        assert_eq!(hap_key("#1"), "#1");
    }

    // ── classify_state tests — all 9 Jacquard delta states ────────────

    fn bs(size: usize, count_a: usize, count_b: usize) -> BlockStat {
        BlockStat { size, count_a, count_b }
    }

    // Delta 1: all 4 haplotypes in one block (all IBD)
    #[test]
    fn classify_state_delta1_all_connected() {
        let blocks = vec![bs(4, 2, 2)];
        assert_eq!(classify_state(&blocks), Some(1));
    }

    // Delta 2: A-pair and B-pair as separate blocks (both pairs IBD within, not between)
    #[test]
    fn classify_state_delta2_a_pair_b_pair() {
        let blocks = vec![bs(2, 2, 0), bs(2, 0, 2)];
        assert_eq!(classify_state(&blocks), Some(2));
    }

    #[test]
    fn classify_state_delta2_reversed_order() {
        // B-pair first, A-pair second — should still be Delta 2
        let blocks = vec![bs(2, 0, 2), bs(2, 2, 0)];
        assert_eq!(classify_state(&blocks), Some(2));
    }

    // Delta 3: triplet (2A + 1B) + singleton (1B)
    #[test]
    fn classify_state_delta3_triplet_2a_1b() {
        let blocks = vec![bs(3, 2, 1), bs(1, 0, 1)];
        assert_eq!(classify_state(&blocks), Some(3));
    }

    #[test]
    fn classify_state_delta3_singleton_first() {
        let blocks = vec![bs(1, 0, 1), bs(3, 2, 1)];
        assert_eq!(classify_state(&blocks), Some(3));
    }

    // Delta 4: A-pair connected + 2 singletons (both B)
    #[test]
    fn classify_state_delta4_a_pair_two_singletons() {
        let blocks = vec![bs(2, 2, 0), bs(1, 0, 1), bs(1, 0, 1)];
        assert_eq!(classify_state(&blocks), Some(4));
    }

    // Delta 5: triplet (1A + 2B) + singleton (1A)
    #[test]
    fn classify_state_delta5_triplet_1a_2b() {
        let blocks = vec![bs(3, 1, 2), bs(1, 1, 0)];
        assert_eq!(classify_state(&blocks), Some(5));
    }

    #[test]
    fn classify_state_delta5_singleton_first() {
        let blocks = vec![bs(1, 1, 0), bs(3, 1, 2)];
        assert_eq!(classify_state(&blocks), Some(5));
    }

    // Delta 6: B-pair connected + 2 singletons (both A)
    #[test]
    fn classify_state_delta6_b_pair_two_singletons() {
        let blocks = vec![bs(2, 0, 2), bs(1, 1, 0), bs(1, 1, 0)];
        assert_eq!(classify_state(&blocks), Some(6));
    }

    // Delta 7: two blocks of 2, each with 1A + 1B (cross-group pairing)
    #[test]
    fn classify_state_delta7_cross_pairs() {
        let blocks = vec![bs(2, 1, 1), bs(2, 1, 1)];
        assert_eq!(classify_state(&blocks), Some(7));
    }

    // Delta 8: mixed pair (1A + 1B) + 2 singletons
    #[test]
    fn classify_state_delta8_mixed_pair_two_singletons() {
        let blocks = vec![bs(2, 1, 1), bs(1, 1, 0), bs(1, 0, 1)];
        assert_eq!(classify_state(&blocks), Some(8));
    }

    // Delta 9: all 4 singletons (no IBD between any pair)
    #[test]
    fn classify_state_delta9_all_singletons() {
        let blocks = vec![bs(1, 1, 0), bs(1, 1, 0), bs(1, 0, 1), bs(1, 0, 1)];
        assert_eq!(classify_state(&blocks), Some(9));
    }

    // ── classify_state edge cases ─────────────────────────────────────

    #[test]
    fn classify_state_empty_returns_none() {
        assert_eq!(classify_state(&[]), None);
    }

    #[test]
    fn classify_state_single_block_not_size4_returns_none() {
        // 1 block with 3 nodes — shouldn't happen normally, but classify handles it
        let blocks = vec![bs(3, 2, 1)];
        assert_eq!(classify_state(&blocks), None);
    }

    #[test]
    fn classify_state_two_blocks_2_2_neither_delta2_nor_delta7() {
        // 2 blocks of 2, but with 2A+0B and 1A+1B — invalid for delta 2 or 7
        let blocks = vec![bs(2, 2, 0), bs(2, 1, 1)];
        assert_eq!(classify_state(&blocks), None);
    }

    #[test]
    fn classify_state_two_blocks_3_1_triplet_2a_0b_returns_none() {
        // triplet with 2A+0B — doesn't match delta 3 (2A+1B) or delta 5 (1A+2B)
        let blocks = vec![bs(3, 2, 0), bs(1, 0, 1)];
        assert_eq!(classify_state(&blocks), None);
    }

    #[test]
    fn classify_state_five_blocks_returns_none() {
        // More than 4 blocks — impossible for 4 haplotypes but handled gracefully
        let blocks = vec![bs(1, 1, 0), bs(1, 0, 1), bs(1, 0, 1), bs(1, 1, 0), bs(1, 0, 0)];
        assert_eq!(classify_state(&blocks), None);
    }

    #[test]
    fn classify_state_three_blocks_with_size3_returns_none() {
        // 3 blocks, but one has size 3 instead of expected (2+1+1) pattern
        let blocks = vec![bs(3, 2, 1), bs(1, 0, 1), bs(1, 0, 0)];
        assert_eq!(classify_state(&blocks), None);
    }

    #[test]
    fn classify_state_four_blocks_not_all_size1_returns_none() {
        let blocks = vec![bs(2, 1, 1), bs(1, 1, 0), bs(1, 0, 1), bs(1, 0, 0)];
        assert_eq!(classify_state(&blocks), None);
    }

    // ── classify_locus integration via classify_state ─────────────────

    // Verify that each delta classification is stable
    // by testing symmetric block orderings
    #[test]
    fn classify_state_delta4_different_ordering() {
        // Singletons first, pair last
        let blocks = vec![bs(1, 0, 1), bs(1, 0, 1), bs(2, 2, 0)];
        assert_eq!(classify_state(&blocks), Some(4));
    }

    #[test]
    fn classify_state_delta6_different_ordering() {
        // Pair in middle
        let blocks = vec![bs(1, 1, 0), bs(2, 0, 2), bs(1, 1, 0)];
        assert_eq!(classify_state(&blocks), Some(6));
    }

    #[test]
    fn classify_state_delta8_different_ordering() {
        // Singletons first
        let blocks = vec![bs(1, 0, 1), bs(1, 1, 0), bs(2, 1, 1)];
        assert_eq!(classify_state(&blocks), Some(8));
    }

    // ── UnionFind tests ───────────────────────────────────────────────

    #[test]
    fn union_find_singletons() {
        let mut uf = UnionFind::new(&["a", "b", "c"]);
        // Each is its own root initially
        assert_ne!(uf.find("a"), uf.find("b"));
        assert_ne!(uf.find("b"), uf.find("c"));
    }

    #[test]
    fn union_find_union_makes_same_root() {
        let mut uf = UnionFind::new(&["a", "b", "c"]);
        uf.union("a", "b");
        assert_eq!(uf.find("a"), uf.find("b"));
        assert_ne!(uf.find("a"), uf.find("c"));
    }

    #[test]
    fn union_find_transitive() {
        let mut uf = UnionFind::new(&["a", "b", "c"]);
        uf.union("a", "b");
        uf.union("b", "c");
        assert_eq!(uf.find("a"), uf.find("c"));
    }

    #[test]
    fn union_find_all_four() {
        let mut uf = UnionFind::new(&["a1", "a2", "b1", "b2"]);
        uf.union("a1", "a2");
        uf.union("b1", "b2");
        uf.union("a1", "b1");
        // All in same component
        let root = uf.find("a1");
        assert_eq!(uf.find("a2"), root);
        assert_eq!(uf.find("b1"), root);
        assert_eq!(uf.find("b2"), root);
    }

    #[test]
    fn union_find_duplicate_union_is_idempotent() {
        let mut uf = UnionFind::new(&["x", "y"]);
        uf.union("x", "y");
        let root1 = uf.find("x");
        uf.union("x", "y");
        let root2 = uf.find("x");
        assert_eq!(root1, root2);
    }

    // ── HaplotypeSet tests ────────────────────────────────────────────

    #[test]
    fn haplotype_set_contains_all_four() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        assert!(haps.contains("A#1"));
        assert!(haps.contains("A#2"));
        assert!(haps.contains("B#1"));
        assert!(haps.contains("B#2"));
        assert!(!haps.contains("C#1"));
    }

    #[test]
    fn haplotype_set_is_a_is_b() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        assert!(haps.is_a("A#1"));
        assert!(haps.is_a("A#2"));
        assert!(!haps.is_a("B#1"));
        assert!(haps.is_b("B#1"));
        assert!(haps.is_b("B#2"));
        assert!(!haps.is_b("A#1"));
    }

    #[test]
    fn haplotype_set_nodes_returns_four() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        let nodes = haps.nodes();
        assert_eq!(nodes.len(), 4);
        assert_eq!(nodes[0], "A#1");
        assert_eq!(nodes[1], "A#2");
        assert_eq!(nodes[2], "B#1");
        assert_eq!(nodes[3], "B#2");
    }

    // ── classify_locus integration tests ──────────────────────────────

    #[test]
    fn classify_locus_delta1_all_pairs_present() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        // All 6 possible pairs → all 4 haplotypes in one block → Delta 1
        let mut data = LocusData::default();
        data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
        data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
        data.pairs.insert(Pair::new("A#1".into(), "B#2".into()));
        data.pairs.insert(Pair::new("A#2".into(), "B#1".into()));
        data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
        data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
        assert_eq!(classify_locus(&data, &haps), Some(1));
    }

    #[test]
    fn classify_locus_delta9_no_pairs() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        // No pairs → 4 singletons → Delta 9
        let data = LocusData::default();
        assert_eq!(classify_locus(&data, &haps), Some(9));
    }

    #[test]
    fn classify_locus_delta2_within_group_ibd() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        // A1-A2 and B1-B2 connected, but no cross-group → Delta 2
        let mut data = LocusData::default();
        data.pairs.insert(Pair::new("A#1".into(), "A#2".into()));
        data.pairs.insert(Pair::new("B#1".into(), "B#2".into()));
        assert_eq!(classify_locus(&data, &haps), Some(2));
    }

    #[test]
    fn classify_locus_delta7_cross_group_pairs() {
        let haps = HaplotypeSet::new(
            "A#1".into(), "A#2".into(), "B#1".into(), "B#2".into(),
        );
        // A1-B1 and A2-B2 connected → two cross-group blocks of size 2 → Delta 7
        let mut data = LocusData::default();
        data.pairs.insert(Pair::new("A#1".into(), "B#1".into()));
        data.pairs.insert(Pair::new("A#2".into(), "B#2".into()));
        assert_eq!(classify_locus(&data, &haps), Some(7));
    }
}
