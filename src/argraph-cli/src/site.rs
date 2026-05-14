//! Per-bubble site emission for downstream ARG inference.
//!
//! A `Site` wraps a `Bubble` with the fields a tsinfer-style inferrer needs:
//!   - `ref_pos`: an anchor coordinate on a single 1D axis (CHM13 here).
//!   - `genotypes`: which branch each panel haplotype takes (or MISSING_DATA).
//!   - `alleles`: a label per branch (e.g. "A", "T", "REF", "INS_3bp").
//!   - `ancestral_branch`: the index of the reference path's branch.
//!   - `bubble_type` and `mu_event`: mechanism + rate metadata.
//!
//! The panel order is the lexicographic order of P-line names from the GFA,
//! with the reference path excluded if requested.

use std::collections::HashMap;

use crate::bubble::Bubble;
use crate::classify::{classify, BubbleType};
use crate::gfa::{Graph, NodeId, Path};

/// `i8::MIN` mirrors tskit's `MISSING_DATA = -1`; we use -1 here for the same
/// semantic: this haplotype does not pass through the bubble source.
pub const MISSING_GENOTYPE: i8 = -1;

/// Per-bubble site emission record.
#[derive(Debug, Clone)]
pub struct Site {
    pub source: NodeId,
    pub sink: NodeId,
    pub bfs_closed: bool,
    pub bubble_type: BubbleType,
    pub mu: f64,
    /// Reference (CHM13) coordinate at the source node, if the reference path
    /// traverses through. None when the reference does not visit this source
    /// (e.g. a non-reference insertion).
    pub ref_pos: Option<u64>,
    /// Which branch the reference path takes (index into `graph.successors(source)`).
    /// None when the reference does not traverse the source.
    pub ancestral_branch: Option<usize>,
    /// One label per branch — for human inspection and downstream tooling.
    /// tsinfer treats these as opaque strings; semantic interpretation is up
    /// to the caller.
    pub alleles: Vec<String>,
    /// `genotypes[i]` = branch index taken by `panel[i]`, or `MISSING_GENOTYPE`.
    pub genotypes: Vec<i8>,
}

/// Index into a path: maps node id → list of (position-in-path,
/// cumulative-bp-up-to-this-node). The list is in case a path visits the
/// same node twice (rare for impg subgraphs of a single region; we keep the
/// data structure general).
pub struct PathIndex {
    pub by_node: HashMap<NodeId, Vec<(usize, u64)>>,
    pub total_bp: u64,
}

impl PathIndex {
    pub fn build(path: &Path, graph: &Graph) -> Self {
        let mut by_node: HashMap<NodeId, Vec<(usize, u64)>> = HashMap::new();
        let mut bp: u64 = 0;
        for (i, &n) in path.nodes.iter().enumerate() {
            by_node.entry(n).or_default().push((i, bp));
            if let Some(seq) = graph.seq.get(&n) {
                bp += seq.len() as u64;
            }
        }
        PathIndex { by_node, total_bp: bp }
    }
}

/// Parse the absolute reference start coordinate from a PanSN path name like
/// `CHM13#0#chr12:60000000-60010000`. Returns the integer after the last `:`
/// and before `-`. Returns None if the name does not match.
fn parse_ref_offset(name: &str) -> Option<u64> {
    let last_seg = name.rsplit('#').next()?;
    // `chr12:60000000-60010000`
    let after_colon = last_seg.split_once(':').map(|(_, b)| b)?;
    let before_dash = after_colon.split_once('-').map(|(a, _)| a)?;
    before_dash.parse().ok()
}

/// A panel is the ordered list of paths whose genotypes go into the per-site
/// vector. The reference (if any) is tracked separately.
pub struct Panel {
    pub names: Vec<String>,
    pub paths: Vec<Path>,
    pub indices: Vec<PathIndex>,
    pub reference: Option<ReferencePath>,
}

pub struct ReferencePath {
    pub name: String,
    pub path: Path,
    pub index: PathIndex,
    /// Absolute coordinate offset parsed from the path name, or 0 if unparseable.
    pub abs_offset: u64,
}

impl Panel {
    /// Build a panel from the graph, excluding any path whose name starts
    /// with `ref_prefix` and using it as the reference. Sample paths are
    /// returned in lexicographic order for deterministic output.
    pub fn from_graph(graph: &Graph, ref_prefix: &str) -> Self {
        let mut ref_path: Option<Path> = None;
        let mut samples: Vec<Path> = Vec::new();

        for p in &graph.paths {
            if p.name.starts_with(ref_prefix) && ref_path.is_none() {
                ref_path = Some(p.clone());
            } else {
                samples.push(p.clone());
            }
        }

        samples.sort_by(|a, b| a.name.cmp(&b.name));
        let indices: Vec<PathIndex> = samples.iter().map(|p| PathIndex::build(p, graph)).collect();
        let names: Vec<String> = samples.iter().map(|p| p.name.clone()).collect();

        let reference = ref_path.map(|p| {
            let abs_offset = parse_ref_offset(&p.name).unwrap_or(0);
            let index = PathIndex::build(&p, graph);
            ReferencePath { name: p.name.clone(), path: p, index, abs_offset }
        });

        Panel { names, paths: samples, indices, reference }
    }
}

/// Compute the genotype vector for a bubble: which branch each panel path
/// takes at the source node.
fn genotypes_for_bubble(
    bubble: &Bubble,
    graph: &Graph,
    panel: &Panel,
) -> Vec<i8> {
    let succs = graph.successors(bubble.source);
    let succ_to_idx: HashMap<NodeId, i8> = succs
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as i8))
        .collect();

    let mut genotypes = Vec::with_capacity(panel.paths.len());
    for (path, idx) in panel.paths.iter().zip(&panel.indices) {
        let g = lookup_branch(path, idx, bubble.source, &succ_to_idx);
        genotypes.push(g);
    }
    genotypes
}

fn lookup_branch(
    path: &Path,
    idx: &PathIndex,
    source: NodeId,
    succ_to_idx: &HashMap<NodeId, i8>,
) -> i8 {
    let positions = match idx.by_node.get(&source) {
        Some(p) => p,
        None => return MISSING_GENOTYPE,
    };
    // First occurrence of source in this path.
    let (pos_in_path, _) = positions[0];
    if pos_in_path + 1 >= path.nodes.len() {
        return MISSING_GENOTYPE;
    }
    let next = path.nodes[pos_in_path + 1];
    *succ_to_idx.get(&next).unwrap_or(&MISSING_GENOTYPE)
}

/// Build a label per branch. For SNPs the label is the single nucleotide;
/// for indels and microsats it's the actual inserted/repeat sequence (up to
/// 24 bp, then summarised as "<n>bp"); for the "skip" branch (succ == sink)
/// the label is "REF". tsinfer treats these as opaque strings, so the
/// labels are for human inspection only.
fn allele_labels(bubble: &Bubble, graph: &Graph, bubble_type: &BubbleType) -> Vec<String> {
    const MAX_LABEL_BP: usize = 24;
    let succs = graph.successors(bubble.source);
    succs
        .iter()
        .map(|&succ| {
            if succ == bubble.sink && bubble.sink != bubble.source {
                return "REF".to_string();
            }
            let seq = walk_simple_chain_seq(graph, succ, bubble.sink);
            match bubble_type {
                BubbleType::Snp | BubbleType::MultiAllelicSnp if seq.len() == 1 => {
                    (seq[0] as char).to_string()
                }
                _ if seq.is_empty() => "ALT_empty".to_string(),
                _ if seq.len() <= MAX_LABEL_BP => {
                    String::from_utf8_lossy(&seq).to_string()
                }
                _ => format!("{}bp", seq.len()),
            }
        })
        .collect()
}

/// Walk forward from `start` through single-out interior nodes, collecting
/// the base sequence at each visited node. Stops when the next step would
/// land on `sink` (sink's bases are excluded) or when the chain branches.
fn walk_simple_chain_seq(graph: &Graph, start: NodeId, sink: NodeId) -> Vec<u8> {
    if start == sink {
        return Vec::new();
    }
    let mut seq: Vec<u8> = Vec::new();
    let mut cur = start;
    for _ in 0..200 {
        if let Some(s) = graph.seq.get(&cur) {
            seq.extend_from_slice(s);
        }
        let nxt = graph.successors(cur);
        if nxt.len() != 1 {
            return seq;
        }
        let nn = nxt[0];
        if nn == sink {
            return seq;
        }
        cur = nn;
    }
    seq
}

/// Build a Site from a Bubble + Panel + Graph.
pub fn build_site(bubble: Bubble, graph: &Graph, panel: &Panel) -> Site {
    let bubble_type = classify(&bubble, graph);
    let mu = bubble_type.mu_event();
    let bfs_closed = !bubble.branches.is_empty() || bubble.source != bubble.sink;

    let (ref_pos, ancestral_branch) = match panel.reference.as_ref() {
        Some(refp) => {
            let positions = refp.index.by_node.get(&bubble.source);
            if let Some(pos) = positions.and_then(|v| v.first()) {
                let (idx_in_path, cum_bp) = *pos;
                let abs_pos = refp.abs_offset + cum_bp;
                // Which branch does the reference take at this source?
                let succs = graph.successors(bubble.source);
                let succ_to_idx: HashMap<NodeId, usize> = succs
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| (n, i))
                    .collect();
                let anc = if idx_in_path + 1 < refp.path.nodes.len() {
                    let next = refp.path.nodes[idx_in_path + 1];
                    succ_to_idx.get(&next).copied()
                } else {
                    None
                };
                (Some(abs_pos), anc)
            } else {
                (None, None)
            }
        }
        None => (None, None),
    };

    let alleles = allele_labels(&bubble, graph, &bubble_type);
    let genotypes = genotypes_for_bubble(&bubble, graph, panel);

    Site {
        source: bubble.source,
        sink: bubble.sink,
        bfs_closed,
        bubble_type,
        mu,
        ref_pos,
        ancestral_branch,
        alleles,
        genotypes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mk_graph(
        nodes: &[(NodeId, &[u8])],
        links: &[(NodeId, NodeId)],
        paths: &[(&str, &[NodeId])],
    ) -> Graph {
        let mut seq = HashMap::new();
        for (id, s) in nodes {
            seq.insert(*id, s.to_vec());
        }
        let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut backward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (a, b) in links {
            forward.entry(*a).or_default().push(*b);
            backward.entry(*b).or_default().push(*a);
        }
        let ps = paths
            .iter()
            .map(|(n, ns)| Path { name: n.to_string(), nodes: ns.to_vec() })
            .collect();
        Graph { seq, forward, backward, paths: ps }
    }

    #[test]
    fn parse_pansn_ref_offset() {
        assert_eq!(parse_ref_offset("CHM13#0#chr12:60000000-60010000"), Some(60_000_000));
        assert_eq!(parse_ref_offset("HG002#1#chr2:1000-2000"), Some(1000));
        assert_eq!(parse_ref_offset("not_a_pansn_name"), None);
    }

    #[test]
    fn snp_genotypes_two_branches() {
        // Source 1, branches 2(A), 3(T). 4 is sink.
        // Three paths: CHM13 takes A; sample_alt takes T; sample_ref takes A.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"R")],
            &[(1, 2), (1, 3), (2, 4), (3, 4)],
            &[
                ("CHM13#0#chr1:100-200", &[1, 2, 4]),
                ("HG002#1#chr1:100-200", &[1, 3, 4]),
                ("HG003#1#chr1:100-200", &[1, 2, 4]),
            ],
        );
        let panel = Panel::from_graph(&g, "CHM13");
        assert_eq!(panel.names, vec!["HG002#1#chr1:100-200", "HG003#1#chr1:100-200"]);
        assert!(panel.reference.is_some());

        let bubble = crate::bubble::find_bubble(&g, 1, 10).unwrap();
        let site = build_site(bubble, &g, &panel);

        assert_eq!(site.bubble_type, BubbleType::Snp);
        assert_eq!(site.alleles, vec!["A".to_string(), "T".to_string()]);
        // CHM13 took successor 2 (= branch index 0 = "A"), so ancestral = 0.
        assert_eq!(site.ancestral_branch, Some(0));
        // HG002 took 3 (= branch 1), HG003 took 2 (= branch 0).
        assert_eq!(site.genotypes, vec![1i8, 0i8]);
        // Position: CHM13 reaches node 1 after 0 bp (it's the first); abs_offset=100.
        assert_eq!(site.ref_pos, Some(100));
    }

    #[test]
    fn path_missing_through_source_yields_missing_data() {
        // Source 1, branches 2,3 → sink 4. Path that doesn't touch source 1
        // (e.g. an alternative path bypasses it entirely) should get -1.
        let g = mk_graph(
            &[(1, b"L"), (2, b"A"), (3, b"T"), (4, b"R"), (5, b"X")],
            &[(1, 2), (1, 3), (2, 4), (3, 4), (5, 4)],
            &[
                ("CHM13#0#chr1:0-100", &[1, 2, 4]),
                ("HG_nopath#1#chr1:0-100", &[5, 4]),
            ],
        );
        let panel = Panel::from_graph(&g, "CHM13");
        let bubble = crate::bubble::find_bubble(&g, 1, 10).unwrap();
        let site = build_site(bubble, &g, &panel);
        assert_eq!(site.genotypes, vec![MISSING_GENOTYPE]);
    }

    #[test]
    fn ref_pos_includes_cumulative_bp_offset() {
        // Path: nodes [1(L), 2(AB), 3], with bubble at node 3.
        // Cumulative bp at node 3 = 1 + 2 = 3. With abs_offset = 500, ref_pos = 503.
        let g = mk_graph(
            &[(1, b"L"), (2, b"AB"), (3, b"X"), (4, b"A"), (5, b"T"), (6, b"R")],
            &[(1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)],
            &[
                ("CHM13#0#chr1:500-700", &[1, 2, 3, 4, 6]),
                ("HG002#1#chr1:500-700", &[1, 2, 3, 5, 6]),
            ],
        );
        let panel = Panel::from_graph(&g, "CHM13");
        let bubble = crate::bubble::find_bubble(&g, 3, 10).unwrap();
        let site = build_site(bubble, &g, &panel);
        assert_eq!(site.ref_pos, Some(503));
    }
}
