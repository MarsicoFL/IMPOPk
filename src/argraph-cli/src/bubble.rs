//! Bubble enumeration on a pangenome graph.
//!
//! A bubble here is a pair (source, sink) such that every successor of `source`
//! reaches `sink` (forward) before reconverging elsewhere. The "branches"
//! are the internal node sequences between source and sink, exclusive of both.
//!
//! v0.1: top-level bubbles only — bubbles nested inside a branch are not
//! enumerated separately. Sufficient for impg-emitted GFAs of short regions.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::gfa::{Graph, NodeId};

/// A bubble found in the graph.
#[derive(Debug, Clone)]
pub struct Bubble {
    pub source: NodeId,
    pub sink: NodeId,
    /// One internal-node list per branch (in order of `Graph::successors(source)`).
    /// An empty branch means the branch goes directly from source to sink.
    pub branches: Vec<Vec<NodeId>>,
}

impl Bubble {
    pub fn n_branches(&self) -> usize {
        self.branches.len()
    }
}

/// Find the bubble rooted at `source` if one closes within `max_depth` steps.
///
/// Uses parallel BFS from each immediate successor, tracking reachable sets
/// per branch. The sink is the node reachable from every branch with the
/// smallest maximum depth across branches.
pub fn find_bubble(graph: &Graph, source: NodeId, max_depth: usize) -> Option<Bubble> {
    let succs = graph.successors(source);
    if succs.len() < 2 {
        return None;
    }
    let n = succs.len();

    let mut depth: Vec<HashMap<NodeId, usize>> = vec![HashMap::new(); n];
    let mut parents: Vec<HashMap<NodeId, NodeId>> = vec![HashMap::new(); n];
    let mut frontier: Vec<Vec<NodeId>> = vec![Vec::new(); n];

    for (i, &s) in succs.iter().enumerate() {
        depth[i].insert(s, 0);
        frontier[i].push(s);
    }

    for step in 0..max_depth {
        // Candidates: nodes present in every branch's reachable set.
        let candidates: Vec<NodeId> = depth[0]
            .keys()
            .copied()
            .filter(|n_| depth[1..].iter().all(|m| m.contains_key(n_)))
            .collect();
        if !candidates.is_empty() {
            // Pick the sink as the node minimizing max-depth across branches
            // (= earliest reachable by the slowest branch).
            let sink = *candidates
                .iter()
                .min_by_key(|n_| depth.iter().map(|d| d[n_]).max().unwrap_or(usize::MAX))
                .unwrap();
            let branches = reconstruct_branches(&parents, succs, sink);
            return Some(Bubble { source, sink, branches });
        }

        // Expand all frontiers by one step.
        let cur_depth = step + 1;
        let mut next_frontier: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut any_progress = false;
        for i in 0..n {
            for &node in &frontier[i] {
                for &m in graph.successors(node) {
                    if let Entry::Vacant(e) = depth[i].entry(m) {
                        e.insert(cur_depth);
                        parents[i].insert(m, node);
                        next_frontier[i].push(m);
                        any_progress = true;
                    }
                }
            }
        }
        frontier = next_frontier;
        if !any_progress {
            return None;
        }
    }
    None
}

fn reconstruct_branches(
    parents: &[HashMap<NodeId, NodeId>],
    succs: &[NodeId],
    sink: NodeId,
) -> Vec<Vec<NodeId>> {
    let n = succs.len();
    let mut branches = Vec::with_capacity(n);
    for i in 0..n {
        // If the immediate successor IS the sink, branch is empty.
        if succs[i] == sink {
            branches.push(Vec::new());
            continue;
        }
        // Walk back from sink through parents[i] until we hit succs[i].
        let mut path = vec![sink];
        let mut cur = sink;
        while let Some(&p) = parents[i].get(&cur) {
            cur = p;
            path.push(cur);
            if cur == succs[i] {
                break;
            }
        }
        path.reverse();
        // path now starts at succs[i] and ends at sink. Drop the sink so the
        // branch is just the internal nodes.
        if path.last() == Some(&sink) {
            path.pop();
        }
        // Sanity: path[0] should be succs[i]. If not, the BFS state was
        // inconsistent — fall back to an empty branch rather than panic.
        if path.first() != Some(&succs[i]) {
            branches.push(Vec::new());
            continue;
        }
        branches.push(path);
    }
    branches
}

/// All nodes with ≥2 outgoing edges, sorted by node id for determinism.
/// These are the candidate sources for classification — every such node
/// represents a bubble in the broad sense (a branching point in the graph),
/// whether or not its BFS sink converges.
pub fn enumerate_sources(graph: &Graph) -> Vec<NodeId> {
    let mut sources: Vec<NodeId> = graph
        .forward
        .iter()
        .filter(|(_, v)| v.len() >= 2)
        .map(|(k, _)| *k)
        .collect();
    sources.sort_unstable();
    sources
}

/// Enumerate sources, returning a Bubble per source. When BFS converges,
/// the bubble has a real sink and reconstructed branches. When it doesn't
/// (e.g. interior structure that doesn't close within `max_depth`), we still
/// emit a bubble with `sink = source` and empty `branches` so the classifier
/// can still walk from the source and decide a type (typically Complex).
pub fn enumerate_bubbles(graph: &Graph, max_depth: usize) -> Vec<Bubble> {
    enumerate_sources(graph)
        .into_iter()
        .map(|s| {
            find_bubble(graph, s, max_depth).unwrap_or(Bubble {
                source: s,
                sink: s,
                branches: Vec::new(),
            })
        })
        .collect()
}
