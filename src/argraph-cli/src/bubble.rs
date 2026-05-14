//! Bubble enumeration on a pangenome graph.
//!
//! A bubble here is a pair (source, sink) such that every successor of `source`
//! reaches `sink` (forward) before reconverging elsewhere. The "branches"
//! are the internal node sequences between source and sink, exclusive of both.
//!
//! v0.1: top-level bubbles only — bubbles nested inside a branch are not
//! enumerated separately. Sufficient for impg-emitted GFAs of short regions.

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

    for _step in 0..max_depth {
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
            let branches = reconstruct_branches(&parents, &depth, succs, sink);
            return Some(Bubble { source, sink, branches });
        }

        // Expand all frontiers by one step.
        let mut next_frontier: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut any_progress = false;
        for i in 0..n {
            let cur_depth = _step + 1;
            for &node in &frontier[i] {
                for &m in graph.successors(node) {
                    if !depth[i].contains_key(&m) {
                        depth[i].insert(m, cur_depth);
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
    depth: &[HashMap<NodeId, usize>],
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
        // path now starts at succs[i] and ends at sink. Internal = drop sink.
        if let Some(&last) = path.last() {
            if last == sink {
                path.pop();
            }
        }
        // Safety: ensure path[0] == succs[i]
        if path.first() != Some(&succs[i]) {
            // Reconstruction failed; emit empty branch as fallback.
            // Should not happen given how depth/parents were populated.
            let _ = depth; // silence unused
            branches.push(Vec::new());
            continue;
        }
        branches.push(path);
    }
    branches
}

/// Enumerate all top-level bubbles in the graph.
///
/// A node is a candidate source iff it has ≥2 outgoing edges. Each candidate
/// is processed independently; bubbles nested inside another bubble's branch
/// are not deduplicated in v0.1.
pub fn enumerate_bubbles(graph: &Graph, max_depth: usize) -> Vec<Bubble> {
    let mut sources: Vec<NodeId> = graph
        .forward
        .iter()
        .filter(|(_, v)| v.len() >= 2)
        .map(|(k, _)| *k)
        .collect();
    sources.sort_unstable();
    sources
        .into_iter()
        .filter_map(|s| find_bubble(graph, s, max_depth))
        .collect()
}
