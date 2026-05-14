//! Minimal GFA 1.0 parser sufficient for impg-produced subgraphs.
//!
//! Only handles S (segment), L (link), and P (path) lines. Strands on links
//! are assumed all-forward (impg-emitted GFAs from `impg query` produce that).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path as FsPath;

pub type NodeId = u64;

/// A path through the graph (one haplotype's traversal).
#[derive(Debug, Clone)]
pub struct Path {
    pub name: String,
    pub nodes: Vec<NodeId>,
}

/// A pangenome subgraph parsed from a GFA file.
#[derive(Debug)]
pub struct Graph {
    /// Sequence per node (one or more bases; impg emits one base per node).
    pub seq: HashMap<NodeId, Vec<u8>>,
    /// Forward adjacency: node → list of immediate successors.
    pub forward: HashMap<NodeId, Vec<NodeId>>,
    /// Backward adjacency: node → list of immediate predecessors.
    pub backward: HashMap<NodeId, Vec<NodeId>>,
    /// Paths in the graph.
    pub paths: Vec<Path>,
}

impl Graph {
    pub fn parse(path: &FsPath) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let r = BufReader::new(f);

        let mut seq: HashMap<NodeId, Vec<u8>> = HashMap::new();
        let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut backward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut paths: Vec<Path> = Vec::new();

        for (i, line) in r.lines().enumerate() {
            let line = line.with_context(|| format!("reading line {}", i + 1))?;
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split('\t');
            let tag = fields.next().unwrap_or("");
            match tag {
                "H" => continue,
                "S" => {
                    let id: NodeId = fields
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("S line missing id (line {})", i + 1))?
                        .parse()
                        .with_context(|| format!("parsing S id on line {}", i + 1))?;
                    let s = fields
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("S line missing seq (line {})", i + 1))?;
                    seq.insert(id, s.as_bytes().to_vec());
                }
                "L" => {
                    let from: NodeId = fields
                        .next()
                        .and_then(|f| f.parse().ok())
                        .ok_or_else(|| anyhow::anyhow!("L line bad from on line {}", i + 1))?;
                    let _from_strand = fields.next();
                    let to: NodeId = fields
                        .next()
                        .and_then(|f| f.parse().ok())
                        .ok_or_else(|| anyhow::anyhow!("L line bad to on line {}", i + 1))?;
                    let _to_strand = fields.next();
                    forward.entry(from).or_default().push(to);
                    backward.entry(to).or_default().push(from);
                }
                "P" => {
                    let name = fields
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("P line missing name on line {}", i + 1))?
                        .to_string();
                    let path_str = fields
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("P line missing path on line {}", i + 1))?;
                    let mut nodes = Vec::new();
                    for token in path_str.split(',') {
                        // Strip optional strand suffix (+ or -).
                        let token = token.trim_end_matches(|c: char| c == '+' || c == '-');
                        let id: NodeId = token
                            .parse()
                            .with_context(|| format!("parsing path node on line {}", i + 1))?;
                        nodes.push(id);
                    }
                    paths.push(Path { name, nodes });
                }
                _ => continue, // ignore other tag types (E, W, etc.)
            }
        }

        if paths.is_empty() {
            bail!("no paths found in {}", path.display());
        }
        Ok(Graph { seq, forward, backward, paths })
    }

    /// Successors of a node, empty slice if none.
    pub fn successors(&self, n: NodeId) -> &[NodeId] {
        self.forward.get(&n).map_or(&[], |v| v.as_slice())
    }

    /// Predecessors of a node, empty slice if none.
    pub fn predecessors(&self, n: NodeId) -> &[NodeId] {
        self.backward.get(&n).map_or(&[], |v| v.as_slice())
    }
}
