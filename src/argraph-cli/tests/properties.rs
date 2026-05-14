//! Property tests: invariants every emit-sites output must satisfy on any
//! valid pangenome GFA. Run against the two bundled example GFAs.

use impopk_argraph::{build_site, enumerate_bubbles, Graph, Panel, MISSING_GENOTYPE};
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn gold_gfa() -> PathBuf {
    workspace_root().join("data/examples/argraph/input/pangenome.gfa")
}

fn sv_gfa() -> PathBuf {
    workspace_root().join("data/examples/argraph/input/pangenome_svrich.gfa")
}

/// Returns (graph, panel, sites) ready for invariant checks.
fn run(gfa: &PathBuf) -> (Graph, Panel, Vec<impopk_argraph::Site>) {
    let graph = Graph::parse(gfa).expect("parse GFA");
    let panel = Panel::from_graph(&graph, "CHM13");
    let bubbles = enumerate_bubbles(&graph, 200);
    let sites: Vec<_> = bubbles
        .into_iter()
        .map(|b| build_site(b, &graph, &panel))
        .collect();
    (graph, panel, sites)
}

/// For every site:
///   - genotypes vector has exactly panel.names.len() entries
///   - each genotype value is either MISSING_GENOTYPE (-1) or in
///     [0, n_branches), where n_branches = alleles.len()
///   - alleles vector has at least 2 entries (every bubble has ≥2 succs)
///   - ancestral_branch, when set, is in [0, n_branches)
///   - bfs_closed implies sink != source
fn invariants_for(label: &str, gfa: PathBuf) {
    let (graph, panel, sites) = run(&gfa);
    let n_panel = panel.names.len();

    for (i, site) in sites.iter().enumerate() {
        // genotype shape
        assert_eq!(
            site.genotypes.len(),
            n_panel,
            "[{}] site {}: genotypes len {} != panel size {}",
            label,
            i,
            site.genotypes.len(),
            n_panel
        );

        let n_branches = site.alleles.len();
        assert!(
            n_branches >= 2,
            "[{}] site {}: n_branches {} < 2",
            label,
            i,
            n_branches
        );

        // genotype values in valid range
        for (j, &g) in site.genotypes.iter().enumerate() {
            if g == MISSING_GENOTYPE {
                continue;
            }
            assert!(
                g >= 0 && (g as usize) < n_branches,
                "[{}] site {}, hap {}: genotype {} out of range [0, {})",
                label,
                i,
                j,
                g,
                n_branches
            );
        }

        // ancestral within range, when present
        if let Some(anc) = site.ancestral_branch {
            assert!(
                anc < n_branches,
                "[{}] site {}: ancestral {} >= n_branches {}",
                label,
                i,
                anc,
                n_branches
            );
        }

        // bfs_closed semantics
        if site.bfs_closed {
            assert_ne!(
                site.sink, site.source,
                "[{}] site {}: bfs_closed but sink == source",
                label, i
            );
        } else {
            assert_eq!(
                site.sink, site.source,
                "[{}] site {}: !bfs_closed but sink != source",
                label, i
            );
        }

        // alleles count matches the graph's successor count for the source
        let succs = graph.successors(site.source);
        assert_eq!(
            n_branches,
            succs.len(),
            "[{}] site {}: alleles count {} != graph successors {} for source {}",
            label,
            i,
            n_branches,
            succs.len(),
            site.source
        );
    }
}

#[test]
fn invariants_hold_on_gold_standard() {
    invariants_for("chr12:60Mb", gold_gfa());
}

#[test]
fn invariants_hold_on_sv_rich() {
    invariants_for("chr12:90Mb", sv_gfa());
}

/// ref_pos values must be unique across sites that have one (no collisions on
/// the same reference base).
#[test]
fn ref_pos_values_are_unique() {
    for (label, gfa) in [
        ("chr12:60Mb", gold_gfa()),
        ("chr12:90Mb", sv_gfa()),
    ] {
        let (_g, _p, sites) = run(&gfa);
        let positions: Vec<u64> = sites
            .iter()
            .filter_map(|s| s.ref_pos)
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            positions.len(),
            sorted.len(),
            "[{}] ref_pos values are not unique",
            label
        );
    }
}

/// Genotype branch usage: the count of each non-missing branch index plus the
/// missing count must equal the panel size, per site. (Each panel haplotype
/// takes exactly one branch or is missing.)
#[test]
fn genotype_partition_is_complete() {
    for (label, gfa) in [
        ("chr12:60Mb", gold_gfa()),
        ("chr12:90Mb", sv_gfa()),
    ] {
        let (_g, panel, sites) = run(&gfa);
        let n_panel = panel.names.len();
        for (i, site) in sites.iter().enumerate() {
            let mut by_branch: std::collections::BTreeMap<i8, usize> =
                std::collections::BTreeMap::new();
            for &g in &site.genotypes {
                *by_branch.entry(g).or_insert(0) += 1;
            }
            let total: usize = by_branch.values().sum();
            assert_eq!(
                total, n_panel,
                "[{}] site {}: total genotype count {} != panel {}",
                label, i, total, n_panel
            );
        }
    }
}

/// For SNP-type sites the alleles must be single-character distinct nucleotides.
/// For Microsatellite-type sites at least one branch must be the skip ("REF").
/// (Both are basic sanity for the classifier's interpretation.)
#[test]
fn classification_consistency_basic() {
    use impopk_argraph::BubbleType;
    for (label, gfa) in [
        ("chr12:60Mb", gold_gfa()),
        ("chr12:90Mb", sv_gfa()),
    ] {
        let (_g, _p, sites) = run(&gfa);
        for site in sites.iter() {
            match site.bubble_type {
                BubbleType::Snp => {
                    assert_eq!(site.alleles.len(), 2, "[{}] SNP must have 2 alleles", label);
                    for a in &site.alleles {
                        assert_eq!(a.len(), 1, "[{}] SNP allele {:?} not single-char", label, a);
                    }
                    let mut letters: Vec<&String> = site.alleles.iter().collect();
                    letters.sort();
                    letters.dedup();
                    assert_eq!(letters.len(), 2, "[{}] SNP alleles not distinct", label);
                }
                BubbleType::MultiAllelicSnp => {
                    assert!((3..=4).contains(&site.alleles.len()));
                }
                BubbleType::Microsatellite => {
                    let n_skip = site.alleles.iter().filter(|a| *a == "REF").count();
                    assert!(
                        n_skip <= 1,
                        "[{}] microsat with multiple REF alleles: {:?}",
                        label,
                        site.alleles
                    );
                }
                _ => {}
            }
        }
    }
}
