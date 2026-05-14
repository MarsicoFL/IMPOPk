#!/usr/bin/env bash
# Example: classify bubbles in a small pangenome subgraph (chr12:60-60.01 Mb).
#
# Input:  input/pangenome.gfa — 21 paths × 10 kb extracted from HPRC v2 via
#                                 `impg query`. Gold-standard region for the
#                                 typology classifier (panarg corpus).
# Output: output/bubbles.tsv  — one row per top-level bubble with type, μ
#                                 estimate, and per-branch lengths.
#
# Expected:
#   bubbles     20
#   indel        1
#   microsat     1
#   snp         18

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

ARG_BIN=${ARG_BIN:-../../../target/release/argraph}

"$ARG_BIN" classify \
    --gfa  input/pangenome.gfa \
    --output output/bubbles.tsv

echo
echo "Bubble counts by type:"
awk -F'\t' 'NR>1 {c[$5]++} END {for (k in c) printf "  %-10s %d\n", k, c[k]}' \
    output/bubbles.tsv | sort

echo
echo "First rows:"
column -t -s $'\t' output/bubbles.tsv | head -10
