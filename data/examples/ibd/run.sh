#!/usr/bin/env bash
# Example: run the 2-state IBD HMM on a precomputed pairwise identity TSV.
#
# Input:  input/ibs.tsv       — windowed pairwise identities (chr12, EUR panel,
#                                5 Mb region at 10 kb resolution)
# Output: output/ibd.tsv      — detected IBD segments per pair
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

IBD_BIN=${IBD_BIN:-../../../target/release/ibd}

"$IBD_BIN" \
    --similarity-file input/ibs.tsv \
    --region          chr12:15000000-20000000 \
    --region-length   5000000 \
    --size            10000 \
    --min-len-bp      2000000 \
    --population      EUR \
    --threads         4 \
    --output          output/ibd.tsv

echo
echo "Segments detected:"
wc -l output/ibd.tsv
echo
head -5 output/ibd.tsv
