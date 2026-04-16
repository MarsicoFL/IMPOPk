#!/usr/bin/env bash
# Example: run the 2-state IBD HMM on a synthetic IBS TSV with known IBD.
#
# Input:  input/ibs.tsv       — synthetic pairwise identities over a 5 Mb
#                                 region with two known IBD segments:
#                                   SIM1#1 × SIM2#1 : 1.25–3.75 Mb (2.5 Mb)
#                                   SIM1#2 × SIM3#1 : 0.50–2.00 Mb (1.5 Mb)
#                                 (regenerable via scripts/generate_synthetic_ibd_example.py)
# Output: output/ibd.tsv      — detected IBD segments per pair
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

IBD_BIN=${IBD_BIN:-../../../target/release/ibd}

"$IBD_BIN" \
    --similarity-file input/ibs.tsv \
    --region          chr12:0-5000000 \
    --region-length   5000000 \
    --size            10000 \
    --min-len-bp      1000000 \
    --population      Generic \
    --threads         4 \
    --output          output/ibd.tsv

echo
echo "Segments detected:"
wc -l output/ibd.tsv
echo
head -5 output/ibd.tsv
