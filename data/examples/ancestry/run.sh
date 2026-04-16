#!/usr/bin/env bash
# Example: run the N-state local ancestry HMM on a precomputed identity TSV.
#
# Input:  input/ibs.tsv           — windowed pairwise identities (chr12, 50 Mb
#                                    at 10 kb resolution)
#         input/populations.tsv   — TSV: population, haplotype_id
#         input/queries.txt       — query haplotypes (one per line)
# Output: output/ancestry.tsv     — painted ancestry tracts per query haplotype
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

ANC_BIN=${ANC_BIN:-../../../target/release/ancestry}

"$ANC_BIN" \
    --similarity-file input/ibs.tsv \
    --window-size     10000 \
    --populations     input/populations.tsv \
    --query-samples   input/queries.txt \
    --emission-model  max \
    --estimate-params \
    --threads         4 \
    --output          output/ancestry.tsv

echo
echo "Ancestry segments per query:"
awk 'NR>1{print $4}' output/ancestry.tsv | sort | uniq -c
echo
head -5 output/ancestry.tsv
