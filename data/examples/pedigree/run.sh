#!/usr/bin/env bash
# Example: founder painting — the ancestry HMM run with individual
# grandparents as states instead of populations. This is the same
# machinery used in the paper to validate against the CEPH 1463
# platinum pedigree.
#
# Input:  input/ibs.tsv           — synthetic pairwise identities (5 Mb) where
#                                    CHILD is a mosaic of 4 grandparents:
#                                      GP1: 0.0–1.25 Mb
#                                      GP2: 1.25–2.5 Mb
#                                      GP3: 2.5–3.75 Mb
#                                      GP4: 3.75–5.0 Mb
#                                    (regenerable via
#                                     scripts/generate_synthetic_pedigree_example.py)
#         input/populations.tsv   — 4 "populations", each = one founder haplotype
#         input/queries.txt       — query haplotypes (CHILD)
# Output: output/painting.tsv     — painted haplotype tracts by founder
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
    --output          output/painting.tsv

echo
echo "Painted segments per founder:"
awk 'NR>1{print $4}' output/painting.tsv | sort | uniq -c
echo
head -5 output/painting.tsv
