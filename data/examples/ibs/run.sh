#!/usr/bin/env bash
# Example: inspect IBS enrichment in a windowed pairwise identity TSV.
#
# This example does NOT run a binary — the IBS table is the input produced
# by `ibs` (which wraps `impg similarity`). Here we show how to compute
# a simple summary: fraction of within-population pairs with identity
# above the IBS threshold (0.9999 by default), per window.
#
# Input:  input/ibs.tsv      — windowed pairwise identities (chr12, EUR, 1 Mb)
#         input/subset.txt   — haplotypes included in the panel
# Output: output/ibs_enrichment.tsv  — fraction of IBS pairs per 10 kb window
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output

THRESHOLD=0.9999

{
    # Header first, then sorted data rows (so `head` always shows the header)
    echo -e "chrom\tstart\tend\tn_pairs\tn_ibs\tfrac_ibs"
    awk -v T=$THRESHOLD '
    BEGIN { FS=OFS="\t" }
    NR==1 { next }
    {
        key = $1 "\t" $2 "\t" $3
        total[key]++
        if ($6+0 >= T) hi[key]++
    }
    END {
        for (k in total) {
            h = (k in hi) ? hi[k] : 0
            printf "%s\t%d\t%d\t%.6f\n", k, total[k], h, h/total[k]
        }
    }' input/ibs.tsv | sort -k1,1 -k2,2n
} > output/ibs_enrichment.tsv

echo "Windows summarised:"
wc -l output/ibs_enrichment.tsv
echo
echo "Top IBS enrichment windows:"
sort -k6,6gr output/ibs_enrichment.tsv | head -10
