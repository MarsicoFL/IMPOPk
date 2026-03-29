#!/usr/bin/env bash
# generate_sample_lists.sh — Generate per-superpopulation sample lists from PAF + IGSR metadata
#
# Creates data/samples/{AFR,EUR,EAS,CSA,AMR}.txt with haplotype IDs (SAMPLE#1, SAMPLE#2).
#
# Data sources:
#   1. PAF alignment file: extract all sample IDs present in the pangenome
#   2. 1000 Genomes 3202-sample PED file: superpopulation for most samples
#   3. HPRC Year 1 metadata (GitHub): superpopulation for HPRC-specific samples
#   4. Manual annotations: GIAB samples (HG002=EUR, HG005=EAS) and others not in above sources
#
# Usage:
#   ./scripts/generate_sample_lists.sh [PAF_FILE]
#
# If PAF_FILE is not provided, defaults to data/alignments/hprc465vschm13.aln.paf.gz
#
# Requirements: curl, zcat (or gzip -dc), awk, sort, comm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/data/samples"
TMPDIR_BASE="${TMPDIR:-/tmp}/impopk_sample_gen_$$"

# --- Configuration ---
PAF_FILE="${1:-}"
if [ -z "$PAF_FILE" ]; then
    # Try common locations
    for candidate in \
        "${PROJECT_DIR}/data/alignments/hprc465vschm13.aln.paf.gz"; do
        if [ -f "$candidate" ]; then
            PAF_FILE="$candidate"
            break
        fi
    done
    if [ -z "$PAF_FILE" ]; then
        echo "ERROR: No PAF file found. Provide path as first argument." >&2
        echo "Usage: $0 <paf_file.gz>" >&2
        exit 1
    fi
fi

echo "=== impopk: Generate Sample Lists ==="
echo "PAF file: $PAF_FILE"
echo "Output:   $OUTPUT_DIR"
echo ""

# --- Setup ---
mkdir -p "$OUTPUT_DIR" "$TMPDIR_BASE"
trap 'rm -rf "$TMPDIR_BASE"' EXIT

# --- Step 1: Extract unique sample IDs from PAF ---
echo "[1/5] Extracting sample IDs from PAF..."
zcat "$PAF_FILE" \
    | cut -f1 \
    | grep -v "^GRCh38" \
    | grep -v "^chm13" \
    | sed 's/#[12]#.*//' \
    | sort -u \
    > "$TMPDIR_BASE/paf_samples.txt"

N_SAMPLES=$(wc -l < "$TMPDIR_BASE/paf_samples.txt")
echo "  Found $N_SAMPLES unique sample IDs"

# --- Step 2: Download 1000 Genomes 3202-sample metadata ---
echo "[2/5] Downloading 1000 Genomes metadata (3202 samples)..."
IGSR_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"
curl -sL "$IGSR_URL" > "$TMPDIR_BASE/igsr_ped.txt"

# Extract SampleID + Superpopulation (columns 2 and 7, skip header)
tail -n +2 "$TMPDIR_BASE/igsr_ped.txt" \
    | awk '{print $2, $7}' \
    > "$TMPDIR_BASE/igsr_superpop.txt"

echo "  Downloaded $(wc -l < "$TMPDIR_BASE/igsr_superpop.txt") sample records"

# --- Step 3: Download HPRC Year 1 metadata ---
echo "[3/5] Downloading HPRC Year 1 sample metadata..."
HPRC_URL="https://raw.githubusercontent.com/human-pangenomics/HPP_Year1_Assemblies/main/sample_metadata/hprc_year1_assemblies_v2_sample_metadata.txt"
curl -sL "$HPRC_URL" > "$TMPDIR_BASE/hprc_y1.txt"

# Extract Sample + Superpopulation (columns 1 and 6, skip header)
# Some HPRC samples have empty superpopulation — those need manual handling
tail -n +2 "$TMPDIR_BASE/hprc_y1.txt" \
    | awk -F'\t' '$6 != "" {print $1, $6}' \
    > "$TMPDIR_BASE/hprc_superpop.txt"

echo "  Downloaded $(wc -l < "$TMPDIR_BASE/hprc_superpop.txt") HPRC sample records with superpopulation"

# --- Step 4: Classify each PAF sample ---
echo "[4/5] Classifying samples by superpopulation..."

# Manual annotations for samples not in IGSR or HPRC metadata
# These are verified from Coriell catalog and GIAB project documentation:
#   HG002 = GIAB Ashkenazi Jewish trio son (NA24385) -> EUR
#   HG005 = GIAB Han Chinese trio son (NA24631) -> EAS
#   HG03471 = Mende in Sierra Leone (MSL) -> AFR
#   HG06807 = African American, St. Louis, MO -> AFR
#   NA21309 = Maasai in Kinyawa, Kenya (MKK) -> AFR
cat > "$TMPDIR_BASE/manual_superpop.txt" << 'MANUAL_EOF'
HG002 EUR
HG005 EAS
HG03471 AFR
HG06807 AFR
NA21309 AFR
MANUAL_EOF

> "$TMPDIR_BASE/classified.txt"
> "$TMPDIR_BASE/unclassified.txt"

while read -r sample; do
    # Try 1KG 3202 first (most comprehensive)
    superpop=$(awk -v s="$sample" '$1 == s {print $2; exit}' "$TMPDIR_BASE/igsr_superpop.txt")

    # Try HPRC Year 1 metadata
    if [ -z "$superpop" ]; then
        superpop=$(awk -v s="$sample" '$1 == s {print $2; exit}' "$TMPDIR_BASE/hprc_superpop.txt")
    fi

    # Try manual annotations
    if [ -z "$superpop" ]; then
        superpop=$(awk -v s="$sample" '$1 == s {print $2; exit}' "$TMPDIR_BASE/manual_superpop.txt")
    fi

    # Normalize SAS -> CSA (project convention: Central/South Asian)
    if [ "$superpop" = "SAS" ]; then
        superpop="CSA"
    fi

    if [ -n "$superpop" ]; then
        echo "$sample $superpop" >> "$TMPDIR_BASE/classified.txt"
    else
        echo "$sample" >> "$TMPDIR_BASE/unclassified.txt"
    fi
done < "$TMPDIR_BASE/paf_samples.txt"

N_CLASSIFIED=$(wc -l < "$TMPDIR_BASE/classified.txt")
N_UNCLASSIFIED=$(wc -l < "$TMPDIR_BASE/unclassified.txt")
echo "  Classified: $N_CLASSIFIED / $N_SAMPLES"

if [ "$N_UNCLASSIFIED" -gt 0 ]; then
    echo "  WARNING: $N_UNCLASSIFIED samples could not be classified:"
    cat "$TMPDIR_BASE/unclassified.txt" | sed 's/^/    /'
    echo "  Add these samples to the manual_superpop.txt section in this script."
fi

# --- Step 5: Generate per-population files ---
echo "[5/5] Writing population files..."

for pop in AFR EUR EAS CSA AMR; do
    outfile="$OUTPUT_DIR/${pop}.txt"
    awk -v p="$pop" '$2 == p {print $1"#1"; print $1"#2"}' "$TMPDIR_BASE/classified.txt" \
        | sort \
        > "$outfile"
    n_haps=$(wc -l < "$outfile")
    n_indiv=$((n_haps / 2))
    echo "  $pop: $n_indiv individuals, $n_haps haplotypes -> $outfile"
done

TOTAL_HAPS=$(cat "$OUTPUT_DIR"/*.txt | wc -l)
TOTAL_INDIV=$((TOTAL_HAPS / 2))

echo ""
echo "=== Done ==="
echo "Total: $TOTAL_INDIV individuals, $TOTAL_HAPS haplotypes"
echo "Files written to: $OUTPUT_DIR/"
echo ""

if [ "$N_UNCLASSIFIED" -gt 0 ]; then
    echo "NOTE: $N_UNCLASSIFIED samples were not classified and are excluded."
    echo "      See warnings above."
    exit 1
fi
