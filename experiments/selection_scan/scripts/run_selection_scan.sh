#!/usr/bin/env bash
set -euo pipefail

# Multi-Region Selection Scan
# Tests IBS patterns across known selection loci
# Compares target population vs AFR control

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs_parallel.sh"
SAMPLE_DIR="$BASE_DIR/sample_lists"
RESULTS_DIR="$SCRIPT_DIR/../regions"

WINDOW_SIZE=5000

# Parallel jobs (leave 3 cores free)
TOTAL_CORES=$(nproc)
JOBS=$((TOTAL_CORES - 3))
if [[ $JOBS -lt 1 ]]; then JOBS=1; fi
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Define regions: NAME|CHROM|START|END|TARGET_POP
REGIONS=(
    "LCT|chr2|130787850|140837183|EUR"
    "SLC24A5|chr15|48000000|50000000|EUR"
    "EDAR|chr2|108000000|110000000|EAS"
    "HBB|chr11|5200000|5300000|AFR"
    "DARC|chr1|159000000|160000000|AFR"
)

run_region() {
    local REGION_DEF=$1
    IFS='|' read -r NAME CHROM START END TARGET_POP <<< "$REGION_DEF"

    local REGION="${CHROM}:${START}-${END}"
    local REGION_DIR="$RESULTS_DIR/$NAME"

    mkdir -p "$REGION_DIR"

    echo "=== Processing $NAME ($REGION) ===" >&2
    echo "    Target: $TARGET_POP, Control: AFR" >&2

    # Run for target population (use full sample lists)
    local TARGET_FILE="$SAMPLE_DIR/HPRCv2_${TARGET_POP}_full.txt"
    local TARGET_OUTPUT="$REGION_DIR/${NAME}_${TARGET_POP}_ibs.tsv"

    if [[ -f "$TARGET_FILE" ]]; then
        echo "  Running $TARGET_POP..." >&2
        "$IBS_SCRIPT" \
            --sequence-files "$SEQ_FILES" \
            -a "$ALIGN_FILE" \
            -r "$REF_NAME" \
            -region "$REGION" \
            -size "$WINDOW_SIZE" \
            --subset-sequence-list "$TARGET_FILE" \
            --output "$TARGET_OUTPUT" \
            -j "$JOBS"
    fi

    # Run for AFR control (if not already target)
    if [[ "$TARGET_POP" != "AFR" ]]; then
        local AFR_FILE="$SAMPLE_DIR/HPRCv2_AFR_full.txt"
        local AFR_OUTPUT="$REGION_DIR/${NAME}_AFR_ibs.tsv"

        echo "  Running AFR control..." >&2
        "$IBS_SCRIPT" \
            --sequence-files "$SEQ_FILES" \
            -a "$ALIGN_FILE" \
            -r "$REF_NAME" \
            -region "$REGION" \
            -size "$WINDOW_SIZE" \
            --subset-sequence-list "$AFR_FILE" \
            --output "$AFR_OUTPUT" \
            -j "$JOBS"
    fi

    echo "  $NAME complete." >&2
}

# Run single region if specified
if [[ $# -ge 1 ]]; then
    for REGION_DEF in "${REGIONS[@]}"; do
        if [[ "$REGION_DEF" == "$1|"* ]]; then
            run_region "$REGION_DEF"
            exit 0
        fi
    done
    echo "ERROR: Unknown region: $1" >&2
    echo "Available: LCT, SLC24A5, EDAR, HBB, DARC" >&2
    exit 1
fi

# Run all regions
echo "Running selection scan across ${#REGIONS[@]} regions..." >&2
for REGION_DEF in "${REGIONS[@]}"; do
    run_region "$REGION_DEF"
done

echo ""
echo "Selection scan complete. Results in: $RESULTS_DIR"
