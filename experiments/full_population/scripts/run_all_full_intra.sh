#!/usr/bin/env bash
set -euo pipefail

# Full Population Intra-Population IBS Analysis
# Uses ALL available haplotypes per ancestry for LCT region

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs.sh"
SAMPLE_DIR="$BASE_DIR/sample_lists"
RESULTS_DIR="$SCRIPT_DIR/../intra"

REGION="chr2:130787850-140837183"
WINDOW_SIZE=5000
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

POPULATIONS=("AFR" "EUR" "EAS" "CSA" "AMR")

run_intra() {
    local POP=$1
    local SAMPLE_FILE="$SAMPLE_DIR/HPRCv2_${POP}_full.txt"
    local OUTPUT_DIR="$RESULTS_DIR/FULL-${POP}-INTRA"
    local OUTPUT_FILE="$OUTPUT_DIR/FULL-${POP}-INTRA_ibs.tsv"

    if [[ ! -f "$SAMPLE_FILE" ]]; then
        echo "ERROR: Sample file not found: $SAMPLE_FILE" >&2
        return 1
    fi

    mkdir -p "$OUTPUT_DIR"

    # New format: 1 line per individual, 2 haplotypes each
    IND_COUNT=$(wc -l < "$SAMPLE_FILE")
    HAP_COUNT=$(( IND_COUNT * 2 ))
    PAIRS=$(( HAP_COUNT * (HAP_COUNT - 1) / 2 ))

    echo "=== Running FULL-${POP}-INTRA: $HAP_COUNT haplotypes, $PAIRS pairs ===" >&2

    START_TIME=$(date +%s.%N)

    "$IBS_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$SAMPLE_FILE" \
        --output "$OUTPUT_FILE"

    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    RECORDS=$(wc -l < "$OUTPUT_FILE")
    RECORDS=$((RECORDS - 1))  # Subtract header

    echo "  Completed: ${RUNTIME}s, ${RECORDS} records" >&2
    echo "${POP},${HAP_COUNT},${PAIRS},${RUNTIME},${RECORDS}"
}

# Run single population if specified, otherwise run all
if [[ $# -ge 1 ]]; then
    run_intra "$1"
else
    echo "population,haplotypes,pairs,runtime_seconds,records" > "$RESULTS_DIR/intra_metrics.csv"
    for POP in "${POPULATIONS[@]}"; do
        run_intra "$POP" >> "$RESULTS_DIR/intra_metrics.csv"
    done
    echo ""
    echo "All intra-population experiments complete."
    cat "$RESULTS_DIR/intra_metrics.csv"
fi
