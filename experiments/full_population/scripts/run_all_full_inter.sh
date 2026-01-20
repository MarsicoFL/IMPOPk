#!/usr/bin/env bash
set -euo pipefail

# Full Population Inter-Population IBS Analysis
# Uses ALL available haplotypes per ancestry for LCT region
# Runs all 10 pairwise inter-population comparisons

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs.sh"
SAMPLE_DIR="$BASE_DIR/sample_lists"
RESULTS_DIR="$SCRIPT_DIR/../inter"

REGION="chr2:130787850-140837183"
WINDOW_SIZE=5000
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

POPULATIONS=("AFR" "EUR" "EAS" "CSA" "AMR")

mkdir -p "$RESULTS_DIR"

run_inter() {
    local POP1=$1
    local POP2=$2
    local SAMPLE_FILE1="$SAMPLE_DIR/HPRCv2_${POP1}_full.txt"
    local SAMPLE_FILE2="$SAMPLE_DIR/HPRCv2_${POP2}_full.txt"
    local OUTPUT_DIR="$RESULTS_DIR/FULL-${POP1}-${POP2}"
    local OUTPUT_FILE="$OUTPUT_DIR/FULL-${POP1}-${POP2}_ibs.tsv"
    local COMBINED_LIST="$OUTPUT_DIR/combined_samples.txt"

    if [[ ! -f "$SAMPLE_FILE1" ]] || [[ ! -f "$SAMPLE_FILE2" ]]; then
        echo "ERROR: Sample files not found" >&2
        return 1
    fi

    mkdir -p "$OUTPUT_DIR"

    # Combine sample lists
    cat "$SAMPLE_FILE1" "$SAMPLE_FILE2" > "$COMBINED_LIST"

    # New format: 1 line per individual, 2 haplotypes each
    IND1=$(wc -l < "$SAMPLE_FILE1")
    IND2=$(wc -l < "$SAMPLE_FILE2")
    HAP1=$(( IND1 * 2 ))
    HAP2=$(( IND2 * 2 ))
    CROSS_PAIRS=$(( HAP1 * HAP2 ))

    echo "=== Running FULL-${POP1}-${POP2}: ${HAP1}×${HAP2} = $CROSS_PAIRS cross-pairs ===" >&2

    START_TIME=$(date +%s.%N)

    "$IBS_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$COMBINED_LIST" \
        --output "$OUTPUT_FILE"

    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    RECORDS=$(wc -l < "$OUTPUT_FILE")
    RECORDS=$((RECORDS - 1))  # Subtract header

    echo "  Completed: ${RUNTIME}s, ${RECORDS} records" >&2
    echo "${POP1}-${POP2},${HAP1},${HAP2},${CROSS_PAIRS},${RUNTIME},${RECORDS}"
}

# Run single pair if specified, otherwise run all
if [[ $# -ge 2 ]]; then
    run_inter "$1" "$2"
else
    echo "comparison,hap1,hap2,cross_pairs,runtime_seconds,records" > "$RESULTS_DIR/inter_metrics.csv"

    # All 10 pairwise comparisons
    for ((i=0; i<${#POPULATIONS[@]}; i++)); do
        for ((j=i+1; j<${#POPULATIONS[@]}; j++)); do
            run_inter "${POPULATIONS[$i]}" "${POPULATIONS[$j]}" >> "$RESULTS_DIR/inter_metrics.csv"
        done
    done

    echo ""
    echo "All inter-population experiments complete."
    cat "$RESULTS_DIR/inter_metrics.csv"
fi
