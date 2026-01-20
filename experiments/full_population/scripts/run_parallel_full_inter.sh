#!/usr/bin/env bash
set -euo pipefail

# Parallel Full Population Inter-Population IBS Analysis
# Uses ibs_parallel.sh for window-level parallelization (fastest)
# Runs all 10 inter-population comparisons sequentially

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs_parallel.sh"
SAMPLE_DIR="$BASE_DIR/sample_lists"
RESULTS_DIR="$SCRIPT_DIR/../inter"

REGION="chr2:130787850-140837183"
WINDOW_SIZE=5000
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Leave 3 cores free
TOTAL_CORES=$(nproc)
MAX_JOBS=$((TOTAL_CORES - 3))
if [[ $MAX_JOBS -lt 1 ]]; then MAX_JOBS=1; fi

mkdir -p "$RESULTS_DIR"

run_single_inter() {
    local PAIR=$1
    local POP1="${PAIR%-*}"
    local POP2="${PAIR#*-}"

    local SAMPLE_FILE1="$SAMPLE_DIR/HPRCv2_${POP1}_full.txt"
    local SAMPLE_FILE2="$SAMPLE_DIR/HPRCv2_${POP2}_full.txt"
    local OUTPUT_DIR="$RESULTS_DIR/FULL-${POP1}-${POP2}"
    local OUTPUT_FILE="$OUTPUT_DIR/FULL-${POP1}-${POP2}_ibs.tsv"
    local COMBINED_LIST="$OUTPUT_DIR/combined_samples.txt"

    mkdir -p "$OUTPUT_DIR"
    cat "$SAMPLE_FILE1" "$SAMPLE_FILE2" > "$COMBINED_LIST"

    IND1=$(wc -l < "$SAMPLE_FILE1")
    IND2=$(wc -l < "$SAMPLE_FILE2")
    HAP1=$(( IND1 * 2 ))
    HAP2=$(( IND2 * 2 ))
    CROSS_PAIRS=$(( HAP1 * HAP2 ))

    echo "[$(date +%H:%M:%S)] Starting ${POP1}-${POP2}: ${HAP1}×${HAP2} = $CROSS_PAIRS cross-pairs" >&2

    START_TIME=$(date +%s.%N)

    "$IBS_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$COMBINED_LIST" \
        --output "$OUTPUT_FILE" \
        -j "$MAX_JOBS"

    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    RECORDS=$(wc -l < "$OUTPUT_FILE")
    RECORDS=$((RECORDS - 1))

    echo "[$(date +%H:%M:%S)] Completed ${POP1}-${POP2}: ${RUNTIME}s, ${RECORDS} records" >&2
    echo "${POP1}-${POP2},${HAP1},${HAP2},${CROSS_PAIRS},${RUNTIME},${RECORDS}"
}

# All 10 pairwise comparisons
COMPARISONS=(
    "AFR-EUR" "AFR-EAS" "AFR-CSA" "AFR-AMR"
    "EUR-EAS" "EUR-CSA" "EUR-AMR"
    "EAS-CSA" "EAS-AMR"
    "CSA-AMR"
)

echo "=== Running ${#COMPARISONS[@]} inter-population experiments ===" >&2
echo "Each experiment uses $MAX_JOBS parallel window jobs" >&2
echo "" >&2

echo "comparison,hap1,hap2,cross_pairs,runtime_seconds,records" > "$RESULTS_DIR/inter_metrics.csv"

for PAIR in "${COMPARISONS[@]}"; do
    POP1="${PAIR%-*}"
    POP2="${PAIR#*-}"
    run_single_inter "$PAIR" >> "$RESULTS_DIR/inter_metrics.csv"
done

echo "" >&2
echo "=== All inter-population experiments complete ===" >&2
cat "$RESULTS_DIR/inter_metrics.csv"
