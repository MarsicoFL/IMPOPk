#!/usr/bin/env bash
set -euo pipefail

# Parallel Full Population Intra-Population IBS Analysis
# Uses ibs_parallel.sh for window-level parallelization (fastest)
# Runs experiments sequentially to avoid CPU oversubscription

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs_parallel.sh"
SAMPLE_DIR="$BASE_DIR/sample_lists"
RESULTS_DIR="$SCRIPT_DIR/../intra"

REGION="chr2:130787850-140837183"
WINDOW_SIZE=5000
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Leave 3 cores free for window parallelization
TOTAL_CORES=$(nproc)
MAX_JOBS=$((TOTAL_CORES - 3))
if [[ $MAX_JOBS -lt 1 ]]; then MAX_JOBS=1; fi

POPULATIONS=("AFR" "EUR" "EAS" "CSA" "AMR")

mkdir -p "$RESULTS_DIR"

run_single_intra() {
    local POP=$1
    local SAMPLE_FILE="$SAMPLE_DIR/HPRCv2_${POP}_full.txt"
    local OUTPUT_DIR="$RESULTS_DIR/FULL-${POP}-INTRA"
    local OUTPUT_FILE="$OUTPUT_DIR/FULL-${POP}-INTRA_ibs.tsv"

    mkdir -p "$OUTPUT_DIR"

    IND_COUNT=$(wc -l < "$SAMPLE_FILE")
    HAP_COUNT=$(( IND_COUNT * 2 ))
    PAIRS=$(( HAP_COUNT * (HAP_COUNT - 1) / 2 ))

    echo "[$(date +%H:%M:%S)] Starting $POP: $HAP_COUNT haplotypes, $PAIRS pairs" >&2

    START_TIME=$(date +%s.%N)

    "$IBS_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$SAMPLE_FILE" \
        --output "$OUTPUT_FILE" \
        -j "$MAX_JOBS"

    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    RECORDS=$(wc -l < "$OUTPUT_FILE")
    RECORDS=$((RECORDS - 1))

    echo "[$(date +%H:%M:%S)] Completed $POP: ${RUNTIME}s, ${RECORDS} records" >&2
    echo "${POP},${HAP_COUNT},${PAIRS},${RUNTIME},${RECORDS}"
}

echo "=== Running ${#POPULATIONS[@]} intra-population experiments ===" >&2
echo "Each experiment uses $MAX_JOBS parallel window jobs" >&2
echo "" >&2

echo "population,haplotypes,pairs,runtime_seconds,records" > "$RESULTS_DIR/intra_metrics.csv"

for POP in "${POPULATIONS[@]}"; do
    run_single_intra "$POP" >> "$RESULTS_DIR/intra_metrics.csv"
done

echo "" >&2
echo "=== All intra-population experiments complete ===" >&2
cat "$RESULTS_DIR/intra_metrics.csv"
