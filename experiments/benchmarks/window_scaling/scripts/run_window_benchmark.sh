#!/usr/bin/env bash
set -euo pipefail

# Window Scaling Benchmark
# Tests IBS runtime/output scaling with window size
# Fixed: 8 haplotypes, LCT region (10 Mb)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs_parallel.sh"
SAMPLE_FILE="$SCRIPT_DIR/../data/fixed_008hap.txt"
RESULTS_DIR="$SCRIPT_DIR/../results"
METRICS_FILE="$RESULTS_DIR/benchmark_metrics.tsv"

REGION="chr2:130787850-140837183"
REGION_SIZE=$((140837183 - 130787850))  # ~10 Mb
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Parallel jobs (leave 3 cores free)
TOTAL_CORES=$(nproc)
JOBS=$((TOTAL_CORES - 3))
if [[ $JOBS -lt 1 ]]; then JOBS=1; fi

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$SAMPLE_FILE" ]]; then
    echo "ERROR: Sample file not found: $SAMPLE_FILE" >&2
    exit 1
fi

# New format: 1 line per individual, 2 haplotypes each
IND_COUNT=$(wc -l < "$SAMPLE_FILE")
HAP_COUNT=$(( IND_COUNT * 2 ))
PAIRS=$(( HAP_COUNT * (HAP_COUNT - 1) / 2 ))

# Header for metrics
echo -e "window_size\twindows\thaplotypes\tpairs\truntime_seconds\toutput_records\toutput_bytes" > "$METRICS_FILE"

for WINDOW_SIZE in 2000 5000 7000 10000; do
    WINDOWS=$(( REGION_SIZE / WINDOW_SIZE ))
    OUTPUT_FILE="$RESULTS_DIR/bench_win${WINDOW_SIZE}_ibs.tsv"

    echo "=== Running benchmark: ${WINDOW_SIZE} bp windows ($WINDOWS windows) ===" >&2

    START_TIME=$(date +%s.%N)

    "$IBS_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$SAMPLE_FILE" \
        --output "$OUTPUT_FILE" \
        -j "$JOBS"

    END_TIME=$(date +%s.%N)
    RUNTIME=$(echo "$END_TIME - $START_TIME" | bc)

    RECORDS=$(wc -l < "$OUTPUT_FILE")
    RECORDS=$((RECORDS - 1))  # Subtract header
    BYTES=$(stat --printf="%s" "$OUTPUT_FILE")

    echo -e "${WINDOW_SIZE}\t${WINDOWS}\t${HAP_COUNT}\t${PAIRS}\t${RUNTIME}\t${RECORDS}\t${BYTES}" >> "$METRICS_FILE"

    echo "  Completed: ${RUNTIME}s, ${RECORDS} records, ${BYTES} bytes" >&2
done

echo ""
echo "Benchmark complete. Metrics saved to: $METRICS_FILE"
echo ""
cat "$METRICS_FILE"
