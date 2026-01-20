#!/usr/bin/env bash
set -euo pipefail

# Haplotype Scaling Benchmark
# Tests IBS runtime/output scaling with sample count
# Fixed: 5kb windows, LCT region (10 Mb)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_DIR="$BASE_DIR/data"
IBS_SCRIPT="$BASE_DIR/scripts/ibs_parallel.sh"
SAMPLE_DIR="$SCRIPT_DIR/../data"
RESULTS_DIR="$SCRIPT_DIR/../results"
METRICS_FILE="$RESULTS_DIR/benchmark_metrics.tsv"

REGION="chr2:130787850-140837183"
WINDOW_SIZE=5000
REF_NAME="CHM13"
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Parallel jobs (leave 3 cores free)
TOTAL_CORES=$(nproc)
JOBS=$((TOTAL_CORES - 3))
if [[ $JOBS -lt 1 ]]; then JOBS=1; fi

mkdir -p "$RESULTS_DIR"

# Header for metrics
echo -e "haplotypes\tpairs\truntime_seconds\toutput_records\toutput_bytes" > "$METRICS_FILE"

for HAP_COUNT in 2 10 50 100 150 200; do
    SAMPLE_FILE="$SAMPLE_DIR/random_$(printf '%03d' $HAP_COUNT)hap.txt"
    OUTPUT_FILE="$RESULTS_DIR/bench_hap${HAP_COUNT}_ibs.tsv"

    if [[ ! -f "$SAMPLE_FILE" ]]; then
        echo "ERROR: Sample file not found: $SAMPLE_FILE" >&2
        continue
    fi

    # New format: 1 line per individual, 2 haplotypes each
    IND_COUNT=$(wc -l < "$SAMPLE_FILE")
    HAP_ACTUAL=$(( IND_COUNT * 2 ))
    PAIRS=$(( HAP_ACTUAL * (HAP_ACTUAL - 1) / 2 ))

    echo "=== Running benchmark: $HAP_ACTUAL haplotypes ($IND_COUNT individuals, $PAIRS pairs) ===" >&2

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

    echo -e "${HAP_ACTUAL}\t${PAIRS}\t${RUNTIME}\t${RECORDS}\t${BYTES}" >> "$METRICS_FILE"

    echo "  Completed: ${RUNTIME}s, ${RECORDS} records, ${BYTES} bytes" >&2
done

echo ""
echo "Benchmark complete. Metrics saved to: $METRICS_FILE"
echo ""
cat "$METRICS_FILE"
