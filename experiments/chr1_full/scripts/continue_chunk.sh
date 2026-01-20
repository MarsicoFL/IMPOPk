#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# continue_chunk.sh - Continue processing a chunk from where it left off
# =============================================================================
#
# Usage: ./continue_chunk.sh <POP> <CHUNK_ID>
#   POP:      EUR or AFR
#   CHUNK_ID: 000-007
#
# Example:
#   ./continue_chunk.sh AFR 001
#
# This script:
#   1. Reads the existing partial output to find max position processed
#   2. Generates a BED file for the remaining windows
#   3. Runs impg similarity on just those windows
#   4. Appends results to the existing output file
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$EXP_DIR/data"
PARTIAL_DIR="$DATA_DIR/partial_runs"

# Paths to data (absolute paths)
# ibd-cli is 4 levels up from scripts: scripts -> chr1_full -> validation -> experiments -> ibd-cli
IBD_CLI_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SEQ_FILES="$IBD_CLI_DIR/../ibs-cli/data/HPRC_r2_assemblies_0.6.1.agc"
ALIGN="$IBD_CLI_DIR/../ibs-cli/data/hprc465vschm13.aln.paf.gz"
REF_NAME="CHM13"
WINDOW_SIZE=5000

# Chunk boundaries
declare -A CHUNK_START CHUNK_END
CHUNK_START[000]=1
CHUNK_END[000]=31120001
CHUNK_START[001]=31120001
CHUNK_END[001]=62240001
CHUNK_START[002]=62240001
CHUNK_END[002]=93360001
CHUNK_START[003]=93360001
CHUNK_END[003]=124480001
CHUNK_START[004]=124480001
CHUNK_END[004]=155600001
CHUNK_START[005]=155600001
CHUNK_END[005]=186720001
CHUNK_START[006]=186720001
CHUNK_END[006]=217840001
CHUNK_START[007]=217840001
CHUNK_END[007]=248387328  # Actual CHM13 chr1 length

usage() {
    echo "Usage: $0 <POP> <CHUNK_ID>"
    echo "  POP:      EUR or AFR"
    echo "  CHUNK_ID: 000-007"
    exit 1
}

if [[ $# -ne 2 ]]; then
    usage
fi

POP="$1"
CHUNK_ID="$2"

# Validate inputs
if [[ ! "$POP" =~ ^(EUR|AFR)$ ]]; then
    echo "ERROR: POP must be EUR or AFR" >&2
    exit 1
fi

if [[ ! "$CHUNK_ID" =~ ^00[0-7]$ ]]; then
    echo "ERROR: CHUNK_ID must be 000-007" >&2
    exit 1
fi

# Set population-specific paths
if [[ "$POP" == "EUR" ]]; then
    SUBSET_LIST="$IBD_CLI_DIR/../ibs-cli/sample_lists/HPRCv2_EUR_full.txt"
    RUN_DIR="$PARTIAL_DIR/EUR_run1"
else
    SUBSET_LIST="$IBD_CLI_DIR/../ibs-cli/sample_lists/HPRCv2_AFR_full.txt"
    RUN_DIR="$PARTIAL_DIR/AFR_run1"
fi

OUTPUT_FILE="$RUN_DIR/out_${CHUNK_ID}.tsv"
CHUNK_FILE="$RUN_DIR/chunk_${CHUNK_ID}"

# Get chunk boundaries
START_POS="${CHUNK_START[$CHUNK_ID]}"
END_POS="${CHUNK_END[$CHUNK_ID]}"

echo "=== Continue Chunk Processing ===" >&2
echo "Population: $POP" >&2
echo "Chunk: $CHUNK_ID ($START_POS - $END_POS)" >&2
echo "" >&2

# Find max position already processed
if [[ -s "$OUTPUT_FILE" ]]; then
    MAX_POS=$(cut -f3 "$OUTPUT_FILE" | sort -n | tail -1)
    echo "Existing output: $(wc -l < "$OUTPUT_FILE") lines" >&2
    echo "Max position processed: $MAX_POS" >&2
else
    MAX_POS=0
    echo "No existing output, starting from beginning" >&2
fi

# Calculate continuation start (next window after max)
if [[ "$MAX_POS" -eq 0 ]] || [[ "$MAX_POS" -lt "$START_POS" ]]; then
    CONTINUE_FROM="$START_POS"
else
    CONTINUE_FROM=$((MAX_POS + 1))
fi

if [[ "$CONTINUE_FROM" -ge "$END_POS" ]]; then
    echo "" >&2
    echo "Chunk $CHUNK_ID already COMPLETE!" >&2
    echo "Max position ($MAX_POS) >= end position ($END_POS)" >&2
    exit 0
fi

# Calculate remaining work
REMAINING_BP=$((END_POS - CONTINUE_FROM))
REMAINING_WINDOWS=$((REMAINING_BP / WINDOW_SIZE))
PROGRESS_PCT=$(( (MAX_POS - START_POS) * 100 / (END_POS - START_POS) ))

echo "" >&2
echo "Progress: $PROGRESS_PCT%" >&2
echo "Continuing from: $CONTINUE_FROM" >&2
echo "Remaining: ~$REMAINING_WINDOWS windows ($REMAINING_BP bp)" >&2
echo "" >&2

# Generate continuation BED file
CONT_BED=$(mktemp)
trap "rm -f $CONT_BED" EXIT

echo "Generating continuation BED..." >&2
pos="$CONTINUE_FROM"
count=0
while [[ "$pos" -lt "$END_POS" ]]; do
    end=$((pos + WINDOW_SIZE))
    if [[ "$end" -gt "$END_POS" ]]; then
        end="$END_POS"
    fi
    printf "%s#0#chr1\t%d\t%d\n" "$REF_NAME" "$pos" "$end" >> "$CONT_BED"
    pos=$((pos + WINDOW_SIZE))
    count=$((count + 1))
done
echo "Generated $count windows" >&2

# Run impg
echo "" >&2
echo "Running impg similarity..." >&2
echo "Output will be appended to: $OUTPUT_FILE" >&2
echo "" >&2

impg similarity \
    --sequence-files "$SEQ_FILES" \
    -a "$ALIGN" \
    --target-bed "$CONT_BED" \
    --force-large-region \
    --subset-sequence-list "$SUBSET_LIST" 2>/dev/null | \
awk -v ref="$REF_NAME" '
    BEGIN { FS=OFS="\t" }
    NR==1 {
        for (i=1; i<=NF; i++) {
            if ($i == "estimated.identity") est=i
            if ($i == "chrom") c_chrom=i
            if ($i == "start") c_start=i
            if ($i == "end") c_end=i
            if ($i == "group.a") c_ga=i
            if ($i == "group.b") c_gb=i
        }
        next
    }
    {
        if ($c_ga == $c_gb) next
        if (index($c_ga, ref "#") == 1) next
        if (index($c_gb, ref "#") == 1) next
        if ($c_ga > $c_gb) next
        print $c_chrom, $c_start, $c_end-1, $c_ga, $c_gb, $est
    }
' >> "$OUTPUT_FILE"

# Report results
NEW_LINES=$(wc -l < "$OUTPUT_FILE")
NEW_MAX=$(cut -f3 "$OUTPUT_FILE" | sort -n | tail -1)
NEW_PCT=$(( (NEW_MAX - START_POS) * 100 / (END_POS - START_POS) ))

echo "" >&2
echo "=== Chunk $CHUNK_ID Complete ===" >&2
echo "Total lines: $NEW_LINES" >&2
echo "Max position: $NEW_MAX" >&2
echo "Progress: $NEW_PCT%" >&2

if [[ "$NEW_MAX" -ge "$((END_POS - WINDOW_SIZE))" ]]; then
    echo "Status: COMPLETE!" >&2
else
    echo "Status: Still incomplete (run again)" >&2
fi
