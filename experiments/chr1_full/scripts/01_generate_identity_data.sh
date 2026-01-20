#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_data_generation.sh - Generate full pairwise identity data for chr1
# =============================================================================
#
# Generates complete pairwise identity matrices for EUR and AFR populations
# across the entire chromosome 1.
#
# Expected runtime: ~10-20 hours (depending on hardware)
# Expected output: ~60 GB total
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IBD_CLI_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
IBS_CLI_DIR="$(cd "$IBD_CLI_DIR/../ibs-cli" && pwd)"

# Paths
IDENTITY_SCRIPT="$IBD_CLI_DIR/scripts/pairwise-identity.sh"
DATA_DIR="$IBS_CLI_DIR/data"
SAMPLE_DIR="$IBS_CLI_DIR/sample_lists"
OUTPUT_DIR="$EXP_DIR/data"

# Data files
SEQ_FILES="$DATA_DIR/HPRC_r2_assemblies_0.6.1.agc"
ALIGN_FILE="$DATA_DIR/hprc465vschm13.aln.paf.gz"

# Region parameters
REF_NAME="CHM13"
REGION="chr1:1-248956422"
WINDOW_SIZE=5000

# Parallelization (leave some cores free)
TOTAL_CORES=$(nproc)
JOBS=$((TOTAL_CORES - 4))
if [[ $JOBS -lt 1 ]]; then JOBS=1; fi

# Populations to process
POPULATIONS=("EUR" "AFR")

# =============================================================================
# Validation
# =============================================================================

echo "=== chr1_full Data Generation ===" >&2
echo "" >&2

# Check script exists
if [[ ! -x "$IDENTITY_SCRIPT" ]]; then
    echo "ERROR: pairwise-identity.sh not found at: $IDENTITY_SCRIPT" >&2
    exit 1
fi

# Check data files
if [[ ! -f "$SEQ_FILES" ]]; then
    echo "ERROR: Sequence files not found: $SEQ_FILES" >&2
    exit 1
fi

if [[ ! -f "$ALIGN_FILE" ]]; then
    echo "ERROR: Alignment file not found: $ALIGN_FILE" >&2
    exit 1
fi

# Check sample lists
for POP in "${POPULATIONS[@]}"; do
    SAMPLE_FILE="$SAMPLE_DIR/HPRCv2_${POP}_full.txt"
    if [[ ! -f "$SAMPLE_FILE" ]]; then
        echo "ERROR: Sample list not found: $SAMPLE_FILE" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "Configuration:" >&2
echo "  Region: $REGION" >&2
echo "  Window size: $WINDOW_SIZE bp" >&2
echo "  Parallel jobs: $JOBS" >&2
echo "  Populations: ${POPULATIONS[*]}" >&2
echo "  Output: $OUTPUT_DIR" >&2
echo "" >&2

# =============================================================================
# Process each population
# =============================================================================

for POP in "${POPULATIONS[@]}"; do
    SAMPLE_FILE="$SAMPLE_DIR/HPRCv2_${POP}_full.txt"
    OUTPUT_FILE="$OUTPUT_DIR/${POP}_chr1_full.tsv"

    # Count samples
    IND_COUNT=$(wc -l < "$SAMPLE_FILE")
    HAP_COUNT=$((IND_COUNT * 2))
    PAIRS=$((HAP_COUNT * (HAP_COUNT - 1) / 2))

    echo "=== Processing $POP ===" >&2
    echo "  Individuals: $IND_COUNT" >&2
    echo "  Haplotypes: $HAP_COUNT" >&2
    echo "  Pairwise comparisons: $PAIRS" >&2
    echo "" >&2

    # Check if already exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        EXISTING_LINES=$(wc -l < "$OUTPUT_FILE")
        echo "  WARNING: Output file exists with $EXISTING_LINES lines" >&2
        echo "  Skipping... (delete file to regenerate)" >&2
        echo "" >&2
        continue
    fi

    START_TIME=$(date +%s)

    "$IDENTITY_SCRIPT" \
        --sequence-files "$SEQ_FILES" \
        -a "$ALIGN_FILE" \
        -r "$REF_NAME" \
        -region "$REGION" \
        -size "$WINDOW_SIZE" \
        --subset-sequence-list "$SAMPLE_FILE" \
        --output "$OUTPUT_FILE" \
        -j "$JOBS"

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))
    RUNTIME_HOURS=$(echo "scale=2; $RUNTIME / 3600" | bc)

    # Get file size
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    RECORDS=$(($(wc -l < "$OUTPUT_FILE") - 1))

    echo "" >&2
    echo "  Completed $POP:" >&2
    echo "    Runtime: ${RUNTIME}s (${RUNTIME_HOURS} hours)" >&2
    echo "    Records: $RECORDS" >&2
    echo "    File size: $FILE_SIZE" >&2
    echo "" >&2
done

echo "=== All populations complete ===" >&2
echo "" >&2
ls -lh "$OUTPUT_DIR"/*.tsv 2>/dev/null || echo "No output files generated"
