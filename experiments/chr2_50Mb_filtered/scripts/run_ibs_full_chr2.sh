#!/bin/bash
#
# Run IBS analysis on FULL chromosome 2
#
# This script runs the IBS tool on the complete chr2 (~243 Mb)
# for multiple populations to enable comprehensive IBD analysis.
#

set -e

# Paths
IBS_CLI="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/target/release/ibs"
AGC="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/data/HPRC_r2_assemblies_0.6.1.agc"
PAF="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/data/hprc465vschm13.aln.paf.gz"
SAMPLE_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/sample_lists"
OUTPUT_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibd-cli/experiments/validation/chr2_full_analysis/data"

# Parameters
REGION="chr2:1-243199373"  # Full chr2
WINDOW_SIZE=5000
CUTOFF=0.999  # High-identity windows for IBD detection
THREADS=8

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "IBS Analysis - Full Chromosome 2"
echo "============================================================"
echo "Region: $REGION"
echo "Window size: $WINDOW_SIZE bp"
echo "Cutoff: $CUTOFF"
echo "Threads: $THREADS"
echo ""

# Function to run IBS for a population
run_ibs_population() {
    local POP=$1
    local SAMPLE_LIST=$2
    local OUTPUT="${OUTPUT_DIR}/${POP}_chr2_full_ibs.tsv"

    echo "------------------------------------------------------------"
    echo "Processing: $POP"
    echo "Sample list: $SAMPLE_LIST"
    echo "Output: $OUTPUT"
    echo "Started: $(date)"
    echo "------------------------------------------------------------"

    if [ -f "$OUTPUT" ]; then
        echo "Output already exists, skipping..."
        return
    fi

    time "$IBS_CLI" \
        --sequence-files "$AGC" \
        -a "$PAF" \
        -r CHM13 \
        --region "$REGION" \
        --size "$WINDOW_SIZE" \
        --subset-sequence-list "$SAMPLE_LIST" \
        --output "$OUTPUT" \
        -c "$CUTOFF" \
        -t "$THREADS" \
        --region-length 243199373

    echo "Completed: $(date)"
    echo ""
}

# Run for each population
# Using subset lists for manageable analysis

# EUR
run_ibs_population "EUR" "${SAMPLE_DIR}/HPRCv2_EURsubset.txt"

# AFR
run_ibs_population "AFR" "${SAMPLE_DIR}/HPRCv2_AFRsubset.txt"

# EAS
run_ibs_population "EAS" "${SAMPLE_DIR}/HPRCv2_EASsubset.txt"

echo "============================================================"
echo "All IBS analyses completed!"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.tsv
