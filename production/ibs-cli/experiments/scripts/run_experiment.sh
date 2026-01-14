#!/bin/bash
# Run IBS experiment for a specific population or population pair
# Usage: ./run_experiment.sh <experiment_type> <pop1> [pop2] <region> <output_dir>

set -e

# Configuration - use project-local paths
PROJECT_ROOT="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
DATA_DIR="$PROJECT_ROOT/experiments/data"
AGC="$PROJECT_ROOT/data/HPRC_r2_assemblies_0.6.1.agc"
PAF="$PROJECT_ROOT/data/hprc465vschm13.aln.paf.gz"
WINDOW_SIZE=5000
CUTOFF=0.999

# Parse arguments
EXPERIMENT_TYPE=$1
POP1=$2
POP2=$3
REGION=$4
OUTPUT_DIR=$5

if [ "$EXPERIMENT_TYPE" == "intra" ]; then
    # Intra-population: use single population sample list
    SAMPLE_LIST="$DATA_DIR/${POP1}_4samples.txt"
    if [ ! -f "$SAMPLE_LIST" ]; then
        SAMPLE_LIST="$DATA_DIR/${POP1}_3samples.txt"  # For AMR
    fi
    OUTPUT_FILE="$OUTPUT_DIR/${POP1}_intra_ibs.tsv"
    echo "Running intra-population IBS for $POP1"
    echo "Sample list: $SAMPLE_LIST"
    echo "Region: $REGION"

elif [ "$EXPERIMENT_TYPE" == "inter" ]; then
    # Inter-population: combine two population sample lists
    SAMPLE_LIST1="$DATA_DIR/${POP1}_4samples.txt"
    SAMPLE_LIST2="$DATA_DIR/${POP2}_4samples.txt"
    if [ ! -f "$SAMPLE_LIST1" ]; then
        SAMPLE_LIST1="$DATA_DIR/${POP1}_3samples.txt"
    fi
    if [ ! -f "$SAMPLE_LIST2" ]; then
        SAMPLE_LIST2="$DATA_DIR/${POP2}_3samples.txt"
    fi

    # Create combined sample list
    COMBINED_LIST="$OUTPUT_DIR/${POP1}_${POP2}_combined.txt"
    cat "$SAMPLE_LIST1" "$SAMPLE_LIST2" > "$COMBINED_LIST"
    SAMPLE_LIST="$COMBINED_LIST"
    OUTPUT_FILE="$OUTPUT_DIR/${POP1}_vs_${POP2}_inter_ibs.tsv"
    echo "Running inter-population IBS: $POP1 vs $POP2"
    echo "Sample list: $SAMPLE_LIST ($(wc -l < "$SAMPLE_LIST") haplotypes)"
    echo "Region: $REGION"
else
    echo "Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Usage: $0 <intra|inter> <pop1> [pop2] <region> <output_dir>"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run IBS detection
echo "Starting at $(date)"
time bash "$SCRIPTS_DIR/ibs.sh" \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -c "$CUTOFF" \
    -r CHM13 \
    -region "$REGION" \
    -size "$WINDOW_SIZE" \
    --subset-sequence-list "$SAMPLE_LIST" \
    --output "$OUTPUT_FILE"

echo "Completed at $(date)"
echo "Output: $OUTPUT_FILE"
wc -l "$OUTPUT_FILE"
