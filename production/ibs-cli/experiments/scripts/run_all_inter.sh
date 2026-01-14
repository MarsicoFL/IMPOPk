#!/bin/bash
# Run all inter-population IBS experiments
# This script runs experiments sequentially to avoid overwhelming the system

SCRIPT_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/experiments/scripts"
RESULTS_BASE="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/experiments/results"
REGION="chr2:130787850-140837183"

# Define population pairs (10 pairs)
declare -a PAIRS=(
    "AFR:EUR"
    "AFR:EAS"
    "AFR:CSA"
    "AFR:AMR"
    "EUR:EAS"
    "EUR:CSA"
    "EUR:AMR"
    "EAS:CSA"
    "EAS:AMR"
    "CSA:AMR"
)

echo "Starting inter-population IBS experiments"
echo "Region: $REGION"
echo "Total pairs: ${#PAIRS[@]}"
echo "==========================================="

for pair in "${PAIRS[@]}"; do
    POP1=$(echo $pair | cut -d: -f1)
    POP2=$(echo $pair | cut -d: -f2)
    OUTPUT_DIR="$RESULTS_BASE/${POP1}_vs_${POP2}_inter"

    echo ""
    echo "Starting: $POP1 vs $POP2"
    echo "Output: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    time bash "$SCRIPT_DIR/run_experiment.sh" inter "$POP1" "$POP2" "$REGION" "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/log.txt"

    echo "Completed: $POP1 vs $POP2"
    echo "-------------------------------------------"
done

echo ""
echo "All inter-population experiments completed!"
echo "Results in: $RESULTS_BASE"
