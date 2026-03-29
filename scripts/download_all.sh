#!/usr/bin/env bash
set -euo pipefail

# Master download orchestrator for impopk
#
# Calls all individual download scripts in sequence, tracks elapsed time,
# and verifies checksums at the end.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="./data"
DRY_RUN=false
ARGS=()

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download all data required by impopk (~10-12GB total)"
    echo ""
    echo "  Component               Approx. size"
    echo "  ─────────────────────────────────────"
    echo "  HPRC AGC assemblies      3.1 GB"
    echo "  HPRC PAF alignments      5.3 GB"
    echo "  CHM13 v2.0 reference     0.9 GB"
    echo "  Genetic maps (22 chr)     10 MB"
    echo "  Platinum pedigree        ~1.0 GB (estimated)"
    echo "  Validation VCFs          ~2.0 GB (estimated)"
    echo "  ─────────────────────────────────────"
    echo "  Total                   ~12.3 GB"
    echo ""
    echo "Options:"
    echo "  --data-dir DIR   Base data directory (default: ./data/)"
    echo "  --dry-run        Show what would be downloaded without downloading"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            ARGS+=(--data-dir "$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            ARGS+=(--dry-run)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

START_TIME=$(date +%s)

echo "================================================================"
echo "  impopk - Download All Data"
echo "================================================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Estimated total download: ~12.3 GB"
if $DRY_RUN; then
    echo "Mode: DRY RUN (no files will be downloaded)"
fi
echo ""
echo "================================================================"

run_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Step ${step_num}/5: ${step_name}"
    echo "────────────────────────────────────────────────────────────"
    echo ""

    bash "${SCRIPT_DIR}/${script}" "${ARGS[@]+"${ARGS[@]}"}"
}

run_step 1 "HPRC assemblies and alignments" "download_hprc.sh"
run_step 2 "CHM13 T2T reference" "download_reference.sh"
run_step 3 "Genetic maps (22 autosomes)" "download_genetic_maps.sh"
run_step 4 "Platinum pedigree assemblies" "download_platinum.sh"
run_step 5 "Validation VCFs" "download_vcf.sh"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================"
echo "  Download complete"
echo "  Elapsed time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "================================================================"

# Verify checksums (unless dry-run)
if ! $DRY_RUN; then
    echo ""
    echo "Running checksum verification..."
    if [[ -f "${SCRIPT_DIR}/checksums.sha256" ]]; then
        bash "${SCRIPT_DIR}/verify_checksums.sh" --data-dir "$DATA_DIR" || {
            echo "WARNING: Some checksums did not match. See output above."
            exit 1
        }
    else
        echo "WARNING: checksums.sha256 not found, skipping verification"
    fi
fi

echo ""
echo "All done. You can now proceed with the tutorials."
