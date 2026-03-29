#!/usr/bin/env bash
set -euo pipefail

# Download HPRC Release 2 data: AGC assemblies and PAF alignments
#
# Sources:
#   - HPRCv2 repository: https://github.com/pangenome/HPRCv2
#   - impop repository: https://github.com/pangenome/impop
#   - Garrison Lab S3: https://garrisonlab.s3.amazonaws.com/hprcv2/
#   - HPRC data portal: https://humanpangenome.org/data/
#   - Zenodo: https://zenodo.org/records/11622477 (minigraph-cactus v2.1)

DATA_DIR="./data"
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download HPRC Release 2 AGC assemblies (~3.1GB) and PAF alignments (~5.3GB)"
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
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

ASSEMBLIES_DIR="${DATA_DIR}/assemblies"
ALIGNMENTS_DIR="${DATA_DIR}/alignments"

# --- AGC assemblies ---
# The AGC file bundles all HPRC Release 2 assemblies (465 haplotypes + CHM13).
# Source: HPRC S3 submission (as referenced in MarsicoFL/IMPOPk README)
# Browse: https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/
AGC_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc"
AGC_FILE="${ASSEMBLIES_DIR}/HPRC_r2_assemblies_0.6.1.agc"
AGC_SHA256="e9b417e4ea49a18009d522e33444f57b7c09773ff9ec0d5ecb942e282f5eba56"
AGC_SIZE="3.1GB"

# --- PAF alignments (WFMASH, vs CHM13) ---
# Primary source: Garrison Lab S3 (from https://github.com/pangenome/HPRCv2)
# These are the WFMASH all-vs-ref alignments used by impg and impop.
PAF_URL="https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz"
PAF_URL_FALLBACK="https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze2/minigraph-cactus/hprc465vschm13.aln.paf.gz"
PAF_FILE="${ALIGNMENTS_DIR}/hprc465vschm13.aln.paf.gz"
PAF_SHA256="cf48ae04e5cab5636dd926287413f555f3755acc14024d714f16be74cfae1b6d"
PAF_SIZE="5.3GB"

# --- PAF index (for impg) ---
PAF_GZI_URL="https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz.gzi"
PAF_GZI_FILE="${ALIGNMENTS_DIR}/hprc465vschm13.aln.paf.gz.gzi"
PAF_GZI_SIZE="200KB"

# --- impg index (pre-built, optional) ---
IMPG_IDX_URL="https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg"
IMPG_IDX_FILE="${ALIGNMENTS_DIR}/hprc465vschm13.aln.paf.gz.impg"
IMPG_IDX_SIZE="varies"

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"
    local size="$4"
    local sha256="$5"

    if [[ -f "$dest" ]]; then
        echo "Already exists: $dest"
        return 0
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would download: $desc (~$size)"
        echo "  URL:  $url"
        echo "  Dest: $dest"
        return 0
    fi

    local dest_dir
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"

    echo "Downloading: $desc (~$size)"
    echo "  URL:  $url"
    echo "  Dest: $dest"
    wget -c --progress=bar:force -O "${dest}.tmp" "$url"
    mv "${dest}.tmp" "$dest"

    if [[ -n "$sha256" ]]; then
        echo "Verifying SHA256 checksum..."
        echo "$sha256  $dest" | sha256sum -c -
    fi

    echo "Done: $dest"
}

echo "=== HPRC Release 2 Data Download ==="
echo "Sources: https://github.com/pangenome/HPRCv2"
echo "         https://github.com/pangenome/impop"
echo ""

download_file "$AGC_URL" "$AGC_FILE" "HPRC r2 AGC assemblies" "$AGC_SIZE" "$AGC_SHA256"
echo ""

# Try primary PAF URL (Garrison Lab), fall back to HPRC S3
if [[ -f "$PAF_FILE" ]]; then
    echo "Already exists: $PAF_FILE"
else
    echo "Downloading: WFMASH PAF alignments (~$PAF_SIZE)"
    echo "  Primary: $PAF_URL"
    echo "  Fallback: $PAF_URL_FALLBACK"
    if $DRY_RUN; then
        echo "[DRY RUN] Would download PAF alignments"
    else
        mkdir -p "$(dirname "$PAF_FILE")"
        if ! wget -c --progress=bar:force -O "${PAF_FILE}.tmp" "$PAF_URL" 2>/dev/null; then
            echo "Primary URL failed, trying fallback..."
            wget -c --progress=bar:force -O "${PAF_FILE}.tmp" "$PAF_URL_FALLBACK"
        fi
        mv "${PAF_FILE}.tmp" "$PAF_FILE"
        if [[ -n "$PAF_SHA256" ]]; then
            echo "Verifying SHA256 checksum..."
            echo "$PAF_SHA256  $PAF_FILE" | sha256sum -c -
        fi
        echo "Done: $PAF_FILE"
    fi
fi
echo ""

# Optional: PAF index for impg
download_file "$PAF_GZI_URL" "$PAF_GZI_FILE" "PAF GZI index (for impg)" "$PAF_GZI_SIZE" ""
echo ""

# Optional: pre-built impg index
download_file "$IMPG_IDX_URL" "$IMPG_IDX_FILE" "Pre-built impg index (optional)" "$IMPG_IDX_SIZE" ""

echo ""
echo "=== HPRC download complete ==="
