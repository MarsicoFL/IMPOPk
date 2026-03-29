#!/usr/bin/env bash
set -euo pipefail

# Download CHM13 T2T v2.0 reference genome and associated files
#
# Sources:
#   - T2T consortium: https://github.com/marbl/CHM13
#   - S3: s3://human-pangenomics/T2T/CHM13/assemblies/analysis_set/

DATA_DIR="./data"
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download CHM13 T2T v2.0 reference genome (~900MB) and annotations"
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

REF_DIR="${DATA_DIR}/reference"

# --- CHM13 v2.0 reference genome ---
REF_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz"
REF_FILE="${REF_DIR}/chm13v2.0.fa.gz"
# TODO: Fill in checksum after first verified download (compute with: sha256sum chm13v2.0.fa.gz)
REF_SHA256=""
REF_SIZE="900MB"

# --- FAI index ---
REF_FAI_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz.fai"
REF_FAI_FILE="${REF_DIR}/chm13v2.0.fa.gz.fai"
REF_FAI_SIZE="2KB"

# --- GZI index ---
REF_GZI_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz.gzi"
REF_GZI_FILE="${REF_DIR}/chm13v2.0.fa.gz.gzi"
REF_GZI_SIZE="200KB"

# --- Segmental duplications ---
# Source: T2T consortium / UCSC Genome Browser
# The CHM13 v2.0 segmental duplication track
SD_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_SD.bed.gz"
SD_FILE="${REF_DIR}/chm13v2.0_SD.bed.gz"
SD_SIZE="5MB"

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"
    local size="$4"
    local sha256="${5:-}"

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
    wget -c --progress=bar:force -O "$dest" "$url"

    if [[ -n "$sha256" ]]; then
        echo "Verifying SHA256 checksum..."
        echo "$sha256  $dest" | sha256sum -c -
    else
        echo "NOTE: No checksum available for $dest — skipping verification"
    fi

    echo "Done: $dest"
}

# Decompress a .gz file, keeping the original compressed copy.
# Skips if the decompressed file already exists.
decompress_gz() {
    local gz_file="$1"
    local out_file="${gz_file%.gz}"

    if [[ -f "$out_file" ]]; then
        echo "Already decompressed: $out_file"
        return 0
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would decompress: $gz_file -> $out_file"
        return 0
    fi

    echo "Decompressing: $gz_file"
    gunzip -k "$gz_file"
    echo "Done: $out_file"
}

echo "=== CHM13 T2T v2.0 Reference Download ==="
echo ""

download_file "$REF_URL" "$REF_FILE" "CHM13 v2.0 reference genome" "$REF_SIZE" ""
echo ""
download_file "$REF_FAI_URL" "$REF_FAI_FILE" "CHM13 v2.0 FAI index" "$REF_FAI_SIZE" ""
echo ""
download_file "$REF_GZI_URL" "$REF_GZI_FILE" "CHM13 v2.0 GZI index" "$REF_GZI_SIZE" ""
echo ""
download_file "$SD_URL" "$SD_FILE" "CHM13 v2.0 segmental duplications" "$SD_SIZE" ""

# Decompress FASTA and BED so downstream tools can use the uncompressed files
echo ""
echo "--- Decompressing reference files ---"
decompress_gz "$REF_FILE"
decompress_gz "$SD_FILE"

echo ""
echo "=== Reference download complete ==="
