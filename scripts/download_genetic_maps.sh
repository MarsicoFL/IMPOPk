#!/usr/bin/env bash
set -euo pipefail

# Download plink-format genetic maps for all 22 autosomes (GRCh38)
#
# Source: Browning lab, University of Washington
# https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/

DATA_DIR="./data"
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download plink genetic maps (GRCh38) for all 22 autosomes (~10MB total)"
    echo "Skips chromosomes already present in the data directory."
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

MAP_DIR="${DATA_DIR}/genetic_maps"
BASE_URL="https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps"

echo "=== Genetic Maps Download (GRCh38, Browning lab) ==="
echo ""

downloaded=0
skipped=0

for chr in $(seq 1 22); do
    file="plink.chr${chr}.GRCh38.map"
    dest="${MAP_DIR}/${file}"
    url="${BASE_URL}/${file}"

    if [[ -f "$dest" ]]; then
        echo "Already exists: $dest"
        skipped=$((skipped + 1))
        continue
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would download: $file"
        echo "  URL:  $url"
        echo "  Dest: $dest"
        downloaded=$((downloaded + 1))
        continue
    fi

    mkdir -p "$MAP_DIR"
    echo "Downloading: $file"
    wget -c --progress=bar:force -O "$dest" "$url"
    echo "Done: $dest"
    downloaded=$((downloaded + 1))
done

echo ""
echo "=== Genetic maps download complete ==="
echo "  Downloaded: $downloaded"
echo "  Skipped (already present): $skipped"
