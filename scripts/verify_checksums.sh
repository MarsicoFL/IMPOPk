#!/usr/bin/env bash
set -euo pipefail

# Verify SHA256 checksums for downloaded data files
#
# Reads checksums.sha256, skips comment/placeholder lines,
# and verifies each file that exists.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="./data"

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Verify SHA256 checksums for all downloaded data files"
    echo ""
    echo "Options:"
    echo "  --data-dir DIR   Base data directory (default: ./data/)"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
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

CHECKSUM_FILE="${SCRIPT_DIR}/checksums.sha256"

if [[ ! -f "$CHECKSUM_FILE" ]]; then
    echo "Error: Checksum file not found: $CHECKSUM_FILE"
    exit 1
fi

echo "=== Checksum Verification ==="
echo "Checksum file: $CHECKSUM_FILE"
echo "Data directory: $DATA_DIR"
echo ""

total=0
verified=0
failed=0
missing=0
skipped=0

while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    # Skip PLACEHOLDER lines
    if [[ "$line" =~ ^PLACEHOLDER ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Parse: HASH  RELATIVE_PATH
    hash=$(echo "$line" | awk '{print $1}')
    relpath=$(echo "$line" | awk '{print $2}')

    # Handle paths relative to data dir or absolute
    if [[ "$relpath" == data/* ]]; then
        # Strip "data/" prefix and prepend DATA_DIR
        subpath="${relpath#data/}"
        filepath="${DATA_DIR}/${subpath}"
    else
        filepath="$relpath"
    fi

    total=$((total + 1))

    if [[ ! -f "$filepath" ]]; then
        echo "MISSING: $filepath"
        missing=$((missing + 1))
        continue
    fi

    echo -n "Checking: $filepath ... "
    actual=$(sha256sum "$filepath" | awk '{print $1}')

    if [[ "$actual" == "$hash" ]]; then
        echo "OK"
        verified=$((verified + 1))
    else
        echo "FAILED"
        echo "  Expected: $hash"
        echo "  Got:      $actual"
        failed=$((failed + 1))
    fi

done < "$CHECKSUM_FILE"

echo ""
echo "=== Summary ==="
echo "  Total entries:   $total"
echo "  Verified OK:     $verified"
echo "  Failed:          $failed"
echo "  Missing files:   $missing"
echo "  Placeholders:    $skipped (update checksums.sha256 after download)"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "ERROR: $failed file(s) failed checksum verification!"
    exit 1
fi

if [[ $missing -gt 0 ]]; then
    echo ""
    echo "NOTE: $missing file(s) not yet downloaded. Run download scripts first."
fi

exit 0
