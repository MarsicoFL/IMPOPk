#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# merge_completed.sh - Merge all completed chunk outputs into final file
# =============================================================================
#
# Usage: ./merge_completed.sh <POP>
#   POP: EUR or AFR
#
# Checks that all chunks are complete before merging.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$EXP_DIR/data"
PARTIAL_DIR="$DATA_DIR/partial_runs"

# Chunk boundaries
declare -A CHUNK_END
CHUNK_END[000]=31120001
CHUNK_END[001]=62240001
CHUNK_END[002]=93360001
CHUNK_END[003]=124480001
CHUNK_END[004]=155600001
CHUNK_END[005]=186720001
CHUNK_END[006]=217840001
CHUNK_END[007]=248387328  # Actual CHM13 chr1 length

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <POP>"
    echo "  POP: EUR or AFR"
    exit 1
fi

POP="$1"

if [[ "$POP" == "EUR" ]]; then
    RUN_DIR="$PARTIAL_DIR/EUR_run1"
    OUTPUT="$DATA_DIR/EUR_chr1_full.tsv"
elif [[ "$POP" == "AFR" ]]; then
    RUN_DIR="$PARTIAL_DIR/AFR_run1"
    OUTPUT="$DATA_DIR/AFR_chr1_full.tsv"
else
    echo "ERROR: POP must be EUR or AFR" >&2
    exit 1
fi

echo "=== Checking chunk completion ===" >&2
echo "" >&2

ALL_COMPLETE=1
for i in 000 001 002 003 004 005 006 007; do
    f="$RUN_DIR/out_$i.tsv"
    if [[ -s "$f" ]]; then
        max_pos=$(cut -f3 "$f" | sort -n | tail -1)
        end_pos="${CHUNK_END[$i]}"
        # Allow 5kb tolerance
        if [[ "$max_pos" -ge "$((end_pos - 5001))" ]]; then
            echo "  chunk_$i: COMPLETE (max=$max_pos)" >&2
        else
            echo "  chunk_$i: INCOMPLETE (max=$max_pos, need=$end_pos)" >&2
            ALL_COMPLETE=0
        fi
    else
        echo "  chunk_$i: MISSING" >&2
        ALL_COMPLETE=0
    fi
done

echo "" >&2

if [[ "$ALL_COMPLETE" -eq 0 ]]; then
    echo "ERROR: Not all chunks are complete. Run continue_chunk.sh first." >&2
    exit 1
fi

echo "All chunks complete. Merging..." >&2

# Add header and merge all outputs, sorting by position
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity" > "$OUTPUT"
cat "$RUN_DIR"/out_*.tsv | sort -k1,1 -k2,2n >> "$OUTPUT"

TOTAL_LINES=$(($(wc -l < "$OUTPUT") - 1))
FILE_SIZE=$(du -h "$OUTPUT" | cut -f1)

echo "" >&2
echo "=== Merge Complete ===" >&2
echo "Output: $OUTPUT" >&2
echo "Records: $TOTAL_LINES" >&2
echo "Size: $FILE_SIZE" >&2
