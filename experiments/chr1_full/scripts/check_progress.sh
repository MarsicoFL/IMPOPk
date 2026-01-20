#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# check_progress.sh - Check current progress of all chunks
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$EXP_DIR/data"
PARTIAL_DIR="$DATA_DIR/partial_runs"

# Chunk boundaries
declare -A CHUNK_START CHUNK_END
CHUNK_START[000]=1
CHUNK_END[000]=31120001
CHUNK_START[001]=31120001
CHUNK_END[001]=62240001
CHUNK_START[002]=62240001
CHUNK_END[002]=93360001
CHUNK_START[003]=93360001
CHUNK_END[003]=124480001
CHUNK_START[004]=124480001
CHUNK_END[004]=155600001
CHUNK_START[005]=155600001
CHUNK_END[005]=186720001
CHUNK_START[006]=186720001
CHUNK_END[006]=217840001
CHUNK_START[007]=217840001
CHUNK_END[007]=248960001

check_pop() {
    local POP="$1"
    local RUN_DIR="$PARTIAL_DIR/${POP}_run1"

    echo "=== $POP ==="

    if [[ ! -d "$RUN_DIR" ]]; then
        echo "  No data yet"
        return
    fi

    local complete=0
    local total=8

    for i in 000 001 002 003 004 005 006 007; do
        f="$RUN_DIR/out_$i.tsv"
        start="${CHUNK_START[$i]}"
        end="${CHUNK_END[$i]}"
        range=$((end - start))

        if [[ -s "$f" ]]; then
            max=$(cut -f3 "$f" 2>/dev/null | sort -n | tail -1)
            lines=$(wc -l < "$f")
            progress=$(( (max - start) * 100 / range ))

            if [[ "$progress" -ge 99 ]]; then
                status="COMPLETE"
                ((complete++))
            elif [[ "$progress" -ge 80 ]]; then
                status=">80%"
            else
                status=""
            fi

            printf "  chunk_%s: %3d%% (%d lines) %s\n" "$i" "$progress" "$lines" "$status"
        else
            printf "  chunk_%s:   0%% (empty)\n" "$i"
        fi
    done

    echo "  ---"
    echo "  Complete: $complete/$total"
    echo ""
}

echo ""
echo "=== Chunk Progress Report ==="
echo "Generated: $(date)"
echo ""

check_pop "EUR"
check_pop "AFR"

# Check for running processes
echo "=== Active Processes ==="
running=$(ps aux | grep "impg similarity" | grep -v grep | wc -l)
if [[ "$running" -gt 0 ]]; then
    echo "  $running impg process(es) running"
    ps aux | grep "impg similarity" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12}' | head -5
else
    echo "  No impg processes running"
fi
echo ""
