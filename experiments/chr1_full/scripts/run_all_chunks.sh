#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_all_chunks.sh - Automatically complete all remaining chunks
# =============================================================================
#
# Monitors progress and launches chunks as slots become available.
# Runs max 2 processes in parallel to avoid resource issues.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_PARALLEL=2
PROGRESS_LOG="$SCRIPT_DIR/../data/progress.log"

# Chunk boundaries
declare -A CHUNK_END
CHUNK_END[000]=31120001
CHUNK_END[001]=62240001
CHUNK_END[002]=93360001
CHUNK_END[003]=124480001
CHUNK_END[004]=155600001
CHUNK_END[005]=186720001
CHUNK_END[006]=217840001
CHUNK_END[007]=248960001

declare -A CHUNK_START
CHUNK_START[000]=1
CHUNK_START[001]=31120001
CHUNK_START[002]=62240001
CHUNK_START[003]=93360001
CHUNK_START[004]=124480001
CHUNK_START[005]=155600001
CHUNK_START[006]=186720001
CHUNK_START[007]=217840001

get_chunk_progress() {
    local pop="$1"
    local chunk="$2"
    local out_file="$SCRIPT_DIR/../data/partial_runs/${pop}_run1/out_${chunk}.tsv"

    if [[ ! -s "$out_file" ]]; then
        echo "0"
        return
    fi

    local max_pos=$(cut -f3 "$out_file" 2>/dev/null | sort -n | tail -1)
    local start="${CHUNK_START[$chunk]}"
    local end="${CHUNK_END[$chunk]}"
    local range=$((end - start))

    if [[ -z "$max_pos" ]]; then
        echo "0"
        return
    fi

    local progress=$(( (max_pos - start) * 100 / range ))
    echo "$progress"
}

is_chunk_complete() {
    local pop="$1"
    local chunk="$2"
    local out_file="$SCRIPT_DIR/../data/partial_runs/${pop}_run1/out_${chunk}.tsv"
    local end="${CHUNK_END[$chunk]}"

    if [[ ! -s "$out_file" ]]; then
        return 1
    fi

    local max_pos=$(cut -f3 "$out_file" 2>/dev/null | sort -n | tail -1)
    # Complete if within 5kb of end
    if [[ "$max_pos" -ge "$((end - 5001))" ]]; then
        return 0
    fi
    return 1
}

get_running_count() {
    pgrep -f "continue_chunk.sh" 2>/dev/null | wc -l
}

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS_LOG"
}

run_chunk() {
    local pop="$1"
    local chunk="$2"
    log_progress "Starting $pop chunk $chunk"
    "$SCRIPT_DIR/continue_chunk.sh" "$pop" "$chunk" >> "$PROGRESS_LOG" 2>&1 &
}

show_status() {
    echo ""
    echo "=== Progress Report $(date '+%H:%M:%S') ==="

    for pop in EUR AFR; do
        echo ""
        echo "=== $pop ==="
        local complete=0
        for chunk in 000 001 002 003 004 005 006 007; do
            local pct=$(get_chunk_progress "$pop" "$chunk")
            local status=""
            if is_chunk_complete "$pop" "$chunk"; then
                status="COMPLETE"
                ((complete++))
            elif pgrep -f "continue_chunk.sh $pop $chunk" > /dev/null 2>&1; then
                status="RUNNING"
            else
                status=""
            fi
            printf "  chunk_%s: %3d%% %s\n" "$chunk" "$pct" "$status"
        done
        echo "  --- Complete: $complete/8"
    done
    echo ""
}

# Main loop
log_progress "=== Starting automated chunk completion ==="
log_progress "Max parallel: $MAX_PARALLEL"

# Priority queue: higher progress first, then EUR (smaller), then AFR
declare -a QUEUE=()

# Build queue sorted by progress (highest first)
for pop in EUR AFR; do
    for chunk in 000 001 002 003 004 005 006 007; do
        if ! is_chunk_complete "$pop" "$chunk"; then
            pct=$(get_chunk_progress "$pop" "$chunk")
            QUEUE+=("${pct}:${pop}:${chunk}")
        fi
    done
done

# Sort by progress descending
IFS=$'\n' SORTED=($(sort -t: -k1 -rn <<< "${QUEUE[*]}")); unset IFS

log_progress "Queue (${#SORTED[@]} chunks remaining):"
for item in "${SORTED[@]}"; do
    log_progress "  $item"
done

while true; do
    # Check how many are running
    running=$(get_running_count)

    # Find next chunk to run
    for item in "${SORTED[@]}"; do
        pct="${item%%:*}"
        rest="${item#*:}"
        pop="${rest%%:*}"
        chunk="${rest#*:}"

        # Skip if complete
        if is_chunk_complete "$pop" "$chunk"; then
            continue
        fi

        # Skip if already running
        if pgrep -f "continue_chunk.sh $pop $chunk" > /dev/null 2>&1; then
            continue
        fi

        # Start if we have slots
        if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
            run_chunk "$pop" "$chunk"
            ((running++))
        fi
    done

    # Show status
    show_status

    # Check if all complete
    all_done=true
    for pop in EUR AFR; do
        for chunk in 000 001 002 003 004 005 006 007; do
            if ! is_chunk_complete "$pop" "$chunk"; then
                all_done=false
                break 2
            fi
        done
    done

    if $all_done; then
        log_progress "=== ALL CHUNKS COMPLETE ==="
        break
    fi

    # Wait before next check
    sleep 60
done

log_progress "=== Done ==="
