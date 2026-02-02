#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Haplotype Relatedness Analysis Script
# =============================================================================
# This script demonstrates how to use the ancestry HMM to determine
# which reference haplotype each segment of a query individual is most
# similar to. This is useful for relatedness/pedigree analysis.
#
# Usage: ./run_impg_ped.sh [WORKDIR]
#
# Default example:
#   Query: HG00344 (both haplotypes)
#   References: HG00099 and HG00097 (both haplotypes each = 4 reference haplotypes)
#   Region: chr1:50,000,001-60,000,000 (10 Mb)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Working directory (can be overridden by first argument)
WORKDIR="${1:-$PROJECT_ROOT/tutorial_relatedness}"

# Configuration
JOBS=8
WINDOW_SIZE=5000

# Region of interest
CHROM="chr1"
START=50000001
END=60000000
REGION="${CHROM}:${START}-${END}"
REGION_LEN=$((END - START + 1))

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

log "=============================================="
log "Haplotype Relatedness Analysis"
log "=============================================="
log "Region: $REGION ($(echo "scale=1; $REGION_LEN / 1000000" | bc) Mb)"
log "Window size: $WINDOW_SIZE bp"
log "Parallel jobs: $JOBS"
log "Working directory: $WORKDIR"
log ""

# Create working directory structure
mkdir -p "$WORKDIR"/{samples,results}

# Data files
AGC="$PROJECT_ROOT/data/assemblies/HPRC_r2_assemblies_0.6.1.agc"
PAF="$PROJECT_ROOT/data/alignments/hprc465vschm13.aln.paf.gz"

# Check data files exist
if [[ ! -e "$AGC" ]]; then
    echo "ERROR: AGC file not found: $AGC"
    echo "Download from: https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/"
    exit 1
fi

if [[ ! -e "$PAF" ]]; then
    echo "ERROR: PAF file not found: $PAF"
    echo "Download from: https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz"
    exit 1
fi

# Sample files
SAMPLES_QUERY="$WORKDIR/samples/query.txt"
SAMPLES_REF="$WORKDIR/samples/references.txt"
SAMPLES_ALL="$WORKDIR/samples/all.txt"

# Create sample files if they don't exist
if [[ ! -f "$SAMPLES_QUERY" ]]; then
    log "Creating default sample files..."
    cat > "$SAMPLES_QUERY" << 'EOF'
HG00344#1
HG00344#2
EOF
    cat > "$SAMPLES_REF" << 'EOF'
HG00099#1
HG00099#2
HG00097#1
HG00097#2
EOF
    cat "$SAMPLES_QUERY" "$SAMPLES_REF" > "$SAMPLES_ALL"
    log "  Query: HG00344 (2 haplotypes)"
    log "  References: HG00099, HG00097 (4 haplotypes)"
fi

# Output directory
OUT_DIR="$WORKDIR/results"

# Output files
SIM_FILE="$OUT_DIR/similarities.tsv"
QUERY_VS_REF_FILE="$OUT_DIR/query_vs_ref.tsv"
ANCESTRY_FILE="$OUT_DIR/relatedness.tsv"
POSTERIORS_FILE="$OUT_DIR/posteriors.tsv"

# Build ancestry binary if needed
ANCESTRY_BIN="$PROJECT_ROOT/target/release/ancestry"
if [[ ! -x "$ANCESTRY_BIN" ]]; then
    log "Building ancestry binary..."
    cargo build --release --bin ancestry --manifest-path "$PROJECT_ROOT/Cargo.toml"
fi

# =============================================================================
# STEP 1: Generate pairwise similarities using impg
# =============================================================================
log ""
log "=============================================="
log "STEP 1: Generate pairwise similarities"
log "=============================================="

if [[ -f "$SIM_FILE" ]] && [[ $(wc -l < "$SIM_FILE") -gt 100 ]]; then
    log "Using existing similarities file ($(wc -l < "$SIM_FILE") lines)"
else
    log "Running impg similarity in parallel..."

    TMPDIR=$(mktemp -d)
    log "Temp dir: $TMPDIR"

    # Generate window list
    WINDOWS_FILE="$TMPDIR/windows.txt"
    pos=$START
    idx=0
    while [[ $pos -le $END ]]; do
        win_end=$((pos + WINDOW_SIZE - 1))
        [[ $win_end -gt $END ]] && win_end=$END
        win_len=$((win_end - pos + 1))
        [[ $win_len -lt $((WINDOW_SIZE / 2)) ]] && break
        echo "$idx $pos $win_end"
        pos=$((win_end + 1))
        idx=$((idx + 1))
    done > "$WINDOWS_FILE"

    TOTAL_WINDOWS=$(wc -l < "$WINDOWS_FILE")
    log "Total windows: $TOTAL_WINDOWS"

    # Header
    echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity" > "$SIM_FILE"

    # Create processing script
    cat > "$TMPDIR/process.sh" << SCRIPT
#!/bin/bash
idx=\$1
start=\$2
end=\$3

region="${CHROM}:\${start}-\${end}"
outfile="$TMPDIR/w_\${idx}.tsv"

impg similarity \\
    --sequence-files "$AGC" \\
    -a "$PAF" \\
    -r "\$region" \\
    --subset-sequence-list "$SAMPLES_ALL" \\
    --force-large-region \\
    -t 1 \\
    -v 0 2>/dev/null | tail -n +2 > "\$outfile"

if (( idx % 500 == 0 )); then
    echo "  Window \$idx / $TOTAL_WINDOWS" >&2
fi
SCRIPT
    chmod +x "$TMPDIR/process.sh"

    START_TIME=$(date +%s)

    cat "$WINDOWS_FILE" | parallel -j "$JOBS" --colsep ' ' "$TMPDIR/process.sh" {1} {2} {3}

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    log "Similarity computation completed in ${ELAPSED}s"

    # Combine results
    log "Combining window results..."
    for f in "$TMPDIR"/w_*.tsv; do
        [[ -s "$f" ]] && cat "$f" >> "$SIM_FILE"
    done

    rm -rf "$TMPDIR"

    FINAL_LINES=$(wc -l < "$SIM_FILE")
    log "Total lines: $FINAL_LINES"
fi

log "Similarities file: $SIM_FILE ($(du -h "$SIM_FILE" | cut -f1))"

# =============================================================================
# STEP 2: Extract query vs reference similarities
# =============================================================================
log ""
log "=============================================="
log "STEP 2: Extract query vs reference matrix"
log "=============================================="

python3 "$SCRIPT_DIR/extract_query_vs_ref_similarities.py" \
    "$SIM_FILE" \
    -o "$QUERY_VS_REF_FILE" \
    --queries "$SAMPLES_QUERY" \
    --references "$SAMPLES_REF"

log "Query vs ref file: $QUERY_VS_REF_FILE ($(wc -l < "$QUERY_VS_REF_FILE") lines)"

# =============================================================================
# STEP 3: Run ancestry/relatedness HMM
# =============================================================================
log ""
log "=============================================="
log "STEP 3: Run HMM inference"
log "=============================================="

$ANCESTRY_BIN \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "chm13#chr1" \
    --region "${CHROM#chr}:${START}-${END}" \
    --region-length "$REGION_LEN" \
    --window-size "$WINDOW_SIZE" \
    --query-samples "$SAMPLES_QUERY" \
    -o "$ANCESTRY_FILE" \
    --similarity-file "$SIM_FILE" \
    --estimate-params \
    --smooth-min-windows 3 \
    --min-posterior 0.7 \
    --posteriors-output "$POSTERIORS_FILE" \
    -t "$JOBS"

log "Relatedness file: $ANCESTRY_FILE ($(wc -l < "$ANCESTRY_FILE") lines)"
log "Posteriors file: $POSTERIORS_FILE ($(wc -l < "$POSTERIORS_FILE") lines)"

# =============================================================================
# STEP 4: Generate plots
# =============================================================================
log ""
log "=============================================="
log "STEP 4: Generate plots"
log "=============================================="

python3 "$SCRIPT_DIR/plot_relatedness.py" \
    "$ANCESTRY_FILE" \
    -o "$OUT_DIR/relatedness" \
    --title "Haplotype Relatedness: Query vs References (${CHROM}:${START}-${END})"

log ""
log "=============================================="
log "ANALYSIS COMPLETE!"
log "=============================================="
log ""
log "Output files:"
log "  Similarities:  $SIM_FILE"
log "  Query vs Ref:  $QUERY_VS_REF_FILE"
log "  Relatedness:   $ANCESTRY_FILE"
log "  Posteriors:    $POSTERIORS_FILE"
log "  Plots:         $OUT_DIR/relatedness_*.png"
log ""
