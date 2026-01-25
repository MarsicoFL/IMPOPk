#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# pairwise-identity.sh - Full pairwise identity computation for IBD inference
# =============================================================================
#
# Computes ALL pairwise sequence identity values without any cutoff filtering.
# This is essential for proper IBD modeling because:
#   - HMM emission parameters require the full distribution
#   - Non-IBD distribution characterization needs all values
#   - Proper d' separability requires both tails
#
# Output format:
#   chrom	start	end	group.a	group.b	estimated.identity
#
# NOTE: Output files will be MUCH larger (~10-50x) than IBS-filtered versions.
#       For chr1 full (~249 Mb) with 60 haplotypes, expect ~50-100 GB output.
# =============================================================================

usage() {
  cat <<EOF
Usage: pairwise-identity.sh [options]

Compute full pairwise identity (no cutoff) for IBD inference.

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln)
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --output FILE                 Output file

Optional:
  -j JOBS                       Number of parallel chunks (default: 4)
  --region-length LEN           Total length of REGION if you use -region chr1
  --subset-sequence-list FILE   Haplotypes to compare

Example (chr1 full, EUR population):
  ./pairwise-identity.sh \\
    --sequence-files ../data/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/hprc465vschm13.aln.paf.gz \\
    -r CHM13 -region chr1:1-248956422 -size 5000 \\
    --subset-sequence-list ../sample_lists/HPRCv2_EUR_full.txt \\
    --output EUR_chr1_full.tsv -j 8
EOF
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
SEQ_FILES=""
ALIGN=""
REF_NAME=""
REGION=""
WINDOW_SIZE=""
SUBSET_LIST=""
OUTPUT=""
REGION_LEN=""
JOBS="4"

if [[ $# -eq 0 ]]; then usage; exit 1; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence-files) SEQ_FILES="$2"; shift 2;;
    -a) ALIGN="$2"; shift 2;;
    -r) REF_NAME="$2"; shift 2;;
    -region) REGION="$2"; shift 2;;
    -size) WINDOW_SIZE="$2"; shift 2;;
    --subset-sequence-list) SUBSET_LIST="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --region-length) REGION_LEN="$2"; shift 2;;
    -j) JOBS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 1;;
  esac
done

# Validate required parameters
for var in SEQ_FILES ALIGN REF_NAME REGION WINDOW_SIZE OUTPUT; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: missing required parameter: $var" >&2; usage; exit 1
  fi
done

if ! command -v impg >/dev/null 2>&1; then
  echo "ERROR: 'impg' is not in PATH" >&2; exit 1
fi

# -----------------------------------------------------------------------------
# Parse region
# -----------------------------------------------------------------------------
if [[ "$REGION" == *:* ]]; then
  REG_CHROM="${REGION%%:*}"
  rest="${REGION#*:}"
  REG_START="${rest%%-*}"
  REG_END="${rest##*-}"
else
  REG_CHROM="$REGION"
  if [[ -z "$REGION_LEN" ]]; then
    echo "ERROR: -region '$REGION' needs --region-length" >&2; exit 1
  fi
  REG_START=1
  REG_END="$REGION_LEN"
fi

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
TMPDIR=$(mktemp -d)
KEEP_TMP=0

cleanup() {
  if [[ $KEEP_TMP -eq 0 ]]; then
    rm -rf "$TMPDIR"
  else
    echo "Keeping temp dir for debugging: $TMPDIR" >&2
  fi
}
trap cleanup EXIT

OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Generate BED file
# -----------------------------------------------------------------------------
BED_FILE="$TMPDIR/windows.bed"
WINDOW_COUNT=0

echo "=== Pairwise Identity Computation ===" >&2
echo "Region: ${REG_CHROM}:${REG_START}-${REG_END}" >&2
echo "Window size: ${WINDOW_SIZE} bp" >&2
echo "Parallel jobs: ${JOBS}" >&2
echo "" >&2
echo "Generating BED file..." >&2

start_pos="$REG_START"
while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE ))
  if [[ "$end_pos" -gt $(( REG_END + 1 )) ]]; then
    end_pos=$(( REG_END + 1 ))
  fi
  if [[ "$start_pos" -ge "$end_pos" ]]; then
    break
  fi
  printf "%s#0#%s\t%d\t%d\n" "$REF_NAME" "$REG_CHROM" "$start_pos" "$end_pos" >> "$BED_FILE"
  start_pos=$(( start_pos + WINDOW_SIZE ))
  WINDOW_COUNT=$((WINDOW_COUNT + 1))
done

echo "Generated $WINDOW_COUNT windows" >&2

# -----------------------------------------------------------------------------
# Split into chunks
# -----------------------------------------------------------------------------
WINDOWS_PER_CHUNK=$(( (WINDOW_COUNT + JOBS - 1) / JOBS ))
echo "Splitting into $JOBS chunks (~$WINDOWS_PER_CHUNK windows each)..." >&2
split -l "$WINDOWS_PER_CHUNK" -d -a 3 "$BED_FILE" "$TMPDIR/chunk_"

# -----------------------------------------------------------------------------
# Process chunks (NO FILTERING - full distribution)
# -----------------------------------------------------------------------------
process_chunk() {
  local chunk_file="$1"
  local chunk_id="$2"
  local out_file="$TMPDIR/out_${chunk_id}.tsv"

  local impg_cmd="impg similarity"
  impg_cmd="$impg_cmd --sequence-files \"$SEQ_FILES\""
  impg_cmd="$impg_cmd -a \"$ALIGN\""
  impg_cmd="$impg_cmd --target-bed \"$chunk_file\""
  impg_cmd="$impg_cmd --force-large-region"

  if [[ -n "$SUBSET_LIST" ]]; then
    impg_cmd="$impg_cmd --subset-sequence-list \"$SUBSET_LIST\""
  fi

  # NO CUTOFF - keep ALL pairwise identity values
  # Only filter:
  #   - Self-self comparisons
  #   - Reference pairs
  #   - Canonical order (A < B to avoid duplicates)
  eval "$impg_cmd" 2>/dev/null | \
  awk -v ref="$REF_NAME" '
    BEGIN { FS=OFS="\t" }
    NR==1 {
      for (i=1; i<=NF; i++) {
        if ($i == "estimated.identity") est=i
        if ($i == "chrom") c_chrom=i
        if ($i == "start") c_start=i
        if ($i == "end") c_end=i
        if ($i == "group.a") c_ga=i
        if ($i == "group.b") c_gb=i
      }
      if (!est || !c_chrom || !c_start || !c_end || !c_ga || !c_gb) {
        print "ERROR: missing columns" > "/dev/stderr"
        exit 1
      }
      next
    }
    {
      # Remove self-self
      if ($c_ga == $c_gb) next

      # Remove reference pairs
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next

      # Canonical order
      if ($c_ga > $c_gb) next

      # Output: chrom, start, end-1, group.a, group.b, identity
      print $c_chrom, $c_start, $c_end-1, $c_ga, $c_gb, $est
    }
  ' > "$out_file"

  echo "Chunk $chunk_id: $(wc -l < "$out_file") records" >&2
}

export -f process_chunk
export SEQ_FILES ALIGN SUBSET_LIST REF_NAME TMPDIR

echo "" >&2
echo "Processing $JOBS chunks in parallel..." >&2

CHUNK_FILES=("$TMPDIR"/chunk_*)
pids=()
for i in "${!CHUNK_FILES[@]}"; do
  chunk_file="${CHUNK_FILES[$i]}"
  chunk_id=$(printf "%03d" "$i")
  process_chunk "$chunk_file" "$chunk_id" &
  pids+=($!)
done

echo "Waiting for chunks to complete..." >&2
FAILED=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    echo "ERROR: chunk $i (pid $pid) failed" >&2
    FAILED=1
  fi
done

if [[ $FAILED -eq 1 ]]; then
  KEEP_TMP=1
  echo "" >&2
  echo "Some chunks failed. Completed chunks saved in: $TMPDIR" >&2
  echo "Completed outputs:" >&2
  ls -lh "$TMPDIR"/out_*.tsv 2>/dev/null | grep -v " 0 " || echo "  (none)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
echo "" >&2
echo "Merging outputs..." >&2
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity" > "$OUTPUT"
cat "$TMPDIR"/out_*.tsv | sort -k1,1 -k2,2n >> "$OUTPUT"

TOTAL=$(( $(wc -l < "$OUTPUT") - 1 ))
FILE_SIZE=$(du -h "$OUTPUT" | cut -f1)

echo "" >&2
echo "=== Complete ===" >&2
echo "Records: $TOTAL" >&2
echo "File size: $FILE_SIZE" >&2
echo "Output: $OUTPUT" >&2
