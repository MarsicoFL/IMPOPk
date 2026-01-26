#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ibs-bed-full.sh - Full IBS output WITHOUT cutoff filtering
# =============================================================================
#
# This script outputs ALL pairwise identity values for proper IBD modeling.
# Unlike ibs-bed-test.sh which filters by cutoff, this preserves the complete
# distribution needed for:
#   - Proper emission parameter estimation
#   - Full non-IBD distribution characterization
#   - Rigorous HMM calibration
#
# Output format (same columns, but ALL pairs):
#   chrom	start	end	group.a	group.b	estimated.identity
#
# WARNING: Output files will be MUCH larger (~10-50x) than filtered version
# =============================================================================

usage() {
  cat <<EOF
Usage: ibs-bed-full.sh [options]

Full IBS output with parallel chunk processing - NO CUTOFF FILTERING.

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

Example:
  ./ibs-bed-full.sh \\
    --sequence-files ../data/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/hprc465vschm13.aln.paf.gz \\
    -r CHM13 -region chr2:1-50000000 -size 5000 \\
    --subset-sequence-list sample_list.txt \\
    --output ibs_chr2_50Mb_FULL.tsv -j 4
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
trap "rm -rf $TMPDIR" EXIT

OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Generate BED file
# -----------------------------------------------------------------------------
BED_FILE="$TMPDIR/windows.bed"
WINDOW_COUNT=0

echo "Generating BED file for ${REG_CHROM}:${REG_START}-${REG_END} (window=${WINDOW_SIZE})..." >&2

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
# Process chunks (NO FILTERING)
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

  # NO CUTOFF FILTERING - keep ALL pairs
  # Only apply:
  #   - Remove self-self comparisons
  #   - Remove reference pairs
  #   - Canonical order (A < B)
  #   - Select output columns
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

echo "Processing $JOBS chunks in parallel (NO CUTOFF)..." >&2

CHUNK_FILES=("$TMPDIR"/chunk_*)
pids=()
for i in "${!CHUNK_FILES[@]}"; do
  chunk_file="${CHUNK_FILES[$i]}"
  chunk_id=$(printf "%03d" "$i")
  process_chunk "$chunk_file" "$chunk_id" &
  pids+=($!)
done

echo "Waiting for chunks..." >&2
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "ERROR: chunk failed" >&2; exit 1; }
done

# -----------------------------------------------------------------------------
# Merge
# -----------------------------------------------------------------------------
echo "Merging outputs..." >&2
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity" > "$OUTPUT"
cat "$TMPDIR"/out_*.tsv | sort -k1,1 -k2,2n >> "$OUTPUT"

TOTAL=$(( $(wc -l < "$OUTPUT") - 1 ))
echo "Done. $TOTAL records written to: $OUTPUT" >&2
echo "WARNING: This file contains ALL pairs (no cutoff filtering)" >&2
