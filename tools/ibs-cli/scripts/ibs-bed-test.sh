#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ibs-bed-test.sh - EXPERIMENTAL: BED-based IBS computation with parallelization
# =============================================================================
#
# Strategy:
#   1. Generate BED file from region and window size
#   2. Split BED into N chunks (based on -j cores)
#   3. Run impg similarity --target-bed in parallel on each chunk
#   4. Merge outputs and apply filtering
#
# This approach amortizes the index loading overhead by processing many windows
# per impg call, resulting in ~2x speedup per window compared to the loop approach.
#
# Output format is IDENTICAL to ibs.sh:
#   chrom	start	end	group.a	group.b	estimated.identity
# =============================================================================

usage() {
  cat <<EOF
Usage: ibs-bed-test.sh [options]

EXPERIMENTAL: BED-based IBS with parallel chunk processing.

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln)
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --output FILE                 Output file

Optional:
  -c CUTOFF                     Cutoff on estimated.identity (default: 0.99)
  -j JOBS                       Number of parallel chunks (default: 4)
  --region-length LEN           Total length of REGION if you use -region chr1
  --subset-sequence-list FILE   Haplotypes to compare

Example:
  ./ibs-bed-test.sh \\
    --sequence-files ../data/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/hprc465vschm13.aln.paf.gz \\
    -r CHM13 -region chr2:1-50000000 -size 5000 \\
    --subset-sequence-list sample_list.txt \\
    --output ibs_chr2_50Mb.tsv -j 4
EOF
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
SEQ_FILES=""
ALIGN=""
CUTOFF="0.99"
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
    -c) CUTOFF="$2"; shift 2;;
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
# Parse region into chromosome, start, end
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
# Create temp directory
# -----------------------------------------------------------------------------
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Step 1: Generate BED file with all windows
# -----------------------------------------------------------------------------
# Window logic matches ibs.sh exactly:
#   - ibs.sh uses: end_pos = start_pos + WINDOW_SIZE - 1
#   - We generate BED with end = start + WINDOW_SIZE (so impg reports end = start + WINDOW_SIZE)
#   - Then awk outputs end-1 to match ibs.sh format
# For the last window, if end exceeds region, cap at REG_END+1 so awk gives REG_END
BED_FILE="$TMPDIR/windows.bed"
WINDOW_COUNT=0

echo "Generating BED file for region ${REG_CHROM}:${REG_START}-${REG_END} with window size ${WINDOW_SIZE}..." >&2

start_pos="$REG_START"
while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE ))
  # Cap at REG_END + 1 so that after awk does end-1, we get REG_END
  if [[ "$end_pos" -gt $(( REG_END + 1 )) ]]; then
    end_pos=$(( REG_END + 1 ))
  fi
  # Skip if window would be empty
  if [[ "$start_pos" -ge "$end_pos" ]]; then
    break
  fi
  # BED format: chrom, start, end (impg will report these coordinates)
  # impg expects: REF_NAME#0#chrom format
  printf "%s#0#%s\t%d\t%d\n" "$REF_NAME" "$REG_CHROM" "$start_pos" "$end_pos" >> "$BED_FILE"
  # Next window starts where this one ends (non-overlapping)
  start_pos=$(( start_pos + WINDOW_SIZE ))
  WINDOW_COUNT=$((WINDOW_COUNT + 1))
done

echo "Generated $WINDOW_COUNT windows" >&2

# -----------------------------------------------------------------------------
# Step 2: Split BED file into N chunks for parallel processing
# -----------------------------------------------------------------------------
WINDOWS_PER_CHUNK=$(( (WINDOW_COUNT + JOBS - 1) / JOBS ))

echo "Splitting into $JOBS chunks (~$WINDOWS_PER_CHUNK windows each)..." >&2

split -l "$WINDOWS_PER_CHUNK" -d -a 3 "$BED_FILE" "$TMPDIR/chunk_"

# -----------------------------------------------------------------------------
# Step 3: Process each chunk in parallel
# -----------------------------------------------------------------------------
process_chunk() {
  local chunk_file="$1"
  local chunk_id="$2"
  local out_file="$TMPDIR/out_${chunk_id}.tsv"

  # Build impg command
  local impg_cmd="impg similarity"
  impg_cmd="$impg_cmd --sequence-files \"$SEQ_FILES\""
  impg_cmd="$impg_cmd -a \"$ALIGN\""
  impg_cmd="$impg_cmd --target-bed \"$chunk_file\""
  impg_cmd="$impg_cmd --force-large-region"

  if [[ -n "$SUBSET_LIST" ]]; then
    impg_cmd="$impg_cmd --subset-sequence-list \"$SUBSET_LIST\""
  fi

  # Run impg and filter output
  eval "$impg_cmd" 2>/dev/null | \
  awk -v cutoff="$CUTOFF" -v ref="$REF_NAME" '
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
      # Filter by identity cutoff
      if ($est+0 < cutoff) next

      # Remove self-self comparisons
      if ($c_ga == $c_gb) next

      # Remove reference pairs (starts with REF_NAME#)
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next

      # Canonical order: keep only A < B
      if ($c_ga > $c_gb) next

      # Output in ibs.sh format: chrom, start, end-1, group.a, group.b, identity
      # Note: end-1 to match ibs.sh format (end is inclusive)
      print $c_chrom, $c_start, $c_end-1, $c_ga, $c_gb, $est
    }
  ' > "$out_file"

  echo "Chunk $chunk_id done: $(wc -l < "$out_file") records" >&2
}

export -f process_chunk
export SEQ_FILES ALIGN SUBSET_LIST CUTOFF REF_NAME TMPDIR

echo "Processing $JOBS chunks in parallel..." >&2

# Get list of chunk files
CHUNK_FILES=("$TMPDIR"/chunk_*)
NUM_CHUNKS=${#CHUNK_FILES[@]}

# Process chunks in parallel using background processes
pids=()
for i in "${!CHUNK_FILES[@]}"; do
  chunk_file="${CHUNK_FILES[$i]}"
  chunk_id=$(printf "%03d" "$i")
  process_chunk "$chunk_file" "$chunk_id" &
  pids+=($!)
done

# Wait for all background processes
echo "Waiting for all chunks to complete..." >&2
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "ERROR: chunk process $pid failed" >&2; exit 1; }
done

# -----------------------------------------------------------------------------
# Step 4: Merge all outputs
# -----------------------------------------------------------------------------
echo "Merging outputs..." >&2

# Write header
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity" > "$OUTPUT"

# Concatenate all chunk outputs and sort by position
cat "$TMPDIR"/out_*.tsv | sort -k1,1 -k2,2n >> "$OUTPUT"

TOTAL_RECORDS=$(( $(wc -l < "$OUTPUT") - 1 ))
echo "Done. $TOTAL_RECORDS IBS records written to: $OUTPUT" >&2
