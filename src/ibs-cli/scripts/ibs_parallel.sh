#!/usr/bin/env bash
set -euo pipefail

# Parallel IBS Script - processes windows in parallel using GNU parallel
# Significant speedup over sequential ibs.sh

usage() {
  cat <<EOF
Usage: ibs_parallel [options]

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln)
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --output FILE                 Output file

Optional:
  -c CUTOFF                     Cutoff on estimated.identity (default: 1.0)
  -j JOBS                       Number of parallel jobs (default: auto, leaves 3 cores free)
  --region-length LEN           Total length of REGION if you use -region chr1
  --subset-sequence-list FILE   Haplotypes to compare

Example:
  ./ibs_parallel.sh \\
    --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/human/hprc465vschm13.aln.paf.gz \\
    -r CHM13 -region chr20:1-15000 -size 5000 \\
    --subset-sequence-list ibs_example.txt \\
    --output ibs_chr20_15kb.out -j 10
EOF
}

SEQ_FILES=""
ALIGN=""
CUTOFF="1.0"
REF_NAME=""
REGION=""
WINDOW_SIZE=""
SUBSET_LIST=""
OUTPUT=""
REGION_LEN=""
JOBS=""

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

for var in SEQ_FILES ALIGN REF_NAME REGION WINDOW_SIZE OUTPUT; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: missing required parameter: $var" >&2; usage; exit 1
  fi
done

if ! command -v impg >/dev/null 2>&1; then
  echo "ERROR: 'impg' is not in PATH" >&2; exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: 'GNU parallel' is not in PATH" >&2; exit 1
fi

# Auto-detect job count
if [[ -z "$JOBS" ]]; then
  TOTAL_CORES=$(nproc)
  JOBS=$((TOTAL_CORES - 3))
  if [[ $JOBS -lt 1 ]]; then JOBS=1; fi
fi

OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Parse region
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

# Create temp directory for parallel output
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Generate window list
WINDOW_FILE="$TMPDIR/windows.txt"
start_pos="$REG_START"
window_idx=0
while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE - 1 ))
  if [[ "$end_pos" -gt "$REG_END" ]]; then end_pos="$REG_END"; fi
  printf "%d\t%d\t%d\n" "$window_idx" "$start_pos" "$end_pos" >> "$WINDOW_FILE"
  start_pos=$(( end_pos + 1 ))
  window_idx=$((window_idx + 1))
done

TOTAL_WINDOWS=$window_idx
echo "Processing $TOTAL_WINDOWS windows with $JOBS parallel jobs..." >&2

# Function to process single window
process_window() {
  local idx=$1
  local start=$2
  local end=$3
  local ref_region="${REF_NAME}#0#${REG_CHROM}:${start}-${end}"
  local out_file="$TMPDIR/window_${idx}.tsv"

  IMPG_CMD="impg similarity --sequence-files \"$SEQ_FILES\" -a \"$ALIGN\" -r \"$ref_region\" --force-large-region"
  if [[ -n "$SUBSET_LIST" ]]; then
    IMPG_CMD="$IMPG_CMD --subset-sequence-list \"$SUBSET_LIST\""
  fi

  eval "$IMPG_CMD" 2>/dev/null | \
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
      next
    }
    {
      if ($est+0 < cutoff) next
      if ($c_ga == $c_gb) next
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next
      if ($c_ga > $c_gb) next
      print $c_chrom, $c_start, $c_end, $c_ga, $c_gb, $est
    }
  ' > "$out_file"
}

export -f process_window
export SEQ_FILES ALIGN SUBSET_LIST CUTOFF REF_NAME REG_CHROM TMPDIR

# Process windows in parallel
cat "$WINDOW_FILE" | parallel -j "$JOBS" --colsep '\t' process_window {1} {2} {3}

# Combine results with header
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity" > "$OUTPUT"
cat "$TMPDIR"/window_*.tsv | sort -k2,2n >> "$OUTPUT"

TOTAL_RECORDS=$(wc -l < "$OUTPUT")
TOTAL_RECORDS=$((TOTAL_RECORDS - 1))
echo "Done. $TOTAL_RECORDS IBS records written to: $OUTPUT" >&2
