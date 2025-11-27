#!/usr/bin/env bash
set -euo pipefail

# ibs: wrapper around `impg similarity` to obtain IBS segments.
#
# Pipeline:
#   1. Slide a window across a reference chromosome.
#   2. Run `impg similarity` in each window.
#   3. For each window, immediately:
#        - filter rows by estimated.identity >= cutoff
#        - reduce to: chrom, start, end, group.a, group.b
#        - append to output (streaming)
#   4. Optionally collapse contiguous segments at the end.
#
# Notes:
#   - IBS is defined via the `estimated.identity` column in the
#     impg similarity output.
#   - Streaming to the *final* output file happens when --colapse F.
#     If --colapse T, we accumulate in a temp file, then sort+collapse.

usage() {
  cat <<EOF
Usage: ibs [options]

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln) [passed to impg as -p]
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --subset-sequence-list FILE   Haplotypes to compare (e.g. ibs_example.txt)
  --output FILE                 Output file

Optional:
  -c CUTOFF                     Cutoff on estimated.identity (default: 1.0)
  -m METRIC                     Only informational for now (default: cosin)
  --colapse F|T                 Collapse contiguous IBS segments (default: F)
  --collapse F|T                Alias of --colapse
  --region-length LEN           Total length of REGION if you use -region chr1

Example (small region):
  ./ibs.sh \\
    --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/human/hprc465vschm13.aln.paf.gz \\
    -c 1.0 \\
    -m cosin \\
    -r CHM13 \\
    -region chr20:1-15000 \\
    -size 5000 \\
    --subset-sequence-list ibs_example.txt \\
    --colapse F \\
    --output ibs_chr20_15kb.out
EOF
}

# Parameters
SEQ_FILES=""
ALIGN=""
CUTOFF="1.0"
METRIC="cosin"
REF_NAME=""
REGION=""
WINDOW_SIZE=""
SUBSET_LIST=""
COLLAPSE="F"
OUTPUT=""
REGION_LEN=""

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence-files)
      SEQ_FILES="$2"; shift 2;;
    -a)
      ALIGN="$2"; shift 2;;
    -c)
      CUTOFF="$2"; shift 2;;
    -m)
      METRIC="$2"; shift 2;;
    -r)
      REF_NAME="$2"; shift 2;;
    -region)
      REGION="$2"; shift 2;;
    -size)
      WINDOW_SIZE="$2"; shift 2;;
    --subset-sequence-list)
      SUBSET_LIST="$2"; shift 2;;
    --colapse|--collapse)
      COLLAPSE="$2"; shift 2;;
    --output)
      OUTPUT="$2"; shift 2;;
    --region-length)
      REGION_LEN="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

# Required arguments
for var in SEQ_FILES ALIGN REF_NAME REGION WINDOW_SIZE SUBSET_LIST OUTPUT; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: missing required parameter: $var" >&2
    usage
    exit 1
  fi
done

if ! command -v impg >/dev/null 2>&1; then
  echo "ERROR: 'impg' is not in PATH" >&2
  exit 1
fi

# Parse REGION:
#   chr1:1-248956422  -> explicit bounds
#   chr1              -> requires --region-length
REG_CHROM=""
REG_START=""
REG_END=""

if [[ "$REGION" == *:* ]]; then
  REG_CHROM="${REGION%%:*}"
  rest="${REGION#*:}"
  REG_START="${rest%%-*}"
  REG_END="${rest##*-}"
else
  REG_CHROM="$REGION"
  if [[ -z "$REGION_LEN" ]]; then
    echo "ERROR: -region '$REGION' needs --region-length" >&2
    exit 1
  fi
  REG_START=1
  REG_END="$REGION_LEN"
fi

# Temp file used only if collapsing
TMP_IBS_COLS="$(mktemp)"
trap 'rm -f "$TMP_IBS_COLS"' EXIT

# Decide where to stream per-chunk IBS
STREAM_TARGET="$OUTPUT"
if [[ "$COLLAPSE" == "T" || "$COLLAPSE" == "t" ]]; then
  STREAM_TARGET="$TMP_IBS_COLS"
fi

# Truncate stream target at start
: > "$STREAM_TARGET"

# 1) Loop over windows and call impg similarity, streaming into STREAM_TARGET
start_pos="$REG_START"
add_header=1   # whether IBS header (chrom, start, end, group.a, group.b) should be printed

while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE - 1 ))
  if [[ "$end_pos" -gt "$REG_END" ]]; then
    end_pos="$REG_END"
  fi

  REF_REGION="${REF_NAME}#0#${REG_CHROM}:${start_pos}-${end_pos}"

  echo "Processing window ${REF_REGION}" >&2

  impg similarity \
    --sequence-files "$SEQ_FILES" \
    -p "$ALIGN" \
    -r "$REF_REGION" \
    --subset-sequence-list "$SUBSET_LIST" \
    --force-large-region | \
  awk -v cutoff="$CUTOFF" -v add_header="$add_header" '
    BEGIN { FS=OFS="\t" }
    NR==1 {
      # Identify columns on the header
      for (i=1; i<=NF; i++) {
        if ($i == "estimated.identity") est=i
        if ($i == "chrom")   c_chrom=i
        if ($i == "start")   c_start=i
        if ($i == "end")     c_end=i
        if ($i == "group.a") c_ga=i
        if ($i == "group.b") c_gb=i
      }
      if (!est || !c_chrom || !c_start || !c_end || !c_ga || !c_gb) {
        print "ERROR: missing required columns in similarity output" > "/dev/stderr"
        exit 1
      }
      if (add_header == "1") {
        print "chrom","start","end","group.a","group.b"
      }
      next
    }
    {
      if ($est+0 >= cutoff) {
        print $c_chrom, $c_start, $c_end, $c_ga, $c_gb
      }
    }
  ' >> "$STREAM_TARGET"

  add_header=0
  start_pos=$(( end_pos + 1 ))
done

# 2) Collapse contiguous segments if requested
if [[ "$COLLAPSE" == "T" || "$COLLAPSE" == "t" ]]; then
  # We have all IBS rows in TMP_IBS_COLS; sort+collapse into final OUTPUT.
  {
    read header
    echo "$header"
    sort -k1,1 -k4,4 -k5,5 -k2,2n
  } < "$TMP_IBS_COLS" | \
  awk '
    BEGIN { FS=OFS="\t" }
    NR==1 {
      print $0     # header
      next
    }
    NR==2 {
      prev_chrom=$1; prev_start=$2; prev_end=$3; prev_ga=$4; prev_gb=$5
      next
    }
    {
      chrom=$1; s=$2; e=$3; ga=$4; gb=$5
      if (chrom==prev_chrom && ga==prev_ga && gb==prev_gb && s <= prev_end+1) {
        if (e > prev_end) prev_end=e
      } else {
        print prev_chrom, prev_start, prev_end, prev_ga, prev_gb
        prev_chrom=chrom; prev_start=s; prev_end=e; prev_ga=ga; prev_gb=gb
      }
    }
    END {
      if (NR >= 2) {
        print prev_chrom, prev_start, prev_end, prev_ga, prev_gb
      }
    }
  ' > "$OUTPUT"
fi

echo "IBS written to: $OUTPUT"

