#!/usr/bin/env bash
set -euo pipefail

# ibs: wrapper around `impg similarity` to obtain IBS segments.
#
# Pipeline:
#   1. Slide a window across a reference chromosome.
#   2. Run `impg similarity` in each window.
#   3. Concatenate results.
#   4. Keep only rows with estimated.identity >= cutoff.
#   5. Reduce to: chrom, start, end, group.a, group.b.
#   6. Optionally collapse contiguous segments per haplotype pair.
#
# IBS is currently defined using the column `estimated.identity`
# from `impg similarity` output.

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

Example:
  ./ibs.sh \\
    --sequence-files HPRC_r2_assemblies_0.6.1.agc \\
    -a hprc465vschm13.aln.paf.gz \\
    -c 1.0 \\
    -m cosin \\
    -r CHM13 \\
    -region chr1:1-248956422 \\
    -size 10000 \\
    --subset-sequence-list ibs_example.txt \\
    --colapse F \\
    --output ibs.out
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

# Parse REGION
#   chr1:1-248956422  -> use those bounds
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

# Temporary files
TMP_SIM="$(mktemp)"
TMP_FILTERED="$(mktemp)"
TMP_IBS_COLS="$(mktemp)"

trap 'rm -f "$TMP_SIM" "$TMP_FILTERED" "$TMP_IBS_COLS"' EXIT

# 1) Loop over windows and call impg similarity
start_pos="$REG_START"
first_window=1

while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE - 1 ))
  if [[ "$end_pos" -gt "$REG_END" ]]; then
    end_pos="$REG_END"
  fi

  # Example: CHM13#0#chr1:1-10000
  REF_REGION="${REF_NAME}#0#${REG_CHROM}:${start_pos}-${end_pos}"

  if [[ "$first_window" -eq 1 ]]; then
    # First window: keep header
    impg similarity \
      --sequence-files "$SEQ_FILES" \
      -p "$ALIGN" \
      -r "$REF_REGION" \
      --subset-sequence-list "$SUBSET_LIST" \
      --force-large-region
    first_window=0
  else
    # Subsequent windows: drop header
    impg similarity \
      --sequence-files "$SEQ_FILES" \
      -p "$ALIGN" \
      -r "$REF_REGION" \
      --subset-sequence-list "$SUBSET_LIST" \
      --force-large-region | tail -n +2
  fi

  start_pos=$(( end_pos + 1 ))
done > "$TMP_SIM"

# 2) Filter rows by estimated.identity >= CUTOFF
awk -v cutoff="$CUTOFF" '
BEGIN { FS=OFS="\t" }
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i == "estimated.identity") est=i
  }
  if (!est) {
    print "ERROR: column estimated.identity not found in header" > "/dev/stderr"
    exit 1
  }
  print  # header
  next
}
{
  if ($est+0 >= cutoff) print
}' "$TMP_SIM" > "$TMP_FILTERED"

# 3) Reduce to IBS-style columns: chrom, start, end, group.a, group.b
awk '
BEGIN { FS=OFS="\t" }
NR==1 {
  for (i=1; i<=NF; i++) {
    if ($i == "chrom")   c_chrom=i
    if ($i == "start")   c_start=i
    if ($i == "end")     c_end=i
    if ($i == "group.a") c_ga=i
    if ($i == "group.b") c_gb=i
  }
  if (!c_chrom || !c_start || !c_end || !c_ga || !c_gb) {
    print "ERROR: missing required columns (chrom/start/end/group.a/group.b)" > "/dev/stderr"
    exit 1
  }
  print "chrom","start","end","group.a","group.b"
  next
}
{
  print $c_chrom, $c_start, $c_end, $c_ga, $c_gb
}' "$TMP_FILTERED" > "$TMP_IBS_COLS"

# 4) Collapse contiguous segments if requested
if [[ "$COLLAPSE" == "T" || "$COLLAPSE" == "t" ]]; then
  sort -k1,1 -k4,4 -k5,5 -k2,2n "$TMP_IBS_COLS" | \
  awk '
  BEGIN { FS=OFS="\t" }
  NR==1 {
    # header
    print $0
    next
  }
  NR==2 {
    # first data row
    prev_chrom=$1; prev_start=$2; prev_end=$3; prev_ga=$4; prev_gb=$5
    next
  }
  {
    chrom=$1; s=$2; e=$3; ga=$4; gb=$5
    # Same chrom and pair, contiguous or overlapping segment
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
  }' > "$OUTPUT"
else
  cp "$TMP_IBS_COLS" "$OUTPUT"
fi

echo "IBS written to: $OUTPUT"
