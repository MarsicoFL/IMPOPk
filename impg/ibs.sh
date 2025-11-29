#!/usr/bin/env bash
set -euo pipefail

# ibs: wrapper around `impg similarity` to obtain IBS segments.
#
# Pipeline:
#   1. Slide a window across a reference chromosome.
#   2. Run `impg similarity` in each window.
#   3. For each window, immediately:
#        - filter rows by estimated.identity >= cutoff
#        - drop self-self and ref-involving comparisons
#        - drop duplicated A–B / B–A (keep canonical order)
#        - reduce to: chrom, start, end, group.a, group.b, estimated.identity
#        - append to output (streaming)
#
# IBS is defined using the `estimated.identity` column
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

# Truncate output at start
: > "$OUTPUT"

# 1) Loop over windows and call impg similarity, streaming into OUTPUT
start_pos="$REG_START"
add_header=1   # whether IBS header (chrom, start, end, group.a, group.b, estimated.identity) should be printed

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
  awk -v cutoff="$CUTOFF" -v add_header="$add_header" -v ref="$REF_NAME" '
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
      if (add_header == 1) {
        print "chrom","start","end","group.a","group.b","estimated.identity"
      }
      next
    }
    {
      # Apply identity threshold
      if ($est+0 < cutoff) next

      # Skip self-self comparisons
      if ($c_ga == $c_gb) next

      # Skip any comparison involving the reference (e.g. CHM13#0#...)
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next

      # Keep only one direction per pair (canonical lexicographic order)
      if ($c_ga > $c_gb) next

      # Keep only "real" haplotype-haplotype comparisons
      print $c_chrom, $c_start, $c_end, $c_ga, $c_gb, $est
    }
  ' >> "$OUTPUT"

  add_header=0
  start_pos=$(( end_pos + 1 ))
done

echo "IBS written to: $OUTPUT"
