#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: ibs [options]

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln)
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --output FILE                 Output file

Optional:
  -c CUTOFF                     Cutoff on estimated.identity (default: 1.0)
  -m METRIC                     Only informational for now (default: cosin)
  --region-length LEN           Total length of REGION if you use -region chr1
  --subset-sequence-list FILE   Haplotypes to compare (if not provided, compares all)

Example:
  ./ibs.sh \\
    --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/human/hprc465vschm13.aln.paf.gz \\
    -c 1.0 -m cosin -r CHM13 \\
    -region chr20:1-15000 -size 5000 \\
    --subset-sequence-list ibs_example.txt \\
    --output ibs_chr20_15kb.out
EOF
}

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

for var in SEQ_FILES ALIGN REF_NAME REGION WINDOW_SIZE OUTPUT; do
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

OUTPUT_DIR=$(dirname "$OUTPUT")
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR" || { echo "ERROR: cannot create output directory: $OUTPUT_DIR" >&2; exit 1; }
fi
if [[ ! -w "$OUTPUT_DIR" ]]; then
  echo "ERROR: output directory is not writable: $OUTPUT_DIR" >&2
  exit 1
fi

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

: > "$OUTPUT"

start_pos="$REG_START"
add_header=1

while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE - 1 ))
  if [[ "$end_pos" -gt "$REG_END" ]]; then
    end_pos="$REG_END"
  fi

  REF_REGION="${REF_NAME}#0#${REG_CHROM}:${start_pos}-${end_pos}"

  echo "Processing window ${REF_REGION}" >&2

  IMPG_CMD="impg similarity --sequence-files \"$SEQ_FILES\" -a \"$ALIGN\" -r \"$REF_REGION\" --force-large-region"

  if [[ -n "$SUBSET_LIST" ]]; then
    IMPG_CMD="$IMPG_CMD --subset-sequence-list \"$SUBSET_LIST\""
  fi

  if ! eval "$IMPG_CMD" | \
  awk -v cutoff="$CUTOFF" -v add_header="$add_header" -v ref="$REF_NAME" '
    BEGIN { FS=OFS="\t" }
    NR==1 {
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
      if ($est+0 < cutoff) next
      if ($c_ga == $c_gb) next
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next
      if ($c_ga > $c_gb) next
      print $c_chrom, $c_start, $c_end, $c_ga, $c_gb, $est
    }
  ' >> "$OUTPUT"; then
    echo "ERROR: impg/awk pipeline failed for window ${REF_REGION}" >&2
    exit 1
  fi

  add_header=0
  start_pos=$(( end_pos + 1 ))
done

echo "IBS written to: $OUTPUT"
