#!/bin/bash
set -euo pipefail

# Paths (adjust as needed)
PAF_FILE="../data/hprc465vschm13.aln.paf.gz"
SEQUENCE_FILES="../data/HPRC_r2_assemblies_0.6.1.agc"
REGION_PREFIX="CHM13#0#"
SUBSET_LIST=""
OUTPUT_FILE=""
VERBOSE=""

usage() {
  cat <<USAGE
Usage: $0 -b <bed> [-p paf] [-s agc] [-u subset.list] [-P prefix] [-o out.tsv] [-v]
Emit all pairwise identities per window as returned by 'impg similarity', plus
REGION/CHR/START/END/LENGTH:
  REGION CHR START END LENGTH group.a group.b estimated.identity
Notes:
- Mirrors the style of existing wrappers (region prefix, subset, etc.).
- Recommended window size for 'impg similarity': <=10kb.
USAGE
  exit 1
}

while getopts "b:p:s:u:P:o:vh" opt; do
  case $opt in
    b) BED_FILE="$OPTARG" ;;
    p) PAF_FILE="$OPTARG" ;;
    s) SEQUENCE_FILES="$OPTARG" ;;
    u) SUBSET_LIST="$OPTARG" ;;
    P) REGION_PREFIX="$OPTARG" ;;
    o) OUTPUT_FILE="$OPTARG" ;;
    v) VERBOSE=1 ;;
    h) usage ;;
    *) usage ;;
  esac
done

[ -z "${BED_FILE:-}" ] && { echo "Error: -b <bed> is required" >&2; usage; }
[ -f "$BED_FILE" ] || { echo "Error: BED '$BED_FILE' not found" >&2; exit 1; }
[ -f "$PAF_FILE" ] || { echo "Error: PAF '$PAF_FILE' not found" >&2; exit 1; }
[ -f "$SEQUENCE_FILES" ] || { echo "Error: AGC '$SEQUENCE_FILES' not found" >&2; exit 1; }
[ -z "${SUBSET_LIST}" ] || [ -f "$SUBSET_LIST" ] || { echo "Error: subset '$SUBSET_LIST' not found" >&2; exit 1; }

out="${OUTPUT_FILE:-/dev/stdout}"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

printed_header=""

while IFS=$'\t' read -r chr start end rest; do
  [[ -z "$chr" || "$chr" =~ ^# ]] && continue
  [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]] || { echo "Warning: non-integer coords $chr:$start-$end, skip" >&2; continue; }
  (( end > start )) || { echo "Warning: invalid interval $chr:$start-$end, skip" >&2; continue; }

  len=$(( end - start ))
  region="${REGION_PREFIX}${chr}:${start}-${end}"
  [[ -n "$VERBOSE" ]] && echo "[impg] $region" >&2

  cmd=(impg similarity -p "$PAF_FILE" -r "$region" --sequence-files "$SEQUENCE_FILES")
  [[ -n "$SUBSET_LIST" ]] && cmd+=(--subset-sequence-list "$SUBSET_LIST")

  if ! "${cmd[@]}" > "$tmp"; then
    echo "Error: 'impg similarity' failed at $region" >&2
    continue
  fi

  if [[ -z "$printed_header" ]]; then
    awk -v OFS="\t" -v region="$region" -v c="$chr" -v s="$start" -v e="$end" -v l="$len" '
      NR==1 { print "REGION","CHR","START","END","LENGTH",$0; next }
      { print region,c,s,e,l,$0 }
    ' "$tmp" > "$out"
    printed_header=1
  else
    awk -v OFS="\t" -v region="$region" -v c="$chr" -v s="$start" -v e="$end" -v l="$len" '
      NR==1 { next }
      { print region,c,s,e,l,$0 }
    ' "$tmp" >> "$out"
  fi
done < "$BED_FILE"
