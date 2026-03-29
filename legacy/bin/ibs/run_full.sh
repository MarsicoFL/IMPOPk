#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLI_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${CLI_ROOT}/.." && pwd)
DATA_ROOT="${REPO_ROOT}/data/human"

AGC="${AGC:-${DATA_ROOT}/HPRC_r2_assemblies_0.6.1.agc}"
PAF="${PAF:-${DATA_ROOT}/hprc465vschm13.aln.paf.gz}"
SUB="${SUB:-${CLI_ROOT}/sample_lists/ibs_example.txt}"

REF="${REF:-CHM13}"
CHR="${CHR:-chr20}"
START=${START:-1}
END=${END:-60000000}
SIZE=${SIZE:-5000}
JOBS=${JOBS:-10}

if [[ ! -f "$AGC" ]]; then
  echo "ERROR: sequence file not found ($AGC). Override AGC=..." >&2
  exit 1
fi
if [[ ! -f "$PAF" ]]; then
  echo "ERROR: alignment file not found ($PAF). Override PAF=..." >&2
  exit 1
fi
if [[ ! -f "$SUB" ]]; then
  echo "ERROR: subset list not found ($SUB). Override SUB=..." >&2
  exit 1
fi

CHUNK=$(( (END-START+1) / JOBS ))
if (( CHUNK <= 0 )); then
  echo "ERROR: invalid region/JOBS combination" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
TMPDIR=$(mktemp -d "run_full.XXXXXX")
trap 'rm -rf "$TMPDIR"' EXIT
REGIONS_FILE="$TMPDIR/regions.tsv"
> "$REGIONS_FILE"
for i in $(seq 0 $((JOBS-1))); do
  s=$((START + i*CHUNK))
  e=$((s + CHUNK - 1))
  printf "%s\t%s\n" "$s" "$e" >> "$REGIONS_FILE"
done

CMD="./ibs.sh --sequence-files '$AGC' -a '$PAF' -c 0.99999 -m cosin -r '$REF' \
  -region '$CHR':{1}-{2} -size '$SIZE' --subset-sequence-list '$SUB' \
  --output $TMPDIR/ibs_part_{#}.out"

if ! parallel -j "$JOBS" --colsep '\t' "$CMD" :::: "$REGIONS_FILE"; then
  echo "ERROR: one or more parallel jobs failed" >&2
  exit 1
fi

for i in $(seq 1 "$JOBS"); do
  if [[ ! -f "$TMPDIR/ibs_part_${i}.out" ]]; then
    echo "ERROR: missing output file for job $i" >&2
    exit 1
  fi
done

cat "$TMPDIR"/ibs_part_*.out | sort -k1,1 -k2,2n -k3,3n > "$CLI_ROOT/ibs_for_ibd.out"
echo "Merged IBS windows -> $CLI_ROOT/ibs_for_ibd.out"
