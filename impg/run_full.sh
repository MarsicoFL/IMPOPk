#!/usr/bin/env bash
set -euo pipefail

AGC="../data/human/HPRC_r2_assemblies_0.6.1.agc"
PAF="../data/human/hprc465vschm13.aln.paf.gz"
SUB="ibs_example.txt"

REF="CHM13"
CHR="chr20"
START=1
END=60000000
SIZE=5000
JOBS=10

CHUNK=$(( (END-START+1) / JOBS ))  # 6000000

# Generar lista de regiones (start end)
> regions.txt
for i in $(seq 0 $((JOBS-1))); do
  s=$((START + i*CHUNK))
  e=$((s + CHUNK - 1))
  echo -e "${s}\t${e}" >> regions.txt
done

# Correr en paralelo
parallel -j "$JOBS" --colsep '\t' \
  './ibs.sh --sequence-files '"$AGC"' -a '"$PAF"' -c 0.99999 -m cosin -r '"$REF"' \
     -region '"$CHR"':{1}-{2} -size '"$SIZE"' --subset-sequence-list '"$SUB"' \
     --output ibs_part_{#}.out' \
  :::: regions.txt

# Merge final
cat ibs_part_*.out | sort -k1,1 -k2,2n -k3,3n > ibs_for_ibd.out

