#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --ibs IBS.tsv \\
       --hap-a1 A1 --hap-a2 A2 --hap-b1 B1 --hap-b2 B2

IBS.tsv is a TAB-separated file with columns:
  1: chrom
  2: start
  3: end
  4: group.a
  5: group.b
  (remaining columns ignored)

Haplotypes must be in the SAME logical space as the IBS:
  e.g. A1 = "HG01167#1", A2 = "HG01167#2",
       B1 = "NA19682#1", B2 = "NA19682#2"

Each line represents an IBS-positive pair (IBS = 1) in that window.
Windows inside [min_start, max_end] that never show IBS between the four
haplotypes are counted as Delta9.

Output:
  Delta1..Delta9 = counts / total_windows (classified windows)
EOF
}

IBS_FILE=""
HAP_A1=""
HAP_A2=""
HAP_B1=""
HAP_B2=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ibs)    IBS_FILE="$2"; shift 2 ;;
    --hap-a1) HAP_A1="$2";  shift 2 ;;
    --hap-a2) HAP_A2="$2";  shift 2 ;;
    --hap-b1) HAP_B1="$2";  shift 2 ;;
    --hap-b2) HAP_B2="$2";  shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1 ;;
  esac
done

if [[ -z "$IBS_FILE" || -z "$HAP_A1" || -z "$HAP_A2" || -z "$HAP_B1" || -z "$HAP_B2" ]]; then
  echo "ERROR: missing required arguments." >&2
  usage
  exit 1
fi

sort -k1,1 -k2,2n -k3,3n "$IBS_FILE" | \
awk -v A1="$HAP_A1" -v A2="$HAP_A2" -v B1="$HAP_B1" -v B2="$HAP_B2" '
BEGIN {
  FS = "\t"; OFS = "\t";

  four[A1] = 1; four[A2] = 1;
  four[B1] = 1; four[B2] = 1;

  first_data = 1;
  min_start = -1;
  max_end   = -1;
  chrom0    = "";

  n_loci_ibsfour = 0;
  n_unclassified = 0;
  for (k = 1; k <= 9; k++) counts[k] = 0;
}

# Extract hap key: sample#hapIndex from "sample#hapIndex#contig:coords"
function hapkey(s,   a, n) {
  n = split(s, a, "#");
  if (n >= 2) {
    return a[1] "#" a[2];
  } else {
    return s;
  }
}

{
  if (NR == 1 && $1 == "chrom") next;  # skip header

  c = $1;
  s = $2 + 0;
  e = $3 + 0;

  if (first_data) {
    chrom0    = c;
    min_start = s;
    max_end   = e;
    win_size  = e - s + 1;
    first_data = 0;
  } else {
    if (s < min_start) min_start = s;
    if (e > max_end)   max_end   = e;
    # sanity: asumimos win_size constante
  }

  raw1 = $4;
  raw2 = $5;

  k1 = hapkey(raw1);
  k2 = hapkey(raw2);

  # si ninguno pertenece a los cuatro haps de interés, no sumamos edges
  if (!(k1 in four) && !(k2 in four)) next;

  locus = c ":" s "-" e;

  # sólo guardamos pares si AMBOS son de los cuatro haps
  if ((k1 in four) && (k2 in four) && k1 != k2) {
    if (k1 < k2) pair = k1 "|" k2;
    else         pair = k2 "|" k1;

    key = locus SUBSEP pair;
    if (!(key in ibs)) {
      ibs[key] = 1;
      if (!(locus in locus_seen)) {
        locus_seen[locus] = 1;
        locus_order[++n_loci_ibsfour] = locus;
      }
      if (locus_pairs[locus] == "") locus_pairs[locus] = pair;
      else                           locus_pairs[locus] = locus_pairs[locus] " " pair;
    }
  }
}

END {
  if (first_data) {
    print "ERROR: no data rows in IBS file." > "/dev/stderr";
    exit 1;
  }

  # Procesamos SOLO los locus que tienen IBS entre los cuatro haps
  for (i = 1; i <= n_loci_ibsfour; i++) {
    locus = locus_order[i];
    process_locus(locus);
  }

  # total de ventanas teórico en el rango [min_start, max_end]
  span = max_end - min_start + 1;
  if (span % win_size != 0) {
    print "WARNING: (max_end - min_start + 1) not divisible by win_size." > "/dev/stderr";
  }
  total_windows = int(span / win_size);

  # ventanas sin IBS entre los cuatro haps -> Delta9
  classified_from_ibsfour = 0;
  for (k = 1; k <= 9; k++) classified_from_ibsfour += counts[k];
  # ojo: classified_from_ibsfour == n_loci_ibsfour - n_unclassified, en la práctica

  missing = total_windows - n_loci_ibsfour;
  if (missing < 0) missing = 0;  # por seguridad

  counts[9] += missing;

  total = 0;
  for (k = 1; k <= 9; k++) total += counts[k];

  if (total == 0) {
    print "ERROR: no loci classified into Jacquard states." > "/dev/stderr";
    exit 1;
  }

  print "# chrom", chrom0, "min_start", min_start, "max_end", max_end, "win_size", win_size > "/dev/stderr";
  print "# total_windows", total_windows, "loci_with_IBS_fourhaps", n_loci_ibsfour, "missing_windows_as_Delta9", missing, "unclassified", n_unclassified > "/dev/stderr";

  for (k = 1; k <= 9; k++) {
    delta = counts[k] / total;
    printf("Delta%d\t%.8f\t(count=%d)\n", k, delta, counts[k]);
  }
}

function process_locus(locus,    parent, nodes, i, r, block, block_size, nb, blk, size, pairs, np, p, arr, key) {
  # union-find
  delete parent;
  parent[A1] = A1;
  parent[A2] = A2;
  parent[B1] = B1;
  parent[B2] = B2;

  pairs = locus_pairs[locus];
  if (pairs != "") {
    np = split(pairs, arr, " ");
    for (i = 1; i <= np; i++) {
      p = arr[i];
      if (p == "") continue;
      split(p, key, "|");
      unite(key[1], key[2], parent);
    }
  }

  # construir bloques
  delete block;
  delete block_size;
  nodes[1] = A1; nodes[2] = A2; nodes[3] = B1; nodes[4] = B2;

  for (i = 1; i <= 4; i++) {
    n = nodes[i];
    r = find(n, parent);
    if (block[r] == "") block[r] = n;
    else                block[r] = block[r] " " n;
    block_size[r]++;
  }

  nb = 0;
  for (r in block) {
    nb++;
    blk[nb]  = block[r];
    size[nb] = block_size[r];
  }

  state = classify_state(nb, blk, size);
  if (state == 0) n_unclassified++;
  else            counts[state]++;
}

function find(x, parent,    p) {
  p = parent[x];
  if (p == x) return x;
  parent[x] = find(p, parent);
  return parent[x];
}

function unite(x, y, parent,    rx, ry) {
  rx = find(x, parent);
  ry = find(y, parent);
  if (rx != ry) parent[ry] = rx;
}

# partición {A1,A2,B1,B2} -> Delta1..9
function classify_state(nb, blk, size,   i,j,arr,nTok,aCount,bCount, bA, bB, pairIndex, trip, condA,condB) {
  for (i = 1; i <= nb; i++) {
    aCount = 0; bCount = 0;
    nTok = split(blk[i], arr, " ");
    for (j = 1; j <= nTok; j++) {
      t = arr[j];
      if (t == "") continue;
      if      (t == A1 || t == A2) aCount++;
      else if (t == B1 || t == B2) bCount++;
    }
    bA[i] = aCount;
    bB[i] = bCount;
  }

  # Delta1: los 4 en un solo bloque
  if (nb == 1) {
    if (size[1] == 4) return 1;
    else return 0;
  }

  # Delta9: cuatro singletons (ojo: esto aquí solo pasaría si no hubo edges)
  if (nb == 4) {
    for (i = 1; i <= 4; i++) if (size[i] != 1) return 0;
    return 9;
  }

  # nb == 2: Delta2,3,5,7
  if (nb == 2) {
    i1 = 1; i2 = 2;

    if (size[i1] == 2 && size[i2] == 2) {
      condA = (bA[i1] == 2 && bB[i1] == 0 && bA[i2] == 0 && bB[i2] == 2);
      condB = (bA[i2] == 2 && bB[i2] == 0 && bA[i1] == 0 && bB[i1] == 2);
      if (condA || condB) return 2;

      if (bA[i1] == 1 && bB[i1] == 1 && bA[i2] == 1 && bB[i2] == 1) return 7;

      return 0;
    }

    if ((size[i1] == 1 && size[i2] == 3) || (size[i1] == 3 && size[i2] == 1)) {
      if (size[i1] == 3) trip = i1; else trip = i2;

      if (bA[trip] == 2 && bB[trip] == 1) return 3;
      if (bA[trip] == 1 && bB[trip] == 2) return 5;

      return 0;
    }

    return 0;
  }

  # nb == 3: Delta4,6,8
  if (nb == 3) {
    pairIndex = 0;
    for (i = 1; i <= 3; i++) {
      if (size[i] == 2) pairIndex = i;
      else if (size[i] != 1) return 0;
    }
    if (pairIndex == 0) return 0;

    if (bA[pairIndex] == 2 && bB[pairIndex] == 0) return 4;
    if (bA[pairIndex] == 0 && bB[pairIndex] == 2) return 6;
    if (bA[pairIndex] == 1 && bB[pairIndex] == 1) return 8;

    return 0;
  }

  return 0;
}
'

