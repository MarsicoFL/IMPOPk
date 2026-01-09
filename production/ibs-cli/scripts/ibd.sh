#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ibd.sh – glue `impg similarity` and a lightweight HMM for IBD calling.
#
# This bash pipeline predates the Rust CLI but remains useful for experimentation
# and serves as the truth-table for the parity tests. It keeps the per-window
# streaming behavior and delegates the segment calling to a compact HMM written
# in R. The documentation here mirrors the README so that researchers can stay
# in the shell without referencing external docs.
# -----------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: ibd [options]

Required:
  --sequence-files FILE         Sequence file(s) for impg (e.g. .agc)
  -a ALIGN_PAF                  Alignment file (.paf/.paf.gz/.1aln) [passed to impg as -p]
  -r REF_NAME                   Reference name (e.g. CHM13)
  -region REGION                Region, e.g. chr1:1-248956422 or chr1
  -size WINDOW_SIZE             Window size in bp
  --subset-sequence-list FILE   Haplotypes to compare (e.g. ibs_example.txt)
  --output FILE                 Output file for IBD segments

Optional:
  --region-length LEN           Total length of REGION if you use -region chr1
  --ibs-output FILE             Where to write per-window identity table
                                (default: OUTPUT.ibs_windows.tsv)
  --min-len-bp N                Minimum IBD segment length (bp) to keep (default: 0)
  --expected-seg-windows N      Expected IBD segment length in windows for HMM (default: 50)

Example:
  ./ibd.sh \\
    --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \\
    -a ../data/human/hprc465vschm13.aln.paf.gz \\
    -r CHM13 \\
    -region chr20:1-64000000 \\
    -size 5000 \\
    --subset-sequence-list ibs_example.txt \\
    --output ibd_chr20_5kb.tsv

EOF
}

# Defaults
SEQ_FILES=""
ALIGN=""
REF_NAME=""
REGION=""
WINDOW_SIZE=""
SUBSET_LIST=""
OUTPUT=""
REGION_LEN=""
IBS_OUTPUT=""
MIN_LEN_BP=0
EXPECTED_SEG_WINDOWS=50

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

# --- CLI argument parsing ----------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence-files)
      SEQ_FILES="$2"; shift 2;;
    -a)
      ALIGN="$2"; shift 2;;
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
    --ibs-output)
      IBS_OUTPUT="$2"; shift 2;;
    --min-len-bp)
      MIN_LEN_BP="$2"; shift 2;;
    --expected-seg-windows)
      EXPECTED_SEG_WINDOWS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

# --- Input validation -------------------------------------------------------
# Ensure the user provided the minimum information required to launch impg.
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

if ! command -v Rscript >/dev/null 2>&1; then
  echo "ERROR: 'Rscript' is not in PATH" >&2
  exit 1
fi

# Validate output directory exists and is writable
OUTPUT_DIR=$(dirname "$OUTPUT")
if [[ ! -d "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR" || { echo "ERROR: cannot create output directory: $OUTPUT_DIR" >&2; exit 1; }
fi
if [[ ! -w "$OUTPUT_DIR" ]]; then
  echo "ERROR: output directory is not writable: $OUTPUT_DIR" >&2
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

# Resolve IBS output name
if [[ -z "$IBS_OUTPUT" ]]; then
  IBS_OUTPUT="${OUTPUT}.ibs_windows.tsv"
fi

echo "Region: ${REG_CHROM}:${REG_START}-${REG_END}" >&2
echo "Window size: ${WINDOW_SIZE} bp" >&2
echo "Per-window IBS output: ${IBS_OUTPUT}" >&2
echo "IBD segments output: ${OUTPUT}" >&2
echo "Min IBD segment length: ${MIN_LEN_BP} bp" >&2
echo "Expected IBD length (windows): ${EXPECTED_SEG_WINDOWS}" >&2

# Truncate outputs
: > "$IBS_OUTPUT"
: > "$OUTPUT"

# 1) Loop over windows and call impg similarity, streaming into IBS_OUTPUT
start_pos="$REG_START"
add_header=1   # whether header should be printed

while [[ "$start_pos" -le "$REG_END" ]]; do
  end_pos=$(( start_pos + WINDOW_SIZE - 1 ))
  if [[ "$end_pos" -gt "$REG_END" ]]; then
    end_pos="$REG_END"
  fi

  REF_REGION="${REF_NAME}#0#${REG_CHROM}:${start_pos}-${end_pos}"

  echo "Processing window ${REF_REGION}" >&2

  if ! impg similarity \
    --sequence-files "$SEQ_FILES" \
    -p "$ALIGN" \
    -r "$REF_REGION" \
    --subset-sequence-list "$SUBSET_LIST" \
    --force-large-region | \
  awk -v add_header="$add_header" -v ref="$REF_NAME" '
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
      # NO identity threshold here: keep all windows

      # Skip self-self comparisons
      if ($c_ga == $c_gb) next

      # Skip any comparison involving the reference (e.g. CHM13#0#...)
      if (index($c_ga, ref "#") == 1) next
      if (index($c_gb, ref "#") == 1) next

      # Keep only one direction per pair (canonical lexicographic order)
      if ($c_ga > $c_gb) next

      # Output per-window identity
      print $c_chrom, $c_start, $c_end, $c_ga, $c_gb, $est
    }
  ' >> "$IBS_OUTPUT"; then
    echo "ERROR: impg/awk pipeline failed for window ${REF_REGION}" >&2
    exit 1
  fi

  add_header=0
  start_pos=$(( end_pos + 1 ))
done

echo "Per-window IBS written to: $IBS_OUTPUT" >&2

# 2) Run HMM in R to obtain IBD segments
export IBS_INPUT="$IBS_OUTPUT"
export IBD_OUTPUT="$OUTPUT"
export MIN_LEN_BP
export EXPECTED_SEG_WINDOWS

echo "Running HMM-based IBD calling..." >&2

Rscript - <<'RSCRIPT'
input <- Sys.getenv("IBS_INPUT")
output <- Sys.getenv("IBD_OUTPUT")
min_len_bp <- as.numeric(Sys.getenv("MIN_LEN_BP"))
expected_seg_windows <- as.numeric(Sys.getenv("EXPECTED_SEG_WINDOWS"))

if (is.na(min_len_bp)) min_len_bp <- 0
if (is.na(expected_seg_windows) || expected_seg_windows <= 1) expected_seg_windows <- 50

suppressWarnings({
  df <- try(read.table(input, header = TRUE, sep = "\t",
                       stringsAsFactors = FALSE, check.names = FALSE),
            silent = TRUE)
})

if (inherits(df, "try-error") || NROW(df) == 0) {
  # Empty or error -> write empty output
  seg_df <- data.frame(
    chrom = character(0),
    start = integer(0),
    end = integer(0),
    group.a = character(0),
    group.b = character(0),
    n_windows = integer(0),
    mean_identity = numeric(0),
    stringsAsFactors = FALSE
  )
  write.table(seg_df, file = output, quote = FALSE, sep = "\t", row.names = FALSE)
  quit(save = "no", status = 0)
}

# Basic type coercions
df$start <- as.numeric(df$start)
df$end <- as.numeric(df$end)
df$estimated.identity <- as.numeric(df$estimated.identity)

df <- df[!is.na(df$start) & !is.na(df$end) & !is.na(df$estimated.identity), ]

# Order by chrom, pair, start
ord <- order(df$chrom, df$group.a, df$group.b, df$start)
df <- df[ord, ]

# Split by (chrom, group.a, group.b)
key <- paste(df$chrom, df$group.a, df$group.b, sep = "||")
groups <- split(df, key)

seg_list <- list()

viterbi_gauss <- function(y, mu, sigma, pi_vec, A) {
  Tn <- length(y)
  S <- length(mu)
  logA <- log(A)
  logPi <- log(pi_vec)

  logB <- matrix(0, nrow = S, ncol = Tn)
  for (s in 1:S) {
    # simple Gaussian; y in [0,1]
    logB[s, ] <- dnorm(y, mean = mu[s], sd = sigma[s], log = TRUE)
  }

  delta <- matrix(NA_real_, nrow = S, ncol = Tn)
  psi <- matrix(NA_integer_, nrow = S, ncol = Tn)

  # t = 1
  delta[, 1] <- logPi + logB[, 1]
  psi[, 1] <- 0L

  if (Tn > 1) {
    for (t in 2:Tn) {
      for (s in 1:S) {
        prev <- delta[, t - 1] + logA[, s]
        j <- which.max(prev)
        delta[s, t] <- prev[j] + logB[s, t]
        psi[s, t] <- j
      }
    }
  }

  states <- integer(Tn)
  states[Tn] <- which.max(delta[, Tn])
  if (Tn > 1) {
    for (t in seq(Tn - 1, 1, by = -1)) {
      states[t] <- psi[states[t + 1], t + 1]
    }
  }
  states
}

for (gname in names(groups)) {
  g <- groups[[gname]]
  y <- g$estimated.identity
  Tn <- length(y)

  if (Tn < 3 || sd(y, na.rm = TRUE) < 1e-6) {
    # Muy poca info para segmentar
    next
  }

  # 1) Estimar dos clusters (bajo identity = no-IBD, alto = IBD)
  km <- try(kmeans(y, centers = c(0.2, 0.8), iter.max = 20), silent = TRUE)

  if (inherits(km, "try-error") || length(km$size) < 2 ||
      any(km$size == 0)) {
    # fallback muy básico
    mu1 <- quantile(y, 0.3, na.rm = TRUE)
    mu2 <- quantile(y, 0.9, na.rm = TRUE)
    sd_all <- sd(y, na.rm = TRUE)
    if (is.na(sd_all) || sd_all <= 0) sd_all <- 0.05
    mu <- c(mu1, mu2)
    sigma <- c(sd_all, sd_all)
  } else {
    # Ordenar clusters por media (1 = no-IBD, 2 = IBD)
    centers <- km$centers[, 1]
    ordc <- order(centers)
    mu <- centers[ordc]
    cluster <- match(km$cluster, ordc)

    sd1 <- sd(y[cluster == 1], na.rm = TRUE)
    sd2 <- sd(y[cluster == 2], na.rm = TRUE)
    if (is.na(sd1) || sd1 <= 0) sd1 <- 0.05
    if (is.na(sd2) || sd2 <= 0) sd2 <- 0.05
    sigma <- c(sd1, sd2)
  }

  # 2) Matriz de transición
  # Estado 1 = no-IBD, 2 = IBD
  p01 <- 1e-4  # no-IBD -> IBD
  p_stay_ibd <- 1 - 1 / expected_seg_windows
  if (p_stay_ibd < 0.5) p_stay_ibd <- 0.5
  if (p_stay_ibd > 0.9999) p_stay_ibd <- 0.9999
  p10 <- 1 - p_stay_ibd      # IBD -> no-IBD

  A <- matrix(c(1 - p01, p01,
                p10,      1 - p10),
              nrow = 2, byrow = TRUE)

  # 3) Distribución inicial (prior fuerte a no-IBD)
  pi_vec <- c(0.99, 0.01)

  # 4) Viterbi
  states <- viterbi_gauss(y, mu, sigma, pi_vec, A)

  # 5) Extraer segmentos donde state == 2 (IBD)
  r <- rle(states)
  ends <- cumsum(r$lengths)
  begins <- c(1, head(ends, -1) + 1)

  for (i in seq_along(r$values)) {
    if (r$values[i] != 2) next  # solo IBD

    idx_start <- begins[i]
    idx_end <- ends[i]

    seg_start_bp <- g$start[idx_start]
    seg_end_bp <- g$end[idx_end]
    seg_len_bp <- seg_end_bp - seg_start_bp + 1

    if (!is.finite(seg_len_bp) || seg_len_bp < min_len_bp) next

    seg_mean_id <- mean(y[idx_start:idx_end], na.rm = TRUE)

    seg_row <- data.frame(
      chrom = g$chrom[1],
      start = seg_start_bp,
      end = seg_end_bp,
      group.a = g$group.a[1],
      group.b = g$group.b[1],
      n_windows = r$lengths[i],
      mean_identity = seg_mean_id,
      stringsAsFactors = FALSE
    )
    seg_list[[length(seg_list) + 1L]] <- seg_row
  }
}

if (length(seg_list) == 0) {
  seg_df <- data.frame(
    chrom = character(0),
    start = integer(0),
    end = integer(0),
    group.a = character(0),
    group.b = character(0),
    n_windows = integer(0),
    mean_identity = numeric(0),
    stringsAsFactors = FALSE
  )
} else {
  seg_df <- do.call(rbind, seg_list)
  # Ordenar salida de forma razonable
  seg_df <- seg_df[order(seg_df$chrom, seg_df$group.a,
                         seg_df$group.b, seg_df$start), ]
}

write.table(seg_df, file = output, quote = FALSE, sep = "\t", row.names = FALSE)

RSCRIPT

echo "IBD segments written to: $OUTPUT"
