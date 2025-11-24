#!/bin/bash

# IBS Detection Script using impg similarity
# Detects Identity-By-State segments by iterating impg across chromosome chunks

set -e
set -o pipefail

# Default values
SIMILARITY_CUTOFF=1.0
CHUNK_SIZE=100000
OUTPUT_FILE=""

usage() {
    cat << EOF
Usage: $0 -a <agc> -p <paf> -c <chr> -s <sequences> [options]

Required:
    -a FILE    AGC sequence file
    -p FILE    PAF alignment file
    -c STRING  Chromosome (e.g., chr20)
    -s FILE    Sequences file (one name per line)

Optional:
    -k INT     Chunk size (default: 100000)
    -t FLOAT   Identity cutoff (default: 1.0)
    -o FILE    Output file (default: stdout)
    -h         Show help

Example:
    $0 -a data.agc -p data.paf.gz -c chr20 -s sequences.txt -o output.txt
EOF
    exit 1
}

# Parse arguments
while getopts "a:p:c:s:k:t:o:h" opt; do
    case $opt in
        a) AGC_FILE="$OPTARG" ;;
        p) PAF_FILE="$OPTARG" ;;
        c) CHROM="$OPTARG" ;;
        s) SEQ_FILE="$OPTARG" ;;
        k) CHUNK_SIZE="$OPTARG" ;;
        t) SIMILARITY_CUTOFF="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check required arguments
if [[ -z "$AGC_FILE" ]] || [[ -z "$PAF_FILE" ]] || [[ -z "$CHROM" ]] || [[ -z "$SEQ_FILE" ]]; then
    echo "Error: Missing required arguments" >&2
    usage
fi

# Validate files
for f in "$AGC_FILE" "$PAF_FILE" "$SEQ_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: File not found: $f" >&2
        exit 1
    fi
done

# Check tools
if ! command -v impg &> /dev/null; then
    echo "Error: impg not found" >&2
    exit 1
fi

if ! command -v agc &> /dev/null; then
    echo "Error: agc not found" >&2
    exit 1
fi

# Get chromosome length
echo "Getting chromosome length..." >&2
CHROM_LENGTH=$(agc listset "$AGC_FILE" | grep "CHM13#0#$CHROM:" | sed 's/.*:\([0-9]*\)-\([0-9]*\)/\2/')

if [[ -z "$CHROM_LENGTH" ]]; then
    echo "Error: Could not find CHM13#0#$CHROM" >&2
    exit 1
fi

echo "Chromosome: CHM13#0#$CHROM (length: $CHROM_LENGTH bp)" >&2
echo "Chunk size: $CHUNK_SIZE bp, Cutoff: $SIMILARITY_CUTOFF" >&2
echo "" >&2

# Create temporary directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Step 1: Run impg similarity for each chunk
echo "Running impg similarity..." >&2

CHUNK_START=1
CHUNK_NUM=0

while [[ $CHUNK_START -le $CHROM_LENGTH ]]; do
    CHUNK_END=$((CHUNK_START + CHUNK_SIZE - 1))
    if [[ $CHUNK_END -gt $CHROM_LENGTH ]]; then
        CHUNK_END=$CHROM_LENGTH
    fi
    
    REGION="CHM13#0#$CHROM:$CHUNK_START-$CHUNK_END"
    
    impg similarity \
        --sequence-files "$AGC_FILE" \
        -p "$PAF_FILE" \
        -r "$REGION" \
        --subset-sequence-list "$SEQ_FILE" \
        --force-large-region \
        > "$TMPDIR/chunk_${CHUNK_NUM}.txt" 2>/dev/null || true
    
    CHUNK_START=$((CHUNK_END + 1))
    CHUNK_NUM=$((CHUNK_NUM + 1))
done

echo "Processed $CHUNK_NUM chunks" >&2

# Step 2: Filter and sort
echo "Filtering IBS segments..." >&2

cat "$TMPDIR"/chunk_*.txt 2>/dev/null | \
    awk -v cutoff="$SIMILARITY_CUTOFF" '
    BEGIN {OFS="\t"}
    /^chrom/ {next}
    NF > 0 {
        if ($12 >= cutoff && $4 != $5) {
            if ($4 < $5) {
                print $1, $2, $3, $4, $5
            } else {
                print $1, $2, $3, $5, $4
            }
        }
    }
    ' | sort -k4,4 -k5,5 -k1,1 -k2,2n > "$TMPDIR/filtered.txt"

echo "Found $(wc -l < "$TMPDIR/filtered.txt") segments" >&2

# Step 3: Merge consecutive segments
echo "Merging consecutive segments..." >&2

awk '
BEGIN {
    OFS="\t"
    print "chrom", "start", "end", "seq1", "seq2", "length"
}
{
    chrom = $1
    start = $2
    end = $3
    seq1 = $4
    seq2 = $5
    pair_key = seq1 SUBSEP seq2
    
    if (pair_key == prev_pair && chrom == prev_chrom && start == prev_end + 1) {
        prev_end = end
    } else {
        if (NR > 1) {
            print prev_chrom, prev_start, prev_end, prev_seq1, prev_seq2, prev_end - prev_start + 1
        }
        prev_chrom = chrom
        prev_start = start
        prev_end = end
        prev_seq1 = seq1
        prev_seq2 = seq2
        prev_pair = pair_key
    }
}
END {
    if (NR > 0) {
        print prev_chrom, prev_start, prev_end, prev_seq1, prev_seq2, prev_end - prev_start + 1
    }
}
' "$TMPDIR/filtered.txt" > "$TMPDIR/merged.txt"

# Output
if [[ -n "$OUTPUT_FILE" ]]; then
    cp "$TMPDIR/merged.txt" "$OUTPUT_FILE"
    echo "Output: $OUTPUT_FILE" >&2
else
    cat "$TMPDIR/merged.txt"
fi

# Summary
NUM_MERGED=$(tail -n +2 "$TMPDIR/merged.txt" | wc -l)
echo "Final: $NUM_MERGED IBS segments" >&2
