#!/bin/bash

# Default values
SEQUENCE_FILES=""
ALIGNMENT=""
CUTOFF=0.95
METRIC="cosine.similarity"
REFERENCE="CHM13"
REGION=""
SIZE=10000
SUBSET_LIST=""
COLLAPSE="F"
OUTPUT="output.ibs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequence-files)
            SEQUENCE_FILES="$2"
            shift 2
            ;;
        -a)
            ALIGNMENT="$2"
            shift 2
            ;;
        -c)
            CUTOFF="$2"
            shift 2
            ;;
        -m)
            METRIC="$2"
            shift 2
            ;;
        -r)
            REFERENCE="$2"
            shift 2
            ;;
        -region)
            REGION="$2"
            shift 2
            ;;
        -size)
            SIZE="$2"
            shift 2
            ;;
        --subset-sequence-list)
            SUBSET_LIST="$2"
            shift 2
            ;;
        --collapse)
            COLLAPSE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$SEQUENCE_FILES" || -z "$ALIGNMENT" || -z "$REGION" || -z "$SUBSET_LIST" ]]; then
    echo "Error: Missing required parameters"
    echo "Usage: ibs --sequence-files <file> -a <alignment> -region <chr> -size <size> --subset-sequence-list <list> [options]"
    exit 1
fi

# Map metric names to column names
case $METRIC in
    jaccard)
        METRIC_COL="jaccard.similarity"
        ;;
    cosine|cosin)
        METRIC_COL="cosine.similarity"
        ;;
    dice)
        METRIC_COL="dice.similarity"
        ;;
    identity)
        METRIC_COL="estimated.identity"
        ;;
    *)
        echo "Unknown metric: $METRIC"
        exit 1
        ;;
esac

# Get chromosome length (this would need to be adapted to your data)
# For now, using a placeholder - you'd need to get actual chr length
# Example for chr1: ~248M bp
case $REGION in
    chr1) CHR_LENGTH=248956422 ;;
    chr2) CHR_LENGTH=242193529 ;;
    chr20) CHR_LENGTH=64444167 ;;
    # Add more chromosomes as needed
    *) 
        echo "Warning: Unknown chromosome length for $REGION, using 250M"
        CHR_LENGTH=250000000
        ;;
esac

# Create temporary file for raw results
TEMP_RAW=$(mktemp)

# Write header
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b" > "$OUTPUT"

# Iterate over chromosome in windows
START=1
while [[ $START -lt $CHR_LENGTH ]]; do
    END=$((START + SIZE))
    
    # Run impg similarity for this window
    COORD="${REFERENCE}#0#${REGION}:${START}-${END}"
    
    impg similarity \
        --sequence-files "$SEQUENCE_FILES" \
        -a "$ALIGNMENT" \
        -r "$COORD" \
        --subset-sequence-list "$SUBSET_LIST" \
        --force-large-region 2>/dev/null | \
    awk -v cutoff="$CUTOFF" -v metric="$METRIC_COL" '
    BEGIN {FS=OFS="\t"}
    NR==1 {
        # Find the column index for the metric
        for(i=1; i<=NF; i++) {
            if($i == metric) metric_idx = i
        }
        next
    }
    NR>1 {
        if($metric_idx >= cutoff) {
            print $1, $2, $3, $4, $5
        }
    }
    ' >> "$TEMP_RAW"
    
    START=$((START + SIZE))
done

# Apply collapse if requested
if [[ "$COLLAPSE" == "T" || "$COLLAPSE" == "TRUE" || "$COLLAPSE" == "true" ]]; then
    # Collapse consecutive segments with same haplotype pairs
    awk 'BEGIN {FS=OFS="\t"}
    NR==1 {print; next}
    {
        key = $4 OFS $5
        if(key == prev_key && $2 == prev_end) {
            # Extend current segment
            prev_end = $3
        } else {
            # Print previous segment if exists
            if(NR > 2) {
                print prev_chr, prev_start, prev_end, prev_grp_a, prev_grp_b
            }
            # Start new segment
            prev_chr = $1
            prev_start = $2
            prev_end = $3
            prev_grp_a = $4
            prev_grp_b = $5
            prev_key = key
        }
    }
    END {
        if(NR > 1) {
            print prev_chr, prev_start, prev_end, prev_grp_a, prev_grp_b
        }
    }' "$TEMP_RAW" >> "$OUTPUT"
else
    # Just concatenate results
    cat "$TEMP_RAW" >> "$OUTPUT"
fi

# Cleanup
rm "$TEMP_RAW"

echo "IBS analysis complete. Results written to $OUTPUT"
