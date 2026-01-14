#!/bin/bash
# Generate final comprehensive report from all IBS experiments

EXPERIMENTS_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/experiments"
RESULTS_DIR="$EXPERIMENTS_DIR/results"
REPORTS_DIR="$EXPERIMENTS_DIR/reports"
SCRIPTS_DIR="$EXPERIMENTS_DIR/scripts"

mkdir -p "$REPORTS_DIR/plots"

echo "Generating IBS Analysis Report..."
echo "=================================="

# Build input arguments for Python script
INTRA_INPUTS=""
INTER_INPUTS=""

# Collect intra-population results
for pop in AFR EUR EAS CSA AMR; do
    file="$RESULTS_DIR/${pop}_intra/${pop}_intra_ibs.tsv"
    if [ -f "$file" ] && [ -s "$file" ]; then
        INTRA_INPUTS="$INTRA_INPUTS ${pop}:$file"
        echo "Found: $pop intra-population ($(wc -l < "$file") lines)"
    fi
done

# Collect inter-population results
for dir in "$RESULTS_DIR"/*_vs_*_inter/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir" | sed 's/_inter$//')
        file="$dir"/*_inter_ibs.tsv
        if [ -f $file ] && [ -s $file ]; then
            INTER_INPUTS="$INTER_INPUTS ${name}:$file"
            echo "Found: $name inter-population ($(wc -l < $file) lines)"
        fi
    fi
done

# Generate intra-population report
if [ -n "$INTRA_INPUTS" ]; then
    echo ""
    echo "Generating intra-population report..."
    python3 "$SCRIPTS_DIR/analyze_ibs.py" \
        --input $INTRA_INPUTS \
        --output "$REPORTS_DIR/intra_population" \
        --name "intra_population_chr2_LCT"
fi

# Generate inter-population report
if [ -n "$INTER_INPUTS" ]; then
    echo ""
    echo "Generating inter-population report..."
    python3 "$SCRIPTS_DIR/analyze_ibs.py" \
        --input $INTER_INPUTS \
        --output "$REPORTS_DIR/inter_population" \
        --name "inter_population_chr2_LCT"
fi

# Generate combined summary
echo ""
echo "Generating combined summary..."

COMBINED_REPORT="$REPORTS_DIR/COMBINED_IBS_REPORT.md"

cat > "$COMBINED_REPORT" << 'EOF'
# Combined IBS Analysis Report: Chromosome 2 LCT Region

## Experiment Overview

**Region**: chr2:130,787,850-140,837,183 (10Mb around LCT gene)
**Window Size**: 5,000 bp
**Reference**: CHM13
**Identity Cutoff**: 0.999

## Population Groups Analyzed

| Population | Code | Samples | Haplotypes |
|------------|------|---------|------------|
| African | AFR | 4 | 8 |
| European | EUR | 4 | 8 |
| East Asian | EAS | 4 | 8 |
| Central/South Asian | CSA | 4 | 8 |
| Americas | AMR | 3 | 6 |

EOF

# Add statistics from each experiment
echo "## Summary Statistics" >> "$COMBINED_REPORT"
echo "" >> "$COMBINED_REPORT"
echo "### Intra-Population Comparisons" >> "$COMBINED_REPORT"
echo "" >> "$COMBINED_REPORT"
echo "| Population | Windows | Mean IBS | Min | Max | High IBS (%) |" >> "$COMBINED_REPORT"
echo "|------------|---------|----------|-----|-----|--------------|" >> "$COMBINED_REPORT"

for pop in AFR EUR EAS CSA AMR; do
    file="$RESULTS_DIR/${pop}_intra/${pop}_intra_ibs.tsv"
    if [ -f "$file" ] && [ -s "$file" ]; then
        stats=$(awk -F'\t' 'NR>1 {
            sum+=$6; count++;
            if(min=="" || $6<min) min=$6;
            if(max=="" || $6>max) max=$6;
            if($6>=0.999) high++;
        } END {
            if(count>0) printf "%d\t%.4f\t%.4f\t%.4f\t%.1f", count, sum/count, min, max, (high/count)*100;
            else print "0\t0\t0\t0\t0"
        }' "$file")

        n=$(echo "$stats" | cut -f1)
        mean=$(echo "$stats" | cut -f2)
        min=$(echo "$stats" | cut -f3)
        max=$(echo "$stats" | cut -f4)
        high=$(echo "$stats" | cut -f5)

        echo "| $pop | $n | $mean | $min | $max | $high |" >> "$COMBINED_REPORT"
    fi
done

echo "" >> "$COMBINED_REPORT"
echo "### Inter-Population Comparisons" >> "$COMBINED_REPORT"
echo "" >> "$COMBINED_REPORT"
echo "| Comparison | Windows | Mean IBS | Min | Max | High IBS (%) |" >> "$COMBINED_REPORT"
echo "|------------|---------|----------|-----|-----|--------------|" >> "$COMBINED_REPORT"

for dir in "$RESULTS_DIR"/*_vs_*_inter/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir" | sed 's/_inter$//' | sed 's/_/ vs /')
        file=$(ls "$dir"/*_inter_ibs.tsv 2>/dev/null | head -1)
        if [ -f "$file" ] && [ -s "$file" ]; then
            stats=$(awk -F'\t' 'NR>1 {
                sum+=$6; count++;
                if(min=="" || $6<min) min=$6;
                if(max=="" || $6>max) max=$6;
                if($6>=0.999) high++;
            } END {
                if(count>0) printf "%d\t%.4f\t%.4f\t%.4f\t%.1f", count, sum/count, min, max, (high/count)*100;
                else print "0\t0\t0\t0\t0"
            }' "$file")

            n=$(echo "$stats" | cut -f1)
            mean=$(echo "$stats" | cut -f2)
            min=$(echo "$stats" | cut -f3)
            max=$(echo "$stats" | cut -f4)
            high=$(echo "$stats" | cut -f5)

            echo "| $name | $n | $mean | $min | $max | $high |" >> "$COMBINED_REPORT"
        fi
    fi
done

echo "" >> "$COMBINED_REPORT"
echo "## Detailed Reports" >> "$COMBINED_REPORT"
echo "" >> "$COMBINED_REPORT"
echo "- [Intra-Population Analysis](intra_population/intra_population_chr2_LCT_report.md)" >> "$COMBINED_REPORT"
echo "- [Inter-Population Analysis](inter_population/inter_population_chr2_LCT_report.md)" >> "$COMBINED_REPORT"
echo "" >> "$COMBINED_REPORT"
echo "---" >> "$COMBINED_REPORT"
echo "*Report generated: $(date)*" >> "$COMBINED_REPORT"

echo ""
echo "=================================="
echo "Report generated: $COMBINED_REPORT"
echo "=================================="
