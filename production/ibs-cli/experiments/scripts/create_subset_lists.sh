#!/bin/bash
# Create 4-sample (8 haplotype) subsets for each population for manageable experiments

SAMPLE_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/sample_lists"
OUTPUT_DIR="/home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/experiments/data"

# AFR - first 4 samples (8 haplotypes)
head -8 "$SAMPLE_DIR/HPRCv2_AFRsubset.txt" > "$OUTPUT_DIR/AFR_4samples.txt"

# EUR - first 4 samples (8 haplotypes)
head -8 "$SAMPLE_DIR/HPRCv2_EURsubset.txt" > "$OUTPUT_DIR/EUR_4samples.txt"

# EAS - first 4 samples (8 haplotypes)
head -8 "$SAMPLE_DIR/HPRCv2_EASsubset.txt" > "$OUTPUT_DIR/EAS_4samples.txt"

# CSA - first 4 samples (8 haplotypes)
head -8 "$SAMPLE_DIR/HPRCv2_CSAsubset.txt" > "$OUTPUT_DIR/CSA_4samples.txt"

# AMR - all 3 samples (6 haplotypes)
cp "$SAMPLE_DIR/HPRCv2_AMRsubset.txt" "$OUTPUT_DIR/AMR_3samples.txt"

echo "Created subset lists:"
wc -l "$OUTPUT_DIR"/*.txt
