#!/usr/bin/env bash
set -euo pipefail

# Download Platinum Pedigree (CEPH 1463) assemblies for validation
#
# The CEPH 1463 family has 28 members with phased assemblies available
# from the platinum-pedigree-data S3 bucket (no authentication required).
#
# Family structure (CEPH 1463):
#   Parents:      NA12877 (father), NA12878 (mother)
#   Maternal GPs: NA12891 (maternal GF), NA12892 (maternal GM)
#   Paternal GPs: NA12889 (paternal GF), NA12890 (paternal GM)
#   Children:     NA12879, NA12881-NA12887 (offspring of NA12877+NA12878)
#
# Source: s3://platinum-pedigree-data/ (AWS, no authentication needed)
# Paper: Ebler et al. (2022) "Pangenome-based genome inference..."
# Also: https://github.com/human-pangenomics/HPP_Year1_Assemblies

DATA_DIR="./data"
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download Platinum Pedigree (CEPH 1463) assemblies for validation."
    echo "Uses verkko assemblies for the 14 sequenced adults (NA128xx)"
    echo "and hifiasm assemblies for the 14 generation-3 members (2000xx)."
    echo ""
    echo "Options:"
    echo "  --data-dir DIR   Base data directory (default: ./data/)"
    echo "  --dry-run        Show what would be downloaded without downloading"
    echo "  -h, --help       Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

PLAT_DIR="${DATA_DIR}/platinumPed"

# Verkko assemblies for the 15 adults (NA12877-NA12892)
# These are available from the HPRC/HPP Year 1 release
VERKKO_SAMPLES=(
    NA12877 NA12878 NA12879 NA12880 NA12881 NA12882 NA12883
    NA12884 NA12885 NA12886 NA12887 NA12889 NA12890
    NA12891 NA12892
)

# Hifiasm assemblies for generation-3 members
HIFIASM_SAMPLES=(
    200080 200081 200082 200084 200085 200086 200087
    200100 200101 200102 200103 200104 200105 200106
)

# HPRC CEPH 1463 assembly URLs (S3 public bucket)
VERKKO_BASE_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/working/HPRC/CEPH1463/assemblies/verkko"
HIFIASM_BASE_URL="https://s3-us-west-2.amazonaws.com/human-pangenomics/working/HPRC/CEPH1463/assemblies/hifiasm"

download_haplotype() {
    local base_url="$1"
    local sample="$2"
    local hap="$3"
    local dest_dir="$4"
    local assembler="$5"

    local filename="${sample}.${hap}.fa.gz"
    local dest="${dest_dir}/${assembler}/${filename}"
    # TODO: Verify exact URL pattern. Common patterns:
    #   ${base_url}/${sample}/${sample}.${hap}.fa.gz
    #   ${base_url}/${sample}.${hap}.fa.gz
    local url="${base_url}/${sample}/${filename}"

    if [[ -f "$dest" ]]; then
        echo "  Already exists: $dest"
        return 0
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would download: $filename"
        echo "    URL:  $url"
        echo "    Dest: $dest"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    echo "  Downloading: $filename"
    wget -c -q --show-progress -O "$dest" "$url" || {
        echo "  WARNING: Failed to download $filename (URL may need updating)"
        rm -f "$dest"
        return 0
    }
}

echo "=== Platinum Pedigree (CEPH 1463) Download ==="
echo ""
echo "Family: 28 members, 56 haplotypes"
echo "  Verkko assemblies: ${#VERKKO_SAMPLES[@]} adults (NA128xx series)"
echo "  Hifiasm assemblies: ${#HIFIASM_SAMPLES[@]} generation-3 (2000xx series)"
echo ""

# Download verkko assemblies (adults)
echo "--- Verkko assemblies (adults) ---"
for sample in "${VERKKO_SAMPLES[@]}"; do
    echo "Sample: $sample"
    download_haplotype "$VERKKO_BASE_URL" "$sample" "1" "$PLAT_DIR" "verkko"
    download_haplotype "$VERKKO_BASE_URL" "$sample" "2" "$PLAT_DIR" "verkko"
done
echo ""

# Download hifiasm assemblies (generation 3)
echo "--- Hifiasm assemblies (generation 3) ---"
for sample in "${HIFIASM_SAMPLES[@]}"; do
    echo "Sample: $sample"
    download_haplotype "$HIFIASM_BASE_URL" "$sample" "1" "$PLAT_DIR" "hifiasm"
    download_haplotype "$HIFIASM_BASE_URL" "$sample" "2" "$PLAT_DIR" "hifiasm"
done

echo ""
echo "=== Platinum pedigree download complete ==="
echo ""
echo "NOTE: After downloading, you need to align these assemblies against CHM13"
echo "using minimap2 to generate PAF files for impopk analysis."
echo "See tutorial 06_platinum_pedigree.md for details."
