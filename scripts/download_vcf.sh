#!/usr/bin/env bash
set -euo pipefail

# Download validation VCFs for IBD and ancestry comparison
#
# These are HPRC phased VCFs used for gold-standard comparison with
# hap-ibd (IBD) and RFMix v2 (local ancestry). We need subsets for
# chromosomes 10, 11, and 12.
#
# Source: 1000 Genomes on GRCh38 / HPRC phased variant calls
#   - 1KG on GRCh38: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
#   - HPRC integrated calls: s3://human-pangenomics/pangenomes/freeze/freeze2/variants/
#
# The subset VCFs contain phased biallelic SNPs for the HPRC samples only,
# used as input to hap-ibd and RFMix for concordance benchmarks.

DATA_DIR="./data"
DRY_RUN=false

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download validation VCFs (chr10, chr11, chr12) for gold-standard comparison"
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

VCF_DIR="${DATA_DIR}/vcf"

# --- 1000 Genomes high-coverage phased VCFs on GRCh38 ---
# These are the 3202-sample phased VCFs from the 1000 Genomes Project
# high-coverage collection aligned to GRCh38.
# After download, subset to HPRC samples using bcftools.
BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"

CHROMOSOMES=(10 11 12)

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [[ -f "$dest" ]]; then
        echo "Already exists: $dest"
        return 0
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would download: $desc"
        echo "  URL:  $url"
        echo "  Dest: $dest"
        return 0
    fi

    local dest_dir
    dest_dir="$(dirname "$dest")"
    mkdir -p "$dest_dir"

    echo "Downloading: $desc"
    echo "  URL:  $url"
    echo "  Dest: $dest"
    wget -c --progress=bar:force -O "$dest" "$url" || {
        echo "WARNING: Failed to download $desc"
        echo "  The URL may have changed. Check:"
        echo "  - https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
        echo "  - https://www.internationalgenome.org/data-portal/data-collection/30x-grch38"
        rm -f "$dest"
        return 1
    }

    echo "Done: $dest"
}

echo "=== Validation VCF Download ==="
echo ""
echo "Downloading 1000 Genomes high-coverage phased VCFs for chr10, chr11, chr12"
echo "These full-chromosome VCFs are ~500MB-1GB each."
echo ""
echo "After download, you can subset to HPRC samples with:"
echo '  bcftools view -S <(cat data/samples/*.txt | sed "s/#[12]$//" | sort -u) \'
echo '    --force-samples -Oz -o ref_chrN_subset.vcf.gz full_chrN.vcf.gz'
echo ""

for chr in "${CHROMOSOMES[@]}"; do
    vcf_name="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    vcf_url="${BASE_URL}/${vcf_name}"
    vcf_dest="${VCF_DIR}/${vcf_name}"

    idx_name="${vcf_name}.tbi"
    idx_url="${BASE_URL}/${idx_name}"
    idx_dest="${VCF_DIR}/${idx_name}"

    download_file "$vcf_url" "$vcf_dest" "Chr${chr} phased VCF"
    download_file "$idx_url" "$idx_dest" "Chr${chr} VCF index"
    echo ""
done

echo "=== VCF download complete ==="
echo ""
echo "Next steps:"
echo "  1. Subset to HPRC samples (see command above)"
echo "  2. Run hap-ibd for IBD gold standard"
echo "  3. Run RFMix v2 for ancestry gold standard"
echo "  See tutorials for detailed instructions."
