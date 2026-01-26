# Tutorial: IBS Detection with `ibs` Binary and `ibs.sh` Script

## Overview

The IBS (Identity-By-State) detection pipeline identifies regions where haplotypes share identical (or near-identical) sequence. This is the foundational step for downstream IBD analysis and Jacquard coefficient computation.

This tutorial covers both:
- **Rust binary**: `cargo run --bin ibs` (recommended for production)
- **Shell script**: `scripts/ibs.sh` (useful for prototyping or parity testing)

---

## How It Works

The IBS pipeline:
1. Divides a genomic region into fixed-size windows
2. For each window, calls `impg similarity` to compare all haplotypes
3. Filters results by identity threshold (default: 0.999)
4. Outputs IBS-positive haplotype pairs per window

---

## Prerequisites

### Required Tools
| Tool | Purpose | Installation |
|------|---------|--------------|
| **impg** | Pangenome similarity queries | `cargo install impg` or build from [github.com/ekg/impg](https://github.com/ekg/impg) |
| **Rust toolchain** | Building the binary | [rustup.rs](https://rustup.rs) |

### Required Data Files
| File | Description |
|------|-------------|
| AGC archive | HPRC assemblies (e.g., `HPRC_r2_assemblies_0.6.1.agc`) |
| PAF alignment | Alignments against reference (e.g., `hprc465vschm13.aln.paf.gz`) |
| Subset list | Text file with haplotype IDs to compare (one per line) |

---

## CLI Reference

### Rust Binary Arguments

```
ibs - wrapper around impg similarity to obtain IBS segments

USAGE:
    ibs [OPTIONS] --sequence-files <FILE> -a <FILE> -r <NAME> --region <REGION> --size <BP> --subset-sequence-list <FILE> --output <FILE>

OPTIONS:
    --sequence-files <FILE>       Path to AGC/FASTA sequence archive (required)
    -a <FILE>                     Alignment file (.paf/.paf.gz) for impg (required)
    -r <NAME>                     Reference name, e.g., CHM13 (required)
    --region <REGION>             Target region: chr1:1-1000000 or chr1 (required)
    --size <BP>                   Window size in base pairs (required)
    --subset-sequence-list <FILE> File with haplotype IDs to compare (required)
    --output <FILE>               Output TSV file path (required)
    -c <FLOAT>                    Identity cutoff [default: 0.999]
    -m <METRIC>                   Metric name (informational) [default: cosin]
    --region-length <BP>          Required if --region omits coordinates
    -t, --threads <N>             Number of parallel threads [default: auto]
    -h, --help                    Print help information
    -V, --version                 Print version information
```

### Shell Script Arguments

The `scripts/ibs.sh` script accepts the same arguments with slightly different syntax:

| Flag | Description |
|------|-------------|
| `--sequence-files` | Path(s) to AGC/FASTA archives |
| `-a` | Alignment file for impg |
| `-r` | Reference name |
| `-region` | Target interval (note: single dash) |
| `-size` | Window size in bp |
| `--output` | Output TSV file |
| `--region-length` | Required when region omits coordinates |
| `--subset-sequence-list` | Haplotype allowlist |
| `-c` | Identity cutoff [default: 1.0 for script] |

---

## Usage Examples

### Example 1: Basic IBS Detection (Small Region)

```bash
cd /path/to/HPRCv2-IBD/production/ibs-cli

# Build the binary
cargo build --release

# Run on a 1MB region with 5kb windows
./target/release/ibs \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-1000000 \
  --size 5000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibs_chr20_1mb.tsv
```

### Example 2: Full Chromosome Analysis

```bash
# For a full chromosome, use --region-length
./target/release/ibs \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20 \
  --region-length 64444167 \
  --size 5000 \
  --subset-sequence-list sample_lists/HPRCv2_AFRsubset.txt \
  --output /results/ibs_chr20_full.tsv
```

### Example 3: Relaxed Identity Threshold

```bash
# Use -c to lower the identity threshold (e.g., for divergent populations)
./target/release/ibs \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-5000000 \
  --size 10000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibs_relaxed.tsv \
  -c 0.995
```

### Example 4: Using the Shell Script

```bash
cd production/ibs-cli/scripts

./ibs.sh \
  --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \
  -a ../data/human/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr20:1-15000000 \
  -size 5000 \
  --subset-sequence-list ../sample_lists/ibs_example.txt \
  --output /tmp/ibs_chr20.tsv
```

---

## Output Format

The output is a tab-separated file with the following columns:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome name |
| `start` | Window start position (1-based) |
| `end` | Window end position |
| `group.a` | First haplotype identifier |
| `group.b` | Second haplotype identifier |
| `estimated.identity` | Sequence identity between haplotypes |

### Example Output

```
chrom	start	end	group.a	group.b	estimated.identity
chr20	1	5000	HG01167#1#chr20:1-5000	NA19682#1#chr20:1-5000	0.9999
chr20	1	5000	HG01167#1#chr20:1-5000	NA19682#2#chr20:1-5000	0.9998
chr20	5001	10000	HG01167#1#chr20:5001-10000	HG01167#2#chr20:5001-10000	0.9999
chr20	5001	10000	HG01167#1#chr20:5001-10000	NA19682#1#chr20:5001-10000	0.9995
```

### Interpreting the Output

- **High identity (>0.999)**: Strong evidence of recent shared ancestry or common haplotype
- **Moderate identity (0.995-0.999)**: Possible IBD with some accumulated mutations
- **Pairs appearing in consecutive windows**: Candidate IBD segments (use `ibd` binary for formal detection)

---

## Performance Considerations

### Window Size Selection

| Window Size | Use Case | Trade-offs |
|-------------|----------|------------|
| 1-5 kb | High resolution | More windows, slower, may fragment IBD segments |
| 5-10 kb | Balanced (recommended) | Good resolution vs. speed |
| 10-50 kb | Large-scale surveys | Fast, may miss short IBD segments |

### Memory and Runtime

- **Memory**: Primarily determined by impg; typically 8-32 GB for HPRC data
- **Runtime**: ~1-5 minutes per 1 Mb region (depends on sample count and window size)
- **Parallelization**: Use `run_full.sh` for parallel processing of large regions

---

## Troubleshooting

### Error: "impg is not in PATH"

```
ERROR: 'impg' is not in PATH
```

**Solution**: Install impg and ensure it is in your PATH:
```bash
# Option 1: cargo install
cargo install impg

# Option 2: Add to PATH
export PATH=$PATH:/path/to/impg/directory
```

### Error: "sequence file not found"

```
ERROR: sequence file not found: /path/to/file.agc
```

**Solution**: Verify the AGC file exists and the path is correct. Download from HPRC if needed:
```bash
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc
```

### Error: "region needs --region-length"

```
ERROR: Region 'chr1' requires --region-length
```

**Solution**: When using chromosome-only region format, provide the length:
```bash
./target/release/ibs --region chr1 --region-length 248956422 ...
```

### Empty Output File

**Possible causes**:
1. Identity cutoff too strict - try lowering `-c` to 0.995
2. No overlapping alignments in the specified region
3. Subset list contains invalid haplotype IDs

**Debugging steps**:
```bash
# Check if impg produces output directly
impg similarity \
  --sequence-files /data/file.agc \
  -a /data/file.paf.gz \
  -r "CHM13#0#chr20:1-5000" \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --force-large-region | head
```

### Slow Performance

**Possible causes**:
1. Very small window size creating many impg calls
2. Large subset list with many pairwise comparisons
3. Missing PAF index file (.gzi)

**Solutions**:
- Increase window size (e.g., 10000 instead of 5000)
- Reduce sample count in subset list
- Ensure `.paf.gz.gzi` index exists alongside the PAF file
- Use `run_full.sh` for parallel processing

---

## Next Steps

After generating IBS windows:

1. **Jacquard Coefficients**: Compute identity state distributions
   ```bash
   ./target/release/jacquard --ibs output.tsv --hap-a1 ... --hap-a2 ... --hap-b1 ... --hap-b2 ...
   ```

2. **IBD Segment Detection**: Run HMM-based IBD calling
   ```bash
   ./target/release/ibd --region chr20:1-1000000 --output ibd_segments.tsv ...
   ```

See the [IBD tutorial](ibd.md) and [Jacquard tutorial](jacquard_coeffs.md) for details.

---

## See Also

- [TESTING_GUIDE.md](../../TESTING_GUIDE.md) - Complete testing procedures
- [run_full.sh tutorial](run_full.md) - Parallel processing for large regions
- [Conceptual Framework](../paper_concepts/conceptual_framework.md) - Theoretical background
