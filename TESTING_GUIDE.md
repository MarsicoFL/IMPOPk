# HPRCv2-IBD Testing Guide

A practical, step-by-step guide for testing the HPRCv2-IBD pipeline.

---

## Table of Contents

1. [Dependencies and Requirements](#1-dependencies-and-requirements)
2. [Test Data Strategy](#2-test-data-strategy)
3. [Unit Tests (Rust Library)](#3-unit-tests-rust-library)
4. [Binary Tests](#4-binary-tests)
5. [Integration Tests](#5-integration-tests)
6. [Test Fixtures](#6-test-fixtures)
7. [Validation Checklist](#7-validation-checklist)
8. [Output Analysis Plan](#8-output-analysis-plan)

---

## 1. Dependencies and Requirements

### 1.1 External Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **impg** | Implicit pangenome graph similarity queries | `cargo install impg` or build from [https://github.com/ekg/impg](https://github.com/ekg/impg) |
| **GNU Parallel** | Parallel execution for `run_full.sh` | `apt install parallel` / `brew install parallel` |
| **Rscript** | HMM-based IBD calling in shell script | `apt install r-base` / `brew install r` |
| **Rust toolchain** | Compiling the Rust binaries | [https://rustup.rs](https://rustup.rs) |

### 1.2 Verify External Tools

```bash
# Check impg installation
impg --version
# Expected: version number (e.g., impg 0.x.x)

# Check GNU Parallel
parallel --version
# Expected: GNU parallel version info

# Check Rscript
Rscript --version
# Expected: R scripting front-end version info
```

### 1.3 Data Files Required

| File | Description | Download URL | Size |
|------|-------------|--------------|------|
| AGC archive | HPRC assemblies in AGC format | `wget https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc` | ~50 GB |
| PAF alignment | Alignment against CHM13 | `wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz` | ~2 GB |
| PAF index | GZI index for PAF | `wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz.gzi` | ~1 MB |
| IMPG index | Pre-built impg index | `wget https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg` | ~5 GB |

### 1.4 Minimum System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 32 GB+ |
| Disk space | 100 GB (for full data) | 200 GB+ |
| CPU cores | 4 | 16+ (for parallel processing) |

---

## 2. Test Data Strategy

### 2.1 Testing Without Full HPRC Data

The full HPRC dataset is ~50GB. For rapid testing, use these strategies:

#### Option A: Use Existing Test Fixtures (Recommended for CI)

The repository includes minimal test fixtures that do not require impg or real data:

```
production/ibs-cli/tests/data/jacquard_toy.tsv  # Pre-computed IBS windows
```

#### Option B: Create Synthetic IBS Data

Create a synthetic IBS file for testing Jacquard and IBD components:

```bash
cat > /tmp/synthetic_ibs.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	SampleA#1#chr1:0-4999	SampleA#2#chr1:0-4999	1.0
chr1	0	4999	SampleB#1#chr1:0-4999	SampleB#2#chr1:0-4999	1.0
chr1	5000	9999	SampleA#1#chr1:5000-9999	SampleB#1#chr1:5000-9999	0.9999
chr1	5000	9999	SampleA#2#chr1:5000-9999	SampleB#2#chr1:5000-9999	0.9999
chr1	10000	14999	SampleA#1#chr1:10000-14999	SampleA#2#chr1:10000-14999	0.9998
chr1	10000	14999	SampleB#1#chr1:10000-14999	SampleB#2#chr1:10000-14999	0.9998
chr1	15000	19999	SampleA#1#chr1:15000-19999	SampleB#1#chr1:15000-19999	0.9995
chr1	20000	24999	SampleA#1#chr1:20000-24999	SampleB#1#chr1:20000-24999	0.9996
chr1	25000	29999	SampleA#1#chr1:25000-29999	SampleB#1#chr1:25000-29999	0.9997
EOF
```

#### Option C: Small Real Data Subset (Requires impg + data)

Use a small genomic region (e.g., 50kb instead of full chromosome):

```bash
# Minimal region for quick test
CHR=chr20
START=1
END=50000
SIZE=5000
```

### 2.2 Sample Lists for Testing

Existing sample lists are in `production/ibs-cli/sample_lists/`:

- `ibs_example.txt` - 2 haplotypes for minimal testing
- `ibs_example2.txt` - Additional example
- `HPRCv2_AFRsubset.txt` - African ancestry subset
- `HPRCv2_EURsubset.txt` - European ancestry subset
- `HPRCv2_EASsubset.txt` - East Asian ancestry subset
- `HPRCv2_AMRsubset.txt` - American ancestry subset
- `HPRCv2_CSAsubset.txt` - Central/South Asian ancestry subset

---

## 3. Unit Tests (Rust Library)

### 3.1 Run All Library Tests

```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli
cargo test --lib
```

### 3.2 Existing Test Coverage

The library includes tests in these modules:

| Module | File | Tests |
|--------|------|-------|
| **lib.rs** | `src/lib.rs` | `test_region_parse_with_coords`, `test_region_parse_without_coords`, `test_window_iterator` |
| **hmm** | `src/hmm.rs` | `test_viterbi_simple`, `test_extract_segments` |
| **stats** | `src/stats.rs` | `test_gaussian_pdf`, `test_kmeans`, `test_online_stats` |
| **segment** | `src/segment.rs` | `test_segment_length` |

### 3.3 Run Specific Module Tests

```bash
# Test HMM module only
cargo test hmm::tests

# Test stats module only
cargo test stats::tests

# Test segment module only
cargo test segment::tests
```

### 3.4 Run Tests with Output

```bash
cargo test --lib -- --nocapture
```

### 3.5 Expected Output

```
running 8 tests
test hmm::tests::test_viterbi_simple ... ok
test hmm::tests::test_extract_segments ... ok
test stats::tests::test_gaussian_pdf ... ok
test stats::tests::test_kmeans ... ok
test stats::tests::test_online_stats ... ok
test segment::tests::test_segment_length ... ok
test tests::test_region_parse_with_coords ... ok
test tests::test_region_parse_without_coords ... ok
test tests::test_window_iterator ... ok

test result: ok. 8 passed; 0 failed; 0 ignored
```

---

## 4. Binary Tests

### 4.1 Build All Binaries

```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli
cargo build --release
```

**Expected output**: Three binaries created in `target/release/`:
- `ibs`
- `jacquard`
- `ibd`

### 4.2 Test `jacquard` Binary (No External Dependencies)

The `jacquard` binary can be tested without impg using pre-computed IBS files.

#### 4.2.1 Show Help

```bash
./target/release/jacquard --help
```

**Expected output**:
```
Compute Jacquard delta coefficients from IBS windows

Usage: jacquard --ibs <IBS> --hap-a1 <HAP_A1> --hap-a2 <HAP_A2> --hap-b1 <HAP_B1> --hap-b2 <HAP_B2>

Options:
      --ibs <IBS>        IBS windows file (TSV with chrom/start/end/group.a/group.b)
      --hap-a1 <HAP_A1>
      --hap-a2 <HAP_A2>
      --hap-b1 <HAP_B1>
      --hap-b2 <HAP_B2>
  -h, --help             Print help
  -V, --version          Print version
```

#### 4.2.2 Run with Test Fixture

```bash
./target/release/jacquard \
  --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" \
  --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" \
  --hap-b2 "HGB#2"
```

**Expected stdout format**:
```
Delta1	0.00000000	(count=0)
Delta2	0.50000000	(count=1)
...
Delta9	0.00000000	(count=0)
```

**Expected stderr format**:
```
# chrom	chr1	min_start	0	max_end	9999	win_size	5000
# total_windows	2	loci_with_IBS_fourhaps	2	missing_windows_as_Delta9	0	unclassified	0
```

### 4.3 Test `ibs` Binary (Requires impg)

#### 4.3.1 Show Help

```bash
./target/release/ibs --help
```

#### 4.3.2 Test with Real Data (if available)

```bash
./target/release/ibs \
  --sequence-files /path/to/HPRC_r2_assemblies_0.6.1.agc \
  -a /path/to/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-15000 \
  --size 5000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibs_test.tsv
```

**Expected output format** (`/tmp/ibs_test.tsv`):
```
chrom	start	end	group.a	group.b	estimated.identity
chr20	1	5000	HG01167#1#chr20:1-5000	NA19682#1#chr20:1-5000	0.9999
...
```

### 4.4 Test `ibd` Binary (Requires impg)

#### 4.4.1 Show Help

```bash
./target/release/ibd --help
```

#### 4.4.2 Test with Real Data (if available)

```bash
./target/release/ibd \
  --sequence-files /path/to/HPRC_r2_assemblies_0.6.1.agc \
  -a /path/to/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-100000 \
  --size 5000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibd_test.tsv \
  --ibs-output /tmp/ibs_windows.tsv \
  --min-len-bp 10000 \
  --min-windows 3
```

**Expected output format** (`/tmp/ibd_test.tsv`):
```
chrom	start	end	group.a	group.b	n_windows	mean_identity
chr20	5000	25000	HG01167#1	NA19682#1	4	0.999800
...
```

---

## 5. Integration Tests

### 5.1 Parity Tests (Script vs Rust)

The parity test framework compares bash scripts with their Rust equivalents:

```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli
cargo test --test parity
```

**What it verifies**:
- `jacquard` binary produces identical output to `jacquard_coeffs.sh`
- Uses test fixtures in `tests/data/`
- Compares stdout, stderr, and exit codes

**Expected output**:
```
running 1 test
test parity_specs ... ok

test result: ok. 1 passed; 0 failed; 0 ignored
```

### 5.2 Full Pipeline Test (Requires Real Data)

#### 5.2.1 Quick Pipeline Test (50kb region)

```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli/scripts

# Set minimal test parameters
export AGC=/path/to/HPRC_r2_assemblies_0.6.1.agc
export PAF=/path/to/hprc465vschm13.aln.paf.gz
export SUB=../sample_lists/ibs_example.txt
export CHR=chr20
export START=1
export END=50000
export SIZE=5000
export JOBS=2

./run_full.sh
```

**Expected runtime**: ~2-5 minutes for 50kb region with 2 samples

**Expected output**: `../ibs_for_ibd.out` with format:
```
chrom	start	end	group.a	group.b	estimated.identity
chr20	1	5000	...	...	0.9999
```

#### 5.2.2 Jacquard Coefficient Calculation

```bash
./jacquard_coeffs.sh \
  --ibs ../ibs_for_ibd.out \
  --hap-a1 "HG01167#1" \
  --hap-a2 "HG01167#2" \
  --hap-b1 "NA19682#1" \
  --hap-b2 "NA19682#2"
```

#### 5.2.3 IBD Segment Detection (Shell Script)

```bash
./ibd.sh \
  --sequence-files $AGC \
  -a $PAF \
  -r CHM13 \
  -region chr20:1-100000 \
  -size 5000 \
  --subset-sequence-list ../sample_lists/ibs_example.txt \
  --output /tmp/ibd_segments.tsv
```

---

## 6. Test Fixtures

### 6.1 Existing Fixtures

| File | Purpose | Location |
|------|---------|----------|
| `jacquard_toy.tsv` | Minimal IBS data for Jacquard testing | `tests/data/jacquard_toy.tsv` |

### 6.2 Proposed Additional Fixtures

#### 6.2.1 Delta State Test Fixtures

Create fixtures that produce each Jacquard state:

**Delta1 (All 4 haplotypes identical)**:
```bash
cat > tests/data/delta1_test.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	A#1#chr1:0-4999	A#2#chr1:0-4999	1.0
chr1	0	4999	A#1#chr1:0-4999	B#1#chr1:0-4999	1.0
chr1	0	4999	A#1#chr1:0-4999	B#2#chr1:0-4999	1.0
chr1	0	4999	A#2#chr1:0-4999	B#1#chr1:0-4999	1.0
chr1	0	4999	A#2#chr1:0-4999	B#2#chr1:0-4999	1.0
chr1	0	4999	B#1#chr1:0-4999	B#2#chr1:0-4999	1.0
EOF
```

**Delta7 (A1=B1, A2=B2 cross-match)**:
```bash
cat > tests/data/delta7_test.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	A#1#chr1:0-4999	B#1#chr1:0-4999	1.0
chr1	0	4999	A#2#chr1:0-4999	B#2#chr1:0-4999	1.0
EOF
```

**Delta9 (No IBS among 4 haplotypes)**:
```bash
cat > tests/data/delta9_test.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	X#1#chr1:0-4999	Y#1#chr1:0-4999	1.0
EOF
```

#### 6.2.2 HMM/IBD Test Fixture

For testing the IBD HMM with known state transitions:

```bash
cat > tests/data/ibd_hmm_test.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	A#1	B#1	0.5
chr1	5000	9999	A#1	B#1	0.6
chr1	10000	14999	A#1	B#1	0.55
chr1	15000	19999	A#1	B#1	0.999
chr1	20000	24999	A#1	B#1	0.998
chr1	25000	29999	A#1	B#1	0.9995
chr1	30000	34999	A#1	B#1	0.999
chr1	35000	39999	A#1	B#1	0.997
chr1	40000	44999	A#1	B#1	0.5
chr1	45000	49999	A#1	B#1	0.4
EOF
```

**Expected**: IBD segment detected in windows 15000-39999 (5 windows)

#### 6.2.3 Edge Case Fixtures

**Empty file**:
```bash
cat > tests/data/empty.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
EOF
```

**Single window**:
```bash
cat > tests/data/single_window.tsv << 'EOF'
chrom	start	end	group.a	group.b	estimated.identity
chr1	0	4999	A#1#chr1:0-4999	B#1#chr1:0-4999	0.9999
EOF
```

---

## 7. Validation Checklist

### 7.1 Environment Setup

- [ ] Rust toolchain installed (`rustc --version`)
- [ ] impg installed and in PATH (`impg --version`)
- [ ] GNU Parallel installed (`parallel --version`)
- [ ] Rscript installed (`Rscript --version`)

### 7.2 Data Files (Optional, for full integration)

- [ ] AGC archive downloaded and accessible
- [ ] PAF alignment file downloaded
- [ ] PAF index file (.gzi) in same directory as PAF
- [ ] IMPG index file downloaded (optional, improves speed)

### 7.3 Build Verification

- [ ] `cargo build --release` completes without errors
- [ ] Binary `target/release/ibs` exists
- [ ] Binary `target/release/jacquard` exists
- [ ] Binary `target/release/ibd` exists

### 7.4 Unit Tests

- [ ] `cargo test --lib` passes all tests
- [ ] HMM module tests pass (`cargo test hmm::tests`)
- [ ] Stats module tests pass (`cargo test stats::tests`)
- [ ] Segment module tests pass (`cargo test segment::tests`)

### 7.5 Parity Tests

- [ ] `cargo test --test parity` passes
- [ ] Jacquard output matches between shell script and Rust

### 7.6 Binary Functional Tests

- [ ] `jacquard --help` displays usage
- [ ] `jacquard` with toy fixture produces expected Delta output
- [ ] `ibs --help` displays usage (impg not required for help)
- [ ] `ibd --help` displays usage (impg not required for help)

### 7.7 Integration Tests (Requires Data)

- [ ] `ibs` produces valid TSV output
- [ ] `jacquard` classifies windows into Delta states
- [ ] `ibd` detects IBD segments with HMM
- [ ] Shell scripts (`ibs.sh`, `jacquard_coeffs.sh`, `ibd.sh`) execute successfully

---

## 8. Output Analysis Plan

### 8.1 IBS Output Validation

**File**: Output from `ibs` binary or `ibs.sh`

| Check | Method | Success Criteria |
|-------|--------|------------------|
| Header present | `head -1 file.tsv` | Contains `chrom start end group.a group.b estimated.identity` |
| Tab-separated | `awk -F'\t' 'NF!=6 {exit 1}' file.tsv` | Exit code 0 |
| Valid coordinates | `awk '$2 >= 0 && $3 > $2' file.tsv \| wc -l` | Equals total data rows |
| Identity range | `awk '$6 >= 0 && $6 <= 1' file.tsv \| wc -l` | Equals total data rows |
| Canonical order | `awk '$4 <= $5' file.tsv \| wc -l` | Equals total data rows |

### 8.2 Jacquard Output Validation

**File**: stdout from `jacquard` binary

| Check | Method | Success Criteria |
|-------|--------|------------------|
| All 9 deltas present | `grep -c "^Delta"` | Returns 9 |
| Fractions sum to 1 | Sum of column 2 | ~1.0 (within floating point tolerance) |
| Non-negative counts | All `count=N` values | N >= 0 |
| Count matches fraction | `count / total` | Equals fraction in column 2 |

**Validation script**:
```bash
./target/release/jacquard --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" --hap-b2 "HGB#2" \
  | awk -F'\t' '
    BEGIN { sum = 0 }
    /^Delta/ {
      sum += $2
      if ($2 < 0 || $2 > 1) { print "ERROR: invalid fraction", $0; exit 1 }
    }
    END {
      if (sum < 0.999 || sum > 1.001) {
        print "ERROR: fractions sum to", sum, "expected ~1.0"
        exit 1
      }
      print "OK: fractions sum to", sum
    }
  '
```

### 8.3 IBD Output Validation

**File**: Output from `ibd` binary

| Check | Method | Success Criteria |
|-------|--------|------------------|
| Header present | `head -1` | Contains `chrom start end group.a group.b n_windows mean_identity` |
| Valid segments | `start < end` | All rows |
| n_windows >= min | User-specified `--min-windows` | All rows |
| length >= min | `end - start >= --min-len-bp` | All rows |
| Mean identity in range | `0 <= mean_identity <= 1` | All rows |

### 8.4 Regression Detection

Compare outputs across versions:

```bash
# Save baseline output
./target/release/jacquard --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" --hap-b2 "HGB#2" \
  > baseline_jacquard.txt 2> baseline_jacquard.stderr

# After code changes, compare
./target/release/jacquard --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" --hap-b2 "HGB#2" \
  > new_jacquard.txt 2> new_jacquard.stderr

diff baseline_jacquard.txt new_jacquard.txt
diff baseline_jacquard.stderr new_jacquard.stderr
```

### 8.5 Performance Benchmarks

Track runtime for standard test cases:

```bash
# Time the jacquard binary
time ./target/release/jacquard --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" --hap-b2 "HGB#2"

# Expected: < 0.1 seconds for toy fixture
# Alert if: > 1 second (indicates regression)
```

---

## Appendix A: Quick Start Commands

```bash
# 1. Navigate to project
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli

# 2. Build
cargo build --release

# 3. Run all unit tests
cargo test --lib

# 4. Run parity tests
cargo test --test parity

# 5. Quick smoke test (no external data needed)
./target/release/jacquard \
  --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" --hap-b2 "HGB#2"
```

## Appendix B: Troubleshooting

### impg not found

```
ERROR: 'impg' is not in PATH
```

**Solution**: Install impg or add to PATH:
```bash
export PATH=$PATH:/path/to/impg/directory
```

### Missing test data

```
ERROR: sequence file not found
```

**Solution**: Download required data files (see Section 1.3) or use synthetic fixtures.

### Parity test failures

```
assertion failed: script_stdout != rust_stdout
```

**Solution**: This indicates a difference between shell script and Rust implementation. Check:
1. Input file format matches expected columns
2. Both implementations handle edge cases identically
3. Floating point precision differences (normalize output)

---

*Last updated: 2026-01-13*
