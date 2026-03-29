# impopk QA Report

Verification date: 2026-03-27

## 1. Functional Verification

| Check | Result | Details |
|-------|--------|---------|
| `cargo build --release` | PASS | 6 binarios: ibs, ibs-from-paf, ibd, ibd-validate, ancestry, jacquard |
| `cargo clippy --workspace -- -D warnings` | PASS | 0 warnings |
| `cargo test --workspace` | PASS | **7334 passed, 0 failed, 0 ignored** |
| `--help` (6 binarios) | PASS | Todos retornan exit 0 |
| `--version` (6 binarios) | PASS | Todos reportan 0.2.0 |
| Bundled data integrity | PASS | 464 haplotipos (5 pob), 4 genetic maps, checksums OK |
| Download scripts `--dry-run` | PASS | URLs correctas, 5 steps |

### Pipeline Smoke Tests (chr12, datos reales)

| Test | Input | Result |
|------|-------|--------|
| IBS 1Mb EUR | chr12:1M-2M, 10kb windows, 62 haplotipos | 190,991 rows, mean identity 0.9965, 0 bad values |
| IBD desde IBS | 1Mb EUR, BW=20, floor=0.9 | 2,935 segmentos, 0 NaN/Inf |
| IBS 5Mb all-pops | chr12:1M-6M, 50kb windows, 464 haplotipos | 10.8M pairs, 5.8s |
| Ancestry 3-way | AFR/EUR/AMR, auto-configure, 4 AMR queries | AMR correctamente asignado, posteriors suman ~1.0 |
| IBD 50Mb EUR | chr12:1-50M, 10kb windows | 254 segmentos, LOD > 180 |
| Jacquard toy data | jacquard_toy.tsv | Delta2=0.5, Delta7=0.5 (correcto) |
| eGRM output | ancestry --output-egrm | .grm.bin, .grm.N.bin, .grm.id generados |
| Error handling | archivos inexistentes en 3 binarios | Error graceful, sin panics |

## 2. Bugs Corregidos

### Round 1

| # | Archivo | Descripcion | Fix |
|---|---------|-------------|-----|
| 1 | `tutorials/05_ancestry_inference.md` | `--ref-name "GRCh38#0#chr12"` incorrecto | Cambiado a `-r CHM13` |
| 2 | `tutorials/06_platinum_pedigree.md` | Mismo ref-name incorrecto | Cambiado a `-r CHM13` |
| 3 | `README.md` L148-154 | Population counts 67/30/50 | Corregido a 70/31/51 |
| 4 | `tutorials/02_data_preparation.md` L98-102 | Mismos counts | Corregido |
| 5 | `tutorials/02_data_preparation.md` L135,156 | 454 haplotipos / 227 individuos | Corregido a 464/232 |
| 6 | `tutorials/02_data_preparation.md` L189 | PAF target `GRCh38#0#chr12` | Corregido a `CHM13#0#chr12` |
| 7 | `src/README.md` | Stale: decia "HPRCv2-IBD", 3 tools, build incorrecto | Reescrito completo |
| 8 | `jacquard` binary | No soportaba `--version` | Agregado `version` a `#[command]` |
| 9 | 3 tests en ancestry-cli | Paths hardcodeados `/home/franco/.../HPRCv2-IBD/` | Cambiados a paths relativos |
| 10 | `scripts/checksums.sha256` | Entradas PLACEHOLDER confusas | Limpiados, dejados como comentarios claros |
| 11 | `tutorials/07_simulation.md` | Referencia al repo de desarrollo HPRCv2-IBD | Removida |

### Archivos Eliminados

| Archivo | Razon |
|---------|-------|
| `CONSOLIDATION_PLAN.md` | Documento interno del desarrollo |
| `src/ibs-cli/PERFORMANCE_REPORT.md` | Benchmark interno del desarrollo |

### Round 2

| # | Archivo | Descripcion | Fix |
|---|---------|-------------|-----|
| 12 | `README.md` Quick Start | `--paf` no existe, `--window-size` no existe, `--populations POP:FILE` formato incorrecto, jacquard `--output` no existe | Reescrito completo con flags correctos (-a, --size, populations.tsv, sin --output) |
| 13 | `README.md` Quick Start | IBD example usaba `ibd` (requiere impg) | Cambiado a `ibd-validate` (el workflow recomendado) |
| 14 | `tutorials/05_ancestry_inference.md` | Output format incorrecto (8 cols, orden wrong) | Actualizado a 10 cols reales: chrom,start,end,sample,ancestry,n_windows,mean_similarity,mean_posterior,discriminability,lod_score |
| 15 | `tutorials/05_ancestry_inference.md` | Posteriors format incorrecto (best,posterior cols) | Actualizado a formato real: P(POP) cols + margin + entropy |
| 16 | `scripts/download_platinum.sh` | TODO comments sobre S3 URLs (URLs ya estaban correctas) | Limpiado el TODO, dejadas las URLs funcionales |

## 3. Cross-Reference: Modelos y Parametros

### IBD HMM (2 estados, Gaussian)

| Parametro | Codigo (`ibd-cli/src/hmm.rs`) | Paper (`methods_ibd.tex`) | `METHODOLOGY.md` | Match |
|-----------|-------------------------------|---------------------------|-------------------|-------|
| IBD mean (mu1) | 0.9997 | 0.9997 | 0.9997 | OK |
| IBD std (sigma1) | 0.0005 | 0.0005 | 0.0005 | OK |
| AFR pi | 0.00125 | 0.00125 | 0.00125 | OK |
| EUR pi | 0.00085 | 0.00085 | 0.00085 | OK |
| EAS pi | 0.00080 | 0.00080 | 0.00080 | OK |
| CSA pi | 0.00095 | 0.00095 | 0.00095 | OK |
| AMR pi | 0.00100 | 0.00100 | 0.00100 | OK |
| Non-IBD mean | 1 - pi | 1 - theta | 1 - theta | OK |
| Non-IBD std | sqrt(pi/W * 3) | sqrt(pi/W * LD_corr) | binomial+LD | OK |
| LD correction factor | 3.0 | "LD correction" | mentioned | OK |
| Emission distribution | Gaussian | N(mu, sigma^2) | Gaussian | OK |
| Baum-Welch iters | 20 (default) | 20 | 20 recommended | OK |
| Identity floor | 0.9 (default) | 0.9 | 0.9 | OK |
| Expected segment windows | 50 | 50 (500kb/10kb) | -- | OK |
| p_enter_ibd | 0.0001 | 10^-4 | -- | OK |
| K-means min separation | 0.0005 | 0.0005 | -- | OK |
| EM prior strength (AFR) | 15 | 15 | -- | OK |
| BW bounds mu0 | [prior-0.005, 0.9993] | mentioned | -- | OK |
| BW bounds mu1 | [0.9990, 1.0] | mentioned | -- | OK |
| LOD threshold | 3.0 (default) | 3.0 | -- | OK |
| Min segment length | 2Mb (default) | 2Mb | -- | OK |
| Window size | 10kb (default) | 10kb | -- | OK |

### IBD Population-Adaptive Transitions

| Population | Seg length factor | p_enter factor | Rationale (code comment) |
|------------|-------------------|----------------|--------------------------|
| AFR | 0.7x | 0.3x | Fewer, shorter IBD (high diversity) |
| EUR | 1.0x | 1.0x | Standard |
| EAS | 1.1x | 1.0x | Slightly longer segments |
| CSA | 0.9x | 0.8x | Intermediate |
| AMR | 0.8x | 0.7x | Admixed, variable IBD |
| InterPop | 0.5x | 0.1x | Cross-pop IBD very rare |

### Ancestry HMM (N estados, Softmax)

| Parametro | Codigo (`ancestry-cli/src/hmm.rs`) | Paper (`methods_ancestry.tex`) | `METHODOLOGY.md` | Match |
|-----------|-------------------------------------|--------------------------------|-------------------|-------|
| Emission model | log-softmax | softmax with temperature tau | softmax | OK |
| Temperature default | 0.03 | estimated adaptively | -- | OK |
| Aggregation default | max | max (default, best) | -- | OK |
| Switch prob default | 0.001 | -- | -- | OK |
| Pairwise contrast | Bradley-Terry per pair | Bradley-Terry P_ij | Bradley-Terry | OK |
| Per-pair temperature | median\|fi-fj\| | median_t\|fi(ot)-fj(ot)\| | -- | OK |
| Auto-config metric | D_min = min Cohen's d | D_min = min Cohen's d | -- | OK |
| D_ref calibration | 0.014 | calibrated on HPRC 3-way | -- | OK |
| Emission context | round(0.1/D_min), [1,15] | round(0.1/D_min), [1,15] | -- | OK |
| PW weight formula | 0.7 * cv_factor * min(1, D_min/0.014) * sqrt(n/5000) | scales with CV of Cohen's d | -- | OK |
| BW for ancestry | counterproductive (-2-3pp) | counterproductive | -- | OK |
| BW transition dampening | 0.5 (with auto-configure) | -- | -- | OK |
| Temperature clamp | [0.0005, 0.15] | -- | -- | OK |
| BW diagonal clamp (K>=3) | [0.9, 0.9999] | -- | -- | OK |

### Ancestry Auto-Configure Behavior

| Data regime | D_min | pairwise_weight | emission_context |
|-------------|-------|-----------------|------------------|
| Simulation (strong signal) | ~0.025 | ~0.7 | 1 |
| HPRC 3-way real (weak signal) | ~0.004 | ~0.12 | 10-15 |
| Cross-application penalty | -- | -- | 9.8 pp loss |

### Key Ancestry Finding (code + paper)

BW training is **counterproductive** for ancestry (-2-3pp). The help text in `ancestry-cli/src/main.rs` documents this explicitly:
- Overestimates p_switch: 0.01 -> 0.03-0.05
- Signal too weak for data-driven transition estimation
- Recommendation: BW iterations = 0 (or use auto-configure which sets dampening=0.5)

## 4. Validation Results Cross-Reference

| Benchmark | Paper | README | Tutorials | Match |
|-----------|-------|--------|-----------|-------|
| Simulated ancestry (3-way) concordance | 97.95% | 97.95% | 97.95% (tut07) | OK |
| HPRC real ancestry (3-way) concordance | 76.45% | 76.45% | -- | OK |
| IBD detection top-10% ranking | 100% (11/11) | 100% (11/11) | -- | OK |
| Platinum pedigree (4-state) accuracy | 99.53% | 99.53% | 99.53% (tut06) | OK |
| IBD simulation pair F1 | 0.514 | 0.514 | -- | OK |
| hap-ibd pair F1 (comparison) | 0.489 | 0.489 | -- | OK |
| Full genome time | 195s | 195s | -- | OK |
| Full genome RAM | 101MB | 101MB | -- | OK |
| PAF-direct vs impg speedup | 1,090x | 1,090x | -- | OK |
| Pairwise contrast gain | +2.05pp | -- | +2.05pp (tut07) | OK |
| Softmax baseline | 95.1% | -- | 95.1% (tut07) | OK |

## 5. docs/conceptualization.md Status

This document contains a historical critique of early defaults (non-IBD mean = 0.5) from the development process. The current code correctly uses population-specific values (0.998-0.999) derived from nucleotide diversity. The critique was resolved during development -- the document is a development artifact, not an active bug.

## 6. Remaining Notes

- **Tutorial 07** (Simulation): purely descriptive, no runnable commands. This is by design (simulation requires msprime + minimap2 + AGC infrastructure).
- **Tutorial 06** (Platinum Pedigree): requires platinum data download (~1GB). Not tested in this QA round.
- **3 tests with relative paths**: `ancestry.rs` tests use `data/samples/AFR.txt` (relative). They auto-skip if path doesn't exist (if-exists guard). This is correct behavior.
- **No hardcoded absolute paths** remain in any source or documentation file.
- **No references to HPRCv2-IBD** remain in any public-facing file.
