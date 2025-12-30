# Haplotype-Based IBS/IBD Analysis for HPRCv2

This repository keeps the tooling that powers the upcoming HPRC reports focused
on haplotype matching. We rely on `impg` to scan selected windows across HPRC
assemblies, measure identity-by-state (IBS) between every pair of haplotypes,
and summarize the IBS tiles into identity-by-descent (IBD) calls. The aim is to
keep a consistent impg-centered workflow so ongoing studies can mix VCF-derived
and haplotype-derived evidence without extra glue code.

## What we are doing
- define window tilings per chromosome and stream `impg` similarity over them;
- record IBS matches per window and aggregate them with Jacquard-style metrics;
- classify IBD segments with the Rust HMM and hand results to reporting
  notebooks.

Each component supports that flow and provides the material that shows up in the
research updates shared with the consortium.

## Quick workflow
1. `cd production/ibs-cli` and run `cargo build --release`.
2. Call `scripts/run_full.sh` (or `cargo run --bin ibs -- ...`) to tile a
   chromosome, execute impg across windows, and write per-window IBS tables.
3. Pass those tables to `cargo run --bin jacquard -- ...` or
   `scripts/jacquard_coeffs.sh`, then finish with `scripts/ibd.sh` when you need
   HMM-based IBD states.
4. Consume the output from notebooks under `analysis/ibd-network` to update the
   HPRC report decks in `docs/reports`.

Override defaults with environment variables such as `AGC`, `CHR`, or
`WINDOW_SIZE` so the same script set can operate on local copies of the HPRC
assemblies. Large AGC/PAF inputs are never checked into git—place them under
`data/` or point the CLI flags to an external location.

## Repository layout
- `production/ibs-cli/` – Rust CLI plus bash wrappers for impg jobs and Jacquard
  reductions. `cargo test --test parity` guards prototypes vs. Rust ports.
- `analysis/ibd-network/` – notebooks and small helpers that read the IBS/IBD
  tables and prepare figures for the consortium.
- `docs/reports/` – shipped report artifacts (PDFs, slide decks, etc.).
- `data/` – lightweight metadata kept in git; drop heavy reference data beside
  it locally.
Together these pieces cover how we run impg-based IBS window scans, connect them
to IBD calls, and study the results for the next HPRCv2 iterations.
