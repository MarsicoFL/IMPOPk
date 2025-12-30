# Haplotype-Based IBS/IBD Analysis for HPRCv2

Public HPRC assemblies plus impg-based similarity provide the raw material for
exploring Identity-By-State (IBS) and Identity-By-Descent (IBD) connections
between haplotypes. The repository separates production tooling, exploratory
analyses, and published reports so that each area can evolve independently.

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

## Analysis and reporting assets
- Notebooks plus helper scripts live under `analysis/ibd-network`. They consume
  the per-window IBS tables generated above and produced the deliverables stored
  in `docs/reports/` (e.g. `HPRCv2_IBD.pdf`).
- Keep heavyweight raw data out of git; store them under `data/` or supply
  explicit paths when executing the scripts.

Each area in this layout focuses on a single audience: production code with
tests, repeatable analysis notebooks, and published reports for broader review.
