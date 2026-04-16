#!/usr/bin/env python3
"""Generate a tiny synthetic IBS TSV with known IBD structure, for the
ibd/ example. Two individuals (SIM1, SIM2, SIM3) are simulated on a
5 Mb region with:

  - SIM1#1 × SIM2#1 share a 2.5 Mb IBD segment (1.25-3.75 Mb)
  - SIM1#2 × SIM3#1 share a 1.5 Mb IBD segment (0.5-2.0 Mb)
  - All other pairs are unrelated

Background identity: ~0.998 (typical HPRC pairwise divergence).
IBD windows: 0.99995 (near-perfect identity over the segment).
"""
from __future__ import annotations
import random
from pathlib import Path

random.seed(42)

REGION_BP = 5_000_000
WINDOW_BP = 10_000
N_WINDOWS = REGION_BP // WINDOW_BP  # 500
CHROM = "CHM13#0#chr12"
HAPS = ["SIM1#1", "SIM1#2", "SIM2#1", "SIM2#2", "SIM3#1", "SIM3#2"]

# IBD truth: list of (hap_a, hap_b, start_win, end_win)
IBD_TRUTH = [
    ("SIM1#1", "SIM2#1", 125, 375),   # 1.25-3.75 Mb
    ("SIM1#2", "SIM3#1",  50, 200),   # 0.50-2.00 Mb
]


def is_ibd(a: str, b: str, w: int) -> bool:
    pair = tuple(sorted([a, b]))
    for ta, tb, ws, we in IBD_TRUTH:
        if tuple(sorted([ta, tb])) == pair and ws <= w < we:
            return True
    return False


def make_row(chrom: str, win: int, a: str, b: str) -> str:
    start = win * WINDOW_BP
    end = start + WINDOW_BP
    if is_ibd(a, b, win):
        identity = random.gauss(0.99995, 0.00005)
    else:
        identity = random.gauss(0.998, 0.0003)
    identity = min(0.999999, max(0.990, identity))
    return f"{chrom}\t{start}\t{end}\t{a}\t{b}\t{identity:.6f}\n"


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n")
        for w in range(N_WINDOWS):
            for i in range(len(HAPS)):
                for j in range(i + 1, len(HAPS)):
                    f.write(make_row(CHROM, w, HAPS[i], HAPS[j]))
    print(f"Wrote {out_path} with {N_WINDOWS} windows × {len(HAPS)*(len(HAPS)-1)//2} pairs")
    print("Ground truth IBD segments:")
    for ta, tb, ws, we in IBD_TRUTH:
        print(f"  {ta} × {tb}: {ws*WINDOW_BP:,}-{we*WINDOW_BP:,} bp "
              f"({(we-ws)*WINDOW_BP/1e6:.2f} Mb)")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ibs_synthetic.tsv")
    main(out)
