#!/usr/bin/env python3
"""Generate a synthetic IBS TSV that mimics a pedigree founder-painting
scenario for the pedigree/ example.

Four founders (GP1-GP4) and one descendant (CHILD) over a 5 Mb region.
The descendant is a 50:50 mosaic: its genome is paired 1:1 with GP1 on
[0, 1.25 Mb), GP2 on [1.25, 2.5 Mb), GP3 on [2.5, 3.75 Mb), GP4 on
[3.75, 5 Mb). Within an assigned block the descendant shares ~0.99995
identity with the matching founder and ~0.998 with the other three.
"""
from __future__ import annotations
import random
from pathlib import Path

random.seed(42)

REGION_BP = 5_000_000
WINDOW_BP = 10_000
N_WINDOWS = REGION_BP // WINDOW_BP
CHROM = "CHM13#0#chr12"
FOUNDERS = ["GP1", "GP2", "GP3", "GP4"]
CHILD = "CHILD"

# Which founder "owns" each window (pedigree ground truth)
BLOCK_SIZE = N_WINDOWS // 4  # 125 windows per founder, 1.25 Mb each


def founder_of_window(w: int) -> str:
    return FOUNDERS[min(len(FOUNDERS) - 1, w // BLOCK_SIZE)]


def identity_between(a: str, b: str, w: int) -> float:
    if a == b:
        return random.gauss(0.99999, 0.00001)
    # CHILD is IBD with one founder per window
    if CHILD in (a, b):
        other = a if b == CHILD else b
        owner = founder_of_window(w)
        if other == owner:
            return random.gauss(0.99995, 0.00005)
    # Founders among themselves are unrelated
    return random.gauss(0.998, 0.0003)


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    haps = FOUNDERS + [CHILD]
    with out_path.open("w") as f:
        f.write("chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n")
        for w in range(N_WINDOWS):
            start = w * WINDOW_BP
            end = start + WINDOW_BP
            for i in range(len(haps)):
                for j in range(i + 1, len(haps)):
                    a, b = haps[i], haps[j]
                    identity = identity_between(a, b, w)
                    identity = min(0.999999, max(0.990, identity))
                    f.write(f"{CHROM}\t{start}\t{end}\t{a}\t{b}\t{identity:.6f}\n")
    print(f"Wrote {out_path}")
    print("Ground-truth founder blocks:")
    for i, gp in enumerate(FOUNDERS):
        s = i * BLOCK_SIZE * WINDOW_BP
        e = (i + 1) * BLOCK_SIZE * WINDOW_BP
        print(f"  {s:>9,} - {e:>9,} bp  →  {gp}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ibs_pedigree.tsv")
    main(out)
