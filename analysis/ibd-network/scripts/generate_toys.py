#!/usr/bin/env python3
import csv, argparse

HEADER = ["REGION","CHR","START","END","LENGTH","group.a","group.b","estimated.identity"]

def write_row(w, chr, start, end, a, b, ident):
    w.writerow([f"CHM13#0#{chr}:{start}-{end}", chr, start, end, end-start, a, b, f"{ident:.4f}"])

def toy1(path):
    win = 5000
    w = csv.writer(open(path,"w",newline=""), delimiter="\t")
    w.writerow(HEADER)
    for i in range(10):
        s = i*win; e = s+win
        ident = 0.9950
        if 1 <= i <= 6:  # windows 1..6 inclusive => 5k..35k
            ident = 0.9998 if i != 4 else 0.9997
        write_row(w, "chr1", s, e, "A", "B", ident)
    for i in range(0,7):
        s = i*win; e = s+win
        write_row(w, "chr1", s, e, "A", "C", 0.9950)

def toy2(path):
    win = 5000
    w = csv.writer(open(path,"w",newline=""), delimiter="\t")
    w.writerow(HEADER)
    vals = [0.9970,0.9990,0.9960,0.9970,0.9960,0.9970,0.9960,0.9970,0.9960,0.9970]
    for i,ident in enumerate(vals):
        s=i*win; e=s+win
        write_row(w, "chr1", s, e, "A", "B", ident)

def toy3(path):
    win = 5000
    w = csv.writer(open(path,"w",newline=""), delimiter="\t")
    w.writerow(HEADER)
    write_row(w, "chr1", 0, 5000, "A", "B", 0.9950)
    write_row(w, "chr1", 5000, 10000, "A", "B", 0.9997)
    write_row(w, "chr1", 10000, 15000, "A", "B", 0.9998)
    write_row(w, "chr1", 25000, 30000, "A", "B", 0.9998)
    write_row(w, "chr1", 30000, 35000, "A", "B", 0.9998)
    write_row(w, "chr1", 35000, 40000, "A", "B", 0.9950)

def main():
    ap = argparse.ArgumentParser(description="Generate toy pairwise per-window identities")
    ap.add_argument("--which", choices=["toy1","toy2","toy3","all"], default="all")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    if args.which in ("toy1","all"): toy1(f"{args.outdir}/toy1.pairwise.tsv")
    if args.which in ("toy2","all"): toy2(f"{args.outdir}/toy2.pairwise.tsv")
    if args.which in ("toy3","all"): toy3(f"{args.outdir}/toy3.pairwise.tsv")
    print("Done. Files:", ", ".join([f"{args.outdir}/toy{i}.pairwise.tsv" for i in (1,2,3)]))

if __name__ == "__main__":
    main()
