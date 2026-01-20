#!/usr/bin/env python3
"""Generate BED file with windows for chr2 analysis."""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=50000000)
    parser.add_argument('--window', type=int, default=5000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--chrom', type=str, default='CHM13#0#chr2')

    args = parser.parse_args()

    with open(args.output, 'w') as f:
        pos = args.start
        while pos + args.window <= args.end:
            f.write(f"{args.chrom}\t{pos}\t{pos + args.window}\n")
            pos += args.window

    n_windows = (args.end - args.start) // args.window
    print(f"Generated {n_windows} windows in {args.output}")

if __name__ == '__main__':
    main()
