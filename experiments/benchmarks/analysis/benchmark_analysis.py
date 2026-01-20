#!/usr/bin/env python3
"""
Benchmark Analysis for IBS Scalability

Analyzes runtime and output scaling:
- Haplotype scaling: O(n²) pairs
- Window scaling: linear with windows

Usage:
    python benchmark_analysis.py [--hap-dir PATH] [--win-dir PATH] [--output-dir PATH]
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_HAP_DIR = Path(__file__).parent.parent / "haplotype_scaling" / "results"
DEFAULT_WIN_DIR = Path(__file__).parent.parent / "window_scaling" / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def load_haplotype_metrics(hap_dir: Path) -> pd.DataFrame:
    """Load haplotype scaling benchmark metrics."""
    metrics_file = hap_dir / "benchmark_metrics.tsv"
    if not metrics_file.exists():
        print(f"  Warning: {metrics_file} not found")
        return pd.DataFrame()
    return pd.read_csv(metrics_file, sep='\t')


def load_window_metrics(win_dir: Path) -> pd.DataFrame:
    """Load window scaling benchmark metrics."""
    metrics_file = win_dir / "benchmark_metrics.tsv"
    if not metrics_file.exists():
        print(f"  Warning: {metrics_file} not found")
        return pd.DataFrame()
    return pd.read_csv(metrics_file, sep='\t')


def plot_haplotype_scaling(df: pd.DataFrame, output_path: Path) -> None:
    """Plot haplotype scaling analysis."""
    if df.empty:
        print("  No haplotype data to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Runtime vs Haplotypes
    ax = axes[0]
    ax.scatter(df['haplotypes'], df['runtime_seconds'], s=80, c='#3498db', edgecolor='black')
    ax.set_xlabel('Haplotypes (n)', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('A. Runtime Scaling', fontweight='bold', loc='left')

    # Fit quadratic: runtime ~ a*n² (due to pairwise comparisons)
    if len(df) >= 3:
        x = df['haplotypes'].values
        y = df['runtime_seconds'].values
        x_fit = np.linspace(min(x), max(x), 100)

        # Linear fit for runtime (impg is optimized)
        slope, intercept, r, p, se = stats.linregress(x, y)
        ax.plot(x_fit, slope * x_fit + intercept, 'r--', alpha=0.7,
                label=f'Linear: R²={r**2:.3f}')
        ax.legend(fontsize=9)

    # Panel B: Runtime vs Pairs
    ax = axes[1]
    ax.scatter(df['pairs'], df['runtime_seconds'], s=80, c='#2ecc71', edgecolor='black')
    ax.set_xlabel('Pairwise Comparisons', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('B. Runtime vs Pairs', fontweight='bold', loc='left')

    if len(df) >= 3:
        x = df['pairs'].values
        y = df['runtime_seconds'].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r--', alpha=0.7,
                label=f'Linear: R²={r**2:.3f}')
        ax.legend(fontsize=9)

    # Panel C: Output vs Pairs
    ax = axes[2]
    ax.scatter(df['pairs'], df['output_records'], s=80, c='#9b59b6', edgecolor='black')
    ax.set_xlabel('Pairwise Comparisons', fontsize=11)
    ax.set_ylabel('Output Records', fontsize=11)
    ax.set_title('C. Output Scaling', fontweight='bold', loc='left')

    if len(df) >= 3:
        x = df['pairs'].values
        y = df['output_records'].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r--', alpha=0.7,
                label=f'Linear: R²={r**2:.3f}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def plot_window_scaling(df: pd.DataFrame, output_path: Path) -> None:
    """Plot window scaling analysis."""
    if df.empty:
        print("  No window data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Runtime vs Windows
    ax = axes[0]
    ax.scatter(df['windows'], df['runtime_seconds'], s=80, c='#e74c3c', edgecolor='black')
    ax.set_xlabel('Number of Windows', fontsize=11)
    ax.set_ylabel('Runtime (seconds)', fontsize=11)
    ax.set_title('A. Runtime vs Windows', fontweight='bold', loc='left')

    if len(df) >= 3:
        x = df['windows'].values
        y = df['runtime_seconds'].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'b--', alpha=0.7,
                label=f'Linear: R²={r**2:.3f}')
        ax.legend(fontsize=9)

    # Panel B: Time per Window
    ax = axes[1]
    df_sorted = df.sort_values('window_size')
    time_per_window = df_sorted['runtime_seconds'] / df_sorted['windows']
    ax.bar(range(len(df_sorted)), time_per_window, color='#f39c12', edgecolor='black')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([f"{int(w/1000)}kb" for w in df_sorted['window_size']])
    ax.set_xlabel('Window Size', fontsize=11)
    ax.set_ylabel('Time per Window (sec)', fontsize=11)
    ax.set_title('B. Time per Window', fontweight='bold', loc='left')

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Benchmark Analysis for IBS Scalability')
    parser.add_argument('--hap-dir', type=Path, default=DEFAULT_HAP_DIR,
                        help='Haplotype scaling results directory')
    parser.add_argument('--win-dir', type=Path, default=DEFAULT_WIN_DIR,
                        help='Window scaling results directory')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for figures')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IBS Benchmark Analysis")
    print("=" * 60)

    # Haplotype scaling
    print("\n[1/2] Analyzing haplotype scaling...")
    hap_df = load_haplotype_metrics(args.hap_dir)
    if not hap_df.empty:
        print(f"  Loaded {len(hap_df)} data points")
        print(hap_df.to_string(index=False))

        hap_fig = args.output_dir / "haplotype_scaling"
        plot_haplotype_scaling(hap_df, hap_fig)
        print(f"  Figure: {hap_fig}.png/.pdf")

    # Window scaling
    print("\n[2/2] Analyzing window scaling...")
    win_df = load_window_metrics(args.win_dir)
    if not win_df.empty:
        print(f"  Loaded {len(win_df)} data points")
        print(win_df.to_string(index=False))

        win_fig = args.output_dir / "window_scaling"
        plot_window_scaling(win_df, win_fig)
        print(f"  Figure: {win_fig}.png/.pdf")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
