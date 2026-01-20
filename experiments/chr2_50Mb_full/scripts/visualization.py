#!/usr/bin/env python3
"""
Visualization Module for IBD Analysis

Creates publication-quality figures for IBD detection results:
- Chromosome-wide IBD tracks
- Posterior probability heatmaps
- Segment length distributions
- Population comparisons

Author: IBD-CLI Project
Date: 2026-01
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Try to import seaborn for nicer aesthetics
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ibd_inference import IBDResult, IBDSegment, Population


# ============================================================
# Style configuration
# ============================================================

def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    if HAS_SEABORN:
        sns.set_palette("colorblind")


# Custom colormap for posterior probabilities
IBD_CMAP = LinearSegmentedColormap.from_list(
    'ibd_posterior',
    [(0, '#FFFFFF'),      # White for P(IBD)=0
     (0.3, '#FFE0E0'),    # Light red
     (0.5, '#FF8080'),    # Medium red
     (0.7, '#FF4040'),    # Darker red
     (1, '#CC0000')],     # Dark red for P(IBD)=1
)


# ============================================================
# Single pair visualization
# ============================================================

def plot_ibd_track(
    result: IBDResult,
    output_path: Optional[Path] = None,
    show_identity: bool = True,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Plot IBD track for a single haplotype pair.

    Shows:
    - Top: IBS identity values along chromosome
    - Middle: Posterior P(IBD) probability
    - Bottom: Detected IBD segments

    Args:
        result: IBDResult from inference
        output_path: Path to save figure
        show_identity: Whether to show identity track
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    setup_style()

    n_panels = 3 if show_identity else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

    # Convert positions to Mb
    pos_mb = result.window_starts / 1e6

    panel_idx = 0

    # Panel 1: Identity values
    if show_identity:
        ax = axes[panel_idx]
        ax.scatter(pos_mb, result.identities, s=1, alpha=0.5, c='gray', rasterized=True)

        # Mark IBD segments
        for seg in result.segments:
            seg_pos = result.window_starts[seg.start_idx:seg.end_idx+1] / 1e6
            seg_id = result.identities[seg.start_idx:seg.end_idx+1]
            ax.scatter(seg_pos, seg_id, s=2, c='red', alpha=0.7, rasterized=True)

        ax.set_ylabel('Sequence\nIdentity')
        ax.set_ylim(0.99, 1.001)
        ax.axhline(y=0.9997, color='red', linestyle='--', alpha=0.5, label='IBD threshold')
        ax.axhline(y=0.999, color='blue', linestyle='--', alpha=0.5, label='Non-IBD mean')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f'IBS/IBD Analysis: {result.sample_a}#{result.hap_a} vs {result.sample_b}#{result.hap_b}',
                    fontsize=12, fontweight='bold')
        panel_idx += 1

    # Panel 2: Posterior probability
    ax = axes[panel_idx]

    # Create heatmap-style plot
    img = ax.scatter(pos_mb, np.ones(len(pos_mb)) * 0.5, c=result.posterior_ibd,
                     cmap=IBD_CMAP, s=3, vmin=0, vmax=1, rasterized=True)

    # Also plot as line
    ax.fill_between(pos_mb, 0, result.posterior_ibd, alpha=0.3, color='red')
    ax.plot(pos_mb, result.posterior_ibd, color='darkred', linewidth=0.5, alpha=0.7)

    ax.set_ylabel('P(IBD)')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('P(IBD)', fontsize=9)

    panel_idx += 1

    # Panel 3: IBD segments
    ax = axes[panel_idx]

    # Draw segments as rectangles
    for seg in result.segments:
        start = seg.start_bp / 1e6
        width = seg.length_bp / 1e6
        rect = mpatches.Rectangle(
            (start, 0.2), width, 0.6,
            facecolor='red', edgecolor='darkred', alpha=0.7,
            linewidth=1
        )
        ax.add_patch(rect)

        # Add length label for long segments
        if seg.length_bp > 100000:
            ax.text(start + width/2, 0.9, f'{seg.length_bp/1000:.0f}kb',
                   ha='center', va='bottom', fontsize=7)

    ax.set_xlim(pos_mb[0], pos_mb[-1])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel(f'Position on {result.chrom} (Mb)')
    ax.set_ylabel('IBD\nSegments')
    ax.set_yticks([])

    # Summary statistics
    summary_text = (
        f"Segments: {result.n_segments} | "
        f"Total IBD: {result.total_ibd_bp/1e6:.2f} Mb | "
        f"Fraction: {result.fraction_ibd:.1%}"
    )
    ax.text(0.5, -0.15, summary_text, transform=ax.transAxes,
           ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_segment_length_distribution(
    results: List[IBDResult],
    labels: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of IBD segment lengths.

    Args:
        results: List of IBDResults
        labels: Labels for each result
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if labels is None:
        labels = [f'Sample {i+1}' for i in range(len(results))]

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    # Panel 1: Length histogram
    ax = axes[0]
    for i, (result, label) in enumerate(zip(results, labels)):
        lengths = np.array([s.length_bp / 1000 for s in result.segments])
        if len(lengths) > 0:
            ax.hist(lengths, bins=30, alpha=0.6, label=label, color=colors[i])

    ax.set_xlabel('Segment Length (kb)')
    ax.set_ylabel('Count')
    ax.set_title('IBD Segment Length Distribution')
    ax.legend()

    # Panel 2: Length vs posterior
    ax = axes[1]
    for i, (result, label) in enumerate(zip(results, labels)):
        lengths = np.array([s.length_bp / 1000 for s in result.segments])
        posteriors = np.array([s.mean_posterior for s in result.segments])
        if len(lengths) > 0:
            ax.scatter(lengths, posteriors, alpha=0.6, label=label, color=colors[i], s=30)

    ax.set_xlabel('Segment Length (kb)')
    ax.set_ylabel('Mean P(IBD)')
    ax.set_title('Segment Length vs Posterior Probability')
    ax.legend()
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


# ============================================================
# Multi-pair comparison
# ============================================================

def plot_pairwise_heatmap(
    results: Dict[Tuple[str, str], IBDResult],
    samples: List[str],
    metric: str = 'fraction_ibd',
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot pairwise IBD sharing heatmap.

    Args:
        results: Dictionary mapping (sample_a, sample_b) to IBDResult
        samples: Ordered list of sample names
        metric: 'fraction_ibd', 'n_segments', or 'total_ibd_bp'
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    setup_style()

    n = len(samples)
    matrix = np.zeros((n, n))

    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i == j:
                matrix[i, j] = np.nan
            else:
                key = (s1, s2) if (s1, s2) in results else (s2, s1)
                if key in results:
                    result = results[key]
                    if metric == 'fraction_ibd':
                        matrix[i, j] = result.fraction_ibd * 100  # as percentage
                    elif metric == 'n_segments':
                        matrix[i, j] = result.n_segments
                    elif metric == 'total_ibd_bp':
                        matrix[i, j] = result.total_ibd_bp / 1e6  # in Mb

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(samples, rotation=45, ha='right')
    ax.set_yticklabels(samples)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if metric == 'fraction_ibd':
        cbar.set_label('IBD Sharing (%)')
    elif metric == 'n_segments':
        cbar.set_label('Number of Segments')
    elif metric == 'total_ibd_bp':
        cbar.set_label('Total IBD (Mb)')

    ax.set_title('Pairwise IBD Sharing', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_population_comparison(
    results_by_pop: Dict[str, List[IBDResult]],
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Plot IBD comparison across populations.

    Args:
        results_by_pop: Dictionary mapping population to list of IBDResults
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    populations = list(results_by_pop.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(populations)))

    # Panel 1: Fraction IBD by population
    ax = axes[0, 0]
    data = []
    for pop in populations:
        fractions = [r.fraction_ibd * 100 for r in results_by_pop[pop]]
        data.append(fractions)

    bp = ax.boxplot(data, labels=populations, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('IBD Sharing (%)')
    ax.set_title('IBD Sharing by Population')

    # Panel 2: Number of segments
    ax = axes[0, 1]
    data = []
    for pop in populations:
        n_segs = [r.n_segments for r in results_by_pop[pop]]
        data.append(n_segs)

    bp = ax.boxplot(data, labels=populations, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Number of IBD Segments')
    ax.set_title('IBD Segments by Population')

    # Panel 3: Mean segment length
    ax = axes[1, 0]
    data = []
    for pop in populations:
        mean_lens = [r.mean_segment_length / 1000 for r in results_by_pop[pop]
                    if r.n_segments > 0]
        if mean_lens:
            data.append(mean_lens)
        else:
            data.append([0])

    bp = ax.boxplot(data, labels=populations, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Mean Segment Length (kb)')
    ax.set_title('Segment Length by Population')

    # Panel 4: Segment length distributions
    ax = axes[1, 1]
    for i, pop in enumerate(populations):
        all_lengths = []
        for r in results_by_pop[pop]:
            all_lengths.extend([s.length_bp / 1000 for s in r.segments])
        if all_lengths:
            ax.hist(all_lengths, bins=30, alpha=0.5, label=pop, color=colors[i])

    ax.set_xlabel('Segment Length (kb)')
    ax.set_ylabel('Count')
    ax.set_title('Segment Length Distribution by Population')
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


# ============================================================
# Chromosome-wide visualization
# ============================================================

def plot_chromosome_ibd_summary(
    results: List[IBDResult],
    chrom: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """
    Create chromosome-wide summary of IBD across multiple pairs.

    Shows:
    - IBD density along chromosome
    - Individual pair tracks
    - Hotspot analysis

    Args:
        results: List of IBDResults for the chromosome
        chrom: Chromosome name
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig = plt.figure(figsize=figsize)

    # Get chromosome extent
    all_starts = np.concatenate([r.window_starts for r in results])
    all_ends = np.concatenate([r.window_ends for r in results])
    chrom_start = np.min(all_starts)
    chrom_end = np.max(all_ends)

    # Create common bin structure for density calculation
    bin_size = 100000  # 100kb bins
    n_bins = int((chrom_end - chrom_start) / bin_size) + 1
    bin_edges = np.linspace(chrom_start, chrom_end, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 1e6  # in Mb

    # Calculate IBD density per bin
    ibd_density = np.zeros(n_bins)
    for result in results:
        for seg in result.segments:
            # Find overlapping bins
            start_bin = int((seg.start_bp - chrom_start) / bin_size)
            end_bin = int((seg.end_bp - chrom_start) / bin_size)
            start_bin = max(0, min(start_bin, n_bins - 1))
            end_bin = max(0, min(end_bin, n_bins - 1))
            ibd_density[start_bin:end_bin+1] += 1

    # Normalize by number of pairs
    ibd_density /= len(results)

    # Layout
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 0.5], hspace=0.1)

    # Panel 1: IBD density
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(bin_centers, 0, ibd_density, alpha=0.6, color='red')
    ax1.plot(bin_centers, ibd_density, color='darkred', linewidth=0.5)
    ax1.set_ylabel('IBD\nDensity')
    ax1.set_xlim(bin_centers[0], bin_centers[-1])
    ax1.set_title(f'{chrom} IBD Summary ({len(results)} pairs)', fontsize=12, fontweight='bold')
    ax1.set_xticklabels([])

    # Panel 2: Individual tracks (show top N pairs by IBD)
    ax2 = fig.add_subplot(gs[1])

    # Sort results by total IBD
    sorted_results = sorted(results, key=lambda r: r.total_ibd_bp, reverse=True)
    max_tracks = min(20, len(sorted_results))

    for i, result in enumerate(sorted_results[:max_tracks]):
        y = max_tracks - i - 1  # Plot from top

        # Draw segments
        for seg in result.segments:
            start = seg.start_bp / 1e6
            width = seg.length_bp / 1e6
            rect = mpatches.Rectangle(
                (start, y + 0.1), width, 0.8,
                facecolor='red', edgecolor='none', alpha=0.7
            )
            ax2.add_patch(rect)

        # Add label
        label = f"{result.sample_a[:6]}-{result.sample_b[:6]}"
        ax2.text(-1, y + 0.5, label, ha='right', va='center', fontsize=7)

    ax2.set_xlim(bin_centers[0], bin_centers[-1])
    ax2.set_ylim(0, max_tracks)
    ax2.set_ylabel('Sample Pairs\n(sorted by total IBD)')
    ax2.set_yticks([])
    ax2.set_xticklabels([])

    # Panel 3: Position axis
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(bin_centers[0], bin_centers[-1])
    ax3.set_xlabel(f'Position on {chrom} (Mb)')
    ax3.set_yticks([])
    ax3.spines['left'].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


# ============================================================
# Test
# ============================================================

def test_visualization():
    """Test visualization with synthetic data."""
    from ibd_inference import test_inference

    print("Generating test data...")
    result = test_inference()

    print("\nCreating visualizations...")

    # Create output directory
    output_dir = Path('test_figures')
    output_dir.mkdir(exist_ok=True)

    # Plot single track
    fig1 = plot_ibd_track(result, output_dir / 'test_ibd_track.png')

    # Create multiple results for comparison
    results = [result]

    fig2 = plot_segment_length_distribution(
        results, ['Test'],
        output_dir / 'test_segment_lengths.png'
    )

    print(f"\nFigures saved to: {output_dir}/")

    plt.close('all')


if __name__ == '__main__':
    test_visualization()
