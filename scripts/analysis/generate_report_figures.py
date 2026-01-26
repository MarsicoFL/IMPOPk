#!/usr/bin/env python3
"""
Generate figures for HPRCv2-IBD Progress Report.

This script generates publication-quality figures for the progress report,
integrating data from both chromosome 1 and chromosome 2 analyses.

Usage:
    python generate_report_figures.py [--output-dir DIR]

Figures generated:
    fig1_emission_distributions.png  - Emission parameter comparison (chr1)
    fig2_chr2_population_comparison.png - Multi-population IBD comparison (chr2)
    fig3_chr1_eur_landscape.png      - EUR IBD landscape (chr1)
    fig4_selection_heatmap.png       - Selection scan results
    fig5_segment_characteristics.png - Segment characteristics (chr2 all pops)
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'AFR': '#E64B35',
    'EUR': '#4DBBD5',
    'EAS': '#00A087',
    'CSA': '#3C5488',
    'AMR': '#F39B7F',
    'IBD': '#2E7D32',
}

# Chromosome 1 constants
CHR1_LENGTH = 248956422
CENTROMERE_START = 121500000
CENTROMERE_END = 142200000


def get_project_root():
    """Find project root by looking for Cargo.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'Cargo.toml').exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def load_all_data(project_root):
    """Load all required data files."""
    data = {}

    # Chr1 data (filtered v2)
    chr1_dir = project_root / 'experiments/chr1_full/results/json'
    if chr1_dir.exists():
        for f in ['EUR_summary_v2.json', 'AFR_summary_v2.json', 'EUR_ibd_results.json']:
            path = chr1_dir / f
            if path.exists():
                key = 'chr1_' + f.replace('.json', '').lower()
                data[key] = json.loads(path.read_text())

    # Chr2 filtered data (valid for all populations)
    chr2_dir = project_root / 'experiments/chr2_50Mb_filtered/results/json'
    if chr2_dir.exists():
        for pop in ['AFR', 'EUR', 'EAS']:
            path = chr2_dir / f'{pop}_chr2_full_results.json'
            if path.exists():
                data[f'chr2_{pop.lower()}'] = json.loads(path.read_text())

    # Selection scan
    selection_file = project_root / 'experiments/selection_scan/analysis/figures/expanded_statistics.json'
    if selection_file.exists():
        data['selection'] = json.loads(selection_file.read_text())

    return data


def extract_segments(results):
    """Extract segment data from IBD results."""
    segments = []
    for pair in results.get('results', []):
        for seg in pair.get('segments', []):
            segments.append({
                'start': seg['start_bp'],
                'end': seg['end_bp'],
                'length': seg['length_bp'],
                'identity': seg['mean_identity'],
                'posterior': seg['mean_posterior'],
            })
    return segments


def fig1_emission_distributions(data, output_dir):
    """
    Figure 1: Emission distributions for EUR and AFR (chr1 v2 filtered).
    """
    if 'chr1_eur_summary_v2' not in data or 'chr1_afr_summary_v2' not in data:
        print("  Skipping fig1: missing chr1 v2 data")
        return

    eur = data['chr1_eur_summary_v2']['emission_params']
    afr = data['chr1_afr_summary_v2']['emission_params']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.linspace(0.994, 1.0005, 1000)

    for ax, params, pop, color in [
        (axes[0], eur, 'EUR', COLORS['EUR']),
        (axes[1], afr, 'AFR', COLORS['AFR'])
    ]:
        y_non_ibd = stats.norm.pdf(x, params['non_ibd']['mean'], params['non_ibd']['std'])
        y_ibd = stats.norm.pdf(x, params['ibd']['mean'], params['ibd']['std'])

        ax.fill_between(x, y_non_ibd, alpha=0.5, color=color, label='non-IBD')
        ax.fill_between(x, y_ibd, alpha=0.5, color=COLORS['IBD'], label='IBD')
        ax.plot(x, y_non_ibd, color=color, linewidth=1)
        ax.plot(x, y_ibd, color=COLORS['IBD'], linewidth=1)

        ax.set_xlabel('Sequence identity')
        ax.set_ylabel('Density')
        ax.set_title(f"{pop} ($d' = {params['d_prime']:.2f}$)", fontweight='bold')
        ax.legend(loc='upper left', frameon=False)
        ax.set_xlim(0.994, 1.0005)

    fig.suptitle('Chromosome 1: Emission Distributions (v2 filtered)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_emission_distributions.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_emission_distributions.pdf', bbox_inches='tight')
    plt.close()
    print("  Generated: fig1_emission_distributions")


def fig2_chr2_population_comparison(data, output_dir):
    """
    Figure 2: Multi-population IBD comparison from chr2 (the main result).
    """
    pops = ['AFR', 'EUR', 'EAS']
    pop_data = {}

    for pop in pops:
        key = f'chr2_{pop.lower()}'
        if key not in data:
            print(f"  Skipping fig2: missing {key}")
            return
        pop_data[pop] = data[key]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel A: Mean IBD per pair
    ax = axes[0, 0]
    mean_ibd = []
    for pop in pops:
        results = pop_data[pop]['results']
        total = sum(p['total_ibd_bp'] for p in results) / 1e6
        mean_ibd.append(total / len(results))

    bars = ax.bar(pops, mean_ibd, color=[COLORS[p] for p in pops])
    ax.set_ylabel('Mean IBD per pair (Mb)')
    ax.set_title('A) IBD sharing by population', loc='left', fontweight='bold')

    for bar, val in zip(bars, mean_ibd):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{val:.1f}', ha='center', fontsize=9)

    # Add ratio annotations
    afr_val = mean_ibd[0]
    ax.text(1, mean_ibd[1] + 1.5, f'{mean_ibd[1]/afr_val:.1f}x AFR', ha='center', fontsize=8)
    ax.text(2, mean_ibd[2] + 1.5, f'{mean_ibd[2]/afr_val:.1f}x AFR', ha='center', fontsize=8)

    # Panel B: Segment count
    ax = axes[0, 1]
    n_segs = [len(extract_segments(pop_data[pop])) for pop in pops]
    bars = ax.bar(pops, n_segs, color=[COLORS[p] for p in pops])
    ax.set_ylabel('Number of segments')
    ax.set_title('B) Segment count', loc='left', fontweight='bold')

    for bar, val in zip(bars, n_segs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
               f'{val:,}', ha='center', fontsize=9)

    # Panel C: Identity distributions
    ax = axes[1, 0]
    for pop in pops:
        segs = extract_segments(pop_data[pop])
        identities = [s['identity'] for s in segs]
        ax.hist(identities, bins=30, alpha=0.5, color=COLORS[pop],
               label=f'{pop} (n={len(identities)})', density=True)

    ax.axvline(0.9999, color='black', linestyle='--', linewidth=1, label='Expected IBD')
    ax.set_xlabel('Segment identity')
    ax.set_ylabel('Density')
    ax.set_title('C) Segment identity distributions', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.set_xlim(0.9993, 1.0001)

    # Panel D: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Population', 'Pairs', 'Segments', 'Mean IBD (Mb)', 'Mean Identity', 'vs AFR'],
    ]
    for i, pop in enumerate(pops):
        segs = extract_segments(pop_data[pop])
        ids = [s['identity'] for s in segs]
        ratio = f'{mean_ibd[i]/afr_val:.2f}x' if i > 0 else '1.00x'
        table_data.append([
            pop,
            str(pop_data[pop]['n_pairs']),
            f'{len(segs):,}',
            f'{mean_ibd[i]:.2f}',
            f'{np.mean(ids):.6f}',
            ratio
        ])

    colors = [['#f0f0f0']*6]
    for pop in pops:
        colors.append([COLORS[pop] + '40']*6)

    table = ax.table(cellText=table_data, cellColours=colors,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax.set_title('D) Summary statistics', loc='left', fontweight='bold', y=0.95)

    fig.suptitle('Chromosome 2 (50 Mb): Valid Multi-Population IBD Comparison',
                fontweight='bold', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_chr2_population_comparison.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_chr2_population_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("  Generated: fig2_chr2_population_comparison")


def fig3_chr1_eur_landscape(data, output_dir):
    """
    Figure 3: EUR IBD landscape across chromosome 1.
    """
    if 'chr1_eur_ibd_results' not in data:
        print("  Skipping fig3: missing chr1 EUR results")
        return

    segments = extract_segments(data['chr1_eur_ibd_results'])
    if not segments:
        print("  Skipping fig3: no EUR segments")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel A: Segment positions
    ax = axes[0]
    for seg in segments:
        mid = (seg['start'] + seg['end']) / 2 / 1e6
        ax.axvline(mid, color=COLORS['EUR'], alpha=0.4, linewidth=1.5)
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_ylabel('Segment\nlocations')
    ax.set_yticks([])
    ax.set_title('A) EUR IBD segment positions', loc='left', fontweight='bold')
    ax.text(132, 0.5, 'CEN', fontsize=8, ha='center', va='center', color='gray')

    # Panel B: Segment density histogram
    ax = axes[1]
    positions = [(s['start'] + s['end']) / 2 / 1e6 for s in segments]
    ax.hist(positions, bins=50, color=COLORS['EUR'], alpha=0.7, edgecolor='white', range=(0, 250))
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_ylabel('Segment count')
    ax.set_title('B) Spatial distribution', loc='left', fontweight='bold')

    # Panel C: Identity by position
    ax = axes[2]
    identities = [s['identity'] for s in segments]
    ax.scatter(positions, identities, c=COLORS['EUR'], alpha=0.6, s=25, edgecolors='none')
    ax.axhline(np.mean(identities), color='red', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(identities):.6f}')
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Segment identity')
    ax.set_ylim(0.9995, 1.0001)
    ax.set_title('C) Segment identity (all valid: 99.95-99.99%)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(0, CHR1_LENGTH / 1e6)

    fig.suptitle('Chromosome 1: EUR IBD Landscape (100 pairs)', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_chr1_eur_landscape.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_chr1_eur_landscape.pdf', bbox_inches='tight')
    plt.close()
    print("  Generated: fig3_chr1_eur_landscape")


def fig4_selection_heatmap(data, output_dir):
    """
    Figure 4: Selection scan IBS rate heatmap.
    """
    if 'selection' not in data:
        print("  Skipping fig4: missing selection data")
        return

    selection = data['selection']
    loci = ['LCT', 'SLC24A5', 'EDAR', 'HBB', 'DARC']
    populations = ['AFR', 'EUR', 'EAS', 'CSA', 'AMR']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Heatmap
    ax = axes[0]
    ibs_matrix = np.array([[selection['ibs_rates'][pop][locus]
                           for pop in populations] for locus in loci])

    im = ax.imshow(ibs_matrix, cmap='YlOrRd', aspect='auto', vmin=0.05, vmax=0.30)

    ax.set_xticks(range(len(populations)))
    ax.set_xticklabels(populations)
    ax.set_yticks(range(len(loci)))
    ax.set_yticklabels(loci)

    # Add values
    for i in range(len(loci)):
        for j in range(len(populations)):
            val = ibs_matrix[i, j]
            color = 'white' if val > 0.2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=8)

    # Mark expected targets
    targets = {'LCT': 'EUR', 'SLC24A5': 'EUR', 'EDAR': 'EAS', 'HBB': 'AFR', 'DARC': 'AFR'}
    for locus, target in targets.items():
        i = loci.index(locus)
        j = populations.index(target)
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor='blue', linewidth=2))

    ax.set_title('A) IBS rates (blue = expected target)', loc='left', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='IBS rate')

    # Panel B: Fold enrichment
    ax = axes[1]
    x = np.arange(len(loci))
    width = 0.18

    for idx, pop in enumerate(['EUR', 'EAS', 'CSA', 'AMR']):
        enrichments = []
        for locus in loci:
            afr_rate = selection['ibs_rates']['AFR'][locus]
            pop_rate = selection['ibs_rates'][pop][locus]
            enrichments.append(pop_rate / afr_rate if afr_rate > 0 else 0)

        bars = ax.bar(x + idx*width - 1.5*width, enrichments, width,
                     label=pop, color=COLORS[pop])

        # Mark targets
        for i, locus in enumerate(loci):
            if targets[locus] == pop:
                ax.scatter([x[i] + idx*width - 1.5*width], [enrichments[i] + 0.08],
                          marker='*', color='black', s=80, zorder=5)

    ax.axhline(1, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(loci)
    ax.set_ylabel('Fold enrichment vs AFR')
    ax.set_title('B) Relative enrichment (* = expected target)', loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_selection_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_selection_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  Generated: fig4_selection_heatmap")


def fig5_segment_characteristics(data, output_dir):
    """
    Figure 5: Segment characteristics from chr2 (all populations).
    """
    pops = ['AFR', 'EUR', 'EAS']
    all_segs = {}

    for pop in pops:
        key = f'chr2_{pop.lower()}'
        if key in data:
            all_segs[pop] = extract_segments(data[key])

    if len(all_segs) < 3:
        print("  Skipping fig5: insufficient chr2 data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel A: Length distributions
    ax = axes[0, 0]
    for pop in pops:
        lengths = [s['length'] / 1e3 for s in all_segs[pop]]  # kb
        ax.hist(lengths, bins=30, alpha=0.5, color=COLORS[pop],
               label=f'{pop} (median: {np.median(lengths):.0f} kb)', density=True)
    ax.set_xlabel('Segment length (kb)')
    ax.set_ylabel('Density')
    ax.set_title('A) Segment length distributions', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Length comparison boxplot
    ax = axes[0, 1]
    length_data = [[s['length']/1e3 for s in all_segs[pop]] for pop in pops]
    bp = ax.boxplot(length_data, labels=pops, patch_artist=True)
    for patch, pop in zip(bp['boxes'], pops):
        patch.set_facecolor(COLORS[pop])
        patch.set_alpha(0.6)
    ax.set_ylabel('Segment length (kb)')
    ax.set_title('B) Length comparison', loc='left', fontweight='bold')

    # Panel C: Posterior distributions
    ax = axes[1, 0]
    for pop in pops:
        posteriors = [s['posterior'] for s in all_segs[pop]]
        ax.hist(posteriors, bins=30, alpha=0.5, color=COLORS[pop],
               label=f'{pop}', density=True)
    ax.axvline(0.95, color='green', linestyle='--', linewidth=1, label='95% threshold')
    ax.set_xlabel('Mean posterior probability')
    ax.set_ylabel('Density')
    ax.set_title('C) Posterior probability distributions', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=8)

    # Panel D: Summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Chromosome 2 Segment Summary\n" + "=" * 40 + "\n\n"
    for pop in pops:
        segs = all_segs[pop]
        lengths = [s['length']/1e3 for s in segs]
        ids = [s['identity'] for s in segs]
        posts = [s['posterior'] for s in segs]
        pct_high = 100 * sum(1 for p in posts if p > 0.95) / len(posts)

        summary_text += f"{pop}:\n"
        summary_text += f"  Segments: {len(segs):,}\n"
        summary_text += f"  Median length: {np.median(lengths):.0f} kb\n"
        summary_text += f"  Mean identity: {np.mean(ids):.6f}\n"
        summary_text += f"  Posterior > 0.95: {pct_high:.1f}%\n\n"

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax.set_title('D) Summary statistics', loc='left', fontweight='bold', y=0.95)

    fig.suptitle('Chromosome 2: Segment Characteristics (All Populations Valid)',
                fontweight='bold', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_segment_characteristics.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_segment_characteristics.pdf', bbox_inches='tight')
    plt.close()
    print("  Generated: fig5_segment_characteristics")


def main():
    parser = argparse.ArgumentParser(description='Generate report figures')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory (default: reports/main/figures)')
    args = parser.parse_args()

    project_root = get_project_root()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = project_root / 'reports/main/figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print()

    print("Loading data...")
    data = load_all_data(project_root)
    print(f"  Loaded: {list(data.keys())}")
    print()

    print("Generating figures...")
    fig1_emission_distributions(data, output_dir)
    fig2_chr2_population_comparison(data, output_dir)
    fig3_chr1_eur_landscape(data, output_dir)
    fig4_selection_heatmap(data, output_dir)
    fig5_segment_characteristics(data, output_dir)

    print()
    print(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
