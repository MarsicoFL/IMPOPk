#!/usr/bin/env python3
"""
Publication figures for 2Mb minimum IBD segment analysis - FULL DATASET.
"""

import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

COLORS = {
    'EUR': '#0072B2',
    'AFR': '#D55E00',
    'EAS': '#009E73',
    'IBD': '#CC79A7',
    'non_IBD': '#56B4E9',
    'gray': '#999999',
    'dark': '#333333',
    'light_gray': '#E5E5E5',
}

RESULTS_DIR = Path(__file__).parent.parent / "results_2Mb"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
    })


def load_full_results(pop):
    path = RESULTS_DIR / 'json' / f'{pop}_2mb_full_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def create_main_figure():
    """Main figure for 2Mb full analysis."""
    setup_style()

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    populations = ['AFR', 'EUR']
    all_data = {p: load_full_results(p) for p in populations}

    # ================================================================
    # Panel A: Number of segments by population
    # ================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    pops_with_data = [p for p in populations if all_data[p]]
    total_segments = [all_data[p].get('total_segments', 0) for p in pops_with_data]

    bars = ax_a.bar(pops_with_data, total_segments,
                    color=[COLORS[p] for p in pops_with_data], edgecolor='none', alpha=0.8)

    for bar, val in zip(bars, total_segments):
        ax_a.text(bar.get_x() + bar.get_width()/2, val + max(total_segments)*0.03,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')

    ax_a.set_ylabel('Number of segments >= 2 Mb')
    ax_a.set_title('A. Total IBD segments detected (>= 2 Mb)', loc='left', fontweight='bold')

    # EUR/AFR ratio annotation
    if len(total_segments) == 2 and total_segments[0] > 0:
        ratio = total_segments[1] / total_segments[0]
        ax_a.text(0.5, max(total_segments) * 0.7, f'EUR/AFR = {ratio:.1f}x',
                 ha='center', fontsize=11, style='italic', fontweight='bold',
                 transform=ax_a.get_xaxis_transform())

    # ================================================================
    # Panel B: Segment length distribution
    # ================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    for pop in pops_with_data:
        results = all_data[pop].get('results', [])
        lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]
        if lengths:
            bins = np.linspace(2, min(12, max(lengths) + 0.5), 30)
            ax_b.hist(lengths, bins=bins, alpha=0.6, label=f'{pop} (n={len(lengths):,})',
                     color=COLORS[pop], edgecolor='white', linewidth=0.5)

    ax_b.axvline(2, color=COLORS['gray'], linestyle='--', linewidth=1.5)
    ax_b.text(2.15, ax_b.get_ylim()[1]*0.92, 'Min 2 Mb', fontsize=8, color=COLORS['gray'])

    ax_b.set_xlabel('Segment length (Mb)')
    ax_b.set_ylabel('Count')
    ax_b.legend(loc='upper right')
    ax_b.set_title('B. Distribution of IBD segment lengths', loc='left', fontweight='bold')

    # ================================================================
    # Panel C: Pairs with IBD
    # ================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    # Calculate totals (approximate from number of unique windows)
    total_pairs = {'AFR': 1953, 'EUR': 1830}
    n_with_ibd = [all_data[p].get('n_pairs_with_segments', 0) for p in pops_with_data]
    n_total = [total_pairs[p] for p in pops_with_data]

    x = np.arange(len(pops_with_data))
    width = 0.35

    bars1 = ax_c.bar(x - width/2, n_total, width, label='Total pairs analyzed',
                     color=COLORS['light_gray'], edgecolor=COLORS['dark'])
    bars2 = ax_c.bar(x + width/2, n_with_ibd, width, label='Pairs with >= 2Mb IBD',
                     color=[COLORS[p] for p in pops_with_data], edgecolor='none', alpha=0.8)

    for i, (total, with_ibd) in enumerate(zip(n_total, n_with_ibd)):
        pct = with_ibd / total * 100
        ax_c.text(i + width/2, with_ibd + max(n_total)*0.02, f'{pct:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(pops_with_data)
    ax_c.set_ylabel('Number of pairs')
    ax_c.legend(loc='upper right')
    ax_c.set_title('C. Fraction of pairs with detectable long IBD', loc='left', fontweight='bold')

    # ================================================================
    # Panel D: Mean segment length by population
    # ================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    mean_lengths = []
    std_lengths = []
    for pop in pops_with_data:
        results = all_data[pop].get('results', [])
        lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]
        if lengths:
            mean_lengths.append(np.mean(lengths))
            std_lengths.append(np.std(lengths) / np.sqrt(len(lengths)))
        else:
            mean_lengths.append(0)
            std_lengths.append(0)

    bars = ax_d.bar(pops_with_data, mean_lengths,
                    color=[COLORS[p] for p in pops_with_data], edgecolor='none', alpha=0.8,
                    yerr=std_lengths, capsize=5, error_kw={'linewidth': 1.5})

    for bar, val in zip(bars, mean_lengths):
        ax_d.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}',
                 ha='center', fontsize=10)

    ax_d.set_ylabel('Mean segment length (Mb)')
    ax_d.set_title('D. Average IBD segment length', loc='left', fontweight='bold')

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig1_2mb_main_analysis.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig1_2mb_main_analysis.png', format='png', dpi=300)
    print("Saved: fig1_2mb_main_analysis.pdf/png")

    plt.close(fig)


def create_population_comparison_figure():
    """Population genetics comparison figure."""
    setup_style()

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    populations = ['AFR', 'EUR']
    all_data = {p: load_full_results(p) for p in populations}
    pops_with_data = [p for p in populations if all_data[p]]

    # ================================================================
    # Panel A: IBD fraction per pair
    # ================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    data_box = []
    labels_box = []
    colors_box = []

    for pop in pops_with_data:
        results = all_data[pop].get('results', [])
        fractions = [r['fraction_ibd'] * 100 for r in results]
        if fractions:
            data_box.append(fractions)
            labels_box.append(pop)
            colors_box.append(COLORS[pop])

    if data_box:
        bp = ax_a.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)

        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color(COLORS['dark'])

        # Add means
        means = [np.mean(d) for d in data_box]
        for i, m in enumerate(means):
            ax_a.scatter(i + 1, m, marker='D', s=60, color='white',
                        edgecolor=COLORS['dark'], linewidth=2, zorder=5)

    ax_a.set_ylabel('IBD fraction per pair (%)')
    ax_a.set_title('A. IBD sharing distribution (pairs with >= 2Mb)', loc='left', fontweight='bold')

    # Stats
    if len(data_box) >= 2:
        u, p = stats.mannwhitneyu(data_box[0], data_box[1])
        y_max = max([max(d) for d in data_box])
        ax_a.plot([1, 2], [y_max * 1.05, y_max * 1.05], color=COLORS['dark'], linewidth=1)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax_a.text(1.5, y_max * 1.08, f'{sig} (p={p:.2e})', ha='center', fontsize=8)

    # ================================================================
    # Panel B: Summary statistics table
    # ================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis('off')

    # Diversity values
    diversity = {'AFR': 0.00125, 'EUR': 0.00085}

    table_data = []
    for pop in pops_with_data:
        results = all_data[pop].get('results', [])
        n_pairs = all_data[pop].get('n_pairs_with_segments', 0)
        total_seg = all_data[pop].get('total_segments', 0)
        lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]
        fractions = [r['fraction_ibd'] * 100 for r in results]

        table_data.append([
            pop,
            f'{diversity[pop]*100:.3f}%',
            f'{n_pairs:,}',
            f'{total_seg:,}',
            f'{np.mean(lengths):.2f}' if lengths else '-',
            f'{max(lengths):.2f}' if lengths else '-',
            f'{np.mean(fractions):.2f}%' if fractions else '-',
        ])

    col_labels = ['Pop', 'Diversity\n(pi)', 'Pairs\nwith IBD', 'Total\nsegs',
                  'Mean\nlength', 'Max\nlength', 'Mean\nIBD%']

    table = ax_b.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=[COLORS['light_gray']]*len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.0)

    for i, pop in enumerate(pops_with_data):
        table[(i+1, 0)].set_facecolor(COLORS[pop])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')

    ax_b.set_title('B. Summary of >= 2Mb IBD segments', loc='left', fontweight='bold', y=0.85)

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig2_2mb_population_comparison.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig2_2mb_population_comparison.png', format='png', dpi=300)
    print("Saved: fig2_2mb_population_comparison.pdf/png")

    plt.close(fig)


def create_top_segments_figure():
    """Figure showing top longest IBD segments."""
    setup_style()

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4)

    populations = ['AFR', 'EUR']

    for idx, pop in enumerate(populations):
        ax = fig.add_subplot(gs[idx])

        data = load_full_results(pop)
        if not data or 'results' not in data:
            continue

        results = data['results']

        # Get all segments
        all_segs = []
        for r in results:
            for s in r['segments']:
                all_segs.append({
                    'pair': f"{r['sample_a']}#{r['hap_a']} - {r['sample_b']}#{r['hap_b']}",
                    'sample_a': r['sample_a'],
                    'sample_b': r['sample_b'],
                    'length_mb': s['length_bp']/1e6,
                    'start_mb': s['start_bp']/1e6,
                    'end_mb': s['end_bp']/1e6,
                    'posterior': s['mean_posterior'],
                })

        # Sort by length and get top 15
        all_segs.sort(key=lambda x: -x['length_mb'])
        top_segs = all_segs[:15]

        # Plot horizontal bars
        y_pos = np.arange(len(top_segs))
        lengths = [s['length_mb'] for s in top_segs]
        labels = [f"{s['sample_a']}-{s['sample_b']}" for s in top_segs]

        bars = ax.barh(y_pos, lengths, color=COLORS[pop], alpha=0.7, edgecolor='none')

        # Add length labels
        for i, (bar, seg) in enumerate(zip(bars, top_segs)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{seg["length_mb"]:.2f} Mb', va='center', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Segment length (Mb)')

        panel = chr(65 + idx)
        total_seg = data.get('total_segments', 0)
        ax.set_title(f'{panel}. {pop}: Top 15 longest segments (of {total_seg:,} total >= 2Mb)',
                    loc='left', fontweight='bold')

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig3_2mb_top_segments.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig3_2mb_top_segments.png', format='png', dpi=300)
    print("Saved: fig3_2mb_top_segments.pdf/png")

    plt.close(fig)


def create_genomic_distribution_figure():
    """Figure showing distribution of segments across the region."""
    setup_style()

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.35)

    populations = ['AFR', 'EUR']

    for idx, pop in enumerate(populations):
        ax = fig.add_subplot(gs[idx])

        data = load_full_results(pop)
        if not data or 'results' not in data:
            continue

        results = data['results']

        # Get segment positions
        starts = []
        ends = []
        for r in results:
            for s in r['segments']:
                starts.append(s['start_bp']/1e6)
                ends.append(s['end_bp']/1e6)

        if not starts:
            continue

        # Create density plot
        positions = np.linspace(0, 50, 500)
        density = np.zeros_like(positions)

        for start, end in zip(starts, ends):
            mask = (positions >= start) & (positions <= end)
            density[mask] += 1

        # Normalize
        max_pairs = 1953 if pop == 'AFR' else 1830
        density_pct = density / max_pairs * 100

        ax.fill_between(positions, 0, density_pct, alpha=0.4, color=COLORS[pop])
        ax.plot(positions, density_pct, linewidth=1, color=COLORS[pop])

        ax.set_xlim(0, 50)
        ax.set_ylabel('% pairs with\nIBD at position')

        if idx == 1:
            ax.set_xlabel('Chromosome 2 position (Mb)')

        panel = chr(65 + idx)
        n_seg = data.get('total_segments', 0)
        ax.set_title(f'{panel}. {pop}: Distribution of {n_seg:,} IBD segments (>= 2 Mb)',
                    loc='left', fontweight='bold')

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig4_2mb_genomic_distribution.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig4_2mb_genomic_distribution.png', format='png', dpi=300)
    print("Saved: fig4_2mb_genomic_distribution.pdf/png")

    plt.close(fig)


if __name__ == '__main__':
    print("=" * 60)
    print("Generating 2Mb Analysis Figures - FULL DATASET")
    print("=" * 60)

    print("\n1. Main analysis figure...")
    create_main_figure()

    print("\n2. Population comparison figure...")
    create_population_comparison_figure()

    print("\n3. Top segments figure...")
    create_top_segments_figure()

    print("\n4. Genomic distribution figure...")
    create_genomic_distribution_figure()

    print("\n" + "=" * 60)
    print("Done! Figures saved to:", FIGURES_DIR)
    print("=" * 60)
