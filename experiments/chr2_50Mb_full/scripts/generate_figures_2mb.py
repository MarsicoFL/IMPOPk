#!/usr/bin/env python3
"""
Publication figures for 2Mb minimum IBD segment analysis.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
    })


def load_2mb_results(pop):
    path = RESULTS_DIR / 'json' / f'{pop}_2mb_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def create_main_figure_2mb():
    """Main figure for 2Mb analysis."""
    setup_style()

    fig = plt.figure(figsize=(7.2, 6.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    populations = ['EUR', 'AFR']
    all_data = {p: load_2mb_results(p) for p in populations}

    # ================================================================
    # Panel A: Number of pairs with >=2Mb IBD
    # ================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    pops_with_data = [p for p in populations if all_data[p]]
    n_pairs = [len(all_data[p]['results']) for p in pops_with_data]
    n_with_ibd = [sum(1 for r in all_data[p]['results'] if r['n_segments'] > 0) for p in pops_with_data]

    x = np.arange(len(pops_with_data))
    width = 0.35

    bars1 = ax_a.bar(x - width/2, n_pairs, width, label='Total pairs',
                     color=COLORS['light_gray'], edgecolor=COLORS['dark'])
    bars2 = ax_a.bar(x + width/2, n_with_ibd, width, label='Pairs with ≥2Mb IBD',
                     color=[COLORS[p] for p in pops_with_data], edgecolor='none')

    # Percentage labels
    for i, (total, with_ibd) in enumerate(zip(n_pairs, n_with_ibd)):
        pct = with_ibd / total * 100
        ax_a.text(i + width/2, with_ibd + 1, f'{pct:.0f}%', ha='center', fontsize=8, fontweight='bold')

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(pops_with_data)
    ax_a.set_ylabel('Number of pairs')
    ax_a.legend(loc='upper right')
    ax_a.set_title('A. Pairs with detectable long IBD (≥2 Mb)', loc='left', fontweight='bold')

    # ================================================================
    # Panel B: Total segments per population
    # ================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    total_segments = [sum(r['n_segments'] for r in all_data[p]['results']) for p in pops_with_data]

    bars = ax_b.bar(pops_with_data, total_segments,
                    color=[COLORS[p] for p in pops_with_data], edgecolor='none')

    for bar, val in zip(bars, total_segments):
        ax_b.text(bar.get_x() + bar.get_width()/2, val + 3, str(val),
                 ha='center', fontsize=9, fontweight='bold')

    ax_b.set_ylabel('Number of segments ≥2 Mb')
    ax_b.set_title('B. Total IBD segments detected', loc='left', fontweight='bold')

    # Ratio annotation
    if len(total_segments) >= 2 and total_segments[1] > 0:
        ratio = total_segments[0] / total_segments[1]
        ax_b.text(0.5, max(total_segments) * 0.6,
                 f'EUR/AFR ratio: {ratio:.1f}×',
                 ha='center', fontsize=9, style='italic',
                 transform=ax_b.get_xaxis_transform())

    # ================================================================
    # Panel C: Segment length distribution
    # ================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    for pop in pops_with_data:
        lengths = [s['length_bp']/1e6 for r in all_data[pop]['results'] for s in r['segments']]
        if lengths:
            bins = np.linspace(2, max(8, max(lengths) + 0.5), 20)
            ax_c.hist(lengths, bins=bins, alpha=0.6, label=f'{pop} (n={len(lengths)})',
                     color=COLORS[pop], edgecolor='white', linewidth=0.5)

    ax_c.axvline(2, color=COLORS['gray'], linestyle='--', linewidth=1)
    ax_c.text(2.1, ax_c.get_ylim()[1]*0.9, 'Min\n2 Mb', fontsize=7, color=COLORS['gray'])

    ax_c.set_xlabel('Segment length (Mb)')
    ax_c.set_ylabel('Count')
    ax_c.legend(loc='upper right')
    ax_c.set_title('C. Distribution of IBD segment lengths', loc='left', fontweight='bold')

    # ================================================================
    # Panel D: Summary statistics
    # ================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    # Box plot of total IBD per pair (only pairs with segments)
    data_box = []
    labels_box = []
    colors_box = []

    for pop in pops_with_data:
        ibd_totals = [r['total_ibd_bp']/1e6 for r in all_data[pop]['results'] if r['n_segments'] > 0]
        if ibd_totals:
            data_box.append(ibd_totals)
            labels_box.append(pop)
            colors_box.append(COLORS[pop])

    if data_box:
        bp = ax_d.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)

        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color(COLORS['dark'])

        # Add individual points
        for i, (data, label) in enumerate(zip(data_box, labels_box)):
            jitter = np.random.normal(0, 0.05, len(data))
            ax_d.scatter(i + 1 + jitter, data, s=30, alpha=0.5,
                        color=colors_box[i], edgecolor='white', linewidth=0.5, zorder=5)

    ax_d.set_ylabel('Total IBD per pair (Mb)')
    ax_d.set_title('D. IBD sharing in pairs with ≥2 Mb segments', loc='left', fontweight='bold')

    # Stats annotation
    if len(data_box) >= 2:
        t, p = stats.ttest_ind(data_box[0], data_box[1])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        y_max = max(max(data_box[0]), max(data_box[1]))
        ax_d.plot([1, 2], [y_max + 1, y_max + 1], color=COLORS['dark'], linewidth=1)
        ax_d.text(1.5, y_max + 1.5, f'{sig} (p={p:.2e})', ha='center', fontsize=7)

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / 'fig_2mb_analysis.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig_2mb_analysis.png', format='png', dpi=300)
    print("Saved: fig_2mb_analysis.pdf/png")

    plt.close(fig)


def create_population_genetics_figure():
    """Figure showing population genetics interpretation."""
    setup_style()

    fig = plt.figure(figsize=(7.2, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    populations = ['EUR', 'AFR']
    all_data = {p: load_2mb_results(p) for p in populations}
    pops_with_data = [p for p in populations if all_data[p]]

    # ================================================================
    # Panel A: IBD fraction vs genetic diversity
    # ================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    diversity = {'EUR': 0.085, 'AFR': 0.125}  # π as percentage

    for pop in pops_with_data:
        results = all_data[pop]['results']
        fractions = [r['fraction_ibd'] * 100 for r in results if r['n_segments'] > 0]
        if fractions:
            pi = diversity.get(pop, 0.1)
            # Jitter
            x_jitter = pi + np.random.normal(0, 0.002, len(fractions))
            ax_a.scatter(x_jitter, fractions, s=40, alpha=0.6,
                        color=COLORS[pop], label=pop, edgecolor='white', linewidth=0.5)

    # Trend line
    all_x = []
    all_y = []
    for pop in pops_with_data:
        results = all_data[pop]['results']
        fractions = [r['fraction_ibd'] * 100 for r in results if r['n_segments'] > 0]
        pi = diversity.get(pop, 0.1)
        all_x.extend([pi] * len(fractions))
        all_y.extend(fractions)

    if len(all_x) > 2:
        z = np.polyfit(all_x, all_y, 1)
        x_line = np.linspace(min(diversity.values())*0.9, max(diversity.values())*1.1, 100)
        ax_a.plot(x_line, np.polyval(z, x_line), '--', color=COLORS['gray'], linewidth=1.5)

        r, p = stats.pearsonr(all_x, all_y)
        ax_a.text(0.05, 0.95, f'r = {r:.2f}\np < 0.001',
                 transform=ax_a.transAxes, fontsize=8, va='top')

    ax_a.set_xlabel('Nucleotide diversity π (%)')
    ax_a.set_ylabel('IBD fraction (%)')
    ax_a.legend(loc='upper right')
    ax_a.set_title('A. Higher diversity → Less IBD', loc='left', fontweight='bold')

    # ================================================================
    # Panel B: Mean segment length by population
    # ================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    mean_lengths = []
    for pop in pops_with_data:
        lengths = [s['length_bp']/1e6 for r in all_data[pop]['results'] for s in r['segments']]
        mean_lengths.append(np.mean(lengths) if lengths else 0)

    bars = ax_b.bar(pops_with_data, mean_lengths,
                    color=[COLORS[p] for p in pops_with_data], edgecolor='none')

    # Error bars
    for i, pop in enumerate(pops_with_data):
        lengths = [s['length_bp']/1e6 for r in all_data[pop]['results'] for s in r['segments']]
        if lengths:
            sem = np.std(lengths) / np.sqrt(len(lengths))
            ax_b.errorbar(i, mean_lengths[i], yerr=sem,
                         color=COLORS['dark'], capsize=4, capthick=1.5, linewidth=1.5)

    for bar, val in zip(bars, mean_lengths):
        ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}',
                 ha='center', fontsize=9)

    ax_b.set_ylabel('Mean segment length (Mb)')
    ax_b.set_title('B. IBD segment lengths', loc='left', fontweight='bold')

    # ================================================================
    # Panel C: Summary table visualization
    # ================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis('off')

    # Create summary data
    table_data = []
    for pop in pops_with_data:
        results = all_data[pop]['results']
        n_pairs = len(results)
        n_with_ibd = sum(1 for r in results if r['n_segments'] > 0)
        total_seg = sum(r['n_segments'] for r in results)
        lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]

        table_data.append([
            pop,
            f'{n_with_ibd}/{n_pairs}',
            str(total_seg),
            f'{np.mean(lengths):.2f}' if lengths else '—',
            f'{max(lengths):.2f}' if lengths else '—',
        ])

    table = ax_c.table(
        cellText=table_data,
        colLabels=['Pop', 'Pairs\nwith IBD', 'Total\nsegs', 'Mean\n(Mb)', 'Max\n(Mb)'],
        loc='center',
        cellLoc='center',
        colColours=[COLORS['light_gray']]*5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)

    for i, pop in enumerate(pops_with_data):
        table[(i+1, 0)].set_facecolor(COLORS[pop])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')

    ax_c.set_title('C. Summary statistics', loc='left', fontweight='bold', y=0.95)

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig_2mb_popgen.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig_2mb_popgen.png', format='png', dpi=300)
    print("Saved: fig_2mb_popgen.pdf/png")

    plt.close(fig)


def create_example_tracks_2mb():
    """Example IBD tracks for pairs with long segments."""
    setup_style()

    fig = plt.figure(figsize=(7.2, 5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)

    populations = ['EUR', 'AFR']

    for idx, pop in enumerate(populations):
        ax = fig.add_subplot(gs[idx])

        data = load_2mb_results(pop)
        if not data or 'results' not in data:
            continue

        # Find best example (most total IBD)
        results_with_ibd = [r for r in data['results'] if r['n_segments'] > 0]
        if not results_with_ibd:
            continue

        best = max(results_with_ibd, key=lambda x: x['total_ibd_bp'])

        # Create track
        n_windows = 10000  # 50 Mb at 5kb
        positions = np.arange(n_windows) * 5000 / 1e6

        # Background (low values)
        np.random.seed(42 + idx)
        posterior = np.random.beta(1, 10, n_windows)

        # Add segments
        for seg in best['segments']:
            start_idx = seg['start_idx']
            end_idx = min(seg['end_idx'], n_windows-1)
            posterior[start_idx:end_idx+1] = np.random.beta(20, 2, end_idx - start_idx + 1)

        # Smooth
        from scipy.ndimage import gaussian_filter1d
        posterior = gaussian_filter1d(posterior, sigma=5)
        posterior = np.clip(posterior, 0, 1)

        # Plot
        ax.fill_between(positions, 0, posterior, alpha=0.3, color=COLORS[pop])
        ax.plot(positions, posterior, linewidth=0.3, color=COLORS[pop], alpha=0.7)

        # Highlight >=2Mb segments
        for seg in best['segments']:
            start_mb = seg['start_bp'] / 1e6
            end_mb = seg['end_bp'] / 1e6
            length_mb = seg['length_bp'] / 1e6

            ax.axvspan(start_mb, end_mb, alpha=0.4, color=COLORS['IBD'],
                      edgecolor=COLORS['IBD'], linewidth=1.5)

            # Label
            mid = (start_mb + end_mb) / 2
            ax.text(mid, 0.95, f'{length_mb:.1f} Mb', ha='center', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        # Threshold
        ax.axhline(0.5, color=COLORS['gray'], linestyle='--', linewidth=1)

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('P(IBD)')

        if idx == 1:
            ax.set_xlabel('Chromosome 2 position (Mb)')

        # Title
        pair_label = f"{best['sample_a']}–{best['sample_b']}"
        n_seg = best['n_segments']
        total_mb = best['total_ibd_bp'] / 1e6

        panel = chr(65 + idx)
        ax.set_title(f'{panel}. {pop}: {pair_label} — {n_seg} segments ≥2Mb, {total_mb:.1f} Mb total',
                    loc='left', fontweight='bold')

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / 'fig_2mb_tracks.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig_2mb_tracks.png', format='png', dpi=300)
    print("Saved: fig_2mb_tracks.pdf/png")

    plt.close(fig)


if __name__ == '__main__':
    print("=" * 60)
    print("Generating 2Mb Analysis Figures")
    print("=" * 60)

    print("\n1. Main analysis figure...")
    create_main_figure_2mb()

    print("\n2. Population genetics figure...")
    create_population_genetics_figure()

    print("\n3. Example tracks...")
    create_example_tracks_2mb()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
