#!/usr/bin/env python3
"""
chr1_full: Full Chromosome 1 IBD Analysis - CORRECTED VERSION

Key fix: The original script included low-quality alignments (identity < 0.9)
which severely biased the emission parameter estimation.

The correct approach:
1. Filter out low-quality windows (identity < 0.9) - these are gaps/poor alignments
2. The non-IBD distribution is the bulk around 0.99-0.999
3. The IBD distribution is the right tail near 1.0

Author: IBD-CLI Project
Date: 2026-01-19
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from ibd_inference import (
    Population, HMMParams, GaussianParams, IBDResult,
    forward_backward, viterbi, extract_segments
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Quality filter: exclude windows with identity < this threshold
# These represent gaps, poor alignments, or structural variants
MIN_QUALITY_THRESHOLD = 0.90

# Theoretical population diversity
THEORETICAL_DIVERSITY = {
    'AFR': 0.00125,  # ~0.125% divergence -> non-IBD mean ~0.99875
    'EUR': 0.00085,  # ~0.085% divergence -> non-IBD mean ~0.99915
}

# Region parameters
CHROM = "chr1"
REGION_END = 248956422
WINDOW_SIZE = 5000


# ============================================================
# CORRECTED Emission Parameter Estimation
# ============================================================

def estimate_emission_parameters_v2(
    identities: np.ndarray,
    population: str,
    verbose: bool = True
) -> Dict:
    """
    CORRECTED emission parameter estimation.

    Key changes from v1:
    1. Filter out low-quality windows (identity < 0.9)
    2. Use robust estimators on the filtered distribution
    3. Properly separate non-IBD bulk from IBD tail
    """
    n_total = len(identities)

    # Step 1: Filter low-quality alignments
    high_quality = identities[identities >= MIN_QUALITY_THRESHOLD]
    n_filtered = len(high_quality)
    frac_filtered = n_filtered / n_total

    if verbose:
        print(f"  Total observations: {n_total:,}")
        print(f"  After quality filter (>={MIN_QUALITY_THRESHOLD}): {n_filtered:,} ({frac_filtered:.1%})")

    # Step 2: Identify the non-IBD bulk
    # The non-IBD distribution is centered around 1 - π (diversity)
    # For AFR: ~0.99875, for EUR: ~0.99915
    # We use values < 0.999 as clearly non-IBD

    non_ibd_mask = high_quality < 0.999
    non_ibd_values = high_quality[non_ibd_mask]

    # Step 3: Identify potential IBD windows
    # IBD windows should have identity very close to 1.0 (>= 0.9999)
    ibd_mask = high_quality >= 0.9999
    ibd_values = high_quality[ibd_mask]

    # Step 4: Estimate non-IBD parameters
    if len(non_ibd_values) > 100:
        # Use percentile method to exclude any IBD contamination
        # The 25th-75th percentile should be pure non-IBD
        p25, p75 = np.percentile(non_ibd_values, [25, 75])
        bulk = non_ibd_values[(non_ibd_values >= p25) & (non_ibd_values <= p75)]

        mean_non_ibd = np.mean(bulk)
        std_non_ibd = np.std(bulk)

        # Also compute robust estimates
        median_non_ibd = np.median(non_ibd_values)
        mad = np.median(np.abs(non_ibd_values - median_non_ibd))
        robust_std = mad * 1.4826
    else:
        # Fallback to theoretical
        pi = THEORETICAL_DIVERSITY.get(population, 0.001)
        mean_non_ibd = 1 - pi
        std_non_ibd = 0.001
        median_non_ibd = mean_non_ibd
        robust_std = std_non_ibd

    # Step 5: Estimate IBD parameters
    if len(ibd_values) > 50:
        mean_ibd = np.mean(ibd_values)
        std_ibd = np.std(ibd_values)
        # Ensure std is not too small (numerical issues)
        std_ibd = max(std_ibd, 0.0001)
    else:
        # Theoretical IBD: nearly identical
        mean_ibd = 0.9999
        std_ibd = 0.0002

    # Step 6: Calculate d' separability
    pooled_std = np.sqrt((std_non_ibd**2 + std_ibd**2) / 2)
    d_prime = (mean_ibd - mean_non_ibd) / pooled_std if pooled_std > 0 else 0

    # Step 7: Fraction statistics
    frac_above_999 = np.mean(high_quality >= 0.999)
    frac_above_9995 = np.mean(high_quality >= 0.9995)
    frac_above_9999 = np.mean(high_quality >= 0.9999)

    result = {
        'n_total': n_total,
        'n_high_quality': n_filtered,
        'quality_threshold': MIN_QUALITY_THRESHOLD,
        'fraction_high_quality': float(frac_filtered),
        'non_ibd': {
            'mean': float(mean_non_ibd),
            'std': float(std_non_ibd),
            'median': float(median_non_ibd),
            'robust_std': float(robust_std),
            'n_samples': len(non_ibd_values),
        },
        'ibd': {
            'mean': float(mean_ibd),
            'std': float(std_ibd),
            'n_samples': len(ibd_values),
        },
        'd_prime': float(d_prime),
        'fraction_above_thresholds': {
            '0.999': float(frac_above_999),
            '0.9995': float(frac_above_9995),
            '0.9999': float(frac_above_9999),
        },
    }

    if verbose:
        print(f"  Non-IBD samples (<0.999): {len(non_ibd_values):,}")
        print(f"  IBD samples (>=0.9999): {len(ibd_values):,}")
        print(f"  Non-IBD: mean={mean_non_ibd:.6f}, std={std_non_ibd:.6f}")
        print(f"  IBD: mean={mean_ibd:.6f}, std={std_ibd:.6f}")
        print(f"  d' separability: {d_prime:.2f}")

    return result


def create_hmm_params_v2(
    emission_params: Dict,
    expected_ibd_length: float = 50.0,
    p_enter_ibd: float = 0.0001,
) -> HMMParams:
    """Create HMM parameters from corrected emission estimates."""

    emission_non_ibd = GaussianParams(
        mean=emission_params['non_ibd']['mean'],
        std=emission_params['non_ibd']['std'],
    )

    emission_ibd = GaussianParams(
        mean=emission_params['ibd']['mean'],
        std=emission_params['ibd']['std'],
    )

    p_exit_ibd = 1.0 / expected_ibd_length
    p_exit_ibd = np.clip(p_exit_ibd, 0.0001, 0.5)

    return HMMParams(
        emission_non_ibd=emission_non_ibd,
        emission_ibd=emission_ibd,
        p_enter_ibd=p_enter_ibd,
        p_exit_ibd=p_exit_ibd,
    )


# ============================================================
# Data Loading (same as original but with quality filter)
# ============================================================

def load_identity_data(
    filepath: Path,
    max_lines: int = None,
    quality_threshold: float = MIN_QUALITY_THRESHOLD,
) -> Tuple[np.ndarray, Dict]:
    """Load identity data with quality filtering."""

    print(f"Loading: {filepath.name}")

    all_identities = []
    all_windows = set()
    pair_windows = defaultdict(dict)
    chrom = None

    line_count = 0
    low_quality_count = 0

    with open(filepath, 'r') as f:
        header = f.readline()

        for line in f:
            line_count += 1

            if max_lines and line_count > max_lines:
                break

            if line_count % 10000000 == 0:
                print(f"  Processed {line_count:,} lines...")

            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            chrom_full, start, end, group_a, group_b, identity = parts[:6]

            try:
                start = int(start)
                end = int(end)
                identity = float(identity)
            except ValueError:
                continue

            # Track all identities for parameter estimation
            all_identities.append(identity)

            # Skip low-quality windows for pair analysis
            if identity < quality_threshold:
                low_quality_count += 1
                continue

            if chrom is None:
                chrom = chrom_full

            # Parse haplotype names
            parts_a = group_a.split('#')
            parts_b = group_b.split('#')
            sample_a = parts_a[0]
            hap_a = int(parts_a[1]) if len(parts_a) > 1 and parts_a[1].isdigit() else 0
            sample_b = parts_b[0]
            hap_b = int(parts_b[1]) if len(parts_b) > 1 and parts_b[1].isdigit() else 0

            # Canonical order
            if sample_a > sample_b or (sample_a == sample_b and hap_a > hap_b):
                sample_a, hap_a, sample_b, hap_b = sample_b, hap_b, sample_a, hap_a

            window_key = (start, end)
            pair_key = (sample_a, hap_a, sample_b, hap_b)

            all_windows.add(window_key)
            pair_windows[pair_key][window_key] = identity

    print(f"  Total lines: {line_count:,}")
    print(f"  Low quality excluded: {low_quality_count:,} ({100*low_quality_count/line_count:.1f}%)")
    print(f"  High quality windows: {len(all_windows):,}")
    print(f"  Unique pairs: {len(pair_windows):,}")

    # Sort windows
    sorted_windows = sorted(all_windows)

    # Build pair data
    pair_data = {}
    for pair_key, windows in pair_windows.items():
        sample_a, hap_a, sample_b, hap_b = pair_key

        starts = []
        ends = []
        identities = []

        for start, end in sorted_windows:
            starts.append(start)
            ends.append(end)
            identities.append(windows.get((start, end), np.nan))

        pair_data[(sample_a, hap_a, sample_b, hap_b, chrom)] = {
            'starts': np.array(starts),
            'ends': np.array(ends),
            'identities': np.array(identities),
        }

    return np.array(all_identities), pair_data


# ============================================================
# IBD Analysis
# ============================================================

def analyze_pair(
    pair_key: Tuple,
    data: Dict,
    hmm_params: HMMParams,
    min_segment_windows: int = 10,
) -> Optional[IBDResult]:
    """Run IBD inference on a single pair."""

    sample_a, hap_a, sample_b, hap_b, chrom = pair_key

    identities = data['identities']
    starts = data['starts']
    ends = data['ends']

    # Handle missing values
    valid_mask = ~np.isnan(identities)
    if np.sum(valid_mask) < 100:
        return None

    # Fill missing with non-IBD mean
    identities_filled = identities.copy()
    identities_filled[~valid_mask] = hmm_params.emission_non_ibd.mean

    try:
        # Run HMM
        posterior_ibd, log_likelihood = forward_backward(identities_filled, hmm_params)
        viterbi_states = viterbi(identities_filled, hmm_params)

        # Extract segments
        segments = extract_segments(
            viterbi_states,
            posterior_ibd,
            identities_filled,
            starts,
            ends,
            min_windows=min_segment_windows,
            min_posterior=0.5,
        )

        result = IBDResult(
            sample_a=sample_a,
            hap_a=hap_a,
            sample_b=sample_b,
            hap_b=hap_b,
            chrom=chrom,
            n_windows=len(identities_filled),
            window_starts=starts,
            window_ends=ends,
            identities=identities_filled,
            posterior_ibd=posterior_ibd,
            viterbi_states=viterbi_states,
            log_likelihood=log_likelihood,
            segments=segments,
        )
        result.compute_summary()

        return result

    except Exception as e:
        return None


def analyze_population(
    pop_name: str,
    max_pairs: int = 100,
    max_lines: int = None,
    expected_ibd_length: float = 50.0,
    min_segment_windows: int = 10,
) -> Tuple[Dict, List[IBDResult]]:
    """Analyze IBD for a population with corrected parameters."""

    print(f"\n{'='*60}")
    print(f"Analyzing {pop_name} - chr1 Full (CORRECTED)")
    print(f"{'='*60}")

    data_file = DATA_DIR / f"{pop_name}_chr1_full.tsv"
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return {}, []

    # Load data
    all_identities, pair_data = load_identity_data(data_file, max_lines)

    # Estimate parameters with corrected method
    print("\nEstimating emission parameters (CORRECTED)...")
    emission_params = estimate_emission_parameters_v2(all_identities, pop_name)

    # Compare with theoretical
    pi_theoretical = THEORETICAL_DIVERSITY.get(pop_name, 0.001)
    mean_theoretical = 1 - pi_theoretical
    print(f"\n  Theoretical non-IBD mean: {mean_theoretical:.6f}")
    print(f"  Empirical non-IBD mean: {emission_params['non_ibd']['mean']:.6f}")
    diff_pct = (emission_params['non_ibd']['mean'] - mean_theoretical) * 100
    print(f"  Difference: {diff_pct:+.4f}%")

    # Create HMM
    hmm_params = create_hmm_params_v2(
        emission_params,
        expected_ibd_length=expected_ibd_length,
    )

    # Select pairs for analysis (those with most variance)
    pair_variance = {}
    for pair_key, data in pair_data.items():
        valid = data['identities'][~np.isnan(data['identities'])]
        if len(valid) > 100:
            pair_variance[pair_key] = np.var(valid)

    sorted_pairs = sorted(pair_variance.items(), key=lambda x: -x[1])
    selected_pairs = [p[0] for p in sorted_pairs[:max_pairs]]

    print(f"\nRunning IBD inference on {len(selected_pairs)} pairs...")

    results = []
    for i, pair_key in enumerate(selected_pairs):
        if (i + 1) % 20 == 0:
            print(f"  Processing pair {i+1}/{len(selected_pairs)}")

        result = analyze_pair(
            pair_key,
            pair_data[pair_key],
            hmm_params,
            min_segment_windows,
        )

        if result:
            results.append(result)

    print(f"  Completed: {len(results)} pairs analyzed")

    return emission_params, results


# ============================================================
# Report Generation
# ============================================================

def generate_report(
    results_by_pop: Dict[str, Tuple[Dict, List[IBDResult]]],
    output_dir: Path,
) -> str:
    """Generate report with corrected parameters."""

    report = []
    report.append("# chr1_full: Full Chromosome 1 IBD Analysis Report (CORRECTED)")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Version**: v2 (corrected emission estimation)")
    report.append(f"**Quality Filter**: identity >= {MIN_QUALITY_THRESHOLD}")
    report.append(f"**Region**: chr1:1-{REGION_END:,} ({REGION_END/1e6:.0f} Mb)")
    report.append(f"**Window Size**: {WINDOW_SIZE/1000:.0f} kb")
    report.append("")

    # Key fix explanation
    report.append("## Key Corrections from v1")
    report.append("")
    report.append("1. **Quality filter**: Exclude windows with identity < 0.90")
    report.append("   - These represent gaps, poor alignments, or structural variants")
    report.append("   - Previously biased the non-IBD mean severely downward")
    report.append("")
    report.append("2. **Proper non-IBD estimation**: Use windows with 0.90 <= identity < 0.999")
    report.append("   - This is the true non-IBD bulk distribution")
    report.append("")
    report.append("3. **Proper IBD estimation**: Use windows with identity >= 0.9999")
    report.append("   - True IBD should be nearly identical")
    report.append("")

    # Emission Parameters
    report.append("## Empirical Emission Parameters")
    report.append("")
    report.append("| Population | Quality % | Non-IBD Mean | Non-IBD Std | IBD Mean | IBD Std | d' |")
    report.append("|------------|-----------|--------------|-------------|----------|---------|-----|")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            params, _ = results_by_pop[pop_name]
            if params:
                quality_pct = params['fraction_high_quality'] * 100
                report.append(
                    f"| {pop_name} | {quality_pct:.1f}% | "
                    f"{params['non_ibd']['mean']:.6f} | {params['non_ibd']['std']:.6f} | "
                    f"{params['ibd']['mean']:.6f} | {params['ibd']['std']:.6f} | "
                    f"{params['d_prime']:.2f} |"
                )

    report.append("")

    # Quality Assessment
    report.append("## Quality Assessment")
    report.append("")
    report.append("d' separability (higher is better):")
    report.append("- d' > 2: Good separation")
    report.append("- d' 1-2: Moderate separation")
    report.append("- d' < 1: Poor separation")
    report.append("")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            params, _ = results_by_pop[pop_name]
            if params:
                d_prime = params['d_prime']
                quality = "Good" if d_prime > 2 else ("Moderate" if d_prime > 1 else "Poor")
                report.append(f"- **{pop_name}**: d' = {d_prime:.2f} ({quality})")

    report.append("")

    # IBD Results
    report.append("## IBD Detection Results")
    report.append("")
    report.append("| Population | Pairs | Segments | Mean IBD (Mb) | Mean Fraction | Mean Length (kb) |")
    report.append("|------------|-------|----------|---------------|---------------|------------------|")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            _, results = results_by_pop[pop_name]
            if results:
                n_pairs = len(results)
                total_segments = sum(r.n_segments for r in results)
                mean_ibd = np.mean([r.total_ibd_bp for r in results]) / 1e6
                mean_frac = np.mean([r.fraction_ibd for r in results])

                all_lengths = [s.length_bp for r in results for s in r.segments]
                mean_length = np.mean(all_lengths) / 1000 if all_lengths else 0

                report.append(
                    f"| {pop_name} | {n_pairs} | {total_segments} | "
                    f"{mean_ibd:.2f} | {mean_frac:.4f} | {mean_length:.1f} |"
                )

    report.append("")

    report_text = '\n'.join(report)

    report_path = output_dir / "REPORT_v2.md"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved: {report_path}")

    return report_text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='chr1_full: Full chromosome 1 IBD analysis (CORRECTED)'
    )
    parser.add_argument('--populations', nargs='+', default=['EUR', 'AFR'])
    parser.add_argument('--max-pairs', type=int, default=100,
                        help='Maximum pairs to analyze per population')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Maximum lines to read (for testing)')
    parser.add_argument('--expected-ibd-length', type=float, default=50.0,
                        help='Expected IBD segment length in windows')
    parser.add_argument('--min-segment-windows', type=int, default=10,
                        help='Minimum segment length to report')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / 'json'
    json_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("chr1_full: FULL CHROMOSOME 1 IBD ANALYSIS (CORRECTED v2)")
    print("=" * 70)
    print(f"Quality filter: identity >= {MIN_QUALITY_THRESHOLD}")
    print(f"Populations: {args.populations}")
    print(f"Max pairs: {args.max_pairs}")

    results_by_pop = {}

    for pop in args.populations:
        emission_params, results = analyze_population(
            pop,
            max_pairs=args.max_pairs,
            max_lines=args.max_lines,
            expected_ibd_length=args.expected_ibd_length,
            min_segment_windows=args.min_segment_windows,
        )

        if emission_params:
            results_by_pop[pop] = (emission_params, results)

            # Save emission parameters
            with open(json_dir / f'{pop}_emission_params_v2.json', 'w') as f:
                json.dump(emission_params, f, indent=2)

            # Save results summary
            if results:
                summary = {
                    'population': pop,
                    'version': 'v2_corrected',
                    'quality_threshold': MIN_QUALITY_THRESHOLD,
                    'emission_params': emission_params,
                    'n_pairs': len(results),
                    'total_segments': sum(r.n_segments for r in results),
                    'mean_ibd_mb': float(np.mean([r.total_ibd_bp for r in results]) / 1e6),
                }
                with open(json_dir / f'{pop}_summary_v2.json', 'w') as f:
                    json.dump(summary, f, indent=2)

    # Generate report
    if results_by_pop:
        generate_report(results_by_pop, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE (CORRECTED v2)")
    print("=" * 70)

    for pop in args.populations:
        if pop in results_by_pop:
            params, results = results_by_pop[pop]
            print(f"\n{pop}:")
            print(f"  Quality retained: {params['fraction_high_quality']:.1%}")
            print(f"  d' separability: {params['d_prime']:.2f}")
            print(f"  Pairs analyzed: {len(results)}")
            if results:
                total_seg = sum(r.n_segments for r in results)
                mean_ibd = np.mean([r.total_ibd_bp for r in results]) / 1e6
                print(f"  Total segments: {total_seg}")
                print(f"  Mean IBD: {mean_ibd:.2f} Mb")


if __name__ == '__main__':
    main()
