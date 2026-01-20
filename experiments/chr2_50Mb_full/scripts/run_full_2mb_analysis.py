#!/usr/bin/env python3
"""
Full IBD Analysis with 2Mb minimum segment length - ALL PAIRS.
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
    Population, HMMParams, GaussianParams, IBDResult, IBDSegment,
    forward_backward, viterbi
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results_2Mb"

WINDOW_SIZE = 5000  # bp
MIN_SEGMENT_BP = 2_000_000  # 2 Mb


def estimate_emission_parameters(identities: np.ndarray) -> Dict:
    """Estimate emission parameters from full distribution."""
    p10, p90 = np.percentile(identities, [10, 90])
    bulk = identities[(identities >= p10) & (identities <= p90)]

    mean_non_ibd = np.mean(bulk)
    std_non_ibd = np.std(bulk)

    high = identities[identities >= 0.9995]
    if len(high) > 100:
        mean_ibd = np.mean(high)
        std_ibd = np.std(high)
    else:
        mean_ibd = 0.9997
        std_ibd = 0.0005

    pooled_std = np.sqrt((std_non_ibd**2 + std_ibd**2) / 2)
    d_prime = (mean_ibd - mean_non_ibd) / pooled_std if pooled_std > 0 else 0

    return {
        'non_ibd': {'mean': float(mean_non_ibd), 'std': float(std_non_ibd)},
        'ibd': {'mean': float(mean_ibd), 'std': float(std_ibd)},
        'd_prime': float(d_prime),
        'n_observations': len(identities),
    }


def create_long_segment_hmm(emission_params: Dict) -> HMMParams:
    """Create HMM optimized for detecting long segments (>=2Mb)."""

    emission_non_ibd = GaussianParams(
        mean=emission_params['non_ibd']['mean'],
        std=emission_params['non_ibd']['std'],
    )

    emission_ibd = GaussianParams(
        mean=emission_params['ibd']['mean'],
        std=emission_params['ibd']['std'],
    )

    # For 2 Mb segments at 5kb windows = 400 windows expected length
    p_exit_ibd = 0.001  # Expected length ~1000 windows = 5 Mb
    p_enter_ibd = 0.00005  # Low entry probability

    return HMMParams(
        emission_non_ibd=emission_non_ibd,
        emission_ibd=emission_ibd,
        p_enter_ibd=p_enter_ibd,
        p_exit_ibd=p_exit_ibd,
        p_initial_ibd=0.001,
    )


def merge_nearby_segments(segments: List[IBDSegment], max_gap_windows: int = 50) -> List[IBDSegment]:
    """Merge segments separated by small gaps."""
    if len(segments) <= 1:
        return segments

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        gap = next_seg.start_idx - current.end_idx - 1

        if gap <= max_gap_windows:
            current = IBDSegment(
                start_idx=current.start_idx,
                end_idx=next_seg.end_idx,
                start_bp=current.start_bp,
                end_bp=next_seg.end_bp,
                n_windows=next_seg.end_idx - current.start_idx + 1,
                length_bp=next_seg.end_bp - current.start_bp,
                mean_identity=(current.mean_identity + next_seg.mean_identity) / 2,
                mean_posterior=(current.mean_posterior + next_seg.mean_posterior) / 2,
                max_posterior=max(current.max_posterior, next_seg.max_posterior),
                min_posterior=min(current.min_posterior, next_seg.min_posterior),
            )
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)
    return merged


def extract_segments_long(
    states: np.ndarray,
    posterior: np.ndarray,
    identities: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    min_length_bp: int = MIN_SEGMENT_BP,
    merge_gap: int = 100,
) -> List[IBDSegment]:
    """Extract and merge long IBD segments."""

    segments = []
    n = len(states)

    if n == 0:
        return segments

    in_segment = False
    start_idx = 0

    for i in range(n):
        if states[i] == 1 and not in_segment:
            in_segment = True
            start_idx = i
        elif states[i] == 0 and in_segment:
            in_segment = False
            end_idx = i - 1

            seg_posterior = posterior[start_idx:end_idx+1]
            segments.append(IBDSegment(
                start_idx=start_idx,
                end_idx=end_idx,
                start_bp=int(window_starts[start_idx]),
                end_bp=int(window_ends[end_idx]),
                n_windows=end_idx - start_idx + 1,
                length_bp=int(window_ends[end_idx] - window_starts[start_idx]),
                mean_identity=float(np.mean(identities[start_idx:end_idx+1])),
                mean_posterior=float(np.mean(seg_posterior)),
                max_posterior=float(np.max(seg_posterior)),
                min_posterior=float(np.min(seg_posterior)),
            ))

    if in_segment:
        end_idx = n - 1
        seg_posterior = posterior[start_idx:end_idx+1]
        segments.append(IBDSegment(
            start_idx=start_idx,
            end_idx=end_idx,
            start_bp=int(window_starts[start_idx]),
            end_bp=int(window_ends[end_idx]),
            n_windows=end_idx - start_idx + 1,
            length_bp=int(window_ends[end_idx] - window_starts[start_idx]),
            mean_identity=float(np.mean(identities[start_idx:end_idx+1])),
            mean_posterior=float(np.mean(seg_posterior)),
            max_posterior=float(np.max(seg_posterior)),
            min_posterior=float(np.min(seg_posterior)),
        ))

    if merge_gap > 0:
        segments = merge_nearby_segments(segments, merge_gap)

    segments = [s for s in segments if s.length_bp >= min_length_bp]

    return segments


def load_ibs_data(filepath: Path) -> Tuple[np.ndarray, Dict]:
    """Load IBS data."""
    print(f"Loading: {filepath.name}")

    all_identities = []
    all_windows = set()
    pair_windows = defaultdict(dict)
    chrom = None

    line_count = 0
    with open(filepath, 'r') as f:
        f.readline()

        for line in f:
            line_count += 1
            if line_count % 2000000 == 0:
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

            if chrom is None:
                chrom = chrom_full

            all_identities.append(identity)

            parts_a = group_a.split('#')
            parts_b = group_b.split('#')
            sample_a = parts_a[0]
            hap_a = int(parts_a[1]) if len(parts_a) > 1 else 0
            sample_b = parts_b[0]
            hap_b = int(parts_b[1]) if len(parts_b) > 1 else 0

            if sample_a > sample_b or (sample_a == sample_b and hap_a > hap_b):
                sample_a, hap_a, sample_b, hap_b = sample_b, hap_b, sample_a, hap_a

            window_key = (start, end)
            pair_key = (sample_a, hap_a, sample_b, hap_b)

            all_windows.add(window_key)
            pair_windows[pair_key][window_key] = identity

    print(f"  Total lines: {line_count:,}")
    print(f"  Unique windows: {len(all_windows):,}")
    print(f"  Unique pairs: {len(pair_windows):,}")

    sorted_windows = sorted(all_windows)

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


def analyze_population_all_pairs(pop_name: str) -> Tuple[Dict, List[Dict]]:
    """Analyze ALL pairs in population with 2Mb minimum segments."""

    print(f"\n{'='*60}")
    print(f"Analyzing {pop_name} - ALL PAIRS - 2Mb Minimum")
    print(f"{'='*60}")

    data_file = DATA_DIR / f"{pop_name}_chr2_50Mb_full.tsv"
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return {}, []

    all_identities, pair_data = load_ibs_data(data_file)

    print("\nEstimating emission parameters...")
    emission_params = estimate_emission_parameters(all_identities)
    print(f"  d' = {emission_params['d_prime']:.2f}")

    hmm_params = create_long_segment_hmm(emission_params)
    print(f"  HMM p_exit = {hmm_params.p_exit_ibd:.4f} (expected length: {1/hmm_params.p_exit_ibd:.0f} windows)")

    total_pairs = len(pair_data)
    print(f"\nRunning IBD inference on ALL {total_pairs} pairs...")

    results = []
    pairs_processed = 0
    segments_found = 0

    for pair_key, data in pair_data.items():
        pairs_processed += 1
        if pairs_processed % 200 == 0:
            print(f"  Processing pair {pairs_processed}/{total_pairs} ({100*pairs_processed/total_pairs:.1f}%) - {segments_found} segments found")

        sample_a, hap_a, sample_b, hap_b, chrom = pair_key

        identities = data['identities'].copy()
        valid_mask = ~np.isnan(identities)
        if np.sum(valid_mask) < 100:
            continue

        identities[~valid_mask] = hmm_params.emission_non_ibd.mean

        try:
            posterior, log_likelihood = forward_backward(identities, hmm_params)
            states = viterbi(identities, hmm_params)

            segments = extract_segments_long(
                states, posterior, identities,
                data['starts'], data['ends'],
                min_length_bp=MIN_SEGMENT_BP,
                merge_gap=100,
            )

            if segments:
                segments_found += len(segments)
                total_ibd_bp = sum(s.length_bp for s in segments)
                total_bp = data['ends'][-1] - data['starts'][0]

                result = {
                    'sample_a': sample_a,
                    'hap_a': hap_a,
                    'sample_b': sample_b,
                    'hap_b': hap_b,
                    'chrom': chrom,
                    'n_segments': len(segments),
                    'total_ibd_bp': total_ibd_bp,
                    'fraction_ibd': total_ibd_bp / total_bp if total_bp > 0 else 0,
                    'segments': [s.to_dict() for s in segments],
                }
                results.append(result)

        except Exception as e:
            pass  # Skip errors silently for full analysis

    print(f"  Completed: {pairs_processed} pairs analyzed")
    print(f"  Pairs with >= 2Mb segments: {len(results)}")
    print(f"  Total segments >= 2Mb: {segments_found}")

    if segments_found > 0:
        all_lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]
        print(f"  Segment lengths: {min(all_lengths):.2f} - {max(all_lengths):.2f} Mb")
        print(f"  Mean segment length: {np.mean(all_lengths):.2f} Mb")

    return emission_params, results


def main():
    parser = argparse.ArgumentParser(description='Full IBD analysis with 2Mb minimum - ALL PAIRS')
    parser.add_argument('--populations', nargs='+', default=['AFR', 'EUR'])
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / 'json'
    json_dir.mkdir(exist_ok=True)
    figures_dir = args.output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FULL IBD ANALYSIS - 2 Mb MINIMUM SEGMENT LENGTH - ALL PAIRS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    for pop in args.populations:
        emission_params, results = analyze_population_all_pairs(pop)

        if emission_params:
            all_results[pop] = {'params': emission_params, 'results': results}

            with open(json_dir / f'{pop}_2mb_full_results.json', 'w') as f:
                json.dump({
                    'population': pop,
                    'min_segment_mb': 2.0,
                    'analysis_type': 'all_pairs',
                    'emission_params': emission_params,
                    'n_pairs_with_segments': len(results),
                    'total_segments': sum(r['n_segments'] for r in results),
                    'results': results,
                }, f, indent=2)

    # Generate summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - 2 Mb MINIMUM SEGMENTS (ALL PAIRS)")
    print("=" * 70)

    for pop in args.populations:
        if pop in all_results:
            results = all_results[pop]['results']
            total_seg = sum(r['n_segments'] for r in results)

            print(f"\n{pop}:")
            print(f"  Pairs with >=2Mb IBD: {len(results)}")
            print(f"  Total segments >=2Mb: {total_seg}")

            if total_seg > 0:
                lengths = [s['length_bp']/1e6 for r in results for s in r['segments']]
                print(f"  Mean segment length: {np.mean(lengths):.2f} Mb")
                print(f"  Max segment length: {max(lengths):.2f} Mb")

                print(f"\n  Top 5 longest segments:")
                all_segs = []
                for r in results:
                    for s in r['segments']:
                        all_segs.append({
                            'pair': f"{r['sample_a']}#{r['hap_a']} - {r['sample_b']}#{r['hap_b']}",
                            'length_mb': s['length_bp']/1e6,
                            'pos': f"{s['start_bp']/1e6:.1f}-{s['end_bp']/1e6:.1f} Mb",
                            'posterior': s['mean_posterior'],
                        })
                all_segs.sort(key=lambda x: -x['length_mb'])
                for i, seg in enumerate(all_segs[:5]):
                    print(f"    {i+1}. {seg['pair']}: {seg['length_mb']:.2f} Mb ({seg['pos']}, P={seg['posterior']:.3f})")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
