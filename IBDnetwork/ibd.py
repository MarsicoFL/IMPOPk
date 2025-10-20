#!/usr/bin/env python3

import sys, csv, argparse
from collections import defaultdict, namedtuple

Row = namedtuple("Row", "region chr start end length a b ident")

def parse_table(path, identity_col="estimated.identity"):
    req = {"REGION","CHR","START","END","LENGTH","group.a","group.b",identity_col}
    rows = []
    with open(path, newline="") as fh:
        dr = csv.DictReader(fh, delimiter="\t")
        if not dr.fieldnames or not set(dr.fieldnames).issuperset(req):
            raise SystemExit(f"Error: missing columns in {path}. Required: {sorted(req)}")
        for r in dr:
            try:
                rows.append(
                    Row(
                        region=r["REGION"],
                        chr=r["CHR"],
                        start=int(r["START"]),
                        end=int(r["END"]),
                        length=int(r["LENGTH"]),
                        a=r["group.a"],
                        b=r["group.b"],
                        ident=float(r[identity_col]),
                    )
                )
            except Exception as e:
                # skip malformed lines
                continue
    return rows

def pair_key(a,b):
    return (a,b) if a<=b else (b,a)

def build_windows(rows):
    # one ordered list per chromosome (keeps windows contiguous by chr)
    uniq = defaultdict(dict)  # chr -> {(start,end,length): None}
    for r in rows:
        uniq[r.chr][(r.start,r.end,r.length)] = None
    Win = namedtuple("Win","chr start end length")
    wins_by_chr = {}
    index_by_chr = {}
    for chr_, d in uniq.items():
        wins = [Win(chr_, s, e, l) for (s,e,l) in sorted(d.keys())]
        wins_by_chr[chr_] = wins
        index_by_chr[chr_] = {(w.start,w.end): i for i,w in enumerate(wins)}
    return wins_by_chr, index_by_chr

def build_pair_tracks(rows, index_by_chr):
    # per pair -> per chr -> list of (win_idx, ident)
    tracks = defaultdict(lambda: defaultdict(list))
    for r in rows:
        idx = index_by_chr[r.chr].get((r.start,r.end))
        if idx is None: continue
        tracks[pair_key(r.a,r.b)][r.chr].append((idx, r.ident, r.length))
    # sort by index
    for p in tracks:
        for chr_ in tracks[p]:
            tracks[p][chr_].sort(key=lambda x: x[0])
    return tracks

def rle_segments_for_pair(track_list, wins, min_id, max_gap, min_windows, min_len_bp,
                          treat_missing_as_gap=True, drop_tolerance=0.0):
    """Simple run-length thresholding with gap tolerance (works on one chr)."""
    ident_by_idx = {idx:ident for idx,ident,_ in track_list}
    segments = []
    n_wins_total = len(wins)

    def qualifies(ident):
        return (ident is not None) and (ident >= min_id or (drop_tolerance>0 and ident >= (min_id - drop_tolerance)))

    i = 0
    current = None
    gaps = 0
    called = 0
    ident_sum = 0.0
    min_ident = 1.0

    while i < n_wins_total:
        ident = ident_by_idx.get(i, None)
        missing = (ident is None)
        good = qualifies(ident) if not missing else False

        if current is None:
            if good:
                current = {"start_idx": i, "end_idx": i}
                gaps = 0; called = 1; ident_sum = ident; min_ident = ident
        else:
            extend_condition = good or (missing and treat_missing_as_gap)
            if extend_condition:
                current["end_idx"] = i
                if missing:
                    gaps += 1
                else:
                    called += 1
                    ident_sum += ident
                    if ident < min_ident: min_ident = ident
                if gaps > max_gap:
                    # finalize without including this window
                    current["end_idx"] = i-1
                    seg = finalize_segment(current, wins, called, ident_sum, min_ident,
                                           min_windows, min_len_bp, gaps_allowed=max_gap)
                    if seg: segments.append(seg)
                    current = None; gaps=0; called=0; ident_sum=0.0; min_ident=1.0
            else:
                seg = finalize_segment(current, wins, called, ident_sum, min_ident,
                                       min_windows, min_len_bp, gaps_allowed=gaps)
                if seg: segments.append(seg)
                current = None; gaps=0; called=0; ident_sum=0.0; min_ident=1.0
                if good:
                    current = {"start_idx": i, "end_idx": i}
                    gaps = 0; called = 1; ident_sum = ident; min_ident = ident
        i += 1

    if current is not None:
        seg = finalize_segment(current, wins, called, ident_sum, min_ident,
                               min_windows, min_len_bp, gaps_allowed=gaps)
        if seg: segments.append(seg)
    return segments

def seed_extend_segments_for_pair(track_list, wins, seed_thr, seed_k,
                                  ext_thr, xdrop, reward, pen_bad, pen_miss,
                                  min_windows, min_len_bp, treat_missing_as_gap=True):
    """Seed-and-extend (x-drop) per chromosome: 
       1) find seeds = runs of >= seed_k windows with ident >= seed_thr
       2) extend left/right using x-drop on a simple scoring scheme
    """
    ident_by_idx = {idx:ident for idx,ident,_ in track_list}
    n = len(wins)
    used = [False]*n  # optional: mark windows already assigned to a called seg to avoid duplicates
    segments = []

    # helper to classify a window
    def classify(i):
        ident = ident_by_idx.get(i, None)
        if ident is None:
            return None, True  # missing
        return ident, False

    # find seeds: runs with ident >= seed_thr
    i = 0
    seeds = []
    while i < n:
        ident = ident_by_idx.get(i, None)
        if ident is not None and ident >= seed_thr:
            j = i
            run = 0
            while j < n and ident_by_idx.get(j, None) is not None and ident_by_idx[j] >= seed_thr:
                run += 1; j += 1
            if run >= seed_k:
                seeds.append( (i, j-1) )  # inclusive indices
            i = j
        else:
            i += 1

    # extend each seed (skip overlapping processed areas)
    for (s, e) in seeds:
        # skip if fully covered by a previous segment (optional)
        if all(used[k] for k in range(s, e+1)):
            continue

        # extend left
        best_left = s
        score = 0.0
        best_score = 0.0
        k = s - 1
        while k >= 0:
            ident, missing = classify(k)
            if missing and not treat_missing_as_gap:
                k -= 1
                continue
            if missing:
                score -= pen_miss
            else:
                if ident >= ext_thr:
                    score += reward
                else:
                    score -= pen_bad
            if score > best_score:
                best_score = score
                best_left = k
            # x-drop stop
            if best_score - score > xdrop:
                break
            k -= 1

        # extend right
        best_right = e
        score = 0.0
        best_score = 0.0
        k = e + 1
        while k < n:
            ident, missing = classify(k)
            if missing and not treat_missing_as_gap:
                k += 1
                continue
            if missing:
                score -= pen_miss
            else:
                if ident >= ext_thr:
                    score += reward
                else:
                    score -= pen_bad
            if score > best_score:
                best_score = score
                best_right = k
            if best_score - score > xdrop:
                break
            k += 1

        # finalize candidate
        seg = summarize_segment(best_left, best_right, wins, ident_by_idx)
        if seg["n_windows"] >= min_windows and seg["covered_bp"] >= min_len_bp:
            segments.append(seg)
            for k in range(best_left, best_right+1):
                used[k] = True

    # merge overlapping/adjacent segments (optional minor cleanup)
    segments = merge_segments(segments, wins)
    return segments

def summarize_segment(s, e, wins, ident_by_idx):
    start_bp = wins[s].start; end_bp = wins[e].end
    n_windows = e - s + 1
    covered_bp = sum(w.length for w in wins[s:e+1])
    called_idents = [ident_by_idx[i] for i in range(s, e+1) if i in ident_by_idx]
    mean_ident = sum(called_idents)/len(called_idents) if called_idents else 0.0
    min_ident = min(called_idents) if called_idents else 0.0
    frac_called = len(called_idents)/n_windows if n_windows>0 else 0.0
    return {
        "chr": wins[s].chr,
        "start": start_bp,
        "end": end_bp,
        "n_windows": n_windows,
        "covered_bp": covered_bp,
        "mean_ident": mean_ident,
        "min_ident": min_ident,
        "frac_called": frac_called
    }

def merge_segments(segs, wins):
    if not segs: return []
    segs = sorted(segs, key=lambda s:(s["chr"], s["start"], s["end"]))
    out = [segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s["chr"]==last["chr"] and s["start"] <= last["end"]:
            # merge
            last["start"] = min(last["start"], s["start"])
            last["end"] = max(last["end"], s["end"])
            # recompute stats approximately via windows span (not re-averaging idents)
        else:
            out.append(s)
    return out

def finalize_segment(current, wins, called, ident_sum, min_ident,
                     min_windows, min_len_bp, gaps_allowed=0, **kwargs):
    s = current["start_idx"]; e = current["end_idx"]
    n_windows = e - s + 1
    start_bp = wins[s].start; end_bp = wins[e].end
    covered_bp = sum(w.length for w in wins[s:e+1])
    mean_ident = (ident_sum / called) if called>0 else 0.0
    frac_called = called / n_windows if n_windows>0 else 0.0

    if n_windows >= min_windows and covered_bp >= min_len_bp:
        return {
          "chr": wins[s].chr,
          "start": start_bp,
          "end": end_bp,
          "n_windows": n_windows,
          "covered_bp": covered_bp,
          "mean_ident": mean_ident,
          "min_ident": min_ident,
          "n_gaps": gaps_allowed,
          "frac_called": frac_called
        }
    return None

def main():
    ap = argparse.ArgumentParser(description="Call IBD segments from per-window pairwise identities")
    ap.add_argument("pairwise_tsv", help="Output of run_pairwise_impg.sh")
    ap.add_argument("--mode", choices=["rle","seed"], default="seed",
                    help="RLE thresholding (rle) or seed-and-extend (seed)")
    # thresholds for both modes
    ap.add_argument("--min-windows", type=int, default=3, help="Minimum number of windows in a segment")
    ap.add_argument("--min-length-bp", type=int, default=5000, help="Minimum segment length (bp)")
    ap.add_argument("--missing-as-gap", action="store_true", help="Count missing windows as gaps")
    # RLE params
    ap.add_argument("--min-identity", type=float, default=0.9995, help="Per-window identity threshold for RLE mode")
    ap.add_argument("--max-gap", type=int, default=1, help="Max consecutive sub-threshold/missing windows in RLE mode")
    ap.add_argument("--drop-tolerance", type=float, default=0.0, help="Allow slight identity dips below threshold (RLE)")
    # Seed-and-extend params
    ap.add_argument("--seed-threshold", type=float, default=0.9998, help="Seed identity threshold")
    ap.add_argument("--seed-k", type=int, default=2, help="Consecutive windows >= seed threshold to form a seed")
    ap.add_argument("--extend-threshold", type=float, default=0.9995, help="Extension identity threshold")
    ap.add_argument("--xdrop", type=float, default=2.0, help="X-drop (stop when running score falls this far from best)")
    ap.add_argument("--reward", type=float, default=1.0, help="Score added for a good (>= extend-threshold) window")
    ap.add_argument("--penalty-bad", type=float, default=1.0, help="Penalty for sub-threshold window during extension")
    ap.add_argument("--penalty-miss", type=float, default=1.0, help="Penalty for missing window during extension")
    ap.add_argument("--identity-col", default="estimated.identity", help="Identity column name")
    ap.add_argument("--pairs", help="Optional file with pairs A<TAB>B to restrict analysis")
    args = ap.parse_args()

    rows = parse_table(args.pairwise_tsv, identity_col=args.identity_col)
    if not rows:
        print("No valid rows", file=sys.stderr); sys.exit(1)
    wins_by_chr, index_by_chr = build_windows(rows)
    tracks = build_pair_tracks(rows, index_by_chr)

    pair_filter = None
    if args.pairs:
        pair_filter = set()
        with open(args.pairs) as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"): continue
                a,b = line.rstrip("\n").split("\t")[:2]
                pair_filter.add(pair_key(a,b))

    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    header = ["CHR","START","END","HAP1","HAP2","N_WINDOWS","COVERED_BP","MEAN_IDENTITY","MIN_IDENTITY","FRACTION_CALLED","MODE"]
    w.writerow(header)

    for (a,b), tracks_by_chr in tracks.items():
        if pair_filter and pair_key(a,b) not in pair_filter:
            continue
        for chr_, track in tracks_by_chr.items():
            wins = wins_by_chr[chr_]
            if args.mode == "rle":
                segs = rle_segments_for_pair(
                    track, wins,
                    min_id=args.min_identity,
                    max_gap=args.max_gap,
                    min_windows=args.min_windows,
                    min_len_bp=args.min_length_bp,
                    treat_missing_as_gap=args.missing_as_gap,
                    drop_tolerance=args.drop_tolerance
                )
            else:
                segs = seed_extend_segments_for_pair(
                    track, wins,
                    seed_thr=args.seed_threshold,
                    seed_k=args.seed_k,
                    ext_thr=args.extend_threshold,
                    xdrop=args.xdrop,
                    reward=args.reward,
                    pen_bad=args.penalty_bad,
                    pen_miss=args.penalty_miss,
                    min_windows=args.min_windows,
                    min_len_bp=args.min_length_bp,
                    treat_missing_as_gap=args.missing_as_gap
                )
            for s in segs:
                w.writerow([chr_, s["start"], s["end"], a, b,
                            s["n_windows"], s["covered_bp"],
                            f"{s['mean_ident']:.6f}", f"{s['min_ident']:.6f}",
                            f"{s['frac_called']:.3f}", args.mode])

if __name__ == "__main__":
    main()
