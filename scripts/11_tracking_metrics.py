"""R07: tracking-quality statistics for one or more tracking pickles.

Per run: number of tracks, track duration, within-track miss rate (frames without a matched detection
inside a track's span), bridged gaps (re-association after 1..tau_gap missed frames), fragmentation
(strict chaining: a track that starts <= 24 h after another ended at the same location, mask IoU >= 0.5),
and an identity-switch proxy (consecutive associations with centroid jump > 0.25 x cap diameter or
area change > 30 %, i.e., physically implausible for a static, slowly growing cap).

Usage: python r07_tracking_metrics.py <out.json> <tracks1.pkl> [tracks2.pkl ...]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle
import numpy as np
from growth.common import to_poly, jdump, HRS_PER_FRAME

OUT = sys.argv[1]; res = {}
for path in sys.argv[2:]:
    D = pickle.load(open(path, "rb")); T = D["tracks"]; meta = D["meta"]
    span = np.array([t["frame"][-1] - t["frame"][0] + 1 for t in T]); nobs = np.array([len(t["frame"]) for t in T])
    long = span >= 20
    gaps = np.concatenate([np.diff(t["frame"]) - 1 for t in T])
    n_assoc = int(sum(len(t["frame"]) - 1 for t in T))
    jumps = 0; area_jumps = 0
    for t in T:
        d = np.hypot(np.diff(t["cx"]), np.diff(t["cy"]))
        diam = np.nan_to_num(t["ell"][:-1, 0], nan=float(np.nanmedian(t["ell"][:, 0])) if np.isfinite(t["ell"][:, 0]).any() else 50.0)
        jumps += int((d > 0.25 * np.maximum(diam, 1)).sum())
        ar = t["area"][1:] / np.maximum(t["area"][:-1], 1)
        area_jumps += int(((ar > 1.3) | (ar < 1 / 1.3)).sum())
    # strict fragmentation chains
    lastp = [to_poly(t["last_pts"]) for t in T]; firstp = [to_poly(t["first_pts"]) for t in T]
    lf = np.array([t["frame"][-1] for t in T]); ff = np.array([t["frame"][0] for t in T])
    LB = np.array([p.bounds if p is not None else (np.inf, np.inf, -np.inf, -np.inf) for p in lastp])
    succ = {}
    for b in np.where(long)[0]:
        pb = firstp[b]
        if pb is None:
            continue
        x0, y0, x1, y1 = pb.bounds
        cand = np.where(long & (lf < ff[b]) & (ff[b] - lf <= 96) & (LB[:, 0] <= x1) & (LB[:, 2] >= x0) & (LB[:, 1] <= y1) & (LB[:, 3] >= y0))[0]
        best, biou = None, 0.5
        for a in cand:
            pa = lastp[a]
            if pa is None or a in succ:
                continue
            inter = pa.intersection(pb).area; iou = inter / (pa.area + pb.area - inter)
            if iou >= biou:
                best, biou = a, iou
        if best is not None:
            succ[best] = b
    has_pred = set(succ.values()); chains = []
    for a in np.where(long)[0]:
        if a in has_pred:
            continue
        c = [a]
        while c[-1] in succ:
            c.append(succ[c[-1]])
        chains.append(c)
    clen = np.array([len(c) for c in chains])
    res[path.split("/")[-1]] = {
        "meta": meta, "n_tracks": len(T), "n_tracks_ge20": int(long.sum()),
        "duration_h_median_ge20": float(np.median(span[long]) * HRS_PER_FRAME),
        "duration_h_p90_ge20": float(np.percentile(span[long], 90) * HRS_PER_FRAME),
        "miss_rate_within_tracks_pct": float((span - nobs).sum() / span.sum() * 100),
        "miss_rate_within_tracks_ge20_pct": float((span[long] - nobs[long]).sum() / span[long].sum() * 100),
        "bridged_gaps": int((gaps > 0).sum()), "associations": n_assoc,
        "bridged_gap_pct_of_assoc": float((gaps > 0).sum() / max(n_assoc, 1) * 100),
        "idswitch_proxy_centroid_per_1000": float(jumps / max(n_assoc, 1) * 1000),
        "area_jump_per_1000": float(area_jumps / max(n_assoc, 1) * 1000),
        "chains": int(len(chains)), "fragments_per_chain_mean": float(clen.mean()),
        "chains_single_fragment_pct": float((clen == 1).mean() * 100),
    }
    print(path, {k: v for k, v in res[path.split("/")[-1]].items() if k != "meta"}, flush=True)
jdump(res, OUT)
print("DONE")
