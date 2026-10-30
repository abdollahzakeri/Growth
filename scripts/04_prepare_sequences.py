"""R02: build growth sequences + mushroom groups from a tracking pickle.

* sizes: major/minor diameters of the convex-hull Fitzgibbon ellipse (px); bbox kept for ablation
* regular 15-min grid: missed frames inside a track (<= tau_gap) are linearly interpolated
* truncation: stop at the first frame where the causally smoothed major diameter (l=10) falls more
  than DROP_FRAC below its running maximum (harvest / cap removal / persistent occlusion)
* minimum length MIN_LEN frames (5 h)
* groups: track fragments that belong to the same physical mushroom (a later track that starts where
  an earlier one ended, mask IoU >= 0.3) share a group id; data splits are done at the group level

Usage: python r02_prepare.py <tracks.pkl> <out_seqs.pkl> [drop_frac=0.10] [min_len=20]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle
import numpy as np
from growth.common import to_poly, causal_ma

INP, OUT = sys.argv[1], sys.argv[2]
DROP_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
MIN_LEN = int(sys.argv[4]) if len(sys.argv) > 4 else 20
GROUP_IOU = 0.3

D = pickle.load(open(INP, "rb")); tracks = D["tracks"]


def regrid(frames, vals):
    full = np.arange(frames[0], frames[-1] + 1)
    out = np.column_stack([np.interp(full, frames, vals[:, k]) for k in range(vals.shape[1])])
    return full, out


def clean_ell(t):
    e = t["ell"].astype(float).copy()
    bb = np.column_stack([np.maximum(t["bw"], t["bh"]), np.minimum(t["bw"], t["bh"])]).astype(float)
    bad = ~np.isfinite(e).all(1) | (e[:, 0] <= 0)
    e[bad] = bb[bad]
    return e, bb, int(bad.sum())


# ---------- groups (union-find) ----------
par = list(range(len(tracks)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
last_polys = [to_poly(t["last_pts"]) for t in tracks]
first_polys = [to_poly(t["first_pts"]) for t in tracks]
last_f = np.array([t["frame"][-1] for t in tracks])
LB = np.array([p.bounds if p is not None else (np.inf, np.inf, -np.inf, -np.inf) for p in last_polys])
for b in range(len(tracks)):
    pb = first_polys[b]
    if pb is None:
        continue
    fb = tracks[b]["frame"][0]
    x0, y0, x1, y1 = pb.bounds
    cand = np.where((last_f < fb) & (LB[:, 0] <= x1) & (LB[:, 2] >= x0) & (LB[:, 1] <= y1) & (LB[:, 3] >= y0))[0]
    for a in cand:
        pa = last_polys[a]
        inter = pa.intersection(pb).area
        if inter / (pa.area + pb.area - inter) >= GROUP_IOU:
            par[find(b)] = find(a)
gid = [find(i) for i in range(len(tracks))]

seqs, n_interp_total, n_obs_total, n_fit_fail = [], 0, 0, 0
for i, t in enumerate(tracks):
    e, bb, nbad = clean_ell(t)
    n_fit_fail += nbad
    fr = t["frame"].astype(int)
    full, eg = regrid(fr, e)
    _, bg = regrid(fr, bb)
    _, cg = regrid(fr, np.column_stack([t["cx"], t["cy"]]).astype(float))
    sm = causal_ma(eg[:, :1], 10)[:, 0]
    runmax = np.maximum.accumulate(sm)
    cut = np.where(sm < (1 - DROP_FRAC) * runmax)[0]
    T = cut[0] if len(cut) else len(full)
    if T < MIN_LEN:
        continue
    obs_mask = np.isin(full[:T], fr)
    n_interp_total += int((~obs_mask).sum()); n_obs_total += int(obs_mask.sum())
    seqs.append({"id": int(t["id"]), "group": int(gid[i]), "f0": int(full[0]), "d": eg[:T].astype(np.float32),
                 "bbox": bg[:T].astype(np.float32), "c": cg[:T].astype(np.float32),
                 "observed": obs_mask, "len": int(T)})

groups = {}
for s in seqs:
    groups.setdefault(s["group"], []).append(s)
lens = np.array([s["len"] for s in seqs])
stats = {"n_tracks_raw": len(tracks), "n_seqs": len(seqs), "n_groups": len(groups),
         "frag_per_group_mean": float(np.mean([len(v) for v in groups.values()])),
         "frag_per_group_max": int(max(len(v) for v in groups.values())),
         "groups_with_multiple_frag": int(sum(len(v) > 1 for v in groups.values())),
         "len_frames_median": float(np.median(lens)), "len_frames_mean": float(lens.mean()),
         "len_h_quantiles": {q: float(np.percentile(lens, q) * 0.25) for q in [10, 25, 50, 75, 90, 99]},
         "n_reach_h": {h: int((lens > h * 4).sum()) for h in [6, 12, 24, 48, 72]},
         "interp_frac": float(n_interp_total / max(n_interp_total + n_obs_total, 1)),
         "ellipse_fit_fail_frac": float(n_fit_fail / max(sum(len(t["frame"]) for t in tracks), 1)),
         "drop_frac": DROP_FRAC, "min_len": MIN_LEN, "meta": D["meta"]}
pickle.dump({"seqs": seqs, "stats": stats}, open(OUT, "wb"))
print({k: v for k, v in stats.items() if k != "meta"}, flush=True)
