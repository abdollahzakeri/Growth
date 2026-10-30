"""R13: statistics requested in the co-author review.
(1) paired comparisons of forecasting models across the 30 repeated group-level splits
    (corrected resampled t-test of Nadeau & Bengio 2003, Wilcoxon signed-rank, win counts);
(2) groups / trajectories / estimated physical mushrooms per partition;
(3) fragment-linking statistics and cross-split proximity check;
(4) scale-consistency counts; (5) how cohort trajectories end (harvest vs. track loss).
Usage: python r13_stats.py <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, glob, pickle
import numpy as np
from scipy import stats
from growth.common import RES, to_poly, jdump, PIX2MM, causal_ma, ROOT
import pandas as pd
OUT = sys.argv[1]
R = []
for f in sorted(glob.glob(os.path.join(RES, "fc_t090_*.pkl"))):
    R += pickle.load(open(f, "rb"))
R = sorted(R, key=lambda r: r["seed"])
res = {"n_seeds": len(R)}
H = [6, 12, 24]


def hz(e, h):
    k = h * 4 - 1
    return np.sqrt(e["h_sq"][k] / e["h_cnt"][k]) if e["h_cnt"][k] >= 3 else np.nan


def metric(name, m):
    e = [r["test"][name] for r in R]
    return np.array([x["mean"] for x in e]) if m == "track" else np.array([hz(x, int(m)) for x in e])


ratio = np.mean([r["n_groups"][2] / r["n_groups"][0] for r in R])   # test/train groups
res["test_train_ratio"] = float(ratio)


def compare(a, b, m):
    x, y = metric(a, m), metric(b, m); ok = np.isfinite(x) & np.isfinite(y); d = x[ok] - y[ok]; J = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / np.sqrt((1 / J + ratio) * sd ** 2) if sd > 0 else np.inf
    p_c = 2 * stats.t.sf(abs(t), J - 1)
    t_n = d.mean() / (sd / np.sqrt(J)) if sd > 0 else np.inf
    try:
        p_w = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_w = 1.0
    return {"a": a, "b": b, "metric": m, "J": J, "mean_a": float(x[ok].mean()), "mean_b": float(y[ok].mean()),
            "diff_mean": float(d.mean()), "diff_sd": float(sd), "rel_diff_pct": float(d.mean() / y[ok].mean() * 100),
            "wins_a": int((d < 0).sum()), "t_corrected": float(t), "p_corrected": float(p_c),
            "p_naive_t": float(2 * stats.t.sf(abs(t_n), J - 1)), "p_wilcoxon": p_w}


pairs = [("overall", "persistence_n1"), ("linear_single", "persistence_n1"), ("paper_et_abs_n1_l10", "persistence_n1"),
         ("linear_single", "paper_et_abs_n1_l10"), ("overall", "paper_et_abs_n1_l10"), ("overall", "overall_l1"),
         ("overall_any", "overall"), ("et_delta_single", "paper_et_abs_n1_l10")]
res["paired"] = [compare(a, b, m) for a, b in pairs for m in ["track"] + [str(h) for h in H]]
for c in res["paired"]:
    print(f"{c['a']:20s} vs {c['b']:20s} [{c['metric']:>5s}] {c['mean_a']:.2f} vs {c['mean_b']:.2f}  d={c['diff_mean']:+.2f}±{c['diff_sd']:.2f} "
          f"({c['rel_diff_pct']:+.1f}%) wins {c['wins_a']}/{c['J']}  p_corr={c['p_corrected']:.3g} p_naive={c['p_naive_t']:.3g} p_wil={c['p_wilcoxon']:.3g}")
# smoothing: same configuration with and without smoothing (only seeds where l differs)
dd = [(r["test"]["overall_l1"]["mean"], r["test"]["overall"]["mean"], r["chosen"]["overall"]) for r in R]
res["smoothing_pairs"] = dd

# ---------------- groups, trajectories, mushrooms ----------------
S = pickle.load(open(os.path.join(RES, "seqs_t090.pkl"), "rb"))["seqs"]
T = pickle.load(open(os.path.join(RES, "tracks_t090.pkl"), "rb"))["tracks"]
tid = {t["id"]: i for i, t in enumerate(T)}
groups = np.array([s["group"] for s in S])
lastp = [to_poly(t["last_pts"]) for t in T]; firstp = [to_poly(t["first_pts"]) for t in T]
lf = np.array([t["frame"][-1] for t in T]); ff = np.array([t["frame"][0] for t in T])
LB = np.array([p.bounds if p is not None else (np.inf, np.inf, -np.inf, -np.inf) for p in lastp])


def iou(a, b):
    if a is None or b is None:
        return 0.0
    i = a.intersection(b).area
    return i / (a.area + b.area - i) if i > 0 else 0.0


# candidate predecessor relations among sequence tracks
sid = [tid[s["id"]] for s in S]; sset = set(sid); s_of_t = {t: k for k, t in enumerate(sid)}
rel = []          # (seq_a, seq_b, iou, gap_frames)
for kb, b in enumerate(sid):
    pb = firstp[b]
    if pb is None:
        continue
    x0, y0, x1, y1 = pb.bounds
    cand = np.where((lf < ff[b]) & (LB[:, 0] <= x1) & (LB[:, 2] >= x0) & (LB[:, 1] <= y1) & (LB[:, 3] >= y0))[0]
    for a in cand:
        if a in sset:
            v = iou(lastp[a], pb)
            if v > 0:
                rel.append((s_of_t[a], kb, v, int(ff[b] - lf[a])))
rel = np.array(rel, float)
same = groups[rel[:, 0].astype(int)] == groups[rel[:, 1].astype(int)]
link = rel[rel[:, 2] >= 0.3]
res["links"] = {"n_links_ge03": int(len(link)), "iou_quantiles": {q: float(np.percentile(link[:, 2], q)) for q in [10, 25, 50, 75]},
                "gap_h_quantiles": {q: float(np.percentile(link[:, 3] * 0.25, q)) for q in [10, 25, 50, 75, 90]},
                "gap_le_24h_pct": float((link[:, 3] <= 96).mean() * 100)}
# strict chains -> estimated physical mushrooms (start <= 24 h after end, IoU >= 0.5, one successor each)
strict = rel[(rel[:, 2] >= 0.5) & (rel[:, 3] <= 96)]
order = np.argsort(-strict[:, 2]); succ, pred = {}, {}
for a, b, v, g in strict[order]:
    a, b = int(a), int(b)
    if a in succ or b in pred:
        continue
    succ[a] = b; pred[b] = a
root = np.arange(len(S))
for b in sorted(pred, key=lambda k: S[k]["f0"]):
    root[b] = root[pred[b]]
res["mushrooms"] = {"n_seqs": len(S), "n_groups": int(len(np.unique(groups))), "n_est_mushrooms": int(len(np.unique(root))),
                    "seqs_per_mushroom": float(len(S) / len(np.unique(root))),
                    "mushrooms_per_group_mean": float(len(np.unique(root)) / len(np.unique(groups))),
                    "mushrooms_per_group_median": float(np.median(pd.Series(root).groupby(groups).nunique()))}
# cross-group near misses (possible missed links): best IoU with an earlier sequence of ANOTHER group, gap <= 24 h
cross = rel[(~same) & (rel[:, 3] <= 96)]
res["cross_group"] = {"pairs_iou_gt0_gap_le24h": int(len(cross)), "max_iou": float(cross[:, 2].max()) if len(cross) else 0.0,
                      "n_iou_ge_0_2": int((cross[:, 2] >= 0.2).sum()), "n_iou_ge_0_1": int((cross[:, 2] >= 0.1).sum()),
                      "n_seqs": len(S)}
# per split: groups, trajectories, mushrooms in test; leakage proximity check (test start near a train end or v.v.)
per = []
for r in R:
    rng = np.random.default_rng(r["seed"]); ug = np.unique(groups); rng.shuffle(ug)
    n1, n2 = int(0.6 * len(ug)), int(0.8 * len(ug))
    gtr, gte = set(ug[:n1]), set(ug[n2:])
    te = np.isin(groups, list(gte)); tr = np.isin(groups, list(gtr))
    near = rel[(rel[:, 3] <= 96) & (((te[rel[:, 0].astype(int)]) & (tr[rel[:, 1].astype(int)])) | ((tr[rel[:, 0].astype(int)]) & (te[rel[:, 1].astype(int)])))]
    per.append({"seed": r["seed"], "test_groups": int(len(gte)), "test_seqs": int(te.sum()), "test_mushrooms": int(len(np.unique(root[te]))),
                "train_groups": int(len(gtr)), "train_seqs": int(tr.sum()),
                "cross_split_pairs_iou_gt0_le24h": int(len(near)), "cross_split_max_iou": float(near[:, 2].max()) if len(near) else 0.0})
res["per_split"] = per
for k in ["test_groups", "test_seqs", "test_mushrooms", "cross_split_pairs_iou_gt0_le24h", "cross_split_max_iou"]:
    v = np.array([p[k] for p in per], float); res.setdefault("per_split_summary", {})[k] = [float(v.mean()), float(v.std()), float(v.min()), float(v.max())]
print(res["links"], res["mushrooms"], res["cross_group"], res["per_split_summary"], flush=True)

# ---------------- how sequences end ----------------
ends_trunc = np.array([s["len"] < (T[tid[s["id"]]]["frame"][-1] - s["f0"] + 1) for s in S])
has_succ = np.zeros(len(S), bool)
for a in range(len(S)):
    ta = sid[a]; pa = lastp[ta]
    if pa is None:
        continue
    x0, y0, x1, y1 = pa.bounds
    for b in np.where((ff > lf[ta]) & (ff - lf[ta] <= 96))[0]:
        pb = firstp[b]
        if pb is None:
            continue
        bx0, by0, bx1, by1 = pb.bounds
        if bx0 > x1 or bx1 < x0 or by0 > y1 or by1 < y0:
            continue
        if iou(pa, pb) >= 0.5:
            has_succ[a] = True; break
res["seq_end"] = {"truncated_by_size_drop_pct": float(ends_trunc.mean() * 100),
                  "track_continues_within_24h_pct": float(has_succ.mean() * 100),
                  "continues_and_not_truncated_pct": float((has_succ & ~ends_trunc).mean() * 100)}
print(res["seq_end"], flush=True)

# ---------------- scale-consistency counts ----------------
long = [s for s in S if s["len"] >= 96]
mx = np.array([np.sqrt(np.prod(s["d"][np.argmax(s["d"][:, 0])])) * PIX2MM for s in long])
mxs = np.array([np.max(np.sqrt(np.prod(causal_ma(s["d"].astype(float), 10), axis=1))) * PIX2MM for s in long])
b = pd.read_csv(f"{ROOT}/brown.csv", encoding="utf-8-sig"); gm = np.sqrt(b.x * b.y)
Q = [10, 25, 50, 75, 90]
res["scale_check"] = {"n_seqs_ge24h": len(long), "n_groups_ge24h": int(len({s['group'] for s in long})),
                      "n_mushrooms_ge24h": int(len({root[i] for i, s in enumerate(S) if s['len'] >= 96})),
                      "raw_max": {q: float(np.percentile(mx, q)) for q in Q}, "smoothed_max": {q: float(np.percentile(mxs, q)) for q in Q},
                      "caliper": {q: float(np.percentile(gm, q)) for q in Q}, "n_caliper": int(len(gm)),
                      "ks_raw": [float(x) for x in stats.ks_2samp(mx, gm)], "ks_smoothed": [float(x) for x in stats.ks_2samp(mxs, gm)]}
print(res["scale_check"], flush=True)
jdump(res, OUT)
print("DONE")
