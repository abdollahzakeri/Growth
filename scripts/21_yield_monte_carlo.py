"""R06c: tray-level yield from a single image, evaluated on MUSHROOM-level trajectories.
Fragments of the same cap (next fragment starts <= 24 h after the previous one ended at the same place, mask IoU >= 0.5)
are chained; gaps between fragments are linearly interpolated. A cap leaves the cohort only when its chain ends.
Forecast models are trained on the fragments with five-fold cross-fitting over the groups (as in R06); Monte Carlo
measurement and regression errors as in R06 (optional env R06_RESID, R06_CORR, R06_SCALE, R06_PARALLAX).
Usage: python r06c_yield_chain.py <seqs.pkl> <tracks.pkl> <family> <n> <l> <weight_model> <occlusion.json> <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, json, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone
from growth.common import ROOT, RES, PIX2MM, HRS_PER_FRAME, causal_ma, jdump, to_poly
from growth.weights import models as wmodels
SEQ, TRK, FAM, N, L, WM, OCC, OUT = sys.argv[1:9]; N, L = int(N), int(L)
HORIZ_H = [0, 3, 6, 12, 24, 48, 72]; STEP0 = 24; B = 400
CORR = os.environ.get("R06_CORR") == "1"; SCALE_SD = float(os.environ.get("R06_SCALE", "0"))
PAR = [float(v) for v in os.environ["R06_PARALLAX"].split(",")] if os.environ.get("R06_PARALLAX") else None
data = pickle.load(open(SEQ, "rb"))["seqs"]; T = pickle.load(open(TRK, "rb"))["tracks"]; tid = {t["id"]: i for i, t in enumerate(T)}
seqs = [s["d"].astype(float) for s in data]; f0 = np.array([s["f0"] for s in data]); lens = np.array([s["len"] for s in data])
groups = np.array([s["group"] for s in data])
# ---------------- chains of fragments (estimated physical mushrooms) ----------------
lp = [to_poly(T[tid[s["id"]]]["last_pts"]) for s in data]; fp = [to_poly(T[tid[s["id"]]]["first_pts"]) for s in data]
end = f0 + lens - 1; rel = []
for b in range(len(data)):
    if fp[b] is None: continue
    for a in np.where((end < f0[b]) & (f0[b] - end <= 96))[0]:
        if lp[a] is None: continue
        i = lp[a].intersection(fp[b]).area
        if i > 0:
            v = i / (lp[a].area + fp[b].area - i)
            if v >= 0.5: rel.append((v, a, b))
succ, pred = {}, {}
for v, a, b in sorted(rel, reverse=True):
    if a in succ or b in pred: continue
    succ[a] = b; pred[b] = a
chains = []
for a in range(len(data)):
    if a in pred: continue
    c = [a]
    while c[-1] in succ: c.append(succ[c[-1]])
    chains.append(c)
C = []
for c in chains:
    s0, e0 = f0[c[0]], end[c[-1]]; fr = np.arange(s0, e0 + 1); d = np.full((len(fr), 2), np.nan); obs = np.zeros(len(fr), bool)
    for k in c:
        d[f0[k] - s0: f0[k] - s0 + lens[k]] = seqs[k]; obs[f0[k] - s0: f0[k] - s0 + lens[k]] = True
    for j in range(2):
        m = np.isfinite(d[:, j]); d[:, j] = np.interp(fr, fr[m], d[m, j])
    C.append({"start": int(s0), "end": int(e0), "d": d, "obs": obs, "frag": c, "group": int(groups[c[0]])})
print(f"{len(data)} fragments -> {len(C)} chains (mushrooms); {sum(len(c['frag']) > 1 for c in C)} with >1 fragment", flush=True)
# ---------------- weight model & error pools ----------------
bd = pd.read_csv(os.path.join(ROOT, "data", "brown.csv"), encoding="utf-8-sig"); Xb, yb = bd[["x", "y"]].values.astype(float), bd["w"].values.astype(float)
wmodel = wmodels()[WM].fit(Xb, yb)
def weight(d_px):
    return np.clip(wmodel.predict(np.asarray(d_px, float).reshape(-1, 2) * PIX2MM), 0, None)
loo = np.zeros(len(yb))
for a, b in LeaveOneOut().split(Xb):
    loo[b] = yb[b] - clone(wmodels()[WM]).fit(Xb[a], yb[a]).predict(Xb[b])
rel_res, gm_res = loo / yb, np.sqrt(Xb[:, 0] * Xb[:, 1])
if os.environ.get("R06_RESID"):
    nj = json.load(open(os.environ["R06_RESID"]))["nested"]; rel_res = np.array(nj["residuals_nested"]) / np.array(nj["w"]); gm_res = np.array(nj["gm"])
rel_res = np.clip(rel_res, -0.9, 0.9)
occ = json.load(open(OCC))["representative"]["signed_hull"]
e_meas = np.concatenate([np.array(occ[k], float) for k in ["10", "20", "30", "40"]]); e_meas = e_meas[np.isfinite(e_meas)]
# ---------------- cross-fitted models (folds over groups, as in R06) ----------------
rng = np.random.default_rng(0); ug = np.unique(groups); rng.shuffle(ug); fold_of = {g: i % 5 for i, g in enumerate(ug)}
Hmax = int(max(HORIZ_H) / HRS_PER_FRAME); starts = np.arange(0, end.max() + 1, STEP0)
pairs = [(ci, int(t0)) for t0 in starts for ci, c in enumerate(C) if c["start"] <= t0 <= c["end"] and c["obs"][t0 - c["start"]]]
pairs = np.array(pairs); pfold = np.array([fold_of[C[ci]["group"]] for ci in pairs[:, 0]])
print(f"{len(starts)} start frames, {len(pairs)} (cap, t0) pairs", flush=True)
pred = np.full((len(pairs), Hmax + 1, 2), np.nan)
for k in range(5):
    X, Y = [], []
    for s, g in zip(seqs, groups):
        if fold_of[g] == k: continue
        sm = causal_ma(s, L) if L > 1 else s
        for i in range(len(sm) - N):
            w = sm[i:i + N]; X.append(w.ravel()); Y.append(sm[i + N] - (w[-1] if FAM == "et_delta" else 0))
    X, Y = np.asarray(X), np.asarray(Y)
    m = None if FAM == "persistence" else (Ridge(alpha=1e-3).fit(X, Y) if FAM == "linear" else ExtraTreesRegressor(n_estimators=100, random_state=k, n_jobs=8).fit(X, Y))
    idx = np.where(pfold == k)[0]
    W = np.stack([C[pairs[j, 0]]["d"][pairs[j, 1] - C[pairs[j, 0]]["start"]][None, :] for j in idx]).astype(float)
    pred[idx, 0] = W[:, -1]
    for h in range(1, Hmax + 1):
        nx = W[:, -1] if m is None else (W[:, -1] + m.predict(W.reshape(len(idx), -1)) if FAM == "et_delta" else m.predict(W.reshape(len(idx), -1)))
        pred[idx, h] = nx; W = np.concatenate([W[:, 1:], nx[:, None, :]], axis=1)
    print(f"fold {k} done", flush=True)
mrng = np.random.default_rng(1); gm_bins = [0, 20, 35, 50, 200]
def mc(obs, prd):
    wp = weight(prd); gm = np.sqrt(np.prod(obs * PIX2MM, axis=1)); bidx = np.digitize(gm, gm_bins) - 1; draws = []
    for b in range(B // 4):
        em = (np.full(len(obs), mrng.choice(e_meas)) if CORR else mrng.choice(e_meas, size=len(obs))) / 100.0
        sc = mrng.normal(0, SCALE_SD) if SCALE_SD > 0 else 0.0
        d_true = obs / (1 + em[:, None]) / (1 + sc)
        if PAR is not None:
            d_true = d_true * (1 - mrng.uniform(PAR[0], PAR[1], size=len(obs))[:, None] / 419.0)
        er = np.empty(len(obs))
        for bi in np.unique(bidx):
            pool = rel_res[(np.digitize(gm_res, gm_bins) - 1) == bi]; pool = pool if len(pool) >= 5 else rel_res
            er[bidx == bi] = mrng.choice(pool, size=(bidx == bi).sum())
        w_true = weight(d_true) * (1 + er); draws.append((wp.sum() - w_true.sum()) / w_true.sum() * 100)
    return draws
cend = np.array([c["end"] for c in C]); cstart = np.array([c["start"] for c in C])
out = {"config": {"family": FAM, "n": N, "l": L, "weight_model": WM, "B": B, "corr": CORR, "scale_sd": SCALE_SD, "parallax": PAR, "resid": os.environ.get("R06_RESID")},
       "n_chains": len(C), "n_fragments": len(data), "n_start_frames": int(len(starts)), "horizons": {}, "cohort": {}}
for mode in ["present", "cohort"]:
    for hh in HORIZ_H:
        h = int(hh / HRS_PER_FRAME); rows = []
        for t0 in starts:
            sel = np.where(pairs[:, 1] == t0)[0]
            rem = cend[pairs[sel, 0]] - t0
            if mode == "present":
                sel = sel[rem >= h]; hl = np.full(len(sel), h)
            else:
                hl = np.minimum(h, rem)
            if len(sel) < 5: continue
            obs = np.stack([C[pairs[j, 0]]["d"][t0 + k - C[pairs[j, 0]]["start"]] for j, k in zip(sel, hl)]); prd = pred[sel, hl]
            new = np.where((cstart > t0) & (cstart <= t0 + h) & (cend >= t0 + h))[0]
            w_new = weight(np.stack([C[i]["d"][t0 + h - C[i]["start"]] for i in new])).sum() if len(new) else 0.0
            rows.append((t0, sel, obs, prd, w_new, float(np.mean(hl < h))))
        if not rows: continue
        det, mcv, coh, tw, ce, nf, hv = [], [], [], [], [], [], []
        for t0, sel, obs, prd, w_new, hf in rows:
            wp, wo = weight(prd), weight(obs)
            det.append((wp.sum() - wo.sum()) / wo.sum() * 100); ce.extend((wp - wo).tolist()); coh.append(len(sel)); tw.append(wo.sum())
            nf.append(w_new / (w_new + wo.sum()) * 100); hv.append(hf); mcv.extend(mc(obs, prd))
        det, mcv = np.array(det), np.array(mcv)
        res = {"n_start_frames": len(rows), "cohort_mean": float(np.mean(coh)), "tray_g_mean": float(np.mean(tw)),
               "forecast_only_abs_mean_pct": float(np.mean(np.abs(det))), "forecast_only_signed_mean_pct": float(det.mean()),
               "mc_abs_mean_pct": float(np.mean(np.abs(mcv))), "mc_signed_mean_pct": float(mcv.mean()),
               "mc_ci95_pct": [float(np.percentile(mcv, 2.5)), float(np.percentile(mcv, 97.5))], "mc_abs_p95_pct": float(np.percentile(np.abs(mcv), 95)),
               "mc_abs_sd_over_starts": float(np.std([np.mean(np.abs(mcv[i * (B // 4):(i + 1) * (B // 4)])) for i in range(len(rows))])),
               "cap_rmse_g": float(np.sqrt(np.mean(np.square(ce)))), "new_cap_biomass_pct_mean": float(np.mean(nf)), "harvested_before_h_frac": float(np.mean(hv)),
               "series": [{"t0": int(t0), "pred_g": float(weight(prd).sum()), "obs_g": float(weight(obs).sum()), "n": int(len(sel))} for t0, sel, obs, prd, _, _ in rows]}
        (out["horizons"] if mode == "present" else out["cohort"])[str(hh)] = res
        print(f"[{mode}] h={hh:2d} starts={len(rows)} caps={res['cohort_mean']:.1f} tray={res['tray_g_mean']:.0f} g det|err|={res['forecast_only_abs_mean_pct']:.1f}% "
              f"signed={res['forecast_only_signed_mean_pct']:+.1f}% MC|err|={res['mc_abs_mean_pct']:.1f}% p95={res['mc_abs_p95_pct']:.1f} MCsigned={res['mc_signed_mean_pct']:+.1f}% "
              f"ended={res['harvested_before_h_frac']*100:.0f}% new={res['new_cap_biomass_pct_mean']:.1f}%", flush=True)
best_t0 = max(starts, key=lambda t: (pairs[:, 1] == t).sum()); sel = np.where(pairs[:, 1] == best_t0)[0]
ex = {"t0": int(best_t0), "t0_hours": float(best_t0 * HRS_PER_FRAME), "n": int(len(sel)), "hours": [], "pred_g": [], "obs_g": [], "true_lo": [], "true_hi": [], "n_still_visible": []}
for h in range(0, Hmax + 1, 4):
    hl = np.minimum(h, cend[pairs[sel, 0]] - best_t0)
    obs = np.stack([C[pairs[j, 0]]["d"][best_t0 + k - C[pairs[j, 0]]["start"]] for j, k in zip(sel, hl)]); wp = weight(pred[sel, hl]).sum(); dr = []
    for b in range(200):
        em = mrng.choice(e_meas, size=len(obs)) / 100.0; er = mrng.choice(rel_res, size=len(obs)); dr.append((weight(obs / (1 + em[:, None])) * (1 + er)).sum())
    ex["hours"].append(h * HRS_PER_FRAME); ex["pred_g"].append(float(wp)); ex["obs_g"].append(float(weight(obs).sum()))
    ex["true_lo"].append(float(np.percentile(dr, 2.5))); ex["true_hi"].append(float(np.percentile(dr, 97.5))); ex["n_still_visible"].append(int((hl == h).sum()))
out["example"] = ex; jdump(out, OUT); print("DONE")
