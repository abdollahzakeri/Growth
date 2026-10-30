"""R11: summarize the forecasting study across split seeds into tables (JSON) for the manuscript.
Usage: python r11_summarize.py "<glob of fc pickles>" <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, glob, pickle, json, collections
import numpy as np
from growth.common import jdump
R = []
for f in sorted(glob.glob(sys.argv[1])): R += pickle.load(open(f, "rb"))
H = [1, 3, 6, 12, 24, 48, 72]
out = {"n_seeds": len(R), "n_test_tracks": [float(np.mean([r["n_test"] for r in R])), float(np.std([r["n_test"] for r in R]))],
       "n_groups": np.mean([r["n_groups"] for r in R], axis=0).tolist()}


def hz(entry, h):
    k = h * 4 - 1
    return np.sqrt(entry["h_sq"][k] / entry["h_cnt"][k]) if entry["h_cnt"][k] >= 3 else np.nan


def block(get):
    vals = [get(r) for r in R]
    tm = [v["mean"] for v in vals]; md = [v["median"] for v in vals]
    res = {"track_mean": [float(np.mean(tm)), float(np.std(tm))], "track_median": [float(np.mean(md)), float(np.std(md))], "h": {}}
    for h in H:
        x = np.array([hz(v, h) for v in vals], float)
        n = [int(v["h_cnt"][h * 4 - 1]) for v in vals]
        # pooled over seeds (sum of squared errors over all test tracks of all splits)
        sq = sum(v["h_sq"][h * 4 - 1] for v in vals); cn = sum(v["h_cnt"][h * 4 - 1] for v in vals)
        res["h"][str(h)] = {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x)), "n_mean": float(np.mean(n)),
                            "n_seeds_valid": int(np.isfinite(x).sum()), "pooled": float(np.sqrt(sq / cn)) if cn else None, "n_pooled": int(cn)}
    return res


out["models"] = {nm: block(lambda r, nm=nm: r["test"][nm]) for nm in
                 ["persistence_n1", "linear_single", "et_abs_single", "et_delta_single", "paper_et_abs_n1_l10", "overall", "overall_l1", "overall_any"]}
out["selection_counts"] = {k: dict(collections.Counter(r["chosen"][k] for r in R)) for k in R[0]["chosen"]}
# smoothing gain (validation-selected single-image model vs the same model without smoothing)
g = [(r["test"]["overall_l1"]["mean"] - r["test"]["overall"]["mean"]) / r["test"]["overall_l1"]["mean"] * 100 for r in R
     if r["test"]["overall"]["cfg"] != r["test"]["overall_l1"]["cfg"]]
out["smoothing_gain_pct"] = [float(np.mean(g)), float(np.std(g)), len(g)] if g else None
gp = [(r["val_grid"]["et_abs|1|1"]["mean"] - r["val_grid"]["et_abs|1|10"]["mean"]) / r["val_grid"]["et_abs|1|1"]["mean"] * 100 for r in R]
out["smoothing_gain_et_val_pct"] = [float(np.mean(gp)), float(np.std(gp))]
gt = []
for r in R:
    pass
# validation grid (mean over seeds)
keys = R[0]["val_grid"].keys()
out["val_grid"] = {k: [float(np.mean([r["val_grid"][k]["mean"] for r in R])), float(np.std([r["val_grid"][k]["mean"] for r in R]))] for k in keys}
# re-anchoring (overall single-image config) and persistence
ra = {}
for R_ in ["None", "96", "48", "24"]:
    for tag, sel in [("model", lambda r: r["chosen"]["overall"]), ("persistence", lambda r: "persistence|1|1")]:
        ra[f"{tag}|R{R_}"] = block(lambda r, R_=R_, sel=sel: r["reanchor"][f"{sel(r)}|R{R_}"])
out["reanchor"] = ra
nz = {}
for sd in ["0.02", "0.05", "0.1"]:
    nz[sd] = block(lambda r, sd=sd: r["noise"][f"{r['chosen']['overall']}|{sd}"])
    nz[sd + "_persistence"] = block(lambda r, sd=sd: r["noise"][f"persistence|1|1|{sd}"])
out["noise"] = nz
jdump(out, sys.argv[2])
m = out["models"]
print(f"{out['n_seeds']} seeds; test tracks {out['n_test_tracks']}")
for nm, v in m.items():
    print(f"{nm:22s} track {v['track_mean'][0]:.2f}±{v['track_mean'][1]:.2f}  " + "  ".join(f"{h}h {v['h'][str(h)]['mean']:.1f}±{v['h'][str(h)]['sd']:.1f}" for h in H))
print("n per horizon:", {h: round(m['persistence_n1']['h'][str(h)]['n_mean'], 1) for h in H})
print("selection:", out["selection_counts"]["overall"])
print("smoothing gain (selected vs l=1, test):", out["smoothing_gain_pct"], " ET l10 vs l1 on validation:", out["smoothing_gain_et_val_pct"])
for k, v in ra.items():
    print("reanchor", k, f"track {v['track_mean'][0]:.2f}", "  ".join(f"{h}h {v['h'][str(h)]['mean']:.1f}" for h in [24, 48, 72]))
for k, v in nz.items():
    print("noise", k, f"track {v['track_mean'][0]:.2f}")
