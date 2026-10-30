"""Aggregate r13b outputs: smoothing per family (paired, corrected t-test) and noise sensitivity."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from growth.common import RES
import glob, pickle, sys, json
import numpy as np
from scipy import stats
R = {}
for f in glob.glob(os.path.join(RES, "m_sn*_*.pkl")):
    for r in pickle.load(open(f, "rb")):
        R[r["seed"]] = r
R = [R[k] for k in sorted(R)]
print(len(R), "splits:", [r["seed"] for r in R])
ratio = 63 / 187
def ttest(d):
    d = np.asarray(d); J = len(d); sd = d.std(ddof=1); t = d.mean() / np.sqrt((1 / J + ratio) * sd ** 2)
    return float(2 * stats.t.sf(abs(t), J - 1))
out = {"n": len(R), "smooth": {}, "noise": {}}
for fam, ls in [("linear", [5, 10]), ("et_abs", [10]), ("et_delta", [10])]:
    for l in ls:
        for m in ["track", 6, 12, 24]:
            a = np.array([r["test"][f"{fam}|1|{l}"]["mean"] if m == "track" else r["test"][f"{fam}|1|{l}"]["h"][m] for r in R])
            b = np.array([r["test"][f"{fam}|1|1"]["mean"] if m == "track" else r["test"][f"{fam}|1|1"]["h"][m] for r in R])
            ok = np.isfinite(a) & np.isfinite(b); d = a[ok] - b[ok]
            out["smooth"][f"{fam}|l{l}|{m}"] = {"smoothed": float(a[ok].mean()), "raw": float(b[ok].mean()), "diff": float(d.mean()),
                                                "rel_pct": float(d.mean() / b[ok].mean() * 100), "wins": int((d < 0).sum()), "J": int(ok.sum()), "p": ttest(d)}
            print(f"{fam} l={l} vs l=1 [{m}] {a[ok].mean():.2f} vs {b[ok].mean():.2f} ({d.mean()/b[ok].mean()*100:+.1f}%) wins {(d<0).sum()}/{ok.sum()} p={ttest(d):.3g}")
for sd in ["0.02", "0.05", "0.1"]:
    for mode in ["iso", "indep"]:
        inc, within = [], []
        for r in R:
            clean = r["test"][r["selected"]]["mean"]; tr = np.array([x["mean"] for x in r["noise"][f"{sd}|{mode}"]])
            inc.append(tr.mean() - clean); within.append(tr.std(ddof=1))
        inc = np.array(inc)
        out["noise"][f"{sd}|{mode}"] = {"increase_mean": float(inc.mean()), "increase_sd_splits": float(inc.std()), "within_split_trial_sd": float(np.mean(within)),
                                        "increase_min": float(inc.min()), "increase_max": float(inc.max())}
        print(f"noise {sd} {mode}: +{inc.mean():.2f} mm (SD over splits {inc.std():.2f}, range {inc.min():.2f}..{inc.max():.2f}; SD over trials within split {np.mean(within):.3f})")
json.dump(out, open(os.path.join(RES, "m_smooth_noise_summary.json"), "w"), indent=1)
