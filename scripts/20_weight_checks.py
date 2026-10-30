"""R15: (a) predictions from original vs. swapped diameters for the same test caps (models trained WITHOUT
augmentation, original 75/25 split); (b) cross-validated residuals of the baby bella weight model with model
selection repeated inside every outer fold (nested), compared with leave-one-out residuals of the fixed
K-nearest-neighbors model. Usage: python r15_weight_extra.py <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, LeaveOneOut
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from growth.common import ROOT, RES, jdump
from growth.weights import models as fixed_models
warnings.filterwarnings("ignore")
from lazypredict.Supervised import LazyRegressor, REGRESSORS
POOL = {n: c for n, c in REGRESSORS}
OUT = sys.argv[1]; res = {"swap": {}, "nested": {}}
load = lambda n: pd.read_csv(os.path.join(ROOT, "data", f"{n}.csv"), encoding="utf-8-sig")[["x", "y", "w"]]
for var in ["white", "brown"]:
    D = load(var); tr, te = train_test_split(D, test_size=0.25, random_state=42)
    X, y = tr[["x", "y"]].values, tr["w"].values; Xt = te[["x", "y"]].values; yt = te["w"].values
    for mn, m0 in fixed_models().items():
        m = clone(m0).fit(X, y); p = m.predict(Xt); ps = m.predict(Xt[:, ::-1]); d = ps - p
        res["swap"][f"{var}|{mn}"] = {"mean_abs_diff_g": float(np.mean(np.abs(d))), "max_abs_diff_g": float(np.max(np.abs(d))),
                                      "mean_abs_diff_pct": float(np.mean(np.abs(d) / np.maximum(p, 0.5)) * 100),
                                      "rmse_orig": float(np.sqrt(np.mean((p - yt) ** 2))), "rmse_swapped": float(np.sqrt(np.mean((ps - yt) ** 2))),
                                      "rmse_avg": float(np.sqrt(np.mean(((p + ps) / 2 - yt) ** 2))),
                                      "test_mean_abs_xy_diff_mm": float(np.mean(np.abs(Xt[:, 0] - Xt[:, 1])))}
        print(var, mn, res["swap"][f"{var}|{mn}"], flush=True)

# (b) nested residuals, brown (variety of the monitored bed)
B = load("brown"); Xb, yb = B[["x", "y"]], B["w"].values; gm = np.sqrt(B.x.values * B.y.values)
loo = np.zeros(len(yb)); knn = fixed_models()["K-nearest neighbors"]
for a, b in LeaveOneOut().split(Xb):
    loo[b] = yb[b] - clone(knn).fit(Xb.values[a], yb[a]).predict(Xb.values[b])
nested = np.zeros(len(yb)); chosen = []
for rep in range(2):
    pr = np.zeros(len(yb))
    for k, (a, b) in enumerate(KFold(n_splits=10, shuffle=True, random_state=rep).split(Xb)):
        tr = B.iloc[a]; tabs = []
        for ia, ib in RepeatedKFold(n_splits=5, n_repeats=5, random_state=0).split(tr):
            t, _ = LazyRegressor(verbose=0, ignore_warnings=True, predictions=False).fit(tr.iloc[ia][["x", "y"]], tr.iloc[ib][["x", "y"]], tr.iloc[ia]["w"], tr.iloc[ib]["w"])
            tabs.append(t[["RMSE"]])
        cat = pd.concat(tabs).groupby(level=0)["RMSE"]; rk = pd.DataFrame({"m": cat.mean(), "n": cat.count()}).query("n >= 20").sort_values("m")
        best = rk.index[0]; chosen.append(best)
        mdl = POOL[best]()
        try:
            mdl.set_params(random_state=42)
        except Exception:
            pass
        pipe = Pipeline([("s", StandardScaler()), ("m", mdl)]).fit(tr[["x", "y"]], tr["w"])
        pr[b] = pipe.predict(B.iloc[b][["x", "y"]])
        print("rep", rep, "fold", k, best, flush=True)
    nested += (yb - pr) / 2
bins = [0, 20, 35, 50, 200]
def binstats(r):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (gm >= lo) & (gm < hi)
        if s.sum():
            out.append({"bin": [lo, hi], "n": int(s.sum()), "rmse": float(np.sqrt(np.mean(r[s] ** 2))), "rel_sd_pct": float(np.std(r[s] / yb[s]) * 100),
                        "rel_mean_pct": float(np.mean(r[s] / yb[s]) * 100)})
    return out
res["nested"] = {"loo_knn_rmse": float(np.sqrt(np.mean(loo ** 2))), "nested_rmse": float(np.sqrt(np.mean(nested ** 2))),
                 "loo_bins": binstats(loo), "nested_bins": binstats(nested), "chosen": pd.Series(chosen).value_counts().to_dict(),
                 "residuals_nested": nested.tolist(), "residuals_loo": loo.tolist(), "w": yb.tolist(), "gm": gm.tolist()}
print({k: v for k, v in res["nested"].items() if not k.startswith("resid") and k not in ("w", "gm")}, flush=True)
jdump(res, OUT)
print("DONE")
