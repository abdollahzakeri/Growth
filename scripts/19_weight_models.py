"""R05b: weight models on the ORIGINAL split (train_test_split test_size=0.25, random_state=42),
with model selection kept separate from testing: the regressor for every scenario is selected by
5x5 repeated cross-validation on the TRAINING data only (LazyPredict pool), then evaluated once on the
held-out test set. 95% CIs by bootstrap over test caps. ME = mean(predicted - actual).
Also: the same fixed model set with and without width-height swapping on identical data (R1.9),
pooled training on both varieties with a variety indicator, and size-binned test errors.

Usage: python r05b_weight_nested.py <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.base import clone
from growth.common import ROOT, RES, jdump
from growth.weights import models as fixed_models, swap_aug
warnings.filterwarnings("ignore")
from lazypredict.Supervised import LazyRegressor, REGRESSORS
OUT = sys.argv[1]
POOL = {name: cls for name, cls in REGRESSORS}


def load(n):
    return pd.read_csv(os.path.join(ROOT, "data", f"{n}.csv"), encoding="utf-8-sig")[["x", "y", "w"]]


def aug(df):
    s = df.rename(columns={"x": "y", "y": "x"})[["x", "y", "w"]]
    return pd.concat([df, s], ignore_index=True)


def boot_ci(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); r, m = [], []
    for _ in range(B):
        i = rng.integers(0, n, n); e = p[i] - y[i]; r.append(np.sqrt(np.mean(e ** 2))); m.append(e.mean())
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def lazy_cv_rank(train, feats, aug_train):
    """5x5 repeated CV on the training data; augmentation applied to inner training folds only."""
    tabs = []
    for k, (a, b) in enumerate(RepeatedKFold(n_splits=5, n_repeats=5, random_state=0).split(train)):
        tr, va = train.iloc[a], train.iloc[b]
        if aug_train:
            tr = aug(tr)
        lr = LazyRegressor(verbose=0, ignore_warnings=True, predictions=False)
        t, _ = lr.fit(tr[feats], va[feats], tr["w"], va["w"])
        tabs.append(t[["RMSE"]])
    cat = pd.concat(tabs).groupby(level=0)["RMSE"]
    return pd.DataFrame({"cv_rmse": cat.mean(), "cv_sd": cat.std(), "n": cat.count()}).query("n >= 20").sort_values("cv_rmse")


def fit_predict(name, train, test, feats):
    m = POOL[name]()
    try:
        m.set_params(random_state=42)
    except Exception:
        pass
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("s", StandardScaler()), ("m", m)])
    pipe.fit(train[feats], train["w"])
    return pipe.predict(test[feats])


W, Bd = load("white"), load("brown")
Wtr, Wte = train_test_split(W, test_size=0.25, random_state=42)
Btr, Bte = train_test_split(Bd, test_size=0.25, random_state=42)
res = {"split": {"white": [len(Wtr), len(Wte)], "brown": [len(Btr), len(Bte)]}, "tables": {}, "fixed": {}, "pooled": {}, "lazy_cv": {}}
scen = {"White->White": (Wtr, Wte), "Brown->Brown": (Btr, Bte), "White->Brown": (W, Bd), "Brown->White": (Bd, W)}
for feats_name, feats in [("xy", ["x", "y"]), ("x", ["x"]), ("y", ["y"])]:
    for augm in [False, True]:
        for sname, (tr, te) in scen.items():
            rank = lazy_cv_rank(tr, feats, augm)
            best = rank.index[0]
            trf = aug(tr) if augm else tr
            p = fit_predict(best, trf, te, feats); y = te["w"].values
            ci_r, ci_m = boot_ci(y, p)
            key = f"{sname}|{feats_name}|{'aug' if augm else 'noaug'}"
            res["tables"][key] = {"model": best, "cv_rmse": float(rank.iloc[0]["cv_rmse"]), "cv_sd": float(rank.iloc[0]["cv_sd"]),
                                  "test_rmse": float(np.sqrt(np.mean((p - y) ** 2))), "test_me": float(np.mean(p - y)),
                                  "rmse_ci": ci_r, "me_ci": ci_m, "n_test": int(len(y))}
            if feats_name == "xy" and sname in ("White->White", "Brown->Brown"):
                res["lazy_cv"][key] = rank.reset_index().rename(columns={"index": "Model"}).to_dict(orient="records")
            print(key, res["tables"][key], flush=True)

# fixed model set, same split, with vs without augmentation (identical train/test data)
for sname, (tr, te) in [("White->White", (Wtr, Wte)), ("Brown->Brown", (Btr, Bte))]:
    for mn, m0 in fixed_models().items():
        for augm in [False, True]:
            X, y = tr[["x", "y"]].values, tr["w"].values
            if augm:
                X, y = swap_aug(X, y)
            p = clone(m0).fit(X, y).predict(te[["x", "y"]].values); yt = te["w"].values
            ci_r, _ = boot_ci(yt, p)
            res["fixed"][f"{sname}|{mn}|{'aug' if augm else 'noaug'}"] = {"rmse": float(np.sqrt(np.mean((p - yt) ** 2))), "me": float(np.mean(p - yt)), "rmse_ci": ci_r}
    print("fixed", sname, "done", flush=True)

# pooled training (both varieties, variety indicator) evaluated on each variety's test split
P_tr = pd.concat([Wtr.assign(v=0), Btr.assign(v=1)], ignore_index=True)
for augm in [False, True]:
    trf = pd.concat([aug(Wtr).assign(v=0), aug(Btr).assign(v=1)], ignore_index=True) if augm else P_tr
    rank = lazy_cv_rank(P_tr, ["x", "y", "v"], augm)
    best = rank.index[0]
    for sname, te, v in [("Pooled->White", Wte, 0), ("Pooled->Brown", Bte, 1)]:
        p = fit_predict(best, trf, te.assign(v=v), ["x", "y", "v"]); y = te["w"].values; ci_r, ci_m = boot_ci(y, p)
        res["pooled"][f"{sname}|{'aug' if augm else 'noaug'}"] = {"model": best, "test_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                                                                  "test_me": float(np.mean(p - y)), "rmse_ci": ci_r}
    print("pooled", augm, best, flush=True)

# size-binned test errors of the selected bivariate same-variety models
res["size_bins"] = {}
for sname, (tr, te) in [("White->White", (Wtr, Wte)), ("Brown->Brown", (Btr, Bte))]:
    sel = res["tables"][f"{sname}|xy|noaug"]["model"]
    p = fit_predict(sel, tr, te, ["x", "y"]); y = te["w"].values; gm = np.sqrt(te.x.values * te.y.values)
    rows = []
    for lo, hi in [(0, 25), (25, 40), (40, 200)]:
        s = (gm >= lo) & (gm < hi)
        if s.sum():
            e = p[s] - y[s]
            rows.append({"bin_mm": [lo, hi], "n": int(s.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))), "me": float(e.mean()),
                         "mape": float(np.mean(np.abs(e) / y[s]) * 100), "mean_w": float(y[s].mean())})
    res["size_bins"][sname] = {"model": sel, "bins": rows}
jdump(res, OUT)
print("DONE", flush=True)
