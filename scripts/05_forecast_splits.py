"""R03: growth-forecasting study with group-level train/validation/test splits.

For each split seed:
  * groups (physical mushrooms) -> 60% train / 20% validation / 20% test
  * model families: persistence, linear AR (ridge), Extra-Trees AR on absolute size (original),
    Extra-Trees AR on growth increments
  * hyperparameters (n_steps x smoothing window l) selected on VALIDATION by mean per-track RMSE
  * TEST: recursive multi-step rollout from raw observations (single image when n_steps=1),
    seed frames excluded from the error, error vs. forecast horizon, re-anchoring, input-noise
    sensitivity, smoothing ablation

Usage: python r03_forecast.py <seqs.pkl> <seed_start> <n_seeds> <out_prefix> [feature=d] [n_jobs=4]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle, json, time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from growth.common import PIX2MM, causal_ma, jdump

SEQ_PKL, S0, NS, OUTP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
FEAT = sys.argv[5] if len(sys.argv) > 5 else "d"
NJ = int(sys.argv[6]) if len(sys.argv) > 6 else 4
NSTEPS = [1, 2, 4, 8]
LWIN = [1, 5, 10, 20]
FAMILIES = ["linear", "et_abs", "et_delta"]
HMAX = 480                       # 120 h horizon bookkeeping
REANCHOR = [None, 96, 48, 24]    # every 24 h, 12 h, 6 h
NOISE = [0.02, 0.05, 0.10]

data = pickle.load(open(SEQ_PKL, "rb"))["seqs"]
seqs = [s[FEAT].astype(float) for s in data]
groups = np.array([s["group"] for s in data])


def make_xy(train, n, l, delta):
    X, Y = [], []
    for s in train:
        sm = causal_ma(s, l) if l > 1 else s
        for i in range(len(sm) - n):
            w = sm[i:i + n]
            X.append(w.ravel()); Y.append(sm[i + n] - (w[-1] if delta else 0))
    return np.asarray(X), np.asarray(Y)


def fit(family, n, l, train, seed):
    if family == "persistence":
        return None
    X, Y = make_xy(train, n, l, family == "et_delta")
    if family == "linear":
        return Ridge(alpha=1e-3).fit(X, Y)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=NJ).fit(X, Y)
    m.set_params(n_jobs=1)
    return m


def rollout(model, family, n, test, reanchor=None, noise=0.0, rng=None):
    """Returns list of (pred_frames, errors_mm per forecast frame, horizon index per frame)."""
    N = len(test); T = np.array([len(s) for s in test])
    W = np.stack([s[:n] for s in test]).astype(float)                    # N x n x 2
    if noise > 0:
        W = W * (1 + rng.normal(0, noise, size=W.shape[:2] + (1,)))
    preds = [[] for _ in range(N)]
    Lmax = int(T.max())
    for t in range(n, Lmax):
        act = np.where(T > t)[0]
        if len(act) == 0:
            break
        if reanchor is not None and (t - n) > 0 and (t - n) % reanchor == 0:
            for i in act:
                W[i] = test[i][t - n:t]
        Wa = W[act]
        if family == "persistence":
            nx = Wa[:, -1]
        else:
            out = model.predict(Wa.reshape(len(act), -1))
            nx = Wa[:, -1] + out if family == "et_delta" else out
        for k, i in enumerate(act):
            preds[i].append(nx[k])
        W[act] = np.concatenate([Wa[:, 1:], nx[:, None, :]], axis=1)
    res = []
    for i, s in enumerate(test):
        p = np.asarray(preds[i]); g = s[n:n + len(p)]
        err = np.linalg.norm(p - g, axis=1) * PIX2MM
        res.append(err)
    return res


def summarize(errs):
    rm = np.array([np.sqrt(np.mean(e ** 2)) for e in errs if len(e)])
    sq = np.zeros(HMAX); cnt = np.zeros(HMAX); ab = np.zeros(HMAX)
    for e in errs:
        k = min(len(e), HMAX)
        sq[:k] += e[:k] ** 2; ab[:k] += e[:k]; cnt[:k] += 1
    return {"track_rmse": rm, "mean": float(rm.mean()), "median": float(np.median(rm)),
            "h_sq": sq, "h_abs": ab, "h_cnt": cnt}


results = []
for seed in range(S0, S0 + NS):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); rng.shuffle(ug)
    n1, n2 = int(0.6 * len(ug)), int(0.8 * len(ug))
    gtr, gva, gte = set(ug[:n1]), set(ug[n1:n2]), set(ug[n2:])
    tr = [s for s, g in zip(seqs, groups) if g in gtr]
    va = [s for s, g in zip(seqs, groups) if g in gva]
    te = [s for s, g in zip(seqs, groups) if g in gte]
    out = {"seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "n_groups": [len(gtr), len(gva), len(gte)], "val_grid": {}, "chosen": {}, "test": {}}
    # ---------------- validation grid ----------------
    for fam in FAMILIES:
        for n in NSTEPS:
            for l in LWIN:
                m = fit(fam, n, l, tr, seed)
                r = summarize(rollout(m, fam, n, va))
                out["val_grid"][f"{fam}|{n}|{l}"] = {"mean": r["mean"], "median": r["median"]}
    for n in NSTEPS:
        r = summarize(rollout(None, "persistence", n, va))
        out["val_grid"][f"persistence|{n}|1"] = {"mean": r["mean"], "median": r["median"]}
    for fam in FAMILIES + ["persistence"]:
        keys = [k for k in out["val_grid"] if k.startswith(fam + "|")]
        out["chosen"][fam] = min(keys, key=lambda k: out["val_grid"][k]["mean"])
        k1 = [k for k in keys if k.split("|")[1] == "1"]
        out["chosen"][fam + "_single"] = min(k1, key=lambda k: out["val_grid"][k]["mean"])
    overall = min([out["chosen"][f + "_single"] for f in FAMILIES], key=lambda k: out["val_grid"][k]["mean"])
    overall_any = min([out["chosen"][f] for f in FAMILIES], key=lambda k: out["val_grid"][k]["mean"])
    out["chosen"]["overall"] = overall            # primary: single-image setting (n_steps = 1)
    out["chosen"]["overall_any"] = overall_any    # secondary: with recent history
    # ---------------- test: chosen configs + paper configuration ----------------
    eval_cfgs = dict(out["chosen"])
    eval_cfgs["paper_et_abs_n1_l10"] = "et_abs|1|10"
    eval_cfgs["persistence_n1"] = "persistence|1|1"
    fam_o, n_o, l_o = overall.split("|"); n_o, l_o = int(n_o), int(l_o)
    eval_cfgs["overall_l1"] = f"{fam_o}|{n_o}|1"
    models = {}
    for name, cfg in eval_cfgs.items():
        fam, n, l = cfg.split("|"); n, l = int(n), int(l)
        if cfg not in models:
            models[cfg] = fit(fam, n, l, tr, seed)
        r = summarize(rollout(models[cfg], fam, n, te))
        out["test"][name] = {"cfg": cfg, **r}
    # re-anchoring & noise (overall best + persistence)
    out["reanchor"] = {}
    for cfg in [overall, overall_any, "persistence|1|1"]:
        fam, n, l = cfg.split("|"); n, l = int(n), int(l)
        for R in REANCHOR:
            r = summarize(rollout(models.get(cfg), fam, n, te, reanchor=R))
            out["reanchor"][f"{cfg}|R{R}"] = r
    out["noise"] = {}
    for sd in NOISE:
        for cfg in [overall, "persistence|1|1"]:
            fam, n, l = cfg.split("|"); n, l = int(n), int(l)
            r = summarize(rollout(models.get(cfg), fam, n, te, noise=sd, rng=np.random.default_rng(seed + 999)))
            out["noise"][f"{cfg}|{sd}"] = r
    # n_steps curves on test for the overall family at l=1 and l=best (Fig. ablation)
    out["nsteps_curve"] = {}
    for n in NSTEPS:
        for l in sorted({1, l_o}):
            cfg = f"{fam_o}|{n}|{l}"
            if cfg not in models:
                models[cfg] = fit(fam_o, n, l, tr, seed)
            r = summarize(rollout(models[cfg], fam_o, n, te))
            out["nsteps_curve"][cfg] = {"mean": r["mean"], "median": r["median"]}
    out["time_s"] = time.time() - t0
    print(f"seed {seed}: chosen={out['chosen']}  test overall mean={out['test']['overall']['mean']:.2f} mm  "
          f"any={out['test']['overall_any']['mean']:.2f}  persist={out['test']['persistence_n1']['mean']:.2f}  paper={out['test']['paper_et_abs_n1_l10']['mean']:.2f}  "
          f"({out['time_s']:.0f}s)", flush=True)
    results.append(out)
    pickle.dump(results, open(f"{OUTP}_{S0}.pkl", "wb"))
print("DONE", flush=True)
