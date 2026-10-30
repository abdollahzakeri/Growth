"""R13b: test-set comparison with and without smoothing for every single-image model family, and repeated
input-noise trials (isotropic: same relative error on both axes; independent: separate error per axis).
Same group-level splits as r03 (seed -> split). Usage: python r13b_smooth_noise.py <seed0> <n> <out.pkl>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from growth.common import RES, PIX2MM, causal_ma
S0, NS, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data = pickle.load(open(os.path.join(RES, "seqs_t090.pkl"), "rb"))["seqs"]
seqs = [s["d"].astype(float) for s in data]; groups = np.array([s["group"] for s in data])
FC = {}
import glob
for f in glob.glob(os.path.join(RES, "fc_t090_*.pkl")):
    for r in pickle.load(open(f, "rb")):
        FC[r["seed"]] = r["chosen"]["overall"]
CFGS = ["linear|1|1", "linear|1|5", "linear|1|10", "et_abs|1|1", "et_abs|1|10", "et_delta|1|1", "et_delta|1|10"]
import os
NOISE = [0.02, 0.05, 0.10]; TRIALS = int(os.environ.get("TRIALS", "10"))


def fit(fam, n, l, tr, seed):
    X, Y = [], []
    for s in tr:
        sm = causal_ma(s, l) if l > 1 else s
        for i in range(len(sm) - n):
            w = sm[i:i + n]; X.append(w.ravel()); Y.append(sm[i + n] - (w[-1] if fam == "et_delta" else 0))
    X, Y = np.asarray(X), np.asarray(Y)
    if fam == "linear":
        return Ridge(alpha=1e-3).fit(X, Y)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=8).fit(X, Y); m.set_params(n_jobs=8); return m


def rollout(m, fam, n, te, noise=0.0, mode="iso", rng=None):
    T = np.array([len(s) for s in te]); W = np.stack([s[:n] for s in te]).astype(float)
    if noise > 0:
        shape = W.shape[:2] + ((1,) if mode == "iso" else (2,))
        W = W * (1 + rng.normal(0, noise, size=shape))
    P = [[] for _ in te]
    for t in range(n, T.max()):
        act = np.where(T > t)[0]; Wa = W[act]
        if fam == "persistence":
            nx = Wa[:, -1]
        else:
            o = m.predict(Wa.reshape(len(act), -1)); nx = Wa[:, -1] + o if fam == "et_delta" else o
        for k, i in enumerate(act):
            P[i].append(nx[k])
        W[act] = np.concatenate([Wa[:, 1:], nx[:, None, :]], axis=1)
    errs = [np.linalg.norm(np.asarray(P[i]) - s[n:], axis=1) * PIX2MM for i, s in enumerate(te)]
    rm = np.array([np.sqrt(np.mean(e ** 2)) for e in errs])
    hz = {}
    for h in [6, 12, 24]:
        k = h * 4 - 1; v = [e[k] for e in errs if len(e) > k]
        hz[h] = float(np.sqrt(np.mean(np.square(v)))) if len(v) >= 3 else np.nan
    return {"mean": float(rm.mean()), "h": hz}


out = []
for seed in range(S0, S0 + NS):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rng.shuffle(ug)
    n1, n2 = int(0.6 * len(ug)), int(0.8 * len(ug))
    gtr, gte = set(ug[:n1]), set(ug[n2:])
    tr = [s for s, g in zip(seqs, groups) if g in gtr]; te = [s for s, g in zip(seqs, groups) if g in gte]
    r = {"seed": seed, "test": {}, "noise": {}}
    models = {}
    for cfg in CFGS + [FC[seed]]:
        if cfg in models:
            continue
        fam, n, l = cfg.split("|"); n, l = int(n), int(l)
        models[cfg] = fit(fam, n, l, tr, seed)
        r["test"][cfg] = rollout(models[cfg], fam, n, te)
    sel = FC[seed]; fam, n, l = sel.split("|"); n, l = int(n), int(l)
    r["selected"] = sel
    nrng = np.random.default_rng(seed + 12345)
    for sd in NOISE:
        for mode in ["iso", "indep"]:
            r["noise"][f"{sd}|{mode}"] = [rollout(models[sel], fam, n, te, sd, mode, nrng) for _ in range(TRIALS)]
            r["noise"][f"{sd}|{mode}|persistence"] = [rollout(None, "persistence", 1, te, sd, mode, nrng) for _ in range(TRIALS)]
    r["noise"]["clean|persistence"] = rollout(None, "persistence", 1, te)
    out.append(r)
    print(seed, sel, {k: round(v["mean"], 2) for k, v in r["test"].items()}, flush=True)
    pickle.dump(out, open(OUT, "wb"))
print("DONE")
