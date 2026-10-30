"""R20: re-anchoring that uses only real detections available at the restart time (no interpolated sizes).
For every split (same group-level splits and validation-selected configuration as r03), the forecast is restarted
every R frames from the most recent OBSERVED size at or before the restart time; between restarts only predictions
are fed back. Also reports how many restart inputs in r03 were interpolated values.
Usage: python r20_reanchor_observed.py <seed0> <n> <out.pkl>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, glob, pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from growth.common import RES, PIX2MM, causal_ma
S0, NS, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
data = pickle.load(open(os.path.join(RES, "seqs_t090.pkl"), "rb"))["seqs"]
seqs = [s["d"].astype(float) for s in data]; obs = [s["observed"] for s in data]; groups = np.array([s["group"] for s in data])
CH = {}
for f in glob.glob(os.path.join(RES, "fc_t090_*.pkl")):
    for r in pickle.load(open(f, "rb")): CH[r["seed"]] = r["chosen"]["overall"]
def fit(fam, n, l, tr, seed):
    if fam == "persistence": return None
    X, Y = [], []
    for s in tr:
        sm = causal_ma(s, l) if l > 1 else s
        for i in range(len(sm) - n):
            w = sm[i:i + n]; X.append(w.ravel()); Y.append(sm[i + n] - (w[-1] if fam == "et_delta" else 0))
    X, Y = np.asarray(X), np.asarray(Y)
    if fam == "linear": return Ridge(alpha=1e-3).fit(X, Y)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=4).fit(X, Y); m.set_params(n_jobs=4); return m
def rollout(m, fam, n, te, ob, R):
    N = len(te); T = np.array([len(s) for s in te]); W = np.stack([s[:n] for s in te]).astype(float); P = [[] for _ in te]
    n_interp_old = 0; n_restart = 0
    for t in range(n, T.max()):
        act = np.where(T > t)[0]
        if R and (t - n) > 0 and (t - n) % R == 0:
            for i in act:
                n_restart += 1; n_interp_old += int(not ob[i][t - 1])
                k = np.where(ob[i][:t])[0][-1]                   # last real detection at or before t-1
                W[i] = te[i][max(0, k - n + 1):k + 1] if k + 1 >= n else te[i][:n]
        Wa = W[act]
        if fam == "persistence": nx = Wa[:, -1]
        else:
            o = m.predict(Wa.reshape(len(act), -1)); nx = Wa[:, -1] + o if fam == "et_delta" else o
        for k_, i in enumerate(act): P[i].append(nx[k_])
        W[act] = np.concatenate([Wa[:, 1:], nx[:, None, :]], axis=1)
    errs = [np.linalg.norm(np.asarray(P[i]) - s[n:], axis=1) * PIX2MM for i, s in enumerate(te)]
    hz = {}
    for h in [24, 48, 72]:
        k = h * 4 - 1; v = [e[k] for e in errs if len(e) > k]
        hz[h] = (float(np.sqrt(np.mean(np.square(v)))) if len(v) >= 3 else np.nan, len(v))
    return hz, n_interp_old, n_restart
out = []
for seed in range(S0, S0 + NS):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rng.shuffle(ug)
    n1, n2 = int(0.6 * len(ug)), int(0.8 * len(ug)); gtr, gte = set(ug[:n1]), set(ug[n2:])
    tr = [s for s, g in zip(seqs, groups) if g in gtr]; idx = [i for i, g in enumerate(groups) if g in gte]
    te = [seqs[i] for i in idx]; ob = [obs[i] for i in idx]
    sel = CH[seed]; fam, n, l = sel.split("|"); n, l = int(n), int(l)
    m = fit(fam, n, l, tr, seed); r = {"seed": seed, "cfg": sel}
    for R in [96, 48, 24]:
        r[f"model|R{R}"], ni, nr = rollout(m, fam, n, te, ob, R)
        r[f"persistence|R{R}"], _, _ = rollout(None, "persistence", 1, te, ob, R)
        r[f"interp_restarts_old|R{R}"] = (ni, nr)
    out.append(r); print(seed, sel, {k: v for k, v in r.items() if "|R" in k}, flush=True)
    pickle.dump(out, open(OUT, "wb"))
print("DONE")
