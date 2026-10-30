"""R07b: forecast accuracy of a FIXED configuration on a sequence set (no tuning), over group splits.
Used for the tracker-threshold and detection-dropout sensitivity analyses.

Usage: python r07b_sensitivity.py <seqs.pkl> <family> <n> <l> <n_splits> <out.json> [feature=d]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from growth.common import PIX2MM, causal_ma, jdump

SEQ, FAM, N, L, NS, OUT = sys.argv[1:7]; N, L, NS = int(N), int(L), int(NS)
FEAT = sys.argv[7] if len(sys.argv) > 7 else "d"
D = pickle.load(open(SEQ, "rb")); data = D["seqs"]
seqs = [s[FEAT].astype(float) for s in data]; groups = np.array([s["group"] for s in data])
H = [6, 12, 24, 48, 72]
means, medians, hz = [], [], {h: [] for h in H}
for seed in range(NS):
    rng = np.random.default_rng(seed); ug = np.unique(groups); rng.shuffle(ug)
    gte = set(ug[int(0.8 * len(ug)):])
    tr = [s for s, g in zip(seqs, groups) if g not in gte]; te = [s for s, g in zip(seqs, groups) if g in gte]
    X, Y = [], []
    for s in tr:
        sm = causal_ma(s, L) if L > 1 else s
        for i in range(len(sm) - N):
            w = sm[i:i + N]; X.append(w.ravel()); Y.append(sm[i + N] - (w[-1] if FAM == "et_delta" else 0))
    m = None if FAM == "persistence" else (Ridge(alpha=1e-3).fit(X, Y) if FAM == "linear" else
                                            ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=8).fit(X, Y))
    if m is not None and FAM != "linear":
        m.set_params(n_jobs=1)
    Tl = np.array([len(s) for s in te]); W = np.stack([s[:N] for s in te]).astype(float)
    P = [[] for _ in te]
    for t in range(N, Tl.max()):
        act = np.where(Tl > t)[0]; Wa = W[act]
        if m is None: nx = Wa[:, -1]
        else:
            o = m.predict(Wa.reshape(len(act), -1)); nx = Wa[:, -1] + o if FAM == "et_delta" else o
        for k, i in enumerate(act): P[i].append(nx[k])
        W[act] = np.concatenate([Wa[:, 1:], nx[:, None, :]], axis=1)
    errs = [np.linalg.norm(np.asarray(P[i]) - s[N:], axis=1) * PIX2MM for i, s in enumerate(te)]
    rm = np.array([np.sqrt(np.mean(e ** 2)) for e in errs]); means.append(rm.mean()); medians.append(np.median(rm))
    for h in H:
        k = h * 4 - 1
        v = [e[k] for e in errs if len(e) > k]
        hz[h].append(float(np.sqrt(np.mean(np.square(v)))) if len(v) >= 3 else np.nan)
res = {"seqs": SEQ, "cfg": f"{FAM}|{N}|{L}", "n_seqs": len(seqs), "stats": D["stats"],
       "mean_rmse": [float(np.mean(means)), float(np.std(means))], "median_rmse": [float(np.mean(medians)), float(np.std(medians))],
       "horizon_rmse": {str(h): [float(np.nanmean(hz[h])), float(np.nanstd(hz[h]))] for h in H}}
jdump(res, OUT)
print(SEQ, res["cfg"], "mean", res["mean_rmse"], "median", res["median_rmse"], "h24", res["horizon_rmse"]["24"], flush=True)
