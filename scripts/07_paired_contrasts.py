"""R21: paired contrasts across the 30 splits (incl. validation-selected minus linear) for Table B.2."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from growth.common import RES
import glob, pickle, json
import numpy as np
from scipy import stats
R = []
for f in sorted(glob.glob(os.path.join(RES, "fc_t090_*.pkl"))): R += pickle.load(open(f, "rb"))
R = sorted(R, key=lambda r: r["seed"]); ratio = np.mean([r["n_groups"][2] / r["n_groups"][0] for r in R])
def hz(e, h):
    k = h * 4 - 1; return np.sqrt(e["h_sq"][k] / e["h_cnt"][k]) if e["h_cnt"][k] >= 3 else np.nan
def met(name, m): return np.array([r["test"][name]["mean"] if m == "track" else hz(r["test"][name], m) for r in R])
pairs = [("overall", "persistence_n1"), ("linear_single", "persistence_n1"), ("paper_et_abs_n1_l10", "persistence_n1"),
         ("overall", "linear_single"), ("overall", "paper_et_abs_n1_l10"), ("linear_single", "paper_et_abs_n1_l10"),
         ("overall", "overall_l1"), ("overall_any", "overall")]
out = []
for a, b in pairs:
    row = {"a": a, "b": b}
    for m in ["track", 12, 24]:
        x, y = met(a, m), met(b, m); ok = np.isfinite(x) & np.isfinite(y); d = x[ok] - y[ok]; J = len(d); sd = d.std(ddof=1)
        t = d.mean() / np.sqrt((1 / J + ratio) * sd ** 2) if sd > 0 else 0.0
        row[str(m)] = {"diff": float(d.mean()), "wins_a": int((d < 0).sum()), "J": J, "p": float(2 * stats.t.sf(abs(t), J - 1))}
    out.append(row); print(a, "-", b, {k: (round(v["diff"], 2), v["wins_a"], round(v["p"], 4)) for k, v in row.items() if isinstance(v, dict)})
json.dump(out, open(os.path.join(RES, "m_paired_table.json"), "w"), indent=1)
