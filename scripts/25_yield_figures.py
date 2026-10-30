"""R10: publication figures (consistent style) for the revised manuscript.

Usage: python r10_figures.py <which> [args]
  which in {weight, occlusion, growth, yield, all}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, json, glob, pickle
import numpy as np, pandas as pd
from growth.common import ROOT, RES, set_style, PALETTE, PIX2MM, HRS_PER_FRAME, causal_ma
plt = set_style()
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.environ.get("GROWTH_RESULTS", os.path.join(ROOT, "results"))
W1, W2 = 3.5, 7.2           # single / double column width (in)
which = sys.argv[1] if len(sys.argv) > 1 else "all"


def fig_yield():
    E = json.load(open(f"{RES}/m_ychain_et.json")); Lr = json.load(open(f"{RES}/m_ychain_linear.json"))
    # (Fig. 12) example cohort: predicted vs image-derived yield of the caps visible in one image
    ex = E["example"]; h = np.array(ex["hours"])
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.6))
    ax.fill_between(h, np.array(ex["true_lo"]) / 1000, np.array(ex["true_hi"]) / 1000, color=PALETTE[0], alpha=0.18, lw=0,
                    label="Reference, 95% interval")
    ax.plot(h, np.array(ex["obs_g"]) / 1000, color=PALETTE[0], lw=1.6, label="Reference (from observed sizes)")
    ax.plot(h, np.array(ex["pred_g"]) / 1000, color=PALETTE[1], ls="--", lw=1.6, label="Forecast from one image")
    ax.set_xlabel("Hours after the image"); ax.set_ylabel("Biomass of the cohort (kg)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(f"{FIG}/predicted_yield.pdf"); plt.close(fig)
    # (Fig. 13) Monte Carlo tray-level error vs horizon
    H = [0, 3, 6, 12, 24, 48, 72]
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.6))
    for D, lab, c, mk in [(E, "Extra-Trees ($n_{\\mathrm{steps}}$=1, $l$=10)", PALETTE[0], "o"), (Lr, "Linear autoregression", PALETTE[2], "s")]:
        y = [D["cohort"][str(x)]["mc_abs_mean_pct"] for x in H]
        ax.plot(H, y, color=c, marker=mk, label=lab)
    ci = np.array([E["cohort"][str(x)]["mc_ci95_pct"] for x in H])
    ax.fill_between(H, np.abs(ci).min(1) * 0, np.percentile(np.abs(ci), 100, axis=1) * 0 + [E["cohort"][str(x)]["mc_abs_p95_pct"] for x in H],
                    color=PALETTE[0], alpha=0.12, lw=0, label="95th percentile, Extra-Trees")
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Error of predicted biomass (%)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.savefig(f"{FIG}/compound_yield_error.pdf"); plt.close(fig)



fig_yield()
print("ok")
