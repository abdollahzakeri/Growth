"""R19: coverage of the training trajectories across cap sizes (all 1,617 trajectories of >= 5 h).
(a) every trajectory: major diameter vs. time since its first frame; (b) number of frames (training pairs) and of
trajectories per 2-mm size bin; (c) density of (current size, size change over the next hour).
Usage: python r19_coverage.py <outdir>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, pickle
import numpy as np
from growth.common import RES, set_style, PIX2MM, HRS_PER_FRAME, PALETTE
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
S = pickle.load(open(os.path.join(RES, "seqs_t090.pkl"), "rb"))["seqs"]
plt = set_style()
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
allD, d1h_x, d1h_y = [], [], []
for s in S:
    d = s["d"][:, 0].astype(float) * PIX2MM; h = np.arange(len(d)) * HRS_PER_FRAME
    ax[0].plot(h, d, lw=0.35, alpha=0.18, color=PALETTE[0])
    allD.append(d)
    if len(d) > 4:
        d1h_x.append(d[:-4]); d1h_y.append(d[4:] - d[:-4])
allD = np.concatenate(allD); X = np.concatenate(d1h_x); Y = np.concatenate(d1h_y)
ax[0].set_xlabel("Time since trajectory start (h)"); ax[0].set_ylabel("Major diameter (mm)")
ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 90)
ax[0].set_title(f"(a) All {len(S):,} trajectories", fontsize=9)
bins = np.arange(0, 92, 2)
cnt, _ = np.histogram(allD, bins)
ntraj = np.array([sum(1 for s in S if np.any((s["d"][:, 0] * PIX2MM >= lo) & (s["d"][:, 0] * PIX2MM < hi))) for lo, hi in zip(bins[:-1], bins[1:])])
ax[1].barh(bins[:-1] + 1, cnt, height=1.8, color=PALETTE[0], alpha=0.8, label="frames")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 90); ax[1].set_xlabel("Count per 2-mm bin"); ax[1].set_ylabel("Major diameter (mm)")
ax[1].plot(ntraj, bins[:-1] + 1, color=PALETTE[1], lw=1.4, label="trajectories")
ax[1].legend(fontsize=7, frameon=True, framealpha=0.9, loc="upper right")
ax[1].set_title("(b) Samples per size", fontsize=9)
hb = ax[2].hexbin(X, Y, gridsize=45, extent=(0, 90, -6, 8), bins="log", cmap="viridis", mincnt=1)
med = [np.median(Y[(X >= lo) & (X < lo + 5)]) if np.sum((X >= lo) & (X < lo + 5)) > 20 else np.nan for lo in range(0, 90, 5)]
ax[2].plot(np.arange(2.5, 92.5, 5), med, color="white", lw=1.6); ax[2].plot(np.arange(2.5, 92.5, 5), med, color="k", lw=0.8, label="median")
ax[2].set_xlabel("Major diameter (mm)"); ax[2].set_ylabel("Change over next 1 h (mm)"); ax[2].set_xlim(0, 90); ax[2].set_ylim(-6, 8)
cb = fig.colorbar(hb, ax=ax[2], fraction=0.05, pad=0.03); cb.set_label("Samples")
ax[2].legend(fontsize=7, frameon=True, framealpha=0.9, loc="lower right")
ax[2].set_title("(c) Size transitions", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectory_size_coverage.png"), dpi=300); fig.savefig(os.path.join(OUT, "trajectory_size_coverage.pdf"))
q = np.percentile(allD, [1, 5, 50, 95, 99])
print("frames", len(allD), "pairs(1h)", len(X), "diam pct 1/5/50/95/99:", np.round(q, 1))
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]:
    k = (allD >= lo) & (allD < hi); print(f"{lo}-{hi} mm: frames {k.sum()}, trajectories {sum(1 for s in S if np.any((s['d'][:,0]*PIX2MM>=lo)&(s['d'][:,0]*PIX2MM<hi)))}")
print("DONE")
