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


def fig_weight():
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for name, c, mk, lab in [("white", PALETTE[0], "o", "White button"), ("brown", PALETTE[1], "s", "Baby bella")]:
        d = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"), encoding="utf-8-sig")
        gm = np.sqrt(d.x * d.y)
        ax.scatter(gm, d.w, s=9, marker=mk, facecolor="none", edgecolor=c, linewidth=0.7, label=lab)
        b, a = np.polyfit(np.log(gm), np.log(d.w), 1); xs = np.linspace(gm.min(), gm.max(), 200)
        ax.plot(xs, np.exp(a) * xs ** b, color=c, lw=1.6, zorder=3, label=f"{lab} fit")
    ax.set_xlabel(r"Geometric mean cap diameter $\sqrt{wh}$ (mm)"); ax.set_ylabel("Fresh weight (g)")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    fig.savefig(f"{FIG}/brown_white_scatter.pdf"); plt.close(fig)


def fig_occlusion():
    o = json.load(open(f"{RES}/occlusion.json"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    meth = [("hull", "Convex hull + ellipse (ours)", PALETTE[0], "o"), ("bbox", "Bounding box", PALETTE[1], "s"),
            ("raw", "Ellipse on raw contour", PALETTE[2], "^")]
    for ax, (sname, title) in zip(axes, [("clean", "(a) Well-formed caps (n = 300)"), ("representative", "(b) Randomly sampled caps (n = 300)")]):
        S = o[sname]["summary"]; lv = sorted(S, key=int)
        x = [S[k]["actual_removed_mean"] for k in lv]
        for m, lab, c, mk in meth:
            y = [S[k][m]["mean"] for k in lv]; e = [S[k][m]["sd"] for k in lv]
            ax.errorbar(x, y, yerr=e, color=c, marker=mk, capsize=2, lw=1.2, ms=4, label=lab)
        ax.axhline(5, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("Removed mask area (%)"); ax.set_title(title, loc="left"); ax.set_xticks([10, 20, 30, 40, 50])
    axes[0].set_ylabel("Diameter error (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(f"{FIG}/occlusion_experiment_results.pdf"); plt.close(fig)


if which in ("weight", "all"):
    fig_weight()
if which in ("occlusion", "all"):
    fig_occlusion()
def load_fc(pat="fc_t090"):
    R = []
    for f in sorted(glob.glob(f"{RES}/{pat}_*.pkl")): R += pickle.load(open(f, "rb"))
    return R


def per_seed_curve(entry, kmax=288, minn=5):
    c = entry["h_cnt"][:kmax]; q = entry["h_sq"][:kmax]
    return np.where(c >= minn, np.sqrt(q / np.maximum(c, 1)), np.nan), c


def fig_horizon(R):
    hrs = np.arange(1, 289) * HRS_PER_FRAME
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8))
    ax = axes[0]
    for nm, lab, c, ls in [("persistence_n1", "Persistence", PALETTE[6], "--"), ("linear_single", "Linear autoregression", PALETTE[2], "-"),
                           ("paper_et_abs_n1_l10", "Extra-Trees (n=1, l=10)", PALETTE[1], "-"), ("overall", "Validation-selected model", PALETTE[0], "-")]:
        C = np.array([per_seed_curve(r["test"][nm])[0] for r in R])
        m = np.nanmean(C, 0); sd = np.nanstd(C, 0)
        ax.plot(hrs, m, color=c, ls=ls, label=lab); ax.fill_between(hrs, m - sd, m + sd, color=c, alpha=0.12, lw=0)
    n = np.mean([r["test"]["persistence_n1"]["h_cnt"][:288] for r in R], 0)
    thin = hrs[np.argmax(n < 10)] if (n < 10).any() else None
    if thin is not None:
        ax.axvspan(thin, 72, color="0.85", alpha=0.5, lw=0); ax.text(thin - 1, 3, "< 10 test\ntrajectories", fontsize=6.5, color="0.35", va="bottom", ha="right")
    ax2 = ax.twinx(); ax2.plot(hrs, n, color="0.6", lw=0.8, ls=":"); ax2.set_ylabel("Test trajectories per split", color="0.45")
    ax2.tick_params(axis="y", colors="0.45"); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Diameter RMSE (mm)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_title("(a) Single-image forecast", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax = axes[1]
    for Rk, lab, c in [("None", "No re-anchoring", PALETTE[6]), ("96", "Every 24 h", PALETTE[1]), ("48", "Every 12 h", PALETTE[2]), ("24", "Every 6 h", PALETTE[0])]:
        C = np.array([per_seed_curve(r["reanchor"][f"{r['chosen']['overall']}|R{Rk}"])[0] for r in R])
        ax.plot(hrs, np.nanmean(C, 0), color=c, label=lab)
    ax.set_xlabel("Time since first image (h)"); ax.set_ylabel("Diameter RMSE (mm)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_title("(b) Validation-selected model with re-anchoring", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/forecast_error_vs_horizon.pdf"); plt.close(fig)


def fig_valgrid(R):
    NS, LW = [1, 2, 4, 8], [1, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    for ax, fam, title in [(axes[0], "et_abs", "(a) Extra-Trees"), (axes[1], "linear", "(b) Linear autoregression")]:
        M = np.array([[np.mean([r["val_grid"][f"{fam}|{n}|{l}"]["mean"] for r in R]) for l in LW] for n in NS])
        im = ax.imshow(M, cmap="viridis_r", aspect="auto")
        for i in range(len(NS)):
            for j in range(len(LW)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > M.mean() else "black")
        ax.set_xticks(range(len(LW))); ax.set_xticklabels(LW); ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
        ax.set_xlabel(r"Smoothing window $l$ (frames)"); ax.set_ylabel(r"History length $n_{\mathrm{steps}}$ (frames)"); ax.set_title(title, loc="left"); ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Validation RMSE (mm)")
    fig.tight_layout(); fig.savefig(f"{FIG}/mean_rmse_surface.pdf"); plt.close(fig)


def fig_ablation(R):
    NS = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    for l, c, mk in [(1, PALETTE[6], "s"), (10, PALETTE[0], "o"), (20, PALETTE[2], "^")]:
        V = np.array([[r["val_grid"][f"et_abs|{n}|{l}"]["mean"] for n in NS] for r in R])
        ax.errorbar(NS, V.mean(0), yerr=V.std(0), color=c, marker=mk, capsize=2, label=f"$l$ = {l}" + (" (no smoothing)" if l == 1 else ""))
    ax.set_xscale("log", base=2); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel(r"History length $n_{\mathrm{steps}}$ (frames)"); ax.set_ylabel("Validation RMSE (mm)"); ax.legend(frameon=False)
    fig.savefig(f"{FIG}/rmse_vs_n_steps_w1_w10_mm.pdf"); plt.close(fig)


def fig_examples(seqs_pkl, R):
    """Representative test trajectories of split 0 (20th, 40th, 60th, 80th percentile of per-track RMSE among
    trajectories longer than 24 h), validation-selected model, raw observations vs. forecasts."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    D = pickle.load(open(seqs_pkl, "rb"))["seqs"]; r0 = R[0]
    fam, n, l = "linear", 1, 5          # deployed configuration (most frequently selected on validation)
    seqs = [s["d"].astype(float) for s in D]; groups = np.array([s["group"] for s in D])
    rng = np.random.default_rng(r0["seed"]); ug = np.unique(groups); rng.shuffle(ug)
    n1, n2 = int(0.6 * len(ug)), int(0.8 * len(ug)); gtr, gte = set(ug[:n1]), set(ug[n2:])
    tr = [s for s, g in zip(seqs, groups) if g in gtr]; te = [s for s, g in zip(seqs, groups) if g in gte]
    X, Y = [], []
    for s in tr:
        sm = causal_ma(s, l) if l > 1 else s
        for i in range(len(sm) - n):
            w = sm[i:i + n]; X.append(w.ravel()); Y.append(sm[i + n] - (w[-1] if fam == "et_delta" else 0))
    m = Ridge(alpha=1e-3).fit(X, Y) if fam == "linear" else ExtraTreesRegressor(n_estimators=100, random_state=r0["seed"], n_jobs=8).fit(X, Y)
    if fam != "linear":
        m.set_params(n_jobs=1)
    cand = [s for s in te if len(s) > 96]
    out = []
    for s in cand:
        W = s[:n][None].copy(); P = []
        for t in range(n, len(s)):
            o = m.predict(W.reshape(1, -1))[0]; nx = W[0, -1] + o if fam == "et_delta" else o
            P.append(nx); W = np.concatenate([W[:, 1:], nx[None, None, :]], axis=1)
        P = np.array(P); e = np.sqrt(np.mean(np.sum((P - s[n:]) ** 2, 1))) * PIX2MM; out.append((e, s, P))
    out.sort(key=lambda z: z[0]); idx = [int(q * (len(out) - 1)) for q in [0.2, 0.4, 0.6, 0.8]]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.2), sharex=False)
    for ax, i, tag in zip(axes.ravel(), idx, ["20th", "40th", "60th", "80th"]):
        e, s, P = out[i]; t = np.arange(len(s)) * HRS_PER_FRAME
        ax.plot(t, s[:, 0] * PIX2MM, color=PALETTE[0], lw=0.8, alpha=0.6, label="Observed major axis")
        ax.plot(t, s[:, 1] * PIX2MM, color=PALETTE[1], lw=0.8, alpha=0.6, label="Observed minor axis")
        ax.plot(t[n:], P[:, 0] * PIX2MM, color=PALETTE[0], ls="--", lw=1.5, label="Forecast major axis")
        ax.plot(t[n:], P[:, 1] * PIX2MM, color=PALETTE[1], ls="--", lw=1.5, label="Forecast minor axis")
        ax.set_title(f"{tag} percentile, RMSE = {e:.2f} mm", loc="left"); ax.set_xlabel("Time since first image (h)"); ax.set_ylabel("Diameter (mm)")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{FIG}/forecast_examples.pdf"); plt.close(fig)
    return {"percentile_rmse_mm": [float(out[i][0]) for i in idx], "n_candidates": len(out), "config": f"{fam}|{n}|{l}"}


def fig_ma(seqs_pkl, l=10):
    D = pickle.load(open(seqs_pkl, "rb"))["seqs"]
    lens = np.array([s["len"] for s in D]); rng = np.random.default_rng(3)
    grow = np.array([(s["d"][-10:, 0].mean() - s["d"][:10, 0].mean()) / s["d"][:10, 0].mean() for s in D])
    long_i = int(rng.choice(np.where((lens > 192) & (grow > 0.5))[0])); short_i = int(rng.choice(np.where((lens >= 30) & (lens <= 48))[0]))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
    for ax, i, title in [(axes[0], long_i, "(a) Long trajectory"), (axes[1], short_i, "(b) Short trajectory")]:
        s = D[i]["d"].astype(float); sm = causal_ma(s, l); t = np.arange(len(s)) * HRS_PER_FRAME
        for k, c, nm in [(0, PALETTE[0], "major"), (1, PALETTE[1], "minor")]:
            ax.plot(t, s[:, k] * PIX2MM, color=c, lw=0.7, alpha=0.55, label=f"Raw {nm} axis")
            ax.plot(t, sm[:, k] * PIX2MM, color=c, lw=1.6, label=f"Moving average ($l$ = {l}), {nm} axis")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Diameter (mm)"); ax.set_title(title, loc="left")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{FIG}/moving_average_examples.pdf"); plt.close(fig)


if which in ("growth", "all"):
    R = load_fc("fc_t090")
    fig_horizon(R); fig_valgrid(R); fig_ablation(R); fig_ma(f"{RES}/seqs_t090.pkl")
    info = fig_examples(f"{RES}/seqs_t090.pkl", R); json.dump(info, open(f"{RES}/fig_examples_info.json", "w")); print(info)

def fig_yield():
    E = json.load(open(f"{RES}/yield_mc_et.json")); Lr = json.load(open(f"{RES}/yield_mc_linear.json"))
    # (Fig. 12) example cohort: predicted vs image-derived yield of the caps visible in one image
    ex = E["example"]; h = np.array(ex["hours"])
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 2.6))
    ax.fill_between(h, np.array(ex["true_lo"]) / 1000, np.array(ex["true_hi"]) / 1000, color=PALETTE[0], alpha=0.18, lw=0,
                    label="Reference, 95% interval")
    ax.plot(h, np.array(ex["obs_g"]) / 1000, color=PALETTE[0], lw=1.6, label="Reference (observed sizes)")
    ax.plot(h, np.array(ex["pred_g"]) / 1000, color=PALETTE[1], ls="--", lw=1.6, label="Forecast from one image")
    ax.set_xlabel("Hours after the image"); ax.set_ylabel("Cohort fresh weight (kg)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
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
    ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel("Tray-level yield error (%)"); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.savefig(f"{FIG}/compound_yield_error.pdf"); plt.close(fig)


if which == "ma":
    fig_ma(f"{RES}/seqs_t090.pkl")
if which in ("yield", "all"):
    fig_yield()
print("figures written:", sorted(os.listdir(FIG)))
