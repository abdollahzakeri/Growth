"""Shared settings and geometry helpers (convex-hull Fitzgibbon ellipse fit, causal moving average, plotting style)."""
import json, os
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon

ROOT = os.environ.get("GROWTH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
RES = os.environ.get("GROWTH_RESULTS", os.path.join(ROOT, "results"))
os.makedirs(RES, exist_ok=True)

# Bed-plane scale: measured lens-to-bed distance / checkerboard-calibrated focal length
Z_BED_MM = 419.0
F_CAL_PX = 879.2            # 50 mm A2 checkerboard, 43 views, processed (undistorted, cropped) image
F_CAL_CI = (847.7, 904.8)   # bootstrap 95% CI
PIX2MM = Z_BED_MM / F_CAL_PX
HRS_PER_FRAME = 15.0 / 60.0

COCO_DEFAULT = os.path.join(RES, "coco_annotations.json")


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def seg_points(seg):
    if not seg:
        return None
    if isinstance(seg, str):
        seg = json.loads(seg)
    s = seg[0] if isinstance(seg[0], (list, tuple)) else seg
    pts = np.asarray(s, float).reshape(-1, 2)
    return pts if len(pts) >= 3 else None


def to_poly(pts):
    p = Polygon(pts)
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty:
        return None
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    return p


def hull_points(pts):
    if pts.shape[0] < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        return pts


def fitzgibbon(points):
    """Direct least-squares ellipse fit (Fitzgibbon et al., 1999) with coordinate normalization."""
    if points is None or points.shape[0] < 6:
        return None
    x = points[:, 0].astype(np.float64); y = points[:, 1].astype(np.float64)
    mx, my = x.mean(), y.mean(); sx, sy = max(x.std(), 1e-8), max(y.std(), 1e-8)
    xn, yn = (x - mx) / sx, (y - my) / sy
    D1 = np.column_stack([xn ** 2, xn * yn, yn ** 2]); D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1 = D1.T @ D1; S2 = D1.T @ D2; S3 = D2.T @ D2
    C1_inv = np.array([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]], np.float64)
    try:
        T = -np.linalg.solve(S3, S2.T); M = C1_inv @ (S1 + S2 @ T)
        evals, evecs = np.linalg.eig(M)
        cond = 4 * evecs[0] * evecs[2] - evecs[1] ** 2
        valid = np.where((cond.real > 0) & np.isfinite(evals))[0]
        if len(valid) == 0:
            return None
        idx = valid[np.argmin(np.abs(evals[valid].real))]
        a1 = evecs[:, idx].real; a2 = (T @ a1).real
        An, Bn, Cn = a1; Dn, En, Fn = a2
        A = An / (sx * sx); B = Bn / (sx * sy); C = Cn / (sy * sy)
        D = -2 * An * mx / (sx * sx) - Bn * my / (sx * sy) + Dn / sx
        E = -Bn * mx / (sx * sy) - 2 * Cn * my / (sy * sy) + En / sy
        F = (An * mx ** 2 / (sx * sx) + Bn * mx * my / (sx * sy) + Cn * my ** 2 / (sy * sy)
             - Dn * mx / sx - En * my / sy + Fn)
        Q = np.array([[A, B / 2], [B / 2, C]]); d_vec = np.array([D, E]) / 2.0
        center = -np.linalg.solve(Q, d_vec)
        lam, V = np.linalg.eigh(Q)
        kappa = center @ Q @ center + np.array([D, E]) @ center + F
        if kappa * lam[0] >= 0 or kappa * lam[1] >= 0:
            return None
        semi = np.sqrt(-kappa / lam)
        return {"center": (float(center[0]), float(center[1])),
                "axes": (float(max(semi)), float(min(semi))),
                "angle": float(np.arctan2(V[1, 0], V[0, 0]))}
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None


def ellipse_diams(pts, use_hull=True):
    """Major/minor diameters (px) of the Fitzgibbon ellipse fitted to the (hull of the) mask contour."""
    fit = fitzgibbon(hull_points(pts) if use_hull else pts)
    if fit is None:
        return None
    a, b = fit["axes"]
    return 2 * a, 2 * b


def causal_ma(seq, l):
    """Causal moving average; uses the min(l, t+1) available past observations."""
    out = np.zeros_like(seq, dtype=float)
    for i in range(len(seq)):
        out[i] = seq[max(0, i - l + 1): i + 1].mean(0)
    return out


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.dpi": 600,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    return plt


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
