"""R04b (straight-edge occluder variant of R04): synthetic-occlusion validation of cap-diameter recovery.

Two reference sets of 300 caps each:
  * 'clean'          : solidity >= 0.95, highest mask/ellipse agreement (original protocol)
  * 'representative' : uniformly random detected caps (not touching the image border, area >= 400 px,
                       solidity >= 0.85, i.e., not visibly occluded already), no ellipse-quality screening
Reference diameters: hull+Fitzgibbon fit on the unoccluded mask.
Methods on the occluded mask: hull+Fitzgibbon (ours), Fitzgibbon on raw contour (no hull), bounding box.
The ACTUAL removed area fraction is recorded for every trial.

Usage: python r04_occlusion.py <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, json, random
import numpy as np
from shapely.geometry import Polygon as SP
from growth.common import COCO_DEFAULT, seg_points, to_poly, fitzgibbon, hull_points, jdump

OUT = sys.argv[1]
LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
W_IMG, H_IMG = 1580, 697


def ell_poly(cx, cy, a, b, ang, n=128):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    p = SP(np.column_stack([cx + a * np.cos(t) * ca - b * np.sin(t) * sa, cy + a * np.cos(t) * sa + b * np.sin(t) * ca]))
    return p if p.is_valid else p.buffer(0)


def occlude(poly, pct, rng):
    """Straight-edge occluder: removes the part of the cap on one side of a line of random orientation."""
    minx, miny, maxx, maxy = poly.bounds; cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    orig = poly.area; ang = rng.uniform(0, 2 * np.pi); R = 4 * max(maxx - minx, maxy - miny)
    nx, ny = np.cos(ang), np.sin(ang)
    def half(off):
        px, py = cx + off * nx, cy + off * ny; tx, ty = -ny, nx
        return SP([(px + R * tx, py + R * ty), (px - R * tx, py - R * ty), (px - R * tx + R * nx, py - R * ty + R * ny), (px + R * tx + R * nx, py + R * ty + R * ny)])
    lo, hi = -R / 4, R / 4; best, bestd = None, 1e9
    for _ in range(30):
        off = (lo + hi) / 2; r = poly.difference(half(off))
        if r.is_empty:
            lo = off; continue
        fr = 1 - r.area / orig
        if abs(fr - pct) < bestd:
            bestd, best = abs(fr - pct), r
        if fr > pct: lo = off
        else: hi = off
    if best is None or best.is_empty:
        return None, None
    if best.geom_type == "MultiPolygon":
        best = max(best.geoms, key=lambda g: g.area)
    if best.geom_type != "Polygon":
        return None, None
    af = 1 - best.area / orig
    if abs(af - pct) > 0.05:
        return None, None
    c = np.array(best.exterior.coords[:-1])
    return (c, af) if len(c) >= 6 else (None, None)


coco = json.load(open(COCO_DEFAULT))
by_img = {}
for a in coco["annotations"]:
    by_img.setdefault(str(a["image_id"]), []).append(a)
imgs = coco["images"]
rng0 = np.random.default_rng(7)
clean, rep_pool = [], []
for img in imgs[::5]:
    for a in by_img.get(str(img["id"]), []):
        pts = seg_points(a.get("segmentation"))
        if pts is None or len(pts) < 10:
            continue
        p = to_poly(pts)
        if p is None or p.area < 400:
            continue
        x0, y0, x1, y1 = p.bounds
        if x0 < 3 or y0 < 3 or x1 > W_IMG - 3 or y1 > H_IMG - 3:
            continue
        sol = p.area / p.convex_hull.area
        fit = fitzgibbon(hull_points(pts))
        if fit is None:
            continue
        if sol >= 0.85:
            rep_pool.append((pts, fit, sol))
        if sol >= 0.95:
            e = ell_poly(*fit["center"], *fit["axes"], fit["angle"])
            iou = p.intersection(e).area / p.union(e).area
            clean.append((pts, fit, iou))
clean.sort(key=lambda x: -x[2])
sets = {"clean": [(p, f) for p, f, _ in clean[:300]],
        "representative": [(rep_pool[i][0], rep_pool[i][1]) for i in rng0.choice(len(rep_pool), 300, replace=False)]}
print({k: len(v) for k, v in sets.items()}, "rep pool", len(rep_pool), flush=True)

out = {}
for sname, items in sets.items():
    rng = np.random.default_rng(42)
    rec = {m: {str(int(l * 100)): [] for l in LEVELS} for m in ["hull", "raw", "bbox"]}
    actual = {str(int(l * 100)): [] for l in LEVELS}
    signed = {str(int(l * 100)): [] for l in LEVELS}
    for pts, tf in items:
        tM, tm = 2 * tf["axes"][0], 2 * tf["axes"][1]
        poly = to_poly(pts)
        for l in LEVELS:
            clip, af = None, None
            for _ in range(8):
                clip, af = occlude(poly, l, rng)
                if clip is not None:
                    break
            if clip is None:
                continue
            key = str(int(l * 100)); actual[key].append(af)
            for m in ["hull", "raw"]:
                f = fitzgibbon(hull_points(clip) if m == "hull" else clip)
                if f is None:
                    rec[m][key].append(np.nan); continue
                M, mn = 2 * f["axes"][0], 2 * f["axes"][1]
                rec[m][key].append(0.5 * (abs(M - tM) / tM + abs(mn - tm) / tm))
                if m == "hull":
                    signed[key].append(0.5 * ((M - tM) / tM + (mn - tm) / tm))
            bw, bh = np.ptp(clip[:, 0]), np.ptp(clip[:, 1])
            rec["bbox"][key].append(0.5 * (abs(max(bw, bh) - tM) / tM + abs(min(bw, bh) - tm) / tm))
    summ = {}
    for key in actual:
        a = np.array(actual[key])
        summ[key] = {"n": int(len(a)), "actual_removed_mean": float(a.mean() * 100), "actual_removed_sd": float(a.std() * 100),
                     "actual_removed_min": float(a.min() * 100), "actual_removed_max": float(a.max() * 100)}
        for m in rec:
            e = np.array(rec[m][key], float); e = e[np.isfinite(e)] * 100
            summ[key][m] = {"mean": float(e.mean()), "sd": float(e.std()), "median": float(np.median(e)),
                            "p90": float(np.percentile(e, 90)), "fail": int(len(rec[m][key]) - len(e))}
        s = np.array(signed[key]) * 100
        summ[key]["hull_signed_mean"] = float(s.mean())
    out[sname] = {"summary": summ, "raw": {m: rec[m] for m in rec}, "actual": actual, "signed_hull": signed}
    print(sname, json.dumps({k: (v["actual_removed_mean"], v["hull"]["mean"], v["raw"]["mean"], v["bbox"]["mean"]) for k, v in summ.items()}), flush=True)
jdump(out, OUT)
print("DONE")
