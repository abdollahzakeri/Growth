"""R01: IoU-Hungarian mask tracking (tau_IoU, tau_gap) with per-detection geometry.

Each detection stores bbox size, centroid, mask area, and the Fitzgibbon ellipse diameters
fitted to (a) the convex hull of the mask contour and (b) the raw contour.
Optional detection dropout (rate, seed) is used for the segmentation-sensitivity analysis.

Usage: python r01_track.py <out.pkl> [tau_iou=0.9] [tau_gap=10] [drop_rate=0] [seed=0] [coco]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, json, pickle, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
from growth.common import COCO_DEFAULT, seg_points, to_poly, ellipse_diams

OUT = sys.argv[1]
TAU_IOU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
TAU_GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 10
DROP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
COCO = sys.argv[6] if len(sys.argv) > 6 else COCO_DEFAULT


def feat(a):
    pts = seg_points(a.get("segmentation"))
    if pts is None:
        return None
    p = to_poly(pts)
    if p is None or p.area < 20:
        return None
    eh = ellipse_diams(pts, True); er = ellipse_diams(pts, False)
    bb = a["bbox"] if not isinstance(a["bbox"], str) else json.loads(a["bbox"])
    return {"pts": pts.astype(np.float32), "bbox": [float(v) for v in bb],
            "cx": float(p.centroid.x), "cy": float(p.centroid.y), "area": float(p.area),
            "solidity": float(p.area / max(p.convex_hull.area, 1e-9)),
            "ell": eh, "ell_raw": er, "conf": float(a.get("confidence", 1.0))}


def frame_feats(args):
    fid, anns = args
    return fid, [f for f in (feat(a) for a in anns) if f is not None]


if __name__ == "__main__":
    t0 = time.time()
    coco = json.load(open(COCO))
    images = sorted(coco["images"], key=lambda x: x.get("timestep", x["id"]))
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(str(a["image_id"]), []).append(a)
    jobs = [(img.get("timestep", img["id"]), by_img.get(str(img["id"]), [])) for img in images]
    del coco
    with Pool(16) as pool:
        frames = dict(pool.map(frame_feats, jobs, chunksize=8))
    print(f"features: {sum(len(v) for v in frames.values())} detections in {len(frames)} frames "
          f"({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    tracks, finished, next_id = [], [], 1
    n_det_total = 0
    for fid in sorted(frames):
        dets = frames[fid]
        if DROP > 0:
            dets = [d for d in dets if rng.random() >= DROP]
        n_det_total += len(dets)
        polys = [to_poly(d["pts"]) for d in dets]
        act = [t for t in tracks if t["lost"] <= TAU_GAP]
        matched_d = set()
        if act and dets:
            C = np.ones((len(act), len(dets)))
            for i, t in enumerate(act):
                tx0, ty0 = t["bbox"][0], t["bbox"][1]; tx1, ty1 = tx0 + t["bbox"][2], ty0 + t["bbox"][3]
                for j, d in enumerate(dets):
                    b = d["bbox"]
                    if b[0] + b[2] < tx0 or b[0] > tx1 or b[1] + b[3] < ty0 or b[1] > ty1:
                        continue
                    inter = t["poly"].intersection(polys[j]).area
                    if inter <= 0:
                        continue
                    iou = inter / (t["poly"].area + polys[j].area - inter)
                    if iou >= TAU_IOU:
                        C[i, j] = 1 - iou
            ri, cj = linear_sum_assignment(C)
            mt = set()
            for i, j in zip(ri, cj):
                if C[i, j] <= 1 - TAU_IOU:
                    t = act[i]; d = dets[j]
                    t["poly"] = polys[j]; t["bbox"] = d["bbox"]; t["lost"] = 0
                    t["obs"].append((fid, d)); mt.add(i); matched_d.add(j)
            for i, t in enumerate(act):
                if i not in mt:
                    t["lost"] += 1
        else:
            for t in act:
                t["lost"] += 1
        for j, d in enumerate(dets):
            if j not in matched_d:
                tracks.append({"id": next_id, "poly": polys[j], "bbox": d["bbox"], "lost": 0,
                               "obs": [(fid, d)]})
                next_id += 1
        still = []
        for t in tracks:
            (finished if t["lost"] > TAU_GAP else still).append(t)
        tracks = still
    finished.extend(tracks)

    out = []
    for t in finished:
        if len(t["obs"]) < 2:
            continue
        obs = t["obs"]
        out.append({
            "id": t["id"],
            "frame": np.array([f for f, _ in obs], np.int32),
            "bw": np.array([d["bbox"][2] for _, d in obs], np.float32),
            "bh": np.array([d["bbox"][3] for _, d in obs], np.float32),
            "cx": np.array([d["cx"] for _, d in obs], np.float32),
            "cy": np.array([d["cy"] for _, d in obs], np.float32),
            "area": np.array([d["area"] for _, d in obs], np.float32),
            "solidity": np.array([d["solidity"] for _, d in obs], np.float32),
            "conf": np.array([d["conf"] for _, d in obs], np.float32),
            "ell": np.array([d["ell"] if d["ell"] else (np.nan, np.nan) for _, d in obs], np.float32),
            "ell_raw": np.array([d["ell_raw"] if d["ell_raw"] else (np.nan, np.nan) for _, d in obs], np.float32),
            "first_pts": obs[0][1]["pts"], "last_pts": obs[-1][1]["pts"],
        })
    meta = {"tau_iou": TAU_IOU, "tau_gap": TAU_GAP, "drop": DROP, "seed": SEED, "coco": COCO,
            "n_frames": len(frames), "n_det": n_det_total,
            "det_per_frame": float(n_det_total / max(len(frames), 1)), "n_tracks": len(out)}
    pickle.dump({"meta": meta, "tracks": out}, open(OUT, "wb"))
    print(f"{meta} -> {OUT} ({time.time()-t0:.0f}s)", flush=True)
