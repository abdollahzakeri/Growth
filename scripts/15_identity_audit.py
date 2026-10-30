"""R14: visual audit of identity decisions. Renders image pairs (before | after) for
 (a) fragment links used for grouping (gap > tau_gap), (b) tracker re-associations after 1-10 missed frames,
 (c) cross-group near misses (0.1 <= IoU < 0.3, gap <= 24 h; candidate missed links).
Usage: python r14_audit.py <outdir>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, json, glob, pickle
import numpy as np, cv2
from growth.common import RES, to_poly, seg_points, COCO_DEFAULT, ROOT
OUTD = sys.argv[1]; os.makedirs(OUTD, exist_ok=True)
W, H = 1920, 1080
f = W / (2 * np.tan(np.deg2rad(110) / 2)); K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
newK, _ = cv2.getOptimalNewCameraMatrix(K, np.array([-0.06, 0, 0, 0, 0.]), (W, H), 1, (W, H))
D = np.array([-0.078, 0, 0, 0, 0.]); files = sorted(glob.glob(os.path.join(ROOT, "images", "*.JPG")))
mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (W, H), cv2.CV_32FC1)
cache = {}
def frame(i):
    if i not in cache:
        u = cv2.remap(cv2.imread(files[i]), mapx, mapy, cv2.INTER_LINEAR); cache[i] = u[165:165 + 697, 170:170 + 1580]
    return cache[i]
coco = json.load(open(COCO_DEFAULT)); by = {}
ts = {im["id"]: im.get("timestep", im["id"]) for im in coco["images"]}
for a in coco["annotations"]:
    by.setdefault(ts[a["image_id"]], []).append(seg_points(a.get("segmentation")))
del coco
def mask_at(fr, cx, cy):
    best, bd = None, 1e9
    for p in by.get(fr, []):
        if p is None: continue
        d = np.hypot(p[:, 0].mean() - cx, p[:, 1].mean() - cy)
        if d < bd: best, bd = p, d
    return best
S = pickle.load(open(os.path.join(RES, "seqs_t090.pkl"), "rb"))["seqs"]
T = pickle.load(open(os.path.join(RES, "tracks_t090.pkl"), "rb"))["tracks"]
tid = {t["id"]: i for i, t in enumerate(T)}; grp = {s["id"]: s["group"] for s in S}
seq_t = [tid[s["id"]] for s in S]
lastp = {i: to_poly(T[i]["last_pts"]) for i in seq_t}; firstp = {i: to_poly(T[i]["first_pts"]) for i in seq_t}
def iou(a, b):
    if a is None or b is None: return 0.0
    i = a.intersection(b).area; return i / (a.area + b.area - i) if i > 0 else 0.0
rel = []
for b in seq_t:
    for a in seq_t:
        if T[a]["frame"][-1] < T[b]["frame"][0]:
            pa, pb = lastp[a], firstp[b]
            if pa is None or pb is None: continue
            ba, bb = pa.bounds, pb.bounds
            if ba[0] > bb[2] or ba[2] < bb[0] or ba[1] > bb[3] or ba[3] < bb[1]: continue
            v = iou(pa, pb)
            if v > 0: rel.append((a, b, v, int(T[b]["frame"][0] - T[a]["frame"][-1])))
rng = np.random.default_rng(3)
A = [r for r in rel if r[2] >= 0.3 and grp[T[r[0]]["id"]] == grp[T[r[1]]["id"]]]
C = [r for r in rel if 0.1 <= r[2] < 0.3 and r[3] <= 96 and grp[T[r[0]]["id"]] != grp[T[r[1]]["id"]]]
cases = {"a_links": [(T[a]["frame"][-1], T[a]["last_pts"], T[b]["frame"][0], T[b]["first_pts"], f"gap {g*0.25:.1f} h IoU {v:.2f}")
                     for a, b, v, g in [A[i] for i in rng.choice(len(A), min(40, len(A)), replace=False)]],
         "c_nearmiss": [(T[a]["frame"][-1], T[a]["last_pts"], T[b]["frame"][0], T[b]["first_pts"], f"gap {g*0.25:.1f} h IoU {v:.2f}")
                        for a, b, v, g in [C[i] for i in rng.choice(len(C), min(32, len(C)), replace=False)]]}
G = []
for i in seq_t:
    t = T[i]; d = np.diff(t["frame"])
    for k in np.where(d > 1)[0]:
        G.append((i, k))
bsel = [G[j] for j in rng.choice(len(G), 40, replace=False)]
cases["b_gaps"] = []
for i, k in bsel:
    t = T[i]; f1, f2 = int(t["frame"][k]), int(t["frame"][k + 1])
    m1, m2 = mask_at(f1, t["cx"][k], t["cy"][k]), mask_at(f2, t["cx"][k + 1], t["cy"][k + 1])
    cases["b_gaps"].append((f1, m1, f2, m2, f"missed {f2-f1-1} frames"))
info = {"n_links": len(A), "n_nearmiss": len(C), "n_gap_events": len(G)}
TS = 220
def tile(fr, pts, box, col):
    x0, y0, x1, y1 = box; im = frame(fr)[y0:y1, x0:x1].copy()
    if pts is not None:
        cv2.polylines(im, [np.round(pts - [x0, y0]).astype(np.int32)], True, col, 2)
    return cv2.resize(im, (TS, TS))
for name, L in cases.items():
    for page in range(0, len(L), 8):
        rows = []
        for j, (f1, p1, f2, p2, lab) in enumerate(L[page:page + 8]):
            P = np.vstack([q for q in (p1, p2) if q is not None]); cx, cy = P.mean(0); r = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) * 0.9 + 20
            box = (int(max(cx - r, 0)), int(max(cy - r, 0)), int(min(cx + r, 1580)), int(min(cy + r, 697)))
            pair = np.hstack([tile(f1, p1, box, (0, 0, 255)), np.full((TS, 6, 3), 255, np.uint8), tile(f2, p2, box, (0, 255, 0))])
            lab = f"#{page+j+1} {lab}"
            hdr = np.full((22, pair.shape[1], 3), 255, np.uint8); cv2.putText(hdr, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            rows.append(np.vstack([hdr, pair]))
        while len(rows) % 2: rows.append(np.full_like(rows[0], 255))
        grid = np.vstack([np.hstack([rows[i], np.full((rows[i].shape[0], 12, 3), 255, np.uint8), rows[i + 1]]) for i in range(0, len(rows), 2)])
        cv2.imwrite(os.path.join(OUTD, f"{name}_{page//8+1}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(info, open(os.path.join(OUTD, "info.json"), "w")); print(info, "DONE")
