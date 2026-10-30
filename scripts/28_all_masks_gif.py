"""R19b: same crop and time window as R18, but every detected mask in view, colored by its track.
Usage: python r18_neighbors_gif.py <outdir> <track_id> [k=10] [frame_step=2]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, json, glob, pickle
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from growth.common import RES, seg_points, to_poly, COCO_DEFAULT, ROOT, PIX2MM, HRS_PER_FRAME
OUT, TID = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 10; STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)
T = pickle.load(open(os.path.join(RES, "tracks_t090.pkl"), "rb"))["tracks"]
tgt = next(t for t in T if t["id"] == TID); F0, F1 = int(tgt["frame"][0]), int(tgt["frame"][-1])
tc = {int(f): (x, y) for f, x, y in zip(tgt["frame"], tgt["cx"], tgt["cy"])}
cands = []
for t in T:
    if t["id"] == TID or t["frame"][-1] < F0 or t["frame"][0] > F1:
        continue
    ov = [(np.hypot(x - tc[int(f)][0], y - tc[int(f)][1])) for f, x, y in zip(t["frame"], t["cx"], t["cy"]) if int(f) in tc]
    if len(ov) >= 8:                                # overlaps at least 2 h with the target
        cands.append((float(np.mean(ov)), t))
cands.sort(key=lambda c: c[0]); nb = [t for _, t in cands[:K]]
tracks = [tgt] + nb
print("target", TID, "frames", F0, F1, "neighbors:", [(t["id"], int(t["frame"][0]), int(t["frame"][-1]), round(d * PIX2MM, 1)) for d, t in cands[:K]], flush=True)
want = set(range(F0, F1 + 1))
coco = json.load(open(COCO_DEFAULT)); ts = {im["id"]: im.get("timestep", im["id"]) for im in coco["images"]}
by = {}
for a in coco["annotations"]:
    f_ = ts[a["image_id"]]
    if f_ in want:
        p = seg_points(a.get("segmentation"))
        if p is not None:
            q = to_poly(p)
            if q is not None and q.area >= 20: by.setdefault(f_, []).append((p, q.centroid.x, q.centroid.y))
del coco
# all tracks that are observed in the crop window; map (frame, centroid) -> track id
tmap = {}
for t in T:
    if t["frame"][-1] < F0 or t["frame"][0] > F1: continue
    for f_, x, y in zip(t["frame"], t["cx"], t["cy"]):
        if F0 <= f_ <= F1: tmap.setdefault(int(f_), []).append((t["id"], x, y))
masks = []
for md_t in [tracks[0]]:
    md = {}
    for f_, x, y in zip(md_t["frame"], md_t["cx"], md_t["cy"]):
        f_ = int(f_)
        for p, qx, qy in by.get(f_, []):
            if np.hypot(qx - x, qy - y) < 2: md[f_] = p
    masks.append(md)
allp = np.vstack([p for md in masks for p in md.values()] + [p for md in [{}] for p in md.values()]) if False else None
nbm = []
for t in tracks:
    for f_, x, y in zip(t["frame"], t["cx"], t["cy"]):
        f_ = int(f_)
        if F0 <= f_ <= F1:
            for p, qx, qy in by.get(f_, []):
                if np.hypot(qx - x, qy - y) < 2: nbm.append(p)
allp = np.vstack(nbm)
x0, y0 = int(max(allp[:, 0].min() - 25, 0)), int(max(allp[:, 1].min() - 25, 0))
x1, y1 = int(min(allp[:, 0].max() + 25, 1580)), int(min(allp[:, 1].max() + 25, 697))
W, H = 1920, 1080; f = W / (2 * np.tan(np.deg2rad(110) / 2))
Km = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
newK, _ = cv2.getOptimalNewCameraMatrix(Km, np.array([-0.06, 0, 0, 0, 0.]), (W, H), 1, (W, H))
m1, m2 = cv2.initUndistortRectifyMap(Km, np.array([-0.078, 0, 0, 0, 0.]), None, newK, (W, H), cv2.CV_16SC2)
files = sorted(glob.glob(os.path.join(ROOT, "images", "*.JPG")))
# colors (RGB): target orange, neighbors from a colorblind-friendly qualitative palette
COLS = [(255, 140, 0), (0, 114, 178), (0, 158, 115), (204, 121, 167), (86, 180, 233), (240, 228, 66),
        (220, 20, 60), (0, 0, 0), (148, 103, 189), (140, 86, 75), (23, 190, 207)]
SZ = 560; sc = SZ / max(x1 - x0, y1 - y0)
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15); small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 12)
except Exception:
    font = small = ImageFont.load_default()
import colorsys
def tcol(tid):
    if tid == TID: return (255, 140, 0)
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95); return (int(r * 255), int(g * 255), int(b * 255))
gif = []; n_det = []; n_untracked = []
for f_ in range(F0, F1 + 1, STEP):
    img = cv2.remap(cv2.imread(files[f_]), m1, m2, cv2.INTER_LINEAR)[165:165 + 697, 170:170 + 1580][y0:y1, x0:x1].copy()
    ov = img.copy(); draw = []
    for p, qx, qy in by.get(f_, []):
        if p[:, 0].max() < x0 or p[:, 0].min() > x1 or p[:, 1].max() < y0 or p[:, 1].min() > y1: continue
        tid = None
        for t_id, x, y in tmap.get(f_, []):
            if np.hypot(qx - x, qy - y) < 2: tid = t_id; break
        poly = np.round(p - [x0, y0]).astype(np.int32)
        if tid is not None:
            c = tcol(tid); cv2.fillPoly(ov, [poly], c[::-1])
        draw.append((tid, poly))
    img = cv2.addWeighted(ov, 0.40, img, 0.60, 0)
    for tid, poly in draw:
        c = (200, 200, 200) if tid is None else tcol(tid)
        cv2.polylines(img, [poly], True, c[::-1], 3 if tid == TID else 2, cv2.LINE_AA)
    n_det.append(len(draw)); n_untracked.append(sum(1 for t_, _ in draw if t_ is None))
    img = cv2.resize(img, (int((x1 - x0) * sc), int((y1 - y0) * sc)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    for tid, poly in draw:
        if tid is None: continue
        cx, cy = poly.mean(0) * sc; tag = str(tid)
        if 6 < cx < pil.width - 6 and 6 < cy < pil.height - 6:
            d.rectangle([cx - 4 * len(tag) - 3, cy - 8, cx + 4 * len(tag) + 3, cy + 8], fill=(255, 255, 255))
            d.text((cx - 4 * len(tag), cy - 7), tag, fill=(0, 0, 0), font=small)
    canvas = Image.new("RGB", (pil.width, pil.height + 28), (255, 255, 255)); canvas.paste(pil, (0, 28))
    ImageDraw.Draw(canvas).text((6, 6), f"All detected masks (color = track; gray outline = not kept in a track)   t = {(f_ - F0) * HRS_PER_FRAME:5.1f} h", fill=(0, 0, 0), font=font)
    gif.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
print("detections per frame in crop: mean", round(float(np.mean(n_det)), 1), "untracked mean", round(float(np.mean(n_untracked)), 2))
p = os.path.join(OUT, f"track_{TID}_all_masks.gif")
gif[0].save(p, save_all=True, append_images=gif[1:], duration=80, loop=0, optimize=True)
print("gif", p, "frames", len(gif), "MB", round(os.path.getsize(p) / 1e6, 1), "crop", (x1 - x0, y1 - y0)); print("DONE")
