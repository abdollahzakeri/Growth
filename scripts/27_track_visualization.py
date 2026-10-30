"""R17: qualitative tracking visualizations for the longest track (tau_IoU = 0.90).
(1) centroid path figure (path over the image crop, colored by time; displacement and size vs. time);
(2) GIF of the isolated mushroom with its tracked mask highlighted in every frame.
Usage: python r17_track_viz.py <outdir> [frame_step=2]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, json, glob, pickle
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from growth.common import RES, set_style, seg_points, COCO_DEFAULT, ROOT, PIX2MM, HRS_PER_FRAME, PALETTE
OUT = sys.argv[1]; STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2; os.makedirs(OUT, exist_ok=True)
TID = int(sys.argv[3]) if len(sys.argv) > 3 else None; TAG = f"track_{TID}" if TID else "track_longest"
T = pickle.load(open(os.path.join(RES, "tracks_t090.pkl"), "rb"))["tracks"]
span = np.array([t["frame"][-1] - t["frame"][0] + 1 for t in T])
cand = [i for i in range(len(T)) if span[i] == span.max()]
grow = [np.nanmax(T[i]["ell"][:, 0]) - np.nanmin(T[i]["ell"][:, 0]) for i in cand]
t = T[cand[int(np.argmax(grow))]] if TID is None else next(x for x in T if x["id"] == TID)
fr = t["frame"].astype(int); cx, cy = t["cx"].astype(float), t["cy"].astype(float)
print("track", t["id"], "frames", fr[0], fr[-1], "observed", len(fr), "of", fr[-1] - fr[0] + 1, "tied tracks", len(cand), flush=True)
# masks per observed frame from the COCO annotations (nearest centroid)
coco = json.load(open(COCO_DEFAULT)); ts = {im["id"]: im.get("timestep", im["id"]) for im in coco["images"]}
by = {}
want = set(fr.tolist())
for a in coco["annotations"]:
    f_ = ts[a["image_id"]]
    if f_ in want:
        p = seg_points(a.get("segmentation"))
        if p is not None: by.setdefault(f_, []).append(p)
del coco
masks = {}
for f_, x, y in zip(fr, cx, cy):
    best, bd = None, 1e9
    for p in by.get(f_, []):
        d = np.hypot(p[:, 0].mean() - x, p[:, 1].mean() - y)
        if d < bd: best, bd = p, d
    masks[f_] = best
# undistortion maps (same as the pipeline)
W, H = 1920, 1080; f = W / (2 * np.tan(np.deg2rad(110) / 2))
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
newK, _ = cv2.getOptimalNewCameraMatrix(K, np.array([-0.06, 0, 0, 0, 0.]), (W, H), 1, (W, H))
m1, m2 = cv2.initUndistortRectifyMap(K, np.array([-0.078, 0, 0, 0, 0.]), None, newK, (W, H), cv2.CV_16SC2)
files = sorted(glob.glob(os.path.join(ROOT, "images", "*.JPG")))
def frame(i):
    return cv2.remap(cv2.imread(files[i]), m1, m2, cv2.INTER_LINEAR)[165:165 + 697, 170:170 + 1580]
allp = np.vstack([m for m in masks.values() if m is not None]); mx, my = allp.mean(0)
half = int(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) * 0.5 + 80)
x0, y0 = int(max(mx - half, 0)), int(max(my - half, 0)); x1, y1 = int(min(mx + half, 1580)), int(min(my + half, 697))
hours = (fr - fr[0]) * HRS_PER_FRAME
# ---------------- (2) GIF ----------------
SZ = 360; sc = SZ / max(x1 - x0, y1 - y0); HL = (255, 140, 0)
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15)
except Exception:
    font = ImageFont.load_default()
gif = []
for k in range(0, len(fr), STEP):
    f_ = fr[k]; img = frame(f_)[y0:y1, x0:x1].copy(); m = masks[f_]
    if m is not None:
        poly = np.round(m - [x0, y0]).astype(np.int32)
        ov = img.copy(); cv2.fillPoly(ov, [poly], HL[::-1]); img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        cv2.polylines(img, [poly], True, HL[::-1], 2, cv2.LINE_AA)
    trail = np.column_stack([cx[:k + 1] - x0, cy[:k + 1] - y0])
    cv2.polylines(img, [np.round(trail).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (int(round(cx[k] - x0)), int(round(cy[k] - y0))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    img = cv2.resize(img, (int((x1 - x0) * sc), int((y1 - y0) * sc)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); canvas = Image.new("RGB", (pil.width, pil.height + 26), (255, 255, 255))
    canvas.paste(pil, (0, 26)); d = ImageDraw.Draw(canvas)
    d.text((6, 5), f"Track {t['id']}   t = {hours[k]:5.1f} h   major axis {t['ell'][k, 0] * PIX2MM:4.1f} mm", fill=(0, 0, 0), font=font)
    gif.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=96))
gif[0].save(os.path.join(OUT, TAG + ".gif"), save_all=True, append_images=gif[1:], duration=70, loop=0, optimize=True)
print("gif frames", len(gif), "size MB", os.path.getsize(os.path.join(OUT, TAG + ".gif")) / 1e6, flush=True)
# ---------------- (1) centroid path figure ----------------
plt = set_style()
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.05, 0.95, 1.35]})
last = frame(fr[-1])[y0:y1, x0:x1]
ax[0].imshow(cv2.cvtColor(last, cv2.COLOR_BGR2RGB), extent=[x0, x1, y1, y0])
for f_, col, lab in [(fr[0], "#56B4E9", "first mask (0 h)"), (fr[-1], "#E69F00", f"last mask ({hours[-1]:.0f} h)")]:
    m = masks[f_]
    if m is not None:
        ax[0].plot(np.r_[m[:, 0], m[0, 0]], np.r_[m[:, 1], m[0, 1]], color=col, lw=1.2, label=lab)
ax[0].plot(cx, cy, color="white", lw=0.8, zorder=3)
ax[0].set_xlim(x0, x1); ax[0].set_ylim(y1, y0); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].grid(False)
ax[0].legend(frameon=True, fontsize=6.5, loc="lower left", framealpha=0.85)
ax[0].set_title("(a) Masks and centroid path", fontsize=9)
dx, dy = (cx - cx[0]) * PIX2MM, (cy - cy[0]) * PIX2MM
sc_ = ax[1].scatter(dx, dy, c=hours, cmap="viridis", s=3, zorder=3); ax[1].plot(dx, dy, color="0.6", lw=0.4, zorder=2)
lim = max(np.abs(dx).max(), np.abs(dy).max()) * 1.15
ax[1].set_xlim(-lim, lim); ax[1].set_ylim(lim, -lim); ax[1].set_aspect("equal")
ax[1].set_xlabel("Horizontal (mm)"); ax[1].set_ylabel("Vertical (mm)")
cb = fig.colorbar(sc_, ax=ax[1], fraction=0.046, pad=0.03); cb.set_label("Time (h)")
ax[1].set_title("(b) Centroid relative to start", fontsize=9)
ax[2].plot(hours, np.hypot(dx, dy), color="k", lw=1.0, label="centroid distance from start")
ax[2].set_xlabel("Time since first detection (h)"); ax[2].set_ylabel("Distance (mm)")
a2 = ax[2].twinx(); a2.plot(hours, t["ell"][:, 0] * PIX2MM, color=PALETTE[2], lw=1.0, label="major diameter")
a2.set_ylabel("Major diameter (mm)", color=PALETTE[2]); a2.tick_params(axis="y", colors=PALETTE[2]); a2.grid(False); a2.spines["right"].set_visible(True)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
ax[2].set_ylim(0, max(np.hypot(dx, dy)) * 1.25)
ax[2].set_title("(c) Distance from start and size", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, TAG + "_centroid.png"), dpi=300); fig.savefig(os.path.join(OUT, TAG + "_centroid.pdf"))
print("max displacement mm", float(np.hypot(dx, dy).max()), "diam mm", float(np.nanmin(t["ell"][:, 0]) * PIX2MM), float(np.nanmax(t["ell"][:, 0]) * PIX2MM))
print("DONE")
