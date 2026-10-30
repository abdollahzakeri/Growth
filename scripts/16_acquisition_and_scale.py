"""R09: acquisition facts, size correlation, flush structure, scale plausibility.
Usage: python r09_misc.py <tracks.pkl> <seqs.pkl> <out.json>
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, glob, re, pickle
import numpy as np, pandas as pd
from PIL import Image
from growth.common import ROOT, PIX2MM, HRS_PER_FRAME, jdump
TR, SQ, OUT = sys.argv[1:4]
res = {}
fs = sorted(glob.glob(os.path.join(ROOT, "images", "*.JPG")))
nums = np.array([int(re.findall(r"(\d+)", os.path.basename(f))[0]) for f in fs])
res["n_frames"] = len(fs); res["first"] = os.path.basename(fs[0]); res["last"] = os.path.basename(fs[-1])
res["numbering_contiguous"] = bool(np.all(np.diff(nums) == 1))
def dt(f):
    try:
        e = Image.open(f)._getexif() or {}
        return e.get(36867) or e.get(306)
    except Exception:
        return None
stamps = [dt(f) for f in fs]
res["exif_available"] = int(sum(s is not None for s in stamps))
if res["exif_available"] > 10:
    ts = pd.to_datetime([s for s in stamps if s], format="%Y:%m:%d %H:%M:%S")
    d = np.diff(ts.values).astype("timedelta64[s]").astype(float) / 60
    res["exif_first"], res["exif_last"] = str(ts[0]), str(ts[-1])
    res["span_h"] = float((ts[-1] - ts[0]).total_seconds() / 3600)
    res["interval_min_median"] = float(np.median(d)); res["interval_min_max"] = float(d.max())
res["span_h_from_count"] = float((len(fs) - 1) * HRS_PER_FRAME)
# detections per frame (flush structure)
D = pickle.load(open(TR, "rb"))
cnt = np.zeros(res["n_frames"])
for t in D["tracks"]:
    cnt[t["frame"]] += 1
k = 24; sm = np.convolve(cnt, np.ones(k) / k, mode="same")
res["tracked_caps_per_frame"] = {"mean": float(cnt.mean()), "max": float(cnt.max())}
res["caps_per_frame_6h_smoothed"] = sm[::24].tolist()
# size correlation (major vs minor along growth) on 50 random sequences
S = pickle.load(open(SQ, "rb"))["seqs"]
rng = np.random.default_rng(0); idx = rng.choice(len(S), 50, replace=False)
cors = []
for i in idx:
    d = S[i]["d"]
    if np.std(d[:, 0]) > 0 and np.std(d[:, 1]) > 0:
        cors.append(float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]))
res["major_minor_corr"] = {"values": cors, "mean": float(np.mean(cors)), "median": float(np.median(cors)),
                           "lengths": [int(S[i]["len"]) for i in idx]}
# scale plausibility: caps at their maximum tracked size vs caliper data of the same variety
mx = np.array([np.sqrt(np.prod(s["d"][np.argmax(s["d"][:, 0])])) * PIX2MM for s in S if s["len"] >= 96])
b = pd.read_csv(os.path.join(ROOT, "data", "brown.csv"), encoding="utf-8-sig"); gm = np.sqrt(b.x * b.y)
res["scale_check"] = {"tracked_max_gm_mm_quantiles": {q: float(np.percentile(mx, q)) for q in [10, 50, 90, 99]},
                      "n_tracked": int(len(mx)),
                      "caliper_gm_mm_quantiles": {q: float(np.percentile(gm, q)) for q in [10, 50, 90, 99]}}
res["fov_mm"] = [1580 * PIX2MM, 697 * PIX2MM]; res["fov_m2"] = 1580 * 697 * PIX2MM ** 2 / 1e6
jdump(res, OUT)
print({k: v for k, v in res.items() if k not in ["caps_per_frame_6h_smoothed"]})
