"""Per-image runtime of the complete pipeline on the Jetson (same stages as r08_runtime.py on the workstation).
Stages: undistortion+crop, YOLOv8-X segmentation (TensorRT FP16 engine), convex-hull Fitzgibbon ellipse fits,
72-h (288-step) recursive forecast of all caps (linear AR and Extra-Trees), weight regression (KNN) of all caps at
all 289 time steps. 100 frames, first 3 excluded as warm-up. Usage: python3 jetson_bench.py <engine> <tag> [n_frames]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, os, glob, time, json, subprocess, platform
import numpy as np, cv2, pandas as pd
from growth.common import ellipse_diams
from growth.weights import models as wmodels
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from ultralytics import YOLO
ENG, TAG = sys.argv[1], sys.argv[2]; NF = int(sys.argv[3]) if len(sys.argv) > 3 else 100
# D: folder with frames/, train_l5.npz, train_l10.npz and brown.csv (set JETSON_PKG to override)
D = os.environ.get("JETSON_PKG", os.path.dirname(os.path.abspath(__file__))); PIX2MM = 419.0 / 879.2
raw = sorted(glob.glob(os.path.join(D, "frames", "*.JPG")))[:NF]
W, H = 1920, 1080; f = W / (2 * np.tan(np.deg2rad(110) / 2))
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
newK, _ = cv2.getOptimalNewCameraMatrix(K, np.array([-0.06, 0, 0, 0, 0.]), (W, H), 1, (W, H))
m1, m2 = cv2.initUndistortRectifyMap(K, np.array([-0.078, 0, 0, 0, 0.]), None, newK, (W, H), cv2.CV_16SC2)
t = time.time()
a = np.load(os.path.join(D, "train_l5.npz")); fc_lin = Ridge(alpha=1e-3).fit(a["X"], a["Y"])
b = np.load(os.path.join(D, "train_l10.npz")); fc_et = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs=-1).fit(b["X"], b["Y"])
fc_et.set_params(n_jobs=os.cpu_count())
bd = pd.read_csv(os.path.join(D, "brown.csv"), encoding="utf-8-sig")
wm = wmodels()["K-nearest neighbors"].fit(bd[["x", "y"]].values.astype(float), bd["w"].values.astype(float))
print(f"models ready ({time.time()-t:.0f}s)", flush=True)
seg = YOLO(ENG, task="segment")
ts = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
T = {k: [] for k in ["undistort", "segment", "ellipse", "forecast_lin", "forecast_et", "weight", "n_caps"]}
def roll(model, d):
    Wd = d.copy(); out = [Wd]
    for _ in range(288):
        Wd = model.predict(Wd); out.append(Wd)
    return np.stack(out, 1)                         # caps x 289 x 2
for k, p in enumerate(raw):
    img = cv2.imread(p)
    t0 = time.perf_counter(); und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)[165:165 + 697, 170:170 + 1580]
    t1 = time.perf_counter(); r = seg.predict(und, conf=0.1, verbose=False, device=0)[0]
    polys = [] if r.masks is None else r.masks.xy
    t2 = time.perf_counter()
    d = [ellipse_diams(np.asarray(q, float), True) for q in polys if len(q) >= 6]
    d = np.array([x for x in d if x is not None], float); t3 = time.perf_counter()
    traj = roll(fc_lin, d) if len(d) else np.zeros((0, 289, 2)); t4 = time.perf_counter()
    if len(d): roll(fc_et, d)
    t5 = time.perf_counter()
    if len(d): wm.predict(traj.reshape(-1, 2) * PIX2MM).reshape(len(d), 289).sum(0)
    t6 = time.perf_counter()
    if k >= 3:
        for key, v in zip(["undistort", "segment", "ellipse", "forecast_lin", "forecast_et", "weight"], [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5]):
            T[key].append(v * 1000)
        T["n_caps"].append(len(d))
    if k % 20 == 0: print(k, {kk: round(vv[-1], 1) for kk, vv in T.items() if vv}, flush=True)
ts.terminate(); lines = ts.stdout.read().splitlines()
ram = [int(x.split("/")[0]) for l in lines for x in [l.split("RAM ")[1].split("MB")[0]] if "RAM " in l]
pw = [int(l.split("VDD_IN ")[1].split("mW")[0]) for l in lines if "VDD_IN " in l]
A = {k: np.array(v) for k, v in T.items()}
tot_lin = A["undistort"] + A["segment"] + A["ellipse"] + A["forecast_lin"] + A["weight"]
tot_et = A["undistort"] + A["segment"] + A["ellipse"] + A["forecast_et"] + A["weight"]
res = {"tag": TAG, "engine": ENG, "frames": len(A["segment"]), **{k: [float(v.mean()), float(v.std())] for k, v in A.items()},
       "total_linear_ms": [float(tot_lin.mean()), float(tot_lin.std())], "total_et_ms": [float(tot_et.mean()), float(tot_et.std())],
       "et_ms_per_step": float(A["forecast_et"].mean() / 288), "ram_mb_peak": max(ram) if ram else None,
       "power_mw_mean": float(np.mean(pw)) if pw else None, "power_mw_peak": max(pw) if pw else None,
       "nvpmodel": subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.strip().splitlines()[:2],
       "l4t": open("/etc/nv_tegra_release").readline().strip(), "python": platform.python_version()}
json.dump(res, open(os.path.join(D, f"bench_{TAG}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
