"""Break down segmentation time: Ultralytics preprocess / TensorRT inference / postprocess (NMS + mask decoding) and
mask-to-polygon conversion (masks.xy), over 40 frames after 3 warm-up frames."""
import sys, os, glob, time, json
import numpy as np, cv2
from ultralytics import YOLO
# D: folder with frames/, train_l5.npz, train_l10.npz and brown.csv (set JETSON_PKG to override)
D = os.environ.get("JETSON_PKG", os.path.dirname(os.path.abspath(__file__))); ENG = sys.argv[1]
raw = sorted(glob.glob(os.path.join(D, "frames", "*.JPG")))[:43]
W, H = 1920, 1080; f = W / (2 * np.tan(np.deg2rad(110) / 2))
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
newK, _ = cv2.getOptimalNewCameraMatrix(K, np.array([-0.06, 0, 0, 0, 0.]), (W, H), 1, (W, H))
m1, m2 = cv2.initUndistortRectifyMap(K, np.array([-0.078, 0, 0, 0, 0.]), None, newK, (W, H), cv2.CV_16SC2)
seg = YOLO(ENG, task="segment"); R = {k: [] for k in ["pre", "inf", "post", "xy", "wall", "n"]}
for k, p in enumerate(raw):
    und = cv2.remap(cv2.imread(p), m1, m2, cv2.INTER_LINEAR)[165:165 + 697, 170:170 + 1580]
    t0 = time.perf_counter(); r = seg.predict(und, conf=0.1, verbose=False, device=0)[0]; t1 = time.perf_counter()
    xy = [] if r.masks is None else r.masks.xy; t2 = time.perf_counter()
    if k >= 3:
        R["pre"].append(r.speed["preprocess"]); R["inf"].append(r.speed["inference"]); R["post"].append(r.speed["postprocess"])
        R["xy"].append((t2 - t1) * 1000); R["wall"].append((t1 - t0) * 1000); R["n"].append(len(xy))
out = {k: [float(np.mean(v)), float(np.std(v))] for k, v in R.items()}
n = np.array(R["n"]); post = np.array(R["post"]); out["post_ms_per_cap"] = float(np.polyfit(n, post, 1)[0])
print(json.dumps(out, indent=1))
