"""
Step 1 — Barrel-distortion correction + crop.

Reproduces the exact procedure that produced the original `undistorted_cropped/`
(verified: undistort with nominal-FOV pinhole K, getOptimalNewCameraMatrix(alpha=1)
computed at the ORIGINAL k1=-0.06 scale, then a fixed 1580x697 crop at offset (170,165);
mean|diff| vs the shipped images = 1.9 / 255, corr = 0.9994 -> JPEG noise only).

The ONLY thing that changes between runs is the radial coefficient k1 used in the
distortion model D=[k1,0,0,0].  Crucially, the new-camera matrix (output scale) is held
FIXED at the k1=-0.06 value so that:
  * the downstream pixel->mm constant (30/63 mm/px) stays valid, and
  * the 1580x697 crop stays inside the valid (non-black) region (verified: 0% invalid).

Usage:  python 01_undistort_crop.py <k1> <out_dir>
Example: python 01_undistort_crop.py -0.078 undistorted_cropped_078
"""
import cv2, numpy as np, os, sys, glob

ROOT = os.environ.get("GROWTH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SRC  = os.path.join(ROOT, "images")
K1   = float(sys.argv[1]) if len(sys.argv) > 1 else -0.078
OUT  = os.path.join(ROOT, sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "undistorted_cropped_078")

W, H = 1920, 1080
f = W / (2 * np.tan(np.deg2rad(110) / 2))          # nominal-FOV pinhole focal
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)

# Fixed output scale: newK from the ORIGINAL k1=-0.06 (preserves px->mm + crop geometry)
D_REF = np.array([-0.06, 0, 0, 0, 0], np.float64)
newK, _ = cv2.getOptimalNewCameraMatrix(K, D_REF, (W, H), 1, (W, H))

D = np.array([K1, 0, 0, 0, 0], np.float64)
X0, Y0, CW, CH = 170, 165, 1580, 697               # verified fixed crop

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*.JPG")) + glob.glob(os.path.join(SRC, "*.jpg")))
print(f"k1={K1}  ->  {OUT}   ({len(files)} images), newK fx={newK[0,0]:.2f}")
for i, fp in enumerate(files):
    img = cv2.imread(fp)
    und = cv2.undistort(img, K, D, None, newK)
    crop = und[Y0:Y0 + CH, X0:X0 + CW]
    cv2.imwrite(os.path.join(OUT, os.path.basename(fp)), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    if i % 400 == 0:
        print(f"  {i}/{len(files)}", flush=True)
print("done")
