import argparse
import json
from typing import Dict, Any, List

import numpy as np
import cv2


def fit_ellipse_axes(points: np.ndarray) -> List[float]:
    if points.shape[0] < 5:
        raise ValueError("At least 5 points are required to fit an ellipse")
    # cv2.fitEllipse expects (N,1,2) int32 or float32
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(pts)
    # Major/minor diameters are the returned sizes (already diameters)
    major = float(max(MA, ma))
    minor = float(min(MA, ma))
    return [major, minor]


def process(in_json: str, out_json: str) -> None:
    with open(in_json, "r") as f:
        coco = json.load(f)

    updated = 0
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation", [])
        points = []
        if seg and isinstance(seg[0], (list, tuple)):
            # Already a list of [x1,y1,x2,y2,...] or list of arrays
            if isinstance(seg[0], (int, float)):
                flat = seg
                pts = np.array([[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)], dtype=np.float32)
                points = pts
            else:
                # list of polygons; pick the largest by area (approx via poly length)
                polys = []
                for poly in seg:
                    arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    polys.append(arr)
                # choose the polygon with max number of points as proxy
                points = max(polys, key=lambda a: a.shape[0])
        if len(points) >= 5:
            try:
                major, minor = fit_ellipse_axes(np.asarray(points))
                ann["x"] = float(major)
                ann["y"] = float(minor)
                updated += 1
            except Exception:
                pass
        else:
            # fallback from bbox
            bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            ann["x"] = float(bw)
            ann["y"] = float(bh)

    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Ellipse-completed diameters written to {out_json} (updated {updated} annotations)")


def main():
    ap = argparse.ArgumentParser(description="Fit ellipses to mask contours to obtain x/y diameters per annotation")
    ap.add_argument("--in", dest="in_json", required=True, help="Input COCO JSON (tracked or untracked)")
    ap.add_argument("--out", dest="out_json", required=True, help="Output COCO JSON with x/y fields")
    args = ap.parse_args()
    process(args.in_json, args.out_json)


if __name__ == "__main__":
    main()


