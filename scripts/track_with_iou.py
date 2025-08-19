import argparse
import json
from typing import Dict, Any, List

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial.distance import euclidean


def load_coco(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def track(
    coco_json: str,
    output_json: str,
    max_lost_frames: int = 30,
    max_center_dist: float = 50.0,
    min_iou: float = 0.9,
    area_drop_threshold: float = 0.99,
) -> None:
    coco = load_coco(coco_json)
    images = sorted(coco["images"], key=lambda x: x.get("timestep", x["id"]))
    annotations = coco["annotations"]

    ann_map: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    tracks: List[Dict[str, Any]] = []
    next_id = 1

    for img in images:
        img_id = img["id"]
        dets: List[Dict[str, Any]] = []

        for ann in ann_map.get(img_id, []):
            seg = ann.get("segmentation", [])
            if seg and isinstance(seg[0], (list, tuple)):
                pts = seg
            elif seg and isinstance(seg[0], (int, float)):
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            else:
                continue
            if len(pts) < 3:
                continue

            raw = Polygon(pts)
            clean = raw.buffer(0)
            if clean.is_empty:
                continue
            if isinstance(clean, MultiPolygon):
                clean = max(clean.geoms, key=lambda p: p.area)
            poly = clean
            center = (poly.centroid.x, poly.centroid.y)
            dets.append({"ann": ann, "poly": poly, "center": center, "track_id": None})

        for tr in tracks:
            tr["matched"] = False

        for det in dets:
            best_tr = None
            best_score = -np.inf
            for tr in tracks:
                if tr["lost"] > max_lost_frames or tr["matched"] or tr.get("finished", False):
                    continue
                dist = euclidean(det["center"], tr["center"])
                if dist > max_center_dist:
                    continue
                tr_poly = tr["poly"].buffer(0)
                inter = det["poly"].intersection(tr_poly).area
                if inter == 0:
                    continue
                union = det["poly"].area + tr_poly.area - inter
                iou = inter / union
                if iou < min_iou:
                    continue
                score = iou - 0.1 * (dist / max_center_dist)
                if score > best_score:
                    best_score = score
                    best_tr = tr

            if best_tr:
                curr_area = det["poly"].area
                if curr_area < area_drop_threshold * best_tr["max_area"]:
                    best_tr["finished"] = True
                    best_tr = None
                else:
                    det["track_id"] = best_tr["id"]
                    best_tr.update({
                        "poly": det["poly"],
                        "center": det["center"],
                        "lost": 0,
                        "matched": True,
                        "max_area": max(best_tr["max_area"], curr_area),
                    })

            if best_tr is None:
                det["track_id"] = next_id
                area0 = det["poly"].area
                tracks.append({
                    "id": next_id,
                    "poly": det["poly"],
                    "center": det["center"],
                    "lost": 0,
                    "matched": True,
                    "max_area": area0,
                    "finished": False,
                })
                next_id += 1

        for tr in tracks:
            if not tr["matched"] and not tr.get("finished", False):
                tr["lost"] += 1

        for det in dets:
            det["ann"]["track_id"] = det["track_id"]

    coco["annotations"] = annotations
    save_json(coco, output_json)


def main():
    p = argparse.ArgumentParser(description="Track masks across frames using IoU and simple motion prior")
    p.add_argument("--coco", required=True, help="Input COCO JSON from YOLO segmentation")
    p.add_argument("--out", required=True, help="Output tracked COCO JSON")
    p.add_argument("--max_lost", type=int, default=30)
    p.add_argument("--max_dist", type=float, default=50.0)
    p.add_argument("--min_iou", type=float, default=0.9)
    p.add_argument("--area_drop", type=float, default=0.99)
    args = p.parse_args()

    track(
        coco_json=args.coco,
        output_json=args.out,
        max_lost_frames=args.max_lost,
        max_center_dist=args.max_dist,
        min_iou=args.min_iou,
        area_drop_threshold=args.area_drop,
    )


if __name__ == "__main__":
    main()


