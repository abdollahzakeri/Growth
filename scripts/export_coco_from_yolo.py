import argparse
import json
import os
from typing import List, Dict, Any
from PIL import Image

from ultralytics import YOLO


def export_coco(
    image_dir: str,
    model_path: str,
    output_json: str,
    confidence: float = 0.1,
) -> None:
    model = YOLO(model_path)

    categories = [{"id": cid, "name": name} for cid, name in model.names.items()]

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 0

    files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )

    for timestep, filename in enumerate(files):
        image_id = timestep
        path = os.path.join(image_dir, filename)
        with Image.open(path) as img:
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "timestep": timestep,
        })

        results = model.predict(path, conf=confidence, stream=False)
        res = results[0]

        boxes = res.boxes
        masks = getattr(res, "masks", None)
        for inst_idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = bbox[2] * bbox[3]

            segmentation: List[List[float]] = []
            if masks is not None and hasattr(masks, "xy"):
                polys = masks.xy[inst_idx]
                for poly in polys:
                    segmentation.append(poly.flatten().tolist())

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [float(x) for x in bbox],
                "area": float(area),
                "segmentation": segmentation,
                "confidence": conf,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run YOLO segmentation and export COCO JSON")
    parser.add_argument("--images", required=True, help="Folder of undistorted & cropped images")
    parser.add_argument("--model", required=True, help="Path to YOLO model (e.g., best_v3.pt)")
    parser.add_argument("--out", required=True, help="Output COCO JSON path")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()

    export_coco(args.images, args.model, args.out, confidence=args.conf)


if __name__ == "__main__":
    main()


