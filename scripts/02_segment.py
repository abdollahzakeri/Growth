"""
Step 3 (GPU) — Instance segmentation with the trained YOLO model (best_v3.pt),
mirroring 02_Segmentation_and_Detection.ipynb.  Produces a COCO-style JSON with
one polygon mask per detected cap, used by the ellipse-fitting / occlusion
experiment (06).

Usage: python 03_segment.py <cropped_dir> <out_coco.json>
"""
import os, sys, json
from PIL import Image
from ultralytics import YOLO

IMAGE_DIR = sys.argv[1]
OUT_JSON = sys.argv[2]
ROOT = os.environ.get("GROWTH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_PATH = os.environ.get("GROWTH_SEG_MODEL", os.path.join(ROOT, "models", "best_v3.pt"))
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

model = YOLO(MODEL_PATH)  # trained with imgsz=1280; the checkpoint keeps this input size for prediction
categories = [{"id": cid, "name": name} for cid, name in model.names.items()]
images = sorted(f for f in os.listdir(IMAGE_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
coco_images, coco_annotations = [], []
ann_id = 0
for timestep, filename in enumerate(images):
    path = os.path.join(IMAGE_DIR, filename)
    with Image.open(path) as im:
        width, height = im.size
    coco_images.append({"id": timestep, "file_name": filename,
                        "width": width, "height": height, "timestep": timestep})
    res = model.predict(path, conf=CONF, stream=False, verbose=False)[0]
    boxes = res.boxes
    masks = getattr(res, "masks", None)
    for inst_idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [x1, y1, x2 - x1, y2 - y1]
        segmentation = []
        if masks is not None and hasattr(masks, "xy"):
            poly = masks.xy[inst_idx]
            segmentation.append(poly.flatten().tolist())
        coco_annotations.append({
            "id": ann_id, "image_id": timestep, "category_id": cls_id,
            "bbox": [float(x) for x in bbox], "area": float(bbox[2] * bbox[3]),
            "segmentation": segmentation, "confidence": conf, "iscrowd": 0})
        ann_id += 1
    if timestep % 300 == 0:
        print(f"  {timestep}/{len(images)}  anns={ann_id}", flush=True)

with open(OUT_JSON, "w") as f:
    json.dump({"images": coco_images, "annotations": coco_annotations,
               "categories": categories}, f)
print(f"Saved {OUT_JSON}  ({len(coco_annotations)} annotations)")
