## Edge-Deployable Growth and Yield Forecasting for Commercial Mushroom Farms

<video src="animation.mp4" autoplay loop muted playsinline width="640">
  Your browser does not support the video tag.
</video>


This repository provides an organized, reproducible implementation of the methodology described in the paper “Edge-Deployable Growth and Yield Forecasting for Commercial Mushroom Farms” (ssrn-5371080.pdf). It assembles code from the original notebooks into focused scripts that follow the paper’s pipeline: undistortion and cropping, YOLO segmentation to COCO JSON, mask tracking via IoU, causal smoothing + Extra-Trees growth forecasting, and species-specific size–weight regression.

### Repository layout

- `scripts/`
  - `undistort_and_crop.py`: Barrel-distortion removal and fixed cropping for the time-lapse images (k1, k2, FOV, crop window).
  - `export_coco_from_yolo.py`: Runs a YOLO segmentation model on undistorted images and exports detections/masks into a COCO-style JSON with `timestep` preserved.
  - `track_with_iou.py`: Associates masks across frames using IoU and a small motion prior, adds `track_id` to annotations; track life-cycle and area-drop logic align with the paper.
  - `build_time_series_and_forecast.py`: Builds per-cap time-series ⟨x, y⟩, applies causal moving average on training windows, trains an Extra-Trees forecaster, and reports mean RMSE as in Section 3.2.
  - `fit_weight_regressors.py`: Trains species-specific size→weight regressors on `white.csv` and `brown.csv`, reporting RMSE and mean error (ME) in line with Section 3.1 and Tables 1–2.
  - `yield_forecast.py`: Wiring for composing growth predictions with a species-specific weight model to generate tray-level yield forecasts (provide a serialized model to operationalize).
- `data/`
  - `white.csv`, `brown.csv`: Species-specific paired size–weight datasets (columns: `x`, `y` in mm; `w` in grams) used for the weight regressors.


### Installation and environment

Create a fresh conda env:

```bash
conda create -n mushroom-forecast python=3.10 -y
conda activate mushroom-forecast
pip install ultralytics shapely scikit-learn pillow pandas numpy opencv-python scipy
```



Note: `ultralytics` expects a compatible PyTorch; install CUDA/CPU variants as appropriate for your machine per Ultralytics documentation.

### Reproducing the methodology

1) Undistort and crop time-lapse images using default parameters (images will be made available upon request)

```bash
python scripts/undistort_and_crop.py --input /path/to/images --output undistorted_cropped \
  --k1 -0.060 --k2 0.0 --fov 110 --crop_x 170 --crop_y 165 --crop_w 1580 --crop_h 697
```

2) Segment frames and export COCO JSON -- model .pt file will be made available upon request

```bash
python scripts/export_coco_from_yolo.py --images undistorted_cropped --model best_v3.pt --out coco_annotations.json --conf 0.1
```

3) Track masks across frames (IoU association, motion prior, area-drop termination)

```bash
python scripts/track_with_iou.py --coco coco_annotations.json --out coco_tracked.json --min_iou 0.9 --max_dist 50 --max_lost 30 --area_drop 0.99
```

4) Build time-series and train the growth forecaster (causal smoothing on training only)

```bash
python scripts/build_time_series_and_forecast.py --tracked coco_tracked.json --n_steps 1 --window 10 --test_split 0.2
```

5) Species-specific weight regressors (using the provided datasets)

```bash
python scripts/fit_weight_regressors.py --white data/white.csv --brown data/brown.csv
```

6) Compose growth predictions with weight regressors to forecast yield

Train or load a serialized weight regressor (e.g., scikit-learn via joblib) and provide it to:

```bash
python scripts/yield_forecast.py --tracked coco_tracked.json --species brown --model path/to/weight_model.joblib
```


### Methodology notes (paper alignment)

- Distortion removal and cropping follow Section 3.2: Brown–Conrady coefficients (k1, k2), FOV-based intrinsics, and a fixed crop window matching the prepared dataset.
- Tracking uses mask IoU with a high threshold (default 0.90), short-range centroid prior, `max_lost` dormancy, and termination on large area drop (0.99 of running max), consistent with the time-lapse interval and near-stationary assumptions.
- Growth forecasting uses a sliding-window representation of past ⟨x, y⟩ (default `n_steps=1`), with a causal moving average on training sequences (`window=10` ~2.5 hours) to reduce segmentation jitter.
- Weight regression is trained per species on the provided paired datasets, reporting RMSE and ME similar to the paper’s tables and discussion, using off-the-shelf regressors.

### How to cite

Please cite the paper as follows (BibTeX):

```bibtex
@misc{zakeri2025edge,
  title        = {Edge-Deployable Growth and Yield Forecasting for Commercial Mushroom Farms},
  author       = {Zakeri, Abdollah and Kang, Jiming and Koirala, Bikram and Silwal, Raman and Balan, Venkatesh and Zhu, Weihang and Benhaddou, Driss and Merchant, Fatima A.},
  year         = {2025},
  howpublished = {SSRN preprint},
  note         = {SSRN: 5371080}
}
```

### License

Data and code are provided for academic and non-commercial use. See the paper and this repository for details.


