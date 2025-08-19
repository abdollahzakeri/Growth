### Method steps mapped to scripts

1) Undistortion + cropping (Section 3.2)
   - `undistort_and_crop.py` with k1, k2, FOV, fixed crop.

2) Segmentation to COCO
   - `export_coco_from_yolo.py` using a YOLO model (e.g., `best_v3.pt`). Outputs `coco_annotations.json`.

3) Mask tracking across frames
   - `track_with_iou.py` adds `track_id` per annotation with IoU matching, centroid prior, dormancy, and area-drop termination. Outputs `coco_tracked.json`.

4) Ellipse-based completion of occluded caps
   - `compute_ellipse_diameters.py` computes major/minor diameters (`x`, `y`) from mask contours using OpenCV’s `fitEllipse` (fallback to bbox when insufficient points).

5) Growth forecasting
   - `build_time_series_and_forecast.py`: builds ⟨x, y⟩ sequences per track, applies causal moving average on training, trains Extra-Trees forecaster, and evaluates mean RMSE.

6) Species-specific weight estimation
   - `fit_weight_regressors.py` trains/evaluates regressors on `data/white.csv` and `data/brown.csv`.

7) Yield composition
   - `yield_forecast.py` demonstrates wiring predicted diameters through a fitted weight regressor to obtain tray-level yield curves.


