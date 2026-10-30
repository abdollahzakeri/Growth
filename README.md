## Edge-Deployable Growth and Yield Forecasting for Commercial Mushroom Farms

![](anim.gif)

Code, data, and trained models for the paper *Edge-Deployable Growth and Yield Forecasting for Commercial Mushroom Farms* (Smart Agricultural Technology, manuscript ATECH-D-26-03398). The scripts reproduce every table and figure of the revised manuscript.

### Contents

| Path | Content |
|---|---|
| `images/` | The 2,121 raw time-lapse frames (1920×1080 px, one every 15 min, 1 to 23 May 2025, one baby bella bed). Distortion correction and cropping are applied by `scripts/01_undistort_crop.py`. |
| `data/white.csv`, `data/brown.csv` | Caliper diameters `x`, `y` (mm) and fresh weight `w` (g) of 203 white button and 203 baby bella caps. |
| `release/tracks_t090.csv.gz` | All tracked detections with persistent track identities (139,687 detections in 3,808 tracks): frame, centroid, mask area, convex-hull ellipse diameters (px), confidence. |
| `release/growth_trajectories.json` | The 1,617 cleaned growth trajectories (major/minor diameters in px, start frame, group id) used for forecasting. |
| `release/growth_splits.json` | The 30 group-level training/validation/test splits (seeds 0 to 29). |
| `release/growth_linear_ar_n1_l5.joblib` | Trained linear autoregressive growth model (n_steps = 1, l = 5). |
| `release/weight_*.joblib`, `release/weight_split_*.json` | Trained size-to-weight regressors and the training/test indices of the weight data. |
| `growth/` | Shared code: settings, convex-hull Fitzgibbon ellipse fit, causal moving average, weight models. |
| `scripts/` | Numbered scripts for every step of the pipeline and every analysis in the paper. |

The trained Extra-Trees growth model (n_steps = 1, l = 10) is 1.9 GB and therefore not stored in the repository; `scripts/29_export_release.py` regenerates it deterministically (`random_state=0`) from the released trajectories. The YOLOv8-X segmentation weights (`best_v3.pt`, 144 MB) are available from the authors on request; place them in `models/`.

### Settings used in the paper

| Step | Setting |
|---|---|
| Distortion correction | Brown–Conrady, k1 = −0.078, k2 = 0, pinhole matrix from the 110° nominal field of view; output camera matrix fixed at the k1 = −0.06 scale |
| Crop | 1580 × 697 px at offset (170, 165) |
| Scale | ρ = Z / f = 419 mm / 879.2 px = 0.4766 mm/px (casing plane) |
| Segmentation | YOLOv8-X instance segmentation (`best_v3.pt`), input size 1280 px, confidence threshold 0.1 |
| Tracking | Mask IoU ≥ 0.90 with Hungarian assignment; gap tolerance 10 frames |
| Trajectories | Convex-hull Fitzgibbon ellipse diameters; missed frames interpolated; end when the smoothed (l = 10) major diameter drops more than 10% below its running maximum; minimum 20 frames (5 h) |
| Grouping for splits | Fragments linked when the first mask of one overlaps the last mask of an earlier one with IoU ≥ 0.3 (no time limit) |
| Forecasting | 60/20/20 group-level splits, 30 seeds; n_steps ∈ {1, 2, 4, 8}, l ∈ {1, 5, 10, 20}, selection on validation; recursive rollout, seed frames excluded from the error |
| Weight models | `train_test_split(test_size=0.25, random_state=42)` (152/51 caps); model selection by 5×5 repeated cross-validation on the training caps |
| Yield analysis | 78 start images every 6 h; fragments chained into mushrooms (IoU ≥ 0.5, gap ≤ 24 h); five-fold cross-fitting over groups; 100 Monte Carlo draws per start image |

### Environment

```bash
conda env create -f environment.yml
conda activate mushroom-forecast
```

The results were produced with Python 3.10, NumPy 1.26, SciPy 1.15, scikit-learn 1.4, pandas 2.3, Shapely 2.1, OpenCV 4.9, PyTorch 2.5, Ultralytics 8.3, and LazyPredict. Paths default to the repository root and can be changed with the environment variables `GROWTH_ROOT` (data root) and `GROWTH_RESULTS` (results folder).

### Reproducing the paper

```bash
python scripts/01_undistort_crop.py -0.078 undistorted_cropped_078          # distortion correction + crop
python scripts/02_segment.py undistorted_cropped_078 results/coco_annotations.json 0.1
python scripts/03_track.py results/tracks_t090.pkl 0.90 10                     # IoU-Hungarian tracking
python scripts/04_prepare_sequences.py results/tracks_t090.pkl results/seqs_t090.pkl
for s in 0 5 10 15 20 25; do python scripts/05_forecast_splits.py results/seqs_t090.pkl $s 5 results/fc_t090; done
python scripts/06_summarize_forecasts.py "results/fc_t090_*.pkl" results/forecast_summary_t090.json
```

| Paper item | Script |
|---|---|
| Table 4, Fig. 6, Fig. 9a, Fig. 10, Table B.1 | `05_forecast_splits.py`, `06_summarize_forecasts.py`, `24_figures.py` |
| Paired contrasts (Section 3.2, Table B.2) | `07_paired_contrasts.py` |
| Smoothing on test data, input-noise trials | `08_smoothing_and_noise.py`, `09_smoothing_noise_summary.py` |
| Re-anchoring (Fig. 9b) | `10_reanchoring.py` |
| Table 5, tracking sensitivity | `11_tracking_metrics.py`, `12_tracking_sensitivity.py` |
| Grouping, leakage check, scale plausibility | `13_grouping_and_scale_checks.py`, `14_loose_grouping.py`, `16_acquisition_and_scale.py` |
| Visual identity audit | `15_identity_audit.py` |
| Fig. 11, occlusion results (Section 3.3) | `17_occlusion_neighbor.py`, `18_occlusion_straight_edge.py` |
| Tables 1 to 3, Appendix A | `19_weight_models.py`, `20_weight_checks.py` |
| Table 6, Figs. 12 and 13, sensitivity analyses | `21_yield_monte_carlo.py` (variants via `R06_CORR=1`, `R06_SCALE=0.0179`, `R06_PARALLAX=10,30`, `R06_RESID=<json>`), `25_yield_figures.py` |
| Embedded runtime (Jetson Orin NX 16 GB, MAXN, TensorRT FP16) | `22_benchmark_jetson.py`, `23_profile_segmentation.py` |
| Supplementary visualizations | `26_size_coverage.py`, `27_track_visualization.py`, `28_all_masks_gif.py` |
| Released models and splits | `29_export_release.py` |

Most analysis scripts take their inputs from `results/` (written by steps 03 to 05) and are documented in their header docstrings.
