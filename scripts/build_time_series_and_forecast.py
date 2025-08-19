import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


def extract_tracks(coco_tracked_path: str) -> List[List[Tuple[int, float, float]]]:
    with open(coco_tracked_path, "r") as f:
        coco = json.load(f)

    image_map = {img["id"]: img.get("timestep", img["id"]) for img in coco["images"]}
    per_track: Dict[int, List[Tuple[int, float, float]]] = {}
    for ann in coco["annotations"]:
        tid = ann.get("track_id")
        if tid is None:
            continue
        t = image_map[ann["image_id"]]

        # Prefer ellipse-completed diameters if provided; else fall back to bbox w/h
        x = float(ann.get("x", 0.0))
        y = float(ann.get("y", 0.0))
        if x == 0.0 or y == 0.0:
            # derive from bbox if not present (approximate major/minor from bbox)
            bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            x = float(bw)
            y = float(bh)

        per_track.setdefault(tid, []).append((t, x, y))

    sequences: List[List[Tuple[int, float, float]]] = []
    for tid, vals in per_track.items():
        vals_sorted = sorted(vals, key=lambda z: z[0])
        sequences.append(vals_sorted)
    return sequences


def causal_moving_average(seq: np.ndarray, window: int) -> np.ndarray:
    L = seq.shape[0]
    out = np.zeros_like(seq)
    for i in range(L):
        s = max(0, i - window + 1)
        out[i] = np.mean(seq[s : i + 1], axis=0)
    return out


def build_windows(seqs: List[np.ndarray], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for seq in seqs:
        L = seq.shape[0]
        for i in range(L - n_steps):
            X.append(seq[i : i + n_steps].reshape(-1))
            y.append(seq[i + n_steps])
    return np.vstack(X), np.vstack(y)


def train_and_rollout(
    sequences: List[List[Tuple[int, float, float]]],
    n_steps: int = 1,
    window: int = 10,
    test_split: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    # Convert to arrays and discard short sequences (< n_steps + 1)
    seq_arrays = []
    for seq in sequences:
        arr = np.array([[x, y] for _, x, y in seq], dtype=float)
        if arr.shape[0] >= n_steps + 1:
            seq_arrays.append(arr)

    # Split sequences
    rng = np.random.default_rng(random_seed)
    idx = np.arange(len(seq_arrays))
    rng.shuffle(idx)
    split = int((1 - test_split) * len(seq_arrays))
    train_idx, test_idx = idx[:split], idx[split:]
    train_seqs = [seq_arrays[i] for i in train_idx]
    test_seqs = [seq_arrays[i] for i in test_idx]

    # Smooth training sequences only
    smoothed_train = [causal_moving_average(seq, window) for seq in train_seqs]

    X_train, y_train = build_windows(smoothed_train, n_steps)
    X_test, y_test = build_windows(test_seqs, n_steps)

    model = ExtraTreesRegressor(random_state=random_seed)
    model.fit(X_train, y_train)

    # Rollout on test sequences
    results = []
    for seq in test_seqs:
        T = seq.shape[0]
        window_vec = seq[:n_steps].reshape(1, -1)
        pred_traj = [seq[j] for j in range(n_steps)]
        for t in range(n_steps, T):
            nxt = model.predict(window_vec).squeeze(0)
            pred_traj.append(nxt)
            window_vec = np.roll(window_vec, -2, axis=1)
            window_vec[0, -2:] = nxt
        pred_arr = np.vstack(pred_traj)
        dists = np.linalg.norm(pred_arr - seq, axis=1)
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        results.append({"rmse": rmse})

    mean_rmse = float(np.mean([r["rmse"] for r in results])) if results else float("nan")
    return {
        "model": model,
        "mean_rmse": mean_rmse,
        "n_test": len(results),
    }


def main():
    ap = argparse.ArgumentParser(description="Build time series from tracked COCO and forecast growth")
    ap.add_argument("--tracked", required=True, help="Tracked COCO JSON path")
    ap.add_argument("--n_steps", type=int, default=1)
    ap.add_argument("--window", type=int, default=10, help="Causal moving average window for training")
    ap.add_argument("--test_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sequences = extract_tracks(args.tracked)
    summary = train_and_rollout(
        sequences,
        n_steps=args.n_steps,
        window=args.window,
        test_split=args.test_split,
        random_seed=args.seed,
    )
    print(f"Mean RMSE on test set: {summary['mean_rmse']:.3f} over {summary['n_test']} sequences")


if __name__ == "__main__":
    main()


