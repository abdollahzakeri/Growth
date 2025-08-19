import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor


def load_xyw(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[["x", "y"]].values.astype(float)
    y = df["w"].values.astype(float)
    return X, y


def evaluate_models(X: np.ndarray, y: np.ndarray, name: str) -> None:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "KNeighborsRegressor": KNeighborsRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42),
        "GammaRegressor": GammaRegressor(max_iter=10000),
        "PoissonRegressor": PoissonRegressor(max_iter=10000),
    }
    print(f"\n{name} models (RMSE, ME):")
    for mname, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        me = float(np.mean(yte - pred))
        print(f"- {mname}: RMSE={rmse:.3f}, ME={me:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Fit species-specific weight regressors from size–weight CSVs")
    ap.add_argument("--white", required=True, help="Path to white.csv")
    ap.add_argument("--brown", required=True, help="Path to brown.csv")
    args = ap.parse_args()

    Xw, yw = load_xyw(args.white)
    Xb, yb = load_xyw(args.brown)

    evaluate_models(Xw, yw, name="White")
    evaluate_models(Xb, yb, name="Brown")


if __name__ == "__main__":
    main()


