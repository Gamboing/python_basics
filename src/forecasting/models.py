from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from xgboost import XGBRegressor


def baseline_naive_lag1(features) -> np.ndarray:
    return features["lag_1"].to_numpy()


def baseline_ma7(features) -> np.ndarray:
    return features["rollmean_7"].to_numpy()


def build_xgb_model(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def train_xgb(X, y) -> XGBRegressor:
    model = build_xgb_model()
    model.fit(X, y)
    return model
