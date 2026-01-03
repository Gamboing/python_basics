from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.features import build_feature_frame
from src.forecasting.models import baseline_ma7, baseline_naive_lag1, train_xgb


@dataclass
class ForecastResult:
    target_date: pd.Timestamp
    product_id: str
    horizon: int
    y_true: float
    preds: Dict[str, float]


def make_supervised(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["target_date"] = df["date"] + pd.to_timedelta(horizon, unit="D")
    target = df[["product_id", "target_date", "units_sold"]].rename(columns={"units_sold": "y_target"})
    features = df.drop(columns=["target_date"], errors="ignore")
    merged = pd.merge(
        features,
        target,
        how="left",
        left_on=["product_id", "date"],
        right_on=["product_id", "target_date"],
        suffixes=("", "_y"),
    )
    merged = merged[~merged["y_target"].isna()]
    return merged


def walk_forward(df: pd.DataFrame, horizon: int, min_train: int = 60) -> Tuple[List[ForecastResult], Dict[str, float]]:
    results: List[ForecastResult] = []
    df_h = make_supervised(df, horizon).sort_values("date")
    feature_cols = [c for c in df_h.columns if c not in {"y_target", "target_date"} and not c.startswith("units_sold")]
    dates = sorted(df_h["date"].unique())
    metrics_accum: Dict[str, List[float]] = {"naive": [], "ma7": [], "xgb": []}
    for cutoff in dates:
        train = df_h[df_h["date"] < cutoff]
        test = df_h[df_h["date"] == cutoff]
        if len(train) < min_train or test.empty:
            continue
        X_train = train[feature_cols]
        y_train = train["y_target"]
        X_test = test[feature_cols]
        y_test = test["y_target"]

        preds = {
            "naive": baseline_naive_lag1(test),
            "ma7": baseline_ma7(test),
        }
        model = train_xgb(X_train, y_train)
        preds["xgb"] = model.predict(X_test)

        for i, (_, row) in enumerate(test.iterrows()):
            res = ForecastResult(
                target_date=row["target_date"],
                product_id=row["product_id"],
                horizon=horizon,
                y_true=y_test.iloc[i],
                preds={k: float(v[i]) for k, v in preds.items()},
            )
            results.append(res)

        for k in preds:
            metrics_accum[k].append(
                {
                    "mae": mean_absolute_error(y_test, preds[k]),
                    "rmse": mean_squared_error(y_test, preds[k], squared=False),
                    "mape": float(np.mean(np.abs((y_test - preds[k]) / (y_test + 1e-6)))),
                }
            )

    metrics: Dict[str, float] = {}
    for model_name, vals in metrics_accum.items():
        if not vals:
            continue
        mae = np.mean([v["mae"] for v in vals])
        rmse = np.mean([v["rmse"] for v in vals])
        mape = np.mean([v["mape"] for v in vals])
        metrics[model_name] = {"mae": mae, "rmse": rmse, "mape": mape}
    return results, metrics


def build_features_for_forecast(df_sales: pd.DataFrame) -> pd.DataFrame:
    df_feat = build_feature_frame(df_sales)
    df_feat = df_feat.dropna(subset=["lag_1"])
    return df_feat
