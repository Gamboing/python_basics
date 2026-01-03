from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("product_id")["units_sold"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 28)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"rollmean_{w}"] = (
            df.groupby("product_id")["units_sold"].shift(1).rolling(window=w, min_periods=1).mean()
        )
    return df


def build_feature_frame(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df
