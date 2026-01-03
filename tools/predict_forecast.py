from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import Table, Column, Date, Integer, MetaData, Numeric, String, DateTime
from sqlalchemy.dialects.postgresql import insert

from tools.db import get_engine
from src.forecasting.eval import build_features_for_forecast
from src.forecasting.features import build_feature_frame
from src.forecasting.models import baseline_ma7, baseline_naive_lag1, train_xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera pronósticos futuros.")
    parser.add_argument("--db-url", type=str, default=None, help="DATABASE_URL")
    parser.add_argument("--run-tag", type=str, required=True, help="Etiqueta del experimento.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90], help="Horizontes de días.")
    return parser.parse_args()


def fetch_sales(engine):
    query = "SELECT date, product_id, units_sold, revenue FROM sales_daily ORDER BY date"
    return pd.read_sql(query, engine, parse_dates=["date"])


def insert_predictions(engine, rows: List[Dict[str, object]]) -> None:
    metadata = MetaData()
    table = Table(
        "forecast_predictions",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("created_at", DateTime(timezone=True)),
        Column("product_id", String),
        Column("horizon", Integer),
        Column("target_date", Date),
        Column("yhat", Numeric),
        Column("model_name", String),
        Column("run_tag", String),
        extend_existing=True,
    )
    if not rows:
        return
    stmt = insert(table).values(rows)
    with engine.begin() as conn:
        conn.execute(stmt)


def forecast_future(feat: pd.DataFrame, horizon: int, run_tag: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    last_date = feat["date"].max()
    target_date = last_date + pd.to_timedelta(horizon, unit="D")
    feature_cols = [c for c in feat.columns if c not in {"units_sold", "date", "revenue"}]
    for pid, group in feat.groupby("product_id"):
        group = group.sort_values("date")
        last_row = group.iloc[-1:]
        X_train = group[feature_cols]
        y_train = group["units_sold"]
        preds_naive = baseline_naive_lag1(last_row)
        preds_ma7 = baseline_ma7(last_row)
        model = train_xgb(X_train, y_train)
        preds_xgb = model.predict(last_row[feature_cols])
        now = datetime.utcnow()
        rows.extend(
            [
                {
                    "created_at": now,
                    "product_id": pid,
                    "horizon": horizon,
                    "target_date": target_date.date(),
                    "yhat": float(preds_naive[0]),
                    "model_name": "naive",
                    "run_tag": run_tag,
                },
                {
                    "created_at": now,
                    "product_id": pid,
                    "horizon": horizon,
                    "target_date": target_date.date(),
                    "yhat": float(preds_ma7[0]),
                    "model_name": "ma7",
                    "run_tag": run_tag,
                },
                {
                    "created_at": now,
                    "product_id": pid,
                    "horizon": horizon,
                    "target_date": target_date.date(),
                    "yhat": float(preds_xgb[0]),
                    "model_name": "xgb",
                    "run_tag": run_tag,
                },
            ]
        )
    return rows


def main() -> None:
    args = parse_args()
    engine = get_engine(args.db_url)
    sales = fetch_sales(engine)
    if sales.empty:
        print("No hay datos en sales_daily.")
        return
    feat = build_features_for_forecast(sales)
    all_rows: List[Dict[str, object]] = []
    for horizon in args.horizons:
        rows = forecast_future(feat, horizon, args.run_tag)
        all_rows.extend(rows)
        print(f"Pronósticos generados: horizon={horizon}, filas={len(rows)}")
    insert_predictions(engine, all_rows)


if __name__ == "__main__":
    main()
