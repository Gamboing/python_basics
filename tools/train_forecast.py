from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import Table, Column, Date, Integer, MetaData, Numeric, String, DateTime
from sqlalchemy.dialects.postgresql import insert

from tools.db import get_engine
from src.forecasting.eval import build_features_for_forecast, walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y evalúa modelos de pronóstico.")
    parser.add_argument("--db-url", type=str, default=None, help="DATABASE_URL")
    parser.add_argument("--run-tag", type=str, required=True, help="Etiqueta del experimento.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[7, 30, 90], help="Horizontes de días.")
    return parser.parse_args()


def fetch_sales(engine):
    query = "SELECT date, product_id, units_sold, revenue FROM sales_daily ORDER BY date"
    return pd.read_sql(query, engine, parse_dates=["date"])


def insert_metrics(engine, run_tag: str, horizon: int, metrics: Dict[str, Dict[str, float]]) -> None:
    metadata = MetaData()
    table = Table(
        "forecast_metrics",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("created_at", DateTime(timezone=True)),
        Column("horizon", Integer),
        Column("model_name", String),
        Column("mae", Numeric),
        Column("rmse", Numeric),
        Column("mape", Numeric),
        Column("run_tag", String),
        extend_existing=True,
    )
    rows = []
    now = datetime.utcnow()
    for model_name, vals in metrics.items():
        rows.append(
            {
                "created_at": now,
                "horizon": horizon,
                "model_name": model_name,
                "mae": vals["mae"],
                "rmse": vals["rmse"],
                "mape": vals["mape"],
                "run_tag": run_tag,
            }
        )
    if not rows:
        return
    stmt = insert(table).values(rows)
    with engine.begin() as conn:
        conn.execute(stmt)


def main() -> None:
    args = parse_args()
    engine = get_engine(args.db_url)
    sales = fetch_sales(engine)
    if sales.empty:
        print("No hay datos en sales_daily.")
        return
    feat = build_features_for_forecast(sales)
    for horizon in args.horizons:
        results, metrics = walk_forward(feat, horizon)
        insert_metrics(engine, args.run_tag, horizon, metrics)
        print(f"Horizon {horizon}: métricas={metrics}")


if __name__ == "__main__":
    main()
