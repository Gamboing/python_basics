from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL no definido. Configura variable de entorno o st.secrets['DATABASE_URL'].")
    return create_engine(url, pool_pre_ping=True)


def fetch_df(engine: Engine, query: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


# Overview
def get_overview_kpis(engine: Engine) -> pd.DataFrame:
    query = """
    SELECT
      COALESCE(SUM(revenue), 0) AS revenue_total,
      COALESCE(SUM(units_sold), 0) AS units_total,
      (SELECT COUNT(*) FROM products) AS products_total,
      (SELECT COALESCE(SUM(stock),0) FROM products) AS stock_total,
      (SELECT AVG(margin_pct) FROM products WHERE margin_pct IS NOT NULL) AS margin_avg
    FROM sales_daily;
    """
    return fetch_df(engine, query)


def get_monthly_series(engine: Engine) -> pd.DataFrame:
    query = """
    SELECT date_trunc('month', date)::date AS month,
           SUM(units_sold) AS units,
           SUM(revenue) AS revenue
    FROM sales_daily
    GROUP BY 1
    ORDER BY 1;
    """
    return fetch_df(engine, query)


def get_top_entities(engine: Engine, group_col: str, limit: int = 10) -> pd.DataFrame:
    query = f"""
    SELECT {group_col} AS label,
           SUM(units_sold) AS units,
           SUM(revenue) AS revenue
    FROM sales_daily sd
    JOIN products p ON p.product_id = sd.product_id
    GROUP BY {group_col}
    ORDER BY units DESC NULLS LAST
    LIMIT :limit;
    """
    return fetch_df(engine, query, {"limit": limit})


# Materials / Categories
def get_rank_by_material(engine: Engine) -> pd.DataFrame:
    return get_top_entities(engine, "p.material", limit=50)


def get_rank_by_category(engine: Engine) -> pd.DataFrame:
    return get_top_entities(engine, "p.category", limit=50)


def get_months_best(engine: Engine, group: Optional[str] = None) -> pd.DataFrame:
    group_clause = f", {group}" if group else ""
    select_group = f"{group}," if group else ""
    query = f"""
    SELECT date_trunc('month', date)::date AS month,
           {select_group}
           SUM(units_sold) AS units,
           SUM(revenue) AS revenue
    FROM sales_daily sd
    JOIN products p ON p.product_id = sd.product_id
    GROUP BY month{group_clause}
    ORDER BY units DESC NULLS LAST
    LIMIT 12;
    """
    return fetch_df(engine, query)


# Productos
def list_products(engine: Engine) -> pd.DataFrame:
    query = "SELECT product_id, product_name FROM products ORDER BY product_id;"
    return fetch_df(engine, query)


def product_history(engine: Engine, product_id: str) -> pd.DataFrame:
    query = """
    SELECT sd.date, sd.units_sold, sd.revenue,
           p.product_name, p.unit_cost, p.unit_price
    FROM sales_daily sd
    JOIN products p ON p.product_id = sd.product_id
    WHERE sd.product_id = :pid
    ORDER BY sd.date;
    """
    return fetch_df(engine, query, {"pid": product_id})


# Forecast
def forecast_metrics(engine: Engine) -> pd.DataFrame:
    query = """
    SELECT horizon, model_name, mae, rmse, mape, run_tag, created_at
    FROM forecast_metrics
    ORDER BY created_at DESC;
    """
    return fetch_df(engine, query)


def forecast_predictions(engine: Engine, product_id: Optional[str] = None, horizons: Optional[List[int]] = None,
                         model_name: Optional[str] = None, run_tag: Optional[str] = None) -> pd.DataFrame:
    clauses = []
    params = {}
    if product_id:
        clauses.append("product_id = :pid")
        params["pid"] = product_id
    if horizons:
        clauses.append("horizon = ANY(:horizons)")
        params["horizons"] = horizons
    if model_name:
        clauses.append("model_name = :model")
        params["model"] = model_name
    if run_tag:
        clauses.append("run_tag = :tag")
        params["tag"] = run_tag
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    query = f"""
    SELECT product_id, horizon, target_date, yhat, model_name, run_tag, created_at
    FROM forecast_predictions
    {where}
    ORDER BY target_date;
    """
    return fetch_df(engine, query, params)


# Video analytics
def list_runs(engine: Engine) -> pd.DataFrame:
    query = "SELECT run_id, started_at, video_path FROM runs ORDER BY started_at DESC;"
    return fetch_df(engine, query)


def tracks_stats(engine: Engine, run_id: Optional[str] = None) -> pd.DataFrame:
    where = "WHERE run_id = :rid" if run_id else ""
    params = {"rid": run_id} if run_id else None
    query = f"""
    SELECT class_name, COUNT(*) AS detections, MIN(timestamp_sec) AS t_min, MAX(timestamp_sec) AS t_max
    FROM tracks
    {where}
    GROUP BY class_name
    ORDER BY detections DESC;
    """
    return fetch_df(engine, query, params)


def tracks_over_time(engine: Engine, run_id: Optional[str] = None) -> pd.DataFrame:
    where = "WHERE run_id = :rid" if run_id else ""
    params = {"rid": run_id} if run_id else None
    query = f"""
    SELECT date_trunc('minute', to_timestamp(timestamp_sec)) AS minute,
           class_name,
           COUNT(*) AS detections
    FROM tracks
    {where}
    GROUP BY minute, class_name
    ORDER BY minute;
    """
    return fetch_df(engine, query, params)
