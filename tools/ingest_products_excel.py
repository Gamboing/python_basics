from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import MetaData, Table, Column, String, Text, Numeric, Integer, DateTime
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql import func

from tools.db import get_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de productos desde Excel a Postgres.")
    parser.add_argument("--db-url", type=str, default=None, help="DATABASE_URL para Postgres (sino usa env).")
    parser.add_argument("--excel", type=Path, required=True, help="Ruta al Excel Muebles.xlsx.")
    parser.add_argument("--errors-out", type=Path, default=Path("outputs/etl_errors_products.csv"), help="Ruta para escribir errores.")
    return parser.parse_args()


def load_allowed_lists(df_lists: pd.DataFrame) -> Dict[str, set]:
    allowed = {"category": set(), "material": set(), "color": set()}
    for col in df_lists.columns:
        key = col.strip().lower()
        if "categor" in key:
            allowed["category"].update(_normalize_series(df_lists[col]))
        elif "material" in key:
            allowed["material"].update(_normalize_series(df_lists[col]))
        elif "color" in key:
            allowed["color"].update(_normalize_series(df_lists[col]))
    return allowed


def _normalize_series(series: pd.Series) -> List[str]:
    vals: List[str] = []
    for x in series.dropna().tolist():
        normed = _norm_label(x)
        if normed:
            vals.append(normed)
    return vals


def _norm_string(value: str) -> str:
    return str(value).strip()


def _norm_label(value) -> str:
    return str(value).strip().title()


def _to_numeric(val) -> Optional[float]:
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def _to_int(val) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        return int(val)
    except Exception:
        return None


def prepare_products(df: pd.DataFrame, allowed: Dict[str, set], errors: List[Dict[str, str]]) -> List[Dict[str, object]]:
    products: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        record: Dict[str, object] = {}
        record["product_id"] = str(row.get("ID Producto", "")).strip()
        record["product_name"] = str(row.get("Nombre del Mueble", "")).strip()
        record["description"] = str(row.get("Descripción", "")).strip()
        record["category"] = _norm_label(row.get("Categoría", ""))
        record["material"] = _norm_label(row.get("Material", ""))
        record["color"] = _norm_label(row.get("Color", ""))
        record["unit_cost"] = _to_numeric(row.get("Costo de Fabricación"))
        record["unit_price"] = _to_numeric(row.get("Costo de Venta"))
        record["margin_pct"] = _to_numeric(row.get("Margen (%)"))
        record["stock"] = _to_int(row.get("Stock"))
        record["created_at"] = pd.to_datetime(row.get("Fecha de Registro"), errors="coerce")

        if not record["product_id"]:
            errors.append({"row": idx, "field": "product_id", "value": "", "error": "product_id vacío"})
            continue
        if not record["product_name"]:
            errors.append({"row": idx, "field": "product_name", "value": "", "error": "product_name vacío"})
            continue
        # Validaciones con listas
        for key in ("category", "material", "color"):
            if record[key] and allowed.get(key) and record[key] not in allowed[key]:
                errors.append(
                    {
                        "row": idx,
                        "field": key,
                        "value": record[key],
                        "error": f"Valor no encontrado en listas ({key})",
                    }
                )
        for num_field in ("unit_cost", "unit_price", "margin_pct"):
            raw_val = row.get(
                {
                    "unit_cost": "Costo de Fabricación",
                    "unit_price": "Costo de Venta",
                    "margin_pct": "Margen (%)",
                }[num_field]
            )
            if pd.notna(raw_val) and record[num_field] is None:
                errors.append({"row": idx, "field": num_field, "value": raw_val, "error": "Valor numérico inválido"})
        if row.get("Stock") is not None and pd.notna(row.get("Stock")) and record["stock"] is None:
            errors.append({"row": idx, "field": "stock", "value": row.get("Stock"), "error": "Valor entero inválido"})
        if record["created_at"] is pd.NaT:
            errors.append({"row": idx, "field": "created_at", "value": row.get("Fecha de Registro"), "error": "Fecha inválida; usando NOW()"})
            record["created_at"] = None
        products.append(record)
    return products


def upsert_products(engine: Engine, products: List[Dict[str, object]]) -> None:
    metadata = MetaData()
    products_table = Table(
        "products",
        metadata,
        Column("product_id", String, primary_key=True),
        Column("product_name", Text, nullable=False),
        Column("description", Text),
        Column("category", Text),
        Column("material", Text),
        Column("color", Text),
        Column("unit_cost", Numeric(12, 2)),
        Column("unit_price", Numeric(12, 2)),
        Column("margin_pct", Numeric(6, 2)),
        Column("stock", Integer),
        Column("created_at", DateTime(timezone=True), server_default=func.now()),
        extend_existing=True,
    )
    stmt = insert(products_table).values(products)
    update_cols = {c: getattr(stmt.excluded, c) for c in [
        "product_name",
        "description",
        "category",
        "material",
        "color",
        "unit_cost",
        "unit_price",
        "margin_pct",
        "stock",
        "created_at",
    ]}
    stmt = stmt.on_conflict_do_update(index_elements=["product_id"], set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)


def write_errors(errors: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        path.write_text("")
        return
    pd.DataFrame(errors).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    engine = get_engine(args.db_url)

    df_muebles = pd.read_excel(args.excel, sheet_name="Muebles")
    df_listas = pd.read_excel(args.excel, sheet_name="Listas")

    allowed = load_allowed_lists(df_listas)
    errors: List[Dict[str, object]] = []
    products = prepare_products(df_muebles, allowed, errors)
    if products:
        upsert_products(engine, products)
    write_errors(errors, args.errors_out)
    print(f"Ingesta completada. Filas procesadas: {len(products)}. Errores: {len(errors)} (ver {args.errors_out}).")


if __name__ == "__main__":
    main()
