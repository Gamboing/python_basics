from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import Column, Date, Integer, MetaData, Numeric, String, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from tools.db import get_engine


COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["fecha", "date", "fecha venta", "fecha_venta"],
    "product_id": ["id producto", "product_id", "id_producto"],
    "units_sold": ["unidades vendidas", "cantidad", "units_sold", "unidades", "venta unidades", "ventas"],
    "revenue": ["ingreso", "total", "venta", "revenue", "monto", "importe"],
}

TEMPLATE_COLUMNS = ["Fecha", "ID Producto", "Unidades Vendidas", "Ingreso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de ventas diarias desde Excel a Postgres.")
    parser.add_argument("--db-url", type=str, default=None, help="DATABASE_URL (sino usa env).")
    parser.add_argument("--excel", type=Path, required=True, help="Ruta al Excel de ventas.")
    parser.add_argument("--sheet", type=str, default="Ventas", help="Nombre de hoja (por defecto 'Ventas').")
    parser.add_argument("--template-out", type=Path, default=Path("outputs/plantilla_ventas.xlsx"), help="Ruta para guardar plantilla si falta hoja/archivo.")
    parser.add_argument("--errors-out", type=Path, default=Path("outputs/etl_errors_sales.csv"), help="Ruta para errores.")
    return parser.parse_args()


def normalize_col(name: str) -> str:
    return name.strip().lower()


def map_columns(df: pd.DataFrame) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for col in df.columns:
        norm = normalize_col(col)
        for target, aliases in COLUMN_ALIASES.items():
            if norm == target or norm in aliases:
                found[target] = col
    return found


def generate_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        [
            {"Fecha": "2024-01-01", "ID Producto": "SKU001", "Unidades Vendidas": 10, "Ingreso": 1500.00},
            {"Fecha": "2024-01-02", "ID Producto": "SKU001", "Unidades Vendidas": 8, "Ingreso": 1200.00},
        ],
        columns=TEMPLATE_COLUMNS,
    )
    sample.to_excel(path, index=False)
    print(f"No se encontró archivo/hoja de ventas. Se generó plantilla en {path}")


def to_int(val) -> Optional[int]:
    if pd.isna(val):
        return None
    try:
        return int(val)
    except Exception:
        return None


def to_float(val) -> Optional[float]:
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def prepare_rows(df: pd.DataFrame, cols: Dict[str, str], errors: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        record: Dict[str, object] = {}
        # date
        date_val = pd.to_datetime(row.get(cols["date"]), errors="coerce")
        if pd.isna(date_val):
            errors.append({"row": idx, "field": "date", "value": row.get(cols["date"]), "error": "Fecha inválida"})
            continue
        record["date"] = date_val.date()
        # product_id
        pid_raw = row.get(cols["product_id"])
        pid = str(pid_raw).strip() if pd.notna(pid_raw) else ""
        if not pid:
            errors.append({"row": idx, "field": "product_id", "value": pid_raw, "error": "product_id vacío"})
            continue
        record["product_id"] = pid
        # units_sold
        units = to_int(row.get(cols["units_sold"]))
        if units is None:
            errors.append({"row": idx, "field": "units_sold", "value": row.get(cols["units_sold"]), "error": "units_sold inválido"})
            continue
        record["units_sold"] = units
        # revenue optional
        revenue_val = None
        if "revenue" in cols:
            revenue_val = to_float(row.get(cols["revenue"]))
            if row.get(cols["revenue"]) is not None and pd.notna(row.get(cols["revenue"])) and revenue_val is None:
                errors.append({"row": idx, "field": "revenue", "value": row.get(cols["revenue"]), "error": "revenue inválido"})
        record["revenue"] = revenue_val
        rows.append(record)
    return rows


def upsert_sales(engine: Engine, rows: List[Dict[str, object]]) -> None:
    metadata = MetaData()
    sales_daily = Table(
        "sales_daily",
        metadata,
        Column("date", Date, primary_key=True),
        Column("product_id", String, primary_key=True),
        Column("units_sold", Integer, nullable=False),
        Column("revenue", Numeric(14, 2)),
        extend_existing=True,
    )
    stmt = insert(sales_daily).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "product_id"],
        set_={"units_sold": stmt.excluded.units_sold, "revenue": stmt.excluded.revenue},
    )
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
    if not args.excel.exists():
        generate_template(args.template_out)
        return

    try:
        df = pd.read_excel(args.excel, sheet_name=args.sheet)
    except ValueError:
        generate_template(args.template_out)
        return

    col_map = map_columns(df)
    required = {"date", "product_id", "units_sold"}
    if not required.issubset(col_map.keys()):
        print("Faltan columnas requeridas en la hoja de ventas. Se genera plantilla.")
        generate_template(args.template_out)
        return

    errors: List[Dict[str, object]] = []
    rows = prepare_rows(df, col_map, errors)
    write_errors(errors, args.errors_out)
    if not rows:
        print(f"No se insertaron filas. Revisa {args.errors_out}")
        return
    engine = get_engine(args.db_url)
    upsert_sales(engine, rows)
    print(f"Ingesta de ventas completada. Filas insertadas/actualizadas: {len(rows)}. Errores: {len(errors)} (ver {args.errors_out}).")


if __name__ == "__main__":
    main()
