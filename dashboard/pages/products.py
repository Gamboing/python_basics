from __future__ import annotations

import streamlit as st

from dashboard import queries


def render(engine) -> None:
    st.header("Productos")
    df_products = queries.list_products(engine)
    if df_products.empty:
        st.info("No hay productos en la base.")
        return
    pid = st.selectbox("Producto", df_products["product_id"])
    hist = queries.product_history(engine, pid)
    if hist.empty:
        st.info("Sin ventas para este producto.")
        return
    st.line_chart(hist, x="date", y=["units_sold", "revenue"] if "revenue" in hist else ["units_sold"])
    st.download_button("CSV histórico", hist.to_csv(index=False), "producto_historico.csv")

    st.subheader("Margen estimado")
    if "unit_cost" in hist and "unit_price" in hist:
        latest = hist.dropna(subset=["unit_cost", "unit_price"]).tail(1)
        if not latest.empty:
            unit_cost = latest.iloc[0]["unit_cost"]
            unit_price = latest.iloc[0]["unit_price"]
            margen = (unit_price - unit_cost) if unit_cost is not None and unit_price is not None else None
            st.metric("Margen unitario", f"{margen:.2f}" if margen is not None else "N/A")
        else:
            st.info("No hay unit_cost / unit_price para calcular margen.")
    else:
        st.info("No se encontraron columnas de costos/precios en productos.")
