from __future__ import annotations

import streamlit as st

from dashboard import queries


def render(engine) -> None:
    st.header("Overview")
    kpis = queries.get_overview_kpis(engine)
    if not kpis.empty:
        k = kpis.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Ventas totales", f"{k['revenue_total'] or 0:,.2f}")
        cols[1].metric("Unidades", f"{k['units_total'] or 0:,.0f}")
        cols[2].metric("# Productos", f"{k['products_total'] or 0}")
        cols[3].metric("Stock total", f"{k['stock_total'] or 0}")
        cols[4].metric("Margen promedio", f"{(k['margin_avg'] or 0):.2f}%")

    st.subheader("Serie mensual")
    df_month = queries.get_monthly_series(engine)
    if df_month.empty:
        st.info("No hay datos en sales_daily.")
    else:
        df_plot = df_month.rename(columns={"month": "Fecha"})
        st.line_chart(df_plot, x="Fecha", y=["units", "revenue"] if "revenue" in df_plot else ["units"])
        st.download_button("Descargar serie (CSV)", df_month.to_csv(index=False), "serie_mensual.csv")

    st.subheader("Top 10 productos")
    top_products = queries.get_top_entities(engine, "p.product_id", limit=10)
    st.dataframe(top_products)
    st.download_button("CSV productos", top_products.to_csv(index=False), "top_productos.csv")

    st.subheader("Top 10 categorías")
    top_cats = queries.get_top_entities(engine, "p.category", limit=10)
    st.dataframe(top_cats)
    st.download_button("CSV categorías", top_cats.to_csv(index=False), "top_categorias.csv")

    st.subheader("Top 10 materiales")
    top_mat = queries.get_top_entities(engine, "p.material", limit=10)
    st.dataframe(top_mat)
    st.download_button("CSV materiales", top_mat.to_csv(index=False), "top_materiales.csv")
