from __future__ import annotations

import streamlit as st

from dashboard import queries


def render(engine) -> None:
    st.header("Predicción")
    df_metrics = queries.forecast_metrics(engine)
    df_preds_all = queries.forecast_predictions(engine)
    if df_preds_all.empty:
        st.info("No hay pronósticos almacenados.")
    else:
        products = sorted(df_preds_all["product_id"].unique())
        pid = st.selectbox("Producto", products)
        horizons = sorted(df_preds_all["horizon"].unique())
        horizon = st.selectbox("Horizonte", horizons, index=0)
        models = sorted(df_preds_all["model_name"].unique())
        model = st.selectbox("Modelo", models, index=0)
        run_tags = sorted(df_preds_all["run_tag"].dropna().unique())
        run_tag = st.selectbox("Run tag", run_tags) if run_tags else None

        df_filtered = queries.forecast_predictions(
            engine,
            product_id=pid,
            horizons=[horizon],
            model_name=model,
            run_tag=run_tag,
        )
        st.subheader("Histórico vs predicción")
        if df_filtered.empty:
            st.info("Sin datos para estos filtros.")
        else:
            st.line_chart(df_filtered, x="target_date", y="yhat")
            st.download_button("CSV predicciones", df_filtered.to_csv(index=False), "predicciones.csv")

    st.subheader("Métricas por modelo/horizonte")
    if df_metrics.empty:
        st.info("No hay métricas.")
    else:
        st.dataframe(df_metrics)
        st.download_button("CSV métricas", df_metrics.to_csv(index=False), "forecast_metrics.csv")
