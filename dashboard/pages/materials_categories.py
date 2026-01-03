from __future__ import annotations

import streamlit as st

from dashboard import queries


def render(engine) -> None:
    st.header("Materiales / Categorías")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ranking materiales")
        df_mat = queries.get_rank_by_material(engine)
        st.dataframe(df_mat)
        st.download_button("CSV materiales", df_mat.to_csv(index=False), "ranking_materiales.csv")
    with col2:
        st.subheader("Ranking categorías")
        df_cat = queries.get_rank_by_category(engine)
        st.dataframe(df_cat)
        st.download_button("CSV categorías", df_cat.to_csv(index=False), "ranking_categorias.csv")

    st.subheader("Meses con más ventas (global)")
    df_months = queries.get_months_best(engine)
    st.dataframe(df_months)
    st.download_button("CSV meses", df_months.to_csv(index=False), "meses_top_global.csv")

    st.subheader("Meses con más ventas por material")
    df_months_mat = queries.get_months_best(engine, group="p.material")
    st.dataframe(df_months_mat)
    st.download_button("CSV meses-material", df_months_mat.to_csv(index=False), "meses_top_material.csv")

    st.subheader("Meses con más ventas por categoría")
    df_months_cat = queries.get_months_best(engine, group="p.category")
    st.dataframe(df_months_cat)
    st.download_button("CSV meses-categoría", df_months_cat.to_csv(index=False), "meses_top_categoria.csv")
