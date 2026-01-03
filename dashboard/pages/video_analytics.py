from __future__ import annotations

import streamlit as st

from dashboard import queries


def render(engine) -> None:
    st.header("Video analytics (opcional)")
    runs = queries.list_runs(engine)
    run_id = None
    if not runs.empty:
        run_choices = ["(Todos)"] + runs["run_id"].tolist()
        choice = st.selectbox("Run", run_choices)
        run_id = None if choice == "(Todos)" else choice
    else:
        st.info("No hay runs registrados.")

    st.subheader("Conteo de detecciones por clase")
    stats = queries.tracks_stats(engine, run_id)
    st.dataframe(stats)
    st.download_button("CSV detecciones", stats.to_csv(index=False), "tracks_stats.csv")

    st.subheader("Detecciones en el tiempo (por minuto)")
    over_time = queries.tracks_over_time(engine, run_id)
    st.dataframe(over_time)
    if not over_time.empty:
        st.line_chart(over_time, x="minute", y="detections", color="class_name")
    st.download_button("CSV tiempo", over_time.to_csv(index=False), "tracks_tiempo.csv")
