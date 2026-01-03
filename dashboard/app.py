from __future__ import annotations

import os
import streamlit as st

from dashboard.queries import get_engine
from dashboard.pages import overview, materials_categories, products, predictions, video_analytics


st.set_page_config(page_title="Retail Dashboard", layout="wide")


def main() -> None:
    st.title("Retail Dashboard (Postgres)")
    db_url = st.secrets.get("DATABASE_URL") if "DATABASE_URL" in st.secrets else os.getenv("DATABASE_URL")
    if not db_url:
        st.error("Define DATABASE_URL en entorno o .streamlit/secrets.toml")
        return
    engine = get_engine(db_url)

    page = st.sidebar.radio(
        "Páginas",
        ["Overview", "Materiales/Categorías", "Productos", "Predicción", "Video analytics"],
    )
    if page == "Overview":
        overview.render(engine)
    elif page == "Materiales/Categorías":
        materials_categories.render(engine)
    elif page == "Productos":
        products.render(engine)
    elif page == "Predicción":
        predictions.render(engine)
    elif page == "Video analytics":
        video_analytics.render(engine)


if __name__ == "__main__":
    main()
