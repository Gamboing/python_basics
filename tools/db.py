from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine(db_url: Optional[str] = None) -> Engine:
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL no está definido. Usa --db-url o variable de entorno.")
    return create_engine(url, pool_pre_ping=True)
