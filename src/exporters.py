from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExportBuffer:
    rows: List[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = []
        if self.events is None:
            self.events = []


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    # Crear archivo incluso si no hay filas, para señalar que se intentó exportar.
    if not rows:
        path.touch()
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
