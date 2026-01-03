from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class ROI:
    roi_id: str
    points: Tuple[Point, ...]

    def contains(self, x: float, y: float) -> bool:
        # Ray casting algorithm for polygons
        num = len(self.points)
        inside = False
        j = num - 1
        for i in range(num):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    @property
    def as_int_points(self) -> List[Tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self.points]


def _rect_to_points(rect: Sequence[float]) -> Tuple[Point, ...]:
    x1, y1, x2, y2 = rect
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def load_rois(path: Path) -> List[ROI]:
    if not path.exists():
        raise FileNotFoundError(f"Archivo de ROIs no encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rois: List[ROI] = []
    for item in data:
        roi_id = item.get("id") or item.get("name")
        if not roi_id:
            raise ValueError("Cada ROI debe tener un 'id' o 'name'")
        if "points" in item:
            points = item["points"]
        elif "rect" in item:
            points = _rect_to_points(item["rect"])
        else:
            raise ValueError(f"ROI {roi_id} debe incluir 'points' o 'rect'")
        rois.append(ROI(roi_id=str(roi_id), points=tuple((float(x), float(y)) for x, y in points)))
    return rois
