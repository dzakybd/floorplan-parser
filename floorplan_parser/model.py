"""Output contract shared by both extractors: one Extraction per page, written as JSON and CSV."""

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from shapely.geometry import LineString

Point = tuple[float, float]


class ExtractionError(Exception):
    """The input could not be read as a PDF page."""


@dataclass(frozen=True)
class Element:
    id: str
    kind: Literal["line", "polyline"]
    geometry: list[Point]
    extractor: str
    source_bbox: tuple[float, float, float, float]

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.geometry, self.geometry[1:]))


@dataclass
class Extraction:
    drawing_id: str
    page: int
    extractor: str
    elements: list[Element]
    coordinate_system: str = "pdf_points"
    page_size: tuple[float, float] = (0.0, 0.0)
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> list[LineString]:
        return [LineString(e.geometry) for e in self.elements]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: Path) -> "Extraction":
        data = json.loads(path.read_text())
        data["elements"] = [
            Element(
                e["id"],
                e["kind"],
                [tuple(p) for p in e["geometry"]],
                e["extractor"],
                tuple(e["source_bbox"]),
            )
            for e in data["elements"]
        ]
        data["page_size"] = tuple(data["page_size"])
        return cls(**data)

    def to_csv(self, path: Path) -> None:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "kind", "extractor", "x0", "y0", "x1", "y1", "n_points", "length"])
            for e in self.elements:
                (x0, y0), (x1, y1) = e.geometry[0], e.geometry[-1]
                w.writerow([e.id, e.kind, e.extractor, x0, y0, x1, y1, len(e.geometry), round(e.length, 4)])


def elements_from_lines(lines: list[LineString], extractor: str) -> list[Element]:
    out = []
    for i, line in enumerate(lines, 1):
        coords = [(float(x), float(y)) for x, y in line.coords]
        kind = "line" if len(coords) == 2 else "polyline"
        out.append(Element(f"ln_{i}", kind, coords, extractor, tuple(float(v) for v in line.bounds)))
    return out
