"""Ground truth for scoring: the line geometry of the original FloorPlanCAD SVG, in PDF points.

Arcs, circles and curves are flattened with the same tolerance used everywhere else. Text is skipped. The
`semantic-id` label rides along only for per-class recall; it never reaches an extractor.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import svgelements as se
from shapely.geometry import LineString

from floorplan_parser.geometry import dedupe_indices, tessellate_bezier
from floorplan_parser.metrics import TESSELLATION_TOL

# FloorPlanCAD semantic ids, as listed in VITA-Group/CADTransformer config/anno_config.py
CLASS_NAMES = {
    1: "single door",
    2: "double door",
    3: "sliding door",
    4: "folding door",
    5: "revolving door",
    6: "rolling door",
    7: "window",
    8: "bay window",
    9: "blind window",
    10: "opening symbol",
    11: "sofa",
    12: "bed",
    13: "chair",
    14: "table",
    15: "TV cabinet",
    16: "wardrobe",
    17: "cabinet",
    18: "gas stove",
    19: "sink",
    20: "refrigerator",
    21: "airconditioner",
    22: "bath",
    23: "bath tub",
    24: "washing machine",
    25: "squat toilet",
    26: "urinal",
    27: "toilet",
    28: "stairs",
    29: "elevator",
    30: "escalator",
    31: "row chairs",
    32: "parking spot",
    33: "wall",
    34: "curtain wall",
    35: "railing",
}


@dataclass
class GroundTruth:
    lines: list[LineString]
    labels: list[str | None]
    page_size: tuple[float, float]


def _arc_points(seg: se.Arc, tol: float) -> list[tuple[float, float]]:
    r = max(float(seg.rx), float(seg.ry), 1e-9)
    step = 2 * math.acos(max(-1.0, 1 - tol / r)) if tol < r else math.pi / 2
    n = max(2, math.ceil(abs(float(seg.sweep)) / step))
    return [(float(x), float(y)) for x, y in seg.npoint(np.linspace(0, 1, n + 1))[1:]]


def _subpaths(shape: se.Shape, tol: float) -> list[list[tuple[float, float]]]:
    out, cur, start = [], [], None
    for seg in se.Path(shape).segments():
        if isinstance(seg, se.Move):
            if len(cur) > 1:
                out.append(cur)
            start = (float(seg.end.x), float(seg.end.y))
            cur = [start]
        elif isinstance(seg, se.Close):
            if start is not None and cur and cur[-1] != start:
                cur.append(start)
        elif isinstance(seg, se.Line):
            cur.append((float(seg.end.x), float(seg.end.y)))
        elif isinstance(seg, se.Arc):
            cur.extend(_arc_points(seg, tol))
        elif isinstance(seg, se.CubicBezier):
            ctrl = [(float(p.x), float(p.y)) for p in (seg.start, seg.control1, seg.control2, seg.end)]
            cur.extend(tessellate_bezier(*ctrl, tol=tol)[1:])
        elif isinstance(seg, se.QuadraticBezier):
            p0, c, p1 = (np.array([float(p.x), float(p.y)]) for p in (seg.start, seg.control, seg.end))
            ctrl = [tuple(p0), tuple(p0 + 2 / 3 * (c - p0)), tuple(p1 + 2 / 3 * (c - p1)), tuple(p1)]
            cur.extend(tessellate_bezier(*ctrl, tol=tol)[1:])
    if len(cur) > 1:
        out.append(cur)
    return out


def load_ground_truth(svg_path: Path, page_pt: float = 800.0) -> GroundTruth:
    svg = se.SVG.parse(str(svg_path), width=page_pt, height=page_pt)
    lines, labels = [], []
    for el in svg.elements():
        if isinstance(el, se.Text) or not isinstance(el, se.Shape):
            continue
        sem = el.values.get("semantic-id")
        label = CLASS_NAMES.get(int(sem)) if sem not in (None, "") else None
        for pts in _subpaths(el, TESSELLATION_TOL):
            lines.append(LineString(pts))
            labels.append(label)
    keep = dedupe_indices(lines)
    return GroundTruth([lines[i] for i in keep], [labels[i] for i in keep], (page_pt, page_pt))
