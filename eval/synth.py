"""Synthetic floor-plan drawings in FloorPlanCAD's SVG style (viewBox 100, black background, 0.1 strokes, semantic-id).

They are used to tune the raster parameters and for every public picture; the FloorPlanCAD drawings are neither.
Each drawing targets one thing that is hard for one of the two routes.

Usage: uv run python -m eval.synth runs/inputs/synth
"""

import itertools
import math
import sys
from pathlib import Path

WALL, SINGLE_DOOR, WINDOW, STAIRS = 33, 1, 7, 28
COLORS = {WALL: "rgb(255,255,255)", SINGLE_DOOR: "rgb(0,255,255)", WINDOW: "rgb(255,0,255)", STAIRS: "rgb(255,255,0)"}


def _f(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


class Drawing:
    def __init__(self) -> None:
        self.items: list[str] = []

    def line(self, a, b, sem: int | None = WALL, color: str | None = None) -> None:
        self._path(f"M {_f(a[0])},{_f(a[1])} L {_f(b[0])},{_f(b[1])}", sem, color)

    def arc(self, a, b, r: float, sweep: int, sem: int | None = SINGLE_DOOR) -> None:
        self._path(f"M {_f(a[0])},{_f(a[1])} A {_f(r)},{_f(r)} 0.0 0,{sweep} {_f(b[0])},{_f(b[1])}", sem)

    def circle(self, c, r: float, sem: int | None = None) -> None:
        attrs = f' semantic-id="{sem}"' if sem else ""
        color = COLORS.get(sem, "rgb(0,178,0)")
        self.items.append(
            f'  <circle cx="{_f(c[0])}" cy="{_f(c[1])}" fill="none" r="{_f(r)}"{attrs} '
            f'stroke="{color}" stroke-width="0.1"/>'
        )

    def wall(self, a, b, thick: float = 1.5) -> None:
        """A wall as two parallel lines `thick` apart."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        ox, oy = -dy / n * thick / 2, dx / n * thick / 2
        self.line((a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy))
        self.line((a[0] - ox, a[1] - oy), (b[0] - ox, b[1] - oy))

    def _path(self, d: str, sem: int | None, color: str | None = None) -> None:
        attrs = f' semantic-id="{sem}"' if sem else ""
        color = color or COLORS.get(sem, "rgb(0,178,0)")
        self.items.append(f'  <path d="{d}" fill="none"{attrs} stroke="{color}" stroke-width="0.1"/>')

    def svg(self) -> str:
        body = "\n".join(self.items)
        return (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<svg style="background-color: #000;" version="1.1" viewBox="0 0 100.0 100.0" '
            'xmlns="http://www.w3.org/2000/svg">\n'
            f' <g id="layer-synthetic">\n{body}\n </g>\n</svg>\n'
        )


def _outer(d: Drawing) -> None:
    for a, b in [((5, 5), (95, 5)), ((95, 5), (95, 95)), ((95, 95), (5, 95)), ((5, 95), (5, 5))]:
        d.wall(a, b)


def walls() -> Drawing:
    d = Drawing()
    _outer(d)
    d.wall((5, 45), (40, 45))
    d.wall((50, 45), (95, 45))  # 10-unit door gap
    d.wall((60, 5), (60, 30))
    d.wall((60, 38), (60, 45))
    d.wall((30, 45), (30, 95))
    for x in (12, 20):  # window in the top wall
        d.line((x, 4.25), (x, 5.75), sem=WINDOW)
    return d


def diagonal() -> Drawing:
    d = Drawing()
    _outer(d)
    for ang, start, length in [(30, (10, 85), 70), (60, (15, 90), 80), (150, (90, 70), 60), (120, (85, 92), 50)]:
        t = math.radians(ang)
        d.wall(start, (start[0] + length * math.cos(t), start[1] - length * math.sin(t)))
    return d


def doors() -> Drawing:
    d = Drawing()
    _outer(d)
    d.wall((5, 50), (35, 50))
    d.wall((45, 50), (95, 50))
    for x0 in (35, 70):  # door leaf plus its quarter swing
        r = 10 if x0 == 35 else 8
        d.line((x0, 50), (x0, 50 - r), sem=SINGLE_DOOR)
        d.arc((x0, 50 - r), (x0 + r, 50), r, 1)
    d.wall((70, 5), (70, 20))
    d.line((70, 28), (78, 28), sem=SINGLE_DOOR)
    d.arc((78, 28), (70, 20), 8, 0)
    for c, r in [((20, 75), 4.0), ((40, 75), 2.0), ((55, 75), 1.0), ((65, 75), 0.5), ((80, 70), 6.0)]:
        d.circle(c, r)
    return d


def stair() -> Drawing:
    d = Drawing()
    _outer(d)
    d.wall((50, 5), (50, 95))
    for i in range(25):  # treads 1 unit apart
        y = 20 + i * 1.0
        d.line((15, y), (40, y), sem=STAIRS)
    d.line((15, 20), (15, 44), sem=STAIRS)
    d.line((40, 20), (40, 44), sem=STAIRS)
    for i in range(60):  # 45-degree hatch filling the box 60..90 x 55..85
        c = 115 + i * 1.0  # line x + y = c
        x0, x1 = max(60, c - 85), min(90, c - 55)
        if x1 - x0 > 0.5:
            d.line((x0, c - x0), (x1, c - x1), sem=None)
    return d


def near_parallel() -> Drawing:
    d = Drawing()
    _outer(d)
    for i, gap in enumerate([0.15, 0.25, 0.35, 0.5, 0.75, 1.0]):
        y = 15 + i * 12
        d.line((15, y), (85, y))
        d.line((15, y + gap), (85, y + gap))
    for i, gap in enumerate([0.2, 0.4]):  # a vertical pair too
        x = 88 + i * 3
        d.line((x, 15), (x, 85))
        d.line((x + gap, 15), (x + gap, 85))
    return d


# Stroke glyphs for the label "1200" (a dimension label drawn as lines, as CAD exports often do)
GLYPHS = {
    "1": [[(0.5, 0), (0.5, 1)], [(0.25, 0.2), (0.5, 0)]],
    "2": [[(0, 0.2), (0.2, 0), (0.8, 0), (1, 0.2), (1, 0.4), (0, 1), (1, 1)]],
    "0": [[(0.2, 0), (0.8, 0), (1, 0.2), (1, 0.8), (0.8, 1), (0.2, 1), (0, 0.8), (0, 0.2), (0.2, 0)]],
}


def text_paths(with_text: bool) -> Drawing:
    d = Drawing()
    _outer(d)
    d.wall((5, 50), (95, 50))
    d.line((20, 60), (80, 60), sem=WINDOW)  # dimension line
    d.line((20, 58), (20, 62), sem=WINDOW)
    d.line((80, 58), (80, 62), sem=WINDOW)
    if with_text:
        x, size = 40.0, 4.0
        for ch in "1200":
            for stroke in GLYPHS[ch]:
                for a, b in itertools.pairwise(stroke):
                    pa = (x + a[0] * size * 0.6, 53 + a[1] * size)
                    pb = (x + b[0] * size * 0.6, 53 + b[1] * size)
                    d.line(pa, pb, sem=None, color="rgb(255,255,0)")
            x += size * 0.8
    return d


def make_synthetic(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    drawings = {
        "synth-1-walls": walls(),
        "synth-2-diagonal": diagonal(),
        "synth-3-doors": doors(),
        "synth-4-stair": stair(),
        "synth-5-near-parallel": near_parallel(),
        "synth-text-paths": text_paths(with_text=True),
    }
    paths = []
    for name, d in drawings.items():
        p = out_dir / f"{name}.svg"
        p.write_text(d.svg())
        paths.append(p)
    (out_dir / "synth-text-paths.gt.svg").write_text(text_paths(with_text=False).svg())
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/inputs/synth")
    print("\n".join(str(p) for p in make_synthetic(out)))
