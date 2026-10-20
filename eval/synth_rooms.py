"""Synthetic floor plans with a room answer key, drawn the CAD way, for the lines -> walls -> rooms step.

FloorPlanCAD has no room annotations, so rooms are scored on these. Each plan is a building split into a grid of
rectangular rooms with walls of one random thickness, a doorway with a swing arc in every inner wall, an entrance
in the outer wall, furniture and windows (parallel lines that are not walls), and sometimes an open kitchen: a
removed wall between two rooms that the answer key still counts as two rooms.

Seeds 0-9 tune the room parameters; seeds 100-119 are held out.

Usage: uv run python -m eval.synth_rooms runs/inputs/rooms
"""

import itertools
import json
import random
import sys
from pathlib import Path

from shapely import union_all
from shapely.geometry import Polygon, box

from eval.synth import SINGLE_DOOR, WINDOW, Drawing

SCALE = 8.0  # SVG units (viewBox 100) to PDF points (800 pt page)
TUNE_SEEDS = list(range(10))
TEST_SEEDS = list(range(100, 120))


def _cuts(rng: random.Random, lo: float, hi: float, n: int, min_size: float) -> list[float]:
    while True:
        xs = sorted(rng.uniform(lo + min_size, hi - min_size) for _ in range(n - 1))
        edges = [lo, *xs, hi]
        if all(b - a >= min_size for a, b in itertools.pairwise(edges)):
            return edges


def plan(seed: int) -> tuple[Drawing, list[Polygon], float]:
    """The drawing, the reference rooms (PDF points) and the wall thickness (PDF points)."""
    rng = random.Random(seed)
    t = rng.choice([1.0, 1.25, 1.5, 2.0])
    x0, y0 = 8.0, 8.0
    x1, y1 = x0 + rng.uniform(60, 84), y0 + rng.uniform(50, 84)
    xs = _cuts(rng, x0, x1, rng.randint(2, 4), 14)
    ys = _cuts(rng, y0, y1, rng.randint(2, 3), 14)
    cells = [(i, j) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]

    def inner(i, j):
        return box(xs[i] + t / 2, ys[j] + t / 2, xs[i + 1] - t / 2, ys[j + 1] - t / 2)

    walls = [
        box(x0 - t / 2, y0 - t / 2, x1 + t / 2, y1 + t / 2).difference(
            box(x0 + t / 2, y0 + t / 2, x1 - t / 2, y1 - t / 2)
        )
    ]
    open_wall = None
    inner_walls = []
    for i, j in cells:  # the wall to the right of and below each cell, shared with the neighbour
        if i + 1 < len(xs) - 1:
            inner_walls.append(("v", xs[i + 1], ys[j], ys[j + 1]))
        if j + 1 < len(ys) - 1:
            inner_walls.append(("h", ys[j + 1], xs[i], xs[i + 1]))
    if rng.random() < 0.4:
        open_wall = rng.randrange(len(inner_walls))
    d = Drawing()
    gaps = []
    for k, (o, at, a, b) in enumerate(inner_walls):
        if k == open_wall:
            continue
        seg = box(at - t / 2, a, at + t / 2, b) if o == "v" else box(a, at - t / 2, b, at + t / 2)
        walls.append(seg)
        width = rng.uniform(5, 9)
        lo, hi = a + t + 1, b - t - 1 - width
        if hi > lo:
            s = rng.uniform(lo, hi)
            gaps.append((o, at, s, s + width))
    # entrance in the bottom outer wall
    width = rng.uniform(6, 9)
    s = rng.uniform(xs[0] + t + 1, xs[1] - t - 1 - width)
    gaps.append(("h", y1, s, s + width))
    wall_geom = union_all(walls)
    for o, at, s, e in gaps:
        cut = box(at - t, s, at + t, e) if o == "v" else box(s, at - t, e, at + t)
        wall_geom = wall_geom.difference(cut)
        # door leaf and swing arc on one side
        if o == "h":
            d.line((s, at), (s, at - (e - s)), sem=SINGLE_DOOR)
            d.arc((s, at - (e - s)), (e, at), e - s, 1)
        else:
            d.line((at, s), (at + (e - s), s), sem=SINGLE_DOOR)
            d.arc((at + (e - s), s), (at, e), e - s, 1)
    for ring_owner in getattr(wall_geom, "geoms", [wall_geom]):
        for ring in [ring_owner.exterior, *ring_owner.interiors]:
            c = list(ring.coords)
            for a, b in itertools.pairwise(c):
                d.line(a, b)
    for _ in range(rng.randint(2, 5)):  # windows: three lines inside the outer wall band
        wx = rng.uniform(x0 + 4, x1 - 10)
        for off in (-t / 6, t / 6):
            d.line((wx, y0 + off), (wx + 6, y0 + off), sem=WINDOW)
    for i, j in cells:  # furniture: a rectangle in most rooms
        r = inner(i, j)
        if rng.random() < 0.7:
            fx0, fy0, fx1, fy1 = r.bounds
            w, h = rng.uniform(4, 10), rng.uniform(4, 8)
            fx, fy = rng.uniform(fx0 + 2, max(fx0 + 2, fx1 - w - 2)), rng.uniform(fy0 + 2, max(fy0 + 2, fy1 - h - 2))
            for a, b in [
                ((fx, fy), (fx + w, fy)),
                ((fx + w, fy), (fx + w, fy + h)),
                ((fx + w, fy + h), (fx, fy + h)),
                ((fx, fy + h), (fx, fy)),
            ]:
                d.line(a, b, sem=None)
    rooms = [Polygon([(x * SCALE, y * SCALE) for x, y in inner(i, j).exterior.coords]) for i, j in cells]
    return d, rooms, t * SCALE


def make(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for seed in TUNE_SEEDS + TEST_SEEDS:
        d, rooms, t = plan(seed)
        name = f"rooms-{seed:03d}"
        (out_dir / f"{name}.svg").write_text(d.svg(), encoding="utf-8")
        (out_dir / f"{name}.rooms.json").write_text(
            json.dumps({"wall_thickness_pt": t, "rooms": [list(r.exterior.coords) for r in rooms]})
        )
        names.append(name)
    return names


if __name__ == "__main__":
    print(len(make(Path(sys.argv[1]))), "plans")
