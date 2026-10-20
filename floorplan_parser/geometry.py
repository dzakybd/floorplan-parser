"""Line clean-up shared by both extractors and the ground truth: dedupe, merge fragments, flatten curves."""

import math

import numpy as np
from shapely import STRtree
from shapely.affinity import affine_transform
from shapely.geometry import LineString

Point = tuple[float, float]


def dedupe(lines: list[LineString], tol: float = 0.5) -> list[LineString]:
    """Drop zero-length lines and lines lying within `tol` of a longer kept line. Longest lines win."""
    return [lines[i] for i in dedupe_indices(lines, tol)]


def dedupe_indices(lines: list[LineString], tol: float = 0.5) -> list[int]:
    """Indices (ascending) of the lines `dedupe` keeps, so callers can keep per-line data aligned."""
    valid = [i for i, ln in enumerate(lines) if ln.length > 1e-9]
    if not valid:
        return []
    tree = STRtree([lines[i] for i in valid])
    kept: set[int] = set()
    buffers: dict[int, object] = {}
    for k in sorted(range(len(valid)), key=lambda k: -lines[valid[k]].length):
        line = lines[valid[k]]
        covered = False
        for j in tree.query(line, predicate="dwithin", distance=tol):
            if j in kept:
                if j not in buffers:
                    buffers[j] = lines[valid[j]].buffer(tol, cap_style="flat")
                if buffers[j].covers(line):
                    covered = True
                    break
        if not covered:
            kept.add(k)
    return sorted(valid[k] for k in kept)


def merge_collinear(lines: list[LineString], dist_tol: float, angle_tol_deg: float, gap: float) -> list[LineString]:
    """Join straight fragments that share a direction (within `angle_tol_deg`), sit on the same line (within
    `dist_tol`) and are at most `gap` apart along it. Polylines pass through unchanged.

    Each merge grows from the longest unused fragment and keeps that fragment's direction, so nearby short pieces at
    other angles (curves, circles) can never tilt a long line."""
    straight = [ln for ln in lines if len(ln.coords) == 2 and ln.length > 1e-9]
    out = [ln for ln in lines if len(ln.coords) != 2]
    if not straight:
        return out
    p = np.array([ln.coords[0] for ln in straight])
    q = np.array([ln.coords[1] for ln in straight])
    d = (q - p) / np.hypot(*(q - p).T)[:, None]
    tree = STRtree(straight)
    cos_tol = math.cos(math.radians(angle_tol_deg))
    used = np.zeros(len(straight), bool)
    for i in np.argsort([-ln.length for ln in straight]):
        if used[i]:
            continue
        used[i] = True
        u = d[i]
        n = np.array([-u[1], u[0]])
        c = float(p[i] @ n)
        t0, t1 = sorted((float(p[i] @ u), float(q[i] @ u)))
        grown = True
        while grown:
            grown = False
            reach = _segment(u, n, c, t0 - gap, t1 + gap)
            for j in tree.query(reach, predicate="dwithin", distance=dist_tol):
                if used[j] or abs(float(d[j] @ u)) < cos_tol:
                    continue
                if abs(float(p[j] @ n) - c) > dist_tol or abs(float(q[j] @ n) - c) > dist_tol:
                    continue
                ta, tb = sorted((float(p[j] @ u), float(q[j] @ u)))
                if ta > t1 + gap or tb < t0 - gap:
                    continue
                t0, t1 = min(t0, ta), max(t1, tb)
                used[j] = grown = True
        out.append(_segment(u, n, c, t0, t1))
    return out


def chain_smooth(
    lines: list[LineString], max_turn_deg: float = 30.0, min_turn_deg: float = 0.1, tol: float = 1e-3
) -> list[LineString]:
    """Join pieces end to end into polylines where exactly two pieces meet (within `tol`) and the path turns by
    less than `max_turn_deg` there: a curve flattened into dozens of short lines comes back as one polyline,
    while corners and junctions (three or more pieces) stay separate. A straight continuation (turn below
    `min_turn_deg`) is not joined either: a flattened curve always turns, while two collinear lines that meet are
    usually two lines in the CAD file. The geometry itself is unchanged."""
    lines = [ln for ln in lines if ln.length > 1e-9]
    coords = [np.asarray(ln.coords, float) for ln in lines]
    nodes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, c in enumerate(coords):
        for end, pt in ((0, c[0]), (1, c[-1])):
            nodes.setdefault((round(pt[0] / tol), round(pt[1] / tol)), []).append((i, end))
    cos_max = math.cos(math.radians(max_turn_deg))
    cos_min = math.cos(math.radians(min_turn_deg))

    def outward(i: int, end: int) -> np.ndarray:
        c = coords[i]
        d = c[0] - c[1] if end == 0 else c[-1] - c[-2]
        return d / np.hypot(*d)

    def partner(i: int, end: int) -> tuple[int, int] | None:
        pt = coords[i][0 if end == 0 else -1]
        here = nodes[(round(pt[0] / tol), round(pt[1] / tol))]
        if len(here) != 2:
            return None
        j, e = here[0] if here[1] == (i, end) else here[1]
        cos_turn = float(-outward(i, end) @ outward(j, e))
        if j == i or cos_turn < cos_max or (min_turn_deg > 0 and cos_turn > cos_min):
            return None
        return j, e

    used = [False] * len(lines)
    out = []
    for i in range(len(lines)):
        if used[i]:
            continue
        cur, end, seen = i, 0, {i}  # walk back to the open start of the chain (or around a loop)
        while (p := partner(cur, end)) is not None and p[0] not in seen:
            seen.add(p[0])
            cur, end = p[0], 1 - p[1]
        first, start_end = cur, end
        path = list(coords[cur] if end == 0 else coords[cur][::-1])
        used[cur] = True
        end = 1 - end
        while (p := partner(cur, end)) is not None and not used[p[0]]:
            nxt = coords[p[0]] if p[1] == 0 else coords[p[0]][::-1]
            path.extend(nxt[1:])
            used[p[0]] = True
            cur, end = p[0], 1 - p[1]
        if p == (first, start_end):
            path[-1] = path[0]  # a closed loop, such as a circle
        out.append(LineString(path))
    return out


def _segment(u, n, c, t0, t1) -> LineString:
    return LineString([tuple(c * n + t0 * u), tuple(c * n + t1 * u)])


def tessellate_bezier(p0: Point, p1: Point, p2: Point, p3: Point, tol: float) -> list[Point]:
    """Flatten a cubic Bézier into a polyline whose chord error is at most `tol`."""
    pts = [tuple(map(float, p0))]
    _subdivide(np.array(p0, float), np.array(p1, float), np.array(p2, float), np.array(p3, float), tol, pts, 0)
    return pts


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _subdivide(a, b, c, d, tol, pts, depth):
    chord = d - a
    norm = np.hypot(*chord)
    if norm < 1e-12:
        flat = max(np.hypot(*(b - a)), np.hypot(*(c - a)))
    else:
        flat = max(abs(_cross(chord, b - a)), abs(_cross(chord, c - a))) / norm
    if flat <= tol or depth >= 16:
        pts.append((float(d[0]), float(d[1])))
        return
    ab, bc, cd = (a + b) / 2, (b + c) / 2, (c + d) / 2
    abc, bcd = (ab + bc) / 2, (bc + cd) / 2
    mid = (abc + bcd) / 2
    _subdivide(a, ab, abc, mid, tol, pts, depth + 1)
    _subdivide(mid, bcd, cd, d, tol, pts, depth + 1)


def affine(lines: list[LineString], matrix: tuple[float, float, float, float, float, float]) -> list[LineString]:
    """Apply a PDF-style matrix (a, b, c, d, e, f): x' = a*x + c*y + e, y' = b*x + d*y + f."""
    a, b, c, d, e, f = matrix
    return [affine_transform(ln, [a, c, b, d, e, f]) for ln in lines]
