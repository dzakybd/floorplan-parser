"""Line clean-up shared by both extractors and the ground truth: dedupe, merge fragments, flatten curves."""

import itertools
import math

import numpy as np
from shapely import STRtree
from shapely.affinity import affine_transform
from shapely.geometry import LineString

Point = tuple[float, float]


def dedupe(lines: list[LineString], tol: float = 0.5) -> list[LineString]:
    """Drop zero-length lines and lines lying within `tol` of a longer kept line. Longest lines win."""
    lines = [ln for ln in lines if ln.length > 1e-9]
    order = sorted(range(len(lines)), key=lambda i: -lines[i].length)
    tree = STRtree(lines)
    kept: set[int] = set()
    buffers: dict[int, object] = {}
    for i in order:
        covered = False
        for j in tree.query(lines[i], predicate="dwithin", distance=tol):
            if j in kept:
                if j not in buffers:
                    buffers[j] = lines[j].buffer(tol, cap_style="flat")
                if buffers[j].covers(lines[i]):
                    covered = True
                    break
        if not covered:
            kept.add(i)
    return [lines[i] for i in sorted(kept)]


def merge_collinear(lines: list[LineString], dist_tol: float, angle_tol_deg: float, gap: float) -> list[LineString]:
    """Join straight fragments that share a direction (within `angle_tol_deg`), sit on the same line (within
    `dist_tol`) and are at most `gap` apart along it. Polylines pass through unchanged."""
    straight = [ln for ln in lines if len(ln.coords) == 2 and ln.length > 1e-9]
    out = [ln for ln in lines if len(ln.coords) != 2]
    if not straight:
        return out
    p = np.array([ln.coords[0] for ln in straight])
    q = np.array([ln.coords[1] for ln in straight])
    ang = np.degrees(np.arctan2(q[:, 1] - p[:, 1], q[:, 0] - p[:, 0])) % 180.0
    ang = np.where(ang >= 180.0 - angle_tol_deg, ang - 180.0, ang)
    for group in _chain_clusters(ang, angle_tol_deg):
        theta = math.radians(float(np.mean(ang[group])))
        u = np.array([math.cos(theta), math.sin(theta)])
        n = np.array([-u[1], u[0]])
        offs = (p[group] @ n + q[group] @ n) / 2
        for sub in _chain_clusters(offs, dist_tol):
            idx = np.asarray(group)[sub]
            t = np.sort(np.stack([p[idx] @ u, q[idx] @ u], axis=1), axis=1)
            c = offs[sub]
            order = np.argsort(t[:, 0])
            run_t0, run_t1, run_c = t[order[0], 0], t[order[0], 1], [c[order[0]]]
            for k in order[1:]:
                if t[k, 0] <= run_t1 + gap:
                    run_t1 = max(run_t1, t[k, 1])
                    run_c.append(c[k])
                else:
                    out.append(_segment(u, n, float(np.mean(run_c)), run_t0, run_t1))
                    run_t0, run_t1, run_c = t[k, 0], t[k, 1], [c[k]]
            out.append(_segment(u, n, float(np.mean(run_c)), run_t0, run_t1))
    return out


def _chain_clusters(values: np.ndarray, tol: float) -> list[list[int]]:
    order = np.argsort(values)
    groups, cur = [], [int(order[0])]
    for a, b in itertools.pairwise(order):
        if values[b] - values[a] <= tol:
            cur.append(int(b))
        else:
            groups.append(cur)
            cur = [int(b)]
    groups.append(cur)
    return groups


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
