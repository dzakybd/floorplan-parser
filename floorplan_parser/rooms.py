"""From line geometry to rooms: walls are pairs of parallel lines one wall thickness apart, rooms are the closed
spaces between them. Geometry only (NumPy + Shapely); nothing here is trained.

1. `wall_thickness`: the most common gap between overlapping parallel lines, weighted by overlap length (CAD
   draws a wall as its two faces, and walls outnumber every other parallel pair).
2. `detect_walls`: every parallel pair about that far apart becomes a wall rectangle.
3. `close_openings`: doorways are gaps between collinear wall pieces; gaps up to `max_gap` are bridged. Only
   collinear pieces are bridged, so a corridor between two parallel walls stays open.
4. `extract_rooms`: the page minus the walls; closed pieces that do not reach the page border are rooms.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from shapely import STRtree, union_all
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class RoomParams:
    max_gap_factor: float = 10.0  # bridge doorways up to this many wall thicknesses wide
    min_area_factor: float = 20.0  # rooms smaller than this many squared wall thicknesses are dropped
    thickness_tol: float = 0.25  # a pair is a wall when its gap is within this share of the wall thickness
    angle_tol_deg: float = 2.0
    max_thickness: float = 40.0  # pt; wider parallel gaps are never walls
    n_peaks: int = 1  # wall thicknesses kept: the strongest peaks of the gap histogram
    peak_share: float = 0.25  # a further peak must weigh at least this share of the strongest one

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RoomParams":
        return cls(**d)


DEFAULT = RoomParams()


@dataclass(frozen=True)
class _Pairs:
    gap: np.ndarray  # perpendicular distance between the two lines
    overlap: np.ndarray  # length of their common span along the first line
    lo: np.ndarray  # common span along u
    hi: np.ndarray
    c0: np.ndarray  # offsets of the two lines along n
    c1: np.ndarray
    u: np.ndarray  # unit direction of the first line
    n: np.ndarray  # its unit normal


def _straight(lines: list[LineString], min_len: float) -> list[LineString]:
    return [ln for ln in lines if len(ln.coords) == 2 and ln.length >= min_len]


def _parallel_pairs(segs: list[LineString], max_gap: float, angle_tol_deg: float, min_overlap: float) -> _Pairs:
    if len(segs) < 2:
        z = np.zeros(0)
        return _Pairs(z, z, z, z, z, z, np.zeros((0, 2)), np.zeros((0, 2)))
    p = np.array([s.coords[0] for s in segs])
    q = np.array([s.coords[1] for s in segs])
    d = (q - p) / np.hypot(*(q - p).T)[:, None]
    i, j = STRtree(segs).query(segs, predicate="dwithin", distance=max_gap)
    keep = i < j
    i, j = i[keep], j[keep]
    u = d[i]
    parallel = np.abs(np.sum(u * d[j], axis=1)) >= math.cos(math.radians(angle_tol_deg))
    i, j, u = i[parallel], j[parallel], u[parallel]
    n = np.stack([-u[:, 1], u[:, 0]], axis=1)
    c0 = np.sum(p[i] * n, axis=1)
    cj_a, cj_b = np.sum(p[j] * n, axis=1), np.sum(q[j] * n, axis=1)
    c1 = (cj_a + cj_b) / 2
    a0, a1 = np.sort(np.stack([np.sum(p[i] * u, 1), np.sum(q[i] * u, 1)], 1), axis=1).T
    b0, b1 = np.sort(np.stack([np.sum(p[j] * u, 1), np.sum(q[j] * u, 1)], 1), axis=1).T
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    gap = np.abs(c1 - c0)
    ok = (hi - lo >= min_overlap) & (gap > 0.25) & (gap <= max_gap) & (np.abs(cj_a - cj_b) <= 0.5)
    return _Pairs(gap[ok], (hi - lo)[ok], lo[ok], hi[ok], c0[ok], c1[ok], u[ok], n[ok])


def wall_thicknesses(lines: list[LineString], p: RoomParams = DEFAULT) -> list[float]:
    """Wall thicknesses (pt), strongest first: peaks of the histogram of gaps between overlapping parallel straight
    lines, weighted by overlap length. Up to `n_peaks`, each at least `peak_share` of the strongest and at least
    1.5 pt from a kept one."""
    pairs = _parallel_pairs(_straight(lines, 4.0), p.max_thickness, p.angle_tol_deg, 4.0)
    if len(pairs.gap) == 0:
        return []
    bins = np.arange(0.25, p.max_thickness + 0.5, 0.5)
    hist, _ = np.histogram(pairs.gap, bins=bins, weights=pairs.overlap)
    out: list[float] = []
    for k in np.argsort(hist)[::-1]:
        if len(out) == p.n_peaks or hist[k] <= 0 or hist[k] < p.peak_share * hist.max():
            break
        in_bin = (pairs.gap >= bins[k]) & (pairs.gap < bins[k + 1])
        t = float(np.average(pairs.gap[in_bin], weights=pairs.overlap[in_bin]))
        if all(abs(t - u) >= 1.5 for u in out):
            out.append(t)
    return out


def wall_thickness(lines: list[LineString], p: RoomParams = DEFAULT) -> float | None:
    """The most common wall thickness (pt), or None when there are no parallel pairs."""
    ts = wall_thicknesses(lines, p)
    return ts[0] if ts else None


@dataclass(frozen=True)
class Band:
    """A straight wall piece: from `lo` to `hi` along unit direction `u`, centred at offset `c` along normal `n`."""

    u: np.ndarray
    n: np.ndarray
    lo: float
    hi: float
    c: float
    width: float

    def polygon(self) -> Polygon:
        h = self.width / 2
        corners = [self.lo * self.u + (self.c - h) * self.n, self.hi * self.u + (self.c - h) * self.n,
                   self.hi * self.u + (self.c + h) * self.n, self.lo * self.u + (self.c + h) * self.n]  # fmt: skip
        return Polygon([tuple(x) for x in corners])


def _band(u: np.ndarray, lo: float, hi: float, c0: float, c1: float) -> Band:
    """A band with one canonical orientation per direction, so collinear pieces share u, n and offset."""
    n = np.array([-u[1], u[0]])
    if u[0] < -1e-12 or (abs(u[0]) <= 1e-12 and u[1] < 0):
        u, n, lo, hi, c0, c1 = -u, -n, -hi, -lo, -c0, -c1
    return Band(u, n, lo, hi, (c0 + c1) / 2, abs(c1 - c0))


def wall_bands(lines: list[LineString], t: float | list[float], p: RoomParams = DEFAULT) -> list[Band]:
    """One band per pair of parallel lines about one of the thicknesses `t` apart, extended by its gap at both ends
    so that walls meeting at a corner overlap instead of touching at a point."""
    ts = [t] if isinstance(t, (int, float)) else list(t)
    if not ts:
        return []
    lo_t, hi_t = min(ts), max(ts)
    pairs = _parallel_pairs(_straight(lines, lo_t / 2), hi_t * (1 + p.thickness_tol), p.angle_tol_deg, lo_t / 2)
    wall = np.zeros(len(pairs.gap), bool)
    for u in ts:
        wall |= np.abs(pairs.gap - u) <= p.thickness_tol * u
    return [
        _band(pairs.u[k], pairs.lo[k] - pairs.gap[k], pairs.hi[k] + pairs.gap[k], pairs.c0[k], pairs.c1[k])
        for k in np.flatnonzero(wall)
    ]


def detect_walls(lines: list[LineString], t: float | list[float], p: RoomParams = DEFAULT) -> BaseGeometry:
    return union_all([b.polygon() for b in wall_bands(lines, t, p)])


def bridges(
    bands: list[Band], max_gap_factor: float, angle_tol_deg: float = 2.0, width_tol: float = 0.25
) -> list[Band]:
    """For each band, a bridge to the nearest band of about the same width that continues it on the same line within
    `max_gap_factor` widths: a doorway closed. Bands on parallel but different lines (a corridor) are never
    bridged."""
    cos_tol = math.cos(math.radians(angle_tol_deg))
    out = []
    for a in bands:
        best = None
        for b in bands:
            if b is a or float(a.u @ b.u) < cos_tol or abs(a.c - b.c) > a.width / 2:
                continue
            if abs(a.width - b.width) > width_tol * a.width:
                continue
            gap = b.lo - a.hi
            if 0 < gap <= max_gap_factor * a.width and (best is None or gap < best):
                best = gap
        if best is not None:
            out.append(Band(a.u, a.n, a.hi, a.hi + best, a.c, a.width))
    return out


def close_openings(lines: list[LineString], t: float | list[float], p: RoomParams = DEFAULT) -> BaseGeometry:
    """Walls with their doorways closed."""
    bands = wall_bands(lines, t, p)
    if not bands:
        return Polygon()
    return union_all([b.polygon() for b in [*bands, *bridges(bands, p.max_gap_factor, p.angle_tol_deg)]])


def extract_rooms(lines: list[LineString], page_size: tuple[float, float], p: RoomParams = DEFAULT) -> list[Polygon]:
    """Room polygons (PDF points), largest first. Empty when no wall thickness can be found."""
    ts = wall_thicknesses(lines, p)
    if not ts:
        return []
    t = ts[0]
    walls = close_openings(lines, ts, p)
    if walls.is_empty:
        return []
    page = box(0, 0, *page_size)
    free = page.difference(walls)
    border = page.exterior.buffer(1e-6)
    rooms = [
        g for g in getattr(free, "geoms", [free]) if not g.intersects(border) and g.area >= p.min_area_factor * t * t
    ]
    return sorted(rooms, key=lambda g: -g.area)


def write_rooms(ex, path: Path, p: RoomParams = DEFAULT) -> dict:
    """rooms.json for an extraction: wall thickness and each room's polygon, area (pt2) and perimeter (pt)."""
    lines = ex.lines()
    t = wall_thickness(lines, p)
    rooms = extract_rooms(lines, ex.page_size, p)
    data = {
        "drawing_id": ex.drawing_id,
        "page": ex.page,
        "coordinate_system": "pdf_points",
        "unit": "pt2",
        "wall_thickness_pt": round(t, 3) if t else None,
        "rooms": [
            {
                "id": f"room_{i}",
                "area_pt2": round(r.area, 2),
                "perimeter_pt": round(r.length, 2),
                "polygon": [[round(x, 3), round(y, 3)] for x, y in r.exterior.coords],
            }
            for i, r in enumerate(rooms, 1)
        ],
    }
    path.write_text(json.dumps(data, indent=1))
    return data
