"""Length-based line matching: how much of one set of lines lies on the other, with direction checked.

Scoring by length (not by segment count) keeps a line split into 50 pieces worth the same as the whole line.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from shapely import STRtree, points
from shapely.geometry import LineString

from floorplan_parser.geometry import dedupe

TESSELLATION_TOL = 0.1  # pt, chord error when flattening arcs, circles and curves
DIST_TOL = 2.0  # pt, a sample matches a line within this distance
ANGLE_TOL_DEG = 5.0  # and within this direction difference
SAMPLE_STEP = 1.0  # pt between samples along a line
ENDPOINT_RADIUS = 5.0  # pt, farthest an endpoint may be from its match
SHORT_SEGMENT = 2 * DIST_TOL  # pt, pieces shorter than this are curve flattening; matched by distance only

CONSTANTS = {
    "tessellation_tol_pt": TESSELLATION_TOL,
    "dist_tol_pt": DIST_TOL,
    "angle_tol_deg": ANGLE_TOL_DEG,
    "sample_step_pt": SAMPLE_STEP,
    "endpoint_radius_pt": ENDPOINT_RADIUS,
    "short_segment_pt": SHORT_SEGMENT,
}


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    endpoint_err_mean: float
    endpoint_err_median: float
    endpoint_match_rate: float
    length_error_pct: float
    gt_length: float
    pred_length: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _segments(lines: list[LineString]) -> tuple[np.ndarray, np.ndarray]:
    """Explode lines into 2-point segments: start points (N, 2) and end points (N, 2)."""
    a, b = [], []
    for ln in lines:
        c = np.asarray(ln.coords)
        a.append(c[:-1])
        b.append(c[1:])
    if not a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a, b = np.concatenate(a), np.concatenate(b)
    keep = np.hypot(*(b - a).T) > 1e-9
    return a[keep], b[keep]


def _samples(a: np.ndarray, b: np.ndarray):
    """Points every SAMPLE_STEP along each segment, with the segment index and the length each sample stands for."""
    lengths = np.hypot(*(b - a).T)
    counts = np.maximum(1, np.ceil(lengths / SAMPLE_STEP)).astype(int)
    seg = np.repeat(np.arange(len(a)), counts)
    k = np.concatenate([(np.arange(n) + 0.5) / n for n in counts]) if len(a) else np.zeros(0)
    pts = a[seg] + (b[seg] - a[seg]) * k[:, None]
    return pts, seg, (lengths / counts)[seg]


def _matched(src: list[LineString], dst: list[LineString]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per sample of `src`: whether it lies on `dst` (distance and direction), its weight, segment index and point."""
    sa, sb = _segments(src)
    pts, seg, w = _samples(sa, sb)
    hit = np.zeros(len(pts), bool)
    da, db = _segments(dst)
    if len(pts) == 0 or len(da) == 0:
        return hit, w, seg, pts
    tree = STRtree([LineString([tuple(x), tuple(y)]) for x, y in zip(da, db)])
    pi, di = tree.query(points(pts), predicate="dwithin", distance=DIST_TOL)
    sdir = sb[seg[pi]] - sa[seg[pi]]
    ddir = db[di] - da[di]
    slen, dlen = np.hypot(*sdir.T), np.hypot(*ddir.T)
    cos = np.abs(np.sum(sdir * ddir, axis=1)) / (slen * dlen)
    # Two flattenings of a tight curve disagree on direction piece by piece; only long pieces get the angle check.
    ok = (cos >= math.cos(math.radians(ANGLE_TOL_DEG))) | (slen < SHORT_SEGMENT) | (dlen < SHORT_SEGMENT)
    hit[pi[ok]] = True
    return hit, w, seg, pts


def _coverage(src: list[LineString], dst: list[LineString]) -> float:
    hit, w, _, _ = _matched(src, dst)
    return float(w[hit].sum() / w.sum()) if w.sum() > 0 else 0.0


def unmatched_points(src: list[LineString], dst: list[LineString]) -> np.ndarray:
    """Sample points of `src` (every SAMPLE_STEP pt) that do not lie on `dst`: missed lines when `src` is the ground
    truth, extra lines when `src` is the prediction."""
    hit, _, _, pts = _matched(src, dst)
    return pts[~hit]


def _endpoints(lines: list[LineString]) -> np.ndarray:
    return np.array([c for ln in lines for c in (ln.coords[0], ln.coords[-1])]).reshape(-1, 2)


def score(pred: list[LineString], gt: list[LineString]) -> Scores:
    recall = _coverage(gt, pred)
    precision = _coverage(pred, gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    errs = np.zeros(0)
    gte, pe = _endpoints(gt), _endpoints(pred)
    if len(gte) and len(pe):
        tree = STRtree(points(pe))
        _, d = tree.query_nearest(points(gte), max_distance=ENDPOINT_RADIUS, return_distance=True, all_matches=False)
        errs = d
    gt_len = sum(ln.length for ln in gt)
    pred_len = sum(ln.length for ln in dedupe(pred)) if pred else 0.0
    return Scores(
        precision=precision,
        recall=recall,
        f1=f1,
        endpoint_err_mean=float(errs.mean()) if len(errs) else float("nan"),
        endpoint_err_median=float(np.median(errs)) if len(errs) else float("nan"),
        endpoint_match_rate=len(errs) / len(gte) if len(gte) else 0.0,
        length_error_pct=abs(pred_len - gt_len) / gt_len * 100 if gt_len else 0.0,
        gt_length=gt_len,
        pred_length=pred_len,
    )


def recall_by_label(pred: list[LineString], gt: list[LineString], labels: list[str | None]) -> dict[str, float]:
    """Recall per ground-truth class. Extracted lines carry no labels, so there is no per-class precision."""
    out = {}
    for name in dict.fromkeys(lb or "unlabelled" for lb in labels):
        subset = [g for g, lb in zip(gt, labels) if (lb or "unlabelled") == name]
        out[name] = round(_coverage(subset, pred), 4)
    return out
