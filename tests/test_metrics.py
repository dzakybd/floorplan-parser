import math

from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import LineString

from floorplan_parser.geometry import affine
from floorplan_parser.metrics import recall_by_label, score

GT = [LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)]), LineString([(20, 10), (80, 40)])]


def test_identical_sets_score_perfect():
    s = score(GT, GT)
    assert s.precision == s.recall == s.f1 == 1.0
    assert s.length_error_pct == 0.0
    assert s.endpoint_err_mean == 0.0 and s.endpoint_match_rate == 1.0


def test_split_line_scores_same_as_whole():
    gt = [LineString([(0, 0), (100, 0)])]
    pieces = [LineString([(2 * i, 0), (2 * i + 2, 0)]) for i in range(50)]
    s = score(pieces, gt)
    assert s.f1 == 1.0
    assert math.isclose(s.length_error_pct, 0.0, abs_tol=1e-6)


def test_offset_beyond_tol_scores_zero():
    s = score([LineString([(0, 3), (100, 3)])], [LineString([(0, 0), (100, 0)])])
    assert s.recall == 0.0 and s.precision == 0.0


def test_perpendicular_line_not_matched():
    s = score([LineString([(50, -50), (50, 50)])], [LineString([(0, 0), (100, 0)])])
    assert s.recall < 0.05


def test_missing_half_gives_recall_half():
    s = score([LineString([(0, 0), (50, 0)])], [LineString([(0, 0), (100, 0)])])
    assert math.isclose(s.recall, 0.5, abs_tol=0.03)
    assert s.precision == 1.0
    assert math.isclose(s.length_error_pct, 50.0, abs_tol=1e-6)


def test_empty_prediction_scores_zero_without_error():
    s = score([], GT)
    assert s.recall == 0.0 and s.precision == 0.0 and s.f1 == 0.0


def test_recall_by_label():
    gt = [LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)]), LineString([(0, 90), (10, 90)])]
    out = recall_by_label([gt[0]], gt, ["wall", "door", None])
    assert out == {"wall": 1.0, "door": 0.0, "unlabelled": 0.0}


@settings(max_examples=25, deadline=None)
@given(st.floats(-300, 300), st.floats(-300, 300))
def test_translation_preserves_scores(tx, ty):
    pred = [LineString([(0, 0), (60, 0)]), LineString([(0, 50.5), (100, 50.5)])]
    base = score(pred, GT)
    m = (1, 0, 0, 1, tx, ty)
    moved = score(affine(pred, m), affine(GT, m))
    assert math.isclose(base.f1, moved.f1, abs_tol=1e-6)


def _circle(r, n, phase=0.0):
    pts = [
        (400 + r * math.cos(phase + 2 * math.pi * i / n), 400 + r * math.sin(phase + 2 * math.pi * i / n))
        for i in range(n)
    ]
    return LineString(pts + [pts[0]])


def test_small_circle_flattened_differently_scores_perfect():
    s = score([_circle(4.0, 40, phase=0.05)], [_circle(4.0, 14)])
    assert s.precision >= 0.99 and s.recall >= 0.99


def test_collapsed_pair_shows_in_length_error_not_f1():
    # Known limit: matching is many-to-one within DIST_TOL, so one line between two true lines 1.2 pt apart scores
    # F1 1.0; the collapse shows only as a 50% length error. Pinned so a change of behaviour is deliberate.
    gt = [LineString([(0, 0), (100, 0)]), LineString([(0, 1.2), (100, 1.2)])]
    s = score([LineString([(0, 0.6), (100, 0.6)])], gt)
    assert s.f1 == 1.0
    assert math.isclose(s.length_error_pct, 50.0, abs_tol=1e-6)
