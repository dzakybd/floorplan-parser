import math

from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import unary_union
from shapely.geometry import LineString

from floorplan_parser.geometry import affine, chain_smooth, dedupe, merge_collinear, tessellate_bezier

coord = st.floats(-500, 500, allow_nan=False)
segment = st.tuples(coord, coord, coord, coord).filter(lambda s: math.dist(s[:2], s[2:]) > 1.0)


def _total(lines):
    return sum(line.length for line in lines)


def test_dedupe_drops_duplicates_and_zero_length():
    a = LineString([(0, 0), (100, 0)])
    lines = [a, LineString([(100, 0), (0, 0)]), LineString([(5, 5), (5, 5)]), LineString([(0, 50), (100, 50)])]
    out = dedupe(lines)
    assert len(out) == 2
    assert math.isclose(_total(out), 200.0)


def test_dedupe_drops_line_covered_by_longer_one():
    out = dedupe([LineString([(0, 0), (100, 0)]), LineString([(20, 0.2), (60, 0.2)])])
    assert len(out) == 1 and math.isclose(out[0].length, 100.0)


def test_merge_collinear_joins_gap():
    out = merge_collinear(
        [LineString([(0, 0), (40, 0)]), LineString([(41.5, 0.3), (100, 0)])], dist_tol=1.0, angle_tol_deg=5, gap=2.0
    )
    assert len(out) == 1
    assert math.isclose(out[0].length, 100.0, rel_tol=0.01)


def test_merge_collinear_keeps_perpendicular_apart():
    out = merge_collinear(
        [LineString([(0, 0), (40, 0)]), LineString([(40, 0), (40, 40)])], dist_tol=1.0, angle_tol_deg=5, gap=2.0
    )
    assert len(out) == 2


def test_tessellate_bezier_quarter_circle():
    r, k = 100.0, 0.5522847498
    pts = tessellate_bezier((r, 0), (r, r * k), (r * k, r), (0, r), tol=0.1)
    assert math.isclose(LineString(pts).length, math.pi * r / 2, rel_tol=0.005)
    assert pts[0] == (r, 0) and pts[-1] == (0, r)


@settings(max_examples=60)
@given(segment, st.floats(0, 360), coord, coord)
def test_rotation_preserves_lengths(s, deg, tx, ty):
    t = math.radians(deg)
    m = (math.cos(t), math.sin(t), -math.sin(t), math.cos(t), tx, ty)
    line = LineString([s[:2], s[2:]])
    assert math.isclose(affine([line], m)[0].length, line.length, rel_tol=1e-9)


@settings(max_examples=60)
@given(st.lists(segment, min_size=1, max_size=5), st.floats(0.1, 10))
def test_uniform_scale_multiplies_length(segs, s):
    lines = [LineString([g[:2], g[2:]]) for g in segs]
    assert math.isclose(_total(affine(lines, (s, 0, 0, s, 0, 0))), _total(lines) * s, rel_tol=1e-6)


def test_merge_collinear_keeps_direction_with_circle_pieces_nearby():
    wall = [LineString([(0, 100), (300, 100)]), LineString([(301, 100), (600, 100)])]
    # short pieces at angles less than 2 degrees apart around a circle, as Hough returns at high DPI
    circle = [
        LineString(
            [
                (450 + 40 * math.cos(t), 300 + 40 * math.sin(t)),
                (450 + 40 * math.cos(t + 0.08), 300 + 40 * math.sin(t + 0.08)),
            ]
        )
        for t in [i * 0.03 for i in range(105)]
    ]
    out = merge_collinear(wall + circle, dist_tol=0.75, angle_tol_deg=2.0, gap=2.0)
    merged = [ln for ln in out if ln.length > 500]
    assert len(merged) == 1
    ys = [y for _, y in merged[0].coords]
    assert max(abs(y - 100) for y in ys) < 0.05


@settings(max_examples=30)
@given(segment, st.floats(0.1, 10), coord, coord)
def test_coordinate_roundtrip(s, scale, tx, ty):
    line = LineString([s[:2], s[2:]])
    moved = affine([line], (scale, 0, 0, scale, tx, ty))
    back = affine(moved, (1 / scale, 0, 0, 1 / scale, -tx / scale, -ty / scale))[0]
    assert line.hausdorff_distance(back) < 1e-8


def _arc(cx, cy, r, a0, a1, n):
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]
    return [LineString([pts[k], pts[k + 1]]) for k in range(n)]


def test_chain_smooth_joins_a_flattened_arc():
    pieces = _arc(0, 0, 50, 0, math.pi / 2, 90)
    out = chain_smooth(pieces)
    assert len(out) == 1
    assert math.isclose(out[0].length, _total(pieces))


def test_chain_smooth_keeps_corners_and_junctions():
    square = [LineString(s) for s in [((0, 0), (10, 0)), ((10, 0), (10, 10)), ((10, 10), (0, 10)), ((0, 10), (0, 0))]]
    assert len(chain_smooth(square)) == 4  # 90 degree turns are corners, not curves
    t = [LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)]), LineString([(10, 0), (10, 10)])]
    assert len(chain_smooth(t)) == 3  # three pieces meet: a junction, never joined


def test_chain_smooth_closes_a_circle_and_respects_gaps():
    circle = _arc(0, 0, 20, 0, 2 * math.pi, 72)
    out = chain_smooth(circle)
    assert len(out) == 1 and out[0].is_closed
    apart = [LineString([(0, 0), (10, 0)]), LineString([(10.5, 0), (20, 0)])]
    assert len(chain_smooth(apart)) == 2  # 0.5 pt gap: not the same point


@settings(max_examples=50, deadline=None)
@given(st.integers(3, 120), st.floats(1, 300), st.floats(0.1, 2 * math.pi))
def test_chain_smooth_never_changes_the_geometry(n, r, sweep):
    pieces = _arc(400, 400, r, 0.3, 0.3 + sweep, n)
    out = chain_smooth(pieces)
    assert math.isclose(_total(out), _total(pieces), rel_tol=1e-9)
    assert unary_union(out).hausdorff_distance(unary_union(pieces)) < 1e-6
