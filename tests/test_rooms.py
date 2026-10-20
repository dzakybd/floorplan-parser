import itertools
import math

from shapely.geometry import LineString, Polygon, box

from floorplan_parser.rooms import (
    RoomParams,
    close_openings,
    detect_walls,
    extract_rooms,
    wall_thickness,
    wall_thicknesses,
)

T = 8.0  # wall thickness in pt


def _outline(poly: Polygon) -> list[LineString]:
    """A wall polygon drawn the CAD way: its outline as separate straight lines."""
    out = []
    for ring in [poly.exterior, *poly.interiors]:
        c = list(ring.coords)
        out += [LineString([a, b]) for a, b in itertools.pairwise(c)]
    return out


def _two_room_plan(door: float = 40.0) -> tuple[list[LineString], list[Polygon]]:
    """Two rooms side by side inside a 400 x 200 pt building, a dividing wall with a doorway of width `door`."""
    outer = box(100, 100, 500 + T, 300 + T).difference(box(100 + T, 100 + T, 500, 300))
    divider = box(300, 100 + T, 300 + T, 300).difference(box(300, 180, 300 + T, 180 + door))
    walls = outer.union(divider)
    rooms = [box(100 + T, 100 + T, 300, 300), box(300 + T, 100 + T, 500, 300)]
    return _outline(walls), rooms


def test_wall_thickness_is_the_most_common_parallel_gap():
    lines, _ = _two_room_plan()
    furniture = [LineString([(150, 150), (250, 150)]), LineString([(150, 190), (250, 190)])]  # 40 pt apart
    assert math.isclose(wall_thickness(lines + furniture), T, abs_tol=0.3)


def test_walls_cover_the_drawn_wall_band():
    lines, _ = _two_room_plan()
    walls = detect_walls(lines, T)
    assert walls.contains(LineString([(100 + T / 2, 120), (100 + T / 2, 280)]))  # inside the left outer wall
    assert not walls.intersects(box(150, 150, 250, 250))  # not in a room


def test_openings_close_between_collinear_wall_pieces_only():
    lines, _ = _two_room_plan(door=40)
    walls = close_openings(lines, T, RoomParams(max_gap_factor=10))
    assert walls.contains(LineString([(300 + T / 2, 185), (300 + T / 2, 215)]))  # the doorway is closed
    assert not walls.intersects(box(150, 150, 250, 250))


def test_rooms_come_back_with_their_areas():
    lines, truth = _two_room_plan()
    rooms = extract_rooms(lines, (800, 800), RoomParams())
    assert len(rooms) == 2
    for gt in truth:
        best = max(rooms, key=lambda r: r.intersection(gt).area)
        assert best.intersection(gt).area / best.union(gt).area > 0.97


def test_no_walls_means_no_rooms():
    assert extract_rooms([LineString([(0, 0), (100, 0)])], (800, 800), RoomParams()) == []


def test_two_wall_thicknesses_are_both_found_when_asked():
    lines, _ = _two_room_plan()  # 8 pt walls
    thin = box(600, 100, 603, 300).union(box(603, 100, 700, 103))  # an L of 3 pt walls elsewhere on the page
    lines += _outline(thin)
    one = wall_thicknesses(lines, RoomParams(n_peaks=1))
    two = wall_thicknesses(lines, RoomParams(n_peaks=2, peak_share=0.1))
    assert len(one) == 1 and math.isclose(one[0], T, abs_tol=0.3)
    assert len(two) == 2 and any(math.isclose(t, 3, abs_tol=0.3) for t in two)
    walls = detect_walls(lines, two)
    assert walls.contains(LineString([(601.5, 150), (601.5, 250)]))  # the thin wall is a wall too
