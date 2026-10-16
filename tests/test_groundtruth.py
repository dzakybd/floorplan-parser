import math
from pathlib import Path

from floorplan_parser.groundtruth import CLASS_NAMES, load_ground_truth

TINY = Path(__file__).parent / "fixtures" / "tiny.svg"


def test_tiny_svg_counts():
    gt = load_ground_truth(TINY)
    assert len(gt.lines) == 4  # 2 lines, 1 arc, 1 circle; text skipped
    assert gt.page_size == (800.0, 800.0)


def test_coordinates_scaled_to_page():
    gt = load_ground_truth(TINY)
    vertical = [ln for ln in gt.lines if len(ln.coords) == 2 and ln.coords[0][0] == ln.coords[1][0]]
    assert len(vertical) == 1
    assert vertical[0].coords[0][0] == 400.0


def test_circle_length_close_to_circumference():
    gt = load_ground_truth(TINY)
    closed = [ln for ln in gt.lines if ln.coords[0] == ln.coords[-1]]
    assert len(closed) == 1
    assert math.isclose(closed[0].length, 2 * math.pi * 64, rel_tol=0.005)  # r = 8 units * 8 pt


def test_arc_length_close_to_quarter_circle():
    gt = load_ground_truth(TINY)
    arc = [ln for ln, lb in zip(gt.lines, gt.labels) if lb == "single door"]
    assert len(arc) == 1
    assert math.isclose(arc[0].length, math.pi * 80 / 2, rel_tol=0.005)


def test_labels_mapped():
    gt = load_ground_truth(TINY)
    assert sorted(lb or "-" for lb in gt.labels) == ["-", "-", "single door", "wall"]
    assert CLASS_NAMES[33] == "wall" and CLASS_NAMES[35] == "railing"


def test_duplicate_lines_counted_once(tmp_path):
    svg = TINY.read_text().replace(
        "</g>", '<path d="M 90.0,20.0 L 10.0,20.0" fill="none" semantic-id="33" stroke="rgb(1,1,1)"/></g>'
    )
    p = tmp_path / "dup.svg"
    p.write_text(svg)
    assert len(load_ground_truth(p).lines) == 4
