import itertools
import math

import pymupdf
import pytest

from floorplan_parser.model import ExtractionError
from floorplan_parser.vector import extract_vector


def _pdf(tmp_path, draw, rotate=0, name="t.pdf"):
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=800)
    draw(page)
    if rotate:
        page.set_rotation(rotate)
    path = tmp_path / name
    doc.save(path)
    return path


def _total(ex):
    return sum(e.length for e in ex.elements)


def test_horizontal_vertical_diagonal(tmp_path):
    def draw(p):
        p.draw_line((100, 100), (500, 100))
        p.draw_line((100, 100), (100, 400))
        p.draw_line((100, 400), (400, 700))

    ex = extract_vector(_pdf(tmp_path, draw))
    assert len(ex.elements) == 3
    assert math.isclose(_total(ex), 400 + 300 + 300 * math.sqrt(2), rel_tol=1e-6)
    assert ex.coordinate_system == "pdf_points" and ex.page == 1 and ex.extractor == "vector"
    assert ex.page_size == (800.0, 800.0)


def test_curve_tessellated_length(tmp_path):
    r, k = 200.0, 0.5522847498

    def draw(p):
        p.draw_bezier((100 + r, 100), (100 + r, 100 + r * k), (100 + r * k, 100 + r), (100, 100 + r))

    ex = extract_vector(_pdf(tmp_path, draw))
    assert math.isclose(_total(ex), math.pi * r / 2, rel_tol=0.005)


def test_rect_gives_four_lines(tmp_path):
    ex = extract_vector(_pdf(tmp_path, lambda p: p.draw_rect(pymupdf.Rect(100, 100, 300, 200))))
    assert len(ex.elements) == 4
    assert math.isclose(_total(ex), 600.0)


def test_empty_page_returns_empty(tmp_path):
    ex = extract_vector(_pdf(tmp_path, lambda p: None))
    assert ex.elements == []
    assert ex.warnings


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 garbage")
    with pytest.raises(ExtractionError):
        extract_vector(bad)


def test_missing_page_raises(tmp_path):
    with pytest.raises(ExtractionError):
        extract_vector(_pdf(tmp_path, lambda p: p.draw_line((0, 0), (10, 10))), page=2)


def test_zero_length_and_duplicates_removed(tmp_path):
    def draw(p):
        p.draw_line((100, 100), (500, 100))
        p.draw_line((500, 100), (100, 100))
        p.draw_line((300, 300), (300, 300))

    ex = extract_vector(_pdf(tmp_path, draw))
    assert len(ex.elements) == 1


def test_image_only_pdf_returns_empty(tmp_path, capsys):
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 100, 100), False)
    pix.clear_with(255)
    ex = extract_vector(_pdf(tmp_path, lambda p: p.insert_image(p.rect, pixmap=pix)))
    assert ex.elements == []
    assert "no vector paths" in capsys.readouterr().err


def test_rotated_page_coordinates(tmp_path):
    ex = extract_vector(_pdf(tmp_path, lambda p: p.draw_line((100, 200), (500, 200)), rotate=90))
    (x0, y0), (x1, y1) = sorted(ex.elements[0].geometry)
    assert (round(x0), round(y0), round(x1), round(y1)) == (100, 200, 500, 200)


def test_chain_joins_a_curve_without_changing_the_linework(tmp_path):
    arc = [(300 + 100 * math.cos(math.pi * k / 120), 300 + 100 * math.sin(math.pi * k / 120)) for k in range(61)]

    def draw(p):
        for a, b in itertools.pairwise(arc):  # a quarter circle stored as 60 short lines, as CAD exports often do
            p.draw_line(a, b)
        p.draw_line(arc[-1], (arc[-1][0], arc[-1][1] + 150))  # meets the arc at a right angle: stays separate

    pdf = _pdf(tmp_path, draw)
    raw, joined = extract_vector(pdf), extract_vector(pdf, chain=True)
    assert len(raw.elements) == 61 and len(joined.elements) == 2
    assert math.isclose(sum(e.length for e in joined.elements), sum(e.length for e in raw.elements), rel_tol=1e-9)
