import pymupdf
import pytest
from shapely.geometry import LineString

from floorplan_parser.metrics import score
from floorplan_parser.prep import rasterize_pdf
from floorplan_parser.raster import RasterParams, extract_raster

AXIS = [((100, 100), (700, 100)), ((100, 100), (100, 700)), ((100, 700), (700, 700)), ((400, 100), (400, 500))]
DIAG = [((100, 100), (600, 500)), ((150, 650), (650, 250))]


def _vector_pdf(tmp_path, segs, name="v.pdf", width=0.8):
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=800)
    for a, b in segs:
        page.draw_line(a, b, width=width)
    path = tmp_path / name
    doc.save(path)
    return path


def _score(ex, segs):
    return score(ex.lines(), [LineString(s) for s in segs])


@pytest.fixture
def axis_raster(tmp_path):
    return rasterize_pdf(_vector_pdf(tmp_path, AXIS), 300, tmp_path / "r.pdf")


def test_recovers_axis_aligned_lines(axis_raster):
    ex = extract_raster(axis_raster)
    assert ex.extractor == "raster" and ex.coordinate_system == "pdf_points"
    assert _score(ex, AXIS).f1 >= 0.95


def test_diagonal_recovered(tmp_path):
    ras = rasterize_pdf(_vector_pdf(tmp_path, DIAG), 300, tmp_path / "r.pdf")
    assert _score(extract_raster(ras), DIAG).f1 >= 0.9


def test_vector_pdf_rendered_at_dpi(tmp_path):
    ex = extract_raster(_vector_pdf(tmp_path, AXIS), dpi=150)
    assert _score(ex, AXIS).f1 >= 0.9


def test_border_line_recovered(tmp_path):
    segs = [((0, 0), (800, 0)), ((0, 400), (800, 400))]
    ras = rasterize_pdf(_vector_pdf(tmp_path, segs, width=1.6), 300, tmp_path / "r.pdf")
    ex = extract_raster(ras)
    assert score(ex.lines(), [LineString(segs[0])]).recall >= 0.8
    _assert_in_bounds(ex)


def test_coords_in_page_bounds(axis_raster):
    _assert_in_bounds(extract_raster(axis_raster))


def _assert_in_bounds(ex):
    w, h = ex.page_size
    assert (w, h) == (800.0, 800.0)
    for e in ex.elements:
        for x, y in e.geometry:
            assert -1e-6 <= x <= w + 1e-6 and -1e-6 <= y <= h + 1e-6


def test_params_roundtrip():
    p = RasterParams(threshold=100, hough_votes_pt=4.0)
    assert RasterParams.from_dict(p.to_dict()) == p


def test_blank_page_returns_empty(tmp_path):
    ras = rasterize_pdf(_vector_pdf(tmp_path, []), 150, tmp_path / "r.pdf")
    assert extract_raster(ras).elements == []


def test_embedded_image_used_at_its_own_resolution(tmp_path):
    from floorplan_parser.raster import _page_image

    ras = rasterize_pdf(_vector_pdf(tmp_path, AXIS), 150, tmp_path / "r.pdf")
    with pymupdf.open(ras) as doc:
        img, rect = _page_image(doc[0], dpi=600)
    assert img.shape == (round(800 / 72 * 150), round(800 / 72 * 150))
    assert rect == pymupdf.Rect(0, 0, 800, 800)


def test_rotated_page_coordinates_match_vector(tmp_path):
    from floorplan_parser.vector import extract_vector

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.draw_line((100, 200), (500, 200), width=1.0)
    page.set_rotation(90)
    path = tmp_path / "rot.pdf"
    doc.save(path)
    ras, vec = extract_raster(path, dpi=300), extract_vector(path)
    assert ras.page_size == vec.page_size == (600.0, 800.0)
    assert score(ras.lines(), vec.lines()).f1 >= 0.95


def test_robust_to_two_degree_rotation(tmp_path):
    import cv2

    from floorplan_parser.geometry import affine
    from floorplan_parser.prep import degrade_rotate

    ras = rasterize_pdf(_vector_pdf(tmp_path, AXIS), 300, tmp_path / "r.pdf", transform=degrade_rotate(2.0))
    m = cv2.getRotationMatrix2D((400, 400), 2.0, 1.0)
    gt = affine([LineString(s) for s in AXIS], (m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2]))
    assert score(extract_raster(ras).lines(), gt).f1 >= 0.9


def test_robust_to_scan_noise(tmp_path):
    from floorplan_parser.prep import degrade_scan

    ras = rasterize_pdf(_vector_pdf(tmp_path, AXIS), 300, tmp_path / "r.pdf", transform=degrade_scan())
    assert _score(extract_raster(ras), AXIS).f1 >= 0.95
