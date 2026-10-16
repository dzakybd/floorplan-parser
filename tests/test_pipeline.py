import json
import subprocess
import sys

import pymupdf
import pytest
from shapely.geometry import LineString

from floorplan_parser.model import Extraction
from floorplan_parser.overlay import draw_overlay
from floorplan_parser.prep import rasterize_pdf
from floorplan_parser.vector import extract_vector


@pytest.fixture
def vector_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=800)
    page.draw_line((100, 100), (700, 100), width=0.8)
    page.draw_line((100, 100), (100, 700), width=0.8)
    path = tmp_path / "a.pdf"
    doc.save(path)
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "floorplan_parser", *map(str, args)], check=False, capture_output=True, text=True
    )


def _check_outputs(out, extractor):
    for name in ("elements.json", "elements.csv", "overlay.png"):
        assert (out / name).exists(), name
    ex = Extraction.from_json(out / "elements.json")
    data = json.loads((out / "elements.json").read_text())
    assert data["coordinate_system"] == "pdf_points" and data["page"] == 1 and data["extractor"] == extractor
    assert ex.elements
    for e in ex.elements:
        assert all(0 <= x <= 800 and 0 <= y <= 800 for x, y in e.geometry)


def test_cli_vector_writes_three_files(tmp_path, vector_pdf):
    r = _run("--input", vector_pdf, "--method", "vector", "--output", tmp_path / "out_v")
    assert r.returncode == 0, r.stderr
    _check_outputs(tmp_path / "out_v", "vector")


def test_cli_raster_writes_three_files(tmp_path, vector_pdf):
    ras = rasterize_pdf(vector_pdf, 300, tmp_path / "a_raster.pdf")
    r = _run("--input", ras, "--method", "raster", "--dpi", 300, "--output", tmp_path / "out_r")
    assert r.returncode == 0, r.stderr
    _check_outputs(tmp_path / "out_r", "raster")


def test_cli_bad_file_exit_1(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    r = _run("--input", bad, "--method", "vector", "--output", tmp_path / "o")
    assert r.returncode == 1
    assert len(r.stderr.strip().splitlines()) == 1


def test_overlay_with_ground_truth(tmp_path, vector_pdf):
    ex = extract_vector(vector_pdf)
    gt = [LineString([(100, 100), (700, 100)]), LineString([(400, 400), (500, 500)])]
    out = draw_overlay(vector_pdf, 1, ex, tmp_path / "o.png", gt=gt, scale=1.0)
    pix = pymupdf.Pixmap(str(out))
    assert (pix.width, pix.height) == (800, 800)


def test_overlay_aligned_on_rotated_page(tmp_path):
    import numpy as np

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.set_rotation(90)
    path = tmp_path / "rot.pdf"
    doc.save(path)
    from floorplan_parser.model import Element

    ex = Extraction(
        "r", 1, "vector", [Element("ln_1", "line", [(100.0, 200.0), (500.0, 200.0)], "vector", (100, 200, 500, 200))]
    )
    out = draw_overlay(path, 1, ex, tmp_path / "o.png", scale=1.0)
    pix = pymupdf.Pixmap(str(out))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    # unrotated (300, 200) shows at (displayed) point (600 - 200, 300) = (600, 300) on the 800 x 600 render
    p = pymupdf.Point(300, 200) * doc[0].rotation_matrix
    x, y = round(p.x), round(p.y)
    assert img[y, min(x, pix.width - 1) - 1].tolist()[:3] != [255, 255, 255]
