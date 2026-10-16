from pathlib import Path

import pymupdf

from floorplan_parser.prep import clean_svg, degrade_rotate, degrade_scan, rasterize_pdf, svg_to_vector_pdf

TINY = Path(__file__).parent / "fixtures" / "tiny.svg"


def test_clean_svg_strips_labels_and_text():
    out = clean_svg(TINY.read_text())
    assert "semantic-id" not in out and "instance-id" not in out
    assert "<text" not in out
    assert 'width="800"' in out and 'height="800"' in out
    assert "rgb(255,0,0)" not in out


def test_vector_pdf_page_is_800pt(tmp_path):
    pdf = svg_to_vector_pdf(clean_svg(TINY.read_text()), tmp_path / "v.pdf")
    with pymupdf.open(pdf) as doc:
        assert doc[0].rect == pymupdf.Rect(0, 0, 800, 800)
        assert doc[0].get_drawings()


def test_raster_pdf_is_image_only(tmp_path):
    vec = svg_to_vector_pdf(clean_svg(TINY.read_text()), tmp_path / "v.pdf")
    ras = rasterize_pdf(vec, 150, tmp_path / "r.pdf")
    with pymupdf.open(ras) as doc:
        page = doc[0]
        assert page.get_drawings() == []
        imgs = page.get_images(full=True)
        assert len(imgs) == 1
        assert imgs[0][2] == round(800 / 72 * 150)


def test_raster_page_same_size_as_vector(tmp_path):
    vec = svg_to_vector_pdf(clean_svg(TINY.read_text()), tmp_path / "v.pdf")
    ras = rasterize_pdf(vec, 300, tmp_path / "r.pdf")
    with pymupdf.open(ras) as doc:
        assert doc[0].rect == pymupdf.Rect(0, 0, 800, 800)


def test_degradations_keep_image_shape(tmp_path):
    vec = svg_to_vector_pdf(clean_svg(TINY.read_text()), tmp_path / "v.pdf")
    for i, t in enumerate([degrade_rotate(2.0), degrade_scan()]):
        ras = rasterize_pdf(vec, 150, tmp_path / f"d{i}.pdf", transform=t)
        with pymupdf.open(ras) as doc:
            assert doc[0].get_images(full=True)[0][2] == round(800 / 72 * 150)
