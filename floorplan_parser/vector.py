"""Vector route: read the line geometry the PDF already stores, with PyMuPDF `page.get_drawings()`."""

import sys
import time
from pathlib import Path

import pymupdf
from shapely.geometry import LineString

from floorplan_parser.geometry import dedupe, tessellate_bezier
from floorplan_parser.metrics import TESSELLATION_TOL
from floorplan_parser.model import Extraction, ExtractionError, elements_from_lines


def open_page(pdf: Path, page: int) -> tuple[pymupdf.Document, pymupdf.Page]:
    """Open `pdf` and return page `page` (1-based), raising ExtractionError for anything unreadable."""
    try:
        doc = pymupdf.open(pdf)
    except Exception as e:
        raise ExtractionError(f"cannot open {pdf}: {e}") from e
    if not doc.is_pdf or doc.page_count == 0:
        doc.close()
        raise ExtractionError(f"{pdf} is not a readable PDF")
    if not 1 <= page <= doc.page_count:
        n = doc.page_count
        doc.close()
        raise ExtractionError(f"{pdf} has {n} page(s); page {page} requested")
    return doc, doc[page - 1]


def _item_lines(item: tuple) -> list[list[tuple[float, float]]]:
    kind = item[0]
    if kind == "l":
        return [[tuple(item[1]), tuple(item[2])]]
    if kind == "c":
        return [tessellate_bezier(*(tuple(p) for p in item[1:5]), tol=TESSELLATION_TOL)]
    if kind == "re":
        r = item[1]
        corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    elif kind == "qu":
        q = item[1]
        corners = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
    else:
        return []
    return [[corners[i], corners[(i + 1) % 4]] for i in range(4)]


def extract_vector(pdf: Path, page: int = 1, drawing_id: str | None = None) -> Extraction:
    t0 = time.perf_counter()
    doc, pg = open_page(pdf, page)
    with doc:
        paths = pg.get_drawings()
        read_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        lines = [
            LineString(pts) for path in paths for item in path["items"] for pts in _item_lines(item) if len(pts) > 1
        ]
        extract_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        lines = dedupe(lines)
        normalize_s = time.perf_counter() - t0
        seconds = read_s + extract_s + normalize_s
        size = (float(pg.cropbox.width), float(pg.cropbox.height))
    warnings = []
    if not lines:
        warnings.append(f"no vector paths on page {page} of {pdf.name}; is it a scanned or image-only PDF?")
        print(f"warning: {warnings[0]}", file=sys.stderr)
    return Extraction(
        drawing_id=drawing_id or pdf.stem,
        page=page,
        extractor="vector",
        elements=elements_from_lines(lines, "vector"),
        page_size=size,
        seconds=seconds,
        warnings=warnings,
        timing_s={"read_s": read_s, "extract_s": extract_s, "normalize_s": normalize_s},
    )
