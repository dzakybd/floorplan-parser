"""Raster route: treat the page as pixels and find straight lines with OpenCV (threshold, close, Hough).

Lengths in the parameters are in PDF points and converted to pixels for the page's resolution, so one tuned
setting means the same physical sizes at 150, 300 or 600 DPI.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from shapely.geometry import LineString

from floorplan_parser.geometry import dedupe, merge_collinear
from floorplan_parser.model import Extraction, elements_from_lines
from floorplan_parser.vector import open_page


@dataclass(frozen=True)
class RasterParams:
    threshold: int = 128  # gray level below which a pixel is ink
    close_px: int = 1  # morphological close radius, bridges 1-2 px breaks
    hough_votes_pt: float = 6.0  # Hough accumulator threshold, as a line length in pt
    min_line_pt: float = 3.0  # shortest line kept
    max_gap_pt: float = 1.0  # gap Hough may bridge inside one line
    merge_gap_pt: float = 2.0  # gap bridged when merging collinear fragments afterwards

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RasterParams":
        return cls(**d)


def _page_image(page: pymupdf.Page, dpi: int) -> tuple[np.ndarray, pymupdf.Rect]:
    """The page as a grayscale array plus the page rect it covers. A page that is one full-page image is decoded at
    its own resolution; anything else is rendered at `dpi`."""
    rect = page.rect
    images = page.get_images(full=True)
    if len(images) == 1 and not page.get_drawings():
        xref = images[0][0]
        rects = page.get_image_rects(xref)
        if len(rects) == 1 and abs(rects[0] & rect) >= 0.99 * abs(rect):
            pix = pymupdf.Pixmap(page.parent, xref)
            if pix.n - pix.alpha != 1:
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, 0]
            return img.copy(), rects[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy(), rect


def detect_lines(img: np.ndarray, px_per_pt: float, params: RasterParams) -> np.ndarray:
    """Hough segments in pixel coordinates, shape (N, 4) as x0, y0, x1, y1."""
    ink = (img < params.threshold).astype(np.uint8) * 255
    if params.close_px > 0:
        k = 2 * params.close_px + 1
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    segs = cv2.HoughLinesP(
        ink,
        rho=1,
        theta=np.pi / 720,
        threshold=max(5, round(params.hough_votes_pt * px_per_pt)),
        minLineLength=max(2, round(params.min_line_pt * px_per_pt)),
        maxLineGap=max(1, round(params.max_gap_pt * px_per_pt)),
    )
    return np.zeros((0, 4)) if segs is None else np.asarray(segs, float).reshape(-1, 4)


def extract_raster(
    pdf: Path, page: int = 1, dpi: int = 300, params: RasterParams | None = None, drawing_id: str | None = None
) -> Extraction:
    params = params or RasterParams()
    doc, pg = open_page(pdf, page)
    with doc:
        img, rect = _page_image(pg, dpi)
        size = (float(pg.rect.width), float(pg.rect.height))
    t0 = time.perf_counter()
    sx, sy = rect.width / img.shape[1], rect.height / img.shape[0]
    segs = detect_lines(img, 1 / sx, params)
    xs = np.clip(rect.x0 + (segs[:, [0, 2]] + 0.5) * sx, 0, size[0])
    ys = np.clip(rect.y0 + (segs[:, [1, 3]] + 0.5) * sy, 0, size[1])
    lines = [LineString([(x[0], y[0]), (x[1], y[1])]) for x, y in zip(xs, ys)]
    lines = dedupe(lines, tol=max(0.5, 1.5 * sx))
    lines = merge_collinear(lines, dist_tol=max(0.75, 2 * sx), angle_tol_deg=2.0, gap=params.merge_gap_pt)
    lines = dedupe(lines, tol=max(0.5, 1.5 * sx))
    seconds = time.perf_counter() - t0
    return Extraction(
        drawing_id=drawing_id or pdf.stem,
        page=page,
        extractor="raster",
        elements=elements_from_lines(lines, "raster"),
        page_size=size,
        seconds=seconds,
    )
