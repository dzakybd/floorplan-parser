"""Overlay: the page in light grey, extracted lines on top, and (with ground truth) what was missed or extra."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf
from shapely.geometry import LineString

from floorplan_parser.metrics import unmatched_points
from floorplan_parser.model import Extraction

# BGR
EXTRACTED = (180, 119, 31)  # blue
MISSED = (40, 39, 214)  # red
EXTRA = (14, 127, 255)  # orange


def draw_overlay(
    pdf: Path, page: int, extraction: Extraction, out: Path, gt: list[LineString] | None = None, scale: float = 2.0
) -> Path:
    with pymupdf.open(pdf) as doc:
        pg = doc[page - 1]
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
        sx = pix.width / pg.rect.width
        sy = pix.height / pg.rect.height
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    faded = (255 - (255 - gray.astype(np.float32)) * 0.3).astype(np.uint8)
    img = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)
    thick = max(1, round(scale))

    def px(pts):
        return np.round(np.asarray(pts) * [sx, sy]).astype(np.int32)

    for e in extraction.elements:
        cv2.polylines(img, [px(e.geometry)], False, EXTRACTED, thick, cv2.LINE_AA)
    if gt is not None:
        pred = extraction.lines()
        for color, pts in ((MISSED, unmatched_points(gt, pred)), (EXTRA, unmatched_points(pred, gt))):
            for x, y in px(pts) if len(pts) else []:
                cv2.circle(img, (int(x), int(y)), thick + 1, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out), img)
    return out
