"""Make the two inputs of the comparison from one source drawing: a vector PDF and image-only PDFs of the same page.

Labels and text are stripped first, so neither extractor ever sees the answer key.
"""

import re
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf

Transform = Callable[[np.ndarray], np.ndarray]


def clean_svg(svg_text: str, page_pt: float = 800.0) -> str:
    """Remove labels and text, draw black on white, and size the page to `page_pt` points."""
    s = re.sub(r'\s(?:semantic-id|instance-id)="[^"]*"', "", svg_text)
    s = re.sub(r"<text\b.*?</text>", "", s, flags=re.DOTALL)
    s = re.sub(r"<text\b[^>]*/>", "", s)
    s = re.sub(r"background-color:\s*#[0-9a-fA-F]+;?", "background-color: #fff;", s)
    s = re.sub(r'stroke="(?!none)[^"]*"', 'stroke="rgb(0,0,0)"', s)
    s = re.sub(r'fill="(?!none)[^"]*"', 'fill="rgb(0,0,0)"', s)
    s = re.sub(r'\s(?:width|height)="[^"]*"(?=[^>]*viewBox)', "", s, count=2)
    return re.sub(r"<svg\b", f'<svg width="{page_pt:g}" height="{page_pt:g}"', s, count=1)


def svg_to_vector_pdf(svg_text: str, out: Path) -> Path:
    with pymupdf.open(stream=svg_text.encode(), filetype="svg") as src:
        pdf_bytes = src.convert_to_pdf()
    with pymupdf.open("pdf", pdf_bytes) as doc:
        doc.save(out)
    return out


def rasterize_pdf(vector_pdf: Path, dpi: int, out: Path, transform: Transform | None = None) -> Path:
    """Render page 1 at `dpi` in grayscale and write it as the only content of a same-size PDF page."""
    with pymupdf.open(vector_pdf) as src:
        page = src[0]
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        rect = page.rect
    if transform is not None:
        img = transform(img)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    with pymupdf.open() as doc:
        doc.new_page(width=rect.width, height=rect.height).insert_image(rect, stream=png.tobytes())
        doc.save(out)
    return out


def degrade_rotate(deg: float) -> Transform:
    """Rotate the rendered page about its centre (a slightly skewed scan), keeping its size."""

    def f(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

    return f


def degrade_scan(noise_sigma: float = 8.0, blur_px: int = 1, seed: int = 0) -> Transform:
    """Gaussian blur plus pixel noise, a rough stand-in for a scanned sheet. Seeded, so runs repeat."""

    def f(img: np.ndarray) -> np.ndarray:
        k = 2 * blur_px + 1
        out = cv2.GaussianBlur(img, (k, k), 0).astype(np.float32)
        out += np.random.default_rng(seed).normal(0, noise_sigma, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    return f
