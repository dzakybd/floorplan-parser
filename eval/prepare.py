"""Make every benchmark input from its source SVG: clean.svg, vector.pdf and image-only PDFs.

Synthetic drawings also get a 2-degree rotated and a scan-like (noise + blur) raster at 300 DPI.

Usage: uv run python -m eval.prepare
"""

from eval.common import DPIS, INPUTS, PAGE_PT, REAL, ROTATE_DEG, SYNTH, SYNTH_SRC, TUNE, source_svg
from eval.synth import make_synthetic
from floorplan_parser.prep import clean_svg, degrade_rotate, degrade_scan, rasterize_pdf, svg_to_vector_pdf


def main() -> None:
    make_synthetic(SYNTH_SRC)
    for d in [*REAL, *SYNTH]:
        out = INPUTS / d
        out.mkdir(parents=True, exist_ok=True)
        clean = clean_svg(source_svg(d).read_text(encoding="utf-8"), PAGE_PT)
        (out / "clean.svg").write_text(clean, encoding="utf-8")
        vec = svg_to_vector_pdf(clean, out / "vector.pdf")
        for dpi in DPIS:
            rasterize_pdf(vec, dpi, out / f"raster-{dpi}.pdf")
        if d in TUNE:
            rasterize_pdf(vec, 300, out / "raster-300-rot2.pdf", transform=degrade_rotate(ROTATE_DEG))
            rasterize_pdf(vec, 300, out / "raster-300-scan.pdf", transform=degrade_scan())
        print(d)


if __name__ == "__main__":
    main()
