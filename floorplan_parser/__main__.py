"""CLI: python -m floorplan_parser --input a.pdf --method vector|raster --output out/a [--chain] [--dpi 300] [--page 1]"""

import argparse
import json
import sys
from pathlib import Path

from floorplan_parser.model import ExtractionError
from floorplan_parser.overlay import draw_overlay
from floorplan_parser.raster import RasterParams, extract_raster
from floorplan_parser.vector import extract_vector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="floorplan_parser", description="Extract line geometry from a drawing PDF.")
    ap.add_argument("--input", type=Path, required=True, help="PDF file")
    ap.add_argument("--method", choices=["vector", "raster"], required=True)
    ap.add_argument("--output", type=Path, required=True, help="directory for elements.json, elements.csv, overlay.png")
    ap.add_argument("--page", type=int, default=1, help="1-based page number")
    ap.add_argument(
        "--chain", action="store_true", help="vector: join the pieces of each smooth curve into one polyline"
    )
    ap.add_argument("--dpi", type=int, default=300, help="raster: render resolution for pages that are not one image")
    ap.add_argument("--params", type=Path, help="raster: JSON file of RasterParams fields")
    ap.add_argument("--coordinate-system", choices=["pdf_points"], default="pdf_points")
    args = ap.parse_args(argv)
    try:
        if args.method == "vector":
            ex = extract_vector(args.input, page=args.page, chain=args.chain)
        else:
            params = RasterParams.from_dict(json.loads(args.params.read_text())) if args.params else None
            ex = extract_raster(args.input, page=args.page, dpi=args.dpi, params=params)
        args.output.mkdir(parents=True, exist_ok=True)
        ex.to_json(args.output / "elements.json")
        ex.to_csv(args.output / "elements.csv")
        draw_overlay(args.input, args.page, ex, args.output / "overlay.png")
    except ExtractionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{len(ex.elements)} {args.method} elements -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
