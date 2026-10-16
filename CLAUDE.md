# floorplan-parser

Construction drawing PDF -> line geometry (JSON / CSV / overlay), two ways: read the vector paths (PyMuPDF) or
render the page and find lines with OpenCV. Plus a benchmark against FloorPlanCAD SVG ground truth.

## Stack
- Python >= 3.11 (dev on 3.12), uv. PyMuPDF (`import pymupdf`), opencv-python-headless, shapely 2, svgelements.
- Tests: pytest + hypothesis.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q` (single file: `uv run pytest tests/test_metrics.py -v`)
- Format / lint: `uvx ruff format . && uvx ruff check .`
- CLI: `uv run python -m floorplan_parser --input a.pdf --method vector --output out/a_vector`
- Samples: `uv run python eval/fetch_samples.py --archive ~/Downloads/test-00.tar.xz`

## Layout
- `floorplan_parser/`: the package (model, geometry, metrics, prep, groundtruth, vector, raster, overlay, CLI).
- `eval/`: synthetic drawings, prep, tuning, benchmark, failure notes. `runs/`: committed JSON/CSV results only.
- `space/`: static Hugging Face Space (synthetic drawings only).

## Forbidden
- Never commit FloorPlanCAD files or anything rendered from them (`samples/`, `runs/inputs/`, images under `runs/`).
  The annotations are CC BY-NC 4.0 and the dataset authors do not own the drawings. Public images use synthetic
  drawings only.
- Raster parameters are tuned on synthetic drawings only (`eval/tune.py` must not read `samples/`).
- Labels (`semantic-id`, `instance-id`) and `<text>` never reach an extractor.
