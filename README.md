# floorplan-parser

Line geometry from construction drawing PDFs, two ways, measured against the drawing's own geometry:

- **vector**: read the paths the PDF already stores (PyMuPDF `page.get_drawings()`).
- **raster**: render the page to pixels and find the lines with classic computer vision (OpenCV threshold, morphological close, probabilistic Hough, then merging of broken fragments). No neural network.

Both write the same output: `elements.json`, `elements.csv` and an `overlay.png`, in PDF points. This recovers **lines, not meaning**: neither route knows a wall from a door.

Demo (synthetic drawings, precomputed): [huggingface.co/spaces/dzakybd/floorplan-parser](https://huggingface.co/spaces/dzakybd/floorplan-parser) · Write-up: [dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster](https://dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster)

## Install and run

```bash
uv sync
uv run python -m floorplan_parser --input plan.pdf --method vector --output out/plan_vector
uv run python -m floorplan_parser --input scan.pdf --method raster --dpi 300 --output out/plan_raster
```

Options: `--page N` (1-based), `--dpi` (raster: render resolution for pages that are not a single image; a page that is one full-page image is read at its own resolution), `--params file.json` (raster parameters, see `runs/params.json`). An unreadable file exits with code 1. A page with no vector paths (a scan) returns no elements and a warning, so use `--method raster` there.

Output (`elements.json`):

```json
{"drawing_id": "plan", "page": 1, "extractor": "vector", "coordinate_system": "pdf_points",
 "page_size": [800.0, 800.0], "seconds": 0.02, "warnings": [],
 "elements": [{"id": "ln_1", "kind": "line", "geometry": [[10.0, 20.0], [110.0, 20.0]],
               "extractor": "vector", "source_bbox": [10.0, 20.0, 110.0, 20.0]}]}
```

## How it is measured

- **Data:**
  - Three FloorPlanCAD drawings from the original test archive, picked by SVG file size: `0674-0822` (simple structural grid), `1010-0022` (medium: doors, stair, close double lines) and `0393-0004` (dense apartment plan).
  - Six synthetic drawings in the same SVG style (`eval/synth.py`).
- **One source, two inputs:**
  - Each SVG is stripped of its labels (`semantic-id`, `instance-id`) and its text, drawn black on white, and scaled to an 800 × 800 pt page.
  - It is converted once to a vector PDF, and that PDF is rendered to image-only PDFs at 150, 300 and 600 DPI.
- **Ground truth:** the SVG's own lines, arcs and circles in the same coordinates.
  - Curves are flattened at 0.1 pt chord error.
  - Duplicate CAD lines are dropped.
  - Labels are used only for per-class recall.
- **Score, by length:**
  - Recall is the share of true linework that lies within 2 pt and 5° of an extracted line; precision is the reverse. A line split into 50 pieces scores the same as the whole line.
  - Pieces under 4 pt, i.e. flattened curves, are matched by distance only.
  - Also reported: median endpoint error, total-length error, piece count, and time. Time is the median of 3 runs of the extract call, Apple M3 Pro; PDF preparation is not timed.
- **No tuning on the test drawings:**
  - Raster parameters were picked by grid search on the five synthetic drawings at 300 DPI and frozen in `runs/params.json`.
  - The FloorPlanCAD drawings were then run once with them.
  - The synthetic set turned out easy: tuned settings score a mean F1 of 0.995 against 0.993 for the defaults.

## Results (held-out FloorPlanCAD drawings)

F1 by length, with the number of extracted pieces in brackets (from `runs/metrics.json`):

| Drawing | True lines | Vector | Raster 150 DPI | Raster 300 DPI | Raster 600 DPI |
|---|---|---|---|---|---|
| 0674-0822 simple | 114 | 1.000 (114) | 0.999 (110) | 1.000 (102) | 1.000 (104) |
| 1010-0022 medium | 229 | 1.000 (667) | 0.878 (1217) | 0.890 (1371) | 0.880 (1675) |
| 0393-0004 dense | 807 | 1.000 (7967) | 0.976 (439) | 0.984 (515) | 0.985 (641) |

What the numbers say:
- **Vector recovers all of the linework, but in pieces.** Every arc comes back as many short straight segments, so 807 true lines become 7,967 pieces on the dense drawing (9.9 per line). Counting "lines" from a vector PDF needs merging first.
- **Raster is close on clean, sparse drawings and weakest on close double lines.** On the medium drawing it adds lines along closely spaced double lines (precision 0.84 at 300 DPI), and more pixels make that worse (length error 2.8% at 150 DPI, 25% at 600 DPI). It also drops small circles and the ends of arcs.
- **Time:** vector takes 5–21 ms on the two simpler drawings against 0.25–0.39 s for raster at 300 DPI. On the dense drawing the two are about even (0.38 s vs 0.40 s), because the vector route has to dedupe 7,967 pieces.
- **Scan robustness (synthetic, 300 DPI, `runs/robustness.csv`):** noise plus blur moves F1 by at most 0.6 points, and a 2° rotation costs 0.3–2.4 points.
- **Text drawn as lines (`runs/text_paths.json`):** both routes pick up a dimension label drawn as line strokes as if it were geometry. Vector precision drops to 0.96.

Case notes: `runs/failure_cases/`. Per-class recall: `runs/per_class.csv`.

## Reproduce

```bash
uv run python -m eval.fetch_samples --archive ~/Downloads/test-00.tar.xz   # from floorplancad.github.io
uv run python -m eval.prepare
uv run python -m eval.tune
uv run python -m eval.benchmark
uv run python -m eval.failures
uv run python space/build.py
uv run pytest -q
```

## Limitations

- Three real drawings. The numbers show how the two routes behave, not how well they do across all drawings.
- The vector PDF is made from the same SVG the score uses, so near-perfect vector numbers are expected by construction. Real CAD-exported PDFs add hatching, text drawn as lines, and clipping.
- Straight lines only on the raster side. Curves come back as short straight pieces.
- Geometry only. Walls, doors and rooms need a separate recognition step.

## Data and licence

Code: MIT.

FloorPlanCAD: Z. Fan, L. Zhu, H. Li, X. Chen, S. Zhu, P. Tan, "FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting", ICCV 2021, https://floorplancad.github.io/. Its annotations are licensed CC BY-NC 4.0, and the dataset authors do not own the underlying drawings.

**No FloorPlanCAD data is included in this repository.** `samples/` and every image rendered from it stay local; only numeric results are committed. Class names follow `config/anno_config.py` in VITA-Group/CADTransformer.
