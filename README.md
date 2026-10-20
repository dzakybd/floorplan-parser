# floorplan-parser

Line geometry from construction drawing PDFs, two ways, measured against the drawing's own geometry:

- **vector**: read the paths the PDF already stores (PyMuPDF `page.get_drawings()`).
- **raster**: render the page to pixels and find the lines with classic computer vision (OpenCV threshold, morphological close, probabilistic Hough, then merging of broken fragments). No neural network.

Both write the same output: `elements.json`, `elements.csv` and an `overlay.png`, in PDF points. This recovers **lines, not meaning**: neither route knows a wall from a door.

Demo (synthetic drawings, precomputed): [huggingface.co/spaces/dzakybd/floorplan-parser](https://huggingface.co/spaces/dzakybd/floorplan-parser) · Write-up: [dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster](https://dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster)

## Install and run

```bash
uv sync
uv run python -m floorplan_parser --input plan.pdf --method vector --chain --rooms --output out/plan_vector
uv run python -m floorplan_parser --input scan.pdf --method raster --dpi 300 --output out/plan_raster
```

Options: `--chain` (vector: join the pieces of each curve into one polyline; see below), `--rooms` (also find walls and rooms; writes `rooms.json` with each room's polygon, area in pt² and perimeter in pt), `--page N` (1-based), `--dpi` (raster: render resolution for pages that are not a single image; a page that is one full-page image is read at its own resolution), `--params file.json` (raster parameters, see `runs/params.json`). An unreadable file exits with code 1. A page with no vector paths (a scan) returns no elements and a warning, so use `--method raster` there.

Output (`elements.json`):

```json
{"drawing_id": "plan", "page": 1, "extractor": "vector", "coordinate_system": "pdf_points",
 "page_size": [800.0, 800.0], "seconds": 0.02, "warnings": [],
 "elements": [{"id": "ln_1", "kind": "line", "geometry": [[10.0, 20.0], [110.0, 20.0]],
               "extractor": "vector", "source_bbox": [10.0, 20.0, 110.0, 20.0]}]}
```

## How it is measured

- **Data:**
  - Ten FloorPlanCAD drawings from the original test archive (`runs/sample_manifest.json`): the first three examples (`0674-0822` simple, `1010-0022` medium, `0393-0004` dense) plus seven drawn with seed 20261019 within SVG-file-size terciles, before any of them were scored.
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
  - Also reported: median endpoint error, total-length error, piece count, and time. Endpoint error is reported with its match rate. Time is the median of 3 runs of the extract call, Apple M1 Pro; PDF preparation is not timed.
- **No tuning on the test drawings:**
  - Raster parameters were picked by grid search on the five synthetic drawings at 300 DPI and frozen in `runs/params.json`.
  - The FloorPlanCAD drawings were run with them unchanged; the expansion to ten drawings kept the same frozen file.
  - The synthetic set turned out easy: tuned settings score a mean F1 of 0.995 against 0.993 for the defaults.

## Results (held-out FloorPlanCAD drawings)

F1 by length, with the number of extracted pieces in brackets (from `runs/metrics.json`):

| Drawing | True lines | Vector | Raster 150 DPI | Raster 300 DPI | Raster 600 DPI |
|---|---|---|---|---|---|
| 0674-0822 | 114 | 1.000 (114) | 0.999 (110) | 1.000 (102) | 1.000 (104) |
| 1010-0022 | 229 | 1.000 (667) | 0.878 (1217) | 0.890 (1371) | 0.880 (1675) |
| 0393-0004 | 807 | 1.000 (7967) | 0.976 (439) | 0.984 (515) | 0.985 (641) |
| 0721-0004 | 123 | 1.000 (1759) | 0.990 (135) | 0.987 (133) | 0.988 (145) |
| 1261-0083 | 91 | 1.000 (91) | 0.999 (76) | 1.000 (66) | 1.000 (68) |
| 0175-0005 | 155 | 1.000 (240) | 0.994 (111) | 0.994 (116) | 0.994 (121) |
| 0515-0016 | 195 | 1.000 (831) | 0.998 (155) | 0.999 (143) | 1.000 (145) |
| 1083-0018 | 135 | 1.000 (135) | 0.999 (80) | 1.000 (75) | 1.000 (74) |
| 0298-0026 | 751 | 1.000 (5859) | 0.973 (459) | 0.977 (499) | 0.982 (668) |
| 0672-0021 | 732 | 1.000 (7066) | 0.978 (522) | 0.983 (568) | 0.985 (647) |

What the numbers say:
- **Vector recovers all of the linework on all ten drawings, but in pieces.** Every arc comes back as many short straight segments: up to 14.3 pieces per true line (`0721-0004`, 1,759 pieces for 123 lines). F1 is by length, so it cannot see this; the structure numbers below can.
- **`--chain` fixes the structure, at a measured cost.** It joins pieces where exactly two meet and the path turns by 0.1° to 30° (curves), never at corners, junctions or exactly straight continuations; the geometry is unchanged. Pieces per reference line: mean 5.34 → 0.93, worst drawing 14.30 → 1.27; reference lines split into 2+ pieces: 10.4% → 2.9%; F1 still 1.000. Cost: elements that cover two or more reference lines rise from 2.6% to 5.5% (the worst drawing's 17.8% is the same as the raw vector route); endpoint coverage 100% → 93.0%. Without the 0.1° floor, joined elements were 10.2%; the floor comes from geometry (a flattened curve always turns), not from these drawings.
- **Raster at 300 DPI averages F1 0.981 (0.890–1.000).** It is close on clean drawings and weakest on close double lines. On the medium drawing it adds lines along closely spaced double lines (precision 0.84 at 300 DPI), and more pixels make that worse (length error 2.4% at 150 DPI, 25.7% at 600 DPI). It also drops small circles and the ends of arcs.
- **Time (median across ten drawings):** vector 26 ms, vector with `--chain` 32 ms, raster at 300 DPI 0.25 s.
- **Lines → walls → rooms (`--rooms`, `floorplan_parser/rooms.py`, Shapely):** wall thickness is the most common gap between overlapping parallel lines; parallel pairs about that far apart become wall bands; gaps between collinear bands up to 14 thicknesses are bridged (doorways); rooms are the closed pieces of page minus walls. Scored by `eval.rooms_eval` (`runs/rooms.json`):
  - *Synthetic plans with a room answer key* (`eval/synth_rooms.py`; room parameters chosen on 10 plans, scored on 20 others): wall thickness within 0.5 pt on 20/20, room F1 mean 0.977 (median 1.000, min 0.870), total area error −4.5% to +1.2%. All imperfect plans are open kitchens found as one room. These plans follow the generator's own conventions.
  - *Real CubiCasa5K layouts* drawn as vector PDFs from their wall, door and room annotations (quantity-lab `eval/export_vector.py`): real walls vary in thickness, and with one thickness the room step found almost nothing (room F1 0.09). The thickness rule (up to 5 histogram peaks, tolerance 0.4) is chosen on ten validation layouts by room F1, among rules that keep the synthetic tune plans (which have furniture) within 0.01 room F1 and 1% total-area error; on the ten test layouts room F1 is **0.55** (median 0.53, range 0.00–0.97) and matched rooms' area error a median 1.6%. The peak count sits at the edge of its grid: the synthetic guard does not constrain it. The walls here are drawn cleanly from annotations, so this tests walls → rooms on real layouts, not reading messy CAD.
  - *FloorPlanCAD drawings with wall labels* (`eval/fetch_walls.py`, `runs/wall_manifest.json`: ten of the 46 test drawings with ≥ 30 wall primitives, seeded): **not scored as accuracy.** The share of wall-labelled linework inside the detected walls has a median 0.78, but on their edges, where wall faces should be, 0.03. Inspection shows the `wall` label marks only some walls (often thin partitions 2–4 pt thick) while thick walls are unlabelled, so both numbers describe the labels (`label_audit` in `runs/rooms.json`). Choosing the thickness rule on these labels had no sound basis; it is chosen on the CubiCasa5K layouts above instead.
- **Scan robustness (synthetic, 300 DPI, `runs/robustness.csv`):** noise plus blur moves F1 by at most 0.6 points, and a 2° rotation costs 0.3–2.4 points.
- **Text drawn as lines (`runs/text_paths.json`):** both routes pick up a dimension label drawn as line strokes as if it were geometry. Vector precision drops to 0.96.

Full distributions (mean, median, min, max) for precision, recall, F1, endpoint error and match rate, length error and stage timings: `runs/summary.json`. Case notes: `runs/failure_cases/`. Per-class recall: `runs/per_class.csv`. Structure (pieces per reference line, reference lines split, elements joining 2+ reference lines; `floorplan_parser.metrics.structure`) is in `runs/metrics.json` per drawing and setting. Verification of every export and an independent rescoring of all 50 held-out settings: `runs/verification.json`.

## Reproduce

```bash
uv run python -m eval.fetch_samples --archive ~/Downloads/test-00.tar.xz   # from floorplancad.github.io
uv run python -m eval.prepare
uv run python -m eval.tune
uv run python -m eval.benchmark
uv run python -m eval.failures
uv run python -m eval.report
uv run python -m eval.verify
uv run python -m eval.fetch_walls --archive ~/Downloads/test-00.tar.xz   # wall-labelled set (list committed)
uv run python -m eval.rooms_eval
uv run python space/build.py
uv run pytest -q
```

## Limitations

- Ten real drawings, three of which were inspected before the set was expanded. The numbers show how the two routes behave, not how well they do across all drawings.
- The vector PDF is made from the same SVG the score uses, so near-perfect vector numbers are expected by construction. Real CAD-exported PDFs add hatching, text drawn as lines, and clipping.
- Matching is many-to-one within 2 pt. An extractor that merges two true lines closer than 2 pt into one still scores F1 1.0; only the length error shows it (50% for one pair). On the near-parallel synthetic drawing the raster length error is at most 0.21%, so no such merge happened there (`tests/test_metrics.py::test_collapsed_pair_shows_in_length_error_not_f1` pins this behaviour).
- Straight lines only on the raster side. Curves come back as short straight pieces.
- Rooms are scored on synthetic plans only: FloorPlanCAD has no room labels.
- No real-drawing score for walls or rooms: FloorPlanCAD labels only some walls and no rooms. Needs vector drawings with wall faces and rooms labelled.
- Geometry only. Walls, doors and rooms need a separate recognition step.

## Data and licence

Code: MIT.

FloorPlanCAD: Z. Fan, L. Zhu, H. Li, X. Chen, S. Zhu, P. Tan, "FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting", ICCV 2021, https://floorplancad.github.io/. Its annotations are licensed CC BY-NC 4.0, and the dataset authors do not own the underlying drawings.

**No FloorPlanCAD data is included in this repository.** `samples/` and every image rendered from it stay local; only numeric results are committed. Class names follow `config/anno_config.py` in VITA-Group/CADTransformer.
