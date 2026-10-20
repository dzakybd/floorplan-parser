"""Pack synthetic overlays and the benchmark numbers into space/data for the Hugging Face Space.

Only synthetic drawings are pictured. FloorPlanCAD results appear as numbers only; no FloorPlanCAD drawing or
anything rendered from one is copied.

Usage: uv run python space/build.py   (after python -m eval.benchmark)
"""

import csv
import json
import shutil
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
SETTINGS = ["vector", "vector-chained", "raster-150", "raster-300", "raster-600"]
# the lines -> walls -> rooms figure is drawn by eval.figures into the portfolio repo next to this one
ROOMS_FIGURE = REPO.parent / "dzakyfawwaz" / "content" / "floorplan-parser" / "media" / "floorplan-parser-rooms.png"
TITLES = {
    "synth-1-walls": "Walls and door gaps",
    "synth-2-diagonal": "Diagonal walls",
    "synth-3-doors": "Door swings and small circles",
    "synth-4-stair": "Stair treads and hatch",
    "synth-5-near-parallel": "Near-parallel lines",
    "synth-text-paths": "Label drawn as lines",
}
WEB = 1100


def main() -> None:
    data = REPO / "space" / "data"
    shutil.rmtree(data, ignore_errors=True)
    metrics = json.loads((RUNS / "metrics.json").read_text())
    drawings = []
    for d, title in TITLES.items():
        assert d.startswith("synth-"), d
        (data / d).mkdir(parents=True)
        for s in SETTINGS:
            Image.open(RUNS / "overlays" / d / f"{s}.png").convert("RGB").resize((WEB, WEB), Image.LANCZOS).save(
                data / d / f"{s}.png", optimize=True
            )
        drawings.append({"id": d, "title": title, "scores": {s: metrics[d][s] for s in SETTINGS}})
    held_out = [
        {"drawing": d, "gt_lines": m["gt_lines"], **{s: m[s] for s in SETTINGS}}
        for d, m in metrics.items()
        if m["split"] == "held-out"
    ]
    out = {
        "drawings": drawings,
        "held_out": held_out,
        "robustness": list(csv.DictReader((RUNS / "robustness.csv").open())),
        "text_paths": json.loads((RUNS / "text_paths.json").read_text()),
        "rooms": json.loads((RUNS / "rooms.json").read_text())["summary"],
        "config": json.loads((RUNS / "experiment_config.json").read_text()),
    }
    (data / "results.json").write_text(json.dumps(out, indent=1))
    shutil.copy(ROOMS_FIGURE, data / "rooms.png")
    print(f"{len(drawings)} synthetic drawings x {len(SETTINGS)} settings -> {data}")


if __name__ == "__main__":
    main()
