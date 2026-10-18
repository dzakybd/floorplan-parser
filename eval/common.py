"""Paths and drawing lists shared by the eval scripts."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "samples"
RUNS = REPO / "runs"
INPUTS = RUNS / "inputs"
SYNTH_SRC = INPUTS / "synth"

PAGE_PT = 800.0
DPIS = (150, 300, 600)
# Held out: never used for tuning. Picked by SVG file size (simple / medium / dense) from the FloorPlanCAD test set.
REAL = json.loads((RUNS / "sample_manifest.json").read_text())["held_out"]
TUNE = ["synth-1-walls", "synth-2-diagonal", "synth-3-doors", "synth-4-stair", "synth-5-near-parallel"]
TEXT = "synth-text-paths"
SYNTH = [*TUNE, TEXT]
ROTATE_DEG = 2.0


def source_svg(drawing: str) -> Path:
    """The SVG a drawing's inputs are made from."""
    return SAMPLES / f"{drawing}.svg" if drawing in REAL else SYNTH_SRC / f"{drawing}.svg"


def truth_svg(drawing: str) -> Path:
    """The SVG its ground truth comes from (the text page scores against its glyph-free twin)."""
    gt = SYNTH_SRC / f"{drawing}.gt.svg"
    return gt if gt.exists() else source_svg(drawing)
