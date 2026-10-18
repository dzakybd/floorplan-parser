"""Run both extractors on every drawing with the frozen raster parameters and score them against ground truth.

Writes runs/metrics.json, per_drawing.csv, per_class.csv, robustness.csv, text_paths.json and
experiment_config.json. Overlays go to runs/overlays/ (local only, gitignored).

Usage: uv run python -m eval.benchmark
"""

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from importlib.metadata import version

import cv2

from eval.common import DPIS, INPUTS, PAGE_PT, REAL, ROTATE_DEG, RUNS, TEXT, TUNE, source_svg, truth_svg
from floorplan_parser.geometry import affine
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import CONSTANTS, recall_by_label, score
from floorplan_parser.overlay import draw_overlay
from floorplan_parser.raster import RasterParams, extract_raster
from floorplan_parser.vector import extract_vector

REPEATS = 3


def timed(fn, *args, **kw):
    """Run an extraction REPEATS times; return the last result and the median wall time in seconds."""
    times, stages = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ex = fn(*args, **kw)
        times.append(time.perf_counter() - t0)
        stages.append(ex.timing_s)
    ex.timing_s = {k: statistics.median(t[k] for t in stages) for k in stages[0]}
    return ex, statistics.median(times)


def settings():
    yield "vector", "vector.pdf", extract_vector, {}
    for dpi in DPIS:
        yield f"raster-{dpi}", f"raster-{dpi}.pdf", extract_raster, {"dpi": dpi}


def hardware() -> dict:
    try:
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
        ram_gb = round(ram / 2**30)
    except (OSError, ValueError, subprocess.CalledProcessError):
        ram_gb = None
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    return {"platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(), "ram_gb": ram_gb}


def main() -> None:
    params = RasterParams.from_dict(json.loads((RUNS / "params.json").read_text())["chosen"])
    rows, metrics, per_class = [], {}, []
    for d in [*REAL, *TUNE, TEXT]:
        gt = load_ground_truth(truth_svg(d), PAGE_PT)
        metrics[d] = {"split": "held-out" if d in REAL else "synthetic", "gt_lines": len(gt.lines)}
        (RUNS / "overlays" / d).mkdir(parents=True, exist_ok=True)
        for name, file, fn, kw in settings():
            pdf = INPUTS / d / file
            if fn is extract_raster:
                kw = {**kw, "params": params}
            ex, seconds = timed(fn, pdf, drawing_id=d, **kw)
            s = score(ex.lines(), gt.lines).to_dict()
            assert all(
                -1e-4 <= x <= PAGE_PT + 1e-4 and -1e-4 <= y <= PAGE_PT + 1e-4
                for e in ex.elements
                for x, y in e.geometry
            ), (d, name)
            out = RUNS / "outputs" / d / name
            out.mkdir(parents=True, exist_ok=True)
            ex.to_json(out / "elements.json")
            ex.to_csv(out / "elements.csv")
            metrics[d][name] = {
                **s,
                "elements": len(ex.elements),
                "seconds": round(seconds, 4),
                "timing_s": {k: round(v, 6) for k, v in ex.timing_s.items()},
            }
            rows.append(
                {
                    "drawing": d,
                    "split": metrics[d]["split"],
                    "setting": name,
                    "elements": len(ex.elements),
                    "seconds": round(seconds, 4),
                    **s,
                }
            )
            if d in REAL:
                for label, rec in recall_by_label(ex.lines(), gt.lines, gt.labels).items():
                    n = sum(1 for lb in gt.labels if (lb or "unlabelled") == label)
                    per_class.append({"drawing": d, "setting": name, "class": label, "gt_lines": n, "recall": rec})
            draw_overlay(pdf, 1, ex, RUNS / "overlays" / d / f"{name}.png", gt=gt.lines)
        print(d, {k: v["f1"] for k, v in metrics[d].items() if isinstance(v, dict)})

    robustness = []
    rot = cv2.getRotationMatrix2D((PAGE_PT / 2, PAGE_PT / 2), ROTATE_DEG, 1.0)
    rot_matrix = (rot[0, 0], rot[1, 0], rot[0, 1], rot[1, 1], rot[0, 2], rot[1, 2])
    for d in TUNE:
        gt = load_ground_truth(truth_svg(d), PAGE_PT).lines
        for name, gt_lines in (
            ("raster-300", gt),
            ("raster-300-rot2", affine(gt, rot_matrix)),
            ("raster-300-scan", gt),
        ):
            pdf = INPUTS / d / f"{name}.pdf"
            ex = extract_raster(pdf, params=params, drawing_id=d)
            robustness.append({"drawing": d, "setting": name, **score(ex.lines(), gt_lines).to_dict()})
            draw_overlay(pdf, 1, ex, RUNS / "overlays" / d / f"{name}.png", gt=gt_lines)

    text = {}
    full = load_ground_truth(source_svg(TEXT), PAGE_PT).lines
    clean = load_ground_truth(truth_svg(TEXT), PAGE_PT).lines
    glyph_len = sum(ln.length for ln in full) - sum(ln.length for ln in clean)
    for name, file, fn, kw in settings():
        if fn is extract_raster:
            kw = {**kw, "params": params}
        ex = fn(INPUTS / TEXT / file, **kw)
        text[name] = {"vs_drawing_without_text": score(ex.lines(), clean).to_dict()}
    text["glyph_stroke_length_pt"] = round(glyph_len, 2)

    _csv(RUNS / "per_drawing.csv", rows)
    _csv(RUNS / "per_class.csv", per_class)
    _csv(RUNS / "robustness.csv", robustness)
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=1, allow_nan=False))
    (RUNS / "text_paths.json").write_text(json.dumps(text, indent=1))
    config = {
        "params_sha256": hashlib.sha256((RUNS / "params.json").read_bytes()).hexdigest(),
        "source_sha256": {d: hashlib.sha256(source_svg(d).read_bytes()).hexdigest() for d in REAL},
        "sample_manifest": json.loads((RUNS / "sample_manifest.json").read_text()),
        "page_pt": PAGE_PT,
        "dpis": list(DPIS),
        "held_out": REAL,
        "synthetic_tuning": TUNE,
        "text_fixture": TEXT,
        "rotate_deg": ROTATE_DEG,
        "scan": {"noise_sigma": 8.0, "blur_px": 1, "seed": 0},
        "scoring": CONSTANTS,
        "raster_params": params.to_dict(),
        "timing": f"median of {REPEATS} runs of the extract call (open + read/decode + detect + clean-up); "
        "PDF preparation and scoring are excluded; read/extract/normalize stage medians are also recorded",
        "packages": {p: version(p) for p in ("pymupdf", "opencv-python-headless", "shapely", "svgelements", "numpy")},
        "python": platform.python_version(),
        "hardware": hardware(),
    }
    (RUNS / "experiment_config.json").write_text(json.dumps(config, indent=1))


def _csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
