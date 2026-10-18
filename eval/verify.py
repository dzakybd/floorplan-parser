"""Validate every export and independently recompute held-out scores from exported geometry."""

import csv
import hashlib
import json
import subprocess
import sys

from PIL import Image

from eval.common import PAGE_PT, REAL, RUNS, source_svg, truth_svg
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import score
from floorplan_parser.model import Extraction


def main():
    config = json.loads((RUNS / "experiment_config.json").read_text())
    metrics = json.loads((RUNS / "metrics.json").read_text())
    assert len(REAL) == len(set(REAL)) == 10
    assert hashlib.sha256((RUNS / "params.json").read_bytes()).hexdigest() == config["params_sha256"]
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False, capture_output=True, text=True)
    (RUNS / "test_results.txt").write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    checked = rescored = 0
    for d, entry in metrics.items():
        if d in REAL:
            assert hashlib.sha256(source_svg(d).read_bytes()).hexdigest() == config["source_sha256"][d]
            gt = load_ground_truth(truth_svg(d), PAGE_PT).lines
        for setting in ("vector", "raster-150", "raster-300", "raster-600"):
            out = RUNS / "outputs" / d / setting
            ex = Extraction.from_json(out / "elements.json")
            assert ex.coordinate_system == "pdf_points" and ex.page == 1
            with (out / "elements.csv").open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == len(ex.elements)
            assert all(r["coordinate_system"] == "pdf_points" and r["page"] == "1" for r in rows)
            assert all(
                -1e-4 <= x <= PAGE_PT + 1e-4 and -1e-4 <= y <= PAGE_PT + 1e-4
                for e in ex.elements
                for x, y in e.geometry
            )
            with Image.open(RUNS / "overlays" / d / f"{setting}.png") as im:
                assert im.width > 0 and im.height > 0
            if d in REAL:
                for k, value in score(ex.lines(), gt).to_dict().items():
                    assert value == entry[setting][k], (d, setting, k)
                rescored += 1
            checked += 1
    (RUNS / "verification.json").write_text(
        json.dumps(
            {
                "pytest": {
                    "command": "python -m pytest -q",
                    "exit_code": tests.returncode,
                    "output": tests.stdout.strip(),
                },
                "exports_checked": checked,
                "held_out_settings_rescored": rescored,
                "params_unchanged": True,
                "source_hashes_verified": True,
                "checks": "JSON/CSV counts and page/unit metadata; coordinates in page bounds; readable overlays; exported geometry scores equal benchmark",
            },
            indent=2,
        )
    )
    print(f"Verified {checked} exports and independently rescored {rescored} held-out settings")


if __name__ == "__main__":
    main()
