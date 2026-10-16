"""Pick raster parameters on the synthetic drawings only, then freeze them in runs/params.json.

The FloorPlanCAD drawings are held out: this script never reads samples/ or their inputs.

Usage: uv run python -m eval.tune
"""

import itertools
import json
import statistics

from eval.common import INPUTS, RUNS, TUNE, truth_svg
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import score
from floorplan_parser.raster import RasterParams, extract_raster

GRID = {
    "threshold": [100, 128, 160],
    "close_px": [0, 1],
    "hough_votes_pt": [3.0, 6.0, 10.0],
    "min_line_pt": [1.5, 3.0, 5.0],
    "max_gap_pt": [0.5, 1.0, 2.0],
}


def main() -> None:
    truth = {d: load_ground_truth(truth_svg(d)).lines for d in TUNE}
    trials = []
    for values in itertools.product(*GRID.values()):
        params = RasterParams(**dict(zip(GRID, values)))
        f1s = [score(extract_raster(INPUTS / d / "raster-300.pdf", params=params).lines(), truth[d]).f1 for d in TUNE]
        trials.append({"params": params.to_dict(), "mean_f1": round(statistics.mean(f1s), 4)})
    best = max(trials, key=lambda t: t["mean_f1"])
    out = {
        "tuned_on": TUNE,
        "dpi": 300,
        "objective": "mean f1 over the tuning drawings",
        "grid": GRID,
        "chosen": best["params"],
        "chosen_mean_f1": best["mean_f1"],
        "default_mean_f1": next(t["mean_f1"] for t in trials if t["params"] == RasterParams().to_dict()),
        "trials": len(trials),
    }
    (RUNS / "params.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chosen", "chosen_mean_f1", "default_mean_f1")}, indent=1))


if __name__ == "__main__":
    main()
