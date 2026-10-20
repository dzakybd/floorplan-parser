"""Score the lines -> walls -> rooms step.

1. Rooms, on synthetic plans with an answer key (eval/synth_rooms.py): the two room parameters are chosen on the
   tune seeds only, frozen in runs/room_params.json, then scored once on the held-out seeds.
2. Walls, on FloorPlanCAD drawings with wall labels (eval/fetch_walls.py): recall of the wall-labelled linework.
   Precision cannot be scored there: many real walls are unlabelled.

Rooms are matched one-to-one at IoU > 0.5 (a match above 0.5 is unique, so no assignment step is needed).
Writes runs/room_params.json and runs/rooms.json.

Usage: uv run python -m eval.rooms_eval
"""

import itertools
import json
import statistics

import numpy as np
from shapely import STRtree, points
from shapely.geometry import Polygon

from eval.common import INPUTS, PAGE_PT, RUNS, SAMPLES
from eval.synth_rooms import TEST_SEEDS, TUNE_SEEDS, make
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import _samples, _segments
from floorplan_parser.prep import clean_svg, svg_to_vector_pdf
from floorplan_parser.rooms import RoomParams, detect_walls, extract_rooms, wall_thickness
from floorplan_parser.vector import extract_vector

GRID = {"max_gap_factor": [4, 6, 8, 10, 14], "min_area_factor": [5, 10, 20, 40]}
ROOMS = INPUTS / "rooms"


def score_rooms(pred: list[Polygon], gt: list[Polygon]) -> dict:
    tree = STRtree(pred) if pred else None
    pairs = []
    for gi, g in enumerate(gt):
        for pi in tree.query(g) if tree is not None else []:
            p = pred[pi]
            iou = p.intersection(g).area / p.union(g).area
            if iou > 0.5:
                pairs.append((pi, gi, iou))
    m = len(pairs)
    prec, rec = (m / len(pred) if pred else 0.0), (m / len(gt) if gt else 0.0)
    errs = [(pred[pi].area - gt[gi].area) / gt[gi].area * 100 for pi, gi, _ in pairs]
    total_p, total_g = sum(p.area for p in pred), sum(g.area for g in gt)
    return {
        "gt_rooms": len(gt),
        "pred_rooms": len(pred),
        "matched": m,
        "room_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "matched_iou_mean": statistics.mean(i for *_, i in pairs) if pairs else None,
        "area_abs_err_pct_median": statistics.median(abs(e) for e in errs) if errs else None,
        "total_area_err_pct": (total_p - total_g) / total_g * 100 if total_g else None,
    }


def _plan(seed: int):
    name = f"rooms-{seed:03d}"
    pdf = ROOMS / f"{name}.pdf"
    if not pdf.exists():
        svg_to_vector_pdf(clean_svg((ROOMS / f"{name}.svg").read_text(encoding="utf-8"), PAGE_PT), pdf)
    ex = extract_vector(pdf, chain=True)
    gt = json.loads((ROOMS / f"{name}.rooms.json").read_text())
    return name, ex, [Polygon(r) for r in gt["rooms"]], gt["wall_thickness_pt"]


def _wall_recall(lines, gt_lines) -> float:
    a, b = _segments(gt_lines)
    pts, _, w = _samples(a, b)
    walls = detect_walls(lines, wall_thickness(lines) or 1.0)
    if walls.is_empty or len(pts) == 0:
        return 0.0
    near = walls.buffer(1.0).contains(points(pts))
    return float(w[near].sum() / w.sum())


def main() -> None:
    make(ROOMS)
    tune = [_plan(s) for s in TUNE_SEEDS]
    trials = []
    for values in itertools.product(*GRID.values()):
        p = RoomParams(**dict(zip(GRID, values, strict=True)))
        ms = [score_rooms(extract_rooms(ex.lines(), ex.page_size, p), gt) for _, ex, gt, _ in tune]
        trials.append({"params": p.to_dict(), "room_f1": round(statistics.mean(x["room_f1"] for x in ms), 4),
                       "abs_total_area_err_pct": round(statistics.mean(abs(x["total_area_err_pct"]) for x in ms), 3)})  # fmt: skip
    best = max(trials, key=lambda t: (t["room_f1"], -t["abs_total_area_err_pct"]))
    (RUNS / "room_params.json").write_text(json.dumps({"tuned_on": [f"rooms-{s:03d}" for s in TUNE_SEEDS],
        "objective": "mean room F1, ties broken by mean |total area error|", "grid": GRID, "chosen": best["params"],
        "chosen_score": best, "all_trials": trials}, indent=1))  # fmt: skip
    params = RoomParams.from_dict(best["params"])

    synthetic = {}
    for seed in TEST_SEEDS:
        name, ex, gt, t_true = _plan(seed)
        t_est = wall_thickness(ex.lines(), params)
        synthetic[name] = score_rooms(extract_rooms(ex.lines(), ex.page_size, params), gt) | {
            "wall_thickness_true_pt": t_true,
            "wall_thickness_est_pt": round(t_est, 3) if t_est else None,
        }

    real = {}
    manifest = json.loads((RUNS / "wall_manifest.json").read_text())
    for d in manifest["drawings"]:
        svg = SAMPLES / "walls" / f"{d}.svg"
        out = INPUTS / "walls" / d
        out.mkdir(parents=True, exist_ok=True)
        pdf = svg_to_vector_pdf(clean_svg(svg.read_text(encoding="utf-8"), PAGE_PT), out / "vector.pdf")
        ex = extract_vector(pdf, chain=True)
        gt = load_ground_truth(svg, PAGE_PT)
        wall_lines = [ln for ln, lb in zip(gt.lines, gt.labels, strict=True) if lb == "wall"]
        t = wall_thickness(ex.lines(), params)
        real[d] = {
            "wall_lines": len(wall_lines),
            "wall_thickness_est_pt": round(t, 3) if t else None,
            "wall_recall": round(_wall_recall(ex.lines(), wall_lines), 4),
            "rooms_found": len(extract_rooms(ex.lines(), ex.page_size, params)),
        }

    def dist(vals):
        vals = [v for v in vals if v is not None]
        return (
            {
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
            if vals
            else {"n": 0}
        )

    summary = {
        "synthetic_held_out": {
            k: dist([v[k] for v in synthetic.values()])
            for k in ("room_f1", "area_abs_err_pct_median", "total_area_err_pct", "matched_iou_mean")
        },
        "thickness_exact_within_0.5pt": sum(
            abs(v["wall_thickness_est_pt"] - v["wall_thickness_true_pt"]) <= 0.5
            for v in synthetic.values()
            if v["wall_thickness_est_pt"]
        ),
        "real_wall_recall": dist([v["wall_recall"] for v in real.values()]),
    }
    (RUNS / "rooms.json").write_text(json.dumps({"params": params.to_dict(), "summary": summary,
                                                 "synthetic": synthetic, "floorplancad_walls": real}, indent=1))  # fmt: skip
    print(json.dumps(summary, indent=1))
    print("chosen", best)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
