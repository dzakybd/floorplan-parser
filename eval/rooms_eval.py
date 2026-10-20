"""Score the lines -> walls -> rooms step.

1. Rooms, on synthetic plans with an answer key (eval/synth_rooms.py): the two room parameters are chosen on the
   tune seeds only, frozen in runs/room_params.json, then scored once on the held-out seeds.
2. FloorPlanCAD drawings with wall labels (eval/fetch_walls.py): a label audit, not an accuracy. Inspection shows
   the 'wall' label marks only some walls (often thin partitions) while many thick walls are unlabelled, so the
   share of labelled linework inside, or on the edges of, the detected walls describes the labels, not detection.

3. Rooms on real layouts: CubiCasa5K plans drawn as vector PDFs from their wall, door and room annotations by
   quantity-lab (eval/export_vector.py). Real walls vary in thickness, so the wall-thickness rule (peaks kept,
   tolerance) is chosen on ten validation layouts by room F1, then everything is scored once on the ten test ones. This tests walls -> rooms on
   real room shapes and openings; the walls are drawn cleanly from the annotation, not read from messy CAD.

Rooms are matched one-to-one at IoU > 0.5 (a match above 0.5 is unique, so no assignment step is needed).
Writes runs/room_params.json and runs/rooms.json.

Usage: uv run python -m eval.rooms_eval
"""

import itertools
import json
import statistics
from pathlib import Path

import numpy as np
from shapely import STRtree, points, wkt
from shapely.geometry import Polygon

from eval.common import INPUTS, PAGE_PT, RUNS, SAMPLES
from eval.synth_rooms import TEST_SEEDS, TUNE_SEEDS, make
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import _samples, _segments
from floorplan_parser.prep import clean_svg, svg_to_vector_pdf
from floorplan_parser.rooms import RoomParams, detect_walls, extract_rooms, wall_thickness, wall_thicknesses
from floorplan_parser.vector import extract_vector

CUBICASA = Path(__file__).resolve().parents[2] / "quantity-lab" / "runs" / "inputs" / "vector"
CUBICASA_VAL = CUBICASA.with_name("vector-val")  # validation layouts: where the wall-thickness rule is chosen
THICKNESS_GRID = {"n_peaks": [1, 2, 3, 4, 5], "thickness_tol": [0.25, 0.4, 0.6], "peak_share": [0.1]}
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


def _label_audit(lines, gt_lines, p: RoomParams) -> dict:
    """Where FloorPlanCAD's wall-labelled lines fall relative to the detected walls. Not an accuracy: the labels mark
    only some walls (often thin partitions), and many thick walls are unlabelled, so neither share is a recall."""
    a, b = _segments(gt_lines)
    pts, _, w = _samples(a, b)
    ts = wall_thicknesses(lines, p)
    if not ts or len(pts) == 0:
        return {"inside_detected_walls": 0.0, "on_detected_wall_edges": 0.0}
    walls = detect_walls(lines, ts, p)
    if walls.is_empty:
        return {"inside_detected_walls": 0.0, "on_detected_wall_edges": 0.0}
    inside = walls.buffer(1.0).contains(points(pts))
    edge = walls.boundary.buffer(1.0).contains(points(pts))
    return {"inside_detected_walls": round(float(w[inside].sum() / w.sum()), 4),
            "on_detected_wall_edges": round(float(w[edge].sum() / w.sum()), 4)}  # fmt: skip


def _real_walls(drawing: str, folder: str):
    svg = SAMPLES / folder / f"{drawing}.svg"
    out = INPUTS / folder / drawing
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "vector.pdf"
    if not pdf.exists():
        svg_to_vector_pdf(clean_svg(svg.read_text(encoding="utf-8"), PAGE_PT), pdf)
    ex = extract_vector(pdf, chain=True)
    gt = load_ground_truth(svg, PAGE_PT)
    return ex, [ln for ln, lb in zip(gt.lines, gt.labels, strict=True) if lb == "wall"]


def _layout(rooms_json: Path):
    info = json.loads(rooms_json.read_text())
    return extract_vector(rooms_json.parent / "plan.pdf", chain=True), [wkt.loads(w) for w in info["rooms"]]


def main() -> None:
    make(ROOMS)
    tune = [_plan(s) for s in TUNE_SEEDS]
    trials = []
    for values in itertools.product(*GRID.values()):
        # step 1 tunes the room parameters with the single-thickness rule; step 2 then chooses the thickness rule
        p = RoomParams(**dict(zip(GRID, values, strict=True)), thickness_tol=0.25, n_peaks=1, peak_share=0.25)
        ms = [score_rooms(extract_rooms(ex.lines(), ex.page_size, p), gt) for _, ex, gt, _ in tune]
        trials.append({"params": p.to_dict(), "room_f1": round(statistics.mean(x["room_f1"] for x in ms), 4),
                       "abs_total_area_err_pct": round(statistics.mean(abs(x["total_area_err_pct"]) for x in ms), 3)})  # fmt: skip
    best = max(trials, key=lambda t: (t["room_f1"], -t["abs_total_area_err_pct"]))
    (RUNS / "room_params.json").write_text(json.dumps({"tuned_on": [f"rooms-{s:03d}" for s in TUNE_SEEDS],
        "objective": "mean room F1, ties broken by mean |total area error|", "grid": GRID, "chosen": best["params"],
        "chosen_score": best, "all_trials": trials}, indent=1))  # fmt: skip
    params = RoomParams.from_dict(best["params"])
    val_layouts = [_layout(d) for d in sorted(CUBICASA_VAL.glob("*/rooms.json"))] if CUBICASA_VAL.exists() else []
    rules = []
    for values in itertools.product(*THICKNESS_GRID.values()):
        p = RoomParams(**(params.to_dict() | dict(zip(THICKNESS_GRID, values, strict=True))))
        f1 = (
            statistics.mean(
                score_rooms(extract_rooms(ex.lines(), ex.page_size, p), gt)["room_f1"] for ex, gt in val_layouts
            )
            if val_layouts
            else 0.0
        )
        # the layouts hold walls and door arcs only, so accepting every parallel gap costs nothing there; the
        # synthetic tune plans have furniture and windows, and guard against a rule that swallows them
        syn = [score_rooms(extract_rooms(ex.lines(), ex.page_size, p), gt) for _, ex, gt, _ in tune]
        rules.append({"params": p.to_dict(), "val_room_f1": round(f1, 4),
                      "synthetic_tune_room_f1": round(statistics.mean(x["room_f1"] for x in syn), 4),
                      "synthetic_tune_abs_total_area_err_pct": round(statistics.mean(abs(x["total_area_err_pct"]) for x in syn), 3)})  # fmt: skip
    if val_layouts:
        ok = [r for r in rules if r["synthetic_tune_room_f1"] >= best["room_f1"] - 0.01
              and r["synthetic_tune_abs_total_area_err_pct"] <= 1.0]  # fmt: skip
        params = RoomParams.from_dict(max(ok, key=lambda r: r["val_room_f1"])["params"])
    rp = json.loads((RUNS / "room_params.json").read_text())
    rp.update({"thickness_rule_chosen_on": "CubiCasa5K validation layouts (quantity-lab eval/export_vector.py --val), room F1, among rules that keep synthetic tune room F1 within 0.01 of the best and mean |total area error| <= 1%",
               "thickness_rules": rules, "final": params.to_dict()})  # fmt: skip
    (RUNS / "room_params.json").write_text(json.dumps(rp, indent=1))

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
        ex, wall_lines = _real_walls(d, "walls")
        ts = wall_thicknesses(ex.lines(), params)
        real[d] = {
            "wall_lines": len(wall_lines),
            "wall_thicknesses_est_pt": [round(t, 3) for t in ts],
            "label_audit": _label_audit(ex.lines(), wall_lines, params),
            "rooms_found": len(extract_rooms(ex.lines(), ex.page_size, params)),
        }

    layouts = {}
    for d in sorted(CUBICASA.glob("*/rooms.json")) if CUBICASA.exists() else []:
        ex, gt = _layout(d)
        ts = wall_thicknesses(ex.lines(), params)
        layouts[json.loads(d.read_text())["plan"]] = score_rooms(
            extract_rooms(ex.lines(), ex.page_size, params), gt
        ) | {"wall_thicknesses_est_pt": [round(t, 2) for t in ts]}

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
        "cubicasa_layouts": {
            k: dist([v[k] for v in layouts.values()])
            for k in ("room_f1", "area_abs_err_pct_median", "total_area_err_pct", "matched_iou_mean")
        }
        if layouts
        else None,
        "real_label_audit_inside": dist([v["label_audit"]["inside_detected_walls"] for v in real.values()]),
        "real_label_audit_edge": dist([v["label_audit"]["on_detected_wall_edges"] for v in real.values()]),
    }
    (RUNS / "rooms.json").write_text(json.dumps({"params": params.to_dict(), "summary": summary,
                                                 "synthetic": synthetic, "cubicasa_layouts": layouts, "floorplancad_walls": real}, indent=1))  # fmt: skip
    print(json.dumps(summary, indent=1))
    print("chosen", best)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
