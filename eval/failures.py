"""Write up three cases from the held-out results: the vector run that fragments most (elements per ground-truth
line; its f1 is 1.0 on every held-out drawing, so fragmentation is how it fails), the weakest raster run (300 DPI),
and a drawing where both do well. Each note gives the numbers and where the missed length sits by class.

Overlays referenced here live in runs/overlays/ (local only: they show FloorPlanCAD drawings).

Usage: uv run python -m eval.failures
"""

import csv
import json
from collections import defaultdict

from eval.common import REAL, RUNS

KEYS = ("precision", "recall", "f1", "endpoint_err_median", "length_error_pct", "elements", "seconds")


def main() -> None:
    m = json.loads((RUNS / "metrics.json").read_text())
    pc = defaultdict(list)
    for r in csv.DictReader((RUNS / "per_class.csv").open()):
        pc[(r["drawing"], r["setting"])].append(r)
    vec = max(REAL, key=lambda d: m[d]["vector"]["elements"] / m[d]["gt_lines"])
    ras = min(REAL, key=lambda d: m[d]["raster-300"]["f1"])
    both = max(REAL, key=lambda d: min(m[d]["vector"]["f1"], m[d]["raster-300"]["f1"]))
    out = RUNS / "failure_cases"
    out.mkdir(exist_ok=True)
    for case, d, setting in (
        ("vector-fragmented", vec, "vector"),
        ("raster-weakest", ras, "raster-300"),
        ("both-succeed", both, "raster-300"),
    ):
        ratio = m[d]["vector"]["elements"] / m[d]["gt_lines"]
        lines = [f"# {case}: {d} ({setting})", "", f"Ground-truth lines: {m[d]['gt_lines']}"]
        lines += [f"Vector elements per ground-truth line: {ratio:.1f}", ""]
        lines += ["| setting | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
        for s in ("vector", "raster-150", "raster-300", "raster-600"):
            lines.append(f"| {s} | " + " | ".join(str(m[d][s][k]) for k in KEYS) + " |")
        rows = sorted(pc[(d, setting)], key=lambda r: float(r["recall"]))
        lines += [
            "",
            f"Recall by class ({setting}), lowest first:",
            "",
            "| class | gt lines | recall |",
            "|---|---|---|",
        ]
        lines += [f"| {r['class']} | {r['gt_lines']} | {r['recall']} |" for r in rows]
        lines += [
            "",
            (
                f"Overlay (local only): runs/overlays/{d}/{setting}.png. Blue: extracted. Red: missed ground "
                "truth. Orange: extracted but not in ground truth."
            ),
        ]
        (out / f"{case}.md").write_text("\n".join(lines) + "\n")
        print(case, d, setting, m[d][setting]["f1"])


if __name__ == "__main__":
    main()
