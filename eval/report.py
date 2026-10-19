"""Equal-weight held-out distributions for every extraction setting."""

import json
import statistics

from eval.common import REAL, RUNS


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    m = json.loads((RUNS / "metrics.json").read_text())
    summary = {
        "n_drawings": len(REAL),
        "aggregation": "Each drawing has equal weight. Endpoint errors are conditional on a nearest predicted endpoint within 5 pt; report endpoint_match_rate alongside them.",
        "settings": {},
    }
    for setting in ("vector", "vector-chained", "raster-150", "raster-300", "raster-600"):
        rows = [m[d][setting] for d in REAL]
        entry = {k: distribution([r[k] for r in rows]) for k in rows[0] if k != "timing_s"}
        entry["timing_s"] = {k: distribution([r["timing_s"][k] for r in rows]) for k in rows[0]["timing_s"]}
        summary["settings"][setting] = entry
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("Summarized", len(REAL), "held-out drawings")


if __name__ == "__main__":
    main()
