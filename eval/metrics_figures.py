"""Plot the ten held-out drawings from numeric results only."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.common import REAL, RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RUNS / "metrics.json").read_text())
    for lang in ("en", "id"):
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        for setting, marker, color in [
            ("vector", "o", "#225ba8"),
            ("raster-150", "s", "#c9913c"),
            ("raster-300", "^", "#d8643a"),
            ("raster-600", "D", "#248866"),
        ]:
            ax.plot(
                REAL, [metrics[d][setting]["f1"] for d in REAL], marker=marker, color=color, label=setting, linewidth=1
            )
        ax.set_ylim(0, 1.035)
        ax.set_ylabel("Length-weighted F1" if lang == "en" else "F1 berbobot panjang")
        ax.set_title(
            "Paths versus pixels · 10 FloorPlanCAD test drawings"
            if lang == "en"
            else "Path versus piksel · 10 gambar uji FloorPlanCAD",
            loc="left",
        )
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, ncol=4, loc="lower left")
        fig.tight_layout()
        fig.savefig(args.out / f"floorplan-parser-ten-results{'-id' if lang == 'id' else ''}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
