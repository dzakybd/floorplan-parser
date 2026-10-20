"""Pick and extract the wall-recall set: FloorPlanCAD test drawings with at least 30 wall-labelled primitives.

FloorPlanCAD labels walls on few drawings (742 of 5,502 test SVGs have any wall label), so walls are scored on
their own set, chosen by label count only: candidates with >= 30 wall primitives, minus the ten line-extraction
drawings, shuffled with seed 20261020, first ten. Files land in samples/walls/ (gitignored); the list is
runs/wall_manifest.json.

Usage: uv run python -m eval.fetch_walls --archive ~/Downloads/test-00.tar.xz
"""

import argparse
import json
import random
import re
import tarfile
from pathlib import Path

from eval.common import REAL, RUNS, SAMPLES

MIN_WALLS, N, SEED = 30, 10, 20261020


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args()
    manifest = RUNS / "wall_manifest.json"
    out = SAMPLES / "walls"
    out.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        chosen = json.loads(manifest.read_text())["drawings"]
    else:
        counts = {}
        with tarfile.open(args.archive, mode="r|xz") as tar:
            for m in tar:
                if m.name.endswith(".svg"):
                    text = tar.extractfile(m).read().decode("utf-8", "ignore")
                    counts[Path(m.name).stem] = len(re.findall(r'semantic-id="33"', text))
        candidates = sorted(k for k, v in counts.items() if v >= MIN_WALLS and k not in REAL)
        chosen = random.Random(SEED).sample(candidates, N)
        manifest.write_text(json.dumps({"rule": f">= {MIN_WALLS} wall primitives (semantic-id 33), not in the line set, "
                                        f"random.Random({SEED}).sample of {len(candidates)} candidates",
                                        "svgs_scanned": len(counts), "candidates": len(candidates),
                                        "drawings": chosen, "wall_primitives": {k: counts[k] for k in chosen}}, indent=1))  # fmt: skip
    want = set(chosen)
    with tarfile.open(args.archive, mode="r|xz") as tar:
        for m in tar:
            if Path(m.name).stem in want and m.name.endswith(".svg"):
                (out / f"{Path(m.name).stem}.svg").write_bytes(tar.extractfile(m).read())
    print(sorted(p.stem for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
