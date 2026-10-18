"""Extract the three held-out FloorPlanCAD drawings from a local copy of the original test archive.

The archive (test-00.tar.xz, about 1.8 GB) is linked from https://floorplancad.github.io/ . Files land in samples/,
which is gitignored: FloorPlanCAD data is never committed.

Usage: uv run python -m eval.fetch_samples --archive ~/Downloads/test-00.tar.xz
"""

import argparse
import json
import random
import tarfile
from pathlib import Path

from eval.common import REAL, RUNS, SAMPLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument(
        "--select", action="store_true", help="Retain the original three drawings and select seven by size stratum"
    )
    args = ap.parse_args()
    if args.select:
        original = ["0674-0822", "1010-0022", "0393-0004"]
        with tarfile.open(args.archive, mode="r|xz") as tar:
            candidates = sorted(
                (m.size, Path(m.name).stem)
                for m in tar
                if m.isfile() and m.name.endswith(".svg") and Path(m.name).stem not in original
            )
        rng = random.Random(20261019)
        selected = []
        for i, n in enumerate((3, 2, 2)):
            group = candidates[i * len(candidates) // 3 : (i + 1) * len(candidates) // 3]
            selected.extend(
                {"drawing": d, "svg_bytes": size, "stratum": ["simple", "medium", "dense"][i]}
                for size, d in rng.sample(group, n)
            )
        manifest = {
            "source": "https://floorplancad.github.io/ (original test-00.tar.xz)",
            "selection_seed": 20261019,
            "retained": original,
            "additional": selected,
            "selection": "Retain three previously inspected examples; seven additional test drawings randomly sampled within SVG-byte-size terciles (3/2/2), before scoring",
            "held_out": original + [r["drawing"] for r in selected],
        }
        (RUNS / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    drawings = (
        json.loads((RUNS / "sample_manifest.json").read_text())["held_out"]
        if (RUNS / "sample_manifest.json").exists()
        else REAL
    )
    want = {f"{d}.{ext}" for d in drawings for ext in ("svg", "png")}
    SAMPLES.mkdir(exist_ok=True)
    with tarfile.open(args.archive, mode="r|xz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in want:
                (SAMPLES / name).write_bytes(tar.extractfile(member).read())
                want.discard(name)
                print(name)
                if not want:
                    break
    if want:
        raise SystemExit(f"not found in archive: {sorted(want)}")


if __name__ == "__main__":
    main()
