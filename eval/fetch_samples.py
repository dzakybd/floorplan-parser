"""Extract the three held-out FloorPlanCAD drawings from a local copy of the original test archive.

The archive (test-00.tar.xz, about 1.8 GB) is linked from https://floorplancad.github.io/ . Files land in samples/,
which is gitignored: FloorPlanCAD data is never committed.

Usage: uv run python -m eval.fetch_samples --archive ~/Downloads/test-00.tar.xz
"""

import argparse
import tarfile
from pathlib import Path

from eval.common import REAL, SAMPLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args()
    want = {f"{d}.{ext}" for d in REAL for ext in ("svg", "png")}
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
