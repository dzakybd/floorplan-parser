"""15-second demo video (1080x1350) from real runs on a synthetic drawing, ending on the held-out FloorPlanCAD numbers.

Scenes: the drawing; the vector route reading its lines; the pixel route at 150, 300 and 600 DPI with misses and
extras; the held-out results. Writes floorplan-parser-demo[-id].mp4 and .gif.

Usage: uv run python -m eval.demo --out ../dzakyfawwaz/content/floorplan-parser/media [--lang en|id]
(macOS fonts, needs ffmpeg; run after python -m eval.benchmark)
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from eval.common import INPUTS, RUNS
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.metrics import unmatched_points
from floorplan_parser.raster import RasterParams, extract_raster
from floorplan_parser.vector import extract_vector

W, H, FPS = 1080, 1350, 20
PANEL, PX, PY = 960, 60, 230  # drawing panel: 960 px square for the 800 pt page
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
INK, MUTED, PRIMARY, SOFT = (9, 9, 11), (82, 82, 91), (67, 56, 202), (238, 240, 255)
BLUE, RED, ORANGE, FAINT = (31, 119, 180), (214, 39, 40), (255, 127, 14), (205, 205, 200)
DRAWING = "synth-3-doors"

TEXT = {
    "en": {
        "kicker": "CONSTRUCTION PDF  ·  PATHS OR PIXELS?",
        "s1": ("A drawing PDF", "Walls, door swings, small circles"),
        "s2": ("Read the vector paths", "Every line is already stored: F1 {f1}"),
        "s3": ("Or find lines in pixels", "OpenCV Hough lines at {dpi} DPI: F1 {f1}"),
        "legend": "blue found · red missed · orange extra",
        "end_title": "Ten real CAD drawings",
        "end_rows": [
            ("Read the vector paths", "F1 {vf1} on all ten"),
            ("Find lines in pixels, 300 DPI", "mean F1 {rf1} ({rmin} to {rmax})"),
            ("But vector lines come in pieces", "up to {pieces} per line: merge first"),
        ],
        "end_note": "Out: line coordinates as JSON and CSV.",
        "end_foot": "Data: FloorPlanCAD, numbers only. Pictures: synthetic drawing.",
        "url": "dzaky.ai.id/en/blog/floorplan-parser-vector-vs-raster",
    },
    "id": {
        "kicker": "PDF KONSTRUKSI  ·  PATH ATAU PIKSEL?",
        "s1": ("Sebuah PDF gambar", "Dinding, ayunan pintu, lingkaran kecil"),
        "s2": ("Baca path vektornya", "Setiap garis sudah tersimpan: F1 {f1}"),
        "s3": ("Atau cari garis di piksel", "Hough lines OpenCV pada {dpi} DPI: F1 {f1}"),
        "legend": "biru terbaca · merah terlewat · oranye kelebihan",
        "end_title": "Sepuluh gambar CAD nyata",
        "end_rows": [
            ("Baca path vektor", "F1 {vf1} di kesepuluhnya"),
            ("Cari garis di piksel, 300 DPI", "rata-rata F1 {rf1} ({rmin} sampai {rmax})"),
            ("Tapi garis vektor terpecah", "sampai {pieces} per garis: gabungkan dulu"),
        ],
        "end_note": "Hasil: koordinat garis dalam JSON dan CSV.",
        "end_foot": "Data: FloorPlanCAD, hanya angka. Gambar: sintetis.",
        "url": "dzaky.ai.id/id/blog/floorplan-parser-vector-vs-raster",
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def page_image(pdf: Path) -> Image.Image:
    with pymupdf.open(pdf) as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(PANEL / 800, PANEL / 800), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.eval(img, lambda v: 255 - (255 - v) * 0.35)  # faded, so the extracted lines stand out


def frame(t: dict, title: str, sub: str, panel: Image.Image | None, legend: bool) -> Image.Image:
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((PX, 56), t["kicker"], fill=PRIMARY, font=font(26, bold=True))
    d.text((PX, 100), title, fill=INK, font=font(56, bold=True))
    d.text((PX, 172), sub, fill=MUTED, font=font(32))
    if panel is not None:
        im.paste(panel, (PX, PY))
        d.rectangle((PX, PY, PX + PANEL - 1, PY + PANEL - 1), outline=(226, 226, 220), width=2)
    if legend:
        d.text((PX, PY + PANEL + 22), t["legend"], fill=MUTED, font=font(28))
    d.text((PX, H - 70), t["url"], fill=MUTED, font=font(26))
    return im


def draw_lines(base: Image.Image, lines, n: int, color, width: int = 3) -> Image.Image:
    im = base.copy()
    d = ImageDraw.Draw(im)
    s = PANEL / 800
    for e in lines[:n]:
        d.line([(x * s, y * s) for x, y in e.geometry], fill=color, width=width)
    return im


def dots(base: Image.Image, pts: np.ndarray, color, r: int = 4) -> Image.Image:
    im = base.copy()
    d = ImageDraw.Draw(im)
    s = PANEL / 800
    for x, y in pts:
        d.ellipse((x * s - r, y * s - r, x * s + r, y * s + r), fill=color)
    return im


def end_card(t: dict, metrics: dict, lang: str) -> Image.Image:
    held = [v for v in metrics.values() if v["split"] == "held-out"]
    vf1 = {round(v["vector"]["f1"], 3) for v in held}
    assert len(vf1) == 1, "end card says the vector F1 is the same on every drawing"
    rf1 = [v["raster-300"]["f1"] for v in held]
    vals = {
        "vf1": f"{vf1.pop():.3f}",
        "rf1": f"{sum(rf1) / len(rf1):.3f}",
        "rmin": f"{min(rf1):.3f}",
        "rmax": f"{max(rf1):.3f}",
        "pieces": f"{max(v['vector']['elements'] / v['gt_lines'] for v in held):.1f}",
    }
    if lang == "id":
        vals = {k: v.replace(".", ",") for k, v in vals.items()}
    im = frame(t, t["end_title"], t["end_note"], None, False)
    d = ImageDraw.Draw(im)
    for i, (label, value) in enumerate(t["end_rows"]):
        y = 300 + i * 190
        d.rounded_rectangle((PX - 10, y - 20, W - PX + 10, y + 130), radius=18, fill=SOFT if i % 2 == 0 else None)
        d.text((PX + 10, y), label, fill=INK, font=font(38, bold=True))
        d.text((PX + 10, y + 56), value.format(**vals), fill=PRIMARY, font=font(42, bold=True))
    d.text((PX, 300 + 3 * 190 + 20), t["end_foot"], fill=MUTED, font=font(28))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "id"], default="en")
    args = ap.parse_args()
    t = TEXT[args.lang]
    sfx = "" if args.lang == "en" else "-id"
    metrics = json.loads((RUNS / "metrics.json").read_text())
    params = RasterParams.from_dict(json.loads((RUNS / "params.json").read_text())["chosen"])
    src = INPUTS / DRAWING
    gt = load_ground_truth(INPUTS / "synth" / f"{DRAWING}.svg").lines
    base = page_image(src / "vector.pdf")
    vec = sorted(extract_vector(src / "vector.pdf").elements, key=lambda e: (e.geometry[0][1], e.geometry[0][0]))
    m = metrics[DRAWING]

    frames: list[Image.Image] = []
    hold = lambda img, sec: frames.extend([img] * round(sec * FPS))
    hold(frame(t, *t["s1"], base, False), 2.0)
    steps = round(2.5 * FPS)
    for k in range(1, steps + 1):
        n = round(len(vec) * k / steps)
        sub = t["s2"][1].format(f1=f"{m['vector']['f1']:.3f}")
        frames.append(frame(t, t["s2"][0], sub, draw_lines(base, vec, n, BLUE), False))
    hold(frames[-1], 1.0)
    for dpi in (150, 300, 600):
        ex = extract_raster(src / f"raster-{dpi}.pdf", params=params)
        lines = sorted(ex.elements, key=lambda e: (e.geometry[0][1], e.geometry[0][0]))
        sub = t["s3"][1].format(dpi=dpi, f1=f"{m[f'raster-{dpi}']['f1']:.3f}")
        steps = round(1.0 * FPS)
        for k in range(1, steps + 1):
            frames.append(frame(t, t["s3"][0], sub, draw_lines(base, lines, round(len(lines) * k / steps), BLUE), True))
        marked = dots(frames[-1].crop((PX, PY, PX + PANEL, PY + PANEL)), unmatched_points(gt, ex.lines()), RED)
        marked = dots(marked, unmatched_points(ex.lines(), gt), ORANGE, r=3)
        hold(frame(t, t["s3"][0], sub, marked, True), 1.2)
    hold(end_card(t, metrics, args.lang), 3.0)

    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):
        f.save(tmp / f"f{i:04d}.png")
    out_mp4 = args.out / f"floorplan-parser-demo{sfx}.mp4"
    out_gif = args.out / f"floorplan-parser-demo{sfx}.gif"
    ff = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png")]
    subprocess.run([*ff, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)], check=True)
    vf = "fps=12,scale=540:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none"
    subprocess.run([*ff, "-vf", vf, str(out_gif)], check=True)
    shutil.rmtree(tmp)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s -> {out_mp4.name}, {out_gif.name}")


if __name__ == "__main__":
    main()
