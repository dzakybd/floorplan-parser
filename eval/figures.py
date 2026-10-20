"""Figures for the write-up, from synthetic drawings and committed numbers only (no FloorPlanCAD drawing is drawn).

One figure is a labelled synthetic recreation of a failure seen on the FloorPlanCAD drawings: an arc coming back from
the vector PDF as many straight pieces.

Usage: uv run python -m eval.figures --out ../dzakyfawwaz/content/floorplan-parser/media [--lang en|id]
(macOS fonts; run after python -m eval.benchmark)
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from eval.common import INPUTS, RUNS
from floorplan_parser.groundtruth import load_ground_truth
from floorplan_parser.overlay import draw_overlay
from floorplan_parser.raster import RasterParams, extract_raster
from floorplan_parser.vector import extract_vector

FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
INK, MUTED, BG = (29, 29, 27), (107, 107, 102), (255, 255, 255)
BLUE, ORANGE = "#2a78d6", "#eb6834"

TEXT = {
    "en": {
        "vector_pdf": "Vector PDF: the lines are stored",
        "raster_pdf": "Image PDF, 150 DPI: only pixels",
        "vector": "Vector paths",
        "raster": "Pixels, 300 DPI",
        "dpi": "Pixels, {} DPI",
        "legend": "blue: extracted   red: missed   orange: extra",
        "recreation": "Synthetic recreation of a failure seen on a FloorPlanCAD drawing",
        "pieces": "One door arc from the vector PDF: {} pieces",
        "pieces_gt": "In the drawing: 1 arc",
        "chain_raw": "As read: {} pieces",
        "chain_joined": "With --chain: {} polyline",
        "chain_note": "Same synthetic door arc, same geometry; F1 1.00 both ways",
        "rooms_titles": ["1. Lines read from the PDF", "2. Walls, doorways closed", "3. Rooms and areas"],
        "rooms_note": "Synthetic held-out plan. Orange: doorways bridged. Red outline: two rooms with no wall between them, found as one.",
        "text": "A label drawn as lines is picked up as geometry (orange)",
        "chart_title": "F1 by length on three held-out FloorPlanCAD drawings",
        "chart_x": "F1 (length-based), axis starts at 0.85",
        "rows": ["Simple: grid\n114 lines", "Medium: doors, double lines\n229 lines", "Dense: apartment\n807 lines"],
        "series": ["Vector paths", "Pixels, 300 DPI"],
    },
    "id": {
        "vector_pdf": "PDF vektor: garisnya tersimpan",
        "raster_pdf": "PDF gambar, 150 DPI: hanya piksel",
        "vector": "Path vektor",
        "raster": "Piksel, 300 DPI",
        "dpi": "Piksel, {} DPI",
        "legend": "biru: terbaca   merah: terlewat   oranye: kelebihan",
        "recreation": "Rekonstruksi sintetis dari kegagalan pada gambar FloorPlanCAD",
        "pieces": "Satu busur pintu dari PDF vektor: {} potongan",
        "pieces_gt": "Di gambar: 1 busur",
        "chain_raw": "Seperti terbaca: {} potongan",
        "chain_joined": "Dengan --chain: {} polyline",
        "chain_note": "Busur pintu sintetis yang sama, geometri sama; F1 1,00 di keduanya",
        "rooms_titles": ["1. Garis dibaca dari PDF", "2. Dinding, pintu ditutup", "3. Ruangan dan luasnya"],
        "rooms_note": "Denah sintetis held-out. Oranye: pintu dijembatani. Garis merah: dua ruangan tanpa dinding di antaranya, terbaca jadi satu.",
        "text": "Label yang digambar sebagai garis ikut terbaca sebagai geometri (oranye)",
        "chart_title": "F1 berbasis panjang pada tiga gambar FloorPlanCAD uji",
        "chart_x": "F1 (berbasis panjang), sumbu mulai 0,85",
        "rows": ["Sederhana: grid\n114 garis", "Sedang: pintu, garis ganda\n229 garis", "Padat: apartemen\n807 garis"],
        "series": ["Path vektor", "Piksel, 300 DPI"],
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def render(pdf: Path, dpi: int) -> Image.Image:
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        if images and not page.get_drawings():
            pix = pymupdf.Pixmap(doc, images[0][0])
        else:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
        return Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.width, pix.height), pix.samples).convert("RGB")


def crop_pt(img: Image.Image, box_pt: tuple[float, float, float, float], page_pt: float = 800.0) -> Image.Image:
    s = img.width / page_pt
    return img.crop(tuple(round(v * s) for v in box_pt))


def panel(images: list[Image.Image], labels: list[str], size: int, footer: str | None = None) -> Image.Image:
    """Images side by side, each scaled to `size` px square, with a label above and an optional footer line."""
    pad, head = 24, 52
    foot = 44 if footer else 0
    w = pad + len(images) * (size + pad)
    out = Image.new("RGB", (w, head + size + pad + foot), BG)
    d = ImageDraw.Draw(out)
    for i, (im, lb) in enumerate(zip(images, labels)):
        x = pad + i * (size + pad)
        d.text((x, 16), lb, fill=INK, font=font(24, bold=True))
        out.paste(im.resize((size, size), Image.NEAREST if im.width < size else Image.LANCZOS), (x, head))
        d.rectangle((x, head, x + size - 1, head + size - 1), outline=(226, 226, 220), width=2)
    if footer:
        d.text((pad, head + size + pad - 4), footer, fill=MUTED, font=font(20))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "id"], default="en")
    args = ap.parse_args()
    t = TEXT[args.lang]
    sfx = "" if args.lang == "en" else "-id"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    params = RasterParams.from_dict(json.loads((RUNS / "params.json").read_text())["chosen"])
    tmp = Path(tempfile.mkdtemp())

    # 1. Same drawing, two files: crisp vector vs 150 DPI pixels, zoomed on a door swing and small circles
    doors = INPUTS / "synth-3-doors"
    box = (270, 300, 380, 410)  # the 10-unit door swing
    vec = crop_pt(render(doors / "vector.pdf", 900), box)
    ras = crop_pt(render(doors / "raster-150.pdf", 150), box)
    panel([vec, ras], [t["vector_pdf"], t["raster_pdf"]], 560).save(out / f"floorplan-parser-two-inputs{sfx}.png")

    # 2. What each route found on the whole synthetic page
    gt = load_ground_truth(INPUTS / "synth" / "synth-3-doors.svg").lines
    ov_v = draw_overlay(doors / "vector.pdf", 1, extract_vector(doors / "vector.pdf"), tmp / "v.png", gt=gt)
    ex300 = extract_raster(doors / "raster-300.pdf", params=params)
    ov_r = draw_overlay(doors / "raster-300.pdf", 1, ex300, tmp / "r.png", gt=gt)
    panel([Image.open(ov_v), Image.open(ov_r)], [t["vector"], t["raster"]], 560, t["legend"]).save(
        out / f"floorplan-parser-overlays{sfx}.png"
    )

    # 3. DPI strip on the small circles
    strip, labels = [], []
    for dpi in (150, 300, 600):
        ex = extract_raster(doors / f"raster-{dpi}.pdf", params=params)
        p = draw_overlay(doors / f"raster-{dpi}.pdf", 1, ex, tmp / f"d{dpi}.png", gt=gt, scale=4.0)
        strip.append(crop_pt(Image.open(p), (300, 570, 540, 630)).resize((640, 160), Image.LANCZOS))
        labels.append(t["dpi"].format(dpi))
    w = Image.new("RGB", (24 + 3 * 664, 52 + 160 + 68), BG)
    dr = ImageDraw.Draw(w)
    for i, (im, lb) in enumerate(zip(strip, labels)):
        x = 24 + i * 664
        dr.text((x, 16), lb, fill=INK, font=font(24, bold=True))
        w.paste(im, (x, 52))
    dr.text((24, 52 + 160 + 18), t["legend"], fill=MUTED, font=font(20))
    w.save(out / f"floorplan-parser-dpi{sfx}.png")

    # 5. Recreation: one arc comes back from the vector PDF as many straight pieces
    ex = extract_vector(doors / "vector.pdf")
    arc_box = (280, 300, 360, 400)  # the 10-unit door swing, x 280..360 pt, y 320..400 pt
    pieces = [
        e
        for e in ex.elements
        if all(arc_box[0] - 1 <= x <= arc_box[2] + 1 and arc_box[1] - 1 <= y <= arc_box[3] + 1 for x, y in e.geometry)
        and abs(e.geometry[0][0] - e.geometry[-1][0]) > 0.01
        and abs(e.geometry[0][1] - e.geometry[-1][1]) > 0.01
    ]
    s = 5.0
    xs = [x for e in pieces for x, _ in e.geometry]
    ys = [y for e in pieces for _, y in e.geometry]
    x0, y0 = min(xs) - 3, min(ys) - 3
    im = Image.new("RGB", (round((max(xs) - x0 + 3) * s), round((max(ys) - y0 + 3) * s)), BG)
    dr = ImageDraw.Draw(im)
    for i, e in enumerate(pieces):
        pts = [((x - x0) * s, (y - y0) * s) for x, y in e.geometry]
        dr.line(pts, fill=BLUE if i % 2 else ORANGE, width=7)
    w = Image.new("RGB", (1200, im.height + 48), BG)
    dr = ImageDraw.Draw(w)
    dr.text((24, 40), t["recreation"], fill=MUTED, font=font(20))
    dr.text((24, 76), t["pieces"].format(len(pieces)), fill=INK, font=font(28, bold=True))
    dr.text((24, 116), t["pieces_gt"], fill=MUTED, font=font(24))
    w.paste(im, (1200 - im.width - 48, 24))
    w.save(out / f"floorplan-parser-recreate-pieces{sfx}.png")

    # 7. The same arc after chaining: pieces of one smooth curve joined into one polyline
    inside = lambda e: all(
        arc_box[0] - 1 <= x <= arc_box[2] + 1 and arc_box[1] - 1 <= y <= arc_box[3] + 1 for x, y in e.geometry
    )
    joined = [e for e in extract_vector(doors / "vector.pdf", chain=True).elements if inside(e) and len(e.geometry) > 2]
    pair = Image.new("RGB", (2 * im.width + 72, im.height + 150), BG)
    dr = ImageDraw.Draw(pair)
    for k, (title, elems, alternate) in enumerate(
        ((t["chain_raw"].format(len(pieces)), pieces, True), (t["chain_joined"].format(len(joined)), joined, False))
    ):
        x = 24 + k * (im.width + 24)
        dr.text((x, 20), title, fill=INK, font=font(28, bold=True))
        for i, e in enumerate(elems):
            pts = [(x + (px - x0) * s, 76 + (py - y0) * s) for px, py in e.geometry]
            dr.line(pts, fill=(ORANGE if i % 2 == 0 else BLUE) if alternate else BLUE, width=7, joint="curve")
    dr.text((24, im.height + 104), t["chain_note"], fill=MUTED, font=font(22))
    pair.save(out / f"floorplan-parser-chain{sfx}.png")
    for name, elems, alternate in (("pieces", pieces, True), ("chained", joined, False)):  # poster panels, no text
        tile = Image.new("RGB", (im.width + 48, im.height + 48), BG)
        dt = ImageDraw.Draw(tile)
        for i, e in enumerate(elems):
            pts = [(24 + (px - x0) * s, 24 + (py - y0) * s) for px, py in e.geometry]
            dt.line(pts, fill=(ORANGE if i % 2 == 0 else BLUE) if alternate else BLUE, width=9, joint="curve")
        tile.save(out / f"floorplan-parser-poster-{name}.png")

    # 8. Lines -> walls -> rooms on a held-out synthetic plan with an open kitchen
    from eval.rooms_eval import _plan
    from floorplan_parser.rooms import RoomParams, bridges, extract_rooms, wall_bands, wall_thicknesses

    rp = RoomParams.from_dict(json.loads((RUNS / "room_params.json").read_text())["chosen"])
    _, rex, rgt, _ = _plan(105)
    lines = rex.lines()
    bands = wall_bands(lines, wall_thicknesses(lines, rp), rp)
    brs = bridges(bands, rp.max_gap_factor, rp.angle_tol_deg)
    found = extract_rooms(lines, rex.page_size, rp)
    bx0 = min(ln.bounds[0] for ln in lines) - 10
    by0 = min(ln.bounds[1] for ln in lines) - 10
    bx1 = max(ln.bounds[2] for ln in lines) + 10
    by1 = max(ln.bounds[3] for ln in lines) + 10
    sc = 560 / max(bx1 - bx0, by1 - by0)
    pw, ph = round((bx1 - bx0) * sc), round((by1 - by0) * sc)
    to = lambda x, y: ((x - bx0) * sc, (y - by0) * sc)
    panels = []
    for k in range(3):
        im3 = Image.new("RGB", (pw, ph), BG)
        dr3 = ImageDraw.Draw(im3, "RGBA")
        if k == 0:
            for ln in lines:
                dr3.line([to(x, y) for x, y in ln.coords], fill=INK, width=2)
        if k == 1:
            for b in bands:
                dr3.polygon([to(x, y) for x, y in b.polygon().exterior.coords], fill=(42, 120, 214, 255))
            for b in brs:
                dr3.polygon([to(x, y) for x, y in b.polygon().exterior.coords], fill=(235, 104, 52, 255))
        if k == 2:
            fills = [
                (42, 120, 214, 70),
                (27, 175, 122, 70),
                (235, 104, 52, 60),
                (164, 123, 232, 70),
                (230, 180, 40, 80),
            ]
            for i, r in enumerate(found):
                dr3.polygon(
                    [to(x, y) for x, y in r.exterior.coords], fill=fills[i % len(fills)], outline=(80, 80, 80, 255)
                )
                merged = sum(r.intersection(g).area / g.area > 0.9 for g in rgt) >= 2
                if merged:
                    dr3.line([to(x, y) for x, y in r.exterior.coords], fill=(214, 39, 40, 255), width=5)
                cx, cy = to(*r.representative_point().coords[0])
                label = (
                    f"{r.area / 1000:.1f}k pt²" if args.lang == "en" else f"{r.area / 1000:.1f}k pt²".replace(".", ",")
                )
                dr3.text((cx - 34, cy - 10), label, fill=INK, font=font(17, bold=True))
            for ln in lines:
                dr3.line([to(x, y) for x, y in ln.coords], fill=(150, 150, 145, 255), width=1)
        panels.append(im3)
    trio = Image.new("RGB", (24 + 3 * (pw + 24), ph + 52 + 60), BG)
    dr = ImageDraw.Draw(trio)
    for k, im3 in enumerate(panels):
        x = 24 + k * (pw + 24)
        dr.text((x, 14), t["rooms_titles"][k], fill=INK, font=font(24, bold=True))
        trio.paste(im3, (x, 52))
    dr.text((24, ph + 52 + 18), t["rooms_note"], fill=MUTED, font=font(19))
    trio.save(out / f"floorplan-parser-rooms{sfx}.png")

    # 6. Text drawn as lines
    tp = INPUTS / "synth-text-paths"
    tgt = load_ground_truth(INPUTS / "synth" / "synth-text-paths.gt.svg").lines
    p = draw_overlay(tp / "vector.pdf", 1, extract_vector(tp / "vector.pdf"), tmp / "t.png", gt=tgt, scale=4.0)
    crop = crop_pt(Image.open(p), (140, 380, 660, 520)).resize((1040, 280), Image.LANCZOS)
    w = Image.new("RGB", (1088, 52 + 280 + 24), BG)
    ImageDraw.Draw(w).text((24, 14), t["text"], fill=INK, font=font(24, bold=True))
    w.paste(crop, (24, 52))
    w.save(out / f"floorplan-parser-text-lines{sfx}.png")

    # 7. Results chart: held-out F1, vector vs pixels at 300 DPI (DPI sweep is in the table)
    m = json.loads((RUNS / "metrics.json").read_text())
    ids = ["0674-0822", "1010-0022", "0393-0004"]
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(ids))[::-1]
    for k, (setting, color, dy) in enumerate((("vector", BLUE, 0.12), ("raster-300", ORANGE, -0.12))):
        vals = [m[d][setting]["f1"] for d in ids]
        ax.scatter(vals, y + dy, s=110, color=color, edgecolor="#fcfcfb", linewidth=2, zorder=3, label=t["series"][k])
        for v, yy in zip(vals, y + dy):
            ax.annotate(
                f"{v:.3f}",
                (v, yy),
                xytext=(-12, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color="#52514e",
                fontsize=12,
            )
    ax.set_yticks(y, t["rows"])
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel(t["chart_x"], color="#52514e")
    ax.set_title(t["chart_title"], loc="left", fontsize=15, fontweight="bold", color="#0b0b0b", pad=40)
    ax.grid(axis="x", color="#e6e5e0", linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(-0.01, 1.0), borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(out / f"floorplan-parser-results{sfx}.png", facecolor=fig.get_facecolor())
    print(f"figures ({args.lang}) -> {out}")


if __name__ == "__main__":
    main()
