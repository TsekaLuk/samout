"""Render the routing decision as something you can actually look at.

Two outputs:
  route_map.png    — the mockup, each region outlined by its verdict colour
  route_sheet.png  — every region cropped, grouped under its verdict, with the
                     model's one-line reason. This is the one to eyeball: it puts
                     the pixels next to the claim about them.

    python render_report.py input/ktv_ui.jpeg assets/assets.json assets/audit_qwen3-vl-flash.json
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

COLOR = {"photography": (0, 214, 255), "spot_illustration": (120, 255, 140),
         "product_icon": (255, 196, 0), "brand_asset": (255, 90, 200),
         "system_icon": (150, 170, 255), "typography": (140, 140, 155),
         "token": (110, 110, 125), "composite": (70, 70, 85),
         "unobserved": (255, 60, 60), "?": (255, 60, 120)}
ORDER = ["brand_asset", "photography", "spot_illustration", "product_icon",
         "system_icon", "typography", "token", "composite", "unobserved", "?"]


def boxes_from(handoff):
    """One source of truth now: the handoff spec. Previously this dug boxes out of
    two differently-shaped arrays in a separate manifest."""
    return {e["id"]: e["box_xyxy"] for e in handoff["spec"]}


def route_map(image, boxes, verdicts, path):
    dim = Image.fromarray((__import__("numpy").array(image) * 0.35).astype("uint8"))
    canvas = dim.convert("RGB")
    d = ImageDraw.Draw(canvas)
    # draw css first so assets land on top
    for want in reversed(ORDER):
        for rid, box in boxes.items():
            v = verdicts.get(str(rid)) or verdicts.get(rid) or {}
            if (v.get("verdict") or "?") != want:
                continue
            x0, y0, x1, y1 = box
            c = COLOR[want]
            d.rectangle([x0, y0, x1, y1], outline=c, width=3)
            tag = str(rid)
            d.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(tag) + 8, y0], fill=c)
            d.text((x0 + 4, max(0, y0 - 14)), tag, fill=(0, 0, 0))
    # legend
    present = [n for n in ORDER if any(v.get("verdict") == n for v in verdicts.values())]
    d.rectangle([8, 8, 320, 8 + 22 * len(present) + 8], fill=(15, 15, 20))
    for k, name in enumerate(present):
        n = sum(1 for v in verdicts.values() if v.get("verdict") == name)
        d.rectangle([16, 16 + 22 * k, 34, 30 + 22 * k], fill=COLOR[name])
        d.text((42, 18 + 22 * k), f"{name}  ({n})", fill=(240, 240, 240))
    canvas.save(path)


def route_sheet(image, boxes, verdicts, path, cell=140, cols=7):
    groups = {k: [] for k in ORDER}
    for rid, box in boxes.items():
        v = verdicts.get(str(rid)) or verdicts.get(rid) or {}
        groups[v.get("verdict") or "?"].append((rid, box, v))
    for g in groups.values():
        g.sort(key=lambda t: -(t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))

    rowh = cell + 46
    total_rows = sum((len(g) + cols - 1) // cols for g in groups.values() if g)
    headers = sum(1 for g in groups.values() if g)
    H = total_rows * rowh + headers * 34 + 20
    W = cols * (cell + 10) + 20
    sheet = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(sheet)

    y = 10
    for name in ORDER:
        g = groups[name]
        if not g:
            continue
        d.rectangle([10, y, W - 10, y + 26], fill=COLOR[name])
        d.text((18, y + 7), f"{name.upper()}  —  {len(g)} regions", fill=(0, 0, 0))
        y += 34
        for k, (rid, box, v) in enumerate(g):
            if k and k % cols == 0:
                y += rowh
            cx = 10 + (k % cols) * (cell + 10)
            crop = image.crop(tuple(box))
            crop.thumbnail((cell, cell), Image.LANCZOS)
            d.rectangle([cx, y, cx + cell, y + cell], fill=(34, 34, 40))
            sheet.paste(crop, (cx + (cell - crop.width) // 2,
                               y + (cell - crop.height) // 2))
            d.text((cx + 3, y + cell + 2), f"#{rid}", fill=COLOR[name])
            note = v.get("reason") or ""
            for li, line in enumerate(textwrap.wrap(note, 22)[:2]):
                d.text((cx + 3, y + cell + 15 + li * 11), line, fill=(165, 165, 175))
        y += rowh
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("handoff", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.handoff.parent
    image = Image.open(args.image).convert("RGB")
    handoff = json.loads(args.handoff.read_text())
    boxes = boxes_from(handoff)
    verdicts = {str(e["id"]): {"verdict": e["class"],
                               "reason": (e.get("subject") or "")}
                for e in handoff["spec"]}

    route_map(image, boxes, verdicts, out / "route_map.png")
    route_sheet(image, boxes, verdicts, out / "route_sheet.png")
    print(f"wrote {out}/route_map.png and {out}/route_sheet.png")


if __name__ == "__main__":
    main()
