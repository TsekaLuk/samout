"""Extract non-CSS-reproducible art assets from a UI mockup.

Pipeline:
  1. SAM 3 concept segmentation over an asset-oriented prompt set
  2. cross-prompt NMS -> one region per physical thing, carrying every label that hit it
  3. containment analysis -> drop layout containers, keep atomic art
  4. texture classification -> photo / 3d_render / flat_ui / text
  5. export RGBA cutouts (mask as alpha) + a numbered context map + assets.json

The point is the manifest + cutouts: feed them to an image model (gpt-image-2,
qwen-image-3-pro, grok-imagine-image-2.0) to regenerate clean assets, then hand
those plus the CSS-reproducible layout to a coding agent.

    python extract_assets.py input/ktv_ui.jpeg --out assets
"""

import argparse
import colorsys
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_ID = "models/sam3" if Path("models/sam3/model.safetensors").exists() else "jetjodh/sam3"

# Prompts chosen to surface raster art, not layout. "thumbnail" catches image
# slots; "3d icon"/"illustration" catch rendered art; "photo"/"portrait" catch
# photography. Layout words ("card", "panel") are deliberately absent — they
# score 0 on SAM 3 anyway and would only add container noise.
ASSET_PROMPTS = [
    "icon",
    "3d icon",   # strongest discriminator: fires on shaded/rendered icons,
                 # skips flat play buttons and the search glyph (both CSS-able)
    "photo",
    "thumbnail",
    "logo",
    "avatar",
    "neon sign",
]
# Scored 0 at every threshold — SAM 3 wants concrete nouns, not style words:
#   illustration, sticker, portrait, cartoon character, glossy 3d object,
#   card, panel, text, chinese text, navigation bar

# Texture class -> whether a browser can draw it without a raster asset.
CSS_REPRODUCIBLE = {
    "flat_ui": True,     # solid fills, borders, simple gradients, glyph icons
    "text": True,        # font rendering
    "3d_render": False,  # shaded/rendered illustration, needs alpha
    "logotype": False,   # stylized wordmark
    "photo": False,      # photography / rendered scene
}

# Label thresholds for the classifier. These are the real discriminators —
# texture statistics are NOT, because everything on a dark neon UI reads as
# high-entropy. Measured on the KTV screen: the 搜索 pill (n_colors=966,
# entropy=7.27) and the 3D podium icon (n_colors=626, entropy=7.32) are
# statistically indistinguishable, but SAM 3 fires "3d icon" on only one.
T_PHOTO = 0.60      # "photo" -> real imagery
T_AVATAR = 0.45     # "avatar" -> portrait crop
T_NEON = 0.35       # "neon sign" -> glow art, usually inside a larger scene
T_3D = 0.35         # "3d icon" -> shaded icon; absent on flat CSS-able glyphs
T_LOGO = 0.35       # "logo" -> wordmark, only when nothing stronger hit

# Seed text for the image model, per asset class. Concatenate with the crop,
# the palette, and whatever product copy the region carries.
PROMPT_HINT = {
    "photo": ("photographic scene, same subject framing and lighting as the "
              "reference, no text, no UI chrome, no watermark"),
    "3d_render": ("glossy 3D rendered icon on a fully transparent background, "
                  "soft studio lighting, same silhouette and material as the "
                  "reference, centered, no drop shadow baked in"),
    "logotype": ("stylized wordmark on a fully transparent background, "
                 "identical letterforms and gradient to the reference, "
                 "crisp edges, no background glow"),
}


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (area_a + area_b - inter)


def contains(outer, inner, tol=0.92):
    """Fraction of `inner`'s area that falls inside `outer`."""
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    cx0, cy0 = max(ox0, ix0), max(oy0, iy0)
    cx1, cy1 = min(ox1, ix1), min(oy1, iy1)
    cw, ch = max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)
    inner_area = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
    return (cw * ch) / inner_area >= tol


def describe_texture(rgb, mask):
    """Cheap texture stats over the masked pixels."""
    px = rgb[mask]
    if len(px) < 32:
        return None

    # Unique colors at 5 bits/channel — flat UI collapses to a handful,
    # photography stays in the thousands.
    q = (px >> 3).astype(np.int32)
    n_colors = len(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]))

    gray = px.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hist = np.bincount(gray.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    mx = px.max(axis=1).astype(np.float32)
    mn = px.min(axis=1).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    return {
        "n_colors": int(n_colors),
        "entropy": round(entropy, 3),
        "sat_mean": round(float(sat.mean()), 3),
        "sat_std": round(float(sat.std()), 3),
        "px": int(len(px)),
    }


def edge_density(rgb, mask, box):
    """Sobel-ish edge fraction inside the box — text and flat vector art spike."""
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = rgb[y0:y1, x0:x1]
    if sub.size == 0 or min(sub.shape[:2]) < 3:
        return 0.0
    g = sub.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    sm = mask[y0:y1, x0:x1]
    if sm.sum() < 16:
        return 0.0
    return round(float((mag[sm] > 28).mean()), 3)


def classify(labels):
    """photo | 3d_render | logotype | flat_ui, decided by which concept hit.

    Priority matters: a region can be both "icon" and "3d icon"; the shaded
    reading wins. Plain "icon" with nothing else is a glyph — CSS draws it.
    "thumbnail" alone is deliberately NOT enough: it fires on layout panels
    (the 会员专享 bar) as readily as on real image slots.
    """
    if labels.get("photo", 0) >= T_PHOTO or labels.get("avatar", 0) >= T_AVATAR:
        return "photo"
    if labels.get("neon sign", 0) >= T_NEON:
        return "3d_render"
    if labels.get("3d icon", 0) >= T_3D:
        return "3d_render"
    if labels.get("logo", 0) >= T_LOGO:
        return "logotype"
    return "flat_ui"


def dominant_colors(rgb, mask, k=5):
    px = rgb[mask]
    if len(px) == 0:
        return []
    q = (px >> 4) << 4  # 4-bit buckets
    keys = [tuple(int(v) for v in c) for c in q[:: max(1, len(q) // 4000)]]
    return ["#%02x%02x%02x" % c for c, _ in Counter(keys).most_common(k)]


def palette(n):
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--prompts", nargs="*", default=ASSET_PROMPTS)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.55)
    ap.add_argument("--min-area", type=int, default=600,
                    help="drop regions smaller than this many mask pixels")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    from transformers import Sam3Model, Sam3Processor

    cut_dir = args.out / "cutouts"
    cut_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size
    rgb = np.array(image, dtype=np.uint8)

    device = pick_device()
    print(f"device={device}  image={W}x{H}  model={args.model}")

    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()

    # ---- 1. detect --------------------------------------------------------
    raw = []
    for prompt in args.prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=args.threshold,
            mask_threshold=args.mask_threshold, target_sizes=[(H, W)])[0]
        masks = res["masks"].cpu().numpy().astype(bool)
        scores = res["scores"].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()
        for m, s, b in zip(masks, scores, boxes):
            # A near-full-canvas hit is the model latching onto the whole
            # screenshot; it would become every other region's parent.
            if m.sum() >= args.min_area and m.sum() <= 0.8 * W * H:
                raw.append({"label": prompt, "score": float(s),
                            "box": [float(v) for v in b], "mask": m})
        print(f"  {prompt:<14} -> {len(masks)} hits")

    # ---- 2. cross-prompt NMS ---------------------------------------------
    raw.sort(key=lambda r: r["score"], reverse=True)
    merged = []
    for r in raw:
        for m in merged:
            if iou(r["box"], m["box"]) >= args.nms_iou:
                m["labels"][r["label"]] = max(
                    m["labels"].get(r["label"], 0.0), r["score"])
                break
        else:
            merged.append({"box": r["box"], "mask": r["mask"],
                           "score": r["score"],
                           "labels": {r["label"]: r["score"]}})
    print(f"\n{len(raw)} detections -> {len(merged)} unique regions after NMS")

    # ---- 3. classify + nest ----------------------------------------------
    for m in merged:
        stats = describe_texture(rgb, m["mask"])
        m["stats"] = stats or {}
        m["edges"] = edge_density(rgb, m["mask"], m["box"]) if stats else 0.0
        m["texture"] = classify(m["labels"])

    # Largest first, so an asset's id is always lower than its children's.
    merged.sort(key=lambda m: -(m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]))
    for i, m in enumerate(merged):
        m["idx"] = i

    # Nesting: the banner is one asset, but "LET'S SING" inside it is another.
    # Record the tree instead of picking for the caller — regenerating the whole
    # banner and regenerating its neon sign separately are both valid, and which
    # you want depends on how the coding agent plans to compose the page.
    for m in merged:
        m["children"] = [o["idx"] for o in merged
                         if o is not m and contains(m["box"], o["box"])]
    for m in merged:
        parents = [o for o in merged if m["idx"] in o["children"]]
        m["parent"] = min(parents, key=lambda o: (o["box"][2] - o["box"][0]) *
                          (o["box"][3] - o["box"][1]))["idx"] if parents else None

    assets = [m for m in merged if not CSS_REPRODUCIBLE[m["texture"]]]
    css_ok = [m for m in merged if CSS_REPRODUCIBLE[m["texture"]]]

    # ---- 4. export cutouts + manifest ------------------------------------
    manifest = {
        "source": str(args.image),
        "size": {"width": W, "height": H},
        "summary": {
            "regions": len(merged),
            "art_assets": len(assets),
            "css_reproducible": len(css_ok),
        },
        "assets": [],
        "css_regions": [],
    }

    for m in assets:
        i = m["idx"]
        x0, y0, x1, y1 = [int(round(v)) for v in m["box"]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        rgba = np.dstack([rgb, (m["mask"] * 255).astype(np.uint8)])
        Image.fromarray(rgba[y0:y1, x0:x1], "RGBA").save(cut_dir / f"{i:02d}_cutout.png")
        image.crop((x0, y0, x1, y1)).save(cut_dir / f"{i:02d}_context.png")

        alpha = m["texture"] in ("3d_render", "logotype")
        manifest["assets"].append({
            "id": i,
            "texture": m["texture"],
            "labels": {k: round(v, 3) for k, v in sorted(
                m["labels"].items(), key=lambda kv: -kv[1])},
            "box_xyxy": [x0, y0, x1, y1],
            "size_px": [x1 - x0, y1 - y0],
            "box_pct": [round(x0 / W, 4), round(y0 / H, 4),
                        round(x1 / W, 4), round(y1 / H, 4)],
            "parent": m["parent"],
            "children": m["children"],
            "mask_area_px": int(m["mask"].sum()),
            "silhouette_fill": round(float(m["mask"].sum()) / max(1, (x1 - x0) * (y1 - y0)), 3),
            "dominant_colors": dominant_colors(rgb, m["mask"]),
            "texture_stats": m["stats"],
            "edge_density": m["edges"],
            "cutout": f"cutouts/{i:02d}_cutout.png",
            "context_crop": f"cutouts/{i:02d}_context.png",
            # Everything below is what you hand to the image model.
            "regen": {
                "needs_transparency": alpha,
                "output_format": "png_rgba" if alpha else "png",
                "target_size_px": [(x1 - x0) * 3, (y1 - y0) * 3],  # @3x for retina
                "reference": f"cutouts/{i:02d}_cutout.png",
                "palette": dominant_colors(rgb, m["mask"], 3),
                "prompt_hint": PROMPT_HINT[m["texture"]],
            },
        })

    # Export a masked cutout for EVERY region, not just the asset-class ones.
    # The downstream observer must see the object without its backdrop: reading a
    # bounding-box crop made it report glow and bevel belonging to the card photo
    # behind a play button, which was 53% of all classification error.
    for m in css_ok:
        i = m["idx"]
        x0, y0, x1, y1 = [int(round(v)) for v in m["box"]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            rgba = np.dstack([rgb, (m["mask"] * 255).astype(np.uint8)])
            Image.fromarray(rgba[y0:y1, x0:x1], "RGBA").save(
                cut_dir / f"{i:02d}_cutout.png")

    for m in css_ok:
        x0, y0, x1, y1 = [int(round(v)) for v in m["box"]]
        manifest["css_regions"].append({
            "id": m["idx"],
            "texture": m["texture"],
            "labels": {k: round(v, 3) for k, v in sorted(
                m["labels"].items(), key=lambda kv: -kv[1])},
            "box_xyxy": [x0, y0, x1, y1],
            "parent": m["parent"],
            "note": "reproduce with CSS/HTML, no raster asset needed",
        })

    # ---- 5. numbered context map -----------------------------------------
    dim = (np.array(image, dtype=np.float32) * 0.4).astype(np.uint8)
    canvas = Image.fromarray(dim)
    canvas.paste(image, (0, 0), Image.fromarray(
        (np.logical_or.reduce([m["mask"] for m in assets]) * 255).astype(np.uint8)
        if assets else np.zeros((H, W), np.uint8), "L"))
    d = ImageDraw.Draw(canvas)
    tex_color = {"photo": (0, 230, 255), "3d_render": (255, 210, 0),
                 "logotype": (255, 90, 200)}
    for m in assets:
        x0, y0, x1, y1 = [int(round(v)) for v in m["box"]]
        color = tex_color[m["texture"]]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = str(m["idx"])
        d.rectangle([x0, max(0, y0 - 17), x0 + 9 * len(tag) + 8, y0], fill=color)
        d.text((x0 + 4, max(0, y0 - 15)), tag, fill=(0, 0, 0))
    canvas.save(args.out / "asset_map.png")

    (args.out / "assets.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    by_tex = Counter(m["texture"] for m in merged)
    print(f"\ntexture mix: {dict(by_tex)}")
    print(f"art assets : {len(assets)}  (exported to {cut_dir}/)")
    print(f"css regions: {len(css_ok)}")
    print(f"map        : {args.out}/asset_map.png")


if __name__ == "__main__":
    main()
