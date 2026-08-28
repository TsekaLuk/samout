"""Segment a UI screenshot with SAM 3 (promptable concept segmentation).

Runs one open-vocabulary text prompt at a time, collects every instance mask
above threshold, and writes per-concept overlays plus a combined overlay.

    python segment_ui.py input/ktv_ui.jpeg --out out
"""

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Local snapshot of the SAM 3 weights (mirrored from jetjodh/sam3, itself a
# copy of the gated facebook/sam3). Falls back to the Hub id if not present.
MODEL_ID = "models/sam3" if Path("models/sam3/model.safetensors").exists() else "jetjodh/sam3"

# Tuned on a KTV app home screen. Notes on wording:
#   "thumbnail" beats "card"/"panel" for UI containers — the latter two score 0,
#   presumably because SAM 3 reads them as playing card / instrument panel.
#   "text" and "chinese text" score 0 at any threshold; use OCR for the text layer.
DEFAULT_PROMPTS = [
    "icon",
    "button",
    "thumbnail",
    "photo",
    "avatar",
    "logo",
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def palette(n):
    """n visually distinct RGB colors."""
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def overlay(base, masks, colors, alpha=0.5):
    """Alpha-blend boolean masks onto a copy of `base` (PIL RGB)."""
    arr = np.array(base.convert("RGB"), dtype=np.float32)
    for mask, color in zip(masks, colors):
        m = mask.astype(bool)
        if not m.any():
            continue
        arr[m] = arr[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(arr.astype(np.uint8))


def draw_boxes(img, boxes, colors, labels):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for box, color, label in zip(boxes, colors, labels):
        x0, y0, x1, y1 = [float(v) for v in box]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tw = 7 * len(label) + 6
        d.rectangle([x0, max(0, y0 - 16), x0 + tw, y0], fill=color)
        d.text((x0 + 3, max(0, y0 - 15)), label, fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="detection score threshold")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    from transformers import Sam3Model, Sam3Processor

    args.out.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    device = pick_device()
    print(f"device={device}  image={W}x{H}  model={args.model}")

    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model).to(device).eval()

    manifest = {"image": str(args.image), "width": W, "height": H, "concepts": []}
    all_masks, all_colors, all_boxes, all_labels = [], [], [], []
    concept_colors = palette(len(args.prompts))

    for prompt, base_color in zip(args.prompts, concept_colors):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        res = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=[(H, W)],
        )[0]

        masks = res["masks"].cpu().numpy().astype(bool)
        scores = res["scores"].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()
        n = len(scores)
        print(f"  {prompt:<8} -> {n} instances")

        if n:
            inst_colors = palette(n)
            ov = overlay(image, masks, inst_colors)
            ov = draw_boxes(ov, boxes, inst_colors,
                            [f"{prompt} {s:.2f}" for s in scores])
            ov.save(args.out / f"concept_{prompt.replace(' ', '_')}.png")

            all_masks.extend(masks)
            all_colors.extend([base_color] * n)
            all_boxes.extend(boxes)
            all_labels.extend([f"{prompt} {s:.2f}" for s in scores])

        manifest["concepts"].append({
            "prompt": prompt,
            "count": int(n),
            "color": base_color,
            "instances": [
                {
                    "score": float(s),
                    "box_xyxy": [round(float(v), 1) for v in b],
                    "area_px": int(m.sum()),
                }
                for s, b, m in zip(scores, boxes, masks)
            ],
        })

    if all_masks:
        combined = overlay(image, all_masks, all_colors, alpha=0.55)
        combined = draw_boxes(combined, all_boxes, all_colors, all_labels)
        combined.save(args.out / "combined.png")

    (args.out / "regions.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))

    total = sum(c["count"] for c in manifest["concepts"])
    print(f"\n{total} instances total -> {args.out}/")


if __name__ == "__main__":
    main()
