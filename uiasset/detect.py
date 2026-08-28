"""Stage 1: SAM 3 segmentation -> list[Region].

This stage detects and nothing else. The version it replaces also carried a
heuristic classifier (colour counts, entropy, `T_3D` thresholds) that decided
asset-vs-CSS inline; that logic is gone, superseded by `taxonomy.py`. Detection
now has one job and no opinion about what it found.

What it knows about SAM 3, from BENCHMARK.md §1: prompts must be concrete physical
nouns. Layout words return nothing at any threshold, because SAM 3 segments things
and a container is not a thing.
"""

import colorsys

import numpy as np
import torch

from .matte import soft_masks
from .model import Region, iou


def _keep_mask(outputs, threshold):
    """The processor's score filter, reproduced so the soft masks line up with the
    binary ones it returned."""
    scores = outputs.pred_logits.sigmoid()
    if getattr(outputs, "presence_logits", None) is not None:
        scores = scores * outputs.presence_logits.sigmoid()
    return (scores > threshold)[0]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def texture_stats(rgb, mask):
    """Descriptive only — kept in the manifest as reference data, never used to
    classify. On a neon UI these numbers are *inverted* relative to intuition: the
    flat CSS pill scores higher entropy than the rendered 3D icon. See INSIGHTS
    aha #2 for the table."""
    px = rgb[mask]
    if len(px) < 32:
        return {}
    q = (px >> 3).astype(np.int32)
    n_colors = len(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]))
    gray = px.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hist = np.bincount(gray.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    mx, mn = px.max(axis=1).astype(np.float32), px.min(axis=1).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {"n_colors": int(n_colors),
            "entropy": round(float(-(p * np.log2(p)).sum()), 3),
            "sat_mean": round(float(sat.mean()), 3),
            "sat_std": round(float(sat.std()), 3)}


def dominant_colors(rgb, mask, k=5):
    px = rgb[mask]
    if len(px) == 0:
        return []
    from collections import Counter
    q = (px >> 4) << 4
    keys = [tuple(int(v) for v in c) for c in q[:: max(1, len(q) // 4000)]]
    return ["#%02x%02x%02x" % c for c, _ in Counter(keys).most_common(k)]


def palette(n):
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.8, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def detect(image, cfg, model_id=None, verbose=True):
    """-> list[Region], sorted largest first so ids are stable and a parent's id
    is always lower than its children's."""
    from transformers import Sam3Model, Sam3Processor

    from .config import Config

    W, H = image.size
    device = pick_device()
    model_id = model_id or Config.model_id()
    if verbose:
        print(f"device={device}  image={W}x{H}  model={model_id}")

    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()

    # The image is identical for every prompt, so encode it once. Running the vision
    # encoder per prompt made detection 81% of local wall clock (7 x 3.3s); only the
    # text side actually varies. `forward` takes either `pixel_values` or a
    # precomputed `vision_embeds`, and requires exactly one of them.
    first = processor(images=image, text=cfg.prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        vision_embeds = model.vision_encoder(first["pixel_values"])

    raw = []
    for prompt in cfg.prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        text_only = {k: v for k, v in inputs.items() if k != "pixel_values"}
        with torch.no_grad():
            outputs = model(vision_embeds=vision_embeds, **text_only)
        res = processor.post_process_instance_segmentation(
            outputs, threshold=cfg.threshold,
            mask_threshold=cfg.mask_threshold, target_sizes=[(H, W)])[0]
        masks = res["masks"].cpu().numpy().astype(bool)
        scores = res["scores"].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()

        # The processor binarises on its last line. Recover the continuous mask the
        # model actually emitted, using the same score filter, so cutouts can have
        # antialiased edges instead of a 2-value staircase.
        keep = _keep_mask(outputs, cfg.threshold)
        soft = soft_masks(outputs, keep, (H, W))

        kept = 0
        for i, (m, s, b) in enumerate(zip(masks, scores, boxes)):
            n = m.sum()
            # A near-full-canvas hit is the model latching onto the whole
            # screenshot; left in, it becomes every other region's parent.
            if cfg.min_area_px <= n <= cfg.max_area_frac * W * H:
                raw.append({"label": prompt, "score": float(s),
                            "box": tuple(float(v) for v in b), "mask": m,
                            "alpha": soft[i] if i < len(soft) else None})
                kept += 1
        if verbose:
            print(f"  {prompt:<12} {len(masks):>3} hits, {kept:>3} kept")

    # Cross-prompt NMS. `icon` and `logo` fire on the same pixels constantly, so
    # merge by box overlap and keep every label that landed — a region being both
    # "icon 0.91" and "3d icon 0.80" is information, not duplication.
    raw.sort(key=lambda r: -r["score"])
    merged = []
    for r in raw:
        for m in merged:
            if iou(r["box"], m["box"]) >= cfg.nms_iou:
                m["labels"][r["label"]] = max(m["labels"].get(r["label"], 0.0),
                                              r["score"])
                break
        else:
            merged.append({"box": r["box"], "mask": r["mask"], "alpha": r["alpha"],
                           "score": r["score"], "labels": {r["label"]: r["score"]}})

    merged.sort(key=lambda m: -(m["box"][2] - m["box"][0]) * (m["box"][3] - m["box"][1]))
    regions = []
    for i, m in enumerate(merged):
        r = Region(id=i, box=m["box"], labels=m["labels"], mask=m["mask"],
                   alpha=m.get("alpha"), score=m["score"])
        rows, cols = np.any(m["mask"], axis=1), np.any(m["mask"], axis=0)
        if rows.any():
            ys, xs = np.nonzero(rows)[0], np.nonzero(cols)[0]
            r.mask_box = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
        regions.append(r)
    if verbose:
        print(f"{len(raw)} detections -> {len(regions)} regions after NMS")
    return regions


def _erode(m, iterations=2):
    e = m
    for _ in range(iterations):
        p = np.pad(e, 1, constant_values=False)
        e = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return e


def _bbox_slice(region, W, H, pad=2):
    """Crop window for per-region work.

    Uses the MASK's extent, not the detection box — they differ, and using the box
    clips mask pixels and changes the statistics. Falls back to the full canvas if
    the extent was never computed, so correctness never depends on the optimisation.
    """
    mb = getattr(region, "mask_box", None)
    if mb is None:
        return slice(0, H), slice(0, W)
    x0, y0, x1, y1 = mb
    return (slice(max(0, y0 - pad), min(H, y1 + pad)),
            slice(max(0, x0 - pad), min(W, x1 + pad)))


def measure(image, regions):
    """Quantities computed from the pixels, used by the classifier.

    Distinct from `describe`, which is reference data nothing reads. These are
    inputs to `taxonomy.classify`. The split matters: anything computable should be
    measured here rather than asked of the VLM — `hue_count` was asked for, came
    back less accurate than measuring it, and then turned out not to separate the
    classes at all.

    Eroded before measuring so that antialiasing at the silhouette edge — which is
    an artifact of the mask, not a property of the object — does not dominate.
    """
    rgb_u8 = np.asarray(image.convert("RGB"))
    H, W = rgb_u8.shape[:2]
    out = {}
    for r in regions:
        if r.mask is None:
            continue
        # Crop before computing. Indexing a full-canvas array with a full-canvas
        # mask, once per region, costs n x canvas; the regions themselves only cover
        # a fraction of it. Measured on the test screen: 83.4M pixel-ops against
        # 1.13M of actual region area — 74x wasted, and it grows linearly with the
        # region count, so a dense screen makes it quadratic in practice.
        ys, xs = _bbox_slice(r, W, H)
        m = r.mask[ys, xs]
        if m.sum() < 40:
            continue
        inner = _erode(m, 2)
        if inner.sum() < 30:
            inner = m
        sub = rgb_u8[ys, xs].astype(np.float32)
        lum = sub @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mx = sub.max(2)
        sat = (mx - sub.min(2)) / np.maximum(mx, 1e-6)
        out[r.id] = {
            "lum_std_inner": round(float(lum[inner].std()), 2),
            "lum_mean_inner": round(float(lum[inner].mean()), 2),
            "sat_std_inner": round(float(sat[inner].std()), 4),
            "sat_mean_inner": round(float(sat[inner].mean()), 4),
            "inner_px": int(inner.sum()),
        }
    return out


def describe(image, regions):
    """Reference statistics per region, for the manifest. Not a classifier — on a
    neon UI these numbers are inverted relative to intuition (INSIGHTS aha #2)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    out = {}
    for r in regions:
        if r.mask is None:
            continue
        ys, xs = _bbox_slice(r, W, H, pad=0)          # crop before computing
        sub, m = rgb[ys, xs], r.mask[ys, xs]
        area = int(m.sum())
        out[r.id] = {"texture": texture_stats(sub, m),
                     "dominant_colors": dominant_colors(sub, m),
                     "mask_area_px": area,
                     "silhouette_fill": round(area / max(1, r.area), 3)}
    return out
