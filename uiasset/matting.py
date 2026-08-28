"""Alpha matting with ViTMatte, using the SAM 3 mask as a trimap.

Segmentation and matting are different tasks and this module exists to stop
conflating them. SAM 3 answers *which object*; its mask head predicts at a low
internal resolution and is bilinearly upsampled, so on a 60px icon the boundary
smears across a third of the sprite. Sharpening that ramp back up — which is what
the first attempt did — is a filter compensating for the wrong tool, not a fix.

ViTMatte answers *how opaque is this pixel*, which is what it was trained for. The
two compose cleanly because a matting model wants a trimap, and a segmentation mask
is exactly where a trimap comes from:

    erode(mask)          -> definitely foreground
    outside dilate(mask) -> definitely background
    the band between     -> unknown, solve here

SAM 3 keeps authority over *what* is being cut out; ViTMatte only decides the edge.
That ordering matters — a standalone background remover (BiRefNet, RMBG) would
re-decide which object is salient inside the crop and can disagree with the region
the pipeline actually asked for.

Falls back to `matte.cut` when the model is unavailable, so this stays optional.

  ViTMatte — Yao et al., "Boosting Image Matting with Pretrained Plain Vision
    Transformers" (2023). `hustvl/vitmatte-small-composition-1k`, Apache-2.0.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_DIR = Path("models/vitmatte-small")
HUB_ID = "hustvl/vitmatte-small-composition-1k"

_MODEL = None
_PROC = None


def available(model_dir=None):
    d = Path(model_dir or DEFAULT_DIR)
    return (d / "model.safetensors").exists()


def _load(model_dir=None, device="cpu"):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        src = str(model_dir or DEFAULT_DIR) if available(model_dir) else HUB_ID
        _PROC = VitMatteImageProcessor.from_pretrained(src)
        _MODEL = VitMatteForImageMatting.from_pretrained(src).to(device).eval()
    return _MODEL, _PROC


def trimap_from_mask(alpha, band=None, fg_thresh=0.65, bg_thresh=0.35):
    """Soft mask -> trimap in {0, 0.5, 1}.

    The unknown band has to be wide enough to contain the true edge and narrow
    enough to leave the matting model a real constraint. Scaling it with the
    object's size keeps both true for a 40px icon and an 800px banner.
    """
    solid = alpha >= fg_thresh
    if not solid.any():
        solid = alpha >= 0.5
    if band is None:
        # ~4% of the object's smaller dimension, clamped to a sane pixel range.
        ys, xs = np.nonzero(alpha > 0.1)
        if len(ys) == 0:
            return np.zeros_like(alpha)
        extent = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        band = int(np.clip(round(extent * 0.04), 2, 12))

    fg = _erode(solid, band)
    bg = ~_dilate(alpha >= bg_thresh, band)
    tri = np.full(alpha.shape, 0.5, dtype=np.float32)
    tri[fg] = 1.0
    tri[bg] = 0.0
    return tri


def _erode(m, it):
    e = m
    for _ in range(it):
        p = np.pad(e, 1, constant_values=False)
        e = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return e


def _dilate(m, it):
    d = m
    for _ in range(it):
        p = np.pad(d, 1, constant_values=False)
        d = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    return d


def matte(image_rgb, alpha, box, model_dir=None, device="cpu", pad=6, min_side=64):
    """-> (alpha_hi float32 HxW crop, tight box) for one region, or (None, None).

    Works on a padded crop rather than the whole page: matting is expensive, and
    the model needs to see the edge context, not the rest of the screen.
    """
    H, W = alpha.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None

    a = alpha[y0:y1, x0:x1].astype(np.float32)
    tri = trimap_from_mask(a)
    if not (tri == 0.5).any():
        return a, (x0, y0, x1, y1)   # nothing ambiguous; the mask is already the answer

    crop = image_rgb[y0:y1, x0:x1]
    # ViTMatte is a ViT with a patch stride; tiny crops give it nothing to work with,
    # so upscale small icons before matting and bring the result back down.
    scale = max(1.0, min_side / max(1, min(crop.shape[0], crop.shape[1])))
    if scale > 1.0:
        nh, nw = int(round(crop.shape[0] * scale)), int(round(crop.shape[1] * scale))
        crop = np.array(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS))
        tri = np.array(Image.fromarray((tri * 255).astype(np.uint8))
                       .resize((nw, nh), Image.NEAREST), dtype=np.float32) / 255.0

    model, proc = _load(model_dir, device)
    inputs = proc(images=Image.fromarray(crop), trimaps=Image.fromarray(
        (tri * 255).astype(np.uint8)), return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).alphas[0, 0].detach().cpu().numpy()

    pred = pred[:crop.shape[0], :crop.shape[1]]
    if scale > 1.0:
        pred = np.array(Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8))
                        .resize((x1 - x0, y1 - y0), Image.LANCZOS),
                        dtype=np.float32) / 255.0
    return np.clip(pred, 0.0, 1.0), (x0, y0, x1, y1)
