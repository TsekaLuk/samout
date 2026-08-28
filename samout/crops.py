"""Pixel access for regions — the one place that knows where a region's image
comes from.

This exists so `observe` does not. Previously a `cut_dir` argument was threaded
through observe -> batch -> contact_sheet, which meant the observation stage knew
about file layout, mask cutouts and compositing. Now it receives `[(id, Image)]`
and knows nothing else, and swapping in a different pixel source touches one file.

Which source is used matters more than it sounds: sending bounding-box crops made
the observer report `bevel` and `glow` from the card photograph *behind* a play
button, worth 11 points of accuracy. See INSIGHTS aha #10.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from . import matting
from .matte import cut, decontaminate


def write_cutouts(image, regions, out_dir, refine=True, use_matting=True,
                  device="cpu"):
    """RGBA cutouts — the asset deliverable and the observation input.

    Written for EVERY region, not just the ones that look like art: the observer
    needs a clean view of the flat glyphs too, and those are exactly the ones whose
    backgrounds were misleading it.

    With `refine`, uses the continuous mask SAM 3 emits rather than the binarised
    one, then decontaminates the boundary and trims to the true extent. Without it,
    sprites carry staircase edges and a rim of whatever they were lifted off.
    Returns (written, matted, refined) so the caller can see which path each
    region took.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    written = refined = matted = 0
    can_matte = use_matting and matting.available()

    for r in regions:
        rgba = None
        # Preferred path: SAM 3 says which object, ViTMatte solves the edge.
        if can_matte and r.alpha is not None:
            try:
                a_hi, box = matting.matte(rgb, r.alpha, r.box, device=device)
                if a_hi is not None:
                    rgba, tight = _compose(rgb, a_hi, box)
                    if rgba is not None:
                        matted += 1
            except Exception:
                rgba = None   # any matting failure falls through to the mask path
        if rgba is None and refine and r.alpha is not None:
            rgba, tight = cut(rgb, r.alpha, r.box)
            if rgba is not None:
                refined += 1
        if rgba is None:
            if r.mask is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in r.box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            # Slice first, then stack. Building a full-canvas RGBA and slicing it
            # allocated ~6MB per region for a few KB of output.
            rgba = np.dstack([rgb[y0:y1, x0:x1],
                              (r.mask[y0:y1, x0:x1] * 255).astype(np.uint8)])
            tight = (x0, y0, x1, y1)

        Image.fromarray(rgba, "RGBA").save(out_dir / f"{r.id:02d}_cutout.png")
        image.crop(tight).save(out_dir / f"{r.id:02d}_context.png")
        written += 1
    return written, matted, refined


def _compose(rgb_u8, alpha, box, min_alpha=0.02):
    """Alpha + source pixels -> a tight RGBA sprite, background bleed removed."""
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(alpha > min_alpha)
    if len(ys) == 0:
        return None, None
    ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = alpha[ty0:ty1, tx0:tx1]
    rgb = rgb_u8[y0 + ty0:y0 + ty1, x0 + tx0:x0 + tx1].astype(np.float32) / 255.0
    # Still needed with a good matte: this fixes colour, not coverage. A boundary
    # pixel is a blend of object and backdrop, so a gold crown lifted off a dark UI
    # keeps a dark rim until the backdrop's contribution is subtracted out.
    rgb = decontaminate(rgb, a)
    rgba = np.dstack([np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
                      np.clip(a * 255.0, 0, 255).astype(np.uint8)])
    return rgba, (x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1)


class CropSource:
    """Resolves a region id to the image the observer should look at."""

    def __init__(self, image, cutout_dir=None, use_cutouts=True,
                 backdrop=(128, 128, 128)):
        self.image = image.convert("RGB")
        self.dir = Path(cutout_dir) if cutout_dir else None
        self.use_cutouts = use_cutouts
        self.backdrop = backdrop

    def cutout_path(self, rid):
        return self.dir / f"{rid:02d}_cutout.png" if self.dir else None

    def get(self, region):
        """Masked cutout on a neutral backdrop when available, else the box crop.

        Neutral grey rather than white or black: both of those imply a lighting
        direction, and lighting is one of the things being observed.
        """
        if self.use_cutouts and self.dir:
            p = self.cutout_path(region.id)
            if p.exists():
                cut = Image.open(p).convert("RGBA")
                flat = Image.new("RGB", cut.size, self.backdrop)
                flat.paste(cut, (0, 0), cut)
                return flat
        return self.image.crop(tuple(int(v) for v in region.box))

    def pairs(self, regions):
        """[(id, Image)] — the only shape `observe` ever sees."""
        return [(r.id, self.get(r)) for r in regions]
