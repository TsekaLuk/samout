"""Pixel-level cutout quality: soft alpha, colour decontamination, tight bounds.

The default path throws away information SAM 3 actually produces. Its mask head
emits a continuous probability per pixel; `post_process_instance_segmentation`
bilinearly resizes it and then binarises on the last line. Taking the binary result
gives sprites with 2 alpha values and 0% soft-edge pixels — staircase edges, and a
dark rim wherever a bright object met a dark backdrop, because boundary pixels are
blends of the two and binary alpha keeps them fully opaque.

Three separate defects, three separate fixes:

  soft alpha        keep the probability instead of thresholding it -> antialiased
  decontamination   boundary pixels are a*F + (1-a)*B; solve for F -> no dark rim
  tight bounds      trim to where alpha actually is, drop speckle

Composited over a light background, the difference is the whole thing: a binary
cutout of a gold crown on a dark UI carries a black halo into every future use.
"""

import numpy as np
import torch


def soft_masks(outputs, keep_mask, target_size):
    """Reproduce the processor's mask path up to — but not including — binarisation.

    Mirrors `Sam3ImageProcessor.post_process_instance_segmentation`: sigmoid, then
    bilinear resize to the source resolution. Returns float32 in [0, 1].
    """
    masks = outputs.pred_masks.sigmoid()[0][keep_mask]
    if len(masks) == 0:
        return masks.cpu().numpy()
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    return masks.detach().cpu().float().numpy()


def _dilate(m, iterations=1):
    d = m
    for _ in range(iterations):
        p = np.pad(d, 1, constant_values=False)
        d = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    return d


def largest_component(binary):
    """Keep the biggest connected blob. Flood fill on a small crop is cheap enough
    that the lack of scipy is not worth a dependency."""
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    best, best_size = None, 0
    for sy in range(h):
        for sx in range(w):
            if not binary[sy, sx] or seen[sy, sx]:
                continue
            stack, comp = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > best_size:
                best, best_size = comp, len(comp)
    out = np.zeros_like(binary)
    if best:
        ys, xs = zip(*best)
        out[np.array(ys), np.array(xs)] = True
    return out


def decontaminate(rgb, alpha, bg_iters=3):
    """Undo background bleed at the boundary.

    A partially covered pixel observes `C = a*F + (1-a)*B`. We know C and a, and can
    estimate B from the nearby fully-transparent pixels, so F follows. Without this a
    gold crown lifted off a dark UI keeps a black rim, which then shows against any
    lighter background it is later placed on.

    `rgb` float32 HxWx3 in [0,1], `alpha` float32 HxW in [0,1]. Returns corrected rgb.
    """
    out = rgb.copy()
    edge = (alpha > 0.02) & (alpha < 0.98)
    if not edge.any():
        return out

    # Local background estimate: blur the known-background pixels outward.
    bg_known = alpha < 0.02
    if not bg_known.any():
        return out
    bg = np.zeros_like(rgb)
    weight = np.zeros(alpha.shape, dtype=np.float32)
    bg[bg_known] = rgb[bg_known]
    weight[bg_known] = 1.0
    for _ in range(bg_iters):
        for arr in (bg, weight):
            pass
        # 4-neighbour box blur, weight-normalised so unknown pixels borrow from known
        def shift_sum(a):
            p = np.pad(a, ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2), mode="edge")
            if a.ndim == 3:
                return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
                        + p[1:-1, 1:-1])
            return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
                    + p[1:-1, 1:-1])
        bg, weight = shift_sum(bg), shift_sum(weight)
        nz = weight > 0
        bg[nz] /= weight[nz][:, None]
        weight[nz] = 1.0

    a = alpha[edge][:, None]
    fg = (rgb[edge] - (1.0 - a) * bg[edge]) / np.maximum(a, 0.08)
    out[edge] = np.clip(fg, 0.0, 1.0)
    return out


def sharpen_alpha(a, contrast=4.0, pivot=0.5):
    """Narrow the alpha ramp without going binary.

    SAM 3 predicts its mask at a low internal resolution and it is bilinearly
    upsampled to source size, so on a 60px icon the transition smears across a third
    of the sprite — 54-76% of pixels came out partially transparent. That width is an
    upsampling artifact, not coverage information: a real object boundary is
    antialiased over roughly one pixel. Rescaling around the 0.5 iso-line keeps the
    subpixel edge and drops the mush.

    `contrast` is the only knob: 1.0 leaves the ramp alone, large values approach the
    binary mask we started from. 4.0 lands the soft-edge fraction near 15%, which is
    what genuine antialiasing looks like.
    """
    return np.clip((a - pivot) * contrast + pivot, 0.0, 1.0)


def cut(rgb_u8, alpha, box, min_alpha=0.06, clean=True, decontam=True, pad=1,
        contrast=4.0):
    """-> (RGBA uint8 array, tight box) or (None, None) if nothing survives.

    `rgb_u8` is the full source image, `alpha` the full-size soft mask, `box` the
    detection box used as a search window.
    """
    H, W = alpha.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None, None

    a = sharpen_alpha(alpha[y0:y1, x0:x1].astype(np.float32), contrast)
    solid = a >= 0.5
    if clean and solid.any():
        # Speckle is a real failure mode here: a mask can pick up a stray blob of
        # background that shares the object's colour. Keeping only the main blob,
        # dilated so its own soft halo survives, removes it without eroding the edge.
        keep = _dilate(largest_component(solid), 2)
        a = np.where(keep, a, 0.0)

    a[a < min_alpha] = 0.0
    if not (a > 0).any():
        return None, None

    # Tight bounds: trim to where alpha actually is, not where the detector guessed.
    ys, xs = np.nonzero(a > 0)
    ty0, ty1 = ys.min(), ys.max() + 1
    tx0, tx1 = xs.min(), xs.max() + 1
    a = a[ty0:ty1, tx0:tx1]

    rgb = rgb_u8[y0 + ty0:y0 + ty1, x0 + tx0:x0 + tx1].astype(np.float32) / 255.0
    if decontam:
        rgb = decontaminate(rgb, a)

    rgba = np.dstack([np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
                      np.clip(a * 255.0, 0, 255).astype(np.uint8)])
    return rgba, (x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1)
