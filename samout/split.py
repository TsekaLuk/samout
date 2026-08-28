"""Split bundled regions into container + inner glyph, with SAM 2 point prompts.

SAM 3 hands back one box for "translucent disc + white play triangle". Downstream
that is unfixable: the disc's shading gets attributed to the glyph, so seven flat
library icons were routed to an image model. It is not a matting problem and not a
rule problem — the region is simply wrong, and the fix belongs at the detection
stage.

The tool for it is already in the SAM family. An interactive segmenter prompted with
a point returns several masks at different granularities — whole object, part,
subpart — which is exactly the container/glyph decomposition we need. SAM 3's own
text-prompted head does not expose that; `Sam2Model` does, and is ungated.

    disc + glyph  --point at centroid-->  [whole disc, glyph, glyph outline]
                                           keep the smallest clean one as a child

Candidates are chosen by the same measurement that exposed the bug: a high interior
luminance variance means something is showing through the object, which is what a
translucent container over a busy backdrop looks like. So the feature that caused
four misclassifications is the one that now triggers the fix.
"""

from pathlib import Path

import numpy as np
import torch

DEFAULT_DIR = Path("models/sam2")
HUB_ID = "facebook/sam2.1-hiera-small"

_MODEL = None
_PROC = None


def available(model_dir=None):
    d = Path(model_dir or DEFAULT_DIR)
    return (d / "model.safetensors").exists()


def _load(model_dir=None, device="cpu"):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import Sam2Model, Sam2Processor
        src = str(model_dir or DEFAULT_DIR) if available(model_dir) else HUB_ID
        _PROC = Sam2Processor.from_pretrained(src)
        _MODEL = Sam2Model.from_pretrained(src).to(device).eval()
    return _MODEL, _PROC


def is_candidate(region, measured, min_lum_std=70.0, max_area=40000, min_area=400):
    """Worth trying to split?

    Restricted deliberately. Splitting a region that was correct costs a spurious
    child and a wrong parent class, so the gate is the signature we actually
    diagnosed rather than "try everything": a small region whose interior varies a
    lot is a container you can see through, not a solid object.
    """
    m = (measured or {}).get(region.id) or {}
    lum = m.get("lum_std_inner")
    if lum is None:
        return False
    return lum >= min_lum_std and min_area <= region.area <= max_area


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def propose(image, region, device="cpu", model_dir=None,
            min_frac=0.06, max_frac=0.40, min_contrast=60.0):
    # Calibrated against the observed separation, with margin on both axes:
    #   true splits  (5 play glyphs)  area_frac 0.177-0.247   contrast 132-190
    #   false split  (a wordmark cut
    #                 into two words) area_frac 0.537         contrast  26.2
    # Both gaps are wide, so the thresholds sit between them rather than on a
    # sample. Compare the `sat_std` term in OPTIMIZATION.md, which had a margin one
    # sample wide and was deliberately not shipped.
    """-> (inner_mask, info) or (None, reason).

    `inner_mask` is full-image sized so it drops straight into a Region.

    A returned mask has to earn it: cover a real but minority share of the parent,
    stay inside it, and differ in luminance from the rest of the parent. Without the
    contrast test the segmenter happily returns a slightly-eroded copy of the disc,
    which would split nothing and add a duplicate.
    """
    if region.mask is None:
        return None, "no mask"
    pt = _centroid(region.mask)
    if pt is None:
        return None, "empty mask"

    model, proc = _load(model_dir, device)
    inputs = proc(images=image, input_points=[[[list(pt)]]],
                  input_labels=[[[1]]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(
        out.pred_masks, inputs["original_sizes"])[0][0].cpu().numpy().astype(bool)

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    parent = region.mask
    parent_area = max(1, int(parent.sum()))

    best, best_info = None, None
    for k, m in enumerate(masks):
        inner = m & parent
        frac = inner.sum() / parent_area
        if not (min_frac <= frac <= max_frac):
            continue
        # Must sit inside the parent, not merely overlap it.
        if m.sum() and (inner.sum() / m.sum()) < 0.75:
            continue
        rest = parent & ~inner
        if rest.sum() < 20 or inner.sum() < 20:
            continue
        contrast = abs(float(lum[inner].mean()) - float(lum[rest].mean()))
        if contrast < min_contrast:
            continue
        info = {"granularity": k, "area_frac": round(float(frac), 3),
                "contrast": round(contrast, 1)}
        # Prefer the smallest qualifying mask: that is the glyph, not the container.
        if best is None or frac < best_info["area_frac"]:
            best, best_info = inner, info

    if best is None:
        return None, "no sub-object met the area/containment/contrast tests"
    return best, best_info


def _shape_differs(image, parent, child_mask, min_dist=0.35):
    """Is the child a different shape, or just a smaller copy of the parent?

    The signal-bars glyph "split" into a shrunken version of itself — a duplicate,
    not a decomposition. Area alone cannot tell those apart; the silhouettes can.
    Reuses the same contrast-normalised shape vector the component grouper uses, so
    "same shape" means the same thing in both places.
    """
    from .components import shape_vector, similarity

    ys, xs = np.nonzero(child_mask)
    if len(ys) == 0:
        return False
    cb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    pb = tuple(int(round(v)) for v in parent.box)
    a = shape_vector(image.crop(cb))
    b = shape_vector(image.crop(pb))
    return similarity(a, b) < (1.0 - min_dist)


def split_component(image, comp, by_id, device="cpu", model_dir=None):
    """Split a component ONCE and map the result onto every instance.

    Splitting each instance independently is what made this feature unusable: SAM 2
    returned a 20x22 child for three play buttons and 19x32 for two others, so the
    grouper then saw two components and gave them two classes. A component is one
    design element; its decomposition must be one decision.

    The representative is split for real; the other instances inherit the child's
    box in *relative* coordinates and take their own pixels from it.
    """
    rep = by_id[comp.representative]
    inner, info = propose(image, rep, device=device, model_dir=model_dir)
    if inner is None:
        return None, info
    if not _shape_differs(image, rep, inner):
        return None, "child is a scaled copy of the parent, not a part of it"

    ys, xs = np.nonzero(inner)
    px0, py0, px1, py1 = [float(v) for v in rep.box]
    pw, ph = max(1.0, px1 - px0), max(1.0, py1 - py0)
    rel = ((xs.min() - px0) / pw, (ys.min() - py0) / ph,
           (xs.max() + 1 - px0) / pw, (ys.max() + 1 - py0) / ph)

    out = {}
    for rid in comp.members:
        r = by_id[rid]
        x0, y0, x1, y1 = [float(v) for v in r.box]
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        box = (x0 + rel[0] * w, y0 + rel[1] * h, x0 + rel[2] * w, y0 + rel[3] * h)
        # The child's pixels come from the instance's own alpha, restricted to the
        # inherited window — same geometry everywhere, real pixels per instance.
        src = r.alpha if r.alpha is not None else r.mask.astype(np.float32)
        m = np.zeros_like(src, dtype=bool)
        bx0, by0 = int(max(0, box[0])), int(max(0, box[1]))
        bx1, by1 = int(min(src.shape[1], box[2])), int(min(src.shape[0], box[3]))
        if bx1 <= bx0 or by1 <= by0:
            continue
        m[by0:by1, bx0:bx1] = src[by0:by1, bx0:bx1] >= 0.5
        if m.sum() >= 20:
            out[rid] = m
    return out, {**info, "instances": len(out)}


def split_regions(image, regions, components, measured, device="cpu", model_dir=None,
                  verbose=True):
    """-> (new_regions, notes). Appends one child Region per split instance.

    Splits per COMPONENT, not per region: one decision, applied to every instance.
    Children get ids after the existing ones; the tree stage picks up the containment
    relationship from geometry on its own, so nothing downstream needs to know a
    split happened.
    """
    from .model import Region

    if not available(model_dir):
        return regions, {"skipped": "sam2 weights not present"}

    by_id = {r.id: r for r in regions}
    todo = [c for c in components
            if is_candidate(by_id[c.representative], measured)]
    if verbose:
        print(f"  {len(todo)} components are split candidates "
              f"({sum(c.count for c in todo)} instances)")

    next_id = max((r.id for r in regions), default=-1) + 1
    added, notes, split_parents = [], {}, set()
    for c in todo:
        masks, info = split_component(image, c, by_id, device=device,
                                      model_dir=model_dir)
        if masks is None:
            notes[c.key] = info
            continue
        ids = []
        for rid, m in masks.items():
            ys, xs = np.nonzero(m)
            box = (float(xs.min()), float(ys.min()),
                   float(xs.max() + 1), float(ys.max() + 1))
            added.append(Region(id=next_id, box=box,
                                labels={f"split:{rid}": 1.0},
                                mask=m, alpha=m.astype(np.float32),
                                score=by_id[rid].score))
            ids.append(next_id)
            next_id += 1
        notes[c.key] = {"parents": list(masks), "children": ids, **info}
        split_parents.update(masks)
        if verbose:
            print(f"    {c.key} x{c.count}: split into {len(ids)} children "
                  f"(area {info.get('area_frac')}, contrast {info.get('contrast')})")

    if verbose and added:
        print(f"  {len(added)} child regions added")
    notes["_split_parents"] = sorted(split_parents)
    return regions + added, notes
