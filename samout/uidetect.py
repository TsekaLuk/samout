"""Stage 1b: UI element detection with OmniParser, merged with SAM 3's concepts.

SAM 3 finds *things it has words for*. On a screen that is mostly artwork that is
nearly everything; on one that is mostly chrome it is a small minority. Measured
against OmniParser on two screens:

    KTV app screen      SAM 3 covered 37 elements, missed  7  (19%)
    Virelle design board SAM 3 covered 153 elements, missed 116 (76%)

Three quarters of the second screen never entered the pipeline — every button, tab
glyph, chip, price and label — which is why it produced zero `token`, zero
`system_icon` and zero `typography` there. Nothing was misclassified; the regions
simply did not exist to classify.

The two detectors are complementary because their training objectives are:

    OmniParser (YOLOv8 on web UI)  "where can you tap"   -> chrome, controls, text
    SAM 3 (open-vocabulary)        "what concept is this" -> photos, icons, artwork

OmniParser returns boxes only, no masks. That is fine for the majority of what it
adds — those regions are CSS and never need a cutout. When one does classify as an
asset, it is flagged for mask refinement rather than shipped box-cropped.

  microsoft/OmniParser-v2.0, MIT. Wang et al., arXiv:2408.00203.
"""

from pathlib import Path

from .model import Region, covers as _covers, iou

DEFAULT_PT = Path("models/omniparser/icon_detect.pt")
HUB_ID = "microsoft/OmniParser-v2.0"

_MODEL = None


def available(weights=None):
    return Path(weights or DEFAULT_PT).exists()


def _load(weights=None):
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(weights or DEFAULT_PT))
    return _MODEL


def detect_elements(image, conf=0.10, iou_nms=0.5, imgsz=1280, weights=None,
                    min_area=200, max_area_frac=0.8):
    """-> list of (box, score), the interactable elements OmniParser sees."""
    model = _load(weights)
    W, H = image.size
    res = model.predict(image, conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    out = []
    for b, s in zip(boxes, scores):
        w, h = float(b[2] - b[0]), float(b[3] - b[1])
        if w * h < min_area or w * h > max_area_frac * W * H:
            continue
        out.append((tuple(float(v) for v in b), float(s)))
    return out


def merge(regions, elements, dup_iou=0.55, contain_cover=0.90, verbose=True):
    """Add OmniParser elements as Regions, skipping ones SAM 3 already has.

    Ids continue after the existing regions and everything is re-sorted largest
    first, so the invariant the tree relies on — a parent's id is lower than its
    children's — still holds after the merge.

    A merged region carries `mask=None`. Downstream that means: crop by box for
    observation, and no cutout to ship. Most of them are chrome and want neither.
    """
    kept, dup, hit_area = [], 0, 0
    for box, score in elements:
        if any(iou(box, r.box) >= dup_iou for r in regions):
            dup += 1
            continue
        # A tap target that wraps exactly ONE existing region is that region's hit
        # area, not its parent. Kept as a region it becomes one, and the thing
        # inside it stops being a leaf — which is how 17 OmniParser boxes turned
        # into `composite` containers and cost 7.5pp on the labelled screen.
        #
        # Area ratio cannot separate the two cases (3.1x to 17.5x, no gap), but the
        # child count can, and it follows from what the detectors mean: a container
        # groups siblings, a hit area wraps one element.
        enclosed = sum(1 for r in regions if _covers(box, r.box) >= contain_cover)
        if enclosed == 1:
            hit_area += 1
            continue
        kept.append((box, score))

    merged = list(regions) + [
        Region(id=-1, box=box, labels={"ui_element": round(score, 3)},
               mask=None, alpha=None, score=score)
        for box, score in kept]
    merged.sort(key=lambda r: -((r.box[2] - r.box[0]) * (r.box[3] - r.box[1])))
    for i, r in enumerate(merged):
        r.id = i

    if verbose:
        print(f"  OmniParser: {len(elements)} elements — {dup} duplicate, "
              f"{hit_area} hit areas around a single region, {len(kept)} added "
              f"-> {len(merged)} regions total")
    return merged


def detect_and_merge(image, regions, cfg=None, weights=None, verbose=True):
    """Convenience: run the detector and merge in one call. No-op if weights absent."""
    if not available(weights):
        if verbose:
            print("  OmniParser weights not present; skipping UI element detection")
        return regions
    conf = getattr(cfg, "conf", 0.10) if cfg else 0.10
    elements = detect_elements(image, conf=conf, weights=weights)
    return merge(regions, elements,
                 dup_iou=getattr(cfg, "dup_iou", 0.55) if cfg else 0.55,
                 contain_cover=getattr(cfg, "contain_cover", 0.90) if cfg else 0.90,
                 verbose=verbose)
