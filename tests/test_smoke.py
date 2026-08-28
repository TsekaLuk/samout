"""Smoke tests: does an install actually work, and do the pure functions hold?

Deliberately not a coverage exercise. These answer the two questions someone
picking this up needs answered — did the weights land, and is the logic that
decides delivery routes behaving — without an API key and without a mockup.

    python -m pytest tests/ -q          (or: python tests/test_smoke.py)

The stages that need a network or 3.5 GB of weights are checked for *presence*
and skipped otherwise, so a fresh clone reports what is missing instead of
failing opaquely.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from samout import handoff, uitree                                   # noqa: E402
from samout.components import consensus, probes_disagree, shape_vector, similarity
from samout.config import Config                                    # noqa: E402
from samout.matte import decontaminate, sharpen_alpha               # noqa: E402
from samout.model import Region, covers, iou                        # noqa: E402
from samout.taxonomy import CLASSES, DELIVERY, classify, deliver    # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# --- geometry ---------------------------------------------------------------

def test_geometry():
    print("geometry")
    check("iou of identical boxes is 1", abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1) < 1e-9)
    check("iou of disjoint boxes is 0", iou((0, 0, 5, 5), (6, 6, 10, 10)) == 0)
    check("covers is asymmetric", covers((0, 0, 10, 10), (2, 2, 4, 4)) == 1.0
          and covers((2, 2, 4, 4), (0, 0, 10, 10)) < 1.0)


# --- taxonomy: every class must be deliverable ------------------------------

def test_taxonomy():
    print("taxonomy")
    check("every class has a known delivery route",
          all(c["delivery"] in DELIVERY for c in CLASSES.values()),
          f"{len(CLASSES)} classes")
    check("brand_asset is never regenerable", not CLASSES["brand_asset"]["regenerable"],
          "a synthesised trademark is a fidelity and a legal failure")
    check("photography routes to a raster asset",
          deliver("photography")["delivery"] == "asset_raster")

    # The Material system-vs-product icon test, on the observations it was fitted to.
    flat = {"content_type": "pictogram", "hue_count": 1, "depth_cues": []}
    check("flat monochrome glyph -> system_icon",
          classify(flat, {"lum_std_inner": 22.0})[0] == "system_icon")
    check("multi-hue pictogram -> product_icon",
          classify({**flat, "hue_count": 3}, {"lum_std_inner": 30.0})[0] == "product_icon")
    check("glyph on a translucent disc -> system_icon (backdrop shows through)",
          classify(flat, {"lum_std_inner": 95.0})[0] == "system_icon")
    check("solid shaded body -> product_icon",
          classify(flat, {"lum_std_inner": 40.0})[0] == "product_icon")
    check("a trademark outranks its drawing style",
          classify({**flat, "is_brand_mark": True})[0] == "brand_asset")


# --- consensus: cue intersection, not union ---------------------------------

def test_consensus():
    print("consensus")
    a = {"content_type": "pictogram", "hue_count": 1, "depth_cues": ["bevel", "glow"]}
    b = {"content_type": "pictogram", "hue_count": 1, "depth_cues": ["bevel"]}
    m = consensus([a, b])
    check("depth cues intersect rather than union", m["depth_cues"] == ["bevel"],
          "a cue seen once in N passes is backdrop, not object")
    check("dropped cues are reported", m["consensus"]["cues_dropped"] == ["glow"])
    check("probe disagreement is detected",
          probes_disagree([a, {**b, "content_type": "text"}]))
    check("agreement is not flagged", not probes_disagree([a, b]))


# --- shape vector: background must not dominate -----------------------------

def test_shape_vector():
    print("shape vector")
    def disc(bg):
        im = Image.new("RGB", (40, 40), bg)
        y, x = np.ogrid[:40, :40]
        a = np.array(im)
        a[((x - 20) ** 2 + (y - 20) ** 2) < 100] = (255, 255, 255)
        return Image.fromarray(a)
    s = similarity(shape_vector(disc((10, 10, 40))), shape_vector(disc((200, 180, 90))))
    check("same shape on different backgrounds stays similar", s > 0.7, f"sim={s:.3f}")


# --- matte ------------------------------------------------------------------

def test_matte():
    print("matte")
    a = np.linspace(0, 1, 100, dtype=np.float32)
    sharp = sharpen_alpha(a, contrast=4.0)
    check("alpha stays in [0,1]", sharp.min() >= 0 and sharp.max() <= 1)
    check("sharpening narrows the ramp",
          ((sharp > 0.02) & (sharp < 0.98)).sum() < ((a > 0.02) & (a < 0.98)).sum())

    rgb = np.zeros((5, 5, 3), np.float32)
    rgb[2, 2] = 0.5                       # a half-covered white pixel on black
    alpha = np.zeros((5, 5), np.float32)
    alpha[2, 2] = 0.5
    out = decontaminate(rgb, alpha)
    check("decontamination brightens a half-covered pixel", out[2, 2, 0] > rgb[2, 2, 0],
          f"{rgb[2,2,0]:.2f} -> {out[2,2,0]:.2f}")


# --- tree -------------------------------------------------------------------

def test_tree():
    print("tree")
    cfg = Config().tree
    page = [0, 0, 200, 200]
    regions = [
        Region(id=0, box=(0, 0, 200, 100), labels={}),      # a row container
        Region(id=1, box=(10, 20, 50, 60), labels={}),      # two children
        Region(id=2, box=(80, 20, 120, 60), labels={}),
        Region(id=3, box=(0, 150, 40, 190), labels={}),     # a sibling of the container
    ]
    nodes, roots = uitree.annotate(regions, {}, page, cfg)
    check("parent is the smallest containing region", nodes[1].parent == 0)
    check("container collects its children", sorted(nodes[0].children) == [1, 2])
    check("a non-contained region stays a root", nodes[3].parent is None)
    check("row layout inferred", (nodes[0].layout or {}).get("direction") == "row")


# --- grid detection ---------------------------------------------------------

def test_grids():
    print("grids")
    spec = [{"id": i, "class": "system_icon", "delivery": "icon_library",
             "children": [], "flags": [],
             "box_xyxy": [100 + i * 25, 50, 120 + i * 25, 70],
             "size_px": [20, 20]} for i in range(8)]
    grids = handoff.find_grids(spec)
    check("an evenly spaced run is found", len(grids) == 1, f"{len(grids)} grid(s)")
    if grids:
        check("all cells captured", grids[0]["cells"] == 8)
        check("gap measured", grids[0]["gap"] == 25)
        check("regularity is low-variance", grids[0]["gap_cv"] < 0.05)

    scattered = [dict(e, box_xyxy=[100 + i * i * 12, 50, 120 + i * i * 12, 70])
                 for i, e in enumerate(spec)]
    check("irregular spacing is not a grid", len(handoff.find_grids(scattered)) == 0)


# --- rule set ---------------------------------------------------------------

def test_rules():
    print("rule set")
    from samout.rules import Context, audit, describe
    from samout.ruleset import RULES, classify as apply_rules

    check("rule set audits clean", not audit(RULES), "; ".join(audit(RULES)))
    check("every rule states a reason", all(r.why for r in RULES))
    check("every rule cites evidence or is definitional",
          all(r.evidence or r.tags for r in RULES))
    check("rule set is printable", len(describe(RULES).splitlines()) >= len(RULES))

    # The decisions the branch chain used to make, now pinned per rule.
    cases = [
        ("brand mark", Context(obs={"is_brand_mark": True}), "brand_asset", "brand_mark"),
        ("region holding a mark",
         Context(obs={"is_brand_mark": True}, children=[1, 2]),
         "composite", "region_containing_a_mark"),
        ("container", Context(obs={"content_type": "pictogram"}, children=[1]),
         "composite", "composite_container"),
        ("split shell", Context(obs={"content_type": "plain"}, is_split_parent=True),
         "token", "split_container_shell"),
        ("filled control", Context(obs={"content_type": "control"}),
         "token", "filled_control"),
        ("live text", Context(obs={"content_type": "text"}),
         "typography", "live_text"),
        ("display lettering", Context(obs={"content_type": "display_lettering"}),
         "spot_illustration", "display_lettering"),
        ("unobserved", Context(obs={}), "unobserved", "unobserved"),
    ]
    for label, ctx, want_cls, want_rule in cases:
        cls, why, rule, _ = apply_rules(ctx)
        check(f"{label} -> {want_cls}", cls == want_cls and rule == want_rule,
              f"got {cls} via {rule}")

    # Identity must beat structure, structure must beat content.
    c = Context(obs={"is_brand_mark": True, "content_type": "photographic"})
    check("identity outranks content", apply_rules(c)[0] == "brand_asset")
    c = Context(obs={"content_type": "photographic"}, children=[1, 2])
    check("structure outranks content", apply_rules(c)[0] == "composite")


# --- weights ----------------------------------------------------------------

def test_weights():
    print("weights (presence only)")
    for name, path, required in (
            ("SAM 3", "models/sam3/model.safetensors", True),
            ("ViTMatte", "models/vitmatte-small/model.safetensors", False),
            ("SAM 2", "models/sam2/model.safetensors", False),
            ("OmniParser", "models/omniparser/icon_detect.pt", False)):
        ok = Path(path).exists()
        if required:
            check(f"{name} present", ok, path)
        else:
            print(f"  {'PASS' if ok else 'SKIP'}  {name} optional  — {path}")


def main():
    for fn in (test_geometry, test_taxonomy, test_consensus, test_shape_vector,
               test_matte, test_tree, test_grids, test_rules, test_weights):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
