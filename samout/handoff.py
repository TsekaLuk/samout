"""Stage 5: join every stage's output into the design handoff spec.

Pure — no IO, no model calls, no globals. It is the only module that imports from
more than one sibling, and it does so to join their results, never to drive them.
That keeps the dependency graph a DAG: everything depends on `model`, `handoff`
depends on the pure stages, `run` depends on everything.

The output is a design handoff spec, which is a real artifact a design team ships
to engineering — not a shape invented here. Per region: what class it is, how it
is delivered, what it costs, its accessibility treatment, and the defects a
designer would raise at review.
"""

from .taxonomy import classify, deliver

# TRIED AND REVERTED — kept as a warning, empty on purpose.
#
# The idea: a region that IS art should keep its class even with children, because a
# neon sign found inside a venue photo is part of that photo. That reasoning is sound
# for `is_brand_mark` and it is measurably wrong for content types.
#
# Measured: promoting `photographic` here scored 58.5% against a 70.6% +- 2.2 baseline
# — a 12pp regression, far outside the noise band. Inspection showed why: 7 of the 9
# regions it promoted were containers. The observer calls a playlist card
# "photographic" because the card is mostly a photograph, so the label does not
# distinguish "a photo" from "a card containing a photo".
#
# The distinction that matters: `is_brand_mark` is an exclusive factual question
# ("is this a trademark"), while `content_type` is a description of degree that fits
# both a container and its contents. Only the former can outrank structure.
#
# The underlying complaint — that venue photos never reach the asset list — is real,
# but it is a region-granularity problem (SAM 3 returns card = photo + caption as one
# box), not a rule-ordering one. It belongs with the bundling work.
ATOMIC_CONTENT = set()

# A brand lockup resolves into at most one separately-detected part. Above this it
# is a container that happens to include a mark. See the `is_brand_mark` branch.
MARK_MAX_CHILDREN = 2


# TRIED AND ABANDONED — left at 0 (off) with the evidence, so it is not re-attempted.
#
# Generating from a small crop of a detailed SCENE fails: a 72x71 nebula came back
# an abstract smear, a 23x93 cosmic banner likewise. A 73x70 icon of the same pixel
# count came back clean, as did four photographs of 5-13k px. The failure is real
# and worth warning about — a bad asset that looks successful is the worst outcome
# in this pipeline.
#
# But no predictor was found. Five candidates, measured on the eight generated
# assets, all interleave successes and failures:
#     pixel count      #39 failed at 5112, #41 succeeded at 5110
#     colour count     849 (fail) vs 700 (success)
#     entropy          7.377 (fail) vs 7.300 (success)
#     information density  11.87 (fail) vs 8.57 (success)
#     post-hoc similarity of result to reference — fully overlapping ranges
#
# And a size gate cannot work here regardless: the median asset on the test board is
# 1599 px, so any threshold covering both failures (5112, 2139) flags 74-90% of all
# assets. A warning on nine assets in ten is not a warning.
#
# What would actually settle it is a screen with enough labelled generations to
# calibrate on, which does not exist yet. Until then this stays off rather than
# shipping a gate that is either useless or arbitrary.
REFERENCE_THIN_PX = 0


def _flags(cls, obs, node, component_size, spec_class_of_ancestor, node_size=None):
    """Handoff defects, in the vocabulary a design review would use."""
    out = []
    if obs.get("text_role") == "live_data":
        out.append("overlay: the region carries runtime values (counts, duration, "
                   "price) — render them in CSS over the asset, not into it")
    elif obs.get("has_baked_text"):
        out.append("i18n: copy baked into pixels, cannot be localized")
    if cls == "brand_asset":
        out.append("brand: extract exactly, do not regenerate")
    if obs.get("theme_dependent"):
        out.append("theming: needs a light and a dark variant")
    if obs.get("needs_transparency") and deliver(cls)["delivery"].startswith("asset"):
        out.append("alpha: deliver with transparency")
    if (obs.get("confidence") or 1.0) < 0.5:
        out.append("review: low observation confidence")
    if cls == "product_icon":
        out.append("prefer SVG redraw if the icon language is systematic")
    # Reference thinness. Generation from a small crop of a detailed SCENE fails —
    # a 72x71 nebula came back as an abstract smear — while a 73x70 icon of the same
    # pixel count came back clean. Five candidate predictors were tried on the eight
    # generated assets (pixel count, colour count, entropy, information density, and
    # post-hoc similarity of the result to the reference) and NONE separated the two
    # groups; the successes and failures interleave on every one.
    #
    # So this does not claim to predict failure. It marks the regions where the
    # reference is thin enough that the question is open, because the alternative —
    # a bad asset entering the handoff looking like a good one — is the worse error.
    # Narrow it when a screen exists that can actually calibrate it.
    if REFERENCE_THIN_PX and deliver(cls)["delivery"].startswith("asset"):
        px = node_size[0] * node_size[1] if node_size else 0
        if 0 < px <= REFERENCE_THIN_PX:
            out.append(f"thin reference: {px}px source; verify the generated asset "
                       "by eye — detailed scenes at this size have failed")
    if component_size > 1 and node.is_leaf:
        out.append(f"component: {component_size} instances share this class; "
                   "build once, reuse")
    # A region sitting inside a photograph or an illustration ships as part of it.
    # Routing it separately bills for art that is already in the parent's pixels.
    if spec_class_of_ancestor in ("photography", "spot_illustration") and node.is_leaf:
        out.append(f"absorbed: inside a {spec_class_of_ancestor} ancestor; "
                   "delivered with the parent, not separately")
    return out


def _ancestor_class(rid, nodes, class_of):
    node = nodes.get(rid)
    seen = set()
    while node is not None and node.parent is not None and node.parent not in seen:
        seen.add(node.parent)
        cls = class_of.get(node.parent)
        if cls in ("photography", "spot_illustration", "brand_asset"):
            return cls
        node = nodes.get(node.parent)
    return None


def build(regions, nodes, observations, components, describe=None, measured=None,
          split_parents=None):
    """-> list[dict], one entry per region, ordered by id.

    `components` is list[Component]; `describe` is the optional reference-statistics
    map from the detection stage.
    """
    describe = describe or {}
    measured = measured or {}
    split_parents = set(split_parents or ())
    by_id = {r.id: r for r in regions}
    comp_of, comp_size = {}, {}
    for c in components:
        for rid in c.members:
            comp_of[rid] = c.key
            comp_size[rid] = c.count

    # First pass: class only, so ancestors can be consulted in the second.
    class_of, why_of = {}, {}
    for rid, node in nodes.items():
        obs = observations.get(rid, {})
        # Content outranks structure. A region that IS art does not stop being art
        # because something was detected inside it — a neon sign spotted within a
        # venue photograph is part of that photograph, not a sibling of it, and the
        # photo still has to ship as a raster asset. Letting `composite` win here
        # turned two venue photos into layout containers and dropped them from the
        # asset list entirely; the same ordering bug previously stripped a brand
        # badge of its never-regenerate flag.
        if obs.get("is_brand_mark") and len(node.children) < MARK_MAX_CHILDREN:
            class_of[rid] = "brand_asset"
            why_of[rid] = "identity-locked mark (outranks structure)"
        elif obs.get("is_brand_mark"):
            # The observer says "brand" for a region that merely CONTAINS the logo:
            # a top app bar (4 children: clock, status, logo, scan), a benefits row
            # (5), a product card (2). Marked as art, a coding agent would extract
            # the whole strip instead of the wordmark inside it — and the error is
            # invisible downstream, because the class looks right.
            #
            # A lockup decomposes into at most one separately-detected part (a crown
            # over "VIP"); a container holds several distinct elements. Measured
            # across two screens: real marks had 0-1 children, every false one had
            # 2, 4 or 5. The mark itself is still classified on its own region.
            class_of[rid] = "composite"
            why_of[rid] = (f"contains a brand mark but holds {len(node.children)} "
                           "classified elements — a region with a logo, not a logo")
        elif obs.get("content_type") in ATOMIC_CONTENT:
            cls, why = classify(obs, measured.get(rid), describe.get(rid))
            class_of[rid] = cls
            why_of[rid] = f"{why} (content outranks structure; children are part of it)"
        elif rid in split_parents:
            # We took this region apart ourselves, so we know what the remainder is.
            # The split gate only fires on high interior variance — a shell you can
            # see the backdrop through — so what is left after lifting the glyph out
            # is a translucent disc or pill, which is CSS.
            #
            # This is safe where the earlier "content outranks structure" attempt was
            # not: that one used `content_type == photographic`, a description that
            # fits a card as readily as the photo inside it. Here the fact is
            # constructed by the pipeline, not inferred from a label.
            class_of[rid] = "token"
            why_of[rid] = "container shell left after its glyph was split out"
        elif node.children:
            class_of[rid] = "composite"
            why_of[rid] = (f"{node.atomic} with {len(node.children)} children, "
                           f"role={node.role}")
        elif not obs:
            class_of[rid] = "unobserved"
            why_of[rid] = "no observation returned; rerun or inspect by hand"
        else:
            class_of[rid], why_of[rid] = classify(obs, measured.get(rid), describe.get(rid))

    spec = []
    for rid in sorted(nodes):
        r, node = by_id[rid], nodes[rid]
        obs = observations.get(rid, {})
        cls = class_of[rid]
        d = deliver(cls)
        anc = _ancestor_class(rid, nodes, class_of)
        entry = {
            "id": rid,
            "subject": obs.get("subject"),
            "class": cls,
            "why": why_of[rid],
            "role": node.role,
            "atomic": node.atomic,
            "depth": node.depth,
            "delivery": d["delivery"],
            "action": d["action"],
            "cost": d["cost"],
            "regenerable": d["regenerable"],
            "a11y": d["a11y"],
            "box_xyxy": [int(v) for v in r.box],
            "size_px": list(r.size),
            "parent": node.parent,
            "children": node.children,
            "layout": node.layout,
            "component": comp_of.get(rid),
            "component_instances": comp_size.get(rid, 1),
            "sam3_labels": {k: round(v, 3) for k, v in
                            sorted(r.labels.items(), key=lambda kv: -kv[1])},
            "observed": {k: obs.get(k) for k in
                         ("content_type", "hue_count", "depth_cues", "is_brand_mark",
                          "has_baked_text", "text_role", "needs_transparency",
                          "theme_dependent", "closest_library_icon", "confidence")},
            "reference": describe.get(rid, {}),
            "measured": measured.get(rid, {}),
            "flags": _flags(cls, obs, node, comp_size.get(rid, 1), anc,
                            node_size=list(r.size)),
        }
        if obs.get("regen_prompt") and d["delivery"].startswith("asset"):
            entry["regen"] = {
                "prompt": obs["regen_prompt"],
                "needs_transparency": bool(obs.get("needs_transparency")),
                "output_format": "png_rgba" if obs.get("needs_transparency") else "png",
                "target_size_px": [r.size[0] * 3, r.size[1] * 3],
                "reference": f"cutouts/{rid:02d}_cutout.png",
                "palette": describe.get(rid, {}).get("dominant_colors", []),
                "forbidden": (["do not synthesise — extract the original"]
                              if cls == "brand_asset" else []),
            }
        spec.append(entry)
    return spec


def find_grids(spec, min_cells=5, size_tol=0.30, gap_cv=0.25):
    """Group evenly-spaced same-size CSS cells into one logical component.

    A calendar arrives as ~19 separate cells and a colour palette as 8 swatches,
    because the detector finds every tappable cell but not the module around them —
    on the test board both were top-level roots with no container to attach to. So
    the grouping cannot come from the tree; it has to be recovered from the cells.

    Regularity is the signal, and it is measurable: the palette's swatches sit at
    x = 1191, 1219, 1245, 1270, 1295, 1321, 1346, 1374 — a gap of 25-26 every time.
    A run of same-sized cells with a low coefficient of variation on the gaps is a
    grid; scattered controls of similar size are not.

    Additive only: cells keep their own entries and classes, and gain a `grid` id.
    Nothing is removed, because "one calendar" and "19 cells" are both true and
    different consumers want different ones.
    """
    import statistics as st

    cells = [e for e in spec
             if not e.get("children")
             and e["delivery"] in ("css", "icon_library")]
    used, grids = set(), []

    for axis, pos, cross in ((0, lambda e: e["box_xyxy"][0], lambda e: e["box_xyxy"][1]),
                             (1, lambda e: e["box_xyxy"][1], lambda e: e["box_xyxy"][0])):
        pool = [e for e in cells if e["id"] not in used]
        # A run shares a cross-axis line and a size.
        lines = {}
        for e in pool:
            lines.setdefault(round(cross(e) / 8), []).append(e)
        for row in lines.values():
            if len(row) < min_cells:
                continue
            row = sorted(row, key=pos)
            med_w = st.median(e["size_px"][axis] for e in row)
            run = [e for e in row
                   if abs(e["size_px"][axis] - med_w) <= size_tol * max(1, med_w)]
            if len(run) < min_cells:
                continue
            gaps = [pos(run[i + 1]) - pos(run[i]) for i in range(len(run) - 1)]
            if not gaps or st.median(gaps) <= 0:
                continue
            cv = (st.pstdev(gaps) / st.median(gaps)) if len(gaps) > 1 else 0.0
            if cv > gap_cv:
                continue
            gid = f"g{len(grids):02d}"
            box = [min(e["box_xyxy"][0] for e in run), min(e["box_xyxy"][1] for e in run),
                   max(e["box_xyxy"][2] for e in run), max(e["box_xyxy"][3] for e in run)]
            from collections import Counter
            grids.append({"id": gid, "axis": "row" if axis == 0 else "column",
                          "cells": len(run), "box_xyxy": box,
                          "gap": round(st.median(gaps)),
                          "gap_cv": round(cv, 3),
                          "cell_classes": dict(Counter(e["class"] for e in run)),
                          "cell_ids": [e["id"] for e in run]})
            for e in run:
                e["grid"] = gid
                e["flags"].append(f"grid {gid}: one of {len(run)} evenly-spaced cells; "
                                  "build as a single component")
                used.add(e["id"])
    return grids


def mark_repeating(spec, nodes, min_children=5, max_child_frac=0.25):
    """Flag containers whose children are a uniform run of CSS-buildable cells.

    A calendar arrives as 1 container + 13 text cells + 5 glyphs; a colour palette
    as 1 container + 8 swatches. Nothing is misclassified — every cell is correctly
    CSS — but a coding agent wants "a calendar", not 19 entries, and the observation
    stage paid for all 19.

    This is aggregation, not correction, so it is additive: the cells stay in the
    spec and in the tree, and the parent gains a `repeats` block plus a
    `collapsible` flag. A consumer that wants one entry can take the parent; one
    that wants pixel detail still has everything.

    The test is deliberately narrow — enough children, all leaves, all delivered by
    CSS or an icon library, and each small relative to the parent. A card holding a
    photo and a caption fails on the delivery check, which is the point.
    """
    by_id = {e["id"]: e for e in spec}
    from collections import Counter
    marked = 0
    for e in spec:
        kids = [by_id[c] for c in e.get("children", []) if c in by_id]
        if len(kids) < min_children:
            continue
        if any(k.get("children") for k in kids):
            continue
        if not all(k["delivery"] in ("css", "icon_library") for k in kids):
            continue
        pa = max(1, e["size_px"][0] * e["size_px"][1])
        if any(k["size_px"][0] * k["size_px"][1] > max_child_frac * pa for k in kids):
            continue
        classes = Counter(k["class"] for k in kids)
        e["repeats"] = {"cells": len(kids),
                        "cell_classes": dict(classes),
                        "child_ids": [k["id"] for k in kids]}
        e["flags"].append(
            f"repeating: {len(kids)} uniform CSS cells; build as one component")
        for k in kids:
            k["collapsible_into"] = e["id"]
        marked += 1
    return marked


def summarize(spec):
    from collections import Counter
    by_class = Counter(e["class"] for e in spec)
    by_delivery = Counter(e["delivery"] for e in spec)
    return {
        "regions": len(spec),
        "by_class": dict(by_class),
        "by_delivery": dict(by_delivery),
        "total_effort": sum(e["cost"] for e in spec),
        "to_generate": sum(1 for e in spec
                           if e["delivery"].startswith("asset") and e["regenerable"]),
        "to_extract": sum(1 for e in spec if e["delivery"] == "asset_exact"),
        "flagged": sum(1 for e in spec if e["flags"]),
    }
