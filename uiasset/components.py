"""Group repeated regions into components before classifying them.

A design system does not contain eight slightly different play buttons; it contains
one Play Button used eight times. Classifying instances instead of components is what
produced this, on visually identical tab-bar glyphs:

    #35 #36 #37 #38  play button -> product_icon   (cues reported: bevel, specular)
    #45 #46 #47      play button -> system_icon    (no cues reported)

The model is not wrong twice; it is being asked the same question eight times and
answering independently. Grouping first makes the answer structural rather than
per-sample, and cuts VLM cost by the repeat factor.

Grouping is perceptual, not semantic: a difference hash over the normalised crop,
plus an aspect-ratio guard so a wide banner never merges with a square icon.
Reference: Atomic Design (Frost) — instances of a component share a class by
definition.
"""

import numpy as np
from PIL import Image

from .model import Component


def shape_vector(crop, n=16):
    """Contrast-normalised luminance patch.

    A difference hash over the raw crop does NOT work here, which is worth
    recording: the eight play buttons are white glyphs on eight different card
    photographs, so the background dominates and same-component distances came out
    at 24-48 bits — indistinguishable from unrelated icons. Subtracting the mean
    and dividing by the standard deviation removes exactly that background
    luminance and leaves the silhouette, which is the actual component invariant.
    """
    g = np.asarray(crop.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float32)
    g -= g.mean()
    sd = g.std()
    return (g / sd).flatten() if sd > 1e-6 else g.flatten()


def similarity(a, b):
    """Normalised correlation in [-1, 1]; 1.0 is identical shape."""
    return float(np.dot(a, b) / len(a))


def group(crop_source, regions, cfg):
    """-> (groups, of_region)

    groups: list of {"key", "members": [ids], "representative": id, "count"}
    of_region: {region_id: group_key}

    min_sim was calibrated on this repo's test screen against known pairs, not
    guessed: seven instances of the play button scored 0.834-0.970 against each
    other, while unrelated icon pairs topped out at 0.546. Anywhere in 0.55-0.75
    gives full recall with no false merge; 0.70 sits in the middle of that gap.
    A false merge assigns a region the wrong class; a false split only costs one
    redundant VLM call, so err high.
    """
    min_sim, aspect_tol, size_tol = (cfg.min_similarity, cfg.aspect_tol, cfg.size_tol)
    feats = {}
    for r in regions:
        w, h = r.size
        feats[r.id] = {"v": shape_vector(crop_source.get(r)),
                       "aspect": max(1, w) / max(1, h), "area": max(1, w * h)}

    groups, of_region = [], {}
    for r in regions:
        rid = r.id
        f = feats[rid]
        best, best_sim = None, min_sim
        for g in groups:
            p = feats[g["representative"]]
            if abs(np.log(f["aspect"] / p["aspect"])) > aspect_tol:
                continue
            if abs(np.log(f["area"] / p["area"])) > size_tol:
                continue
            s = similarity(f["v"], p["v"])
            if s >= best_sim:
                best, best_sim = g, s
        if best is not None:
            best["members"].append(rid)
            of_region[rid] = best["key"]
        else:
            key = f"c{len(groups):02d}"
            groups.append({"key": key, "members": [rid], "representative": rid})
            of_region[rid] = key

    # Sample up to `probes` instances per component rather than trusting one.
    # Grouping makes a single bad observation contaminate every instance: the
    # largest play button sits on a card photograph, and reading only that one
    # reported bevel/specular/glow that belong to the background, reclassifying
    # all seven. Spreading the probes across the size range and reconciling them
    # (see consensus) keeps the consistency win without the contamination.
    out = []
    for g in groups:
        ordered = sorted(g["members"], key=lambda i: -feats[i]["area"])
        probes = ([ordered[0]] if len(ordered) == 1
                  else [ordered[0], ordered[len(ordered) // 2], ordered[-1]])
        out.append(Component(key=g["key"], members=g["members"],
                             representative=ordered[0],
                             probes=list(dict.fromkeys(probes))[:cfg.max_probes]))
    return out, of_region


def probes_disagree(observations, hue_slack=1):
    """Do the probes describe the same thing?

    Disagreement is evidence the visual grouper merged two different components,
    not noise to be averaged away. The pink 3D gradient music note in the feature
    rail and the flat white music note in the tab bar are similar enough in
    silhouette to merge, and voting then assigned both the same class. Treat a
    content_type conflict, or a hue spread wider than `hue_slack`, as a split
    signal — the group is dissolved and its members observed individually.
    """
    obs = [o for o in observations if o]
    if len(obs) < 2:
        return False
    if len({o.get("content_type") for o in obs}) > 1:
        return True
    hues = [o.get("hue_count") for o in obs if o.get("hue_count") is not None]
    if hues and max(hues) - min(hues) > hue_slack:
        return True
    if len({bool(o.get("is_brand_mark")) for o in obs}) > 1:
        return True
    return False


def consensus(observations):
    """Reconcile several observations of one component into one.

    Categorical fields take a majority vote. Depth cues take the INTERSECTION,
    not the union: a bevel that is really part of the object appears on every
    instance, while a highlight bleeding in from one instance's backdrop appears
    on exactly one. Union would import every background artifact; intersection
    keeps only what the component itself carries.
    """
    obs = [o for o in observations if o]
    if not obs:
        return None
    if len(obs) == 1:
        return dict(obs[0])

    from collections import Counter

    def vote(key, default=None):
        vals = [o.get(key) for o in obs if o.get(key) is not None]
        return Counter(vals).most_common(1)[0][0] if vals else default

    cue_sets = [set(o.get("depth_cues") or []) for o in obs]
    shared = set.intersection(*cue_sets) if cue_sets else set()
    dropped = sorted(set.union(*cue_sets) - shared) if cue_sets else []

    merged = dict(max(obs, key=lambda o: len(o.get("subject") or "")))
    merged.update({
        "content_type": vote("content_type"),
        "hue_count": sorted(o.get("hue_count", 1) for o in obs)[len(obs) // 2],
        "depth_cues": sorted(shared),
        "is_brand_mark": vote("is_brand_mark", False),
        "has_baked_text": vote("has_baked_text", False),
        "needs_transparency": vote("needs_transparency", False),
        "theme_dependent": vote("theme_dependent", False),
        "confidence": min(o.get("confidence", 1.0) for o in obs),
        "consensus": {"probes": len(obs), "cues_dropped": dropped},
    })
    return merged


def containment(regions, cover=0.90, min_children=2, explained=0.45):
    """Recompute the nesting tree geometrically over ALL regions.

    The tree from the detector under-reports: it only relates regions that one
    prompt happened to find nested inside another. A card holding a photo and a
    caption is a container even when the detector logged no children for it.

    A region is a container when it geometrically covers >= min_children other
    regions AND those children explain a real share of its area (so a big
    illustration that merely overlaps two small badges is not reclassified).
    """
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def inside(outer, inner):
        ix0, iy0 = max(outer[0], inner[0]), max(outer[1], inner[1])
        ix1, iy1 = min(outer[2], inner[2]), min(outer[3], inner[3])
        return (max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)) / max(1e-6, area(inner))

    out = {}
    for a in regions:
        kids = [b["id"] for b in regions
                if b["id"] != a["id"]
                and area(b["box_xyxy"]) < area(a["box_xyxy"]) * 0.9
                and inside(a["box_xyxy"], b["box_xyxy"]) >= cover]
        kid_area = sum(area(b["box_xyxy"]) for b in regions if b["id"] in kids)
        out[a["id"]] = {
            "children": kids,
            "is_container": len(kids) >= min_children
            and kid_area >= explained * area(a["box_xyxy"]),
            "child_area_ratio": round(kid_area / max(1.0, area(a["box_xyxy"])), 3),
        }
    return out
