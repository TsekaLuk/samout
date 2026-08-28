"""The contract between stages.

Every stage takes one of these and returns another, keyed by region id. Nothing
downstream re-derives the region list from a JSON shape, which is what previously
coupled the stages: each one dug the regions back out of `assets` + `css_regions`
using different field names, so changing the manifest broke three modules.

Dependency direction is one-way — every module imports this, this imports nothing
from the package. Stages do not import each other; `handoff` joins their outputs
and `run` orchestrates.

    detect    -> list[Region]
    group     -> list[Component],       {region_id: component_key}
    observe   -> {region_id: Observation}
    uitree    -> {region_id: Node}
    taxonomy  -> (class, why) per Observation
    handoff   -> list[HandoffEntry]

`Observation` and `HandoffEntry` stay plain dicts on purpose: they cross the VLM
and JSON boundaries, where a dataclass buys nothing and costs a conversion at every
edge. The records below are the ones that carry invariants worth naming.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Region:
    """One detected thing. Produced by detect, never mutated afterwards."""

    id: int
    box: tuple                      # x0, y0, x1, y1 in source pixels
    labels: dict                    # SAM 3 prompt -> score, all prompts that hit
    mask: Optional[np.ndarray] = None   # bool HxW, dropped once cutouts are written
    alpha: Optional[np.ndarray] = None  # float HxW in [0,1], the pre-threshold mask
    score: float = 0.0
    # Extent of the mask itself, which is NOT the same as `box`: SAM 3 emits the two
    # independently and the mask routinely spills outside the box. Cropping to `box`
    # to save work silently clipped those pixels and changed every texture statistic.
    # Computed once at detection, where the mask is already in hand.
    mask_box: Optional[tuple] = None

    @property
    def size(self):
        return (int(self.box[2] - self.box[0]), int(self.box[3] - self.box[1]))

    @property
    def area(self):
        w, h = self.size
        return w * h

    def as_json(self):
        return {"id": self.id, "box_xyxy": [int(v) for v in self.box],
                "size_px": list(self.size), "score": round(self.score, 4),
                "labels": {k: round(v, 3) for k, v in
                           sorted(self.labels.items(), key=lambda kv: -kv[1])}}

    @staticmethod
    def from_json(d):
        return Region(id=d["id"], box=tuple(d["box_xyxy"]),
                      labels=d.get("labels", {}), score=d.get("score", 0.0))


@dataclass
class Component:
    """A set of regions that are the same design-system component.

    A design system has one Play Button used seven times, not seven play buttons.
    Classifying the component rather than each instance is both a consistency
    guarantee and an N-fold saving on observation calls.
    """

    key: str
    members: list
    representative: int
    probes: list = field(default_factory=list)   # instances actually observed

    @property
    def count(self):
        return len(self.members)

    def as_json(self):
        return {"key": self.key, "members": self.members, "count": self.count,
                "representative": self.representative, "probes": self.probes}


@dataclass
class Node:
    """A region's place in the UI tree. Produced by uitree from boxes alone."""

    id: int
    parent: Optional[int] = None
    children: list = field(default_factory=list)
    depth: int = 0
    role: str = "generic"           # WAI-ARIA role
    atomic: str = "atom"            # Atomic Design tier
    layout: Optional[dict] = None   # None for leaves

    @property
    def is_leaf(self):
        return not self.children

    def as_json(self):
        d = {"id": self.id, "parent": self.parent, "children": self.children,
             "depth": self.depth, "role": self.role, "atomic": self.atomic}
        if self.layout:
            d["layout"] = self.layout
        return d


# --- helpers shared by more than one stage, kept here so no stage imports another ---

def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    return inter / (area(a) + area(b) - inter)


def covers(outer, inner):
    """Fraction of `inner` that falls inside `outer`."""
    ix0, iy0 = max(outer[0], inner[0]), max(outer[1], inner[1])
    ix1, iy1 = min(outer[2], inner[2]), min(outer[3], inner[3])
    return (max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)) / max(1e-6, area(inner))
