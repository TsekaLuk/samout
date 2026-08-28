"""Stage 4: reconstruct the UI tree. list[Region] -> {id: Node}.

Pure geometry and naming — no model calls, no IO. This is the backbone the rest of
the pipeline hangs off, not a post-processing step. A coding agent cannot build a
screen from a flat list of boxes; it needs what contains what, in what order, laid
out how. That structure also decides asset routing: an internal node is layout, a
leaf is content. No threshold required — every threshold we deleted here was
compensating for structure we had not built.

The target shape is the accessibility tree, the one standardised description of a
UI's structure *and* semantics:

  W3C WAI-ARIA — role vocabulary (banner, navigation, tablist, list, img, heading)
    and the accessible-name computation screen readers consume.
  CSS Flexbox / Grid — what a container's child relationship compiles down to;
    row / column / grid is measurable from geometry.
  Brad Frost, Atomic Design — atoms / molecules / organisms.
  RICO / view-hierarchy research — precedent for screenshot -> UI tree extraction.

Known gap: SAM 3 detects things, and a layout container is not a thing, so most
internal nodes are missing and the forest is wide. See INDEX.md "Known gaps".
"""

from statistics import median

from .model import Node, area, covers


def build_forest(regions, cfg):
    """Parent of a region is the SMALLEST region that contains it.

    Smallest, not first: a play button sits inside a card inside a list inside the
    page. Only the card is its parent.
    """
    nodes = {r.id: Node(id=r.id) for r in regions}
    box = {r.id: r.box for r in regions}
    by_area = sorted(regions, key=lambda r: r.area)

    for r in by_area:
        candidates = [o for o in by_area
                      if o.id != r.id
                      and o.area > r.area / cfg.shrink
                      and covers(o.box, r.box) >= cfg.cover]
        if candidates:
            p = min(candidates, key=lambda o: o.area)
            nodes[r.id].parent = p.id
            nodes[p.id].children.append(r.id)

    roots = [r.id for r in by_area if nodes[r.id].parent is None]

    def set_depth(rid, d):
        nodes[rid].depth = d
        for c in nodes[rid].children:
            set_depth(c, d + 1)
    for rid in roots:
        set_depth(rid, 0)

    # Reading order: top to bottom, then left to right within a visual row.
    def order(ids):
        return sorted(ids, key=lambda i: (round(box[i][1] / 16), box[i][0]))
    for n in nodes.values():
        n.children = order(n.children)
    return nodes, order(roots)


def infer_layout(node, box, cfg):
    """What CSS would produce this arrangement of children."""
    kids = [box[c] for c in node.children]
    if len(kids) < 2:
        return {"display": "block", "children": len(kids)}

    rows = []
    for b in sorted(kids, key=lambda k: k[1]):
        for row in rows:
            ref = row[0]
            overlap = min(ref[3], b[3]) - max(ref[1], b[1])
            if overlap > cfg.row_overlap * min(ref[3] - ref[1], b[3] - b[1]):
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda k: k[0])

    widths = [len(r) for r in rows]
    if len(rows) == 1:
        kind, cols = "row", widths[0]
    elif all(w == 1 for w in widths):
        kind, cols = "column", 1
    elif len(set(widths)) == 1 and widths[0] > 1:
        kind, cols = "grid", widths[0]
    else:
        kind, cols = "mixed", max(widths)

    gaps_x, gaps_y = [], []
    for row in rows:
        gaps_x += [row[i + 1][0] - row[i][2] for i in range(len(row) - 1)]
    for i in range(len(rows) - 1):
        gaps_y.append(min(r[1] for r in rows[i + 1]) - max(r[3] for r in rows[i]))

    out = {"display": "flex" if kind in ("row", "column")
           else ("grid" if kind == "grid" else "block"),
           "direction": kind, "rows": len(rows), "columns": cols,
           "children": len(kids)}
    if gaps_x:
        out["gap_x"] = round(median(gaps_x))
    if gaps_y:
        out["gap_y"] = round(median(gaps_y))

    px0, _, px1, _ = node_box = box[node.id]
    left = min(k[0] for k in kids) - px0
    right = px1 - max(k[2] for k in kids)
    if abs(left - right) <= max(4, 0.05 * (px1 - px0)):
        out["justify"] = "center" if left > 4 else "stretch"
    else:
        out["justify"] = "flex-start" if left < right else "flex-end"
    return out


def infer_role(node, nodes, box, obs, page, cfg, component_of=None):
    """A conservative ARIA role. Geometry decides structure, the observation decides
    leaf semantics. Anything uncertain stays generic rather than inventing a role a
    screen reader would then announce wrongly."""
    x0, y0, x1, y1 = box[node.id]
    pw, ph = page[2] - page[0], page[3] - page[1]
    full_width = (x1 - x0) >= 0.9 * pw
    o = obs.get(node.id, {})

    if node.children:
        L = node.layout or {}
        if full_width and y1 > 0.88 * ph and L.get("direction") == "row" \
                and len(node.children) >= 3:
            return "tablist"
        if full_width and y0 < 0.08 * ph:
            return "banner"
        if L.get("direction") in ("row", "grid") and len(node.children) >= 3 \
                and component_of:
            sibs = {component_of.get(c) for c in node.children}
            if len(sibs) == 1 and None not in sibs:
                return "list"
        return "group"

    ct = o.get("content_type")
    if ct == "text":
        return "heading" if (y1 - y0) >= cfg.heading_height_frac * ph else "text"
    if ct == "control":
        return "button"
    if ct in ("photographic", "illustration", "pictogram"):
        return "img"
    return "generic"


def annotate(regions, observations, page_box, cfg, component_of=None):
    """-> ({id: Node}, roots). The only entry point callers need."""
    nodes, roots = build_forest(regions, cfg)
    box = {r.id: r.box for r in regions}
    for n in nodes.values():
        n.layout = infer_layout(n, box, cfg) if n.children else None
    for n in nodes.values():
        n.role = infer_role(n, nodes, box, observations, page_box, cfg, component_of)
        n.atomic = ("organism" if len(n.children) >= 3
                    else "molecule" if n.children else "atom")
    return nodes, roots


def to_text(nodes, roots, spec_by_id=None, max_depth=8):
    """Indented rendering — the cheap, high-signal form to paste into a coding
    agent's context."""
    spec_by_id = spec_by_id or {}
    lines = []

    def walk(rid, indent):
        n = nodes[rid]
        if indent > max_depth:
            return
        s = spec_by_id.get(rid, {})
        box = s.get("box_xyxy", [])
        bits = [f"<{n.role}", f"#{rid}"]
        if n.layout:
            L = n.layout
            bits.append(f"{L['display']}:{L.get('direction', '')}")
            if L.get("display") == "grid" and L.get("columns", 1) > 1:
                bits.append(f"cols={L['columns']}")
            for k in ("gap_x", "gap_y"):
                if L.get(k):
                    bits.append(f"{k.replace('_', '-')}={L[k]}")
        if box:
            bits.append(f"box={box}")
        if s.get("class"):
            bits.append(f"class={s['class']}")
        if s.get("delivery") and s["delivery"] != "css":
            bits.append(f"deliver={s['delivery']}")
        if s.get("component_instances", 1) > 1:
            bits.append(f"component={s['component']}x{s['component_instances']}")
        if s.get("subject"):
            bits.append(f'"{s["subject"][:34]}"')
        lines.append("  " * indent + " ".join(bits) + ">")
        for c in n.children:
            walk(c, indent + 1)

    for rid in roots:
        walk(rid, 0)
    return "\n".join(lines)


def to_json(nodes, roots, spec_by_id=None):
    spec_by_id = spec_by_id or {}

    def build(rid):
        n = nodes[rid]
        s = spec_by_id.get(rid, {})
        out = {"id": rid, "role": n.role, "atomic": n.atomic}
        if s.get("box_xyxy"):
            out["box_xyxy"] = s["box_xyxy"]
        if n.layout:
            out["layout"] = n.layout
        if s.get("component_instances", 1) > 1:
            out["component"] = {"key": s.get("component"),
                                "instances": s["component_instances"]}
        for k in ("class", "delivery", "action", "cost", "regenerable", "a11y",
                  "subject", "flags", "regen"):
            if s.get(k) not in (None, [], ""):
                out[k] = s[k]
        if n.children:
            out["children"] = [build(c) for c in n.children]
        return out

    return [build(r) for r in roots]
