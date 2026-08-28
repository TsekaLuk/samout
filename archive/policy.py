"""Routing policy: observations -> how the coding agent should rebuild each region.

This module is deliberately model-free. Everything subjective lives here as data
you can read and tune, not inside a prompt where it gets rationalised away.

Two ideas carry the generality:

1. ROUTES, not categories. "asset / library / css" describes the means, not the
   question. The question is which reconstruction path is cheapest while still
   landing close enough to the design. Adding a path is adding a row, not
   rewriting a taxonomy.

2. COMPOSITION. A region is atomic or it is a container of other regions. A
   container is never itself an asset — its children are routed individually and
   it becomes CSS layout. This is what stops whole nav bars and promo strips from
   being sent to an image model.
"""

from dataclasses import dataclass, field

# cost is ordinal effort for the downstream coding agent, not currency:
#   0  write CSS you were going to write anyway
#   1  add an icon-library import
#   2  slice the mockup and ship a bitmap (free, but inherits mockup resolution
#      and any text baked into it)
#  10  a round trip through an image model, then art-direction review
ROUTES = [
    ("css_only", 0),
    ("library_icon", 1),
    ("mockup_crop", 2),
    ("generate", 10),
]
ROUTE_COST = dict(ROUTES)


@dataclass
class Policy:
    """The subjective knobs, all in one place."""

    # Max visual drift tolerated from a route, 0..1. Lower = more assets get
    # generated, higher = the agent hand-builds more.
    tau: float = 0.3

    # Below this VLM confidence the region is sent to review rather than routed.
    min_confidence: float = 0.45

    # mockup_crop inherits the mockup's pixel density. Refuse it for regions that
    # would need upscaling past this factor to hit the target raster size.
    crop_max_upscale: float = 2.0

    # A container is a region that geometrically holds >= this many other regions.
    # Containers route to css_only regardless of what they look like.
    container_min_children: int = 2

    # Routes that are disabled for this project, e.g. drop "library_icon" if the
    # product has no icon-set dependency.
    disabled: set = field(default_factory=set)


def is_container(region, policy):
    """Layout scaffolding, not art. Its children carry the actual content."""
    return len(region.get("children") or []) >= policy.container_min_children


def crop_viable(region, policy, target_scale=3):
    """Can we just slice the mockup? Only if it has the pixels to spare."""
    w, h = region.get("size_px", [0, 0])
    if min(w, h) <= 0:
        return False
    # target is @Nx of the region's CSS size; the mockup gives us w*h as-is
    return target_scale <= policy.crop_max_upscale


def resolve(region, obs, policy=None):
    """-> (route, reason)

    `obs` is the VLM's per-route risk estimate plus its confidence. Everything
    here is a comparison against knobs; no heuristics about what the pixels are.
    """
    policy = policy or Policy()

    if is_container(region, policy):
        return "css_only", (f"container of {len(region['children'])} regions; "
                            "children routed individually")

    conf = obs.get("confidence", 1.0)
    if conf < policy.min_confidence:
        return "review", f"low confidence {conf:.2f} < {policy.min_confidence}"

    risk = obs.get("reproduction_risk") or {}
    viable = []
    for name, cost in ROUTES:
        if name in policy.disabled:
            continue
        if name == "mockup_crop" and not crop_viable(region, policy):
            continue
        r = risk.get(name)
        if r is None:
            continue
        if r <= policy.tau:
            viable.append((cost, name, r))

    if not viable:
        best = min(risk.items(), key=lambda kv: kv[1]) if risk else None
        if best and best[1] <= policy.tau * 2:
            return best[0], f"no route under tau={policy.tau}; best is {best[0]} at {best[1]:.2f}"
        return "review", (f"every route exceeds tau={policy.tau} "
                          f"(min {best[1]:.2f} via {best[0]})" if best else "no risk data")

    viable.sort()
    cost, name, r = viable[0]
    return name, f"cheapest route under tau: {name} at risk {r:.2f}"


def resolve_all(regions, observations, policy=None):
    """regions: list of dicts with id/children/size_px. observations: {id: obs}."""
    policy = policy or Policy()
    out = {}
    for reg in regions:
        obs = observations.get(str(reg["id"])) or observations.get(reg["id"]) or {}
        route, why = resolve(reg, obs, policy)
        out[reg["id"]] = {"route": route, "why": why,
                          "cost": ROUTE_COST.get(route, 0),
                          "confidence": obs.get("confidence"),
                          "risk": obs.get("reproduction_risk")}
    return out


def summarize(resolved):
    from collections import Counter
    c = Counter(v["route"] for v in resolved.values())
    total_cost = sum(v["cost"] for v in resolved.values())
    return {"by_route": dict(c), "total_cost": total_cost,
            "needs_generation": c.get("generate", 0),
            "needs_review": c.get("review", 0)}
