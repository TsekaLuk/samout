"""The classification rule set.

Every rule that used to be a branch in `taxonomy.classify` or in `handoff.build`'s
first pass, moved here as a record with its reason and the measurement that forced
it. Priority order is the order the branches ran in; the behaviour is unchanged and
`tests/test_smoke.py` pins that.

Reading this file top to bottom is the answer to "how does a region get its class",
which previously required reading two functions in two modules and simulating the
control flow.

Grouping by `tags` shows what the rule set is really made of:

    identity    a mark is a mark regardless of how it is drawn
    structure   a region holding other regions is layout, not art
    artifact    placeholders and pipeline failures are not design classes
    material    the Material system-vs-product icon test
    content     what the pixels depict

`structure` and `identity` keep colliding — a container that holds a logo, a tap
target that wraps one icon, a disc that wraps a glyph. That collision is the
symptom of a missing abstraction (region extent vs semantic extent), noted in
TODO.md. Naming the tags at least makes the collision visible.
"""

from .rules import Rule
from .taxonomy import SYSTEM_ICON_CRITERIA as C
from .taxonomy import is_placeholder

MARK_MAX_CHILDREN = 2


def _pictogram_class(ctx):
    """The Material system-vs-product test, kept as one function because its three
    branches are one decision and splitting them would obscure the ordering that
    matters: the measurement is consulted at its confident end FIRST."""
    hues = ctx.o("hue_count", 1)
    cues = set(ctx.o("depth_cues") or [])
    lum = ctx.m("lum_std_inner")

    if lum is not None and lum >= C["lum_std_max"]:
        return "system_icon"
    if hues > C["max_hues"]:
        return "product_icon"
    if cues & C["product_cues"]:
        return "product_icon"
    if lum is not None:
        return "product_icon" if lum > C["lum_std_min"] else "system_icon"
    return "product_icon" if len(cues) >= C["min_cues_for_product"] else "system_icon"


RULES = [
    Rule(
        name="unobserved",
        priority=5,
        when=lambda c: not c.obs,
        then="unobserved",
        why="no observation returned; rerun or inspect by hand",
        evidence="an empty class is not 'plain' — silently defaulting loses assets",
        tags=("artifact",),
    ),
    Rule(
        name="loading_placeholder",
        priority=10,
        when=lambda c: is_placeholder(c.measured, c.reference),
        then="token",
        why="achromatic featureless block — a loading placeholder",
        evidence="saturation 0.009-0.015 vs 0.227-0.737 for content; "
                 "entropy 2.5 vs 7.4. Both gaps an order of magnitude",
        tags=("artifact",),
    ),
    Rule(
        name="brand_mark",
        priority=20,
        when=lambda c: c.o("is_brand_mark") and c.n_children < MARK_MAX_CHILDREN,
        then="brand_asset",
        why="identity-locked mark (outranks structure)",
        evidence="a synthesised trademark is a fidelity and a legal failure",
        tags=("identity",),
    ),
    Rule(
        name="region_containing_a_mark",
        priority=25,
        when=lambda c: c.o("is_brand_mark") and c.n_children >= MARK_MAX_CHILDREN,
        then="composite",
        why="contains a brand mark but holds several elements — a region with a "
            "logo, not a logo",
        evidence="real marks had 0-1 children across two screens; every false "
                 "positive had 2, 4 or 5",
        tags=("identity", "structure"),
    ),
    Rule(
        name="split_container_shell",
        priority=30,
        when=lambda c: c.is_split_parent,
        then="token",
        why="container shell left after its glyph was split out",
        evidence="the split gate only fires on high interior variance — a shell "
                 "the backdrop shows through, which is CSS",
        tags=("structure",),
    ),
    Rule(
        name="composite_container",
        priority=40,
        when=lambda c: c.n_children > 0,
        then="composite",
        why="holds other classified regions — layout, not art",
        evidence="Atomic Design: an organism carries layout; its children carry "
                 "the content",
        tags=("structure",),
    ),
    Rule(
        name="photographic",
        priority=50,
        when=lambda c: c.o("content_type") == "photographic",
        then="photography",
        why="camera or photoreal imagery",
        tags=("content",),
    ),
    Rule(
        name="illustration",
        priority=55,
        when=lambda c: c.o("content_type") == "illustration",
        then="spot_illustration",
        why="illustrative artwork, not a pictogram",
        tags=("content",),
    ),
    Rule(
        name="display_lettering",
        priority=60,
        when=lambda c: c.o("content_type") == "display_lettering",
        then="spot_illustration",
        why="custom letterforms, must ship as art",
        evidence="without this class, 3D headlines fell into `text` and were "
                 "rebuilt with a web font",
        tags=("content",),
    ),
    Rule(
        name="live_text",
        priority=65,
        when=lambda c: c.o("content_type") == "text",
        then="typography",
        why="live text in a type style",
        tags=("content",),
    ),
    Rule(
        name="filled_control",
        priority=70,
        when=lambda c: c.o("content_type") == "control",
        then="token",
        why="filled control; shape is CSS, label is live copy",
        evidence="reading a pill-with-a-label as `text` loses the pill — four of "
                 "thirteen errors before this class existed",
        tags=("content",),
    ),
    Rule(
        name="pictogram",
        priority=80,
        when=lambda c: c.o("content_type") == "pictogram",
        then=None,                       # resolved by _pictogram_class
        why="Material system-vs-product icon test",
        evidence="`bevel`+`specular_highlight` appear identically on both classes "
                 "and carry no signal; interior luminance variance does",
        tags=("material",),
    ),
    Rule(
        name="plain_shape",
        priority=90,
        when=lambda c: c.o("content_type") == "plain",
        then="token",
        why="describable with tokens alone",
        tags=("content",),
    ),
    Rule(
        name="fallback",
        priority=999,
        when=lambda c: True,
        then="token",
        why="no pictorial content identified",
        tags=("content",),
    ),
]


def classify(ctx):
    """-> (class, why, rule_name, shadowed_rule_names)"""
    from .rules import evaluate

    cls, why, name, shadowed = evaluate(RULES, ctx)
    if name == "pictogram":
        cls = _pictogram_class(ctx)
        lum = ctx.m("lum_std_inner")
        why = (f"{cls} by the Material test"
               + (f" (interior variance {lum:.0f})" if lum is not None else ""))
    return cls, why, name, shadowed
