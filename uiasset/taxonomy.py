"""Design-system asset taxonomy and delivery rules.

Nothing in this file is invented for this project. The classes below are the ones
mainstream design systems already ship and hand off, and the delivery rules mirror
how design teams actually deliver each class to engineering. Sources are named per
class so the boundaries can be argued with a designer rather than with a threshold.

Why this replaces an ad-hoc taxonomy: an invented axis ("how 3D does it look") is
fitted to whatever screen it was invented on. These classes already survived contact
with thousands of products, and — critically — the industry has already decided how
each one is delivered, which is the question we actually need answered.

References
  W3C Design Tokens Community Group, Format Module — the formal list of visual
    properties expressible as tokens: color, dimension, border, shadow, gradient,
    typography, strokeStyle. Anything fully describable in tokens is code, not asset.
  Material Design 3 — "System icons" vs "Product icons". System icons are monochrome,
    geometric, drawn on a 24dp grid with a 20dp live area and consistent stroke weight.
    Product icons are expressive, multi-colour, dimensional and brand-specific.
  Apple Human Interface Guidelines — SF Symbols (weight-matched to text, monochrome /
    hierarchical / palette / multicolour) vs app icons vs custom artwork.
  Brad Frost, Atomic Design — atoms / molecules / organisms. Containers are organisms;
    they carry layout, not art.
"""

# ---------------------------------------------------------------------------
# Asset classes
# ---------------------------------------------------------------------------
# delivery : how a design team hands this class to engineering
# regenerable : may an image model produce it? brand marks may not.
# a11y : default accessibility treatment
CLASSES = {
    "token": {
        "definition": "Fully describable with design tokens — fill, gradient, radius, "
                      "border, elevation. No pictorial content.",
        "source": "W3C Design Tokens Format Module",
        "delivery": "css",
        "regenerable": False,
        "a11y": "presentation",
        "examples": "pills, bars, dividers, plain containers, solid or gradient panels",
    },
    "typography": {
        "definition": "Live text set in a type style. Includes text over artwork.",
        "source": "W3C Design Tokens (typography composite type)",
        "delivery": "css",
        "regenerable": False,
        "a11y": "text",
        "examples": "labels, headings, body copy, numerals, badges with live counts",
    },
    "system_icon": {
        "definition": "Monochrome or single-hue functional pictogram, geometric, "
                      "grid-aligned, uniform stroke. Communicates an action or object.",
        "source": "Material 3 system icons; Apple SF Symbols",
        "delivery": "icon_library",
        "regenerable": False,
        "a11y": "informative",
        "examples": "search, play, home, chevron, battery, wifi, notification badge",
    },
    "product_icon": {
        "definition": "Expressive multi-colour pictogram with dimension or material — "
                      "the brand's own icon language, not a library glyph.",
        "source": "Material 3 product icons; Apple app/custom icons",
        "delivery": "asset_svg_or_raster",
        "regenerable": True,
        "a11y": "informative",
        "examples": "the shaded 3D feature icons on a home screen, coupon medallions",
    },
    "spot_illustration": {
        "definition": "Standalone illustrative artwork that is not a pictogram — "
                      "decorative or narrative.",
        "source": "Material 3 illustration guidance; Shopify Polaris illustrations",
        "delivery": "asset_raster",
        "regenerable": True,
        "a11y": "decorative",
        "examples": "empty-state art, hero scenes, decorative neon, mascots",
    },
    "photography": {
        "definition": "Camera imagery or photorealistic rendered content.",
        "source": "Material 3 imagery; HIG images",
        "delivery": "asset_raster",
        "regenerable": True,
        "a11y": "informative",
        "examples": "artist portraits, venue photos, album covers, food shots",
    },
    "brand_asset": {
        "definition": "Identity-locked mark: logo, wordmark, lockup, brand badge.",
        "source": "Brand guidelines; Material 3 branding; HIG app identity",
        "delivery": "asset_exact",
        # A regenerated trademark is both a fidelity failure and a legal one.
        # This is the one class that must be extracted, never synthesised.
        "regenerable": False,
        "a11y": "informative",
        "examples": "the product wordmark, a partner logo, a certification badge",
    },
    "unobserved": {
        "definition": "No observation was returned for this region.",
        "source": "pipeline failure mode, not a design class",
        "delivery": "review",
        "regenerable": False,
        "a11y": "unknown",
        "examples": "batches lost to transport errors",
    },
    "composite": {
        "definition": "A region that contains other classified regions. Layout, not art.",
        "source": "Atomic Design — organisms",
        "delivery": "css",
        "regenerable": False,
        "a11y": "presentation",
        "examples": "nav bars, cards, promo strips, section wrappers",
    },
}

# Delivery -> what the coding agent actually does, and what it costs in effort.
DELIVERY = {
    "css":                 {"cost": 0,  "action": "write CSS/HTML using design tokens"},
    "icon_library":        {"cost": 1,  "action": "import from the project icon set"},
    "asset_exact":         {"cost": 2,  "action": "extract from mockup; never regenerate"},
    "asset_svg_or_raster": {"cost": 6,  "action": "redraw as SVG, or generate raster"},
    "asset_raster":        {"cost": 10, "action": "generate with an image model"},
    "review":              {"cost": 0,  "action": "unresolved — rerun or inspect by hand"},
}

# The discipline's own test for system vs product icon, expressed as the observable
# properties Material actually specifies. Kept here so it can be cited and tuned,
# not embedded in a prompt.
SYSTEM_ICON_CRITERIA = {
    # Material system icons are monochrome. Holding to that literally separates the
    # labelled data; the earlier allowance of 2 was a guess that cost accuracy.
    "max_hues": 1,

    # Calibrated against eval/labels, not assumed. Counted over 24 labelled
    # pictograms, `bevel` and `specular_highlight` appear on BOTH classes in exactly
    # the same pairing — 6 product icons and 9 system icons all report
    # {bevel, specular_highlight} and nothing else. That pair carries no signal for
    # this observer: it fires on any gradient with a lit edge, which includes the
    # translucent disc bundled behind a flat play glyph.
    #
    # These, by contrast, appeared only on product icons:
    "product_cues": {"occlusion", "inner_shadow", "cast_shadow",
                     "material_texture", "perspective"},

    # A weak cue alone means nothing; three or more of anything means the shape is
    # genuinely modelled. Only used when the measured feature is unavailable.
    "weak_cues": {"bevel", "specular_highlight", "glow"},
    "min_cues_for_product": 3,

    # Interior luminance standard deviation, measured from the mask cutout.
    # Calibrated on 24 labelled pictograms:
    #   flat library glyphs        20-34   (uniform fill)
    #   solid product icons        25-67   (smooth shading across a solid body)
    #   glyph on translucent disc  83-101  (the backdrop shows through)
    # The upper bound has a 20.7-wide margin (product max 66.6, next system 87.3).
    # The lower bound does not separate cleanly and is set conservatively; the
    # `sat_std` term that would have closed it had a margin one sample wide and was
    # deliberately not shipped. See OPTIMIZATION.md "calibration is not a licence
    # to overfit".
    "lum_std_max": 83.0,
    "lum_std_min": 24.0,
}


def classify(obs, measured=None):
    """Observable properties (+ computed pixel measurements) -> design-system class.

    Applies the published criteria in the order a designer would: identity first
    (a mark is a mark regardless of how it is drawn), then content type, then the
    Material system-vs-product icon test.

    `measured` carries quantities computed from the pixels rather than reported by
    the model. Anything computable belongs there: asking a VLM to count hues was
    both less accurate than measuring and, as it turned out, measuring the wrong
    thing. See OPTIMIZATION.md L2.
    """
    measured = measured or {}

    if obs.get("is_brand_mark"):
        return "brand_asset", "identity-locked mark"

    content = obs.get("content_type")
    if content == "photographic":
        return "photography", "camera or photoreal imagery"
    if content == "illustration":
        return "spot_illustration", "illustrative artwork, not a pictogram"
    if content == "display_lettering":
        # Custom letterforms: a 3D extruded headline, a neon word, a wordmark.
        # Without this class such regions fell into `text` (and got rebuilt with a
        # web font) or `photographic`, which was three of twelve errors.
        return "spot_illustration", "custom letterforms, must ship as art"
    if content == "text":
        return "typography", "live text in a type style"
    if content == "control":
        # A button is a filled shape with a label on it: the shape is a token and
        # the label is live copy. Reading the pair as "text" loses the shape, which
        # was four of thirteen errors before this class existed.
        return "token", "filled control; shape is CSS, label is live copy"
    if content == "plain":
        return "token", "describable with tokens alone"

    if content == "pictogram":
        C = SYSTEM_ICON_CRITERIA
        hues = obs.get("hue_count", 1)
        cues = set(obs.get("depth_cues") or [])
        strong = cues & C["product_cues"]
        lum_std = measured.get("lum_std_inner")

        # Measured FIRST, because it is deterministic and the VLM fields are not.
        # With the observed fields ahead of it, a flicker in `hue_count` or
        # `depth_cues` preempted a stable measurement: the seven play buttons sit at
        # lum_std 83-90, unambiguously above the threshold, yet their class flipped
        # between runs and — because they are one component — took six regions with
        # it. That single flip was the entire bimodal spread (71.7 <-> 81.1).
        # Deterministic sources outrank noisy ones; the rule order has to say so. A solid rendered object has a smooth interior; a
        # library glyph is either flat (very low variance) or sits on a translucent
        # disc that lets the backdrop through (very high variance). Product icons
        # occupy the middle band. Calibrated on 24 labelled pictograms with a
        # 20.7-wide margin at the upper bound — see OPTIMIZATION.md.
        if lum_std is not None and lum_std >= C["lum_std_max"]:
            return "system_icon", (f"interior variance {lum_std:.0f} — backdrop shows "
                                   "through, not a solid object")

        # Below the ceiling the measurement alone is ambiguous (flat glyphs and solid
        # icons overlap in 24-67), so the observed fields decide, then the floor.
        if hues > C["max_hues"]:
            return "product_icon", f"{hues} hues exceeds the monochrome system-icon rule"
        if strong:
            return "product_icon", f"modelled form: {'/'.join(sorted(strong))}"
        if lum_std is not None:
            if lum_std > C["lum_std_min"]:
                return "product_icon", f"solid shaded interior (variance {lum_std:.0f})"
            return "system_icon", f"flat interior (variance {lum_std:.0f})"

        if len(cues) >= C["min_cues_for_product"]:
            return "product_icon", f"{len(cues)} depth cues"
        bad = set()
        if bad:
            return "product_icon", f"dimensional cues absent from grid glyphs: {'/'.join(sorted(bad))}"
        return "system_icon", f"monochrome geometric glyph ({hues} hue)"

    return "token", "no pictorial content"


def deliver(cls):
    spec = CLASSES[cls]
    d = DELIVERY[spec["delivery"]]
    return {"class": cls, "delivery": spec["delivery"], "action": d["action"],
            "cost": d["cost"], "regenerable": spec["regenerable"],
            "a11y": spec["a11y"]}
