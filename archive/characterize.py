"""Stage 2: observe each region, then classify it with taxonomy.py.

The VLM answers only countable or pointable questions — how many hues, is there a
cast shadow, is this a trademark, is text baked into the pixels. It is never asked
what class the region is or what to do with it, because those are exactly the
judgements it rationalises away. Classification lives in taxonomy.py, against
published design-system criteria.

Also runs a recall audit: SAM 3 finds only concepts it has vocabulary for, so a
bespoke mascot can be missed entirely. The VLM is shown what was found and asked
what is missing — the answer feeds back as new SAM 3 prompts.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from components import consensus, group
from taxonomy import CLASSES, classify, deliver
from uitree import annotate, to_json, to_text
from vlm import chat, encode_image, parse_json

OBSERVE_SYSTEM = """You are doing a design-system audit of regions cut from a UI mockup.

Report only what you can COUNT or POINT AT. Do not classify, recommend, or decide how
anything should be built — a downstream rule does that against published design-system
criteria. Your accuracy on the observations is the whole job.

For each region report:

`content_type` — exactly one:
  "photographic"  a camera image, or a rendered scene indistinguishable from one
  "illustration"  drawn or composed artwork that is not a pictogram: scenes, decorative
                  art, mascots, neon signage, banner art
  "pictogram"     a small symbol standing for a concept or action
  "text"          live copy set in an ordinary typeface
  "plain"         no pictorial content at all — a pill, bar, divider, empty container

`hue_count` — how many DISTINCT HUES appear, ignoring tints and shades of the same hue.
  A white glyph = 1. A pink-to-purple gradient = 2. A calendar icon with an orange body,
  white check and grey shadow = 2 (grey is neutral, not a hue). Count carefully: this
  number decides whether the icon can come from an icon library. Neutrals (white, black,
  grey) do not count as hues.

`depth_cues` — every one you can actually SEE, using ONLY these terms:
  specular_highlight   a bright spot where a light source reflects off the surface
  cast_shadow          a shadow the object throws onto what is behind it
  bevel                a rounded or chamfered edge that catches light along one side
  inner_shadow         shading inside the shape implying thickness
  occlusion            one part of the object overlapping and darkening another
  material_texture     visible metal, glass, plastic, fabric, gold
  perspective          the object is drawn at an angle rather than flat-on
  glow                 light bleeding outward from the shape
  Empty list if the shape is flat or flat-gradient with none of the above.

`is_brand_mark` — true only for an identity-locked mark: the product's own logo or
  wordmark, a partner logo, a certification badge. A generic icon is not a brand mark
  even when it is on-brand. When true, say which brand in `subject`.

`has_baked_text` — is readable copy rendered INTO the pixels, such that translating the
  product would require a new image? Text next to the region does not count.

`needs_transparency` — must this sit on a varying background, requiring an alpha channel?

`theme_dependent` — would this need a different version in light vs dark theme?

OUTPUT: JSON array, one entry per id shown, no prose.
[
  {"id": <int>,
   "subject": "<what it depicts, max 8 words>",
   "content_type": "photographic"|"illustration"|"pictogram"|"text"|"plain",
   "hue_count": <int>,
   "depth_cues": [<terms from the list above>],
   "is_brand_mark": <bool>,
   "has_baked_text": <bool>,
   "needs_transparency": <bool>,
   "theme_dependent": <bool>,
   "closest_library_icon": "<e.g. 'lucide:map-pin', or null if not a pictogram>",
   "confidence": 0.0-1.0,
   "regen_prompt": "<standalone prompt for an image model to recreate this region — \
subject, material, lighting, palette, background handling. null for text and plain>"}
]
Return an entry for EVERY id shown. Do not skip any."""

RECALL_SYSTEM = """You are checking an open-vocabulary segmenter for misses.

You get a UI mockup with every already-detected region outlined in green. The segmenter
only finds concepts it has words for, so bespoke artwork gets missed entirely.

Name any PICTORIAL content that is NOT already outlined — photographs, illustrations,
rendered objects, mascots, logos, decorative artwork, textured backgrounds.

Ignore: live text, flat shapes, plain glyph icons, anything already outlined, and a
plain gradient page background.

OUTPUT: JSON array. An empty array is a normal and expected answer.
[
  {"what": "<short description>",
   "box_pct": [x0, y0, x1, y1],
   "sam3_prompt": "<a concrete physical noun that would make an open-vocabulary \
segmenter find it — 1-3 words, e.g. 'disco ball'. Never an abstract or style word.>",
   "confidence": 0.0-1.0}
]"""


def region_crop(image, r, cut_dir=None):
    """Prefer the mask cutout over the bounding-box crop, composited onto neutral grey.

    This matters more than any prompt wording: with box crops, the observer read
    `bevel` and `glow` off the card photograph *behind* a play button and called
    seven flat glyphs product icons. That single confusion was 53% of all error.
    Neutral grey rather than white or black so it adds no implied lighting.
    """
    if cut_dir:
        p = cut_dir / f"{r['id']:02d}_cutout.png"
        if p.exists():
            cut = Image.open(p).convert("RGBA")
            flat = Image.new("RGB", cut.size, (128, 128, 128))
            flat.paste(cut, (0, 0), cut)
            return flat
    return image.crop(tuple(r["box_xyxy"])).convert("RGB")


def contact_sheet(image, regions, cell=190, cols=6, pad=10, cut_dir=None):
    rows = (len(regions) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad,
                              rows * (cell + pad + 18) + pad), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(regions):
        crop = region_crop(image, r, cut_dir)
        crop.thumbnail((cell, cell), Image.LANCZOS)
        cx = pad + (k % cols) * (cell + pad)
        cy = pad + (k // cols) * (cell + pad + 18)
        d.rectangle([cx - 2, cy - 2, cx + cell + 2, cy + cell + 2], outline=(90, 90, 100))
        sheet.paste(crop, (cx + (cell - crop.width) // 2, cy + (cell - crop.height) // 2))
        d.text((cx + 2, cy + cell + 4), f"#{r['id']}", fill=(255, 220, 0))
    return sheet


def _batch(job):
    """Never raises. A dead batch must not discard the batches that succeeded —
    the endpoint drops connections often enough that all-or-nothing runs rarely
    finish on a large mockup."""
    image, chunk, model, ctx, n, cut_dir = job
    meta = "\n".join(f"#{r['id']}: {r['size_px'][0]}x{r['size_px'][1]}px" for r in chunk)
    try:
        text, usage, dt = chat(
            model,
            [ctx, "Full mockup for context. Below, the regions to observe, each labelled.",
             encode_image(contact_sheet(image, chunk, cut_dir=cut_dir), max_side=1500),
             f"Regions:\n{meta}\n\nObserve ids {[r['id'] for r in chunk]}. JSON array only."],
            system=OBSERVE_SYSTEM, max_tokens=8000, temperature=0.0)
        parsed = parse_json(text)
        print(f"  batch {n}: {len(parsed)}/{len(chunk)}, {dt}s")
        return parsed, usage
    except Exception as e:
        print(f"  batch {n}: FAILED ({type(e).__name__}) — will retry")
        return None, {}


def observe(image, regions, model, batch=10, workers=6, sweeps=3, cut_dir=None):
    """Re-sweeps whatever is still missing, shrinking the batch each pass. A batch
    that fails on transport succeeds alone; one that fails on output length succeeds
    when split."""
    ctx = encode_image(image, max_side=900)
    obs, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()

    pending = list(regions)
    for sweep in range(sweeps):
        if not pending:
            break
        size = max(1, batch // (2 ** sweep))
        chunks = [pending[i:i + size] for i in range(0, len(pending), size)]
        if sweep:
            print(f"  sweep {sweep + 1}: {len(pending)} regions left, batch={size}")
        jobs = [(image, c, model, ctx, f"{sweep + 1}.{i + 1}", cut_dir)
                for i, c in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for parsed, u in ex.map(_batch, jobs):
                for v in (parsed or []):
                    try:
                        obs[int(v["id"])] = v
                    except (KeyError, TypeError, ValueError):
                        continue
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
        pending = [r for r in pending if r["id"] not in obs]

    if pending:
        print(f"  !! {len(pending)} regions unobserved after {sweeps} sweeps: "
              f"{[r['id'] for r in pending]}")
    return obs, usage, round(time.time() - t0, 1)


def recall_audit(image, regions, model):
    marked = image.copy()
    d = ImageDraw.Draw(marked)
    for r in regions:
        d.rectangle(tuple(r["box_xyxy"]), outline=(0, 255, 60), width=3)
    text, _, dt = chat(
        model,
        [encode_image(marked, max_side=1400),
         f"{len(regions)} regions are already outlined. What pictorial content was "
         "missed? JSON array only; empty array if nothing."],
        system=RECALL_SYSTEM, max_tokens=2000, temperature=0.0)
    return parse_json(text), dt


def build_handoff(tree, roots, obs, groups=None):
    """Per-region design handoff spec — the artifact a design team actually ships.

    Internal vs leaf comes from the UI tree, so "is this layout or is this art" is
    a structural fact rather than a threshold. A node with children is layout.
    """
    spec = []
    gsize = {g["key"]: g["count"] for g in (groups or [])}
    for r in tree.values():
        o = obs.get(r["id"], {})
        if r["children"]:
            cls, why = "composite", (f"{r['atomic']} with {len(r['children'])} children, "
                                     f"role={r['role']}")
        elif not o:
            # Unobserved is not the same as "plain". Surface it instead of
            # defaulting into the cheapest class and silently losing an asset.
            cls, why = "unobserved", "no observation returned; needs a rerun or a human"
        else:
            cls, why = classify(o)
        d = deliver(cls)
        entry = {
            "id": r["id"], "subject": o.get("subject"),
            "class": cls, "why": why,
            "role": r["role"], "atomic": r["atomic"], "depth": r["depth"],
            "delivery": d["delivery"], "action": d["action"], "cost": d["cost"],
            "regenerable": d["regenerable"], "a11y": d["a11y"],
            "box_xyxy": [int(v) for v in r["box_xyxy"]], "size_px": r["size_px"],
            "parent": r.get("parent"), "children": r.get("children", []),
            "layout": r.get("layout"),
            "component": r.get("component"),
            "component_instances": gsize.get(r.get("component"), 1),
            "observed": {k: o.get(k) for k in
                         ("content_type", "hue_count", "depth_cues", "is_brand_mark",
                          "has_baked_text", "needs_transparency", "theme_dependent",
                          "closest_library_icon", "confidence")},
            "flags": [],
        }
        if entry["component_instances"] > 1 and not r["children"]:
            entry["flags"].append(
                f"component: {entry['component_instances']} instances share this "
                "class; build once, reuse")
        # Handoff defects a designer would raise at review.
        if o.get("has_baked_text"):
            entry["flags"].append("i18n: copy baked into pixels, cannot be localized")
        if cls == "brand_asset":
            entry["flags"].append("brand: extract exactly, do not regenerate")
        if o.get("theme_dependent"):
            entry["flags"].append("theming: needs a light and a dark variant")
        if o.get("needs_transparency") and d["delivery"].startswith("asset"):
            entry["flags"].append("alpha: deliver with transparency")
        if (o.get("confidence") or 1.0) < 0.5:
            entry["flags"].append("review: low observation confidence")
        if cls == "product_icon":
            entry["flags"].append("prefer SVG redraw if the icon language is systematic")
        spec.append(entry)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--model", default="qwen3-vl-flash")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--skip-recall", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    man = json.loads(args.manifest.read_text())
    out = args.out or args.manifest.parent

    regions = [{"id": a["id"], "box_xyxy": a["box_xyxy"], "size_px": a["size_px"],
                "parent": a.get("parent"), "children": a.get("children", [])}
               for a in man["assets"]]
    regions += [{"id": c["id"], "box_xyxy": c["box_xyxy"],
                 "size_px": [c["box_xyxy"][2] - c["box_xyxy"][0],
                             c["box_xyxy"][3] - c["box_xyxy"][1]],
                 "parent": c.get("parent"), "children": []}
                for c in man["css_regions"]]
    regions.sort(key=lambda r: r["id"])

    # 1. Group repeated regions. A design system has one Play Button used eight
    #    times, not eight play buttons — so observe the component, not the instance.
    groups, of_group = group(image, regions)
    reps = {g["representative"] for g in groups}
    repeated = [g for g in groups if g["count"] > 1]
    print(f"{len(regions)} regions -> {len(groups)} components "
          f"({len(repeated)} repeated, saving {len(regions) - len(groups)} calls)")
    for g in sorted(repeated, key=lambda g: -g["count"])[:6]:
        print(f"  {g['key']}: x{g['count']}  ids {g['members'][:8]}")

    # 2. Observe a few probes per component, not every instance.
    probe_ids = {i for g in groups for i in g["probes"]}
    to_observe = [r for r in regions if r["id"] in probe_ids]
    print(f"observing {len(to_observe)} probes across {len(groups)} components")
    obs_probes, usage, dt = observe(image, to_observe, args.model, args.batch,
                                    cut_dir=args.manifest.parent / 'cutouts')
    print(f"  {dt}s  in={usage['prompt_tokens']} out={usage['completion_tokens']}")

    # 3. Reconcile probes, then propagate the component's verdict to every instance.
    obs = {}
    for g in groups:
        merged = consensus([obs_probes.get(i) for i in g["probes"]])
        if not merged:
            continue
        if merged.get("consensus", {}).get("cues_dropped"):
            print(f"  {g['key']} x{g['count']}: dropped background cues "
                  f"{merged['consensus']['cues_dropped']}")
        for rid in g["members"]:
            obs[rid] = merged

    # 4. The UI tree is the backbone: internal node = layout, leaf = content.
    page_box = [0, 0, image.width, image.height]
    tree, roots = annotate(regions, obs, page_box)
    for rid, n in tree.items():
        n["component"] = of_group.get(rid)
        n["component_count"] = next(
            (g["count"] for g in groups if g["key"] == n["component"]), 1)
        n["size_px"] = next(r["size_px"] for r in regions if r["id"] == rid)
    print(f"tree: {len(roots)} roots, depth {max(n['depth'] for n in tree.values())}, "
          f"{sum(1 for n in tree.values() if n['children'])} internal / "
          f"{sum(1 for n in tree.values() if not n['children'])} leaves")

    spec = build_handoff(tree, roots, obs, groups)

    missed = []
    if not args.skip_recall:
        try:
            missed, rdt = recall_audit(image, regions, args.model)
            print(f"recall audit: {rdt}s, {len(missed)} missed")
            for m in missed:
                print(f"    {str(m.get('what'))[:44]:<46} retry SAM3 with "
                      f"'{m.get('sam3_prompt')}'")
        except Exception as e:
            print(f"recall audit failed: {e}")

    by_class = Counter(e["class"] for e in spec)
    print("\nclass mix:")
    for cls in CLASSES:
        if by_class.get(cls):
            print(f"  {cls:<18} {by_class[cls]:>3}   {CLASSES[cls]['delivery']}")
    gen = [e for e in spec if e["delivery"].startswith("asset") and e["regenerable"]]
    exact = [e for e in spec if e["delivery"] == "asset_exact"]
    print(f"\ngenerate: {len(gen)}   extract-exact: {len(exact)}   "
          f"total effort: {sum(e['cost'] for e in spec)}")
    flagged = [e for e in spec if e["flags"]]
    if flagged:
        print(f"\n{len(flagged)} regions carry handoff flags:")
        for e in flagged[:10]:
            print(f"  #{e['id']:<3} {e['class']:<18} {e['flags'][0][:58]}")

    spec_by_id = {e["id"]: e for e in spec}
    tree_txt = to_text(tree, roots, spec_by_id)
    (out / "uitree.txt").write_text(tree_txt)
    (out / "uitree.json").write_text(json.dumps(
        to_json(tree, roots, spec_by_id), indent=2, ensure_ascii=False))
    (out / "handoff.json").write_text(json.dumps(
        {"model": args.model, "spec": spec, "missed": missed,
         "components": groups, "class_mix": dict(by_class),
         "usage": usage, "latency_s": dt}, indent=2, ensure_ascii=False))

    print("\nUI tree (first 24 lines):")
    print("\n".join(tree_txt.splitlines()[:24]))
    print(f"\nwrote {out}/handoff.json, {out}/uitree.txt, {out}/uitree.json")


if __name__ == "__main__":
    main()
