"""Stage C: let a VLM decide, per region, whether CSS can draw it.

Replaces the hand-tuned label thresholds in extract_assets.py. Those thresholds
(T_3D=0.35 and friends) were fitted to one neon KTV screen; they do not transfer.
A VLM looking at the actual pixels does, and it writes a far better regeneration
prompt than a per-class template can.

Regions are sent as numbered contact sheets — one call per batch rather than one
call per region, which is ~15x cheaper for the same coverage.

    python classify_vlm.py input/ktv_ui.jpeg assets/assets.json --model qwen3-vl-plus
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from vlm import chat, encode_image, parse_json

SYSTEM = """You audit regions cut from a UI mockup and route each one to the cheapest \
way a coding agent can rebuild it faithfully.

The agent's cost order, cheapest first:
  1. "css"     — plain CSS/HTML. Fills, gradients, borders, radius, shadows, blurs,
                 geometric shapes, pills, bars, and ALL live text in a normal typeface.
  2. "library" — an off-the-shelf icon from Material Symbols / Lucide / Heroicons /
                 Font Awesome / Iconfont, recolored and sized with CSS.
  3. "asset"   — must be produced by an image model, because nothing off the shelf
                 reproduces it. This is the expensive path; use it only when earned.

YOU DO NOT DECIDE THE VERDICT. You REPORT WHAT YOU SEE. A rule downstream turns your
observations into the routing decision. Do not try to infer or optimise the outcome —
just describe the pixels accurately.

Report these observable properties per region:

`kind` — what the region physically is:
  "photograph"   a camera image or a photorealistic rendered scene
  "artwork"      painted/illustrated/composed graphic, banner art, a whole scene
  "lettering"    text whose letterforms are custom-drawn or heavily styled (a wordmark,
                 a display headline with 3D extrusion or gradient fill)
  "icon"         a small pictogram standing for a concept
  "text"         live product copy in an ordinary typeface
  "shape"        a plain pill / bar / circle / container with no pictorial content

`render_style` — ONLY for kind "icon", "lettering", "artwork". How it is drawn:
  "flat"          one flat colour, no shading
  "flat_gradient" a linear or radial gradient fill, still visually 2D
  "soft_3d"       gradient plus soft shading that implies volume — a bevel, a rounded
                  edge catching light, a subtle inner shadow
  "rendered_3d"   unmistakably a rendered object: specular highlights, cast shadows,
                  material (metal, glass, plastic, gold), occlusion between parts

`depth_cues` — list every one you can actually SEE, from exactly this vocabulary:
  specular_highlight, cast_shadow, bevel, inner_shadow, occlusion, material_texture,
  perspective, glow. Empty list if none.

`concept_in_libraries` — for kind "icon" only. Does a mainstream icon set (Material
Symbols, Lucide, Heroicons, Font Awesome, Iconfont) ship an icon for THIS MEANING?
Judge the meaning only, ignore the styling. home/search/play/message/user/location/
calendar/heart/star/gear/ticket/coin/percent/crown/microphone/music-note are all yes.
A product-specific mascot or a bespoke composite is no.

`closest_library_icon` — the nearest match by meaning, e.g. "lucide:map-pin". Always
fill this for icons. Naming it implies nothing about whether it will be used.

Be literal about `render_style` and `depth_cues`. If an icon has a bevel catching light
and a drop shadow, say so — do not describe it as flat because the underlying shape is
simple. This is the single most important thing you do here.

OUTPUT: JSON array, one entry per region id shown, no prose.
[
  {"id": <int>,
   "kind": "photograph"|"artwork"|"lettering"|"icon"|"text"|"shape",
   "render_style": "flat"|"flat_gradient"|"soft_3d"|"rendered_3d"|null,
   "depth_cues": [<from the vocabulary above>],
   "concept_in_libraries": <bool or null>,
   "closest_library_icon": "<e.g. 'lucide:map-pin', or null>",
   "confidence": 0.0-1.0,
   "reason": "<max 12 words describing the pixels, not a recommendation>",
   "regen_prompt": "<a complete standalone prompt for an image model to recreate this \
region — subject, material, lighting, palette, background handling. Always fill it for \
photograph/artwork/lettering and for any icon with soft_3d or rendered_3d styling; \
null otherwise>"}
]
Return an entry for EVERY id shown. Do not skip any."""


# The subjective call lives HERE, in code you own and can tune — not inside the model,
# where it gets rationalised away. The model reports observables; this maps them to a
# route. Two earlier attempts asked the VLM for the verdict directly and it talked
# itself out of the same six 3D icons both times, in opposite directions ("not a
# standard glyph" -> library, then "3D is just styling" -> library).
ASSET_STYLES = {"soft_3d", "rendered_3d"}
STRONG_CUES = {"specular_highlight", "cast_shadow", "material_texture", "occlusion"}


def route(obs, min_cues=1):
    """observations -> ('asset'|'library'|'css', why)"""
    kind = obs.get("kind")
    style = obs.get("render_style")
    cues = set(obs.get("depth_cues") or [])
    strong = cues & STRONG_CUES

    if kind in ("photograph", "artwork"):
        return "asset", f"{kind}"
    if kind == "lettering":
        return "asset", "custom letterforms"
    if kind in ("text", "shape"):
        return "css", f"{kind}, no pictorial content"

    # kind == "icon"
    if style in ASSET_STYLES:
        return "asset", f"{style}" + (f" + {'/'.join(sorted(strong))}" if strong else "")
    if len(strong) >= min_cues:
        return "asset", f"depth cues: {'/'.join(sorted(strong))}"
    if obs.get("concept_in_libraries") is False:
        return "asset", "bespoke concept, no library equivalent"
    return "library", f"{style or 'flat'} glyph, standard concept"


def contact_sheet(image, regions, cell=190, cols=6, pad=10):
    """Numbered grid of region crops, so one call covers many regions."""
    rows = (len(regions) + cols - 1) // cols
    W = cols * (cell + pad) + pad
    H = rows * (cell + pad + 18) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(regions):
        x0, y0, x1, y1 = r["box_xyxy"]
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((cell, cell), Image.LANCZOS)
        cx = pad + (k % cols) * (cell + pad)
        cy = pad + (k // cols) * (cell + pad + 18)
        d.rectangle([cx - 2, cy - 2, cx + cell + 2, cy + cell + 2], outline=(90, 90, 100))
        sheet.paste(crop, (cx + (cell - crop.width) // 2,
                           cy + (cell - crop.height) // 2))
        d.text((cx + 2, cy + cell + 4), f"#{r['id']}", fill=(255, 220, 0))
    return sheet


def _audit_batch(args_tuple):
    image, chunk, model, ctx_url, n = args_tuple
    sheet = contact_sheet(image, chunk)
    meta = "\n".join(
        f"#{r['id']}: {r['size_px'][0]}x{r['size_px'][1]}px, "
        f"SAM3 labels {r['labels']}" for r in chunk)
    text, usage, dt = chat(
        model,
        [ctx_url,
         "Full mockup above, for context. Below: the regions to audit, "
         "each labelled with its id.",
         encode_image(sheet, max_side=1500),
         f"Region metadata:\n{meta}\n\nAudit ids: "
         f"{[r['id'] for r in chunk]}. JSON array only."],
        system=SYSTEM, max_tokens=6000, temperature=0.0)
    parsed = parse_json(text)
    for v in parsed:
        v["verdict"], v["route_why"] = route(v)
    print(f"  batch {n}: {len(parsed)}/{len(chunk)} ids, {dt}s")
    return parsed, usage, dt


def audit(image, regions, model, batch=12, workers=6):
    """Batches are independent, so run them concurrently — this is the single
    biggest latency win in the pipeline (serial 170s -> ~35s on 53 regions)."""
    chunks = [regions[i:i + batch] for i in range(0, len(regions), batch)]
    ctx_url = encode_image(image, max_side=900)  # encode the mockup once
    jobs = [(image, c, model, ctx_url, i + 1) for i, c in enumerate(chunks)]

    verdicts = {}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for parsed, usage, _ in ex.map(_audit_batch, jobs):
            for v in parsed:
                verdicts[int(v["id"])] = v
            usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_total["completion_tokens"] += usage.get("completion_tokens", 0)

    return verdicts, usage_total, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--models", nargs="+", default=["qwen3-vl-plus"])
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    man = json.loads(args.manifest.read_text())

    # Audit every region SAM 3 found, assets and css alike — the point is to
    # re-decide the split, so feeding only the current assets would beg the question.
    regions = [{"id": a["id"], "box_xyxy": a["box_xyxy"],
                "size_px": a["size_px"], "labels": a["labels"],
                "heuristic": a["texture"]} for a in man["assets"]]
    regions += [{"id": c["id"],
                 "box_xyxy": c["box_xyxy"],
                 "size_px": [c["box_xyxy"][2] - c["box_xyxy"][0],
                             c["box_xyxy"][3] - c["box_xyxy"][1]],
                 "labels": c["labels"], "heuristic": c["texture"]}
                for c in man["css_regions"]]
    regions.sort(key=lambda r: r["id"])
    print(f"{len(regions)} regions to audit")

    out = args.out or args.manifest.parent
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for model in args.models:
        print(f"\n=== {model}")
        try:
            verdicts, usage, latency = audit(image, regions, model, args.batch)
        except Exception as e:
            print(f"  FAILED: {e}")
            report[model] = {"error": str(e)}
            continue

        missing = [r["id"] for r in regions if r["id"] not in verdicts]
        split = Counter(v["verdict"] for v in verdicts.values())
        # Where the VLM overrules the heuristic classifier
        flips = [(r["id"], r["heuristic"], verdicts[r["id"]]["verdict"],
                  verdicts[r["id"]].get("reason", ""))
                 for r in regions if r["id"] in verdicts
                 and (verdicts[r["id"]]["verdict"] == "asset") !=
                 (r["heuristic"] in ("photo", "3d_render", "logotype"))]

        print(f"  {latency}s  in={usage['prompt_tokens']} out={usage['completion_tokens']}")
        print(f"  asset={split['asset']} library={split['library']} "
              f"css={split['css']} missing={len(missing)}")

        lib = [(k, v) for k, v in sorted(verdicts.items())
               if v["verdict"] == "library"]
        if lib:
            print(f"  library swaps ({len(lib)}) — coding agent imports these:")
            for k, v in lib[:12]:
                print(f"    #{k:<3} {str(v.get('library_suggestion')):<22} {v.get('reason','')[:46]}")

        print(f"  disagrees with heuristic on {len(flips)} regions:")
        for fid, h, v, why in flips[:14]:
            print(f"    #{fid:<3} {h:<10} -> {v:<8} {why[:52]}")

        report[model] = {"verdicts": verdicts, "usage": usage,
                         "latency_s": latency, "missing": missing,
                         "split": dict(split), "flips": flips}
        (out / f"audit_{model.replace('/', '_')}.json").write_text(
            json.dumps(report[model], indent=2, ensure_ascii=False))

    (out / "audit_comparison.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "verdicts"}
                    for k, v in report.items()}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
