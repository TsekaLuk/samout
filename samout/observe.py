"""Stage 3: VLM observation -> {region_id: Observation}.

Takes `[(id, PIL.Image)]` and returns a dict. It does not know what a Region is,
where the pixels came from, whether they are masked, or what will be done with the
answers. That is the whole point: the pixel source is `crops.py`'s problem, the
classification is `taxonomy.py`'s, and swapping either touches one file.

The prompt asks only for things that can be COUNTED or POINTED AT. It never asks
for a verdict. Two earlier designs did, and the model rationalised the same six
icons into the wrong bucket twice — in opposite directions, each time with a
convincing reason. INSIGHTS aha #3-4.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw

from .vlm import chat, encode_image, parse_json

OBSERVE_SYSTEM = """You are doing a design-system audit of regions cut from a UI mockup.

Report only what you can COUNT or POINT AT. Do not classify, recommend, or decide how
anything should be built — a downstream rule does that against published design-system
criteria. Your accuracy on the observations is the whole job.

Regions are shown cut out on a flat grey backdrop. The grey is not part of the region;
never report cues that belong to it.

For each region report:

`content_type` — exactly one:
  "photographic"  a camera image, or a rendered scene indistinguishable from one
  "illustration"  drawn or composed artwork that is not a pictogram: scenes, decorative
                  art, mascots, banner art
  "display_lettering"  WORDS whose letterforms are custom-drawn rather than typed: a
                  3D-extruded headline, a neon word, a stylised wordmark, a badge whose
                  text is part of the artwork. Choose this over "text" whenever the
                  letterforms could not be reproduced by setting a web font, and over
                  "illustration" whenever the region is mostly the words themselves.
  "pictogram"     a small symbol standing for a concept or action
  "text"          live copy set in an ordinary typeface, and nothing else
  "control"       a button, pill, chip or badge — a plain shape WITH text or a glyph on
                  it. Choose this over "text" whenever the copy sits on a filled shape;
                  the shape is CSS and the copy is a label, and calling the whole thing
                  text loses the shape.
  "plain"         no pictorial content at all — a bar, divider, empty container

`hue_count` — how many DISTINCT HUES appear, ignoring tints and shades of the same hue.
  A white glyph = 1. A pink-to-purple gradient = 2. A calendar icon with an orange body,
  white check and grey shadow = 2 (grey is neutral, not a hue). Count carefully: this
  number decides whether the icon can come from an icon library. Neutrals (white, black,
  grey) do not count as hues.

`depth_cues` — every one you can actually SEE ON THE OBJECT, using ONLY these terms:
  specular_highlight   a bright spot where a light source reflects off the surface
  cast_shadow          a shadow the object throws onto what is behind it
  bevel                a rounded or chamfered edge that catches light along one side
  inner_shadow         shading inside the shape implying thickness
  occlusion            one part of the object overlapping and darkening another
  material_texture     visible metal, glass, plastic, fabric, gold
  perspective          the object is drawn at an angle rather than flat-on
  glow                 light bleeding outward from the shape
  Empty list if the shape is flat or flat-gradient with none of the above. A flat icon
  in a translucent circle has NO cues — the circle is not the icon.

`is_brand_mark` — true only for an identity-locked mark: the product's own logo or
  wordmark, a partner logo, a certification badge. Decorative signage inside a
  photograph is not a brand mark; it is part of the photograph.

`has_baked_text` — is readable copy rendered INTO the pixels, such that translating the
  product would require new art? Text beside the region does not count.

`text_role` — if there IS text in the region, what kind, because the two get opposite
  treatment and conflating them bakes live data into an asset:
  "artwork"   the letterforms ARE the art and must be reproduced exactly: a neon sign
              reading LET'S SING, a stylised badge, a wordmark, a headline drawn as
              part of the illustration. Remove it and the asset is wrong.
  "live_data" a value the product renders OVER the image and updates at runtime: view
              and comment counts, durations, prices, dates, ratings, unread badges,
              usernames. It belongs to the code, not the asset. It is usually in a
              corner, in a plain UI typeface, over a scrim.
  "none"      no readable text.
  When both appear, answer "live_data" — the overlay is the part that must not be
  baked in.

`needs_transparency` — must this sit on a varying background, requiring an alpha channel?

`theme_dependent` — would this need a different version in light vs dark theme?

OUTPUT: JSON array, one entry per id shown, no prose.
[
  {"id": <int>,
   "subject": "<what it depicts, max 8 words>",
   "content_type": "photographic"|"illustration"|"display_lettering"|"pictogram"|"text"|"control"|"plain",
   "hue_count": <int>,
   "depth_cues": [<terms from the list above>],
   "is_brand_mark": <bool>,
   "has_baked_text": <bool>,
   "text_role": "artwork"|"live_data"|"none",
   "needs_transparency": <bool>,
   "theme_dependent": <bool>,
   "closest_library_icon": "<e.g. 'lucide:map-pin', or null if not a pictogram>",
   "confidence": 0.0-1.0,
   "regen_prompt": "<standalone prompt for an image model to recreate this region — \
subject, material, lighting, palette, background handling. Describe ONLY the artwork: \
never mention overlaid counts, durations, prices, badges or timestamps, because naming \
them makes the model paint them in. null for text/control/plain>"}
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


def contact_sheet(pairs, cell=190, cols=6, pad=10, max_upscale=2.0):
    """Numbered grid of `[(id, Image)]` — one call covers many regions.

    Never upscales a region beyond `max_upscale`; small crops are centred on the
    cell instead of being blown up. A fixed cell size was stretching a 42x24 neon
    sign 5x into blur, and the model read text off the blur that was not there —
    it reported "NEXUS" and "50%" on two signs that say neither. Padding keeps a
    small region small and honest.
    """
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + pad) + pad,
                              rows * (cell + pad + 18) + pad), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for k, (rid, img) in enumerate(pairs):
        crop = img.copy()
        limit = min(cell, int(max(img.width, img.height) * max_upscale))
        crop.thumbnail((limit, limit), Image.LANCZOS)
        cx = pad + (k % cols) * (cell + pad)
        cy = pad + (k // cols) * (cell + pad + 18)
        d.rectangle([cx - 2, cy - 2, cx + cell + 2, cy + cell + 2], outline=(90, 90, 100))
        sheet.paste(crop, (cx + (cell - crop.width) // 2, cy + (cell - crop.height) // 2))
        d.text((cx + 2, cy + cell + 4), f"#{rid}", fill=(255, 220, 0))
    return sheet


def _batch(job):
    """Never raises. A dead batch must not discard the ones that succeeded — the
    endpoint drops connections often enough that all-or-nothing runs rarely finish."""
    chunk, model, ctx, tag, sheet_side = job
    meta = "\n".join(f"#{rid}: {img.width}x{img.height}px" for rid, img in chunk)
    try:
        text, usage, dt = chat(
            model,
            [ctx, "Full mockup for context. Below, the regions to observe, cut out on "
                  "grey and labelled by id.",
             encode_image(contact_sheet(chunk), max_side=sheet_side),
             f"Regions:\n{meta}\n\nObserve ids {[r for r, _ in chunk]}. JSON array only."],
            system=OBSERVE_SYSTEM, max_tokens=8000, temperature=0.0)
        parsed = parse_json(text)
        print(f"  batch {tag}: {len(parsed)}/{len(chunk)}, {dt}s")
        return parsed, usage
    except Exception as e:
        print(f"  batch {tag}: FAILED ({type(e).__name__}) — will re-sweep")
        return None, {}


def observe(context_image, pairs, cfg):
    """`pairs` is [(region_id, PIL.Image)]. Returns {region_id: observation dict}.

    Raises if the API key is missing. `_batch` deliberately swallows per-batch
    failures so one dead request cannot discard a whole run — but with no key EVERY
    batch fails, and the swallow turned that into an empty observation set and a
    spec reading `generate 0, effort 0`. A user reads that as "this mockup has no
    assets", which is worse than a crash. Check the precondition before the retry
    machinery gets a chance to hide it.

    Re-sweeps whatever is short with a halved batch each pass: a batch that failed on
    transport succeeds alone, and one that failed on output length succeeds when split.

    Samples each region `cfg.samples` times and reconciles them, because this is the
    only non-deterministic stage in the pipeline and its spread swamped every result
    measured before it was quantified: three passes over byte-identical inputs scored
    67.9 / 81.1 / 71.7, sd 2.2. `temperature=0` does not make a hosted MoE model
    deterministic.

    Reconciliation reuses `components.consensus` unchanged — majority vote on the
    categorical fields, intersection on the depth cues. It was written for several
    instances of one component, and repeated looks at one region want the same
    treatment: a cue reported once in three passes is noise, not evidence.
    """
    from collections import defaultdict

    from .components import consensus

    if pairs and not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set — the observation stage cannot run.\n"
            "  export DASHSCOPE_API_KEY=sk-...   (Aliyun Bailian / DashScope)\n"
            "Without it every region would be returned unobserved and the handoff "
            "spec would look empty rather than failing.")

    ctx = encode_image(context_image, max_side=cfg.context_max_side)
    raw = defaultdict(list)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    want = max(1, getattr(cfg, "samples", 1))

    # Batch by region size. A batch spanning a 42px badge and an 860px banner has to
    # pick one cell size and ruins one of them; grouping similar sizes lets each
    # batch render at a scale that suits it.
    pending = sorted(pairs, key=lambda p: -(p[1].width * p[1].height))
    for sweep in range(cfg.sweeps):
        if not pending:
            break
        size = max(1, cfg.batch // (2 ** sweep))
        chunks = [pending[i:i + size] for i in range(0, len(pending), size)]
        # First pass submits every chunk `want` times; later sweeps only top up
        # whatever came back short, so a transport failure costs one sample, not all.
        reps = want if sweep == 0 else 1
        if sweep:
            print(f"  sweep {sweep + 1}: {len(pending)} short, batch={size}")
        jobs = [(c, cfg.model, ctx, f"{sweep + 1}.{i + 1}.{s + 1}", cfg.sheet_max_side)
                for i, c in enumerate(chunks) for s in range(reps)]
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(jobs))) as ex:
            for parsed, u in ex.map(_batch, jobs):
                for v in (parsed or []):
                    try:
                        raw[int(v["id"])].append(v)
                    except (KeyError, TypeError, ValueError):
                        continue
                usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                usage["completion_tokens"] += u.get("completion_tokens", 0)
        pending = [(rid, img) for rid, img in pending if len(raw[rid]) < want]

    obs, disagreed = {}, 0
    for rid, samples in raw.items():
        merged = consensus(samples)
        if merged is None:
            continue
        kinds = {s.get("content_type") for s in samples}
        if len(kinds) > 1:
            disagreed += 1
            merged["sample_disagreement"] = sorted(k for k in kinds if k)
        obs[rid] = merged

    missing = [rid for rid, _ in pairs if rid not in obs]
    counts = [len(raw[rid]) for rid in obs]
    print(f"  {len(obs)} regions from {sum(counts)} samples "
          f"(min {min(counts) if counts else 0}/{want}); "
          f"{disagreed} disagreed on content_type between samples")
    if missing:
        print(f"  !! unobserved after {cfg.sweeps} sweeps: {missing}")
    return obs, usage, round(time.time() - t0, 1)


def recall_audit(image, boxes, model, max_side=1400):
    """What did the detector have no vocabulary for? Returns proposed SAM 3 prompts."""
    marked = image.copy()
    d = ImageDraw.Draw(marked)
    for b in boxes:
        d.rectangle(tuple(int(v) for v in b), outline=(0, 255, 60), width=3)
    text, _, dt = chat(
        model,
        [encode_image(marked, max_side=max_side),
         f"{len(boxes)} regions are already outlined. What pictorial content was "
         "missed? JSON array only; empty array if nothing."],
        system=RECALL_SYSTEM, max_tokens=2000, temperature=0.0)
    return parse_json(text), dt
