"""Stage A: let a VLM read a mockup and decide which SAM 3 prompts to run.

This is what makes the pipeline domain-generic. Instead of a hardcoded prompt
list tuned to one app, the VLM names the concrete things actually depicted.

The system prompt carries our measured priors about SAM 3's vocabulary — which
word classes work and which return zero — because a VLM left to itself proposes
exactly the abstract style words that fail.

    python plan_prompts.py input/ktv_ui.jpeg --models qwen3-vl-flash qwen3-vl-plus qwen3.8-max
"""

import argparse
import json
from pathlib import Path

from .vlm import chat, encode_image, parse_json

# Measured on this repo — see INDEX.md. Handing the model real negative results
# is far more effective than telling it to "be concrete".
WORKS = ["icon", "3d icon", "photo", "thumbnail", "logo", "avatar",
         "neon sign", "button", "microphone", "disco ball"]
FAILS = ["illustration", "sticker", "portrait", "cartoon character",
         "glossy 3d object", "card", "panel", "text", "chinese text",
         "navigation bar", "section", "container"]

SYSTEM = f"""You plan text prompts for SAM 3, an open-vocabulary segmentation model, \
run over a UI mockup.

PURPOSE: find the raster art assets a browser CANNOT reproduce with CSS/HTML \
(photos, 3D-rendered icons, stylized wordmarks, glow art), so an image model can \
regenerate them as production assets. Flat fills, gradients, borders, shadows, \
plain glyph icons and all live text ARE reproducible in CSS and are NOT assets.

HARD CONSTRAINTS, measured empirically on SAM 3 — obey them exactly:

1. SAM 3 responds ONLY to concrete physical nouns naming a depicted thing.
2. These returned ZERO detections at every threshold: {', '.join(FAILS)}.
   Abstract style words and layout/container words DO NOT WORK. Never emit them.
3. These worked well: {', '.join(WORKS)}.
4. "3d icon" is the key discriminator. It fires on shaded/rendered icons (which \
need a raster asset) and stays silent on flat vector glyphs (which CSS draws). \
Always include it.
5. Prompts are single nouns or short noun phrases, lowercase, English, 1-3 words.

OUTPUT: JSON only, no prose.
{{
  "domain": "<what this product/screen is, one line>",
  "visual_style": "<flat | material | neon | 3d-illustrated | photographic | mixed>",
  "prompts": [
    {{"text": "<noun>", "role": "core|discriminator|content",
      "expect": "<what it should hit in THIS mockup>", "confidence": 0.0-1.0}}
  ],
  "expected_css_only": ["<things here that CSS can draw, for sanity checking>"],
  "notes": "<one line>"
}}

RULES:
- 6 to 10 prompts, ordered most reliable first.
- Always include the core four: "icon", "3d icon", "photo", "logo".
- Add 2-6 content nouns for what is ACTUALLY depicted in this specific mockup \
(e.g. a food app might need "hamburger", "bowl of noodles"; a finance dashboard \
might need "line chart", "credit card"). These are what make the plan domain-aware.
- If the UI is entirely flat/textual with no art, say so in notes and return \
only the core prompts."""


def plan(image_path, model, max_side=1400):
    url = encode_image(image_path, max_side=max_side)
    text, usage, dt = chat(
        model,
        [url, "Plan the SAM 3 prompts for this mockup. JSON only."],
        system=SYSTEM, max_tokens=2000, temperature=0.1)
    return parse_json(text), usage, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--models", nargs="+", default=["qwen3-vl-flash"])
    ap.add_argument("--out", type=Path, default=Path("plans"))
    ap.add_argument("--max-side", type=int, default=1400)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    for model in args.models:
        print(f"\n=== {model}")
        try:
            p, usage, dt = plan(args.image, model, args.max_side)
        except Exception as e:
            print(f"  FAILED: {e}")
            results[model] = {"error": str(e)}
            continue

        tok_in = usage.get("prompt_tokens", 0)
        tok_out = usage.get("completion_tokens", 0)
        print(f"  {dt}s  in={tok_in} out={tok_out}")
        print(f"  domain: {p.get('domain')}")
        print(f"  style : {p.get('visual_style')}")
        for q in p.get("prompts", []):
            print(f"    {q.get('text','?'):<18} {q.get('role',''):<14}"
                  f" {q.get('confidence','')}  {q.get('expect','')[:52]}")
        bad = [q["text"] for q in p.get("prompts", []) if q.get("text") in FAILS]
        if bad:
            print(f"  !! emitted known-dead prompts: {bad}")

        results[model] = {"plan": p, "usage": usage, "latency_s": dt,
                          "violations": bad}
        (args.out / f"plan_{model.replace('/', '_')}.json").write_text(
            json.dumps(p, indent=2, ensure_ascii=False))

    (args.out / "comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}/")


