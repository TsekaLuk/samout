"""Stage 6: hand the handoff spec to an image model and get assets back.

This closes the loop the whole pipeline exists to serve. Everything upstream —
detection, matting, classification — produces a `regen` block per asset; until now
nothing consumed it, so asset quality was entirely unmeasured.

Two rules are enforced here rather than left to the caller, because getting them
wrong is expensive in different ways:

  brand_asset is never sent.   A synthesised trademark is a fidelity failure and a
                               legal one; those regions are extracted, not generated.
  the cutout goes with the     Text alone reproduces the subject but not the art
  prompt.                      direction. The reference image is what makes the
                               output match *this* product rather than the category.

Qwen image models are served from the multimodal-generation endpoint, not the
`text2image/image-synthesis` path that `wan*` uses and not the OpenAI-compatible
`/images/generations` route (404).
"""

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = ("https://dashscope.aliyuncs.com/api/v1/services/aigc/"
            "multimodal-generation/generation")
# Measured A/B on three icons, same prompts and references:
#   qwen-image-3.0      24.8 / 33.0 / 77.9s   mean 45.2s
#   qwen-image-3.0-pro  71.0 / 110.9 / 117.2s mean 99.7s
# Output quality was close enough that the 2.2x latency is not worth paying by
# default for icon-scale assets. Pass `model=` for the pro tier when it matters.
DEFAULT_MODEL = "qwen-image-3.0"
PRO_MODEL = "qwen-image-3.0-pro"


def _data_url(path, max_side=1024):
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def generate(prompt, reference=None, model=DEFAULT_MODEL, size="1024*1024",
             timeout=300, retries=2):
    """-> (image_url, latency_s). `reference` is a path used for image-to-image."""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    content = []
    if reference:
        content.append({"image": _data_url(reference)})
    content.append({"text": prompt})
    payload = {"model": model,
               "input": {"messages": [{"role": "user", "content": content}]},
               "parameters": {"size": size, "n": 1}}

    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = requests.post(ENDPOINT, timeout=timeout, json=payload,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            d = r.json()
            if r.status_code != 200 or "output" not in d:
                last = f"HTTP {r.status_code}: {str(d)[:200]}"
                time.sleep(3 * (attempt + 1))
                continue
            parts = d["output"]["choices"][0]["message"]["content"]
            url = next(p["image"] for p in parts if "image" in p)
            return url, round(time.time() - t0, 1)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{model} failed — {last}")


def fetch(url, out_path):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(r.content)
    return out_path


def build_prompt(entry):
    """Assemble the regen prompt from the handoff entry.

    The palette and pixel size come from measurement, not from the observer's
    description, so the model is anchored to what is actually in the image.
    """
    regen = entry.get("regen") or {}
    bits = [regen.get("prompt") or entry.get("subject") or "UI asset"]
    pal = regen.get("palette") or []
    if pal:
        bits.append("Palette: " + ", ".join(pal[:4]) + ".")
    if regen.get("needs_transparency"):
        bits.append("Isolated on a plain white background, centred, no shadow "
                    "touching the frame edge, so it can be cut out.")
    if entry.get("class") == "product_icon":
        bits.append("A single app icon, no text, no border, no frame, no background "
                    "scene.")
    if entry.get("observed", {}).get("has_baked_text"):
        bits.append("Reproduce any lettering exactly as shown in the reference.")
    return " ".join(bits)


def cut_out_generated(path, out_path, device="cpu", min_frac=0.03, max_frac=0.92):
    """Lift the generated object off whatever backdrop the model baked in.

    Image models return a filled square: the three test icons came back on navy,
    brown and white grounds respectively, none of them usable as a sprite. Asking
    for a transparent background in the prompt is unreliable — it is a request, not
    a guarantee — so the background is removed after the fact by the same two-stage
    path used on the source screenshot: SAM 2 finds the object, ViTMatte solves its
    edge.

    A centre point prompt is sound here in a way it would not be on a screenshot,
    because the generation prompt asks for one centred subject.
    """
    import numpy as np
    import torch

    from . import matting
    from .crops import _compose
    from .split import _load

    img = Image.open(path).convert("RGB")
    W, H = img.size
    model, proc = _load(device=device)

    # Two prompt styles, because neither works alone. A single centre point misses
    # the outer parts of a multi-piece subject — on the generated podium it returned
    # only the central pillar. Adding negative corner points fixes that and breaks a
    # subject that reaches into a corner: the microphone runs diagonally, a corner
    # hint landed on it, and the mask inverted.
    #
    # So do not pick a heuristic — generate candidates from both and score them.
    mx, my = W * 0.5, H * 0.5
    span = 0.22
    grid = [[mx + dx * W * span, my + dy * H * span]
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    corners = [[W * f, H * g] for f, g in ((0.03, 0.03), (0.97, 0.03),
                                           (0.03, 0.97), (0.97, 0.97))]
    prompts = [([[mx, my]], [1]),
               (grid, [1] * len(grid)),
               (grid + corners, [1] * len(grid) + [0] * len(corners))]

    cands = []
    for pts, labels in prompts:
        inputs = proc(images=img, input_points=[[pts]], input_labels=[[labels]],
                      return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        for m in proc.post_process_masks(
                out.pred_masks, inputs["original_sizes"])[0][0].cpu().numpy():
            m = m.astype(bool)
            if min_frac <= m.mean() <= max_frac:
                cands.append(m)
    if not cands:
        return None, "no subject mask in range"

    # Objective test, not a heuristic: if the subject was cut correctly, what is
    # LEFT is the generated backdrop, which is flat by construction. So score each
    # candidate by the variance of the pixels outside it and take the flattest,
    # penalising masks that run along the frame edge — a subject on a backdrop does
    # not touch all four sides.
    arr = np.asarray(img, dtype=np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def score(m):
        outside = lum[~m]
        if outside.size < 100:
            return 1e9
        edges = sum(bool(e.any()) for e in
                    (m[0, :], m[-1, :], m[:, 0], m[:, -1]))
        return float(outside.std()) + 12.0 * edges

    mask = min(cands, key=score)

    rgb = np.asarray(img, dtype=np.uint8)
    alpha = mask.astype(np.float32)
    ys, xs = np.nonzero(mask)
    box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    hi, hbox = matting.matte(rgb, alpha, box, device=device)
    if hi is None:
        hi, hbox = alpha, (0, 0, W, H)
    rgba, tight = _compose(rgb, hi, hbox)
    if rgba is None:
        return None, "empty after matting"
    Image.fromarray(rgba, "RGBA").save(out_path)
    return out_path, {"coverage": round(float(mask.mean()), 3),
                      "size": [int(tight[2] - tight[0]), int(tight[3] - tight[1])]}


def generate_for(entry, cutouts_dir, out_dir, model=DEFAULT_MODEL, size="1024*1024",
                 cut_out=True, device="cpu", skip_existing=True):
    """-> dict describing what happened for one handoff entry."""
    if not entry.get("regenerable"):
        return {"id": entry["id"], "skipped": f"{entry['class']} is not regenerable"}
    out = Path(out_dir) / f"{entry['id']:02d}_generated.png"
    if skip_existing and out.exists():
        return {"id": entry["id"], "class": entry["class"], "skipped": "already exists",
                "path": str(out)}

    ref = Path(cutouts_dir) / f"{entry['id']:02d}_cutout.png"
    prompt = build_prompt(entry)
    url, dt = generate(prompt, reference=ref if ref.exists() else None,
                       model=model, size=size)
    fetch(url, out)
    res = {"id": entry["id"], "class": entry["class"], "prompt": prompt,
           "latency_s": dt, "path": str(out)}

    # Photographs keep their frame; everything else is meant to sit on the page.
    if cut_out and entry.get("class") != "photography":
        sprite = Path(out_dir) / f"{entry['id']:02d}_generated_cutout.png"
        got, info = cut_out_generated(out, sprite, device=device)
        res["sprite"] = str(sprite) if got else None
        res["sprite_info"] = info
    return res


def generate_all(spec, cutouts_dir, out_dir, model=DEFAULT_MODEL, workers=6,
                 device="cpu", skip_existing=True, verbose=True):
    """Generate every regenerable asset, concurrently.

    The image API takes 120-173s per call, which is server-side latency and cannot be
    shortened — but the calls are independent, so 22 assets need not take 45 minutes.
    Same reasoning that took observation from 170s to 44s.

    Generation runs in parallel; the SAM 2 + ViTMatte cutout that follows each one does
    NOT — those models are loaded once into a module global and are not thread-safe, so
    sprites are cut serially after the API work is done. That split keeps the expensive,
    parallelisable part parallel without racing the local models.

    `brand_asset` never reaches here: `generate_for` refuses anything not marked
    `regenerable`, and that flag comes from the taxonomy. A batch path must not become
    the hole that rule leaks through.
    """
    todo = [e for e in spec if e.get("regenerable")
            and e.get("delivery", "").startswith("asset")]
    if verbose:
        blocked = [e["id"] for e in spec if e.get("delivery") == "asset_exact"]
        print(f"{len(todo)} regenerable assets, {workers} workers"
              + (f"; {len(blocked)} brand assets excluded: {blocked}" if blocked else ""))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []

    def _api(entry):
        try:
            return generate_for(entry, cutouts_dir, out_dir, model=model,
                                cut_out=False, skip_existing=skip_existing)
        except Exception as e:
            return {"id": entry["id"], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as ex:
        for r in ex.map(_api, todo):
            results.append(r)
            if verbose:
                note = r.get("error") or r.get("skipped") or f"{r.get('latency_s')}s"
                print(f"  #{r['id']:<3} {note}")
    api_s = time.time() - t0

    # Cut sprites serially — one shared SAM 2 / ViTMatte instance, not thread-safe.
    by_id = {e["id"]: e for e in spec}
    cut = 0
    if cut_out_generated is not None:
        for r in results:
            if r.get("error") or by_id[r["id"]].get("class") == "photography":
                continue
            src = Path(r.get("path", ""))
            if not src.exists():
                continue
            sprite = Path(out_dir) / f"{r['id']:02d}_generated_cutout.png"
            if skip_existing and sprite.exists():
                r["sprite"] = str(sprite)
                continue
            got, info = cut_out_generated(src, sprite, device=device)
            r["sprite"] = str(sprite) if got else None
            r["sprite_info"] = info
            cut += 1 if got else 0

    total = time.time() - t0
    if verbose:
        ok = sum(1 for r in results if not r.get("error"))
        print(f"{ok}/{len(todo)} generated, {cut} sprites cut  "
              f"— api {api_s:.0f}s, total {total:.0f}s")
    return results
