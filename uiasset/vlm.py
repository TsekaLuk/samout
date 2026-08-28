"""Minimal Bailian / DashScope VLM client (OpenAI-compatible endpoint).

The API key is read from $DASHSCOPE_API_KEY and never written to disk.
"""

import base64
import io
import json
import os
import re
import time

import requests
from PIL import Image

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def encode_image(img, max_side=1400, fmt="JPEG", quality=88):
    """PIL image or path -> data URL. Downscaled: prompt planning does not need
    full resolution and the token bill scales with pixels."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def chat(model, parts, system=None, max_tokens=4000, temperature=0.1, retries=4):
    """parts: list of str (text) and/or data-URL str for images, in order.

    Returns (text, usage_dict, latency_seconds).
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    content = []
    for p in parts:
        if isinstance(p, str) and p.startswith("data:image"):
            content.append({"type": "image_url", "image_url": {"url": p}})
        else:
            content.append({"type": "text", "text": p})

    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": content})

    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}

    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(
                ENDPOINT, timeout=180,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload)
            dt = time.time() - t0
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(2 * (attempt + 1))
                continue
            d = r.json()
            return (d["choices"][0]["message"]["content"],
                    d.get("usage", {}), round(dt, 2))
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model} failed after {retries} tries — {last}")


def parse_json(text):
    """Pull the first JSON object/array out of a model reply."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in reply: {text[:400]}")
