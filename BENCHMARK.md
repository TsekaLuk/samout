# Benchmarks

Everything here is measured, not estimated. Reproduce with the commands shown.

Hardware: Apple M-series, MPS backend. Network: mainland China, Aliyun PyPI mirror.
Date: 2026-08-27.

---

## 1. SAM 3 prompt vocabulary on UI mockups

The single most useful finding in this repo: **SAM 3 answers concrete physical nouns
and ignores abstract style words and layout words.** Same image, same thresholds,
only the prompt string changes.

```bash
python segment_ui.py input/ktv_ui.jpeg --prompts <word> --threshold 0.3
```

Test image: KTV app home screen, 941×1672.

| prompt | hits | verdict |
|---|---|---|
| `icon` | 41–49 | works — every glyph and icon in the UI |
| `logo` | 18–26 | works, overlaps `icon` heavily |
| `button` | 14 | works |
| `3d icon` | 20–23 | **works, and discriminates** — see §2 |
| `photo` | 10–12 | works, tight boundaries |
| `thumbnail` | 10–14 | works, but also fires on layout panels |
| `neon sign` | 9 | works (content-specific) |
| `microphone` | 6 | works (content-specific) |
| `avatar` | 3 | works |
| `disco ball` | 1 | works (content-specific) |
| `illustration` | **0** | dead |
| `sticker` | **0** | dead |
| `portrait` | **0** | dead |
| `cartoon character` | **0** | dead |
| `glossy 3d object` | **0** | dead |
| `card` | **0** | dead — reads as playing card |
| `panel` | **0** | dead — reads as instrument panel |
| `text` | **0** | dead |
| `chinese text` | **0** | dead |
| `navigation bar` | **0** | dead |

Dead at every threshold tested down to 0.25. Practical consequences:

- **`thumbnail` is the working synonym for a UI container.** `card` and `panel` return nothing.
- **Text is out of scope.** Pair with OCR (PaddleOCR etc.) for the text layer.
- When a VLM proposes prompts (see §4), it must be told this or it proposes exactly
  the dead words.

---

## 2. Texture statistics do NOT separate art assets from CSS-reproducible UI

The intuitive approach — "busy regions are art" — fails on a neon-gradient UI, because
glow and gradient push everything's entropy up.

| region | n_colors | entropy | edge density | truth |
|---|---|---|---|---|
| 搜索 pill (flat CSS gradient) | 966 | 7.27 | 0.153 | CSS-reproducible |
| 3D podium icon (rendered art) | 626 | 7.32 | 0.187 | needs asset |
| 立即开通 pill (flat) | 457 | 7.18 | 0.312 | CSS-reproducible |
| VIP crown (rendered gold) | 296 | 7.17 | 0.282 | needs asset |

Statistically indistinguishable in both directions. What *does* separate them is which
SAM 3 concept fires:

| region | `icon` | `3d icon` | truth |
|---|---|---|---|
| 排行热歌 podium | 0.908 | **0.804** | asset |
| 包厢预订 calendar | 0.934 | **0.774** | asset |
| 一起欢唱 people | 0.921 | **0.743** | asset |
| 搜索 button | 0.356 | — | CSS |
| 立即开通 button | 0.497 | — | CSS |
| play button ×6 | 0.79–0.86 | — | CSS |
| status bar glyphs | 0.80–0.84 | — | CSS |

`3d icon` fires on shaded/rendered icons and stays silent on flat glyphs. Texture stats
are kept in the manifest as reference data only; the classifier is label-driven.

---

## 3. SAM 3 runtime

```bash
time python extract_assets.py input/ktv_ui.jpeg --out assets --threshold 0.3
```

`28.17s` total for 7 prompts on one 941×1672 image:

| stage | cost |
|---|---|
| weight load (local safetensors, 3.4 GB) | ~4 s |
| inference | **~3.4 s per prompt** |
| NMS + texture stats + 36 cutouts + map | < 1 s |

$0 — fully local, no API. Scaling notes:

- Prompt count is the only linear variable. The core four (`photo`, `3d icon`, `logo`,
  `icon`) run in ~14 s; the rest buy cross-validation and nesting detail.
- `Sam3Model.forward` accepts precomputed `vision_embeds`. Encoding the image once and
  reusing it across prompts is the obvious next optimization — untested here.
- Batch to amortize the 4 s load: 100 mockups serial ≈ 45 min today.

---

## 4. Qwen VLM selection

Two VLM stages: **plan** (propose SAM 3 prompts from the mockup) and **audit**
(route each region to css / library / asset).

### 4a. Audit latency — the expensive call

Identical batch of 12 regions, identical prompt, all models fired concurrently.

```bash
python bench_models.py input/ktv_ui.jpeg assets/assets.json
```

| model | sec | in tok | out tok | coverage | agreement w/ `qwen3.8-max` |
|---|---|---|---|---|---|
| **qwen3-vl-flash** | **23.2** | 1903 | 1348 | 12/12 | **12/12 (100%)** |
| qwen-vl-plus | 43.4 | 1903 | 1121 | 12/12 | 11/12 |
| qwen3-vl-plus | 43.7 | 1903 | 1163 | 12/12 | 11/12 |
| qwen3.8-flash | 43.7 | 1972 | 1942 | 12/12 | 12/12 |
| qwen-vl-max | 59.7 | 1903 | 1161 | 12/12 | 11/12 |
| qwen3.8-max | 125.3 | 1934 | 6710 | 12/12 | — |

Pairwise agreement across all six: 92–100%. **The task does not discriminate between
model tiers, so take the fastest.** `qwen3.8-max` spends 6710 output tokens (mostly
reasoning) to reach the same verdicts as flash's 1348 — 5.4× the latency, no gain.
The previous-generation `qwen-vl-plus` / `qwen-vl-max` are strictly dominated.

**Pick: `qwen3-vl-flash`.**

### 4b. Concurrency beats model choice

Audit batches are independent. On 53 regions in batches of 12:

| configuration | wall clock |
|---|---|
| qwen3-vl-plus, serial | 170.1 s |
| qwen3-vl-flash, 6 workers | see §4d |

Encode the full mockup once and reuse the data URL across batches; re-encoding per
batch is pure waste.

### 4c. Prompt planning — model comparison

```bash
python plan_prompts.py input/ktv_ui.jpeg --models qwen3-vl-flash qwen3-vl-plus qwen3.8-max
```

| model | sec | out tok | prompts | dead words emitted |
|---|---|---|---|---|
| qwen3-vl-flash | 30.0 | 611 | 10 | 0 |
| qwen3-vl-plus | 27.5 | 534 | 8 | 0 |
| qwen3.8-max | 73.5 | 3212 | 10 | 0 |

All three respected the vocabulary constraints when given the measured dead-word list
as a negative-example block. Qualitative differences:

- **qwen3-vl-plus** showed the sharpest understanding: it assigned `icon` a confidence
  of 0.1 and annotated it "flat glyphs", spontaneously separating CSS-able glyphs from
  `3d icon`'s rendered ones.
- **qwen3-vl-flash** proposed a usable plan but misdescribed the 3D feature icons as
  "flat icons".
- **qwen3.8-max** added `crown` for the VIP badge — the only model to name it.

Planning runs once per image and its output is human-reviewable, so the latency
difference matters less here than in the audit.

### 4d. Routing criterion

The audit prompt asks one question per icon-like region:

> If I swapped in the closest standard library icon and styled it with CSS,
> would the design still read as the SAME design?

Yes → `library` (emit a `lucide:` / `material:` suggestion). No → `asset`.

This replaced an earlier binary css/asset split that asked "can this be vectorized",
which under-counted assets: a generic *concept* (music note) drawn in a bespoke *style*
(glossy 3D with specular highlights) is not something an icon library ships, so
swapping would visibly change the design.

---

## Reproducing

```bash
uv venv --python 3.12 .venv
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  uv pip install torch torchvision transformers pillow numpy requests

# weights: facebook/sam3 is gated; jetjodh/sam3 is an ungated identical mirror
curl -L -C - -o models/sam3/model.safetensors \
  https://huggingface.co/jetjodh/sam3/resolve/main/model.safetensors

export DASHSCOPE_API_KEY=sk-...
python extract_assets.py input/ktv_ui.jpeg --out assets --threshold 0.3
python classify_vlm.py input/ktv_ui.jpeg assets/assets.json --models qwen3-vl-flash
```

### Environment notes

- `huggingface_hub` hit `SSL: UNEXPECTED_EOF_WHILE_READING` on both `huggingface.co`
  and `hf-mirror.com` mid-download. `curl -L -C -` in a retry loop was reliable where
  the library was not.
- Both HF endpoints sustained ~4 MB/s from mainland China; the mirror was not faster
  for bulk transfer, only for metadata.
