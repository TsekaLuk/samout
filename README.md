# samout

*SAM-based asset extraction for UI mockups.*

Turn a UI mockup into a **design handoff spec**: a UI tree whose leaves each carry the
cheapest route a coding agent can use to rebuild them — write CSS, import a library
icon, extract the exact asset, or generate one with an image model.

```
mockup.png
  ├─ SAM 3            what art is here          (open-vocabulary segmentation)
  ├─ OmniParser       where the UI elements are (tap-target detection)
  ├─ ViTMatte         cut it out cleanly        (alpha matting from SAM's trimap)
  ├─ Qwen3-VL         countable facts per region
  ├─ taxonomy.py      Material/HIG design-system class -> delivery route
  └─ qwen-image-3.0   regenerate what must be raster
```

Two detectors, because neither spans a screen. SAM 3 answers "what concept is this"
and OmniParser answers "where can you tap"; on a design board SAM 3 alone missed 76%
of the elements — every button, tab glyph, chip and label — and produced zero
CSS-reproducible regions as a result.

Output: `uitree.txt` (paste into a coding agent), `handoff.json` (per-region class,
delivery, cost, a11y, flags, regeneration prompt), and RGBA cutouts.

**Read [`INSIGHTS.md`](INSIGHTS.md) first.** It is the honest record of 18 findings and
7 dead ends, most of them measured. The dead ends are the useful half.

---

## Install

```bash
uv venv --python 3.12 .venv
uv pip install torch torchvision transformers pillow numpy requests
```

Weights (about 3.5 GB, all ungated):

```bash
# SAM 3 — facebook/sam3 is gated; jetjodh/sam3 is an identical public mirror
curl -L -C - --create-dirs -o models/sam3/model.safetensors \
  https://huggingface.co/jetjodh/sam3/resolve/main/model.safetensors
for f in config.json processor_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.json merges.txt; do
  curl -L --create-dirs -o models/sam3/$f https://huggingface.co/jetjodh/sam3/resolve/main/$f
done

# ViTMatte — alpha matting
for f in config.json preprocessor_config.json model.safetensors; do
  curl -L --create-dirs -o models/vitmatte-small/$f \
    https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/$f
done
```

Optional: `models/sam2/` (`facebook/sam2.1-hiera-small`) for the experimental region
splitter, `models/omniparser/` (`microsoft/OmniParser-v2.0`) for UI element detection.

`huggingface_hub` hit SSL EOF mid-download on both endpoints during development;
`curl -L -C -` in a retry loop was reliable where the library was not.

## Run

```bash
export DASHSCOPE_API_KEY=sk-...          # Aliyun Bailian / DashScope
python run.py path/to/mockup.png --out assets
python run.py path/to/mockup.png --out assets --generate      # also regenerate assets
python run.py path/to/mockup.png --out assets --from assemble # reuse cached stages
```

No mockup ships with this repo — supply your own. The benchmark numbers below were
measured on a 941×1672 mobile app screen that is not included.

~40 s per screen for analysis (Apple M-series, MPS), plus ~9 min if generating 22
assets. Every stage caches through `store`, so `--from assemble` re-runs classification
with no API call.

## Architecture

```
run.py              orchestration only, no logic

samout/
  model.py          records shared by every stage; imports nothing from the package
  config.py         every tunable, each annotated calibrated-or-guessed
  vlm.py            DashScope client

  # pure — no IO, no model calls. The logic lives here and is testable.
  taxonomy.py       design-system class + delivery rules, each citing a published source
  uitree.py         containment forest, layout inference, WAI-ARIA roles
  components.py     visual grouping of repeated regions, probe consensus
  handoff.py        joins every stage's output into the spec

  # effectful
  detect.py         SAM 3 segmentation
  uidetect.py       OmniParser UI element detection, merged into the region set
  matting.py        ViTMatte alpha; matte.py is the fallback + colour decontamination
  crops.py          the only module that knows where a region's pixels come from
  observe.py        VLM observation; takes [(id, Image)] and knows nothing else
  generate.py       image-model regeneration, concurrent
  split.py          experimental: un-bundle "container + glyph" regions
  store.py          the only module that opens a file

eval/score.py       severity-weighted scoring against eval/labels/
tools/              probe_prompts.py (SAM 3 vocabulary), bench_models.py
archive/            superseded approaches, kept for provenance — see its README
```

No stage imports another stage. `handoff` imports the pure modules to join their
results; `run.py` orchestrates. The dependency graph is a DAG rooted at `model.py`.

## Two ideas carry the design

**The VLM is never asked for a verdict.** It reports only what it can count or point at
— how many hues, is there a cast shadow, is this a trademark. Every judgment lives in
`taxonomy.py`, in code you can read and diff. Two earlier designs asked for the verdict
directly and the model rationalised the same six icons into the wrong bucket twice, in
opposite directions, each time with a convincing reason.

**The categories are borrowed, not invented.** Material Design already specifies the
system-icon vs product-icon boundary this project needed, with observable criteria *and*
the delivery decision attached. Adopting it brought a rule we had outright wrong: brand
marks must never be regenerated — a synthesised trademark is a fidelity failure and a
legal one.

## Accuracy

53 hand-labelled regions on one screen. Class = the design-system class; delivery = the
production route, which is what a coding agent actually consumes.

| | class | delivery |
|---|---|---|
| bounding-box crops | 64.2% | 66.0% |
| masked cutouts | 75.5% | 81.1% |
| + calibrated icon rule | 77.4% | 81.1% |
| + ViTMatte cutouts | 81.1% | 84.9% |
| + deterministic-first rule order | 83.0% | 86.8% |
| **+ OmniParser element detection** | **86.8–90.6%** | **88.7–92.5%** |

The last row is measured on labels that cover only the SAM 3 regions, so the elements
OmniParser adds earn no credit — better recall improved *precision*, because a more
complete sibling set makes the tree's leaf-vs-container calls more accurate.

On a second screen with no ground truth (a design-system board, light background,
mostly chrome) the same change took CSS-reproducible regions from **0 to 89 of 154**.
That screen is where the single-screen caveat stopped being hypothetical: thresholds
calibrated on the first screen held, but recall did not.

Caveats, stated plainly: the labelled set is **one screen**, several thresholds are
calibrated on it, and the second screen is judged by eye. See [`TODO.md`](TODO.md).

Ground truth is matched to regions by box IoU, not by region id. Keying it on ids —
which are assigned by sorting detections by area — made the eval set score 22.6%
against an 83.0% baseline the first time a second detector was added, while every
labelled region was still present and correctly classified. An eval set anchored to
something the pipeline renumbers fails silently, and does so precisely on the changes
it exists to evaluate.

Measurement itself was the largest trap. Before it was quantified, three passes over
byte-identical inputs scored 67.9 / 81.1 / 71.7 — `temperature=0` does not make a hosted
MoE model deterministic, and that 13.2-point spread swallowed nearly every improvement
claimed up to that point. Read [`INSIGHTS.md`](INSIGHTS.md) aha #14 before trusting any
single-run comparison, here or elsewhere.

## Documents

| | |
|---|---|
| [`INSIGHTS.md`](INSIGHTS.md) | what we learned — 18 findings, 7 dead ends, mostly measured |
| [`BENCHMARK.md`](BENCHMARK.md) | SAM 3 prompt vocabulary, model selection, runtime |
| [`OPTIMIZATION.md`](OPTIMIZATION.md) | error attribution framework and priorities |
| [`TODO.md`](TODO.md) | current state, what blocks what |

## License

Apache-2.0. Model weights are under their own licenses; SAM 3's official repository is
gated and this project uses a public mirror.
