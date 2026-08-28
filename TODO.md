# State, WIP, and what to do next

Written 2026-08-28. Ordered by what unblocks what, not by appeal.

Baseline: **70.6% ± 2.2 class / 73.2% ± 2.1 delivery** (n=5, one screen, 53 labelled
regions). The ± is the point — see B1.

---

## Done and verified

| | evidence |
|---|---|
| SAM 3 detection, cross-prompt NMS | deterministic: 53/53 identical boxes across runs |
| ViTMatte cutouts (SAM mask → trimap) | deterministic: 53/53 byte-identical; edges clean, no halos |
| Component grouping | calibrated on known pairs, 0.29-wide margin |
| Design-system taxonomy + delivery rules | each class cites a published source |
| UI tree: containment, layout inference, ARIA roles | pure geometry, no thresholds left |
| Package layering (DAG, no stage imports a stage) | imports verified |
| Stage resume via `store` | `--from assemble` re-runs classification with no VLM call |
| **Generation loop** | qwen-image-3.0-pro, 4 assets, reference-faithful, higher-res than source |
| **Generated-asset cutout** | SAM 2 candidates scored by leftover-background variance; 3/3 clean |

## Done but NOT verified

Everything measured as a single run before B1 was understood. The engineering may be
right; the evidence is not.

- masked cutouts vs box crops (+11.3pp claimed — the only delta that plausibly exceeds noise)
- ViTMatte vs binary mask (+3.7pp claimed — inside noise)
- calibrated icon rule (inside noise)
- the package refactor's "regression" (inside noise)

---

## B — Blocking

### B1. Cut the measurement noise
**Nothing below can be evaluated until this is done.** Three observation passes over
byte-identical inputs scored 67.9 / 81.1 / 71.7 — a 13.2pp spread, sd 2.2. The VLM is
the only non-deterministic stage; `temperature=0` does not make a hosted MoE model
deterministic.

Fix: sample each observation k times and take the majority. The machinery already
exists — `components.consensus` does exactly this across instances of one component;
it needs to apply to singletons too. Expect variance to fall ~√k, and accuracy to rise,
since majority voting also corrects one-off observation errors.

Then: re-measure every claim in "Done but NOT verified" as mean ± sd over n≥5.

*Effort: small. Confidence: high. Cost: k× the observation bill.*

---

## H — High value, unblocked by B1

### H2. Region granularity (bundling)
SAM 3 returns one box for "translucent disc + play glyph" and one for "card = photo +
caption". Four to ten errors trace here, and it is why two venue photographs never
reach the asset list. No matte and no rule can recover a wrong region.

`samout/split.py` exists, is **gated off**, and works in the core: 5 of 6 attempts
pulled the play glyph cleanly out of its disc, thresholds calibrated with wide margins
(true splits area_frac 0.18–0.25 / contrast 132–190; the one false split 0.54 / 26).

Three defects to fix before enabling:
1. **Non-deterministic extent.** SAM 2 returned 20×22 for three instances of one play
   button and 19×32 for two others, so grouping split them into two components with two
   classes. Split the *component* once and map it onto instances, as observation does.
2. **False split on self-similar icons.** The signal-bars glyph "split" into a smaller
   copy of itself. Needs a shape-difference test, not just a size test.
3. **Parent lands in `composite`** where `token` (a CSS disc) is the useful answer.

Note the config comment currently blames a measured regression for the gating — that
justification is void under B1 and should be corrected to "gated pending these three
defects".

*Effort: medium. Confidence: high that it is the right target.*

### H3. Layout skeleton from OmniParser
The tree has 24 flat roots because SAM 3 does not model containers — `panel`,
`navigation bar`, `section` all return zero at every threshold.

OmniParser v2 (MIT, downloaded, verified on this screen) supplies exactly the missing
layer. Measured complementarity:

| | |
|---|---|
| SAM 3 regions inside an OmniParser element | **50/53 (94%)** |
| SAM 3 regions with no OmniParser parent | 3, all large containers |
| OmniParser elements with no SAM 3 region | **7/37** — pure layout and text |

Merge: OmniParser elements become internal nodes, SAM 3 regions hang off them by
geometry. Changes the detection contract, so it touches everything downstream.

**Caveat found by testing:** OmniParser detects *tap targets*, which are coarser than
the art — a quick-entry cell is icon + title + subtitle. It does **not** solve H2.

*Effort: medium. Confidence: high.*

### H4. Shrink the VLM to what has no deterministic source
Agreed direction. Each field it answers is a source of non-determinism, and most have
better sources:

| field | replace with | evidence |
|---|---|---|
| `hue_count` | delete | measured: computing beat asking, *and* the feature does not separate the classes |
| `depth_cues` | `measured.lum_std_inner` | `bevel`+`specular` appear identically on both classes — zero signal |
| `has_baked_text` | OCR | deterministic, more accurate, returns the copy too |
| `content_type` | partly from H3 + OCR | a UI detector knows "button", OCR knows "text" |
| `is_brand_mark` | **keep** | needs world knowledge |
| `regen_prompt` | **keep** | generative |

Compounds with B1: fewer VLM fields is directly less variance.

*Effort: medium (OCR is a new dependency). Confidence: high.*

---

## M — Medium

### M5. Wire generation into `run.py` as a stage
Currently `generate.py` is exercised by hand. Needs: a `--generate` flag, batching with
concurrency, a manifest of what was produced, and skip-if-exists. Brand assets must stay
excluded — that rule is in the code and must not be bypassed by a batch path.

### M6. Backdrop channel
Detection and verdict are pure computation and were prototyped: 40.3% of pixels belong
to no region, linear-gradient fit residual RMS 25.1/255 → the backdrop is raster, not a
CSS gradient. A generated replacement already looks right.

Not implemented as a stage. Note it must **not** enter the region tree — a backdrop is
below everything in z-order, not a parent in containment. It needs its own channel.

Extraction of a *true* backdrop layer (foreground removed) needs inpainting (LaMa, SD
inpaint) — a new dependency, deferred until wanted.

### M7. Extend the eval set
53 fixed regions cannot score regions that H2/H3 create, so detection-stage work is
partly outside the metric's range. Also: two labels are genuinely arguable (a count
badge is circle + number; a battery glyph vs a plain bar) — write the convention into
the labels file rather than letting the metric absorb the ambiguity.

### M8. Adopt Design2Code-style metrics
We hand-rolled an eval protocol. Design2Code/WebSight already define screenshot→code
metrics — block match, text similarity, position error, colour distance. Reinventing
this, worse, is not a good use of time.

### M9. Cost and latency
SAM 3 is ~3.4s per prompt and the image is re-encoded per prompt. `Sam3Model.forward`
accepts precomputed `vision_embeds`; encode once, reuse across prompts. Untested.

---

## The two abstractions the rules are standing in for

Forty branches had accumulated across four functions. They are now records in
`ruleset.py` — listable, testable, explainable — but that is containment, not a
cure. Auditing what forced each one, they fall into two groups, and each group is
one missing concept:

### A. Z-order — the pipeline has no model of layers

| patch | what it really is |
|---|---|
| `text_role: live_data` excludes view counts | a data layer over an image layer |
| stripping overlay clauses from the prompt | same |
| splitting "translucent disc + glyph" | a container layer under an icon layer |
| popover occlusion (unfixed) | an overlay layer over the page |

Four patches, one cause: **a UI is composited, and this treats it as one flat
image.** The correct primitive is that a region's asset is what sits on *its* layer,
not everything visible inside its rectangle. Until that exists, every new kind of
stacking costs another rule.

### B. Region extent vs semantic extent

| patch | what it really is |
|---|---|
| `brand_asset` with >=2 children downgrades | the box is larger than the mark |
| a tap target enclosing one region is a hit area | the interaction box is larger than the icon |
| `split.py` | one box holding two things |
| `find_grids` | many boxes holding one thing |

**A detected box is assumed to be one semantic unit and frequently is not** — in
both directions. `split` and `find_grids` are one-off answers to the two halves.

### Why neither is being built yet

Three screens of evidence. Both were verified against the patches above rather than
argued for — the layer model explains 4 of 6 and does *not* explain the other 2,
which is how the second abstraction was found. Building either on three screens
would repeat the mistake this file already records twice: a model fitted to the data
in front of it. The next thing to do is more screens, not more architecture.

## Open, and not yet answerable

### Generation quality on thin references
Two of eight generated assets came back unusable — both small crops of detailed
*scenes* (a nebula, a cosmic banner). Icons and photographs of the same pixel count
came back clean, so the failure is about scene complexity against resolution, not
size alone.

No predictor survives measurement. Pixel count, colour count, entropy, information
density and post-hoc similarity between the result and its reference were each
tabulated across the eight; successes and failures interleave on every one. A size
gate is doubly dead — the median asset on that board is 1599 px, so any threshold
covering the two failures flags 74-90% of everything.

The gate is left off in `handoff.py` with the numbers recorded, rather than shipping
one that is useless or arbitrary. What would settle it is a screen with enough
labelled generations to calibrate against. Until then a bad asset can enter the
handoff looking correct, which is the honest state of this stage.

## L — Low / deferred

- **L10. Disco ball (#19)** — a hard segmentation failure, mask is a vertical sliver.
  One region, one screen. May disappear under H2/H3.
- **L11. Second and third screen.** Every number in this repo comes from one mockup,
  and several thresholds are calibrated on it. Nothing is trustworthy until it survives
  a screen it was not tuned on — especially anything labelled "calibrated". This is low
  in *order* only because B1 must land first; it is high in importance.
- **L12. `plan.py`** (VLM proposes SAM 3 prompts per screen) is written and benchmarked
  but not in the pipeline. Only matters once L11 shows the fixed prompt set failing on
  a different domain.

---

## Sequencing

```
B1  ──> re-measure everything claimed
     │
     ├─> H4  (fewer VLM fields = less variance; compounds with B1)
     ├─> H2  (region granularity — biggest single error source)
     └─> H3  (layout skeleton — biggest structural gap)
              │
              └─> M7 (eval must cover the regions H2/H3 create)
                   └─> L11 (second screen — the real test)
```

M5, M6, M9 are independent and can go any time.

**The one thing to avoid:** optimising accuracy before B1. With sd = 2.2pp, any change
smaller than ~4.4pp is unmeasurable, and this repo has already shipped one feature
disabled on the strength of a "regression" that was noise.
