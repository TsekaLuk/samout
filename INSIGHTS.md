# What we learned building this

Extracting production art assets from a UI mockup with SAM 3 + a VLM, so a coding
agent can rebuild the screen at high fidelity.

Everything below is from measurement on a real screen, including the parts where we
were wrong. The dead ends are the useful half — each one cost hours and each one has
a generalizable shape.

---

## The one-paragraph version

We tried to get a model to answer "does this need an art asset?" It kept saying yes
and no to the same six icons depending on how we phrased the question, and it argued
convincingly both times. The fix was not a better prompt or a bigger model. It was to
stop asking. Models are reliable at reporting what they can **count or point at** —
how many hues, is there a cast shadow, is this a trademark — and unreliable at
judgments that can be rationalized. So we moved every judgment into code, grounded the
categories in taxonomies the design industry already published, and left the model
reporting observables. Accuracy stopped depending on prompt wording.

---

## Aha moments

### 1. A segmentation model's vocabulary is itself a classifier

SAM 3 fires `3d icon` on shaded, rendered icons and stays completely silent on flat
vector glyphs — which is precisely the CSS-reproducible boundary:

| region | `icon` | `3d icon` | truth |
|---|---|---|---|
| 3D podium feature icon | 0.908 | **0.804** | needs asset |
| 3D calendar feature icon | 0.934 | **0.774** | needs asset |
| search pill | 0.356 | *silent* | CSS |
| play button ×6 | 0.79–0.86 | *silent* | CSS |
| status-bar glyphs | 0.80–0.84 | *silent* | CSS |

We went looking for a detector and found one already inside the tool. Worth asking of
any open-vocabulary model: **is there a prompt whose hit/miss pattern already encodes
my decision boundary?** It is free and it needs no training.

### 2. The obvious image-statistics approach is not merely weak — it carries no signal

Intuition says busy regions are art. On a neon-gradient UI, glow pushes everything's
entropy up:

| region | n_colors | entropy | edge density | truth |
|---|---|---|---|---|
| search pill (flat CSS gradient) | 966 | 7.27 | 0.153 | CSS |
| 3D podium icon (rendered art) | 626 | 7.32 | 0.187 | asset |
| CTA pill (flat) | 457 | 7.18 | 0.312 | CSS |
| VIP crown (rendered gold) | 296 | 7.17 | 0.282 | asset |

Not just overlapping — **inverted**. The flat things score higher. Any threshold on
these numbers is fitted noise. We shipped the stats into the manifest as reference
data and stopped deciding with them.

### 3. A model will rationalize a subjective verdict in whichever direction the prompt leans

Same six icons, two prompt designs, opposite reasoning, same wrong answer:

```
v1  "3D beveled icon with gradient; not a standard glyph"   → verdict: library
v2  "core glyph matches; the 3D is styling, not structure"  → verdict: library
```

In v1 the reason contradicted the verdict outright. We "fixed" it by forcing an
explicit `swap_would_change_design` boolean and adding a deterministic guard that
flipped contradictions. The model responded by flipping the boolean instead — it found
a new escape hatch through the same hole.

**The lesson is not "write a stricter prompt."** It is that any question with a
defensible answer on both sides will get one, and no amount of scaffolding around a
bad question fixes it. Ask something with a fact of the matter.

### 4. Ask for observables; keep the judgment in code

The rewrite asks only what can be counted or pointed at:

```
content_type   photographic | illustration | pictogram | text | plain
hue_count      distinct hues, neutrals excluded   ← countable
depth_cues     from a closed 8-term vocabulary    ← pointable
is_brand_mark  is this a trademark                ← factual
```

Classification then happens in `taxonomy.py`, in code you can read, diff and argue
with. A model cannot rationalize "there is a cast shadow" into "the cast shadow is
just styling." The subjective line still exists — it lives in
`SYSTEM_ICON_CRITERIA["max_hues"] = 2` — but it is now one visible, tunable constant
instead of a sentence buried in a prompt.

### 5. The design industry already published the taxonomy we were inventing

We spent hours inventing axes (`render_style`, "how 3D does it look"). Material Design
had already specified this exact boundary:

- **System icons** — monochrome, geometric, 24dp grid, uniform stroke → ship from an icon library
- **Product icons** — expressive, multi-colour, dimensional, brand-specific → ship as an asset

That is the whole question, already answered, with observable criteria, and — the part
that mattered most — **the industry has also already decided how each class is
delivered**. Which is what we actually needed.

Adopting it brought rules we would not have thought of, and one we had outright wrong:

- **Brand marks must never be regenerated.** We were sending the wordmark to an image
  model. A synthesized trademark is both a fidelity failure and a legal one. It is the
  one class that must be extracted exactly.
- **Text baked into an image is an i18n defect**, flagged at design review, because
  translating the product requires new art.
- **Decorative vs informative** determines `aria-hidden` vs `alt`.

Generalizable: before inventing a taxonomy, check whether the discipline you are
automating already has one. A category that survived a thousand products beats one
fitted to your test file — and it usually arrives with the downstream decision attached.

### 6. The UI tree is the backbone, not a post-processing step

We were producing a flat region list and patching containment on afterwards with
thresholds (`min_children >= 2`, `child_area_ratio >= 0.45`). Both were compensating
for a missing structure.

With a real tree — parent is the **smallest** region containing you — "layout vs art"
stops being a judgment: **an internal node is layout, a leaf is content.** No
threshold. Sibling geometry then compiles straight to CSS: single visual row → `row`,
one-per-row → `column`, uniform row widths → `grid`, with median gaps and alignment.

Target the **accessibility tree** (W3C WAI-ARIA roles: `banner`, `navigation`,
`tablist`, `list`, `img`, `heading`). It is the one standardized description of UI
structure *and* semantics, it is what screen readers consume, and it is what a coding
agent needs to emit anyway.

### 7. Component reuse is a free consistency mechanism

Seven visually identical play buttons were being classified independently, and got
different answers:

```
#35 #36 #37 #38  play button → product_icon   (cues reported: bevel, specular, glow)
#45 #46 #47      play button → system_icon    (no cues reported)
```

The model was not wrong twice. It was asked the same question seven times and answered
independently. But **a design system does not contain seven slightly different play
buttons — it contains one Play Button used seven times.** Grouping by visual similarity
first and classifying the *component* makes consistency structural, and cuts VLM cost
by the repeat factor.

The domain's own principles are often available as engineering constraints, not just
as vocabulary.

### 8. Consistency introduces its own failure mode — reconcile, don't just deduplicate

Grouping made things worse before better. Picking the largest instance as
representative selected the play button sitting on a card photograph; the model read
`bevel/specular/glow` off the *background* and one bad observation contaminated all
seven. `system_icon` went 10 → 1.

Fix: probe several instances and take the **intersection** of depth cues, not the
union. A bevel belonging to the object appears on every instance; a highlight bleeding
in from one instance's backdrop appears on exactly one.

It reports what it discards, which is how we know it is doing the intended thing:

```
c31 x7: dropped background cues ['bevel', 'glow']      ← the seven play buttons
c24 x2: dropped background cues ['bevel']
c28 x2: dropped background cues ['material_texture']
```

`spot_illustration` 16 → 7, `system_icon` 1 → 6. Exactly the contamination predicted,
removed by a set intersection.

**Make reconciliation steps announce what they threw away.** Silent consensus is
indistinguishable from a bug; this one prints its evidence and cost nothing extra.

### 9. Normalize before you fingerprint

Our first component grouper (difference hash over the raw crop) found **zero** repeats
among seven identical buttons — distances of 24–48 bits, indistinguishable from
unrelated icons. Cause: white glyphs on seven different card photographs. The
background dominated the hash.

Subtracting the mean and dividing by the standard deviation removes exactly that and
leaves the silhouette — the actual component invariant:

| | min | median | max |
|---|---|---|---|
| same component (21 pairs) | 0.834 | 0.921 | 0.970 |
| different component (66 pairs) | -0.300 | -0.004 | 0.546 |

A 0.29-wide gap. Any threshold in 0.55–0.75 gives full recall and zero false merges.

**Calibrate thresholds against known pairs.** It took one script and replaced a guess
with a measured margin — and the margin is what tells you the feature works at all.

### 10. We used a pixel-accurate segmenter and then threw the pixels away

The single largest error source was not the taxonomy, the prompt, or the model. It was
`image.crop(box)` — one line, unchanged since the first version, that fed the observer a
**rectangular crop including the background** instead of the mask cutout SAM 3 had
already produced.

A play button's box is "translucent circle + whatever card photograph is behind it". The
observer dutifully reported `bevel` and `glow` — which were really there, in the
backdrop — and seven flat glyphs were routed to an image model.

Compositing the SAM 3 mask onto neutral grey (not white or black, which imply lighting):

| metric | box crop | masked cutout | Δ |
|---|---|---|---|
| class accuracy | 64.2% | **75.5%** | **+11.3pp** |
| delivery accuracy | 66.0% | **81.1%** | **+15.1pp** |
| dominant confusion `system_icon → product_icon` | ×10 | **×3** | −70% |

Two lessons. First, **when you pay for a precise signal, check that it survives to the
consumer.** We ran a state-of-the-art segmentation model and then degraded its masks to
bounding boxes three stages downstream. Second, this is the same meta-lesson as every
dead end in this project — *we assumed a component worked and built on top of it.* Weeks
of work went into consensus, taxonomy and tree structure while the biggest single defect
sat in a line nobody re-read.

It was also invisible until the eval set existed. Squinting at contact sheets, this
looked like model error.

### 11. Check that your features are discriminative — count them, per class

The rule "any depth cue that a grid-drawn glyph could not have implies a product
icon" sounds airtight. Counting the labelled data killed it. Across 24 labelled
pictograms:

| reported cues | system_icon | product_icon |
|---|---|---|
| `{bevel, specular_highlight}` | **9** | **6** |
| `{}` (none) | 5 | 0 |
| contains `occlusion` / `inner_shadow` | 0 | 4 |
| `hue_count > 1` | 0 | 3 |

`bevel` + `specular_highlight` is reported **identically on both classes**. It carries
zero signal for this observer — it fires on any gradient with a lit edge, which includes
the translucent disc bundled behind a flat play glyph. Gating on it made nine flat icons
into assets.

What does separate: `occlusion`, `inner_shadow`, `cast_shadow`, `material_texture`,
`perspective` (product only), and more than one hue (product only). Holding Material's
monochrome rule *literally* — `max_hues = 1`, not the 2 we had allowed as a hedge — was
itself worth accuracy. **The hedge was the error.**

This is the same discipline as calibrating the similarity threshold (aha #9), applied to
a categorical feature: do not ask whether a feature *sounds* discriminative, tabulate it
against labels. Ours had a term with an entropy of zero sitting in the middle of the rule.

### 12. A refactor is only safe if something can catch it regressing

Reorganising into a package regressed accuracy from 75.5% to 67.9% — the same
`system_icon → product_icon` confusion, back from ×3 to ×9, because the observation
prompt changed along with the file layout. Squinting at contact sheets, this would have
shipped.

The eval set caught it in one command, and the stage-resumable design meant testing the
fix cost nothing: `--from assemble` re-ran classification against cached observations
with no VLM call. Recovering to 77.4% took three iterations of a rule, each free.

**Separate the expensive stages from the cheap ones early.** The value is not tidiness,
it is that the slow, paid stage stops being in the loop when you are iterating on the
part that is neither.

### 13. Segmentation is not matting — do not sharpen your way out of the wrong tool

The cutouts had staircase edges and a dark rim around every bright object. The obvious
read is "SAM 3's masks are coarse", and the obvious fix is a filter: take the soft mask
it emits before binarisation, rescale the alpha ramp to narrow it, subtract the
background bleed at the boundary. That is what the first attempt did, and it did help.

It was still a filter compensating for using the wrong tool. **SAM 3 answers *which
object*.** Its mask head predicts at a low internal resolution and is bilinearly
upsampled, so on a 60px icon the boundary smears across a third of the sprite — 54-76%
of pixels came out partially transparent. Sharpening that ramp is inventing detail that
was never predicted.

**Matting is a different task with its own models.** ViTMatte takes an image and a
trimap and outputs a real alpha. And the two compose without adaptation, because a
trimap is exactly what a segmentation mask gives you for free:

    erode(mask)          -> definitely foreground
    outside dilate(mask) -> definitely background
    the band between     -> unknown, solve here

| | soft-edge pixels | edges | dark rim |
|---|---|---|---|
| binary SAM mask | 0% | staircase | yes |
| soft mask + alpha sharpening | 54-76% → 6% | smooth but mushy | reduced |
| **ViTMatte on a SAM trimap** | **14-27%** | **crisp** | **gone** |

Two things worth carrying:

**Choose the composition that preserves authority.** BiRefNet and RMBG are stronger
standalone background removers, but they re-decide which object is salient inside the
crop and can disagree with the region the pipeline asked for. ViTMatte's need for a
trimap looks like a limitation and is actually the guarantee: it cannot change *what* is
being cut out, only where the edge falls. SAM 3 keeps the semantic decision.

**And the payoff was not where it was aimed.** Better cutouts raised *classification*
accuracy from 77.4% to 81.1%, because the observation stage's input is those same
cutouts. Deliverable quality and pipeline accuracy were being treated as two concerns;
one is upstream of the other.

### 14. Verify the *system* is reproducible, not just the metric

Every accuracy number in this project up to this point was a single run, and every
comparison between them was made as if the difference were signal. Then two runs of
identical code on identical inputs came back 71.7% and 67.9%.

Bisecting the pipeline for the source:

| stage | test | deterministic? |
|---|---|---|
| SAM 3 detection | 2 runs, compare every box and measurement | ✅ 53/53 identical |
| ViTMatte cutouts | 2 runs, compare file hashes | ✅ 53/53 byte-identical |
| VLM observation | 3 runs, same cutouts, `temperature=0` | ❌ **3 different hashes** |

Three observation passes over byte-identical inputs scored **67.9% / 81.1% / 71.7%** —
a **13.2-point spread**. `temperature=0` does not make a hosted MoE model deterministic;
server-side batching and expert routing move the output regardless.

That noise band is ±6.6pp, which swallows nearly every result reported here:

| claimed | delta | verdict |
|---|---|---|
| masked cutouts, 64.2% → 75.5% | +11.3 | marginally outside the band; still not a single-run claim |
| ViTMatte, 77.4% → 81.1% | +3.7 | **inside the noise** |
| package refactor, 75.5% → 67.9% | −7.6 | **inside the noise** |
| region splitting, 81.1% → 71.7% | −9.4 | **inside the noise** |

The engineering is not necessarily wrong — the *evidence for it* was. And the cost was
not just cosmetic: the region-splitting feature was disabled on the strength of a
"measured regression" that a single sample cannot establish.

Three corrections, in order of importance:

1. **Report mean ± spread over N runs.** A difference smaller than the band is not a
   result. This should have been the first thing established after building the eval
   set, not the last.
2. **Reduce the variance rather than tolerate it.** Sampling each observation k times
   and taking the majority is the standard fix, and it is already half-built here — the
   multi-probe consensus does exactly this across component instances. Extend it to
   singletons.
3. **Bisect for determinism before trusting any pipeline metric.** Each stage in
   isolation, with a cheap identity check. It took three commands and would have been
   worth running on day one.

The deeper pattern is the one this project keeps re-teaching: *a component was assumed
to work and everything was built on top of it.* Here the assumed component was the
measurement itself — and this happened after "validate your components before building
on them" was already written down in this very file, twice.

### 15. Ask the VLM only what has no deterministic source

"Ask for observables, not verdicts" (aha #3-4) was the right first move. It is not the
end of the rule, because a lot of observables have a better source than a language
model looking at a picture.

The VLM turned out to be **the only non-deterministic stage in the pipeline** —
detection is bit-identical across runs, matting is byte-identical, and the entire
5.7pp spread comes from the observation call. So every field handed to it costs
reproducibility, and should have to earn that.

Auditing the eight fields it was being asked for:

| field | deterministic source? | evidence |
|---|---|---|
| `hue_count` | **yes**, computable | measured: computing it beat asking, and the feature does not separate the classes at all |
| `depth_cues` | **yes**, measurable | `bevel`+`specular_highlight` appear identically on both classes — zero signal; `lum_std_inner` matched it |
| `has_baked_text` | **yes**, OCR | deterministic, more accurate, and returns the copy as a bonus |
| `content_type` | partly | a UI detector knows "button"/"icon", OCR knows "text"; only the art distinctions remain |
| `needs_transparency` | partly | mostly derivable from geometry and context |
| `is_brand_mark` | **no** | requires world knowledge — recognising *which* trademark |
| `regen_prompt` | **no** | generative by nature |

Most of what it was being asked had a cheaper, more accurate, deterministic answer. So
the rule sharpens:

> ask for observables, not verdicts
> → **ask only for the observables nothing else can produce**

Which reshapes the architecture around what each component is actually for:

```
UI detector   -> element inventory, type, structure     deterministic
concept seg   -> art regions                            deterministic
OCR           -> where the text is and what it says     deterministic
measurement   -> shading, colour, size, contrast        deterministic
─────────────────────────────────────────────────────────────────────
VLM           -> is this a trademark? what should the   non-deterministic,
                 regeneration prompt say?               and now minimal
```

Shrinking the VLM to its irreducible semantic core narrows the noise band *and* raises
accuracy, because every other field moves to the tool that is best at it.

One distinction worth keeping: a VLM used for *description* is not the same liability as
a VLM used for *judgement*. OmniParser ships a Florence-2 captioner, and adding it does
not reintroduce the problem — its output is a human-readable label that never feeds the
routing decision. Non-determinism only matters where it propagates into a verdict.

### 16. Check what a borrowed model was *trained to find*, not what it is called

"UI element detection" sounds like the finer-grained detector this pipeline needed for
its bundling problem — one box for "translucent disc + play glyph" that no downstream
stage can undo. OmniParser is purpose-built for UI screenshots, so the expectation was
that it would separate the button from the glyph inside it.

It does the opposite. OmniParser is trained on **interactable regions — tap targets** —
so its boxes are *coarser* than the art: a quick-entry cell is one element covering
icon + title + subtitle, and a tab is icon + label. For its own purpose that is the
correct answer. The bundling problem is untouched by it.

What it does deliver is the thing that was actually missing. Measured on the test screen:

| | |
|---|---|
| SAM 3 regions falling inside an OmniParser element | **50 / 53 (94%)** |
| SAM 3 regions with no OmniParser parent | 3, all large containers |
| OmniParser elements containing no SAM 3 region | **7 / 37** — pure layout and text |

Those 7 are the internal nodes SAM 3 never produces, and the 94% containment means the
art regions hang off the skeleton cleanly rather than crossing it. The two models are
complementary precisely *because* their training objectives differ: one was taught
"where can you tap", the other "what concept is this", and neither alone spans a screen.

The transferable check, before adopting any pretrained component: **read what it was
trained to output, and predict where it will disagree with what you want.** Its domain
matching yours is not sufficient — OmniParser and this pipeline are both "UI parsing"
and still target different granularities. The prediction here was wrong, and one run
of the model on one image was enough to find out.

### 17. Do not key an eval set on anything the pipeline can renumber

Adding a second detector scored **22.6% against an 83.0% baseline**. Nothing was
broken: all 53 labelled regions were still detected and still classified correctly.
The label file was keyed by region id, ids are assigned by sorting detections by
area, and inserting 22 new regions renumbered everything.

The failure mode is the dangerous kind — it does not raise, it returns a plausible
number. And it lands precisely on the changes an eval set exists to evaluate: any
work on the detection stage renumbers regions, so the measurement silently stops
meaning anything exactly when detection is what you are testing.

Fix: match labels to regions by **box IoU**, not by id. Boxes are stable; ids are an
implementation detail. One consequence worth stating — an earlier decision to
disable region splitting on the strength of a "measured regression" was made with
this bug live, and has to be re-measured.

**Ask what your ground truth is anchored to, and whether the thing under test can
move it.**

### 18. When a threshold will not separate, the discriminator is wrong

Merging OmniParser's tap targets with SAM 3's regions cost 7.5pp: 17 of its boxes
enclosed a single existing region, became that region's parent, and turned it from a
leaf into a `composite` container.

The obvious lever is an area-ratio cutoff — a hit area is only a little larger than
what it wraps, a real container is much larger. Measured, the ratios ran **3.06x to
17.45x with no gap anywhere**. No threshold exists.

What separates them is not size but *how many siblings they hold*, and that follows
from what the two detectors were trained to find:

> a container groups several elements; a hit area wraps exactly one.

Dropping boxes that enclose exactly one region recovered the loss and went past it —
**86.8-90.6% class against 83.0% for SAM 3 alone**, on an eval set that gives no
credit for the regions OmniParser adds. Better recall improved precision, because a
more complete sibling set makes the tree's leaf-vs-container calls more accurate.

Same shape as the `bevel`/`specular_highlight` finding (#11): a feature that cannot
separate the classes is not a threshold problem. Tabulate before tuning.

### 19. Design mockups and live screenshots fail differently

Two screens in, the pipeline looked general — a dark neon app screen and a light
design board, and the thresholds calibrated on the first held on the second. Both
were *design mockups*: clean edges, nothing occluded, nothing mid-load.

A screenshot of a live page broke three things none of them could have exposed:

| | mockup | live page |
|---|---|---|
| skeleton loaders | absent | present, and classed `photography` — an empty grey rectangle queued for regeneration |
| overlays | absent | a login popover sits across the nav, and nothing in the pipeline models z-order |
| compression artifacts | absent | present |

The placeholder case was the fixable one, and cleanly: a skeleton loader is
achromatic and flat *by definition*, so the separation is not a fitted threshold —
saturation 0.009-0.015 against 0.227-0.737 for real content, entropy 2.5 against
7.4. Both gaps are an order of magnitude wide.

The lesson is about test-set composition, not about placeholders. **Two inputs that
share a provenance are close to one input.** Every screen tested had been *designed*;
the first one that had been *rendered* moved the failure mode somewhere the existing
tests could not reach. Vary the axis that generates the data, not just the data.

### 20. A model's blind spots are information, not just limitations

We logged `panel`, `card`, `navigation bar`, `section`, `container` returning zero hits
as a vocabulary quirk. It was not a quirk. **SAM 3 does not model layout at all** — it
segments *things*, and a container is not a thing.

That reframes the architecture. The tree currently has content leaves and 24 flat roots
because no layout container was ever detected. The fix follows from the division of
competence:

| | good at | contributes |
|---|---|---|
| **SAM 3** | pixel-accurate instance masks for concepts | content leaves |
| **VLM** | layout semantics, reading order, roles | the section skeleton |

Have the VLM propose the skeleton, SAM 3 supply the leaves, and hang one on the other.
Neither tool alone produces a usable tree.

---

## Dead ends, and what each one actually taught

| dead end | hours | the transferable lesson |
|---|---|---|
| Entropy / colour-count classifier | ~1 | Validate that your features separate the classes *before* building on them. Ours were inverted. |
| Asking the VLM for the verdict (3 rewrites) | ~3 | Rewording a rationalizable question moves the failure, never removes it. |
| Deterministic guard over a bad question | ~1 | Scaffolding a bad question just relocates the escape hatch. Fix the question. |
| dHash on raw crops | ~1 | Fingerprints need normalization against the nuisance variable. Test on known pairs. |
| Trusting the detector's nesting output | ~1 | A detector reports what one prompt happened to find nested. Recompute structure geometrically. |
| Reaching for a bigger model | ~1 | Verify the task discriminates between tiers first. Ours did not. |
| `huggingface_hub` for a 3.4 GB download | ~1 | `curl -L -C -` in a retry loop is resumable where the library was not. |

The pattern across all seven: **we assumed a component worked and built on top of it.**
Every one would have been caught by a ten-minute check of the component in isolation —
do these features separate? do these thresholds have margin? does this model tier
matter?

---

## What the benchmarks actually said

### Model tier does not matter for this task; latency does

Identical batch of 12 regions, identical prompt, six models fired concurrently:

| model | sec | out tok | agreement with `qwen3.8-max` |
|---|---|---|---|
| **qwen3-vl-flash** | **23.2** | 1348 | **12/12 (100%)** |
| qwen-vl-plus | 43.4 | 1121 | 11/12 |
| qwen3-vl-plus | 43.7 | 1163 | 11/12 |
| qwen3.8-flash | 43.7 | 1942 | 12/12 |
| qwen-vl-max | 59.7 | 1161 | 11/12 |
| qwen3.8-max | 125.3 | 6710 | — |

Pairwise agreement across all six: **92–100%**. The flagship spent 6710 output tokens —
mostly reasoning — to reach flash's answers at 5.4× the latency. Previous-generation
`qwen-vl-*` were strictly dominated.

**Measure agreement before paying for a tier.** When the cheapest model agrees with the
most expensive one, the task is not tier-limited, and reaching for a bigger model to fix
a *judgment* problem is treating a design flaw as a capacity problem.

### Concurrency beat model choice by more than model choice did

53 regions: **170s serial → 44s** with 6 workers. Larger than the entire flash-vs-max
gap. Independent batches, encode the shared context image once.

### Where each model was genuinely better

Real qualitative differences, on prompt planning:

- **qwen3-vl-plus** — sharpest semantics. Unprompted, it gave `icon` a confidence of
  0.1 and annotated it "flat glyphs", separating CSS-able glyphs from `3d icon`'s
  rendered ones. It reached our core insight on its own.
- **qwen3.8-max** — best recall on the long tail. Only model to name `crown` for the
  VIP badge.
- **qwen3-vl-flash** — fastest, and its *observations* were as good as anyone's. It was
  weaker at *interpretation* (called the 3D feature icons "flat icons").

Which lines up with the final architecture: flash is asked only to observe, and
observation is where it is already at parity.

### SAM 3 prompt vocabulary

| works | dead (0 hits at every threshold) |
|---|---|
| `icon`, `3d icon`, `photo`, `thumbnail`, `logo`, `avatar`, `button`, `neon sign`, `microphone`, `disco ball` | `illustration`, `sticker`, `portrait`, `cartoon character`, `glossy 3d object`, `card`, `panel`, `text`, `chinese text`, `navigation bar`, `section`, `container` |

Concrete physical nouns only. Abstract style words and layout words return nothing.
`card` reads as playing card, `panel` as instrument panel. `thumbnail` is the working
synonym for a UI image slot — but it fires on layout panels too, so it is not on its
own evidence of an image.

A VLM asked to propose prompts will produce exactly the dead words unless you hand it
the measured dead list as negative examples. With the list, all three models emitted
zero dead prompts.

### Local inference cost

28.2s for 7 prompts on a 941×1672 image, on Apple MPS, $0. Weight load ~4s; **~3.4s per
prompt**. Prompt count is the only linear variable. `Sam3Model.forward` accepts
precomputed `vision_embeds`, so encoding once and reusing across prompts is the obvious
next win.

---

## The method, generalized

Five principles, in the order they matter:

1. **Ask models for observables, not verdicts.** Anything with a defensible answer on
   both sides will get one. Move every judgment into code you can read, diff and tune.
   The subjective line does not disappear — it becomes a visible constant.

2. **Borrow the domain's taxonomy before inventing one.** Established categories arrive
   with observable criteria *and* the downstream decision already attached. Ours brought
   the never-regenerate-a-trademark rule we had gotten wrong.

3. **Get the structure first.** With a real tree, several judgments stop being judgments.
   Every threshold we deleted was compensating for structure we had not built.

4. **Exploit the domain's own invariants as engineering constraints.** "A design system
   has one component used N times" is simultaneously a correctness guarantee and an
   N-fold cost saving.

5. **Calibrate every threshold against known pairs, and measure the margin.** One script
   turned a guessed 14-bit cutoff into a measured 0.29-wide gap — and the margin, not
   the number, is what tells you the feature works.

And one on process: **we iterated three times on prompt wording by rendering a sheet and
squinting at it.** That is unmeasurable and overfits to whichever mockup is open. Build
the eval set first. `evalset.py` weights confusions by severity — sending a photo to CSS
breaks the page, sending a flat glyph to an image model only wastes money, and averaging
them equally hides exactly the difference you care about.

---

## Still open

- **The skeleton gap.** SAM 3 supplies no layout containers, so the tree has 24 flat
  roots. Needs the VLM-proposed section skeleton described in aha #10.
- **Granularity.** SAM 3 returns one box for "translucent circle + play glyph", which is
  really `token` + `system_icon`. Bundled regions cannot be routed correctly.
- **Single screen.** Every number here comes from one mockup. The eval set exists to make
  the second and third screens honest; they have not been run.
