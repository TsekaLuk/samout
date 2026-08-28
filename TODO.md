# State and what to do next

Updated 2026-08-28. Ordered by what unblocks what, not by appeal.

Baseline: **90.6% class / 92.5% delivery** on 53 labelled regions, one screen,
reproducible across runs. ~50 s analysis per screen, ~9 min to generate 22 assets.

---

## Done and verified

| | evidence |
|---|---|
| SAM 3 detection | deterministic, 53/53 identical boxes across runs |
| OmniParser element detection | closes a 76% recall gap on chrome-heavy screens |
| ViTMatte cutouts | deterministic, 53/53 byte-identical, clean edges |
| Component grouping | calibrated on known pairs, 0.29-wide margin |
| Deterministic-first rule order | removed the bimodal 13.2pp spread entirely |
| Design-system taxonomy | every class cites a published source |
| UI tree: containment, layout, ARIA roles | pure geometry, no thresholds left |
| Rules as data (`ruleset.py`) | listable, testable, auditable; behaviour pinned |
| Generation loop | 22/22 generated, 20/20 matted, reference-faithful |
| Fidelity clamps | recovered dropped props; overlay text no longer baked in |
| Vision-embedding reuse | detection 23.2 s → 9.7 s, output identical |
| Crop-before-compute | 74× fewer pixel ops, output identical |
| Smoke tests | 42 checks, needs neither key nor mockup |
| Input validation | small/huge/missing images fail with the real cause |

---

## B — Blocking

### B1. More screens
**Everything below is unjustifiable without this, and it is the only item that has
repeatedly changed conclusions.**

Three screens so far, one labelled. Each new one overturned something:

| screen | what it broke |
|---|---|
| KTV app (labelled) | the baseline |
| Virelle design board | SAM 3 alone missed 76% of elements — forced a second detector |
| bilibili live page | skeleton loaders classed as photography; overlays baked into assets |

Two of the three were *designed* mockups; the first *rendered* page moved the failure
mode somewhere the existing tests could not reach. The next screens should vary
provenance deliberately — an admin dashboard, a form-heavy flow, a dark-mode app, a
right-to-left layout.

Each needs ground truth to be worth much: `python eval/score.py stub <handoff> >
eval/labels/<name>.json`, then correct by hand. Labels match by box IoU, so they
survive detection changes.

*Effort: a few hours per screen. Confidence: high that it finds something.*

---

## H — High value

### H2. Enable region splitting
`split.py` works — per-component now, so one decision applies to every instance, and
5 of 6 attempts pulled a play glyph cleanly out of its disc. It is **gated off**
because the eval set covers 53 fixed regions and cannot score the children a split
creates; the parents also change class, so the measured "regression" that first
disabled it was never trustworthy.

Needs M5 before it can be judged. The three original defects are fixed:
per-component splitting, a shape-difference test against self-similar icons, and the
parent correctly landing in `token`.

### H3. Shrink the VLM to what has no deterministic source
Partly done — `text_role` split out, placeholder detection moved to measurement.
Still outstanding:

| field | replace with | evidence |
|---|---|---|
| `hue_count` | delete or measure | computing beat asking, and it barely separates the classes |
| `depth_cues` | mostly `lum_std_inner` | `bevel`+`specular` carry zero signal |
| `has_baked_text` | OCR | deterministic, and returns the copy too |

The VLM is the only non-deterministic stage; every field removed is variance removed.

### H4. Z-order and region extent
The two abstractions standing behind most of the accumulated rules, verified against
the patches rather than argued for — see below. **Blocked on B1**: designing either
on three screens would repeat a mistake this repo already records twice.

---

## M — Medium

### M5. Extend the eval set
53 fixed regions cannot score regions that splitting or a new detector creates, and
two labels are genuinely arguable (a count badge is a circle plus a number). Write
the convention into the labels file rather than letting the metric absorb it.

### M6. Backdrop channel
Detection and verdict are pure computation and were prototyped: 40.3% of pixels
belong to no region, gradient-fit residual 25.1/255 → the backdrop is raster, not a
CSS gradient. Not wired as a stage. It must **not** enter the region tree — a
backdrop is below everything in z-order, not a parent in containment.

Extracting a true backdrop layer needs inpainting; deferred.

### M7. Adopt Design2Code-style metrics
The eval protocol here is hand-rolled. Design2Code and WebSight already define
screenshot→code metrics — block match, text similarity, position error, colour
distance. Reinventing that, worse, is not a good use of time.

### M8. Occlusion handling
A popover's contents currently merge with the page beneath it. Part of H4 but worth
listing separately: it is the one occlusion case already observed in the wild rather
than reasoned about.

---

## L — Low / deferred

- **L9. Two-dimensional grids.** `find_grids` finds rows and columns; a calendar is a
  7×5 grid and only one of its rows is detected. The cells are all correctly CSS, so
  this is legibility, not correctness.
- **L10. Disco ball (#19).** A hard segmentation failure, mask is a vertical sliver.
  One region on one screen; may disappear under H4.
- **L11. `plan.py`** — VLM proposes SAM 3 prompts per screen. Written and benchmarked,
  not in the pipeline. Only matters once B1 shows the fixed prompt set failing on a
  new domain.
- **L12. Sprite parallelism in generation.** Cutouts run serially because SAM 2 and
  ViTMatte are module globals and not thread-safe. Worth ~40 s in a 9-minute run;
  giving each worker its own model costs 1.6 GB. Calculated and declined.

---

## The two abstractions the rules are standing in for

Forty branches, now records in `ruleset.py` — listable, testable, explainable, but
that is containment rather than a cure. Auditing what forced each one, they fall
into two groups.

### A. Z-order — no model of layers

| patch | what it really is |
|---|---|
| `text_role: live_data` excludes view counts | a data layer over an image layer |
| stripping overlay clauses from the prompt | same |
| splitting "translucent disc + glyph" | a container layer under an icon layer |
| popover occlusion (unfixed) | an overlay layer over the page |

**A UI is composited and this treats it as one flat image.** A region's asset is what
sits on *its* layer, not everything visible inside its rectangle.

### B. Region extent vs semantic extent

| patch | what it really is |
|---|---|
| `brand_asset` with ≥2 children downgrades | the box is larger than the mark |
| a tap target enclosing one region is a hit area | the interaction box is larger than the icon |
| `split.py` | one box holding two things |
| `find_grids` | many boxes holding one thing |

**A detected box is assumed to be one semantic unit and often is not**, in both
directions.

The layer model explains 4 of 6 patches and not the other 2 — which is how the second
abstraction was found. Both wait on B1.

---

## Open, and not yet answerable

### Generation quality on thin references
Two of eight generated assets came back unusable, both small crops of detailed
*scenes*. Icons and photographs of the same pixel count came back clean.

No predictor survives measurement. Pixel count, colour count, entropy, information
density and post-hoc similarity to the reference were each tabulated; successes and
failures interleave on every one. A size gate is doubly dead — the median asset on
that board is 1599 px, so any threshold covering the two failures flags 74–90% of
everything.

The gate is left off in `handoff.py` with the numbers recorded. A bad asset can still
enter the handoff looking correct; that is the honest state of this stage.

---

## Sequencing

```
B1 (more screens)
 ├─> M5 (eval covers what they add)
 │    └─> H2 (splitting becomes judgeable)
 ├─> H4 (enough evidence to design layers / extent)
 │    └─> M8 (occlusion)
 └─> H3 (fewer VLM fields, less variance)
```

M6 and M7 are independent.

**The thing to avoid:** designing H4 now. Three screens has already produced two
thresholds fitted to whatever was in front of them, and both had to be withdrawn.
