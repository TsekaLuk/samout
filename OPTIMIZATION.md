# Where to optimize

A prioritized map, derived from attributing every measured error to a stage rather
than from intuition about which component feels weakest.

Baseline at time of writing: **77.4% class / 81.1% delivery**, 53 regions, one screen.

---

## The framework: attribute each error to exactly one stage

The pipeline is `detect → group → observe → rule`. An error belongs to the **earliest
stage that could have prevented it**. That single convention is what makes the budget
actionable — otherwise every error looks like it could be fixed anywhere.

| stage | owns | an error here looks like |
|---|---|---|
| detect | which regions exist, and their boundaries | one box holding two things; content never found |
| group | which regions are instances of one component | a false merge propagates one class to many |
| observe | what the pixels are, factually | right region, wrong description |
| rule | class and delivery given the description | right description, wrong class |
| *(labels)* | — | the ground truth itself is arguable |

## Current error budget: 12 misses / 53

| stage | count | ids |
|---|---|---|
| **observe** | **8** | 7, 16, 26, 27, 28, 30, 32, 43, 48 |
| rule | 1 | 21 |
| labels debatable | 2 | 49, 51 |
| detect | 0 *measured* | — see the blind spot below |

Within observation:

| failure | count | ids |
|---|---|---|
| flat-vs-shaded not separable from the reported features | 4 | 27, 28, 30, 32 |
| stylized lettering read as `text` / `photographic` | 3 | 7, 16, 26 |
| small region hallucinated | 2 | 43, 48 |

---

## The blind spot that matters most

**The eval set only scores classification of regions that were detected.** There are two
independent axes and only one is measured:

| axis | question | metric today |
|---|---|---|
| 1. classification | given a region, is its class right? | 77.4% |
| 2. coverage | are the right regions there at all? | **none** |

SAM 3's errors live almost entirely on axis 2 — missing layout containers, bundled
regions, art it has no vocabulary for. They score as zero errors because the regions
they would have produced do not exist to be scored.

**So the first investment in detection is a recall metric, not a model change.** Label
the regions that *should* exist on one screen, and report coverage alongside accuracy.
Until then, any claim that "SAM is fine" is an artifact of what we chose to count.

---

## Levers, ranked by attributed error

### Observation — 8 of 12 errors

**L1. Normalize region scale before showing it.** Contact-sheet cells are a fixed 190px,
so a 42×24 neon sign is upscaled 5× into blur while an 860px banner is downscaled 4.5×
into mush. Both hallucinations (#43 "50%", #48 "NEXUS") are small regions; both read
plausible text that is not there. Fix: bucket regions by size and run a pass per bucket
at an appropriate resolution, padding small crops rather than upscaling them.
*Confidence: high. Cost: low.*

**L2. Compute what is computable; ask only what is not.** `hue_count` is a measurement,
not an observation, and asking a VLM for it was worse than useless — it does not separate
the classes at all (7 of 10 product icons are monochrome too). A real separator exists in
the pixels: interior luminance standard deviation, `lum_std_inner`, gives **20/24 against
a 14/24 majority baseline**, with a mechanism — a translucent disc lets the photo behind
it through and spikes the variance, while a solid rendered object is smooth.
*Confidence: medium-high for `lum_std` alone. Cost: low. See the overfitting note below.*

**L3. Add `display_lettering` to `content_type`.** Three errors are stylized headline art
falling into `text` or `photographic` because the enum has no home for it. A wordmark, a
3D extruded headline and a neon sign are lettering that must ship as art.
*Confidence: high. Cost: trivial.*

**L4. Few-shot the observation prompt from the ground truth.** 53 labelled regions exist
now; four or five exemplar crops with their correct `content_type` cost little context
and target exactly the confusions above.
*Confidence: medium. Cost: low.*

### Rule — 1 of 12

**L5. Order the checks by authority.** `composite` currently overrides everything, so a
brand badge that happens to contain sub-regions (#21) loses its `brand_asset` class and
its never-regenerate flag. Identity should outrank structure.
*Confidence: high. Cost: trivial.*

### Detection — unmeasured, likely the largest true cost

**L6. Build the coverage metric first.** Everything below is unjustifiable without it.

**L7. Split bundled regions.** SAM 3 returns one box for "translucent disc + flat glyph";
the disc's shading is then attributed to the glyph, which is the mechanism behind four
errors. SAM 3 accepts geometric prompts, so the fix is available: point-prompt the
interior of a detected region to segment the inner object separately, and keep both as
parent and child. This is the highest-value *structural* fix in the whole pipeline.
*Confidence: medium. Cost: medium.*

**L8. Supply the layout skeleton from the VLM.** SAM 3 does not model containers at all
(`panel`, `navigation bar`, `section` return zero at every threshold), so the tree is 24
flat roots with no header, no search bar, no sections. Have the VLM propose sections with
boxes and ARIA roles, and hang the SAM 3 leaves on that skeleton.
*Confidence: high that it is needed. Cost: medium.*

### Labels — 2 of 12

**L9. Adjudicate the disputed labels.** A notification badge is a circle plus a live
number; calling it `typography` or `token` are both defensible. Write the convention
down in the labels file rather than letting the metric absorb the ambiguity.

---

## Method note: calibration is not a licence to overfit

Two threshold-fitting exercises in this project, one sound and one not. The difference
is worth stating because they look identical from the outside.

**Sound — the component similarity threshold.** One feature, one threshold, evaluated on
21 positive and 66 negative pairs. Same-component scores ran 0.834–0.970, different-
component topped out at 0.546. **A 0.29-wide gap**, and any threshold inside it works.
The margin, not the accuracy, is what justified shipping it.

**Not sound — a two-feature exhaustive search.** Searching 6 features × every threshold ×
both directions × `and`/`or` is on the order of 10⁴ hypotheses against 24 binary points.
It returned `lum_std < 83.06 and sat_std > 0.111` at **24/24**, which is meaningless on
its own: with that many hypotheses a perfect split is expected by chance.

Checking the margins separates the two halves of that rule:

| term | gap | verdict |
|---|---|---|
| `lum_std < 83` | product max 66.6 → system min 87.3 (**20.7 wide**) | real, and mechanistically explicable |
| `sat_std > 0.111` | system max 0.110 → product min 0.118 (**one sample wide**) | knife-edge, do not ship |

So: adopt the first, hold the second until a second screen exists. **Report the margin
whenever you report a calibrated threshold** — accuracy alone cannot distinguish a
measurement from a coincidence.

---

## Sequence

1. **L5, L3** — trivial, immediate, no new measurement needed.
2. **L1** — scale normalization; addresses the hallucination class.
3. **L6** — coverage metric; unblocks all detection work and ends the blind spot.
4. **L2** — `lum_std_inner` as a computed feature, `sat_std` held back.
5. **L7** — bundling split; the structural fix behind the largest single confusion.
6. **L8** — layout skeleton.
7. **Second and third screen.** Every number in this file comes from one mockup. Nothing
   above is trustworthy until it survives a screen it was not tuned on — particularly
   anything with "calibrated" next to it.
