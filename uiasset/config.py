"""Every tunable in one place, each with where its value came from.

Thresholds were previously scattered across four modules, which made it impossible
to answer "what would I change to get more assets and fewer library icons?" without
reading all of them. Calibrated values say so; guesses say so too.
"""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_DIR = Path("models/sam3")
HUB_FALLBACK = "jetjodh/sam3"  # ungated mirror; facebook/sam3 needs manual approval


@dataclass
class DetectConfig:
    """SAM 3 segmentation."""

    # Concrete physical nouns only. Abstract style words and layout words return
    # zero at every threshold — see BENCHMARK.md §1. "3d icon" is kept because its
    # hit/miss pattern tracks the rendered-vs-flat boundary on its own.
    prompts: tuple = ("icon", "3d icon", "photo", "thumbnail", "logo", "avatar",
                      "neon sign")
    threshold: float = 0.30        # detection score; 0.3 keeps low-confidence tab icons
    mask_threshold: float = 0.50
    nms_iou: float = 0.55          # cross-prompt merge; icon/logo overlap heavily
    min_area_px: int = 600
    max_area_frac: float = 0.80    # a near-full-canvas hit is the model latching
                                   # onto the whole screenshot


@dataclass
class SplitConfig:
    """Un-bundling "container + glyph" regions with SAM 2 point prompts.

    OFF by default: measured, it regresses class accuracy 81.1% -> 71.7%. The
    extraction itself works — 5 of 6 attempts pulled the play glyph cleanly out of
    its disc — but three defects make it a net loss today:

      1. Non-deterministic extent. SAM 2 returned 20x22 for three instances of the
         same play button and 19x32 for two others, so component grouping split them
         into two groups and gave them two different classes. The split must be run
         once per COMPONENT and mapped onto its instances, exactly as observation is.
      2. False splits on self-similar icons. The signal-bars glyph "split" into a
         smaller copy of itself, which is a duplicate, not a decomposition. Needs a
         test that the child is a different shape, not just a smaller region.
      3. The parent lands in `composite` when `token` (a CSS disc) is the useful
         answer, because having children currently outranks everything.

    Also: the eval set labels 53 fixed regions, so it cannot score the children this
    creates. Any detection-stage change is partly outside its range — see
    OPTIMIZATION.md, "the blind spot".
    """

    enabled: bool = False
    min_lum_std: float = 70.0      # interior variance implying something shows through
    min_frac: float = 0.06         # child share of the parent
    max_frac: float = 0.40         # true splits ran 0.18-0.25, the false one 0.54
    min_contrast: float = 60.0     # true splits ran 132-190, the false one 26


@dataclass
class UIDetectConfig:
    """OmniParser UI element detection, merged with SAM 3's concept regions.

    On by default because the recall gap it closes is not marginal: SAM 3 alone
    missed 76% of the elements on a design-board screenshot (19% on a single app
    screen). Turn it off to reproduce SAM-3-only numbers.
    """

    enabled: bool = True
    conf: float = 0.10       # 153 elements at 0.10, 128 at 0.20 on the test board
    dup_iou: float = 0.55    # overlap at which an element is already a SAM 3 region
    # A tap target enclosing exactly one existing region is that region's hit area,
    # not a container, and is dropped. Keeping them turned 17 boxes into `composite`
    # parents and cost 7.5pp. Area ratio does not separate the cases (3.1-17.5x,
    # no gap); the number of enclosed siblings does.
    contain_cover: float = 0.90


@dataclass
class ComponentConfig:
    """Grouping repeated regions before observing them."""

    # Calibrated, not guessed: seven play-button instances scored 0.834-0.970 against
    # each other while unrelated icon pairs topped out at 0.546. Anything in 0.55-0.75
    # gives full recall with no false merge; 0.70 sits mid-gap. A false merge assigns a
    # wrong class, a false split costs one redundant call — so err high.
    min_similarity: float = 0.70
    aspect_tol: float = 0.25       # log-ratio
    size_tol: float = 0.90         # log-ratio; instances legitimately vary in size
    max_probes: int = 3            # observations per component, reconciled by consensus
    hue_slack: int = 1             # probe hue spread beyond this splits the group


@dataclass
class ObserveConfig:
    """VLM observation pass."""

    # Six Qwen models agreed 92-100% on this task (BENCHMARK.md §4a), so it is not
    # tier-limited. flash is 5.4x faster than the flagship for identical verdicts.
    model: str = "qwen3-vl-flash"
    batch: int = 10
    workers: int = 6               # batches are independent; 170s -> 44s
    sweeps: int = 3                # re-sweep misses with a halved batch each pass
    # Self-consistency sampling, kept but off by default.
    #
    # Added to fight a 13.2pp spread across identical inputs, and it did raise
    # accuracy — but it was treating a symptom. The spread came from noisy VLM fields
    # being consulted BEFORE a deterministic measurement in `taxonomy.classify`; six
    # of the eight flipping regions were one 7-member component whose consensus
    # toggled. Reordering that rule made the outcome insensitive to VLM jitter, and
    # k=1 then scored identically on 3/3 runs.
    #
    # Measured, 3 runs each: k=1 26.5s 83.0%/86.8% stable | k=2 47.7s 84.9%/87.4%
    # | k=3 66.9s 83.7% and less stable. k=2 buys ~1.9pp (one region) for 1.8x the
    # observation bill; raise it when accuracy matters more than throughput.
    samples: int = 1
    context_max_side: int = 900    # the whole-mockup context image
    sheet_max_side: int = 1500     # the contact sheet of region crops
    # Composite the SAM 3 mask onto neutral grey instead of sending a box crop.
    # Measured: +11.3pp class accuracy, +15.1pp delivery accuracy. INSIGHTS aha #10.
    use_mask_cutouts: bool = True
    # Segmentation and matting are different tasks. SAM 3 decides which object;
    # ViTMatte solves the edge from a trimap derived from SAM's own mask. Without
    # it the sprite carries SAM's low-resolution boundary and a rim of whatever it
    # was lifted off.
    use_matting: bool = True
    cutout_backdrop: tuple = (128, 128, 128)   # grey implies no lighting direction


@dataclass
class TreeConfig:
    """UI tree reconstruction."""

    cover: float = 0.88            # fraction of a child that must fall inside a parent
    shrink: float = 0.92           # a parent must be meaningfully larger than its child
    row_overlap: float = 0.50      # vertical overlap for two children to share a row
    heading_height_frac: float = 0.028   # leaf text taller than this reads as a heading


@dataclass
class Config:
    detect: DetectConfig = field(default_factory=DetectConfig)
    uidetect: UIDetectConfig = field(default_factory=UIDetectConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    observe: ObserveConfig = field(default_factory=ObserveConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)

    @staticmethod
    def model_id():
        return (str(DEFAULT_MODEL_DIR)
                if (DEFAULT_MODEL_DIR / "model.safetensors").exists()
                else HUB_FALLBACK)
