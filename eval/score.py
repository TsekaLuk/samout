"""Ground truth and scoring, so changes can be measured instead of eyeballed.

Without this the loop is: change a prompt, render a sheet, squint, decide it looks
better. That overfits to whichever mockup is open. With it, every change to
taxonomy.py or the observation prompt produces a number on a fixed set.

Labels are matched to regions by GEOMETRY, not by region id. Ids come from sorting
detections by area, so any change to the detection stage renumbers everything.
Keying the labels on ids made the eval set silently useless for exactly the changes
it is most needed for — adding a second detector scored 22.6% against an 83.0%
baseline while all 53 labelled regions were still present and correctly classified.

    python eval/score.py score assets/handoff.json
    python eval/score.py stub  assets/handoff.json > eval/labels/assets.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uiasset.taxonomy import CLASSES  # noqa: E402

LABEL_DIR = Path(__file__).resolve().parent / "labels"
MIN_IOU = 0.80

# Which confusions matter. Sending a photo to CSS breaks the page; sending a flat
# glyph to an image model wastes money. Weighting them equally hides the difference.
SEVERITY = {
    ("photography", "token"): 3.0,
    ("photography", "typography"): 3.0,
    ("spot_illustration", "token"): 3.0,
    ("spot_illustration", "typography"): 3.0,
    ("brand_asset", "product_icon"): 2.5,      # regenerating a trademark
    ("brand_asset", "system_icon"): 3.0,       # swapping a trademark for a glyph
    ("brand_asset", "spot_illustration"): 2.5,
    ("product_icon", "system_icon"): 2.0,      # a library swap changes the design
    ("system_icon", "product_icon"): 1.0,      # wasteful, not wrong
    ("composite", "spot_illustration"): 2.0,   # sending a nav bar to an image model
    ("composite", "photography"): 2.0,
}
DEFAULT_SEVERITY = 1.0


def worst_case(truth_class):
    """Heaviest penalty this truth class can incur — a fixed per-region denominator
    so weighted error is comparable across runs."""
    return max([w for (t, _), w in SEVERITY.items() if t == truth_class]
               + [DEFAULT_SEVERITY])


def iou(p, q):
    ix0, iy0 = max(p[0], q[0]), max(p[1], q[1])
    ix1, iy1 = min(p[2], q[2]), min(p[3], q[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    ap = (p[2] - p[0]) * (p[3] - p[1])
    aq = (q[2] - q[0]) * (q[3] - q[1])
    return inter / max(1e-6, ap + aq - inter)


def load_labels(name):
    p = LABEL_DIR / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"no ground truth at {p} — run `eval/score.py stub` first")
    return json.loads(p.read_text())


def align(truth, spec_list, min_iou=MIN_IOU):
    """{label_key: matched spec entry or None}, by box overlap.

    Falls back to id matching for label files written before boxes were recorded,
    so old files still work — they just remain brittle.
    """
    boxes = truth.get("boxes") or {}
    by_id = {e["id"]: e for e in spec_list}
    out = {}
    for key in truth["labels"]:
        want = boxes.get(key)
        if want is None:
            out[key] = by_id.get(int(key))
            continue
        best, best_iou = None, min_iou
        for e in spec_list:
            v = iou(want, e["box_xyxy"])
            if v >= best_iou:
                best, best_iou = e, v
        out[key] = best
    return out


def score(handoff_path, label_name=None, verbose=True):
    handoff = json.loads(Path(handoff_path).read_text())
    spec_list = handoff["spec"]
    name = label_name or Path(handoff_path).parent.name
    truth = load_labels(name)
    matched = align(truth, spec_list)

    rows, confusion = [], defaultdict(int)
    weighted_err = weighted_total = 0.0
    unmatched = []
    for key, want in truth["labels"].items():
        hit = matched.get(key)
        got = hit["class"] if hit else "MISSING"
        if hit is None:
            unmatched.append(key)
        w = SEVERITY.get((want, got), DEFAULT_SEVERITY)
        ok = got == want
        weighted_total += worst_case(want)
        if not ok:
            weighted_err += w
            confusion[(want, got)] += 1
        rows.append((key, want, got, ok, hit))

    n = len(rows)
    correct = sum(1 for r in rows if r[3])
    deliv_ok = sum(1 for _, want, got, _, _ in rows
                   if got in CLASSES and want in CLASSES
                   and CLASSES[got]["delivery"] == CLASSES[want]["delivery"])

    if verbose:
        print(f"ground truth: {LABEL_DIR.name}/{name}.json  ({n} regions, "
              f"{len(spec_list)} detected)")
        print(f"  class accuracy    {correct}/{n}  ({correct / n:.1%})")
        print(f"  delivery accuracy {deliv_ok}/{n}  ({deliv_ok / n:.1%})")
        print(f"  severity-weighted error  {weighted_err:.1f}/{weighted_total:.1f}"
              f"  ({weighted_err / weighted_total:.1%})")
        if unmatched:
            print(f"  !! {len(unmatched)} labelled regions had no match above "
                  f"IoU {MIN_IOU}: {unmatched[:10]}")

        if confusion:
            print("\n  confusions, worst first:")
            for (want, got), c in sorted(
                    confusion.items(),
                    key=lambda kv: -SEVERITY.get(kv[0], DEFAULT_SEVERITY) * kv[1]):
                w = SEVERITY.get((want, got), DEFAULT_SEVERITY)
                print(f"    {want:<18} -> {got:<18} x{c}  (severity {w})")

        misses = [(k, want, got, hit) for k, want, got, ok, hit in rows if not ok]
        if misses:
            print("\n  per-region misses:")
            for k, want, got, hit in misses[:25]:
                subj = ((hit or {}).get("subject") or "")[:34]
                print(f"    [{k}] want {want:<18} got {got:<18} {subj}")

    return {"n": n, "class_accuracy": correct / n,
            "delivery_accuracy": deliv_ok / n,
            "weighted_error": weighted_err / weighted_total,
            "unmatched": len(unmatched)}


def stub(handoff_path):
    """Emit a labelling file pre-filled with the model's guesses, INCLUDING boxes.

    Reviewing pre-filled rows takes minutes; labelling from scratch does not. The
    boxes are what make the file survive a detection change.
    """
    handoff = json.loads(Path(handoff_path).read_text())
    out = {
        "_doc": "Correct the `labels` map by hand. Valid classes: "
                + ", ".join(k for k in CLASSES if k != "unobserved"),
        "_matching": f"Labels are matched to regions by box IoU >= {MIN_IOU}, not by "
                     "id, so this file survives changes to the detection stage.",
        "_criteria": {k: v["definition"] for k, v in CLASSES.items()},
        "source": handoff_path,
        "labels": {str(e["id"]): e["class"] for e in handoff["spec"]},
        "boxes": {str(e["id"]): e["box_xyxy"] for e in handoff["spec"]},
        "_subjects": {str(e["id"]): e.get("subject") for e in handoff["spec"]},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "stub"])
    ap.add_argument("handoff")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()
    if args.cmd == "stub":
        stub(args.handoff)
    else:
        score(args.handoff, args.labels)


if __name__ == "__main__":
    main()
