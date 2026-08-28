"""Latency/quality bench: same audit batch, many models, run concurrently.

The audit prompt is the expensive one (two images + a JSON array out), so it is
the right thing to time. Candidates are probed for vision support first — several
Qwen text models accept the call and then fail on the image part.

    python bench_models.py input/ktv_ui.jpeg assets/assets.json
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from classify_vlm import SYSTEM, contact_sheet
from vlm import chat, encode_image, parse_json

CANDIDATES = [
    "qwen3-vl-flash",
    "qwen3-vl-plus",
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen3.8-flash",
    "qwen3.8-max",
]


def run_one(model, image, sheet, chunk):
    meta = "\n".join(
        f"#{r['id']}: {r['size_px'][0]}x{r['size_px'][1]}px, SAM3 labels {r['labels']}"
        for r in chunk)
    t0 = time.time()
    try:
        text, usage, dt = chat(
            model,
            [encode_image(image, max_side=900),
             "Full mockup above, for context. Below: the regions to audit.",
             encode_image(sheet, max_side=1500),
             f"Region metadata:\n{meta}\n\nAudit ids: {[r['id'] for r in chunk]}. "
             "JSON array only."],
            system=SYSTEM, max_tokens=6000, temperature=0.0, retries=1)
        parsed = parse_json(text)
        ids = {int(v["id"]) for v in parsed}
        want = {r["id"] for r in chunk}
        return {
            "model": model, "ok": True, "latency_s": dt,
            "in": usage.get("prompt_tokens", 0),
            "out": usage.get("completion_tokens", 0),
            "coverage": f"{len(ids & want)}/{len(want)}",
            "n_asset": sum(1 for v in parsed if v.get("verdict") == "asset"),
            "verdicts": {int(v["id"]): v.get("verdict") for v in parsed},
            "sample": [(v.get("id"), v.get("verdict"), (v.get("reason") or "")[:44])
                       for v in parsed[:4]],
        }
    except Exception as e:
        return {"model": model, "ok": False, "latency_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {str(e)[:160]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--models", nargs="+", default=CANDIDATES)
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    man = json.loads(args.manifest.read_text())
    regions = [{"id": a["id"], "box_xyxy": a["box_xyxy"],
                "size_px": a["size_px"], "labels": a["labels"]}
               for a in man["assets"]][:args.batch]
    sheet = contact_sheet(image, regions)

    print(f"benchmarking {len(args.models)} models on {len(regions)} regions, concurrent\n")
    with ThreadPoolExecutor(max_workers=len(args.models)) as ex:
        results = list(ex.map(lambda m: run_one(m, image, sheet, regions), args.models))

    results.sort(key=lambda r: (not r["ok"], r.get("latency_s", 999)))
    print(f"{'model':<18} {'ok':<4} {'sec':>6} {'in':>6} {'out':>6} {'cover':>7} {'asset':>6}")
    for r in results:
        if r["ok"]:
            print(f"{r['model']:<18} {'y':<4} {r['latency_s']:>6} {r['in']:>6} "
                  f"{r['out']:>6} {r['coverage']:>7} {r['n_asset']:>6}")
        else:
            print(f"{r['model']:<18} {'n':<4} {r['latency_s']:>6}   {r['error'][:60]}")

    # Agreement matrix across the models that worked
    ok = [r for r in results if r["ok"]]
    if len(ok) > 1:
        print("\npairwise verdict agreement:")
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                shared = set(a["verdicts"]) & set(b["verdicts"])
                if not shared:
                    continue
                same = sum(1 for k in shared if a["verdicts"][k] == b["verdicts"][k])
                print(f"  {a['model']:<18} vs {b['model']:<18} "
                      f"{same}/{len(shared)} ({same / len(shared):.0%})")

    Path("plans").mkdir(exist_ok=True)
    Path("plans/bench.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
