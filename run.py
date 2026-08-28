"""Single entry point. Orchestration only — no logic lives here.

    python run.py input/ktv_ui.jpeg --out assets
    python run.py input/ktv_ui.jpeg --out assets --from observe   # reuse detection

Stages are resumable because each writes its output through `store`. Detection is
the slow local stage and observation is the paid one, so being able to re-run either
alone is what makes iterating on `taxonomy.py` cheap.
"""

import argparse
from pathlib import Path

from PIL import Image

from uiasset import handoff, uitree
from uiasset.components import consensus, group, probes_disagree
from uiasset.config import Config
from uiasset.crops import CropSource, write_cutouts
from uiasset.detect import describe, detect, measure
from uiasset.split import split_regions
from uiasset.uidetect import detect_and_merge
from uiasset.observe import observe, recall_audit
from uiasset.store import Store


def ref_measured(ref):
    """`measure` output, pulled back out of the reference map `store` persists."""
    return {rid: v.get("measured", {}) for rid, v in (ref or {}).items()}

STAGES = ["detect", "group", "observe", "assemble"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--from", dest="start", choices=STAGES, default="detect",
                    help="resume from this stage using cached earlier output")
    ap.add_argument("--model", default=None, help="override the VLM")
    ap.add_argument("--sam3", default=None, help="override the SAM 3 checkpoint")
    ap.add_argument("--no-uidetect", action="store_true",
                    help="SAM 3 only, skip OmniParser UI element detection")
    ap.add_argument("--split", action="store_true",
                    help="un-bundle container+glyph regions (experimental, see config)")
    ap.add_argument("--generate", action="store_true",
                    help="generate every regenerable asset with the image model")
    ap.add_argument("--image-model", default=None,
                    help="override the image model (default qwen-image-3.0; "
                         "qwen-image-3.0-pro is ~2.2x slower for marginal gain)")
    ap.add_argument("--recall", action="store_true",
                    help="ask the VLM what the detector missed")
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.observe.model = args.model
    if args.split:
        cfg.split.enabled = True
    if args.no_uidetect:
        cfg.uidetect.enabled = False

    image = Image.open(args.image).convert("RGB")
    store = Store(args.out)
    page = [0, 0, image.width, image.height]
    split_parents = set()
    begin = STAGES.index(args.start)

    # --- 1. detect -------------------------------------------------------
    if begin <= 0:
        print("[detect]")
        regions = detect(image, cfg.detect, model_id=args.sam3)
        # SAM 3 finds concepts; OmniParser finds tap targets. Neither spans a screen
        # alone — see uidetect.py for the measured gap.
        if cfg.uidetect.enabled:
            regions = detect_and_merge(image, regions, cfg.uidetect)
        ref = describe(image, regions)
        meas = measure(image, regions)

        # Un-bundle "container + glyph" regions before anything downstream sees
        # them. A wrong region cannot be recovered by better mattes or better
        # rules, so it has to be fixed here.
        for rid, m in meas.items():
            ref.setdefault(rid, {})["measured"] = m
        n, nmat, nref = write_cutouts(image, regions, store.cutouts,
                                      use_matting=cfg.observe.use_matting)
        print(f"  {n} cutouts -> {store.cutouts}  "
              f"({nmat} ViTMatte, {nref} soft mask, {n - nmat - nref} binary)")
        store.save_regions(regions, args.image, image.size, ref)
    else:
        regions, _, ref = store.load_regions()
        if not regions:
            raise SystemExit(f"no cached regions in {args.out}; run --from detect")
        print(f"[detect] cached: {len(regions)} regions")

    crops = CropSource(image, store.cutouts, cfg.observe.use_mask_cutouts,
                       cfg.observe.cutout_backdrop)

    # --- 2. group --------------------------------------------------------
    if begin <= 1:
        print("[group]")
        by_id_count = {r.id for r in regions}
        components, comp_of = group(crops, regions, cfg.components)
        repeated = [c for c in components if c.count > 1]
        print(f"  {len(regions)} regions -> {len(components)} components "
              f"({len(repeated)} repeated)")
        for c in sorted(repeated, key=lambda c: -c.count)[:6]:
            print(f"    {c.key} x{c.count}  {c.members[:8]}")
        # Split AFTER grouping, so a repeated element is decomposed once and the
        # decision is applied to every instance. Splitting per region produced
        # different child extents for identical buttons, which then regrouped into
        # two components with two classes.
        if cfg.split.enabled:
            regions, notes = split_regions(image, regions, components, ref_measured(ref))
            split_parents = set(notes.get("_split_parents") or ())
            if len(regions) > len(by_id_count):
                new = [r for r in regions if r.id not in ref]
                ref.update(describe(image, new))
                for rid, m in measure(image, new).items():
                    ref.setdefault(rid, {})["measured"] = m
                write_cutouts(image, new, store.cutouts,
                              use_matting=cfg.observe.use_matting)
                store.save_regions(regions, args.image, image.size, ref)
                components, comp_of = group(crops, regions, cfg.components)
                print(f"  regrouped: {len(components)} components after split")
        store.save_components(components)
    else:
        components = store.load_components()
        comp_of = {rid: c.key for c in components for rid in c.members}
        print(f"[group] cached: {len(components)} components")

    # --- 3. observe ------------------------------------------------------
    if begin <= 2:
        print(f"[observe] {cfg.observe.model}")
        by_id = {r.id: r for r in regions}
        probe_ids = [i for c in components for i in c.probes]
        probe_obs, usage, dt = observe(
            image, crops.pairs([by_id[i] for i in probe_ids]), cfg.observe)
        print(f"  {dt}s  in={usage['prompt_tokens']} out={usage['completion_tokens']}")

        # Reconcile probes into one observation per component, then propagate.
        # Disagreement between probes means the grouper merged two different
        # components — dissolve the group rather than vote the difference away.
        obs, dissolved, orphans = {}, [], []
        for c in components:
            probes = [probe_obs.get(i) for i in c.probes]
            if c.count > 1 and probes_disagree(probes, cfg.components.hue_slack):
                dissolved.append(c.key)
                for rid in c.members:
                    if probe_obs.get(rid):
                        obs[rid] = probe_obs[rid]
                    else:
                        # Members that were never probed have no observation of
                        # their own, and the group's consensus is exactly what we
                        # just rejected. Leaving them would silently mark them
                        # unobserved, so collect them for a second pass.
                        orphans.append(rid)
                continue
            merged = consensus(probes)
            if not merged:
                continue
            dropped = (merged.get("consensus") or {}).get("cues_dropped")
            if dropped:
                print(f"  {c.key} x{c.count}: dropped background cues {dropped}")
            for rid in c.members:
                obs[rid] = merged
        if dissolved:
            print(f"  {len(dissolved)} groups dissolved on probe disagreement: "
                  f"{dissolved}")
        if orphans:
            print(f"  re-observing {len(orphans)} members of dissolved groups")
            extra, u2, dt2 = observe(
                image, crops.pairs([by_id[i] for i in orphans]), cfg.observe)
            obs.update(extra)
            usage["prompt_tokens"] += u2["prompt_tokens"]
            usage["completion_tokens"] += u2["completion_tokens"]
            dt += dt2
        split = len(dissolved)
        store.save_observations(obs, {"model": cfg.observe.model, "latency_s": dt,
                                      "usage": usage, "groups_split": split})
    else:
        obs = store.load_observations() or {}
        print(f"[observe] cached: {len(obs)} observations")

    # --- 4. assemble -----------------------------------------------------
    print("[assemble]")
    nodes, roots = uitree.annotate(regions, obs, page, cfg.tree, comp_of)
    internal = sum(1 for n in nodes.values() if n.children)
    print(f"  tree: {len(roots)} roots, depth {max(n.depth for n in nodes.values())}, "
          f"{internal} internal / {len(nodes) - internal} leaves")

    meas = {rid: v.get('measured', {}) for rid, v in (ref or {}).items()}
    spec = handoff.build(regions, nodes, obs, components, ref, meas, split_parents)
    grids = handoff.find_grids(spec)
    n_rep = handoff.mark_repeating(spec, nodes)
    if grids:
        for g in grids:
            print(f"  grid {g['id']}: {g['cells']} {g['axis']} cells, gap {g['gap']}px "
                  f"(cv {g['gap_cv']}), {g['cell_classes']}")
    if n_rep:
        print(f"  {n_rep} repeating containers marked collapsible")
    summary = handoff.summarize(spec)
    spec_by_id = {e["id"]: e for e in spec}

    text = uitree.to_text(nodes, roots, spec_by_id)
    store.save_tree(text, uitree.to_json(nodes, roots, spec_by_id))

    missed = []
    if args.recall:
        try:
            missed, rdt = recall_audit(image, [r.box for r in regions],
                                       cfg.observe.model)
            print(f"  recall audit: {rdt}s, {len(missed)} missed")
            for m in missed:
                print(f"    {str(m.get('what'))[:42]:<44} retry SAM 3 with "
                      f"'{m.get('sam3_prompt')}'")
        except Exception as e:
            print(f"  recall audit failed: {e}")

    gen = []
    if args.generate:
        from uiasset.generate import generate_all
        print("[generate]")
        gen = generate_all(spec, store.cutouts, args.out / "generated",
                           **({"model": args.image_model} if args.image_model else {}))

    store.save_handoff(spec, summary,
                       {"missed": missed, "generated": gen, "grids": grids})

    print("\nclass mix:")
    for k, v in sorted(summary["by_class"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v:>3}")
    print(f"\ngenerate {summary['to_generate']}  extract {summary['to_extract']}  "
          f"effort {summary['total_effort']}  flagged {summary['flagged']}")
    print(f"\n{args.out}/  regions.json components.json observations.json "
          f"handoff.json uitree.txt uitree.json cutouts/")


if __name__ == "__main__":
    main()
