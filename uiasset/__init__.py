"""UI mockup -> design handoff spec.

Layering, strictly one-way:

    model.py      records shared by every stage; imports nothing from the package
    config.py     every tunable, with its provenance
    vlm.py        DashScope client

  pure (no IO, no model calls — this is where the logic lives, and it is testable):
    taxonomy.py   design-system class + delivery rules, each with a published source
    uitree.py     containment forest, layout inference, WAI-ARIA roles
    components.py visual grouping of repeated regions, probe consensus
    handoff.py    joins every stage's output into the spec

  effectful:
    detect.py     SAM 3 segmentation
    crops.py      the only module that knows where a region's pixels come from
    observe.py    VLM observation; takes [(id, Image)] and knows nothing else
    plan.py       VLM proposes SAM 3 prompts for a screen
    store.py      the only module that opens a file
    report.py     rendering

No stage imports another stage. `handoff` imports the pure modules to join their
results; `run.py` orchestrates. That keeps the dependency graph a DAG.
"""
