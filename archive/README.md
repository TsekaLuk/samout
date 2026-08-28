# Superseded

Kept for provenance, not imported by anything. See `../INSIGHTS.md`.

- `classify_vlm.py` — asked the VLM for a css/library/asset **verdict**. Replaced by
  `uiasset/observe.py` + `uiasset/taxonomy.py` after the model rationalised the same
  six icons into the wrong bucket twice, in opposite directions. INSIGHTS aha #3-4.
- `policy.py` — routing as a cost model: the VLM estimates per-path reproduction risk,
  code picks the cheapest path under a tolerance `tau`. A good framing, and a better one
  than what it replaced, but `taxonomy.py` subsumed it: the design-system class already
  determines the delivery path, so the risk estimate became a second opinion nobody read.
  Worth revisiting if delivery ever needs to depend on project context (no icon set
  available, mockup resolution too low to crop) rather than on the class alone.
