"""All persistence. No other module opens a file.

Stages used to write their own JSON in their own shapes — `assets.json`,
`handoff.json`, `characterization.json`, `audit_*.json` — and the next stage dug
the region list back out with different field names. Centralising it means the
on-disk format is one file's problem, and a stage can be run from cached input
without knowing how that input was stored.
"""

import json
from pathlib import Path

from .model import Component, Node, Region


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def cutouts(self):
        return self.root / "cutouts"

    def _write(self, name, obj):
        p = self.root / name
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        return p

    def _read(self, name):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else None

    # --- detection -------------------------------------------------------
    def save_regions(self, regions, source, size, describe=None):
        return self._write("regions.json", {
            "source": str(source),
            "size": {"width": size[0], "height": size[1]},
            "regions": [r.as_json() for r in regions],
            "reference": {str(k): v for k, v in (describe or {}).items()},
        })

    def load_regions(self):
        d = self._read("regions.json")
        if not d:
            return None, None, None
        regions = [Region.from_json(r) for r in d["regions"]]
        ref = {int(k): v for k, v in (d.get("reference") or {}).items()}
        return regions, (d["size"]["width"], d["size"]["height"]), ref

    # --- components ------------------------------------------------------
    def save_components(self, components):
        return self._write("components.json",
                           [c.as_json() for c in components])

    def load_components(self):
        d = self._read("components.json")
        return [Component(key=c["key"], members=c["members"],
                          representative=c["representative"],
                          probes=c.get("probes", [])) for c in (d or [])] or None

    # --- observation -----------------------------------------------------
    def save_observations(self, obs, meta):
        return self._write("observations.json",
                           {"meta": meta,
                            "observations": {str(k): v for k, v in obs.items()}})

    def load_observations(self):
        d = self._read("observations.json")
        if not d:
            return None
        return {int(k): v for k, v in d["observations"].items()}

    # --- handoff ---------------------------------------------------------
    def save_handoff(self, spec, summary, extra=None):
        return self._write("handoff.json",
                           {"summary": summary, "spec": spec, **(extra or {})})

    def load_handoff(self):
        return self._read("handoff.json")

    def save_tree(self, text, tree_json):
        (self.root / "uitree.txt").write_text(text)
        return self._write("uitree.json", tree_json)
