"""Classification rules as data, not as an if/elif chain.

Forty branches had accumulated across four functions — `classify`, `_flags`,
`build`'s first pass, and `build_prompt` — and none of them could answer the
question that matters when a result looks wrong: *which rules fired on this region,
in what order, and why?* The only way to find out was to read the functions and
simulate them.

Making rules records instead of branches buys four things that a chain cannot:

  inspect   list every rule, ordered, with its stated reason and its evidence
  explain   record the exact rule that decided each region, in the output
  test      exercise one rule in isolation, without constructing a whole pipeline
  audit     find rules that overlap or contradict before they ship

This deliberately does NOT reduce the rule count. The count is not the problem —
`INSIGHTS.md` records that most of these were forced by measurement, and deleting
them would just restore the bugs. The problem was that they were invisible. Two
deeper abstractions would genuinely collapse some of them (z-order layers, and
region-vs-semantic-extent; see TODO.md), and both need more evidence than three
screens. When they land, they land as rules here.

Every rule carries `why` and, where one exists, `evidence` — the measurement that
forced it. A rule nobody can justify is a rule nobody can safely delete.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Rule:
    """One classification decision.

    `when(ctx)` returns True if the rule applies; `then` is the class it assigns.
    Lower `priority` runs first. `why` becomes the region's stated reason, so the
    output explains itself.
    """

    name: str
    priority: int
    when: Callable
    then: str
    why: str
    evidence: str = ""
    tags: tuple = ()

    def applies(self, ctx):
        try:
            return bool(self.when(ctx))
        except Exception:
            return False


@dataclass
class Context:
    """Everything a rule may look at. Deliberately explicit: a rule that needs a
    new input has to add it here, which makes the dependency visible."""

    obs: dict = field(default_factory=dict)          # VLM observations
    measured: dict = field(default_factory=dict)     # computed pixel statistics
    reference: dict = field(default_factory=dict)    # descriptive statistics
    children: list = field(default_factory=list)     # ids of contained regions
    size_px: tuple = (0, 0)
    is_split_parent: bool = False
    atomic: str = "atom"
    role: str = "generic"

    def o(self, key, default=None):
        return self.obs.get(key, default)

    def m(self, key, default=None):
        return self.measured.get(key, default)

    @property
    def n_children(self):
        return len(self.children)


def evaluate(rules, ctx):
    """-> (class, why, rule_name, [names of rules that also matched])

    Returns the first match by priority AND the rules that would have matched
    later. Overlap is not an error — later rules are often deliberate fallbacks —
    but it is worth surfacing, because a rule that is always shadowed is dead and a
    rule that shadows unexpectedly is a bug.
    """
    matched = [r for r in sorted(rules, key=lambda r: r.priority) if r.applies(ctx)]
    if not matched:
        return None, "no rule matched", None, []
    winner = matched[0]
    return winner.then, winner.why, winner.name, [r.name for r in matched[1:]]


def audit(rules):
    """Static checks over the rule set, for the problems a chain hides."""
    problems = []
    seen_priority = {}
    for r in rules:
        if r.priority in seen_priority:
            problems.append(
                f"tied priority {r.priority}: {seen_priority[r.priority]} and "
                f"{r.name} — order between them is undefined")
        seen_priority[r.priority] = r.name
        if not r.why:
            problems.append(f"{r.name}: no stated reason")
        if r.then is None and r.name != "pictogram":
            problems.append(f"{r.name}: no target class and no resolver")
    names = [r.name for r in rules]
    for n in set(names):
        if names.count(n) > 1:
            problems.append(f"duplicate rule name: {n}")
    return problems


def describe(rules):
    """The rule set as a table — the thing an if/elif chain cannot produce."""
    out = []
    for r in sorted(rules, key=lambda r: r.priority):
        target = r.then if r.then is not None else "<computed>"
        tags = ",".join(r.tags)
        out.append(f"{r.priority:>3}  {r.name:<26} -> {target:<18} [{tags}] {r.why}")
        if r.evidence:
            out.append(f"     {'':<26}    evidence: {r.evidence}")
    return "\n".join(out)
