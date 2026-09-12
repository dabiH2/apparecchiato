"""Planner interface and the prompt the VLM is held to.

The planner turns (instruction, scene observation) into a TaskGraph. Two
backends implement it:

  vlm    a multi-modal model, run through OpenVINO where available, that sees
         the rendered camera views and reasons about what to do.
  rules  a deterministic parser over the same instruction grammar.

`build_planner("vlm+rules")` composes them: the VLM plans, and if it returns
something unparseable, cyclic, or referencing objects that are not on the table,
the rule planner takes over and the fallback is recorded in the graph notes.
That is what stops a demo from dying live on stage because a 2B model produced
a stray token, without hiding the fact that it happened.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..tasks import TaskGraph


class PlannerError(RuntimeError):
    pass


SYSTEM_PROMPT = """\
You are the task planner for a two-armed robot setting a dinner table.

Arm A is on the left, arm B is on the right. Each arm can only reach part of the
table. You are given, for every object, which arms can reach it now and which
arms can reach where it must end up. If the arm that can pick an object cannot
reach its destination, you MUST route it through a hand-off in the shared zone.

Reply with JSON only. No prose, no markdown fence. Schema:

{"nodes": [
  {"id": "n1", "skill": "...", "args": {...}, "arm": "A"|"B"|"any",
   "deps": ["n0"], "rationale": "one short clause"}
]}

Skills and their required args:
  open_drawer  {}                          - pull the cutlery drawer open
  pick         {"object": name}            - grasp an object
  place        {"object": name, "target": slot}  - put it in a place-setting slot
  handoff      {"object": name}            - pass the held object to the other arm
  pour         {"source": name, "into": name}    - one arm steadies, one pours
  home         {}                          - return to the ready pose

Rules:
- Cutlery is inside the drawer: open_drawer must precede picking spoon or fork.
- Every place must be preceded by a pick of the same object.
- A handoff sits between a pick and a place when the two need different arms.
- pour requires the destination to already be placed and both arms free.
- deps must reference ids you have already defined. No cycles.
- An arm holds ONE object at a time. Between picking something and picking the
  next thing, that arm must place or hand off what it is holding.
- Keep each rationale to six words or fewer. A long answer gets cut off before
  it is finished, and a cut-off plan is no plan.

Worked example. Copy this shape exactly: every node has the same five keys, the
object and target values are bare names in quotes, and the whole reply is one
JSON object with no text before or after it.

Observation: plate reachable by A, plate slot reachable by B.

{"nodes": [
  {"id": "n0", "skill": "pick", "args": {"object": "plate"}, "arm": "A",
   "deps": [], "rationale": "only arm A reaches the plate"},
  {"id": "n1", "skill": "handoff", "args": {"object": "plate"}, "arm": "any",
   "deps": ["n0"], "rationale": "the slot is out of arm A's reach"},
  {"id": "n2", "skill": "place", "args": {"object": "plate", "target": "plate"},
   "arm": "B", "deps": ["n1"], "rationale": "arm B reaches the slot"}
]}
"""


def observation_text(scene, world_state: dict | None = None) -> str:
    """The textual half of the VLM's observation: reachability, not coordinates.

    We deliberately hand the model *affordances* ("arm A can reach the plate")
    rather than raw XYZ. Small VLMs reason far better over relational facts than
    over float triples, and it keeps the planner honest: it cannot fake success
    by echoing numbers it was given.
    """
    from ..sim.layout import reaching_arms, pick_pos
    import numpy as np

    lines = ["Objects on the table right now:"]
    for o in scene.objects:
        pos = pick_pos(scene, o.name, assume_drawer_open=True)
        arms = reaching_arms(pos) or ["none"]
        where = " (in the drawer, reachable once it is open)" if o.inside_drawer else ""
        lines.append(f"  - {o.name}{where}: reachable by {'+'.join(arms)}")
    lines.append("Place-setting slots:")
    for name, g in scene.goals.items():
        arms = reaching_arms(np.asarray(g)) or ["none"]
        lines.append(f"  - slot '{name}': reachable by {'+'.join(arms)}")
    lines.append(f"Drawer: {'open' if scene.drawer_open > 0 else 'closed'}.")
    return "\n".join(lines)


class Planner(ABC):
    name = "planner"

    @abstractmethod
    def plan(self, instruction: str, scene, images=None) -> TaskGraph:
        ...


class FallbackPlanner(Planner):
    """Try each planner in order; the first valid TaskGraph wins."""

    def __init__(self, planners: list[Planner]):
        if not planners:
            raise ValueError("FallbackPlanner needs at least one planner")
        self.planners = planners
        self.name = "+".join(p.name for p in planners)

    def plan(self, instruction: str, scene, images=None) -> TaskGraph:
        notes: list[str] = []
        for p in self.planners:
            try:
                g = p.plan(instruction, scene, images=images)
                g.notes = notes + g.notes
                return g
            except Exception as exc:                     # noqa: BLE001 - report and continue
                notes.append(f"{p.name} failed: {type(exc).__name__}: {exc}")
        raise PlannerError("every planner backend failed:\n  " + "\n  ".join(notes))


def build_planner(spec: str = "vlm+rules", **kw) -> Planner:
    """`spec` is a '+'-separated chain, e.g. 'vlm+rules', 'rules', 'vlm'."""
    from .rules import RulePlanner
    from .vlm import VLMPlanner

    made = []
    for part in spec.split("+"):
        part = part.strip().lower()
        if part == "rules":
            made.append(RulePlanner())
        elif part == "vlm":
            made.append(VLMPlanner(**kw))
        else:
            raise ValueError(f"unknown planner backend {part!r}")
    return made[0] if len(made) == 1 else FallbackPlanner(made)
