"""Deterministic instruction parser.

This is not a toy stand-in for the VLM -- it is the safety net that makes the
demo reproducible, and it is what the VLM's output is validated against. It
understands the instruction grammar the challenge brief uses ("open the top
drawer, pick up the plate with arm A, place it on the table, pick up the mug
with arm B, pour water into the mug with arm A") plus the shorter forms people
actually say out loud when the command arrives via speech.

Crucially it does NOT hardcode arm assignments: it reads reachability from the
scene and inserts a hand-off whenever the picking arm cannot reach the slot.
That logic is shared with the VLM path, so both backends produce graphs that
respect the same physical constraints.
"""
from __future__ import annotations

import re

import numpy as np

from ..tasks import TaskGraph, Node
from .base import Planner, PlannerError

OBJECT_WORDS = {
    "plate": "plate", "dish": "plate",
    "mug": "mug", "cup": "mug",
    "bottle": "bottle", "jug": "bottle", "carafe": "bottle",
    "spoon": "spoon",
    "fork": "fork",
}

# Phrases that mean "do the whole thing", which is how most people will actually
# talk to it -- especially through the microphone.
FULL_TASK_PATTERNS = (
    r"\bset (?:up )?(?:the )?(?:dinner )?table\b",
    r"\blay (?:the )?table\b",
    r"\bmise en place\b",
    r"\bprepare (?:the )?(?:dinner )?(?:table|setting)\b",
    r"\bapparecchia\w*\b",        # apparecchia / apparecchiare / apparecchiato
)

POUR_PATTERNS = (r"\bpour\b", r"\bfill\b", r"\bversa\b")


class RulePlanner(Planner):
    name = "rules"

    def plan(self, instruction: str, scene, images=None) -> TaskGraph:
        text = (instruction or "").lower().strip()
        if not text:
            raise PlannerError("empty instruction")

        if any(re.search(p, text) for p in FULL_TASK_PATTERNS):
            nodes = self._full_setting(scene, pour=any(re.search(p, text) for p in POUR_PATTERNS)
                                       or "water" in text or "drink" in text)
            return TaskGraph(instruction, nodes, source=self.name,
                             notes=["expanded a whole-table command into the standard setting"])

        nodes = self._explicit(text, scene)
        if not nodes:
            raise PlannerError(
                f"could not parse {instruction!r}; no known object or action mentioned"
            )
        return TaskGraph(instruction, nodes, source=self.name)

    # -- helpers ------------------------------------------------------------

    def _mentioned(self, text: str) -> list[str]:
        found, seen = [], set()
        for m in re.finditer(r"[a-z]+", text):
            w = OBJECT_WORDS.get(m.group())
            if w and w not in seen:
                seen.add(w)
                found.append(w)
        return found

    def _route(self, obj: str, scene, nodes: list[Node], deps: list[str], idx: int) -> list[str]:
        """Emit pick -> (handoff) -> place for one object; return the new deps."""
        from ..sim.layout import reaching_arms, pick_pos

        start = pick_pos(scene, obj, assume_drawer_open=True)
        goal = np.asarray(scene.goals[obj], dtype=float)
        pick_arms = reaching_arms(start)
        place_arms = reaching_arms(goal)
        if not pick_arms:
            raise PlannerError(f"no arm can pick the {obj}")
        if not place_arms:
            raise PlannerError(f"no arm can reach the {obj} slot")

        direct = sorted(set(pick_arms) & set(place_arms))
        pid, hid, plid = f"pick_{obj}", f"hand_{obj}", f"place_{obj}"

        if direct:
            arm = direct[0]
            nodes.append(Node(pid, "pick", {"object": obj}, arm=arm, deps=list(deps),
                              rationale=f"arm {arm} reaches both the {obj} and its slot"))
            nodes.append(Node(plid, "place", {"object": obj, "target": obj}, arm=arm,
                              deps=[pid], rationale="same arm can finish the move"))
            return [plid]

        pa, qa = pick_arms[0], place_arms[0]
        nodes.append(Node(pid, "pick", {"object": obj}, arm=pa, deps=list(deps),
                          rationale=f"only arm {pa} can reach the {obj}"))
        nodes.append(Node(hid, "handoff", {"object": obj}, arm="any", deps=[pid],
                          rationale=f"slot is out of arm {pa}'s reach, pass to arm {qa}"))
        nodes.append(Node(plid, "place", {"object": obj, "target": obj}, arm=qa, deps=[hid],
                          rationale=f"arm {qa} reaches the slot"))
        return [plid]

    def _full_setting(self, scene, pour: bool) -> list[Node]:
        nodes: list[Node] = [Node("open", "open_drawer", {}, arm="A",
                                  rationale="cutlery is in the drawer")]
        tails: list[str] = []
        # Plate first (the cutlery is laid around it), then cutlery, then the mug.
        for i, obj in enumerate(("plate", "fork", "spoon", "mug")):
            deps = ["open"] if obj in ("fork", "spoon") else []
            if obj != "plate":
                deps = deps + [t for t in tails if t == "place_plate"]
            tails += self._route(obj, scene, nodes, deps, i)
        if pour:
            nodes.append(Node("pour_water", "pour", {"source": "bottle", "into": "mug"},
                              arm="any", deps=[t for t in tails if t.endswith("mug")] or tails,
                              rationale="one arm steadies the mug while the other pours"))
        nodes.append(Node("done", "home", {}, arm="any",
                          deps=[nodes[-1].id], rationale="clear the table for the diner"))
        return nodes

    def _explicit(self, text: str, scene) -> list[Node]:
        nodes: list[Node] = []
        deps: list[str] = []
        if re.search(r"\b(open|apri)\b.{0,20}\bdrawer|cassett", text):
            nodes.append(Node("open", "open_drawer", {}, arm="A",
                              rationale="explicitly requested"))
            deps = ["open"]
        for i, obj in enumerate(self._mentioned(text)):
            if obj == "bottle":
                continue
            if obj not in scene.goals:
                continue
            deps = self._route(obj, scene, nodes, deps if i == 0 else [], i)
        if any(re.search(p, text) for p in POUR_PATTERNS):
            nodes.append(Node("pour_water", "pour", {"source": "bottle", "into": "mug"},
                              arm="any", deps=deps, rationale="explicitly requested"))
        return nodes
