"""The task graph: the contract between the reasoning layer and the robots.

The planner (VLM or rule-based) emits a TaskGraph. The scheduler consumes it and
decides which arm runs which node when. Keeping this a plain, validated data
structure is what lets us swap the planner backend -- OpenVINO VLM, hosted LLM,
or deterministic parser -- without touching execution, and lets us unit-test
planning and scheduling with no simulator in the loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Literal

Arm = Literal["A", "B", "any"]

SKILLS = {
    # skill        required args
    "open_drawer": (),
    "pick":        ("object",),
    "place":       ("object", "target"),
    "handoff":     ("object",),
    "pour":        ("source", "into"),
    "home":        (),
}

# Skills that need both arms at once. The scheduler treats these specially: it
# reserves A and B together rather than letting one arm proceed alone.
BIMANUAL_SKILLS = {"handoff", "pour"}


@dataclass
class Node:
    id: str
    skill: str
    args: dict = field(default_factory=dict)
    arm: Arm = "any"
    deps: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self):
        if self.skill not in SKILLS:
            raise ValueError(f"unknown skill {self.skill!r}; known: {sorted(SKILLS)}")
        missing = [a for a in SKILLS[self.skill] if a not in self.args]
        if missing:
            raise ValueError(f"node {self.id}: skill {self.skill} missing args {missing}")
        if self.arm not in ("A", "B", "any"):
            raise ValueError(f"node {self.id}: bad arm {self.arm!r}")

    @property
    def bimanual(self) -> bool:
        return self.skill in BIMANUAL_SKILLS


@dataclass
class TaskGraph:
    instruction: str
    nodes: list[Node]
    source: str = "unknown"      # which planner backend produced this
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        ids = [n.id for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate node ids: {sorted(dupes)}")
        known = set(ids)
        for n in self.nodes:
            bad = [d for d in n.deps if d not in known]
            if bad:
                raise ValueError(f"node {n.id} depends on unknown node(s) {bad}")
        self.topological_order()   # raises on a cycle

    def __len__(self) -> int:
        return len(self.nodes)

    def by_id(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def topological_order(self) -> list[str]:
        indeg = {n.id: len(n.deps) for n in self.nodes}
        children: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for d in n.deps:
                children[d].append(n.id)
        ready = sorted(i for i, c in indeg.items() if c == 0)
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for c in sorted(children[nid]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(order) != len(self.nodes):
            stuck = sorted(set(indeg) - set(order))
            raise ValueError(f"task graph has a cycle involving {stuck}")
        return order

    def objects(self) -> set[str]:
        out = set()
        for n in self.nodes:
            for k in ("object", "source", "into"):
                if k in n.args:
                    out.add(n.args[k])
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"instruction": self.instruction, "source": self.source,
             "notes": self.notes, "nodes": [asdict(n) for n in self.nodes]},
            indent=indent,
        )

    @classmethod
    def from_json(cls, text: str | dict) -> "TaskGraph":
        d = json.loads(text) if isinstance(text, str) else text
        return cls(
            instruction=d.get("instruction", ""),
            nodes=[Node(**n) for n in d["nodes"]],
            source=d.get("source", "json"),
            notes=list(d.get("notes", [])),
        )
