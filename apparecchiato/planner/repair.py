"""Repair what is repairable in a model's plan, and refuse to invent the rest.

A validator that only ever says no is cheap to build and throws away real work.
Both failures this 2B planner actually commits are REFERENTIAL, not semantic --
the model knows what it wants to do and mis-writes the bookkeeping:

    node n3 depends on unknown node(s) ['n2']      a dep pointing at nothing
    unknown skill ''                               a node whose skill came out empty

Neither is a wrong plan. The first is a dangling pointer; the second is a node
that carries no instruction at all. Rejecting a whole episode's plan over either
is the software equivalent of failing a compile on a stray semicolon.

So this module repairs exactly those two things, and nothing else. The boundary
it holds is the one that makes the repair honest rather than a ghostwriter:

    IT MAY   drop a reference to a node that does not exist
    IT MAY   drop a node that names no skill, or is missing a required argument
    IT MAY   add an ordering edge between two nodes THE MODEL ALREADY WROTE

    IT MAY NOT  create a node
    IT MAY NOT  change a node's skill
    IT MAY NOT  change, add or remove an argument
    IT MAY NOT  change which arm a node was assigned

`assert_nothing_invented` enforces that on the way out, and every repair is
recorded as a note that ends up in the episode report. A judge can therefore
read, per seed, exactly how much of the executed plan was the model's.

Everything downstream is unchanged: a repaired plan still has to survive
`validate_against_scene` and the scheduler. The repair buys the model a hearing,
not a pass.
"""
from __future__ import annotations

from ..tasks import SKILLS, Node


def sanitise_nodes(raw: list) -> tuple[list[dict], list[str]]:
    """Drop node dicts that cannot become a Node at all. Returns (kept, notes)."""
    kept: list[dict] = []
    dropped_skill: list[str] = []
    dropped_args: list[str] = []

    for i, n in enumerate(raw):
        if not isinstance(n, dict):
            dropped_skill.append(f"#{i}")
            continue
        skill = str(n.get("skill", "")).strip().lower()
        nid = str(n.get("id") or f"n{i}")
        if skill not in SKILLS:
            # An empty or unknown skill is not a wrong instruction, it is the
            # absence of one. There is nothing to execute and nothing to guess.
            dropped_skill.append(f"{nid}({skill or 'empty'})")
            continue
        args = {str(k).strip(): v for k, v in (n.get("args") or {}).items()}
        missing = [a for a in SKILLS[skill] if a not in args or str(args[a]).strip() == ""]
        if missing:
            dropped_args.append(f"{nid}({skill} missing {','.join(missing)})")
            continue
        kept.append(n)

    notes: list[str] = []
    if dropped_skill:
        notes.append(f"repair: dropped {len(dropped_skill)} node(s) naming no known "
                     f"skill [{', '.join(dropped_skill)}]")
    if dropped_args:
        notes.append(f"repair: dropped {len(dropped_args)} node(s) missing a required "
                     f"argument [{', '.join(dropped_args)}]")
    return kept, notes


def assert_repair_was_a_minority(kept: int, dropped: int) -> None:
    """Refuse to rescue an answer that was mostly noise.

    This guard exists because the first version of this module did something
    indefensible and the measurement caught it. Fed a degenerate generation --
    the 4-bit model looping a token until it hit the limit -- salvage recovered
    the single well-formed node at the start, dep-pruning removed its references
    to the forty nodes that never materialised, and the result was a valid,
    "schedulable" one-step plan: open the drawer, then stop. The scheduler
    accepted it. An episode would have run it and failed the task.

    That is worse than a rejection, because a rejection hands the episode to the
    deterministic planner and a one-step plan does not. So: repair is for a plan
    with mistakes in it, not for wreckage with a plan in it. If more of the
    answer was discarded than kept, there was no plan to repair.
    """
    if dropped > kept:
        raise ValueError(
            f"repair would have discarded more of the answer than it kept "
            f"({dropped} dropped vs {kept} kept) -- this is a degenerate "
            f"generation, not a plan with mistakes in it")


def prune_dangling_deps(nodes: list[Node]) -> list[str]:
    """Remove deps that point at node ids which do not exist. Mutates `nodes`.

    A dependency on a node the model never emitted constrains nothing -- it is a
    pointer into empty space. Dropping it can only widen the set of legal
    orderings, and every ordering the scheduler then picks still has to satisfy
    the grasp-state and workspace preconditions, so this cannot sneak an illegal
    plan through.
    """
    known = {n.id for n in nodes}
    pruned: list[str] = []
    for n in nodes:
        bad = [d for d in n.deps if d not in known]
        if bad:
            n.deps = [d for d in n.deps if d in known]
            pruned.append(f"{n.id}->{'/'.join(bad)}")
    return ([f"repair: dropped {len(pruned)} dependency reference(s) to nodes that "
             f"do not exist [{', '.join(pruned)}]"] if pruned else [])


def break_cycles(nodes: list[Node]) -> list[str]:
    """Remove the minimum back-edges needed to make the graph acyclic.

    Kept deliberately dumb: walk the nodes in emission order and drop any dep
    that points forward. The model writes its plan in the order it intends it to
    happen, so emission order is its own answer to "what comes first" -- this
    keeps that answer and discards only the edges that contradict it.
    """
    pos = {n.id: i for i, n in enumerate(nodes)}
    dropped: list[str] = []
    for n in nodes:
        bad = [d for d in n.deps if pos.get(d, -1) > pos[n.id]]
        if bad:
            n.deps = [d for d in n.deps if pos.get(d, -1) <= pos[n.id]]
            dropped.append(f"{n.id}->{'/'.join(bad)}")
    return ([f"repair: dropped {len(dropped)} dependency edge(s) that pointed "
             f"forward and would have formed a cycle [{', '.join(dropped)}]"]
            if dropped else [])


def _fingerprint(nodes: list[Node]) -> list[tuple]:
    return sorted((n.skill, tuple(sorted((k, str(v)) for k, v in n.args.items())), n.arm)
                  for n in nodes)


def assert_nothing_invented(before: list[Node], after: list[Node]) -> None:
    """The repair may only remove instructions and reorder them; never add one.

    This is the claim the whole repair layer rests on, so it is checked rather
    than asserted in prose. If it ever fires, the repair has started
    ghostwriting and the plan is no longer the model's.
    """
    b, a = _fingerprint(before), _fingerprint(after)
    invented = [x for x in a if a.count(x) > b.count(x)]
    if invented:
        raise AssertionError(
            f"plan repair invented {len(invented)} instruction(s) that the model "
            f"never emitted: {invented[:3]}")
