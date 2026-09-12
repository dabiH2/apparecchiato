"""Two-arm scheduler: turns a TaskGraph into a collision-aware execution order.

Three things have to hold at once, and this is where they are enforced rather
than hoped for:

1. Dependencies. A node runs only after every node it depends on has finished.
2. Reachability. A node is assigned to an arm that can actually reach every
   waypoint it touches -- which is what forces hand-offs when an object starts
   in one arm's workspace and belongs in the other's.
3. Shared-workspace safety. The two annuli overlap in a lens. Two arms may work
   concurrently only while at most one of them is inside that lens; a hand-off
   is the one exception, because both arms are meant to be there and the skill
   itself sequences the grasp/release.

The scheduler is deliberately greedy and deterministic: given the same graph and
scene it always produces the same plan, which is what makes a 10-seed evaluation
interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tasks import TaskGraph, Node
from .sim.layout import SceneSpec, reaching_arms, in_shared_workspace, drawer_knob_pos


@dataclass
class Step:
    """One scheduled unit of work."""
    order: int
    node: Node
    arms: tuple[str, ...]
    concurrent_with: list[str] = field(default_factory=list)
    uses_shared_zone: bool = False
    waypoints: list = field(default_factory=list)

    @property
    def label(self) -> str:
        a = "+".join(self.arms)
        args = " ".join(f"{k}={v}" for k, v in self.node.args.items())
        return f"[{a}] {self.node.skill}({args})".strip()


class SchedulingError(RuntimeError):
    pass


def node_waypoints(node: Node, scene: SceneSpec, world: dict) -> list[np.ndarray]:
    """Every table point this node needs to touch, in world coordinates.

    `world` maps object name -> current position, so the scheduler reasons about
    where things are *now*, not where they started.
    """
    a = node.args
    if node.skill == "open_drawer":
        shut = drawer_knob_pos(scene)
        opened = shut.copy()
        opened[1] -= _drawer_travel(scene)
        return [shut, opened]
    if node.skill == "pick":
        return [world[a["object"]]]
    if node.skill == "place":
        return [world[a["object"]], np.asarray(scene.goals[a["target"]], dtype=float)
                if a["target"] in scene.goals else world[a["target"]]]
    if node.skill == "handoff":
        # The rendezvous, not where the object was picked up: the holding arm
        # carries it into the shared lens and releases it there.
        return [np.asarray(scene.handoff_point, dtype=float)]
    if node.skill == "pour":
        return [world[a["source"]], world[a["into"]]]
    return []


def _drawer_travel(scene: SceneSpec) -> float:
    from .sim.layout import DRAWER_OPEN_TRAVEL
    return DRAWER_OPEN_TRAVEL


def _feasible_arms(node: Node, wps: list[np.ndarray]) -> list[str]:
    if not wps:
        return ["A", "B"]
    sets = [set(reaching_arms(w)) for w in wps]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def schedule(graph: TaskGraph, scene: SceneSpec) -> list[Step]:
    """Assign arms and an execution order. Raises SchedulingError if infeasible."""
    from .sim.layout import drawer_content_pos
    world = {o.name: (drawer_content_pos(scene, o.name) if o.inside_drawer
                      else o.grasp_point.copy()) for o in scene.objects}

    remaining = {n.id: n for n in graph.nodes}
    done: set[str] = set()
    steps: list[Step] = []
    order = 0
    last_arm_finish = {"A": 0, "B": 0}
    holding: dict[str, str | None] = {"A": None, "B": None}   # arm -> object it grips
    topo = graph.topological_order()

    while remaining:
        ready = [n for n in remaining.values() if all(d in done for d in n.deps)]
        if not ready:
            raise SchedulingError(
                f"deadlock: {sorted(remaining)} are all waiting on unfinished dependencies"
            )
        ready.sort(key=lambda n: topo.index(n.id))

        # Take the first ready node whose grasp preconditions can be met right
        # now. A node that needs a busy arm is not an error -- it just waits for
        # a later round, once a place or hand-off has freed that arm.
        chosen = None
        blocked: list[str] = []
        for node in ready:
            wps = node_waypoints(node, scene, world)
            try:
                arms = _assign(node, wps, holding, last_arm_finish)
            except SchedulingError as exc:
                blocked.append(f"{node.id}: {exc}")
                continue
            chosen = (node, arms, wps)
            break

        if chosen is None:
            raise SchedulingError(
                "no ready node can be executed with the current grasp state "
                f"(A holds {holding['A']}, B holds {holding['B']}):\n  - "
                + "\n  - ".join(blocked)
            )

        node, arms, wps = chosen
        shared = any(in_shared_workspace(w) for w in wps)
        order += 1
        for a in arms:
            last_arm_finish[a] = order
        steps.append(Step(order=order, node=node, arms=arms, uses_shared_zone=shared,
                          waypoints=wps))

        _apply_grasp(node, arms, holding)
        _apply_effects(node, scene, world)
        done.add(node.id)
        del remaining[node.id]

    _annotate_concurrency(steps, graph)
    return steps


def _assign(node: Node, wps, holding: dict, last_arm_finish: dict) -> tuple[str, ...]:
    """Pick the arm(s) for a node, or raise if its preconditions are unmet.

    This is where the physical facts live: an arm holds at most one object, you
    cannot place what you are not holding, and a hand-off needs one full hand and
    one empty one.
    """
    obj = node.args.get("object")

    if node.skill == "handoff":
        givers = [a for a in ("A", "B") if holding[a] == obj]
        if not givers:
            raise SchedulingError(f"nothing to hand off -- no arm is holding the {obj}")
        giver = givers[0]
        taker = "B" if giver == "A" else "A"
        if holding[taker] is not None:
            raise SchedulingError(
                f"arm {taker} is holding the {holding[taker]} and cannot receive the {obj}")
        return (giver, taker)

    if node.skill == "pour":
        busy = {a: h for a, h in holding.items() if h is not None}
        if busy:
            raise SchedulingError(
                "pour needs both arms free, but "
                + ", ".join(f"{a} holds the {h}" for a, h in busy.items()))
        # wps is [source, destination]. One arm steadies the destination, the
        # other lifts and tips the source -- so the roles are decided by who can
        # reach what, not by a fixed A-steadies convention.
        src, dst = wps[0], wps[1]
        can_src = set(reaching_arms(src))
        can_dst = set(reaching_arms(dst))
        # The pouring arm must reach the source AND hover over the destination;
        # the steadying arm only needs the destination.
        for steady, pourer in (("A", "B"), ("B", "A")):
            if steady in can_dst and pourer in can_src and pourer in can_dst:
                return (steady, pourer)
        raise SchedulingError(
            f"cannot split the pour: {sorted(can_dst) or 'no arm'} can reach the "
            f"destination and {sorted(can_src) or 'no arm'} can reach the source"
        )

    if node.skill == "place":
        holders = [a for a in ("A", "B") if holding[a] == obj]
        if not holders:
            raise SchedulingError(f"cannot place the {obj}: no arm is holding it")
        arm = holders[0]
        if node.arm != "any" and node.arm != arm:
            raise SchedulingError(
                f"plan wants arm {node.arm} to place the {obj}, but arm {arm} is holding it")
        if arm not in _feasible_arms(node, wps):
            raise SchedulingError(
                f"arm {arm} holds the {obj} but cannot reach its slot -- a hand-off is needed")
        return (arm,)

    # pick / open_drawer / home: need a free arm that can reach the waypoints.
    cands = _feasible_arms(node, wps)
    if node.arm != "any":
        if node.arm not in cands:
            raise SchedulingError(
                f"pinned to arm {node.arm}, but only {cands or 'neither arm'} can reach")
        cands = [node.arm]
    if not cands:
        raise SchedulingError(
            f"no arm can reach {[np.round(w, 3).tolist() for w in wps]}")
    if node.skill in ("pick", "open_drawer"):
        free = [a for a in cands if holding[a] is None]
        if not free:
            raise SchedulingError(
                f"every candidate arm is busy ({', '.join(f'{a} holds the {holding[a]}' for a in cands)})")
        cands = free
    return (min(cands, key=lambda a: last_arm_finish[a]),)


def _apply_grasp(node: Node, arms: tuple[str, ...], holding: dict) -> None:
    if node.skill == "pick":
        holding[arms[0]] = node.args["object"]
    elif node.skill == "place":
        holding[arms[0]] = None
    elif node.skill == "handoff":
        giver, taker = arms
        holding[giver] = None
        holding[taker] = node.args["object"]


def _apply_effects(node: Node, scene: SceneSpec, world: dict) -> None:
    """Update the believed world state so later nodes plan against it."""
    from .sim.layout import DRAWER_OPEN_TRAVEL, drawer_content_pos
    if node.skill == "open_drawer":
        scene.drawer_open = DRAWER_OPEN_TRAVEL
        for o in scene.objects:
            if o.inside_drawer:
                world[o.name] = drawer_content_pos(scene, o.name)
    elif node.skill == "handoff":
        world[node.args["object"]] = np.asarray(scene.handoff_point, dtype=float)
    elif node.skill == "place":
        tgt = node.args["target"]
        if tgt in scene.goals:
            world[node.args["object"]] = np.asarray(scene.goals[tgt], dtype=float)


def _annotate_concurrency(steps: list[Step], graph: TaskGraph) -> None:
    """Mark which steps could legally run at the same time.

    Two steps may overlap when they share no arm, neither depends on the other,
    and they are not both inside the shared lens.
    """
    def ancestors(nid: str, cache: dict) -> set[str]:
        if nid in cache:
            return cache[nid]
        out: set[str] = set()
        for d in graph.by_id(nid).deps:
            out.add(d)
            out |= ancestors(d, cache)
        cache[nid] = out
        return out

    cache: dict[str, set[str]] = {}
    for i, s in enumerate(steps):
        for t in steps[i + 1:]:
            if set(s.arms) & set(t.arms):
                continue
            if s.node.id in ancestors(t.node.id, cache) or t.node.id in ancestors(s.node.id, cache):
                continue
            if s.uses_shared_zone and t.uses_shared_zone:
                continue
            s.concurrent_with.append(t.node.id)
            t.concurrent_with.append(s.node.id)


def describe(steps: list[Step]) -> str:
    lines = []
    for s in steps:
        tag = "  [shared zone]" if s.uses_shared_zone else ""
        par = f"  || {','.join(s.concurrent_with)}" if s.concurrent_with else ""
        lines.append(f"{s.order:2d}. {s.label}{tag}{par}")
    return "\n".join(lines)


def parallel_fraction(steps: list[Step]) -> float:
    """Share of steps that can overlap with another -- a dual-arm utilisation proxy."""
    if not steps:
        return 0.0
    return sum(1 for s in steps if s.concurrent_with) / len(steps)
