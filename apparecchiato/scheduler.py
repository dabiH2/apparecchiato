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


def _reserve_the_lens_for_handoffs(graph: TaskGraph) -> None:
    """Make every `place` wait for every `handoff`.

    The shared lens -- the only region both arms reach -- is about 200 x 90 mm,
    and a finished place setting is 190 mm wide. Measured on all ten evaluation
    seeds: once the fork, spoon and mug are in their slots, there is nowhere left
    in the lens to set an object down for a transfer, and the search for one
    falls back to a spot that is already occupied. The symptom was the taker's
    pad meeting a fork at -0.19 mm and stalling its descent 13 mm above a plate
    that had been set down perfectly.

    Nothing about the task requires the setting to be laid before the transfer,
    so the transfer goes first and the lens is empty when it needs to be. This
    is a resource constraint, and it belongs here rather than in a planner: it is
    true of every plan, including ones the VLM writes, and the scheduler is where
    the other two shared-workspace rules already live.
    """
    handoffs = [n.id for n in graph.nodes if n.skill == "handoff"]
    if not handoffs:
        return
    by_id = {n.id: n for n in graph.nodes}

    # Everything the hand-offs already wait on, transitively. Those must not be
    # made to wait on the hand-off in turn -- that is a deadlock, and it is the
    # one this produced on the first attempt: both arms ended up holding objects
    # whose places were waiting for a transfer that could no longer start.
    protected: set[str] = set(handoffs)
    stack = list(handoffs)
    while stack:
        cur = by_id.get(stack.pop())
        if cur is None:
            continue
        for d in cur.deps:
            if d not in protected:
                protected.add(d)
                stack.append(d)

    # Gate the PICKS, not just the places. Gating places alone leaves both arms
    # holding something with nowhere to put it.
    for n in graph.nodes:
        if n.id in protected or n.skill not in ("pick", "place"):
            continue
        for h in handoffs:
            if h not in n.deps:
                n.deps.append(h)


def _pick_is_premature(node: Node, graph: TaskGraph, done: set[str]) -> bool:
    """Would this pick leave an arm holding something it cannot yet put down?

    Do not pick up what you cannot put down. A grasp is not a shelf: an object
    held across other steps slides. Measured on seed 57, arm A picked the spoon,
    then parked holding it while the other arm did three plate steps, and the
    spoon crept 10 mm through the pads in the first of those and 63 mm by the
    time it was placed -- a failed place for a grasp that had been perfect.

    Holding got longer when the scheduler started reserving the shared lens for
    hand-offs, so this is the other half of that change rather than a separate
    idea. A pick waits until everything its own place needs -- other than the
    pick/hand-off/place chain for that same object -- is finished.
    """
    obj = node.args.get("object")
    if not obj:
        return False
    by_id = {n.id: n for n in graph.nodes}
    chain = {n.id for n in graph.nodes if n.args.get("object") == obj}
    places = [n for n in graph.nodes
              if n.skill == "place" and n.args.get("object") == obj]
    if not places:
        return False
    stack = [d for p in places for d in p.deps]
    seen: set[str] = set()
    while stack:
        d = stack.pop()
        if d in seen or d in chain or d in done:
            continue
        seen.add(d)
        dep = by_id.get(d)
        if dep is None:
            continue
        return True
    return False


def schedule(graph: TaskGraph, scene: SceneSpec) -> list[Step]:
    """Assign arms and an execution order. Raises SchedulingError if infeasible."""
    from .sim.layout import drawer_content_pos
    _reserve_the_lens_for_handoffs(graph)
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
        # Put down what is already in a hand before starting anything else.
        # Topological order alone let arm A pick the spoon and then wait three
        # steps for its turn: the spoon crept 10 mm through the pads in the
        # first of those and 63 mm by the time it was placed -- a failed place
        # for a grasp that had been perfect. A gripper is not a shelf.
        def _frees_a_hand(n: Node) -> bool:
            obj = n.args.get("object")
            return (n.skill in ("place", "handoff") and obj is not None
                    and obj in holding.values())

        ready.sort(key=lambda n: (0 if _frees_a_hand(n) else 1, topo.index(n.id)))

        # Take the first ready node whose grasp preconditions can be met right
        # now. A node that needs a busy arm is not an error -- it just waits for
        # a later round, once a place or hand-off has freed that arm.
        # Two passes. The first skips picks that would leave an arm holding
        # something it cannot yet put down; the second drops that preference, so
        # a graph where every remaining node is such a pick still makes progress
        # instead of deadlocking.
        chosen = None
        blocked: list[str] = []
        for defer_premature_picks in (True, False):
            for node in ready:
                if (defer_premature_picks and node.skill == "pick"
                        and _pick_is_premature(node, graph, done)):
                    continue
                wps = node_waypoints(node, scene, world)
                try:
                    arms = _assign(node, wps, holding, last_arm_finish)
                except SchedulingError as exc:
                    if not defer_premature_picks:
                        blocked.append(f"{node.id}: {exc}")
                    continue
                chosen = (node, arms, wps)
                break
            if chosen is not None:
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
