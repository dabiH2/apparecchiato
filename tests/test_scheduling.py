"""The scheduler is where physical impossibility has to surface."""
import numpy as np
import pytest

from apparecchiato.tasks import TaskGraph, Node
from apparecchiato.planner import build_planner
from apparecchiato.scheduler import schedule, SchedulingError, parallel_fraction
from apparecchiato.sim.layout import sample_scene

INSTRUCTION = "Set up the dinner table and pour water into the mug."
SEEDS = list(range(10))


def _plan(seed):
    spec = sample_scene(seed)
    return build_planner("rules").plan(INSTRUCTION, spec), spec


@pytest.mark.parametrize("seed", SEEDS)
def test_every_seed_schedules(seed):
    g, spec = _plan(seed)
    steps = schedule(g, spec)
    assert len(steps) == len(g)


@pytest.mark.parametrize("seed", SEEDS)
def test_an_arm_never_holds_two_things(seed):
    g, spec = _plan(seed)
    holding = {"A": None, "B": None}
    for s in schedule(g, spec):
        n = s.node
        if n.skill == "pick":
            assert holding[s.arms[0]] is None, f"arm {s.arms[0]} already holds something"
            holding[s.arms[0]] = n.args["object"]
        elif n.skill == "place":
            assert holding[s.arms[0]] == n.args["object"]
            holding[s.arms[0]] = None
        elif n.skill == "handoff":
            giver, taker = s.arms
            assert holding[giver] == n.args["object"]
            assert holding[taker] is None
            holding[giver], holding[taker] = None, n.args["object"]
    assert holding == {"A": None, "B": None}, "an object was left in a gripper"


@pytest.mark.parametrize("seed", SEEDS)
def test_handoffs_happen_in_the_shared_workspace(seed):
    g, spec = _plan(seed)
    for s in schedule(g, spec):
        if s.node.skill == "handoff":
            assert s.uses_shared_zone


@pytest.mark.parametrize("seed", SEEDS)
def test_both_arms_are_used(seed):
    g, spec = _plan(seed)
    used = {a for s in schedule(g, spec) for a in s.arms}
    assert used == {"A", "B"}, "a bimanual task that only used one arm"


@pytest.mark.parametrize("seed", SEEDS)
def test_plans_expose_real_parallelism(seed):
    g, spec = _plan(seed)
    assert parallel_fraction(schedule(g, spec)) > 0.3


def test_two_shared_zone_steps_are_never_marked_concurrent():
    g, spec = _plan(0)
    steps = schedule(g, spec)
    by_id = {s.node.id: s for s in steps}
    for s in steps:
        for other in s.concurrent_with:
            assert not (s.uses_shared_zone and by_id[other].uses_shared_zone)


def test_placing_without_holding_is_rejected():
    spec = sample_scene(0)
    g = TaskGraph("x", [Node("n1", "place", {"object": "plate", "target": "plate"})])
    with pytest.raises(SchedulingError, match="no arm is holding"):
        schedule(g, spec)


def test_pinning_a_pick_to_an_arm_that_cannot_reach_is_rejected():
    spec = sample_scene(0)
    from apparecchiato.sim.layout import reaching_arms, pick_pos
    wrong = "B" if reaching_arms(pick_pos(spec, "plate")) == ["A"] else "A"
    g = TaskGraph("x", [Node("n1", "pick", {"object": "plate"}, arm=wrong)])
    with pytest.raises(SchedulingError):
        schedule(g, spec)
