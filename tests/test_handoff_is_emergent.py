"""The hand-off is derived from geometry -- and here is exactly how far that goes.

This file exists because "the hand-off is emergent" is the easiest claim in the
project to make and the easiest to overstate, so it is pinned down here instead
of being asserted in a README.

What is true: nothing in the planner, the scheduler or the skills names an object
as the one that gets passed. The scheduler emits a hand-off exactly when the set
of arms that can pick an object and the set that can place it do not intersect.
Move an object so one arm can do the whole job and the hand-off disappears, with
no code change and no re-planning.

What is NOT true, and is worth knowing before anyone claims otherwise: across a
hundred seeds the answer always comes out the same -- the plate, from arm A to
arm B. That is not scripting, it is the task. The mug's slot has to sit in the
shared lens so both arms can work the pour, which means whichever arm picks the
mug can also place it; the cutlery is deliberately routed to one arm; and the
bottle is never put away. The plate is the only object left whose pick and place
can land in different workspaces, so it is the only one that can need passing.
What the seed varies is WHERE the transfer happens, not what gets transferred.
"""
import numpy as np
import pytest

from apparecchiato.planner import build_planner
from apparecchiato.scheduler import schedule
from apparecchiato.sim.layout import sample_scene, reaching_arms

INSTRUCTION = "Set the plate and the mug on the table."
SEEDS = list(range(6))


def _handoffs(spec):
    graph = build_planner("rules").plan(INSTRUCTION, spec)
    return [(s.node.args["object"], "".join(s.arms))
            for s in schedule(graph, spec) if s.node.skill == "handoff"]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_handoff_appears_exactly_when_pick_and_place_reach_do_not_intersect(seed):
    spec = sample_scene(seed)
    handed = {o for o, _ in _handoffs(spec)}
    for name in ("plate", "mug"):
        can_pick = set(reaching_arms(spec.by_name(name).grasp_point))
        can_place = set(reaching_arms(np.asarray(spec.goals[name])))
        needs_passing = not (can_pick & can_place)
        assert (name in handed) == needs_passing, (
            f"{name}: pick-reach {sorted(can_pick)}, place-reach {sorted(can_place)}, "
            f"handed={name in handed}")


@pytest.mark.parametrize("seed", SEEDS)
def test_move_the_plate_within_reach_of_its_own_slot_and_the_handoff_vanishes(seed):
    """The counterfactual that makes the claim falsifiable. Nothing else changes:
    same planner, same scheduler, same skills, same seed."""
    spec = sample_scene(seed)
    assert "plate" in {o for o, _ in _handoffs(spec)}, "premise of the seed changed"

    slot = np.asarray(spec.goals["plate"], dtype=float)
    spec.by_name("plate").pos = (float(slot[0]), float(slot[1]), float(slot[2]))
    assert set(reaching_arms(np.asarray(spec.by_name("plate").pos))) & \
        set(reaching_arms(slot))

    assert "plate" not in {o for o, _ in _handoffs(spec)}, (
        "a transfer was scheduled for an object one arm could carry the whole way")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_transfer_point_is_not_a_fixed_spot_on_the_table(seed):
    """It is sampled per seed and re-chosen at motion-build time against the live
    table, so two seeds do not hand over in the same place."""
    here = np.asarray(sample_scene(seed).handoff_point, dtype=float)
    others = [np.asarray(sample_scene(s).handoff_point, dtype=float)
              for s in SEEDS if s != seed]
    assert all(np.linalg.norm(here[:2] - o[:2]) > 0.002 for o in others)
