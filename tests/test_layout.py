"""Every sampled seed must be physically solvable before it reaches the robot."""
import numpy as np
import pytest

from apparecchiato.sim.layout import (
    sample_scene, validate_scene, reaching_arms, in_shared_workspace,
    to_arm_frame, to_world, drawer_knob_pos, DRAWER_OPEN_TRAVEL, pick_pos,
)


@pytest.mark.parametrize("seed", range(12))
def test_sampled_scenes_are_feasible(seed):
    validate_scene(sample_scene(seed))


@pytest.mark.parametrize("seed", range(8))
def test_every_object_and_slot_is_reachable(seed):
    spec = sample_scene(seed)
    for o in spec.objects:
        assert reaching_arms(pick_pos(spec, o.name)), f"{o.name} unreachable"
    for name, g in spec.goals.items():
        assert reaching_arms(np.asarray(g)), f"slot {name} unreachable"


@pytest.mark.parametrize("seed", range(8))
def test_handoff_point_is_shared_and_mug_slot_allows_pouring(seed):
    spec = sample_scene(seed)
    assert in_shared_workspace(np.asarray(spec.handoff_point))
    # Pouring needs one arm steadying the mug and the other tipping over it.
    assert in_shared_workspace(np.asarray(spec.goals["mug"]))


def test_arm_frames_round_trip():
    p = np.array([0.03, 0.17, 0.02])
    for arm in ("A", "B"):
        assert np.allclose(to_world(to_arm_frame(p, arm), arm), p)


def test_drawer_knob_travels_with_the_drawer():
    spec = sample_scene(0)
    shut = drawer_knob_pos(spec)
    spec.drawer_open = DRAWER_OPEN_TRAVEL
    opened = drawer_knob_pos(spec)
    assert np.isclose(shut[1] - opened[1], DRAWER_OPEN_TRAVEL)


def test_scenes_are_randomised_but_deterministic():
    a, b = sample_scene(1), sample_scene(1)
    assert np.allclose(a.by_name("plate").pos, b.by_name("plate").pos)
    c = sample_scene(2)
    assert not np.allclose(a.by_name("plate").pos, c.by_name("plate").pos)


@pytest.mark.parametrize("seed", range(6))
def test_randomisation_actually_varies_physics(seed):
    spec = sample_scene(seed)
    masses = {o.name: o.mass for o in spec.objects}
    frictions = {o.name: o.friction for o in spec.objects}
    assert len(set(np.round(list(masses.values()), 3))) > 1
    assert len(set(np.round(list(frictions.values()), 3))) > 1
