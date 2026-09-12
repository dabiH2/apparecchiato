"""Every waypoint of every skill must be IK-solvable before we ever step physics."""
import numpy as np
import pytest

from apparecchiato.planner import build_planner
from apparecchiato.scheduler import schedule
from apparecchiato.skills import build_motion, GRIPPER_OPEN, GRIPPER_CLOSED, grip_for
from apparecchiato.kinematics import HOME_Q, fk, JOINT_LIMITS
from apparecchiato.sim.layout import (
    sample_scene, drawer_content_pos, DRAWER_OPEN_TRAVEL,
)

INSTRUCTION = "Set up the dinner table and pour water into the mug."


def _walk(seed):
    """Build every motion for a seed, advancing the believed world as we go."""
    spec = sample_scene(seed)
    graph = build_planner("rules").plan(INSTRUCTION, spec)
    steps = schedule(graph, spec)

    ex = sample_scene(seed)
    world = {o.name: (drawer_content_pos(ex, o.name) if o.inside_drawer
                      else o.grasp_point.copy()) for o in ex.objects}
    motions = []
    for st in steps:
        motions.append((st, build_motion(st, ex, world)))
        if st.node.skill == "open_drawer":
            ex.drawer_open = DRAWER_OPEN_TRAVEL
            for o in ex.objects:
                if o.inside_drawer:
                    world[o.name] = drawer_content_pos(ex, o.name)
        elif st.node.skill == "handoff":
            world[st.node.args["object"]] = np.asarray(ex.handoff_point)
        elif st.node.skill == "place":
            world[st.node.args["object"]] = np.asarray(ex.goals[st.node.args["target"]])
    return motions


@pytest.mark.parametrize("seed", range(10))
def test_all_motions_build_and_respect_joint_limits(seed):
    for st, m in _walk(seed):
        assert len(m) > 0, f"{st.label} produced no waypoints"
        for w in m.waypoints:
            assert np.all(w.q >= JOINT_LIMITS[:, 0] - 1e-9)
            assert np.all(w.q <= JOINT_LIMITS[:, 1] + 1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_trajectories_never_drive_through_the_table(seed):
    start = {"A": HOME_Q.copy(), "B": HOME_Q.copy()}
    for st, m in _walk(seed):
        for arm, q, _g, label in m.trajectory(start):
            z = fk(q)[0][2]
            assert z > -0.005, f"{st.label} / {label} went below the table (z={z:.3f})"
            start[arm] = q


@pytest.mark.parametrize("seed", range(6))
def test_grasps_close_before_lifting(seed):
    """Approach open, close onto the object, stay closed through the retreat.

    The closed value is object-specific -- jaws are commanded a few mm inside the
    object's real width rather than shut to nothing, because crushing straight
    through a thin object is how a grasp silently fails. So this asserts the
    behaviour, not one magic number.
    """
    for st, m in _walk(seed):
        if st.node.skill != "pick":
            continue
        obj = st.node.args["object"]
        want = grip_for(sample_scene(seed).by_name(obj).kind)
        grips = [w.gripper for w in m.waypoints]
        assert grips[0] == GRIPPER_OPEN, "approached with a closed gripper"
        assert want in grips, f"never closed onto the {obj}"
        assert grips[-1] == want, "let go while retreating"
        assert GRIPPER_CLOSED < want < GRIPPER_OPEN, (
            f"{obj}: jaw target {want * 1000:.0f} mm should sit between fully "
            "closed and fully open")


@pytest.mark.parametrize("seed", range(6))
def test_handoff_is_table_mediated_and_never_drops_the_object(seed):
    """The object is always supported, and the arms are never in the lens together.

    An in-air exchange self-collides on this arm (both bases at the near edge,
    both grasping top-down), so the transfer goes through the table: the giver
    sets the object down and retreats all the way home BEFORE the taker moves in.
    "Supported" therefore means resting on the table at release, not held by the
    other gripper.
    """
    for st, m in _walk(seed):
        if st.node.skill != "handoff":
            continue
        labels = [w.label for w in m.waypoints]
        order = {lab: i for i, lab in enumerate(labels)}
        for lab in ("handoff:set-down", "handoff:release", "handoff:giver-clear",
                    "handoff:taker-approach", "handoff:taker-grasp"):
            assert lab in order, f"missing {lab}"
        # Released only once it is down on the table.
        assert order["handoff:set-down"] < order["handoff:release"]
        # The giver is fully out of the way before the taker enters the lens.
        assert order["handoff:giver-clear"] < order["handoff:taker-approach"]
        assert order["handoff:taker-approach"] < order["handoff:taker-grasp"]
        # Nothing the giver does happens after the taker starts moving.
        givers = [i for i, w in enumerate(m.waypoints) if w.arm == st.arms[0]]
        takers = [i for i, w in enumerate(m.waypoints) if w.arm == st.arms[1]]
        assert max(givers) < min(takers), "arms overlap in the shared lens"


@pytest.mark.parametrize("seed", range(6))
def test_pour_uses_both_arms(seed):
    for st, m in _walk(seed):
        if st.node.skill == "pour":
            assert m.arms() == {"A", "B"}
