"""The simulator must agree with the geometry the planner reasons about.

This is the failure class that silently ruins manipulation demos: the MJCF says
the knob is at y=0.26, `drawer_knob_pos()` says 0.185, the IK aims at 0.185, the
gripper closes on air, and the only symptom is "the drawer didn't open". It cost
a day-one debugging session to find once. These tests assert the correspondence
directly, so it can never drift again silently.

They need MuJoCo; the rest of the suite does not.
"""
import numpy as np
import pytest

from apparecchiato.kinematics import fk, HOME_Q, JOINT_NAMES
from apparecchiato.sim.layout import (
    sample_scene, drawer_knob_pos, drawer_content_pos, to_world,
    DRAWER_OPEN_TRAVEL,
)
from apparecchiato.sim.scene import build_mjcf, DRAWER_FLOOR_TOP

mujoco = pytest.importorskip("mujoco")


def _model(seed=0):
    spec = sample_scene(seed)
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return spec, m, d


def _geom_pos(m, d, name):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"geom {name!r} missing"
    return np.array(d.geom_xpos[gid])


def _site_pos(m, d, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    assert sid >= 0, f"site {name!r} missing"
    return np.array(d.site_xpos[sid])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_drawer_knob_is_where_the_planner_aims(seed):
    """The bug that broke day one: knob geom 75 mm from its planned position."""
    spec, m, d = _model(seed)
    expected = drawer_knob_pos(spec)
    actual = _geom_pos(m, d, "drawer_knob")
    err = float(np.linalg.norm(actual - expected))
    assert err < 0.002, (
        f"drawer knob is {err * 1000:.0f} mm from where the IK aims "
        f"(sim {actual.round(3)} vs planned {expected.round(3)})"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_knob_tracks_the_drawer_when_opened(seed):
    """Pull the slide joint fully open; the knob must land on the planned spot."""
    spec, m, d = _model(seed)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    d.qpos[m.jnt_qposadr[jid]] = DRAWER_OPEN_TRAVEL
    mujoco.mj_forward(m, d)

    spec.drawer_open = DRAWER_OPEN_TRAVEL
    err = float(np.linalg.norm(_geom_pos(m, d, "drawer_knob") - drawer_knob_pos(spec)))
    assert err < 0.002, f"knob {err * 1000:.0f} mm off after opening"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cutlery_starts_inside_the_shut_drawer(seed):
    """Cutlery must begin on the drawer floor, not on the table in front of it."""
    spec, m, d = _model(seed)
    for name in ("spoon", "fork"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = np.array(d.xpos[bid])
        planned_shut = drawer_content_pos(spec, name)   # drawer_open is 0 here
        assert abs(pos[1] - planned_shut[1]) < 0.004, (
            f"{name} starts at y={pos[1]:.3f}, planner expects {planned_shut[1]:.3f}"
        )
        assert pos[2] >= DRAWER_FLOOR_TOP - 0.001, (
            f"{name} starts below the drawer floor (z={pos[2]:.3f})"
        )


@pytest.mark.parametrize("arm", ["A", "B"])
def test_python_fk_matches_the_simulator(arm):
    """The README's central claim, asserted rather than asserted-in-prose.

    The MJCF is generated from the same link constants kinematics.py uses, so
    the grasp site MuJoCo computes must coincide with fk() for any pose.
    """
    _spec, m, d = _model(0)
    rng = np.random.default_rng(0)
    prefix = {"A": "armA", "B": "armB"}[arm]

    for _ in range(12):
        q = HOME_Q + rng.uniform(-0.35, 0.35, size=5)
        for name, v in zip(JOINT_NAMES, q):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{name}")
            d.qpos[m.jnt_qposadr[jid]] = float(v)
        mujoco.mj_forward(m, d)

        sim = _site_pos(m, d, f"{prefix}_grasp")
        predicted = to_world(fk(q)[0], arm)
        err = float(np.linalg.norm(sim - predicted))
        assert err < 0.0015, (
            f"arm {arm}: FK and MuJoCo disagree by {err * 1000:.1f} mm "
            f"(sim {sim.round(4)} vs fk {predicted.round(4)}) -- the link "
            f"constants and the MJCF have drifted apart"
        )
