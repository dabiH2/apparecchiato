"""FK/IK must agree exactly: a drift here is silently mis-aimed grasps."""
import math
import numpy as np
import pytest

from apparecchiato.kinematics import (
    fk, solve_ik, ee_pitch, JOINT_LIMITS, HOME_Q, TOP_DOWN, interpolate,
    workspace_bounds,
)


def test_home_pose_is_above_the_table():
    p, _ = fk(HOME_Q)
    assert p[2] > 0.02
    assert 0.0 < ee_pitch(HOME_Q) < math.pi / 2


@pytest.mark.parametrize("seed", range(5))
def test_ik_round_trip_is_exact(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(400):
        q = np.array([rng.uniform(*JOINT_LIMITS[i]) for i in range(5)])
        p, _ = fk(q)
        res = solve_ik(p, pitch=ee_pitch(q), roll=q[4], yaw=q[0])
        if not res.success:
            continue
        p2, _ = fk(res.q)
        assert np.linalg.norm(p2 - p) < 1e-9
        checked += 1
    assert checked > 100, "IK solved too few poses to be a meaningful test"


def test_ik_respects_joint_limits():
    rng = np.random.default_rng(0)
    lo, hi = workspace_bounds(TOP_DOWN, 0.04)
    for _ in range(300):
        r = rng.uniform(max(lo, 0.10), hi)
        th = rng.uniform(-0.9, 0.9)
        t = np.array([r * math.cos(th), r * math.sin(th), 0.04])
        res = solve_ik(t, pitch=TOP_DOWN)
        if res.success:
            assert np.all(res.q >= JOINT_LIMITS[:, 0] - 1e-9)
            assert np.all(res.q <= JOINT_LIMITS[:, 1] + 1e-9)


def test_unreachable_target_fails_cleanly():
    res = solve_ik(np.array([2.0, 0.0, 0.0]), pitch=TOP_DOWN)
    assert not res.success and "workspace" in res.reason


def test_wrist_roll_does_not_move_the_grasp_point():
    """Roll spins the jaws about the tool axis; the grasp point must not move."""
    base = np.array([0.2, 0.1, 0.6, 0.5, 0.0])
    p0, _ = fk(base)
    for roll in (-2.0, -0.5, 0.7, 2.5):
        q = base.copy()
        q[4] = roll
        p, _ = fk(q)
        assert np.linalg.norm(p - p0) < 1e-12


def test_interpolation_starts_and_ends_where_asked():
    a = np.zeros(5)
    b = np.array([0.4, -0.3, 0.7, 0.2, 1.0])
    path = interpolate(a, b, 25)
    assert np.allclose(path[0], a) and np.allclose(path[-1], b)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert steps.max() < 2.5 * steps.mean(), "easing should avoid a large jump"
