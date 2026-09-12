"""Forward kinematics and analytic IK for the SO-101 5-DoF arm.

Pure NumPy: importable and testable without MuJoCo, so the planner/scheduler/IK
test suite runs fast and in CI without a simulator.

SO-101 joint order (matches the MJCF written by apparecchiato/sim/scene.py):
    0 shoulder_pan   revolute about +Z   (base yaw)
    1 shoulder_lift  revolute about +Y
    2 elbow_flex     revolute about +Y
    3 wrist_flex     revolute about +Y
    4 wrist_roll     revolute about the tool axis (+Z of the wrist frame)
    (the gripper jaw is actuated separately and is not part of the IK chain)

Key structural fact that makes this tractable: joints 1-3 are all parallel (+Y),
and joint 4 spins the gripper about its own axis. So the grasp point depends
only on (q0, q1, q2, q3), and the tool direction depends only on
(q0, q1+q2+q3). That gives an exact closed-form IK for position + tool pitch,
with wrist_roll left free to orient the jaws. No numerical solver in the hot
loop, no singularities inside the workspace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Link geometry (metres), nominal SO-ARM101.
#
# These constants are ALSO written into the generated MJCF, so FK here and the
# simulator agree by construction. If you swap in the official SO-101 MJCF
# (apparecchiato/sim/scene.py --mjcf), run
#     python scripts/calibrate_kinematics.py --mjcf <path>
# to re-fit them against that model before trusting the IK.
# ---------------------------------------------------------------------------
L_BASE_Z = 0.0542      # arm mount -> shoulder_pan axis
L_SHOULDER_Z = 0.0280  # shoulder_pan -> shoulder_lift axis
L_UPPER = 0.1159       # shoulder_lift -> elbow_flex
L_FORE = 0.1350        # elbow_flex -> wrist_flex
L_WRIST = 0.0424       # wrist_flex -> wrist_roll
L_TOOL = 0.0810        # wrist_roll -> grasp point between the jaws

L_SHOULDER_H = L_BASE_Z + L_SHOULDER_Z   # shoulder height above the arm mount
L_HAND = L_WRIST + L_TOOL                # wrist_flex -> grasp point, along tool axis
MAX_REACH = L_UPPER + L_FORE             # shoulder -> wrist_flex, fully extended
MIN_REACH = abs(L_UPPER - L_FORE)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Conservative software limits (radians), slightly inside the hardware limits so
# that a solution valid here is also valid on a real SO-101.
JOINT_LIMITS = np.array([
    [-1.91, 1.91],   # shoulder_pan
    [-1.75, 1.75],   # shoulder_lift
    [-1.69, 1.69],   # elbow_flex
    [-1.79, 1.79],   # wrist_flex
    [-2.79, 2.79],   # wrist_roll
])

# "Ready" pose: tool angled down over the table, clear of the surface.
HOME_Q = np.array([0.0, 0.55, 0.95, 1.15, 0.0])

TOP_DOWN = 0.0                 # pitch convention: 0 rad = tool pointing straight down
HORIZONTAL = math.pi / 2.0     # pitch = 90 deg = tool pointing horizontally outward


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def fk(q) -> tuple[np.ndarray, np.ndarray]:
    """Grasp-point position [3] and tool rotation [3x3] in the arm base frame."""
    q = np.asarray(q, dtype=float).reshape(5)
    s1 = q[1]
    s2 = q[1] + q[2]
    s3 = q[1] + q[2] + q[3]

    # In the arm's vertical plane, measured from +Z.
    r = L_UPPER * math.sin(s1) + L_FORE * math.sin(s2) + L_HAND * math.sin(s3)
    z = L_SHOULDER_H + L_UPPER * math.cos(s1) + L_FORE * math.cos(s2) + L_HAND * math.cos(s3)

    c0, s0 = math.cos(q[0]), math.sin(q[0])
    pos = np.array([r * c0, r * s0, z])
    R = _rot_z(q[0]) @ _rot_y(s3) @ _rot_z(q[4])
    return pos, R


def tool_axis(q) -> np.ndarray:
    """Unit vector the tool points along (from wrist toward the jaws)."""
    _, R = fk(q)
    return R @ np.array([0.0, 0.0, 1.0])


def ee_pitch(q) -> float:
    """Angle of the tool axis from straight down. 0 = top-down, pi/2 = horizontal."""
    q = np.asarray(q, dtype=float).reshape(5)
    return _wrap(math.pi - (q[1] + q[2] + q[3]))


def ee_yaw(q) -> float:
    """Direction the arm is facing in the horizontal plane."""
    return float(np.asarray(q, dtype=float).reshape(5)[0])


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    pos_error: float
    reason: str = "ok"
    elbow: str = "up"
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


def _limits_ok(q: np.ndarray, slack: float = 1e-9) -> bool:
    return bool(
        np.all(q >= JOINT_LIMITS[:, 0] - slack) and np.all(q <= JOINT_LIMITS[:, 1] + slack)
    )


def _solve_branch(r: float, z: float, s3: float, q0: float, roll: float, elbow_up: bool):
    """Closed-form 2-link solve in the arm's vertical plane. Returns q or None."""
    # Back out the wrist_flex centre by walking L_HAND back along the tool axis.
    rw = r - L_HAND * math.sin(s3)
    zw = z - L_HAND * math.cos(s3)

    a = rw
    b = zw - L_SHOULDER_H
    D = math.hypot(a, b)
    if D > MAX_REACH - 1e-9 or D < MIN_REACH + 1e-9:
        return None

    cos_q2 = (D * D - L_UPPER * L_UPPER - L_FORE * L_FORE) / (2.0 * L_UPPER * L_FORE)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = math.acos(cos_q2)
    if not elbow_up:
        q2 = -q2

    phi = math.atan2(a, b)                                    # angle of wrist centre from +Z
    psi = math.atan2(L_FORE * math.sin(q2), L_UPPER + L_FORE * math.cos(q2))
    q1 = phi - psi
    q3 = s3 - q1 - q2

    q = np.array([q0, _wrap(q1), _wrap(q2), _wrap(q3), _wrap(roll)])
    return q if _limits_ok(q) else None


def solve_ik(
    target_pos,
    *,
    pitch: float = TOP_DOWN,
    roll: float = 0.0,
    yaw: float | None = None,
    prefer: np.ndarray | None = None,
    verify_tol: float = 1e-6,
) -> IKResult:
    """Exact IK for grasp position + tool pitch. wrist_roll is passed through.

    pitch  angle of the tool axis from straight down (0 = top-down grasp).
    roll   wrist_roll, orienting the jaws about the tool axis.
    yaw    base rotation; defaults to facing the target.
    prefer if both elbow branches are valid, the one closer to this pose wins.
    """
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    x, y, z = (float(v) for v in target_pos)

    q0 = math.atan2(y, x) if yaw is None else float(yaw)
    if not (JOINT_LIMITS[0, 0] <= q0 <= JOINT_LIMITS[0, 1]):
        return IKResult(HOME_Q.copy(), False, float("inf"), "base yaw out of range")

    r = math.hypot(x, y)
    # If yaw was forced, the target may sit off the arm's plane -- project onto it.
    if yaw is not None:
        r = x * math.cos(q0) + y * math.sin(q0)

    s3 = math.pi - float(pitch)   # tool-axis angle from +Z

    candidates = []
    for elbow_up in (True, False):
        q = _solve_branch(r, z, s3, q0, roll, elbow_up)
        if q is not None:
            candidates.append(("up" if elbow_up else "down", q))

    if not candidates:
        return IKResult(
            HOME_Q.copy(), False, float("inf"),
            "target outside the reachable workspace for this pitch",
        )

    ref = HOME_Q if prefer is None else np.asarray(prefer, dtype=float).reshape(5)
    label, q = min(candidates, key=lambda c: float(np.linalg.norm(c[1] - ref)))

    achieved, _ = fk(q)
    err = float(np.linalg.norm(achieved - target_pos))
    if err > max(verify_tol, 1e-4):
        return IKResult(q, False, err, "FK/IK disagreement -- check link constants", label)
    return IKResult(q, True, err, "ok", label)


def reachable(target_pos, *, pitch: float = TOP_DOWN, **kw) -> bool:
    return solve_ik(target_pos, pitch=pitch, **kw).success


def workspace_bounds(pitch: float = TOP_DOWN, z: float = 0.04) -> tuple[float, float]:
    """(r_min, r_max) reachable radii at height `z` for a given tool pitch."""
    s3 = math.pi - float(pitch)
    zw = z - L_HAND * math.cos(s3)
    b = zw - L_SHOULDER_H
    lo, hi = 0.0, 0.0
    if abs(b) < MAX_REACH:
        hi = math.sqrt(MAX_REACH ** 2 - b ** 2)
    if abs(b) < MIN_REACH:
        lo = math.sqrt(MIN_REACH ** 2 - b ** 2)
    return lo + L_HAND * math.sin(s3), hi + L_HAND * math.sin(s3)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def interpolate(q_a, q_b, steps: int) -> np.ndarray:
    """Cosine-eased joint-space trajectory: smooth start and stop, no overshoot."""
    q_a = np.asarray(q_a, dtype=float).reshape(5)
    q_b = np.asarray(q_b, dtype=float).reshape(5)
    t = np.linspace(0.0, 1.0, max(int(steps), 2))
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return q_a[None, :] + s[:, None] * (q_b - q_a)[None, :]


def path_clear_of_table(q_path: np.ndarray, min_z: float = 0.012) -> bool:
    """Reject trajectories that would drag the gripper through the table."""
    return all(fk(q)[0][2] >= min_z for q in np.asarray(q_path, dtype=float))
