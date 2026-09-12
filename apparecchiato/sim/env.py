"""MuJoCo environment wrapper.

Thin on purpose. Everything that can be reasoned about without a physics engine
-- kinematics, layout, planning, scheduling, skill construction -- lives outside
this file and is unit-tested without MuJoCo. What is left here is the part that
genuinely needs a simulator: stepping, contacts, rendering, and reading back
where objects actually ended up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import JOINT_NAMES, HOME_Q
from .layout import SceneSpec, DRAWER_OPEN_TRAVEL
from .scene import build_mjcf, GRIPPER_TRAVEL

ARM_PREFIX = {"A": "armA", "B": "armB"}


class MuJoCoMissing(RuntimeError):
    def __init__(self):
        super().__init__(
            "MuJoCo is not installed. Run `pip install -r requirements.txt`, "
            "or `python scripts/verify_env.py` for a full environment check."
        )


@dataclass
class StepResult:
    t: float
    contacts: int
    settled: bool


class TableEnv:
    """Dual SO-101 table-setting environment."""

    def __init__(self, spec: SceneSpec, *, mjcf: str | None = None,
                 render_size: tuple[int, int] = (640, 480), substeps: int = 10,
                 markers: bool = True):
        try:
            import mujoco                                    # noqa: PLC0415
        except ImportError as exc:                           # pragma: no cover
            raise MuJoCoMissing() from exc
        self._mj = mujoco
        self.spec = spec
        self.substeps = substeps
        xml = mjcf if mjcf is not None else build_mjcf(spec, markers=markers)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = render_size
        self._renderer = None
        self._jid = {}
        self._aid = {}
        for arm, pre in ARM_PREFIX.items():
            for j in JOINT_NAMES:
                self._jid[(arm, j)] = self._name2id("joint", f"{pre}_{j}")
                self._aid[(arm, j)] = self._name2id("actuator", f"{pre}_{j}")
            for side in ("left", "right"):
                self._aid[(arm, f"{side}_jaw")] = self._name2id("actuator", f"{pre}_{side}_jaw")
        self.reset()

    # -- ids ----------------------------------------------------------------

    def _name2id(self, kind: str, name: str) -> int:
        objtype = {"joint": self._mj.mjtObj.mjOBJ_JOINT,
                   "actuator": self._mj.mjtObj.mjOBJ_ACTUATOR,
                   "body": self._mj.mjtObj.mjOBJ_BODY,
                   "site": self._mj.mjtObj.mjOBJ_SITE,
                   "camera": self._mj.mjtObj.mjOBJ_CAMERA}[kind]
        i = self._mj.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise KeyError(f"{kind} {name!r} not found in the model")
        return i

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self._mj.mj_resetData(self.model, self.data)
        for arm in ("A", "B"):
            self.set_arm_target(arm, HOME_Q, gripper=GRIPPER_TRAVEL)
            for j, v in zip(JOINT_NAMES, HOME_Q):
                self.data.qpos[self.model.jnt_qposadr[self._jid[(arm, j)]]] = float(v)
        self._mj.mj_forward(self.model, self.data)
        self.settle(0.4)

    def settle(self, seconds: float = 0.3) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._mj.mj_step(self.model, self.data)

    # -- control ------------------------------------------------------------

    def set_arm_target(self, arm: str, q, gripper: float | None = None) -> None:
        q = np.asarray(q, dtype=float).reshape(5)
        for j, v in zip(JOINT_NAMES, q):
            self.data.ctrl[self._aid[(arm, j)]] = float(v)
        if gripper is not None:
            # `gripper` is the commanded jaw separation; each finger travels half.
            half = float(np.clip(GRIPPER_TRAVEL / 2 - gripper / 2, 0.0, GRIPPER_TRAVEL / 2))
            for side in ("left", "right"):
                self.data.ctrl[self._aid[(arm, f"{side}_jaw")]] = half

    def step(self, n: int | None = None) -> StepResult:
        for _ in range(n if n is not None else self.substeps):
            self._mj.mj_step(self.model, self.data)
        vel = float(np.abs(self.data.qvel).max()) if self.data.qvel.size else 0.0
        return StepResult(float(self.data.time), int(self.data.ncon), vel < 0.02)

    # -- state --------------------------------------------------------------

    def object_pos(self, name: str) -> np.ndarray:
        return np.array(self.data.xpos[self._name2id("body", name)], dtype=float)

    def object_quat(self, name: str) -> np.ndarray:
        return np.array(self.data.xquat[self._name2id("body", name)], dtype=float)

    def object_yaw(self, name: str) -> float:
        """Heading about +z, in the same convention as ObjectSpec.yaw.

        Long thin items are grasped across the shaft, so the skills need the
        object's CURRENT heading, not the one it spawned with: carrying an
        object turns it with the base pan, and a stale yaw makes the gripper
        close along the fork instead of across it.
        """
        w, x, y, z = self.object_quat(name)
        return float(np.arctan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)))

    def grasp_site(self, arm: str) -> np.ndarray:
        return np.array(self.data.site_xpos[self._name2id("site", f"{ARM_PREFIX[arm]}_grasp")],
                        dtype=float)

    def arm_q(self, arm: str) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[self._jid[(arm, j)]]]
                         for j in JOINT_NAMES], dtype=float)

    def drawer_open(self) -> float:
        jid = self._name2id("joint", "drawer_slide")
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    def is_grasped(self, arm: str, name: str, tol: float = 0.045) -> bool:
        """Held if the object is close to the grasp site and moving with it."""
        return float(np.linalg.norm(self.object_pos(name) - self.grasp_site(arm))) < tol

    def world_state(self) -> dict:
        return {o.name: self.object_pos(o.name) for o in self.spec.objects}

    # -- rendering ----------------------------------------------------------

    def render(self, camera: str = "overhead") -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, height=self.height,
                                               width=self.width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# --- success criteria ------------------------------------------------------

PLACE_TOL_XY = 0.045       # how close to its slot an item must land
UPRIGHT_TOL = 0.35         # radians of tilt allowed for the mug and bottle


def _yaw_tilt(quat: np.ndarray) -> float:
    """Angle between the body's local +z and world +z."""
    w, x, y, z = quat
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    v = np.array([zx, zy, zz])
    n = np.linalg.norm(v)
    return float(np.arccos(np.clip(zz / (n if n else 1.0), -1.0, 1.0)))


def subgoals(env: TableEnv) -> dict[str, bool]:
    """Per-subgoal success, which is what the eval report breaks down.

    Reporting subgoals rather than one pass/fail is the difference between
    "40% success" and "every failure is the spoon hand-off" -- and only the
    second one tells you what to fix.
    """
    spec = env.spec
    out: dict[str, bool] = {
        "drawer_open": env.drawer_open() >= DRAWER_OPEN_TRAVEL * 0.75,
    }
    for name, goal in spec.goals.items():
        p = env.object_pos(name)
        g = np.asarray(goal, dtype=float)
        out[f"{name}_placed"] = bool(np.linalg.norm(p[:2] - g[:2]) < PLACE_TOL_XY
                                     and p[2] > -0.01)
    for name in ("mug", "bottle"):
        out[f"{name}_upright"] = bool(_yaw_tilt(env.object_quat(name)) < UPRIGHT_TOL)
    return out


def episode_success(env: TableEnv) -> bool:
    """The whole task: drawer opened, all four items placed, nothing knocked over."""
    s = subgoals(env)
    return all(s.values())
