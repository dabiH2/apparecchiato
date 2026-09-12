"""Run a scheduled plan in the simulator.

Between the scheduler's arm assignment and the physics there is one more job:
driving the two arms through their motions, holding the arm that is not moving
where it is, watching whether each skill actually achieved what it claimed, and
recording frames for the demo video.

Skill-level verification matters more than it sounds. A grasp that silently
failed will otherwise produce a confident "place" of nothing at all, and the
episode reports a placement error twelve steps downstream of the real cause.
Here each skill is checked the moment it ends, and the failure is attributed to
the step that caused it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .kinematics import HOME_Q
from .scheduler import Step
from .skills import build_motion, park_idle, Motion, GRIPPER_OPEN
from .sim.env import TableEnv, subgoals, episode_success
from .sim.layout import DRAWER_OPEN_TRAVEL


@dataclass
class StepReport:
    order: int
    label: str
    skill: str
    arms: tuple[str, ...]
    ok: bool
    detail: str = ""
    wall_s: float = 0.0
    sim_s: float = 0.0


@dataclass
class EpisodeReport:
    seed: int
    instruction: str
    planner: str
    success: bool
    steps: list[StepReport] = field(default_factory=list)
    subgoals: dict = field(default_factory=dict)
    wall_s: float = 0.0
    frames: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def first_failure(self) -> StepReport | None:
        return next((s for s in self.steps if not s.ok), None)

    def summary(self) -> str:
        head = f"seed {self.seed}: {'SUCCESS' if self.success else 'FAILURE'}"
        if not self.success and self.first_failure:
            head += f" -- first failure at step {self.first_failure.order} " \
                    f"({self.first_failure.label}): {self.first_failure.detail}"
        done = sum(1 for v in self.subgoals.values() if v)
        return f"{head}  [{done}/{len(self.subgoals)} subgoals]"


def run_episode(env: TableEnv, steps: list[Step], *, instruction: str = "",
                planner: str = "", record: str | None = "cinematic",
                frame_every: int = 12, max_seconds: float = 180.0,
                verbose: bool = False, viewer=None,
                realtime: bool = False, detector=None,
                perception_camera: str = "overhead",
                perceive_only: set | None = None) -> EpisodeReport:
    """Run a scheduled plan.

    `viewer` is an optional live MuJoCo viewer (mujoco.viewer.launch_passive);
    it is synced after every physics step so you can watch the episode happen.
    `realtime` paces the loop to wall-clock time, which only matters when a human
    is watching -- otherwise it runs as fast as the solver allows.
    """
    t_start = time.perf_counter()
    rep = EpisodeReport(seed=env.spec.seed, instruction=instruction, planner=planner,
                        success=False)
    world = {o.name: o.grasp_point.copy() for o in env.spec.objects}
    from .sim.layout import drawer_content_pos
    for o in env.spec.objects:
        if o.inside_drawer:
            world[o.name] = drawer_content_pos(env.spec, o.name)

    # Perception, when asked for. The default is the simulator's own state, and
    # saying so plainly matters: every headline number in this repo was produced
    # that way, so it measures planning, scheduling and control, NOT perception.
    # Pass a detector and the episode instead starts from where that detector
    # says the objects are -- the same numbers the rest of the pipeline consumes,
    # errors and all. `scripts/run_eval.py --detector colour` does this, and the
    # gap between the two runs is the honest cost of closing the loop.
    if detector is not None:
        rep.notes.extend(_perceive(env, detector, perception_camera, world,
                                   only=perceive_only))

    hold = {"A": env.arm_q("A"), "B": env.arm_q("B")}
    grip = {"A": GRIPPER_OPEN, "B": GRIPPER_OPEN}
    held: dict[str, str | None] = {"A": None, "B": None}
    tick = 0

    for st in steps:
        t0 = time.perf_counter()
        sim0 = env.data.time
        try:
            motion = build_motion(st, env.spec, world)
        except Exception as exc:                            # noqa: BLE001
            rep.steps.append(StepReport(st.order, st.label, st.node.skill, st.arms,
                                        False, f"could not build motion: {exc}"))
            break
        motion = _with_park(motion, st.arms, hold, grip)

        for arm, q, g, _label in motion.trajectory(hold):
            hold[arm] = q
            grip[arm] = g
            env.set_arm_target(arm, q, g)
            # The idle arm is actively commanded to hold its pose rather than
            # left uncontrolled, so it does not sag into the workspace.
            other = "B" if arm == "A" else "A"
            env.set_arm_target(other, hold[other], grip[other])
            env.step()
            tick += 1
            if viewer is not None:
                if not viewer.is_running():
                    rep.notes.append("aborted: viewer closed")
                    return _finish(env, rep, t_start, record)
                viewer.sync()
                if realtime:
                    time.sleep(env.model.opt.timestep * env.substeps)
            if record and tick % frame_every == 0:
                rep.frames.append(env.render(record))
            if time.perf_counter() - t_start > max_seconds:
                rep.notes.append(f"aborted: exceeded {max_seconds:.0f}s wall clock")
                break
        env.settle(0.25)

        ok, detail = _verify(env, st, world)
        rep.steps.append(StepReport(st.order, st.label, st.node.skill, st.arms, ok,
                                    detail, time.perf_counter() - t0,
                                    float(env.data.time - sim0)))
        if verbose:
            print(f"  {'ok ' if ok else 'FAIL'} {st.order:2d}. {st.label} {detail}")
        held = _held_after(st, held)
        notes = _update_world(env, st, world, detector, perception_camera,
                              skip={v for v in held.values() if v},
                              only=perceive_only)
        if notes:
            rep.notes.extend(notes)
        if not ok:
            break
        if time.perf_counter() - t_start > max_seconds:
            break

    return _finish(env, rep, t_start, record)


def _finish(env: TableEnv, rep: EpisodeReport, t_start: float,
            record: str | None) -> EpisodeReport:
    if record:
        rep.frames.append(env.render(record))
    rep.subgoals = subgoals(env)
    rep.success = episode_success(env)
    rep.wall_s = time.perf_counter() - t_start
    return rep


def _verify(env: TableEnv, st: Step, world: dict) -> tuple[bool, str]:
    """Did this skill actually do what it claimed?"""
    n = st.node
    if n.skill == "open_drawer":
        d = env.drawer_open()
        return (d >= DRAWER_OPEN_TRAVEL * 0.75,
                f"drawer at {d * 1000:.0f}/{DRAWER_OPEN_TRAVEL * 1000:.0f} mm")
    if n.skill == "pick":
        obj = n.args["object"]
        held = env.is_grasped(st.arms[0], obj)
        return held, f"{'holding' if held else 'lost'} the {obj}"
    if n.skill == "place":
        obj, tgt = n.args["object"], n.args["target"]
        goal = np.asarray(env.spec.goals.get(tgt, world.get(tgt)), dtype=float)
        err = float(np.linalg.norm(env.object_pos(obj)[:2] - goal[:2]))
        return err < 0.045, f"{obj} landed {err * 1000:.0f} mm from its slot"
    if n.skill == "handoff":
        obj = n.args["object"]
        taker = st.arms[1]
        held = env.is_grasped(taker, obj)
        return held, f"arm {taker} {'received' if held else 'did not receive'} the {obj}"
    if n.skill == "pour":
        from .sim.env import _yaw_tilt
        tilt = _yaw_tilt(env.object_quat(n.args["into"]))
        return tilt < 0.35, f"{n.args['into']} tilt {np.degrees(tilt):.0f} deg after pouring"
    return True, ""


def _with_park(motion: Motion, active, hold: dict, grip: dict) -> Motion:
    """Prepend "get the other arm out of the way" to a step's motion.

    Done here rather than inside the skills because only the execution loop
    knows where the arms actually are and what they are holding. See
    skills.park_idle for the measurement that made this necessary.
    """
    pk = park_idle(tuple(active), hold, grip)
    if pk is None:
        return motion
    return Motion(motion.skill, pk.waypoints + motion.waypoints,
                  motion.notes + [f"parked idle arm(s) {[w.arm for w in pk.waypoints]} first"])


def _perceive(env: TableEnv, detector, camera: str, world: dict,
              skip: set | None = None, only: set | None = None) -> list[str]:
    """Replace believed positions with what the detector says, and say how wrong.

    Only for objects the detector could actually be looking at: anything still
    inside a shut drawer is not visible, and overriding it with whatever blob the
    segmenter found instead is not perception, it is noise. Everything else takes
    the detector's answer, errors included.

    This runs after EVERY step, not just at the start. Running it once and then
    reading simulator state for the rest of the episode produces a perfectly
    healthy-looking 100% that measures nothing: the first bad reading gets
    silently corrected by the next update, and the run is indistinguishable from
    having no perception at all.
    """
    notes: list[str] = []
    skip = skip or set()
    seen = detector.detect(env, camera)
    hidden = {o.name for o in env.spec.objects
              if o.inside_drawer and env.drawer_open() < 0.02}
    for name, det in seen.items():
        if name not in world or det is None or name in hidden or name in skip:
            continue
        if only and name not in only:
            continue
        truth = env.object_pos(name)
        # x and y from the detector; z from the object library. The detector
        # back-projects onto the table plane, so its z is the plane, not the
        # grasp height -- feeding that straight through asks the arm to close its
        # jaws at z = 0 and it fails with "outside the reachable workspace",
        # which is a units bug masquerading as a perception result. A real cell
        # knows how tall a mug is; it does not measure it every time.
        world[name] = np.array([float(det.pos[0]), float(det.pos[1]),
                                float(world[name][2])])
        notes.append(f"{name} located by {detector.name} "
                     f"{1000 * float(np.linalg.norm(world[name][:2] - truth[:2])):.0f} mm "
                     f"from truth")
    return notes


def _held_after(st: Step, held: dict) -> dict:
    """Grasp state after this step, so perception can skip what is in a gripper.

    An overhead detector back-projects onto the TABLE plane. An object 90 mm
    above it, in a moving gripper, therefore lands somewhere else entirely --
    re-locating it there is worse than not looking. A real cell has the same
    rule: while you are holding something, you know where it is.
    """
    n = st.node
    obj = n.args.get("object")
    if n.skill == "pick":
        held[st.arms[0]] = obj
    elif n.skill == "place":
        held[st.arms[0]] = None
    elif n.skill == "handoff":
        held[st.arms[0]], held[st.arms[1]] = None, obj
    return held


def _update_world(env: TableEnv, st: Step, world: dict,
                  detector=None, camera: str = "overhead",
                  skip: set | None = None, only: set | None = None) -> None:
    for o in env.spec.objects:
        world[o.name] = env.object_pos(o.name)
        # Heading, not just position: carrying an object turns it, and the next
        # skill has to straddle the shaft in the heading it actually has now.
        # Measured from the simulator rather than predicted, so a slip in the
        # jaws corrects itself at the next step instead of compounding.
        world[f"{o.name}:yaw"] = env.object_yaw(o.name)
    if st.node.skill == "open_drawer":
        env.spec.drawer_open = env.drawer_open()
    if detector is not None:
        return _perceive(env, detector, camera, world, skip, only)
    return None
