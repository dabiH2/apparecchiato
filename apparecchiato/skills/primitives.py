"""Skill primitives: a scheduled node becomes a joint-space motion.

Each primitive returns a Motion -- an ordered list of Waypoints, each carrying a
target joint vector for one arm plus a gripper command. The executor interpolates
between waypoints with `kinematics.interpolate`, so motions are smooth and the
same code drives both the simulator and (in principle) a real SO-101.

Every primitive approaches and retreats along the tool axis rather than diving
straight at the object. That is what makes the grasps robust to the randomised
object heights and scales: the pre-grasp is always a fixed standoff above the
grasp point, never a pose tuned to one scene.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..kinematics import (
    HOME_Q, TOP_DOWN, JOINT_LIMITS, solve_ik, interpolate, fk,
)
from ..sim.layout import (
    to_arm_frame, DRAWER_OPEN_TRAVEL, drawer_knob_pos, pick_pos, PARK_Q,
)

GRIPPER_OPEN = 0.035      # jaw separation, metres
GRIPPER_CLOSED = 0.002

# Commanded jaw separation per object, a few mm INSIDE the object's real width so
# the pads squeeze rather than crush. Closing every grasp to 2 mm looks harmless
# and is not: against a 16 mm fork the actuator simply drives the pads through
# the object and it squirts out -- visible in the grasp probe as both fingers
# touching the fork, then the fork gone and the fingers touching each other.
GRIP_WIDTH = {
    "plate": 0.021,       # 26 mm saucer, gripped across the disc
    "mug": 0.024,         # 28 mm cup body
    "bottle": 0.022,      # 28 mm bottle body
    # 16 mm shaft, squeezed 1.5 mm per pad. There is a real optimum here, and it
    # is narrow: at 10 mm the shaft slid 21 mm down its own length during a
    # carry, and at 6 mm the near-rigid pads wedged it and popped it up OUT of
    # the jaws to land on top of the fingers. Both were measured, not guessed.
    "spoon": 0.013,
    "fork": 0.013,
}
KNOB_GRIP = 0.014         # the drawer knob is 18 mm across

# One control tick is TableEnv.substeps (10) physics steps of 2 ms, so 50 ticks
# of held setpoint is one second of settling. Used to turn Waypoint.dwell into a
# number of repeated setpoints; keep in step with sim.env.TableEnv.
TICKS_PER_S = 50.0
SETTLE = 0.30             # dwell after arriving somewhere, before moving on

# Cap on how far any joint may be commanded to move in one control tick, i.e. a
# joint speed limit of MAX_JOINT_STEP * TICKS_PER_S = 0.6 rad/s.
#
# A waypoint's `steps` is a floor, not the schedule: the same 50 steps that are
# gentle for a 10 cm descent are a fling across a 2 rad shoulder sweep, and a
# fling is exactly what it looked like -- the fork left the jaws mid-swing on the
# way to its slot, and an arm sent home while holding the mug arrived empty. A
# speed limit makes long moves slow and short moves quick without hand-tuning a
# step count per waypoint.
MAX_JOINT_STEP = 0.012


def _steps_for(q_from, q_to) -> int:
    """Interpolation steps needed to honour the joint speed limit."""
    return int(np.ceil(float(np.max(np.abs(np.asarray(q_to, dtype=float)
                                          - np.asarray(q_from, dtype=float))))
                       / MAX_JOINT_STEP))


def grip_for(kind: str) -> float:
    return float(GRIP_WIDTH.get(kind, GRIPPER_CLOSED))

# Preferred standoffs. The reachable annulus narrows as the tool rises, so these
# are ceilings, not fixed values: _standoff() takes the highest clearance that is
# actually reachable, down to MIN_CLEARANCE, and raises only if even that fails.
APPROACH_H = 0.075        # preferred standoff above a top-down grasp
LIFT_H = 0.090            # preferred carry height
MIN_H = 0.022             # must match layout.MIN_CLEARANCE
POUR_TILT = 1.15          # wrist roll applied when tipping the bottle
KNOB_ROLL = 0.0           # wrist roll used for the drawer knob (see open_drawer)


class SkillError(RuntimeError):
    pass


@dataclass
class Waypoint:
    arm: str
    q: np.ndarray
    gripper: float
    label: str = ""
    dwell: float = 0.0          # seconds to hold once reached
    steps: int = 40             # interpolation steps to get here

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float).reshape(5)


@dataclass
class Motion:
    skill: str
    waypoints: list[Waypoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waypoints)

    def arms(self) -> set[str]:
        return {w.arm for w in self.waypoints}

    def trajectory(self, start_q: dict[str, np.ndarray], ticks_per_second: float = TICKS_PER_S):
        """Expand waypoints into per-arm setpoint streams.

        Yields (arm, q, gripper, label) tuples in execution order.

        `dwell` is honoured by repeating the final setpoint, which is the whole
        point of it: the interpolation only says where to AIM, and the arm is
        still catching up when the last interpolated setpoint is issued. Without
        this the jaws close, and the descent starts, while the tool is tens of
        millimetres from where it is supposed to be -- measured on seed 0, the
        gripper arrived 40 mm to the side of the mug and swiped it off its spot
        before closing on empty air.
        """
        cur = {a: np.asarray(q, dtype=float).reshape(5) for a, q in start_q.items()}
        for w in self.waypoints:
            steps = max(int(w.steps), _steps_for(cur[w.arm], w.q))
            for q in interpolate(cur[w.arm], w.q, steps)[1:]:
                yield w.arm, q, w.gripper, w.label
            cur[w.arm] = w.q
            for _ in range(int(round(max(0.0, w.dwell) * ticks_per_second))):
                yield w.arm, w.q, w.gripper, w.label


# --- helpers ---------------------------------------------------------------

def _ik_world(world_xyz, arm: str, *, pitch: float = TOP_DOWN, roll: float = 0.0,
              prefer=None, what: str = "target") -> np.ndarray:
    res = solve_ik(to_arm_frame(world_xyz, arm), pitch=pitch, roll=roll, prefer=prefer)
    if not res.success:
        raise SkillError(
            f"arm {arm} cannot reach {what} at {np.round(np.asarray(world_xyz), 3).tolist()}: "
            f"{res.reason}"
        )
    return res.q


def _above(p, h: float) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    p[2] += h
    return p


def base_pan(arm: str, target) -> float:
    """The shoulder-pan angle needed to point that arm at a table point."""
    t = to_arm_frame(target, arm)
    return math.atan2(float(t[1]), float(t[0]))


def carried_yaw(arm: str, frm, to, yaw0: float) -> float:
    """World heading of an object after being carried from one point to another.

    The tool's heading is (base pan + wrist roll). You must hold the wrist roll
    fixed for the whole carry -- twisting it under load just rotates the object
    inside the pads and it walks out of the jaws -- so the object turns with the
    base pan, by exactly the change in pan between the two points.

    Corollary, and the reason this exists: with the roll held fixed,
    cross_shaft_roll(arm, to, carried_yaw(...)) == cross_shaft_roll(arm, frm,
    yaw0). Computing the set-down roll from the object's ORIGINAL yaw instead
    was a real bug -- it asked the wrist to twist mid-carry, and the fork slipped
    18 mm in the jaws between the drawer and the transfer point, then was
    released 26 mm above the table.
    """
    return float(yaw0) + base_pan(arm, to) - base_pan(arm, frm)


def held_yaw(scene, obj: str, world: dict | None) -> float:
    """The object's CURRENT heading: measured if we have it, spawned otherwise."""
    if world is not None and f"{obj}:yaw" in world:
        return float(world[f"{obj}:yaw"])
    return float(scene.by_name(obj).yaw)


def cross_shaft_roll(arm: str, target, yaw: float) -> float:
    """Wrist roll that puts the jaws ACROSS a long thin object, not along it.

    Derived rather than guessed, because a fixed constant cannot be right: the
    jaw axis depends on where the arm is pointing, so the same roll straddles a
    fork correctly in one part of the workspace and squeezes it lengthways in
    another.

    For a top-down grasp the tool rotation is Rz(q0)*Ry(pi)*Rz(q4), so the finger
    offset (local +y) comes out in the arm frame as [sin(q4-q0), cos(q4-q0), 0],
    and the arm's +90 deg mounting turns that into world [-cos(q4-q0),
    sin(q4-q0), 0]. An object with yaw `t` has its long axis along world
    [-sin t, cos t]. Requiring the jaw axis to be perpendicular to it gives

        sin(t + q4 - q0) = 0   ->   q4 = q0 - t

    which is what this returns, wrapped into the wrist's range.
    """
    t = to_arm_frame(target, arm)
    q0 = math.atan2(float(t[1]), float(t[0]))
    roll = (q0 - float(yaw) + math.pi / 2) % math.pi - math.pi / 2
    return float(np.clip(roll, JOINT_LIMITS[4][0], JOINT_LIMITS[4][1]))


def _standoff(arm: str, target, h_pref: float, *, roll: float = 0.0, prefer=None,
              what: str = "standoff") -> np.ndarray:
    """Highest reachable pose above `target`, between h_pref and MIN_H."""
    target = np.asarray(target, dtype=float).reshape(3)
    for h in np.linspace(h_pref, MIN_H, 9):
        res = solve_ik(to_arm_frame(_above(target, float(h)), arm),
                       pitch=TOP_DOWN, roll=roll, prefer=prefer)
        if res.success:
            return res.q
    raise SkillError(
        f"arm {arm} has no clearance above {what} at "
        f"{np.round(target, 3).tolist()} (tried {MIN_H * 100:.0f}-{h_pref * 100:.0f} cm)"
    )


def turn_pose(arm: str, target, roll: float = 0.0) -> np.ndarray:
    """The park pose, re-aimed at `target`: same tucked shape, new base heading.

    This exists because waypoints are interpolated in JOINT space, and a joint-
    space straight line between two very different configurations does not keep
    the tool anywhere sensible in between. Going from park straight to a pose
    above a plate on the far side of the table swings the tool out through the
    middle of the table and DIPS it: measured at the moment of failure, arm A's
    tool was at z = 38 mm with both fingers 5 mm inside the cabinet, on its way
    to a plate it never reached. Three of ten seeds ended there.

    Turning first fixes it geometrically rather than by luck. Park keeps the
    tool 96 mm from the base, inside BASE_KEEPOUT where nothing is allowed to
    stand, and at that radius the swing stays at y <= 0.096 -- in front of the
    cabinet, which starts at y = 0.14. So the arm pans while tucked, and only
    then extends radially outward, which is a motion in one vertical plane.
    """
    q = np.asarray(PARK_Q[arm], dtype=float).copy()
    q[0] = base_pan(arm, target)
    q[4] = roll
    return q


def _grasp_sequence(arm: str, target, *, label: str, roll: float = 0.0,
                    prefer=None, release: bool = False,
                    grip: float = GRIPPER_CLOSED) -> list[Waypoint]:
    """Turn, approach from above, close (or open) the jaws, retreat, tuck back.

    Used by pick, place and both halves of a hand-off, so the approach geometry
    is identical everywhere and a grasp that works in one skill works in all.
    """
    turn = turn_pose(arm, target, roll)
    if prefer is None:
        prefer = turn
    pre = _standoff(arm, target, APPROACH_H, roll=roll, prefer=prefer, what=f"pre-{label}")
    at = _ik_world(target, arm, roll=roll, prefer=pre, what=label)
    post = _standoff(arm, target, LIFT_H, roll=roll, prefer=at, what=f"lift-{label}")
    hold_before = GRIPPER_OPEN if not release else grip
    hold_after = grip if not release else GRIPPER_OPEN
    return [
        # Every arrival gets a settle. The arm is a position servo with real
        # inertia: the last interpolated setpoint is where it is AIMING, not
        # where it is. Descending or closing from a pose it has not reached yet
        # is how the gripper ends up swiping an object off the table and then
        # closing on empty air.
        Waypoint(arm, turn, hold_before, f"{label}:turn", steps=40, dwell=0.15),
        Waypoint(arm, pre, hold_before, f"{label}:approach", steps=45, dwell=SETTLE),
        Waypoint(arm, at, hold_before, f"{label}:descend", steps=30, dwell=SETTLE),
        Waypoint(arm, at, hold_after, f"{label}:{'release' if release else 'grasp'}",
                 dwell=0.45, steps=2),
        Waypoint(arm, post, hold_after, f"{label}:retreat", steps=30, dwell=0.2),
        # Tuck straight back in along the same heading before anything pans. The
        # outward trip is only safe because it happens in one vertical plane;
        # the return trip has to as well, or the arm sweeps back across the
        # table at carry height with the object still in the jaws.
        Waypoint(arm, turn_pose(arm, target, roll), hold_after,
                 f"{label}:withdraw", steps=40, dwell=0.1),
    ]


# --- primitives ------------------------------------------------------------

def open_drawer(arm: str, scene) -> Motion:
    """Grasp the knob top-down and pull the drawer toward the arms."""
    shut = drawer_knob_pos(scene)
    opened = shut.copy()
    opened[1] -= DRAWER_OPEN_TRAVEL

    # Wrist roll is threaded through every knob waypoint because the knob sits in
    # a tight pocket: the fingers are 52 mm long and reach ~17 mm below the grasp
    # point, so anything protruding from the drawer front is something they can
    # jam into. See docs/02-build-plan.md, day-1 log.
    pre = _standoff(arm, shut, APPROACH_H, roll=KNOB_ROLL, what="pre-knob")
    at = _ik_world(shut, arm, roll=KNOB_ROLL, prefer=pre, what="knob")
    # Pull in small increments: every intermediate pose is IK-checked, so a
    # drawer that would drag the arm out of its workspace fails here, loudly,
    # instead of silently slipping off the knob mid-pull.
    pull = []
    prev = at
    for f in np.linspace(0.2, 1.0, 5):
        p = shut.copy()
        p[1] -= DRAWER_OPEN_TRAVEL * float(f)
        q = _ik_world(p, arm, roll=KNOB_ROLL, prefer=prev, what=f"knob pulled {f:.0%}")
        pull.append(Waypoint(arm, q, KNOB_GRIP, f"drawer:pull{f:.0%}", steps=18))
        prev = q
    clear = _standoff(arm, opened, LIFT_H, roll=KNOB_ROLL, prefer=prev,
                      what="clear of drawer")

    return Motion("open_drawer", [
        Waypoint(arm, turn_pose(arm, shut, KNOB_ROLL), GRIPPER_OPEN,
                 "drawer:turn", steps=40, dwell=0.15),
        Waypoint(arm, pre, GRIPPER_OPEN, "drawer:approach", steps=45),
        Waypoint(arm, at, GRIPPER_OPEN, "drawer:descend", steps=25),
        Waypoint(arm, at, KNOB_GRIP, "drawer:grasp-knob", dwell=0.3, steps=2),
        *pull,
        Waypoint(arm, prev, GRIPPER_OPEN, "drawer:release-knob", dwell=0.2, steps=2),
        Waypoint(arm, clear, GRIPPER_OPEN, "drawer:clear", steps=25),
        Waypoint(arm, turn_pose(arm, opened, KNOB_ROLL), GRIPPER_OPEN,
                 "drawer:withdraw", steps=40),
    ], notes=[f"pulled {DRAWER_OPEN_TRAVEL * 100:.0f} cm in 5 IK-checked increments"])


def pick(arm: str, scene, obj: str, world: dict | None = None) -> Motion:
    target = (world or {}).get(obj)
    if target is None:
        target = pick_pos(scene, obj, assume_drawer_open=scene.drawer_open > 0)
    o = scene.by_name(obj)
    # Long thin items are grasped ACROSS the shaft. Closing along its length just
    # squeezes the fork out from between the pads -- which is precisely what the
    # grasp probe caught: both fingers touching the fork, then the fork gone and
    # the fingers touching each other.
    roll = (cross_shaft_roll(arm, target, held_yaw(scene, obj, world))
            if o.kind in ("spoon", "fork") else 0.0)
    return Motion("pick", _grasp_sequence(arm, target, label=f"pick-{obj}", roll=roll,
                                          grip=grip_for(o.kind)),
                  notes=[f"grasp roll {np.degrees(roll):.0f} deg, "
                         f"jaws to {grip_for(o.kind) * 1000:.0f} mm for a {o.kind}"])


def place(arm: str, scene, obj: str, target_slot: str, world: dict | None = None) -> Motion:
    if target_slot in scene.goals:
        target = np.asarray(scene.goals[target_slot], dtype=float)
    elif world and target_slot in world:
        target = np.asarray(world[target_slot], dtype=float)
    else:
        raise SkillError(f"unknown place target {target_slot!r}")
    o = scene.by_name(obj)
    # The roll is the one the object was PICKED with, carried unchanged to the
    # slot. Recomputing it here from the target would twist the wrist mid-carry;
    # see carried_yaw() for the measurement that caught it.
    src = np.asarray((world or {}).get(obj, o.grasp_point), dtype=float)
    roll = (cross_shaft_roll(arm, src, held_yaw(scene, obj, world))
            if o.kind in ("spoon", "fork") else 0.0)
    wps = _grasp_sequence(arm, target, label=f"place-{obj}", roll=roll, release=True,
                          grip=grip_for(o.kind))
    return Motion("place", wps, notes=[
        f"carried roll {np.degrees(roll):.0f} deg held from the pick; the {o.kind} "
        f"ends up at yaw {np.degrees(carried_yaw(arm, src, target, held_yaw(scene, obj, world))):.0f} deg"])


def handoff(giver: str, taker: str, scene, obj: str, world: dict | None = None) -> Motion:
    """Transfer an object from one arm's workspace to the other's.

    The giver sets it down at the shared-lens point and withdraws completely;
    only then does the taker move in. See the comment below for why this is a
    table transfer rather than an in-air hand-off -- it is forced by the arm's
    geometry, and the measurement that proves it is recorded there.
    """
    o = scene.by_name(obj)
    grip = grip_for(o.kind)

    # The transfer happens ON the table, at the shared-lens point, not in mid-air.
    #
    # This is a geometric fact about the arm, not a shortcut. Both arms are
    # mounted at the near edge and both grasp top-down, so to meet in mid-air
    # their forearms have to occupy the same volume: measured, arm A's elbow
    # ends up 5.7 mm inside arm B's wrist, with A's shoulder saturated at its
    # 30 N limit trying to push through, and the taker never gets closer than
    # 29 mm to the object. Raising the rendezvous does not help either -- above
    # z=0.12 the top-down annulus has closed to nothing.
    #
    # So the giver sets the object down in the lens and withdraws completely,
    # then the taker picks it up. The transfer is still what the scheduler
    # decided was necessary, and it is still both arms cooperating on one object
    # neither could handle alone; only the mechanism is table-mediated.
    rendezvous = np.asarray(scene.handoff_point, dtype=float)
    transfer = rendezvous.copy()
    transfer[2] = float(o.grasp_point[2])      # resting height, not mid-air

    # The giver is already holding the object, so its roll is fixed: the one it
    # picked with. Carrying to the transfer point turns the object with the base
    # pan, and the taker -- a different arm, at a different pan angle -- has to
    # straddle the shaft in its NEW heading.
    src = np.asarray((world or {}).get(obj, o.grasp_point), dtype=float)
    yaw0 = held_yaw(scene, obj, world)
    yaw_at_transfer = carried_yaw(giver, src, transfer, yaw0)
    g_roll = (cross_shaft_roll(giver, src, yaw0)
              if o.kind in ("spoon", "fork") else 0.0)
    t_roll = (cross_shaft_roll(taker, transfer, yaw_at_transfer)
              if o.kind in ("spoon", "fork") else 0.0)

    g_pre = _standoff(giver, transfer, APPROACH_H, roll=g_roll, what="giver pre-transfer")
    g_at = _ik_world(transfer, giver, roll=g_roll, prefer=g_pre, what="transfer point (giver)")
    g_away = _standoff(giver, transfer, LIFT_H, roll=g_roll, prefer=g_at,
                       what="giver retreat")
    # Fully out of the taker's way -- and out of the cabinet's way. HOME puts the
    # tool at y = 0.253, inside the cabinet, so retreating there shut the drawer.
    g_home = np.asarray(PARK_Q[giver], dtype=float).copy()
    g_home[4] = g_roll                         # never twist the payload

    t_pre = _standoff(taker, transfer, APPROACH_H, roll=t_roll, what="taker pre-transfer")
    t_at = _ik_world(transfer, taker, roll=t_roll, prefer=t_pre, what="transfer point (taker)")
    t_away = _standoff(taker, transfer, LIFT_H, roll=t_roll, prefer=t_at,
                       what="taker retreat")

    return Motion("handoff", [
        # Giver: carry in, set down, let go, and get completely clear. The retreat
        # to HOME matters -- withdrawing only to a standoff leaves the giver's
        # forearm exactly where the taker needs to be.
        # Turn while tucked, then extend: a joint-space line from park to a pose
        # above the transfer point dips the tool through the middle of the table
        # (see turn_pose). Carrying the payload round at park radius is safe --
        # nothing may stand inside BASE_KEEPOUT.
        Waypoint(giver, turn_pose(giver, transfer, g_roll), grip,
                 "handoff:giver-turn", steps=45, dwell=0.15),
        Waypoint(giver, g_pre, grip, "handoff:carry-in", steps=130),
        Waypoint(giver, g_at, grip, "handoff:set-down", steps=28),
        Waypoint(giver, g_at, GRIPPER_OPEN, "handoff:release", dwell=0.35, steps=2),
        Waypoint(giver, g_away, GRIPPER_OPEN, "handoff:giver-lift", steps=25),
        Waypoint(giver, turn_pose(giver, transfer, g_roll), GRIPPER_OPEN,
                 "handoff:giver-withdraw", steps=40),
        Waypoint(giver, g_home, GRIPPER_OPEN, "handoff:giver-clear", steps=45),
        # Taker: only now does the second arm move in.
        Waypoint(taker, turn_pose(taker, transfer, t_roll), GRIPPER_OPEN,
                 "handoff:taker-turn", steps=40, dwell=0.15),
        Waypoint(taker, t_pre, GRIPPER_OPEN, "handoff:taker-approach", steps=50),
        Waypoint(taker, t_at, GRIPPER_OPEN, "handoff:taker-descend", steps=28),
        Waypoint(taker, t_at, grip, "handoff:taker-grasp", dwell=0.4, steps=2),
        Waypoint(taker, t_away, grip, "handoff:taker-lift", steps=30),
        Waypoint(taker, turn_pose(taker, transfer, t_roll), grip,
                 "handoff:taker-withdraw", steps=40),
    ], notes=[f"table-mediated transfer at {np.round(transfer, 3).tolist()} "
              f"in the shared lens (in-air hand-off self-collides on this arm)"])


def pour(steady_arm: str, pour_arm: str, scene, source: str, into: str,
         world: dict | None = None) -> Motion:
    """One arm holds the mug steady while the other tips the bottle over it.

    This is the complementary dual-arm action the brief asks for: the two arms
    are doing different jobs at the same time on the same object pair, not just
    taking turns.
    """
    mug = np.asarray((world or {}).get(into, scene.goals.get(into)), dtype=float)
    if mug is None or mug.shape != (3,):
        raise SkillError(f"do not know where the {into} is")
    bottle = np.asarray((world or {}).get(source, scene.by_name(source).grasp_point),
                        dtype=float)
    s_pre = _standoff(steady_arm, mug, APPROACH_H, what="pre-steady")
    s_at = _ik_world(mug, steady_arm, prefer=s_pre, what="steady the mug")
    b_pre = _standoff(pour_arm, bottle, APPROACH_H, what="pre-bottle")
    b_at = _ik_world(bottle, pour_arm, prefer=b_pre, what="bottle")
    b_up = _standoff(pour_arm, bottle, LIFT_H, prefer=b_at, what="lift bottle")
    b_over = _standoff(pour_arm, mug, 0.085, prefer=b_up, what="over the mug")
    b_tip = b_over.copy()
    b_tip[4] = float(np.clip(b_over[4] + POUR_TILT, -2.79, 2.79))
    b_back = _standoff(pour_arm, bottle, LIFT_H, prefer=b_over, what="return bottle")
    b_down = _ik_world(bottle, pour_arm, prefer=b_back, what="set bottle down")

    return Motion("pour", [
        # Both arms turn while tucked before extending -- same reason as
        # everywhere else (see turn_pose). The bottle's own trip out and back
        # stays on one heading throughout, so it needs no extra turn.
        Waypoint(steady_arm, turn_pose(steady_arm, mug), GRIPPER_OPEN,
                 "pour:steady-turn", steps=40, dwell=0.15),
        Waypoint(steady_arm, s_pre, GRIPPER_OPEN, "pour:steady-approach", steps=45),
        Waypoint(steady_arm, s_at, GRIPPER_OPEN, "pour:steady-descend", steps=25),
        Waypoint(steady_arm, s_at, GRIPPER_CLOSED, "pour:steady-hold", dwell=0.3, steps=2),
        Waypoint(pour_arm, turn_pose(pour_arm, bottle), GRIPPER_OPEN,
                 "pour:bottle-turn", steps=40, dwell=0.15),
        Waypoint(pour_arm, b_pre, GRIPPER_OPEN, "pour:bottle-approach", steps=45),
        Waypoint(pour_arm, b_at, GRIPPER_OPEN, "pour:bottle-descend", steps=25),
        Waypoint(pour_arm, b_at, GRIPPER_CLOSED, "pour:bottle-grasp", dwell=0.3, steps=2),
        Waypoint(pour_arm, b_up, GRIPPER_CLOSED, "pour:bottle-lift", steps=30),
        Waypoint(pour_arm, b_over, GRIPPER_CLOSED, "pour:bottle-over-mug", steps=40),
        Waypoint(pour_arm, b_tip, GRIPPER_CLOSED, "pour:tip", dwell=1.2, steps=35),
        Waypoint(pour_arm, b_over, GRIPPER_CLOSED, "pour:upright", steps=35),
        Waypoint(pour_arm, b_back, GRIPPER_CLOSED, "pour:return", steps=40),
        Waypoint(pour_arm, b_down, GRIPPER_CLOSED, "pour:set-down", steps=25),
        Waypoint(pour_arm, b_down, GRIPPER_OPEN, "pour:bottle-release", dwell=0.2, steps=2),
        Waypoint(pour_arm, b_up, GRIPPER_OPEN, "pour:bottle-clear", steps=25),
        Waypoint(pour_arm, turn_pose(pour_arm, bottle), GRIPPER_OPEN,
                 "pour:bottle-withdraw", steps=40),
        Waypoint(steady_arm, s_at, GRIPPER_OPEN, "pour:steady-release", dwell=0.2, steps=2),
        Waypoint(steady_arm, s_pre, GRIPPER_OPEN, "pour:steady-clear", steps=25),
        Waypoint(steady_arm, turn_pose(steady_arm, mug), GRIPPER_OPEN,
                 "pour:steady-withdraw", steps=40),
    ])


def home(arm: str, scene=None) -> Motion:
    return Motion("home", [Waypoint(arm, HOME_Q, GRIPPER_OPEN, "home", steps=55)])


def park_idle(active: tuple[str, ...] | set[str], hold: dict, grip: dict,
              tol: float = 3e-3) -> Motion | None:
    """Get any arm that is not part of the next step out of the way first.

    Two arms sharing one table means an arm left wherever its last step ended is
    not "idle", it is an obstacle. Measured on seed 0: with arm B parked where it
    finished its pick, arm A's shoulder saturated at its 30 N limit for the whole
    of the next step with 0.68 rad of tracking error, its elbow 3.6 mm inside arm
    B's elbow, and the fork was released 121 mm from its slot -- while the grasp
    itself was perfect the entire time. The gripper is left exactly as it is, so
    an arm parks WITH whatever it is holding rather than dropping it.

    Returns None when every idle arm is already parked, so this costs nothing on
    the steps that do not need it.
    """
    wps = []
    for a in ("A", "B"):
        if a in active:
            continue
        # Park the ARM, not the payload: PARK_Q has wrist_roll = 0, and driving
        # the roll to zero while holding a fork twists it between the pads until
        # it walks out. Keep whatever roll the arm is carrying.
        q = np.asarray(PARK_Q[a], dtype=float).copy()
        q[4] = float(np.asarray(hold[a], dtype=float)[4])
        if np.allclose(hold[a], q, atol=tol):
            continue
        wps.append(Waypoint(a, q, grip[a], f"park-{a}", steps=50))
    return Motion("park", wps) if wps else None


# --- dispatch --------------------------------------------------------------

def build_motion(step, scene, world: dict | None = None) -> Motion:
    """Turn one scheduled Step into a Motion."""
    n = step.node
    a = n.args
    if n.skill == "open_drawer":
        return open_drawer(step.arms[0], scene)
    if n.skill == "pick":
        return pick(step.arms[0], scene, a["object"], world)
    if n.skill == "place":
        return place(step.arms[0], scene, a["object"], a["target"], world)
    if n.skill == "handoff":
        return handoff(step.arms[0], step.arms[1], scene, a["object"], world)
    if n.skill == "pour":
        return pour(step.arms[0], step.arms[1], scene, a["source"], a["into"], world)
    if n.skill == "home":
        return home(step.arms[0], scene)
    raise SkillError(f"no primitive for skill {n.skill!r}")
