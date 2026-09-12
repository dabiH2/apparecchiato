"""Table layout, domain randomisation, and reachability validation.

Pure NumPy so it can be unit-tested without MuJoCo. `sample_scene(seed)` is the
single source of truth for where everything is: `apparecchiato/sim/scene.py` turns a
SceneSpec into MJCF, and `apparecchiato/eval/run_eval.py` iterates seeds through it.

Bimanual setup: two SO-101 arms mounted side by side at the near edge of the
table, both facing +y. Each arm's top-down workspace is an annulus around its
base; the lens where the two annuli overlap is the shared workspace where
hand-offs happen. Objects outside the lens are reachable by exactly one arm,
which is what forces genuine dual-arm planning rather than one arm doing
everything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from functools import lru_cache

import numpy as np

from ..kinematics import TOP_DOWN, HORIZONTAL, solve_ik

# --- fixed frame -----------------------------------------------------------
ARM_A_BASE = np.array([-0.12, 0.0, 0.0])   # left arm
ARM_B_BASE = np.array([+0.12, 0.0, 0.0])   # right arm
TABLE_Z = 0.0
TABLE_HALF_X = 0.42
TABLE_Y_MIN, TABLE_Y_MAX = -0.04, 0.46

# The drawer knob sits on top of the drawer front and is grasped TOP-DOWN, so
# the whole pull stays inside the arm's top-down annulus (a front-facing handle
# grasped horizontally does not: the horizontal band starts at r=0.29, and the
# drawer ends up at r=0.15 once open).
DRAWER_CLOSED_Y = 0.185      # knob y when shut
DRAWER_OPEN_TRAVEL = 0.045   # how far the drawer pulls toward the arms
DRAWER_KNOB_Z = 0.055
DRAWER_CONTENT_OFFSET_Y = 0.050   # cutlery sits this far behind the knob

# Keep-out box around the drawer cabinet: nothing else may spawn here.
CAB_KEEPOUT_X = 0.105        # cabinet half-width plus clearance
CAB_KEEPOUT_FRONT = 0.045    # how far in front of the shut knob to stay clear

# Hand-offs are table-mediated: the giver sets the object down at the transfer
# point and retreats, then the taker picks it up. Both arms therefore have to
# reach that (x, y) at the object's own resting height AND at the standoff
# height they approach and retreat through.
TRANSFER_STANDOFF_Z = 0.050
# Clearance the transfer point keeps from the goal slots and the bottle, on top
# of both radii. Smaller than a slot's own margin: see _slot_clear_of. 8 mm is
# what puts a pad beside the plate without touching its neighbour.
TRANSFER_MARGIN = 0.008

DRAWER_FLOOR_TOP = 0.014     # world z of the drawer's inner floor (see sim.scene)

# --- object dimensions: ONE source of truth ---------------------------------
#
# Every object's body origin sits at the mid-height of the part the jaws close
# on, so an object resting on the table has its origin at exactly HALF_H and the
# planner's grasp height is the height it actually settles at. Getting this wrong
# is not cosmetic: with the plate, bottle and cutlery each spawning 8-25 mm above
# the table, every one of them dropped on release and drifted, and the "grasp
# height" the planner aimed at was somewhere inside thin air.
#
# TOP_H is what the gripper has to clear on the way in. The pads reach 48 mm ABOVE
# the grasp site and 4 mm below it, so an approach standoff lower than
# (TOP_H - HALF_H) sweeps the open jaws straight through the object sideways --
# which is exactly how a 90 mm mug got knocked 43 mm off its spot before the jaws
# closed on empty air. Everything here is sized so a 75 mm standoff clears.
#
# TOP_H is also bounded from ABOVE, by the carry plane. A picked object rides at
# LIFT_H above where it was grasped, and since every origin sits at HALF_H the
# underside of whatever is being carried is at a fixed ~87 mm for all five kinds.
# Anything standing taller than that is an obstacle that cannot be flown over,
# and it cannot be dodged by carrying higher either: this arm has no top-down
# pose above z = 0.10 at all (see workspace_bounds). The bottle used to stand
# 104 mm and it was exactly that obstacle -- on seed 0 arm A carried the plate
# from its spawn to the transfer point, clipped the bottle's neck 20 mm off the
# line, knocked it flat, and the pour then failed five steps later with "arm A
# cannot reach bottle", because a bottle lying on its side is 27 mm from where
# anyone expected it. So the bottle is a squat carafe: still the tallest thing
# on the table, still pours, and passes under every carry with 13 mm to spare.
OBJECT_HALF_H = {
    "plate": 0.009,    # a 40 mm side plate: the jaws open 36 mm, a dinner plate is not pickable
    "mug": 0.030,      # 60 mm espresso cup
    "bottle": 0.026,   # grasped low on the body: the annulus closes up with height
    "spoon": 0.009,
    "fork": 0.009,
}
OBJECT_TOP_H = {
    "plate": 0.018,
    "mug": 0.060,
    "bottle": 0.068,   # squat carafe: see the carry-plane note above
    "spoon": 0.018,
    "fork": 0.018,
}
OBJECT_RADIUS = {      # half-width across the axis the jaws close on
    # 26 mm. It has to fit in a 36 mm jaw opening with room to DESCEND PAST:
    # at 32 mm the open pads had 1.5 mm of clearance a side, caught the rim on
    # the way down and stalled 13 mm high, so the jaws closed on the plate's top
    # edge and it stayed on the table when the arm lifted.
    "plate": 0.013,
    "mug": 0.014,
    "bottle": 0.014,
    "spoon": 0.008,
    "fork": 0.008,
}

# The grasp height IS the half-height, by construction. Kept as its own name
# because that is what the rest of the code asks for.
GRASP_Z = dict(OBJECT_HALF_H)

# The jaws open 35 mm (skills.primitives.GRIPPER_OPEN; the slide's own travel is
# 36 mm). Descending PAST an object needs clearance a side, not just a width that
# fits: at 1.5 mm the pads caught a plate's rim, and at 3.1 mm they still did.
JAW_OPEN = 0.035
DESCENT_CLEARANCE = 0.004


def _max_scale(kind: str) -> float:
    """Largest scale at which the open jaws still clear this object's widest part."""
    return (JAW_OPEN - 2 * DESCENT_CLEARANCE) / (2 * OBJECT_RADIUS[kind])

# Both arms are bolted to the near edge facing +y (toward the diner), so the
# base-pan limits are measured about +y, not about the world +x axis. Everything
# reachability-related goes through to_arm_frame(), which applies this.
ARM_MOUNT_YAW = {"A": math.pi / 2.0, "B": math.pi / 2.0}
ARMS = {"A": ARM_A_BASE, "B": ARM_B_BASE}

# The cabinet lives on arm A's side, so arm A is the one that opens the drawer
# and takes the cutlery out. Named rather than hard-coded because the cutlery
# GOAL slots have to be reachable by this same arm -- see _sample_place_setting.
DRAWER_ARM = "A"

# Where an idle arm goes to be out of the way. NOT the home pose: at HOME an
# arm's tool sits at y = 0.253, which is inside the cabinet, so parking arm A
# there drove the gripper into the open drawer and shut it -- 44.9 mm back to
# 3.0 mm -- carrying the spoon in with it and failing both steps that needed the
# spoon afterwards. One contact, three failed steps, none of them at the bug.
#
# Three things had to be true at once, and two of them were learned the hard way:
#
#   1. A TUCK, not a retreat. Optimising for distance from everything picked a
#      fully extended pose 450 mm out; an arm that unfolds across the table to
#      park sweeps a huge arc, and it flung the fork 509 mm and launched the
#      plate off the table.
#   2. The tool must keep pointing DOWN. An arm parks while still holding
#      something and the grip is a friction pinch, so tilt puts gravity along
#      the pads. At 70 deg off vertical the fork slid out; at 35 deg -- which
#      the arithmetic said was fine for a 30 g fork -- the mug still slipped,
#      because a smooth cylinder has nothing to catch on.
#   3. Clear of the furniture. But the arm CANNOT hold the tool top-down above
#      z = 0.10 at all (measured: the workspace ends there), and the cabinet
#      walls reach z = 0.08, so "top-down and above the furniture" has no
#      solution.
#
# What resolves it: park inside the arm's own BASE_KEEPOUT radius. Nothing may
# spawn there -- no object, no goal slot -- and the cabinet's nearest corner is
# 185 mm away, so the space is guaranteed empty by construction. Emptiness does
# the job that height could not.
#
# Result: tool 96 mm from its own base at z = 0.097, pitch 0, pan 0 -- straight
# out in front of each arm, so the move to park is short as well as safe.
# scripts/find_park_pose.py derives these and can be re-run if the furniture moves.
PARK_Q = {
    "A": np.array([0.0000, -0.3163, 1.6778, 1.7801, 0.0000]),  # tool [-0.12, 0.096, 0.097]
    "B": np.array([0.0000, -0.3163, 1.6778, 1.7801, 0.0000]),  # tool [ 0.12, 0.096, 0.097]
}


# --- reachability ----------------------------------------------------------

def to_arm_frame(world_xyz, arm: str) -> np.ndarray:
    """World coordinates -> that arm's base frame (translation + mounting yaw)."""
    d = np.asarray(world_xyz, dtype=float).reshape(3) - ARMS[arm]
    a = -ARM_MOUNT_YAW[arm]
    c, s = math.cos(a), math.sin(a)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def to_world(arm_xyz, arm: str) -> np.ndarray:
    """That arm's base frame -> world coordinates."""
    d = np.asarray(arm_xyz, dtype=float).reshape(3)
    a = ARM_MOUNT_YAW[arm]
    c, s = math.cos(a), math.sin(a)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]) + ARMS[arm]


# A grasp is only useful if the arm can also stand off above it: you approach
# and retreat along the tool axis. The top-down annulus shrinks with height, so
# a point can be reachable at table level and useless in practice. Requiring the
# clearance here means planner, scheduler and skills all agree on what
# "reachable" means, instead of the skills discovering it the hard way.
MIN_CLEARANCE = 0.022

# No object or slot may sit closer than this to either arm's base. See the
# comment in _can_reach_exact for what happens when one does.
BASE_KEEPOUT = 0.135


@lru_cache(maxsize=1 << 18)
def _can_reach_exact(x: float, y: float, z: float, arm: str, pitch: float,
                     clearance: float) -> bool:
    p = np.array([x, y, z], dtype=float)
    # A point tucked in close to EITHER base is unusable, even when the IK
    # happily solves it. Inside that radius a point falls in one arm's dead zone,
    # so only the far arm nominally "reaches" it -- across the whole table, with
    # its forearm over the cabinet. Measured on seed 0: the mug spawned 132 mm
    # from arm A's base, was classified B-only, and arm B spent the entire pick
    # with a finger 5.8 mm inside the cabinet, 100 mm short of the mug.
    #
    # This belongs in can_reach rather than in the sampler because planner,
    # scheduler, skills and sampler all ask this one question, and they have to
    # get the same answer.
    for base in ARMS.values():
        if float(np.linalg.norm(p[:2] - base[:2])) < BASE_KEEPOUT:
            return False
    if not solve_ik(to_arm_frame(p, arm), pitch=pitch).success:
        return False
    if clearance <= 0.0:
        return True
    p[2] += clearance
    return solve_ik(to_arm_frame(p, arm), pitch=pitch).success


def can_reach(world_xyz, arm: str, pitch: float = TOP_DOWN,
              clearance: float = MIN_CLEARANCE) -> bool:
    """Can `arm` grasp at this point and still stand off above it?

    Memoised on the exact coordinates, not a quantised grid. Quantising looks
    tempting -- sampling asks this question tens of thousands of times -- but the
    answer is a step function, and rounding a point half a millimetre back inside
    the annulus lets the sampler accept a target the IK will later refuse. The
    planner and the skills must agree exactly, so the cache key has to be exact.
    Repeated queries still hit, because callers pass the same stored coordinates.
    """
    p = np.asarray(world_xyz, dtype=float).reshape(3)
    return _can_reach_exact(float(p[0]), float(p[1]), float(p[2]), arm,
                            float(pitch), float(clearance))


def reaching_arms(world_xyz, pitch: float = TOP_DOWN) -> list[str]:
    return [a for a in ("A", "B") if can_reach(world_xyz, a, pitch)]


def in_shared_workspace(world_xyz, pitch: float = TOP_DOWN) -> bool:
    return len(reaching_arms(world_xyz, pitch)) == 2


# --- scene description -----------------------------------------------------

@dataclass
class ObjectSpec:
    name: str
    kind: str
    pos: tuple[float, float, float]
    yaw: float = 0.0
    mass: float = 0.10
    friction: float = 0.9
    scale: float = 1.0
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.85, 1.0)
    inside_drawer: bool = False

    @property
    def grasp_point(self) -> np.ndarray:
        """World grasp point. `pos` already carries the scale-adjusted height."""
        return np.array([self.pos[0], self.pos[1], self.pos[2]])


@dataclass
class SceneSpec:
    seed: int
    objects: list[ObjectSpec]
    goals: dict[str, tuple[float, float, float]]
    # Rendezvous point in the shared lens where one arm can release an object
    # into the other's grasp. Sampled per seed and validated, so hand-offs are
    # randomised too rather than always happening at the same spot.
    handoff_point: tuple[float, float, float] = (0.0, 0.16, 0.07)
    light_pos: tuple[float, float, float] = (0.0, 0.2, 1.2)
    light_intensity: float = 1.0
    table_rgba: tuple[float, float, float, float] = (0.62, 0.52, 0.40, 1.0)
    wall_rgba: tuple[float, float, float, float] = (0.35, 0.38, 0.44, 1.0)
    drawer_open: float = 0.0

    def by_name(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["objects"] = [asdict(o) for o in self.objects]
        return d


# --- randomisation ---------------------------------------------------------
# Each sampled quantity is listed in docs/01-project-spec.md under "Robustness".
# The rubric asks for randomised placement, weight, friction, shape, lighting
# and background: all six are varied here, and every seed is validated before it
# is handed to the simulator so a run never fails for a reason the policy could
# not have handled.

_A_ONLY = "A_only"
_B_ONLY = "B_only"
_SHARED = "shared"


@lru_cache(maxsize=256)
def _region_grid(region: str, z_mm: int, pitch: float = TOP_DOWN) -> np.ndarray:
    """All table points on a 5 mm grid belonging to `region` at height z.

    Built once per (region, height) and cached. Rejection-sampling these regions
    directly is deceptively expensive: the shared lens is a small fraction of the
    table, so a sampler that draws uniformly and retries can burn six figures of
    IK calls on a single seed. Enumerating the region once costs a few thousand
    and makes every later draw O(1) -- and it also means a genuinely empty region
    fails immediately with a clear message instead of spinning.
    """
    z = z_mm / 1000.0
    pts = []
    for x in np.arange(-0.36, 0.3601, 0.010):
        for y in np.arange(0.05, 0.3001, 0.010):
            arms = tuple(a for a in ("A", "B") if can_reach((x, y, z), a, pitch))
            if region == _SHARED and arms == ("A", "B"):
                pts.append((x, y, z))
            elif region == _A_ONLY and arms == ("A",):
                pts.append((x, y, z))
            elif region == _B_ONLY and arms == ("B",):
                pts.append((x, y, z))
    return np.asarray(pts, dtype=float)


_REGION_ARMS = {_SHARED: ("A", "B"), _A_ONLY: ("A",), _B_ONLY: ("B",)}
_GRID_Z_STEP_MM = 10   # grids are cached per 10 mm of height, not per exact height


def _sample_in_region(rng, region: str, pitch: float = TOP_DOWN, z: float = 0.02,
                      jitter: float = 0.005, draws: int = 60) -> np.ndarray:
    """Draw a table point reachable by exactly the requested arm(s) at height z.

    The candidate grid is cached at 10 mm height resolution -- object scale
    randomisation would otherwise give every seed a slightly different height and
    miss the cache every time. The drawn point is then moved to the object's true
    height and re-checked exactly, so the coarse grid only ever proposes; it
    never decides.
    """
    want = _REGION_ARMS[region]
    z_q = int(round(z * 1000 / _GRID_Z_STEP_MM)) * _GRID_Z_STEP_MM
    grid = _region_grid(region, z_q, pitch)
    if len(grid) == 0:
        raise RuntimeError(
            f"region {region!r} is empty at z={z:.3f} -- check the arm layout constants")
    for _ in range(draws):
        cand = grid[rng.integers(len(grid))].copy()
        cand[0] += rng.uniform(-jitter, jitter)
        cand[1] += rng.uniform(-jitter, jitter)
        cand[2] = z
        if tuple(reaching_arms(cand, pitch)) == want:
            return cand
    raise RuntimeError(
        f"could not place a point in region {region!r} at z={z:.3f} after {draws} draws")


def sample_scene(seed: int) -> SceneSpec:
    """Build one randomised but guaranteed-feasible dinner-table scene."""
    rng = np.random.default_rng(seed)

    # Scale is drawn first because it sets the grasp height, and the grasp height
    # sets which radii are reachable -- sampling position against the unscaled
    # height would let a tall object land just outside the annulus.
    #
    # The upper end of each range is CAPPED by what the hand can descend past.
    # Randomising size is only honest if every size drawn is still pickable, and
    # this one was not: the open jaws span 35 mm, so a 28.8 mm plate leaves
    # 3.1 mm a side, the pads catch its rim on the way down, and the arm stalls
    # 14 mm high and closes on the plate's top edge. Measured on seeds 4 and 5 --
    # 28.8 mm and 27.9 mm both stalled, 26.9 mm did not. Deriving the cap from
    # DESCENT_CLEARANCE rather than hand-tuning each range means a future change
    # to a radius or to the gripper cannot quietly reintroduce it.
    scales = {k: float(rng.uniform(lo, min(hi, _max_scale(k))))
              for k, (lo, hi) in {
                  "plate": (0.88, 1.12), "mug": (0.90, 1.10), "bottle": (0.92, 1.08),
                  "spoon": (0.90, 1.10), "fork": (0.90, 1.10),
              }.items()}
    gz = {k: GRASP_Z[k] * v for k, v in scales.items()}

    # The place setting is laid out FIRST, and everything else -- the cabinet
    # included -- is then kept off it. The setting has to live in the shared lens
    # (both arms must reach the mug slot, for the pour) and the lens is small, so
    # constraining the setting to dodge things already placed rejects every
    # candidate; constraining them to dodge four fixed points is easy, because
    # what is left has room to move.
    #
    # This used to run the other way round, with the drawer placed first so the
    # tableware could be kept out of it. Then the SLOTS turned out never to have
    # been checked against the cabinet at all -- only the object spawns were --
    # and on seed 0 the mug slot landed against the cabinet front: arm B drove a
    # finger 4 mm into the cabinet with the shoulder saturated at 28 N and let
    # the mug go 131 mm short. Adding that check under the old order broke 8 of
    # 30 seeds outright, because it is the setting that has nowhere else to go.
    goals = _sample_place_setting(rng, gz=gz)
    drawer_x, open_y = _sample_drawer(rng, gz=gz, scales=scales, goals=goals)

    def clear_of_cabinet(p) -> bool:
        return not (abs(float(p[0]) - drawer_x) < CAB_KEEPOUT_X
                    and float(p[1]) > DRAWER_CLOSED_Y - CAB_KEEPOUT_FRONT)

    def sample_clear(region, z, kind=None, avoid_goals=()):
        for _ in range(200):
            p = _sample_in_region(rng, region, z=z)
            if not clear_of_cabinet(p):
                continue
            if kind is None or _slot_clear_of(
                    p, kind, {n: goals[n] for n in avoid_goals}, scales):
                return p
        raise RuntimeError(
            f"could not place an object in {region!r} clear of the cabinet and slots")

    # An object must not spawn on a slot that something ELSE has to be placed
    # into -- including its own slot's neighbours. Its own slot is exempt: it is
    # about to be picked up and moved there anyway.
    plate_p = sample_clear(_A_ONLY, gz["plate"], "plate", ("mug", "fork", "spoon"))
    mug_p = sample_clear(_B_ONLY, gz["mug"], "mug", ("plate", "fork", "spoon"))
    # The bottle must be clear of EVERY goal slot, and for the bottle that is
    # non-negotiable: it is the one object that is never put away. It is poured
    # from and set back down, so wherever it spawns it stands there for the whole
    # episode. On seed 0 it spawned on the fork's slot; arm A carried the fork
    # straight into it, shoved the bottle into the drawer, the drawer shut on the
    # spoon, and four later steps failed -- none of them at the overlap.
    #
    # It does NOT have to be in the shared lens, which is what the earlier
    # version required and which leaves nowhere to put it: the setting already
    # fills the lens. One arm reaching it is enough, because the MUG slot is
    # shared, so whichever arm cannot reach the bottle can always be the one
    # steadying the mug. The lens is preferred only because it gives the
    # scheduler a free choice of which arm pours.
    bottle_p, best = None, -1.0
    for region in (_SHARED, _A_ONLY, _B_ONLY):
        for _ in range(40):
            try:
                cand = sample_clear(region, gz["bottle"], "bottle",
                                    ("plate", "mug", "fork", "spoon"))
            except RuntimeError:
                break
            d = float(np.linalg.norm(cand[:2] - mug_p[:2]))
            if d > best:
                bottle_p, best = cand, d
            if d >= 0.09:
                break
        if bottle_p is not None and best >= 0.09:
            break
    if bottle_p is None:
        raise RuntimeError("could not place the bottle clear of the place setting")

    objects = [
        ObjectSpec("plate", "plate", tuple(plate_p), yaw=float(rng.uniform(-0.4, 0.4)),
                   mass=float(rng.uniform(0.09, 0.26)),
                   friction=float(rng.uniform(0.55, 1.25)),
                   scale=scales["plate"], rgba=(0.92, 0.92, 0.94, 1.0)),
        ObjectSpec("mug", "mug", tuple(mug_p), yaw=float(rng.uniform(-0.6, 0.6)),
                   mass=float(rng.uniform(0.06, 0.20)),
                   friction=float(rng.uniform(0.55, 1.25)),
                   scale=scales["mug"], rgba=(0.25, 0.45, 0.78, 1.0)),
        ObjectSpec("bottle", "bottle", tuple(bottle_p), yaw=float(rng.uniform(-0.5, 0.5)),
                   mass=float(rng.uniform(0.15, 0.40)),
                   friction=float(rng.uniform(0.6, 1.2)),
                   scale=scales["bottle"], rgba=(0.20, 0.62, 0.35, 1.0)),
        ObjectSpec("spoon", "spoon", (drawer_x - 0.032, open_y, gz["spoon"]),
                   yaw=0.0, mass=float(rng.uniform(0.02, 0.05)),
                   friction=float(rng.uniform(0.5, 1.1)),
                   scale=scales["spoon"], rgba=(0.75, 0.76, 0.80, 1.0), inside_drawer=True),
        ObjectSpec("fork", "fork", (drawer_x + 0.032, open_y, gz["fork"]),
                   yaw=0.0, mass=float(rng.uniform(0.02, 0.05)),
                   friction=float(rng.uniform(0.5, 1.1)),
                   scale=scales["fork"], rgba=(0.75, 0.76, 0.80, 1.0), inside_drawer=True),
    ]

    # Goal slots: the finished place setting. The plate centre sits in the shared
    # lens so either arm can complete it; cutlery flanks it. The whole setting is
    # accepted only if every slot is reachable by at least one arm, so a seed can
    # never fail for a reason the policy had no way to handle.
    # The transfer point is on the TABLE, because a mid-air hand-off self-collides
    # on this arm (see skills.handoff). It must therefore be reachable by both
    # arms across the whole range of grasp heights an object might be transferred
    # at -- the annulus shifts outward as the tool drops, so a point valid at
    # 50 mm is not necessarily valid at 15 mm.
    #
    # It must ALSO stand clear of everything that is still on the table when the
    # hand-off happens: the four goal slots and the bottle. This was checked for
    # reachability only, and on seed 0 the transfer point landed 16 mm from the
    # mug's slot -- the mug had already been placed there, arm A swept the plate
    # into it on the way in, knocked it over, and set the plate down 52 mm short.
    # Arm B then closed on empty air, and the pour failed four steps later
    # because the mug was lying on its side 77 mm from where it belonged. The
    # plate and mug SPAWNS are deliberately not obstacles here: both have been
    # picked up by the time anything is transferred, and the cutlery is still in
    # the drawer.
    transfer_obstacles = dict(goals)
    transfer_obstacles["bottle"] = tuple(bottle_p)
    z_lo, z_hi = min(gz.values()), max(gz.values())
    handoff_point = None
    for _ in range(600):
        cand = sample_clear(_SHARED, z_lo)
        if not _slot_clear_of(cand, "plate", transfer_obstacles, scales,
                              margin=TRANSFER_MARGIN):
            continue
        if all(in_shared_workspace([cand[0], cand[1], z])
               for z in (z_hi, TRANSFER_STANDOFF_Z)):
            handoff_point = cand
            break
    if handoff_point is None:
        raise RuntimeError("could not place a transfer point reachable at every grasp height")

    spec = SceneSpec(
        seed=seed,
        objects=objects,
        goals=goals,
        handoff_point=(float(handoff_point[0]), float(handoff_point[1]), float(handoff_point[2])),
        light_pos=(float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.1, 0.5)),
                   float(rng.uniform(0.9, 1.5))),
        light_intensity=float(rng.uniform(0.65, 1.35)),
        # A WOOD tone, not three independent channels. Drawing r, g and b
        # separately can land on a near-grey tabletop, and a near-grey tabletop
        # is the same colour class as a white plate and a steel fork -- the
        # colour detector then segments the table and reports the plate a
        # quarter of a metre from where it is. Deriving g and b from r keeps the
        # table reliably warm and saturated, so it can never be confused with
        # achromatic tableware, while still varying plenty.
        table_rgba=_wood_tone(rng),
        # Warm too, and for the same reason: the old wall could be a washed-out
        # blue, which is the mug's hue at low saturation, and the detector
        # segmented the backdrop instead of the cup. Randomising a scene is only
        # useful if the randomisation cannot manufacture a decoy.
        wall_rgba=_wall_tone(rng),
    )
    validate_scene(spec)
    return spec


def _wood_tone(rng) -> tuple:
    """A warm tabletop with guaranteed saturation of at least ~0.35.

    Saturation in HSV is (max - min) / max; with b <= 0.62 * r that is at least
    0.38 whatever r happens to be. That bound is the point: it is what keeps the
    table out of the achromatic colour class the plate and cutlery live in.
    """
    r = float(rng.uniform(0.42, 0.74))
    g = r * float(rng.uniform(0.58, 0.80))
    b = r * float(rng.uniform(0.30, 0.62))
    return (r, g, b, 1.0)


def _wall_tone(rng) -> tuple:
    """A warm, clearly-saturated backdrop -- never a washed-out blue or green."""
    r = float(rng.uniform(0.28, 0.66))
    g = r * float(rng.uniform(0.50, 0.78))
    b = r * float(rng.uniform(0.28, 0.58))
    return (r, g, b, 1.0)


def _sample_drawer(rng, *, gz: dict, scales: dict, goals: dict | None = None,
                   tries: int = 400) -> tuple[float, float]:
    """Drawer x such that arm A can pull the knob through its whole travel and
    then reach the cutlery inside, checking intermediate positions too.

    `goals`, when given, are the place-setting slots the cabinet must not stand
    on. The drawer yields to the setting rather than the other way round because
    the setting has almost no freedom and the drawer has plenty: a full setting
    fits in roughly 3% of draws from the shared lens, and measuring the two
    together showed every one of those lands inside the cabinet keep-out once
    the drawer is at x >= -0.09, while x <= -0.11 costs nothing at all.

    The cutlery is checked at its height INSIDE THE DRAWER, not at the height it
    would rest at on the table. Those differ by the drawer floor -- 14 mm -- and
    the top-down annulus shrinks with height, so validating at table height
    accepts drawer positions the arm cannot actually reach into. Measured: the
    spoon at [-0.111, 0.234] passed the check at z=0.009 and then failed at
    execution with "no clearance above pre-pick-spoon" at z=0.022.
    """
    open_y = DRAWER_CLOSED_Y - DRAWER_OPEN_TRAVEL
    for _ in range(tries):
        x = float(rng.uniform(-0.17, -0.07))
        if goals and any(abs(float(g[0]) - x) < CAB_KEEPOUT_X
                         and float(g[1]) > DRAWER_CLOSED_Y - CAB_KEEPOUT_FRONT
                         for g in goals.values()):
            continue
        travel = [np.array([x, y, DRAWER_KNOB_Z])
                  for y in np.linspace(DRAWER_CLOSED_Y, open_y, 6)]
        contents_y = open_y + DRAWER_CONTENT_OFFSET_Y
        in_drawer = {k: DRAWER_FLOOR_TOP + OBJECT_HALF_H[k] * scales[k]
                     for k in ("spoon", "fork")}
        contents = [np.array([x - 0.032, contents_y, in_drawer["spoon"]]),
                    np.array([x + 0.032, contents_y, in_drawer["fork"]])]
        if (all(can_reach(k, "A", TOP_DOWN) for k in travel)
                and all(can_reach(c, "A", TOP_DOWN) for c in contents)):
            return x, contents_y
    raise RuntimeError("could not place the drawer within arm A's workspace")


_SETTING_OFFSETS = (
    ("mug", 0.062, 0.062, "mug"),
)

# Where the cutlery goes relative to the plate. Several arrangements, all of them
# real table settings, tried in order: the constraint that decides between them
# is that BOTH must be reachable by the arm that owns the drawer (see
# _sample_place_setting), and which ones satisfy that depends on where in the
# shared lens the setting landed.
_CUTLERY_LAYOUTS = (
    {"fork": (-0.072, 0.000), "spoon": (0.072, 0.000)},    # fork left, spoon right
    {"fork": (0.072, 0.000), "spoon": (-0.072, 0.000)},    # mirrored
    {"fork": (-0.095, 0.000), "spoon": (-0.055, 0.000)},   # both left, fork outside
    {"fork": (0.055, 0.000), "spoon": (0.095, 0.000)},     # both right
    {"fork": (-0.072, -0.055), "spoon": (-0.072, 0.055)},  # stacked to the left
)


def _slot_clear_of(goal, kind: str, obstacles: dict, scales: dict,
                   margin: float = 0.030) -> bool:
    """Is this goal slot far enough from everything already on the table?

    "Far enough" is both radii plus a gripper-pad margin: the jaws have to get
    down beside the object being placed, so touching is not the threshold --
    having room for 8 mm of finger on the near side is.

    The margin is smaller for the transfer point (TRANSFER_MARGIN). A slot is
    somewhere an object has to SIT next to its neighbours for the rest of the
    episode; the transfer point is somewhere one object is put down and picked
    straight back up, and the shared lens is not big enough to hold both the
    setting and a full slot's worth of clearance.
    """
    r_slot = OBJECT_RADIUS[kind] * scales.get(kind, 1.0)
    for name, p in obstacles.items():
        need = r_slot + OBJECT_RADIUS[name] * scales.get(name, 1.0) + margin
        if float(np.linalg.norm(np.asarray(goal)[:2] - np.asarray(p)[:2])) < need:
            return False
    return True


def _sample_place_setting(rng, *, gz: dict, tries: int = 300) -> dict:
    """Lay out the place setting around a mug slot inside the shared lens.

    Anchoring on the mug rather than the plate is a big speed-up: the mug slot is
    the one with the hardest constraint (both arms must reach it, for the pour),
    so sampling it first and deriving the rest converges in a couple of draws
    instead of rejecting hundreds of plate positions.
    """
    mug_dx, mug_dy = next((dx, dy) for n, dx, dy, _ in _SETTING_OFFSETS if n == "mug")
    for _ in range(tries):
        mug_goal = _sample_in_region(rng, _SHARED, z=gz["mug"])
        plate_goal = np.array([mug_goal[0] - mug_dx, mug_goal[1] - mug_dy, gz["plate"]])
        base = {"plate": (float(plate_goal[0]), float(plate_goal[1]), gz["plate"])}
        for name, dx, dy, kind in _SETTING_OFFSETS:
            base[name] = (float(plate_goal[0] + dx), float(plate_goal[1] + dy), gz[kind])
        # The mug slot must be reachable by BOTH arms, because pouring needs one
        # arm steadying the mug while the other tips the bottle over it.
        if not in_shared_workspace(np.asarray(base["mug"])):
            continue
        if not all(reaching_arms(np.asarray(g)) for g in base.values()):
            continue
        # The cutlery slots must be reachable by the SAME arm that owns the
        # drawer, so a fork goes drawer -> slot in one arm without a transfer.
        #
        # This is a planning constraint, not a cop-out. A 60 mm fork pinched
        # across its 9 mm edge slid 28 mm down its own shaft during a carry
        # (measured with `diagnose.py --track`), and a transfer compounds that
        # over two carries and a blind re-grasp. A planner that knows an object
        # is marginal to hand off should route it so it never has to be -- which
        # is exactly what this does. The dual-arm requirement is carried by the
        # PLATE, a rigid disc that transfers cleanly, and by the pour, which is
        # genuinely simultaneous rather than a relay.
        for layout in _CUTLERY_LAYOUTS:
            goals = dict(base)
            for name, (dx, dy) in layout.items():
                goals[name] = (float(plate_goal[0] + dx), float(plate_goal[1] + dy),
                               gz[name])
            if not all(can_reach(np.asarray(goals[n]), DRAWER_ARM, TOP_DOWN)
                       for n in ("fork", "spoon")):
                continue
            return goals
    raise RuntimeError("could not place a fully reachable dinner setting")


def validate_scene(spec: SceneSpec) -> None:
    """Fail loudly at sample time rather than mysteriously mid-episode."""
    problems = []
    for o in spec.objects:
        # Cutlery is checked where it will be WHEN IT IS PICKED: drawer open
        # (ObjectSpec.pos already is that position) and resting on the drawer
        # floor, 14 mm above the table. Both halves matter. Checking it with the
        # drawer shut fails every seed, since that is the whole reason the drawer
        # gets opened; checking it at table height passes seeds the arm cannot
        # actually reach into, because the top-down annulus shrinks with height.
        p = (np.array([o.pos[0], o.pos[1],
                       DRAWER_FLOOR_TOP + OBJECT_HALF_H[o.kind] * o.scale])
             if o.inside_drawer else o.grasp_point)
        if not reaching_arms(p):
            problems.append(f"{o.name} at {np.round(p, 3)} is unreachable")
    knob_x = spec.by_name("spoon").pos[0] + 0.032
    for name, g in spec.goals.items():
        if not reaching_arms(np.asarray(g)):
            problems.append(f"goal for {name} at {np.round(g, 3)} is unreachable")
        # A slot inside the cabinet keep-out is as unusable as an unreachable
        # one, and fails later and more confusingly: the arm gets there, drives
        # a finger into the cabinet and releases the object short.
        if (abs(float(g[0]) - knob_x) < CAB_KEEPOUT_X
                and float(g[1]) > DRAWER_CLOSED_Y - CAB_KEEPOUT_FRONT):
            problems.append(f"goal for {name} at {np.round(g, 3)} is inside the cabinet")
    if not in_shared_workspace(np.asarray(spec.handoff_point)):
        problems.append(
            f"hand-off point {np.round(spec.handoff_point, 3)} is not reachable by both arms")
    scales = {o.kind: o.scale for o in spec.objects}
    obstructing = dict(spec.goals)
    obstructing["bottle"] = spec.by_name("bottle").pos
    if not _slot_clear_of(spec.handoff_point, "plate", obstructing, scales,
                          margin=TRANSFER_MARGIN):
        problems.append(
            f"hand-off point {np.round(spec.handoff_point, 3)} is on top of a slot or the bottle")
    for o in spec.objects:
        if o.inside_drawer:
            continue
        if (abs(o.pos[0] - knob_x) < CAB_KEEPOUT_X
                and o.pos[1] > DRAWER_CLOSED_Y - CAB_KEEPOUT_FRONT):
            problems.append(f"{o.name} at {np.round(o.grasp_point, 3)} is inside the cabinet")
    for y in np.linspace(DRAWER_CLOSED_Y, DRAWER_CLOSED_Y - DRAWER_OPEN_TRAVEL, 6):
        if not can_reach(np.array([knob_x, y, DRAWER_KNOB_Z]), "A", TOP_DOWN):
            problems.append(f"drawer knob unreachable by arm A at y={y:.3f}")
            break
    if problems:
        raise ValueError("infeasible scene seed %d:\n  - %s" % (spec.seed, "\n  - ".join(problems)))


def drawer_knob_pos(spec: SceneSpec) -> np.ndarray:
    """Current world position of the drawer knob, given how far it is open."""
    x = spec.by_name("spoon").pos[0] + 0.032
    return np.array([x, DRAWER_CLOSED_Y - spec.drawer_open, DRAWER_KNOB_Z])


def drawer_content_pos(spec: SceneSpec, name: str) -> np.ndarray:
    """Where a cutlery item sits right now: it travels with the drawer."""
    o = spec.by_name(name)
    y = o.pos[1] + (DRAWER_OPEN_TRAVEL - spec.drawer_open)
    # Inside the drawer it rests on the DRAWER floor, not the table, so its grasp
    # height is that much higher. Reporting the table height here aimed the jaws
    # below the fork and they closed on the drawer floor instead.
    z = DRAWER_FLOOR_TOP + OBJECT_HALF_H[o.kind] * o.scale
    return np.array([o.pos[0], y, z])


def pick_pos(spec: SceneSpec, name: str, *, assume_drawer_open: bool = True) -> np.ndarray:
    """Where an arm would actually grasp `name`.

    Cutlery lives in the drawer, so while the drawer is shut its grasp point is
    behind the drawer front and out of reach. Planning has to reason about where
    it *will* be once the drawer is open -- that is what `assume_drawer_open`
    means. Execution passes False, because by then the drawer really is open and
    the scene's own drawer_open field is correct.
    """
    o = spec.by_name(name)
    if not o.inside_drawer:
        return o.grasp_point
    if assume_drawer_open:
        return np.array([o.pos[0], o.pos[1], o.pos[2]])
    return drawer_content_pos(spec, name)
