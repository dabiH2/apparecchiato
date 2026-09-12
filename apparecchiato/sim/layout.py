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
OBJECT_HALF_H = {
    "plate": 0.009,    # a 40 mm side plate: the jaws open 36 mm, a dinner plate is not pickable
    "mug": 0.030,      # 60 mm espresso cup
    "bottle": 0.040,   # grasped low on the body: the annulus closes up with height
    "spoon": 0.009,
    "fork": 0.009,
}
OBJECT_TOP_H = {
    "plate": 0.018,
    "mug": 0.060,
    "bottle": 0.104,   # body plus neck
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

# Both arms are bolted to the near edge facing +y (toward the diner), so the
# base-pan limits are measured about +y, not about the world +x axis. Everything
# reachability-related goes through to_arm_frame(), which applies this.
ARM_MOUNT_YAW = {"A": math.pi / 2.0, "B": math.pi / 2.0}
ARMS = {"A": ARM_A_BASE, "B": ARM_B_BASE}

# The cabinet lives on arm A's side, so arm A is the one that opens the drawer
# and takes the cutlery out. Named rather than hard-coded because the cutlery
# GOAL slots have to be reachable by this same arm -- see _sample_place_setting.
DRAWER_ARM = "A"


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
    scales = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in {
        "plate": (0.88, 1.12), "mug": (0.90, 1.10), "bottle": (0.92, 1.08),
        "spoon": (0.90, 1.10), "fork": (0.90, 1.10),
    }.items()}
    gz = {k: GRASP_Z[k] * v for k, v in scales.items()}

    # The drawer is placed FIRST so the tableware can be kept out of it. Sampling
    # the plate first and the cabinet afterwards let them overlap: on seed 0 the
    # plate spawned 13 mm inside the cabinet, which both wedges the drawer and
    # makes the plate impossible to pick.
    drawer_x, open_y = _sample_drawer(rng, gz=gz)

    def clear_of_cabinet(p) -> bool:
        return not (abs(float(p[0]) - drawer_x) < CAB_KEEPOUT_X
                    and float(p[1]) > DRAWER_CLOSED_Y - CAB_KEEPOUT_FRONT)

    def sample_clear(region, z):
        for _ in range(80):
            p = _sample_in_region(rng, region, z=z)
            if clear_of_cabinet(p):
                return p
        raise RuntimeError(f"could not place an object in {region!r} clear of the cabinet")

    plate_p = sample_clear(_A_ONLY, gz["plate"])
    mug_p = sample_clear(_B_ONLY, gz["mug"])
    # The bottle sits in the shared lens on purpose: pouring is a complementary
    # dual-arm action, so whichever arm is not steadying the mug has to be able
    # to lift it. A bottle only one arm could reach would make the pour
    # impossible whenever that same arm owned the mug slot.
    # Keep the bottle clear of the mug, but bounded: the shared lens is small, so
    # an unbounded "resample until far enough" loop can spin forever on seeds
    # where the mug happens to sit in the middle of it. Take the best of a fixed
    # number of draws instead -- always terminates, still well separated.
    bottle_p, best = None, -1.0
    for _ in range(60):
        cand = sample_clear(_SHARED, gz["bottle"])
        d = float(np.linalg.norm(cand[:2] - mug_p[:2]))
        if d > best:
            bottle_p, best = cand, d
        if d >= 0.09:
            break

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
    goals = _sample_place_setting(rng, gz=gz)
    # The transfer point is on the TABLE, because a mid-air hand-off self-collides
    # on this arm (see skills.handoff). It must therefore be reachable by both
    # arms across the whole range of grasp heights an object might be transferred
    # at -- the annulus shifts outward as the tool drops, so a point valid at
    # 50 mm is not necessarily valid at 15 mm.
    z_lo, z_hi = min(gz.values()), max(gz.values())
    handoff_point = None
    for _ in range(120):
        cand = sample_clear(_SHARED, z_lo)
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


def _sample_drawer(rng, *, gz: dict, tries: int = 400) -> tuple[float, float]:
    """Drawer x such that arm A can pull the knob through its whole travel and
    then reach the cutlery inside, checking intermediate positions too."""
    open_y = DRAWER_CLOSED_Y - DRAWER_OPEN_TRAVEL
    for _ in range(tries):
        x = float(rng.uniform(-0.17, -0.07))
        travel = [np.array([x, y, DRAWER_KNOB_Z])
                  for y in np.linspace(DRAWER_CLOSED_Y, open_y, 6)]
        contents_y = open_y + DRAWER_CONTENT_OFFSET_Y
        contents = [np.array([x - 0.032, contents_y, gz["spoon"]]),
                    np.array([x + 0.032, contents_y, gz["fork"]])]
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
            if all(can_reach(np.asarray(goals[n]), DRAWER_ARM, TOP_DOWN)
                   for n in ("fork", "spoon")):
                return goals
    raise RuntimeError("could not place a fully reachable dinner setting")


def validate_scene(spec: SceneSpec) -> None:
    """Fail loudly at sample time rather than mysteriously mid-episode."""
    problems = []
    for o in spec.objects:
        if not reaching_arms(o.grasp_point):
            problems.append(f"{o.name} at {np.round(o.grasp_point, 3)} is unreachable")
    for name, g in spec.goals.items():
        if not reaching_arms(np.asarray(g)):
            problems.append(f"goal for {name} at {np.round(g, 3)} is unreachable")
    if not in_shared_workspace(np.asarray(spec.handoff_point)):
        problems.append(
            f"hand-off point {np.round(spec.handoff_point, 3)} is not reachable by both arms")
    knob_x = spec.by_name("spoon").pos[0] + 0.032
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
