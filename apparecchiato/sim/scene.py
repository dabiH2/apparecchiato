"""Generate the MuJoCo scene (MJCF) for a SceneSpec.

The arm model is written from the SAME link constants the IK uses
(apparecchiato/kinematics.py), so forward kinematics in Python and the simulator agree by
construction -- there is no hidden URDF whose numbers have drifted from the
solver's. That is deliberate: a mismatch there is the single most common reason
a "working" manipulation demo misses its grasps.

If you prefer the official SO-101 MJCF (LeRobot / mujoco_menagerie), pass
--mjcf to scripts/run_episode.py and run scripts/calibrate_kinematics.py first
to re-fit the constants against it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ..kinematics import (
    L_BASE_Z, L_SHOULDER_Z, L_UPPER, L_FORE, L_WRIST, L_TOOL, JOINT_LIMITS,
)
from .layout import (
    SceneSpec, ARM_A_BASE, ARM_B_BASE, ARM_MOUNT_YAW, DRAWER_CLOSED_Y,
    DRAWER_OPEN_TRAVEL, DRAWER_KNOB_Z, TABLE_HALF_X, TABLE_Y_MIN, TABLE_Y_MAX,
    DRAWER_FLOOR_TOP, OBJECT_HALF_H, OBJECT_RADIUS,
)

GRIPPER_TRAVEL = 0.036

# Cabinet and drawer offsets. Named because three different geoms depend on
# them and an inconsistency here is invisible until a grasp mysteriously misses.
CAB_HALF_Y = 0.075        # cabinet centre sits this far behind the shut knob
# The drawer must SIT ABOVE the cabinet base, not inside it. At DRAWER_Z=0.006 the
# drawer floor (half-thickness 3 mm) overlapped the cabinet base by 3 mm across
# its whole footprint: MuJoCo then fought that penetration with enormous contact
# friction and the drawer would not slide at all -- it crept 3 mm and stopped,
# while the gripper, which had a perfect grasp, was simply dragged off the knob.
CAB_BASE_TOP = 0.006      # world z of the cabinet base's top surface
DRAWER_Z = 0.011          # drawer body height above the cabinet origin (1 mm clearance)
# DRAWER_FLOOR_TOP is defined in layout.py (= DRAWER_Z + 0.003, the floor's
# half-thickness) because the PLANNER needs it to know how high a fork sits while
# it is still in the drawer, and layout must stay importable without MuJoCo.
assert abs(DRAWER_FLOOR_TOP - (DRAWER_Z + 0.003)) < 1e-9, (
    "layout.DRAWER_FLOOR_TOP and the drawer geometry here have drifted apart")


def _e(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k.replace("_", ""): _fmt(v) for k, v in kw.items()})


def _fmt(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(f"{float(x):.6g}" for x in np.asarray(v).ravel())
    if isinstance(v, (int, float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _arm(parent, name: str, base_xyz, mount_yaw: float, rgba):
    """One SO-101: 5 revolute joints plus a symmetric two-finger gripper."""
    body = _e(parent, "body", name=f"{name}_base", pos=base_xyz,
              euler=[0.0, 0.0, float(mount_yaw)])
    # The base takes the arm's own colour, not a light grey. At [0.25,0.25,0.28]
    # it was a large, bright, desaturated disc -- the same colour class as the
    # white plate -- and the colour detector happily reported the plate sitting
    # on top of a shoulder.
    _e(body, "geom", type="cylinder", size=[0.035, L_BASE_Z / 2],
       pos=[0, 0, L_BASE_Z / 2], rgba=rgba, mass=0.4)

    pan = _e(body, "body", name=f"{name}_pan", pos=[0, 0, L_BASE_Z])
    _e(pan, "joint", name=f"{name}_shoulder_pan", type="hinge", axis=[0, 0, 1],
       range=JOINT_LIMITS[0], damping=0.15, armature=0.012)
    _e(pan, "geom", type="cylinder", size=[0.028, L_SHOULDER_Z / 2],
       pos=[0, 0, L_SHOULDER_Z / 2], rgba=rgba, mass=0.18)

    lift = _e(pan, "body", name=f"{name}_lift", pos=[0, 0, L_SHOULDER_Z])
    _e(lift, "joint", name=f"{name}_shoulder_lift", type="hinge", axis=[0, 1, 0],
       range=JOINT_LIMITS[1], damping=0.15, armature=0.014)
    _e(lift, "geom", type="capsule", fromto=[0, 0, 0, 0, 0, L_UPPER], size=0.019,
       rgba=rgba, mass=0.22)

    elbow = _e(lift, "body", name=f"{name}_elbow", pos=[0, 0, L_UPPER])
    _e(elbow, "joint", name=f"{name}_elbow_flex", type="hinge", axis=[0, 1, 0],
       range=JOINT_LIMITS[2], damping=0.12, armature=0.011)
    _e(elbow, "geom", type="capsule", fromto=[0, 0, 0, 0, 0, L_FORE], size=0.017,
       rgba=rgba, mass=0.19)

    wrist = _e(elbow, "body", name=f"{name}_wrist", pos=[0, 0, L_FORE])
    _e(wrist, "joint", name=f"{name}_wrist_flex", type="hinge", axis=[0, 1, 0],
       range=JOINT_LIMITS[3], damping=0.08, armature=0.008)
    _e(wrist, "geom", type="capsule", fromto=[0, 0, 0, 0, 0, L_WRIST], size=0.015,
       rgba=rgba, mass=0.09)

    roll = _e(wrist, "body", name=f"{name}_roll", pos=[0, 0, L_WRIST])
    _e(roll, "joint", name=f"{name}_wrist_roll", type="hinge", axis=[0, 0, 1],
       range=JOINT_LIMITS[4], damping=0.05, armature=0.005)
    _e(roll, "geom", type="cylinder", size=[0.014, 0.012], pos=[0, 0, 0.012],
       rgba=rgba, mass=0.05)
    # The grasp site sits exactly L_TOOL from the roll axis: this is the point
    # kinematics.fk() returns, and what every IK target refers to.
    _e(roll, "site", name=f"{name}_grasp", pos=[0, 0, L_TOOL], size=0.006,
       rgba=[1, 0.2, 0.2, 0.35])

    for side, sgn in (("left", 1.0), ("right", -1.0)):
        # The pads end level with the grasp point, overhanging it by only ~4 mm.
        # At the original 17 mm of overhang a flat object lying on a surface was
        # ungraspable: to bring the grasp point down to the fork's mid-height the
        # pads would have to pass through the drawer floor, so the gripper stalled
        # ~15 mm high and the jaws closed above the fork every time.
        fng = _e(roll, "body", name=f"{name}_{side}_finger",
                 pos=[0, sgn * GRIPPER_TRAVEL / 2, L_TOOL - 0.048])
        # limited + a stiff limit solref, because the default soft joint limit is
        # not a limit at all under contact loads: with a finger jammed 6.6 mm
        # into the cabinet, this joint was driven to -51 mm inside a 0-18 mm
        # range -- the gripper had effectively been torn open, and every grasp
        # after that was meaningless.
        _e(fng, "joint", name=f"{name}_{side}_jaw", type="slide", axis=[0, -sgn, 0],
           range=[0, GRIPPER_TRAVEL / 2], limited="true",
           solreflimit=[0.002, 1], solimplimit=[0.99, 0.9999, 0.0001],
           damping=0.6, armature=0.004)
        # 20 mm long along the tool x axis, not 12. When the jaws straddle a fork
        # they press on its 6 mm-thick edge, so the contact patch is only as tall
        # as the fork: all of the resistance to the fork YAWING between the pads
        # comes from the patch's LENGTH. At 12 mm the fork simply pivoted and slid
        # 21 mm down its own shaft during a carry (measured with
        # `diagnose.py --track`, which resolves the slip into along/across/up).
        _e(fng, "geom", type="box", size=[0.010, 0.004, 0.026], pos=[0, 0, 0.026],
           rgba=[0.15, 0.15, 0.18, 1], mass=0.012,
           friction=[2.2, 0.05, 0.002], condim=4,
           # Stiff, near-rigid pad contact. At the default softness a 25 N jaw
           # actuator simply pushes the pads THROUGH a thin object: the grasp
           # probe shows both fingers biting the fork, then the fingers touching
           # each other with the fork left behind.
           solimp=[0.995, 0.9999, 0.0001], solref=[0.002, 1])


def _object(parent, o):
    # ObjectSpec.pos for cutlery is where it ends up once the drawer is OPEN
    # (that is the position the planner reasons about). In the simulator it has
    # to start shut: inside the drawer, resting on the drawer floor, so the
    # floor's friction carries it out when the drawer is pulled.
    pos = list(o.pos)
    if o.inside_drawer:
        pos[1] += DRAWER_OPEN_TRAVEL
        pos[2] = DRAWER_FLOOR_TOP + OBJECT_HALF_H[o.kind] * o.scale
    body = _e(parent, "body", name=o.name, pos=pos, euler=[0, 0, float(o.yaw)])
    _e(body, "freejoint", name=f"{o.name}_free")
    common = dict(rgba=o.rgba, friction=[o.friction, 0.02, 0.002], condim=4,
                  solimp=[0.99, 0.9995, 0.0005], solref=[0.004, 1])
    s = o.scale
    # Everything is sized for a 36 mm gripper. A 110 mm plate and a 64 mm mug
    # simply cannot be picked up by this hand -- the jaws do not open that far --
    # so the props are a small plate, an espresso cup and a slim bottle.
    #
    # Every geom below is positioned by its WORLD height above the table and then
    # shifted into the body frame by `at()`. The body origin sits at
    # OBJECT_HALF_H, so an object resting on the table has its origin exactly
    # where the planner thinks the grasp point is. Writing local offsets by hand
    # is what let the plate, bottle and cutlery each spawn floating above the
    # table; tests/test_object_geometry.py now asserts bottom == 0 for each kind.
    half = OBJECT_HALF_H[o.kind] * s

    def at(centre_above_table: float) -> list[float]:
        return [0.0, 0.0, centre_above_table - half]

    if o.kind == "plate":
        r, h = OBJECT_RADIUS["plate"] * s, 0.009 * s
        _e(body, "geom", type="cylinder", size=[r, h], pos=at(h), mass=o.mass, **common)
    elif o.kind == "mug":
        h = 0.030 * s
        _e(body, "geom", type="cylinder", size=[OBJECT_RADIUS["mug"] * s, h],
           pos=at(h), mass=o.mass, **common)
    elif o.kind == "bottle":
        h = 0.040 * s                      # body: 0 .. 80 mm
        _e(body, "geom", type="cylinder", size=[OBJECT_RADIUS["bottle"] * s, h],
           pos=at(h), mass=o.mass * 0.8, **common)
        nh = 0.012 * s                     # neck: 80 .. 104 mm
        _e(body, "geom", type="cylinder", size=[0.008 * s, nh],
           pos=at(2 * h + nh), mass=o.mass * 0.2, **common)
    else:  # spoon / fork: a shaft with a head, grasped across the shaft
        # Kept short enough to fit inside the drawer. At the original 45 mm
        # half-length the tail poked 8 mm through the drawer's own front panel.
        # 18 mm thick rather than 6: the pads grip the shaft's EDGE, so the
        # contact patch is exactly as tall as the shaft, and a taller patch is
        # what stops the fork rotating and sliding out of the jaws mid-carry.
        sh = 0.009 * s
        _e(body, "geom", type="box", size=[OBJECT_RADIUS[o.kind] * s, 0.030 * s, sh],
           pos=at(sh), mass=o.mass * 0.6, **common)
        _e(body, "geom", type="box", size=[0.016 * s, 0.016 * s, sh],
           pos=[0, 0.038 * s, 0], mass=o.mass * 0.4, **common)


def _drawer(parent, spec: SceneSpec, knob_x: float):
    """A cabinet fixed to the table with a single sliding drawer along -y."""
    cab = _e(parent, "body", name="cabinet", pos=[knob_x, DRAWER_CLOSED_Y + CAB_HALF_Y, 0.0])
    _e(cab, "geom", type="box", size=[0.085, 0.075, 0.004], pos=[0, 0, 0.002],
       rgba=[0.35, 0.28, 0.22, 1], mass=2.0)
    for sx in (-1, 1):
        _e(cab, "geom", type="box", size=[0.005, 0.075, 0.040],
           pos=[sx * 0.080, 0, 0.040], rgba=[0.35, 0.28, 0.22, 1], mass=0.4)
    _e(cab, "geom", type="box", size=[0.085, 0.005, 0.040], pos=[0, 0.070, 0.040],
       rgba=[0.35, 0.28, 0.22, 1], mass=0.4)

    dr = _e(cab, "body", name="drawer", pos=[0, 0, DRAWER_Z])
    # Sized against what the arm can actually deliver: the shoulder/elbow
    # actuators are capped at 12 N, so a drawer that needs more than a few
    # newtons to move simply never opens.
    _e(dr, "joint", name="drawer_slide", type="slide", axis=[0, -1, 0],
       range=[0, DRAWER_OPEN_TRAVEL], damping=3.0, frictionloss=0.25)
    _e(dr, "geom", type="box", size=[0.072, 0.068, 0.003], pos=[0, 0, 0],
       rgba=[0.55, 0.44, 0.32, 1], mass=0.30)
    # Low front panel: the knob has to stand proud of the drawer or the fingers
    # cannot get either side of it. At the original 28 mm the panel top was above
    # the finger tips at grasp depth and both fingers jammed into the box.
    _e(dr, "geom", type="box", size=[0.072, 0.004, 0.011],
       pos=[0, -0.064, 0.011], rgba=[0.55, 0.44, 0.32, 1], mass=0.12)
    for sx in (-1, 1):
        _e(dr, "geom", type="box", size=[0.004, 0.068, 0.011],
           pos=[sx * 0.068, 0, 0.011], rgba=[0.55, 0.44, 0.32, 1], mass=0.08)
    # The knob the arm grasps top-down. Its world position when shut MUST equal
    # layout.drawer_knob_pos(), which is what the IK aims at. The cabinet body
    # sits at y = DRAWER_CLOSED_Y + CAB_HALF_Y and the drawer at z = DRAWER_Z,
    # so the knob's local offsets are exactly those two, negated -- and
    # tests/test_scene_alignment.py asserts the resulting world position rather
    # than trusting this arithmetic.
    _e(dr, "geom", name="drawer_knob", type="cylinder", size=[0.009, 0.012],
       pos=[0, -CAB_HALF_Y, DRAWER_KNOB_Z - DRAWER_Z],
       rgba=[0.85, 0.78, 0.35, 1], mass=0.03, friction=[2.4, 0.02, 0.001],
       condim=4, solimp=[0.97, 0.99, 0.001])


def build_mjcf(spec: SceneSpec, *, timestep: float = 0.002) -> str:
    root = ET.Element("mujoco", model=f"apparecchiato_seed{spec.seed}")
    _e(root, "compiler", angle="radian", autolimits="true")
    # Offscreen framebuffer big enough for a detector-sized render. MuJoCo's
    # default is 640x480, and it refuses any larger render outright -- which
    # matters because a 26 mm plate across a 680 mm field of view is under one
    # 32 px patch at that size, so the detector has nothing to look at.
    vis = _e(root, "visual")
    _e(vis, "global", offwidth=1280, offheight=1280)
    _e(root, "option", timestep=timestep, integrator="implicitfast",
       cone="elliptic", impratio=10)
    _e(root, "size", njmax=4000, nconmax=1500)

    asset = _e(root, "asset")
    _e(asset, "texture", name="grid", type="2d", builtin="checker",
       width=300, height=300, rgb1=[0.22, 0.24, 0.28], rgb2=[0.28, 0.30, 0.34])
    _e(asset, "material", name="floor", texture="grid", texrepeat=[6, 6],
       reflectance=0.08)
    _e(asset, "material", name="tablemat", rgba=spec.table_rgba, reflectance=0.03)
    _e(asset, "material", name="wallmat", rgba=spec.wall_rgba)

    world = _e(root, "worldbody")
    _e(world, "light", name="key", pos=spec.light_pos, dir=[0, -0.2, -1],
       diffuse=[spec.light_intensity] * 3, specular=[0.15] * 3, castshadow="true")
    _e(world, "light", name="fill", pos=[0, -0.4, 0.9], dir=[0, 0.4, -1],
       diffuse=[0.28 * spec.light_intensity] * 3, castshadow="false")
    _e(world, "geom", name="floor", type="plane", size=[3, 3, 0.1], pos=[0, 0, -0.001],
       material="floor")
    # Randomised backdrop: changes the visual background the perception stack sees
    # without changing anything about the physics.
    _e(world, "geom", name="backdrop", type="box", size=[1.2, 0.01, 0.5],
       pos=[0, TABLE_Y_MAX + 0.25, 0.5], material="wallmat")
    _e(world, "geom", name="table", type="box",
       size=[TABLE_HALF_X, (TABLE_Y_MAX - TABLE_Y_MIN) / 2, 0.01],
       pos=[0, (TABLE_Y_MAX + TABLE_Y_MIN) / 2, -0.01], material="tablemat",
       friction=[0.9, 0.01, 0.001], condim=4)

    # Cameras. `overhead` and `front` are what the VLM planner and the perception
    # stack consume; `cinematic` is for the demo video only.
    _e(world, "camera", name="overhead", pos=[0, 0.17, 0.62], euler=[0, 0, 0], fovy=58)
    _e(world, "camera", name="front", pos=[0, -0.34, 0.30], xyaxes=[1, 0, 0, 0, 0.55, 0.84],
       fovy=55)
    _e(world, "camera", name="wrist_a", pos=[0, 0, 0], euler=[0, 0, 0], fovy=70)
    _e(world, "camera", name="cinematic", pos=[0.40, -0.34, 0.40],
       xyaxes=[0.64, 0.77, 0, -0.36, 0.30, 0.88], fovy=50)

    # The arms are deliberately dark and desaturated, and deliberately NOT blue.
    # Arm A used to be [0.30, 0.48, 0.78] -- 0.06 away from the mug's blue in RGB
    # -- so a colour-segmenting detector locked onto the forearm instead of the
    # mug and reported it 450 mm from where the mug actually was. A robot cell
    # whose gripper is the same colour as the parts is a badly designed cell.
    _arm(world, "armA", ARM_A_BASE, ARM_MOUNT_YAW["A"], [0.16, 0.17, 0.19, 1])
    _arm(world, "armB", ARM_B_BASE, ARM_MOUNT_YAW["B"], [0.34, 0.16, 0.16, 1])

    _drawer(world, spec, knob_x=spec.by_name("spoon").pos[0] + 0.032)
    for o in spec.objects:
        _object(world, o)

    # Visual-only markers for the place-setting slots, so a viewer can see what
    # "success" means. contype/conaffinity 0 keeps them out of the physics.
    for name, g in spec.goals.items():
        _e(world, "site", name=f"goal_{name}", pos=[g[0], g[1], 0.002],
           # Violet, deliberately. These markers were green [0.2,0.9,0.4], whose
           # hue is 0.381 -- the bottle's hue is 0.38. They are 44 mm across and
           # the bottle is 28 mm, so the colour detector picked the marker every
           # time and located the bottle at the goal slot it had not reached yet.
           # A debug overlay must not be a decoy for the perception it sits under.
           size=[0.022, 0.0005], type="cylinder", rgba=[0.72, 0.24, 0.85, 0.25])
    _e(world, "site", name="handoff_zone", pos=spec.handoff_point, size=0.018,
       rgba=[0.95, 0.85, 0.2, 0.20])

    # Actuator gains.
    #
    # These are sized from the static load, not guessed. A position actuator
    # holds a pose by generating torque proportional to its tracking error, so
    # kp sets the steady-state sag: the arm needs roughly 1.2 Nm at the shoulder
    # to hold itself out over the table, and at kp=32 that is 0.038 rad of error
    # -- about 25 mm of droop at the gripper. That is enough to make every
    # top-down grasp close on air just above the object, which is exactly how
    # this first failed: the drawer knob was reached to within 25 mm and never
    # gripped.
    #
    # kp is therefore chosen so the holding error stays under ~0.005 rad, and
    # dampratio=1 critically damps each joint so the stiffer gains do not ring.
    # forcerange stays realistic for SO-101 class servos (~3 Nm).
    act = _e(root, "actuator")
    for arm in ("armA", "armB"):
        for j, kp, fmax, lim in (
            ("shoulder_pan", 150, 30, JOINT_LIMITS[0]),
            ("shoulder_lift", 300, 30, JOINT_LIMITS[1]),
            ("elbow_flex", 220, 25, JOINT_LIMITS[2]),
            ("wrist_flex", 120, 15, JOINT_LIMITS[3]),
            ("wrist_roll", 60, 10, JOINT_LIMITS[4]),
        ):
            _e(act, "position", name=f"{arm}_{j}", joint=f"{arm}_{j}", kp=kp,
               dampratio=1.0, ctrlrange=lim, forcerange=[-fmax, fmax])
        for side in ("left", "right"):
            # The jaws are commanded well past the object's surface and are
            # held there by force, so the grip strength is forcerange, not kp.
            _e(act, "position", name=f"{arm}_{side}_jaw", joint=f"{arm}_{side}_jaw",
               kp=800, dampratio=1.0, ctrlrange=[0, GRIPPER_TRAVEL / 2],
               forcerange=[-25, 25])

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def write_mjcf(spec: SceneSpec, path: str) -> str:
    xml = build_mjcf(spec)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path
