#!/usr/bin/env python3
"""Choose the pour camera by looking at candidates, not by arguing about them.

The committed cinematic shot sits at x = +0.30. Arm B's base is at x = +0.12, so
for anything in the shared lens near x = 0 the camera looks straight through B's
own column -- and the pour, the one step where both arms move at once, is the
step that column hides. A reviewer watching the montage sees an arm, not a pour.

This runs seed 0 once, freezes the simulator at the moment the pour is happening,
and then re-renders that single frozen instant from every candidate eye. One
physics run, N cheap renders, so adding a candidate costs nothing.

    python scripts/find_pour_camera.py                 # render the default set
    python scripts/find_pour_camera.py --seed 0 --out out/cams

Each candidate is written as a PNG and scored: what fraction of a window around
the pour point is arm-coloured. Lower is better. The score is a tie-breaker for
the eye, not a substitute for it -- look at the PNGs.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.executor import run_episode                       # noqa: E402
from apparecchiato.planner import build_planner                      # noqa: E402
from apparecchiato.scheduler import schedule                         # noqa: E402
from apparecchiato.sim.env import TableEnv                           # noqa: E402
from apparecchiato.sim.layout import sample_scene                    # noqa: E402

DEFAULT_INSTRUCTION = ("Open the top drawer, set the plate, fork and spoon on the "
                       "table, put the mug beside them and pour water into the mug.")

# Arm A is dark grey-blue [0.16,0.17,0.19]; arm B is dark red [0.34,0.16,0.16].
ARM_RGB = {"A": (0.16, 0.17, 0.19), "B": (0.34, 0.16, 0.16)}

# What the shot has to show: the mug being held and the bottle being tipped into
# it. Scored on those, not on the arms.
#
# The first version of this script scored "arm-coloured pixels near the pour" and
# ranked the committed camera BEST -- the very camera whose whole problem is that
# arm B's column stands in front of the pour. It was measuring occlusion by
# counting the occluder, so a shot that hid everything behind one arm scored well
# and a shot with both arms visibly working scored badly. The metric agreed with
# the wrong answer, and only looking at the renders caught it.
#
# Count what the viewer is supposed to see instead. More is better.
OBJECT_RGB = {"mug": (0.25, 0.45, 0.78), "bottle": (0.20, 0.62, 0.35)}

# Candidate eyes. The committed one is first so the comparison is honest.
CANDIDATES = {
    "committed":      ((0.30, -0.30, 0.34), None),
    "front_high":     ((0.00, -0.40, 0.44), None),
    "front_slight":   ((0.10, -0.42, 0.40), None),
    "left_mirror":    ((-0.30, -0.30, 0.34), None),
    "left_soft":      ((-0.22, -0.38, 0.38), None),
    "high_three_q":   ((0.20, -0.36, 0.50), None),
    # The one that won on seeds 0 and 3, with the FIXED target the shipped
    # camera uses rather than this seed's mug -- i.e. exactly what the clips get.
    "new_default":    ((0.10, -0.42, 0.40), (-0.01, 0.16, 0.04)),
}


def _colour_mask(img: np.ndarray, rgb, tol: float = 0.10) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    return np.abs(f - np.asarray(rgb, dtype=np.float32)).max(axis=2) < tol


def _visible_px(img: np.ndarray) -> dict:
    """How many pixels of the mug and the bottle the viewer can actually see."""
    return {name: int(_colour_mask(img, rgb).sum())
            for name, rgb in OBJECT_RGB.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/cams")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--phase", type=float, default=0.55,
                    help="how far through the pour motion to freeze, 0-1")
    ap.add_argument("--window", type=int, default=180,
                    help="side of the square around the pour point, in pixels, "
                         "that the occlusion score looks at")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = sample_scene(a.seed)
    graph = build_planner("rules").plan(a.instruction, spec)
    steps = schedule(graph, spec)
    pour = next((s for s in steps if s.node.skill == "pour"), None)
    if pour is None:
        print("this seed has no pour step", file=sys.stderr)
        return 2
    print(f"seed {a.seed}: pour is step {pour.order} of {len(steps)}")

    # Run everything up to AND INCLUDING the pour, then keep the final state.
    before = [s for s in steps if s.order < pour.order]
    # Re-sample, exactly as the eval harness does. `schedule()` walks a believed
    # world forward and mutates `drawer_open` on the spec it was given, so
    # building the simulator from that same object starts the episode with the
    # drawer already open and arm A then cannot reach a knob that has moved.
    env = TableEnv(sample_scene(a.seed), render_size=(a.width, a.height),
                   markers=False)
    rep = run_episode(env, before, instruction=a.instruction, planner="rules",
                      record=None)
    print(f"ran {len(rep.steps)} of {len(before)} step(s) before the pour")
    if len(rep.steps) < len(before):
        print(rep.summary())
        if rep.first_failure:
            print(f"first failure: {rep.first_failure.label} -- "
                  f"{rep.first_failure.detail}")
        return 2

    # Now drive the pour by hand, so the frame can be taken WHILE the bottle is
    # tipped rather than after both arms have parked. Running the step through
    # run_episode and keeping the end state was the first attempt, and it
    # produced a picture of two idle arms -- which is exactly the picture the
    # montage already has and the reason this script exists.
    qpos = _drive_to_pour_moment(env, pour, a.phase)
    mug = env.object_pos("mug")
    env.close()
    print(f"mug at the captured instant: {np.round(mug, 3).tolist()}")

    # The pour point is where the mug is: that is what the shot has to show.
    target = (float(mug[0]), float(mug[1]), float(mug[2]) + 0.03)

    rows = []
    for name, (eye, tgt) in CANDIDATES.items():
        tgt = tgt or target
        e = TableEnv(sample_scene(a.seed), render_size=(a.width, a.height),
                     markers=False, cinematic_eye=eye, cinematic_target=tgt)
        e.data.qpos[:] = qpos
        e._mj.mj_forward(e.model, e.data)
        img = np.asarray(e.render("cinematic"))

        cx, cy = _project(e, target)
        vis = _visible_px(img)
        score = vis["mug"] + vis["bottle"]

        path = out / f"pour_{name}.png"
        _save(img, path)
        e.close()
        rows.append({"name": name, "eye": list(eye), "target": list(tgt),
                     "visible_px": vis, "visible_total": score,
                     "pour_px": [cx, cy], "png": str(path)})
        print(f"  {name:<14} eye={eye}  visible: mug {vis['mug']:>5} px, "
              f"bottle {vis['bottle']:>5} px, total {score:>6}  -> {path}")

    rows.sort(key=lambda r: -r["visible_total"])
    (out / "pour_cameras.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nmost of the pour visible: {rows[0]['name']} "
          f"({rows[0]['visible_total']} px)")
    print("Now LOOK at the PNGs. This score says the two objects are on screen; "
          "it cannot say the shot reads as two arms working together, and an "
          "earlier version of it confidently ranked the broken camera first.")
    return 0


def _drive_to_pour_moment(env, step, phase: float) -> np.ndarray:
    """Step the pour motion and stop at `phase` of the way through it.

    Mirrors the executor's own drive loop rather than calling it, because the
    thing wanted here is a state part-way through a step, and the executor only
    ever hands back the state after one. The waypoint labels are printed so the
    phase can be chosen by name -- the pour is built as tuck / swing / tip /
    swing-back, and only the tip is worth photographing.
    """
    from apparecchiato.executor import _with_park                    # noqa: PLC0415
    from apparecchiato.skills import build_motion, GRIPPER_OPEN      # noqa: PLC0415

    hold = {arm: env.arm_q(arm) for arm in ("A", "B")}
    grip = {"A": GRIPPER_OPEN, "B": GRIPPER_OPEN}
    world = env.world_state()
    motion = _with_park(build_motion(step, env.spec, world), step.arms, hold, grip)
    traj = list(motion.trajectory(hold))
    labels = []
    for _, _, _, lb in traj:
        if lb and (not labels or labels[-1] != lb):
            labels.append(lb)
    print(f"pour waypoint labels: {labels}")

    stop = max(1, int(len(traj) * phase))
    print(f"driving {stop} of {len(traj)} ticks ({phase:.0%}); "
          f"label there: {traj[stop - 1][3]!r}")
    for arm, q, g, _lb in traj[:stop]:
        hold[arm] = q
        grip[arm] = g
        env.set_arm_target(arm, q, g)
        other = "B" if arm == "A" else "A"
        env.set_arm_target(other, hold[other], grip[other])
        env.step()
    return np.array(env.data.qpos, copy=True)


def _project(env, xyz) -> tuple[int, int]:
    """World point -> pixel, using the camera the env was built with."""
    m, d = env.model, env.data
    cam = env._name2id("camera", "cinematic")
    pos = np.array(d.cam_xpos[cam])
    mat = np.array(d.cam_xmat[cam]).reshape(3, 3)
    # MuJoCo cameras look down -z of their own frame.
    rel = mat.T @ (np.asarray(xyz, dtype=float) - pos)
    if rel[2] >= -1e-6:
        return env.width // 2, env.height // 2
    fovy = float(m.cam_fovy[cam])
    f = (env.height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    px = env.width / 2.0 + f * (rel[0] / -rel[2])
    py = env.height / 2.0 - f * (rel[1] / -rel[2])
    return int(round(px)), int(round(py))


def _save(img: np.ndarray, path: pathlib.Path) -> None:
    try:
        from PIL import Image                                        # noqa: PLC0415
        Image.fromarray(img).save(path)
    except ImportError:
        import imageio.v2 as imageio                                 # noqa: PLC0415
        imageio.imwrite(path, img)


if __name__ == "__main__":
    raise SystemExit(main())
