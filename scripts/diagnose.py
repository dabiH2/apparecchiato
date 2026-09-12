#!/usr/bin/env python3
"""Measure why a skill is failing, instead of guessing.

Every bug found on day one was found with three numbers, and this script prints
all three automatically:

  1. how far the grasp site is from what the IK was aiming at, at every waypoint
  2. per-joint commanded-vs-actual error and actuator force, when a step fails
  3. which bodies are actually in contact with the arm at that moment

That third one is usually the answer. "The drawer didn't open" is not a
diagnosis; "both fingers are 6.8 mm inside the drawer front panel while the
shoulder pushes 18 N into it" is.

Examples
--------
  python scripts/diagnose.py --seed 0                 # all steps, stop at first failure
  python scripts/diagnose.py --seed 0 --steps 1       # just the drawer
  python scripts/diagnose.py --seed 0 --steps 2-4 --keep-going
  python scripts/diagnose.py --seed 3 --quiet         # failures only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.executor import _verify, _update_world, _with_park   # noqa: E402
from apparecchiato.kinematics import JOINT_NAMES                  # noqa: E402
from apparecchiato.planner import build_planner                   # noqa: E402
from apparecchiato.scheduler import schedule, node_waypoints       # noqa: E402
from apparecchiato.skills import build_motion, GRIPPER_OPEN, GRIPPER_CLOSED  # noqa: E402
from apparecchiato.sim.layout import sample_scene, drawer_content_pos  # noqa: E402

DEFAULT = ("Open the top drawer, set the plate, fork and spoon on the table, "
           "put the mug beside them and pour water into the mug.")


def parse_steps(text: str, n: int) -> list[int]:
    if not text or text == "all":
        return list(range(1, n + 1))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [s for s in out if 1 <= s <= n]


def report_tracking(env, arms) -> None:
    """Commanded vs actual joint angles. Large errors mean sag or obstruction."""
    import mujoco
    m, d = env.model, env.data
    for arm in arms:
        print(f"    joint tracking, arm {arm}:")
        for j in JOINT_NAMES:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm{arm}_{j}")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm{arm}_{j}")
            cmd = float(d.ctrl[aid])
            act = float(d.qpos[m.jnt_qposadr[jid]])
            force = float(d.actuator_force[aid])
            flag = "  <-- large" if abs(act - cmd) > 0.02 else ""
            print(f"      {j:15s} cmd={cmd:+.4f} act={act:+.4f} "
                  f"err={act - cmd:+.4f} rad  force={force:+7.2f}{flag}")


def report_contacts(env, arms) -> None:
    """Who is touching the arm. Penetration depth is the negative distance."""
    import mujoco
    m, d = env.model, env.data

    def bname(gid):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or "?"

    wanted = tuple(f"arm{a}" for a in arms)
    found = []
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = bname(con.geom1), bname(con.geom2)
        if any(w in b1 or w in b2 for w in wanted):
            found.append((b1, b2, float(con.dist)))
    if not found:
        print("    contacts: none involving the arm "
              "(so nothing is blocking it -- the error is control, not collision)")
        return
    print("    contacts involving the arm:")
    for b1, b2, dist in sorted(found, key=lambda t: t[2])[:14]:
        note = "  <-- penetrating" if dist < -0.001 else ""
        print(f"      {b1:22s} <-> {b2:22s} dist={dist * 1000:+6.2f} mm{note}")


def probe_grasp(env, arm: str, obj: str | None, when: str) -> None:
    """What is between the jaws at the moment they close?

    This is the question that matters for every failed pick, and it can only be
    answered at the instant of closing -- by the time a step fails the arm has
    already retreated and the contact list is empty. Prints each finger's
    distance to the target object and every contact the arm is making.
    """
    import mujoco
    m, d = env.model, env.data
    prefix = f"arm{arm}"

    print(f"    -- grasp probe ({when}) --")
    if obj:
        try:
            opos = env.object_pos(obj)
        except Exception:                                     # noqa: BLE001
            opos = None
        if opos is not None:
            site = env.grasp_site(arm)
            print(f"       {obj} at {np.round(opos, 4)}  "
                  f"site->{obj} = {np.linalg.norm(site - opos) * 1000:5.1f} mm")
            for side in ("left", "right"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{side}_finger")
                if bid >= 0:
                    fp = np.array(d.xpos[bid])
                    print(f"       {side:5s} finger at {np.round(fp, 4)}  "
                          f"-> {obj} = {np.linalg.norm(fp - opos) * 1000:5.1f} mm")

    def bname(gid):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or "?"

    touching = []
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = bname(con.geom1), bname(con.geom2)
        if prefix in b1 or prefix in b2:
            touching.append((b1, b2, float(con.dist)))
    if not touching:
        print("       NOTHING is touching the gripper -- it closed on empty air")
    else:
        seen = set()
        for b1, b2, dist in sorted(touching, key=lambda t: t[2]):
            key = (b1, b2)
            if key in seen:
                continue
            seen.add(key)
            print(f"       touching: {b1:22s} <-> {b2:22s} dist={dist * 1000:+6.2f} mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="all", help="e.g. 1, 2-4, 1,3 (default: all)")
    ap.add_argument("--instruction", default=DEFAULT)
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue past a failing step instead of stopping")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    ap.add_argument("--track", action="store_true",
                    help="print the manipulated object's pose and its offset from "
                         "the grasp site at every waypoint -- this is how you tell "
                         "a slip in the jaws from a bad set-down from a bad IK target")
    args = ap.parse_args()

    try:
        from apparecchiato.sim.env import TableEnv
    except Exception as exc:                                      # noqa: BLE001
        print(f"cannot start the simulator: {exc}", file=sys.stderr)
        return 2

    spec = sample_scene(args.seed)
    graph = build_planner(args.planner).plan(args.instruction, spec)
    steps = schedule(graph, spec)
    wanted = parse_steps(args.steps, len(steps))
    print(f"seed {args.seed}: {len(steps)} steps, diagnosing {wanted}\n")

    env = TableEnv(sample_scene(args.seed))
    world = {o.name: (drawer_content_pos(env.spec, o.name) if o.inside_drawer
                      else o.grasp_point.copy()) for o in env.spec.objects}
    hold = {"A": env.arm_q("A"), "B": env.arm_q("B")}
    grip = {"A": GRIPPER_OPEN, "B": GRIPPER_OPEN}
    failures = 0

    for st in steps:
        run_this = st.order in wanted
        try:
            motion = build_motion(st, env.spec, world)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{st.order:2d}. {st.label}\n    MOTION BUILD FAILED: {exc}")
            failures += 1
            if not args.keep_going:
                return 1
            continue

        motion = _with_park(motion, st.arms, hold, grip)
        targets = node_waypoints(st.node, env.spec, world)
        target = np.asarray(targets[0]) if targets else None
        if run_this and (not args.quiet or args.track):
            print(f"{st.order:2d}. {st.label}")

        seen: set[str] = set()
        probe_obj = st.node.args.get("object") or st.node.args.get("source")
        closed_at: int | None = None
        tick = 0
        for arm, q, g, label in motion.trajectory(hold):
            # Fire on the gripper CLOSING, whatever width it closes to -- jaw
            # targets are object-specific now, so testing for "fully shut" would
            # silently never trigger.
            was_open = g < grip[arm] - 1e-6
            hold[arm], grip[arm] = q, g
            env.set_arm_target(arm, q, g)
            other = "B" if arm == "A" else "A"
            env.set_arm_target(other, hold[other], grip[other])
            env.step()
            tick += 1
            # The jaws just closed: look now, while there is still something to see.
            if run_this and was_open and closed_at is None:
                closed_at = tick
                probe_grasp(env, arm, probe_obj, "jaws closing")
            elif run_this and closed_at is not None and tick == closed_at + 20:
                probe_grasp(env, arm, probe_obj, "20 steps later")
            if run_this and (not args.quiet or args.track) and label not in seen:
                seen.add(label)
                site = env.grasp_site(arm)
                extra = ""
                if target is not None:
                    extra = f" site->target={np.linalg.norm(site - target) * 1000:6.1f}mm"
                if st.node.skill == "open_drawer":
                    extra += f" drawer={env.drawer_open() * 1000:5.1f}mm"
                if args.track and probe_obj:
                    # Object pose AND its offset from the site that is supposed to
                    # be carrying it. A constant offset means a clean carry; a
                    # growing one means the object is slipping or has been left
                    # behind, and the waypoint where it starts growing is the bug.
                    try:
                        op = env.object_pos(probe_obj)
                        # Offset resolved in the OBJECT's own frame: "along" is
                        # sliding down the shaft, "across" is rolling out
                        # sideways. They have different causes and different
                        # fixes, so one scalar distance is not enough.
                        oy = env.object_yaw(probe_obj)
                        d = site - op
                        along = float(-np.sin(oy) * d[0] + np.cos(oy) * d[1])
                        across = float(np.cos(oy) * d[0] + np.sin(oy) * d[1])
                        extra += (f" | {probe_obj}={np.round(op, 4)}"
                                  f" yaw={np.degrees(oy):+6.1f} off={np.linalg.norm(d) * 1000:5.1f}"
                                  f" (along={along * 1000:+5.1f} across={across * 1000:+5.1f}"
                                  f" up={d[2] * 1000:+5.1f})mm")
                    except Exception:                             # noqa: BLE001
                        pass
                print(f"    [{arm}] {label:24s}{extra}  grip={g * 1000:4.1f}mm")
        env.settle(0.25)

        ok, detail = _verify(env, st, world)
        if not ok:
            failures += 1
            print(f"{st.order:2d}. {st.label}\n    FAILED: {detail}")
            if target is not None:
                site = env.grasp_site(st.arms[0])
                print(f"    grasp site {np.round(site, 4)} vs target "
                      f"{np.round(target, 4)}  delta={np.round(site - target, 4)}")
            report_tracking(env, st.arms)
            report_contacts(env, st.arms)
            if not args.keep_going:
                print("\nstopping at the first failure (use --keep-going to continue)")
                env.close()
                return 1
        elif run_this and not args.quiet:
            print(f"    ok: {detail}")
        _update_world(env, st, world)

    env.close()
    print(f"\n{len(steps) - failures}/{len(steps)} steps passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
