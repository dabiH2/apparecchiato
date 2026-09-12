#!/usr/bin/env python3
"""Find the exact tick a carried object leaves the jaws, and what is touching.

`diagnose.py --track` prints one row per waypoint, which localises a loss to a
leg of the motion but not to an instant. This walks the same episode tick by
tick, watches the site-to-object offset, and dumps the full contact set and joint
state at the first tick the offset opens up past a threshold.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.executor import _update_world, _with_park        # noqa: E402
from apparecchiato.kinematics import JOINT_NAMES                    # noqa: E402
from apparecchiato.planner import build_planner                     # noqa: E402
from apparecchiato.scheduler import schedule                        # noqa: E402
from apparecchiato.skills import build_motion, GRIPPER_OPEN         # noqa: E402
from apparecchiato.sim.layout import sample_scene, drawer_content_pos  # noqa: E402

DEFAULT = ("Open the top drawer, set the plate, fork and spoon on the table, "
           "put the mug beside them and pour water into the mug.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=0,
                    help="0 = watch every step until the object is lost")
    ap.add_argument("--arm", default="",
                    help="arm whose grasp site to measure from; default: the "
                         "arm of the current waypoint")
    ap.add_argument("--object", required=True)
    ap.add_argument("--tol", type=float, default=0.010)
    args = ap.parse_args()

    import mujoco
    from apparecchiato.sim.env import TableEnv

    spec = sample_scene(args.seed)
    steps = schedule(build_planner("rules").plan(DEFAULT, spec), spec)
    env = TableEnv(sample_scene(args.seed))
    world = {o.name: (drawer_content_pos(env.spec, o.name) if o.inside_drawer
                      else o.grasp_point.copy()) for o in env.spec.objects}
    hold = {"A": env.arm_q("A"), "B": env.arm_q("B")}
    grip = {"A": GRIPPER_OPEN, "B": GRIPPER_OPEN}
    m, d = env.model, env.data
    armed = [False]    # only report a LOSS, i.e. after the object was in the jaws

    def bname(gid):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or "?"

    for st in steps:
        motion = _with_park(build_motion(st, env.spec, world), st.arms, hold, grip)
        watch = args.step in (0, st.order)
        fired = False
        tick = 0
        for arm, q, g, label in motion.trajectory(hold):
            hold[arm], grip[arm] = q, g
            env.set_arm_target(arm, q, g)
            other = "B" if arm == "A" else "A"
            env.set_arm_target(other, hold[other], grip[other])
            env.step()
            tick += 1
            if not watch or fired:
                continue
            probe_arm = args.arm or arm
            off = float(np.linalg.norm(env.grasp_site(probe_arm)
                                       - env.object_pos(args.object)))
            if off <= args.tol:
                armed[0] = True          # it is in the jaws; now watch for it leaving
                continue
            if not armed[0]:
                continue
            fired = True
            arm = probe_arm
            print(f"step {st.order} '{label}' tick {tick}: "
                  f"site->{args.object} opened to {off * 1000:.1f} mm  "
                  f"grip cmd {g * 1000:.1f} mm")
            print(f"  {args.object} at {np.round(env.object_pos(args.object), 4)} "
                  f"yaw {np.degrees(env.object_yaw(args.object)):+.1f} deg")
            for j in JOINT_NAMES + ("left_jaw", "right_jaw"):
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm{arm}_{j}")
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"arm{arm}_{j}")
                if aid < 0 or jid < 0:
                    continue
                print(f"  {j:14s} cmd={float(d.ctrl[aid]):+.4f} "
                      f"act={float(d.qpos[m.jnt_qposadr[jid]]):+.4f} "
                      f"force={float(d.actuator_force[aid]):+7.2f}")
            print("  contacts:")
            for c in range(d.ncon):
                con = d.contact[c]
                b1, b2 = bname(con.geom1), bname(con.geom2)
                if args.object in (b1, b2) or f"arm{arm}" in b1 or f"arm{arm}" in b2:
                    print(f"    {b1:22s} <-> {b2:22s} dist={float(con.dist) * 1000:+6.2f} mm")
        env.settle(0.25)
        _update_world(env, st, world)
        if fired or (watch and args.step):
            if not fired:
                print(f"step {st.order}: {args.object} never left the jaws")
            env.close()
            return 0
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
