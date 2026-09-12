#!/usr/bin/env python3
"""Does the cutlery actually travel with the drawer when it opens?

The planner assumes it does: ObjectSpec.pos for cutlery is the position it will
have once the drawer is OPEN, and the whole pick depends on the drawer's floor
friction carrying it there. If an item is left behind, every later step aims at
a place it never reached.

  python scripts/probe_drawer.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.sim.layout import (                                # noqa: E402
    sample_scene, DRAWER_OPEN_TRAVEL, DRAWER_FLOOR_TOP, OBJECT_HALF_H,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco
    from apparecchiato.sim.env import TableEnv

    spec = sample_scene(args.seed)
    env = TableEnv(spec)
    env.settle(0.8)

    def show(when: str):
        print(f"\n{when}")
        print(f"  drawer opened {env.drawer_open() * 1000:5.1f} mm")
        for name in ("spoon", "fork"):
            o = spec.by_name(name)
            want_z = DRAWER_FLOOR_TOP + OBJECT_HALF_H[o.kind] * o.scale
            p = env.object_pos(name)
            print(f"  {name:6s} at {np.round(p, 4)}   "
                  f"spec says open-drawer y={o.pos[1]:.3f} z={want_z:.4f}")

    show("after settling, drawer shut")

    # Drive the drawer joint directly. This isolates "does the floor carry the
    # cutlery" from "can the arm pull the knob" -- two failures that look the
    # same from the outside.
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    adr = env.model.jnt_qposadr[jid]
    steps = 400
    for i in range(steps):
        env.data.qpos[adr] = DRAWER_OPEN_TRAVEL * (i + 1) / steps
        mujoco.mj_step(env.model, env.data)
    env.settle(0.6)
    show("after driving the drawer fully open")

    print("\ncontacts on the cutlery:")
    for c in range(env.data.ncon):
        con = env.data.contact[c]
        b1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                               env.model.geom_bodyid[con.geom1]) or "?"
        b2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                               env.model.geom_bodyid[con.geom2]) or "?"
        if "spoon" in (b1, b2) or "fork" in (b1, b2):
            print(f"  {b1:12s} <-> {b2:12s} dist={float(con.dist) * 1000:+6.2f} mm")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
