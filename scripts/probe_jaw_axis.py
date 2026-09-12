#!/usr/bin/env python3
"""Measure the jaw axis in the simulator, instead of trusting the derivation.

cross_shaft_roll() was derived on paper as q4 = q0 - yaw. A sign error there is
invisible in a static grasp -- the jaw axis is only defined mod 180 deg, and a
42 deg misalignment still straddles a 16 mm x 60 mm bar -- but it shows up the
moment the arm CARRIES the object, because the wrist then rotates the object the
wrong way. This script drives the arm to a grid of (q0, q4) and reports the true
world heading of the line joining the two finger bodies.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.sim.env import TableEnv                      # noqa: E402
from apparecchiato.sim.layout import sample_scene               # noqa: E402


def main() -> int:
    import mujoco
    env = TableEnv(sample_scene(0))
    m, d = env.model, env.data
    print(f"{'q0':>8} {'q4':>8} | {'jaw axis':>10} {'q0-q4':>8} {'q0+q4':>8}")
    for q0 in (-0.6, -0.3, 0.0, 0.3):
        for q4 in (-0.6, -0.3, 0.0, 0.3, 0.6):
            q = np.array([q0, 0.55, 0.95, 1.15, q4])
            env.set_arm_target("A", q, 0.035)
            env.settle(1.2)
            p = {}
            for side in ("left", "right"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"armA_{side}_finger")
                p[side] = np.array(d.xpos[bid])
            v = p["left"] - p["right"]
            # The jaw axis is perpendicular to the shaft a cross-shaft grasp
            # straddles, so report the SHAFT heading it corresponds to.
            jaw = np.degrees(np.arctan2(v[1], v[0]))
            shaft = (jaw + 90.0 + 90.0) % 180.0 - 90.0
            print(f"{np.degrees(q0):8.1f} {np.degrees(q4):8.1f} | {shaft:10.1f} "
                  f"{np.degrees(q0 - q4):8.1f} {np.degrees(q0 + q4):8.1f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
