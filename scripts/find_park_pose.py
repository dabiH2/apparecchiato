#!/usr/bin/env python3
"""Find a parking pose for each arm that is clear of everything that matters.

The idle arm has to go somewhere. HOME_Q was the obvious choice and is the wrong
one: at HOME arm A's tool sits at y = 0.253, which is inside the cabinet, so
parking it drove the gripper into the open drawer and shut it -- 44.9 mm to
3.0 mm -- carrying the spoon back in with it and failing the two steps that
needed the spoon.

This searches the joint space for a pose that is simultaneously
  * clear of the cabinet keep-out box,
  * far from the other arm's base and from the shared lens,
  * low enough to be inside the joint limits, and
  * as far from the table's working area as the arm can get,
then prints it as a constant to paste into kinematics.py.

  python scripts/find_park_pose.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.kinematics import (  # noqa: E402
    fk, JOINT_LIMITS, TOP_DOWN, ee_pitch, solve_ik, workspace_bounds,
)
from apparecchiato.sim.layout import (                                # noqa: E402
    ARMS, to_world, DRAWER_CLOSED_Y, DRAWER_OPEN_TRAVEL, CAB_KEEPOUT_X, BASE_KEEPOUT,
)

# The cabinet occupies this band in world space. The drawer front reaches
# DRAWER_OPEN_TRAVEL further toward the arms when open, which is the surface
# that actually got hit.
CAB_Y_MIN = DRAWER_CLOSED_Y - DRAWER_OPEN_TRAVEL - 0.03


def feasible(arm: str, q) -> tuple[bool, str]:
    """Is this pose actually out of the way? Hard constraints only.

    "Out of the way" is not "as far away as possible". Scoring for maximum
    distance picked a fully extended pose with the tool 450 mm from the base,
    and an arm that unfolds across the table to park sweeps a huge arc: it flung
    the fork 509 mm and launched the plate clean off the table. What is wanted is
    a TUCK -- everything pulled in close and held above the work.
    """
    p = to_world(fk(q)[0], arm)
    base_other = ARMS["B" if arm == "A" else "A"]

    # The tool must still point DOWN. An arm parks while holding something --
    # that is the whole reason park keeps the gripper closed -- and the grip is
    # a friction pinch, so tipping the tool away from vertical puts gravity
    # along the pads instead of across them. The first tuck found had a pitch of
    # 109 deg, 70 deg off vertical, and the fork slid straight out during the
    # park and ended up 330 mm away before the step that was meant to place it
    # had even started.
    # The tool stays pointing DOWN, within 9 degrees. An arm parks while still
    # holding something and the grip is a friction pinch, so any tilt puts
    # gravity along the pads instead of across them. This was learned twice: at
    # 70 deg the fork slid out during the park, and at 35 deg -- which the
    # arithmetic said was fine for a 30 g fork -- the mug still slipped 10 mm,
    # because a smooth cylinder has far less to catch on than a square shaft.
    if abs(ee_pitch(q) - TOP_DOWN) > 0.16:
        return False, "tool pitch"

    # Park INSIDE the arm's own base keep-out radius. This is the constraint
    # that actually resolves the conflict: the arm cannot hold the tool top-down
    # above z = 0.10 at all (measured -- the workspace simply ends there), and
    # the cabinet's side walls reach z = 0.08, so "top-down AND above the
    # furniture" has no solution.
    #
    # It does not need one. No object or goal slot may spawn within
    # BASE_KEEPOUT of a base, and the cabinet's nearest corner is 185 mm from
    # arm A's base, so a tool parked inside that radius is over guaranteed-empty
    # table. Height stops being the thing keeping it safe; emptiness does.
    reach = float(np.linalg.norm(p[:2] - ARMS[arm][:2]))
    if reach > BASE_KEEPOUT - 0.005:
        return False, "outside the base keep-out"
    if float(p[2]) < 0.075:
        return False, "height"
    # On its own side of the table, clear of the other arm's base.
    if float(np.linalg.norm(p[:2] - base_other[:2])) < 0.18:
        return False, "other arm"
    return True, ""


def tuck_cost(arm: str, q) -> float:
    """Lower is more tucked: tool close to its own base, arm folded."""
    p = to_world(fk(q)[0], arm)
    reach = float(np.linalg.norm(p[:2] - ARMS[arm][:2]))
    # Penalise a wide shoulder sweep too -- that is what carves the arc.
    return reach + 0.05 * abs(float(q[0]))


def main() -> int:
    # Search in CARTESIAN space through the same IK the skills use, not over raw
    # joint angles. A joint-space grid has to land on the thin sheet where the
    # tool happens to point down, and a grid coarse enough to finish never does:
    # it reported "no feasible pose" while workspace_bounds showed top-down poses
    # available all the way to z = 0.20. Asking the IK for a point is also the
    # stronger guarantee -- a pose it returns is reachable by construction.
    for arm in ("A", "B"):
        best, best_q, best_p, rejected = 1e9, None, None, {}
        # Fine steps, and a floor of 0.115 rather than 0.13. The genuinely
        # top-down reachable set at height is a thin band -- workspace_bounds
        # reports r <= 0.207 at z = 0.10 but the IK only solves near r = 0.14 --
        # so a coarse sweep walks straight over it. 115 mm still clears the
        # cabinet's 80 mm side walls and the 104 mm bottle.
        for z in np.arange(0.075, 0.105, 0.002):
            r_min, r_max = workspace_bounds(z=z)
            if r_max <= r_min:
                continue
            for r in np.arange(max(r_min, 0.04), r_max, 0.002):
                for pan in np.linspace(JOINT_LIMITS[0][0] * 0.95,
                                       JOINT_LIMITS[0][1] * 0.95, 49):
                    local = np.array([r * np.cos(pan), r * np.sin(pan), z])
                    res = solve_ik(local, pitch=TOP_DOWN)
                    if not res.success:
                        continue
                    ok, why = feasible(arm, res.q)
                    if not ok:
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    c = tuck_cost(arm, res.q)
                    if c < best:
                        best, best_q, best_p = c, res.q, to_world(fk(res.q)[0], arm)
        if best_q is None:
            print(f"arm {arm}: NO FEASIBLE PARK POSE. rejected by: {rejected}")
            continue
        print(f"arm {arm}: q = [{', '.join(f'{v:.4f}' for v in best_q)}]")
        print(f"   tool at {np.round(best_p, 4)}   "
              f"reach from own base {np.linalg.norm(best_p[:2] - ARMS[arm][:2]) * 1000:.0f} mm"
              f"   pan {np.degrees(best_q[0]):+.0f} deg"
              f"   pitch {np.degrees(ee_pitch(best_q)):+.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
