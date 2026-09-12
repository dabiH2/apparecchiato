#!/usr/bin/env python3
"""Re-fit the link constants in apparecchiato/kinematics.py against an external SO-101 MJCF.

The project ships a self-consistent model: the MJCF is generated from the same
constants the IK uses. If you swap in the official SO-101 model instead, those
constants must be re-measured or every grasp will be off by the difference.

  python scripts/calibrate_kinematics.py --mjcf path/to/so101.xml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--prefix", default="", help="joint name prefix in that model")
    ap.add_argument("--site", default="grasp", help="site marking the grasp point")
    args = ap.parse_args()

    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("mujoco is required for calibration", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("Measured offsets along the kinematic chain (metres):\n")
    prev = None
    for j in CHAIN:
        name = args.prefix + j
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"  joint {name!r} not found -- pass --prefix", file=sys.stderr)
            return 1
        pos = np.array(data.xanchor[jid])
        if prev is not None:
            print(f"  {prev[0]:>14s} -> {j:<14s} {np.linalg.norm(pos - prev[1]):.4f}")
        prev = (j, pos)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.prefix + args.site)
    if sid >= 0 and prev is not None:
        d = np.linalg.norm(np.array(data.site_xpos[sid]) - prev[1])
        print(f"  {prev[0]:>14s} -> {'grasp point':<14s} {d:.4f}")
    else:
        print(f"\n  site {args.prefix + args.site!r} not found; measure L_TOOL by hand.")

    print("\nCopy these into L_UPPER / L_FORE / L_WRIST / L_TOOL in apparecchiato/kinematics.py,")
    print("then re-run: python -m pytest tests/test_kinematics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
