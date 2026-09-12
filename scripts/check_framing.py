"""Is the demo shot actually watchable? Measured, not eyeballed.

Reports, for a rendered frame: how much of it is near-black, how bright the
median pixel is, and where the table's bounding box sits in the frame. Written
after a review found ~60% of every recorded frame was unlit floor with the table
off to one side at a third of the width -- the kind of thing that is obvious on
screen and invisible from inside the code.

    python scripts/check_framing.py --seeds 0,3,7
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from apparecchiato.sim.env import TableEnv
from apparecchiato.sim.layout import sample_scene, TABLE_HALF_X


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,3,7")
    ap.add_argument("--camera", default="cinematic")
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--save", default=None, help="write each frame as a PNG here")
    a = ap.parse_args()

    print(f"{'seed':>5} {'dark %':>7} {'median L':>9} {'table x':>16} {'table y':>16}")
    for seed in [int(s) for s in a.seeds.split(",")]:
        spec = sample_scene(seed)
        env = TableEnv(spec, render_size=(960, 720))
        env.settle(a.settle)
        img = np.asarray(env.render(a.camera), dtype=float)
        env.close()

        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        dark = float((lum < 60).mean())

        # The table is the one large mid-saturation warm region. Find it by its
        # own material colour rather than by guessing at pixel coordinates.
        # The backdrop is also a warm tone, so match on the table AND reject
        # anything closer to the wall -- otherwise the whole frame "is table".
        table = np.asarray(spec.table_rgba[:3], dtype=float) * 255.0
        wall = np.asarray(spec.wall_rgba[:3], dtype=float) * 255.0
        d_table = np.abs(img - table).max(axis=2)
        d_wall = np.abs(img - wall).max(axis=2)
        near = (d_table < 40) & (d_table < d_wall)
        ys, xs = np.nonzero(near)
        if len(xs) < 500:
            span = ("no table found", "no table found")
        else:
            h, w = lum.shape
            span = (f"{xs.min() / w:.2f}-{xs.max() / w:.2f}",
                    f"{ys.min() / h:.2f}-{ys.max() / h:.2f}")
        print(f"{seed:>5} {100 * dark:>6.1f}% {np.median(lum):>9.0f} "
              f"{span[0]:>16} {span[1]:>16}")

        if a.save:
            from PIL import Image                                # noqa: PLC0415
            d = pathlib.Path(a.save)
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.astype(np.uint8)).save(d / f"frame{seed:03d}.png")

    print("\nWanted: dark below ~30%, median luminance above ~70, and the table "
          "spanning most of the frame in x. Anything else and the viewer is "
          "looking at a room, not at a robot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
