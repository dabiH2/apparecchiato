#!/usr/bin/env python3
"""Write out what the colour detector is actually looking at.

Tuning colour thresholds by watching an error number go up and down is guessing.
This writes the rendered frame and one image per colour mask, with the chosen
blob centroid and the true object position marked, so the failure is visible
rather than inferred.

  python scripts/dump_masks.py --seed 0 --out out/masks
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.perception.detect import (                          # noqa: E402
    ColourDetector, OracleDetector, _largest_blobs,
)
from apparecchiato.sim.layout import sample_scene                      # noqa: E402


def project(env, camera: str, world_xyz, shape):
    """World point -> pixel. The inverse of detect.backproject, for marking truth."""
    import mujoco
    h, w = shape[0], shape[1]
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    cam_pos = np.array(env.data.cam_xpos[cid], dtype=float)
    R = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)
    fovy = float(env.model.cam_fovy[cid]) * np.pi / 180.0
    f = (h / 2.0) / np.tan(fovy / 2.0)
    d = R.T @ (np.asarray(world_xyz, dtype=float) - cam_pos)
    if d[2] >= -1e-9:
        return None
    u = w / 2.0 + f * (d[0] / -d[2])
    v = h / 2.0 - f * (d[1] / -d[2])
    return u, v


def cross(img, u, v, rgb, size=7):
    h, w = img.shape[:2]
    u, v = int(round(u)), int(round(v))
    for d in range(-size, size + 1):
        for (yy, xx) in ((v + d, u), (v, u + d)):
            if 0 <= yy < h and 0 <= xx < w:
                img[yy, xx] = rgb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--render", type=int, default=640)
    ap.add_argument("--out", default="out/masks")
    args = ap.parse_args()

    from PIL import Image
    from apparecchiato.sim.env import TableEnv

    os.makedirs(args.out, exist_ok=True)
    env = TableEnv(sample_scene(args.seed), render_size=(args.render, args.render))
    env.settle(0.6)
    frame = env.render(args.camera)
    img = np.asarray(frame, dtype=float) / 255.0
    value = img.max(axis=2)

    det = ColourDetector()
    truth = OracleDetector().detect(env, args.camera)

    annotated = np.array(frame)
    for name, t in truth.items():
        p = project(env, args.camera, t.pos, img.shape)
        if p:
            cross(annotated, p[0], p[1], (255, 0, 0))
    Image.fromarray(annotated).save(os.path.join(args.out, f"seed{args.seed}_frame.png"))

    h, s, v = det._hsv(img)
    for label, chan in (("hue", h), ("sat", s), ("val", v)):
        Image.fromarray((np.clip(chan, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"seed{args.seed}_{label}.png"))

    print(f"seed {args.seed}: red crosses in *_frame.png are TRUE positions\n")
    for kind, mask in det.masks(img).items():
        n = int(mask.sum())
        vis = np.array(frame)
        vis[~mask] = (vis[~mask] * 0.18).astype(np.uint8)
        blobs = _largest_blobs(mask, det.min_pixels, k=6, value=value)
        for b in blobs:
            cross(vis, b["x"], b["y"], (0, 255, 255), size=5)
        Image.fromarray(vis).save(os.path.join(args.out, f"seed{args.seed}_{kind}.png"))
        desc = "  ".join(f"({b['x']:.0f},{b['y']:.0f}) n={b['n']} v={b['value']:.2f} "
                         f"aspect={b['aspect']:.2f}" for b in blobs)
        print(f"{kind:12s} {n:7d} px  blobs: {desc or 'none'}")

    print(f"\nfor reference, TRUE pixel positions:")
    for name, t in truth.items():
        p = project(env, args.camera, t.pos, img.shape)
        print(f"  {name:8s} ({p[0]:.0f},{p[1]:.0f})" if p else f"  {name:8s} off-frame")
    env.close()
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
