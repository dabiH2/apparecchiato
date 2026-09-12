#!/usr/bin/env python3
"""Does the exported detector actually find the objects on the table?

The export script proves the IR matches torch. That is a different question from
whether the detector WORKS, and this is the one that decides whether the
perception path can be used at all: it runs the OpenVINO detector on rendered
scenes and compares every detection against the simulator's own ground truth --
the oracle detector -- reporting per-object recall and back-projection error in
millimetres on the table plane.

  python scripts/check_detector.py --seeds 3
  python scripts/check_detector.py --device CPU --precision f32 --threshold 0.1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.perception.detect import (                          # noqa: E402
    OpenVINODetector, OracleDetector, build_detector,
)
from apparecchiato.sim.layout import sample_scene                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--detector", default="openvino",
                    choices=("openvino", "colour"),
                    help="which perception path to measure against the oracle")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default=None, help="f32 / bf16 / f16; default: plugin's choice")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--render", type=int, default=960,
                    help="square render size fed to the detector")
    ap.add_argument("--prompts", default=None,
                    help="semicolon-separated replacement prompts, in the SAME "
                         "order and count as the exported ones. The text tokens "
                         "are a graph INPUT, not baked into the weights, so "
                         "prompts can be retuned without re-exporting anything.")
    args = ap.parse_args()

    from apparecchiato.sim.env import TableEnv

    if args.detector == "openvino":
        det = OpenVINODetector(model=args.model, device=args.device,
                               score_threshold=args.threshold,
                               precision_hint=args.precision)
    else:
        det = build_detector(args.detector)
    oracle = OracleDetector()
    if args.prompts:
        det._load()
        new = tuple(p.strip() for p in args.prompts.split(";") if p.strip())
        if len(new) != len(det.prompts):
            print(f"need exactly {len(det.prompts)} prompts, got {len(new)}",
                  file=sys.stderr)
            return 2
        det.override_prompts(new)
        print("prompts: " + " | ".join(new) + "\n")

    hits: dict[str, int] = {}
    errs: dict[str, list] = {}
    seen: dict[str, int] = {}
    latencies = []

    for seed in range(args.seeds):
        # Render square and large. The default 640x480 is for watching; for
        # detection it costs twice over -- the processor squashes a 4:3 frame
        # into a square input, distorting every object, and a 26 mm plate across
        # a 680 mm field of view lands on well under one 32 px patch.
        env = TableEnv(sample_scene(seed), render_size=(args.render, args.render))
        env.settle(0.6)
        truth = oracle.detect(env, args.camera)
        found = det.detect(env, args.camera)
        latencies.append(det.last_latency_s)
        line = []
        for name, t in truth.items():
            seen[name] = seen.get(name, 0) + 1
            d = found.get(name)
            if d is None:
                line.append(f"{name}: MISS")
                continue
            hits[name] = hits.get(name, 0) + 1
            # Back-projection error in the table plane. z is not compared: the
            # detector assumes objects rest on the table, which is the whole
            # point of back-projecting onto a known plane.
            e = float(np.linalg.norm(d.pos[:2] - t.pos[:2])) * 1000.0
            errs.setdefault(name, []).append(e)
            line.append(f"{name}: {e:5.0f}mm p={d.score:.2f}")
        print(f"seed {seed}: " + "  ".join(line))
        env.close()

    if isinstance(det, OpenVINODetector):
        print(f"\ndetector   {det.name}  {det.model}  device={det.device}  "
              f"precision={det.precision_hint or 'plugin default'}")
    else:
        print(f"\ndetector   {det.name}")
    print(f"latency    {np.mean(latencies) * 1000:.0f} ms/frame "
          f"(median {np.median(latencies) * 1000:.0f} ms over {len(latencies)} frames)")
    print(f"\n{'object':10s} {'recall':>8s} {'median err':>12s} {'worst':>8s}")
    for name in sorted(seen):
        n, h = seen[name], hits.get(name, 0)
        if h:
            e = errs[name]
            print(f"{name:10s} {h}/{n:<6d} {np.median(e):9.0f} mm {max(e):6.0f} mm")
        else:
            print(f"{name:10s} {h}/{n:<6d} {'--':>12s} {'--':>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
