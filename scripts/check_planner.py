#!/usr/bin/env python3
"""Does the quantised VLM actually produce a valid, executable plan?

A model that loads and emits tokens is not a planner. This runs the real
prompt through the OpenVINO IR, parses the answer, validates it against the
scene's own reachability, and prints the resulting task graph -- so the claim
"the planner runs on Intel silicon" is backed by a plan the scheduler accepted
rather than by a latency number alone.

  python scripts/check_planner.py
  python scripts/check_planner.py --device CPU --seeds 2 --show-raw
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.planner import build_planner                       # noqa: E402
from apparecchiato.scheduler import schedule                          # noqa: E402
from apparecchiato.sim.layout import sample_scene                     # noqa: E402

DEFAULT = ("Open the top drawer, set the plate, fork and spoon on the table, "
           "put the mug beside them and pour water into the mug.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--instruction", default=DEFAULT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--show-raw", action="store_true",
                    help="print the model's raw output, which is where a bad "
                         "plan is actually diagnosed")
    ap.add_argument("--with-image", action="store_true",
                    help="also feed the rendered overhead frame, exercising the "
                         "vision tower rather than the language model alone")
    args = ap.parse_args()

    planner = build_planner("vlm", backend="openvino", model=args.model,
                            device=args.device, max_new_tokens=args.max_new_tokens)
    print(f"model   {planner.model}\ndevice  {planner.device}\n")

    ok = 0
    for seed in range(args.seeds):
        spec = sample_scene(seed)
        images = None
        if args.with_image:
            from apparecchiato.sim.env import TableEnv
            env = TableEnv(sample_scene(seed))
            env.settle(0.4)
            images = [env.render("overhead")]
            env.close()
        try:
            graph = planner.plan(args.instruction, spec, images=images)
        except Exception as exc:                                       # noqa: BLE001
            print(f"seed {seed}: FAILED -- {type(exc).__name__}: {exc}")
            if args.show_raw and planner.last_raw is not None:
                print("--- raw model output ---")
                print(planner.last_raw[:2000])
                print("--- end ---")
            continue
        steps = schedule(graph, spec)
        ok += 1
        print(f"seed {seed}: {len(graph.nodes)} nodes -> {len(steps)} scheduled steps "
              f"in {planner.last_latency_s:.1f} s")
        for st in steps:
            print(f"   {st.order:2d}. {st.label}")
        if args.show_raw:
            print("--- raw model output ---")
            print((planner.last_raw or "")[:2000])
            print("--- end ---")

    print(f"\n{ok}/{args.seeds} seeds produced a valid, schedulable plan")
    return 0 if ok == args.seeds else 1


if __name__ == "__main__":
    raise SystemExit(main())
