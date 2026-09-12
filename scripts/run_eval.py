#!/usr/bin/env python3
"""Evaluate across randomised seeds and write the report the submission cites.

  python scripts/run_eval.py --seeds 0-9 --planner rules --out out/eval
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.eval.run_eval import evaluate, DEFAULT_INSTRUCTION   # noqa: E402


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,7 (default: 0-9)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--out", default="out/eval")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--camera", default="cinematic")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--detector", default=None,
                    help="close the perception loop: 'colour' or 'openvino'. "
                         "Without it, object positions come from simulator state "
                         "and the run measures planning and control, not vision.")
    ap.add_argument("--perception-camera", default="overhead")
    ap.add_argument("--perceive", default=None,
                    help="comma-separated objects perception is allowed to drive, "
                         "e.g. 'bottle,mug'. Everything else keeps simulator "
                         "state. Use it to report which objects the detector can "
                         "actually place, instead of one pass/fail for all five.")
    ap.add_argument("--no-markers", action="store_true",
                    help="drop the goal-slot overlay -- for the hero shot, where "
                         "the laid table should read without annotation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = evaluate(
        parse_seeds(args.seeds),
        instruction=args.instruction,
        planner_spec=args.planner,
        record_dir=None if args.no_video else os.path.join(args.out, "video"),
        record_camera=None if args.no_video else args.camera,
        max_seconds=args.max_seconds,
        markers=not args.no_markers,
        detector_spec=args.detector,
        perception_camera=args.perception_camera,
        perceive_only=({s.strip() for s in args.perceive.split(",") if s.strip()}
                       if args.perceive else None),
    )
    with open(os.path.join(args.out, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(summary.to_json())
    with open(os.path.join(args.out, "eval_report.md"), "w", encoding="utf-8") as fh:
        fh.write(summary.to_markdown())
    print("\n" + summary.to_markdown())
    print(f"\nwrote {args.out}/eval_report.json and eval_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
