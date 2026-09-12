#!/usr/bin/env python3
"""Run one episode end to end: instruction -> plan -> dual-arm execution.

Examples
--------
  python scripts/run_episode.py --seed 3 --plan-only
  python scripts/run_episode.py --seed 3 --planner rules --record out/seed3.mp4
  python scripts/run_episode.py --voice mic --planner vlm+rules
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.planner import build_planner                        # noqa: E402
from apparecchiato.scheduler import schedule, describe, parallel_fraction   # noqa: E402
from apparecchiato.sim.layout import sample_scene                      # noqa: E402
from apparecchiato.sim.scene import write_mjcf                         # noqa: E402

DEFAULT = ("Open the top drawer, set the plate, fork and spoon on the table, "
           "put the mug beside them and pour water into the mug.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default=DEFAULT)
    ap.add_argument("--planner", default="rules",
                    help="rules | vlm | vlm+rules (default: rules)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the plan and schedule; do not launch MuJoCo")
    ap.add_argument("--dump-mjcf", metavar="PATH", help="write the generated scene XML")
    ap.add_argument("--record", metavar="PATH", help="write an mp4 of the episode")
    ap.add_argument("--view", action="store_true",
                    help="watch it live in the MuJoCo viewer (close the window to stop)")
    ap.add_argument("--slow", type=float, default=1.0,
                    help="with --view, playback speed (0.5 = half speed)")
    ap.add_argument("--camera", default="cinematic")
    ap.add_argument("--voice", choices=("mic", "file"), help="speak the command instead")
    ap.add_argument("--voice-file", help="WAV file when --voice file")
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    args = ap.parse_args()

    instruction = args.instruction
    if args.voice:
        from apparecchiato.voice import transcribe_file, transcribe_microphone, VoiceError
        try:
            tr = (transcribe_file(args.voice_file, language=args.language)
                  if args.voice == "file"
                  else transcribe_microphone(language=args.language,
                                             on_partial=lambda t: print(f"  ... {t}",
                                                                        flush=True)))
        except VoiceError as exc:
            print(f"voice input failed: {exc}", file=sys.stderr)
            return 2
        instruction = tr.text
        print(f'heard: "{instruction}"  ({tr.audio_seconds:.1f}s audio, '
              f'{tr.latency_s:.1f}s round trip)\n')

    spec = sample_scene(args.seed)
    if args.dump_mjcf:
        write_mjcf(spec, args.dump_mjcf)
        print(f"wrote {args.dump_mjcf}")

    graph = build_planner(args.planner).plan(instruction, spec)
    print(f"plan from {graph.source}: {len(graph)} nodes")
    for n in graph.notes:
        print(f"  note: {n}")
    steps = schedule(graph, spec)
    print(describe(steps))
    print(f"\n{parallel_fraction(steps):.0%} of steps can run on both arms at once")

    if args.plan_only:
        return 0

    from apparecchiato.executor import run_episode
    from apparecchiato.sim.env import TableEnv, MuJoCoMissing
    try:
        env = TableEnv(sample_scene(args.seed))
    except MuJoCoMissing as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3

    print("\nexecuting:")
    viewer_cm = None
    if args.view:
        import mujoco.viewer
        viewer_cm = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        v = viewer_cm.__enter__() if viewer_cm else None
        if v is not None:
            print("  (watching live -- close the viewer window to stop early)")
        rep = run_episode(env, steps, instruction=instruction, planner=graph.source,
                          record=args.camera if args.record else None,
                          max_seconds=args.max_seconds, verbose=True,
                          viewer=v, realtime=args.view)
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
    print("\n" + rep.summary())

    if args.record and rep.frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)) or ".", exist_ok=True)
        import imageio.v2 as imageio
        imageio.mimsave(args.record, rep.frames, fps=30, quality=8)
        print(f"wrote {args.record} ({len(rep.frames)} frames)")
    env.close()
    return 0 if rep.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
