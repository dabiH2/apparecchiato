"""Per-seed shot sheet for the demo video.

For every seed: what the scheduler decided, which arm does what, where the
hand-off lands, and the randomisation the clip is meant to show off. This is
what the captions in docs/04-demo-video-script.md are cut from, so it is
generated rather than transcribed by hand.

    python scripts/clip_sheet.py --seeds 0-9 --out out/clip_sheet.md
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from apparecchiato.eval.run_eval import DEFAULT_INSTRUCTION
from apparecchiato.planner import build_planner
from apparecchiato.scheduler import schedule, parallel_fraction
from apparecchiato.sim.layout import sample_scene, reaching_arms


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--out", default="out/clip_sheet.md")
    a = ap.parse_args()

    planner = build_planner(a.planner)
    lines = ["# Clip sheet", "",
             f"Instruction: `{a.instruction}`", ""]
    for seed in _seeds(a.seeds):
        scene = sample_scene(seed)
        graph = planner.plan(a.instruction, scene)
        steps = schedule(graph, scene)
        o = {x.name: x for x in scene.objects}

        hand = [s for s in steps if s.node.skill == "handoff"]
        ho = ", ".join(f"{s.node.args['object']} {s.arms[0]}->{s.arms[1]}" for s in hand)
        lines += [
            f"## seed {seed:03d}  (`out/eval_vlm/video/seed{seed:03d}.mp4`)",
            "",
            f"- **{len(steps)} steps**, {parallel_fraction(steps):.0%} parallelisable",
            f"- **hand-off:** {ho or 'none'}"
            + (f" at {np.round(scene.handoff_point, 3).tolist()}" if hand else ""),
            f"- drawer at x={scene.by_name('spoon').pos[0] + 0.032:+.3f}, "
            f"cutlery reached by arm A",
            "- randomisation: "
            + ", ".join(f"{n} {o[n].scale:.2f}x {o[n].mass * 1000:.0f} g mu={o[n].friction:.2f}"
                        for n in ("plate", "mug", "bottle")),
            f"- light at {np.round(scene.light_pos, 2).tolist()} "
            f"intensity {scene.light_intensity:.2f}; "
            f"table rgb {np.round(scene.table_rgba[:3], 2).tolist()}, "
            f"wall rgb {np.round(scene.wall_rgba[:3], 2).tolist()}",
            "",
            "| # | arms | action | why this arm |",
            "| --- | --- | --- | --- |",
        ]
        for s in steps:
            obj = s.node.args.get("object") or s.node.args.get("source") or ""
            why = ""
            if obj and obj in o:
                reach = reaching_arms(np.asarray(o[obj].grasp_point))
                why = ("both arms reach it" if len(reach) == 2
                       else f"only arm {reach[0]} reaches it" if reach else "")
            lines.append(f"| {s.order} | {'+'.join(s.arms)} | {s.label.split('] ')[-1]} "
                         f"| {s.node.rationale or why} |")
        lines.append("")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
