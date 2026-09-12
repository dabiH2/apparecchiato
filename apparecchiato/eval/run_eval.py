"""Randomised multi-seed evaluation -- the number the submission is judged on.

The brief asks for results across 10 randomised seeds. This produces exactly
that, plus the breakdown that makes the number useful: per-subgoal success
rates, which step failed first, and how much of the plan could run on both arms
at once.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from ..executor import run_episode, EpisodeReport
from ..planner import build_planner
from ..scheduler import schedule, parallel_fraction
from ..sim.env import TableEnv
from ..sim.layout import sample_scene

DEFAULT_INSTRUCTION = (
    "Open the top drawer, set the plate, fork and spoon on the table, "
    "put the mug beside them and pour water into the mug."
)


@dataclass
class EvalSummary:
    seeds: list[int]
    instruction: str
    planner: str
    success_rate: float
    subgoal_rates: dict
    mean_steps: float
    mean_wall_s: float
    mean_parallel_fraction: float
    failures: list = field(default_factory=list)
    per_seed: list = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=float)

    def to_markdown(self) -> str:
        lines = [
            "# Evaluation -- randomised seeds",
            "",
            f"- Instruction: `{self.instruction}`",
            f"- Planner: `{self.planner}`",
            f"- Seeds: {self.seeds}",
            "",
            f"**Task success: {self.success_rate:.0%}** "
            f"({sum(1 for s in self.per_seed if s['success'])}/{len(self.seeds)} seeds)",
            "",
            f"- Mean steps per episode: {self.mean_steps:.1f}",
            f"- Mean wall clock per episode: {self.mean_wall_s:.1f} s",
            f"- Mean parallelisable share of the plan: {self.mean_parallel_fraction:.0%}",
            "",
            "## Subgoal success",
            "",
            "| Subgoal | Rate |",
            "| --- | --- |",
        ]
        for k, v in sorted(self.subgoal_rates.items(), key=lambda kv: kv[1]):
            lines.append(f"| {k} | {v:.0%} |")
        if self.failures:
            lines += ["", "## First failure per failed seed", "",
                      "| Seed | Step | Detail |", "| --- | --- | --- |"]
            for f in self.failures:
                lines.append(f"| {f['seed']} | {f['label']} | {f['detail']} |")
        return "\n".join(lines)


def evaluate(seeds=range(10), *, instruction: str = DEFAULT_INSTRUCTION,
             planner_spec: str = "vlm+rules", record_dir: str | None = None,
             record_camera: str | None = "cinematic", verbose: bool = True,
             max_seconds: float = 180.0) -> EvalSummary:
    seeds = list(seeds)
    planner = build_planner(planner_spec)
    reports: list[EpisodeReport] = []
    pfracs: list[float] = []

    for seed in seeds:
        spec = sample_scene(seed)
        graph = planner.plan(instruction, spec)
        steps = schedule(graph, spec)
        pfracs.append(parallel_fraction(steps))

        # Re-sample so execution starts from a pristine scene: scheduling walks a
        # believed world state forward and mutates drawer_open along the way.
        env = TableEnv(sample_scene(seed))
        rep = run_episode(env, steps, instruction=instruction, planner=graph.source,
                          record=record_camera, max_seconds=max_seconds)
        if verbose:
            print(rep.summary(), flush=True)
        if record_dir and rep.frames:
            _write_frames(rep, record_dir, seed)
        rep.frames = []
        env.close()
        reports.append(rep)

    keys = sorted({k for r in reports for k in r.subgoals})
    return EvalSummary(
        seeds=seeds,
        instruction=instruction,
        planner=planner.name,
        success_rate=float(np.mean([r.success for r in reports])) if reports else 0.0,
        subgoal_rates={k: float(np.mean([bool(r.subgoals.get(k)) for r in reports]))
                       for k in keys},
        mean_steps=float(np.mean([len(r.steps) for r in reports])) if reports else 0.0,
        mean_wall_s=float(np.mean([r.wall_s for r in reports])) if reports else 0.0,
        mean_parallel_fraction=float(np.mean(pfracs)) if pfracs else 0.0,
        failures=[{"seed": r.seed, "label": r.first_failure.label,
                   "detail": r.first_failure.detail}
                  for r in reports if not r.success and r.first_failure],
        per_seed=[{"seed": r.seed, "success": r.success, "steps": len(r.steps),
                   "wall_s": round(r.wall_s, 2), "subgoals": r.subgoals}
                  for r in reports],
    )


def _write_frames(rep: EpisodeReport, record_dir: str, seed: int) -> None:
    os.makedirs(record_dir, exist_ok=True)
    out = os.path.join(record_dir, f"seed{seed:03d}.mp4")
    try:
        import imageio.v2 as imageio                          # noqa: PLC0415
        imageio.mimsave(out, rep.frames, fps=30, quality=8)
        return
    except Exception:                                         # noqa: BLE001
        pass
    try:
        from PIL import Image                                 # noqa: PLC0415
        d = os.path.join(record_dir, f"seed{seed:03d}")
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(rep.frames):
            Image.fromarray(f).save(os.path.join(d, f"{i:05d}.png"))
    except Exception as exc:                                  # noqa: BLE001
        rep.notes.append(f"could not save frames: {exc}")
