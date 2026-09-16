"""Randomised multi-seed evaluation -- the number the submission is judged on.

The brief asks for results across 10 randomised seeds. This produces exactly
that, plus the breakdown that makes the number useful: per-subgoal success
rates, which step failed first, how much of the plan the dependency graph would
permit to overlap, and -- because with a fallback chain this is the whole
question -- which planner's output was actually executed.
"""
from __future__ import annotations

import json
import os
import time
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
    # Where object positions came from. Named in the report because "100% success"
    # means something quite different with and without perception in the loop.
    perception: str = "simulator state (no perception in the loop)"
    success_rate: float = 0.0
    subgoal_rates: dict = field(default_factory=dict)
    mean_steps: float = 0.0
    mean_wall_s: float = 0.0
    mean_parallel_fraction: float = 0.0
    mean_plan_s: float = 0.0
    # Which backend's plan was actually EXECUTED, counted. With a fallback chain
    # this is the difference between "we used a VLM" and "we called one".
    plan_sources: dict = field(default_factory=dict)
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
            f"- Object positions from: **{self.perception}**",
            f"- Plan repair: **{'ON' if os.environ.get('APPARECCHIATO_VLM_REPAIR') == '1' else 'off'}**"
            f" (`--repair-vlm-plan`: the validator may delete and reorder what the "
            f"model wrote, and may never add to it; every repair is named in the "
            f"per-episode notes)",
            f"- Seeds: {self.seeds}",
            "",
            f"**Task success: {self.success_rate:.0%}** "
            f"({sum(1 for s in self.per_seed if s['success'])}/{len(self.seeds)} seeds)",
            "",
            f"- Mean steps per episode: {self.mean_steps:.1f}",
            f"- Mean wall clock per episode: {self.mean_wall_s:.1f} s "
            f"execution + {self.mean_plan_s:.1f} s planning",
            f"- Parallelisable share of the plan: {self.mean_parallel_fraction:.0%} "
            f"(steps the dependency graph permits to overlap; the executor runs "
            f"them one at a time)",
            "",
            "## Which planner's output was executed",
            "",
        ]
        if self.plan_sources:
            total = sum(self.plan_sources.values())
            for src, n in sorted(self.plan_sources.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{src}` -- {n}/{total} episodes")
            rejected = [note for s in self.per_seed for note in s.get("planner_notes", [])]
            if rejected:
                lines += ["", "Rejected upstream plans (the fallback chain recorded these):",
                          ""]
                for note in sorted(set(rejected))[:6]:
                    lines.append(f"- {note}")
        else:
            lines.append("- not recorded")

        located = [n for s in self.per_seed for n in s.get("episode_notes", [])
                   if "from truth" in n]
        if located:
            lines += ["", "## What perception actually located", "",
                      "Every object the detector placed, and how far that was from "
                      "the simulator's own answer. Objects it did not report kept "
                      "their true position -- the cutlery starts inside a shut "
                      "drawer and no camera can see it.", ""]
            per_obj: dict[str, list[float]] = {}
            for note in located:
                obj = note.split(" located", 1)[0]
                try:
                    mm = float(note.rsplit(" mm", 1)[0].rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                per_obj.setdefault(obj, []).append(mm)
            lines += ["| Object | Readings used | Mean error | Worst |",
                      "| --- | --- | --- | --- |"]
            for obj, errs in sorted(per_obj.items()):
                lines.append(f"| {obj} | {len(errs)} | "
                             f"{sum(errs) / len(errs):.0f} mm | {max(errs):.0f} mm |")
        lines += [
            "",
            "## Subgoal success",
            "",
            "| Subgoal | Rate |",
            "| --- | --- |",
        ]
        for k, v in sorted(self.subgoal_rates.items(), key=lambda kv: kv[1]):
            lines.append(f"| {k} | {v:.0%} |")
        if self.failures:
            n_fail = sum(1 for s in self.per_seed if not s.get("success")) \
                if self.per_seed else len(self.failures)
            lines += ["", "## Every failed seed", "",
                      f"{len(self.failures)} row(s) for {n_fail} failed seed(s) — "
                      "these two numbers must match, and the table below is keyed on "
                      "seeds rather than on failing steps so that they do. A seed can "
                      "run every step, have no step report a failure, and still not "
                      "meet the task criterion; those are marked `end-state`.",
                      "",
                      "| Seed | Kind | Where | Detail |", "| --- | --- | --- | --- |"]
            for f in self.failures:
                lines.append(f"| {f['seed']} | {f.get('kind', 'step')} | "
                             f"{f['label']} | {f['detail']} |")
        return "\n".join(lines)


def evaluate(seeds=range(10), *, instruction: str = DEFAULT_INSTRUCTION,
             planner_spec: str = "vlm+rules", record_dir: str | None = None,
             record_camera: str | None = "cinematic", verbose: bool = True,
             max_seconds: float = 180.0, markers: bool = True,
             detector_spec: str | None = None,
             perception_camera: str = "overhead",
             perceive_only: set | None = None) -> EvalSummary:
    seeds = list(seeds)
    planner = build_planner(planner_spec)
    detector = None
    if detector_spec:
        from ..perception.detect import build_detector          # noqa: PLC0415
        detector = build_detector(detector_spec)
    reports: list[EpisodeReport] = []
    pfracs: list[float] = []

    plan_s: list[float] = []
    sources: list[str] = []
    plan_notes: list[list[str]] = []

    for seed in seeds:
        spec = sample_scene(seed)
        # Planning is timed and its PROVENANCE is recorded. Neither was, and the
        # omission was load-bearing: with a `vlm+rules` chain the report said
        # "planner: vlm+rules" whether the VLM's plan was used or thrown away, and
        # mean_wall_s started inside run_episode, so it excluded the ~50 s the VLM
        # spent producing a plan that was then rejected. A report that cannot tell
        # you which planner actually ran is not evidence about either of them.
        t0 = time.perf_counter()
        graph = planner.plan(instruction, spec)
        plan_s.append(time.perf_counter() - t0)
        sources.append(graph.source)
        plan_notes.append(list(graph.notes))
        steps = schedule(graph, spec)
        pfracs.append(parallel_fraction(steps))

        # Re-sample so execution starts from a pristine scene: scheduling walks a
        # believed world state forward and mutates drawer_open along the way.
        env = TableEnv(sample_scene(seed), markers=markers)
        rep = run_episode(env, steps, instruction=instruction, planner=graph.source,
                          record=record_camera, max_seconds=max_seconds,
                          detector=detector, perception_camera=perception_camera,
                          perceive_only=perceive_only)
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
        perception=(f"{detector_spec} detector"
                    + (f", driving {sorted(perceive_only)} only" if perceive_only else "")
                    if detector_spec
                    else "simulator state (no perception in the loop)"),
        success_rate=float(np.mean([r.success for r in reports])) if reports else 0.0,
        subgoal_rates={k: float(np.mean([bool(r.subgoals.get(k)) for r in reports]))
                       for k in keys},
        mean_steps=float(np.mean([len(r.steps) for r in reports])) if reports else 0.0,
        mean_wall_s=float(np.mean([r.wall_s for r in reports])) if reports else 0.0,
        mean_parallel_fraction=float(np.mean(pfracs)) if pfracs else 0.0,
        # EVERY unsuccessful seed appears here, including the ones that have no
        # `first_failure`. A seed can run all eleven steps, have every step report
        # success, and still fail the task -- the bottle ends up on its side, or an
        # object drifts out of tolerance after the gripper lets go. Those have no
        # failing STEP, so keying this list on `first_failure` silently dropped them:
        # the perception report listed two rows for three failed seeds, which makes
        # the table look like it is hiding something even though the summary line
        # above it was right. End-state failures are labelled as such and name the
        # subgoals that were false at the end.
        failures=[_failure_row(r) for r in reports if not r.success],
        mean_plan_s=float(np.mean(plan_s)) if plan_s else 0.0,
        plan_sources={s: sources.count(s) for s in sorted(set(sources))},
        per_seed=[{"seed": r.seed, "success": r.success, "steps": len(r.steps),
                   "wall_s": round(r.wall_s, 2), "plan_s": round(p, 2),
                   "planner_source": src, "planner_notes": notes,
                   "episode_notes": r.notes, "subgoals": r.subgoals}
                  for r, p, src, notes in zip(reports, plan_s, sources, plan_notes)],
    )


def _failure_row(rep: EpisodeReport) -> dict:
    """One row per failed seed, whether or not a step reported the failure.

    Two shapes of failure exist and both have to be visible:

      step       a skill reported failure mid-episode; `first_failure` names it.
      end-state  every step succeeded and the task is still not done. This is
                 what an unstable bottle or a post-release drift looks like, and
                 it is the more interesting of the two, because the executor's
                 own per-step verification did not catch it.
    """
    if rep.first_failure is not None:
        return {"seed": rep.seed, "kind": "step",
                "label": rep.first_failure.label,
                "detail": rep.first_failure.detail}
    unmet = sorted(k for k, v in rep.subgoals.items() if not v)
    return {"seed": rep.seed, "kind": "end-state",
            "label": f"end state ({len(rep.steps)} step(s) ran, none reported failure)",
            "detail": ("subgoals false at the end: " + ", ".join(unmet)) if unmet
                      else "no subgoal was false; success criterion not met"}


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
