#!/usr/bin/env python3
"""Precision vs plan validity vs latency for the VLM planner on OpenVINO.

A latency table alone does not answer the question that matters for putting a
planner on edge silicon: quantising a 2B VLM to 4 bits makes it faster and
smaller, but does it still produce a plan the robot can execute? This measures
both at once, over the same seeds, and reports where each attempt died:

  emitted     the model produced a JSON object at all
  parsed      it parsed as valid JSON with no repair
  repaired    it parsed only after syntax repair (see planner.vlm._repair_json)
  validated   every object, slot and arm assignment exists and is reachable
  scheduled   the plan is physically executable -- this is the one that counts

  python scripts/bench_planner.py --seeds 4
  python scripts/bench_planner.py --models models/qwen2-vl-2b-ov-int4,models/qwen2-vl-2b-ov-int8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.planner.vlm import VLMPlanner                      # noqa: E402
from apparecchiato.planner.base import PlannerError                   # noqa: E402
from apparecchiato.sim.layout import sample_scene                     # noqa: E402

DEFAULT = ("Open the top drawer, set the plate, fork and spoon on the table, "
           "put the mug beside them and pour water into the mug.")


def stage_of(planner: VLMPlanner, exc: Exception | None) -> str:
    """Which stage the attempt reached. Ordered worst to best."""
    if exc is None:
        if planner.last_dropped:
            return "scheduled (nodes dropped)"
        return "scheduled (repaired)" if planner.last_repaired else "scheduled"
    msg = str(exc)
    if "no JSON object" in msg:
        return "no JSON"
    if "invalid JSON" in msg or "no node survived" in msg:
        return "unparseable"
    if "grasp state" in msg or "ready node" in msg or "SchedulingError" in type(exc).__name__:
        return "validated, not schedulable"
    return "rejected by validator"


def run_one(model: str, device: str | None, seeds: int, instruction: str,
            attempts: int, max_new_tokens: int) -> dict:
    planner = VLMPlanner(backend="openvino", model=model, device=device,
                         max_attempts=attempts, max_new_tokens=max_new_tokens)
    rows, latencies = [], []
    for seed in range(seeds):
        spec = sample_scene(seed)
        t0 = time.perf_counter()
        exc = None
        try:
            planner.plan(instruction, spec)
        except Exception as e:                                         # noqa: BLE001
            exc = e
        wall = time.perf_counter() - t0
        latencies.append(wall)
        stage = stage_of(planner, exc)
        rows.append({"seed": seed, "stage": stage, "wall_s": round(wall, 2),
                     "repaired": planner.last_repaired,
                     "dropped": planner.last_dropped,
                     "error": ("" if exc is None else str(exc)[:200])})
        print(f"  seed {seed}: {stage:32s} {wall:6.1f} s"
              + (f"   {str(exc)[:90]}" if exc else ""))
    good = [r for r in rows if r["stage"].startswith("scheduled")]
    clean = [r for r in rows if r["stage"] == "scheduled"]
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(model) for f in fs) / 1e6
    return {
        "model": model, "device": device or "AUTO", "size_mb": round(size_mb),
        "seeds": seeds, "schedulable": len(good), "clean_json": len(clean),
        "median_s": round(statistics.median(latencies), 1),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models/qwen2-vl-2b-ov-int4,models/qwen2-vl-2b-ov-int8")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--instruction", default=DEFAULT)
    ap.add_argument("--out", default="out/planner_bench")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = [m for m in models if os.path.isdir(m)]
    if not models:
        print("no exported models found -- run scripts/export_vlm_openvino.py first",
              file=sys.stderr)
        return 2

    report = []
    for m in models:
        print(f"\n{m}")
        report.append(run_one(m, args.device, args.seeds, args.instruction,
                              args.attempts, args.max_new_tokens))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# VLM planner on OpenVINO: precision vs plan validity", "",
             f"Instruction: _{args.instruction}_", "",
             f"{args.seeds} seeds, up to {args.attempts} attempt(s) each "
             "(a rejected plan is handed back to the model with the reason).", "",
             "| Model | Weights | Size | Schedulable | Clean JSON | Median wall time |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in report:
        fmt = "INT4" if "int4" in r["model"] else ("INT8" if "int8" in r["model"] else "?")
        lines.append(f"| `{os.path.basename(r['model'])}` | {fmt} | {r['size_mb']} MB "
                     f"| {r['schedulable']}/{r['seeds']} | {r['clean_json']}/{r['seeds']} "
                     f"| {r['median_s']} s |")
    lines += ["", "## Where each seed ended up", ""]
    for r in report:
        lines.append(f"**{os.path.basename(r['model'])}**")
        lines.append("")
        for row in r["rows"]:
            lines.append(f"- seed {row['seed']}: {row['stage']} ({row['wall_s']} s)"
                         + (f" -- {row['error']}" if row["error"] else ""))
        lines.append("")
    md = "\n".join(lines)
    with open(args.out + ".md", "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"\nwrote {args.out}.md and {args.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
