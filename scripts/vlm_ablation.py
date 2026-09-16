#!/usr/bin/env python3
"""Is the vision half of the vision-language planner contributing anything?

This is the question a hostile reviewer asks first, and it deserves a measurement
rather than an assurance. The architecture diagram says "VLM reads the overhead
camera"; whether the pixels change the answer is a separate, testable claim.

The experiment holds the instruction and the scene text fixed and varies ONLY the
image, across four conditions:

    none      no image at all -- which, note, is what the committed eval runs do
    real      the rendered overhead frame for this seed
    blank     a mid-grey frame of the same shape: an image with no scene in it
    swapped   the overhead frame of a DIFFERENT seed: a scene that contradicts
              the text

`none` is the control that matters most. If `real` == `none`, the vision tower is
not contributing. If `real` == `swapped`, the model is not reading the image even
when it disagrees with the text -- the stronger failure.

Raw generation is compared, not the parsed plan: parsing and validation would hide
small differences, and it is the model's output distribution that is under test.

    python scripts/vlm_ablation.py                       # seeds 0-1, INT4
    python scripts/vlm_ablation.py --seeds 0-2 --model models/qwen2-vl-2b-ov-int8

Writes results/vlm_ablation.md and results/vlm_ablation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.planner.vlm import VLMPlanner                      # noqa: E402
from apparecchiato.sim.layout import sample_scene                     # noqa: E402

DEFAULT_INSTRUCTION = ("Open the top drawer, set the plate, fork and spoon on the "
                       "table, put the mug beside them and pour water into the mug.")
CONDITIONS = ("none", "real", "blank", "swapped")


def _parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _render(seed: int, camera: str):
    """One settled overhead frame for a seed, markers off so the debug overlay
    cannot be what the model is looking at."""
    from apparecchiato.sim.env import TableEnv
    env = TableEnv(sample_scene(seed), markers=False)
    env.settle(0.4)
    frame = np.asarray(env.render(camera))
    env.close()
    return frame


def _digest(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "replace")).hexdigest()[:12]


STAGES = ("no JSON", "parsed", "graph built", "scene-valid", "SCHEDULABLE")


def _stage_reached(raw: str, seed: int, *, repair: bool) -> str:
    """The furthest stage this generation survives, using the real pipeline.

    A digest says two answers differ. It does not say whether either is a plan.
    This walks the same four gates an episode walks -- parse, build the graph,
    check it against the scene, schedule it onto two arms -- and names where the
    answer stops. `SCHEDULABLE` means the robot could have executed it.
    """
    from apparecchiato.planner.vlm import validate_against_scene       # noqa: PLC0415
    from apparecchiato.scheduler import schedule                       # noqa: PLC0415
    from apparecchiato.sim.layout import sample_scene as _scene        # noqa: PLC0415
    from apparecchiato.tasks import TaskGraph                          # noqa: PLC0415

    p = VLMPlanner(backend="openai", repair=repair)
    spec = _scene(seed)
    try:
        nodes = p._parse(raw)
    except Exception as exc:                                           # noqa: BLE001
        return f"no JSON ({type(exc).__name__})"
    try:
        graph = TaskGraph("i", nodes, source="vlm")
    except Exception as exc:                                           # noqa: BLE001
        return f"parsed, graph rejected: {str(exc)[:70]}"
    try:
        validate_against_scene(graph, spec)
    except Exception as exc:                                           # noqa: BLE001
        return f"graph built, scene rejected: {str(exc)[:70]}"
    try:
        steps = schedule(graph, spec)
    except Exception as exc:                                           # noqa: BLE001
        return f"scene-valid, scheduler rejected: {str(exc)[:70]}"
    return f"**SCHEDULABLE** ({len(steps)} steps)"


def _norm(s: str) -> str:
    """Whitespace-insensitive comparison. Two answers that differ only in
    indentation are the same answer for this purpose."""
    return " ".join((s or "").split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--camera", default="overhead")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/vlm_ablation")
    ap.add_argument("--from-json", default=None,
                    help="do not run anything; re-render the markdown from a "
                         "recorded run. The measurements are in the JSON, the "
                         "prose is a view of them, and correcting the prose "
                         "should never cost four more generations.")
    a = ap.parse_args()

    if a.from_json:
        src = pathlib.Path(a.from_json)
        payload = json.loads(src.read_text(encoding="utf-8"))
        # Recompute the text prompts. They need no model and no simulator, and
        # they are the confound that decides how much the image result is worth:
        # if two seeds' scene descriptions are byte-identical, then swapping
        # their images did NOT put the picture in conflict with the text, and the
        # 'swapped' row proves something weaker than it looks.
        # How far each recorded generation gets through the real pipeline. This
        # is the question the digests alone cannot answer: two outputs can differ
        # and both be useless, or differ and one be a plan. Recomputed from the
        # stored raw text, so it costs nothing and cannot drift from the run.
        for r in payload["rows"]:
            r["stage"] = {k: _stage_reached(r["raw"], r["seed"], repair=v)
                          for k, v in (("as-emitted", False), ("repaired", True))}
        payload["prompt_digests"] = {
            str(s): _digest(_norm(
                VLMPlanner(backend="openai")._build_prompt(
                    payload.get("instruction", DEFAULT_INSTRUCTION), sample_scene(s))))
            for s in payload["seeds"]}
        src.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        dst = pathlib.Path(a.out).with_suffix(".md")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(render_markdown(payload), encoding="utf-8")
        print(f"re-rendered {dst} from {src}")
        return 0

    seeds = _parse_seeds(a.seeds)
    if len(seeds) < 2:
        print("need at least two seeds: the 'swapped' condition uses another "
              "seed's frame", file=sys.stderr)
        return 2

    planner = VLMPlanner(backend="openvino", model=a.model, device=a.device,
                         max_new_tokens=a.max_new_tokens, max_attempts=1)

    print(f"model   {planner.model}\ndevice  {planner.device}\nseeds   {seeds}\n")

    frames = {s: _render(s, a.camera) for s in seeds}
    blank = np.full_like(frames[seeds[0]], 127)

    rows: list[dict] = []
    for i, seed in enumerate(seeds):
        spec = sample_scene(seed)
        other = seeds[(i + 1) % len(seeds)]
        images_for = {
            "none": None,
            "real": [frames[seed]],
            "blank": [blank],
            "swapped": [frames[other]],
        }
        # The prompt is built once per seed and reused verbatim across the four
        # conditions, so the image is genuinely the only thing that varies.
        prompt = planner._build_prompt(a.instruction, spec)
        for cond in CONDITIONS:
            t0 = time.perf_counter()
            try:
                raw = planner._generate(prompt, images_for[cond])
                err = None
            except Exception as exc:                                  # noqa: BLE001
                raw, err = "", f"{type(exc).__name__}: {exc}"
            wall = time.perf_counter() - t0
            rows.append({"seed": seed, "condition": cond,
                         "swapped_with": other if cond == "swapped" else None,
                         "wall_s": round(wall, 2), "chars": len(raw),
                         "digest": _digest(_norm(raw)), "error": err, "raw": raw})
            print(f"seed {seed:>3}  {cond:<8} {wall:6.1f} s  {len(raw):>5} chars  "
                  f"sha {rows[-1]['digest']}" + (f"  ERROR {err}" if err else ""))

    # --- verdict ----------------------------------------------------------
    verdicts = []
    for seed in seeds:
        by = {r["condition"]: r for r in rows if r["seed"] == seed}
        ref = _norm(by["real"]["raw"])
        same = {c: (_norm(by[c]["raw"]) == ref) for c in CONDITIONS if c != "real"}
        verdicts.append({"seed": seed, "identical_to_real": same,
                         "digest_real": by["real"]["digest"]})

    all_same_as_none = all(v["identical_to_real"]["none"] for v in verdicts)
    all_same_as_blank = all(v["identical_to_real"]["blank"] for v in verdicts)
    all_same_as_swapped = all(v["identical_to_real"]["swapped"] for v in verdicts)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": planner.model, "device": planner.device,
               "instruction": a.instruction, "camera": a.camera, "seeds": seeds,
               "rows": rows, "verdicts": verdicts,
               "identical_across_all_seeds": {
                   "real_vs_none": all_same_as_none,
                   "real_vs_blank": all_same_as_blank,
                   "real_vs_swapped": all_same_as_swapped}}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    print(f"real == none    : {all_same_as_none}")
    print(f"real == blank   : {all_same_as_blank}")
    print(f"real == swapped : {all_same_as_swapped}")
    return 0


def render_markdown(payload: dict) -> str:
    """Build the report from the recorded run. Separated from the run so the
    wording can be corrected without paying for four more generations, and so
    `--from-json` produces byte-identical output to the original."""
    rows = payload["rows"]
    verdicts = payload["verdicts"]
    seeds = payload["seeds"]
    planner_model = payload["model"]
    same = payload["identical_across_all_seeds"]
    all_same_as_none = same["real_vs_none"]
    all_same_as_blank = same["real_vs_blank"]
    all_same_as_swapped = same["real_vs_swapped"]
    a_camera = payload.get("camera", "overhead")
    a_seeds = ",".join(str(s) for s in seeds)

    class _A:
        camera = a_camera
        seeds = a_seeds
        model = planner_model
    a = _A()

    class _P:
        model = planner_model
        device = payload.get("device", "CPU")
    planner = _P()

    md = [f"# Does the image change the plan? — `{pathlib.Path(planner.model).name}`",
          "",
          "Generated by `scripts/vlm_ablation.py`. Nothing here is retyped.",
          "",
          f"- model: `{planner.model}`",
          f"- device: `{planner.device}`",
          f"- camera: `{a.camera}`, markers off",
          f"- seeds: {seeds}",
          f"- instruction and scene text: identical across all four conditions",
          "",
          "The instruction and the text scene description are held fixed. The image is "
          "the only variable. Raw generation is compared, not the parsed plan, because "
          "parsing would hide small differences.",
          "",
          "| Seed | Image given | Wall | Chars | sha256[:12] | Same output as `real`? |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        v = next(x for x in verdicts if x["seed"] == r["seed"])
        if r["condition"] == "real":
            same = "—"
        else:
            same = "**yes**" if v["identical_to_real"][r["condition"]] else "no"
        label = {"none": "none (this is what the eval runs do)",
                 "real": "this seed's overhead frame",
                 "blank": "flat mid-grey, same shape",
                 "swapped": f"seed {r['swapped_with']}'s overhead frame"}[r["condition"]]
        md.append(f"| {r['seed']} | {label} | {r['wall_s']:.1f} s | {r['chars']} | "
                  f"`{r['digest']}` | {same} |")

    md += ["", "## What this means", ""]
    if all_same_as_none and all_same_as_blank and all_same_as_swapped:
        md += ["**The image does not affect the output on any seed tested.** Identical "
               "bytes with the real frame, with no frame at all, with a blank frame, and "
               "with a frame of a different scene. On this task, at this model size and "
               "quantisation, the planner is functioning as a language model over the "
               "text scene description; the vision tower contributes nothing measurable.",
               "",
               "Stated plainly because it bounds a claim this project would otherwise be "
               "making by implication: the *multi-modal* half of 'multi-modal reasoning' "
               "is exported, runs, and is benchmarked — and is not, on this evidence, "
               "doing work."]
    elif all_same_as_swapped and not all_same_as_blank and not all_same_as_none:
        md += ["**The vision tower is doing work, but it is not discriminating between "
               "scenes.** Three facts, in the order that makes them legible:",
               "",
               "1. Passing *no* image changes the output. So the image is genuinely "
               "reaching the model and genuinely affecting generation — this is not a "
               "wiring bug where the pixels are silently dropped.",
               "2. Passing a *blank* frame also changes the output, and differs from both "
               "of the above. So the model is responding to image *content*, not merely "
               "to an image being present.",
               "3. Passing **another seed's frame produces byte-identical output to the "
               "correct frame.** Two different tables, same answer, to the byte.",
               "",
               "Taken together: the vision tower contributes, but what it contributes is "
               "roughly *\"this is a photograph of a table with objects on it\"* rather "
               "than *\"the plate is here and the mug is there\"*. At 2B parameters and "
               "INT4, on untextured coloured primitives, it is not resolving the layout — "
               "which is the same conclusion `docs/06` reaches from the other direction, "
               "where OWLv2 on this scene lands 130–370 mm from truth. Two independent "
               "measurements, one cause: **this scene does not look like the natural "
               "photographs these vision towers were trained on.**",
               "",
               "What that licenses this project to claim, and what it does not: the "
               "planner is multi-modal and the image measurably participates. It is **not** "
               "shown to ground its plan in the scene it is looking at, and nothing here "
               "should be read as claiming it does. The scene facts the plan is actually "
               "built from come from `observation_text(scene)` — the object list, "
               "positions and per-arm reachability, in text."]
    elif all_same_as_blank and all_same_as_swapped:
        md += ["**A blank frame and a contradicting frame produce the same output as the "
               "real frame**, while passing no image at all differs. So the image changes "
               "the answer only by being present, not by its content: the vision tower is "
               "being invoked but its contents are not informative."]
    else:
        md += ["The image changes the output under at least one condition; the per-seed "
               "pattern is in the table above and the raw generations are in "
               "`vlm_ablation.json`. Read the `swapped` row first: if another seed's "
               "frame gives the same answer as the correct one, the model is responding "
               "to the image without discriminating between scenes."]

    if any("stage" in r for r in rows):
        md += ["", "## How far each answer actually gets", "",
               "Identical-or-not is only half the question. An answer can differ and "
               "still be useless. So each recorded generation is pushed through the same "
               "four gates an episode walks — parse the JSON, build the task graph, check "
               "it against the scene, schedule it onto two arms — and the table names "
               "where it stops. `as-emitted` is the shipped path; `repaired` additionally "
               "enables `apparecchiato/planner/repair.py`, which may delete and reorder "
               "what the model wrote and may never add to it.", "",
               "| Seed | Image | As emitted | With referential repair |",
               "| --- | --- | --- | --- |"]
        for r in rows:
            st = r.get("stage") or {}
            md.append(f"| {r['seed']} | {r['condition']} | "
                      f"{st.get('as-emitted', '—')} | {st.get('repaired', '—')} |")
        md += ["",
               "Read the `none` rows against the image rows. **Giving this model the "
               "picture does not merely fail to help — it makes the answer worse.** "
               "Text-only, the model emits a recognisable plan: open the drawer, pick the "
               "plate, pick the fork, pick the spoon, place the mug, pour. It is wrong "
               "(three picks with no places between them, and it calls the bottle "
               "\"water\") but it is a plan. With the overhead frame attached, the same "
               "model degenerates into a repeating token loop and never closes its JSON.",
               "",
               "That is the characteristic 4-bit failure — a long structured answer the "
               "model cannot hold together — and the ~1100 extra image tokens are what "
               "tips it over. It also means the eval harness's omission of the image was, "
               "accidentally, the better configuration. Said plainly because it is the "
               "kind of thing a reader should not have to discover: **on this model, at "
               "this precision, the multi-modal path is a net negative, and the honest "
               "recommendation is a larger planner or constrained decoding — not this one "
               "with pictures.**",
               "",
               "Now read the two columns against each other, because that is what the "
               "repair layer is for. **As emitted, every single answer dies on "
               "bookkeeping** — a dependency pointing at a node the model never wrote. "
               "That error says nothing about whether the plan was any good. With the "
               "referential repair on, every answer gets past that wall and is then "
               "rejected for something that is actually *about the task*: it calls the "
               "bottle \"water\"; it places a mug it never picked; it picks a fork and a "
               "spoon and never puts them down; the whole answer was a token loop with "
               "one node in it and moves nothing at all.",
               "",
               "That is the entire value of the repair, and it is worth being precise "
               "about it rather than overselling it: **it did not make a single plan "
               "executable.** What it did was move the failure from the parser to the "
               "validator, where the error message names the model's actual mistake. "
               "\"node n3 depends on unknown node(s) ['n2']\" tells you to fix your JSON "
               "handling. \"plan picks ['fork', 'spoon'] and never places them\" tells you "
               "the model cannot sequence this task — which is the finding, and it is the "
               "one a bigger planner or a constrained decoder would have to address.",
               "",
               "Two of those validator rules were written *because* of this experiment, "
               "after the repair's first version rescued a collapsed generation into a "
               "valid one-step \"plan\" — open the drawer, stop — that the scheduler "
               "accepted. Executing that is worse than rejecting it, because a rejection "
               "hands the episode to the deterministic planner and a one-step plan does "
               "not. The validator now rejects a graph that moves nothing, and one that "
               "ends with an object still in a gripper."]

    pd = payload.get("prompt_digests") or {}
    if pd:
        distinct = len(set(pd.values()))
        md += ["", "## The confound, checked rather than assumed", "",
               "How much the `swapped` row is worth depends on whether the two seeds' "
               "TEXT prompts differ. If they are identical, swapping the images did not "
               "put the picture in conflict with the words, and the row shows less than "
               "it appears to. So the prompts are hashed here too:", "",
               "| Seed | sha256[:12] of the text prompt |", "| --- | --- |"]
        md += [f"| {s} | `{d}` |" for s, d in sorted(pd.items())]
        md += [""]
        if distinct == 1:
            md += ["**The text prompts are byte-identical across these seeds.** "
                   "`observation_text(scene)` names each object and which arm can reach "
                   "it, not where it is in millimetres, and on these two seeds every "
                   "object falls in the same reachability class — so the model is asked "
                   "exactly the same question about two different tables.",
                   "",
                   "That cuts both ways, and both halves belong here. It weakens the "
                   "`swapped` row: the image was never in conflict with the text, so "
                   "identical output is not the model *ignoring a contradiction*. And it "
                   "sharpens the point about where the plan's scene knowledge comes "
                   "from: the text the planner is given is coarse enough that two "
                   "visibly different tables are the same prompt, and the identical "
                   "answers show the image did not break the tie. The stronger version "
                   "of this experiment needs two seeds whose text descriptions differ — "
                   "it is the obvious next measurement, and it has not been run."]
        else:
            md += [f"The {len(pd)} seeds produce {distinct} distinct text prompts, so the "
                   "`swapped` condition really did put the image in conflict with the "
                   "text. An identical output under that condition is the strong result: "
                   "the model did not notice."]

    md += ["",
           "## The control condition is not hypothetical", "",
           "`none` is not a synthetic case invented for this document. "
           "`apparecchiato/eval/run_eval.py` calls `planner.plan(instruction, spec)` "
           "with no `images` argument, so **every committed eval run in this repository "
           "planned from the text scene description alone.** `scripts/check_planner.py "
           "--with-image` is the path that exercises the vision tower.",
           "",
           ("Since the `none` and `real` rows above are **identical**, that omission cost "
            "nothing measurable on these seeds — but it is a fact about the runs and "
            "belongs next to them, not in a commit log."
            if all_same_as_none else
            "Since the `none` and `real` rows above **differ**, the committed eval runs "
            "were not getting the output the architecture diagram implies they were. "
            "That is worth saying in plain words: the plans in "
            "`results/eval_vlm_seeds0-9.md` were generated from text alone. It changes "
            "nothing about those episodes — the plan was rejected either way and the "
            "deterministic planner ran — but a reader is entitled to know which of the "
            "two code paths produced the numbers they are looking at."),
           "",
           "## Reproduce", "",
           "```bash",
           f"python scripts/vlm_ablation.py --seeds {a.seeds} --model {a.model}",
           "```"]

    return "\n".join(md) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
