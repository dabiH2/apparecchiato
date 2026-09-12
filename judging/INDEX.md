# Apparecchiato — judging packet

Everything a judging session needs, in one folder, as of **Sat 12 Sep 2026, late evening**,
repo commit on `main`. Point a fresh session at this folder and nothing else.

This is a **copy**, made by `scripts/build_judging_packet.py`. The live repo is one level up.
Rebuild it after any change, or the packet quietly goes stale — which is exactly the failure
mode that cost 61/100 last time.

---

## Start here

| Read | Why |
| --- | --- |
| `README.md` | The submission's own claims, with the three qualifications up front |
| `results/` | Every number, as produced by the harness — not retyped |
| `docs/06-openvino-findings.md` | The most rigorous document in the project |
| `docs/03-submission-copy.md` | What a judge on lablab actually reads |
| `video/` | 30 s montage, one full episode, cover image |
| `docs/07-judge-simulation-prompt.md` | The prompt that produced the last review |

---

## The four result files, and what each one measures

| File | Positions from | Planner executed | Seeds | Success |
| --- | --- | --- | --- | --- |
| `eval_rules_seeds0-99` | simulator state | `rules` 100/100 | 0–99 | 100% |
| `eval_vlm_seeds0-9` | simulator state | `rules` 10/10, VLM rejected each time | 0–9 | 100% |
| `eval_perception_partial` | colour detector, bottle + mug | `rules` 10/10 | 0–9 | 70% |
| `eval_perception` | colour detector, all five objects | `rules` 10/10 | 0–9 | 0% |

Each report now states its own conditions in its own header. That was the single biggest
change since the last review: a report that could not say which planner ran, and did not say
where object positions came from, was evidence for nothing.

---

## Known weaknesses — stated, not hidden

A judge should find these in the first ten minutes. They are here so the finding is
confirmation rather than discovery.

1. **The VLM never plans a successful episode.** It is called on every seed, produces a task
   graph, and the validator rejects it: `node n3 depends on unknown node(s) ['n2']` on the
   first attempt, `unknown skill ''` on the retry. The deterministic planner then finishes.
   The report quotes those errors verbatim. The claim being made is about the *boundary* —
   a planner that fails does not stop the robot — not about the model.
2. **Perception is measured and does not clear the bar.** Bottle 10 mm, mug 8 mm, plate
   141 mm, cutlery 87–157 mm, against a 45 mm placement tolerance. 0/10 driving everything,
   7/10 driving the two objects it can place.
3. **No hardware, no sim-to-real.** Everything is MuJoCo.
4. **The Intel numbers are from an AMD Ryzen AI 9 HX 370.** The sweep's `GPU` row is this
   machine's NVIDIA RTX 4070 dGPU — the device string is in `bench_report.json`. There is no
   iGPU row and no NPU row. The export path and the sweep harness do run on target silicon.
5. **The hand-off is always the plate, always A→B**, on all 100 seeds. Derived, not scripted,
   but invariant — because the task makes it so. `tests/test_handoff_is_emergent.py` pins
   down exactly how far the claim goes, including the counterfactual where it disappears.
6. **64% "parallelisable" is a property of the plan, not the execution.** The executor runs
   steps in order and parks the idle arm. Genuine two-arm concurrency happens inside the
   hand-off and the pour.
7. **100/100 is a rate over a screened distribution.** Seeds are validated for reachability
   with 35 mm of standoff margin before they run, and the placement tolerance is 45 mm on a
   26 mm plate. Deliberate — an episode should only be able to fail for a reason the policy
   could have handled — but it is a conditioned number, not an unconditional one.
8. **Seeds 0–9 were the debugging set.** Three fixes were chosen by counting failures across
   all 100, so "never looked at" would overstate it. Nothing was tuned per seed.
9. **No Speechmatics artifact yet.** The client and the `--save-transcript` flag exist; a
   real session has not been recorded. The bonus award currently rests on code, not evidence.

---

## What changed since the 61/100 review

| Finding | Status |
| --- | --- |
| Report could not name the executed planner | **Fixed** — per-episode provenance + rejection reasons + planning time |
| Perception outside the reported pipeline | **Fixed** — `--detector colour` closes the loop, two runs committed |
| Footage unwatchable (60% dark, table off-centre, markers dominant) | **Fixed** — ambient light, look-at camera, dimmed/optional markers; `scripts/check_framing.py` measures it |
| README's seed-7 plan block stale | **Fixed** — real `--plan-only` output pasted in |
| Test count wrong in three places | **Fixed** — 208 everywhere |
| "GPU" row presented as an Intel iGPU | **Fixed** — named as the NVIDIA dGPU it is |
| "64% on both arms at once" | **Fixed** — described as a plan property |
| "only seeds 0–9 were looked at" | **Fixed** — scoped to "not tuned per seed" |
| ColourDetector errors understated in status notes | **Fixed** — real per-object table generated by the harness |
| Speechmatics artifact | **Open** — needs one real session with an API key |
| Sim-to-real | **Open** — out of scope for this weekend |
| Hand-off direction never varies | **Open by choice** — mirroring the scene needs the drawer side to become a per-seed property; measured as infeasible on 49/100 seeds without that |

---

## Reproducing anything here

From the repo root, one level up:

```bash
python -m pytest -q                                    # 208 passed
python scripts/run_eval.py --seeds 0-99 --planner rules --out out/eval100 --no-video
python scripts/run_eval.py --seeds 0-9 --planner vlm+rules --out out/eval_vlm --camera cinematic
python scripts/run_eval.py --seeds 0-9 --planner rules --detector colour --out out/eval_perception --no-video
python scripts/run_eval.py --seeds 0-9 --planner rules --detector colour --perceive bottle,mug \
       --out out/eval_perception_partial --no-video
python scripts/check_framing.py --seeds 0,3,7          # is the shot watchable
python scripts/clip_sheet.py --seeds 0-9               # what happens in each clip
python scripts/build_judging_packet.py                 # rebuild this folder
```
