# Apparecchiato — judging packet

Everything a judging session needs, in one folder, as of **Tue 15 Sep 2026**, repo commit on
`main`. Point a fresh session at this folder and nothing else.

This is a **copy**, made by `scripts/build_judging_packet.py`. The live repo is one level up.
Rebuild it after any change, or the packet quietly goes stale — which is exactly the failure
mode that cost 61/100 two reviews ago.

**How to read this packet.** Two reviews have already been run against it and both are in
the repo, with every finding either fixed or listed below as open. So the useful posture is
not "find the overclaim" — the overclaims have been found, and the interesting ones were
found by us. It is: *does the list below actually cover everything, and does each number
survive its own conditions?* The section "Known weaknesses" is written to be the first thing
a hostile reader confirms rather than the last thing they discover.

---

## Start here

| Read | Why |
| --- | --- |
| `README.md` | The submission's own claims, with the four qualifications up front |
| `results/` | Every number, as produced by the harness — not retyped |
| `docs/06-openvino-findings.md` | The most rigorous document in the project |
| `docs/03-submission-copy.md` | What a judge on lablab actually reads |
| `video/` | 30 s montage, one full episode, cover image |
| `docs/07-judge-simulation-prompt.md` | The prompt that produced the last review |

---

## The five result files, and what each one measures

| File | Positions from | Planner executed | Seeds | Success |
| --- | --- | --- | --- | --- |
| `eval_rules_seeds0-99` | simulator state | `rules` 100/100 | 0–99 | 100% |
| `eval_vlm_seeds0-9` | simulator state | `rules` 10/10, VLM rejected each time | 0–9 | 100% |
| `eval_perception_partial` | colour detector, bottle + mug | `rules` 10/10 | 0–9 | 70% |
| `eval_perception` | colour detector, all five objects | `rules` 10/10 | 0–9 | 0% |
| `eval_vlm_repair_seeds0-9` | simulator state | `rules` 10/10, VLM rejected each time — **now for semantic reasons, not JSON ones** | 0–9 | 100% |

Each report now states its own conditions in its own header. That was the single biggest
change since the last review: a report that could not say which planner ran, and did not say
where object positions came from, was evidence for nothing.

Three more files in `results/` are not success rates and are worth opening anyway:

| File | What it answers |
| --- | --- |
| `vlm_ablation.md` | Does the image change what the planner writes? Real frame vs no frame vs blank frame vs another seed's frame, same instruction, raw output diffed. |
| `bench_detector.json` | The raw sweep the markdown is rendered from, including the device product strings. `scripts/rerender_bench.py --check` proves the two agree. |
| `clip_sheet.md` | What actually happens, step by step, in each of the ten clips — generated from the run, not transcribed. |

---

## Known weaknesses — stated, not hidden

A judge should find these in the first ten minutes. They are here so the finding is
confirmation rather than discovery.

1. **The VLM never plans a successful episode.** It is called on every seed, produces a task
   graph, and the validator rejects it: `node n3 depends on unknown node(s) ['n2']` on the
   first attempt, `unknown skill ''` on the retry. The deterministic planner then finishes.
   The report quotes those errors verbatim. The claim being made is about the *boundary* —
   a planner that fails does not stop the robot — not about the model.
2. **Perception is measured and does not clear the bar.** In-loop: bottle 10 mm, mug 8 mm,
   plate 141 mm, cutlery 87–157 mm, against a 45 mm placement tolerance. 0/10 driving
   everything, 7/10 driving the two objects it can place. Two conditions that belong with
   those numbers and are in `docs/06`: only *x* and *y* come from pixels — *z* is read from
   the object library — and an object the detector misses keeps its last believed position,
   which on the first tick is ground truth. So the perception runs are **oracle-backed for
   whatever the detector fails to find**. `docs/06` also carries a *static* table (plate
   87 mm, spoon 46 mm, fork 23 mm); that is one frame with the arms parked, i.e. the
   detector's ceiling, not what the robot ran on. The in-loop numbers are the real ones.
3. **No hardware, no sim-to-real.** Everything is MuJoCo.
4. **The Intel numbers are from an AMD Ryzen AI 9 HX 370.** The sweep's `GPU` row is this
   machine's NVIDIA RTX 4070 dGPU — the device string is now printed in
   `results/bench_detector.md` itself, and the raw `results/bench_detector.json` is
   committed beside it. There is no iGPU row and no NPU row, and none is expected: this host
   has no Intel NPU, which is a fact about the box and not a missing driver. The export path
   and the sweep harness do run on target silicon.
5. **The hand-off is always the plate, always A→B**, on all 100 seeds. Derived, not scripted,
   but invariant — because the task makes it so. `tests/test_handoff_is_emergent.py` pins
   down exactly how far the claim goes, including the counterfactual where it disappears.
6. **64% "parallelisable" is a property of the plan, not the execution — and the hand-off is
   not a simultaneous two-arm action.** The executor runs steps in order and parks the idle
   arm. The hand-off is a *place-and-pick through a shared cell*: arm A sets the plate down
   in the lens and withdraws completely before arm B approaches, because an in-air transfer
   self-collides. Both grippers are never on the plate at the same instant, and the footage
   shows exactly that. **The pour is the one step in which both arms are moving at once**
   (B holds the mug, A tips the bottle). If "bimanual" to you means two arms contacting one
   object simultaneously, this system does not do that and does not claim to; what it does
   is *derive* when two arms are necessary, which is the harder half.
7. **100/100 is a rate over a screened distribution.** Seeds are validated for reachability
   with 35 mm of standoff margin before they run, and the placement tolerance is 45 mm on a
   26 mm plate. Deliberate — an episode should only be able to fail for a reason the policy
   could have handled — but it is a conditioned number, not an unconditional one.
8. **Seeds 0–9 were the debugging set.** Three fixes were chosen by counting failures across
   all 100, so "never looked at" would overstate it. Nothing was tuned per seed.
9. **The Speechmatics session is real; the voice in it is not.**
   `results/speechmatics_en.json` is a genuine real-time session against the Speechmatics
   API — every partial hypothesis in order, 3.1 s of audio, 1.3 s round trip, the server's
   own quota frame included — and the transcript it produced drove a real plan. But the
   audio is Windows SAPI text-to-speech, not a human, and the artifact says so in its own
   `audio_note` field. A human-spoken session (and the Italian one) is the artifact this
   should be replaced with. Finding that bug was worth the trip on its own: the client
   read exactly one message after `StartRecognition` and Speechmatics sends an `Info`
   frame first, so the very first real session it ever attempted died with "Speechmatics
   refused the session" — live, on stage, in front of judges, had it not been run here.
10. **The vision half of the vision-language planner is not resolving the scene.** Measured,
   not assumed — `results/vlm_ablation.md`. Feeding the planner **another seed's overhead
   frame** produces byte-identical output to the correct frame; feeding it no image, or a
   blank one, does not. So the pixels reach the model and change what it writes, but what
   they contribute is "a photograph of a table", not "the plate is here". The report also
   names its own confound (on seeds 0–1 the text prompts are byte-identical, so the swap
   never contradicted the words) and one fact about the committed runs: `run_eval.py` calls
   the planner **with no image**, so `eval_vlm_seeds0-9` was planned from text alone.
11. **Plan repair does not make the VLM load-bearing either, and the run proving it is
   committed.** `apparecchiato/planner/repair.py` lets the validator fix the model's
   *referential* mistakes — a dependency on a node it never wrote, a node naming no skill —
   instead of only rejecting them. It may delete and reorder what the model wrote and may
   never add to it; `tests/test_plan_repair.py` enforces that, including that changing an
   argument or an arm counts as invention. `results/eval_vlm_repair_seeds0-9.md` is the
   10-seed run with it on: still `rules` 10/10, still 100% task success. **It made no plan
   executable.** What it changed is where the plan dies — from
   `node n3 depends on unknown node(s) ['n2']` (an error about JSON) to
   `plan picks ['fork', 'spoon'] and never places them` (an error about the task). That is
   worth having and is not a success rate, and this list says so rather than letting the
   file be read as one. Off by default, so every earlier number reproduces exactly.

---

## What changed since the 56/100 review (the most recent one)

That review audited 28 claims and found **9 contradicted** — every one of them a sentence
that was true of an earlier draft and had been corrected in one file but not another. None
of them was a claim about the code being wrong; all of them were the copy drifting. So the
fix is two things: the sentences, and a test that stops it happening a third time.

<!-- audit-allow-block: this table quotes the retracted phrasings on purpose -->

| Finding | Status |
| --- | --- |
| "hand the spoon across" — it is always the plate | **Fixed** — copy + cover spec |
| Cover subtitle credited the VLM planner for the 100/100 | **Fixed** — now "plans validated against physics" |
| Video hook: "the plan you just saw was written by the model" | **Fixed** — hook now states the rejection, which is the stronger claim |
| Montage VO: "we only ever looked at the first ten" | **Fixed** — cut; the README already retracted it |
| README put an OpenVINO INT8 detector in the pipeline | **Fixed** — HSV colour named as the in-loop detector; the INT8 one marked exported-and-benchmarked-only, in the table and in the diagram |
| "a 2B planner reaches the scheduler reliably" unscoped | **Fixed** — scoped to the INT8 bench build; the shipped INT4 build's rejection stated |
| "real two-arm concurrency happens within the hand-off" | **Fixed** — hand-off described as place-and-pick through a shared cell; the pour named as the one simultaneous action |
| `eval_perception_partial` listed 2 failures for 3 failed seeds | **Fixed** — the harness keyed the table on failing *steps*, so a seed that ran all 11 steps and still failed (seed 4, `bottle_upright`) vanished. Now keyed on seeds, with `end-state` failures labelled; both reports re-run |
| docs/06's static detector table read as contradicting the eval tables | **Fixed** — both are now printed with their conditions, and the reader is told which to quote |
| "no privileged simulator state" was true of the detector, not of the runs | **Fixed** — docs/06 now states that *z* comes from the object library and that a missed object keeps ground truth |
| "the NPU driver is missing" on a host with no NPU | **Fixed** — the bench now says which of the two reasons applies, and names every device by its product string |
| `bench_report.json` not shipped | **Fixed** — committed as `results/bench_detector.json`; `scripts/rerender_bench.py --check` proves the report matches it |
| Does the image change the VLM's output? | **Measured** — `results/vlm_ablation.md`; see weakness 10 below |
| Nothing stopped a corrected claim from reappearing | **Fixed** — `scripts/audit_claims.py` fails on any retracted phrasing, anywhere in repo, docs or packet |

<!-- /audit-allow-block -->

---

## What changed since the 61/100 review

| Finding | Status |
| --- | --- |
| Report could not name the executed planner | **Fixed** — per-episode provenance + rejection reasons + planning time |
| Perception outside the reported pipeline | **Fixed** — `--detector colour` closes the loop, two runs committed |
| Footage unwatchable (60% dark, table off-centre, markers dominant) | **Fixed** — ambient light, look-at camera, dimmed/optional markers; `scripts/check_framing.py` measures it |
| README's seed-7 plan block stale | **Fixed** — real `--plan-only` output pasted in |
| Test count wrong in three places | **Fixed** — 221 everywhere |
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
python -m pytest -q                                    # 221 passed
python scripts/run_eval.py --seeds 0-99 --planner rules --out out/eval100 --no-video
python scripts/run_eval.py --seeds 0-9 --planner vlm+rules --out out/eval_vlm --camera cinematic
python scripts/run_eval.py --seeds 0-9 --planner rules --detector colour --out out/eval_perception --no-video
python scripts/run_eval.py --seeds 0-9 --planner rules --detector colour --perceive bottle,mug \
       --out out/eval_perception_partial --no-video
python scripts/vlm_ablation.py --seeds 0-1             # does the image change the plan
python scripts/check_framing.py --seeds 0,3,7          # is the shot watchable
python scripts/clip_sheet.py --seeds 0-9               # what happens in each clip
python scripts/build_judging_packet.py                 # rebuild this folder
```

And the three checks that keep this packet honest between edits:

```bash
python scripts/audit_claims.py                  # has a retracted claim come back?
python scripts/rerender_bench.py --check        # does bench_detector.md match its .json?
python scripts/build_judging_packet.py --check  # has this folder drifted from the repo?
```
