# Apparecchiato

**Bimanual table setting on dual simulated SO-101 arms, with a VLM planner exported to
OpenVINO IR and a validator between the model and the robot.**

Submission for the **Intel — Bimanual VLA Manipulation with Multi-Modal Reasoning** online
track of the AI Infra Summit Hackathon (lablab.ai), plus the **Best Use of Speechmatics**
bonus award.

You say *"set the table and pour me some water."* Two SO-101 arms in MuJoCo work out
between them who can reach what, pass the plate across when neither arm can do the whole
job alone, lay the setting, and pour.

> **Read this before the numbers.** Four things are true and easy to miss, so they are
> here rather than in a footnote.
>
> 1. **The VLM's plan is rejected on every seed.** Qwen2-VL-2B at INT4 is called, produces
>    a plan, and the validator throws it out; the deterministic planner finishes the
>    episode. The 100% figures are the fallback's. `results/eval_vlm_seeds0-9.md` names the
>    executed planner per episode, and `docs/06-openvino-findings.md` measures why. That
>    boundary *is* the project — a planner that fails does not stop the robot — but it
>    means this is not a demonstration of a VLM driving a robot. Two things were measured
>    rather than assumed, and both are below: whether the *image* changes what the model
>    writes (`results/vlm_ablation.md` — it does, and not for the better), and whether the
>    model's mistakes are repairable without writing the plan for it (`--repair-vlm-plan` —
>    they move the failure from the parser to the validator, and no further).
> 2. **The headline numbers use simulator state for object positions, not perception.**
>    Perception is measured separately and reported in `results/eval_perception*.md`.
> 3. **The Intel figures were measured on an AMD Ryzen AI 9 HX 370**, because no Core
>    Ultra hardware was available. The export path and the device/precision sweep are real
>    and reproduce with one command on target hardware; the numbers are not from target
>    hardware. The "GPU" row in the sweep is this machine's NVIDIA dGPU, not an Intel iGPU.
> 4. **"100/100" is a rate over a screened distribution, at a stated tolerance.** Seeds are
>    rejected at sample time unless every object, slot, hand-off point and drawer position is
>    reachable with 35 mm of standoff margin, and a placement counts as correct within 45 mm
>    on a 26 mm-radius plate. Both are deliberate — an episode should only be able to fail
>    for a reason the policy could have handled — but 100/100 is a *conditioned* number and
>    should never be quoted without those two conditions attached.

---

## What it does

| Stage | What happens |
| --- | --- |
| **Hear** | Speechmatics real-time STT turns speech into an instruction (`--voice mic`, or `--voice file` for a WAV). English or Italian. One real session is committed: [`results/speechmatics_en.json`](results/speechmatics_en.json) — final text, every partial hypothesis in order, 3.1 s of audio, 1.3 s round trip. Its `audio_note` says plainly that the audio was text-to-speech rather than a human. |
| **Observe** | Overhead and front cameras. The detector that actually closes the loop is a classical HSV `ColourDetector` (`--detector colour`); `results/eval_perception*.md` reports exactly what it costs. The open-vocabulary detector (OWLv2) **is** exported to OpenVINO IR at INT8 and benchmarked (`results/bench_detector.md`), but it is **not in the loop for any number in this repo** — `docs/06-openvino-findings.md` measures why. Perception is off by default, and the headline numbers say so. |
| **Understand** | Qwen2-VL-2B, OpenVINO IR at INT4, reads the instruction and the scene and emits a task graph. On every seed so far that graph is rejected downstream. |
| **Plan** | The task graph is validated against physics, then scheduled across two arms: reachability, grasp state, and shared-workspace safety. Rejected plans fall through to a deterministic planner, and the report names which one ran. |
| **Act** | Skill primitives — `open_drawer`, `pick`, `place`, `handoff`, `pour` — drive both arms, verified after every step. |
| **Report** | 10 and 100 randomised seeds, per-subgoal success rates, per-episode planner provenance, and an OpenVINO device/precision sweep. |

---

## The idea worth stealing

Most VLA demos put one model in charge of everything and hope. This one draws a hard line
between **reasoning** and **physics**, and puts a validator on the boundary.

The model decides *what* to do. It never decides *whether that is possible* — the scheduler
does, from geometry. If the model says "arm A, pick up the mug" and the mug is outside arm
A's annulus, the plan is rejected with a specific reason before a single joint moves. If the
object starts in one arm's workspace and belongs in the other's, a hand-off is inserted
because the geometry demands one, not because the model remembered to.

That boundary is why this runs unattended across randomised seeds instead of being a demo
that works on one carefully arranged table. It is also why a 2B model quantised to INT4 is
enough: it is doing task decomposition, not inverse kinematics.

```
instruction ──► VLM (OpenVINO, INT4) ──► task graph ──┐   ← rejected on every seed so far
                                                       ├──► validator ──► scheduler ──► skills ──► MuJoCo
scene ─────────► detector (HSV colour)  ──────────────┘        │
                                                               └── rejects: unreachable arms, missing
                                                                   hand-offs, placing what is not held,
                                                                   picking from a closed drawer, cycles

        OWLv2 → OpenVINO IR INT8: exported, swept, benchmarked — and NOT in the loop above.
        Nothing in this repo's success numbers depends on it. Said plainly so you do not
        have to find it: of the two models exported to OpenVINO, one is rejected every time
        and the other is never called at run time.
```

When the VLM returns something unparseable, a deterministic planner takes over and the
fallback is recorded in the plan's notes. The demo does not die live on stage, and it does
not pretend the model succeeded when it did not.

### Is the vision half of the vision-language model doing anything?

The obvious hostile question, so it gets a measurement rather than an assurance:
`scripts/vlm_ablation.py` holds the instruction and the scene text fixed and varies only the
image — the real overhead frame, no image at all, a blank grey frame, and **another seed's
frame**. Full result in [`results/vlm_ablation.md`](results/vlm_ablation.md); the short
version:

| Image given to the planner | Output |
| --- | --- |
| this seed's overhead frame | the reference |
| none | **different** — so the pixels do reach the model and do change generation |
| flat grey | **different** again — so it responds to image *content*, not just presence |
| a **different seed's** frame | **byte-identical** to the reference |

So the vision tower contributes something like *"a photograph of a table with objects on
it"* and not *"the plate is here"*. It is not resolving the layout — the same conclusion
`docs/06` reaches from the other side, where OWLv2 on this scene lands 130–370 mm from
truth. One cause for both: **this scene does not look like the natural photographs these
vision towers were trained on.**

The report also states the confound rather than waiting to be caught by it — on seeds 0 and
1 the *text* prompts are byte-identical, so the swap never put the image in conflict with
the words — and one fact about the committed runs that is easy to miss: `run_eval.py` calls
`planner.plan(instruction, spec)` with **no image**, so the plans in
`results/eval_vlm_seeds0-9.md` were generated from the text scene description alone.

There is a sharper finding in that report than the one above: **giving this model the
picture makes its answer worse.** Text-only it emits a recognisable plan — open the drawer,
pick the plate, pick the fork, pick the spoon, place the mug, pour. With the overhead frame
attached, the same model degenerates into a repeating token loop and never closes its JSON.
The ~1100 extra image tokens tip a 4-bit 2B model past what it can hold together.

### When the validator can repair instead of reject

A validator that only ever says *no* throws away real work. Both mistakes this model
actually commits are **referential, not semantic** — a dependency on a node it never wrote,
a node whose `skill` came out empty. Neither is a wrong plan; both are bookkeeping, and both
cost the whole episode's plan.

`apparecchiato/planner/repair.py` (`--repair-vlm-plan`, **off by default**) fixes exactly
those, under a boundary enforced in code rather than promised in prose:

> **It may delete instructions and reorder them. It may never add one.**

`assert_nothing_invented` compares every node's skill, arguments and arm before and after
and raises if anything appeared; `tests/test_plan_repair.py` pins that down, including that
changing an argument or an arm counts as invention. Every repair is named in the plan's
notes and therefore in the episode report, so a reader can see per seed how much of the
executed plan was the model's.

**What it buys, stated without inflation: it has not made a single plan executable.** What
it changed is *where* the plan dies. Before, every rejection read
`node n3 depends on unknown node(s) ['n2']` — an error about JSON. After, they read
`plan references objects that are not in the scene: ['water']` and
`plan picks ['fork', 'spoon'] and never places them — the episode would end with them still
in a gripper`. Those are errors about the *task*, and they are the ones a bigger planner or
a constrained decoder would have to fix. Moving a failure from the parser to the validator
is not a success rate; it is the difference between knowing your JSON handling is fragile
and knowing your model cannot sequence the task.

The experiment also caught the repair layer being wrong. Its first version rescued a
collapsed generation into a valid, schedulable one-step plan — *open the drawer, stop* —
which the scheduler accepted and an episode would have run and failed. Executing that is
strictly worse than rejecting it, because a rejection hands the episode to the deterministic
planner and a one-step plan does not. The validator now rejects a graph that moves nothing,
and one that ends with an object still in a gripper.

---

## Quick start

**Windows, one command** (copies out of OneDrive, sets up a venv, installs, tests,
and runs the drawer diagnostic):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dayone.ps1
```

Or by hand:

```bash
pip install -r requirements.txt
python scripts/verify_env.py          # says exactly what is missing and how to fix it

# Plan only -- no simulator needed, runs anywhere in under a second
python scripts/run_episode.py --seed 3 --plan-only

# Full episode with video
python scripts/run_episode.py --seed 3 --record out/seed3.mp4

# The submission numbers: 10 randomised seeds
python scripts/run_eval.py --seeds 0-9 --out out/eval

# Why is a skill failing? Measure, don't guess
python scripts/diagnose.py --seed 0 --steps 1

# Speak the command instead of typing it
export SPEECHMATICS_API_KEY=...
python scripts/run_episode.py --voice mic --language en
```

### On Intel Core Ultra

```bash
python scripts/export_vlm_openvino.py --weight-format int4     # planner  -> OpenVINO IR
python scripts/export_detector_openvino.py --weight-format int8  # detector -> OpenVINO IR
python scripts/bench_openvino.py --list-devices                 # expect CPU, GPU, NPU
python scripts/bench_openvino.py --model models/owlv2-base-ov-int8/openvino_model.xml
```

`bench_openvino.py` sweeps every device against every precision and prints the host CPU it
ran on, stating plainly whether that host is a Core Ultra Series 2/3 part. A benchmark that
does not identify its own silicon is not evidence of anything.

**What the committed sweep is and is not.** `results/bench_detector.md` was measured on an
AMD Ryzen AI 9 HX 370. Its `CPU` row is that CPU; its `GPU` row is this machine's **NVIDIA
RTX 4070 laptop dGPU** — both device strings are printed in the report's own Notes section,
and the raw `results/bench_detector.json` the report is rendered from is committed beside it
(`python scripts/rerender_bench.py --check` proves the two agree) — and there is no NPU row
and no Intel iGPU row, because this box has neither. Everything Intel-specific here is the
export path and the sweep harness, both of which run on target hardware with the commands
above. The latency numbers are not from target hardware and are not presented as if they
were. See `docs/06-openvino-findings.md` for the one finding that does transfer: the CPU
plugin silently runs bf16, so a latency number without its precision is not a measurement.

---

## What a plan looks like

`python scripts/run_episode.py --seed 7 --plan-only`

```
plan from rules: 11 nodes
 1. [A] open_drawer()
 2. [A] pick(object=plate)
 3. [A+B] handoff(object=plate)  [shared zone]
 4. [B] place(object=plate target=plate)  [shared zone]  || pick_fork,pick_spoon,place_spoon
 5. [A] pick(object=fork)  || place_plate,pick_mug,place_mug
 6. [A] place(object=fork target=fork)  [shared zone]  || pick_mug
 7. [B] pick(object=mug)  || pick_fork,place_fork,pick_spoon,place_spoon
 8. [B] place(object=mug target=mug)  [shared zone]  || pick_fork,pick_spoon,place_spoon
 9. [A] pick(object=spoon)  || place_plate,pick_mug,place_mug
10. [A] place(object=spoon target=spoon)  || place_plate,pick_mug,place_mug
11. [B+A] pour(source=bottle into=mug)  [shared zone]

64% of steps can run on both arms at once
```

Step 3 is the interesting one, and nobody wrote it. On this seed the plate starts where
only arm A can reach and its slot lies where only arm B can reach. The intersection is
empty, so the scheduler found that out from the annulus geometry and inserted an exchange
through the shared zone.

Being exact about how far that goes, because it is the claim most worth checking: across a
hundred seeds it is **always the plate and always A to B**. That is the task, not a script.
The mug's slot has to sit in the shared lens for the pour, so whichever arm picks the mug
can also place it; the cutlery is routed to one arm on purpose; the bottle is never put
away. The plate is the only object whose pick and place can land in different workspaces.
What the seed varies is *where* the transfer happens.

`tests/test_handoff_is_emergent.py` states that as tests rather than prose: a hand-off
appears exactly when pick-reach and place-reach do not intersect, and moving the plate
within reach of its own slot makes the step vanish — same seed, same code, one fewer step.

The `||` column lists the steps the dependency graph would permit to run concurrently.
**The executor does not run them concurrently** — it walks the list in order and parks the
idle arm. Be precise about the two steps that involve both arms, because "bimanual" is doing
a lot of work in this sentence elsewhere: the hand-off is a **place-and-pick through a shared
cell** — arm A sets the plate down in the lens and fully withdraws *before* arm B approaches,
by design, because in-air transfer self-collides — so the two arms are never on the plate at
the same instant. The **pour is the one step where both arms are in motion simultaneously**:
B holds the mug steady while A tips the bottle. Overlap *between* steps is a property of the
plan, not of the execution.

---

## Robustness

Every seed randomises object **placement**, **mass**, **friction**, **scale**, **lighting**
(position and intensity), and **background colour** — the six axes the challenge brief asks
for. Seeds are validated at sample time: if any object, slot, hand-off point or drawer
position were unreachable, the seed is rejected before it reaches the simulator. An episode
can therefore only fail for reasons the policy could actually have handled, which is what
makes the success rate mean something.

Reported per subgoal, not as a single number. "40% success" tells you nothing; "every
failure is the spoon hand-off" tells you what to fix.

### Results

| Run | Object positions from | Planner executed | Seeds | Task success |
| --- | --- | --- | --- | --- |
| [eval_rules_seeds0-99](results/eval_rules_seeds0-99.md) | simulator state | `rules` 100/100 | 0–99 | **100%** (100/100) |
| [eval_vlm_seeds0-9](results/eval_vlm_seeds0-9.md) | simulator state | `rules` 10/10 after the VLM is rejected | 0–9 | **100%** (10/10) |
| [eval_perception_partial](results/eval_perception_partial.md) | **colour detector**, driving bottle + mug | `rules` 10/10 | 0–9 | **70%** (7/10) |
| [eval_perception](results/eval_perception.md) | **colour detector**, driving all five objects | `rules` 10/10 | 0–9 | **0%** (0/10) |
| [eval_vlm_repair_seeds0-9](results/eval_vlm_repair_seeds0-9.md) | simulator state | `rules` 10/10 — VLM rejected, but now for a *semantic* reason | 0–9 | **100%** (10/10) |

Read those rows together; separately they each mislead. **Row 5 is not a fifth success
rate** — it is the same task with plan repair enabled, and it is in the table so that the
honest conclusion is visible rather than buried: repair did not make one VLM plan
executable. What it changed is the *reason* for the rejection. See "When the validator can
repair instead of reject" above.

**Rows 1–2 measure planning, scheduling and control.** All seven subgoals at 100%, mean 11.0
steps of 11, no episode ending early. The hundred-seed sweep matters more than the ten: only
seeds 0–9 were ever opened while debugging. Three specific fixes were chosen by counting
failures across all hundred, so "never looked at" would overstate it — but nothing was tuned
per seed, and the per-seed failures that drove those fixes are in the commit log.

**Rows 3–4 measure perception, and it is the weaker half.** With the colour detector driving
every object the task fails outright: it places the bottle to 10 mm and the mug to 8 mm, but
the plate to 141 mm and the cutlery to 87–157 mm, against a 45 mm placement tolerance. The
plate and cutlery are small and achromatic; HSV segmentation has nothing to lock onto. Give
perception only the two objects it can actually place and the task finishes 7 of 10. That
boundary is the honest state of the perception path, and `docs/06-openvino-findings.md`
records why the open-vocabulary detectors do not clear it either.

> **If you are comparing this against the detector table in `docs/06`, the numbers differ
> on purpose.** That table is *static*: one frame per seed, arms parked, nothing occluded —
> plate 87 mm, spoon 46 mm, fork 23 mm. The numbers above are *in-loop*: the mean over every
> perception tick of a running episode, with two arms crossing the overhead view and a
> gripper on the object — plate 141 mm, spoon 157 mm, fork 87 mm. The static table is the
> detector's ceiling; the in-loop table is what the robot ran on. **The in-loop numbers are
> the real ones.** Also worth stating: only *x* and *y* come from pixels — *z* is read from
> the object library, and an object the detector misses keeps its last believed position,
> which on tick one is ground truth. So these runs are perception-driven for what the
> detector sees and oracle-backed for what it misses.

64% of each plan is **parallelisable on paper** — steps the dependency graph would permit to
overlap. The executor runs them one at a time. The hand-off is a **place-and-pick through a
shared cell**, not a mid-air exchange: arm A sets the plate down and withdraws before arm B
moves in, so both grippers are never on the object at once. **The pour is the one step in
which both arms are moving at the same time.** That figure used to read 82%, and the 18 points are a deliberate
purchase: the shared lens is about 200 × 90 mm, a laid place setting is 190 mm wide, so a
hand-off has nowhere to set an object down once the setting is finished. The scheduler
reserves the lens — everything waits for the hand-off — which took success from 50% to 80%
in one change. See `_reserve_the_lens_for_handoffs` in `apparecchiato/scheduler.py`.

Reproduce:

```bash
python scripts/run_eval.py --seeds 0-99 --planner rules     --out out/eval100 --no-video
python scripts/run_eval.py --seeds 0-9  --planner vlm+rules --out out/eval_vlm --camera cinematic
python scripts/run_eval.py --seeds 0-9  --planner rules --detector colour \
       --out out/eval_perception --no-video
python scripts/run_eval.py --seeds 0-9  --planner rules --detector colour \
       --perceive bottle,mug --out out/eval_perception_partial --no-video
```

---

## Layout

```
apparecchiato/
  kinematics.py      analytic FK/IK for the 5-DoF SO-101. No solver in the hot loop.
  tasks.py           TaskGraph -- the contract between reasoning and physics
  scheduler.py       arm assignment, grasp state, shared-workspace safety
  executor.py        drives both arms, verifies each skill as it completes
  planner/           base.py (interface + prompt), vlm.py (OpenVINO/OpenAI), rules.py (fallback),
                     repair.py (may delete and reorder the model's plan; may never add to it)
  skills/            open_drawer, pick, place, handoff, pour, home
  perception/        oracle and OpenVINO open-vocabulary detectors
  sim/               layout.py (geometry + randomisation), scene.py (MJCF), env.py (MuJoCo)
  voice/             Speechmatics real-time client
  bench/             OpenVINO device x precision sweep
  eval/              multi-seed harness and report
scripts/             run_episode, run_eval, diagnose, bench_openvino, verify_env, export_*, calibrate,
                     vlm_ablation (does the image change the plan?), audit_claims (has a
                     retracted claim come back?), rerender_bench, build_judging_packet,
                     dayone.ps1 (one-command Windows setup)
tests/               221 tests, none of which need MuJoCo
```

Everything that can be reasoned about without physics — kinematics, geometry, planning,
scheduling, skill construction — lives outside the simulator and is unit-tested without it.
`pytest` runs the whole suite in seconds on any machine.

```bash
python -m pytest -q     # 221 passed
```

### The claims are tested too

Two review rounds scored this project down for the same class of bug: a sentence that was
true of an earlier draft, corrected in one file, and left standing in another. The cover
image said the VLM planned the episodes after the README had already retracted it. That is
drift, and drift is a bug, so it has a test rather than a resolution to be careful:

```bash
python scripts/audit_claims.py       # every retracted claim, and where it came back
python scripts/rerender_bench.py --check   # the benchmark .md still matches its .json
python scripts/build_judging_packet.py --check   # the review copy still matches the repo
```

`scripts/audit_claims.py` holds one rule per claim a reviewer has caught — the banned
phrasing, why it is false, which file refutes it, and what to say instead — and fails if any
of them reappears anywhere in the repo, the docs or the judging packet. Rules are added when
a claim is caught and never removed, because the wording that was wrong once is exactly the
wording that will feel natural again at 2 a.m.

---

## Notes on the arm model

The MJCF is **generated from the same link constants the IK uses**
(`apparecchiato/kinematics.py`), so forward kinematics in Python and the simulator agree by
construction. A drift between a hand-written URDF and the solver's assumptions is the single
most common reason a "working" manipulation demo quietly misses its grasps.

To use the official SO-101 MJCF instead, run `scripts/calibrate_kinematics.py --mjcf <path>`
to re-measure the link lengths first, then re-run the kinematics tests.

## Licence

MIT.
