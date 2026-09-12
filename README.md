# Apparecchiato

**Bimanual VLA table setting on dual simulated SO-101 arms, deployed on Intel Core Ultra with OpenVINO.**

Submission for the **Intel — Bimanual VLA Manipulation with Multi-Modal Reasoning** online
track of the AI Infra Summit Hackathon (lablab.ai), plus the **Best Use of Speechmatics**
bonus award.

You say *"set the table and pour me some water."* Two SO-101 arms in MuJoCo work out
between them who can reach what, hand the spoon across when neither arm can do the whole
job alone, lay the setting, and pour.

---

## What it does

| Stage | What happens |
| --- | --- |
| **Hear** | Speechmatics real-time STT turns speech into an instruction (`--voice mic`). English or Italian. |
| **Observe** | Overhead and front cameras. An open-vocabulary detector compiled to OpenVINO IR grounds the objects on the table plane. |
| **Understand** | A vision-language model (OpenVINO IR, INT4, running on CPU / iGPU / NPU) reads the instruction and the scene and emits a task graph. |
| **Plan** | The task graph is validated against physics, then scheduled across two arms: reachability, grasp state, and shared-workspace safety. |
| **Act** | Skill primitives — `open_drawer`, `pick`, `place`, `handoff`, `pour` — drive both arms, verified after every step. |
| **Report** | 10 randomised seeds, per-subgoal success rates, and an OpenVINO device/precision sweep. |

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
instruction ──► VLM (OpenVINO, INT4) ──► task graph ──┐
                                                       ├──► validator ──► scheduler ──► skills ──► MuJoCo
scene ─────────► detector (OpenVINO, INT8) ───────────┘        │
                                                               └── rejects: unreachable arms, missing
                                                                   hand-offs, placing what is not held,
                                                                   picking from a closed drawer, cycles
```

When the VLM returns something unparseable, a deterministic planner takes over and the
fallback is recorded in the plan's notes. The demo does not die live on stage, and it does
not pretend the model succeeded when it did not.

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

---

## What a plan looks like

`python scripts/run_episode.py --seed 7 --plan-only`

```
 1. [A] open_drawer()  || place_fork,pick_mug,place_mug
 2. [A] pick(object=fork)  || pick_mug,place_mug,place_plate
 3. [A+B] handoff(object=fork)  [shared zone]
 4. [A] pick(object=plate)  || place_fork,pick_mug,place_mug
 5. [B] place(object=fork target=fork)  [shared zone]  || open,pick_plate,pick_spoon,place_spoon
 6. [B] pick(object=mug)  || open,pick_fork,pick_plate,pick_spoon,place_spoon
 7. [B] place(object=mug target=mug)  [shared zone]  || open,pick_fork,pick_plate,pick_spoon,place_spoon
 8. [A+B] handoff(object=plate)  [shared zone]
 9. [A] pick(object=spoon)  || place_fork,pick_mug,place_mug,place_plate
10. [B] place(object=plate target=plate)  [shared zone]  || pick_fork,pick_spoon,place_spoon
11. [A] place(object=spoon target=spoon)  || place_fork,pick_mug,place_mug,place_plate
12. [A+B] pour(source=bottle into=mug)  [shared zone]

75% of steps can run on both arms at once
```

Steps 3 and 8 are the interesting ones, and nobody wrote them. On this seed the fork starts
in the drawer and the plate starts on the left — both reachable only by arm A — while both
of their slots lie where only arm B can reach. The intersection is empty, so the scheduler
found that out from the annulus geometry and inserted two exchanges through the shared zone.

Change the seed and the hand-offs move, involve different objects, or disappear entirely.
That is the whole point: nothing here is tuned to one table.

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

| Run | Planner | Seeds | Task success |
| --- | --- | --- | --- |
| [results/eval_vlm_seeds0-9.md](results/eval_vlm_seeds0-9.md) | `vlm+rules` (Qwen2-VL-2B, OpenVINO INT4) | 0–9 | **100%** (10/10) |
| [results/eval_rules_seeds0-99.md](results/eval_rules_seeds0-99.md) | `rules` | 0–99 | **100%** (100/100) |

All seven subgoals — drawer, plate, fork, spoon, mug, mug-upright, bottle-upright — are at
100% in every run, and the mean is 11.0 steps of 11: no episode ends early. The hundred-seed
sweep matters more than the ten: only seeds 0–9 were ever looked at while debugging, so the
other ninety are the number the code was *not* tuned against.

64% of each plan is parallelisable across the two arms. That figure used to read 82%, and
the 18 points are a deliberate purchase: the shared lens both arms can reach is about
200 × 90 mm and a laid place setting is 190 mm wide, so a hand-off has nowhere to put an
object down once the setting is finished. The scheduler therefore reserves the lens —
everything waits for the hand-off — which took the success rate from 50% to 80% in one
change. See `_reserve_the_lens_for_handoffs` in `apparecchiato/scheduler.py`.

Reproduce:

```bash
python scripts/run_eval.py --seeds 0-9  --planner vlm+rules --out out/eval_vlm --camera cinematic
python scripts/run_eval.py --seeds 0-99 --planner rules     --out out/eval100 --no-video
```

---

## Layout

```
apparecchiato/
  kinematics.py      analytic FK/IK for the 5-DoF SO-101. No solver in the hot loop.
  tasks.py           TaskGraph -- the contract between reasoning and physics
  scheduler.py       arm assignment, grasp state, shared-workspace safety
  executor.py        drives both arms, verifies each skill as it completes
  planner/           base.py (interface + prompt), vlm.py (OpenVINO/OpenAI), rules.py (fallback)
  skills/            open_drawer, pick, place, handoff, pour, home
  perception/        oracle and OpenVINO open-vocabulary detectors
  sim/               layout.py (geometry + randomisation), scene.py (MJCF), env.py (MuJoCo)
  voice/             Speechmatics real-time client
  bench/             OpenVINO device x precision sweep
  eval/              multi-seed harness and report
scripts/             run_episode, run_eval, diagnose, bench_openvino, verify_env, export_*, calibrate,
                     dayone.ps1 (one-command Windows setup)
tests/               190 tests, none of which need MuJoCo
```

Everything that can be reasoned about without physics — kinematics, geometry, planning,
scheduling, skill construction — lives outside the simulator and is unit-tested without it.
`pytest` runs the whole suite in seconds on any machine.

```bash
python -m pytest -q     # 190 passed
```

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
