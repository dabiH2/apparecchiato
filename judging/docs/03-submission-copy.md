# Submission copy — ready to paste

All numbers filled in from the runs of **Sat 12 Sep, evening** — after a judge simulation
scored the earlier draft 61/100 and caught three overstatements. Sources, all committed:

| Claim | File |
| --- | --- |
| 100/100 seeds, rules planner | `results/eval_rules_seeds0-99.md` |
| 10/10 seeds, full stack, VLM rejected each time | `results/eval_vlm_seeds0-9.md` |
| 7/10 with perception driving bottle + mug | `results/eval_perception_partial.md` |
| 0/10 with perception driving all five | `results/eval_perception.md` |
| Device × precision sweep (AMD host, dGPU not iGPU) | `results/bench_detector.md` |

**Do not re-add the three claims that were cut**: that the VLM plans the episodes, that
perception is in the loop for the headline numbers, and that the sweep covers an Intel iGPU
or NPU. Each is contradicted by a file in this repo.

---

## Project title

```
Apparecchiato
```

## Short description (one line)

```
Two SO-101 arms set a dinner table from a spoken command, handing the plate across when neither can reach alone -- a VLM on OpenVINO proposes the plan, geometry decides whether it is possible, and the robot finishes the job even when the model is wrong.
```

## Technology tags

```
OpenVINO, Intel Core Ultra, MuJoCo, SO-101, Vision-Language-Action, Bimanual Manipulation,
Speechmatics, Robotics, Physical AI, Python
```

## Category tags

```
Robotics, Physical AI, Edge AI, Voice
```

---

## Long description

```
You say "set the table and pour me some water." Two simulated SO-101 arms work out between
them who can reach what, hand the PLATE across when neither arm can do the whole job alone,
lay the place setting, and pour.

THE PROBLEM WITH THE OBVIOUS APPROACH

One end-to-end VLA policy for a multi-step bimanual task is the obvious answer, and it is
the one that fails quietly. A policy that memorises a table layout looks excellent on the
scene it was tuned on and collapses when the plate moves four centimetres. With randomised
mass, friction, scale, lighting and placement, a demo either generalises or it is theatre.

So Apparecchiato draws a hard line. The model decides WHAT to do. Geometry decides WHETHER
IT IS POSSIBLE. A validator sits on that boundary.

HOW IT WORKS

Speech goes through Speechmatics real-time transcription. A vision-language model --
compiled to OpenVINO IR at INT4, running on CPU here and selectable to iGPU or NPU -- reads the overhead
camera and the instruction and emits a task graph: nodes, arm hints, dependencies.

That graph is then checked against the world before anything moves. Plans that reference
objects not on the table, pick cutlery from a drawer they never opened, place what was never
picked, or pin an arm to a target outside its workspace are rejected with a specific reason.
If the model returns something unparseable, a deterministic planner takes over and the
fallback is recorded in the plan -- the demo does not die live, and it does not pretend the
model succeeded when it did not.

The scheduler then assigns arms under three constraints: reachability including approach
clearance, grasp state (an arm holds one object; you cannot place what you are not holding),
and shared-workspace safety (the two annuli overlap in a lens, and only one arm may be in it
at a time -- except during a hand-off, where both belong there).

THE HAND-OFF IS EMERGENT

Nothing names the object that gets passed. The scheduler emits a hand-off exactly when the
set of arms that can PICK something and the set that can PLACE it do not intersect, and on
this table that is the plate: it spawns where only the left arm reaches and its slot is
where only the right arm reaches. The hand-off skill then sets it down in the shared zone
and withdraws completely before the other arm moves in.

Being precise about this, because it is the claim most worth checking: across a hundred
seeds it is always the plate and always left-to-right. That is the task, not a script. The
mug's slot has to sit in the shared lens so both arms can work the pour, which means
whichever arm picks the mug can also place it; the cutlery is deliberately routed to one
arm; the bottle is never put away. The plate is the only object whose pick and place can
land in different workspaces. What the seed varies is WHERE the transfer happens -- it is
sampled per seed and then re-chosen at motion-build time against the live table, because by
then some slots are full.

The claim is tested rather than asserted: tests/test_handoff_is_emergent.py moves the plate
within reach of its own slot and the hand-off disappears -- same seed, same planner, same
code, one fewer step in the plan.

RESULTS, AND WHAT THEY DO AND DO NOT MEASURE

100% across 100 randomised seeds, and 100% across the 10 seeds run with the full stack.
Every one of the seven subgoals at 100%; mean 11.0 steps of 11, so no episode ends early on
any seed. The randomisation covers placement, mass (0.02-0.40 kg), friction (0.5-1.25),
object scale (+/-10%), lighting position and intensity, and background colour.

Four qualifications, stated here rather than buried, because each of them changes what that
number means.

FIRST: the VLM's plan is rejected on every seed. Qwen2-VL-2B at INT4 is called, produces a
task graph, and the validator throws it out; the deterministic planner then finishes the
episode. The report names the executed planner per episode and quotes the rejection reason.
So this is not a demonstration of a VLM driving a robot -- it is a demonstration that a
robot keeps working when its model fails, which is the boundary the whole system is built
around and the reason it can run unattended at all. Be exact about which model does what,
because the two are easy to blur: at INT8 -- the bench configuration, not the shipped one --
Qwen2-VL-2B reaches the scheduler on the first attempt every time and then makes one
specific physical error (it plans every pick first, so the second pick asks an arm to grasp
while it is already holding something). The INT4 build that actually ships is worse: it
often does not emit a usable plan at all, and in the committed ten-seed run its output is
rejected at the validator on every seed. docs/06 measures both and concludes the fix is
capacity or constrained decoding, not more bits.

And the question a sceptic asks next, answered by measurement rather than assurance: IS THE
VISION HALF DOING ANYTHING? results/vlm_ablation.md holds the instruction and the scene text
fixed and varies only the image. Passing no image changes the output; passing a blank grey
frame changes it again; passing ANOTHER SEED'S overhead frame produces byte-identical
output to the correct one. So the pixels do reach the model and do change what it writes,
but what they contribute is "a photograph of a table with objects on it" rather than "the
plate is here" -- the same conclusion docs/06 reaches from the other direction, where OWLv2
on this scene lands 130-370 mm from truth. One cause for both: this scene does not look like
the natural photographs these vision towers were trained on. The report names its own
confound too (on seeds 0-1 the text prompts are byte-identical, so the swap never
contradicted the words), and one fact about the committed runs that is easy to miss: the
eval harness calls the planner with no image at all, so those plans came from text alone.
There is a sharper result in it than that: GIVING THIS MODEL THE PICTURE MAKES ITS ANSWER
WORSE. Text-only it emits a recognisable plan -- open the drawer, pick the plate, pick the
fork, pick the spoon, place the mug, pour. With the overhead frame attached the same model
degenerates into a repeating token loop and never closes its JSON. The ~1100 extra image
tokens tip a 4-bit 2B model past what it can hold together.

A VALIDATOR THAT REPAIRS INSTEAD OF ONLY REJECTING

Both mistakes this model actually commits are referential, not semantic: a dependency on a
node it never wrote, and a node whose skill came out empty. Neither is a wrong plan -- both
are bookkeeping, and both cost the whole episode's plan. So the validator can now repair
them, under a boundary enforced in code rather than promised in prose: IT MAY DELETE
INSTRUCTIONS AND REORDER THEM; IT MAY NEVER ADD ONE. A fingerprint of every node's skill,
arguments and arm is compared before and after, and the repair raises if anything appeared.
Every repair is named in the episode report, so a reader sees per seed how much of the
executed plan was the model's. It is off by default, so every previously committed number
still reproduces exactly.

What it buys, stated without inflation: it has not made a single plan executable. What it
changed is WHERE the plan dies. Before, every rejection read "node n3 depends on unknown
node(s) ['n2']" -- an error about JSON. After, they read "plan references objects that are
not in the scene: ['water']" and "plan picks ['fork', 'spoon'] and never places them -- the
episode would end with them still in a gripper". Those are errors about the TASK, and they
are the ones a larger planner or a constrained decoder would have to fix. That experiment
also caught the repair layer being wrong: its first version rescued a collapsed generation
into a valid one-step "plan" (open the drawer, stop) that the scheduler accepted and an
episode would have run and failed. The validator now rejects a graph that moves nothing, and
one that ends with an object still in a gripper.

SECOND: those numbers use simulator state for object positions, not perception. Perception
is measured separately and reported. With the colour detector driving all five objects the
task fails outright, 0 of 10: it places the bottle to 10 mm and the mug to 8 mm, but the
plate to 141 mm and the cutlery to 87-157 mm against a 45 mm tolerance -- they are small and
achromatic and HSV has nothing to lock onto. Driving only the two objects it can actually
place, the task finishes 7 of 10. That is the honest edge of the perception path.

Those are IN-LOOP errors -- the mean across every perception tick of a running episode, with
two arms crossing the overhead view. docs/06 also reports a STATIC table (one frame, arms
parked: plate 87 mm, spoon 46 mm, fork 23 mm). Both are in the repo and they are not in
conflict: static is the detector's ceiling, in-loop is what the robot ran on. Quote the
in-loop numbers. Two further conditions: only x and y come from pixels -- z is read from the
object library -- and an object the detector misses keeps its last believed position. So
these runs are perception-driven for what is seen and oracle-backed for what is missed.

Neither exported OpenVINO model is load-bearing for any success number here, and it is
better to say so than to have it found: the INT4 planner is rejected on every seed, and the
INT8 detector is exported, swept and benchmarked but never called at run time -- an HSV
colour detector is what closes the perception loop. What the OpenVINO work buys is the
export path, the device x precision sweep, and one finding that transfers to any deployment
(the CPU plugin silently runs bf16, so a latency without its precision is not a
measurement).

THIRD: the Intel figures were measured on an AMD Ryzen AI 9 HX 370, because Core Ultra
hardware was not available to us. The export path and the device-precision sweep are real
and reproduce with one command on target hardware. The sweep's "GPU" row is this machine's
NVIDIA dGPU; there is no iGPU row and no NPU row, because this box has neither.

FOURTH, and the condition to attach to the headline: 100/100 is a rate over a SCREENED
distribution at a STATED tolerance. Every seed is validated before it runs -- if any object,
slot, hand-off point or drawer position is unreachable with 35 mm of standoff margin, the
seed is rejected -- and a placement counts as correct within 45 mm on a plate of 26 mm
radius. Both are deliberate: an episode should only be able to fail for a reason the policy
could actually have handled. But it is a conditioned number and is not quoted here without
its conditions.

64% of each plan is parallelisable ON PAPER: steps the dependency graph would permit to
overlap. The executor walks them in order and parks the idle arm. The hand-off is a
PLACE-AND-PICK THROUGH A SHARED CELL, not a mid-air exchange -- arm A sets the plate down in
the lens and withdraws completely before arm B moves in, because in-air transfer
self-collides -- so both grippers are never on the object at the same instant. THE POUR IS
THE ONE STEP IN WHICH BOTH ARMS MOVE SIMULTANEOUSLY: B holds the mug while A tips the
bottle. That figure used to read 82%, and the 18
points were bought deliberately -- the region both arms can reach is about 200 x 90 mm and a
laid place setting is 190 mm wide, so once the setting is down a hand-off has nowhere to put
an object. The scheduler treats that lens as a resource and makes everything wait for the
hand-off. That one change took success from 50% to 80%.

INTEL / OPENVINO

The planner is exported at INT4 (bandwidth-bound, called once per episode, so weight
compression buys the most). The detector is exported at INT8 (box regression tolerates
8-bit well) and swept, but -- stated plainly -- it is not the detector in the loop; OWLv2
scores 130-370 mm on untextured primitives, so an HSV colour detector closes the loop
instead. The control loop stays fp32 on CPU, where it costs microseconds and quantising
would only add risk.

bench_openvino.py sweeps every available device against every precision rather than trusting
AUTO, so the report shows the trade-off instead of one figure -- and it prints the host CPU,
stating whether it is a Core Ultra Series 2/3 part.

[If benchmarked off-target, keep this paragraph and delete this line:]
Hardware note: the reported figures come from an AMD Ryzen AI 9 HX 370, because Core Ultra
hardware was not available to us. The NPU and iGPU selection path is implemented and
auto-selects on a Core Ultra; the Series 2/3 numbers reproduce with one command on target
hardware. We would rather report the machine we actually measured than a number we did not.

ENGINEERING

The MJCF is generated from the same link constants the IK uses, so forward kinematics and
the simulator agree by construction -- a drift between a hand-written URDF and the solver is
the most common reason a "working" manipulation demo quietly misses its grasps, and it is
invisible until contact.

The IK is closed-form. Joints 1-3 are parallel and joint 4 spins the gripper about its own
axis, so position depends on four joints and tool direction on their sum: no iterative
solver, no singularities inside the workspace, FK(IK(p)) = p to 1e-16.

Two thirds of the system -- kinematics, geometry, planning, scheduling, skill construction --
needs no simulator and is unit-tested without one. 221 tests run in seconds on any machine.

Most of what took this from 0% to 100% was not tuning. It was finding places where two parts
of the system disagreed about geometry, and making them agree: goal slots that were never
checked against the cabinet, a bottle taller than the height an arm can carry over, a fork
described to the clearance checks as an 8 mm disc when it was 108 mm long, a transfer point
chosen before anything had moved. Each of those failed several steps away from its cause,
which is why the repository ships the tools that localise them -- per-waypoint object
tracking in the object's own frame, per-tick slip detection with contact dumps, a drawer
watcher -- and why every constant in the geometry file carries the measurement that set it.

VOICE

Nobody sitting at a table types a JSON task graph. Speechmatics real-time transcription
feeds the same planner the typed path uses, in English or Italian, with partial hypotheses
shown while the person is still speaking.

Run it: pip install -r requirements.txt && python scripts/run_episode.py --seed 3
```

---

---

## Team on lablab (create this first — no team, no submit button)

`lablab.ai/ai-hackathons/ai-infra-summit-hackathon/team/create`

**Team name**

```
Apparecchiato
```

> Named deliberately. Another entrant on this same track had already registered a team
> called **"Mise"**, building a very similar simulation-first bimanual system — two
> near-identical names side by side in the judging list would have read as a fork.
> *Apparecchiato* ("laid", as in a laid table) is unmistakably yours, and the Italian
> also lands the voice demo: `apparecchia la tavola` is a real command the planner parses.

**Description** (928 characters, fits the 1000 limit)

```
Apparecchiato - two simulated SO-101 arms set a dinner table from a spoken command, handing objects to each other when neither arm can reach alone. Building for the online Intel track (Bimanual VLA Manipulation with Multi-Modal Reasoning, dinner-table option), plus the Speechmatics bonus award. Approach: a vision-language model running on OpenVINO reads the camera views and the instruction and emits a task graph. That graph is validated against the real workspace geometry before anything moves, then scheduled across both arms under reachability, grasp-state and shared-workspace constraints. Hand-offs are emergent, not scripted: one appears whenever the arm that can pick an object cannot reach where it belongs. Speechmatics real-time transcription supplies the spoken command (English and Italian). Stack: MuJoCo, OpenVINO (INT4 planner, INT8 detector), Intel Core Ultra deployment target. Solo builder, based in Italy.
```

**Other fields**

- Who can join: **Closed** (solo). Switch to *Open* only if you decide you want a teammate.
- Timezone: **UTC +2:00**
- Cover image: optional, skip for now


## Cover image

What shipped (`out/cover.png`, rendered by `scripts/make_cover.py --at 9.4`): both arms
converging over the laid setting, arm B holding the mug, the drawer open on arm A's side,
the bottle waiting at the left. Markers off — the table reads without annotation.

If you re-pick the moment, note that the hand-off object is **always the plate**, never the
spoon, and that the cover camera is the same corrected cinematic shot the clips use — see
`CINEMATIC_EYE` in `apparecchiato/sim/scene.py` for why the old one was wrong.

Subtitle line, verbatim — this is the one sentence every judge is guaranteed to read, so it
must not claim anything the README retracts:

```
two SO-101 arms . plans validated against physics . 100/100 seeds
```

Do **not** restore "VLM planner on OpenVINO" to this line (audit-allow). The VLM's plan is rejected on
every seed; the 100/100 belongs to the deterministic fallback. `scripts/make_cover.py`
holds the string.

---

## Submission checklist

- [ ] Team created on lablab (required before the submit button appears)
- [ ] **Online Intel track selected** — one track per project
- [ ] Speechmatics bonus flagged
- [ ] Public GitHub repo, cloned and tested from a clean folder
- [ ] `out/eval/eval_report.md` committed and linked from the README
- [ ] `out/bench/bench_report.md` committed and linked from the README
- [ ] Demo video (< 3 min) covering 10 seeds, the hand-off, and the voice command
- [ ] Slides
- [ ] Cover image
- [ ] All bracketed numbers replaced
- [ ] Submitted by **17:00 CEST Tue 16 Sep** — three hours of slack, not three minutes
