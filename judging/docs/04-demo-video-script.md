# Demo video — script and shot list

**Target: under 3 minutes.** Judges watch dozens. The first fifteen seconds decide whether
they watch the rest.

The brief asks the video to make *the command, the scene variation, and the robot outcome
easy to verify*. Every shot below serves one of those three.

---

## 0:00–0:15 — the hook, no preamble

Cold open on the simulator. A person speaks:

> "Set the table and pour me some water."

On screen, the Speechmatics transcript appears word by word as they speak. Then the task
graph renders. Then both arms start moving. No title card, no logo, no "hi, I'm…".

Voice-over, over the motion:

> "No script. The plan you just saw was written by the model, and checked against the
> geometry of the table before a single joint moved."

---

## 0:15–0:50 — one full episode, uncut

Cinematic camera. Let it play. Caption the steps as they happen:

The real order, from `scripts/clip_sheet.py` (run it — the captions are generated, not
transcribed, so they cannot drift from what the clip shows):

1. `open drawer` — arm A
2. `pick plate` — arm A (only A reaches it)
3. **`hand-off` — A → B** ← hold two beats on this
4. `place plate` — arm B (only B reaches the slot)
5. `pick fork` → 6. `place fork` — arm A, both ends in its own workspace
7. `pick mug` → 8. `place mug` — arm B
9. `pick spoon` → 10. `place spoon` — arm A
11. `pour` — B steadies the mug, A tips the bottle

Voice-over lands on the hand-off:

> "Nobody scripted this exchange. The plate lands where only the left arm reaches, and it
> belongs where only the right arm reaches. The scheduler worked that out from the
> workspace geometry — a hand-off appears exactly when pick-reach and place-reach don't
> overlap. Move the plate within reach of its own slot and the step simply disappears."

That sentence is the single most persuasive thing in the video. Do not rush it.

---

## 0:50–1:20 — the same command, ten different tables

Fast montage, 3 seconds per seed, 10 seeds. Corner overlay per clip:

```
seed 04   mass 0.18 kg   friction 0.91   light 1.22   ✓
```

Voice-over:

> "Ten randomised seeds. Different placements, masses, frictions, object sizes, lighting
> and backgrounds — same instruction, same code, no per-scene tuning. Ten out of ten. And
> a hundred seeds, of which we only ever looked at the first ten: a hundred out of a
> hundred, eleven steps of eleven, every time."

Clips are already rendered: `out/eval_vlm/video/seed000.mp4` … `seed009.mp4`, recorded from
the cinematic camera during the `vlm+rules` run that produced
`results/eval_vlm_seeds0-9.md`. Re-render with
`python scripts/run_eval.py --seeds 0-9 --planner vlm+rules --out out/eval_vlm --camera cinematic`.

Then hold the subgoal table on screen for three seconds.

> "Reported per subgoal, not as one number, because 'the pour fails' and 'the grasps fail'
> are different problems."

---

## 1:20–2:00 — the Intel layer

Screen recording, not slides.

1. `python scripts/bench_openvino.py --list-devices` — **cut this shot.** On this machine it
   lists CPU and an NVIDIA dGPU, not an Intel iGPU or NPU. Showing it invites exactly the
   question whose honest answer is "we did not have the hardware". Say that in the
   voice-over instead, over the export command, which is the part that is real.
2. The export commands running — that path is genuinely Intel-specific and genuinely works
3. The device × precision table filling in, with the host CPU line visible at the top
4. The perception numbers: `results/eval_perception.md` beside `eval_perception_partial.md`

Voice-over:

> "The planner is a 2B vision-language model at INT4. The detector is INT8. The control
> loop stays fp32 on CPU, where it costs microseconds and quantising would only add risk.
> The benchmark sweeps every device against every precision, and prints the CPU it ran on —
> which here is an AMD Ryzen, because we did not have Core Ultra hardware. The export path
> and the sweep run on target silicon with one command. The latencies on screen do not."

**Say the AMD sentence inside the first 20 seconds of this section, not at the end of it.**
A judge who spots it themselves scores it as something you hid; a judge who hears it from
you scores it as something you controlled for.

Then the perception line, which is the other thing worth saying out loud:

> "Those success rates use simulator state for object positions. With the colour detector
> actually driving the arms, the bottle lands within ten millimetres and the mug within
> eight — but the plate is a hundred and forty off, and the task fails. Give perception
> only the two objects it can place and seven of ten episodes finish. That is the edge of
> it, and we would rather show you the edge than the middle."

---

## 2:00–2:30 — voice, properly

Show a person actually speaking to it. Face or hands visible — it has to be obvious this is
a microphone and not a config flag. Do one command in English and one in Italian:

> "Apparecchia la tavola e versa l'acqua nella tazza."

Same pipeline, same plan structure, different language.

> "Speechmatics real-time transcription feeds the same planner. The interface to a robot
> that sets your table should be talking to it."

---

## 2:30–2:50 — close

One slide, four lines, six seconds:

- Emergent hand-offs from workspace geometry, not scripts
- Validated plans: physically impossible ones are rejected before execution
- OpenVINO INT4 planner + INT8 detector, exported and swept on CPU (AMD host — say so)
- 208 tests, reproducible in one command

Then: repo URL on screen, held for five full seconds.

---

## Production notes

- **Record at 1080p minimum.** Compressed simulator footage turns to mush.
- **Subtitle everything.** Judges watch on mute more often than anyone admits.
- **Show one failure.** A seed that fails, with the subgoal table explaining exactly where —
  it costs ten seconds and buys more credibility than any success montage.
- Capture with `--record`; assemble with any editor. `imageio` writes mp4 directly.
- Do not narrate the architecture diagram. Show the robot; explain while it moves.
