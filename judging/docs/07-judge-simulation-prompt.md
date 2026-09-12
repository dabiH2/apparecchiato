# Judge simulation — prompt to paste

Paste the block below into a fresh Claude session (Cowork, with the repo folder
connected, or claude.ai with the README + results + submission copy attached).
It runs an adversarial panel against Apparecchiato and publishes a dashboard.

Run it **before** submitting, and again after any change to the copy. The point
is not the score — it is the "kill shots" section, which is where a real judge
would push and where this project has already been caught once.

**Known soft spots to make sure the panel actually probes** (do not put these in
the prompt — see whether it finds them on its own; if it misses them, the panel
was too easy):

- The hand-off is always the plate, always A→B. Derived, but invariant.
- Success is measured in simulation only. No hardware, no sim-to-real.
- The Intel numbers come from an AMD Ryzen AI 9 HX 370, not a Core Ultra.
- The VLM's plan is discarded whenever the validator rejects it — how often is
  the headline number actually the *rules* planner's?
- 64% parallel, down from 82%: a judge may read that as a regression.
- Open-vocabulary detection does not work on this scene; the shipped perception
  path is HSV colour segmentation.

---

```
You are simulating the judging panel for the Intel track of the AI Infra Summit
Hackathon (lablab.ai, Sep 2026): "Bimanual VLA Manipulation with Multi-Modal
Reasoning", online, simulation-first. You are scoring one submission,
Apparecchiato.

Constitute a panel of four judges, each with a distinct and consistent bias, and
keep them in character throughout:

  1. INTEL PLATFORM ENGINEER. Cares about OpenVINO used properly, device and
     precision choices justified by measurement, and whether anything was
     actually run on Core Ultra hardware. Suspicious of benchmark numbers
     without the precision and host stated.
  2. ROBOTICS RESEARCHER. Cares about whether the manipulation is real or
     choreographed, whether randomisation is meaningful or cosmetic, whether the
     success metric can be gamed, and what would break on hardware.
  3. ML / VLA SPECIALIST. Cares about how much work the learned model is really
     doing versus the hand-written fallback, and whether "VLA" is earned.
  4. PRODUCT JUDGE. Cares about the demo video, the story, whether a
     non-specialist understands it in 90 seconds, and whether the claims in the
     copy survive contact with the repo.

SCORE against the published rubric, out of 100:
  - Task completion .......... 30
  - Bimanual / VLA quality ... 20
  - OpenVINO / Intel usage ... 20
  - Robustness ............... 15
  - Technical quality ........ 10
  - Innovation ............... 5

METHOD, in this order:

A. Read the material provided (README, results/, docs/, the submission copy, the
   video script, and any code you are given access to). List, in one line each,
   every factual claim the submission makes that a judge could check.

B. For each claim, mark it VERIFIED (the material shows it), UNVERIFIABLE (would
   need something not provided), or CONTRADICTED (the material shows otherwise).
   Quote the specific line that settles it. Do not be generous: a claim
   supported only by the submission restating itself is UNVERIFIABLE.

C. Each judge scores each rubric category independently, with one sentence of
   reasoning, then the panel converges on a single number per category. Where
   judges disagree by more than 3 points, record the disagreement rather than
   averaging it away — that gap is information.

D. KILL SHOTS. The most important section. List the five questions that would do
   the most damage if asked in a live Q&A, hardest first. For each: the question
   as a judge would phrase it, what the honest answer is, and how bad the honest
   answer looks on a scale of "fine" / "awkward" / "damaging". Be adversarial
   here. Look specifically for: claims that are true on the demo seeds but not in
   general; metrics that could be satisfied without doing the hard thing;
   anything measured on the wrong hardware; and any place where a learned
   component could be removed without changing the result.

E. HIGHEST-LEVERAGE FIXES. Rank what to change before submitting by
   (points gained) / (hours of work). Be specific — "improve the video" is not a
   fix, "cut the first 15 seconds so the hand-off lands before 0:20" is. Mark
   anything that is a copy change rather than a code change, since those are
   nearly free.

F. PREDICTED PLACING against a field of roughly 30 submissions, with the
   reasoning in two sentences.

THEN produce the output as a single self-contained HTML dashboard, published as
an artifact, containing:

  - A header with the total score out of 100 and the predicted placing.
  - A rubric table: category, max, score, and a bar, with the panel's one-line
    reason per row. Colour by fraction earned, not by absolute score.
  - A claims audit table: claim, verdict, and the quoted evidence. Sortable or at
    least grouped by verdict, with CONTRADICTED first.
  - The kill shots as cards, hardest first, each showing the question, the honest
    answer, and the damage rating.
  - The fix list as a ranked table with an effort column and a "copy only" flag.
  - A short panel-disagreement section wherever judges differed by more than 3.

Design rules for the dashboard: work at phone width; readable in light and dark;
no external assets; numbers large enough to read across a room. Do not invent
data to fill a chart — if a section has nothing in it, say so in the section.

Be harsh. A panel that gives this 90+ without argument has not done its job, and
a flattering simulation is worse than none: its only purpose is to find what
breaks before a real judge does.
```

---

## Feeding it the material

With the repo folder connected, point it at:

- `README.md` — the claims
- `results/eval_vlm_seeds0-9.md`, `results/eval_rules_seeds0-99.md` — the numbers
- `results/bench_detector.md` — the OpenVINO sweep
- `docs/06-openvino-findings.md` — the three findings
- `../hackathon-docs/03-submission-copy.md` — what the judges actually read
- `out/montage.mp4` and `out/eval_vlm/video/seed000.mp4` — the demo
- `apparecchiato/scheduler.py` and `apparecchiato/sim/layout.py` — where the
  interesting reasoning lives, and where the constants carry their measurements

## Running it twice

Run once with only the submission copy and the video, which is what most judges
will actually see. Run again with the repo. If the two scores differ a lot, the
repo is carrying the submission and the copy is underselling it — or the copy is
claiming more than the repo supports, which is the dangerous direction.
