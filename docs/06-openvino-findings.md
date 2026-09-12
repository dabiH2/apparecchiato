# OpenVINO: what we exported, what we measured, what broke

Everything here was produced by this project's own scripts on an **AMD Ryzen AI
9 HX 370** host. That is not the Intel Core Ultra Series 2/3 part the challenge
scores, and nothing here pretends otherwise: every script prints its own host CPU
and states plainly whether it is the target part. Re-running them on a Core Ultra
reproduces the tables with real NPU and iGPU rows.

    .\.venv\Scripts\python.exe scripts\verify_openvino.py

## 1. The detector had to be converted by hand

`optimum-cli export openvino --task zero-shot-object-detection` does not work,
and not because of a version skew: optimum-intel has **no OpenVINO export
configuration for that task at all**. The CLI reports the model as "a custom or
unsupported architecture".

So `scripts/export_detector_openvino.py` goes through the OpenVINO API directly:
`torch.export` to capture the graph, `ov.convert_model` to lower it,
`nncf.compress_weights(..., INT8_ASYM)` to quantise. More code than a CLI call,
and considerably more control -- the traced graph, the input shapes and the
quantisation mode are all visible in one file.

Two things were needed to make the capture work at all:

- **eager attention.** The SDPA path in current transformers builds its masks
  with code that is not traceable; `torch.jit.trace` dies inside `sdpa_mask` with
  `IndexError: tuple index out of range`.
- **`torch.export` rather than tracing.** Tracing fails even with eager
  attention; `torch.export(strict=False)` captures cleanly. Both paths are kept
  in the script and the fidelity of whichever one ran is reported, because
  neither is reliably correct across models.

## 2. The CPU plugin silently runs bf16, and it looks exactly like a broken export

The first fidelity check compared the converted IR against the PyTorch model and
reported a **mean logit difference of 0.17** -- far too large for a faithful
conversion. The obvious conclusion was that the export was wrong.

It was not. OpenVINO's CPU plugin defaults to **bf16 inference precision** on any
host that supports it, and bf16 carries about three decimal digits. Pinning
`INFERENCE_PRECISION_HINT: f32` for the comparison brought the difference to
**2.1e-05**, which is what a correct conversion looks like.

The lesson generalises past this project: **a latency number without its
precision is not a measurement.** The plugin will quietly pick a faster, less
accurate path, and if you are benchmarking you are probably getting bf16 without
having asked for it.

## 3. Max-diff is the wrong accuracy metric for a detector

Even in f32, the *maximum* logit difference over all 576 patches stayed around
4.0 while the mean was 2e-05. OWL-ViT's class head divides by an embedding norm
with a 1e-6 floor, so on empty background patches -- most of the image -- a 1e-6
numerical difference is amplified into whole logits. The maximum is dominated by
patches no detection will ever come from.

What the export script reports instead is the decision: **does the same patch
still win each prompt, and does its box still land in the same place.**

| Comparison | mean abs dlogit | winning patch unchanged | abs dbox at the winner |
| --- | --- | --- | --- |
| IR (f32) vs PyTorch | 2.1e-05 | 6/6 | 1e-06 |
| INT8 vs IR | 1.5e-01 | 5/6 | 2.6e-03 |

INT8 weight compression costs about a quarter of a percent of image width on the
boxes that matter. That is the honest quantisation cost for this model.

## 4. The open-vocabulary detector does not work on this scene, and that is the finding

OWL-ViT (patch32, 768 px) and OWLv2 (patch16, 960 px) were both exported and both
measured against the simulator's ground truth with `scripts/check_detector.py`.
Across seeds, viewpoints, thresholds down to 0.005, and prompts reworded for
appearance ("a small white round plate", "a blue cup"), back-projected positions
were **130-370 mm from truth** -- no signal at all.

This is not a bug in the export; the IR matches PyTorch to 2e-05. It is that
OWL-ViT and OWLv2 are trained on natural photographs and this scene is untextured
coloured primitives. Scaling the render up and the patch size down (3.7x the
patch density) did not change it.

So the perception path that actually closes the loop is `ColourDetector`: HSV
segmentation on the rendered frame, reading pixels with **no privileged simulator
state**.

| Object | Recall | Median error |
| --- | --- | --- |
| bottle | 10/10 | 1 mm |
| mug | 9/10 | 6 mm |
| fork | 10/10 | 23 mm |
| spoon | 10/10 | 46 mm |
| plate | 10/10 | 87 mm |

Three bugs were found getting there, and all three are the same shape -- something
in the scene impersonating something else:

- **Arm A was the same colour as the mug** (0.06 apart in RGB). The detector
  segmented the forearm and put the mug 450 mm away. A robot cell whose gripper
  is the colour of the parts is a badly designed cell; the arms are now dark and
  desaturated.
- **The goal markers were the bottle's hue.** The debug overlay drawn to show
  where objects *should* go was green at hue 0.381; the bottle is hue 0.38, and
  the markers are 44 mm across against the bottle's 28 mm, so the detector found
  the marker every time. They are violet now.
- **Back-projection assumed everything lies flat on the table.** An overhead
  camera sees the *top* of a 60 mm mug, and a ray through that pixel continued to
  z = 0 lands to one side by height x tan(off-axis angle) -- 240 mm at the table
  edge, even with the blob centroid two pixels from truth. Each object is now
  back-projected through its own known silhouette height.

The randomised table and backdrop colours were also constrained to warm, clearly
saturated tones. Independent per-channel randomisation could produce a near-grey
tabletop, which is the same colour class as a white plate and a steel fork.
**Randomising a scene is only useful if the randomisation cannot manufacture a
decoy.**

## 5. The VLM planner: quantisation, and what a 2B model gets wrong

`Qwen/Qwen2-VL-2B-Instruct` exported through `optimum-cli` at both INT4 and INT8.
Note that optimum-intel requires **transformers < 5.0** for this export.

| Weights | Size on disk | Language model |
| --- | --- | --- |
| INT4 (ratio 1.0) | 1844 MB | 873 MB |
| INT8 | 2475 MB | ~1.5 GB |

Two failures had to be fixed before the model produced anything usable, and
neither was about quantisation:

- **No chat template.** The planner was feeding the model a bare string. An
  instruct-tuned model given no assistant turn has nothing to end: it opened its
  JSON correctly and then repeated "the fork is in the drawer" until it hit the
  token limit. Going through `apply_chat_template` fixed it.
- **The system prompt was being sent twice** -- once inside the user text and once
  as a system message. A small model given the same rules twice tends to answer
  twice.

What remains is a real quality limit. At both INT4 and INT8 the model produces
**structurally correct plans with damaged punctuation**: `"skill":": "pick"`, a
stray quote before a key, a missing comma between objects. The tokenizer was
checked (`scripts/probe_tokenizer.py`) and round-trips this JSON exactly, so it is
the model, not the encoding.

The architecture already had the right answer to this, which is why it was built
this way: **the model decides what to do, geometry decides whether it is possible,
and a validator sits on the boundary.** Concretely there are four gates, each
measured separately:

1. **syntax repair** -- a fixed set of rules for the artefacts this model actually
   produces, touching separators only, never inventing keys or values;
2. **per-node salvage** -- nodes that cannot be parsed are dropped and counted
   rather than costing the whole plan;
3. **`validate_against_scene`** -- every object, slot and arm assignment must
   exist and be reachable;
4. **the scheduler** -- a well-formed plan can still be physically impossible
   ("pick the fork while still holding the plate"), and only the scheduler knows.

A plan rejected at any gate is handed **back to the model with the reason**, once.
That retry is the cheap half: a small model's failures are precisely stateable,
and one sentence of feedback is worth more than either a larger token budget or
higher weight precision. It is bounded at one retry, after which the
deterministic rules planner takes the episode -- so the robot always has a valid
plan, and the VLM is an improvement rather than a dependency.

Measured results: `out/planner_bench.md`, produced by `scripts/bench_planner.py`.

## Reproducing all of this

```powershell
.\.venv\Scripts\python.exe scripts\verify_openvino.py
.\.venv\Scripts\python.exe scripts\export_detector_openvino.py
.\.venv\Scripts\python.exe scripts\export_vlm_openvino.py --weight-format int4
.\.venv\Scripts\python.exe scripts\check_detector.py --detector colour --seeds 10
.\.venv\Scripts\python.exe scripts\bench_planner.py --seeds 4
.\.venv\Scripts\python.exe scripts\bench_openvino.py
```

On a Core Ultra Series 2/3 the same commands produce NPU and iGPU rows. Nothing
needs to change; the scripts sweep whatever devices OpenVINO reports.
