#!/usr/bin/env python3
"""Export the open-vocabulary detector to OpenVINO IR, weight-compressed to INT8.

  python scripts/export_detector_openvino.py
  python scripts/export_detector_openvino.py --model google/owlv2-base-patch16-ensemble \
      --out models/owlv2-ov-int8 --weight-format int8

Why this converts by hand instead of calling `optimum-cli export openvino`:
optimum-intel has no OpenVINO export configuration for the
`zero-shot-object-detection` task at all -- the CLI fails with "custom or
unsupported architecture" -- so the model goes through `ov.convert_model` and
`nncf.compress_weights` directly. That is a few more lines and rather more
control: the traced graph, the input shapes and the quantisation mode are all
visible here rather than hidden behind a task string.

INT8 weight compression is the right default for detection: it runs every
perception tick, weights dominate the footprint, and box regression tolerates
8 bits far better than the planner's token generation does.

The text prompts are baked in at a FIXED count. This is a fixed-vocabulary
application -- six things can be on this table -- so a static text batch lets the
graph specialise, and it removes a whole class of dynamic-shape surprises on the
NPU, which is the device this is ultimately meant to run on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.perception.detect import PROMPTS                    # noqa: E402

IMAGE_SIZE = {"owlvit": 768, "owlv2": 960}


def _wrap(model):
    """Wrap the detector so the traced graph returns plain tensors.

    `ov.convert_model` traces a callable; a HuggingFace model returns a
    dataclass, and the two do not mix. Only logits and boxes are needed
    downstream, so this returns exactly those, in a fixed order.
    """
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, pixel_values, attention_mask):
            out = self.m(input_ids=input_ids, pixel_values=pixel_values,
                         attention_mask=attention_mask)
            return out.logits, out.pred_boxes

    return Wrapper(model).eval()


def sample_frame(side: int):
    """One rendered overhead frame from the simulator, as the detector will see it.

    Falls back to noise if MuJoCo is unavailable, which is still far better than
    a blank image for checking numerical fidelity.
    """
    import numpy as np
    try:
        from apparecchiato.sim.env import TableEnv
        from apparecchiato.sim.layout import sample_scene
        env = TableEnv(sample_scene(0))
        env.settle(0.5)
        frame = env.render("overhead")
        env.close()
        from PIL import Image
        return np.asarray(Image.fromarray(frame).resize((side, side)))
    except Exception as exc:                                           # noqa: BLE001
        print(f"  (no simulator frame available: {exc}; using noise)", file=sys.stderr)
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (side, side, 3), dtype=np.uint8)


def agreement(logits, boxes, ref_logits, ref_boxes, np) -> float:
    """How much two versions of this detector actually disagree.

    Not max|diff|. OWL-ViT's class head divides by an embedding norm with a
    1e-6 floor, so on empty background patches -- most of the image -- a 1e-6
    numerical difference is amplified into whole logits. The raw maximum over
    576 patches is therefore dominated by patches no detection will ever come
    from, and reports 4.0 for a conversion that is bit-for-bit sound everywhere
    that matters.

    What matters is the decision: does the same patch still win each prompt, and
    does its box still land in the same place. Returns the logit difference
    restricted to the winning patches, which is the number worth thresholding.
    """
    d = np.abs(logits - ref_logits)
    print(f"  |logit| diff over all patches: mean {d.mean():.2e}  "
          f"median {np.median(d):.2e}  max {d.max():.4f}")
    top_ref = ref_logits[0].argmax(axis=0)
    top_new = logits[0].argmax(axis=0)
    same = int((top_ref == top_new).sum())
    n = len(top_ref)
    print(f"  winning patch per prompt unchanged: {same}/{n}")
    at_top = float(np.abs(logits[0][top_ref, np.arange(n)]
                          - ref_logits[0][top_ref, np.arange(n)]).max())
    box_at_top = float(np.abs(boxes[0][top_ref] - ref_boxes[0][top_ref]).max())
    print(f"  |logit| diff at the winning patches: {at_top:.6f}")
    print(f"  |box|   diff at the winning patches: {box_at_top:.6f}")
    if same < n:
        return float("inf")
    return at_top


def convert(wrapper, example, ov, torch, how: str = "auto"):
    """torch module -> OpenVINO model, by whichever capture path is faithful.

    Two paths exist and they are not equivalent. `torch.export` captures the
    graph symbolically and copes with control flow that `torch.jit.trace`
    chokes on; tracing, in exchange, records exactly what ran. Neither is
    reliably correct for a given model, so the caller can force one and
    main() checks the result against the torch reference before trusting it.
    """
    if how in ("auto", "export", "export-strict"):
        try:
            strict = (how == "export-strict")
            print(f"converting via torch.export (strict={strict}) ...", flush=True)
            with torch.no_grad():
                ep = torch.export.export(wrapper, args=(), kwargs=dict(example),
                                         strict=strict)
            return ov.convert_model(ep)
        except Exception as exc:                                       # noqa: BLE001
            if how == "export":
                raise
            print(f"  torch.export failed ({type(exc).__name__}: {exc})"[:300],
                  file=sys.stderr)
    print("converting via torch.jit.trace ...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (example["input_ids"], example["pixel_values"],
                      example["attention_mask"]), strict=False)
    return ov.convert_model(traced, example_input=dict(example))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/owlvit-base-patch32",
                    help="HF id of an OWL-ViT / OWLv2 open-vocabulary detector")
    ap.add_argument("--out", default="models/owlvit-ov-int8")
    ap.add_argument("--weight-format", default="int8", choices=("fp32", "fp16", "int8"))
    ap.add_argument("--converter", default="auto",
                    choices=("auto", "export", "export-strict", "trace"),
                    help="graph capture path; they are not equivalent, and the "
                         "script reports the fidelity of whichever one ran")
    ap.add_argument("--max-logit-drift", type=float, default=0.05,
                    help="fail the export if the uncompressed IR disagrees with "
                         "torch by more than this")
    args = ap.parse_args()

    import numpy as np
    import openvino as ov
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    # eager attention, because the SDPA path builds its masks with code that is
    # not traceable: transformers' sdpa_mask indexes a shape that is a plain int
    # under torch.jit.trace and dies with "tuple index out of range".
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model, attn_implementation="eager")
    model.eval()

    side = IMAGE_SIZE.get(model.config.model_type, 768)
    # A REAL frame, not a black square. Conversion fidelity checked on a blank
    # image is worthless: with no signal the boxes are arbitrary, so a 1e-6
    # change in logits reorders them and the comparison reports half the image
    # of "error" whether or not anything is wrong.
    img = Image.fromarray(sample_frame(side))
    inputs = processor(text=[list(PROMPTS)], images=img, return_tensors="pt")
    example = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs["attention_mask"],
    }
    for k, v in example.items():
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")

    wrapper = _wrap(model)
    with torch.no_grad():
        ref_logits, ref_boxes = wrapper(**example)
    print(f"  logits         {tuple(ref_logits.shape)}")
    print(f"  pred_boxes     {tuple(ref_boxes.shape)}")

    ov_model = convert(wrapper, example, ov, torch, args.converter)

    # Two separate questions, checked separately, because conflating them tells
    # you nothing: (1) is the CONVERSION faithful -- compare the uncompressed IR
    # against torch, expect ~1e-5; (2) how much does QUANTISATION cost -- compare
    # INT8 against that same IR. A single number against torch cannot distinguish
    # a broken export from an aggressive but working one.
    #
    # INFERENCE_PRECISION_HINT f32 is not optional here. The CPU plugin defaults
    # to bf16 on any host that supports it -- this one does -- and bf16 carries
    # about three decimal digits, which showed up as a uniform ~0.1 shift across
    # the whole logit field and looked exactly like a broken export. It is not:
    # it is the runtime quietly choosing a faster precision. Fidelity has to be
    # checked in f32 or it measures the plugin's default rather than the graph.
    np_example = {k: v.numpy() for k, v in example.items()}
    f32 = {"INFERENCE_PRECISION_HINT": "f32"}
    fp = ov.Core().compile_model(ov_model, "CPU", f32)
    fp_out = fp(np_example)
    fp_logits, fp_boxes = fp_out[fp.output(0)], fp_out[fp.output(1)]
    print("conversion fidelity (uncompressed IR vs torch):")
    drift = agreement(fp_logits, fp_boxes, ref_logits.numpy(), ref_boxes.numpy(), np)
    if drift > args.max_logit_drift:
        # Stop here rather than shipping a model that is quietly wrong. A broken
        # conversion looks exactly like heavy quantisation loss downstream, and
        # it would be measured, benchmarked and written up as if it were real.
        print(f"\nthe conversion is NOT faithful (logit drift {drift:.3f} > "
              f"{args.max_logit_drift}). Nothing was saved.\n"
              f"Try --converter {'trace' if args.converter != 'trace' else 'export'}.",
              file=sys.stderr)
        return 1

    if args.weight_format == "int8":
        import nncf
        print("compressing weights to INT8 ...", flush=True)
        ov_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)

    os.makedirs(args.out, exist_ok=True)
    xml = os.path.join(args.out, "openvino_model.xml")
    ov.save_model(ov_model, xml, compress_to_fp16=(args.weight_format == "fp16"))
    processor.save_pretrained(args.out)

    # The runtime needs to know what the graph was built for: which prompts, in
    # which order (the detector returns prompt INDICES), and at what image size.
    # Writing it next to the IR means the loader never has to guess.
    with open(os.path.join(args.out, "apparecchiato_detector.json"), "w") as f:
        json.dump({
            "source_model": args.model,
            "model_type": model.config.model_type,
            "weight_format": args.weight_format,
            "prompts": list(PROMPTS),
            "image_size": side,
            "input_order": ["input_ids", "pixel_values", "attention_mask"],
            "output_order": ["logits", "pred_boxes"],
        }, f, indent=2)

    size_mb = sum(os.path.getsize(os.path.join(args.out, f))
                  for f in os.listdir(args.out)
                  if f.endswith((".xml", ".bin"))) / 1e6
    print(f"\nwrote {xml}  ({size_mb:.0f} MB of IR)")

    compiled = ov.Core().compile_model(ov_model, "CPU", f32)
    res = compiled(np_example)
    q_logits, q_boxes = res[compiled.output(0)], res[compiled.output(1)]
    print(f"quantisation cost ({args.weight_format} vs uncompressed IR):")
    agreement(q_logits, q_boxes, fp_logits, fp_boxes, np)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
