#!/usr/bin/env python3
"""Benchmark the pipeline's models on Intel CPU / iGPU / NPU via OpenVINO.

  python scripts/bench_openvino.py --model models/owlv2-base-ov-int8/openvino_model.xml
  python scripts/bench_openvino.py --list-devices
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detector_shapes(model_xml: str) -> dict:
    """Recover the detector's real input shapes from the metadata beside its IR.

    The export writes apparecchiato_detector.json next to the model recording
    the prompt list and image size it was traced with. Reading it here means the
    benchmark measures the workload the pipeline actually runs, rather than
    whatever shape a human happened to type on the command line.
    """
    import json
    meta_path = os.path.join(os.path.dirname(model_xml), "apparecchiato_detector.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    n_prompts = len(meta.get("prompts") or [])
    side = int(meta.get("image_size") or 768)
    if not n_prompts:
        return {}
    # 16 is the tokenizer's padded prompt length for this fixed vocabulary.
    return {"input_ids": [n_prompts, 16],
            "attention_mask": [n_prompts, 16],
            "pixel_values": [1, 3, side, side]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to an OpenVINO IR .xml")
    ap.add_argument("--devices", help="comma-separated, e.g. CPU,GPU,NPU")
    ap.add_argument("--precisions", default="native,f32,f16,bf16",
                    help="INFERENCE_PRECISION_HINT values to sweep. 'native' "
                         "means whatever the plugin picks -- which on a CPU that "
                         "supports it is bf16, not f32")
    ap.add_argument("--shape", action="append", default=[],
                    help="pin an input, e.g. --shape pixel_values=1,3,768,768 "
                         "(repeatable). Without this a dynamic model is "
                         "benchmarked at 1x1 and the numbers are meaningless.")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--out", default="out/bench")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    try:
        from apparecchiato.bench import benchmark, describe_devices
    except ImportError as exc:
        print(f"OpenVINO is not installed: {exc}", file=sys.stderr)
        return 2

    if args.list_devices:
        version, devices, names = describe_devices()
        print(f"OpenVINO {version}")
        for d in devices:
            print(f"  {d:6s} {names[d]}")
        return 0

    if not args.model:
        ap.error("--model is required unless --list-devices is given")

    shapes = {}
    for s in args.shape:
        name, _, dims = s.partition("=")
        shapes[name.strip()] = [int(d) for d in dims.split(",") if d.strip()]
    if not shapes:
        shapes = detector_shapes(args.model)

    rep = benchmark(args.model,
                    devices=args.devices.split(",") if args.devices else None,
                    precisions=tuple(args.precisions.split(",")),
                    iterations=args.iterations,
                    shapes=shapes or None)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bench_report.json"), "w", encoding="utf-8") as fh:
        fh.write(rep.to_json())
    with open(os.path.join(args.out, "bench_report.md"), "w", encoding="utf-8") as fh:
        fh.write(rep.to_markdown())
    print(rep.to_markdown())
    print(f"\nwrote {args.out}/bench_report.json and bench_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
