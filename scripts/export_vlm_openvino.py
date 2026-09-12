#!/usr/bin/env python3
"""Export a vision-language planner to OpenVINO IR and quantise it.

  python scripts/export_vlm_openvino.py --model Qwen/Qwen2-VL-2B-Instruct \
      --out models/qwen2-vl-2b-ov-int4 --weight-format int4

The exported directory is what APPARECCHIATO_VLM_MODEL points at. INT4 weights are the
right default for a 2B VLM on a Core Ultra: the NPU has limited memory bandwidth
and the planner is a once-per-episode call, so weight compression buys far more
than activation precision does.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out", default="models/qwen2-vl-2b-ov-int4")
    ap.add_argument("--weight-format", default="int4",
                    choices=("fp32", "fp16", "int8", "int4"))
    ap.add_argument("--ratio", default="1.0", help="int4 ratio for mixed precision")
    args = ap.parse_args()

    cmd = ["optimum-cli", "export", "openvino", "--model", args.model,
           "--task", "image-text-to-text", "--weight-format", args.weight_format]
    if args.weight_format == "int4":
        cmd += ["--ratio", args.ratio]
    cmd += [args.out]

    print(" ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("optimum-cli not found. Install it with:\n"
              "  pip install 'optimum[openvino]'", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
