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
import os
import subprocess
import sys


def optimum_cli() -> list[str]:
    """How to invoke optimum-cli from THIS interpreter.

    Not the bare name: `optimum-cli` is only on PATH inside an activated venv,
    and these scripts are run as `.\\.venv\\Scripts\\python.exe scripts\\...`
    without activating anything. Use the executable beside the running
    interpreter so the export happens in the same environment that will load it.
    """
    exe = os.path.join(os.path.dirname(sys.executable),
                       "optimum-cli.exe" if os.name == "nt" else "optimum-cli")
    if os.path.exists(exe):
        return [exe]
    return [sys.executable, "-m", "optimum.commands.optimum_cli"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out", default="models/qwen2-vl-2b-ov-int4")
    ap.add_argument("--weight-format", default="int4",
                    choices=("fp32", "fp16", "int8", "int4"))
    ap.add_argument("--ratio", default="1.0", help="int4 ratio for mixed precision")
    args = ap.parse_args()

    cmd = optimum_cli() + ["export", "openvino", "--model", args.model,
                           "--task", "image-text-to-text",
                           "--weight-format", args.weight_format]
    if args.weight_format == "int4":
        cmd += ["--ratio", args.ratio]
    cmd += [args.out]

    print(" ".join(cmd), flush=True)
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError:
        print("optimum-cli not found. Install it with:\n"
              "  .\\.venv\\Scripts\\pip install \"optimum[openvino]\" nncf",
              file=sys.stderr)
        return 2
    if rc == 0:
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(args.out) for f in fs) / 1e6
        print(f"\nwrote {args.out}  ({size:.0f} MB on disk)")
        print("point the planner at it with:\n"
              f"  $env:APPARECCHIATO_VLM_MODEL = '{args.out}'")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
