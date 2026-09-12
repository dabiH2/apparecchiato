#!/usr/bin/env python3
"""Export the open-vocabulary detector to OpenVINO IR.

  python scripts/export_detector_openvino.py --model google/owlv2-base-patch16-ensemble \
      --out models/owlv2-base-ov-int8 --weight-format int8

Points APPARECCHIATO_DET_MODEL at the result. INT8 is a good default here: detection runs
every perception tick, so activation-level speedups matter, and box regression
tolerates 8-bit far better than the planner's token generation tolerates it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--out", default="models/owlv2-base-ov-int8")
    ap.add_argument("--weight-format", default="int8",
                    choices=("fp32", "fp16", "int8"))
    args = ap.parse_args()

    cmd = ["optimum-cli", "export", "openvino", "--model", args.model,
           "--task", "zero-shot-object-detection",
           "--weight-format", args.weight_format, args.out]
    print(" ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("optimum-cli not found. Install it with:\n"
              "  pip install 'optimum[openvino]'", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
