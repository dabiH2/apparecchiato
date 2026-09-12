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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to an OpenVINO IR .xml")
    ap.add_argument("--devices", help="comma-separated, e.g. CPU,GPU,NPU")
    ap.add_argument("--precisions", default="native,f16,i8")
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

    rep = benchmark(args.model,
                    devices=args.devices.split(",") if args.devices else None,
                    precisions=tuple(args.precisions.split(",")),
                    iterations=args.iterations)
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
