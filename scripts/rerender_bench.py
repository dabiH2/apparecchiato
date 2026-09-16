#!/usr/bin/env python3
"""Re-render a committed `bench_report.json` to markdown, without re-benchmarking.

The JSON is the measurement; the markdown is a view of it. Keeping a script that
regenerates one from the other means the prose around a number can be corrected
without anyone being tempted to retype the number itself -- and it makes the
committed pair verifiable: if `--check` says they disagree, one of them was
edited by hand.

    python scripts/rerender_bench.py                       # results/bench_detector.json
    python scripts/rerender_bench.py --check               # fail if the .md is stale
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apparecchiato.bench.openvino_bench import (BenchReport, DeviceResult,   # noqa: E402
                                                finalise_notes)


def load(path: pathlib.Path) -> BenchReport:
    d = json.loads(path.read_text(encoding="utf-8"))
    rep = BenchReport(host=d.get("host", {}),
                      is_core_ultra=bool(d.get("is_core_ultra")),
                      core_ultra_series_2_3=bool(d.get("core_ultra_series_2_3")),
                      openvino_version=d.get("openvino_version", "unknown"),
                      available_devices=list(d.get("available_devices", [])),
                      results=[DeviceResult(**r) for r in d.get("results", [])],
                      notes=list(d.get("notes", [])))
    return finalise_notes(rep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="results/bench_detector.json")
    ap.add_argument("--out", default=None, help="defaults to the .json path with .md")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit non-zero if the markdown is stale")
    a = ap.parse_args()

    src = pathlib.Path(a.json)
    if not src.exists():
        print(f"{src} not found", file=sys.stderr)
        return 2
    dst = pathlib.Path(a.out) if a.out else src.with_suffix(".md")

    md = load(src).to_markdown() + "\n"
    if a.check:
        cur = dst.read_text(encoding="utf-8") if dst.exists() else ""
        if cur != md:
            print(f"STALE: {dst} does not match {src}. Run without --check.",
                  file=sys.stderr)
            return 1
        print(f"{dst} matches {src}")
        return 0

    dst.write_text(md, encoding="utf-8")
    print(f"wrote {dst} from {src}")

    # Write the derived notes back into the JSON too. The notes are a VIEW of the
    # measurements, not measurements themselves, and leaving a corrected sentence
    # in the markdown while the JSON beside it still carries the old one is the
    # exact drift this repo has been scored down for twice. Measured fields are
    # never touched.
    raw = json.loads(src.read_text(encoding="utf-8"))
    fresh = load(src).notes
    if raw.get("notes") != fresh:
        raw["notes"] = fresh
        src.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"refreshed the derived 'notes' in {src} (measurements untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
