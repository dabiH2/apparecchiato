#!/usr/bin/env python3
"""Check that this machine can run the project, and say exactly what is missing."""
from __future__ import annotations

import importlib
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = [
    ("numpy", "required", "pip install numpy"),
    ("mujoco", "required", "pip install mujoco"),
    ("imageio", "video", "pip install imageio imageio-ffmpeg"),
    ("openvino", "Intel track", "pip install openvino"),
    ("optimum.intel", "VLM/detector on OpenVINO", "pip install optimum[openvino]"),
    ("transformers", "VLM/detector", "pip install transformers"),
    ("torch", "VLM/detector", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
    ("websockets", "Speechmatics voice", "pip install websockets"),
    ("sounddevice", "microphone input", "pip install sounddevice"),
    ("PIL", "image handling", "pip install pillow"),
]


def main() -> int:
    print(f"python {platform.python_version()} on {platform.system()} {platform.release()}")
    from apparecchiato.bench.openvino_bench import _cpu_name, CORE_ULTRA_RE, SERIES_2_3_RE
    cpu = _cpu_name()
    print(f"cpu: {cpu}")
    if SERIES_2_3_RE.search(cpu):
        print("  -> Intel Core Ultra Series 2/3: this is the challenge target platform.")
    elif CORE_ULTRA_RE.search(cpu):
        print("  -> Intel Core Ultra, but not Series 2/3.")
    else:
        print("  -> not an Intel Core Ultra. Everything runs, but the Core Ultra "
              "benchmark must be produced on target hardware "
              "(see docs/05-intel-benchmark-handoff.md).")

    missing_required = []
    print("\npackages:")
    for mod, why, how in CHECKS:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "")
            print(f"  ok      {mod:16s} {v}")
        except Exception:                                      # noqa: BLE001
            print(f"  MISSING {mod:16s} ({why})  ->  {how}")
            if why == "required":
                missing_required.append(mod)

    print("\ncore pipeline (no simulator needed):")
    try:
        from apparecchiato.sim.layout import sample_scene
        from apparecchiato.planner import build_planner
        from apparecchiato.scheduler import schedule, parallel_fraction
        from apparecchiato.skills import build_motion
        spec = sample_scene(0)
        g = build_planner("rules").plan("set the table and pour water into the mug", spec)
        steps = schedule(g, spec)
        n = sum(len(build_motion(s, sample_scene(0))) for s in steps[:2])
        print(f"  ok      plan={len(g)} nodes, schedule={len(steps)} steps, "
              f"{parallel_fraction(steps):.0%} parallelisable, motions build ({n}+ waypoints)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        return 1

    try:
        import openvino as ov
        core = ov.Core()
        print(f"\nopenvino {ov.__version__} devices: {', '.join(core.available_devices)}")
        if "NPU" not in core.available_devices:
            print("  note: no NPU listed. On a Core Ultra, install the NPU driver.")
    except Exception:                                          # noqa: BLE001
        pass

    if missing_required:
        print(f"\nInstall the required packages first: {', '.join(missing_required)}")
        return 1
    print("\nready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
