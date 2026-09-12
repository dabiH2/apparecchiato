#!/usr/bin/env python3
"""Is the OpenVINO toolchain actually here, and what can it run on?

Run this before any export. It answers, in order: which packages are installed,
which inference devices the runtime can see on THIS machine, and whether the
host CPU is the Intel Core Ultra Series 2/3 part the challenge asks for. The
last one is the point -- a benchmark number is only meaningful next to the
device it was produced on, and this prints that context whether it is flattering
or not.
"""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys

PACKAGES = ("openvino", "nncf", "optimum", "transformers", "torch",
            "openvino_genai", "openvino_tokenizers")


def cpu_name() -> str:
    """Host CPU model. PowerShell CIM, because wmic is gone on Windows 11."""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except Exception:                                          # noqa: BLE001
            pass
    return platform.processor() or "unknown"


def is_target_part(name: str) -> tuple[bool, str]:
    """Core Ultra Series 2/3 or not, and why.

    The rule that matters: the number after "Core Ultra 5/7/9" must start with 2
    or 3. A 155H (Series 1, Meteor Lake) and a 255H (Series 2, Arrow Lake) look
    almost identical and only one of them counts.
    """
    low = name.lower()
    if "core ultra" not in low:
        return False, "not an Intel Core Ultra part"
    import re
    m = re.search(r"core\s+ultra\s+\d\s+(\d)", low)
    if not m:
        return False, "could not read the series digit from the CPU name"
    if m.group(1) in ("2", "3"):
        return True, f"Core Ultra Series {m.group(1)} -- on target"
    return False, (f"Core Ultra Series {m.group(1)} (100-series is Series 1) -- "
                   "not the Series 2/3 part the challenge scores")


def main() -> int:
    print("packages")
    missing = []
    for m in PACKAGES:
        try:
            mod = importlib.import_module(m)
            print(f"  {m:22s} {getattr(mod, '__version__', 'installed')}")
        except Exception:                                          # noqa: BLE001
            print(f"  {m:22s} MISSING")
            missing.append(m)

    name = cpu_name()
    ok, why = is_target_part(name)
    print(f"\nhost CPU\n  {name}\n  {'TARGET' if ok else 'OFF TARGET'}: {why}")

    print("\ninference devices")
    try:
        import openvino as ov
        core = ov.Core()
        for d in core.available_devices:
            try:
                full = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:                                      # noqa: BLE001
                full = "?"
            print(f"  {d:8s} {full}")
        if not core.available_devices:
            print("  none")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  cannot query: {exc}")

    if missing:
        print("\ninstall the missing pieces with:\n"
              "  .\\.venv\\Scripts\\pip install 'optimum[openvino]' nncf "
              "openvino-genai", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
