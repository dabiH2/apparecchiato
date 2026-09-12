"""OpenVINO inference benchmark across Intel CPU / iGPU / NPU.

This is the artefact the Intel rubric asks for: run the pipeline's models on an
Intel Core Ultra Series 2/3 system and report latency, throughput, the device
actually selected, and the precision used.

Two things this script does that a bare `benchmark_app` invocation does not:

1. It reports the host it ran on, and says plainly whether that host is a Core
   Ultra Series 2/3 part. A benchmark that does not identify its own silicon is
   not evidence of anything, and a number from the wrong machine presented as if
   it came from the right one is worse than no number.
2. It sweeps every available device and precision rather than trusting AUTO, so
   the report shows the trade-off (NPU wins on power and steady-state latency,
   iGPU on burst throughput, CPU on first-token) instead of a single figure.
"""
from __future__ import annotations

import json
import os
import platform
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field, asdict

import numpy as np

# Core Ultra Series 2 = Lunar Lake / Arrow Lake; Series 3 = Panther Lake.
CORE_ULTRA_RE = re.compile(r"Core\(TM\)?\s*Ultra\s*(\d)?", re.I)
SERIES_2_3_RE = re.compile(r"Ultra\s*[579]\s*2\d{2}|Ultra\s*[579]\s*3\d{2}", re.I)


@dataclass
class DeviceResult:
    device: str
    precision: str
    ok: bool
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    throughput_fps: float = 0.0
    first_infer_ms: float = 0.0
    iterations: int = 0
    error: str = ""


@dataclass
class BenchReport:
    host: dict
    is_core_ultra: bool
    core_ultra_series_2_3: bool
    openvino_version: str
    available_devices: list
    results: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=float)

    def to_markdown(self) -> str:
        lines = ["# OpenVINO inference benchmark", "",
                 f"- Host CPU: `{self.host.get('cpu', 'unknown')}`",
                 f"- OS: `{self.host.get('os', 'unknown')}`",
                 f"- OpenVINO: `{self.openvino_version}`",
                 f"- Devices reported by OpenVINO: `{', '.join(self.available_devices)}`",
                 ""]
        if self.core_ultra_series_2_3:
            lines.append("**Target hardware confirmed: Intel Core Ultra Series 2/3.**")
        elif self.is_core_ultra:
            lines.append("> Intel Core Ultra detected, but not identified as Series 2/3. "
                         "Numbers below are indicative, not the target-platform result.")
        else:
            lines.append("> **Not an Intel Core Ultra Series 2/3 host.** "
                         "The OpenVINO path below is exercised on this machine's CPU; "
                         "the Core Ultra numbers must be produced on target hardware "
                         "with this same script. See docs/05-intel-benchmark-handoff.md.")
        lines += ["", "| Device | Precision | p50 latency (ms) | p95 (ms) | Throughput (inf/s) "
                  "| First infer (ms) | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in self.results:
            # Errors are worth reading, not truncating to a shrug: a failed
            # device/precision pair is a fact about the silicon and the only way
            # to tell "this NPU will not take bf16" from "the shapes were wrong".
            status = "ok" if r.ok else f"failed: {' '.join(r.error.split())[:160]}"
            lines.append(
                f"| {r.device} | {r.precision} | {r.latency_ms_p50:.2f} | {r.latency_ms_p95:.2f} "
                f"| {r.throughput_fps:.1f} | {r.first_infer_ms:.1f} | {status} |")
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


# --- host identification ---------------------------------------------------

def _cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            # PowerShell/CIM first: wmic is deprecated and absent on recent
            # Windows 11 builds, where it silently yields nothing and the CPU
            # ends up reported as "AMD64 Family 26 Model 36" -- useless for
            # deciding whether this is a Core Ultra Series 2/3 part.
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"],
                    capture_output=True, text=True, timeout=20).stdout.strip()
                if out:
                    return out.splitlines()[0].strip()
            except Exception:  # noqa: BLE001
                pass
            out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                                 text=True, timeout=10).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                          # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def describe_devices() -> tuple[str, list, dict]:
    """OpenVINO version, device list, and per-device full names."""
    import openvino as ov                                      # noqa: PLC0415
    core = ov.Core()
    devices = list(core.available_devices)
    names = {}
    for d in devices:
        try:
            names[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception as exc:                               # noqa: BLE001
            names[d] = f"<unavailable: {exc}>"
    return ov.__version__, devices, names


# --- benchmark -------------------------------------------------------------

def _dummy_inputs(compiled) -> dict:
    """Plausible input tensors of the right dtype for every port.

    Dtype matters, not just shape: this model's text ports are int64 token ids,
    and handing them float32 noise either throws or silently benchmarks a
    conversion that the real pipeline never performs. Integer ports get small
    non-negative values -- valid token ids and a mask of ones -- rather than
    arbitrary integers that could index out of an embedding table.
    """
    feed = {}
    for port in compiled.inputs:
        shape = [d.get_length() if d.is_static else 1 for d in port.partial_shape]
        dtype = np.dtype(port.element_type.to_dtype())
        if np.issubdtype(dtype, np.integer):
            feed[port] = np.ones(shape, dtype=dtype)
        elif dtype == np.bool_:
            feed[port] = np.ones(shape, dtype=np.bool_)
        else:
            feed[port] = np.random.rand(*shape).astype(dtype)
    return feed


def _bench_one(core, model, device: str, precision: str, iterations: int,
               warmup: int) -> DeviceResult:
    res = DeviceResult(device=device, precision=precision, ok=False)
    try:
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if precision != "native":
            cfg["INFERENCE_PRECISION_HINT"] = precision
        compiled = core.compile_model(model, device, cfg)
        req = compiled.create_infer_request()

        feed = _dummy_inputs(compiled)

        t0 = time.perf_counter()
        req.infer(feed)
        res.first_infer_ms = (time.perf_counter() - t0) * 1000.0
        for _ in range(warmup):
            req.infer(feed)

        times = []
        for _ in range(iterations):
            t = time.perf_counter()
            req.infer(feed)
            times.append((time.perf_counter() - t) * 1000.0)

        times.sort()
        res.latency_ms_p50 = statistics.median(times)
        res.latency_ms_p95 = times[max(int(0.95 * len(times)) - 1, 0)]
        res.throughput_fps = 1000.0 / res.latency_ms_p50 if res.latency_ms_p50 else 0.0
        res.iterations = iterations
        res.ok = True
    except Exception as exc:                                   # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
    return res


def benchmark(model_path: str, *, devices=None, precisions=("native", "f32", "f16", "bf16"),
              iterations: int = 60, warmup: int = 8, shapes: dict | None = None) -> BenchReport:
    """Sweep devices x precisions for one OpenVINO IR model.

    `shapes` pins the input shapes before compiling. This is not optional in
    practice: `torch.export` leaves the detector's ports fully dynamic ([?,?]),
    so a benchmark that guesses 1 for every unknown dimension feeds the model a
    1x1 image and measures nothing -- which is exactly what happened, as a wall
    of failed inferences. Pinning is also what deployment looks like, since the
    NPU plugin wants static shapes.
    """
    import openvino as ov                                      # noqa: PLC0415

    cpu = _cpu_name()
    version, available, names = describe_devices()
    report = BenchReport(
        host={"cpu": cpu, "os": f"{platform.system()} {platform.release()}",
              "python": platform.python_version(), "devices": names},
        is_core_ultra=bool(CORE_ULTRA_RE.search(cpu)),
        core_ultra_series_2_3=bool(SERIES_2_3_RE.search(cpu)),
        openvino_version=version,
        available_devices=available,
    )
    if not os.path.exists(model_path):
        report.notes.append(f"model not found: {model_path}")
        return report

    core = ov.Core()
    model = core.read_model(model_path)
    dynamic = [p.any_name for p in model.inputs if p.partial_shape.is_dynamic]
    if shapes:
        try:
            model.reshape({k: ov.PartialShape(list(v)) for k, v in shapes.items()})
            report.notes.append(
                "input shapes pinned to " +
                ", ".join(f"{k}={tuple(v)}" for k, v in shapes.items()))
        except Exception as exc:                               # noqa: BLE001
            report.notes.append(f"could not pin input shapes: {exc}")
    elif dynamic:
        report.notes.append(
            f"WARNING: dynamic input shapes {dynamic} and none were pinned -- "
            "every unknown dimension defaults to 1, so these numbers describe a "
            "degenerate input, not the real workload")
    targets = list(devices) if devices else [d for d in available if d != "AUTO"]
    if not targets:
        report.notes.append("OpenVINO reported no devices")
        return report

    for d in targets:
        for p in precisions:
            # NPU and GPU do not accept every precision hint; a failed combination
            # is recorded rather than hidden, so the table shows what the silicon
            # will and will not run.
            report.results.append(_bench_one(core, model, d, p, iterations, warmup))

    ok = [r for r in report.results if r.ok]
    if ok:
        best = min(ok, key=lambda r: r.latency_ms_p50)
        report.notes.append(
            f"lowest latency: {best.device} @ {best.precision} "
            f"= {best.latency_ms_p50:.2f} ms (p50)")
    if "NPU" not in available:
        report.notes.append(
            "No NPU device reported. On a Core Ultra this usually means the NPU "
            "driver is missing -- see Intel's Hack-a-thon Resources page.")
    return report
