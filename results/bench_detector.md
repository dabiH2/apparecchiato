# OpenVINO inference benchmark

- Host CPU: `AMD Ryzen AI 9 HX 370 w/ Radeon 890M`
- OS: `Windows 11`
- OpenVINO: `2026.3.1-22476-759c5a6ab8c-releases/2026/3`
- Devices reported by OpenVINO: `CPU, GPU`

> **Not an Intel Core Ultra Series 2/3 host.** The OpenVINO path below is exercised on this machine's CPU; the Core Ultra numbers must be produced on target hardware with this same script. See docs/05-intel-benchmark-handoff.md.

| Device | Precision | p50 latency (ms) | p95 (ms) | Throughput (inf/s) | First infer (ms) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | native | 64.33 | 66.76 | 15.5 | 74.6 | ok |
| CPU | f32 | 78.75 | 100.90 | 12.7 | 82.9 | ok |
| CPU | f16 | 69.32 | 77.37 | 14.4 | 64.9 | ok |
| CPU | bf16 | 82.14 | 88.63 | 12.2 | 66.9 | ok |
| GPU | native | 722.24 | 724.90 | 1.4 | 848.1 | ok |
| GPU | f32 | 722.78 | 731.74 | 1.4 | 724.6 | ok |
| GPU | f16 | 723.97 | 726.65 | 1.4 | 725.1 | ok |
| GPU | bf16 | 0.00 | 0.00 | 0.0 | 0.0 | failed: RuntimeError: Exception from src\inference\src\cpp\core.cpp:120: Exception from src\inference\src\dev\plugin.cpp:54: Exception from src\inference\src\dev\plugin |

## Notes

- input shapes pinned to input_ids=(6, 16), attention_mask=(6, 16), pixel_values=(1, 3, 768, 768)
- lowest latency: CPU @ native = 64.33 ms (p50)
- No NPU device reported. On a Core Ultra this usually means the NPU driver is missing -- see Intel's Hack-a-thon Resources page.