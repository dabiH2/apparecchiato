"""Scene grounding: camera frames -> object positions.

Two detectors behind one interface.

  oracle    reads positions straight out of the simulator. Useful for isolating
            manipulation failures from perception failures while developing, and
            as the control condition in the ablation table.
  openvino  runs an open-vocabulary detector compiled to OpenVINO IR on the
            rendered overhead view, then back-projects each box centre onto the
            table plane. This is the path that runs on Intel CPU / iGPU / NPU and
            the one the benchmark measures.

Reporting both is the honest way to present the numbers: it separates "the
policy cannot do the task" from "the detector missed the fork".
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

TABLE_Z = 0.0
PROMPTS = ("a plate", "a mug", "a bottle", "a spoon", "a fork", "a drawer knob")
PROMPT_TO_NAME = {"a plate": "plate", "a mug": "mug", "a bottle": "bottle",
                  "a spoon": "spoon", "a fork": "fork", "a drawer knob": "knob"}


@dataclass
class Detection:
    name: str
    pos: np.ndarray          # world position on the table plane
    score: float
    bbox: tuple | None = None


class Detector(ABC):
    name = "detector"

    @abstractmethod
    def detect(self, env, camera: str = "overhead") -> dict[str, Detection]:
        ...


class OracleDetector(Detector):
    """Privileged state. No vision, no error -- the control condition."""
    name = "oracle"

    def detect(self, env, camera: str = "overhead") -> dict[str, Detection]:
        return {o.name: Detection(o.name, env.object_pos(o.name), 1.0)
                for o in env.spec.objects}


class OpenVINODetector(Detector):
    """Open-vocabulary detection on the rendered view, running on Intel silicon.

    Back-projection assumes objects rest on the known table plane, which is true
    here and removes the need for a depth sensor: a ray is cast through each box
    centre and intersected with z = TABLE_Z.
    """
    name = "openvino"

    def __init__(self, model: str | None = None, device: str | None = None,
                 score_threshold: float = 0.20):
        self.model = model or os.environ.get("APPARECCHIATO_DET_MODEL", "models/owlv2-base-ov-int8")
        self.device = device or os.environ.get("APPARECCHIATO_DET_DEVICE", "AUTO")
        self.score_threshold = score_threshold
        self._model = None
        self._processor = None
        self.last_latency_s: float | None = None

    def _load(self):
        if self._model is not None:
            return
        from optimum.intel import OVModelForZeroShotObjectDetection   # noqa: PLC0415
        from transformers import AutoProcessor                        # noqa: PLC0415
        if not os.path.isdir(self.model):
            raise FileNotFoundError(
                f"OpenVINO IR not found at {self.model!r}. "
                "Run: python scripts/export_detector_openvino.py"
            )
        self._model = OVModelForZeroShotObjectDetection.from_pretrained(
            self.model, device=self.device)
        self._processor = AutoProcessor.from_pretrained(self.model)

    def detect(self, env, camera: str = "overhead") -> dict[str, Detection]:
        self._load()
        import torch                                                   # noqa: PLC0415
        from PIL import Image                                          # noqa: PLC0415

        frame = env.render(camera)
        img = Image.fromarray(frame)
        inputs = self._processor(text=[list(PROMPTS)], images=img, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model(**inputs)
        self.last_latency_s = time.perf_counter() - t0

        res = self._processor.post_process_grounded_object_detection(
            outputs=out, target_sizes=torch.tensor([img.size[::-1]]),
            threshold=self.score_threshold)[0]

        best: dict[str, Detection] = {}
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            prompt = PROMPTS[int(label)]
            name = PROMPT_TO_NAME[prompt]
            s = float(score)
            if name in best and best[name].score >= s:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            best[name] = Detection(name, backproject(env, camera, u, v, frame.shape),
                                   s, (x0, y0, x1, y1))
        return best


def backproject(env, camera: str, u: float, v: float, shape) -> np.ndarray:
    """Pixel -> world point on the table plane, using the camera's own pose."""
    import mujoco                                                      # noqa: PLC0415

    h, w = shape[0], shape[1]
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise KeyError(f"camera {camera!r} not in model")
    cam_pos = np.array(env.data.cam_xpos[cid], dtype=float)
    R = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)

    fovy = float(env.model.cam_fovy[cid]) * np.pi / 180.0
    f = (h / 2.0) / np.tan(fovy / 2.0)
    x = (u - w / 2.0) / f
    y = -(v - h / 2.0) / f
    # MuJoCo cameras look down their local -z.
    ray = R @ np.array([x, y, -1.0])
    n = np.linalg.norm(ray)
    ray = ray / (n if n else 1.0)
    if abs(ray[2]) < 1e-9:
        return cam_pos
    t = (TABLE_Z - cam_pos[2]) / ray[2]
    return cam_pos + t * ray


def build_detector(kind: str = "oracle", **kw) -> Detector:
    kind = kind.lower()
    if kind == "oracle":
        return OracleDetector()
    if kind == "openvino":
        return OpenVINODetector(**kw)
    raise ValueError(f"unknown detector {kind!r}; use 'oracle' or 'openvino'")
