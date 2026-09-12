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
# Prompt slot -> object name. The detector answers with an index, and the prompt
# WORDING is tunable at runtime while the slots are not, so this positional
# mapping is the stable one.
PROMPT_NAMES = tuple(PROMPT_TO_NAME[p] for p in PROMPTS)


@dataclass
class _DetectorOutput:
    """What the processor's post-processing expects to be handed.

    The IR returns bare tensors; `post_process_grounded_object_detection` wants
    something with `.logits` and `.pred_boxes`. This is that, and nothing else --
    no reason to pull in a HuggingFace output class for two fields.
    """
    logits: object
    pred_boxes: object


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
                 score_threshold: float = 0.20, precision_hint: str | None = None):
        self.model = model or os.environ.get("APPARECCHIATO_DET_MODEL",
                                             "models/owlvit-ov-int8")
        self.device = device or os.environ.get("APPARECCHIATO_DET_DEVICE", "AUTO")
        # The CPU plugin picks bf16 by default on hardware that supports it, which
        # is a real speedup and a real accuracy change. Leave it to the plugin
        # unless a caller pins it -- but record what was used, because a latency
        # number without its precision is not a measurement.
        self.precision_hint = precision_hint or os.environ.get(
            "APPARECCHIATO_DET_PRECISION", "")
        self.score_threshold = score_threshold
        self._model = None
        self._processor = None
        self._meta: dict = {}
        self._prompt_override: tuple | None = None
        self.last_latency_s: float | None = None

    def _load(self):
        if self._model is not None:
            return
        import json                                                    # noqa: PLC0415
        import openvino as ov                                          # noqa: PLC0415
        from transformers import AutoProcessor                         # noqa: PLC0415

        xml = os.path.join(self.model, "openvino_model.xml")
        if not os.path.isfile(xml):
            raise FileNotFoundError(
                f"OpenVINO IR not found at {xml!r}. "
                "Run: python scripts/export_detector_openvino.py"
            )
        meta_path = os.path.join(self.model, "apparecchiato_detector.json")
        self._meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        # The IR is loaded through the plain OpenVINO runtime, not optimum's
        # wrapper: optimum-intel has no OpenVINO export for zero-shot detection,
        # so this graph was built by hand (see scripts/export_detector_openvino.py)
        # and the prompt order it was traced with is recorded beside it.
        cfg = {"INFERENCE_PRECISION_HINT": self.precision_hint} if self.precision_hint else {}
        self._model = ov.Core().compile_model(xml, self.device, cfg)
        self._processor = AutoProcessor.from_pretrained(self.model)

    @property
    def prompts(self) -> tuple:
        """The prompts this IR was built for -- indices in its output refer to these."""
        return tuple(self._prompt_override or self._meta.get("prompts") or PROMPTS)

    def override_prompts(self, prompts) -> None:
        """Swap the text prompts without re-exporting.

        The tokens are a graph INPUT, not baked into the weights, so the same IR
        serves any prompt set of the same shape. That matters here: the objects
        are rendered primitives, and finding wording an open-vocabulary detector
        actually responds to is an experiment you want to run in seconds, not
        once per 157 MB export. The count must match what the graph was traced
        with, because the text batch dimension is static on purpose.
        """
        prompts = tuple(prompts)
        expected = len(self._meta.get("prompts") or PROMPTS)
        if len(prompts) != expected:
            raise ValueError(
                f"this IR was traced with {expected} prompts; got {len(prompts)}")
        self._prompt_override = prompts

    def detect(self, env, camera: str = "overhead") -> dict[str, Detection]:
        self._load()
        import torch                                                   # noqa: PLC0415
        from PIL import Image                                          # noqa: PLC0415

        frame = env.render(camera)
        img = Image.fromarray(frame)
        inputs = self._processor(text=[list(self.prompts)], images=img,
                                 return_tensors="pt")
        feed = {k: inputs[k].numpy()
                for k in self._meta.get("input_order",
                                        ("input_ids", "pixel_values", "attention_mask"))}
        t0 = time.perf_counter()
        raw = self._model(feed)
        self.last_latency_s = time.perf_counter() - t0

        out = _DetectorOutput(
            logits=torch.from_numpy(raw[self._model.output(0)]),
            pred_boxes=torch.from_numpy(raw[self._model.output(1)]),
        )
        res = self._processor.post_process_grounded_object_detection(
            outputs=out, target_sizes=torch.tensor([img.size[::-1]]),
            threshold=self.score_threshold)[0]

        best: dict[str, Detection] = {}
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            # By POSITION, not by prompt text. The detector returns an index into
            # whichever prompt list was fed in, and that list is swappable
            # (override_prompts) -- so slot i always means the same object
            # regardless of how slot i happens to be worded today.
            name = PROMPT_NAMES[int(label)]
            s = float(score)
            if name in best and best[name].score >= s:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            u, v = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            best[name] = Detection(name, backproject(env, camera, u, v, frame.shape),
                                   s, (x0, y0, x1, y1))
        return best


def backproject(env, camera: str, u: float, v: float, shape,
                plane_z: float = TABLE_Z) -> np.ndarray:
    """Pixel -> world point on a known horizontal plane, via the camera's pose.

    `plane_z` is not decoration. Intersecting with the TABLE is only right for
    something flat: an overhead camera sees the TOP of a 60 mm mug, and a ray
    through that pixel continued all the way down to z = 0 lands to one side by
    roughly (height x tan of the off-axis angle) -- 240 mm at the edge of this
    table, measured, even though the blob centroid was two pixels from truth.
    Every caller knows what it is looking at, so every caller knows which plane
    to use.
    """
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
    t = (plane_z - cam_pos[2]) / ray[2]
    hit = cam_pos + t * ray
    # Report the position the planner uses: the point on the TABLE under the
    # object, not the height the silhouette was seen at.
    hit[2] = TABLE_Z
    return hit


class ColourDetector(Detector):
    """Colour segmentation on the rendered frame. No privileged state.

    This exists because the open-vocabulary detector does not work on this scene,
    and that is worth stating plainly rather than hiding behind the oracle.
    OWL-ViT and OWLv2 are trained on natural photographs; the objects here are
    untextured coloured primitives, and measured over three seeds at every
    threshold, viewpoint and prompt wording tried, the neural detector's
    back-projected positions were 130-370 mm from truth -- i.e. no signal at all.
    Bigger renders, a finer patch size (patch32 -> patch16, 768 -> 960 px) and
    appearance-worded prompts all failed to change that.

    So the perception path that actually closes the loop is this one: each
    tableware type has a fixed colour, the way an industrial cell is calibrated
    to the parts it handles, and a centroid back-projected onto the known table
    plane gives the position. It reads PIXELS, not simulator state -- which is
    the property that matters, because it means the policy is not being handed
    the answer.

    Spoon and fork are deliberately the same colour, as real cutlery is. They are
    separated by x order, which is sound here: they come out of one drawer at
    known relative positions.
    """
    name = "colour"

    # Matched in HSV, not RGB. Scene lighting is randomised over a 2:1 intensity
    # range, which moves every rendered RGB value bodily -- a raw RGB match
    # against the material colour missed the bottle entirely on all five seeds.
    # Hue survives that; brightness does not. Chromatic objects are matched by
    # hue with a saturation floor, achromatic ones (white plate, steel cutlery)
    # by having almost no saturation, separated from each other and from the
    # dark arms by brightness.
    #
    # hue in turns (0..1): red 0, green 1/3, blue 2/3.
    HUES = {
        "mug": (0.60, 0.06),        # blue,  +/- tolerance
        "bottle": (0.38, 0.07),     # green
    }
    # Two thresholds, not one. A single MIN_SAT has to be low enough that a white
    # plate falls below it and high enough that a washed-out blue wall falls
    # below it too, and no value does both: at 0.18 the wall started matching the
    # mug's hue and the mug error went from 17 mm to 217 mm. So there is a floor
    # for "definitely coloured" and a separate, lower ceiling for "definitely
    # not", with a deliberate dead band between them that belongs to neither.
    CHROMA_MIN_SAT = 0.28
    ACHROMA_MAX_SAT = 0.16
    # Plate renders at 0.92 and steel at 0.80 before lighting, which spans
    # 0.65-1.35, so the dimmest tableware lands near 0.52. Everything structural
    # is darker than that by construction.
    MIN_VALUE = 0.45
    ROUND_ASPECT = 1.7              # bounding-box elongation: plate below, cutlery above

    # The height the blob centroid is actually SEEN at, per object, which is the
    # plane to back-project through. Looking down, a tall object shows mostly its
    # top face plus some near wall, so the silhouette centroid sits a little
    # below its top -- not at its mid-height, and certainly not on the table.
    # These come from OBJECT_TOP_H in sim.layout.
    SILHOUETTE_Z = {
        "plate": 0.016,
        "mug": 0.052,
        "bottle": 0.058,   # 68 mm carafe, same ~0.86 x TOP_H as the others
        "spoon": 0.016,
        "fork": 0.016,
    }

    def __init__(self, min_pixels: int = 12):
        self.min_pixels = min_pixels
        self.last_latency_s: float | None = None

    @staticmethod
    def _hsv(img):
        """RGB float image -> (hue in turns, saturation, value), vectorised."""
        mx = img.max(axis=2)
        mn = img.min(axis=2)
        c = mx - mn
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        h = np.zeros_like(mx)
        safe = c > 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = ((g - b) / c) % 6.0
            hg = (b - r) / c + 2.0
            hb = (r - g) / c + 4.0
        pick_r = safe & (mx == r)
        pick_g = safe & (mx == g) & ~pick_r
        pick_b = safe & (mx == b) & ~pick_r & ~pick_g
        h[pick_r] = hr[pick_r]
        h[pick_g] = hg[pick_g]
        h[pick_b] = hb[pick_b]
        h = (h / 6.0) % 1.0
        s = np.where(mx > 1e-6, c / np.where(mx > 1e-6, mx, 1.0), 0.0)
        return h, s, mx

    def masks(self, img) -> dict:
        """One boolean mask per colour class. Exposed so it can be inspected."""
        h, s, v = self._hsv(img)
        out = {}
        for kind, (hue, tol) in self.HUES.items():
            dh = np.abs(((h - hue + 0.5) % 1.0) - 0.5)      # wrap-around distance
            out[kind] = (dh < tol) & (s > self.CHROMA_MIN_SAT) & (v > 0.15)
        # Plate and cutlery are both achromatic, so they share one mask and are
        # told apart afterwards by RELATIVE brightness. Absolute thresholds do
        # not survive this scene: the light is randomised over a 2:1 range, so a
        # dimly-lit plate is darker than a brightly-lit fork. Under one exposure
        # the plate is always the brighter of the two, whatever the exposure is.
        out["achromatic"] = (s < self.ACHROMA_MAX_SAT) & (v > self.MIN_VALUE)
        return out

    def detect(self, env, camera: str = "overhead") -> dict[str, Detection]:
        frame = env.render(camera)
        t0 = time.perf_counter()
        img = np.asarray(frame, dtype=float) / 255.0
        value = img.max(axis=2)
        out: dict[str, Detection] = {}

        for kind, mask in self.masks(img).items():
            if kind == "achromatic":
                continue
            blob = _largest_blob(mask, self.min_pixels)
            if blob is None:
                continue
            out[kind] = Detection(
                kind, backproject(env, camera, blob["x"], blob["y"], img.shape,
                                  self.SILHOUETTE_Z[kind]),
                min(1.0, blob["n"] / 400.0))

        # Plate, spoon and fork are all the same colour, so which blob is which
        # has to come from arrangement rather than appearance. Shape alone does
        # not do it -- a spoon seen from above is dominated by its bowl and comes
        # out with an aspect ratio of 1.03, indistinguishable from a saucer --
        # and neither does size or brightness.
        #
        # What is reliable is that the two pieces of cutlery are a PAIR: they
        # come out of one drawer side by side and end up flanking the plate, so
        # they share a row and are near-identical in area, which is exactly what
        # the plate is not. So: find the best-matching pair, call it the cutlery,
        # and the plate is whatever is left.
        blobs = _largest_blobs(self.masks(img)["achromatic"], self.min_pixels,
                               k=12, value=value)
        pair, rest = _best_pair(blobs, img.shape[0])
        pair = sorted(pair, key=lambda b: b["x"])       # spoon sits left of fork
        for name, b in zip(("spoon", "fork"), pair):
            out[name] = Detection(
                name, backproject(env, camera, b["x"], b["y"], img.shape,
                                  self.SILHOUETTE_Z[name]),
                min(1.0, b["n"] / 200.0))
        if rest:
            plate = max(rest, key=lambda b: b["n"])
            out["plate"] = Detection(
                "plate", backproject(env, camera, plate["x"], plate["y"], img.shape,
                                     self.SILHOUETTE_Z["plate"]),
                min(1.0, plate["n"] / 400.0))
        self.last_latency_s = time.perf_counter() - t0
        return out


def _largest_blobs(mask, min_pixels: int, k: int = 1, value=None) -> list:
    """The k biggest connected components of a boolean mask.

    Each is {"y", "x", "n", "value"} -- centroid, pixel count, and mean
    brightness when a value image is supplied, which is what separates the plate
    from the cutlery.

    Flood fill rather than scipy.ndimage: one fewer dependency, and the masks
    here are a handful of small blobs in a mostly-empty image.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    h, w = mask.shape
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pix) >= min_pixels:
            arr = np.asarray(pix, dtype=float)
            idx = (arr[:, 0].astype(int), arr[:, 1].astype(int))
            hh = arr[:, 0].max() - arr[:, 0].min() + 1.0
            ww = arr[:, 1].max() - arr[:, 1].min() + 1.0
            found.append({
                "y": float(arr[:, 0].mean()),
                "x": float(arr[:, 1].mean()),
                "n": len(pix),
                "value": float(value[idx].mean()) if value is not None else 0.0,
                # Elongation of the bounding box. This is what tells a round
                # plate from a long fork, and unlike brightness it does not
                # depend on the lighting: measured, the plate and the fork came
                # out at 0.99 and 0.98 mean value, which is no separation at
                # all, while their aspect ratios are about 1.1 and 3.5.
                "aspect": float(max(hh, ww) / max(1.0, min(hh, ww))),
            })
    found.sort(key=lambda b: -b["n"])
    return found[:k]


def _largest_blob(mask, min_pixels: int):
    got = _largest_blobs(mask, min_pixels, k=1)
    return got[0] if got else None


def _best_pair(blobs: list, image_h: int) -> tuple:
    """Split blobs into the most sibling-like pair and everything else.

    Siblings share a row and have nearly the same area. Both terms are needed:
    once the setting is laid out the plate sits between its own cutlery at the
    same y, so row alone would happily pair the plate with a fork -- but a
    26 mm saucer and a fork are not the same size, and the two pieces of cutlery
    are. Distances are normalised so the two terms are comparable.
    """
    if len(blobs) < 2:
        return [], blobs
    best, score = None, float("inf")
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            a, b = blobs[i], blobs[j]
            dy = abs(a["y"] - b["y"]) / max(1.0, image_h)
            dn = abs(a["n"] - b["n"]) / max(a["n"], b["n"])
            s = dy + 0.5 * dn
            if s < score:
                best, score = (i, j), s
    i, j = best
    return ([blobs[i], blobs[j]],
            [b for k, b in enumerate(blobs) if k not in (i, j)])


def build_detector(kind: str = "oracle", **kw) -> Detector:
    kind = kind.lower()
    if kind == "oracle":
        return OracleDetector()
    if kind == "openvino":
        return OpenVINODetector(**kw)
    if kind in ("colour", "color"):
        return ColourDetector(**kw)
    raise ValueError(
        f"unknown detector {kind!r}; use 'oracle', 'colour' or 'openvino'")
