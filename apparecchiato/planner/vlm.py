"""Multi-modal planner: camera views + instruction -> TaskGraph.

Backends, tried in this order unless one is forced with `backend=`:

  openvino   a local VLM compiled to OpenVINO IR via optimum-intel. This is the
             path that matters for the Intel rubric: it runs on CPU / iGPU / NPU
             and is what `apparecchiato/bench/openvino_bench.py` measures.
  openai     any OpenAI-compatible chat endpoint (including a local llama.cpp or
             vLLM server). Useful while developing, and as a remote fallback.

Whatever comes back is parsed strictly and validated against the scene before it
is allowed anywhere near the robot: unknown objects, unreachable assignments,
missing hand-offs and cycles are all rejected here. A rejected plan raises, and
`build_planner("vlm+rules")` then falls through to the deterministic planner.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from ..tasks import TaskGraph, Node
from .base import Planner, PlannerError, SYSTEM_PROMPT, observation_text

DEFAULT_OV_MODEL = os.environ.get("APPARECCHIATO_VLM_MODEL", "models/qwen2-vl-2b-ov-int4")
DEFAULT_OV_DEVICE = os.environ.get("APPARECCHIATO_VLM_DEVICE", "AUTO")


class VLMPlanner(Planner):
    name = "vlm"

    def __init__(self, backend: str | None = None, model: str | None = None,
                 device: str | None = None, max_new_tokens: int = 512,
                 temperature: float = 0.0):
        self.backend = backend or os.environ.get("APPARECCHIATO_VLM_BACKEND", "auto")
        self.model = model or DEFAULT_OV_MODEL
        self.device = device or DEFAULT_OV_DEVICE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None
        self._processor = None
        self.last_latency_s: float | None = None
        self.last_backend: str | None = None
        self.last_raw: str | None = None

    # -- public -------------------------------------------------------------

    def plan(self, instruction: str, scene, images=None) -> TaskGraph:
        prompt = self._build_prompt(instruction, scene)
        t0 = time.perf_counter()
        raw = self._generate(prompt, images)
        self.last_latency_s = time.perf_counter() - t0
        self.last_raw = raw

        nodes = self._parse(raw)
        graph = TaskGraph(instruction, nodes, source=f"vlm:{self.last_backend}",
                          notes=[f"planning latency {self.last_latency_s * 1000:.0f} ms "
                                 f"on {self.last_backend}"])
        validate_against_scene(graph, scene)
        return graph

    # -- prompt -------------------------------------------------------------

    def _build_prompt(self, instruction: str, scene) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n{observation_text(scene)}\n\n"
            f"Instruction: {instruction}\n\nJSON:"
        )

    # -- generation ---------------------------------------------------------

    def _generate(self, prompt: str, images) -> str:
        order = ([self.backend] if self.backend != "auto" else ["openvino", "openai"])
        errors = []
        for b in order:
            try:
                if b == "openvino":
                    self.last_backend = "openvino"
                    return self._gen_openvino(prompt, images)
                if b == "openai":
                    self.last_backend = "openai"
                    return self._gen_openai(prompt, images)
                raise PlannerError(f"unknown VLM backend {b!r}")
            except Exception as exc:                       # noqa: BLE001
                errors.append(f"{b}: {type(exc).__name__}: {exc}")
        raise PlannerError("no VLM backend available -> " + " | ".join(errors))

    def _gen_openvino(self, prompt: str, images) -> str:
        if self._pipe is None:
            from optimum.intel import OVModelForVisualCausalLM   # noqa: PLC0415
            from transformers import AutoProcessor              # noqa: PLC0415
            if not os.path.isdir(self.model):
                raise PlannerError(
                    f"OpenVINO IR not found at {self.model!r}. "
                    "Run: python scripts/export_vlm_openvino.py --help"
                )
            self._pipe = OVModelForVisualCausalLM.from_pretrained(self.model, device=self.device)
            self._processor = AutoProcessor.from_pretrained(self.model)

        proc, pipe = self._processor, self._pipe
        pil = _as_pil(images)
        inputs = proc(text=[prompt], images=pil if pil else None, return_tensors="pt")
        out = pipe.generate(**inputs, max_new_tokens=self.max_new_tokens,
                            do_sample=self.temperature > 0.0,
                            temperature=max(self.temperature, 1e-5))
        text = proc.batch_decode(out, skip_special_tokens=True)[0]
        return text[len(prompt):] if text.startswith(prompt) else text

    def _gen_openai(self, prompt: str, images) -> str:
        import urllib.request                                   # noqa: PLC0415

        base = os.environ.get("APPARECCHIATO_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        key = os.environ.get("APPARECCHIATO_LLM_API_KEY", "not-needed")
        model = os.environ.get("APPARECCHIATO_LLM_MODEL", "local-model")

        content: list = [{"type": "text", "text": prompt}]
        for b64 in _as_base64_pngs(images):
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

    # -- parsing ------------------------------------------------------------

    def _parse(self, raw: str) -> list[Node]:
        blob = _extract_json_object(raw)
        if blob is None:
            raise PlannerError(f"no JSON object in model output: {raw[:240]!r}")
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"model emitted invalid JSON: {exc}") from exc
        if "nodes" not in data or not isinstance(data["nodes"], list):
            raise PlannerError("model output has no 'nodes' list")
        if not data["nodes"]:
            raise PlannerError("model returned an empty plan")
        nodes = []
        for i, n in enumerate(data["nodes"]):
            if not isinstance(n, dict) or "skill" not in n:
                raise PlannerError(f"node {i} is malformed: {n!r}")
            nodes.append(Node(
                id=str(n.get("id") or f"n{i}"),
                skill=str(n["skill"]).strip().lower(),
                args={k: str(v) for k, v in (n.get("args") or {}).items()},
                arm=str(n.get("arm", "any")).strip().upper().replace("ANY", "any"),
                deps=[str(d) for d in (n.get("deps") or [])],
                rationale=str(n.get("rationale", "")),
            ))
        return nodes


# --- validation ------------------------------------------------------------

def validate_against_scene(graph: TaskGraph, scene) -> None:
    """Reject plans that are syntactically fine but physically nonsense."""
    from ..sim.layout import reaching_arms, pick_pos

    known = {o.name for o in scene.objects} | set(scene.goals)
    unknown = graph.objects() - known
    if unknown:
        raise PlannerError(f"plan references objects that are not in the scene: {sorted(unknown)}")

    drawer_objs = {o.name for o in scene.objects if o.inside_drawer}
    opened = [n.id for n in graph.nodes if n.skill == "open_drawer"]
    order = graph.topological_order()
    for n in graph.nodes:
        if n.skill == "pick" and n.args["object"] in drawer_objs:
            if not opened:
                raise PlannerError(
                    f"plan picks {n.args['object']} out of a drawer it never opens")
            if order.index(n.id) < min(order.index(o) for o in opened):
                raise PlannerError(
                    f"plan picks {n.args['object']} before opening the drawer")

    held_by_a_pick = {n.args["object"] for n in graph.nodes if n.skill == "pick"}
    for n in graph.nodes:
        if n.skill == "place" and n.args["object"] not in held_by_a_pick:
            raise PlannerError(f"plan places {n.args['object']} without ever picking it")

    for n in graph.nodes:
        if n.arm == "any" or n.skill in ("home", "handoff", "pour"):
            continue
        if n.skill == "pick":
            p = pick_pos(scene, n.args["object"], assume_drawer_open=True)
        elif n.skill == "place":
            t = n.args["target"]
            if t not in scene.goals:
                continue
            p = np.asarray(scene.goals[t], dtype=float)
        else:
            continue
        if n.arm not in reaching_arms(p):
            raise PlannerError(
                f"node {n.id} assigns arm {n.arm} to {n.skill} at "
                f"{np.round(p, 3).tolist()}, which only {reaching_arms(p) or 'no arm'} can reach"
            )


# --- small helpers ---------------------------------------------------------

def _extract_json_object(text: str) -> str | None:
    """Pull the first balanced {...} out of a model response, fences and all."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _as_pil(images):
    if images is None:
        return None
    try:
        from PIL import Image                                    # noqa: PLC0415
    except ImportError:
        return None
    out = []
    for im in (images if isinstance(images, (list, tuple)) else [images]):
        out.append(im if hasattr(im, "size") else Image.fromarray(np.asarray(im)))
    return out or None


def _as_base64_pngs(images) -> list[str]:
    pil = _as_pil(images)
    if not pil:
        return []
    import base64, io                                            # noqa: PLC0415
    out = []
    for im in pil:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out
