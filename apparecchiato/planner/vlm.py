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
                 temperature: float = 0.0, max_attempts: int = 2):
        self.backend = backend or os.environ.get("APPARECCHIATO_VLM_BACKEND", "auto")
        self.model = model or DEFAULT_OV_MODEL
        self.device = device or DEFAULT_OV_DEVICE
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_attempts = max(1, int(max_attempts))
        self._pipe = None
        self._processor = None
        self.last_latency_s: float | None = None
        self.last_backend: str | None = None
        self.last_raw: str | None = None
        self.last_repaired: bool = False
        self.last_dropped: int = 0

    # -- public -------------------------------------------------------------

    def plan(self, instruction: str, scene, images=None) -> TaskGraph:
        """Propose, check, and give the model one chance to fix its own mistake.

        The retry is the cheap half of this architecture. A small model's plans
        fail for reasons that are precisely stateable -- "you told arm A to pick
        the fork while it is still holding the plate" -- and handing that
        sentence back is far more effective than raising the token budget or the
        weight precision. It is also bounded: one retry, then the caller falls
        back to the deterministic planner, so a confused model can never spin.
        """
        prompt = self._build_prompt(instruction, scene)
        attempts = []
        for attempt in range(self.max_attempts):
            t0 = time.perf_counter()
            raw = self._generate(prompt, images)
            self.last_latency_s = time.perf_counter() - t0
            self.last_raw = raw
            try:
                return self._finish(raw, instruction, scene, attempt)
            except Exception as exc:                          # noqa: BLE001
                attempts.append(f"attempt {attempt + 1}: {exc}")
                if attempt + 1 >= self.max_attempts:
                    raise PlannerError(" | ".join(attempts)) from exc
                prompt = (f"{self._build_prompt(instruction, scene)}\n\n"
                          f"Your previous answer was rejected: {exc}\n"
                          "Fix exactly that and reply with the corrected JSON only.")
        raise PlannerError(" | ".join(attempts))

    def _finish(self, raw: str, instruction: str, scene, attempt: int) -> TaskGraph:
        nodes = self._parse(raw)
        notes = [f"planning latency {self.last_latency_s * 1000:.0f} ms "
                 f"on {self.last_backend}"]
        if self.last_repaired:
            # Recorded, not hidden: a plan that needed syntax repair is a
            # different quality claim from one that parsed as emitted, and the
            # benchmark reports the two separately.
            notes.append("model JSON needed syntax repair before parsing")
        if self.last_dropped:
            notes.append(f"{self.last_dropped} node(s) were unparseable and dropped")
        if attempt:
            notes.append(f"accepted on attempt {attempt + 1} after feedback")
        graph = TaskGraph(instruction, nodes, source=f"vlm:{self.last_backend}",
                          notes=notes)
        validate_against_scene(graph, scene)
        # Schedule it here too. A plan can be perfectly well-formed and still
        # physically impossible -- "pick the fork while still holding the
        # plate" -- and the only thing that knows is the scheduler. Checking now
        # means the model gets told, rather than the episode failing later.
        from ..scheduler import schedule                      # noqa: PLC0415
        schedule(graph, scene)
        return graph

    # -- prompt -------------------------------------------------------------

    def _build_prompt(self, instruction: str, scene) -> str:
        """The USER turn only. SYSTEM_PROMPT is sent as a system message by each
        backend, so it is not repeated here -- an instruct model given the same
        rules twice tends to answer twice."""
        return (f"{observation_text(scene)}\n\nInstruction: {instruction}\n\n"
                "Reply with the JSON object and nothing else.")

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

        # Go through the model's own chat template. Feeding an instruct-tuned
        # VLM a bare string is the difference between a plan and a monologue:
        # without the template there is no assistant turn to end, so the model
        # opened its JSON correctly and then repeated "the fork is in the
        # drawer" until it hit the token limit. The template also supplies the
        # image placeholder in the position the vision tower expects.
        content = ([{"type": "image"}] if pil else []) + [{"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        try:
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        except Exception:                                    # noqa: BLE001
            text = prompt                                    # processor without a template
        inputs = proc(text=[text], images=pil if pil else None, return_tensors="pt")

        out = pipe.generate(**inputs, max_new_tokens=self.max_new_tokens,
                            do_sample=self.temperature > 0.0,
                            temperature=max(self.temperature, 1e-5),
                            # A small repetition penalty is cheap insurance on a
                            # 2B model quantised to 4 bits: degenerate loops are
                            # the characteristic failure, and one loop costs the
                            # whole plan.
                            repetition_penalty=1.05)
        # Decode ONLY the generated continuation. Slicing the prompt off the
        # decoded string instead is unreliable -- the template rewrites it.
        n_in = inputs["input_ids"].shape[1]
        return proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]

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
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
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
        self.last_repaired = False
        self.last_dropped = 0
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            # A 2B model quantised for an NPU emits structurally-correct plans
            # with damaged punctuation -- '"skill":": "pick"', a stray quote
            # before a key, a missing comma between objects. Repairing that is
            # not the same as inventing a plan: the repair only touches syntax,
            # and everything downstream (schema check, then
            # validate_against_scene) still has to pass. Refusing a good plan
            # over a doubled colon would just hand the episode to the rules
            # planner for no reason.
            fixed = _repair_json(blob)
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError:
                # One irreparable node should not cost the whole plan. Pull the
                # node objects out individually and keep the ones that parse:
                # the model's failures are local (a missing value in the last
                # node), so the earlier nodes are still its real intent. What is
                # dropped is counted and reported, and the surviving plan still
                # has to satisfy validate_against_scene and the scheduler's
                # preconditions -- if too much was lost, it fails there and the
                # deterministic planner takes over.
                nodes_raw, dropped = _salvage_nodes(fixed)
                if not nodes_raw:
                    raise PlannerError(
                        f"model emitted invalid JSON and no node survived repair")
                data = {"nodes": nodes_raw}
                self.last_dropped = dropped
            self.last_repaired = True
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
                args={k.strip(): _clean_value(v)
                      for k, v in (n.get("args") or {}).items()},
                arm=_clean_arm(n.get("arm", "any")),
                deps=[str(d).strip() for d in (n.get("deps") or [])],
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

def _salvage_nodes(blob: str) -> tuple[list, int]:
    """Parse the node objects that CAN be parsed; count the ones that cannot.

    Scans for balanced {...} at the node nesting level and tries each one on its
    own. Nothing is guessed: a node that will not parse is discarded, never
    patched up with defaults.
    """
    out, dropped = [], 0
    depth, start = 0, None
    in_str, esc = False, False
    for i, c in enumerate(blob):
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            # Nodes sit one level inside the {"nodes": [...]} wrapper, so a node
            # opens at depth 1 and closes back to it. Scanning at depth 0 would
            # only ever find the wrapper, which is the thing that failed to parse.
            if depth == 1:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 1 and start is not None:
                piece = blob[start:i + 1]
                try:
                    obj = json.loads(piece)
                except json.JSONDecodeError:
                    dropped += 1
                else:
                    if isinstance(obj, dict) and "skill" in obj:
                        out.append(obj)
                    else:
                        dropped += 1
                start = None
            elif depth < 0:
                depth = 0
    return out, dropped


def _clean_value(v) -> str:
    """Normalise an arg value to the bare name the rest of the system uses.

    The model likes to answer a slot as "slot 'mug'" or "the mug slot" rather
    than "mug". That is a naming convention it was never given, not a wrong
    plan, so it is normalised here -- and anything that is still not a real
    object or slot afterwards is caught by validate_against_scene.
    """
    s = str(v).strip().strip("'\"").strip()
    m = re.fullmatch(r"(?:the\s+)?slot\s*['\"]?([\w\-]+)['\"]?", s, flags=re.I)
    if m:
        return m.group(1).lower()
    m = re.fullmatch(r"(?:the\s+)?([\w\-]+)\s+slot", s, flags=re.I)
    if m:
        return m.group(1).lower()
    return s


def _clean_arm(v) -> str:
    """'A', 'B', or 'any'. The model writes 'A+B' when it means 'either'."""
    s = str(v).strip().upper()
    if s in ("A", "B"):
        return s
    return "any"


def _repair_json(blob: str) -> str:
    """Fix the punctuation damage a small quantised VLM produces. Syntax only.

    Every rule here corresponds to an artefact actually observed in this
    model's output, not to a general-purpose "fix any JSON" ambition -- a
    lenient repairer that guesses at structure would let a wrong plan through
    looking right, which is the one outcome worth avoiding. Keys, values and
    nesting are never invented; only stray separators are removed.
    """
    s = blob
    # '"skill":": "pick"'  ->  '"skill": "pick"'      (doubled colon)
    s = re.sub(r':\s*":\s*"', ': "', s)
    # '"deps":": [' -> '"deps": ['
    s = re.sub(r':\s*":\s*\[', ': [', s)
    # ', " "arm": "A"'  ->  ', "arm": "A"'            (stray quote before a key)
    s = re.sub(r'([,{]\s*)"\s+"', r'\1"', s)
    # '"skill": " "pick"'  ->  '"skill": "pick"'      (stray quote before a value)
    s = re.sub(r':\s*"\s+"([^"]*)"', r': "\1"', s)
    # '"pick",",'  ->  '"pick",'                      (stray quoted comma)
    s = re.sub(r',\s*",\s*', ', ', s)
    # '}  {'  ->  '}, {'                              (missing comma between objects)
    s = re.sub(r'}\s*{', '}, {', s)
    # trailing commas before a close
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


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
