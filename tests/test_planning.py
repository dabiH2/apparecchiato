"""Plans must be well formed, and plans that are physically impossible must fail."""
import numpy as np
import pytest

from apparecchiato.tasks import TaskGraph, Node
from apparecchiato.planner import build_planner
from apparecchiato.planner.base import PlannerError
from apparecchiato.planner.vlm import _extract_json_object, validate_against_scene
from apparecchiato.sim.layout import sample_scene

INSTRUCTION = "Set up the dinner table and pour water into the mug."


def test_cycles_are_rejected():
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph("c", [Node("a", "home", deps=["b"]), Node("b", "home", deps=["a"])])


def test_missing_skill_args_are_rejected():
    with pytest.raises(ValueError, match="missing args"):
        Node("x", "pick", {})


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="unknown node"):
        TaskGraph("d", [Node("a", "home", deps=["ghost"])])


@pytest.mark.parametrize("seed", range(8))
def test_rule_planner_produces_a_valid_plan(seed):
    spec = sample_scene(seed)
    g = build_planner("rules").plan(INSTRUCTION, spec)
    assert len(g) >= 8
    assert g.topological_order()
    # The drawer must be opened before any cutlery is picked.
    order = g.topological_order()
    opens = [n.id for n in g.nodes if n.skill == "open_drawer"]
    for n in g.nodes:
        if n.skill == "pick" and n.args["object"] in ("fork", "spoon"):
            assert order.index(n.id) > order.index(opens[0])


def test_planner_handles_italian():
    spec = sample_scene(3)
    g = build_planner("rules").plan("apparecchia la tavola e versa l'acqua", spec)
    assert any(n.skill == "pour" for n in g.nodes)


def test_unparseable_instruction_raises():
    with pytest.raises(PlannerError):
        build_planner("rules").plan("what is the weather like", sample_scene(0))


def test_json_extraction_survives_fences_and_braces_in_strings():
    assert _extract_json_object('```json\n{"nodes":[]}\n```') == '{"nodes":[]}'
    assert _extract_json_object('hi {"a":"}"} bye') == '{"a":"}"}'
    assert _extract_json_object("no json") is None


def test_vlm_plan_referencing_a_missing_object_is_rejected():
    spec = sample_scene(0)
    g = TaskGraph("x", [Node("n1", "pick", {"object": "teapot"})])
    with pytest.raises(PlannerError, match="not in the scene"):
        validate_against_scene(g, spec)


def test_vlm_plan_that_skips_opening_the_drawer_is_rejected():
    spec = sample_scene(0)
    g = TaskGraph("x", [Node("n1", "pick", {"object": "fork"})])
    with pytest.raises(PlannerError, match="drawer"):
        validate_against_scene(g, spec)


def test_vlm_plan_that_places_without_picking_is_rejected():
    spec = sample_scene(0)
    g = TaskGraph("x", [Node("n1", "place", {"object": "plate", "target": "plate"})])
    with pytest.raises(PlannerError, match="without ever picking"):
        validate_against_scene(g, spec)
