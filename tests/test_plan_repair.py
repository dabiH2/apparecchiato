"""The repair layer may delete and reorder the model's plan. It may never add to it.

That sentence is the whole claim the repair makes, so it is tested rather than
documented. If these pass, "the executed plan is the model's, minus what it
mis-wrote" is true by construction; if the invention test ever fails, the honest
description of the system has changed and the copy has to change with it.
"""
from __future__ import annotations

import pytest

from apparecchiato.planner.repair import (assert_nothing_invented,
                                          assert_repair_was_a_minority, break_cycles,
                                          prune_dangling_deps, sanitise_nodes)
from apparecchiato.tasks import Node, TaskGraph


def _nodes(*specs):
    return [Node(id=i, skill=s, args=a, deps=list(d)) for i, s, a, d in specs]


# --- sanitise_nodes --------------------------------------------------------

def test_drops_a_node_that_names_no_skill():
    kept, notes = sanitise_nodes([
        {"id": "n0", "skill": "open_drawer", "args": {}},
        {"id": "n1", "skill": "", "args": {}},
    ])
    assert [k["id"] for k in kept] == ["n0"]
    assert notes and "no known skill" in notes[0]


def test_drops_a_node_missing_a_required_argument():
    kept, notes = sanitise_nodes([
        {"id": "n0", "skill": "pick", "args": {"object": "plate"}},
        {"id": "n1", "skill": "place", "args": {"object": "plate"}},   # no target
    ])
    assert [k["id"] for k in kept] == ["n0"]
    assert any("missing a required argument" in n for n in notes)


def test_keeps_every_well_formed_node_and_says_nothing():
    raw = [{"id": "n0", "skill": "pick", "args": {"object": "mug"}},
           {"id": "n1", "skill": "place", "args": {"object": "mug", "target": "mug"}}]
    kept, notes = sanitise_nodes(raw)
    assert kept == raw and notes == []


# --- prune_dangling_deps ---------------------------------------------------

def test_prunes_a_dependency_on_a_node_that_was_never_emitted():
    """This is the exact failure the committed run records: `node n3 depends on
    unknown node(s) ['n2']`. Before the repair it cost the whole episode's plan."""
    nodes = _nodes(("n0", "open_drawer", {}, []),
                   ("n3", "pick", {"object": "fork"}, ["n0", "n2"]))
    notes = prune_dangling_deps(nodes)
    assert nodes[1].deps == ["n0"]
    assert notes and "do not exist" in notes[0]
    TaskGraph("i", nodes)          # now constructible, which it was not before


def test_leaves_a_valid_dependency_alone():
    nodes = _nodes(("n0", "open_drawer", {}, []),
                   ("n1", "pick", {"object": "fork"}, ["n0"]))
    assert prune_dangling_deps(nodes) == []
    assert nodes[1].deps == ["n0"]


# --- break_cycles ----------------------------------------------------------

def test_breaks_a_cycle_by_keeping_the_models_own_emission_order():
    nodes = _nodes(("n0", "pick", {"object": "mug"}, ["n1"]),
                   ("n1", "place", {"object": "mug", "target": "mug"}, ["n0"]))
    notes = break_cycles(nodes)
    assert notes
    g = TaskGraph("i", nodes)                       # no cycle left
    assert g.topological_order() == ["n0", "n1"]    # pick still precedes place


# --- the boundary ----------------------------------------------------------

def test_deleting_and_reordering_is_allowed():
    before = _nodes(("n0", "pick", {"object": "mug"}, []),
                    ("n1", "place", {"object": "mug", "target": "mug"}, []))
    after = [before[1], before[0]]                  # reordered
    assert_nothing_invented(before, after)
    assert_nothing_invented(before, [before[0]])    # one deleted


def test_inventing_a_node_raises():
    before = _nodes(("n0", "pick", {"object": "mug"}, []))
    after = before + _nodes(("n1", "place", {"object": "mug", "target": "mug"}, []))
    with pytest.raises(AssertionError, match="invented"):
        assert_nothing_invented(before, after)


def test_changing_an_argument_counts_as_invention():
    before = _nodes(("n0", "pick", {"object": "mug"}, []))
    after = _nodes(("n0", "pick", {"object": "plate"}, []))
    with pytest.raises(AssertionError, match="invented"):
        assert_nothing_invented(before, after)


def test_changing_the_arm_counts_as_invention():
    before = [Node(id="n0", skill="pick", args={"object": "mug"}, arm="A")]
    after = [Node(id="n0", skill="pick", args={"object": "mug"}, arm="B")]
    with pytest.raises(AssertionError, match="invented"):
        assert_nothing_invented(before, after)


def test_a_mostly_discarded_answer_is_refused_rather_than_rescued():
    """The failure this guard was written for: a degenerate generation whose one
    surviving node repaired into a valid one-step plan the scheduler accepted.
    A one-step plan runs and fails the task; a rejection hands the episode to the
    deterministic planner. Rejecting is strictly better."""
    with pytest.raises(ValueError, match="discarded more of the answer than it kept"):
        assert_repair_was_a_minority(kept=1, dropped=40)


def test_a_plan_with_a_few_mistakes_is_still_repaired():
    assert_repair_was_a_minority(kept=6, dropped=1)     # does not raise
    assert_repair_was_a_minority(kept=3, dropped=3)     # a tie is still repairable


def test_repair_is_off_unless_asked_for():
    """The committed numbers were produced without it, so the default must not
    change under anyone's feet."""
    from apparecchiato.planner.vlm import VLMPlanner
    assert VLMPlanner(backend="openai").repair is False
    assert VLMPlanner(backend="openai", repair=True).repair is True
