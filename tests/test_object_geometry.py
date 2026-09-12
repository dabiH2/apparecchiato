"""The object dimension table and the MJCF geoms must agree.

These are the invariants that, when they quietly stopped holding, cost the most
debugging time in this project:

  * an object resting on the table has its body origin at exactly the grasp
    height the planner aims at. When the plate, bottle and cutlery each spawned
    8-25 mm above the table, every release dropped and drifted, and every "grasp
    point" was a height nothing was ever at;
  * the gripper can actually close on the object -- the jaws open 36 mm, so an
    object wider than that is scenery, not a manipulable;
  * a 75 mm approach standoff clears the top of the object. The pads reach 48 mm
    ABOVE the grasp site, so an object taller than the standoff gets swept off
    the table sideways by the open jaws before they ever close.

They are checked against the generated MJCF, not against a second copy of the
numbers, so a change to the geoms with no change to the table fails here.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from apparecchiato.skills import GRIPPER_OPEN, grip_for
from apparecchiato.sim.layout import (
    OBJECT_HALF_H, OBJECT_TOP_H, OBJECT_RADIUS, GRASP_Z, sample_scene,
)
from apparecchiato.sim.scene import build_mjcf

KINDS = ("plate", "mug", "bottle", "spoon", "fork")
APPROACH_H = 0.075


def _geoms(xml: str, body_name: str):
    """(centre_z, half_z) of every geom of one object body, in the body frame."""
    root = ET.fromstring(xml)
    for body in root.iter("body"):
        if body.get("name") != body_name:
            continue
        out = []
        for g in body.findall("geom"):
            pos = [float(v) for v in (g.get("pos") or "0 0 0").split()]
            size = [float(v) for v in g.get("size").split()]
            half_z = size[-1] if g.get("type") in ("cylinder", "box") else size[0]
            out.append((pos[2], half_z))
        return out
    raise AssertionError(f"no body named {body_name!r} in the scene")


@pytest.mark.parametrize("kind", KINDS)
def test_grasp_height_equals_half_height(kind):
    """The grasp point is the object's mid-height, by construction."""
    assert GRASP_Z[kind] == pytest.approx(OBJECT_HALF_H[kind])


@pytest.mark.parametrize("kind", KINDS)
def test_jaws_open_wide_enough(kind):
    width = 2 * OBJECT_RADIUS[kind]
    # Fitting is not enough: the open pads have to descend PAST the object to
    # reach its mid-height. At 1.5 mm a side the pads caught the plate's rim on
    # the way down and stalled 13 mm high. 3 mm a side, minimum.
    assert width + 2 * 0.003 <= GRIPPER_OPEN, (
        f"a {width * 1000:.0f} mm {kind} leaves the open pads "
        f"{(GRIPPER_OPEN - width) / 2 * 1000:.1f} mm a side to descend past it")
    assert grip_for(kind) < width, (
        f"the commanded jaw target for a {kind} must squeeze it, not hover")


@pytest.mark.parametrize("kind", KINDS)
def test_approach_standoff_clears_the_object(kind):
    """The open jaws must pass OVER the object on the way in, not through it."""
    stick_up = OBJECT_TOP_H[kind] - OBJECT_HALF_H[kind]
    assert stick_up < APPROACH_H, (
        f"a {kind} stands {stick_up * 1000:.0f} mm above its own grasp point, so "
        f"the pads sweep through it at a {APPROACH_H * 1000:.0f} mm standoff")


@pytest.mark.parametrize("seed", range(4))
def test_objects_rest_on_the_table_not_above_it(seed):
    """Bottom of the lowest geom == the table, with the origin at GRASP_Z."""
    spec = sample_scene(seed)
    xml = build_mjcf(spec)
    for o in spec.objects:
        bottom = min(cz - hz for cz, hz in _geoms(xml, o.name))
        origin_z = OBJECT_HALF_H[o.kind] * o.scale
        # MJCF is written to 6 significant figures, so compare at 1 micrometre
        # -- far tighter than anything that matters, far looser than float noise.
        assert origin_z + bottom == pytest.approx(0.0, abs=1e-6), (
            f"{o.name} spawns {(origin_z + bottom) * 1000:+.1f} mm off the table")


@pytest.mark.parametrize("seed", range(4))
def test_declared_top_matches_the_geoms(seed):
    spec = sample_scene(seed)
    xml = build_mjcf(spec)
    for o in spec.objects:
        top = max(cz + hz for cz, hz in _geoms(xml, o.name))
        origin_z = OBJECT_HALF_H[o.kind] * o.scale
        assert origin_z + top == pytest.approx(OBJECT_TOP_H[o.kind] * o.scale, abs=1e-6), (
            f"{o.name}: OBJECT_TOP_H disagrees with its geoms")


@pytest.mark.parametrize("seed", range(4))
def test_goal_slots_sit_at_the_object_resting_height(seed):
    """A slot at the wrong height means the object is released in mid-air."""
    spec = sample_scene(seed)
    for name, g in spec.goals.items():
        o = spec.by_name(name)
        assert float(np.asarray(g)[2]) == pytest.approx(
            OBJECT_HALF_H[o.kind] * o.scale, abs=1e-6)
