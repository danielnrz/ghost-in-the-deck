"""Builds the DJ workstation's Panda3D geometry.

Procedural, not a downloaded model: the goal is a silhouette any viewer reads
instantly as "DJ equipment" - a table, two deck platters, a mixer strip with a
few knobs and faders between them - not a faithful reproduction of any real
hardware. Every position and dimension here comes from
``animation.workstation``, which is also what the behaviour layer reads to
decide where to look and reach. This module only turns those same numbers into
boxes and cylinders; it never invents a coordinate of its own.
"""

from __future__ import annotations

from panda3d.core import NodePath

from ..animation.workstation import (
    CONTROL_DEPTH,
    CONTROLLER_THICKNESS,
    DECK_DEPTH,
    DECK_SPACING,
    SURFACE_HEIGHT,
    DJWorkstationTargets,
)
from .primitives import box, cylinder

TABLE_WIDTH = 1.10
TABLE_DEPTH = 0.34
LEG_SIZE = 0.05

DECK_RADIUS = 0.14
DECK_HEIGHT = CONTROLLER_THICKNESS

MIXER_WIDTH = 0.30
MIXER_DEPTH = TABLE_DEPTH * 0.7
MIXER_HEIGHT = CONTROLLER_THICKNESS * 1.4

KNOB_RADIUS = 0.012
KNOB_HEIGHT = 0.02
FADER_SIZE = (0.012, 0.05, 0.012)

TABLE_COLOR = (0.16, 0.15, 0.17, 1.0)
DECK_COLOR = (0.08, 0.08, 0.09, 1.0)
MIXER_COLOR = (0.20, 0.20, 0.23, 1.0)
KNOB_COLOR = (0.55, 0.55, 0.58, 1.0)
FADER_COLOR = (0.75, 0.35, 0.20, 1.0)


def build_workstation(
    parent: NodePath, targets: DJWorkstationTargets | None = None
) -> NodePath:
    """Attach the workstation geometry under ``parent`` and return its root.

    ``targets`` defaults to the module-level ``DEFAULT_TARGETS`` that the
    behaviour layer also uses; passing a different one only makes sense in a
    test that wants to prove the geometry actually follows the targets rather
    than a hard-coded layout of its own.
    """
    from ..animation.workstation import DEFAULT_TARGETS

    targets = targets or DEFAULT_TARGETS
    root = parent.attachNewNode("dj-workstation")

    _build_table(root)
    _build_deck(root, targets.left_deck)
    _build_deck(root, targets.right_deck)
    _build_mixer(root, targets.mixer_center)

    return root


def _build_table(root: NodePath) -> None:
    top = box(TABLE_WIDTH, TABLE_DEPTH, LEG_SIZE, TABLE_COLOR)
    top.reparentTo(root)
    top.setPos(0.0, DECK_DEPTH + TABLE_DEPTH * 0.15, SURFACE_HEIGHT - LEG_SIZE)

    leg_positions = [
        (x, DECK_DEPTH + dy)
        for x in (-TABLE_WIDTH / 2 + LEG_SIZE, TABLE_WIDTH / 2 - LEG_SIZE)
        for dy in (TABLE_DEPTH * 0.15 - TABLE_DEPTH / 2 + LEG_SIZE, TABLE_DEPTH * 0.15 + TABLE_DEPTH / 2 - LEG_SIZE)
    ]
    for x, y in leg_positions:
        leg = box(LEG_SIZE, LEG_SIZE, SURFACE_HEIGHT - LEG_SIZE, TABLE_COLOR)
        leg.reparentTo(root)
        leg.setPos(x, y, 0.0)


def _build_deck(root: NodePath, center) -> None:
    x, y, top_z = center
    base = box(DECK_RADIUS * 2.2, DECK_RADIUS * 2.2, DECK_HEIGHT * 0.4, DECK_COLOR)
    base.reparentTo(root)
    base.setPos(x - DECK_RADIUS * 1.1, y - DECK_RADIUS * 1.1, top_z - DECK_HEIGHT)

    platter = cylinder(DECK_RADIUS, DECK_HEIGHT * 0.6, (0.25, 0.25, 0.27, 1.0))
    platter.reparentTo(root)
    platter.setPos(x, y, top_z - DECK_HEIGHT)


def _build_mixer(root: NodePath, center) -> None:
    x, y, top_z = center
    body = box(MIXER_WIDTH, MIXER_DEPTH, MIXER_HEIGHT, MIXER_COLOR)
    body.reparentTo(root)
    body.setPos(x - MIXER_WIDTH / 2, y - MIXER_DEPTH / 2, top_z - MIXER_HEIGHT)

    for i in range(3):
        knob = cylinder(KNOB_RADIUS, KNOB_HEIGHT, KNOB_COLOR)
        knob.reparentTo(root)
        kx = x + (i - 1) * MIXER_WIDTH * 0.28
        knob.setPos(kx, y - MIXER_DEPTH * 0.25, top_z)

    for i in range(2):
        fader = box(*FADER_SIZE, FADER_COLOR)
        fader.reparentTo(root)
        fx = x + (i - 0.5) * MIXER_WIDTH * 0.4
        fader.setPos(fx, y + MIXER_DEPTH * 0.2, top_z)
