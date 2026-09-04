"""Builds the DJ workstation's Panda3D geometry.

Procedural, not a downloaded model: the goal is a silhouette any viewer reads
instantly as "DJ equipment" - a table, two deck platters, a mixer strip with a
few knobs and faders between them, and a small control cluster on each side -
not a faithful reproduction of any real hardware. Every position and dimension
here comes from ``animation.workstation``, which is also what the behaviour
layer reads to decide where to look and reach: the side control clusters are
built directly from ``targets.left_controls``/``right_controls``, the same
points ``hand_to_deck`` reaches toward, so a viewer can see exactly which
control the hand is aiming at rather than a bare patch of tabletop. This
module only turns those numbers into boxes and cylinders; it never invents a
coordinate of its own.

Nodes that a test or the review renderer might need to find again are named:
the tabletop slab is ``"tabletop"``, and each side's cluster root is
``"control-cluster-l"`` / ``"control-cluster-r"`` - ``NodePath.find()`` or
``getTightBounds()`` on those reads the actual built geometry, not a constant
copied out of this file.
"""

from __future__ import annotations

from panda3d.core import NodePath

from ..animation.workstation import (
    CONTROL_DEPTH,
    CONTROLLER_THICKNESS,
    DECK_DEPTH,
    DECK_SPACING,
    SURFACE_HEIGHT,
    TABLE_DEPTH,
    TABLE_TOP_THICKNESS,
    TABLE_WIDTH,
    DJWorkstationTargets,
)
from .primitives import box, cylinder

# The table top's thickness and each leg's square cross-section happen to be
# the same value; TABLE_TOP_THICKNESS (animation.workstation's single source
# of truth for the slab) is reused for both rather than this module keeping a
# second, independent number.
LEG_SIZE = TABLE_TOP_THICKNESS

DECK_RADIUS = 0.14
DECK_HEIGHT = CONTROLLER_THICKNESS

MIXER_WIDTH = 0.30
MIXER_DEPTH = TABLE_DEPTH * 0.7
MIXER_HEIGHT = CONTROLLER_THICKNESS * 1.4

KNOB_RADIUS = 0.012
KNOB_HEIGHT = 0.02
FADER_SIZE = (0.012, 0.05, 0.012)

# The side control clusters: smaller than the central mixer, since each one
# only needs to read as "a control", not as the whole mixer strip.
CONTROL_PANEL_WIDTH = 0.16
CONTROL_PANEL_DEPTH = 0.12
CONTROL_PANEL_HEIGHT = CONTROLLER_THICKNESS

TABLE_COLOR = (0.16, 0.15, 0.17, 1.0)
DECK_COLOR = (0.08, 0.08, 0.09, 1.0)
MIXER_COLOR = (0.20, 0.20, 0.23, 1.0)
CONTROL_PANEL_COLOR = (0.18, 0.18, 0.21, 1.0)
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
    _build_control_cluster(root, targets.left_controls, "l")
    _build_control_cluster(root, targets.right_controls, "r")

    return root


def _build_table(root: NodePath) -> None:
    top = box(TABLE_WIDTH, TABLE_DEPTH, LEG_SIZE, TABLE_COLOR)
    top.setName("tabletop")
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
    # box() is centred on X and Y (see primitives.box), so a box centred on the
    # platter is just setPos(x, y, ...) - no half-extent offset. Earlier code
    # subtracted a half-width here (and in the mixer and control panels below),
    # a leftover from assuming corner-origin geometry; it left every slab
    # sitting half its own width off the piece it belongs to, which is what let
    # the right hand's reach pass through the mis-placed central mixer.
    base.setPos(x, y, top_z - DECK_HEIGHT)

    platter = cylinder(DECK_RADIUS, DECK_HEIGHT * 0.6, (0.25, 0.25, 0.27, 1.0))
    platter.reparentTo(root)
    platter.setPos(x, y, top_z - DECK_HEIGHT)


def _build_mixer(root: NodePath, center) -> None:
    x, y, top_z = center
    body = box(MIXER_WIDTH, MIXER_DEPTH, MIXER_HEIGHT, MIXER_COLOR)
    body.reparentTo(root)
    body.setPos(x, y, top_z - MIXER_HEIGHT)

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


def _build_control_cluster(root: NodePath, target, label: str) -> None:
    """A small, visible control cluster built directly at ``target``.

    ``target`` is ``targets.left_controls`` or ``targets.right_controls``
    verbatim - not a coordinate re-derived or hard-coded here - so the panel's
    top sits exactly at the point ``hand_to_deck`` reaches toward, the same
    convention the central mixer already uses for its own knobs. Left and
    right are visually identical, mirrored only in X.
    """
    x, y, top_z = target
    cluster = root.attachNewNode(f"control-cluster-{label}")

    panel = box(CONTROL_PANEL_WIDTH, CONTROL_PANEL_DEPTH, CONTROL_PANEL_HEIGHT, CONTROL_PANEL_COLOR)
    panel.setName(f"control-panel-{label}")
    panel.reparentTo(cluster)
    panel.setPos(x, y, top_z - CONTROL_PANEL_HEIGHT)

    knob = cylinder(KNOB_RADIUS * 1.3, KNOB_HEIGHT, KNOB_COLOR)
    knob.setName(f"control-knob-{label}")
    knob.reparentTo(cluster)
    knob.setPos(x - CONTROL_PANEL_WIDTH * 0.18, y - CONTROL_PANEL_DEPTH * 0.15, top_z)

    fader = box(*FADER_SIZE, FADER_COLOR)
    fader.setName(f"control-fader-{label}")
    fader.reparentTo(cluster)
    fader.setPos(x + CONTROL_PANEL_WIDTH * 0.15, y + CONTROL_PANEL_DEPTH * 0.15, top_z)
