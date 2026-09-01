"""Where the DJ workstation sits, in the avatar's own world space.

This is the single source of truth for the controller's layout. The Panda3D
scene builder (``scene/workstation.py``) places its geometry from these same
numbers, and the DJ behaviour layer (``dj_behavior.py``, ``gesture_pose.py``)
reads them to decide where to look and reach. Neither of those may hard-code a
competing set of coordinates - if the table moves, it moves here once.

Coordinates match the convention the rest of the animation code already uses,
established empirically while building the neutral stance and confirmed again
for this phase by probing the rig directly rather than assuming:

    +X   the avatar's own left (the side the ``_l`` joints are on)
    -Y   forward - the direction the avatar faces
    +Z   up, from the floor the avatar's feet stand on

Nothing here imports Panda3D. A world-space point is a plain ``(x, y, z)``
tuple of metres, not a ``Point3``, so this module stays usable from the
Panda3D-free behaviour and test code the same way ``MusicFeatures`` is.
"""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float, float]

# Measured from the neutral rig: the head joint's resting world position. Used
# by the attention system as a cheap stand-in for an eye point - close enough
# for a lightweight look-toward calculation, not a substitute for real IK.
HEAD_PIVOT: Point = (0.0, -0.035, 1.56)

# The tabletop's height above the floor, and its footprint. Chosen so the
# controller sits at roughly hip height for this avatar (whose standing height
# is about 1.67 m) - comfortable to look and reach down to, not so low that a
# glance reads as a bow.
SURFACE_HEIGHT = 0.95
CONTROLLER_THICKNESS = 0.05

# Half the distance between the two deck centres, and how far in front of the
# avatar the deck platters and the nearer control row sit.
DECK_SPACING = 0.34
DECK_DEPTH = -0.46      # platter centre: further from the avatar
CONTROL_DEPTH = -0.36   # front control row: closer to the avatar


@dataclass(frozen=True)
class DJWorkstationTargets:
    """Meaningful world-space points on the workstation.

    ``deck_center`` is the midpoint between the two platters - what "the deck"
    means when a gesture has no particular side. ``left_deck``/``right_deck``
    are the platter centres; ``left_controls``/``right_controls`` sit closer to
    the avatar, roughly where a fader or a jog wheel's front edge would be.
    """

    surface_height: float
    deck_center: Point
    mixer_center: Point
    left_deck: Point
    right_deck: Point
    left_controls: Point
    right_controls: Point

    def deck_for(self, side: str | None) -> Point:
        """The deck a gesture on ``side`` should reach toward."""
        if side == "l":
            return self.left_deck
        if side == "r":
            return self.right_deck
        return self.deck_center

    def controls_for(self, side: str | None) -> Point:
        if side == "l":
            return self.left_controls
        if side == "r":
            return self.right_controls
        return self.mixer_center


def _surface_point(x: float, y: float) -> Point:
    return (x, y, SURFACE_HEIGHT + CONTROLLER_THICKNESS)


DEFAULT_TARGETS = DJWorkstationTargets(
    surface_height=SURFACE_HEIGHT,
    deck_center=_surface_point(0.0, DECK_DEPTH),
    mixer_center=_surface_point(0.0, CONTROL_DEPTH),
    left_deck=_surface_point(DECK_SPACING, DECK_DEPTH),
    right_deck=_surface_point(-DECK_SPACING, DECK_DEPTH),
    left_controls=_surface_point(DECK_SPACING * 0.6, CONTROL_DEPTH),
    right_controls=_surface_point(-DECK_SPACING * 0.6, CONTROL_DEPTH),
)
