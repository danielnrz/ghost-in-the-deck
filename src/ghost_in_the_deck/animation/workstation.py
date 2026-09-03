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

# The tabletop slab itself: its footprint and thickness. ``scene/workstation.py``
# builds the actual table geometry from these same numbers (its "LEG_SIZE" is
# this thickness, reused for the legs' cross-section too) - this is also the
# single source of truth a reach path checks itself against so a hand can be
# routed around the table rather than through it. Chosen to comfortably
# enclose both deck platters and the control row in front of them.
TABLE_WIDTH = 1.10
TABLE_DEPTH = 0.34
TABLE_TOP_THICKNESS = 0.05
_TABLE_Y_CENTER = DECK_DEPTH + TABLE_DEPTH * 0.15
TABLE_NEAR_EDGE_Y = _TABLE_Y_CENTER + TABLE_DEPTH / 2.0   # edge closest to the avatar

# Measured from the real rig, the same way HEAD_PIVOT above is: where the
# calibrated clearance pose (scripts/solve_arm_ik.py) actually leaves the
# wrist. Not a point the reach is aimed at - see gesture_pose.py's module
# docstring for why an independently-aimed hover point does not work on this
# rig - but the real, measured result of that pose, kept here for the same
# reason HEAD_PIVOT is: so other code and tests have one place to read it
# from rather than a number copied out of a script's console output.
LEFT_CLEARANCE: Point = (0.492, -0.240, 1.078)
RIGHT_CLEARANCE: Point = (-0.492, -0.240, 1.078)


@dataclass(frozen=True)
class DJWorkstationTargets:
    """Meaningful world-space points on the workstation.

    ``deck_center`` is the midpoint between the two platters - what "the deck"
    means when a gesture has no particular side. ``left_deck``/``right_deck``
    are the platter centres; ``left_controls``/``right_controls`` sit closer to
    the avatar, roughly where a fader or a jog wheel's front edge would be.
    ``left_clearance``/``right_clearance`` are not physical controls at all -
    they are where the calibrated clearance pose leaves the wrist, a waypoint
    a reach passes through to route around the table instead of through it.
    """

    surface_height: float
    deck_center: Point
    mixer_center: Point
    left_deck: Point
    right_deck: Point
    left_controls: Point
    right_controls: Point
    left_clearance: Point
    right_clearance: Point

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

    def clearance_for(self, side: str | None) -> Point:
        """The hover waypoint a reach on ``side`` passes through en route."""
        if side == "r":
            return self.right_clearance
        return self.left_clearance


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
    left_clearance=LEFT_CLEARANCE,
    right_clearance=RIGHT_CLEARANCE,
)


def tabletop_bounds() -> tuple[Point, Point]:
    """The tabletop slab's axis-aligned world-space corners, ``(min, max)``.

    Matches the geometry ``scene/workstation.py`` actually builds - useful for
    calibration and as a quick sanity check, not as a substitute for measuring
    the built scene: the reach-path tests measure the constructed geometry
    directly rather than trusting this formula reproduces it.
    """
    return (
        (-TABLE_WIDTH / 2.0, _TABLE_Y_CENTER - TABLE_DEPTH / 2.0, SURFACE_HEIGHT - TABLE_TOP_THICKNESS),
        (TABLE_WIDTH / 2.0, _TABLE_Y_CENTER + TABLE_DEPTH / 2.0, SURFACE_HEIGHT),
    )
