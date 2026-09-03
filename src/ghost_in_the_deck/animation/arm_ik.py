"""Two-bone arm IK geometry: shoulder -> elbow -> wrist, aimed at a target.

Standard analytic 2-bone IK (law of cosines), operating on plain ``(x, y, z)``
tuples - no Panda3D, no dependency on this project's rig at all, so it is
testable with synthetic arm lengths and targets and reusable if the avatar
ever changes. It answers one question: given a shoulder position, a target,
and the two bone lengths, where should the elbow go?

It does **not** answer "what Panda3D joint rotation achieves that" - this
rig's ``controlJoint`` transform does not correspond to a plain scene-graph
direction (confirmed by testing ``NodePath.lookAt()`` on it directly: it
produced a direction with dot -0.78 against the intended target, i.e. nearly
backwards). The only way found to convert a world-space aim into a correct
joint offset was a numerical damped-least-squares solve against the real,
live rig, watching actual world positions - not any assumed Euler convention.
That solve is a one-time calibration, not runtime work: see
``scripts/solve_arm_ik.py``, which used this module's ``solve_elbow`` for the
geometry and produced the constants ``gesture_pose.py`` uses. This keeps the
per-frame cost of a reach gesture down to a handful of multiplications - the
same "peak offset times envelope weight" shape every other gesture already
uses - while the actual aiming math stays honest about the real rig rather
than a guessed formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float, float]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a: Point) -> float:
    return math.sqrt(_dot(a, a))


def _normalized(a: Point) -> Point:
    length = _length(a)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    return _scale(a, 1.0 / length)


@dataclass(frozen=True)
class ElbowSolution:
    """Where the elbow goes, and how the target was handled.

    ``aim_point`` is the target actually used for the geometry - equal to
    ``target`` when it was reachable, otherwise the closest point on the
    reachable sphere along the shoulder-to-target direction. A caller that
    wants to know how far short a clamp fell can compare
    ``distance(shoulder, aim_point)`` against ``distance(shoulder, target)``.
    """

    elbow: Point
    aim_point: Point
    was_clamped: bool
    reach_distance: float   # distance from shoulder to aim_point, after clamping


def distance(a: Point, b: Point) -> float:
    return _length(_sub(a, b))


def solve_elbow(
    shoulder: Point,
    target: Point,
    upper_length: float,
    fore_length: float,
    pole: Point = (0.0, 0.0, -1.0),
) -> ElbowSolution:
    """Two-bone analytic IK: where should the elbow sit to reach ``target``?

    ``pole`` is a bend-plane preference - roughly, which way the elbow should
    point - given as a direction, not a hard constraint; it is projected
    perpendicular to the shoulder-target axis to find the actual bend plane.

    An unreachable target (farther than ``upper_length + fore_length``, or
    closer than their difference, which would need the arm to fold back past
    straight) is clamped to the nearest reachable distance along the same
    direction, rather than producing an invalid rotation - the elbow solution
    is always geometrically consistent: exactly ``upper_length`` from the
    shoulder and exactly ``fore_length`` from ``aim_point``.
    """
    if upper_length <= 0.0 or fore_length <= 0.0:
        raise ValueError("bone lengths must be positive")

    offset = _sub(target, shoulder)
    raw_distance = _length(offset)

    max_reach = upper_length + fore_length
    min_reach = abs(upper_length - fore_length)
    # A hair inside the exact bounds: exactly at max/min reach the triangle
    # degenerates (zero-area) and the bend-plane split below becomes
    # numerically unstable.
    epsilon = 1e-6
    clamped_distance = max(min_reach + epsilon, min(max_reach - epsilon, raw_distance))
    was_clamped = abs(clamped_distance - raw_distance) > 1e-9

    if raw_distance < 1e-9:
        # Degenerate: target sits on the shoulder. Any direction is as good
        # as any other; fall back to the pole direction itself.
        direction = _normalized(pole)
    else:
        direction = _scale(offset, 1.0 / raw_distance)

    aim_point = _add(shoulder, _scale(direction, clamped_distance))

    # Law of cosines: angle at the shoulder between shoulder->aim_point and
    # shoulder->elbow.
    cos_shoulder_angle = (
        upper_length**2 + clamped_distance**2 - fore_length**2
    ) / (2.0 * upper_length * clamped_distance)
    cos_shoulder_angle = max(-1.0, min(1.0, cos_shoulder_angle))
    shoulder_angle = math.acos(cos_shoulder_angle)

    pole_dir = _normalized(pole)
    pole_perp = _sub(pole_dir, _scale(direction, _dot(pole_dir, direction)))
    if _length(pole_perp) < 1e-6:
        # Pole is parallel to the reach direction; fall back to a fixed "up"
        # reference so the bend plane is still well-defined.
        fallback = (0.0, 0.0, 1.0)
        pole_perp = _sub(fallback, _scale(direction, _dot(fallback, direction)))
        if _length(pole_perp) < 1e-6:
            fallback = (1.0, 0.0, 0.0)
            pole_perp = _sub(fallback, _scale(direction, _dot(fallback, direction)))
    pole_perp = _normalized(pole_perp)

    elbow_direction = _add(
        _scale(direction, math.cos(shoulder_angle)),
        _scale(pole_perp, math.sin(shoulder_angle)),
    )
    elbow = _add(shoulder, _scale(elbow_direction, upper_length))

    return ElbowSolution(
        elbow=elbow,
        aim_point=aim_point,
        was_clamped=was_clamped,
        reach_distance=clamped_distance,
    )
