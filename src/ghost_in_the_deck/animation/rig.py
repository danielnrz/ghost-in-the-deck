"""Skeletal control of the avatar inside Panda3D.

This is the only module that manipulates bones. Everything above it works in
terms of named joints and angles, so the rig can be swapped without touching the
behaviour or animation logic.

Two offsets stack on the asset's rest pose:

    neutral   a fixed correction from the A-pose the MPFB asset is built in to a
              natural standing position, applied on load
    offset    the movement, measured from that neutral pose and clamped

Callers only ever see the second one. Angles are relative because the MPFB
skeleton has non-zero rest rotations, so absolute values would be meaningless.
"""

from __future__ import annotations

from pathlib import Path

from direct.actor.Actor import Actor
from panda3d.core import NodePath


class AvatarRig:
    """Wraps an Actor and exposes a small set of directly driven joints."""

    # Everything the groove drives. Fingers, feet and toes are deliberately left
    # out: they are hard to move convincingly and a bad foot reads worse than a
    # still one.
    CONTROLLED = (
        "pelvis",
        "spine_01",
        "spine_02",
        "spine_03",
        "neck_01",
        "head",
        "clavicle_l",
        "clavicle_r",
        "upperarm_l",
        "upperarm_r",
        "lowerarm_l",
        "lowerarm_r",
        "thigh_l",
        "thigh_r",
        "calf_l",
        "calf_r",
    )

    # The A-pose correction, in degrees from the asset's rest pose. Solved offline
    # against this skeleton by searching for upper-arm angles that put the arms
    # 12 degrees off vertical, then choosing an elbow bend of about 20 degrees;
    # the search is recorded in the Phase 1A notes rather than run at startup.
    #
    # Applied at runtime rather than baked into the asset, so the stance shares
    # one coordinate convention with the movement stacked on top of it and the
    # exported mesh and its bind weights stay untouched.
    NEUTRAL_POSE = {
        # Upper-arm pitch is what swings the arms down out of the A-pose; roll
        # is mirrored so both elbows point back rather than out. Larger values
        # reach the target skeleton angle but the linear-blend skin cannot take
        # it - the shoulder splays into a wing - so these were chosen by
        # rendering candidates rather than by the geometric solve alone.
        "upperarm_l": (0.0, 30.0, 15.0),
        "upperarm_r": (0.0, 30.0, -15.0),
        "lowerarm_l": (0.0, -6.0, 0.0),
        "lowerarm_r": (0.0, -6.0, 0.0),
        "clavicle_l": (0.0, -3.0, 0.0),     # shoulders dropped, not braced
        "clavicle_r": (0.0, -3.0, 0.0),
        "spine_01": (0.0, 3.0, 0.0),        # a touch forward, as if over a deck
        "calf_l": (0.0, 5.0, 0.0),          # knees unlocked rather than locked
        "calf_r": (0.0, 5.0, 0.0),
    }

    # Degrees a joint may move *from the neutral pose*, per axis. These are not
    # anatomy, they are a guard rail: they stop a bad coefficient upstream from
    # folding the avatar in half, and they are what the tests assert against.
    LIMITS = {
        "pelvis": (6.0, 8.0, 6.0),
        "spine_01": (10.0, 30.0, 10.0),
        "spine_02": (12.0, 20.0, 12.0),
        "spine_03": (12.0, 20.0, 12.0),
        "neck_01": (15.0, 18.0, 12.0),
        "head": (22.0, 22.0, 15.0),
        "clavicle_l": (12.0, 15.0, 12.0),
        "clavicle_r": (12.0, 15.0, 12.0),
        "upperarm_l": (25.0, 25.0, 45.0),
        "upperarm_r": (25.0, 25.0, 45.0),
        "lowerarm_l": (15.0, 35.0, 15.0),
        "lowerarm_r": (15.0, 35.0, 15.0),
        "thigh_l": (10.0, 15.0, 10.0),
        "thigh_r": (10.0, 15.0, 10.0),
        "calf_l": (8.0, 20.0, 8.0),
        "calf_r": (8.0, 20.0, 8.0),
    }
    DEFAULT_LIMIT = (15.0, 15.0, 15.0)

    def __init__(self, model_path: Path | str, parent: NodePath | None = None):
        self.actor = Actor(str(model_path))
        if parent is not None:
            self.actor.reparentTo(parent)

        self._joints: dict[str, NodePath] = {}
        self._rest: dict[str, tuple[float, float, float]] = {}
        for name in self.CONTROLLED:
            node = self.actor.controlJoint(None, "modelRoot", name)
            if node is None or node.isEmpty():
                continue
            self._joints[name] = node
            hpr = node.getHpr()
            # The neutral correction folds into the stored rest pose, so every
            # angle above this line is measured from a naturally standing body.
            base_h, base_p, base_r = self.NEUTRAL_POSE.get(name, (0.0, 0.0, 0.0))
            self._rest[name] = (hpr.x + base_h, hpr.y + base_p, hpr.z + base_r)

        missing = [n for n in self.CONTROLLED if n not in self._joints]
        if missing:
            raise RuntimeError(f"rig is missing expected joints: {missing}")

        self.reset()

    def joint_names(self) -> list[str]:
        """Every joint in the skeleton, not just the controlled ones."""
        names: list[str] = []

        def walk(node) -> None:
            names.append(node.getName())
            for i in range(node.getNumChildren()):
                walk(node.getChild(i))

        walk(self.actor.getPartBundle("modelRoot"))
        return names

    def joint(self, name: str) -> NodePath:
        return self._joints.get(name, NodePath())

    def rest_hpr(self, name: str) -> tuple[float, float, float]:
        return self._rest.get(name, (0.0, 0.0, 0.0))

    def limit_for(self, name: str) -> tuple[float, float, float]:
        return self.LIMITS.get(name, self.DEFAULT_LIMIT)

    def set_offset(self, name: str, heading: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
        """Rotate a joint by degrees relative to its rest pose.

        Clamped to the joint's limits here rather than in the caller, so nothing
        upstream - now or later - can drive the skeleton somewhere it should not
        go.
        """
        node = self._joints.get(name)
        if node is None:
            return
        max_h, max_p, max_r = self.limit_for(name)
        heading = min(max(heading, -max_h), max_h)
        pitch = min(max(pitch, -max_p), max_p)
        roll = min(max(roll, -max_r), max_r)
        rest_h, rest_p, rest_r = self._rest[name]
        node.setHpr(rest_h + heading, rest_p + pitch, rest_r + roll)

    def offset_of(self, name: str) -> tuple[float, float, float]:
        node = self._joints.get(name)
        if node is None:
            return (0.0, 0.0, 0.0)
        hpr = node.getHpr()
        rest_h, rest_p, rest_r = self._rest[name]
        return (hpr.x - rest_h, hpr.y - rest_p, hpr.z - rest_r)

    def reset(self) -> None:
        for name, node in self._joints.items():
            node.setHpr(*self._rest[name])

    def expose(self, name: str) -> NodePath:
        """A scene-graph node that follows ``name`` as the skeleton deforms.

        The nodes returned by controlJoint are inputs to the character; they do
        not move when a parent bone rotates. An exposed joint is the output side
        and is what to read when checking that a rotation actually propagated.
        """
        return self.actor.exposeJoint(None, "modelRoot", name)

    def force_update(self) -> None:
        """Recompute the skeleton now, without waiting for a rendered frame."""
        self.actor.getPartBundle("modelRoot").forceUpdate()
