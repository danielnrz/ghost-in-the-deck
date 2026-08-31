"""Skeletal control of the avatar inside Panda3D.

This is the only module that manipulates bones. Everything above it works in
terms of named joints and angles, so the rig can be swapped without touching the
behaviour or animation logic.

Poses are expressed as *offsets from the rest pose*: the MPFB skeleton has
non-zero rest rotations, so absolute angles would be meaningless to callers.
"""

from __future__ import annotations

from pathlib import Path

from direct.actor.Actor import Actor
from panda3d.core import NodePath


class AvatarRig:
    """Wraps an Actor and exposes a small set of directly driven joints."""

    # Deliberately short: Phase 0 only needs to prove that named bones move.
    CONTROLLED = (
        "head",
        "neck_01",
        "spine_03",
        "spine_02",
        "spine_01",
        "clavicle_l",
        "clavicle_r",
        "upperarm_l",
        "upperarm_r",
    )

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
            self._rest[name] = (hpr.x, hpr.y, hpr.z)

        missing = [n for n in self.CONTROLLED if n not in self._joints]
        if missing:
            raise RuntimeError(f"rig is missing expected joints: {missing}")

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

    def set_offset(self, name: str, heading: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> None:
        """Rotate a joint by degrees relative to its rest pose."""
        node = self._joints.get(name)
        if node is None:
            return
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
