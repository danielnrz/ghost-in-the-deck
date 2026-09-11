"""Skeletal control of the avatar inside Panda3D.

This is the only module that manipulates bones. Everything above it works in
terms of named joints and angles, so the rig can be swapped without touching the
behaviour or animation logic.

The avatar's neutral standing pose lives in the asset itself: the Blender
generation script (``scripts/blender/make_avatar.py``) aligns the skeleton with
the mesh it deforms, poses it into a relaxed stance, and bakes that as the
asset's own rest pose before export. Everything callers see through
``set_offset`` is therefore already relative to a naturally standing body, and
is the movement alone - not a stance correction stacked underneath it. Angles
are relative rather than absolute because the MPFB skeleton still carries
non-zero rest rotations of its own, which absolute values would ignore.

``NEUTRAL_POSE`` is kept as an empty extension point in case a future asset
needs a runtime correction on top of its own rest pose; nothing populates it
today, and if it ever does, that offset folds into the stored rest pose the same
way the (now unused) A-pose correction used to.
"""

from __future__ import annotations

from pathlib import Path

from direct.actor.Actor import Actor
from panda3d.core import NodePath


class AvatarRig:
    """Wraps an Actor and exposes a small set of directly driven joints."""

    # Everything the groove drives, plus the two wrists. Individual fingers,
    # feet and toes are deliberately left out: they are hard to move
    # convincingly and a bad foot reads worse than a still one. ``hand_l`` /
    # ``hand_r`` are driven only by ``hand_to_deck`` (Phase 1B.2): the reach
    # needs to pitch the whole hand up so the fifteen un-posed finger joints it
    # carries angle across the controls instead of draping down into them. The
    # groove itself never writes to a wrist.
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
        "hand_l",
        "hand_r",
        "thigh_l",
        "thigh_r",
        "calf_l",
        "calf_r",
    )

    # A per-joint (heading, pitch, roll) correction applied on top of the
    # asset's own rest pose, in degrees. Empty: the neutral standing pose is
    # baked into the asset itself (see scripts/blender/make_avatar.py), so
    # there is nothing left to correct here. This stays as the mechanism for
    # doing so if a future asset ever needs one - see the module docstring.
    NEUTRAL_POSE: dict[str, tuple[float, float, float]] = {}

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
        # Roll is arm abduction - lifting the whole arm out from the body.
        # hand_to_deck's clearance pose needs a real lift to carry the trailing
        # finger joints over the tabletop's near edge (Phase 1B.2 F1); 55 is
        # that headroom, still far short of a shoulder's true abduction range
        # and well clear of small_hype's 42.
        "upperarm_l": (25.0, 25.0, 60.0),
        "upperarm_r": (25.0, 25.0, 60.0),
        # Elbow flexion. The reach's own bend needs ~40; the clearance pose
        # folds the forearm further up (CLEARANCE_ELBOW_PITCH, -60) to carry the
        # hand clear of the table without swinging it out, so the guard rail is
        # 60.
        "lowerarm_l": (15.0, 60.0, 15.0),
        "lowerarm_r": (15.0, 60.0, 15.0),
        # Wrist: only hand_to_deck moves it - mostly in pitch (fingers up
        # across the controls - base tilt plus a transient extra while crossing
        # the slab, ~55 deg peak) and a smaller hold-time roll (~12 deg) that
        # tips the pinky edge up off the deck platter. The bounds are the guard
        # rails for that one gesture, not an anatomical claim.
        "hand_l": (20.0, 60.0, 20.0),
        "hand_r": (20.0, 60.0, 20.0),
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
            # Any NEUTRAL_POSE correction folds into the stored rest pose, so
            # every angle below this line is measured from a naturally standing
            # body - the asset's own rest pose, currently, since the dict above
            # is empty.
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


class PerformanceRig(AvatarRig):
    """Live hand articulation; legacy diagnostic rig/calibration stays unchanged.

    Forearm rotation supplies palm orientation instead of forcing the wrist to
    its rail. Finger flexion is modest and always reset with the rest of the rig.
    """
    CONTROLLED = AvatarRig.CONTROLLED + tuple(
        f'{finger}_0{segment}_{side}'
        for side in ('l','r') for finger in ('thumb','index','middle','ring','pinky')
        for segment in (1,2,3))
    LIMITS = {
        **AvatarRig.LIMITS,
        **{f'upperarm_{s}': (35,45,95) for s in ('l','r')},
        **{f'lowerarm_{s}': (45,100,45) for s in ('l','r')},
        **{f'hand_{s}': (25,45,35) for s in ('l','r')},
        **{f'{f}_0{i}_{s}': (20,45,30) for s in ('l','r')
           for f in ('thumb','index','middle','ring','pinky') for i in (1,2,3)},
    }
