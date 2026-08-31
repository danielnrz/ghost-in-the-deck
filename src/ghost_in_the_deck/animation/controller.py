"""Drives the rig from motion cues.

The controller keeps a small amount of state (a decaying impulse plus a slow
sway phase) and writes joint offsets every frame. It knows nothing about beats
or audio; it only consumes MotionCue objects.
"""

from __future__ import annotations

import math

from .cues import MotionCue
from .rig import AvatarRig

# Peak rotation in degrees at full cue strength.
NOD_HEAD = 13.0
NOD_NECK = 6.0
NOD_SPINE = 4.5
SHOULDER_DROP = 7.0
ARM_SWING = 5.0
SWAY_SPINE = 3.5
SWAY_HEAD = 2.0


class AvatarAnimator:
    """Turns cues into visible skeletal movement."""

    def __init__(self, rig: AvatarRig, decay: float = 0.16, sway_period: float = 2.0):
        self.rig = rig
        self.decay = decay          # seconds for the impulse to fall to ~37%
        self.sway_period = sway_period
        self._impulse = 0.0
        self._sway_phase = 0.0

    def reset(self) -> None:
        """Return to the rest pose and forget any movement in flight."""
        self._impulse = 0.0
        self._sway_phase = 0.0
        self.rig.reset()

    def apply_cue(self, cue: MotionCue) -> None:
        """Start a new movement. Re-triggering takes the stronger of the two so
        a fast beat cannot make the avatar stutter mid-nod."""
        self._impulse = max(self._impulse, cue.strength)

    def update(self, dt: float) -> None:
        if dt > 0.25:  # after a hitch, do not integrate a huge step
            dt = 0.25

        self._impulse *= math.exp(-dt / self.decay)
        if self._impulse < 1e-3:
            self._impulse = 0.0

        self._sway_phase += dt * (2.0 * math.pi / self.sway_period)
        if self._sway_phase > 2.0 * math.pi:
            self._sway_phase -= 2.0 * math.pi

        self._write_pose()

    @property
    def impulse(self) -> float:
        return self._impulse

    def _write_pose(self) -> None:
        hit = self._impulse
        sway = math.sin(self._sway_phase)

        # Nod: the head leads, the neck and spine follow with less travel, which
        # reads as a body movement rather than a detached head.
        self.rig.set_offset("head", pitch=-NOD_HEAD * hit, roll=SWAY_HEAD * sway)
        self.rig.set_offset("neck_01", pitch=-NOD_NECK * hit)
        self.rig.set_offset("spine_03", pitch=-NOD_SPINE * hit, roll=SWAY_SPINE * sway * 0.4)
        self.rig.set_offset("spine_02", pitch=-NOD_SPINE * 0.5 * hit, roll=SWAY_SPINE * sway * 0.3)
        self.rig.set_offset("spine_01", roll=SWAY_SPINE * sway * 0.3)

        # Shoulders bounce on the beat, mirrored left and right.
        self.rig.set_offset("clavicle_l", pitch=-SHOULDER_DROP * hit)
        self.rig.set_offset("clavicle_r", pitch=-SHOULDER_DROP * hit)
        self.rig.set_offset("upperarm_l", pitch=ARM_SWING * hit, roll=ARM_SWING * sway * 0.5)
        self.rig.set_offset("upperarm_r", pitch=ARM_SWING * hit, roll=-ARM_SWING * sway * 0.5)
