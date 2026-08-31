"""Drives the rig from the motion timeline.

The pose is a pure function of absolute playback time. Nothing is integrated
across frames, so the sequence of render frames leading up to time T cannot
change the pose produced at T. A renderer that stalls for a second simply misses
frames; when it comes back it draws the pose the music calls for *now*, with no
catching up and no accumulated drift.

This module knows nothing about beats, audio files or Panda3D windows; it reads
cues from a BeatTimeline and writes joint offsets to an AvatarRig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cues import BeatTimeline

# Peak rotation in degrees at full cue strength.
NOD_HEAD = 13.0
NOD_NECK = 6.0
NOD_SPINE = 4.5
SHOULDER_DROP = 7.0
ARM_SWING = 5.0
SWAY_SPINE = 3.5
SWAY_HEAD = 2.0

# Below this the impulse is treated as finished, so a pose that has decayed to
# nothing compares exactly equal to the rest pose.
IMPULSE_EPSILON = 1e-3


@dataclass(frozen=True)
class MotionState:
    """The avatar's movement evaluated at one absolute playback time."""

    time: float
    impulse: float
    sway: float
    beat_index: int      # -1 before the first cue
    beat_age: float      # seconds since that cue; inf when there is none

    @property
    def has_beat(self) -> bool:
        return self.beat_index >= 0


class AvatarAnimator:
    """Evaluates the pose for a playback time and writes it to the rig."""

    def __init__(
        self,
        rig,
        timeline: BeatTimeline | None = None,
        decay: float = 0.16,
        sway_period: float = 2.0,
    ):
        self.rig = rig
        self.timeline = timeline
        self.decay = decay          # seconds for the impulse to fall to ~37%
        self.sway_period = sway_period

    @property
    def visible_for(self) -> float:
        """How long a cue's response stays visible.

        Three decay constants leaves about 5% of the movement, which is the
        point past which a frame would show nothing worth calling a response.
        """
        return self.decay * 3.0

    def state_at(self, time: float) -> MotionState:
        """The pose at ``time``, derived only from ``time`` and the timeline."""
        impulse = 0.0
        beat_index = -1
        beat_age = math.inf

        if self.timeline is not None:
            latest = self.timeline.cue_before(time)
            if latest is not None:
                beat_index = latest.index
                beat_age = time - latest.scheduled_time

            # Older cues can still be fading, so the strongest surviving one
            # wins. The window is wide enough that anything outside it has
            # already decayed below IMPULSE_EPSILON.
            window = self.decay * 8.0
            for cue in self.timeline.cues_in(time - window, time):
                age = time - cue.scheduled_time
                impulse = max(impulse, cue.strength * math.exp(-age / self.decay))

        if impulse < IMPULSE_EPSILON:
            impulse = 0.0

        sway = math.sin(2.0 * math.pi * time / self.sway_period)
        return MotionState(
            time=time,
            impulse=impulse,
            sway=sway,
            beat_index=beat_index,
            beat_age=beat_age,
        )

    def apply_at(self, time: float) -> MotionState:
        """Evaluate the pose for ``time`` and write it to the rig."""
        state = self.state_at(time)
        self._write_pose(state)
        return state

    def reset(self) -> None:
        """Return the rig to its rest pose."""
        self.rig.reset()

    def _write_pose(self, state: MotionState) -> None:
        hit = state.impulse
        sway = state.sway

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
