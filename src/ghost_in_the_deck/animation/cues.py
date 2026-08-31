"""Turning analysed music into motion cues.

This is the seam reserved for the DJ behaviour engine. Today a cue is emitted
for every beat and its strength comes straight from the track's energy. Later a
real behaviour layer will sit here and decide *what the DJ does* (nod, reach for
the mixer, ride the filter) from the same MusicFeatures input, without the
animation code below it having to change.

Nothing here imports librosa or Panda3D.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..audio.features import MusicFeatures


@dataclass(frozen=True)
class MotionCue:
    """A single request for the avatar to do something."""

    kind: str
    scheduled_time: float
    strength: float
    index: int


class BeatCueSource:
    """Emits one cue per beat, in playback order.

    The caller polls with the current playback position; the source returns the
    cues that have come due since the previous poll. Cues that are already late
    by more than ``max_lateness`` are dropped rather than fired in a burst, which
    is what keeps the avatar from convulsing after a stall.
    """

    def __init__(self, features: MusicFeatures, max_lateness: float = 0.15):
        self.features = features
        self.max_lateness = max_lateness
        self._next = 0

    @property
    def total(self) -> int:
        return len(self.features.beats)

    @property
    def pending(self) -> int:
        return max(0, self.total - self._next)

    def reset(self, time: float = 0.0) -> None:
        beats = self.features.beats
        self._next = 0
        while self._next < len(beats) and beats[self._next] < time:
            self._next += 1

    def strength_at(self, time: float) -> float:
        """How hard the avatar should move, from the music's own energy."""
        bass = self.features.envelope_at("bass_energy", time)
        onset = self.features.envelope_at("onset_strength", time)
        level = 0.65 * bass + 0.35 * onset
        return max(0.25, min(1.0, level))

    def poll(self, time: float) -> list[MotionCue]:
        """Return cues due at or before ``time``."""
        beats = self.features.beats
        due: list[MotionCue] = []
        while self._next < len(beats) and beats[self._next] <= time:
            scheduled = beats[self._next]
            index = self._next
            self._next += 1
            if time - scheduled > self.max_lateness:
                continue  # too stale to be worth animating
            due.append(
                MotionCue(
                    kind="beat",
                    scheduled_time=scheduled,
                    strength=self.strength_at(scheduled),
                    index=index,
                )
            )
        return due
