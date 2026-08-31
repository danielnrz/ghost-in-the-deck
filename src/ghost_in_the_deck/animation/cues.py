"""Turning analysed music into motion cues.

This is the seam reserved for the DJ behaviour engine. Today a cue is emitted
for every beat and its strength comes straight from the track's energy. Later a
real behaviour layer will sit here and decide *what the DJ does* (nod, reach for
the mixer, ride the filter) from the same MusicFeatures input, without the
animation code below it having to change.

Nothing here imports librosa or Panda3D.

The timeline is addressed by absolute playback time and holds no playback state.
Asking it about time T gives the same answer whether or not anything was asked
before, which is what lets the renderer stall without the music falling behind.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from ..audio.features import MusicFeatures


@dataclass(frozen=True)
class MotionCue:
    """A single request for the avatar to do something."""

    kind: str
    scheduled_time: float
    strength: float
    index: int


class BeatTimeline:
    """Every motion cue in a track, looked up by absolute playback time.

    All cues are built up front. That keeps lookup to a binary search and, more
    importantly, makes a cue's strength a property of the music rather than of
    when it happened to be requested.
    """

    def __init__(self, features: MusicFeatures):
        self.features = features
        self._times: list[float] = list(features.beats)
        self._cues: list[MotionCue] = [
            MotionCue(
                kind="beat",
                scheduled_time=time,
                strength=self._strength_at(time),
                index=index,
            )
            for index, time in enumerate(self._times)
        ]

    def _strength_at(self, time: float) -> float:
        """How hard the avatar should move, from the music's own energy."""
        bass = self.features.envelope_at("bass_energy", time)
        onset = self.features.envelope_at("onset_strength", time)
        level = 0.65 * bass + 0.35 * onset
        return max(0.25, min(1.0, level))

    def __len__(self) -> int:
        return len(self._cues)

    @property
    def end_time(self) -> float:
        """Playback time of the last cue, or 0.0 for an empty timeline."""
        return self._times[-1] if self._times else 0.0

    def index_before(self, time: float) -> int:
        """Index of the latest cue at or before ``time``; -1 if there is none."""
        return bisect_right(self._times, time) - 1

    def cue_before(self, time: float) -> MotionCue | None:
        index = self.index_before(time)
        return self._cues[index] if index >= 0 else None

    def cues_in(self, start: float, end: float) -> list[MotionCue]:
        """Cues scheduled in the half-open window ``(start, end]``."""
        low = bisect_right(self._times, start)
        high = bisect_right(self._times, end)
        return self._cues[low:high]

    def cue(self, index: int) -> MotionCue:
        return self._cues[index]
