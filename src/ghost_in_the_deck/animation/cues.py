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


# Beats per bar. Assumed rather than detected: the four-on-the-floor bar is the
# right default for the dance music this is aimed at, and a wrong guess only
# changes which slow variation lands on which beat, never the beat timing.
BEATS_PER_BAR = 4

# Used when a track has too few beats to measure an interval from.
FALLBACK_INTERVAL = 0.5


@dataclass(frozen=True)
class BeatPhase:
    """Where a playback time sits between two beats.

    ``phase`` runs 0 at the previous beat to 1 at the next, which is what lets
    movement be continuous instead of a decaying twitch after each beat.
    """

    index: int              # previous beat; -1 before the first
    phase: float            # 0.0 .. 1.0 through the current beat
    interval: float         # seconds between the surrounding beats
    previous_time: float
    next_time: float

    @property
    def bar_phase(self) -> float:
        """0.0 .. 1.0 through a bar, so slower motion can span several beats."""
        return ((self.index % BEATS_PER_BAR) + self.phase) / BEATS_PER_BAR

    @property
    def bar_index(self) -> int:
        return self.index // BEATS_PER_BAR


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

    @property
    def nominal_interval(self) -> float:
        """Typical seconds between beats, for extrapolating outside the beats."""
        if len(self._times) >= 2:
            return (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        bpm = getattr(self.features, "bpm", 0.0)
        return 60.0 / bpm if bpm else FALLBACK_INTERVAL

    def phase_at(self, time: float) -> BeatPhase:
        """Continuous rhythmic position at ``time``.

        Stateless like the rest of the timeline: the answer depends only on the
        time asked about. Before the first beat and after the last one the grid
        is extrapolated at the nominal interval, so the avatar keeps moving
        through an intro or an outro rather than standing still.
        """
        nominal = self.nominal_interval
        if not self._times:
            phase = (time / nominal) % 1.0 if nominal > 0 else 0.0
            return BeatPhase(
                index=-1,
                phase=phase,
                interval=nominal,
                previous_time=time - phase * nominal,
                next_time=time + (1.0 - phase) * nominal,
            )

        index = self.index_before(time)

        if index < 0:                       # before the first beat
            first = self._times[0]
            behind = (first - time) / nominal
            phase = (1.0 - (behind % 1.0)) % 1.0
            return BeatPhase(
                index=-1,
                phase=phase,
                interval=nominal,
                previous_time=time - phase * nominal,
                next_time=time + (1.0 - phase) * nominal,
            )

        previous = self._times[index]
        if index + 1 < len(self._times):
            following = self._times[index + 1]
        else:                               # past the last beat
            following = previous + nominal

        interval = following - previous
        if interval <= 0:
            interval = nominal
        phase = (time - previous) / interval
        if phase >= 1.0:                    # only reachable past the last beat
            whole = int(phase)
            index += whole
            previous += whole * interval
            following = previous + interval
            phase -= whole

        return BeatPhase(
            index=index,
            phase=min(max(phase, 0.0), 1.0),
            interval=interval,
            previous_time=previous,
            next_time=following,
        )
