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

import math
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

    ``index`` numbers a position on the beat grid and keeps counting outside the
    detected beats, so the rhythm stays continuous through an intro or outro.
    That makes it unsafe to use on its own as "this is a real beat": a virtual
    index can be positive too, immediately after the last detected beat, so a
    check like ``index >= 0`` would call a virtual beat real. ``is_real`` is the
    explicit answer - true only when ``index`` names an actually detected beat.
    """

    index: int              # position on the grid; virtual before/after the detected beats
    is_real: bool           # True only when index names a detected beat, not a virtual one
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

    def beat_time(self, index: int) -> float:
        """Absolute playback time of beat ``index``, real or virtual.

        Uses the same nominal-interval extrapolation as ``phase_at`` outside the
        detected range, so a beat index and a time computed from it agree with
        what ``phase_at`` reports for that same beat. Used by anything that
        needs to name a time from a beat count - the gesture scheduler wants
        "the start of bar N" without duplicating this extrapolation itself.
        """
        if not self._times:
            return index * self.nominal_interval
        if 0 <= index < len(self._times):
            return self._times[index]
        if index < 0:
            return self._times[0] + index * self.nominal_interval
        return self._times[-1] + (index - (len(self._times) - 1)) * self.nominal_interval

    @property
    def nominal_interval(self) -> float:
        """Typical seconds between beats, for extrapolating outside the beats."""
        if len(self._times) >= 2:
            return (self._times[-1] - self._times[0]) / (len(self._times) - 1)
        bpm = getattr(self.features, "bpm", 0.0)
        return 60.0 / bpm if bpm else FALLBACK_INTERVAL

    def phase_at(self, time: float) -> BeatPhase:
        """Continuous rhythmic position at ``time``.

        Outside the detected beats the grid is extended at the nominal interval,
        so an intro or an outro still has a pulse to move to. Those virtual beats
        carry real indices - negative before the first detected beat, continuing
        upwards after the last - which is what keeps the bar continuous across
        them. Pinning the index at -1 through the whole intro, as this used to,
        made bar phase jump every time the virtual beat rolled over, and the
        whole body twitched with it.

        ``index`` is therefore only a beat number on the detected grid when
        ``is_real`` is true; outside that range it names a virtual beat used
        purely to keep the rhythm going. Callers that need to know whether a
        beat was actually detected read ``is_real``, not the sign of ``index``.

        Stateless like the rest of the timeline: the answer depends only on the
        time asked about.
        """
        nominal = self.nominal_interval

        if not self._times:
            position = time / nominal if nominal > 0 else 0.0
            index = math.floor(position)
            phase = position - index
            previous = index * nominal
            return BeatPhase(
                index=index,
                is_real=False,
                phase=phase,
                interval=nominal,
                previous_time=previous,
                next_time=previous + nominal,
            )

        index = self.index_before(time)

        if index < 0:
            # Before the first detected beat: continue the grid backwards.
            first = self._times[0]
            position = (time - first) / nominal
            step = math.floor(position)
            phase = position - step
            previous = first + step * nominal
            return BeatPhase(
                index=step,
                is_real=False,
                phase=phase,
                interval=nominal,
                previous_time=previous,
                next_time=previous + nominal,
            )

        previous = self._times[index]
        if index + 1 < len(self._times):
            following = self._times[index + 1]
        else:
            following = previous + nominal

        interval = following - previous
        if interval <= 0:
            interval = nominal
        position = (time - previous) / interval

        if position >= 1.0:
            # Past the last detected beat: continue the grid forwards.
            step = math.floor(position)
            index += step
            previous += step * interval
            following = previous + interval
            position -= step

        return BeatPhase(
            index=index,
            is_real=0 <= index < len(self._times),
            phase=min(max(position, 0.0), 1.0),
            interval=interval,
            previous_time=previous,
            next_time=following,
        )
