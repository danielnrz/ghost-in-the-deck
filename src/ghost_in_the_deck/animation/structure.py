"""A lightweight, deterministic musical-structure layer.

This adds nothing to the audio analysis - no new librosa pass, no schema bump.
It reads the *same* smoothed-energy curve and beat grid the groove already uses,
just at a broader timescale, and turns that into a few honest, non-semantic
signals:

* ``broad_energy`` / ``broad_trend`` - the energy level and its slope seen over
  an eight-second window instead of the groove's ~1.5 s and the DJ scheduler's
  3 s, so a whole build or release registers as one movement rather than as a
  string of bar-scale wobbles.
* ``regime`` - one of ``build`` / ``release`` / ``peak`` / ``stable``. These are
  descriptions of what the broad energy is doing right now, not section labels:
  nothing here claims to know a "drop" from a "breakdown".
* ``section_change_likelihood`` - a continuous 0..1 reading of how fast the
  broad energy is moving, as a stand-in for "something structural is probably
  happening around here". Not a detector, not a boolean.
* ``phrase_position`` - ``bar_index`` folded into an eight-bar cycle. This is a
  *convention* asserted by ``PHRASE_LENGTH_BARS``, not detected phrasing: real
  music does not always phrase in eights and this code never measures whether
  it does. It is computed and tested but deliberately wired into no decision.

Panda3D-free, and it does not import ``dj_planner`` - the dependency runs one
way only, exactly as ``dj_behavior`` -> ``dj_planner`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import dj_behavior
from .cues import BeatTimeline
from .dj_behavior import trend_at
from .energy import EnergyTrack

# Smoothing window for the broad EnergyTrack, in seconds. Several times the
# groove's SMOOTHING_SECONDS (1.5) so a single loud bar cannot move it.
PHRASE_SMOOTHING_SECONDS = 8.0

# How far back the broad trend looks, in seconds. Matches the broad smoothing
# window: reading a slope over less time than the curve was smoothed over would
# just re-expose the noise the smoothing removed.
PHRASE_TREND_LOOKBACK = 8.0

# How much the broad energy must climb / fall over PHRASE_TREND_LOOKBACK to
# count as a build / a release. Independently named on purpose: it happens to
# equal neither dj_behavior.TREND_RISING nor any groove constant, and it should
# be free to move without dragging those with it.
BUILD_SLOPE = 0.15
RELEASE_SLOPE = 0.15

# Divisor mapping the absolute broad slope to section_change_likelihood: a broad
# move of this size over the lookback reads as "almost certainly something
# structural here" (likelihood 1.0).
SECTION_CHANGE_SCALE = 0.3

# Bars per phrase, as an assumed convention. NOT detected - see the module
# docstring. phrase_position is bar_index modulo this.
PHRASE_LENGTH_BARS = 8


@dataclass(frozen=True)
class MusicalStructure:
    """The structure layer's reading at one absolute playback time.

    A pure function of ``time`` and the two analysed inputs - the same inputs
    twice give an identical instance.
    """

    time: float
    bar_index: int
    phrase_position: int          # bar_index % PHRASE_LENGTH_BARS - an assumed cycle, not detected
    broad_energy: float           # smoothed energy over PHRASE_SMOOTHING_SECONDS, 0..1
    broad_trend: float            # broad_energy now minus PHRASE_TREND_LOOKBACK seconds ago
    regime: str                   # "build" / "release" / "peak" / "stable"
    section_change_likelihood: float  # 0..1, min(abs(broad_trend) / SECTION_CHANGE_SCALE, 1.0)


def structure_at(
    broad_energy: EnergyTrack, timeline: BeatTimeline, time: float
) -> MusicalStructure:
    """The musical-structure reading at ``time``.

    ``broad_energy`` must be an ``EnergyTrack`` built with
    ``smoothing_seconds=PHRASE_SMOOTHING_SECONDS`` - this function does not build
    it, so a caller can share one broad track across many lookups.
    """
    bar_index = timeline.phase_at(time).bar_index
    phrase_position = bar_index % PHRASE_LENGTH_BARS

    value = broad_energy.at(time)
    broad_trend = trend_at(broad_energy, time, PHRASE_TREND_LOOKBACK)

    if broad_trend >= BUILD_SLOPE:
        regime = "build"
    elif broad_trend <= -RELEASE_SLOPE:
        regime = "release"
    elif value >= dj_behavior.HIGH_ENERGY:
        regime = "peak"
    else:
        regime = "stable"

    section_change_likelihood = min(abs(broad_trend) / SECTION_CHANGE_SCALE, 1.0)

    return MusicalStructure(
        time=time,
        bar_index=bar_index,
        phrase_position=phrase_position,
        broad_energy=value,
        broad_trend=broad_trend,
        regime=regime,
        section_change_likelihood=section_change_likelihood,
    )
