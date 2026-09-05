"""Two-deck planning primitives - raw measurable quantities only.

This module answers the two-track question "how could deck A move into deck B"
and is kept deliberately separate from the single-track decision layer: it never
imports ``dj_planner`` or ``dj_behavior``. It is Panda3D-free.

``TwoDeckContext`` pairs two ``TrackDeck`` objects and exposes only raw,
deterministic measurements derived live from the two decks - no boolean
"these tracks are compatible" verdict, no key or harmonic guessing, no
section detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .animation.cues import BEATS_PER_BAR
from .animation.structure import MusicalStructure
from .deck import TrackDeck

# A candidate cue point must sit at least this far from either edge of the
# track, so a transition built on it has room to breathe on both decks.
EDGE_MARGIN_SECONDS = 16.0

# Upper bound on how many cue points one deck contributes to planning.
MAX_CANDIDATES_PER_DECK = 5


@dataclass(frozen=True)
class TwoDeckContext:
    """Two analysed tracks placed side by side for transition planning.

    Every property is computed live from ``deck_a`` and ``deck_b``; none is a
    separately stored field. The measurements are direction-sensitive where the
    quantity itself is (``bpm_ratio``) and symmetric where it is
    (``bpm_difference``).
    """

    deck_a: TrackDeck
    deck_b: TrackDeck

    @property
    def bpm_ratio(self) -> float:
        """``deck_b`` tempo as a multiple of ``deck_a`` tempo."""
        return self.deck_b.bpm / self.deck_a.bpm

    @property
    def bpm_difference(self) -> float:
        """Absolute tempo gap in BPM, independent of deck order."""
        return abs(self.deck_b.bpm - self.deck_a.bpm)

    @property
    def duration_a(self) -> float:
        return self.deck_a.duration

    @property
    def duration_b(self) -> float:
        return self.deck_b.duration

    @property
    def bar_seconds_a(self) -> float:
        """Seconds per bar on ``deck_a`` at its nominal beat interval."""
        return self.deck_a.timeline.nominal_interval * BEATS_PER_BAR

    @property
    def bar_seconds_b(self) -> float:
        """Seconds per bar on ``deck_b`` at its nominal beat interval."""
        return self.deck_b.timeline.nominal_interval * BEATS_PER_BAR


@dataclass(frozen=True)
class CuePoint:
    """One whole-bar position on a deck, scored once for transition planning.

    ``score`` is precomputed at construction (``1.0 - section_change_likelihood``)
    so that selecting between cue points is a pure comparison - higher is a
    calmer, more stable place to move through. It carries no semantic verdict.
    """

    time: float
    bar_index: int
    structure: MusicalStructure
    score: float


def candidate_cue_points(
    deck: TrackDeck,
    *,
    margin_seconds: float = EDGE_MARGIN_SECONDS,
    limit: int = MAX_CANDIDATES_PER_DECK,
) -> list[CuePoint]:
    """The most stable whole-bar cue points on ``deck``, best first.

    Walks the bar grid via ``timeline.beat_time(bar * BEATS_PER_BAR)``, skips any
    bar within ``margin_seconds`` of the start or of ``deck.duration``, scores
    each remaining bar as ``1.0 - structure.section_change_likelihood``, and
    returns the top ``limit`` by score. Ties break by earliest ``bar_index``.
    Fully deterministic: no hashing, no randomness.
    """
    timeline = deck.timeline
    duration = deck.duration
    latest = duration - margin_seconds

    bar_seconds = timeline.nominal_interval * BEATS_PER_BAR
    max_bars = int(duration / bar_seconds) + 2 if bar_seconds > 0 else 0

    candidates: list[CuePoint] = []
    for bar in range(max_bars):
        time = timeline.beat_time(bar * BEATS_PER_BAR)
        if time > latest:
            break
        if time >= margin_seconds:
            structure = deck.structure_at(time)
            score = 1.0 - structure.section_change_likelihood
            candidates.append(CuePoint(time, bar, structure, score))

    candidates.sort(key=lambda cue: (-cue.score, cue.bar_index))
    return candidates[:limit]
