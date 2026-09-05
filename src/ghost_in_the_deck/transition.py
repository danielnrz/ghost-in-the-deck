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
from .deck import TrackDeck


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
