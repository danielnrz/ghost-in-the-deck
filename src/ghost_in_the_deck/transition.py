"""Two-deck planning primitives - raw measurable quantities only.

This module answers the two-track question "how could deck A move into deck B"
and is kept deliberately separate from the single-track decision layer: it never
imports ``dj_planner`` or ``dj_behavior``. It is Panda3D-free.

``TwoDeckContext`` pairs two ``TrackDeck`` objects and exposes only raw,
deterministic measurements derived live from the two decks - no boolean
"these tracks are compatible" verdict, no key or harmonic guessing, no
section detection.

``TransitionPlan`` is a frozen data object recording one already-chosen pair of
cue points plus the measured quantities around it. Building one does no audio
processing, loads no sound and touches no effects code.
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

# Default span of a planned transition, in whole bars. Named independently of
# ``structure.PHRASE_LENGTH_BARS``: the two constants currently share a value but
# answer different questions - "how many bars to blend across" versus "the
# length of the assumed phrase cycle" - so neither should track the other.
DEFAULT_TRANSITION_LENGTH_BARS = 8


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


@dataclass(frozen=True)
class TransitionPlan:
    """One already-chosen way for ``deck_a`` to move into ``deck_b``.

    A pure data object: constructing or holding a ``TransitionPlan`` performs no
    audio processing, no crossfade and no EQ - it only records measured
    quantities taken from the two decks and the two chosen cue points. Every
    field traces back to a measurement or to a named module constant; nothing
    here is a compatibility verdict or a guessed song section.
    """

    outgoing_track: str
    incoming_track: str
    outgoing_time: float
    incoming_time: float
    outgoing_bar_index: int
    incoming_bar_index: int
    bpm_a: float
    bpm_b: float
    bpm_ratio: float
    expected_duration_bars: int
    expected_duration_seconds: float
    score: float
    reason: str

    @classmethod
    def from_selection(
        cls,
        context: TwoDeckContext,
        outgoing: CuePoint,
        incoming: CuePoint,
        score: float,
        reason: str,
    ) -> "TransitionPlan":
        """Assemble a plan from an already-picked pair of cue points.

        ``outgoing`` is a cue point on ``context.deck_a`` (the track being mixed
        out); ``incoming`` is a cue point on ``context.deck_b`` (the track being
        mixed in). ``score`` and ``reason`` are supplied by the caller that chose
        this pair - this constructor neither generates candidates nor ranks
        them. ``expected_duration_bars`` is fixed at
        ``DEFAULT_TRANSITION_LENGTH_BARS`` and converted to seconds at
        ``deck_a``'s bar length.
        """
        return cls(
            outgoing_track=context.deck_a.track,
            incoming_track=context.deck_b.track,
            outgoing_time=outgoing.time,
            incoming_time=incoming.time,
            outgoing_bar_index=outgoing.bar_index,
            incoming_bar_index=incoming.bar_index,
            bpm_a=context.deck_a.bpm,
            bpm_b=context.deck_b.bpm,
            bpm_ratio=context.bpm_ratio,
            expected_duration_bars=DEFAULT_TRANSITION_LENGTH_BARS,
            expected_duration_seconds=(
                DEFAULT_TRANSITION_LENGTH_BARS * context.bar_seconds_a
            ),
            score=score,
            reason=reason,
        )


# Weights behind a pair's combined score: the outgoing cue's own stability
# score, the incoming cue's own stability score, and the tempo-closeness term
# from ``bpm_ratio``. Equal weights - a plain mean - is the documented default;
# nothing here is tuned, fitted or learned. Change the tuple to reweight.
PAIR_SCORE_WEIGHTS = (1.0, 1.0, 1.0)


def tempo_closeness(bpm_ratio: float) -> float:
    """How close two decks' tempos are, as a 0..1 score - 1.0 at an exact match.

    ``1.0 - min(abs(bpm_ratio - 1.0), 1.0)``: a half- or double-speed pairing
    (ratio 0.5 or 2.0) still scores 0.5, and anything a whole multiple or more
    apart floors at 0.0. Direction-sensitive because ``bpm_ratio`` is: the score
    for ``r`` and for ``1 / r`` are generally different numbers.
    """
    return 1.0 - min(abs(bpm_ratio - 1.0), 1.0)


def _combined_score(
    outgoing_score: float, incoming_score: float, tempo_score: float
) -> float:
    """Fold the three measured terms into one scalar via ``PAIR_SCORE_WEIGHTS``.

    A weighted mean, evaluated in a fixed term order so the same three inputs
    always produce a byte-identical float.
    """
    w_out, w_in, w_tempo = PAIR_SCORE_WEIGHTS
    weighted = w_out * outgoing_score + w_in * incoming_score + w_tempo * tempo_score
    return weighted / (w_out + w_in + w_tempo)


def plan_transition(
    context: TwoDeckContext,
    *,
    margin_seconds: float = EDGE_MARGIN_SECONDS,
    limit_per_deck: int = MAX_CANDIDATES_PER_DECK,
) -> "TransitionPlan | None":
    """Pick the single best outgoing/incoming cue pair for ``context``.

    Generates ``candidate_cue_points`` for ``context.deck_a`` (outgoing) and
    ``context.deck_b`` (incoming), scores every ``(outgoing, incoming)`` pair as
    ``_combined_score`` of the outgoing cue's score, the incoming cue's score and
    ``tempo_closeness(context.bpm_ratio)``, and returns the top pair as a
    ``TransitionPlan``. Ties break deterministically: earliest ``outgoing.time``,
    then earliest ``incoming.time``.

    Returns ``None`` when either deck yields zero candidates - it never
    fabricates a cue point to force a plan. Fully deterministic: the same two
    ``TrackDeck`` objects give a byte-identical result every call.
    """
    outgoing_candidates = candidate_cue_points(
        context.deck_a, margin_seconds=margin_seconds, limit=limit_per_deck
    )
    incoming_candidates = candidate_cue_points(
        context.deck_b, margin_seconds=margin_seconds, limit=limit_per_deck
    )
    if not outgoing_candidates or not incoming_candidates:
        return None

    tempo_score = tempo_closeness(context.bpm_ratio)

    best: tuple[CuePoint, CuePoint, float] | None = None
    best_key: tuple[float, float, float] | None = None
    for outgoing in outgoing_candidates:
        for incoming in incoming_candidates:
            score = _combined_score(outgoing.score, incoming.score, tempo_score)
            key = (-score, outgoing.time, incoming.time)
            if best_key is None or key < best_key:
                best_key = key
                best = (outgoing, incoming, score)

    assert best is not None  # both candidate lists are non-empty
    outgoing, incoming, score = best
    reason = (
        f"combined {score:.4f} = weighted mean("
        f"outgoing {outgoing.score:.4f}, incoming {incoming.score:.4f}, "
        f"tempo {tempo_score:.4f} from bpm_ratio {context.bpm_ratio:.4f}) "
        f"weights {PAIR_SCORE_WEIGHTS}"
    )
    return TransitionPlan.from_selection(context, outgoing, incoming, score, reason)
