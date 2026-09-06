"""Two-deck planning primitives - raw measurable quantities only.

This module answers the two-track question "how could deck A move into deck B"
and is kept deliberately separate from the single-track decision layer: it never
imports ``dj_planner`` or ``dj_behavior``. It is Panda3D-free.

``TwoDeckContext`` pairs two ``TrackDeck`` objects and exposes only raw,
deterministic measurements derived live from the two decks - tempo ratio,
tempo gap, durations and bar lengths. Nothing here is an interpretation.

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
# track, so a transition built on it has room to breathe on both decks. At the
# tail this is only a floor: ``candidate_cue_points`` reserves the larger of
# this margin and the planned transition span, so the whole transition still
# fits on the deck after the cue even on a slow or short track.
EDGE_MARGIN_SECONDS = 16.0

# Upper bound on how many cue points one deck contributes to planning.
MAX_CANDIDATES_PER_DECK = 5

# Default span of a planned transition, in whole bars. This is a planner/config
# policy value - "how many bars this planner chooses to blend across" - not a
# measured property of either track; a caller may override it via the
# ``transition_bars`` keyword on ``candidate_cue_points``,
# ``TransitionPlan.from_selection`` and ``plan_transition``. 8 bars is the
# current documented default. Named independently of ``structure.PHRASE_LENGTH_BARS``:
# the two constants currently share a value but answer different questions -
# "how many bars this planner blends across" versus "the length of the assumed
# phrase cycle" - so neither should track the other.
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
        """``deck_b`` tempo as a multiple of ``deck_a`` tempo.

        ``0.0`` when ``deck_a`` has no measured tempo (a silent or beatless
        track analyses to ``bpm == 0``): the ratio is undefined, and a caller
        that needs a real tempo pairing should check ``bpm`` first.
        """
        if self.deck_a.bpm <= 0.0:
            return 0.0
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
    transition_bars: int = DEFAULT_TRANSITION_LENGTH_BARS,
) -> list[CuePoint]:
    """The most stable whole-bar cue points on ``deck``, best first.

    Walks the bar grid via ``timeline.beat_time(bar * BEATS_PER_BAR)``, skips any
    bar within ``margin_seconds`` of the start, skips any bar that does not leave
    room after it for a ``transition_bars``-bar transition at this deck's bar
    length (or ``margin_seconds``, whichever is larger), scores each remaining
    bar as ``1.0 - structure.section_change_likelihood``, and returns the top
    ``limit`` by score. Ties break by earliest ``bar_index``. Fully
    deterministic: no hashing, no randomness. Returns ``[]`` when the two edge
    constraints leave no whole bar in between - a slow or short track simply has
    nowhere a full transition fits.
    """
    timeline = deck.timeline
    duration = deck.duration

    bar_seconds = timeline.nominal_interval * BEATS_PER_BAR
    # The tail must hold the whole planned transition, not just the edge margin:
    # on a slow track ``transition_bars`` bars is much longer than the margin.
    tail_room = max(margin_seconds, transition_bars * bar_seconds)
    latest = duration - tail_room

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
    audio processing, no crossfade and no EQ - it only records quantities taken
    from the two decks and the two chosen cue points.

    Every field except ``transition_length_bars`` is a genuine measurement of one
    of the two tracks (or a plain arithmetic combination of measurements).
    ``transition_length_bars`` is the exception: it is a planner/config policy
    value - the number of bars this planner chose to blend across, defaulting to
    ``DEFAULT_TRANSITION_LENGTH_BARS`` (currently 8) - not a property measured
    from either track. The two seconds figures derived from it,
    ``outgoing_duration_seconds`` and ``incoming_duration_seconds``, are that
    policy bar count converted at deck A's and deck B's own bar lengths
    respectively: with no time-stretching anywhere in this project, the same bar
    count is not the same number of seconds on two decks at different tempos, so
    there is deliberately no single shared "transition duration" field.
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
    transition_length_bars: int
    outgoing_duration_seconds: float
    incoming_duration_seconds: float
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
        *,
        transition_bars: int = DEFAULT_TRANSITION_LENGTH_BARS,
    ) -> "TransitionPlan":
        """Assemble a plan from an already-picked pair of cue points.

        ``outgoing`` is a cue point on ``context.deck_a`` (the track being mixed
        out); ``incoming`` is a cue point on ``context.deck_b`` (the track being
        mixed in). ``score`` and ``reason`` are supplied by the caller that chose
        this pair - this constructor neither generates candidates nor ranks
        them. ``transition_length_bars`` is ``transition_bars`` (default
        ``DEFAULT_TRANSITION_LENGTH_BARS``) - a planner/config policy value, not
        a measurement of either track - and is converted to seconds twice, once
        at ``deck_a``'s bar length (``outgoing_duration_seconds``) and once at
        ``deck_b``'s (``incoming_duration_seconds``), because the same bar count
        is not the same number of seconds on two decks at different tempos.
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
            transition_length_bars=transition_bars,
            outgoing_duration_seconds=transition_bars * context.bar_seconds_a,
            incoming_duration_seconds=transition_bars * context.bar_seconds_b,
            score=score,
            reason=reason,
        )


# Weights behind a pair's combined score: the outgoing cue's own stability
# score, the incoming cue's own stability score, and the tempo-similarity term
# from ``bpm_ratio``. Equal weights - a plain mean - is the documented default;
# nothing here is tuned, fitted or learned. Change the tuple to reweight.
PAIR_SCORE_WEIGHTS = (1.0, 1.0, 1.0)


def tempo_similarity(bpm_a: float, bpm_b: float) -> float:
    """How close two decks' tempos are, as a symmetric 0..1 score.

    ``min(bpm_a, bpm_b) / max(bpm_a, bpm_b)`` - equivalently ``min(r, 1 / r)``
    for the tempo ratio ``r`` - so it is ``1.0`` at an exact tempo match and
    falls off the same way whichever deck is the faster one. Because ``min`` and
    ``max`` do not depend on argument order, ``tempo_similarity(x, y)`` and
    ``tempo_similarity(y, x)`` compute a byte-identical float; it is taken over
    the two raw BPMs rather than over ``context.bpm_ratio`` precisely so that
    swapping the decks changes nothing (``b / a`` and ``a / b`` are not exact
    floating-point reciprocals, so a single-ratio form would differ in the last
    bit under a swap). Returns ``0.0`` when either BPM is non-positive - an
    undefined tempo pairing, e.g. a deck analysed to ``bpm == 0``.

    This is a plain tempo-gap term for ranking cue pairs only; it deliberately
    does not treat a half-time or double-time pairing as "similar" - beat
    matching belongs to a later phase.
    """
    hi = max(bpm_a, bpm_b)
    lo = min(bpm_a, bpm_b)
    if lo <= 0.0:
        return 0.0
    return lo / hi


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
    transition_bars: int = DEFAULT_TRANSITION_LENGTH_BARS,
) -> "TransitionPlan | None":
    """Pick the single best outgoing/incoming cue pair for ``context``.

    Generates ``candidate_cue_points`` for ``context.deck_a`` (outgoing) and
    ``context.deck_b`` (incoming) - each already requiring room after the cue for
    the full ``transition_bars``-bar span - scores every ``(outgoing, incoming)``
    pair as ``_combined_score`` of the outgoing cue's score, the incoming cue's
    score and ``tempo_similarity(deck_a.bpm, deck_b.bpm)`` (symmetric under a
    deck swap), and returns the top pair as a ``TransitionPlan``. Ties break
    deterministically: earliest
    ``outgoing.time``, then earliest ``incoming.time``.

    Returns ``None`` when either deck yields zero candidates - including when a
    deck is too slow or too short for a full transition to fit after any cue, or
    when a deck analysed to ``bpm == 0`` (silent or beatless) so there is no
    tempo to plan a beat-matched move against. It never fabricates a cue point
    to force a plan. Fully deterministic: the same two ``TrackDeck`` objects give
    a byte-identical result every call.
    """
    if context.deck_a.bpm <= 0.0 or context.deck_b.bpm <= 0.0:
        return None

    outgoing_candidates = candidate_cue_points(
        context.deck_a,
        margin_seconds=margin_seconds,
        limit=limit_per_deck,
        transition_bars=transition_bars,
    )
    incoming_candidates = candidate_cue_points(
        context.deck_b,
        margin_seconds=margin_seconds,
        limit=limit_per_deck,
        transition_bars=transition_bars,
    )
    if not outgoing_candidates or not incoming_candidates:
        return None

    tempo_score = tempo_similarity(context.deck_a.bpm, context.deck_b.bpm)

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
        f"tempo_similarity {tempo_score:.4f} from bpm {context.deck_a.bpm:.4f}/"
        f"{context.deck_b.bpm:.4f}) "
        f"weights {PAIR_SCORE_WEIGHTS}"
    )
    return TransitionPlan.from_selection(
        context, outgoing, incoming, score, reason, transition_bars=transition_bars
    )
