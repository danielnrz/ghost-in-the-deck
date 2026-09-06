"""Two-deck planning primitives: determinism, directionality, honest labels.

Runs entirely on synthetic features - no private music, no Panda3D, no cloud.
Every property the phase promises is proven here by an assertion, not merely
exercised.
"""

from __future__ import annotations

import dataclasses
import inspect
import unittest

from ghost_in_the_deck import transition as T
from ghost_in_the_deck.animation.cues import BEATS_PER_BAR, BeatTimeline
from ghost_in_the_deck.animation.structure import MusicalStructure
from ghost_in_the_deck.deck import TrackDeck
from ghost_in_the_deck.transition import (
    DEFAULT_TRANSITION_LENGTH_BARS,
    EDGE_MARGIN_SECONDS,
    CuePoint,
    TransitionPlan,
    TwoDeckContext,
    candidate_cue_points,
    plan_transition,
    tempo_similarity,
)

from synthetic import make_features, regular_beats

# The only regime words the structure layer is ever allowed to emit. A "drop"
# or a "chorus" appearing anywhere would mean someone taught this code to guess
# song sections, which the phase forbids.
REGIME_VOCABULARY = {"build", "release", "peak", "stable"}
FORBIDDEN_SEMANTIC_LABELS = (
    "chorus",
    "verse",
    "drop",
    "breakdown",
    "bridge",
    "hook",
    "prechorus",
    "refrain",
)


def _ramp_then_plateau(t: float) -> float:
    # Climbs early, flattens later: the broad-structure slope - and therefore
    # every cue score - genuinely varies along the track.
    return min(0.30 + 0.020 * t, 0.90)


def _plateau_then_fall(t: float) -> float:
    return max(0.90 - 0.015 * max(t - 30.0, 0.0), 0.30)


def _features(track: str, bpm: float, duration: float, energy):
    interval = 60.0 / bpm
    beats = regular_beats(bpm=bpm, count=int(duration / interval))
    features = make_features(beats, duration=duration, bpm=bpm, energy=energy)
    return dataclasses.replace(features, track=track)


def _deck_a() -> TrackDeck:
    return TrackDeck.from_features(
        _features("deck-a", bpm=120.0, duration=96.0, energy=_ramp_then_plateau)
    )


def _deck_b() -> TrackDeck:
    return TrackDeck.from_features(
        _features("deck-b", bpm=124.0, duration=88.0, energy=_plateau_then_fall)
    )


def _twin_decks() -> tuple[TrackDeck, TrackDeck]:
    """Two decks identical in every measurable way but their track name.

    Lets a swap test isolate exactly the fields that follow deck order from the
    fields that follow the tracks' contents.
    """
    base = _features("deck-x", bpm=120.0, duration=96.0, energy=_ramp_then_plateau)
    return (
        TrackDeck.from_features(dataclasses.replace(base, track="deck-x")),
        TrackDeck.from_features(dataclasses.replace(base, track="deck-y")),
    )


class TwoDeckContextMeasurements(unittest.TestCase):
    def test_raw_measurements_match_hand_computation(self):
        a, b = _deck_a(), _deck_b()
        ctx = TwoDeckContext(a, b)
        self.assertAlmostEqual(ctx.bpm_ratio, b.bpm / a.bpm)
        self.assertAlmostEqual(ctx.bpm_difference, abs(b.bpm - a.bpm))
        self.assertEqual(ctx.duration_a, a.duration)
        self.assertEqual(ctx.duration_b, b.duration)
        self.assertAlmostEqual(
            ctx.bar_seconds_a, a.timeline.nominal_interval * BEATS_PER_BAR
        )
        self.assertAlmostEqual(
            ctx.bar_seconds_b, b.timeline.nominal_interval * BEATS_PER_BAR
        )

    def test_context_stores_no_derived_fields(self):
        # Only the two decks are fields; every measurement is a live property.
        field_names = {f.name for f in dataclasses.fields(TwoDeckContext)}
        self.assertEqual(field_names, {"deck_a", "deck_b"})

    def test_swapping_decks_moves_exactly_the_directional_measurements(self):
        a, b = _deck_a(), _deck_b()
        ab = TwoDeckContext(a, b)
        ba = TwoDeckContext(b, a)

        # Symmetric quantity: unchanged by order.
        self.assertAlmostEqual(ab.bpm_difference, ba.bpm_difference)

        # Directional quantities: reciprocal / swapped.
        self.assertAlmostEqual(ab.bpm_ratio, 1.0 / ba.bpm_ratio)
        self.assertEqual(ab.duration_a, ba.duration_b)
        self.assertEqual(ab.duration_b, ba.duration_a)
        self.assertAlmostEqual(ab.bar_seconds_a, ba.bar_seconds_b)
        self.assertAlmostEqual(ab.bar_seconds_b, ba.bar_seconds_a)


class CandidateCuePoints(unittest.TestCase):
    def test_candidates_are_bar_aligned_against_the_real_timeline(self):
        deck = _deck_a()
        timeline = deck.timeline
        cues = candidate_cue_points(deck)
        self.assertGreater(len(cues), 0)
        for cue in cues:
            # The time is exactly the bar's beat time on the real BeatTimeline.
            self.assertEqual(
                cue.time, timeline.beat_time(cue.bar_index * BEATS_PER_BAR)
            )
            phase = timeline.phase_at(cue.time)
            self.assertEqual(phase.bar_index, cue.bar_index)
            # Landing on a bar line means fractional bar phase ~ 0.
            self.assertLess(min(phase.bar_phase, 1.0 - phase.bar_phase), 1e-6)

    def test_every_candidate_time_is_inside_the_edge_margins(self):
        deck = _deck_a()
        margin = EDGE_MARGIN_SECONDS
        for cue in candidate_cue_points(deck):
            self.assertGreaterEqual(cue.time, margin)
            self.assertLessEqual(cue.time, deck.duration - margin)

    def test_custom_margin_is_honoured(self):
        deck = _deck_a()
        margin = 24.0
        for cue in candidate_cue_points(deck, margin_seconds=margin):
            self.assertGreaterEqual(cue.time, margin)
            self.assertLessEqual(cue.time, deck.duration - margin)

    def test_limit_caps_the_candidate_count(self):
        deck = _deck_a()
        self.assertLessEqual(len(candidate_cue_points(deck, limit=3)), 3)

    def test_score_is_one_minus_section_change_likelihood(self):
        deck = _deck_a()
        for cue in candidate_cue_points(deck):
            self.assertAlmostEqual(
                cue.score, 1.0 - cue.structure.section_change_likelihood
            )
            self.assertIsInstance(cue.structure, MusicalStructure)

    def test_candidates_are_sorted_by_score_then_bar_index(self):
        cues = candidate_cue_points(_deck_a(), limit=5)
        keys = [(-cue.score, cue.bar_index) for cue in cues]
        self.assertEqual(keys, sorted(keys))

    def test_candidate_set_is_deterministic(self):
        a = _deck_a()
        first = candidate_cue_points(a)
        second = candidate_cue_points(TrackDeck.from_features(a.features))
        self.assertEqual(first, second)

    def test_decks_a_and_b_yield_independent_candidate_sets(self):
        a, b = _deck_a(), _deck_b()
        cues_a = candidate_cue_points(a)
        cues_b = candidate_cue_points(b)
        # Different tracks -> different bar grids / scores.
        self.assertNotEqual(
            [(c.bar_index, c.score) for c in cues_a],
            [(c.bar_index, c.score) for c in cues_b],
        )
        # Building a's candidates again is unaffected by b existing.
        self.assertEqual(cues_a, candidate_cue_points(TrackDeck.from_features(a.features)))

    def test_no_candidates_when_margins_leave_no_room(self):
        short = TrackDeck.from_features(
            _features("tiny", bpm=120.0, duration=20.0, energy=_ramp_then_plateau)
        )
        self.assertEqual(candidate_cue_points(short), [])

    def test_every_candidate_leaves_room_for_a_full_transition(self):
        # A slow deck: 8 bars at bpm=60 is 32 s, far longer than the 16 s edge
        # margin. Every candidate must still leave that whole span after it.
        slow = TrackDeck.from_features(
            _features("slow", bpm=60.0, duration=140.0, energy=_ramp_then_plateau)
        )
        bar_seconds = slow.timeline.nominal_interval * BEATS_PER_BAR
        span = DEFAULT_TRANSITION_LENGTH_BARS * bar_seconds
        cues = candidate_cue_points(slow)
        self.assertGreater(len(cues), 0)
        for cue in cues:
            self.assertLessEqual(cue.time + span, slow.duration)

    def test_slow_short_deck_yields_no_candidates(self):
        # bpm=60, 40 s: an 8-bar (32 s) transition plus the 16 s lead-in does
        # not fit anywhere, so there is no honest cue to offer.
        slow_short = TrackDeck.from_features(
            _features("slow-short", bpm=60.0, duration=40.0, energy=_ramp_then_plateau)
        )
        self.assertEqual(candidate_cue_points(slow_short), [])


class TempoSimilarity(unittest.TestCase):
    def test_exact_match_scores_one(self):
        self.assertEqual(tempo_similarity(120.0, 120.0), 1.0)

    def test_score_is_exactly_symmetric_in_its_two_arguments(self):
        # The term feeds pair scoring, so it must not depend on which deck is A.
        # 100/120 and 120/100 are the same tempo gap and must score exactly (not
        # merely almost) the same.
        for bpm_a, bpm_b in ((100.0, 120.0), (128.0, 90.0), (120.0, 124.0), (174.0, 87.0)):
            self.assertEqual(
                tempo_similarity(bpm_a, bpm_b), tempo_similarity(bpm_b, bpm_a)
            )

    def test_score_is_min_over_max_of_the_two_tempos(self):
        self.assertAlmostEqual(tempo_similarity(100.0, 120.0), 100.0 / 120.0)
        self.assertAlmostEqual(tempo_similarity(120.0, 100.0), 100.0 / 120.0)
        # A wider gap scores lower, symmetrically.
        self.assertAlmostEqual(tempo_similarity(60.0, 120.0), 0.5)
        self.assertAlmostEqual(tempo_similarity(120.0, 60.0), 0.5)

    def test_non_positive_bpm_scores_zero(self):
        # An undefined tempo pairing (a deck analysed to bpm == 0).
        self.assertEqual(tempo_similarity(0.0, 120.0), 0.0)
        self.assertEqual(tempo_similarity(120.0, 0.0), 0.0)


class ThreeTempoQuantitiesHaveDistinctSymmetry(unittest.TestCase):
    """bpm_ratio is directional; bpm_difference and the scoring term are not."""

    def _pairs(self):
        for bpm_a, bpm_b in ((100.0, 120.0), (120.0, 100.0), (128.0, 90.0), (120.0, 124.0)):
            a = TrackDeck.from_features(
                _features("a", bpm=bpm_a, duration=96.0, energy=_ramp_then_plateau)
            )
            b = TrackDeck.from_features(
                _features("b", bpm=bpm_b, duration=96.0, energy=_plateau_then_fall)
            )
            yield bpm_a, bpm_b, TwoDeckContext(a, b), TwoDeckContext(b, a)

    def test_bpm_ratio_is_directional_and_inverts_under_swap(self):
        for _, _, ab, ba in self._pairs():
            self.assertNotAlmostEqual(ab.bpm_ratio, ba.bpm_ratio)
            self.assertAlmostEqual(ab.bpm_ratio, 1.0 / ba.bpm_ratio)

    def test_bpm_difference_is_symmetric_and_unchanged_under_swap(self):
        for bpm_a, bpm_b, ab, ba in self._pairs():
            self.assertEqual(ab.bpm_difference, ba.bpm_difference)
            self.assertAlmostEqual(ab.bpm_difference, abs(bpm_a - bpm_b))

    def test_tempo_similarity_term_is_exactly_symmetric_under_swap(self):
        for bpm_a, bpm_b, ab, ba in self._pairs():
            forward = tempo_similarity(ab.deck_a.bpm, ab.deck_b.bpm)
            reverse = tempo_similarity(ba.deck_a.bpm, ba.deck_b.bpm)
            self.assertEqual(forward, reverse)
            self.assertAlmostEqual(forward, min(bpm_a, bpm_b) / max(bpm_a, bpm_b))


class PlanTransition(unittest.TestCase):
    def test_plan_is_fully_deterministic(self):
        a, b = _deck_a(), _deck_b()
        one = plan_transition(TwoDeckContext(a, b))
        two = plan_transition(
            TwoDeckContext(
                TrackDeck.from_features(a.features),
                TrackDeck.from_features(b.features),
            )
        )
        self.assertEqual(one, two)
        self.assertEqual(one.reason, two.reason)

    def test_plan_times_are_bar_aligned_and_inside_both_margins(self):
        a, b = _deck_a(), _deck_b()
        plan = plan_transition(TwoDeckContext(a, b))
        self.assertIsNotNone(plan)

        self.assertEqual(
            plan.outgoing_time, a.timeline.beat_time(plan.outgoing_bar_index * BEATS_PER_BAR)
        )
        self.assertEqual(
            plan.incoming_time, b.timeline.beat_time(plan.incoming_bar_index * BEATS_PER_BAR)
        )
        self.assertGreaterEqual(plan.outgoing_time, EDGE_MARGIN_SECONDS)
        self.assertLessEqual(plan.outgoing_time, a.duration - EDGE_MARGIN_SECONDS)
        self.assertGreaterEqual(plan.incoming_time, EDGE_MARGIN_SECONDS)
        self.assertLessEqual(plan.incoming_time, b.duration - EDGE_MARGIN_SECONDS)

    def test_plan_records_only_measured_quantities(self):
        a, b = _deck_a(), _deck_b()
        ctx = TwoDeckContext(a, b)
        plan = plan_transition(ctx)

        self.assertEqual(plan.outgoing_track, a.track)
        self.assertEqual(plan.incoming_track, b.track)
        self.assertEqual(plan.bpm_a, a.bpm)
        self.assertEqual(plan.bpm_b, b.bpm)
        self.assertAlmostEqual(plan.bpm_ratio, ctx.bpm_ratio)
        # transition_length_bars is a planner/config policy value, not measured.
        self.assertEqual(plan.transition_length_bars, DEFAULT_TRANSITION_LENGTH_BARS)
        # The two seconds figures convert that bar count at each deck's own bar
        # length - there is no single shared "transition duration".
        self.assertAlmostEqual(
            plan.outgoing_duration_seconds,
            DEFAULT_TRANSITION_LENGTH_BARS * ctx.bar_seconds_a,
        )
        self.assertAlmostEqual(
            plan.incoming_duration_seconds,
            DEFAULT_TRANSITION_LENGTH_BARS * ctx.bar_seconds_b,
        )
        # Every field is a plain scalar or string - a pure data object.
        for value in dataclasses.astuple(plan):
            self.assertIsInstance(value, (str, int, float))

    def test_plan_score_matches_the_documented_combination(self):
        a, b = _deck_a(), _deck_b()
        ctx = TwoDeckContext(a, b)
        plan = plan_transition(ctx)

        outgoing = next(
            c for c in candidate_cue_points(a) if c.bar_index == plan.outgoing_bar_index
        )
        incoming = next(
            c for c in candidate_cue_points(b) if c.bar_index == plan.incoming_bar_index
        )
        tempo = tempo_similarity(a.bpm, b.bpm)
        expected = (outgoing.score + incoming.score + tempo) / 3.0
        self.assertAlmostEqual(plan.score, expected)

    def test_swapping_decks_swaps_exactly_the_directional_plan_fields(self):
        a, b = _deck_a(), _deck_b()
        ab = plan_transition(TwoDeckContext(a, b))
        ba = plan_transition(TwoDeckContext(b, a))

        self.assertEqual(ab.outgoing_track, ba.incoming_track)
        self.assertEqual(ab.incoming_track, ba.outgoing_track)
        self.assertEqual(ab.bpm_a, ba.bpm_b)
        self.assertEqual(ab.bpm_b, ba.bpm_a)
        self.assertAlmostEqual(ab.bpm_ratio, 1.0 / ba.bpm_ratio)
        # The policy bar count is not directional; the two seconds figures are,
        # and a swap turns the outgoing one into the incoming one and back.
        self.assertEqual(ab.transition_length_bars, ba.transition_length_bars)
        self.assertAlmostEqual(
            ab.outgoing_duration_seconds, ba.incoming_duration_seconds
        )
        self.assertAlmostEqual(
            ab.incoming_duration_seconds, ba.outgoing_duration_seconds
        )
        self.assertNotAlmostEqual(
            ab.outgoing_duration_seconds, ab.incoming_duration_seconds
        )

    def test_twin_decks_isolate_the_directional_fields(self):
        x, y = _twin_decks()
        xy = plan_transition(TwoDeckContext(x, y))
        yx = plan_transition(TwoDeckContext(y, x))

        # Only the track names follow deck order; the twins are otherwise equal.
        self.assertEqual(xy.outgoing_track, "deck-x")
        self.assertEqual(xy.incoming_track, "deck-y")
        self.assertEqual(yx.outgoing_track, "deck-y")
        self.assertEqual(yx.incoming_track, "deck-x")

        for field in (
            "outgoing_time",
            "incoming_time",
            "outgoing_bar_index",
            "incoming_bar_index",
            "bpm_a",
            "bpm_b",
            "bpm_ratio",
            "transition_length_bars",
            "outgoing_duration_seconds",
            "incoming_duration_seconds",
            "score",
        ):
            self.assertEqual(
                getattr(xy, field), getattr(yx, field), msg=f"{field} should not move"
            )

    def test_returns_none_rather_than_fabricating_a_cue(self):
        short = TrackDeck.from_features(
            _features("tiny", bpm=120.0, duration=20.0, energy=_ramp_then_plateau)
        )
        full = _deck_a()
        self.assertIsNone(plan_transition(TwoDeckContext(short, full)))
        self.assertIsNone(plan_transition(TwoDeckContext(full, short)))

    def test_planned_span_fits_on_both_decks(self):
        # The advertised transition span, taken from each cue, must end on or
        # before the end of the deck it runs on - never past it.
        pairs = [
            (_deck_a(), _deck_b()),
            (
                TrackDeck.from_features(
                    _features("slow-a", bpm=64.0, duration=150.0, energy=_ramp_then_plateau)
                ),
                TrackDeck.from_features(
                    _features("slow-b", bpm=68.0, duration=150.0, energy=_plateau_then_fall)
                ),
            ),
        ]
        for a, b in pairs:
            ctx = TwoDeckContext(a, b)
            plan = plan_transition(ctx)
            self.assertIsNotNone(plan)
            self.assertLessEqual(
                plan.outgoing_time + plan.outgoing_duration_seconds, a.duration
            )
            self.assertLessEqual(
                plan.incoming_time + plan.incoming_duration_seconds, b.duration
            )

    def test_different_bpm_decks_get_distinct_per_deck_transition_seconds(self):
        # 120 BPM out, 90 BPM in: 8 bars is 16.0 s on deck A and ~21.33 s on
        # deck B. The plan must carry both, each correct on its own deck.
        a = TrackDeck.from_features(
            _features("fast-out", bpm=120.0, duration=200.0, energy=_ramp_then_plateau)
        )
        b = TrackDeck.from_features(
            _features("slow-in", bpm=90.0, duration=200.0, energy=_plateau_then_fall)
        )
        ctx = TwoDeckContext(a, b)
        plan = plan_transition(ctx)
        self.assertIsNotNone(plan)
        self.assertNotAlmostEqual(
            plan.outgoing_duration_seconds, plan.incoming_duration_seconds
        )
        self.assertAlmostEqual(
            plan.outgoing_duration_seconds,
            plan.transition_length_bars * ctx.bar_seconds_a,
        )
        self.assertAlmostEqual(
            plan.incoming_duration_seconds,
            plan.transition_length_bars * ctx.bar_seconds_b,
        )
        self.assertAlmostEqual(plan.outgoing_duration_seconds, 16.0)
        self.assertAlmostEqual(plan.incoming_duration_seconds, 8 * 4 * 60.0 / 90.0)

    def test_slow_short_track_plans_nothing_instead_of_overrunning(self):
        # Regression: bpm=60 / 40 s once returned a plan whose 32 s span ran to
        # 48.5 s, well past the 40 s track. It must now decline instead.
        slow_short = _features(
            "slow-short", bpm=60.0, duration=40.0, energy=_ramp_then_plateau
        )
        deck = TrackDeck.from_features(slow_short)
        self.assertIsNone(plan_transition(TwoDeckContext(deck, _deck_b())))
        self.assertIsNone(plan_transition(TwoDeckContext(_deck_a(), deck)))

    def test_from_selection_does_not_rank_or_generate(self):
        a, b = _deck_a(), _deck_b()
        ctx = TwoDeckContext(a, b)
        outgoing = candidate_cue_points(a)[-1]
        incoming = candidate_cue_points(b)[-1]
        plan = TransitionPlan.from_selection(ctx, outgoing, incoming, 0.5, "hand picked")
        # It records the pair it was handed, verbatim.
        self.assertEqual(plan.outgoing_bar_index, outgoing.bar_index)
        self.assertEqual(plan.incoming_bar_index, incoming.bar_index)
        self.assertEqual(plan.score, 0.5)
        self.assertEqual(plan.reason, "hand picked")
        # The policy bar count is the default; each seconds figure is that count
        # at its own deck's bar length.
        self.assertEqual(plan.transition_length_bars, DEFAULT_TRANSITION_LENGTH_BARS)
        self.assertAlmostEqual(
            plan.outgoing_duration_seconds,
            DEFAULT_TRANSITION_LENGTH_BARS * ctx.bar_seconds_a,
        )
        self.assertAlmostEqual(
            plan.incoming_duration_seconds,
            DEFAULT_TRANSITION_LENGTH_BARS * ctx.bar_seconds_b,
        )


class BeatlessDeck(unittest.TestCase):
    """A silent or beatless track analyses to bpm=0 and zero beats."""

    def _deck(self) -> TrackDeck:
        features = make_features([], duration=60.0, bpm=0.0, energy=_ramp_then_plateau)
        return TrackDeck.from_features(dataclasses.replace(features, track="silent"))

    def test_bpm_ratio_is_defined_when_a_deck_has_no_tempo(self):
        silent = self._deck()
        self.assertEqual(TwoDeckContext(silent, _deck_b()).bpm_ratio, 0.0)
        # deck_b beatless is still a plain division, no crash.
        self.assertEqual(TwoDeckContext(_deck_a(), silent).bpm_ratio, 0.0)

    def test_plan_transition_declines_rather_than_raising(self):
        silent = self._deck()
        self.assertIsNone(plan_transition(TwoDeckContext(silent, _deck_b())))
        self.assertIsNone(plan_transition(TwoDeckContext(_deck_a(), silent)))


class NoInventedSemantics(unittest.TestCase):
    def _sample_times(self, duration: float):
        return [i * 1.0 for i in range(1, int(duration))]

    def test_regime_stays_within_the_allowed_vocabulary_end_to_end(self):
        for deck in (_deck_a(), _deck_b()):
            for t in self._sample_times(deck.duration):
                self.assertIn(deck.structure_at(t).regime, REGIME_VOCABULARY)
            for cue in candidate_cue_points(deck):
                self.assertIn(cue.structure.regime, REGIME_VOCABULARY)

    def test_plan_reason_carries_no_section_label(self):
        plan = plan_transition(TwoDeckContext(_deck_a(), _deck_b()))
        lowered = plan.reason.lower()
        for label in FORBIDDEN_SEMANTIC_LABELS:
            self.assertNotIn(label, lowered)

    def test_transition_module_source_has_no_section_detection(self):
        for module in (T, __import__("ghost_in_the_deck.deck", fromlist=["x"])):
            source = inspect.getsource(module).lower()
            for label in FORBIDDEN_SEMANTIC_LABELS:
                self.assertNotIn(label, source, msg=f"{label} in {module.__name__}")

    def test_transition_module_never_imports_the_single_track_layer(self):
        source = inspect.getsource(T)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("dj_planner", stripped)
                self.assertNotIn("dj_behavior", stripped)


if __name__ == "__main__":
    unittest.main()
