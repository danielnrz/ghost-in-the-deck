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
    tempo_closeness,
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


class TempoCloseness(unittest.TestCase):
    def test_exact_match_scores_one_and_is_direction_sensitive(self):
        self.assertEqual(tempo_closeness(1.0), 1.0)
        # r and 1/r generally disagree because bpm_ratio itself is directional.
        self.assertNotAlmostEqual(tempo_closeness(1.5), tempo_closeness(1 / 1.5))
        # Below an exact match the score degrades linearly...
        self.assertAlmostEqual(tempo_closeness(0.5), 0.5)
        # ...while a whole multiple or more above it floors at zero.
        self.assertEqual(tempo_closeness(2.0), 0.0)
        self.assertEqual(tempo_closeness(3.0), 0.0)


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
        self.assertEqual(plan.expected_duration_bars, DEFAULT_TRANSITION_LENGTH_BARS)
        self.assertAlmostEqual(
            plan.expected_duration_seconds,
            DEFAULT_TRANSITION_LENGTH_BARS * ctx.bar_seconds_a,
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
        tempo = tempo_closeness(ctx.bpm_ratio)
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
        # The transition span is measured on whichever deck is outgoing.
        self.assertEqual(ab.expected_duration_bars, ba.expected_duration_bars)
        self.assertNotAlmostEqual(
            ab.expected_duration_seconds, ba.expected_duration_seconds
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
            "expected_duration_bars",
            "expected_duration_seconds",
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
