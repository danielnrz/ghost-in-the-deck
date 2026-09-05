"""Phase 2A: the DJ action planner.

Proves that the audio action chosen at a scheduled moment depends only on the
music's own energy there and the planner's own seed - never on the candidate
``GestureEvent``'s ``kind``, ``side`` or ``strength`` - and that its notion of
"energy band" and "trend" is the same one ``DJBehaviorEngine`` already uses,
not a re-derivation that could drift.

Runs entirely on synthetic features; no private music.
"""

from __future__ import annotations

import unittest

from ghost_in_the_deck.animation.dj_behavior import (
    HIGH_ENERGY,
    LOW_ENERGY,
    TREND_RISING,
    GestureEvent,
    trend_at,
)
from ghost_in_the_deck.animation.energy import EnergyTrack
from ghost_in_the_deck.dj_planner import (
    ACTIVATION_BASE,
    ACTIVATION_TREND_BOOST,
    AUDIO_ACTIONS,
    PLANNER_STRENGTH_FLOOR,
    DJActionPlanner,
    MusicalContext,
    _band_for,
    activation_probability,
    context_at,
    decide_action,
)

from synthetic import behavior_for, make_features, regular_beats


def _energy(energy, duration=120.0):
    features = make_features(
        regular_beats(bpm=120.0, count=int(duration / 0.5)),
        duration=duration,
        energy=energy,
    )
    return EnergyTrack(features)


# A flat input normalises to the mid band; a low/high step gives real bands.
MID = lambda: _energy(0.8)
STEP = lambda: _energy(lambda t: 0.05 if t < 40.0 else 0.9)


class TestContextMatchesDJBehaviour(unittest.TestCase):
    """context_at reads the same energy/trend maths DJBehaviorEngine does."""

    def test_energy_and_trend_agree_with_the_engine(self):
        engine = behavior_for(
            regular_beats(bpm=124.0, count=360, offset=0.5),
            duration=180.0,
            bpm=124.0,
            seed="ctx",
            energy=lambda t: 0.1 + 0.8 * t / 180.0,
        )
        for t in (0.0, 2.0, 5.5, 40.2, 90.0, 179.0):
            context = context_at(engine.energy, t)
            with self.subTest(t=t):
                self.assertEqual(context.energy, engine.energy.at(t))
                self.assertEqual(context.trend, engine._trend(t))
                self.assertEqual(context.trend, trend_at(engine.energy, t))

    def test_context_is_always_marked_on_the_bar_grid_this_phase(self):
        context = context_at(MID(), 12.0)
        self.assertTrue(context.at_bar_boundary)


class TestEnergyBandBoundaries(unittest.TestCase):
    """The band edges are dj_behavior's own constants, not restated literals."""

    def test_boundaries_are_exactly_low_and_high_energy(self):
        self.assertEqual(_band_for(LOW_ENERGY - 1e-9), "low")
        self.assertEqual(_band_for(LOW_ENERGY), "mid")
        self.assertEqual(_band_for(HIGH_ENERGY - 1e-9), "mid")
        self.assertEqual(_band_for(HIGH_ENERGY), "high")

    def test_decide_action_maps_each_band(self):
        def ctx(band):
            return MusicalContext(0.0, 0.5, 0.0, band, True)

        self.assertEqual(decide_action(ctx("high")), "gain_riser")
        self.assertEqual(decide_action(ctx("mid")), "filter_sweep")
        self.assertEqual(decide_action(ctx("low")), "none")
        for action in ("gain_riser", "filter_sweep", "none"):
            self.assertIn(action, AUDIO_ACTIONS)


class TestInversion(unittest.TestCase):
    """The core claim: an audio decision ignores the candidate event's content."""

    def test_two_events_sharing_only_timing_get_identical_actions(self):
        planner = DJActionPlanner("invert")
        energy = MID()

        quiet_glance = GestureEvent(30.0, 1.4, "deck_glance", None, 0.61)
        loud_reach = GestureEvent(30.0, 1.4, "hand_to_deck", "l", 0.98)

        [a] = planner.plan([quiet_glance], energy)
        [b] = planner.plan([loud_reach], energy)

        self.assertEqual(a, b)
        self.assertEqual((a.action, a.side, a.strength), (b.action, b.side, b.strength))

    def test_changing_only_the_events_strength_does_not_move_planned_strength(self):
        planner = DJActionPlanner("strength")
        energy = MID()
        weak = GestureEvent(18.0, 0.8, "small_hype", "l", 0.05)
        strong = GestureEvent(18.0, 0.8, "small_hype", "l", 1.0)

        [a] = planner.plan([weak], energy)
        [b] = planner.plan([strong], energy)

        self.assertEqual(a.strength, b.strength)
        expected = PLANNER_STRENGTH_FLOOR + (1.0 - PLANNER_STRENGTH_FLOOR) * energy.at(18.0)
        self.assertAlmostEqual(a.strength, expected, places=12)

    def test_changing_only_the_events_side_does_not_move_planned_side(self):
        planner = DJActionPlanner("side-indep")
        energy = MID()   # mid band -> filter_sweep, which needs a side
        sides = set()
        for event_side in ("l", "r", None):
            event = GestureEvent(9.0, 1.0, "hand_to_deck", event_side, 0.7)
            [planned] = planner.plan([event], energy)
            self.assertEqual(planned.action, "filter_sweep")
            sides.add(planned.side)
        self.assertEqual(len(sides), 1, "planner side followed the candidate event's side")


class TestFilterSweepSide(unittest.TestCase):
    """The sweep side is a real deterministic function of start and seed."""

    def test_same_start_and_seed_give_the_same_side(self):
        planner = DJActionPlanner("stable")
        self.assertEqual(planner._side_for(3.0), planner._side_for(3.0))

    def test_different_seeds_can_disagree_for_the_same_start(self):
        one = {DJActionPlanner(f"seed-{i}")._side_for(5.0) for i in range(40)}
        self.assertEqual(one, {"l", "r"})

    def test_side_is_not_constant_across_starts(self):
        planner = DJActionPlanner("spread")
        seen = {planner._side_for(float(i) + 0.25) for i in range(1, 60)}
        self.assertEqual(seen, {"l", "r"}, "every start hashed to the same side")

    def test_plan_side_is_deterministic_run_to_run(self):
        events = [GestureEvent(t, 1.0, "lean_in", None, 0.5) for t in (12.0, 20.0, 33.0)]
        first = DJActionPlanner("repeat").plan(events, MID())
        second = DJActionPlanner("repeat").plan(events, MID())
        self.assertEqual(first, second)


class TestPlanShape(unittest.TestCase):

    def test_plan_of_no_events_is_empty(self):
        self.assertEqual(DJActionPlanner("empty").plan([], MID()), [])

    def test_none_decisions_are_dropped_entirely(self):
        planner = DJActionPlanner("drop")
        energy = STEP()
        events = [
            GestureEvent(6.0, 1.0, "deck_glance", None, 0.5),    # low band -> none
            GestureEvent(60.0, 1.0, "deck_glance", None, 0.5),   # high band -> riser
        ]
        planned = planner.plan(events, energy)
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].action, "gain_riser")
        self.assertEqual(planned[0].start, 60.0)
        for action in planned:
            self.assertNotEqual(action.action, "none")

    def test_high_energy_plans_a_riser_with_no_side(self):
        planner = DJActionPlanner("riser")
        [planned] = planner.plan(
            [GestureEvent(70.0, 0.7, "deck_glance", "l", 0.5)], STEP()
        )
        self.assertEqual(planned.action, "gain_riser")
        self.assertIsNone(planned.side)


class TestActivationProbability(unittest.TestCase):
    """Phase 2C: the per-eligible-bar activation chance is a musically-justified
    function of the context - busier in a higher energy band, busier still on a
    rising trend - and stays a real probability at every input."""

    BANDS = ("low", "mid", "high")

    def _ctx(self, band, trend=0.0):
        return MusicalContext(
            time=0.0, energy=0.5, trend=trend, energy_band=band, at_bar_boundary=True
        )

    def test_monotonic_non_decreasing_low_to_mid_to_high_at_a_fixed_trend(self):
        for trend in (0.0, TREND_RISING, TREND_RISING + 0.5, 5.0):
            probs = [activation_probability(self._ctx(b, trend)) for b in self.BANDS]
            with self.subTest(trend=trend):
                for lower, higher in zip(probs, probs[1:]):
                    self.assertLessEqual(lower, higher)

    def test_the_bands_actually_differ_so_the_ordering_is_not_vacuous(self):
        probs = {b: activation_probability(self._ctx(b)) for b in self.BANDS}
        self.assertLess(probs["low"], probs["mid"])
        self.assertLess(probs["mid"], probs["high"])

    def test_a_rising_trend_never_lowers_a_bands_probability_and_never_passes_one(self):
        for band in self.BANDS:
            flat = activation_probability(self._ctx(band, 0.0))
            for trend in (TREND_RISING + 1e-6, TREND_RISING + 0.3, 12.0):
                rising = activation_probability(self._ctx(band, trend))
                with self.subTest(band=band, trend=trend):
                    self.assertGreaterEqual(rising, flat)
                    self.assertLessEqual(rising, 1.0)

    def test_a_trend_exactly_at_the_threshold_is_not_yet_rising(self):
        # dj_behavior's own `trend > TREND_RISING` reading - the boundary is open.
        for band in self.BANDS:
            self.assertEqual(
                activation_probability(self._ctx(band, TREND_RISING)),
                activation_probability(self._ctx(band, 0.0)),
            )

    def test_probability_is_in_the_unit_interval_and_never_zero_for_any_band(self):
        for band in self.BANDS:
            for trend in (-3.0, 0.0, TREND_RISING, 1.0, 50.0):
                p = activation_probability(self._ctx(band, trend))
                with self.subTest(band=band, trend=trend):
                    self.assertGreater(p, 0.0)
                    self.assertLessEqual(p, 1.0)

    def test_the_boost_is_exactly_the_band_base_plus_the_trend_boost_capped(self):
        for band in self.BANDS:
            expected = min(ACTIVATION_BASE[band] + ACTIVATION_TREND_BOOST, 1.0)
            self.assertAlmostEqual(
                activation_probability(self._ctx(band, TREND_RISING + 1e-6)),
                expected,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
