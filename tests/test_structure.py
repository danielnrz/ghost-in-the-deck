"""Phase 3A subtask 1: the broad-structure signal.

Proves that ``structure.structure_at`` is deterministic, that its broad trend is
a genuinely longer-timescale reading than ``dj_behavior.trend_at``'s default
3-second one (not a renamed copy), that all four honestly-named regimes are
reachable, and that ``phrase_position`` is exactly ``bar_index`` folded into an
assumed eight-bar cycle.

Runs entirely on synthetic features; no private music.
"""

from __future__ import annotations

import statistics
import unittest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.animation.dj_behavior import (
    HIGH_ENERGY,
    TREND_LOOKBACK,
    trend_at,
)
from ghost_in_the_deck.animation.energy import EnergyTrack
from ghost_in_the_deck.animation import structure as S

from synthetic import make_features, regular_beats

BPM = 120.0
DURATION = 300.0
BEATS = regular_beats(bpm=BPM, count=int(DURATION / (60.0 / BPM)), offset=0.5)


def _broad(energy):
    features = make_features(BEATS, duration=DURATION, bpm=BPM, energy=energy)
    track = EnergyTrack(features, smoothing_seconds=S.PHRASE_SMOOTHING_SECONDS)
    return track, BeatTimeline(features)


class TestTrendKeywordIsBackwardCompatible(unittest.TestCase):
    def test_default_lookback_is_unchanged(self):
        track, _ = _broad(lambda t: 0.1 + 0.8 * t / DURATION)
        for t in (0.0, 3.0, 12.5, 90.0, 250.0):
            self.assertEqual(trend_at(track, t), trend_at(track, t, TREND_LOOKBACK))


class TestStructureIsDeterministic(unittest.TestCase):
    def test_same_inputs_give_identical_structure(self):
        track, timeline = _broad(lambda t: 0.2 + 0.6 * (t / DURATION))
        for t in (0.0, 7.3, 40.0, 123.4, 299.0):
            self.assertEqual(
                S.structure_at(track, timeline, t),
                S.structure_at(track, timeline, t),
            )


class TestBroadTrendIsNotTheShortTrend(unittest.TestCase):
    """A slow multi-minute rise carrying small fast periodic spikes: the broad
    trend follows the rise almost everywhere, while the default 3-second
    ``trend_at`` is dominated by the spikes and swings negative constantly.
    """

    @staticmethod
    def _slow_rise_with_spikes(t):
        base = min(0.12 + 0.85 * (t / 180.0), 0.97)
        spike = 0.30 if (t % 4.0) < 0.5 else 0.0
        return min(base + spike, 1.0)

    def test_broad_trend_tracks_the_slow_rise_and_the_short_one_does_not(self):
        features = make_features(
            BEATS, duration=DURATION, bpm=BPM, energy=self._slow_rise_with_spikes
        )
        short = EnergyTrack(features, smoothing_seconds=1.5)
        broad = EnergyTrack(features, smoothing_seconds=S.PHRASE_SMOOTHING_SECONDS)

        # Sample only the genuinely-rising stretch, off the spike-aligned grid.
        times = [10.0 + 0.37 * i for i in range(500)]
        times = [t for t in times if t < 170.0]

        broad_trends = [trend_at(broad, t, S.PHRASE_TREND_LOOKBACK) for t in times]
        short_trends = [trend_at(short, t) for t in times]

        broad_positive = sum(x > 0 for x in broad_trends) / len(times)
        short_negative = sum(x < 0 for x in short_trends) / len(times)

        self.assertGreater(broad_positive, 0.9, "broad trend should follow the rise")
        self.assertGreater(statistics.mean(broad_trends), 0.03)
        self.assertGreater(
            short_negative, 0.15,
            "the 3-second trend should be pulled negative by the spikes often",
        )
        self.assertGreater(
            statistics.mean(broad_trends), 2.0 * statistics.mean(short_trends)
        )


class TestAllFourRegimes(unittest.TestCase):
    """A crafted profile: steep rise, high plateau, steep fall, low plateau -
    hitting build, peak, release and stable in turn.
    """

    @staticmethod
    def _profile(t):
        if t < 30.0:
            base = 0.10 + 0.85 * (t / 30.0)
        elif t < 110.0:
            base = 0.95
        elif t < 150.0:
            base = 0.95 - 0.85 * ((t - 110.0) / 40.0)
        else:
            base = 0.35
        spike = 0.06 if (t % 37.0) < 0.5 else 0.0
        return min(base + spike, 1.0)

    def test_each_regime_appears_at_least_once(self):
        track, timeline = _broad(self._profile)
        regimes = {
            S.structure_at(track, timeline, t).regime
            for t in range(0, int(DURATION), 2)
        }
        self.assertEqual(regimes, {"build", "release", "peak", "stable"})

    def test_regime_definitions_hold_at_sample_points(self):
        track, timeline = _broad(self._profile)

        build = S.structure_at(track, timeline, 22.0)
        self.assertEqual(build.regime, "build")
        self.assertGreaterEqual(build.broad_trend, S.BUILD_SLOPE)

        peak = S.structure_at(track, timeline, 80.0)
        self.assertEqual(peak.regime, "peak")
        self.assertLess(peak.broad_trend, S.BUILD_SLOPE)
        self.assertGreater(peak.broad_trend, -S.RELEASE_SLOPE)
        self.assertGreaterEqual(peak.broad_energy, HIGH_ENERGY)

        release = S.structure_at(track, timeline, 135.0)
        self.assertEqual(release.regime, "release")
        self.assertLessEqual(release.broad_trend, -S.RELEASE_SLOPE)

        stable = S.structure_at(track, timeline, 260.0)
        self.assertEqual(stable.regime, "stable")
        self.assertLess(stable.broad_energy, HIGH_ENERGY)
        self.assertLess(abs(stable.broad_trend), S.BUILD_SLOPE)


class TestSectionChangeLikelihood(unittest.TestCase):
    def test_it_is_the_scaled_absolute_broad_trend_clamped(self):
        track, timeline = _broad(TestAllFourRegimes._profile)
        for t in (10.0, 22.0, 80.0, 135.0, 260.0):
            ms = S.structure_at(track, timeline, t)
            self.assertEqual(
                ms.section_change_likelihood,
                min(abs(ms.broad_trend) / S.SECTION_CHANGE_SCALE, 1.0),
            )
            self.assertGreaterEqual(ms.section_change_likelihood, 0.0)
            self.assertLessEqual(ms.section_change_likelihood, 1.0)


class TestPhrasePosition(unittest.TestCase):
    def test_phrase_position_is_bar_index_modulo_phrase_length(self):
        track, timeline = _broad(lambda t: 0.3 + 0.5 * (t / DURATION))
        for t in [i * 0.5 for i in range(0, 600)]:
            ms = S.structure_at(track, timeline, t)
            expected_bar = timeline.phase_at(t).bar_index
            self.assertEqual(ms.bar_index, expected_bar)
            self.assertEqual(
                ms.phrase_position, expected_bar % S.PHRASE_LENGTH_BARS
            )
            self.assertIn(ms.phrase_position, range(S.PHRASE_LENGTH_BARS))

    def test_phrase_position_runs_through_the_whole_cycle(self):
        track, timeline = _broad(lambda t: 0.4)
        seen = {
            S.structure_at(track, timeline, t).phrase_position
            for t in [i * 0.5 for i in range(0, 600)]
        }
        self.assertEqual(seen, set(range(S.PHRASE_LENGTH_BARS)))


if __name__ == "__main__":
    unittest.main()
