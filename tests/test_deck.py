"""TrackDeck: deterministic, self-contained wrapping of one analysed track.

Runs entirely on synthetic features; no private music, no Panda3D.
"""

from __future__ import annotations

import unittest

from ghost_in_the_deck.animation import structure as S
from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.animation.energy import EnergyTrack
from ghost_in_the_deck.deck import TrackDeck

from synthetic import make_features, regular_beats


def _rising(t: float) -> float:
    # A slow climb so the broad-structure layer has a real trend to read.
    return 0.2 + 0.05 * t


def _falling(t: float) -> float:
    return 1.0 - 0.04 * t


def _features_a():
    return make_features(
        regular_beats(bpm=120.0, count=48), duration=24.0, bpm=120.0, energy=_rising
    )


def _features_b():
    return make_features(
        regular_beats(bpm=128.0, count=48, offset=0.25),
        duration=20.0,
        bpm=128.0,
        energy=_falling,
    )


def _sample_times(duration: float):
    return [i * 0.5 for i in range(1, int(duration / 0.5))]


class TrackDeckConstruction(unittest.TestCase):
    def test_convenience_properties_mirror_features(self):
        features = _features_a()
        deck = TrackDeck.from_features(features)
        self.assertEqual(deck.track, features.track)
        self.assertEqual(deck.duration, features.duration_seconds)
        self.assertEqual(deck.bpm, features.bpm)

    def test_broad_energy_uses_phrase_smoothing_window(self):
        features = _features_a()
        deck = TrackDeck.from_features(features)
        reference = EnergyTrack(
            features, smoothing_seconds=S.PHRASE_SMOOTHING_SECONDS
        )
        for t in _sample_times(features.duration_seconds):
            self.assertAlmostEqual(deck.broad_energy.at(t), reference.at(t))
        # The broad curve is genuinely broader than the groove-scale one.
        groove = EnergyTrack(features)
        self.assertNotAlmostEqual(
            deck.broad_energy.at(12.0), groove.at(12.0), places=6
        )

    def test_energy_and_timeline_match_app_construction(self):
        features = _features_a()
        deck = TrackDeck.from_features(features)
        groove_energy = EnergyTrack(features)
        timeline = BeatTimeline(features)
        for t in _sample_times(features.duration_seconds):
            self.assertAlmostEqual(deck.energy.at(t), groove_energy.at(t))
            self.assertEqual(
                deck.timeline.phase_at(t).bar_index,
                timeline.phase_at(t).bar_index,
            )


class TrackDeckDeterminism(unittest.TestCase):
    def test_same_features_gives_identical_fields_twice(self):
        features = _features_a()
        one = TrackDeck.from_features(features)
        two = TrackDeck.from_features(features)
        for t in _sample_times(features.duration_seconds):
            self.assertEqual(one.energy.at(t), two.energy.at(t))
            self.assertEqual(one.broad_energy.at(t), two.broad_energy.at(t))
            self.assertEqual(one.structure_at(t), two.structure_at(t))


class TrackDeckIndependence(unittest.TestCase):
    def test_two_decks_are_fully_independent(self):
        features_a = _features_a()
        features_b = _features_b()

        deck_a = TrackDeck.from_features(features_a)
        deck_b = TrackDeck.from_features(features_b)

        # No shared derived objects.
        self.assertIsNot(deck_a.features, deck_b.features)
        self.assertIsNot(deck_a.timeline, deck_b.timeline)
        self.assertIsNot(deck_a.energy, deck_b.energy)
        self.assertIsNot(deck_a.broad_energy, deck_b.broad_energy)

        # Each deck's readings match a standalone build from its own features,
        # and are unchanged by the other deck existing.
        solo_a = TrackDeck.from_features(features_a)
        for t in _sample_times(features_a.duration_seconds):
            self.assertEqual(deck_a.energy.at(t), solo_a.energy.at(t))
            self.assertEqual(deck_a.structure_at(t), solo_a.structure_at(t))

        # The two decks genuinely describe different tracks.
        self.assertNotEqual(deck_a.bpm, deck_b.bpm)
        self.assertNotEqual(deck_a.duration, deck_b.duration)
        self.assertNotAlmostEqual(
            deck_a.broad_energy.at(8.0), deck_b.broad_energy.at(8.0), places=6
        )


class TrackDeckStructureWrapper(unittest.TestCase):
    def test_structure_at_agrees_with_direct_call(self):
        features = _features_a()
        deck = TrackDeck.from_features(features)
        broad = EnergyTrack(features, smoothing_seconds=S.PHRASE_SMOOTHING_SECONDS)
        timeline = BeatTimeline(features)
        for t in _sample_times(features.duration_seconds):
            self.assertEqual(
                deck.structure_at(t),
                S.structure_at(broad, timeline, t),
            )


if __name__ == "__main__":
    unittest.main()
