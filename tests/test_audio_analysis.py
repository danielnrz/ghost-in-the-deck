"""Validation of the audio analysis stage.

Run with:  PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -v

Everything mandatory here runs on generated audio, so a fresh clone with an
empty ``testMusic/`` gets a full, meaningful result. The checks against the
user's own music are an optional extra and skip when there is none.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghost_in_the_deck.audio.analysis import analyse
from ghost_in_the_deck.audio.features import MusicFeatures
from ghost_in_the_deck.audio.library import choose_track, find_tracks

from synthetic import make_beat_track, make_silent_track

ENVELOPES = ("onset_strength", "rms", "bass_energy", "mid_energy", "high_energy")


def check_feature_invariants(case: unittest.TestCase, features: MusicFeatures) -> None:
    """Everything that must hold for any analysed track."""
    case.assertGreater(features.duration_seconds, 1.0)
    case.assertTrue(
        50.0 <= features.bpm <= 220.0, f"implausible bpm {features.bpm}"
    )

    beats = features.beats
    case.assertGreater(len(beats), 10, "too few beats to be a real detection")
    for earlier, later in zip(beats, beats[1:]):
        case.assertLess(earlier, later, "beat timestamps are not increasing")
    case.assertGreaterEqual(beats[0], 0.0)
    case.assertLessEqual(beats[-1], features.duration_seconds)

    frames = len(features.frame_times)
    case.assertGreater(frames, 0)
    for name in ENVELOPES:
        values = np.asarray(getattr(features, name), dtype=float)
        case.assertEqual(values.size, frames, f"{name} length mismatch")
        case.assertTrue(np.all(np.isfinite(values)), f"{name} has non-finite values")
        case.assertGreaterEqual(float(values.min()), 0.0, name)
        case.assertGreater(float(values.max()), 0.0, f"{name} is silent")

    expected_interval = 60.0 / features.bpm
    case.assertAlmostEqual(features.beat_interval, expected_interval, delta=0.06)


class TestTrackDiscovery(unittest.TestCase):
    """Discovery is tested against generated files, never the user's music."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        make_silent_track(self.directory / "bravo.wav", seconds=3.0)
        make_silent_track(self.directory / "alpha.wav", seconds=1.0)   # smallest
        (self.directory / "notes.txt").write_text("not audio")
        self.addCleanup(self._tmp.cleanup)

    def test_finds_only_supported_audio(self):
        names = [p.name for p in find_tracks(self.directory)]
        self.assertEqual(names, ["alpha.wav", "bravo.wav"])

    def test_chooses_the_smallest_file_by_default(self):
        self.assertEqual(choose_track(self.directory).name, "alpha.wav")

    def test_hint_selects_matching_track(self):
        self.assertEqual(choose_track(self.directory, "brav").name, "bravo.wav")

    def test_unknown_hint_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            choose_track(self.directory, "nothing-like-this")

    def test_empty_directory_reports_clearly(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(find_tracks(empty), [])
            with self.assertRaises(FileNotFoundError):
                choose_track(empty)

    def test_missing_directory_is_not_a_crash(self):
        self.assertEqual(find_tracks(self.directory / "does-not-exist"), [])


class TestSyntheticAnalysis(unittest.TestCase):
    """The mandatory correctness checks, on audio we generate ourselves."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.wav = make_beat_track(
            Path(cls._tmp.name) / "synthetic.wav", bpm=120.0, seconds=20.0
        )
        cls.features = analyse(cls.wav)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_detects_the_known_bpm(self):
        self.assertAlmostEqual(self.features.bpm, 120.0, delta=6.0)
        intervals = np.diff(self.features.beats)
        self.assertAlmostEqual(float(np.median(intervals)), 0.5, delta=0.05)

    def test_feature_invariants_hold(self):
        check_feature_invariants(self, self.features)

    def test_analysis_names_the_source_file(self):
        self.assertEqual(self.features.track, self.wav.name)

    def test_bands_reflect_the_generated_content(self):
        """The fixture has a low kick and a high tick, so both bands respond."""
        bass = np.asarray(self.features.bass_energy)
        high = np.asarray(self.features.high_energy)
        self.assertGreater(bass.max(), 0.5, "kick did not show up in the bass band")
        self.assertGreater(high.max(), 0.5, "tick did not show up in the high band")

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            self.features.save(path)
            restored = MusicFeatures.load(path)
        self.assertEqual(restored.track, self.features.track)
        self.assertEqual(len(restored.beats), len(self.features.beats))
        self.assertAlmostEqual(restored.bpm, self.features.bpm, places=1)
        self.assertFalse(math.isnan(restored.duration_seconds))

    def test_a_different_tempo_is_also_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = make_beat_track(Path(tmp) / "fast.wav", bpm=90.0, seconds=20.0)
            features = analyse(wav)
        self.assertAlmostEqual(features.bpm, 90.0, delta=6.0)


@unittest.skipUnless(
    find_tracks(), "no audio in testMusic/ - optional local-music integration test"
)
class TestLocalMusicIntegration(unittest.TestCase):
    """Optional: the same invariants against the user's own library.

    Skips cleanly on a fresh clone. Nothing here is required for the suite to be
    meaningful; it exists so real files can be checked when they are present.
    """

    @classmethod
    def setUpClass(cls):
        cls.track = choose_track()
        cls.features = analyse(cls.track)

    def test_analysis_completes(self):
        self.assertEqual(self.features.track, self.track.name)
        self.assertGreater(self.features.duration_seconds, 10.0)

    def test_feature_invariants_hold(self):
        check_feature_invariants(self, self.features)


if __name__ == "__main__":
    unittest.main()
