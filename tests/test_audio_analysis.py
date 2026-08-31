"""Validation of the audio analysis stage.

Run with:  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ghost_in_the_deck.audio.analysis import analyse
from ghost_in_the_deck.audio.features import MusicFeatures
from ghost_in_the_deck.audio.library import choose_track, find_tracks

ENVELOPES = ("onset_strength", "rms", "bass_energy", "mid_energy", "high_energy")


def make_click_track(path: Path, bpm: float, seconds: float, sr: int = 22050) -> None:
    """Write a metronome-like WAV with a known tempo."""
    samples = np.zeros(int(seconds * sr), dtype=np.float32)
    interval = 60.0 / bpm
    click = np.exp(-np.linspace(0, 12, int(0.03 * sr))).astype(np.float32)
    click *= np.sin(2 * np.pi * 1200 * np.arange(click.size) / sr).astype(np.float32)
    t = 0.0
    while t < seconds - 0.05:
        start = int(t * sr)
        samples[start : start + click.size] += click
        t += interval
    sf.write(path, samples, sr)


class TestTrackDiscovery(unittest.TestCase):
    def test_finds_audio_files(self):
        tracks = find_tracks()
        self.assertTrue(tracks, "no audio files discovered in testMusic/")
        for track in tracks:
            self.assertTrue(track.is_file())

    def test_chooses_a_track_without_hard_coded_name(self):
        self.assertIn(choose_track(), find_tracks())

    def test_hint_selects_matching_track(self):
        first = find_tracks()[0]
        self.assertEqual(choose_track(name_hint=first.stem[:6]), first)


class TestSyntheticTempo(unittest.TestCase):
    """A generated click track has a tempo we know exactly."""

    def test_detects_known_bpm(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "click.wav"
            make_click_track(wav, bpm=120.0, seconds=20.0)
            features = analyse(wav)

        self.assertAlmostEqual(features.bpm, 120.0, delta=6.0)
        intervals = np.diff(features.beats)
        self.assertAlmostEqual(float(np.median(intervals)), 0.5, delta=0.05)


class TestRealTrackAnalysis(unittest.TestCase):
    """The full chain on an actual file from testMusic/."""

    @classmethod
    def setUpClass(cls):
        cls.track = choose_track()
        cls.features = analyse(cls.track)

    def test_analysis_completes(self):
        self.assertEqual(self.features.track, self.track.name)
        self.assertGreater(self.features.duration_seconds, 10.0)

    def test_bpm_is_plausible(self):
        self.assertTrue(
            50.0 <= self.features.bpm <= 220.0, f"implausible bpm {self.features.bpm}"
        )

    def test_beats_are_monotonic(self):
        beats = self.features.beats
        self.assertGreater(len(beats), 20)
        for earlier, later in zip(beats, beats[1:]):
            self.assertLess(earlier, later)

    def test_beats_stay_inside_track(self):
        self.assertGreaterEqual(self.features.beats[0], 0.0)
        self.assertLessEqual(self.features.beats[-1], self.features.duration_seconds)

    def test_energy_arrays_are_finite_and_aligned(self):
        frames = len(self.features.frame_times)
        self.assertGreater(frames, 0)
        for name in ENVELOPES:
            values = np.asarray(getattr(self.features, name), dtype=float)
            self.assertEqual(values.size, frames, f"{name} length mismatch")
            self.assertTrue(np.all(np.isfinite(values)), f"{name} has non-finite values")
            self.assertGreaterEqual(float(values.min()), 0.0, name)
            self.assertGreater(float(values.max()), 0.0, f"{name} is silent")

    def test_beat_interval_matches_bpm(self):
        expected = 60.0 / self.features.bpm
        self.assertAlmostEqual(self.features.beat_interval, expected, delta=0.06)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            self.features.save(path)
            restored = MusicFeatures.load(path)
        self.assertEqual(restored.track, self.features.track)
        self.assertEqual(len(restored.beats), len(self.features.beats))
        self.assertAlmostEqual(restored.bpm, self.features.bpm, places=1)
        self.assertFalse(math.isnan(restored.duration_seconds))


if __name__ == "__main__":
    unittest.main()
