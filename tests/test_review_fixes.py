"""Regressions for review findings.

Phase 1A.1:
F1  bar phase jumped at virtual beat boundaries before the first detected beat
F2  an arbitrarily tiny constant envelope produced medium movement
F3  --debug-every 0 divided by zero

Phase 1C correction round 1 (hand_to_deck audio effect):
F3  a zero-event schedule reached playback re-encoded as PCM_16 instead of
    being served as the untouched source file
F4  the effect cache key truncated mtime to the second, so a same-size,
    same-second file replacement could serve a stale render

Phase 1C correction round 2 (hand_to_deck audio effect):
F5  the effect cache key carried no effect/schema version, so a render
    cached before an ``audio.effects`` change could still be served after it

Phase 1C correction round 3 (hand_to_deck audio effect):
F8  processed playback unconditionally re-encoded to PCM_16, silently
    saturating valid FLOAT-format source samples above +-1
"""

from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.animation.energy import AUDIBLE_RMS, SILENCE_RMS, EnergyTrack
from ghost_in_the_deck.animation.groove import MIN_INTENSITY
from ghost_in_the_deck.app import processed_audio_path

from synthetic import RecordingRig, groove_for, make_features, regular_beats

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-6


def worst_pose_step(animator, around: float, span: float = 0.5) -> float:
    """Largest joint change between samples straddling ``around``."""
    worst = 0.0
    previous = None
    steps = int(span / EPSILON) if span <= EPSILON * 4 else 400
    times = [around - span / 2 + i * (span / steps) for i in range(steps + 1)]
    for time in times:
        offsets = animator.pose_offsets(animator.state_at(time))
        if previous is not None:
            for joint, values in offsets.items():
                for a, b in zip(values, previous.get(joint, (0.0, 0.0, 0.0))):
                    worst = max(worst, abs(a - b))
        previous = offsets
    return worst


class TestVirtualBeatContinuity(unittest.TestCase):
    """F1: the grid before the first beat must be a real, continuous grid."""

    def timeline_from(self, first: float, interval: float = 0.5, count: int = 24):
        beats = [first + i * interval for i in range(count)]
        return BeatTimeline(make_features(beats, duration=first + count * interval + 4.0))

    def test_virtual_indices_decrease_before_the_first_beat(self):
        timeline = self.timeline_from(2.0)
        self.assertEqual(timeline.phase_at(1.9).index, -1)
        self.assertEqual(timeline.phase_at(1.4).index, -2)
        self.assertEqual(timeline.phase_at(0.9).index, -3)
        self.assertEqual(timeline.phase_at(2.1).index, 0)

    def test_bar_phase_is_continuous_across_a_virtual_boundary(self):
        for first in (0.75, 2.0, 5.0):
            timeline = self.timeline_from(first)
            interval = timeline.nominal_interval
            for step in range(1, 5):
                boundary = first - step * interval
                if boundary <= 0.0:
                    continue
                before = timeline.phase_at(boundary - EPSILON)
                after = timeline.phase_at(boundary + EPSILON)
                gap = abs(after.bar_phase - before.bar_phase)
                gap = min(gap, 1.0 - gap)   # a wrap from 1.0 to 0.0 is continuous
                with self.subTest(first=first, boundary=round(boundary, 3)):
                    self.assertLess(gap, 1e-4, f"bar phase jumped by {gap:.5f}")

    def test_bar_phase_is_continuous_at_the_first_real_beat(self):
        for first in (0.75, 2.0, 5.0):
            timeline = self.timeline_from(first)
            before = timeline.phase_at(first - EPSILON)
            after = timeline.phase_at(first + EPSILON)
            gap = abs(after.bar_phase - before.bar_phase)
            gap = min(gap, 1.0 - gap)
            with self.subTest(first=first):
                self.assertLess(gap, 1e-4)

    def test_pose_does_not_jump_in_the_intro(self):
        """The visible symptom: the body twitched at every virtual boundary."""
        for first in (0.75, 2.0, 5.0):
            beats = [first + i * 0.5 for i in range(24)]
            groove = groove_for(beats, duration=first + 16.0)
            animator = AvatarAnimator(RecordingRig(), groove)
            interval = groove.timeline.nominal_interval
            for step in range(1, 4):
                boundary = first - step * interval
                if boundary <= 0.05:
                    continue
                before = animator.pose_offsets(animator.state_at(boundary - EPSILON))
                after = animator.pose_offsets(animator.state_at(boundary + EPSILON))
                worst = max(
                    abs(a - b)
                    for joint, values in after.items()
                    for a, b in zip(values, before.get(joint, (0.0, 0.0, 0.0)))
                )
                with self.subTest(first=first, boundary=round(boundary, 3)):
                    self.assertLess(worst, 0.01, f"pose jumped {worst:.3f} deg")

    def test_post_final_extrapolation_is_also_continuous(self):
        timeline = self.timeline_from(1.0, count=8)
        last = 1.0 + 7 * 0.5
        interval = timeline.nominal_interval
        for step in range(1, 4):
            boundary = last + step * interval
            before = timeline.phase_at(boundary - EPSILON)
            after = timeline.phase_at(boundary + EPSILON)
            self.assertEqual(after.index, before.index + 1)
            gap = abs(after.bar_phase - before.bar_phase)
            gap = min(gap, 1.0 - gap)
            with self.subTest(boundary=round(boundary, 3)):
                self.assertLess(gap, 1e-4)

    def test_real_beats_stay_identifiable(self):
        timeline = self.timeline_from(2.0)
        self.assertIsNone(timeline.cue_before(1.9))
        self.assertEqual(timeline.cue_before(2.1).index, 0)
        groove = groove_for([2.0 + i * 0.5 for i in range(12)], duration=12.0)
        self.assertFalse(groove.state_at(1.5).has_detected_beat)
        self.assertTrue(groove.state_at(2.5).has_detected_beat)

    def test_intro_phase_still_advances(self):
        timeline = self.timeline_from(2.0)
        phases = [timeline.phase_at(t).phase for t in (0.10, 0.22, 0.34, 0.46)]
        self.assertEqual(len(set(round(p, 6) for p in phases)), 4)


class TestSilenceEnergy(unittest.TestCase):
    """F2: approaching silence must approach stillness, smoothly."""

    def intensity(self, energy, duration: float = 14.0) -> float:
        groove = groove_for(regular_beats(bpm=120.0, count=24),
                            duration=duration, energy=energy)
        return groove.state_at(duration * 0.5).intensity

    def test_true_silence_is_restrained(self):
        self.assertAlmostEqual(self.intensity(0.0), MIN_INTENSITY, places=6)

    def test_an_arbitrarily_tiny_signal_is_also_restrained(self):
        """The finding: 1e-12 used to land at medium intensity."""
        for level in (1e-12, 1e-9, 1e-6):
            with self.subTest(level=level):
                self.assertAlmostEqual(self.intensity(level), MIN_INTENSITY, places=4)

    def test_intensity_rises_monotonically_out_of_silence(self):
        levels = [0.0, 1e-6, 2e-4, 5e-4, 1e-3, 3e-3, 0.05]
        values = [self.intensity(level) for level in levels]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later + 1e-9, f"not monotonic: {values}")
        self.assertAlmostEqual(values[0], MIN_INTENSITY, places=6)
        self.assertGreater(values[-1], MIN_INTENSITY + 0.2)

    def test_the_ramp_has_no_step_in_it(self):
        levels = [SILENCE_RMS * (AUDIBLE_RMS / SILENCE_RMS) ** (i / 24.0)
                  for i in range(25)]
        values = [self.intensity(level) for level in levels]
        steps = [abs(b - a) for a, b in zip(values, values[1:])]
        self.assertLess(max(steps), 0.12, f"largest step {max(steps):.3f}")

    def test_a_quiet_but_real_track_still_moves(self):
        quiet = self.intensity(lambda t: 0.004 * (0.2 + 0.8 * t / 14.0), duration=14.0)
        self.assertGreater(quiet, MIN_INTENSITY)

    def test_normal_and_loud_tracks_reach_full_intensity(self):
        for scale in (0.25, 0.4):
            groove = groove_for(regular_beats(bpm=120.0, count=24), duration=14.0,
                                energy=lambda t, s=scale: s * (0.2 + 0.8 * t / 14.0))
            self.assertGreater(groove.state_at(13.0).intensity, 0.9)

    def test_silence_inside_a_loud_track_is_restrained(self):
        """A gap in a loud track is silent in absolute terms, not just relative."""
        def profile(time):
            return 0.0 if 6.0 <= time <= 10.0 else 0.3

        groove = groove_for(regular_beats(bpm=120.0, count=32), duration=18.0,
                            energy=profile)
        self.assertLess(groove.state_at(8.0).intensity, 0.45)
        self.assertGreater(groove.state_at(2.0).intensity, 0.5)

    def test_features_without_a_loudness_reference_are_left_alone(self):
        """Analysis written before the statistic existed must still work."""
        features = make_features(regular_beats(bpm=120.0, count=12), duration=8.0)
        features.peak_rms = 0.0
        self.assertFalse(features.has_absolute_loudness)
        track = EnergyTrack(features)
        self.assertEqual(track.summary()["frames"], len(features.frame_times))


class TestDebugArgumentValidation(unittest.TestCase):
    """F3: --debug-every is used as a divisor."""

    def run_app(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "ghost_in_the_deck.app", *args],
            cwd=ROOT, capture_output=True, text=True,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        )

    def test_zero_is_rejected_cleanly(self):
        result = self.run_app("--debug-every", "0", "--seconds", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be 1 or greater", result.stderr)
        self.assertNotIn("ZeroDivisionError", result.stderr)

    def test_negative_is_rejected_cleanly(self):
        result = self.run_app("--debug-every", "-5", "--seconds", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be 1 or greater", result.stderr)

    def test_non_numeric_is_rejected_cleanly(self):
        result = self.run_app("--debug-every", "many", "--seconds", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)

    def test_positive_int_accepts_sensible_values(self):
        from ghost_in_the_deck.app import positive_int

        self.assertEqual(positive_int("1"), 1)
        self.assertEqual(positive_int("240"), 240)
        with self.assertRaises(Exception):
            positive_int("0")


class _FakeBehavior:
    """Stands in for DJBehaviorEngine: processed_audio_path only reads these."""

    def __init__(self, events, seed="test"):
        self.events = events
        self.seed = seed


def _write_wav(path: Path, sample_count: int, amplitude: float) -> None:
    t = np.arange(sample_count) / 44100.0
    samples = (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), samples, 44100, subtype="FLOAT")


class TestHandToDeckEffectWiring(unittest.TestCase):
    """Phase 1C correction round 1: F3 (no-op re-encoding) and F4 (stale cache)."""

    def test_no_events_returns_the_source_file_untouched(self):
        """F3: a schedule with no hand_to_deck events must not be rendered
        into a PCM_16 copy - the source file itself is the correct output."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "track.wav"
            _write_wav(wav, 44100, 0.5)
            before = wav.read_bytes()

            result = processed_audio_path(wav, _FakeBehavior(events=[]))

            self.assertEqual(result, wav)
            self.assertEqual(wav.read_bytes(), before)

    def test_stale_cache_is_not_served_after_a_same_second_replacement(self):
        """F4: two different-content, same-size files that land in the same
        integer second of mtime (but different nanoseconds) must not share a
        cache entry."""
        import tempfile

        from ghost_in_the_deck.animation.dj_behavior import GestureEvent

        events = [GestureEvent(0.1, 0.2, "hand_to_deck", "l", 0.9)]
        behavior = _FakeBehavior(events=events)

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "track.wav"

            _write_wav(wav, 4410, 0.3)
            import os
            os.utime(wav, (1_700_000_000.1, 1_700_000_000.1))
            first_target = processed_audio_path(wav, behavior)
            first_bytes = first_target.read_bytes()

            _write_wav(wav, 4410, 0.9)
            os.utime(wav, (1_700_000_000.9, 1_700_000_000.9))
            second_target = processed_audio_path(wav, behavior)

            self.assertNotEqual(
                first_target, second_target,
                "same-second replacement reused the previous cache entry",
            )
            self.assertNotEqual(second_target.read_bytes(), first_bytes)

    def test_effect_version_bump_invalidates_the_cache(self):
        """F5: a render cached under one EFFECT_VERSION must not be served
        once the effect implementation's version changes, even though the
        wav and the schedule are unchanged."""
        import tempfile
        from unittest import mock

        from ghost_in_the_deck import app as app_module
        from ghost_in_the_deck.animation.dj_behavior import GestureEvent

        events = [GestureEvent(0.1, 0.2, "hand_to_deck", "l", 0.9)]
        behavior = _FakeBehavior(events=events)

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "track.wav"
            _write_wav(wav, 4410, 0.3)

            with mock.patch.object(app_module, "EFFECT_VERSION", 1):
                first_target = processed_audio_path(wav, behavior)

            with mock.patch.object(app_module, "EFFECT_VERSION", 2):
                second_target = processed_audio_path(wav, behavior)

            self.assertNotEqual(
                first_target, second_target,
                "an EFFECT_VERSION bump reused the previous cache entry",
            )

    def test_events_differing_below_the_old_rounding_do_not_collide(self):
        """F6: two schedules that used to serialise identically at 6 decimal
        places must still resolve to distinct cache keys."""
        import tempfile

        from ghost_in_the_deck.animation.dj_behavior import GestureEvent

        events_a = [GestureEvent(0.10000001, 0.2, "hand_to_deck", "l", 0.9)]
        events_b = [GestureEvent(0.10000009, 0.2, "hand_to_deck", "l", 0.9)]

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "track.wav"
            _write_wav(wav, 4410, 0.3)

            target_a = processed_audio_path(wav, _FakeBehavior(events=events_a))
            target_b = processed_audio_path(wav, _FakeBehavior(events=events_b))

            self.assertNotEqual(
                target_a, target_b,
                "schedules differing below the sixth decimal shared a cache entry",
            )

    def test_a_float_source_above_full_scale_is_not_saturated_to_pcm16(self):
        """F8: a FLOAT-format source wav is allowed to carry samples beyond
        +-1 (unlike PCM formats, which cannot represent them at all). Writing
        the processed render back as PCM_16 - as an earlier version of this
        function unconditionally did - silently saturates such samples to
        PCM_16's ceiling (~0.99997). Preserving the source's own subtype
        must carry them through untouched."""
        import tempfile

        from ghost_in_the_deck.animation.dj_behavior import GestureEvent

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "hot.wav"
            t = np.arange(int(2.0 * 44100)) / 44100.0
            samples = (1.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            sf.write(str(wav), samples, 44100, subtype="FLOAT")

            events = [GestureEvent(0.5, 0.6, "hand_to_deck", "l", 0.9)]
            target = processed_audio_path(wav, _FakeBehavior(events=events))

            info = sf.info(str(target))
            self.assertEqual(
                info.subtype, "FLOAT",
                "processed render did not keep the source's FLOAT subtype",
            )
            processed, _ = sf.read(str(target), dtype="float64")
            self.assertGreater(
                float(np.max(np.abs(processed))), 1.05,
                "full-scale FLOAT samples were saturated by a PCM_16 write",
            )


if __name__ == "__main__":
    unittest.main()
