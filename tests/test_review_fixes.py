"""Regressions for the Phase 1A.1 review findings.

F1  bar phase jumped at virtual beat boundaries before the first detected beat
F2  an arbitrarily tiny constant envelope produced medium movement
F3  --debug-every 0 divided by zero
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.animation.energy import AUDIBLE_RMS, SILENCE_RMS, EnergyTrack
from ghost_in_the_deck.animation.groove import MIN_INTENSITY

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


if __name__ == "__main__":
    unittest.main()
