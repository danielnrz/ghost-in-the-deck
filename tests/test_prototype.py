"""Cue generation, timing measurement, and proof that movement is visible."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ghost_in_the_deck.animation.cues import BeatCueSource, MotionCue
from ghost_in_the_deck.audio.features import MusicFeatures
from ghost_in_the_deck.sync import SyncRecorder

import panda_env

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "avatar"
BAM = ASSETS / "ghost_test.bam"


def make_features(beats, duration=10.0) -> MusicFeatures:
    frames = [i * 0.05 for i in range(int(duration / 0.05))]
    ones = [1.0] * len(frames)
    return MusicFeatures(
        track="synthetic", duration_seconds=duration, sample_rate=22050,
        hop_length=512, bpm=120.0, beats=list(beats), frame_times=frames,
        onset_strength=ones, rms=ones, bass_energy=ones,
        mid_energy=ones, high_energy=ones,
    )


class TestBeatCueSource(unittest.TestCase):
    def setUp(self):
        self.source = BeatCueSource(make_features([0.5, 1.0, 1.5, 2.0]))

    def test_cues_fire_in_order_once_each(self):
        self.assertEqual(self.source.poll(0.4), [])
        first = self.source.poll(0.52)
        self.assertEqual([c.index for c in first], [0])
        self.assertEqual(self.source.poll(0.52), [], "a cue fired twice")
        self.assertEqual([c.index for c in self.source.poll(1.02)], [1])

    def test_stale_cues_are_dropped_not_burst(self):
        """After a stall the avatar must not fire every missed beat at once."""
        cues = self.source.poll(2.0)
        self.assertEqual([c.index for c in cues], [3], "missed beats were replayed")
        self.assertEqual(self.source.pending, 0)

    def test_strength_stays_in_range(self):
        for cue in self.source.poll(2.0):
            self.assertGreaterEqual(cue.strength, 0.25)
            self.assertLessEqual(cue.strength, 1.0)

    def test_reset_seeks_to_a_position(self):
        self.source.reset(1.2)
        self.assertEqual([c.index for c in self.source.poll(1.6)], [2])


class TestSyncRecorder(unittest.TestCase):
    def test_summary_statistics(self):
        recorder = SyncRecorder()
        recorder.record(0, 1.000, 1.010)   # +10 ms
        recorder.record(1, 2.000, 1.996)   #  -4 ms
        recorder.record(2, 3.000, 3.001)   #  +1 ms
        summary = recorder.summary()
        self.assertEqual(summary["beats"], 3)
        self.assertAlmostEqual(summary["mean_abs_error_ms"], 5.0, places=1)
        self.assertAlmostEqual(summary["median_abs_error_ms"], 4.0, places=1)
        self.assertAlmostEqual(summary["max_abs_error_ms"], 10.0, places=1)

    def test_empty_recorder_is_reported_not_crashed(self):
        self.assertEqual(SyncRecorder().summary(), {"beats": 0})


@unittest.skipUnless(BAM.is_file(), "avatar asset not built")
class TestVisibleMovement(unittest.TestCase):
    """Renders real frames and compares pixels: the movement must be on screen."""

    @classmethod
    def setUpClass(cls):
        if not panda_env.has_window():
            raise unittest.SkipTest("no display available for offscreen rendering")

        from ghost_in_the_deck.animation.controller import AvatarAnimator
        from ghost_in_the_deck.animation.rig import AvatarRig
        from panda3d.core import AmbientLight, DirectionalLight, Vec4

        cls.base = panda_env.get_base()
        cls.base.setBackgroundColor(0.05, 0.05, 0.08)
        cls.base.camera.setPos(0.0, -3.4, 1.15)
        cls.base.camera.lookAt(0.0, 0.0, 0.95)

        key = DirectionalLight("key")
        key.setColor(Vec4(1.2, 1.15, 1.1, 1))
        key_np = cls.base.render.attachNewNode(key)
        key_np.setHpr(-35, -25, 0)
        cls.base.render.setLight(key_np)
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.3, 0.3, 0.35, 1))
        cls.base.render.setLight(cls.base.render.attachNewNode(ambient))

        cls.rig = AvatarRig(BAM, parent=cls.base.render)
        # Sway is disabled here so the pixel comparisons isolate the beat movement.
        cls.animator = AvatarAnimator(cls.rig, sway_period=1.0e9)

    def test_avatar_is_actually_drawn(self):
        self.animator.reset()
        frame = panda_env.render_screenshot()
        self.assertGreater(frame.std(), 4.0, "frame looks like an empty background")

    def test_beat_changes_the_rendered_image(self):
        self.animator.reset()
        self.animator.update(1 / 60)
        rest = panda_env.render_screenshot().astype(np.int16)

        self.animator.apply_cue(MotionCue("beat", 0.0, 1.0, 0))
        self.animator.update(1 / 60)
        moved = panda_env.render_screenshot().astype(np.int16)

        changed = np.abs(moved - rest).max(axis=2)
        ratio = float((changed > 12).mean())
        self.assertGreater(ratio, 0.005, f"only {ratio:.4%} of pixels changed on a beat")

    def test_pose_returns_between_beats(self):
        self.animator.reset()
        self.animator.update(1 / 60)
        rest = panda_env.render_screenshot().astype(np.int16)

        self.animator.apply_cue(MotionCue("beat", 0.0, 1.0, 0))
        for _ in range(60):  # one second, longer than the decay
            self.animator.update(1 / 60)
        settled = panda_env.render_screenshot().astype(np.int16)

        changed = float((np.abs(settled - rest).max(axis=2) > 12).mean())
        self.assertLess(changed, 0.01, "avatar did not return to rest after a beat")


if __name__ == "__main__":
    unittest.main()
