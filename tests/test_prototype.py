"""Timeline construction, timing reports, and proof that movement is visible."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.sync import TimingRecorder

import panda_env
from synthetic import make_features, regular_beats

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "avatar"
BAM = ASSETS / "ghost_test.bam"


class TestBeatTimeline(unittest.TestCase):
    def setUp(self):
        self.timeline = BeatTimeline(make_features([0.5, 1.0, 1.5, 2.0], duration=4.0))

    def test_one_cue_per_beat(self):
        self.assertEqual(len(self.timeline), 4)
        self.assertEqual(
            [self.timeline.cue(i).scheduled_time for i in range(4)],
            [0.5, 1.0, 1.5, 2.0],
        )

    def test_cue_before_picks_the_latest_past_beat(self):
        self.assertIsNone(self.timeline.cue_before(0.4))
        self.assertEqual(self.timeline.cue_before(0.5).index, 0)
        self.assertEqual(self.timeline.cue_before(1.4).index, 1)
        self.assertEqual(self.timeline.cue_before(99.0).index, 3)

    def test_strength_stays_in_range(self):
        for index in range(len(self.timeline)):
            strength = self.timeline.cue(index).strength
            self.assertGreaterEqual(strength, 0.25)
            self.assertLessEqual(strength, 1.0)

    def test_end_time_is_the_last_beat(self):
        self.assertEqual(self.timeline.end_time, 2.0)

    def test_empty_timeline_is_usable(self):
        empty = BeatTimeline(make_features([], duration=4.0))
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.end_time, 0.0)
        self.assertIsNone(empty.cue_before(1.0))


class TestTimingRecorder(unittest.TestCase):
    def _state(self, time, beat_index, beat_age):
        from ghost_in_the_deck.animation.controller import MotionState

        return MotionState(
            time=time, impulse=0.5, sway=0.0, beat_index=beat_index, beat_age=beat_age
        )

    def test_first_frame_showing_a_beat_records_its_latency(self):
        recorder = TimingRecorder(visible_for=0.48)
        response = recorder.record_frame(self._state(1.01, 0, 0.01), observed_at=1.01)
        self.assertIsNotNone(response)
        self.assertAlmostEqual(response.latency_ms, 10.0, places=6)
        self.assertAlmostEqual(response.beat_time, 1.0, places=6)

    def test_a_beat_is_only_recorded_once(self):
        recorder = TimingRecorder(visible_for=0.48)
        recorder.record_frame(self._state(1.01, 0, 0.01), observed_at=1.01)
        again = recorder.record_frame(self._state(1.10, 0, 0.10), observed_at=1.10)
        self.assertIsNone(again)
        self.assertEqual(len(recorder.responses), 1)

    def test_beats_with_no_frame_are_counted_as_never_rendered(self):
        recorder = TimingRecorder(visible_for=0.48)
        recorder.record_frame(self._state(1.01, 0, 0.01), observed_at=1.01)
        recorder.record_frame(self._state(3.01, 4, 0.01), observed_at=3.01)
        self.assertEqual(recorder.summary()["beats_displayed"], 2)
        self.assertEqual(recorder.beats_never_rendered, 3)   # indices 1, 2, 3

    def test_a_beat_seen_too_late_is_not_counted_as_displayed(self):
        recorder = TimingRecorder(visible_for=0.48)
        recorder.record_frame(self._state(2.00, 0, 1.00), observed_at=2.00)
        self.assertEqual(recorder.summary()["beats_displayed"], 0)

    def test_state_lag_reports_the_pose_age(self):
        recorder = TimingRecorder()
        recorder.record_frame(self._state(1.0, 0, 0.01), observed_at=1.004)
        self.assertAlmostEqual(recorder.summary()["state_lag"]["max_ms"], 4.0, places=3)

    def test_empty_recorder_is_reported_not_crashed(self):
        summary = TimingRecorder().summary()
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["beats_displayed"], 0)
        self.assertEqual(summary["beats_never_rendered"], 0)
        self.assertIn("Frames rendered", TimingRecorder().format_summary())


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
        # One beat at t=1.0. Sway is disabled so the pixel comparisons isolate
        # the beat movement.
        cls.timeline = BeatTimeline(make_features([1.0], duration=4.0))
        cls.animator = AvatarAnimator(
            cls.rig, cls.timeline, sway_period=1.0e9
        )

    def test_avatar_is_actually_drawn(self):
        self.animator.apply_at(0.0)
        frame = panda_env.render_screenshot()
        self.assertGreater(frame.std(), 4.0, "frame looks like an empty background")

    def test_beat_changes_the_rendered_image(self):
        self.animator.apply_at(0.5)               # before the beat
        rest = panda_env.render_screenshot().astype(np.int16)

        self.animator.apply_at(1.01)              # just after it
        moved = panda_env.render_screenshot().astype(np.int16)

        changed = np.abs(moved - rest).max(axis=2)
        ratio = float((changed > 12).mean())
        self.assertGreater(ratio, 0.005, f"only {ratio:.4%} of pixels changed on a beat")

    def test_pose_returns_between_beats(self):
        self.animator.apply_at(0.5)
        rest = panda_env.render_screenshot().astype(np.int16)

        self.animator.apply_at(2.5)               # well past the decay
        settled = panda_env.render_screenshot().astype(np.int16)

        changed = float((np.abs(settled - rest).max(axis=2) > 12).mean())
        self.assertLess(changed, 0.01, "avatar did not return to rest after a beat")

    def test_rendered_pose_does_not_depend_on_frame_history(self):
        """The schedule-independence invariant, on the real rig and real pixels."""
        probe = 1.08

        self.animator.apply_at(probe)
        direct = panda_env.render_screenshot().astype(np.int16)

        for frame in regular_beats(bpm=300.0, count=40, offset=0.0):
            if frame >= probe:
                break
            self.animator.apply_at(frame)
        self.animator.apply_at(probe)
        after_history = panda_env.render_screenshot().astype(np.int16)

        self.assertEqual(
            int(np.abs(after_history - direct).max()),
            0,
            "the pose at a playback time depended on the frames drawn before it",
        )


if __name__ == "__main__":
    unittest.main()
