"""Timeline construction, timing reports, and proof that movement is visible."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.sync import TimingRecorder

import panda_env
from synthetic import SampleState, groove_for, make_features, regular_beats

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
    """Per-sample bookkeeping. Coverage accounting lives in test_coverage_metrics."""

    def _state(self, time, beat_index, beat_age):
        return SampleState(time=time, beat_index=beat_index, beat_age=beat_age)

    def test_first_sample_carrying_a_beat_records_its_latency(self):
        recorder = TimingRecorder(beat_times=[1.0], response_window=0.48)
        response = recorder.record_sample(self._state(1.01, 0, 0.01), observed_at=1.01)
        self.assertIsNotNone(response)
        self.assertAlmostEqual(response.latency_ms, 10.0, places=6)
        self.assertAlmostEqual(response.beat_time, 1.0, places=6)
        self.assertAlmostEqual(response.sampled_at, 1.01, places=6)

    def test_a_beat_is_only_recorded_once(self):
        recorder = TimingRecorder(beat_times=[1.0], response_window=0.48)
        recorder.record_sample(self._state(1.01, 0, 0.01), observed_at=1.01)
        again = recorder.record_sample(self._state(1.10, 0, 0.10), observed_at=1.10)
        self.assertIsNone(again)
        self.assertEqual(len(recorder.responses), 1)

    def test_a_beat_carried_too_late_is_not_counted_as_sampled(self):
        recorder = TimingRecorder(beat_times=[1.0], response_window=0.48)
        recorder.record_sample(self._state(2.00, 0, 1.00), observed_at=2.00)
        self.assertEqual(recorder.summary()["beats_sampled"], 0)

    def test_state_lag_reports_the_pose_age(self):
        recorder = TimingRecorder()
        recorder.record_sample(self._state(1.0, 0, 0.01), observed_at=1.004)
        self.assertAlmostEqual(recorder.summary()["state_lag"]["max_ms"], 4.0, places=3)

    def test_empty_recorder_is_reported_not_crashed(self):
        summary = TimingRecorder().summary()
        self.assertEqual(summary["update_samples"], 0)
        self.assertEqual(summary["beats_sampled"], 0)
        self.assertEqual(summary["beats_missed"], 0)
        self.assertIn("Update samples", TimingRecorder().format_summary())

    def test_summary_wording_does_not_claim_presentation(self):
        """Naming must describe update samples, not verified display output."""
        recorder = TimingRecorder(beat_times=[1.0], response_window=0.48)
        recorder.record_sample(self._state(1.01, 0, 0.01), observed_at=1.01)
        text = recorder.format_summary()
        self.assertIn("Update samples", text)
        self.assertIn("not verified monitor presentation", text)
        for banned in ("Frames rendered", "displayed", "never rendered", "on screen"):
            self.assertNotIn(banned, text, f"{banned!r} overclaims what was measured")


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
        cls.groove = groove_for(beats=[1.0], duration=4.0, seed="prototype")
        cls.animator = AvatarAnimator(cls.rig, cls.groove)

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

    def test_pose_stays_within_bounded_travel(self):
        """The body keeps moving, but never wanders far from neutral."""
        self.animator.reset()
        neutral = panda_env.render_screenshot().astype(np.int16)

        worst = 0.0
        for step in range(40):
            self.animator.apply_at(step * 0.1)
            frame = panda_env.render_screenshot().astype(np.int16)
            moved = float((np.abs(frame - neutral).max(axis=2) > 12).mean())
            worst = max(worst, moved)
        self.assertLess(worst, 0.25, "pose drifted far from the neutral stance")

    def test_groove_moves_the_body_between_beats(self):
        """Movement continues where no beat accent is firing."""
        self.groove.pulse_decay = 0.02
        self.groove.pulse_attack = 0.005
        quiet = [2.30, 2.42, 2.54]
        for time in quiet:
            self.assertEqual(self.animator.state_at(time).pulse, 0.0)

        frames = []
        for time in quiet:
            self.animator.apply_at(time)
            frames.append(panda_env.render_screenshot().astype(np.int16))

        for earlier, later in zip(frames, frames[1:]):
            changed = float((np.abs(later - earlier).max(axis=2) > 12).mean())
            self.assertGreater(changed, 0.001, "no visible movement between beats")

    def test_louder_music_produces_larger_movement_on_screen(self):
        from ghost_in_the_deck.animation.controller import AvatarAnimator
        from synthetic import groove_for

        self.animator.reset()
        neutral = panda_env.render_screenshot().astype(np.int16)

        def travel(energy):
            animator = AvatarAnimator(
                self.rig,
                groove_for(beats=regular_beats(bpm=120.0, count=40), duration=20.0,
                           seed="energy", energy=energy),
            )
            worst = 0.0
            for step in range(24):
                animator.apply_at(4.0 + step * 0.05)
                frame = panda_env.render_screenshot().astype(np.int16)
                worst = max(worst, float((np.abs(frame - neutral).max(axis=2) > 12).mean()))
            return worst

        ramp = lambda t: t / 20.0
        quiet = travel(lambda t: max(0.02, ramp(t) * 0.05))
        self.animator.reset()
        loud = travel(lambda t: min(1.0, 0.6 + ramp(t)))
        self.assertGreater(loud, quiet, f"loud {loud:.4f} did not exceed quiet {quiet:.4f}")
        self.animator.reset()

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
