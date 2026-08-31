"""Timing architecture: the pose at a playback time must not depend on history.

These tests drive the real BeatTimeline, AvatarAnimator and TimingRecorder with
deliberately irregular render schedules. The rig is a recording stand-in so the
whole file runs without a display, a sound card or any private music.

The invariant under test:

    Given the same timeline and the same absolute playback time T, the pose is
    the same regardless of which frames were rendered before T.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ghost_in_the_deck.animation.controller import AvatarAnimator
from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.sync import TimingRecorder

from synthetic import RecordingRig, make_beat_track, make_features, regular_beats

BPM = 120.0
BEATS = regular_beats(bpm=BPM, count=24, offset=0.5)   # 0.5 s apart, 0.5 .. 12.0
TIMELINE = BeatTimeline(make_features(BEATS, duration=14.0, bpm=BPM))

# Probe times chosen to land at awkward places: just after a beat, mid-decay,
# long after a beat, and between beats.
PROBES = [0.51, 0.62, 1.03, 2.5, 3.017, 4.44, 5.9, 7.25, 9.999, 11.5]


def steady(fps: float, duration: float = 13.0) -> list[float]:
    step = 1.0 / fps
    count = int(duration / step)
    return [i * step for i in range(1, count + 1)]


def with_stalls(stall: float, duration: float = 13.0, fps: float = 60.0) -> list[float]:
    """A 60 fps schedule interrupted by a stall every second."""
    step = 1.0 / fps
    times: list[float] = []
    now = 0.0
    next_stall = 1.0
    while now < duration:
        now += step
        if now >= next_stall:
            now += stall
            next_stall = now + 1.0
        times.append(now)
    return times


SCHEDULES = {
    "60 fps": steady(60.0),
    "30 fps": steady(30.0),
    "15 fps": steady(15.0),
    "5 fps": steady(5.0),
    "stall 250 ms": with_stalls(0.25),
    "stall 500 ms": with_stalls(0.50),
    "stall 1000 ms": with_stalls(1.00),
}


def pose_after(schedule: list[float], probe: float) -> dict:
    """Replay ``schedule`` up to ``probe``, then draw ``probe``."""
    rig = RecordingRig()
    animator = AvatarAnimator(rig, TIMELINE)
    for frame in schedule:
        if frame >= probe:
            break
        animator.apply_at(frame)
    animator.apply_at(probe)
    return rig.pose()


def reference_pose(probe: float) -> dict:
    """The pose with no history at all."""
    rig = RecordingRig()
    AvatarAnimator(rig, TIMELINE).apply_at(probe)
    return rig.pose()


class TestScheduleIndependence(unittest.TestCase):
    """The headline invariant, across every frame cadence."""

    def test_pose_matches_reference_for_every_schedule(self):
        for name, schedule in SCHEDULES.items():
            for probe in PROBES:
                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(
                        pose_after(schedule, probe),
                        reference_pose(probe),
                        f"{name} at t={probe} produced a history-dependent pose",
                    )

    def test_schedules_agree_with_each_other(self):
        for probe in PROBES:
            poses = {
                name: pose_after(schedule, probe) for name, schedule in SCHEDULES.items()
            }
            first = poses["60 fps"]
            for name, pose in poses.items():
                with self.subTest(schedule=name, probe=probe):
                    self.assertEqual(pose, first)

    def test_replaying_the_same_frame_is_idempotent(self):
        rig = RecordingRig()
        animator = AvatarAnimator(rig, TIMELINE)
        animator.apply_at(3.3)
        once = rig.pose()
        for _ in range(5):
            animator.apply_at(3.3)
        self.assertEqual(rig.pose(), once)


class TestNoDriftAfterStalls(unittest.TestCase):
    """The specific failure the old accumulate-and-clamp design had."""

    def test_sway_has_no_accumulated_phase_error(self):
        """Sway is a function of playback time, not of summed frame deltas."""
        for name in ("stall 250 ms", "stall 500 ms", "stall 1000 ms"):
            schedule = SCHEDULES[name]
            probe = 11.5
            rig = RecordingRig()
            animator = AvatarAnimator(rig, TIMELINE)
            for frame in schedule:
                if frame >= probe:
                    break
                animator.apply_at(frame)
            state = animator.apply_at(probe)
            with self.subTest(schedule=name):
                self.assertAlmostEqual(
                    state.sway,
                    math.sin(2.0 * math.pi * probe / animator.sway_period),
                    places=12,
                )

    def test_first_frame_after_a_stall_shows_the_current_beat(self):
        animator = AvatarAnimator(RecordingRig(), TIMELINE)
        for stall in (0.25, 0.5, 1.0, 2.0):
            before = 3.01
            after = before + stall
            animator.apply_at(before)
            state = animator.apply_at(after)
            with self.subTest(stall=stall):
                self.assertEqual(state.beat_index, TIMELINE.index_before(after))
                self.assertAlmostEqual(
                    state.beat_age,
                    after - TIMELINE.cue_before(after).scheduled_time,
                    places=12,
                )
                self.assertLessEqual(
                    state.beat_age,
                    60.0 / BPM,
                    "state resumed on a stale beat instead of the current one",
                )

    def test_missed_beats_are_not_replayed_in_a_burst(self):
        """A stall must not leave a queue of beats to fire on the next frame."""
        animator = AvatarAnimator(RecordingRig(), TIMELINE)
        animator.apply_at(2.0)
        state = animator.apply_at(5.0)   # six beats went by with no frame
        self.assertEqual(state.beat_index, TIMELINE.index_before(5.0))
        self.assertLessEqual(state.impulse, 1.0)
        self.assertEqual(state.impulse, animator.state_at(5.0).impulse)


class TestTimelineIsStateless(unittest.TestCase):
    def test_lookups_do_not_depend_on_order(self):
        forwards = [TIMELINE.index_before(t) for t in PROBES]
        backwards = [TIMELINE.index_before(t) for t in reversed(PROBES)]
        self.assertEqual(forwards, list(reversed(backwards)))

    def test_repeated_lookup_is_stable(self):
        for _ in range(3):
            self.assertEqual(TIMELINE.index_before(4.44), TIMELINE.index_before(4.44))

    def test_before_the_first_beat_there_is_no_cue(self):
        self.assertIsNone(TIMELINE.cue_before(0.0))
        self.assertEqual(TIMELINE.index_before(0.0), -1)
        state = AvatarAnimator(RecordingRig(), TIMELINE).state_at(0.1)
        self.assertFalse(state.has_beat)
        self.assertEqual(state.impulse, 0.0)

    def test_window_lookup_is_half_open(self):
        cues = TIMELINE.cues_in(0.5, 1.5)
        self.assertEqual([c.scheduled_time for c in cues], [1.0, 1.5])


class TestRecorderSeparatesTheTwoConcepts(unittest.TestCase):
    """State currency and sample coverage must be reported separately."""

    def _run(self, schedule: list[float]) -> TimingRecorder:
        rig = RecordingRig()
        animator = AvatarAnimator(rig, TIMELINE)
        recorder = TimingRecorder(
            beat_times=BEATS, response_window=animator.response_window
        )
        for frame in schedule:
            recorder.record_sample(
                animator.apply_at(frame), observed_at=frame, wall_time=frame
            )
        return recorder

    def test_state_lag_is_zero_at_every_frame_rate(self):
        for name, schedule in SCHEDULES.items():
            summary = self._run(schedule).summary()
            with self.subTest(schedule=name):
                self.assertAlmostEqual(summary["state_lag"]["max_ms"], 0.0, places=6)

    def test_high_sample_rate_catches_every_beat(self):
        recorder = self._run(SCHEDULES["60 fps"])
        self.assertEqual(recorder.beats_missed, 0)
        self.assertEqual(recorder.summary()["beats_sampled"], len(BEATS))

    def test_sparse_sampling_misses_beats_while_state_stays_current(self):
        """The distinction, made concrete.

        Samples every 0.833 s against beats every 0.5 s: some response windows
        fall entirely between samples and are provably uncatchable. Coverage must
        report exactly those, while the pose at every later time is still exact.
        """
        window = AvatarAnimator(RecordingRig(), TIMELINE).response_window
        schedule = [0.833 * i for i in range(1, 16)]

        # Independent oracle: which beats had no sample inside their window.
        first, last = schedule[0], schedule[-1]
        oracle = [
            index
            for index, beat in enumerate(BEATS)
            if first <= beat <= last - window
            and not any(beat <= s <= beat + window for s in schedule)
        ]
        self.assertTrue(oracle, "fixture must actually miss something")

        recorder = self._run(schedule)
        self.assertEqual(recorder.missed_indices(), oracle)
        self.assertGreater(recorder.beats_missed, 0)
        self.assertGreater(recorder.summary()["beats_sampled"], 0)

        # Coverage suffered; musical state did not.
        self.assertAlmostEqual(recorder.summary()["state_lag"]["max_ms"], 0.0, places=6)
        rig = RecordingRig()
        animator = AvatarAnimator(rig, TIMELINE)
        for frame in schedule:
            animator.apply_at(frame)
        for probe in (11.0, 11.37, 11.5):
            rig.reset()
            animator.apply_at(probe)
            after_sparse = rig.pose()
            self.assertEqual(
                after_sparse,
                reference_pose(probe),
                f"state at {probe} was wrong after a sparse schedule",
            )

    def test_long_stalls_are_reported_as_missed(self):
        recorder = self._run(SCHEDULES["stall 1000 ms"])
        summary = recorder.summary()
        self.assertGreater(summary["beats_missed"], 0, "a 1 s stall should miss beats")
        self.assertAlmostEqual(summary["state_lag"]["max_ms"], 0.0, places=6)

    def test_beat_latency_stays_within_a_frame_at_60fps(self):
        latency = self._run(SCHEDULES["60 fps"]).summary()["beat_response_latency"]
        self.assertLess(latency["max_ms"], 1000.0 / 60.0 + 1.0)


class TestEndToEndSyntheticTiming(unittest.TestCase):
    """Generated audio -> real analysis -> timeline -> pose under a bad schedule."""

    @classmethod
    def setUpClass(cls):
        from ghost_in_the_deck.audio.analysis import analyse

        cls._tmp = tempfile.TemporaryDirectory()
        wav = make_beat_track(Path(cls._tmp.name) / "beat.wav", bpm=120.0, seconds=16.0)
        cls.features = analyse(wav)
        cls.timeline = BeatTimeline(cls.features)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_analysis_recovers_the_synthetic_tempo(self):
        self.assertAlmostEqual(self.features.bpm, 120.0, delta=6.0)
        self.assertGreater(len(self.timeline), 20)

    def test_cue_strength_follows_the_music(self):
        strengths = [self.timeline.cue(i).strength for i in range(len(self.timeline))]
        self.assertTrue(all(0.25 <= s <= 1.0 for s in strengths))
        self.assertTrue(any(s > 0.25 for s in strengths), "no cue responded to energy")

    def test_pose_is_schedule_independent_on_analysed_audio(self):
        rig = RecordingRig()
        animator = AvatarAnimator(rig, self.timeline)
        probes = [2.35, 5.05, 8.5, 11.1]

        for probe in probes:
            reference = RecordingRig()
            AvatarAnimator(reference, self.timeline).apply_at(probe)

            for name in ("60 fps", "5 fps", "stall 1000 ms"):
                rig.reset()
                animator = AvatarAnimator(rig, self.timeline)
                for frame in SCHEDULES[name]:
                    if frame >= probe:
                        break
                    animator.apply_at(frame)
                animator.apply_at(probe)
                with self.subTest(probe=probe, schedule=name):
                    self.assertEqual(rig.pose(), reference.pose())

    def test_every_beat_is_sampled_when_the_loop_keeps_up(self):
        rig = RecordingRig()
        animator = AvatarAnimator(rig, self.timeline)
        recorder = TimingRecorder(
            beat_times=self.features.beats, response_window=animator.response_window
        )
        for frame in steady(60.0, duration=self.features.duration_seconds):
            recorder.record_sample(
                animator.apply_at(frame), observed_at=frame, wall_time=frame
            )
        self.assertEqual(recorder.beats_missed, 0)


if __name__ == "__main__":
    unittest.main()
