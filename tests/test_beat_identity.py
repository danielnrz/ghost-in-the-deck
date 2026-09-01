"""N1: real vs virtual beat identity.

BeatTimeline.phase_at() keeps ``index`` moving outside the detected beats so the
rhythm stays continuous through an intro or an outro - that behaviour is correct
and must stay. The bug was inferring "this is a real, detected beat" from
``index >= 0``. Post-final virtual beats continue counting upward from the last
detected index, so they are positive too, and a track's outro could report a
beat response that had never actually been detected.

``BeatPhase.is_real`` / ``GrooveState.has_detected_beat`` are the explicit fix:
identity no longer comes from the sign of a number that also serves a different
purpose (grid position).
"""

from __future__ import annotations

import unittest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.animation.groove import GrooveEngine
from ghost_in_the_deck.sync import TimingRecorder

from synthetic import make_features


def timeline_of(beats, duration=None):
    if duration is None:
        duration = (beats[-1] if beats else 0.0) + 6.0
    return BeatTimeline(make_features(beats, duration=duration))


class TestBeatPhaseIdentity(unittest.TestCase):
    """The nine scenarios required by the review, against BeatPhase directly."""

    def test_1_empty_timeline_is_never_real(self):
        timeline = timeline_of([], duration=10.0)
        for time in (0.0, 0.3, 1.7, 5.0, 9.9):
            with self.subTest(time=time):
                self.assertFalse(timeline.phase_at(time).is_real)

    def test_2_before_the_first_real_beat_is_not_real(self):
        timeline = timeline_of([2.0, 2.5, 3.0, 3.5])
        for time in (0.0, 0.5, 1.0, 1.4999, 1.9999):
            with self.subTest(time=time):
                phase = timeline.phase_at(time)
                self.assertFalse(phase.is_real)
                self.assertLess(phase.index, 0)

    def test_3_each_real_beat_is_real_at_its_own_timestamp(self):
        beats = [2.0, 2.5, 3.0, 3.5, 4.0]
        timeline = timeline_of(beats)
        for index, time in enumerate(beats):
            with self.subTest(index=index, time=time):
                phase = timeline.phase_at(time)
                self.assertTrue(phase.is_real)
                self.assertEqual(phase.index, index)

    def test_4_between_real_beats_is_real(self):
        beats = [2.0, 2.5, 3.0, 3.5]
        timeline = timeline_of(beats)
        for time, expected_index in ((2.1, 0), (2.4, 0), (2.9, 1), (3.49, 2)):
            with self.subTest(time=time):
                phase = timeline.phase_at(time)
                self.assertTrue(phase.is_real)
                self.assertEqual(phase.index, expected_index)

    def test_5_exactly_the_last_real_beat_is_real(self):
        beats = [2.0, 2.5, 3.0]
        timeline = timeline_of(beats)
        phase = timeline.phase_at(3.0)
        self.assertTrue(phase.is_real)
        self.assertEqual(phase.index, 2)
        # And the whole final beat's window up to (not including) the next
        # grid line stays real too.
        for time in (3.01, 3.2, 3.499999):
            with self.subTest(time=time):
                self.assertTrue(timeline.phase_at(time).is_real)

    def test_6_first_post_final_virtual_beat_is_not_real(self):
        beats = [2.0, 2.5, 3.0]   # nominal interval 0.5, last real beat at 3.0
        timeline = timeline_of(beats)
        phase = timeline.phase_at(3.5)   # one interval past the last real beat
        self.assertFalse(phase.is_real)
        self.assertEqual(phase.index, 3)   # continues counting, but not real

    def test_7_multiple_post_final_virtual_beats_stay_unreal(self):
        beats = [2.0, 2.5, 3.0]
        timeline = timeline_of(beats, duration=20.0)
        for steps_past in range(1, 8):
            time = 3.0 + steps_past * 0.5
            with self.subTest(steps_past=steps_past):
                phase = timeline.phase_at(time)
                self.assertFalse(phase.is_real)
                self.assertEqual(phase.index, 2 + steps_past)

    def test_8_fast_track_virtual_outro_inside_response_window(self):
        """The exact scenario the review demonstrated: interval < response_window.

        With a 200 BPM grid (0.3 s apart) and the default 0.48 s response
        window, the first virtual beat after the last real one is only 0.3 s
        past it - well inside the window that would normally mark a response as
        current. It must still be flagged unreal.
        """
        beats = [1.0, 1.3, 1.6]
        engine = GrooveEngine(make_features(beats, duration=6.0, bpm=200.0), seed="n1-f8")
        self.assertLess(engine.timeline.nominal_interval, engine.response_window)

        virtual_time = 1.6 + engine.timeline.nominal_interval + 1e-4
        state = engine.state_at(virtual_time)
        self.assertFalse(state.has_detected_beat)
        self.assertLessEqual(state.beat_age, engine.response_window,
                             "fixture must actually land inside the window")

    def test_9_synchronization_summary_reports_exactly_n_beats(self):
        """The end-to-end proof: coverage must equal the real beat count."""
        for n, bpm in ((3, 200.0), (5, 180.0), (8, 150.0), (24, 120.0)):
            beats = [1.0 + i * (60.0 / bpm) for i in range(n)]
            engine = GrooveEngine(make_features(beats, duration=beats[-1] + 6.0, bpm=bpm),
                                  seed=f"n1-f9-{n}")
            recorder = TimingRecorder(beat_times=beats, response_window=engine.response_window)
            time = 0.0
            while time <= beats[-1] + 5.0:
                recorder.record_sample(engine.state_at(time), observed_at=time, wall_time=time)
                time += 0.01

            summary = recorder.summary()
            with self.subTest(n=n, bpm=bpm):
                self.assertEqual(summary["beats_sampled"], n)
                self.assertEqual(len(recorder.responses), n)
                self.assertEqual(
                    sorted(r.index for r in recorder.responses), list(range(n))
                )


class TestPhantomBeatReproduction(unittest.TestCase):
    """The reviewer's exact case: a three-beat timeline reporting a fourth beat."""

    def build(self):
        beats = [1.0, 1.3, 1.6]   # 200 BPM
        engine = GrooveEngine(make_features(beats, duration=6.0, bpm=200.0), seed="n1-repro")
        recorder = TimingRecorder(beat_times=beats, response_window=engine.response_window)
        time = 0.0
        while time <= 4.0:
            recorder.record_sample(engine.state_at(time), observed_at=time, wall_time=time)
            time += 0.01
        return engine, recorder

    def test_no_phantom_fourth_beat(self):
        engine, recorder = self.build()
        summary = recorder.summary()

        self.assertEqual(len(engine.timeline), 3, "fixture must have exactly 3 real beats")
        self.assertEqual(summary["beats_sampled"], 3,
                         f"phantom beat reintroduced: {summary['beats_sampled']} != 3")
        self.assertEqual(summary["beats_in_scope"], 3)
        self.assertEqual(sorted(r.index for r in recorder.responses), [0, 1, 2])

    def test_the_virtual_fourth_index_is_visible_but_not_a_response(self):
        """The grid still counts to a virtual beat 3; it just is not detected."""
        engine, recorder = self.build()
        state = engine.state_at(1.95)   # clearly past the 1.9 virtual boundary
        self.assertEqual(state.beat_index, 3)
        self.assertFalse(state.has_detected_beat)
        self.assertNotIn(3, [r.index for r in recorder.responses])


class TestGrooveContinuesThroughVirtualBeats(unittest.TestCase):
    """Required: rhythm/bar phase still works through virtual beats; pulse does not."""

    def test_bar_phase_advances_through_a_virtual_post_final_beat(self):
        beats = [1.0, 1.5, 2.0]
        engine = GrooveEngine(make_features(beats, duration=8.0, bpm=120.0), seed="n1-cont")
        phases = [engine.timeline.phase_at(t).bar_phase for t in (2.6, 2.7, 2.8, 2.9)]
        self.assertEqual(len(set(round(p, 6) for p in phases)), 4,
                         "bar phase stalled once the beats ran out")

    def test_pulse_never_fires_from_a_virtual_beat(self):
        """MotionCues only exist for real beats, so a virtual beat has no accent."""
        beats = [1.0, 1.5, 2.0]
        engine = GrooveEngine(make_features(beats, duration=8.0, bpm=120.0), seed="n1-pulse")
        # Long after the last real cue's accent has fully decayed, but still
        # inside where a phantom "beat 3" would sit.
        state = engine.state_at(2.5 + 8 * engine.pulse_decay)
        self.assertEqual(state.pulse, 0.0)
        self.assertFalse(state.has_detected_beat)

    def test_sway_and_weight_shift_are_unaffected_by_the_identity_fix(self):
        """These read bar_index/bar_phase, never has_detected_beat."""
        beats = [1.0, 1.5, 2.0]
        engine = GrooveEngine(make_features(beats, duration=8.0, bpm=120.0), seed="n1-sway")
        for time in (0.3, 1.2, 2.7, 4.0):
            state = engine.state_at(time)
            self.assertTrue(-1.0 <= state.sway <= 1.0)
            self.assertTrue(-1.0 <= state.weight_shift <= 1.0)


class TestHasBeatCompatibilityAlias(unittest.TestCase):
    """has_beat must not silently regress back to the old, wrong semantics."""

    def test_has_beat_agrees_with_has_detected_beat(self):
        beats = [1.0, 1.5, 2.0]
        engine = GrooveEngine(make_features(beats, duration=8.0, bpm=120.0), seed="n1-alias")
        for time in (0.2, 0.8, 1.2, 1.9, 2.4, 3.0, 5.0):
            state = engine.state_at(time)
            with self.subTest(time=time):
                self.assertEqual(state.has_beat, state.has_detected_beat)


if __name__ == "__main__":
    unittest.main()
