"""Coverage accounting: which beats a run is responsible for, and which it caught.

The recorder is given the beat times up front, so coverage is a comparison
between the beats a run passed through and the beats a sample actually carried.
Inferring the range from successful samples instead is what previously hid every
beat missed before the first success, and reported zero when nothing was caught
at all.
"""

from __future__ import annotations

import unittest

from ghost_in_the_deck.animation.controller import MotionState
from ghost_in_the_deck.sync import TimingRecorder

WINDOW = 0.48
BEATS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def state_at(time: float, beats=BEATS) -> MotionState:
    """The MotionState a real animator would produce at ``time``."""
    index = -1
    age = float("inf")
    for i, beat in enumerate(beats):
        if beat <= time:
            index, age = i, time - beat
    return MotionState(time=time, impulse=0.0, sway=0.0, beat_index=index, beat_age=age)


def play(samples, beats=BEATS, window=WINDOW) -> TimingRecorder:
    recorder = TimingRecorder(beat_times=beats, response_window=window)
    for time in samples:
        recorder.record_sample(state_at(time, beats), observed_at=time, wall_time=time)
    return recorder


def expected_missed(samples, beats=BEATS, window=WINDOW) -> list[int]:
    """Independent brute-force oracle, deliberately not the recorder's algorithm.

    A beat is in scope when the run was sampling before it and still sampling
    once its window had passed; it is caught when any sample falls inside
    [beat, beat + window].
    """
    if not samples:
        return []
    first, last = samples[0], samples[-1]
    missed = []
    for index, beat in enumerate(beats):
        if beat < first or beat > last - window:
            continue
        if not any(beat <= s <= beat + window for s in samples):
            missed.append(index)
    return missed


class TestMissedBeatAccounting(unittest.TestCase):
    def test_beats_missed_before_the_first_success_are_counted(self):
        """The headline bug: nothing was sampled until beat 3."""
        # Nothing sampled between 0.1 and 2.02, then dense to the end.
        samples = [0.0, 0.1] + [2.02 + i / 60.0 for i in range(int(1.8 * 60))]
        recorder = play(samples)

        self.assertEqual([r.index for r in recorder.responses], [3, 4, 5])
        self.assertEqual(recorder.missed_indices(), [0, 1, 2])
        self.assertEqual(recorder.beats_missed, 3)
        self.assertEqual(recorder.missed_indices(), expected_missed(samples))

    def test_no_beat_sampled_at_all_still_reports_the_misses(self):
        """Previously returned zero, because the span began at the first success."""
        # One gap swallowing every beat, resuming past the last beat's window.
        samples = [0.0, 0.05, 3.6, 3.65]
        recorder = play(samples)

        self.assertEqual(recorder.responses, [])
        self.assertEqual(recorder.beats_missed, 6)
        self.assertEqual(recorder.missed_indices(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(recorder.missed_indices(), expected_missed(samples))

    def test_dense_sampling_misses_nothing(self):
        samples = [i / 60.0 for i in range(int(3.6 * 60))]
        recorder = play(samples)

        self.assertEqual(recorder.beats_missed, 0)
        self.assertEqual(len(recorder.responses), len(BEATS))
        self.assertEqual(recorder.missed_indices(), expected_missed(samples))

    def test_stall_spanning_several_beats_counts_each_one(self):
        dense_start = [i / 60.0 for i in range(0, 40)]        # 0 .. 0.65
        dense_end = [2.6 + i / 60.0 for i in range(0, 70)]    # 2.6 .. 3.75
        samples = dense_start + dense_end                     # 1.95 s gap

        recorder = play(samples)
        self.assertEqual(recorder.missed_indices(), [1, 2, 3])
        self.assertEqual(recorder.missed_indices(), expected_missed(samples))
        self.assertEqual([r.index for r in recorder.responses], [0, 4, 5])

    def test_scope_excludes_beats_outside_the_observed_span(self):
        """A run that starts late or stops early is not blamed for either end."""
        samples = [1.2 + i / 60.0 for i in range(int(1.3 * 60))]   # 1.2 .. 2.5
        recorder = play(samples)

        scope = recorder.beats_in_scope()
        self.assertNotIn(0, scope, "beat before the run began is not in scope")
        self.assertNotIn(1, scope, "beat before the run began is not in scope")
        self.assertNotIn(5, scope, "beat after the run stopped is not in scope")
        self.assertEqual(recorder.missed_indices(), expected_missed(samples))

    def test_summary_reports_scope_alongside_coverage(self):
        samples = [0.0, 0.1] + [2.02 + i / 60.0 for i in range(int(1.8 * 60))]
        summary = play(samples).summary()
        self.assertEqual(summary["beats_sampled"], 3)
        self.assertEqual(summary["beats_missed"], 3)
        self.assertEqual(summary["beats_in_scope"], 6)

    def test_no_samples_at_all_is_not_a_crash(self):
        recorder = TimingRecorder(beat_times=BEATS, response_window=WINDOW)
        self.assertEqual(recorder.beats_in_scope(), [])
        self.assertEqual(recorder.beats_missed, 0)
        self.assertEqual(recorder.summary()["update_samples"], 0)

    def test_recorder_without_beat_times_reports_no_scope(self):
        """Coverage needs the beat list; without it nothing is claimed."""
        recorder = TimingRecorder()
        recorder.record_sample(state_at(1.02), observed_at=1.02, wall_time=1.02)
        self.assertEqual(recorder.beats_missed, 0)
        self.assertEqual(recorder.summary()["beats_in_scope"], 0)


class TestClockSkewDiagnostic(unittest.TestCase):
    """A stall must not fabricate skew between clocks that moved together."""

    def test_a_one_second_stall_does_not_invent_a_one_second_skew(self):
        recorder = TimingRecorder(beat_times=BEATS, response_window=WINDOW)
        # Playback and wall advance identically, including across a 1 s freeze.
        offsets = [0.0, 0.016, 0.032, 1.032, 1.048, 1.064]
        for offset in offsets:
            recorder.record_sample(
                state_at(offset), observed_at=offset, wall_time=100.0 + offset
            )

        skew = recorder.summary()["playback_vs_wall_skew"]
        self.assertEqual(skew["count"], len(offsets) - 1)
        self.assertLess(skew["max_ms"], 1e-6, f"fabricated skew: {skew}")

    def test_real_divergence_is_still_reported(self):
        recorder = TimingRecorder(beat_times=BEATS, response_window=WINDOW)
        recorder.record_sample(state_at(0.0), observed_at=0.0, wall_time=100.0)
        # Playback advanced 0.5 s while the wall advanced 0.6 s.
        recorder.record_sample(state_at(0.5), observed_at=0.5, wall_time=100.6)
        self.assertAlmostEqual(
            recorder.summary()["playback_vs_wall_skew"]["max_ms"], 100.0, places=3
        )

    def test_skew_needs_both_readings(self):
        recorder = TimingRecorder(beat_times=BEATS, response_window=WINDOW)
        recorder.record_sample(state_at(0.0), observed_at=0.0)
        recorder.record_sample(state_at(0.5), observed_at=0.5)
        self.assertEqual(recorder.summary()["playback_vs_wall_skew"], {"count": 0})
        self.assertEqual(recorder.summary()["sample_interval"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
