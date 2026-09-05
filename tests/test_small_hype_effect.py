"""Phase 1D: the small_hype gesture's gain-riser audio effect.

Everything here runs on generated signals and hand-built event schedules, so
it needs no private audio - the same approach test_audio_effects.py uses for
the hand_to_deck filter sweep.
"""

from __future__ import annotations

import unittest

import numpy as np

from ghost_in_the_deck.animation.dj_behavior import GestureEvent
from ghost_in_the_deck.audio.effects import (
    GAIN_RISER_CEILING,
    apply_hand_to_deck_effects,
    apply_small_hype_effects,
)

from synthetic import make_broadband_signal

SAMPLE_RATE = 44100


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


class TestDeterminism(unittest.TestCase):
    def test_same_input_and_schedule_gives_byte_identical_output(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 0.9, "small_hype", "l", 0.85),
            GestureEvent(2.6, 0.7, "small_hype", "r", 0.6),
        ]
        first = apply_small_hype_effects(signal, SAMPLE_RATE, events)
        second = apply_small_hype_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(first, second))

    def test_input_array_is_not_mutated(self):
        signal = make_broadband_signal(3.0, sr=SAMPLE_RATE)
        original = signal.copy()
        events = [GestureEvent(1.0, 0.9, "small_hype", "l", 0.9)]
        apply_small_hype_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(signal, original))


class TestEffectIsRealAndLocated(unittest.TestCase):
    """A measurable amplitude increase appears inside the window, and only there."""

    def _hold_slice(self, start: float, duration: float, attack: float, hold: float, pad=1500):
        """Samples deep in the event's hold, where weight is exactly 1.0."""
        mid_hold = start + duration * (attack + hold * 0.5)
        mid_idx = round(mid_hold * SAMPLE_RATE)
        return slice(max(0, mid_idx - pad), mid_idx + pad)

    def test_gain_increases_energy_inside_the_hold(self):
        signal = make_broadband_signal(6.0, sr=SAMPLE_RATE)
        event = GestureEvent(2.0, 0.7, "small_hype", "l", 1.0)
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, [event])

        # small_hype's ENVELOPE_SHAPE is (0.25, 0.15, 0.60): attack 0.25, hold 0.15.
        inside = self._hold_slice(2.0, 0.7, 0.25, 0.15)
        rms_before = rms(signal[inside])
        rms_after = rms(processed[inside])
        self.assertGreater(
            rms_after / rms_before, 1.05,
            "small_hype riser did not measurably raise amplitude inside its hold",
        )

    def test_effect_is_absent_well_outside_the_window(self):
        signal = make_broadband_signal(8.0, sr=SAMPLE_RATE)
        event = GestureEvent(4.0, 0.7, "small_hype", "l", 1.0)
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, [event])

        far = slice(0, 8000)  # well before the 4.0 s start
        self.assertTrue(np.array_equal(processed[far], signal[far]))


class TestNoLeakage(unittest.TestCase):
    def test_samples_outside_every_window_are_bit_identical(self):
        signal = make_broadband_signal(10.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 0.7, "small_hype", "l", 0.9),
            GestureEvent(3.5, 0.6, "small_hype", "r", 0.6),
            GestureEvent(6.0, 0.8, "small_hype", "l", 1.0),
        ]
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, events)

        touched = np.zeros(signal.shape[0], dtype=bool)
        for event in events:
            start_idx = round(event.start * SAMPLE_RATE)
            end_idx = start_idx + round(event.duration * SAMPLE_RATE) + 1
            touched[start_idx:end_idx] = True

        outside = ~touched
        self.assertTrue(np.array_equal(processed[outside], signal[outside]))

    def test_non_small_hype_events_are_ignored(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 1.2, "deck_glance", None, 0.9),
            GestureEvent(2.5, 1.0, "lean_in", None, 0.9),
            GestureEvent(3.0, 0.5, "hand_to_deck", "l", 0.9),
        ]
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(processed, signal))

    def test_no_scheduled_small_hype_events_is_fully_inert(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, [])
        self.assertTrue(np.array_equal(processed, signal))

    def test_no_scheduled_events_preserves_dtype_and_is_byte_identical(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE).astype(np.float32)
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, [])
        self.assertEqual(processed.dtype, signal.dtype)
        self.assertEqual(processed.tobytes(), signal.tobytes())


class TestNoBoundaryClick(unittest.TestCase):
    def test_contribution_is_exactly_identity_at_start_and_end_of_every_event(self):
        signal = make_broadband_signal(10.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 0.7, "small_hype", "l", 0.9),
            GestureEvent(3.5, 0.6, "small_hype", "r", 0.6),
            GestureEvent(6.0, 0.8, "small_hype", "l", 1.0),
            GestureEvent(8.0, 0.9, "small_hype", "r", 1.0),
        ]
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, events)

        for event in events:
            start_idx = round(event.start * SAMPLE_RATE)
            end_idx = start_idx + round(event.duration * SAMPLE_RATE)
            with self.subTest(start=event.start, side=event.side):
                self.assertTrue(np.array_equal(processed[start_idx], signal[start_idx]))
                self.assertTrue(np.array_equal(processed[end_idx], signal[end_idx]))


class TestSignalIntegrity(unittest.TestCase):
    """Population sweep: several events, both sides, a range of strengths and seeds."""

    def test_no_clipping_overflow_or_non_finite_values_across_a_population(self):
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            signal = make_broadband_signal(20.0, sr=SAMPLE_RATE, seed=seed + 1)
            events = []
            start = 1.0
            for i in range(12):
                side = "l" if i % 2 == 0 else "r"
                strength = float(0.5 + 0.5 * rng.random())
                duration = float(0.5 + 0.4 * rng.random())
                events.append(GestureEvent(start, duration, "small_hype", side, strength))
                start += 1.5

            with self.subTest(seed=seed):
                processed = apply_small_hype_effects(signal, SAMPLE_RATE, events)
                self.assertTrue(np.all(np.isfinite(processed)), "non-finite sample present")
                peak_out = float(np.max(np.abs(processed)))
                ceiling = max(GAIN_RISER_CEILING, float(np.max(np.abs(signal))))
                self.assertLessEqual(
                    peak_out, ceiling + 1e-9,
                    f"processed peak {peak_out:.6f} exceeds ceiling {ceiling:.6f}",
                )

    def test_headroom_bound_on_near_full_scale_input(self):
        """A hot, near-full-scale synthetic signal (not a quiet cherry-picked
        case) must still come out under GAIN_RISER_CEILING at maximum boost -
        the case that would clip under a naive fixed dB multiply."""
        sr = SAMPLE_RATE
        t = np.arange(int(2.5 * sr)) / sr
        # -0.1 dBFS: 10 ** (-0.1 / 20) ~= 0.9886
        hot = (0.9886 * np.sin(2 * np.pi * 300.0 * t))[:, None]
        event = GestureEvent(1.0, 0.7, "small_hype", "l", 1.0)
        processed = apply_small_hype_effects(hot, sr, [event])
        self.assertTrue(np.all(np.isfinite(processed)))
        self.assertLessEqual(float(np.max(np.abs(processed))), GAIN_RISER_CEILING + 1e-9)

    def test_zero_strength_is_an_exact_identity(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        event = GestureEvent(1.0, 0.7, "small_hype", "l", 0.0)
        processed = apply_small_hype_effects(signal, SAMPLE_RATE, [event])
        self.assertTrue(np.array_equal(processed, signal))


class TestCombinedWithHandToDeck(unittest.TestCase):
    """Both effects scheduled on the same track, in non-overlapping windows."""

    def test_both_effects_present_and_neither_corrupts_the_others_window(self):
        # Scaled down from the raw broadband fixture so the small_hype window
        # has real headroom to boost into - the raw fixture's noise floor
        # occasionally peaks just over 1.0 within any given 0.7s window,
        # which would (correctly, by the headroom bound) leave zero room for
        # a boost and make this assertion flaky by chance of window placement.
        signal = make_broadband_signal(10.0, sr=SAMPLE_RATE) * 0.5
        events = [
            GestureEvent(1.0, 1.2, "hand_to_deck", "l", 0.9),
            GestureEvent(5.0, 0.7, "small_hype", "r", 1.0),
        ]

        combined = apply_small_hype_effects(
            apply_hand_to_deck_effects(signal, SAMPLE_RATE, events), SAMPLE_RATE, events
        )
        hand_only = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        hype_only = apply_small_hype_effects(signal, SAMPLE_RATE, events)

        htd_start = round(1.0 * SAMPLE_RATE)
        htd_end = htd_start + round(1.2 * SAMPLE_RATE) + 1
        hype_start = round(5.0 * SAMPLE_RATE)
        hype_end = hype_start + round(0.7 * SAMPLE_RATE) + 1

        # The hand_to_deck window in the combined render is untouched by the
        # small_hype pass, since the two windows never overlap.
        self.assertTrue(np.array_equal(
            combined[htd_start:htd_end], hand_only[htd_start:htd_end]
        ))
        # The small_hype window in the combined render sees the same dry input
        # the standalone render did, since hand_to_deck never touched it.
        self.assertTrue(np.array_equal(
            combined[hype_start:hype_end], hype_only[hype_start:hype_end]
        ))

        # Both effects actually did something inside their own windows.
        self.assertFalse(np.array_equal(
            hand_only[htd_start:htd_end], signal[htd_start:htd_end]
        ))
        self.assertFalse(np.array_equal(
            hype_only[hype_start:hype_end], signal[hype_start:hype_end]
        ))


if __name__ == "__main__":
    unittest.main()
