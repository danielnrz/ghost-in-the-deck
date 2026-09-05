"""Phase 1C: the hand_to_deck gesture's audio effect.

Everything here runs on generated signals and a hand-built event schedule, so
it needs no private audio.
"""

from __future__ import annotations

import unittest

import numpy as np

from ghost_in_the_deck.animation.dj_behavior import GestureEvent
from ghost_in_the_deck.audio.effects import apply_hand_to_deck_effects

from synthetic import make_broadband_signal

SAMPLE_RATE = 44100
LOW_BAND = (20.0, 200.0)
HIGH_BAND = (6000.0, 15000.0)


def band_energy(signal: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    """Total power in one channel's spectrum within [low, high) Hz."""
    spectrum = np.abs(np.fft.rfft(signal[:, 0]))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / sample_rate)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(spectrum[mask] ** 2))


class TestDeterminism(unittest.TestCase):
    def test_same_input_and_schedule_gives_byte_identical_output(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 1.2, "hand_to_deck", "l", 0.85),
            GestureEvent(2.6, 0.9, "hand_to_deck", "r", 0.7),
        ]
        first = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        second = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(first, second))

    def test_input_array_is_not_mutated(self):
        signal = make_broadband_signal(3.0, sr=SAMPLE_RATE)
        original = signal.copy()
        events = [GestureEvent(1.0, 1.2, "hand_to_deck", "l", 0.9)]
        apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(signal, original))


class TestEffectIsRealAndLocated(unittest.TestCase):
    """A measurable spectral change appears inside the window, in the right direction."""

    def _window_slice(self, signal, start, duration, pad=4000):
        start_idx = round(start * SAMPLE_RATE)
        end_idx = start_idx + round(duration * SAMPLE_RATE)
        mid = (start_idx + end_idx) // 2
        return slice(mid - pad, mid + pad)

    def test_left_hand_lowpass_cuts_highs_inside_the_window(self):
        signal = make_broadband_signal(6.0, sr=SAMPLE_RATE)
        event = GestureEvent(2.0, 1.2, "hand_to_deck", "l", 1.0)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [event])

        inside = self._window_slice(signal, 2.0, 1.2)
        hi_before = band_energy(signal[inside], SAMPLE_RATE, *HIGH_BAND)
        hi_after = band_energy(processed[inside], SAMPLE_RATE, *HIGH_BAND)
        self.assertLess(
            hi_after / hi_before, 0.3,
            "left hand_to_deck did not measurably cut high-frequency energy",
        )

    def test_right_hand_highpass_cuts_lows_inside_the_window(self):
        signal = make_broadband_signal(6.0, sr=SAMPLE_RATE)
        event = GestureEvent(2.0, 1.2, "hand_to_deck", "r", 1.0)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [event])

        inside = self._window_slice(signal, 2.0, 1.2)
        lo_before = band_energy(signal[inside], SAMPLE_RATE, *LOW_BAND)
        lo_after = band_energy(processed[inside], SAMPLE_RATE, *LOW_BAND)
        self.assertLess(
            lo_after / lo_before, 0.3,
            "right hand_to_deck did not measurably cut low-frequency energy",
        )

    def test_left_hand_effect_is_absent_well_outside_the_window(self):
        signal = make_broadband_signal(8.0, sr=SAMPLE_RATE)
        event = GestureEvent(4.0, 1.2, "hand_to_deck", "l", 1.0)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [event])

        far = slice(0, 8000)  # well before the 4.0 s start
        hi_before = band_energy(signal[far], SAMPLE_RATE, *HIGH_BAND)
        hi_after = band_energy(processed[far], SAMPLE_RATE, *HIGH_BAND)
        self.assertAlmostEqual(hi_after / hi_before, 1.0, places=9)

    def test_right_hand_effect_is_absent_well_outside_the_window(self):
        signal = make_broadband_signal(8.0, sr=SAMPLE_RATE)
        event = GestureEvent(4.0, 1.2, "hand_to_deck", "r", 1.0)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [event])

        far = slice(0, 8000)
        lo_before = band_energy(signal[far], SAMPLE_RATE, *LOW_BAND)
        lo_after = band_energy(processed[far], SAMPLE_RATE, *LOW_BAND)
        self.assertAlmostEqual(lo_after / lo_before, 1.0, places=9)


class TestNoLeakage(unittest.TestCase):
    def test_samples_outside_every_window_are_bit_identical(self):
        signal = make_broadband_signal(10.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 1.2, "hand_to_deck", "l", 0.9),
            GestureEvent(3.5, 0.8, "hand_to_deck", "r", 0.6),
            GestureEvent(6.0, 1.1, "hand_to_deck", "l", 1.0),
        ]
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)

        touched = np.zeros(signal.shape[0], dtype=bool)
        for event in events:
            start_idx = round(event.start * SAMPLE_RATE)
            end_idx = start_idx + round(event.duration * SAMPLE_RATE) + 1
            touched[start_idx:end_idx] = True

        outside = ~touched
        self.assertTrue(np.array_equal(processed[outside], signal[outside]))

    def test_non_hand_to_deck_events_are_ignored(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 1.2, "deck_glance", None, 0.9),
            GestureEvent(2.5, 1.0, "lean_in", None, 0.9),
            GestureEvent(3.0, 0.5, "small_hype", "l", 0.9),
        ]
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        self.assertTrue(np.array_equal(processed, signal))

    def test_no_scheduled_hand_to_deck_events_is_fully_inert(self):
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [])
        self.assertTrue(np.array_equal(processed, signal))

    def test_no_scheduled_events_preserves_dtype_and_is_byte_identical(self):
        """Equal values are not enough: a float32 track with nothing scheduled
        must come back as float32, byte-for-byte, not silently upcast to
        float64 and doubled in storage."""
        signal = make_broadband_signal(4.0, sr=SAMPLE_RATE).astype(np.float32)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [])
        self.assertEqual(processed.dtype, signal.dtype)
        self.assertEqual(processed.tobytes(), signal.tobytes())


class TestNoBoundaryClick(unittest.TestCase):
    def test_contribution_is_exactly_zero_at_start_and_end_of_every_event(self):
        signal = make_broadband_signal(10.0, sr=SAMPLE_RATE)
        events = [
            GestureEvent(1.0, 1.2, "hand_to_deck", "l", 0.9),
            GestureEvent(3.5, 0.8, "hand_to_deck", "r", 0.6),
            GestureEvent(6.0, 1.1, "hand_to_deck", "l", 1.0),
            GestureEvent(8.0, 0.9, "hand_to_deck", "r", 1.0),
        ]
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)

        for event in events:
            start_idx = round(event.start * SAMPLE_RATE)
            end_idx = start_idx + round(event.duration * SAMPLE_RATE)
            with self.subTest(start=event.start, side=event.side):
                self.assertTrue(np.array_equal(processed[start_idx], signal[start_idx]))
                self.assertTrue(np.array_equal(processed[end_idx], signal[end_idx]))

    def test_highpass_release_has_no_step_discontinuity_before_the_pin(self):
        """The pinned last sample of a window must be *reached* smoothly, not
        jumped to. A slowly-varying dry signal changes by almost nothing from
        one sample to the next, so the processed signal's own step into the
        pinned endpoint should match that, not the multi-percent jump an
        unconditional ``y[-1] = x[-1]`` produces once the high-pass recursion
        has drifted away from the dry signal."""
        sr = SAMPLE_RATE
        t = np.arange(int(2.5 * sr)) / sr
        signal = (0.8 * np.sin(2 * np.pi * 20.0 * t))[:, None]
        event = GestureEvent(1.0, 1.2, "hand_to_deck", "r", 1.0)
        processed = apply_hand_to_deck_effects(signal, sr, [event])

        end_idx = round(1.0 * sr) + round(1.2 * sr)
        dry_step = signal[end_idx, 0] - signal[end_idx - 1, 0]
        wet_step = processed[end_idx, 0] - processed[end_idx - 1, 0]
        self.assertLess(
            abs(wet_step - dry_step), 1e-4,
            f"boundary step {wet_step:.6f} does not match the dry step {dry_step:.6f}",
        )


class TestSignalIntegrity(unittest.TestCase):
    """Population sweep: several events, both sides, a range of strengths."""

    def test_no_clipping_overflow_or_non_finite_values_across_a_population(self):
        signal = make_broadband_signal(30.0, sr=SAMPLE_RATE)
        events = []
        start = 1.0
        for i in range(18):
            side = "l" if i % 2 == 0 else "r"
            strength = 0.6 + 0.4 * ((i * 7) % 10) / 10.0
            duration = 0.8 + 0.5 * ((i * 3) % 5) / 5.0
            events.append(GestureEvent(start, duration, "hand_to_deck", side, strength))
            start += 1.4

        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)

        self.assertTrue(np.all(np.isfinite(processed)), "non-finite sample present")
        peak_in = float(np.max(np.abs(signal)))
        peak_out = float(np.max(np.abs(processed)))
        # The effect must never introduce a peak beyond what the dry audio
        # already required to represent (or the valid PCM ceiling, if the dry
        # audio happens to already exceed it, as this synthetic noise does) -
        # not merely "close to" the input peak.
        limit = max(peak_in, 1.0)
        self.assertLessEqual(
            peak_out, limit + 1e-9,
            f"processed peak {peak_out:.6f} exceeds the {limit:.6f} the dry audio used",
        )

    def test_highpass_never_exceeds_valid_pcm_range_on_a_normalised_signal(self):
        """Reproduces the reviewer's finding: real scheduled events on audio
        that is already normalised to +-1 must not push samples outside it,
        since app.py writes the result as PCM_16 and silently saturates."""
        rng = np.random.default_rng(4)
        signal = rng.normal(0.0, 0.3, (int(2.5 * SAMPLE_RATE), 2))
        signal = signal / np.max(np.abs(signal)) * 0.999
        events = [
            GestureEvent(0.5, 1.0, "hand_to_deck", "r", 1.0),
            GestureEvent(1.8, 0.9, "hand_to_deck", "l", 0.8),
        ]
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, events)
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0 + 1e-9)

    def test_highpass_survives_a_full_scale_step_without_overshoot(self):
        n = int(2.5 * SAMPLE_RATE)
        signal = np.concatenate([-np.ones(n // 2), np.ones(n - n // 2)])[:, None]
        event = GestureEvent(1.0, 1.2, "hand_to_deck", "r", 1.0)
        processed = apply_hand_to_deck_effects(signal, SAMPLE_RATE, [event])
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
