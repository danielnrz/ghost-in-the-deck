"""Synthetic tests for the offline incoming transition material boundary."""

from __future__ import annotations

import numpy as np

from ghost_in_the_deck.audio.tempo_match import stretch_incoming_transition
from ghost_in_the_deck.transition import TransitionPlan


SAMPLE_RATE = 8000


def _plan(*, bpm_a: float = 120.0, bpm_b: float = 150.0) -> TransitionPlan:
    return TransitionPlan(
        outgoing_track="synthetic-outgoing",
        incoming_track="synthetic-incoming",
        outgoing_time=0.0,
        incoming_time=0.5,
        outgoing_bar_index=1,
        incoming_bar_index=2,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        bpm_ratio=bpm_b / bpm_a,
        transition_length_bars=8,
        outgoing_duration_seconds=1.25,
        incoming_duration_seconds=1.0,
        score=1.0,
        reason="synthetic fixture",
    )


def test_only_incoming_material_is_stretched_and_cue_stays_anchored():
    plan = _plan()
    anchor = int(plan.incoming_time * SAMPLE_RATE)
    source_count = int(plan.incoming_duration_seconds * SAMPLE_RATE)
    output_count = int(plan.outgoing_duration_seconds * SAMPLE_RATE)
    samples = np.arange(3 * SAMPLE_RATE, dtype=np.float64)
    original = samples.copy()

    result = stretch_incoming_transition(samples, plan, SAMPLE_RATE)

    assert result.shape == (anchor + output_count + len(samples) - anchor - source_count,)
    np.testing.assert_array_equal(result[:anchor], original[:anchor])
    np.testing.assert_array_equal(
        result[anchor + output_count :], original[anchor + source_count :]
    )
    assert np.array_equal(samples, original)
    assert not np.shares_memory(result, samples)


def test_pitch_is_preserved_while_duration_matches_authoritative_window():
    plan = _plan()
    anchor = int(plan.incoming_time * SAMPLE_RATE)
    source_count = int(plan.incoming_duration_seconds * SAMPLE_RATE)
    output_count = int(plan.outgoing_duration_seconds * SAMPLE_RATE)
    timeline = np.arange(3 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    source = np.sin(2.0 * np.pi * 440.0 * timeline)

    result = stretch_incoming_transition(source, plan, SAMPLE_RATE)
    transformed = result[anchor : anchor + output_count]
    analysis_window = transformed[SAMPLE_RATE // 8 : -SAMPLE_RATE // 8]
    frequencies = np.fft.rfftfreq(len(analysis_window), 1.0 / SAMPLE_RATE)
    peak_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(analysis_window)))]

    assert len(transformed) == output_count
    assert len(source[anchor : anchor + source_count]) == source_count
    assert abs(peak_frequency - 440.0) <= 8.0


def test_transform_is_repeatable_for_same_pcm_and_plan():
    plan = _plan(bpm_a=120.0, bpm_b=100.0)
    timeline = np.arange(4 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    source = np.sin(2.0 * np.pi * 330.0 * timeline)

    first = stretch_incoming_transition(source, plan, SAMPLE_RATE)
    second = stretch_incoming_transition(source, plan, SAMPLE_RATE)

    np.testing.assert_array_equal(first, second)
