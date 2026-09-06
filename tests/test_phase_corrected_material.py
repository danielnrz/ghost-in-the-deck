"""Synthetic tests for the Phase 4E corrected incoming-material boundary."""

from __future__ import annotations

import numpy as np
import pytest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.audio.phase_corrected_material import (
    apply_initial_incoming_cue_correction,
    phase_correct_incoming_transition,
    phase_relationship_for_match,
)
from ghost_in_the_deck.audio.tempo_match import tempo_match
from ghost_in_the_deck.transition import TransitionPlan

from synthetic import make_features


def _timeline(beats: list[float], *, bpm: float = 120.0) -> BeatTimeline:
    return BeatTimeline(make_features(beats, duration=8.0, bpm=bpm))


def _plan(
    *,
    bpm_a: float = 120.0,
    bpm_b: float = 120.0,
    outgoing_time: float = 1.0,
    incoming_time: float = 1.0,
) -> TransitionPlan:
    return TransitionPlan(
        outgoing_track="synthetic-outgoing",
        incoming_track="synthetic-incoming",
        outgoing_time=outgoing_time,
        incoming_time=incoming_time,
        outgoing_bar_index=1,
        incoming_bar_index=1,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        bpm_ratio=bpm_b / bpm_a,
        transition_length_bars=1,
        outgoing_duration_seconds=1.5,
        incoming_duration_seconds=1.5 * bpm_a / bpm_b,
        score=1.0,
        reason="synthetic fixture",
    )


def test_positive_correction_delays_only_the_matched_window():
    plan = _plan()
    samples = np.arange(40.0)
    original = samples.copy()

    corrected = apply_initial_incoming_cue_correction(
        samples, plan, sample_rate=10, correction_samples=2
    )

    np.testing.assert_array_equal(corrected[:10], original[:10])
    np.testing.assert_array_equal(corrected[10:25], [10, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    np.testing.assert_array_equal(corrected[25:], original[25:])
    np.testing.assert_array_equal(samples, original)
    assert not np.shares_memory(corrected, samples)


def test_negative_correction_advances_only_the_matched_window():
    plan = _plan()
    samples = np.arange(40.0)

    corrected = apply_initial_incoming_cue_correction(
        samples, plan, sample_rate=10, correction_samples=-2
    )

    np.testing.assert_array_equal(corrected[10:25], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24])
    np.testing.assert_array_equal(corrected[:10], samples[:10])
    np.testing.assert_array_equal(corrected[25:], samples[25:])


def test_phase_correction_uses_the_phase4d_mapping_and_returns_stable_material():
    plan = _plan(outgoing_time=1.24, incoming_time=2.36)
    outgoing = _timeline([1.24])
    incoming = _timeline([2.36])
    samples = np.arange(60.0)
    original = samples.copy()

    first = phase_correct_incoming_transition(samples, plan, 10, outgoing, incoming)
    second = phase_correct_incoming_transition(
        samples, plan, 10, outgoing, incoming
    )
    relationship = phase_relationship_for_match(
        plan,
        tempo_match(plan, 10),
        outgoing,
        incoming,
    )

    np.testing.assert_array_equal(first, second)
    assert relationship.initial_correction_samples == 1
    assert first.shape == samples.shape
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize(
    ("bpm_a", "bpm_b", "expected_rate"),
    [(120.0, 150.0, 0.8), (120.0, 96.0, 1.25)],
)
def test_supported_tempo_direction_is_inherited_by_phase4e(
    bpm_a: float, bpm_b: float, expected_rate: float
):
    plan = _plan(bpm_a=bpm_a, bpm_b=bpm_b)
    match = tempo_match(plan, sample_rate=10)
    relationship = phase_relationship_for_match(
        plan,
        match,
        _timeline([1.0], bpm=bpm_a),
        _timeline([1.0], bpm=bpm_b),
    )

    assert match.incoming_playback_rate == expected_rate
    assert relationship.transformed_mapping.transformed_count == match.matched_sample_count
    if expected_rate < 1.0:
        assert match.matched_sample_count > match.incoming_sample_count
    else:
        assert match.matched_sample_count < match.incoming_sample_count


def test_correction_rejects_a_shift_that_discards_the_entire_window():
    with pytest.raises(ValueError, match="leave part of the matched window"):
        apply_initial_incoming_cue_correction(
            np.arange(30.0), _plan(), sample_rate=10, correction_samples=15
        )
