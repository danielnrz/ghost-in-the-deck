"""Synthetic deterministic tests for the Phase 4D tempo-match contract."""

from __future__ import annotations

import dataclasses

import pytest

from ghost_in_the_deck.audio.tempo_match import (
    SUPPORTED_PLAYBACK_RATE_MAX,
    SUPPORTED_PLAYBACK_RATE_MIN,
    TempoMatch,
    tempo_match,
)
from ghost_in_the_deck.transition import TransitionPlan


def _plan(*, bpm_a: float, bpm_b: float, outgoing: float, incoming: float):
    return TransitionPlan(
        outgoing_track="synthetic-outgoing",
        incoming_track="synthetic-incoming",
        outgoing_time=4.0,
        incoming_time=8.0,
        outgoing_bar_index=1,
        incoming_bar_index=2,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        bpm_ratio=bpm_b / bpm_a,
        transition_length_bars=8,
        outgoing_duration_seconds=outgoing,
        incoming_duration_seconds=incoming,
        score=1.0,
        reason="synthetic fixture",
    )


def test_equal_bpm_is_a_no_op():
    plan = _plan(bpm_a=120.0, bpm_b=120.0, outgoing=16.0, incoming=16.0)

    result = tempo_match(plan, sample_rate=100)

    assert result == TempoMatch(
        target_bpm=120.0,
        incoming_bpm=120.0,
        incoming_playback_rate=1.0,
        incoming_duration_seconds=16.0,
        matched_duration_seconds=16.0,
        sample_rate=100,
        incoming_sample_count=1600,
        matched_sample_count=1600,
    )
    assert result.rate == 1.0
    assert result.output_duration_seconds == result.incoming_duration_seconds
    assert result.output_sample_count == result.incoming_sample_count


def test_faster_incoming_track_slows_and_expands_to_outgoing_duration():
    # 150 BPM over 120 BPM means the incoming source must play at 0.8x.
    plan = _plan(bpm_a=120.0, bpm_b=150.0, outgoing=16.0, incoming=12.8)

    result = tempo_match(plan, sample_rate=10)

    assert result.target_bpm == 120.0
    assert result.incoming_playback_rate == 0.8
    assert result.incoming_duration_seconds / result.rate == 16.0
    assert result.matched_duration_seconds == plan.outgoing_duration_seconds
    assert result.incoming_sample_count == 128
    assert result.matched_sample_count == 160


def test_slower_incoming_track_speeds_up_and_contracts_to_outgoing_duration():
    # 90 BPM over 120 BPM means the incoming source must play at 4/3x.
    plan = _plan(bpm_a=120.0, bpm_b=90.0, outgoing=16.0, incoming=21.333333333333332)

    result = tempo_match(plan, sample_rate=10)

    assert result.incoming_playback_rate == pytest.approx(4.0 / 3.0)
    assert result.incoming_duration_seconds / result.rate == pytest.approx(16.0)
    assert result.matched_duration_seconds == 16.0
    assert result.incoming_sample_count == 213
    assert result.matched_sample_count == 160


def test_sample_counts_use_deterministic_half_up_rounding():
    plan = _plan(bpm_a=120.0, bpm_b=120.0, outgoing=2.5, incoming=2.5)

    first = tempo_match(plan, sample_rate=1)
    second = tempo_match(dataclasses.replace(plan), sample_rate=1)

    assert first == second
    assert first.incoming_sample_count == 3
    assert first.matched_sample_count == 3


@pytest.mark.parametrize(
    "rate",
    [
        SUPPORTED_PLAYBACK_RATE_MIN,
        SUPPORTED_PLAYBACK_RATE_MAX,
    ],
)
def test_supported_rate_boundaries_are_inclusive(rate):
    result = tempo_match(
        _plan(bpm_a=120.0, bpm_b=120.0 / rate, outgoing=8.0, incoming=8.0 * rate),
        sample_rate=10,
    )

    assert result.incoming_playback_rate == rate


@pytest.mark.parametrize("bpm_a, bpm_b", [(120.0, 50.0), (120.0, 300.0)])
def test_rate_outside_supported_range_is_rejected(bpm_a, bpm_b):
    with pytest.raises(ValueError, match="supported range"):
        tempo_match(
            _plan(bpm_a=bpm_a, bpm_b=bpm_b, outgoing=8.0, incoming=8.0),
            sample_rate=10,
        )
