import dataclasses

import pytest

from ghost_in_the_deck.audio.mixer import TransitionClock
from ghost_in_the_deck.transition import TransitionPlan


def _plan(
    *,
    outgoing_time: float = 12.5,
    incoming_time: float = 37.25,
    outgoing_duration: float = 16.0,
    incoming_duration: float = 21.333333333333332,
) -> TransitionPlan:
    return TransitionPlan(
        outgoing_track="outgoing.wav",
        incoming_track="incoming.wav",
        outgoing_time=outgoing_time,
        incoming_time=incoming_time,
        outgoing_bar_index=3,
        incoming_bar_index=7,
        bpm_a=120.0,
        bpm_b=90.0,
        bpm_ratio=0.75,
        transition_length_bars=8,
        outgoing_duration_seconds=outgoing_duration,
        incoming_duration_seconds=incoming_duration,
        score=1.0,
        reason="test fixture",
    )


def test_clock_exposes_exact_source_anchors_and_is_frozen():
    plan = _plan()
    clock = TransitionClock(plan, sample_rate=1000)

    assert clock.outgoing_anchor_seconds == plan.outgoing_time
    assert clock.incoming_anchor_seconds == plan.incoming_time
    assert clock.outgoing_time == plan.outgoing_time
    assert clock.incoming_time == plan.incoming_time
    with pytest.raises(dataclasses.FrozenInstanceError):
        clock.sample_rate = 2000


def test_source_mappings_are_absolute_and_monotonic():
    clock = TransitionClock(_plan(), sample_rate=1000)
    elapsed = [0.0, 1.25, 8.0, clock.duration_seconds]
    outgoing = [clock.outgoing_time_at(value) for value in elapsed]
    incoming = [clock.incoming_time_at(value) for value in elapsed]

    assert outgoing == [clock.outgoing_anchor_seconds + value for value in elapsed]
    assert incoming == [clock.incoming_anchor_seconds + value for value in elapsed]
    assert all(left < right for left, right in zip(outgoing, outgoing[1:]))
    assert all(left < right for left, right in zip(incoming, incoming[1:]))
    assert clock.source_times_at(1.25) == (
        clock.outgoing_anchor_seconds + 1.25,
        clock.incoming_anchor_seconds + 1.25,
    )


def test_sample_count_is_deterministic_from_duration_and_sample_rate():
    clock = TransitionClock(_plan(outgoing_duration=2.345), sample_rate=44100)

    assert clock.executable_duration_seconds == 2.345
    assert clock.duration_seconds == 2.345
    assert clock.sample_count == int(2.345 * 44100)
    assert clock.sample_count == TransitionClock(
        _plan(outgoing_duration=2.345), sample_rate=44100
    ).sample_count


def test_unequal_tempo_plan_uses_shortest_window_without_stretching():
    plan = _plan(outgoing_duration=16.0, incoming_duration=21.333333333333332)
    clock = TransitionClock(plan, sample_rate=100)

    assert clock.duration_seconds == 16.0
    assert clock.sample_count == 1600
    assert clock.outgoing_time_at(clock.duration_seconds) == 28.5
    assert clock.incoming_time_at(clock.duration_seconds) == 53.25
    assert clock.incoming_time_at(clock.duration_seconds) < (
        plan.incoming_time + plan.incoming_duration_seconds
    )


@pytest.mark.parametrize("sample_rate", [0, -1, float("nan"), float("inf")])
def test_sample_rate_must_be_positive_and_finite(sample_rate):
    with pytest.raises(ValueError):
        TransitionClock(_plan(), sample_rate=sample_rate)


def test_elapsed_mapping_rejects_times_outside_executable_window():
    clock = TransitionClock(_plan(outgoing_duration=4.0, incoming_duration=6.0), 1000)

    with pytest.raises(ValueError):
        clock.outgoing_time_at(-0.001)
    with pytest.raises(ValueError):
        clock.incoming_time_at(clock.duration_seconds + 0.001)
