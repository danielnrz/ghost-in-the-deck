import dataclasses

import numpy as np
import pytest

from ghost_in_the_deck.audio.mixer import (
    TransitionClock,
    execute_transition,
    linear_crossfade_gains,
)
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
    assert clock.sample_count == int(2.345 * 44100 + 0.5)
    assert clock.sample_count == TransitionClock(
        _plan(outgoing_duration=2.345), sample_rate=44100
    ).sample_count


def test_sample_count_uses_half_up_quantization():
    clock = TransitionClock(
        _plan(outgoing_duration=2.5, incoming_duration=3.5), sample_rate=1
    )

    assert clock.sample_count == 3


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


@pytest.mark.parametrize(
    "sample_rate", [0, -1, 1.5, True, "2", float("nan"), float("inf")]
)
def test_sample_rate_must_be_positive_and_finite(sample_rate):
    with pytest.raises(ValueError):
        TransitionClock(_plan(), sample_rate=sample_rate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outgoing_time", None),
        ("incoming_time", None),
        ("outgoing_duration_seconds", None),
        ("incoming_duration_seconds", None),
        ("outgoing_duration_seconds", "2.0"),
        ("incoming_duration_seconds", "2.0"),
        ("outgoing_duration_seconds", float("inf")),
        ("incoming_duration_seconds", float("nan")),
    ],
)
def test_malformed_plan_fields_raise_value_error(field, value):
    plan = dataclasses.replace(_plan(), **{field: value})

    with pytest.raises(ValueError):
        TransitionClock(plan, sample_rate=2)


def test_sample_count_rejects_duration_rate_overflow():
    clock = TransitionClock(
        _plan(outgoing_duration=1e308, incoming_duration=1e308),
        sample_rate=2,
    )

    with pytest.raises(ValueError, match="addressable sample range"):
        _ = clock.sample_count


def test_elapsed_mapping_rejects_times_outside_executable_window():
    clock = TransitionClock(_plan(outgoing_duration=4.0, incoming_duration=6.0), 1000)

    with pytest.raises(ValueError):
        clock.outgoing_time_at(-0.001)
    with pytest.raises(ValueError):
        clock.incoming_time_at(clock.duration_seconds + 0.001)


def test_linear_crossfade_requires_distinct_endpoints():
    with pytest.raises(ValueError, match="at least two"):
        linear_crossfade_gains(1)


def test_linear_crossfade_is_bounded_complementary_and_linear():
    outgoing, incoming = linear_crossfade_gains(5)

    np.testing.assert_array_equal(outgoing, [1.0, 0.75, 0.5, 0.25, 0.0])
    np.testing.assert_array_equal(incoming, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.all((0.0 <= outgoing) & (outgoing <= 1.0))
    assert np.all((0.0 <= incoming) & (incoming <= 1.0))
    np.testing.assert_array_equal(outgoing + incoming, np.ones(5))


@pytest.mark.parametrize("sample_count", [0, -1, 1, 1.5, "4", True, False])
def test_linear_crossfade_rejects_invalid_sample_counts(sample_count):
    with pytest.raises(ValueError, match="at least two"):
        linear_crossfade_gains(sample_count)


def test_linear_crossfade_returns_independent_arrays():
    outgoing, incoming = linear_crossfade_gains(3)
    outgoing[1] = 0.0

    np.testing.assert_array_equal(incoming, [0.0, 0.5, 1.0])


def test_execute_transition_slices_at_anchors_and_owns_endpoints():
    plan = _plan(
        outgoing_time=2.0,
        incoming_time=3.0,
        outgoing_duration=4.0,
        incoming_duration=5.0,
    )
    outgoing = np.arange(8, dtype=np.float32)
    incoming = np.arange(8, dtype=np.float32) + 100.0

    result = execute_transition(outgoing, incoming, plan, sample_rate=1)

    np.testing.assert_allclose(result, [2.0, 110.0 / 3.0, 214.0 / 3.0, 106.0])
    assert result.shape == (4,)
    assert result[0] == outgoing[2]
    assert result[-1] == incoming[6]
    assert result.flags.owndata
    assert not np.shares_memory(result, outgoing)
    assert not np.shares_memory(result, incoming)


def test_execute_transition_mixes_stereo_channels_independently():
    plan = _plan(
        outgoing_time=1.0,
        incoming_time=2.0,
        outgoing_duration=3.0,
        incoming_duration=4.0,
    )
    outgoing = np.array(
        [[-9.0, 9.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        dtype=np.float32,
    )
    incoming = np.array(
        [
            [-8.0, 8.0],
            [-7.0, 7.0],
            [100.0, 1000.0],
            [200.0, 2000.0],
            [300.0, 3000.0],
        ],
        dtype=np.float32,
    )

    result = execute_transition(outgoing, incoming, plan, sample_rate=1)

    np.testing.assert_allclose(
        result,
        [[1.0, 10.0], [101.0, 1010.0], [300.0, 3000.0]],
    )
    assert result.shape == (3, 2)


@pytest.mark.parametrize("stereo", [False, True], ids=["mono", "stereo"])
def test_execute_transition_is_deterministic_and_preserves_sources(stereo):
    plan = _plan(outgoing_time=1.0, incoming_time=1.0, outgoing_duration=4.0, incoming_duration=4.0)
    outgoing_mono = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    incoming_mono = np.linspace(1.0, -1.0, 8, dtype=np.float32)
    if stereo:
        outgoing = np.column_stack((outgoing_mono, -outgoing_mono))
        incoming = np.column_stack((incoming_mono, -incoming_mono))
    else:
        outgoing = outgoing_mono
        incoming = incoming_mono
    outgoing_before = outgoing.copy()
    incoming_before = incoming.copy()

    first = execute_transition(outgoing, incoming, plan, sample_rate=1)
    second = execute_transition(outgoing, incoming, plan, sample_rate=1)
    first[0] = 999.0

    np.testing.assert_array_equal(first[1:], second[1:])
    np.testing.assert_array_equal(outgoing, outgoing_before)
    np.testing.assert_array_equal(incoming, incoming_before)
    assert first.flags.owndata
    assert not np.shares_memory(second, outgoing)
    assert not np.shares_memory(second, incoming)


@pytest.mark.parametrize(
    "outgoing, incoming, plan, sample_rate, message",
    [
        (
            np.zeros((8, 2)),
            np.zeros(8),
            _plan(outgoing_duration=2.0, incoming_duration=2.0),
            1,
            "channel",
        ),
        (
            np.zeros((8, 1, 1)),
            np.zeros(8),
            _plan(outgoing_duration=2.0, incoming_duration=2.0),
            1,
            "shape",
        ),
        (
            np.zeros(2),
            np.zeros(8),
            _plan(outgoing_duration=4.0, incoming_duration=4.0),
            1,
            "complete transition",
        ),
        (
            np.zeros(8),
            np.zeros(8),
            _plan(outgoing_time=-1.0, outgoing_duration=2.0, incoming_duration=2.0),
            1,
            "anchor",
        ),
        (
            np.zeros(8),
            np.zeros(8),
            _plan(outgoing_duration=2.0, incoming_duration=2.0),
            0,
            "sample_rate",
        ),
        (
            np.zeros(8),
            np.zeros(8),
            _plan(outgoing_duration=0.0, incoming_duration=0.0),
            1,
            "at least two",
        ),
    ],
)
def test_execute_transition_rejects_invalid_inputs(
    outgoing, incoming, plan, sample_rate, message
):
    with pytest.raises(ValueError, match=message):
        execute_transition(outgoing, incoming, plan, sample_rate)


def test_execute_transition_rejects_missing_plan_with_value_error():
    with pytest.raises(ValueError, match="plan must be a TransitionPlan"):
        execute_transition(np.zeros(8), np.zeros(8), None, sample_rate=1)


def test_execute_transition_rounds_anchors_and_duration_half_up():
    plan = _plan(
        outgoing_time=2.5,
        incoming_time=3.5,
        outgoing_duration=2.5,
        incoming_duration=4.5,
    )
    outgoing = np.arange(8, dtype=np.float64)
    incoming = np.arange(8, dtype=np.float64) + 100.0

    result = execute_transition(outgoing, incoming, plan, sample_rate=1)

    assert result.shape == (3,)
    np.testing.assert_allclose(result, [3.0, 54.5, 106.0])
    assert result[0] == outgoing[3]
    assert result[-1] == incoming[6]


def test_execute_transition_rejects_non_finite_pcm():
    plan = _plan(outgoing_duration=2.0, incoming_duration=2.0)

    with pytest.raises(ValueError, match="finite"):
        execute_transition(
            np.array([0.0, np.nan, 0.0]),
            np.zeros(3),
            plan,
            sample_rate=1,
        )
