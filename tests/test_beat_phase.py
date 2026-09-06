"""Synthetic tests for the pure beat-phase correction contract."""

from __future__ import annotations

import pytest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.audio.beat_phase import (
    SampleSpan,
    TransformedSampleMapping,
    anchor_cue,
    build_beat_phase_contract,
    half_open_time_span,
    measure_beat_phase,
)

from synthetic import make_features


def _timeline(beats: list[float]) -> BeatTimeline:
    return BeatTimeline(make_features(beats, duration=8.0, bpm=120.0))


def test_cue_anchor_preserves_fractional_timeline_timestamp():
    anchor = anchor_cue(_timeline([1.24, 2.36]), cue_index=0, sample_rate=10)

    assert anchor.cue_time_seconds == pytest.approx(1.24)
    assert anchor.sample_position == pytest.approx(12.4)
    assert anchor.sample_index == 12
    assert anchor.fractional_sample_offset == pytest.approx(0.4)


def test_time_and_subspan_mapping_are_half_open():
    assert half_open_time_span(1.24, 1.74, 10) == SampleSpan(12, 17)

    mapping = TransformedSampleMapping(
        source_span=SampleSpan(12, 17),
        transformed_span=SampleSpan(30, 37),
    )
    assert mapping.source_boundary_to_transformed(12) == 30
    assert mapping.source_boundary_to_transformed(17) == 37
    assert mapping.map_source_span(SampleSpan(13, 16)) == SampleSpan(31, 35)
    assert mapping.map_source_span(SampleSpan(12, 17)).count == 7


def test_signed_phase_offset_and_initial_correction_use_transformed_mapping():
    outgoing = anchor_cue(_timeline([1.24]), cue_index=0, sample_rate=10)
    incoming = anchor_cue(_timeline([2.36]), cue_index=0, sample_rate=10)
    mapping = TransformedSampleMapping(SampleSpan(24, 29), SampleSpan(0, 10))

    result = measure_beat_phase(
        outgoing,
        incoming,
        mapping,
        beat_interval_seconds=0.5,
    )

    # Incoming is -0.4 source samples from its rounded anchor; 2x expansion
    # maps that to -0.8 output samples.  Outgoing is +0.4, so incoming is
    # 1.2 samples early and the initial correction delays it by one sample.
    assert result.signed_offset_samples == pytest.approx(-1.2)
    assert result.signed_offset_seconds == pytest.approx(-0.12)
    assert result.signed_offset_beats == pytest.approx(-0.24)
    assert result.initial_correction_samples == 1
    assert result.initial_correction_seconds == pytest.approx(0.1)
    assert result.residual_offset_samples == pytest.approx(-0.2)


def test_builder_is_repeatable_and_anchors_phase4d_source_window():
    outgoing = _timeline([1.24])
    incoming = _timeline([2.36])

    first = build_beat_phase_contract(
        outgoing,
        incoming,
        outgoing_cue_index=0,
        incoming_cue_index=0,
        sample_rate=10,
        source_sample_count=5,
        transformed_sample_count=10,
    )
    second = build_beat_phase_contract(
        outgoing,
        incoming,
        outgoing_cue_index=0,
        incoming_cue_index=0,
        sample_rate=10,
        source_sample_count=5,
        transformed_sample_count=10,
    )

    assert first == second
    assert first.transformed_mapping.source_span == SampleSpan(24, 29)
    assert first.transformed_mapping.transformed_span == SampleSpan(24, 34)


def test_non_cue_index_cannot_be_used_as_an_anchor():
    with pytest.raises(ValueError, match="detected timeline cue"):
        anchor_cue(_timeline([1.24]), cue_index=1, sample_rate=10)
