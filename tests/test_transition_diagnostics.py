"""Focused checks for deterministic no-time-stretch transition diagnostics."""

from __future__ import annotations

import dataclasses

import pytest

from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.transition import TransitionPlan
from ghost_in_the_deck.transition_diagnostics import (
    TransitionDiagnostics,
    diagnose_transition_drift,
)

from synthetic import make_features, regular_beats


def _timeline(track: str, bpm: float, *, offset: float = 0.5) -> BeatTimeline:
    duration = 80.0
    features = make_features(
        regular_beats(bpm=bpm, count=int(duration / (60.0 / bpm)), offset=offset),
        duration=duration,
        bpm=bpm,
    )
    return BeatTimeline(dataclasses.replace(features, track=track))


def _plan(
    outgoing: BeatTimeline,
    incoming: BeatTimeline,
    *,
    duration_a: float,
    duration_b: float,
) -> TransitionPlan:
    return TransitionPlan(
        outgoing_track=outgoing.features.track,
        incoming_track=incoming.features.track,
        outgoing_time=outgoing.beat_time(8),
        incoming_time=incoming.beat_time(8),
        outgoing_bar_index=2,
        incoming_bar_index=2,
        bpm_a=outgoing.features.bpm,
        bpm_b=incoming.features.bpm,
        bpm_ratio=incoming.features.bpm / outgoing.features.bpm,
        transition_length_bars=8,
        outgoing_duration_seconds=duration_a,
        incoming_duration_seconds=duration_b,
        score=1.0,
        reason="test selection",
    )


def test_equal_tempo_reports_aligned_zero_drift():
    outgoing = _timeline("outgoing", 120.0)
    incoming = _timeline("incoming", 120.0)
    plan = _plan(outgoing, incoming, duration_a=16.0, duration_b=16.0)

    result = diagnose_transition_drift(outgoing, incoming, plan)

    assert isinstance(result, TransitionDiagnostics)
    assert result.bpm_a == 120.0
    assert result.bpm_b == 120.0
    assert result.bpm_difference == 0.0
    assert result.bpm_ratio == 1.0
    assert result.initial_alignment_seconds == pytest.approx(0.0)
    assert result.initial_alignment_beat_fraction == pytest.approx(0.0)
    assert result.transition_duration_seconds == 16.0
    assert result.predicted_end_drift_seconds == pytest.approx(0.0)
    assert result.predicted_end_drift_beat_fraction == pytest.approx(0.0)


def test_slower_incoming_tempo_reports_signed_analytic_drift():
    outgoing = _timeline("fast", 120.0)
    incoming = _timeline("slow", 90.0)
    plan = _plan(outgoing, incoming, duration_a=16.0, duration_b=21.333333333333332)

    result = diagnose_transition_drift(outgoing, incoming, plan)

    # In 16 shared seconds, the incoming grid advances 24 beats and the
    # outgoing grid advances 32.  The incoming clock is therefore 8 outgoing
    # beats behind, or 4 seconds at the outgoing deck's 120 BPM.
    assert result.bpm_difference == pytest.approx(30.0)
    assert result.bpm_ratio == pytest.approx(0.75)
    assert result.transition_duration_seconds == pytest.approx(16.0)
    assert result.predicted_end_drift_seconds == pytest.approx(-4.0)
    assert result.predicted_end_drift_outgoing_beats == pytest.approx(-8.0)
    assert result.predicted_end_drift_incoming_beats == pytest.approx(-6.0)


def test_initial_alignment_is_omitted_for_a_virtual_anchor():
    outgoing = _timeline("outgoing", 120.0)
    incoming = _timeline("incoming", 120.0, offset=0.25)
    plan = _plan(outgoing, incoming, duration_a=16.0, duration_b=16.0)
    plan = dataclasses.replace(plan, incoming_time=incoming.beat_time(8) + 0.1)

    result = diagnose_transition_drift(outgoing, incoming, plan)

    assert result.initial_alignment_seconds is None
    assert result.initial_alignment_beat_fraction is None
