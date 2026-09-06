"""Focused checks for the explicit phase-aligned offline preview path."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

import ghost_in_the_deck.transition_preview as transition_preview
from ghost_in_the_deck.animation.cues import BeatTimeline
from ghost_in_the_deck.transition import TransitionPlan

from synthetic import SAMPLE_RATE, make_features, make_beat_track


def test_phase_aligned_preview_reports_timestamped_mapping_and_reduces_phase_error(
    tmp_path: Path, monkeypatch
):
    outgoing_time = (10000.51) / SAMPLE_RATE
    incoming_time = (20000.49) / SAMPLE_RATE
    outgoing_features = make_features(
        [outgoing_time, outgoing_time + 0.5], duration=5.0, bpm=120.0
    )
    incoming_features = make_features(
        [incoming_time, incoming_time + 0.6], duration=5.0, bpm=100.0
    )
    plan = TransitionPlan(
        outgoing_track="outgoing",
        incoming_track="incoming",
        outgoing_time=outgoing_time,
        incoming_time=incoming_time,
        outgoing_bar_index=1,
        incoming_bar_index=1,
        bpm_a=120.0,
        bpm_b=100.0,
        bpm_ratio=100.0 / 120.0,
        transition_length_bars=1,
        outgoing_duration_seconds=1.5,
        incoming_duration_seconds=1.25,
        score=1.0,
        reason="phase fixture",
    )
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=5.0, bpm=120.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=5.0, bpm=100.0)
    monkeypatch.setattr(
        transition_preview,
        "_build_plan",
        lambda *args, **kwargs: (outgoing_features, incoming_features, plan),
    )

    report = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "phase-aligned.wav",
        refresh=False,
        analysis_dir=tmp_path / "analysis",
        margin_seconds=0.0,
        limit_per_deck=1,
        transition_bars=1,
        mode="phase-aligned",
    )

    assert report.mode == "phase-aligned"
    assert report.output_path.is_file()
    assert report.phase_relationship is not None
    assert report.after_diagnostics is not None
    matched = report.after_diagnostics
    assert matched.outgoing_cue_time_seconds == outgoing_time
    assert matched.incoming_cue_time_seconds == incoming_time
    assert matched.transformed_sample_count == matched.outgoing_sample_count
    assert matched.initial_correction_samples != 0
    assert abs(matched.phase_error_after_samples) < abs(
        matched.phase_error_before_samples
    )
    assert matched.transformed_anchor_sample_position is not None
    assert BeatTimeline(outgoing_features).cue(0).scheduled_time == outgoing_time
    assert sf.info(str(report.output_path)).samplerate == SAMPLE_RATE


def test_phase_aligned_mode_is_deterministic_for_existing_generated_tracks(
    tmp_path: Path,
):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0, bpm=120.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0, bpm=100.0)
    first = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "first.wav",
        refresh=False,
        analysis_dir=tmp_path / "analysis",
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
        mode="phase-aligned",
    )
    second = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "second.wav",
        refresh=False,
        analysis_dir=tmp_path / "analysis",
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
        mode="phase-aligned",
    )

    assert first.output_path.read_bytes() == second.output_path.read_bytes()
    assert first.preview_diagnostics == second.preview_diagnostics
