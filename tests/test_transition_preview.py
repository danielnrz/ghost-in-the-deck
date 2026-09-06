"""Focused end-to-end checks for the file-level transition preview seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import ghost_in_the_deck.transition_preview as transition_preview
from ghost_in_the_deck.transition_preview import (
    IncompatiblePCMError,
    NoTransitionPlanError,
    render_transition_preview,
)
from synthetic import SAMPLE_RATE, make_beat_track


def test_generated_tracks_render_deterministic_pcm16_wav_and_report_drift(
    tmp_path: Path,
):
    outgoing = make_beat_track(
        tmp_path / "outgoing.wav", bpm=120.0, seconds=40.0
    )
    incoming = make_beat_track(
        tmp_path / "incoming.wav", bpm=90.0, seconds=40.0
    )
    outgoing_before = outgoing.read_bytes()
    incoming_before = incoming.read_bytes()
    analysis_dir = tmp_path / "analysis"

    first_report = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "preview-one.wav",
        refresh=False,
        analysis_dir=analysis_dir,
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
    )
    second = render_transition_preview(
        outgoing, incoming, tmp_path / "preview-two.wav", analysis_dir=analysis_dir
    )

    assert first_report.output_path.read_bytes() == second.read_bytes()
    info = sf.info(str(first_report.output_path))
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 1
    samples, sample_rate = sf.read(
        str(first_report.output_path), dtype="float64", always_2d=True
    )
    assert sample_rate == info.samplerate
    assert samples.shape[0] > 0
    assert np.any(samples != 0.0)
    assert first_report.diagnostics.predicted_end_drift_seconds < -3.0
    assert first_report.diagnostics.predicted_end_drift_outgoing_beats < -7.0
    assert outgoing.read_bytes() == outgoing_before
    assert incoming.read_bytes() == incoming_before


def test_preview_delegates_mixing_to_the_phase_4b_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    calls: list[tuple[object, object, object, int]] = []
    original = transition_preview.execute_transition

    def record_call(outgoing_pcm, incoming_pcm, plan, sample_rate):
        calls.append((outgoing_pcm, incoming_pcm, plan, sample_rate))
        return original(outgoing_pcm, incoming_pcm, plan, sample_rate)

    monkeypatch.setattr(transition_preview, "execute_transition", record_call)
    report = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "preview.wav",
        refresh=False,
        analysis_dir=tmp_path / "analysis",
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
    )

    assert len(calls) == 1
    assert calls[0][2] == report.plan
    assert calls[0][3] == SAMPLE_RATE
    assert isinstance(calls[0][0], np.ndarray)
    assert isinstance(calls[0][1], np.ndarray)


def test_preview_reports_when_planner_has_no_plan(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "short-outgoing.wav", seconds=8.0)
    incoming = make_beat_track(tmp_path / "short-incoming.wav", seconds=8.0)

    with pytest.raises(NoTransitionPlanError, match="no transition plan"):
        render_transition_preview(
            outgoing,
            incoming,
            tmp_path / "unused.wav",
            analysis_dir=tmp_path / "analysis",
        )
    assert not (tmp_path / "unused.wav").exists()


def test_preview_reports_mismatched_pcm_channels(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "mono-outgoing.wav", seconds=40.0)
    mono_incoming = make_beat_track(tmp_path / "mono-incoming.wav", seconds=40.0)
    samples, sample_rate = sf.read(str(mono_incoming), dtype="float32")
    incoming = tmp_path / "stereo-incoming.wav"
    sf.write(str(incoming), np.column_stack((samples, samples)), sample_rate)

    with pytest.raises(IncompatiblePCMError, match="same PCM channel count"):
        render_transition_preview(
            outgoing,
            incoming,
            tmp_path / "unused.wav",
            analysis_dir=tmp_path / "analysis",
        )


def test_preview_reports_mismatched_pcm_sample_rates(tmp_path: Path):
    outgoing = make_beat_track(
        tmp_path / "rate-outgoing.wav", seconds=40.0, sr=44100
    )
    incoming = make_beat_track(
        tmp_path / "rate-incoming.wav", seconds=40.0, sr=22050
    )

    with pytest.raises(IncompatiblePCMError, match="same PCM sample rate"):
        render_transition_preview(
            outgoing,
            incoming,
            tmp_path / "unused.wav",
            analysis_dir=tmp_path / "analysis",
        )
