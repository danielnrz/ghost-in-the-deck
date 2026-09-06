"""Focused end-to-end checks for the file-level transition preview seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ghost_in_the_deck.transition_preview import (
    IncompatiblePCMError,
    NoTransitionPlanError,
    render_transition_preview,
)
from synthetic import SAMPLE_RATE, make_beat_track


def test_generated_tracks_render_a_deterministic_owned_wav(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    outgoing_before = outgoing.read_bytes()
    incoming_before = incoming.read_bytes()
    analysis_dir = tmp_path / "analysis"

    first = render_transition_preview(
        outgoing, incoming, tmp_path / "preview-one.wav", analysis_dir=analysis_dir
    )
    second = render_transition_preview(
        outgoing, incoming, tmp_path / "preview-two.wav", analysis_dir=analysis_dir
    )

    assert first.read_bytes() == second.read_bytes()
    samples, sample_rate = sf.read(str(first), dtype="float64", always_2d=True)
    assert sample_rate == SAMPLE_RATE
    assert samples.shape[0] > 0
    assert np.any(samples != 0.0)
    assert outgoing.read_bytes() == outgoing_before
    assert incoming.read_bytes() == incoming_before


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
