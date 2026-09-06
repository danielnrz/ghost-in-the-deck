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
    format_preview_summary,
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


def test_bpm_matched_preview_reduces_residual_drift_without_claiming_alignment(
    tmp_path: Path,
):
    outgoing = make_beat_track(
        tmp_path / "outgoing.wav", bpm=120.0, seconds=40.0
    )
    incoming = make_beat_track(
        tmp_path / "incoming.wav", bpm=90.0, seconds=40.0
    )
    report = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "matched-preview.wav",
        refresh=False,
        analysis_dir=tmp_path / "analysis",
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
        mode="bpm-matched",
    )

    assert report.mode == "bpm-matched"
    assert report.diagnostics.predicted_end_drift_seconds < -3.0
    assert report.before_diagnostics == report.diagnostics
    assert report.after_diagnostics is not None
    matched = report.after_diagnostics
    assert matched.source_sample_count != matched.transformed_sample_count
    assert matched.transformed_sample_count == matched.outgoing_sample_count
    assert abs(matched.residual_end_drift_seconds) < 1.0 / report.sample_rate
    summary = format_preview_summary(report)
    assert "BPM-matched sample mapping" in summary
    assert "semantic alignment is not claimed" in summary


def test_same_named_sources_use_distinct_feature_cache_entries(tmp_path: Path):
    outgoing = make_beat_track(
        tmp_path / "outgoing" / "same.wav", bpm=120.0, seconds=40.0
    )
    incoming = make_beat_track(
        tmp_path / "incoming" / "same.wav", bpm=90.0, seconds=40.0
    )
    analysis_dir = tmp_path / "analysis"

    first = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "preview-one.wav",
        refresh=False,
        analysis_dir=analysis_dir,
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
    )
    second = transition_preview._render_transition_preview(
        outgoing,
        incoming,
        tmp_path / "preview-two.wav",
        refresh=False,
        analysis_dir=analysis_dir,
        margin_seconds=16.0,
        limit_per_deck=5,
        transition_bars=8,
    )

    assert len(list(analysis_dir.glob("same-*.json"))) == 2
    assert first.plan == second.plan
    assert first.diagnostics == second.diagnostics
    assert first.output_path.read_bytes() == second.output_path.read_bytes()
    assert first.plan.bpm_a > first.plan.bpm_b
    assert first.diagnostics.predicted_end_drift_seconds < -3.0


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


def test_preview_output_temp_path_cannot_alias_source(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing-source.wav", seconds=40.0)
    aliased_source = tmp_path / "preview.wav.partial"
    outgoing.rename(aliased_source)
    outgoing = aliased_source
    incoming = make_beat_track(tmp_path / "incoming-source.wav", seconds=40.0)
    incoming_path = tmp_path / "incoming.audio"
    incoming.rename(incoming_path)
    incoming = incoming_path
    original = outgoing.read_bytes()
    output = tmp_path / "preview.wav"

    render_transition_preview(
        outgoing,
        incoming,
        output,
        analysis_dir=tmp_path / "analysis",
    )

    assert output.is_file()
    assert outgoing.read_bytes() == original


def test_preview_cleans_unique_temp_file_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    output = tmp_path / "preview.wav"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced preview write failure")

    monkeypatch.setattr(sf, "write", fail_write)
    with pytest.raises(transition_preview.TransitionPreviewError, match="could not write"):
        render_transition_preview(
            outgoing,
            incoming,
            output,
            analysis_dir=tmp_path / "analysis",
        )

    assert not output.exists()
    assert list(tmp_path.glob(".preview.wav.*.partial")) == []


def test_feature_cache_path_cannot_alias_incoming_source(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", bpm=120.0, seconds=40.0)
    incoming_wav = make_beat_track(
        tmp_path / "incoming.wav", bpm=90.0, seconds=40.0
    )
    analysis_dir = tmp_path / "analysis"
    cache_path = transition_preview._feature_cache_path(outgoing, analysis_dir)
    cache_path.parent.mkdir()
    incoming_wav.rename(cache_path)
    incoming_before = cache_path.read_bytes()
    output = tmp_path / "preview.wav"

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="generated feature-cache path must not alias an input source",
    ):
        render_transition_preview(
            outgoing, cache_path, output, analysis_dir=analysis_dir
        )

    assert cache_path.read_bytes() == incoming_before
    assert not output.exists()
    assert sorted(path.name for path in analysis_dir.iterdir()) == [cache_path.name]


def test_feature_cache_symlink_alias_is_rejected_without_artifacts(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    analysis_dir = tmp_path / "analysis"
    cache_path = transition_preview._feature_cache_path(outgoing, analysis_dir)
    cache_path.parent.mkdir()
    cache_path.symlink_to(incoming)

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="generated feature-cache path must not alias an input source",
    ):
        render_transition_preview(
            outgoing, incoming, tmp_path / "preview.wav", analysis_dir=analysis_dir
        )

    assert cache_path.is_symlink()
    assert not (tmp_path / "preview.wav").exists()


def test_feature_cache_hardlink_alias_is_rejected_without_artifacts(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    analysis_dir = tmp_path / "analysis"
    cache_path = transition_preview._feature_cache_path(outgoing, analysis_dir)
    cache_path.parent.mkdir()
    cache_path.hardlink_to(incoming)

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="generated feature-cache path must not alias an input source",
    ):
        render_transition_preview(
            outgoing, incoming, tmp_path / "preview.wav", analysis_dir=analysis_dir
        )

    assert cache_path.read_bytes() == incoming.read_bytes()
    assert not (tmp_path / "preview.wav").exists()


def test_preview_rejects_output_symlink_to_source_without_artifacts(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    output = tmp_path / "preview.wav"
    output.symlink_to(outgoing)
    outgoing_before = outgoing.read_bytes()

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="output must not replace a source",
    ):
        render_transition_preview(
            outgoing, incoming, output, analysis_dir=tmp_path / "analysis"
        )

    assert outgoing.read_bytes() == outgoing_before
    assert output.is_symlink()
    assert not (tmp_path / "analysis").exists()


def test_preview_rejects_output_hardlink_to_source_without_artifacts(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = make_beat_track(tmp_path / "incoming.wav", seconds=40.0)
    output = tmp_path / "preview.wav"
    output.hardlink_to(outgoing)
    outgoing_before = outgoing.read_bytes()

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="output must not replace a source",
    ):
        render_transition_preview(
            outgoing, incoming, output, analysis_dir=tmp_path / "analysis"
        )

    assert outgoing.read_bytes() == outgoing_before
    assert output.read_bytes() == outgoing_before
    assert not (tmp_path / "analysis").exists()


def test_preview_rejects_hardlinked_input_sources_without_artifacts(tmp_path: Path):
    outgoing = make_beat_track(tmp_path / "outgoing.wav", seconds=40.0)
    incoming = tmp_path / "incoming.wav"
    incoming.hardlink_to(outgoing)

    with pytest.raises(
        transition_preview.TransitionPreviewError,
        match="sources must be distinct",
    ):
        render_transition_preview(
            outgoing,
            incoming,
            tmp_path / "preview.wav",
            analysis_dir=tmp_path / "analysis",
        )

    assert not (tmp_path / "analysis").exists()
    assert not (tmp_path / "preview.wav").exists()


def test_safe_analysis_directory_is_accepted_by_writable_path_preflight(
    tmp_path: Path,
):
    outgoing = tmp_path / "outgoing.wav"
    incoming = tmp_path / "incoming.wav"
    outgoing.write_bytes(b"outgoing")
    incoming.write_bytes(b"incoming")
    cache_dir = tmp_path / "safe" / "analysis"
    output = tmp_path / "preview.wav"

    validated = transition_preview._validated_paths(outgoing, incoming, output)
    cache_paths = transition_preview._validate_writable_paths(
        *validated, cache_dir
    )

    assert cache_paths == (
        transition_preview._feature_cache_path(outgoing, cache_dir),
        transition_preview._feature_cache_path(incoming, cache_dir),
    )
    assert not cache_dir.exists()
    assert not output.exists()
