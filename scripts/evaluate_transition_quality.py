"""Evaluate frozen transition modes on deterministic generated PCM fixtures.

This command is deliberately an observational report: its measurements do not
change preview policy and no synthetic score is used as an application gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ghost_in_the_deck.audio.metric_primitives import (
    energy_continuity,
    measure_peak_headroom,
    measure_transient_interference,
    rms_energy,
)
from ghost_in_the_deck.transition_preview import _render_transition_preview

SCHEMA_VERSION = 1
MODES = ("no-stretch", "bpm-matched", "phase-aligned")
OUTGOING_BPM = 120.0
INCOMING_BPM = 100.0
LIMITATIONS = (
    "Generated WAV fixtures only; no private or user audio is read.",
    "Metrics describe the rendered PCM and do not establish perceptual or semantic alignment.",
    "The report is observational; synthetic scores never gate or alter application behavior.",
    "Only the bounded transition window is evaluated, not a complete musical mix.",
)


@dataclass(frozen=True)
class _SyntheticPreviewCase:
    mode: str
    output_path: Path
    report: Any


def _burst(frequency: float, seconds: float, falloff: float, sample_rate: int) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return (np.sin(2 * np.pi * frequency * time) * np.exp(-falloff * time)).astype(np.float32)


def _make_beat_track(path: Path, *, bpm: float, seconds: float, sample_rate: int = 22050) -> Path:
    """Write a deterministic, broadband generated source for one evaluation."""
    samples = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    layers = (
        0.9 * _burst(55.0, 0.18, 26.0, sample_rate),
        0.25 * _burst(420.0, 0.07, 55.0, sample_rate),
        0.18 * _burst(4200.0, 0.03, 160.0, sample_rate),
    )
    interval = 60.0 / bpm
    time = 0.0
    while time < seconds - 0.25:
        start = int(time * sample_rate)
        for layer in layers:
            samples[start : start + layer.size] += layer[: samples.size - start]
        time += interval
    rng = np.random.default_rng(7)
    samples += rng.normal(0.0, 0.002, samples.size).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(samples, -1.0, 1.0), sample_rate)
    return path


def _run_synthetic_preview_matrix(output_dir: Path, *, seconds: float) -> list[_SyntheticPreviewCase]:
    """Render the frozen preview matrix using only locally generated WAV sources."""
    sources = output_dir / "sources"
    analysis = output_dir / "analysis"
    outgoing = _make_beat_track(sources / "outgoing.wav", bpm=OUTGOING_BPM, seconds=seconds)
    incoming = _make_beat_track(sources / "incoming.wav", bpm=INCOMING_BPM, seconds=seconds)
    source_bytes = (outgoing.read_bytes(), incoming.read_bytes())
    cases: list[_SyntheticPreviewCase] = []
    for mode in MODES:
        output = output_dir / "previews" / f"{mode}.wav"
        report = _render_transition_preview(
            outgoing,
            incoming,
            output,
            refresh=False,
            analysis_dir=analysis,
            margin_seconds=16.0,
            limit_per_deck=5,
            transition_bars=8,
            mode=mode,
        )
        cases.append(_SyntheticPreviewCase(mode, output, report))
    if (outgoing.read_bytes(), incoming.read_bytes()) != source_bytes:
        raise AssertionError("synthetic preview modified a source WAV")
    return cases


def _number(value: float | int | None) -> float | int | None:
    """Return JSON-friendly finite numeric values, preserving missing data."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, 12)


def _diagnostics(report: Any) -> dict[str, Any]:
    before = report.before
    result: dict[str, Any] = {
        "no_stretch_end_drift_seconds": _number(before.predicted_end_drift_seconds),
        "no_stretch_end_drift_outgoing_beats": _number(
            before.predicted_end_drift_outgoing_beats
        ),
        "initial_alignment_beat_fraction": _number(before.initial_alignment_beat_fraction),
    }
    after = report.after_diagnostics
    if after is not None:
        result.update(
            {
                "matched_end_drift_seconds": _number(after.residual_end_drift_seconds),
                "matched_end_drift_outgoing_beats": _number(
                    after.residual_end_drift_outgoing_beats
                ),
                "incoming_playback_rate": _number(after.incoming_playback_rate),
                "initial_phase_offset_samples": _number(after.initial_phase_offset_samples),
                "residual_phase_offset_samples": _number(after.residual_phase_offset_samples),
                "initial_correction_samples": after.initial_correction_samples,
            }
        )
    return result


def _case_record(case: Any, *, outgoing_bpm: float, incoming_bpm: float) -> dict[str, Any]:
    samples, sample_rate = sf.read(case.output_path, dtype="float64", always_2d=False)
    pcm = np.asarray(samples, dtype=np.float64)
    if pcm.ndim == 2:
        mono = np.mean(pcm, axis=1)
        channels = int(pcm.shape[1])
    else:
        mono = pcm
        channels = 1
    boundary = mono.size // 2
    window = min(1024, boundary, mono.size - boundary)
    continuity = energy_continuity(mono, boundary, window=window)
    peak = measure_peak_headroom(mono)
    left = mono[boundary - window : boundary]
    right = mono[boundary : boundary + window]
    interference = measure_transient_interference(left, right)
    case_id = f"synthetic-{int(outgoing_bpm)}-{int(incoming_bpm)}-{case.mode}"
    return {
        "case_id": case_id,
        "mode": case.mode,
        "tempo": {
            "outgoing_bpm": outgoing_bpm,
            "incoming_bpm": incoming_bpm,
            "ratio": _number(incoming_bpm / outgoing_bpm),
        },
        "output": {
            "path": case.output_path.name,
            "sample_rate": int(sample_rate),
            "channels": channels,
            "frames": int(mono.size),
        },
        "metrics": {
            "rms": _number(rms_energy(mono)),
            "peak": _number(peak.peak),
            "headroom": _number(peak.headroom),
            "headroom_db": _number(peak.headroom_db),
            "boundary": {
                "sample": boundary,
                "window": window,
                "before_rms": _number(continuity.before_rms),
                "after_rms": _number(continuity.after_rms),
                "dip_rms": _number(continuity.dip_rms),
                "dip_ratio": _number(continuity.dip_ratio),
                "boundary_jump": _number(continuity.boundary_jump),
            },
            "transient_interference": {
                "correlation": _number(interference.correlation),
                "sum_rms": _number(interference.sum_rms),
                "expected_rms": _number(interference.expected_rms),
                "interference_ratio": _number(interference.interference_ratio),
                "cancellation": _number(interference.cancellation),
            },
        },
        "mapping_diagnostics": _diagnostics(case.report),
    }
    return result


def evaluate(out_dir: Path, *, seconds: float = 40.0) -> dict[str, Any]:
    """Render and aggregate the fixed three-mode synthetic evaluation matrix."""
    cases = _run_synthetic_preview_matrix(out_dir, seconds=seconds)
    records = [_case_record(case, outgoing_bpm=OUTGOING_BPM, incoming_bpm=INCOMING_BPM) for case in cases]
    records.sort(key=lambda item: MODES.index(item["mode"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "synthetic-transition-quality",
        "status": "PASS",
        "fixture": {
            "kind": "generated",
            "sample_rate": 22050,
            "seconds": seconds,
            "outgoing_bpm": OUTGOING_BPM,
            "incoming_bpm": INCOMING_BPM,
            "modes": list(MODES),
        },
        "limitations": list(LIMITATIONS),
        "cases": records,
    }


def format_report_summary(report: dict[str, Any]) -> str:
    """Produce concise, stable human-readable CLI output."""
    lines = [
        f"synthetic transition quality: {report['status']}",
        f"fixture: {report['fixture']['outgoing_bpm']:.0f} -> {report['fixture']['incoming_bpm']:.0f} BPM",
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            f"{case['mode']}: case={case['case_id']} rms={metrics['rms']:.6f} "
            f"peak={metrics['peak']:.6f} boundary_jump={metrics['boundary']['boundary_jump']:.6f}"
        )
    lines.append(f"limitations: {len(report['limitations'])} explicit limitations; observational only")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen transition modes on generated PCM.")
    parser.add_argument("--out-dir", type=Path, required=True, help="ignored directory for fixtures and report")
    parser.add_argument("--seconds", type=float, default=40.0, help="generated fixture duration")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.out_dir, seconds=args.seconds)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.out_dir / "transition-quality-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(format_report_summary(report))
        print(f"report: {report_path}")
        return 0
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
