"""Fixtures the test suite can rely on without any private music.

Everything here is generated, so a fresh clone containing only
``testMusic/.gitkeep`` still exercises the real analysis, timeline and
animation code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Iterator

import numpy as np
import soundfile as sf

from ghost_in_the_deck.animation.dj_behavior import DJBehaviorEngine
from ghost_in_the_deck.animation.groove import GrooveEngine
from ghost_in_the_deck.audio.features import MusicFeatures

SAMPLE_RATE = 22050


@dataclass(frozen=True)
class SyntheticPreviewCase:
    """One bounded generated-input preview evaluation."""

    mode: str
    output_path: Path
    report: object


@contextmanager
def isolated_synthetic_workspace(root: Path | None = None) -> Iterator[Path]:
    """Yield an ignored workspace and remove it when the evaluation ends.

    When ``root`` is supplied it is the caller-owned ignored output directory;
    only files created by this harness are removed.  A temporary directory is
    used otherwise, so fixtures can never fall back to the repository root.
    """
    import tempfile

    if root is None:
        with tempfile.TemporaryDirectory(prefix="synthetic-transition-") as name:
            yield Path(name)
        return

    workspace = Path(root)
    workspace.mkdir(parents=True, exist_ok=True)
    marker = workspace / ".synthetic-evaluation"
    marker.write_text("isolated synthetic evaluation workspace\\n")
    try:
        yield workspace
    finally:
        for child in workspace.iterdir():
            if child == marker:
                continue
            if child.is_dir() and not child.is_symlink():
                rmtree(child)
            else:
                child.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)


def run_synthetic_preview_matrix(
    output_dir: Path, *, seconds: float = 40.0
) -> list[SyntheticPreviewCase]:
    """Run each frozen preview path using only generated WAV sources.

    All inputs, feature caches, and outputs live beneath ``output_dir``.  The
    deliberately small matrix exercises equal tempo, a supported tempo ratio,
    and the explicit phase-aligned spelling without reading private audio.
    """
    from ghost_in_the_deck.transition_preview import _render_transition_preview

    root = Path(output_dir)
    sources = root / "sources"
    analysis = root / "analysis"
    outgoing = make_beat_track(sources / "outgoing.wav", bpm=120.0, seconds=seconds)
    incoming = make_beat_track(sources / "incoming.wav", bpm=100.0, seconds=seconds)
    source_bytes = (outgoing.read_bytes(), incoming.read_bytes())
    cases: list[SyntheticPreviewCase] = []
    for mode in ("no-stretch", "bpm-matched", "phase-aligned"):
        output = root / "previews" / f"{mode}.wav"
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
        cases.append(SyntheticPreviewCase(mode, output, report))
    if (outgoing.read_bytes(), incoming.read_bytes()) != source_bytes:
        raise AssertionError("synthetic preview modified a source WAV")
    return cases


def assert_unsupported_tempo_preview(output_dir: Path, *, seconds: float = 40.0) -> str:
    """Verify an out-of-policy generated tempo is rejected before output write."""
    from ghost_in_the_deck.transition_preview import (
        IncompatiblePCMError,
        _render_transition_preview,
    )

    root = Path(output_dir)
    outgoing = make_beat_track(root / "unsupported-outgoing.wav", bpm=120.0, seconds=seconds)
    incoming = make_beat_track(root / "unsupported-incoming.wav", bpm=90.0, seconds=seconds)
    output = root / "unsupported-preview.wav"
    try:
        _render_transition_preview(
            outgoing, incoming, output,
            refresh=False, analysis_dir=root / "unsupported-analysis",
            margin_seconds=16.0, limit_per_deck=5, transition_bars=8,
            mode="bpm-matched",
        )
    except IncompatiblePCMError as exc:
        if output.exists():
            raise AssertionError("unsupported tempo left a preview artifact")
        return str(exc)
    raise AssertionError("unsupported generated tempo was accepted")


def _burst(sr: int, frequency: float, seconds: float, falloff: float) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (np.sin(2 * np.pi * frequency * t) * np.exp(-falloff * t)).astype(np.float32)


def make_beat_track(
    path: Path, bpm: float = 120.0, seconds: float = 16.0, sr: int = SAMPLE_RATE
) -> Path:
    """Write a WAV with a known tempo and energy in every analysis band.

    A bare click would leave the bass band empty, so each beat is a low kick
    plus a mid body and a high tick. That gives the band analysis something
    real to measure while keeping the tempo exactly known.
    """
    samples = np.zeros(int(seconds * sr), dtype=np.float32)
    kick = 0.9 * _burst(sr, 55.0, 0.18, 26.0)
    body = 0.25 * _burst(sr, 420.0, 0.07, 55.0)
    tick = 0.18 * _burst(sr, 4200.0, 0.03, 160.0)

    interval = 60.0 / bpm
    time = 0.0
    while time < seconds - 0.25:
        start = int(time * sr)
        for layer in (kick, body, tick):
            samples[start : start + layer.size] += layer[: samples.size - start]
        time += interval

    # A little noise keeps every band above zero between hits.
    rng = np.random.default_rng(7)
    samples += rng.normal(0.0, 0.002, samples.size).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(samples, -1.0, 1.0), sr)
    return path


def make_silent_track(path: Path, seconds: float = 2.0, sr: int = SAMPLE_RATE) -> Path:
    """A valid but featureless audio file, for discovery tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


def make_features(
    beats, duration: float = 12.0, bpm: float = 120.0, energy=None
) -> MusicFeatures:
    """A MusicFeatures with an exactly known beat timeline.

    ``energy`` may be a constant or a callable taking the frame time, so a test
    can shape the loudness curve the groove will read.
    """
    step = 0.05
    frames = [i * step for i in range(int(duration / step))]
    if energy is None:
        raw = [1.0] * len(frames)
    elif callable(energy):
        raw = [float(energy(t)) for t in frames]
    else:
        raw = [float(energy)] * len(frames)

    # Analysis normalises every envelope to its own peak and records the true
    # peak separately, so the fixture has to do the same or the absolute
    # loudness gate would see a doubly-scaled signal.
    peak = max(raw) if raw else 0.0
    ones = [v / peak for v in raw] if peak > 0 else list(raw)
    return MusicFeatures(
        track="synthetic",
        duration_seconds=duration,
        sample_rate=SAMPLE_RATE,
        hop_length=512,
        bpm=bpm,
        beats=list(beats),
        peak_rms=peak,
        frame_times=frames,
        onset_strength=ones,
        rms=ones,
        bass_energy=ones,
        mid_energy=ones,
        high_energy=ones,
    )


def make_broadband_signal(
    seconds: float, sr: int = 44100, channels: int = 2, seed: int = 3
) -> np.ndarray:
    """Deterministic white-ish noise spanning the audible band.

    Audio effects tests need a signal with real energy at both low and high
    frequencies to prove a filter moved it - a pure tone would only prove the
    filter affects *a* frequency, not that it reshapes a real spectrum. Fixed
    seed, so two calls with the same arguments are byte-identical.
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, 0.2, (int(seconds * sr), channels))
    return samples.astype(np.float64)


def regular_beats(bpm: float = 120.0, count: int = 24, offset: float = 0.5) -> list[float]:
    interval = 60.0 / bpm
    return [offset + i * interval for i in range(count)]


class RecordingRig:
    """Stands in for AvatarRig, recording joint offsets instead of moving bones.

    Lets the timing tests exercise the real animator with no Panda3D window.
    """

    def __init__(self) -> None:
        self.offsets: dict[str, tuple[float, float, float]] = {}

    def set_offset(
        self, name: str, heading: float = 0.0, pitch: float = 0.0, roll: float = 0.0
    ) -> None:
        self.offsets[name] = (heading, pitch, roll)

    def reset(self) -> None:
        self.offsets.clear()

    def pose(self) -> dict[str, tuple[float, float, float]]:
        return dict(self.offsets)


def groove_for(
    beats=None,
    duration: float = 14.0,
    bpm: float = 120.0,
    seed: str = "test",
    energy=None,
) -> GrooveEngine:
    """A GrooveEngine over a known beat grid, for tests that need behaviour."""
    if beats is None:
        beats = regular_beats(bpm=bpm)
    return GrooveEngine(
        make_features(beats, duration=duration, bpm=bpm, energy=energy), seed=seed
    )


@dataclass(frozen=True)
class SampleState:
    """Minimal stand-in for the pose state TimingRecorder reads.

    The recorder only looks at four attributes; building a whole GrooveState in
    a coverage test would obscure what is being tested.
    """

    time: float
    beat_index: int
    beat_age: float

    @property
    def has_detected_beat(self) -> bool:
        """This stand-in never carries virtual beats, so the sign is enough."""
        return self.beat_index >= 0

    @property
    def has_beat(self) -> bool:
        return self.has_detected_beat


def behavior_for(
    beats=None,
    duration: float = 120.0,
    bpm: float = 120.0,
    seed: str = "test",
    energy=None,
) -> DJBehaviorEngine:
    """A DJBehaviorEngine over a known beat grid, for gesture-scheduling tests."""
    if beats is None:
        beats = regular_beats(bpm=bpm, count=int(duration / (60.0 / bpm)))
    features = make_features(beats, duration=duration, bpm=bpm, energy=energy)
    return DJBehaviorEngine(features, seed=seed)
