"""Fixtures the test suite can rely on without any private music.

Everything here is generated, so a fresh clone containing only
``testMusic/.gitkeep`` still exercises the real analysis, timeline and
animation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ghost_in_the_deck.animation.groove import GrooveEngine
from ghost_in_the_deck.audio.features import MusicFeatures

SAMPLE_RATE = 22050


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
        ones = [1.0] * len(frames)
    elif callable(energy):
        ones = [float(energy(t)) for t in frames]
    else:
        ones = [float(energy)] * len(frames)
    return MusicFeatures(
        track="synthetic",
        duration_seconds=duration,
        sample_rate=SAMPLE_RATE,
        hop_length=512,
        bpm=bpm,
        beats=list(beats),
        frame_times=frames,
        onset_strength=ones,
        rms=ones,
        bass_energy=ones,
        mid_energy=ones,
        high_energy=ones,
    )


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
    def has_beat(self) -> bool:
        return self.beat_index >= 0
