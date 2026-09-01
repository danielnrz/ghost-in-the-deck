"""The data contract between audio analysis and everything downstream.

Nothing in this module knows about librosa or Panda3D. It is the hand-off point
where a DJ behaviour engine will later be inserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2


@dataclass
class MusicFeatures:
    """Pre-analysed description of one track."""

    track: str
    duration_seconds: float
    sample_rate: int
    hop_length: int
    bpm: float
    beats: list[float]
    # RMS at the loudest point, before the envelopes were normalised. The
    # envelopes alone cannot tell a quiet recording from a loud one, because
    # each is scaled to its own peak; this is what lets a near-silent passage be
    # recognised as near-silent rather than as "quiet relative to itself".
    # 0.0 means unknown, which is how features written before schema 2 read.
    peak_rms: float = 0.0
    frame_times: list[float] = field(default_factory=list)
    onset_strength: list[float] = field(default_factory=list)
    rms: list[float] = field(default_factory=list)
    bass_energy: list[float] = field(default_factory=list)
    mid_energy: list[float] = field(default_factory=list)
    high_energy: list[float] = field(default_factory=list)

    @property
    def has_absolute_loudness(self) -> bool:
        """Whether an absolute loudness reference was recorded."""
        return self.peak_rms > 0.0

    def absolute_rms(self, index: int) -> float:
        """RMS at a frame in the original recording's own scale."""
        if not self.has_absolute_loudness or not self.rms:
            return 0.0
        index = max(0, min(index, len(self.rms) - 1))
        return float(self.rms[index]) * self.peak_rms

    @property
    def beat_interval(self) -> float:
        """Average seconds between beats, or 0.5 s if there are too few."""
        if len(self.beats) < 2:
            return 0.5
        return (self.beats[-1] - self.beats[0]) / (len(self.beats) - 1)

    def envelope_at(self, name: str, time: float) -> float:
        """Sample a per-frame envelope at ``time`` seconds (nearest frame)."""
        values = getattr(self, name)
        if not values or not self.frame_times:
            return 0.0
        index = int(round(time / (self.frame_times[1] - self.frame_times[0]))) if len(self.frame_times) > 1 else 0
        index = max(0, min(index, len(values) - 1))
        return float(values[index])

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "track": self.track,
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "bpm": round(self.bpm, 2),
            "peak_rms": round(self.peak_rms, 9),
            "beat_count": len(self.beats),
            "beats": [round(t, 4) for t in self.beats],
            "frame_times": [round(t, 4) for t in self.frame_times],
            "onset_strength": [round(v, 5) for v in self.onset_strength],
            "rms": [round(v, 6) for v in self.rms],
            "bass_energy": [round(v, 6) for v in self.bass_energy],
            "mid_energy": [round(v, 6) for v in self.mid_energy],
            "high_energy": [round(v, 6) for v in self.high_energy],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MusicFeatures":
        return cls(
            track=data["track"],
            duration_seconds=float(data["duration_seconds"]),
            sample_rate=int(data["sample_rate"]),
            hop_length=int(data["hop_length"]),
            bpm=float(data["bpm"]),
            beats=[float(t) for t in data["beats"]],
            peak_rms=float(data.get("peak_rms", 0.0)),
            frame_times=[float(t) for t in data.get("frame_times", [])],
            onset_strength=[float(v) for v in data.get("onset_strength", [])],
            rms=[float(v) for v in data.get("rms", [])],
            bass_energy=[float(v) for v in data.get("bass_energy", [])],
            mid_energy=[float(v) for v in data.get("mid_energy", [])],
            high_energy=[float(v) for v in data.get("high_energy", [])],
        )

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "MusicFeatures":
        return cls.from_dict(json.loads(Path(path).read_text()))


def normalise(values: np.ndarray) -> np.ndarray:
    """Scale an envelope into 0..1 so animation code can stay unit-agnostic."""
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(values)
    return values / peak
