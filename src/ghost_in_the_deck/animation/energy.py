"""A stable movement-intensity signal derived from the analysed music.

The per-frame envelopes in MusicFeatures are far too jumpy to drive a body with:
a single quiet analysis frame would make the avatar twitch. This smooths them
into one slow signal and rescales it against the track's *own* dynamic range, so
a quiet recording still reaches full intensity in its loudest passage and a
loud one still drops in its intro. Nothing here is tuned to a particular song.

The whole curve is built once, when the groove engine is created, so evaluating
it during playback is a single indexed lookup. It is a pure function of the
analysis, so the same track always yields the same intensity at the same time.

This is not a section classifier. It says how energetic the music is right now,
not whether this is a build-up, a drop or a breakdown.
"""

from __future__ import annotations

import numpy as np

from ..audio.features import MusicFeatures

# How the envelopes combine. Loudness carries most of the feel, bass is what a
# body actually moves to, and onset activity distinguishes a busy passage from a
# sustained loud one.
WEIGHTS = {"rms": 0.5, "bass_energy": 0.3, "onset_strength": 0.2}

# Seconds of smoothing. Long enough that individual hits do not register as
# energy changes, short enough to follow a section change within a bar or two.
SMOOTHING_SECONDS = 1.5

# Percentiles the track's own range is mapped from. Trimming the extremes stops
# one loud transient from flattening everything else.
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 95.0


class EnergyTrack:
    """Smoothed 0..1 intensity, addressable by absolute playback time."""

    def __init__(
        self,
        features: MusicFeatures,
        smoothing_seconds: float = SMOOTHING_SECONDS,
    ):
        self.features = features
        self._times = np.asarray(features.frame_times, dtype=float)
        self._values = self._build(features, smoothing_seconds)

        if self._times.size >= 2:
            self._step = float(self._times[1] - self._times[0])
        else:
            self._step = 0.0

    @staticmethod
    def _build(features: MusicFeatures, smoothing_seconds: float) -> np.ndarray:
        frames = len(features.frame_times)
        if frames == 0:
            return np.zeros(0)

        combined = np.zeros(frames, dtype=float)
        for name, weight in WEIGHTS.items():
            values = np.asarray(getattr(features, name), dtype=float)
            if values.size != frames:
                continue
            combined += weight * values

        if not np.any(combined):
            return np.zeros(frames)

        step = 0.0
        if frames >= 2:
            step = float(features.frame_times[1] - features.frame_times[0])
        width = max(1, int(round(smoothing_seconds / step))) if step > 0 else 1

        if width > 1:
            # Edge-padded moving average, so the intro and outro are not dragged
            # towards zero by an implicit silent surround.
            pad = width // 2
            padded = np.pad(combined, pad, mode="edge")
            kernel = np.ones(width) / width
            combined = np.convolve(padded, kernel, mode="same")[pad : pad + frames]

        low = float(np.percentile(combined, LOW_PERCENTILE))
        high = float(np.percentile(combined, HIGH_PERCENTILE))
        if high - low < 1e-9:
            return np.full(frames, 0.5)

        return np.clip((combined - low) / (high - low), 0.0, 1.0)

    def at(self, time: float) -> float:
        """Intensity at ``time``, in 0..1. Constant outside the analysed range."""
        if self._values.size == 0:
            return 0.5
        if self._step <= 0.0:
            return float(self._values[0])
        index = int(round(time / self._step))
        index = max(0, min(index, self._values.size - 1))
        return float(self._values[index])

    def summary(self) -> dict:
        """Spread of the curve, for reporting and for sanity checks in tests."""
        if self._values.size == 0:
            return {"frames": 0}
        return {
            "frames": int(self._values.size),
            "min": round(float(self._values.min()), 4),
            "mean": round(float(self._values.mean()), 4),
            "max": round(float(self._values.max()), 4),
        }
