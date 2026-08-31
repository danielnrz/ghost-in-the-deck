"""Offline (pre-playback) analysis of a track with librosa.

Phase 0 deliberately analyses the whole file before playback starts. Live input
is out of scope.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .decode import to_wav
from .features import MusicFeatures, normalise

ANALYSIS_SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048

# Band edges in Hz. Broad on purpose: the animation layer only needs to know
# roughly where the energy sits, not an exact spectral breakdown.
BANDS = {
    "bass_energy": (20.0, 250.0),
    "mid_energy": (250.0, 4000.0),
    "high_energy": (4000.0, 16000.0),
}


def _band_rms(magnitude: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.zeros(magnitude.shape[1])
    return np.sqrt(np.mean(magnitude[mask] ** 2, axis=0))


def analyse(path: Path | str, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> MusicFeatures:
    """Analyse one audio file and return its features."""
    source = Path(path)
    wav = to_wav(source)

    y, sr = librosa.load(str(wav), sr=sample_rate, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH, units="frames"
    )
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH).tolist()

    magnitude = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    rms = librosa.feature.rms(S=magnitude, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    frame_count = magnitude.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(frame_count), sr=sr, hop_length=HOP_LENGTH
    )

    bands = {
        name: normalise(_band_rms(magnitude, freqs, low, high)).tolist()
        for name, (low, high) in BANDS.items()
    }

    # onset_strength can be one frame off from the STFT frame count; trim to match
    onset = normalise(onset_env)[:frame_count]
    if onset.size < frame_count:
        onset = np.pad(onset, (0, frame_count - onset.size))

    return MusicFeatures(
        track=source.name,
        duration_seconds=duration,
        sample_rate=sr,
        hop_length=HOP_LENGTH,
        bpm=bpm,
        beats=[float(t) for t in beats],
        frame_times=[float(t) for t in frame_times],
        onset_strength=[float(v) for v in onset],
        rms=[float(v) for v in normalise(rms)],
        **bands,
    )
