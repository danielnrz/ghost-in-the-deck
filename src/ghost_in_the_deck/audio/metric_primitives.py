"""Pure deterministic PCM metrics for offline transition evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _samples(values: Sequence[float] | np.ndarray, name: str = "samples") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty one-dimensional finite array")
    return array


@dataclass(frozen=True)
class PeakHeadroom:
    peak: float
    headroom: float
    headroom_db: float


@dataclass(frozen=True)
class EnergyContinuity:
    before_rms: float
    after_rms: float
    dip_rms: float
    dip_ratio: float
    boundary_jump: float


@dataclass(frozen=True)
class Interference:
    correlation: float
    sum_rms: float
    expected_rms: float
    interference_ratio: float
    cancellation: float


def rms_energy(samples: Sequence[float] | np.ndarray) -> float:
    values = _samples(samples)
    return float(np.sqrt(np.mean(values * values)))


def measure_peak_headroom(samples: Sequence[float] | np.ndarray, *, full_scale: float = 1.0) -> PeakHeadroom:
    values = _samples(samples)
    scale = float(full_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("full_scale must be finite and positive")
    peak = float(np.max(np.abs(values)))
    headroom = scale - peak
    ratio = headroom / scale
    headroom_db = math.inf if ratio == 0.0 else 20.0 * math.log10(abs(ratio))
    return PeakHeadroom(peak, headroom, headroom_db)


def peak_headroom(samples, *, full_scale: float = 1.0) -> PeakHeadroom:
    return measure_peak_headroom(samples, full_scale=full_scale)


def measure_energy_continuity(samples: Sequence[float] | np.ndarray, boundary: int, *, window: int = 1) -> EnergyContinuity:
    values = _samples(samples)
    if not isinstance(boundary, (int, np.integer)) or not 0 < boundary < values.size:
        raise ValueError("boundary must be an interior sample index")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if boundary < window or values.size - boundary < window:
        raise ValueError("window must fit on both sides of boundary")
    before = values[boundary - window:boundary]
    after = values[boundary:boundary + window]
    before_rms, after_rms = rms_energy(before), rms_energy(after)
    dip_rms = rms_energy(np.concatenate((before, after)))
    reference = max(before_rms, after_rms)
    dip_ratio = dip_rms / reference if reference else 1.0
    return EnergyContinuity(before_rms, after_rms, dip_rms, float(dip_ratio), float(abs(after[0] - before[-1])))


def energy_continuity(samples, boundary: int, *, window: int = 1) -> EnergyContinuity:
    return measure_energy_continuity(samples, boundary, window=window)


def measure_transient_interference(first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray) -> Interference:
    left, right = _samples(first, "first"), _samples(second, "second")
    if left.size != right.size:
        raise ValueError("first and second must have equal lengths")
    left_rms, right_rms = rms_energy(left), rms_energy(right)
    sum_rms = rms_energy(left + right)
    energy = left_rms * left_rms + right_rms * right_rms
    norm_product = float(np.linalg.norm(left) * np.linalg.norm(right))
    correlation = float(np.dot(left, right) / norm_product) if norm_product else 0.0
    interference_ratio = sum_rms * sum_rms / energy if energy else 1.0
    expected_rms = math.sqrt(energy)
    total = left_rms + right_rms
    cancellation = 1.0 - sum_rms / total if total else 0.0
    return Interference(correlation, sum_rms, expected_rms, float(interference_ratio), float(cancellation))


def transient_phase_interference(first, second) -> Interference:
    return measure_transient_interference(first, second)


__all__ = ["PeakHeadroom", "EnergyContinuity", "Interference", "rms_energy", "measure_peak_headroom", "peak_headroom", "measure_energy_continuity", "energy_continuity", "measure_transient_interference", "transient_phase_interference"]
