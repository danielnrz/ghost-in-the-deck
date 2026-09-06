"""Deterministic two-source transition timing primitives.

This module contains the offline timing contract for a transition.  It does
not load audio, stretch time, apply a crossfade, or depend on Panda3D.  A
``TransitionClock`` turns a planned pair of source cues into one executable
window in which both source clocks advance by the same elapsed amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..transition import TransitionPlan


def linear_crossfade_gains(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return outgoing and incoming gains for a linear crossfade.

    The first sample belongs entirely to the outgoing source and the last
    sample belongs entirely to the incoming source.  A one-sample crossfade
    therefore has outgoing gain ``1.0`` and incoming gain ``0.0``: the only
    sample is owned by the outgoing endpoint.  The returned arrays are new,
    independent ``float64`` arrays.

    Args:
        sample_count: Number of samples in the crossfade; must be a positive
            integer.

    Raises:
        ValueError: If ``sample_count`` is not a positive integer.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)):
        raise ValueError("sample_count must be a positive integer")
    if sample_count < 1:
        raise ValueError("sample_count must be a positive integer")

    outgoing = np.linspace(1.0, 0.0, int(sample_count), dtype=np.float64)
    incoming = 1.0 - outgoing
    return outgoing, incoming


@dataclass(frozen=True)
class TransitionClock:
    """The frozen, absolute-time clock contract for one transition.

    ``plan`` supplies the two source anchors and their independently measured
    durations.  The executable window is deliberately the shorter of those
    durations: no source is time-stretched to make unequal tempos line up.
    Both source clocks then advance by exactly ``elapsed_seconds`` from their
    own anchor.  ``sample_count`` is the deterministic number of samples in
    that window, using floor semantics for a partial final sample.
    """

    plan: TransitionPlan
    sample_rate: float

    def __post_init__(self) -> None:
        rate = float(self.sample_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("sample_rate must be a finite positive number")
        object.__setattr__(self, "sample_rate", rate)

        for name in ("outgoing_duration_seconds", "incoming_duration_seconds"):
            duration = float(getattr(self.plan, name))
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def outgoing_anchor_seconds(self) -> float:
        """Absolute source time at the start of the outgoing transition."""
        return self.plan.outgoing_time

    @property
    def incoming_anchor_seconds(self) -> float:
        """Absolute source time at the start of the incoming transition."""
        return self.plan.incoming_time

    @property
    def outgoing_anchor(self) -> float:
        """Short alias for ``outgoing_anchor_seconds``."""
        return self.outgoing_anchor_seconds

    @property
    def incoming_anchor(self) -> float:
        """Short alias for ``incoming_anchor_seconds``."""
        return self.incoming_anchor_seconds

    @property
    def executable_duration_seconds(self) -> float:
        """The shortest source window, with no time-stretching."""
        return min(
            self.plan.outgoing_duration_seconds,
            self.plan.incoming_duration_seconds,
        )

    @property
    def duration_seconds(self) -> float:
        """The transition's executable duration."""
        return self.executable_duration_seconds

    @property
    def sample_count(self) -> int:
        """Number of whole output samples in the executable duration."""
        return int(self.executable_duration_seconds * self.sample_rate)

    def _check_elapsed(self, elapsed_seconds: float) -> float:
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed):
            raise ValueError("elapsed_seconds must be finite")
        if not 0.0 <= elapsed <= self.executable_duration_seconds:
            raise ValueError(
                "elapsed_seconds must be within the executable transition window"
            )
        return elapsed

    def outgoing_time_at(self, elapsed_seconds: float) -> float:
        """Map elapsed transition time to an absolute outgoing source time."""
        return self.outgoing_anchor_seconds + self._check_elapsed(elapsed_seconds)

    def incoming_time_at(self, elapsed_seconds: float) -> float:
        """Map elapsed transition time to an absolute incoming source time."""
        return self.incoming_anchor_seconds + self._check_elapsed(elapsed_seconds)

    def source_times_at(self, elapsed_seconds: float) -> tuple[float, float]:
        """Return outgoing and incoming absolute source times at elapsed time."""
        elapsed = self._check_elapsed(elapsed_seconds)
        return (
            self.outgoing_anchor_seconds + elapsed,
            self.incoming_anchor_seconds + elapsed,
        )

    @property
    def outgoing_time(self) -> float:
        """Compatibility alias for the outgoing source anchor."""
        return self.outgoing_anchor_seconds

    @property
    def incoming_time(self) -> float:
        """Compatibility alias for the incoming source anchor."""
        return self.incoming_anchor_seconds
