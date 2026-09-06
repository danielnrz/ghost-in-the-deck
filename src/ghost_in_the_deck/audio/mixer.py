"""Deterministic two-source transition timing and array mixing primitives.

This module does not load audio, stretch time, or depend on Panda3D.  A
``TransitionClock`` turns a planned pair of source cues into one executable
window in which both source clocks advance by the same elapsed amount, and
``execute_transition`` renders that window from already-decoded PCM arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..transition import TransitionPlan


def _as_float_channels(samples: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Validate one PCM buffer and return an owned, channel-normalized view.

    The internal representation always has shape ``(frames, channels)``.  The
    second return value records whether the caller supplied a one-dimensional
    mono buffer so the executor can return the corresponding shape.
    """
    try:
        raw = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric PCM array") from exc

    if raw.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (frames,) or (frames, channels)")
    if raw.dtype.kind in "bc":
        raise ValueError(f"{name} must contain real numeric samples")
    try:
        pcm = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain float-compatible samples") from exc
    if not np.all(np.isfinite(pcm)):
        raise ValueError(f"{name} must contain only finite samples")

    was_mono = pcm.ndim == 1
    if was_mono:
        pcm = pcm[:, None]
    elif pcm.shape[1] < 1:
        raise ValueError(f"{name} must have at least one channel")
    return pcm, was_mono


def _source_start_index(anchor_seconds: float, sample_rate: float, name: str) -> int:
    """Map an absolute source-time anchor to its nearest PCM frame."""
    anchor = float(anchor_seconds)
    if not math.isfinite(anchor) or anchor < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    frame_position = anchor * sample_rate
    if not math.isfinite(frame_position):
        raise ValueError(f"{name} is outside the addressable sample range")
    return int(round(frame_position))


def execute_transition(
    outgoing_samples: np.ndarray,
    incoming_samples: np.ndarray,
    plan: TransitionPlan,
    sample_rate: float,
) -> np.ndarray:
    """Render one planned transition from two decoded PCM sample arrays.

    The two buffers are interpreted as complete source tracks beginning at
    source time zero.  The plan's absolute cue times select the first outgoing
    and incoming frames; every subsequent output frame reads the next frame
    from each source, with no resampling or time-stretching.  The executable
    window is the shorter duration recorded by the plan and uses the linear
    endpoint-owned gain envelope from :func:`linear_crossfade_gains`.

    Mono arrays have shape ``(frames,)`` and multi-channel arrays have shape
    ``(frames, channels)``.  The buffers must have the same channel count and
    enough frames after their respective anchors for the complete transition.
    Inputs are copied before processing and the returned array owns its data.
    """
    if not isinstance(plan, TransitionPlan):
        raise TypeError("plan must be a TransitionPlan")

    outgoing, outgoing_was_mono = _as_float_channels(
        outgoing_samples, "outgoing_samples"
    )
    incoming, incoming_was_mono = _as_float_channels(
        incoming_samples, "incoming_samples"
    )
    if outgoing.shape[1] != incoming.shape[1]:
        raise ValueError(
            "outgoing and incoming arrays must have the same channel count"
        )

    clock = TransitionClock(plan, sample_rate)
    sample_count = clock.sample_count
    if sample_count < 1:
        raise ValueError("transition must contain at least one output sample")

    outgoing_start = _source_start_index(
        clock.outgoing_anchor_seconds, clock.sample_rate, "outgoing anchor"
    )
    incoming_start = _source_start_index(
        clock.incoming_anchor_seconds, clock.sample_rate, "incoming anchor"
    )
    if outgoing_start + sample_count > outgoing.shape[0]:
        raise ValueError(
            "outgoing_samples do not contain the complete transition window"
        )
    if incoming_start + sample_count > incoming.shape[0]:
        raise ValueError(
            "incoming_samples do not contain the complete transition window"
        )

    outgoing_gain, incoming_gain = linear_crossfade_gains(sample_count)
    outgoing_window = outgoing[outgoing_start : outgoing_start + sample_count]
    incoming_window = incoming[incoming_start : incoming_start + sample_count]
    mixed = (
        outgoing_window * outgoing_gain[:, None]
        + incoming_window * incoming_gain[:, None]
    )

    if outgoing_was_mono and incoming_was_mono:
        return np.array(mixed[:, 0], dtype=np.float64, copy=True)
    return np.array(mixed, dtype=np.float64, copy=True)


# ``mix_transition`` is the concise name used by callers that treat this
# module as a mixer; keep the descriptive executor name as the implementation.
mix_transition = execute_transition


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
