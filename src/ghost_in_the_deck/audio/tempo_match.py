"""Deterministic BPM-match timing contract for an incoming transition source.

This module describes the input a later pitch-preserving time-stretcher will
need.  It does not process PCM, resample audio or change ``TransitionPlan`` or
the Phase 4B mixer.

The contract targets the outgoing deck's BPM.  ``incoming_playback_rate`` is
the conventional playback-speed multiplier applied to the incoming source:

``incoming_playback_rate = plan.bpm_a / plan.bpm_b``

Consequently, a faster incoming track gets a rate below ``1.0`` and plays for
longer, while a slower incoming track gets a rate above ``1.0`` and plays for
less time.  The rate is not folded through a half-time or double-time
interpretation; the two BPM values are used directly.

The supported offline policy is an inclusive playback-rate range of
``0.5 <= incoming_playback_rate <= 2.0``.  A request outside that range is
rejected rather than silently changing the target.  The outgoing transition
duration is authoritative after matching.  Source and matched sample counts
are both rounded to the nearest sample with explicit half-up ties:
``floor(seconds * sample_rate + 0.5)``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..transition import TransitionPlan


# This is a contract boundary for the later offline stretcher, not a claim
# that every future real-time backend will support the same quality envelope.
SUPPORTED_PLAYBACK_RATE_MIN = 0.5
SUPPORTED_PLAYBACK_RATE_MAX = 2.0


def _validate_number(value: object, name: str, *, positive: bool = False) -> float:
    description = "positive" if positive else "non-negative"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite {description} number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {description} number") from exc
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        raise ValueError(f"{name} must be a finite {description} number")
    return number


def _half_up_sample_count(duration_seconds: float, sample_rate: int) -> int:
    try:
        sample_position = duration_seconds * sample_rate
    except OverflowError as exc:
        raise ValueError(
            "duration and sample_rate exceed the addressable sample range"
        ) from exc
    if not math.isfinite(sample_position):
        raise ValueError("duration and sample_rate exceed the addressable sample range")
    return math.floor(sample_position + 0.5)


@dataclass(frozen=True)
class TempoMatch:
    """The resolved timing values for matching one incoming plan window.

    ``incoming_duration_seconds`` and ``incoming_sample_count`` describe the
    unprocessed incoming source window in the existing plan.  The output
    duration and sample count describe the incoming window after applying the
    pitch-preserving rate.  ``sample_rate`` is retained so the sample-count
    convention is visible in the immutable result.
    """

    target_bpm: float
    incoming_bpm: float
    incoming_playback_rate: float
    incoming_duration_seconds: float
    matched_duration_seconds: float
    sample_rate: int
    incoming_sample_count: int
    matched_sample_count: int

    @property
    def rate(self) -> float:
        """The incoming playback-rate multiplier."""
        return self.incoming_playback_rate

    @property
    def output_duration_seconds(self) -> float:
        """The duration of the matched incoming output window."""
        return self.matched_duration_seconds

    @property
    def output_sample_count(self) -> int:
        """The deterministic sample count of the matched output window."""
        return self.matched_sample_count


def tempo_match(plan: "TransitionPlan", sample_rate: int) -> TempoMatch:
    """Resolve the incoming BPM-match contract for an existing plan.

    ``plan.bpm_a`` is the target because deck A is the outgoing source and
    ``plan.bpm_b`` is the incoming source.  The plan's outgoing duration is
    the authoritative matched wall-clock duration; for plans produced by the
    planner it equals the incoming duration divided by the returned rate.

    No half-time/double-time folding is performed.  Both source BPMs must be
    positive, and the direct rate must fall inside the documented inclusive
    supported range.
    """
    from ..transition import TransitionPlan

    if not isinstance(plan, TransitionPlan):
        raise ValueError("plan must be a TransitionPlan")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, numbers.Integral):
        raise ValueError("sample_rate must be a positive integer")
    rate_hz = int(sample_rate)
    if rate_hz <= 0:
        raise ValueError("sample_rate must be a positive integer")

    target_bpm = _validate_number(plan.bpm_a, "target BPM", positive=True)
    incoming_bpm = _validate_number(plan.bpm_b, "incoming BPM", positive=True)
    incoming_duration = _validate_number(
        plan.incoming_duration_seconds, "incoming duration"
    )
    matched_duration = _validate_number(
        plan.outgoing_duration_seconds, "outgoing duration"
    )

    incoming_playback_rate = target_bpm / incoming_bpm
    if not (
        SUPPORTED_PLAYBACK_RATE_MIN
        <= incoming_playback_rate
        <= SUPPORTED_PLAYBACK_RATE_MAX
    ):
        raise ValueError(
            "incoming playback rate is outside the supported range "
            f"[{SUPPORTED_PLAYBACK_RATE_MIN}, {SUPPORTED_PLAYBACK_RATE_MAX}]"
        )

    return TempoMatch(
        target_bpm=target_bpm,
        incoming_bpm=incoming_bpm,
        incoming_playback_rate=incoming_playback_rate,
        incoming_duration_seconds=incoming_duration,
        matched_duration_seconds=matched_duration,
        sample_rate=rate_hz,
        incoming_sample_count=_half_up_sample_count(incoming_duration, rate_hz),
        matched_sample_count=_half_up_sample_count(matched_duration, rate_hz),
    )


# Descriptive spelling for callers that want to emphasize this is a plan
# calculation rather than an audio-processing operation.
tempo_match_for_plan = tempo_match
