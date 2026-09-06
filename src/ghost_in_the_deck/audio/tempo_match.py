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
``0.80 <= incoming_playback_rate <= 1.25``.  A request outside that range is
rejected rather than silently changing the target.  The outgoing transition
duration is authoritative after matching.  Source and matched sample counts
are both rounded to the nearest sample with explicit half-up ties:
``floor(seconds * sample_rate + 0.5)``.
"""

from __future__ import annotations

import math
import numbers
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..transition import TransitionPlan


# This is a contract boundary for the later offline stretcher, not a claim
# that every future real-time backend will support the same quality envelope.
SUPPORTED_PLAYBACK_RATE_MIN = 0.80
SUPPORTED_PLAYBACK_RATE_MAX = 1.25

# Rubber Band needs a short non-silent lookahead to flush the end of a finite
# input window.  Without it, the ffmpeg filter can return tens of milliseconds
# fewer frames than the requested wall-clock window.
_RUBBERBAND_LOOKAHEAD_SECONDS = 0.25


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


def _as_float_pcm(samples: object) -> tuple[np.ndarray, bool]:
    """Validate PCM and return an owned float64 array plus its mono shape flag."""
    try:
        raw = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("incoming_samples must be a numeric PCM array") from exc
    if raw.ndim not in (1, 2):
        raise ValueError(
            "incoming_samples must have shape (frames,) or (frames, channels)"
        )
    if raw.dtype.kind not in "fiu":
        raise ValueError("incoming_samples must contain real numeric samples")
    try:
        pcm = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("incoming_samples must contain float-compatible samples") from exc
    if pcm.ndim == 2 and pcm.shape[1] < 1:
        raise ValueError("incoming_samples must have at least one channel")
    if pcm.shape[0] < 1:
        raise ValueError("incoming_samples must not be empty")
    if not np.all(np.isfinite(pcm)):
        raise ValueError("incoming_samples must contain only finite samples")
    return pcm, pcm.ndim == 1


def _incoming_anchor_sample(plan: "TransitionPlan", sample_rate: int) -> int:
    anchor_seconds = _validate_number(plan.incoming_time, "incoming anchor")
    return _half_up_sample_count(anchor_seconds, sample_rate)


def _rubberband_transform(
    material: np.ndarray,
    *,
    sample_rate: int,
    channels: int,
    tempo: float,
    output_sample_count: int,
    lookahead: np.ndarray | None = None,
) -> np.ndarray:
    """Run the local pitch-neutral Rubber Band filter on one PCM window.

    A finite filter input needs deterministic non-silent lookahead so the
    backend can flush its final analysis blocks.  The lookahead is never
    returned: only the requested transformed window belongs to the result.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg with the rubberband filter is required")

    flush_lookahead_count = max(
        1, math.ceil(sample_rate * _RUBBERBAND_LOOKAHEAD_SECONDS)
    )
    # The plan's outgoing duration is authoritative even for a manually
    # constructed plan whose BPM/duration fields are not algebraically paired.
    # Supply enough continuation for the requested output first, then add the
    # backend flush margin.
    required_input_count = math.ceil(output_sample_count * tempo)
    lookahead_count = max(
        flush_lookahead_count,
        required_input_count - material.shape[0] + flush_lookahead_count,
    )
    continuation = np.empty((lookahead_count, channels), dtype=np.float64)
    supplied = 0
    if lookahead is not None:
        if lookahead.ndim != 2 or lookahead.shape[1] != channels:
            raise ValueError("rubberband lookahead has an incompatible channel shape")
        supplied = min(lookahead.shape[0], lookahead_count)
        continuation[:supplied] = lookahead[:supplied]
    if supplied < lookahead_count:
        # A source is allowed to end exactly at its transition window.  A
        # repeated tail is deterministic and keeps the backend flush material
        # non-silent without inventing zero padding at the audible boundary.
        tail = material[-min(material.shape[0], lookahead_count) :]
        repetitions = math.ceil((lookahead_count - supplied) / tail.shape[0])
        repeated = np.tile(tail, (repetitions, 1))
        continuation[supplied:] = repeated[: lookahead_count - supplied]
    filter_material = np.concatenate((material, continuation), axis=0)

    # Raw float PCM keeps this boundary independent of file codecs and gives
    # the filter no opportunity to replace or rewrite either source file.
    command = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-filter_threads",
        "1",
        "-f",
        "f64le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-af",
        (
            f"rubberband=tempo={tempo:.17g}:pitch=1.0:"
            "formant=preserved:channels=together"
        ),
        "-f",
        "f64le",
        "-acodec",
        "pcm_f64le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=np.ascontiguousarray(filter_material, dtype=np.float64).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = f": {exc.stderr.decode(errors='replace').strip()}"
        raise RuntimeError(f"rubberband could not transform incoming PCM{detail}") from exc

    raw_output = completed.stdout
    frame_width = channels * np.dtype("<f8").itemsize
    if len(raw_output) == 0 or len(raw_output) % frame_width:
        raise RuntimeError("rubberband returned an invalid PCM buffer")
    transformed = np.frombuffer(raw_output, dtype="<f8").reshape(-1, channels)
    if not np.all(np.isfinite(transformed)):
        raise RuntimeError("rubberband returned non-finite PCM")

    if transformed.shape[0] < output_sample_count:
        raise RuntimeError(
            "rubberband produced "
            f"{transformed.shape[0]} frames for a {output_sample_count}-frame "
            "window even after deterministic lookahead"
        )
    return np.array(transformed[:output_sample_count], dtype=np.float64, copy=True)


def stretch_incoming_transition(
    incoming_samples: np.ndarray,
    plan: "TransitionPlan",
    sample_rate: int,
) -> np.ndarray:
    """Pitch-preserve only the incoming transition window of one PCM source.

    The input is the complete incoming source beginning at source time zero.
    The window starts at ``plan.incoming_time`` and has the plan's incoming
    duration.  It is transformed to the plan's authoritative outgoing duration
    using the documented playback-rate policy.  The returned array is a new
    source buffer: the prefix remains before the same cue frame, and the suffix
    remains untouched after the transformed window.  No file or source path is
    accessed by this function.

    Mono input retains shape ``(frames,)``; multi-channel input retains shape
    ``(frames, channels)``.  The returned transition window has exactly the
    half-up sample count from :func:`tempo_match`.
    """
    match = tempo_match(plan, sample_rate)
    pcm, was_mono = _as_float_pcm(incoming_samples)
    channels = 1 if was_mono else pcm.shape[1]
    source_count = match.incoming_sample_count
    output_count = match.matched_sample_count
    if source_count < 1 or output_count < 1:
        raise ValueError("incoming transition material must contain samples")

    anchor = _incoming_anchor_sample(plan, match.sample_rate)
    end = anchor + source_count
    if end > pcm.shape[0]:
        raise ValueError(
            "incoming_samples do not contain the complete transition window"
        )

    material = pcm[anchor:end]
    if was_mono:
        material = material[:, None]
    if match.incoming_playback_rate == 1.0 and source_count == output_count:
        transformed = np.array(material, dtype=np.float64, copy=True)
    else:
        lookahead = pcm[end : end + max(
            1, math.ceil(match.sample_rate * _RUBBERBAND_LOOKAHEAD_SECONDS)
        )]
        if was_mono:
            lookahead = lookahead[:, None]
        transformed = _rubberband_transform(
            material,
            sample_rate=match.sample_rate,
            channels=channels,
            tempo=match.incoming_playback_rate,
            output_sample_count=output_count,
            lookahead=lookahead,
        )
    prefix = pcm[:anchor, None] if was_mono else pcm[:anchor]
    suffix = pcm[end:, None] if was_mono else pcm[end:]
    result = np.concatenate((prefix, transformed, suffix), axis=0)
    if was_mono:
        return np.array(result[:, 0], dtype=np.float64, copy=True)
    return np.array(result, dtype=np.float64, copy=True)


# Descriptive aliases for callers that name the operation by its material
# boundary rather than by the selected backend.
pitch_preserving_incoming_transition = stretch_incoming_transition
transform_incoming_transition = stretch_incoming_transition
