"""Deterministic initial sample correction for matched incoming material.

The Phase 4D transform owns the incoming window's tempo and duration.  This
module applies the separate Phase 4E initial cue correction to that already
transformed window.  Only the transformed window is shifted; the incoming
source prefix and suffix remain byte-for-byte represented in the returned
buffer.  The correction is an integer sample displacement, so no second time
stretch, resampling, or pitch operation is introduced.
"""

from __future__ import annotations

import numpy as np

from ..animation.cues import BeatTimeline
from ..transition import TransitionPlan
from .beat_phase import BeatPhaseRelationship, build_beat_phase_contract
from .tempo_match import (
    TempoMatch,
    _half_up_sample_count,
    stretch_incoming_transition,
    tempo_match,
)


def _as_float_pcm(samples: object) -> tuple[np.ndarray, bool]:
    """Validate PCM and return an owned float64 array plus its mono flag."""
    try:
        raw = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed_samples must be a numeric PCM array") from exc
    if raw.ndim not in (1, 2):
        raise ValueError(
            "transformed_samples must have shape (frames,) or (frames, channels)"
        )
    if raw.dtype.kind not in "fiu":
        raise ValueError("transformed_samples must contain real numeric samples")
    try:
        pcm = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("transformed_samples must contain float-compatible samples") from exc
    if pcm.ndim == 2 and pcm.shape[1] < 1:
        raise ValueError("transformed_samples must have at least one channel")
    if pcm.shape[0] < 1:
        raise ValueError("transformed_samples must not be empty")
    if not np.all(np.isfinite(pcm)):
        raise ValueError("transformed_samples must contain only finite samples")
    was_mono = pcm.ndim == 1
    if was_mono:
        pcm = pcm[:, None]
    return pcm, was_mono


def _correction_samples(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("correction_samples must be an integer")
    return int(value)


def _cue_index_for_plan_time(timeline: BeatTimeline, time: float, name: str) -> int:
    """Resolve a plan's selected timestamp to an actually detected cue."""
    if not isinstance(timeline, BeatTimeline):
        raise TypeError(f"{name} must be a BeatTimeline")
    index = timeline.index_before(time)
    if index < 0:
        raise ValueError(f"{name} plan anchor is not a detected timeline cue")
    cue_time = timeline.cue(index).scheduled_time
    if cue_time != time:
        raise ValueError(f"{name} plan anchor is not a detected timeline cue")
    return index


def phase_relationship_for_match(
    plan: TransitionPlan,
    match: TempoMatch,
    outgoing_timeline: BeatTimeline,
    incoming_timeline: BeatTimeline,
) -> BeatPhaseRelationship:
    """Measure the selected-cue relationship for an accepted Phase 4D match."""
    if not isinstance(plan, TransitionPlan):
        raise ValueError("plan must be a TransitionPlan")
    if not isinstance(match, TempoMatch):
        raise ValueError("match must be a TempoMatch")
    if match.sample_rate <= 0:
        raise ValueError("match sample rate must be positive")
    outgoing_index = _cue_index_for_plan_time(
        outgoing_timeline, plan.outgoing_time, "outgoing"
    )
    incoming_index = _cue_index_for_plan_time(
        incoming_timeline, plan.incoming_time, "incoming"
    )
    return build_beat_phase_contract(
        outgoing_timeline,
        incoming_timeline,
        outgoing_cue_index=outgoing_index,
        incoming_cue_index=incoming_index,
        sample_rate=match.sample_rate,
        source_sample_count=match.incoming_sample_count,
        transformed_sample_count=match.matched_sample_count,
    )


def _apply_initial_incoming_cue_correction(
    transformed_samples: np.ndarray,
    plan: TransitionPlan,
    sample_rate: int,
    correction_samples: int,
    match: TempoMatch,
) -> np.ndarray:
    """Shift the matched incoming window by an integer number of samples.

    ``correction_samples`` uses the phase contract's sign convention: a
    positive value delays incoming material and a negative value advances it.
    The vacated edge is held at the nearest window sample.  This keeps the
    output length and the Phase 4D window mapping deterministic while leaving
    every interior sample in its original order, which preserves pitch.

    The input is copied before processing.  The returned buffer has the same
    source prefix and suffix lengths as ``transformed_samples`` and owns its
    memory.  A correction that would consume the complete matched window is
    rejected because it would no longer retain the selected incoming cue.
    """
    if not isinstance(plan, TransitionPlan):
        raise ValueError("plan must be a TransitionPlan")
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
        sample_rate, (int, np.integer)
    ) or int(sample_rate) <= 0:
        raise ValueError("sample_rate must be a positive integer")
    correction = _correction_samples(correction_samples)
    pcm, was_mono = _as_float_pcm(transformed_samples)
    anchor = _half_up_sample_count(plan.incoming_time, match.sample_rate)
    window_count = match.matched_sample_count
    end = anchor + window_count
    if end > pcm.shape[0]:
        raise ValueError(
            "transformed_samples do not contain the complete matched transition window"
        )
    if abs(correction) >= window_count:
        raise ValueError("correction_samples must leave part of the matched window")
    if correction == 0:
        return np.array(pcm[:, 0] if was_mono else pcm, dtype=np.float64, copy=True)

    window = pcm[anchor:end]
    corrected = np.empty_like(window)
    if correction > 0:
        corrected[:correction] = window[0]
        corrected[correction:] = window[:-correction]
    else:
        advance = -correction
        corrected[:-advance] = window[advance:]
        corrected[-advance:] = window[-1]

    prefix = pcm[:anchor]
    suffix = pcm[end:]
    result = np.concatenate((prefix, corrected, suffix), axis=0)
    if was_mono:
        return np.array(result[:, 0], dtype=np.float64, copy=True)
    return np.array(result, dtype=np.float64, copy=True)


def apply_initial_incoming_cue_correction(
    transformed_samples: np.ndarray,
    plan: TransitionPlan,
    sample_rate: int,
    correction_samples: int,
) -> np.ndarray:
    """Apply an initial correction using the plan's accepted Phase 4D match."""
    match = tempo_match(plan, sample_rate)
    return _apply_initial_incoming_cue_correction(
        transformed_samples, plan, sample_rate, correction_samples, match
    )


def phase_correct_incoming_transition(
    incoming_samples: np.ndarray,
    plan: TransitionPlan,
    sample_rate: int,
    outgoing_timeline: BeatTimeline,
    incoming_timeline: BeatTimeline,
) -> np.ndarray:
    """Tempo-match then apply the deterministic initial cue correction.

    The Phase 4D stretcher is called exactly once, and the resulting material
    is handed to the integer-only boundary operation above.  The relationship
    is available from :func:`phase_relationship_for_match` when a caller needs
    to report the applied correction.
    """
    match = tempo_match(plan, sample_rate)
    relationship = phase_relationship_for_match(
        plan, match, outgoing_timeline, incoming_timeline
    )
    transformed = stretch_incoming_transition(incoming_samples, plan, sample_rate)
    return _apply_initial_incoming_cue_correction(
        transformed,
        plan,
        sample_rate,
        relationship.initial_correction_samples,
        match,
    )


def phase_correct_incoming_transition_with_relationship(
    incoming_samples: np.ndarray,
    plan: TransitionPlan,
    sample_rate: int,
    outgoing_timeline: BeatTimeline,
    incoming_timeline: BeatTimeline,
) -> tuple[np.ndarray, BeatPhaseRelationship]:
    """Return corrected incoming PCM together with its deterministic relation."""
    match = tempo_match(plan, sample_rate)
    relationship = phase_relationship_for_match(
        plan, match, outgoing_timeline, incoming_timeline
    )
    transformed = stretch_incoming_transition(incoming_samples, plan, sample_rate)
    corrected = _apply_initial_incoming_cue_correction(
        transformed,
        plan,
        sample_rate,
        relationship.initial_correction_samples,
        match,
    )
    return corrected, relationship


# Descriptive aliases for callers that name the material boundary directly.
correct_incoming_transition_phase = phase_correct_incoming_transition
phase_corrected_material = phase_correct_incoming_transition
