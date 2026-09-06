"""Deterministic diagnostics for one planned two-deck transition.

The diagnostic reads the two existing beat grids and a selected
``TransitionPlan``.  It reports the tempo relationship, whether both source
anchors are actual detected beats, and the amount by which the two beat clocks
will separate while the frozen mixer advances them by the same elapsed time.
It performs no audio processing and does not choose or revise a plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .animation.cues import BeatTimeline
from .transition import TransitionPlan


def _finite_non_negative(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _real_anchor(timeline: BeatTimeline, time: float) -> bool:
    """Whether ``time`` names a detected cue rather than a virtual beat."""
    cue = timeline.cue_before(time)
    return cue is not None and math.isclose(
        cue.scheduled_time, time, rel_tol=0.0, abs_tol=1e-9
    )


def _wrapped_phase_difference(first: float, second: float) -> float:
    """Return ``second - first`` as the nearest signed beat fraction."""
    difference = second - first
    if difference > 0.5:
        difference -= 1.0
    elif difference <= -0.5:
        difference += 1.0
    return difference


@dataclass(frozen=True)
class TransitionDiagnostics:
    """Measured tempo and predicted no-time-stretch drift for one plan.

    Positive drift means the incoming beat clock is ahead of the outgoing
    clock at the end of the shared transition window.  ``predicted_end_drift``
    is tempo drift only; any initial anchor alignment is reported separately.
    The seconds value uses the outgoing deck's nominal beat as its reference,
    and both beat-fraction values describe that same signed seconds value in
    each deck's beat units.
    """

    bpm_a: float
    bpm_b: float
    bpm_difference: float
    bpm_ratio: float | None
    initial_alignment_seconds: float | None
    initial_alignment_beat_fraction: float | None
    transition_duration_seconds: float
    predicted_end_drift_seconds: float | None
    predicted_end_drift_outgoing_beats: float | None
    predicted_end_drift_incoming_beats: float | None

    @property
    def duration_seconds(self) -> float:
        """The executable shared duration used by the PCM mixer."""
        return self.transition_duration_seconds

    @property
    def predicted_end_drift_beats(self) -> float | None:
        """End drift expressed in outgoing-deck beat units."""
        return self.predicted_end_drift_outgoing_beats

    @property
    def predicted_end_drift_beat_fraction(self) -> float | None:
        """Short name for the outgoing-deck end drift in beat units."""
        return self.predicted_end_drift_outgoing_beats

    @property
    def initial_alignment(self) -> float | None:
        """Initial signed phase difference in beat units, when both anchors are real."""
        return self.initial_alignment_beat_fraction


def diagnose_transition_drift(
    outgoing_timeline: BeatTimeline,
    incoming_timeline: BeatTimeline,
    plan: TransitionPlan,
) -> TransitionDiagnostics:
    """Report deterministic tempo and end-drift measurements for ``plan``.

    The plan's executable duration is the shorter of its outgoing and incoming
    durations, matching ``TransitionClock``.  With positive tempos, the
    predicted signed seconds drift is

    ``duration * (incoming_bpm - outgoing_bpm) / outgoing_bpm``.

    Initial phase is reported only when each plan anchor is exactly one of its
    timeline's detected beats.  A virtual or between-beat anchor therefore
    produces ``None`` instead of being presented as a detected alignment.
    Non-positive BPMs have a defined zero ratio/difference report where
    possible, but no drift prediction because a beat rate is unavailable.
    """
    if not isinstance(outgoing_timeline, BeatTimeline):
        raise TypeError("outgoing_timeline must be a BeatTimeline")
    if not isinstance(incoming_timeline, BeatTimeline):
        raise TypeError("incoming_timeline must be a BeatTimeline")
    if not isinstance(plan, TransitionPlan):
        raise TypeError("plan must be a TransitionPlan")

    bpm_a = _finite_non_negative(outgoing_timeline.features.bpm, "outgoing BPM")
    bpm_b = _finite_non_negative(incoming_timeline.features.bpm, "incoming BPM")
    bpm_difference = abs(bpm_b - bpm_a)
    bpm_ratio = bpm_b / bpm_a if bpm_a > 0.0 else None

    transition_duration = min(
        _finite_non_negative(
            plan.outgoing_duration_seconds, "outgoing transition duration"
        ),
        _finite_non_negative(
            plan.incoming_duration_seconds, "incoming transition duration"
        ),
    )

    initial_alignment_seconds: float | None = None
    initial_alignment_beats: float | None = None
    if _real_anchor(outgoing_timeline, plan.outgoing_time) and _real_anchor(
        incoming_timeline, plan.incoming_time
    ):
        outgoing_phase = outgoing_timeline.phase_at(plan.outgoing_time)
        incoming_phase = incoming_timeline.phase_at(plan.incoming_time)
        initial_alignment_beats = _wrapped_phase_difference(
            outgoing_phase.phase, incoming_phase.phase
        )
        initial_alignment_seconds = (
            initial_alignment_beats * outgoing_timeline.nominal_interval
        )

    predicted_seconds: float | None = None
    predicted_outgoing_beats: float | None = None
    predicted_incoming_beats: float | None = None
    interval_a = outgoing_timeline.nominal_interval
    interval_b = incoming_timeline.nominal_interval
    if (
        bpm_a > 0.0
        and bpm_b > 0.0
        and math.isfinite(interval_a)
        and interval_a > 0.0
        and math.isfinite(interval_b)
        and interval_b > 0.0
    ):
        # Use the actual timelines' nominal intervals for the beat-unit
        # conversion.  This keeps the diagnostic about the supplied beat grids
        # while regular grids reduce exactly to the BPM formula above.
        beat_count_difference = transition_duration / interval_b - transition_duration / interval_a
        predicted_outgoing_beats = beat_count_difference
        predicted_incoming_beats = beat_count_difference * interval_a / interval_b
        predicted_seconds = beat_count_difference * interval_a

    return TransitionDiagnostics(
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        bpm_difference=bpm_difference,
        bpm_ratio=bpm_ratio,
        initial_alignment_seconds=initial_alignment_seconds,
        initial_alignment_beat_fraction=initial_alignment_beats,
        transition_duration_seconds=transition_duration,
        predicted_end_drift_seconds=predicted_seconds,
        predicted_end_drift_outgoing_beats=predicted_outgoing_beats,
        predicted_end_drift_incoming_beats=predicted_incoming_beats,
    )


diagnose_transition = diagnose_transition_drift


__all__ = ["TransitionDiagnostics", "diagnose_transition", "diagnose_transition_drift"]
