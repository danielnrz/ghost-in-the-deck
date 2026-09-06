"""Pure beat-phase and initial sample-correction contract.

The contract sits after cue selection and after the Phase 4D incoming-window
transform.  It only relates existing ``BeatTimeline`` cue timestamps to the
sample boundaries used by that transform; it does not choose cues, inspect
musical structure, process PCM, or continuously warp a beat grid.

All sample spans are half-open: ``[start, end)`` includes ``start`` and
excludes ``end``.  Timestamps are converted to sample positions with the same
deterministic half-up convention as the Phase 4D tempo-match contract.  A cue
may therefore sit fractionally before or after the rounded sample used as its
anchor.  That signed fractional position is the only phase quantity needed for
the initial correction.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from ..animation.cues import BeatTimeline


def _sample_rate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("sample_rate must be a positive integer")
    rate = int(value)
    if rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return rate


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _half_up(value: float) -> int:
    """Round a non-negative sample position to the nearest sample, half-up."""
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("sample position must be finite and non-negative")
    return math.floor(value + 0.5)


def _signed_nearest(value: float) -> int:
    """Round a signed sample offset to nearest integer, ties away from zero."""
    if not math.isfinite(value):
        raise ValueError("signed sample offset must be finite")
    magnitude = math.floor(abs(value) + 0.5)
    return magnitude if value >= 0.0 else -magnitude


@dataclass(frozen=True)
class SampleSpan:
    """A non-negative half-open sample span ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be a non-negative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.end < self.start:
            raise ValueError("sample span end must not precede start")

    @property
    def count(self) -> int:
        """Number of samples in the half-open span."""
        return self.end - self.start


def half_open_time_span(
    start_seconds: float, end_seconds: float, sample_rate: int
) -> SampleSpan:
    """Map a non-negative time span to a deterministic half-open sample span.

    Both boundaries use half-up nearest-sample rounding.  The result is a
    boundary mapping, not an inclusive end index: a span ending at sample 12
    is represented by ``end == 12`` and does not include sample 12.
    """
    rate = _sample_rate(sample_rate)
    start = _finite_number(start_seconds, "start_seconds")
    end = _finite_number(end_seconds, "end_seconds")
    if start < 0.0 or end < 0.0:
        raise ValueError("time span boundaries must be non-negative")
    if end < start:
        raise ValueError("end_seconds must not precede start_seconds")
    return SampleSpan(_half_up(start * rate), _half_up(end * rate))


@dataclass(frozen=True)
class TransformedSampleMapping:
    """Half-open source-to-transformed mapping for one Phase 4D window.

    The mapping is affine over sample *boundaries*.  It maps source boundary
    ``source_span.start`` to transformed boundary ``transformed_span.start``
    and the source end boundary to the transformed end boundary.  Mapping a
    source sub-span floors both transformed boundaries, preserving the
    half-open convention.  This object describes indices only; it does not
    transform audio and is not a continuous beat-grid correction.
    """

    source_span: SampleSpan
    transformed_span: SampleSpan

    def __post_init__(self) -> None:
        if self.source_span.count < 1:
            raise ValueError("source span must contain at least one sample")
        if self.transformed_span.count < 1:
            raise ValueError("transformed span must contain at least one sample")

    @property
    def source_count(self) -> int:
        return self.source_span.count

    @property
    def transformed_count(self) -> int:
        return self.transformed_span.count

    @property
    def scale(self) -> float:
        """Output samples per source sample for boundary interpolation."""
        return self.transformed_count / self.source_count

    def source_boundary_to_transformed(self, source_boundary: int) -> int:
        """Map a source boundary, including either half-open endpoint."""
        if isinstance(source_boundary, bool) or not isinstance(
            source_boundary, numbers.Integral
        ):
            raise ValueError("source boundary must be an integer")
        boundary = int(source_boundary)
        if not self.source_span.start <= boundary <= self.source_span.end:
            raise ValueError("source boundary is outside the source span")
        relative = boundary - self.source_span.start
        return self.transformed_span.start + math.floor(relative * self.scale)

    def map_source_span(self, source_span: SampleSpan) -> SampleSpan:
        """Map a contained source span using half-open boundary semantics."""
        if not self.source_span.start <= source_span.start <= source_span.end <= self.source_span.end:
            raise ValueError("source sub-span is outside the source span")
        return SampleSpan(
            self.source_boundary_to_transformed(source_span.start),
            self.source_boundary_to_transformed(source_span.end),
        )

    def map_source_position(self, source_position: float) -> float:
        """Map a continuous source boundary position to output samples.

        Positions at either endpoint are accepted.  This small continuous
        helper is used only for the cue's fractional sample offset around the
        transform's anchored start; it does not imply ongoing time warping.
        """
        position = _finite_number(source_position, "source_position")
        if not self.source_span.start <= position <= self.source_span.end:
            raise ValueError("source position is outside the source span")
        return self.transformed_span.start + (
            position - self.source_span.start
        ) * self.scale


# A singular spelling is convenient at call sites and keeps the public name
# aligned with the Phase 4D wording.
TransformedSampleMap = TransformedSampleMapping


@dataclass(frozen=True)
class CueAnchor:
    """One detected ``BeatTimeline`` cue and its sample-domain anchor."""

    cue_index: int
    cue_time_seconds: float
    sample_rate: int
    sample_position: float
    sample_index: int
    fractional_sample_offset: float


def anchor_cue(timeline: BeatTimeline, cue_index: int, sample_rate: int) -> CueAnchor:
    """Anchor one real timeline cue at its deterministic nearest sample.

    ``cue_index`` addresses ``BeatTimeline.cue`` directly, so this contract
    cannot silently replace a detected cue with a virtual beat or a fabricated
    section boundary.
    """
    if not isinstance(timeline, BeatTimeline):
        raise TypeError("timeline must be a BeatTimeline")
    if isinstance(cue_index, bool) or not isinstance(cue_index, numbers.Integral):
        raise ValueError("cue_index must be an integer")
    index = int(cue_index)
    try:
        cue = timeline.cue(index)
    except IndexError as exc:
        raise ValueError("cue_index must identify a detected timeline cue") from exc
    rate = _sample_rate(sample_rate)
    timestamp = _finite_number(cue.scheduled_time, "cue timestamp")
    if timestamp < 0.0:
        raise ValueError("cue timestamp must be non-negative")
    position = timestamp * rate
    sample_index = _half_up(position)
    return CueAnchor(
        cue_index=index,
        cue_time_seconds=timestamp,
        sample_rate=rate,
        sample_position=position,
        sample_index=sample_index,
        fractional_sample_offset=position - sample_index,
    )


# Explicit alias for callers that prefer the noun-first spelling.
cue_anchor = anchor_cue


@dataclass(frozen=True)
class BeatPhaseRelationship:
    """Measured initial relationship between outgoing and incoming cues.

    ``signed_offset_samples`` is incoming minus outgoing at the first output
    sample of the transition.  Positive means the incoming cue is later.  The
    correction has the opposite sign: positive delays the incoming material,
    negative advances it.  Since this is an initial integer offset, residual
    magnitude below half a sample is intentionally left unchanged.
    """

    outgoing_anchor: CueAnchor
    incoming_anchor: CueAnchor
    transformed_mapping: TransformedSampleMapping
    signed_offset_samples: float
    signed_offset_seconds: float
    signed_offset_beats: float
    initial_correction_samples: int

    @property
    def initial_correction_seconds(self) -> float:
        """Initial incoming shift in seconds at the shared sample rate."""
        return self.initial_correction_samples / self.outgoing_anchor.sample_rate

    @property
    def correction_samples(self) -> int:
        """Alias for the deterministic initial incoming correction."""
        return self.initial_correction_samples

    @property
    def phase_offset_samples(self) -> float:
        """Alias for the signed incoming-minus-outgoing phase offset."""
        return self.signed_offset_samples

    @property
    def phase_offset_seconds(self) -> float:
        return self.signed_offset_seconds

    @property
    def residual_offset_samples(self) -> float:
        """Signed offset remaining after the integer initial correction."""
        return self.signed_offset_samples + self.initial_correction_samples


def measure_beat_phase(
    outgoing_anchor: CueAnchor,
    incoming_anchor: CueAnchor,
    transformed_mapping: TransformedSampleMapping,
    *,
    beat_interval_seconds: float,
) -> BeatPhaseRelationship:
    """Measure phase and derive one deterministic initial correction.

    The incoming cue's signed fractional sample offset is mapped through the
    already-resolved Phase 4D window scale.  It is compared with the outgoing
    cue's fractional offset at its unchanged sample anchor.  No later beat is
    inspected and no continuous correction is proposed.
    """
    if not isinstance(outgoing_anchor, CueAnchor):
        raise TypeError("outgoing_anchor must be a CueAnchor")
    if not isinstance(incoming_anchor, CueAnchor):
        raise TypeError("incoming_anchor must be a CueAnchor")
    if not isinstance(transformed_mapping, TransformedSampleMapping):
        raise TypeError("transformed_mapping must be a TransformedSampleMapping")
    if outgoing_anchor.sample_rate != incoming_anchor.sample_rate:
        raise ValueError("cue anchors must use the same sample rate")
    if transformed_mapping.source_span.start != incoming_anchor.sample_index:
        raise ValueError("mapping source span must start at the incoming cue anchor")
    interval = _finite_number(beat_interval_seconds, "beat_interval_seconds")
    if interval <= 0.0:
        raise ValueError("beat_interval_seconds must be positive")

    incoming_position = (
        transformed_mapping.transformed_span.start
        + incoming_anchor.fractional_sample_offset * transformed_mapping.scale
    )
    outgoing_position = outgoing_anchor.fractional_sample_offset
    signed_samples = incoming_position - transformed_mapping.transformed_span.start - outgoing_position
    signed_seconds = signed_samples / outgoing_anchor.sample_rate
    return BeatPhaseRelationship(
        outgoing_anchor=outgoing_anchor,
        incoming_anchor=incoming_anchor,
        transformed_mapping=transformed_mapping,
        signed_offset_samples=signed_samples,
        signed_offset_seconds=signed_seconds,
        signed_offset_beats=signed_seconds / interval,
        initial_correction_samples=-_signed_nearest(signed_samples),
    )


def build_beat_phase_contract(
    outgoing_timeline: BeatTimeline,
    incoming_timeline: BeatTimeline,
    *,
    outgoing_cue_index: int,
    incoming_cue_index: int,
    sample_rate: int,
    source_sample_count: int,
    transformed_sample_count: int,
    transformed_start: int | None = None,
) -> BeatPhaseRelationship:
    """Build the relationship directly from timelines and Phase 4D counts.

    By default the transformed span starts at the same absolute sample as the
    source span, matching ``stretch_incoming_transition``: the incoming prefix
    remains before the transformed window.  A caller using transition-local
    coordinates may provide ``transformed_start=0`` explicitly.
    """
    outgoing = anchor_cue(outgoing_timeline, outgoing_cue_index, sample_rate)
    incoming = anchor_cue(incoming_timeline, incoming_cue_index, sample_rate)
    if isinstance(source_sample_count, bool) or not isinstance(
        source_sample_count, numbers.Integral
    ) or source_sample_count < 1:
        raise ValueError("source_sample_count must be a positive integer")
    if isinstance(transformed_sample_count, bool) or not isinstance(
        transformed_sample_count, numbers.Integral
    ) or transformed_sample_count < 1:
        raise ValueError("transformed_sample_count must be a positive integer")
    if transformed_start is None:
        transformed_start = incoming.sample_index
    elif isinstance(transformed_start, bool) or not isinstance(
        transformed_start, numbers.Integral
    ) or transformed_start < 0:
        raise ValueError("transformed_start must be a non-negative integer")
    source_span = SampleSpan(
        incoming.sample_index, incoming.sample_index + int(source_sample_count)
    )
    transformed_span = SampleSpan(
        int(transformed_start), int(transformed_start) + int(transformed_sample_count)
    )
    mapping = TransformedSampleMapping(source_span, transformed_span)
    return measure_beat_phase(
        outgoing,
        incoming,
        mapping,
        beat_interval_seconds=outgoing_timeline.nominal_interval,
    )


# Concise aliases for callers that treat the result as the correction API.
beat_phase_contract = build_beat_phase_contract
