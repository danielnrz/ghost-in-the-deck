"""What the DJ is doing, as distinct from how the body feels the beat.

GrooveEngine answers "how is the body feeling the music right now" - a
continuous question, answered every sample. This module answers a different
one: "is the DJ doing something purposeful right now" - deck_glance, lean_in,
hand_to_deck, small_hype, or nothing. That is occasional, not continuous, and
it is kept as a separate concept on purpose: later phases will replace or
extend this scheduler with real DJ intelligence without groove ever needing to
change, and the groove keeps animating underneath whatever this layer decides.

Like BeatTimeline, the whole gesture schedule is built once from the analysed
track and then looked up by absolute time via binary search - no stateful
cursor, no dependence on which times were asked about before. A renderer stall
simply skips part of a gesture and resumes at the progress the schedule says is
current for wherever the clock has moved to.

Selection is deterministic: whether a bar hosts an event, and roughly how far
apart events fall, are decided by hashing the track's own seed with the bar
number - the same primitive GrooveEngine's per-bar variation uses, but salted
independently so an event's placement and its groove personality are two
different, uncorrelated numbers, not two views of one hidden random source.
Nothing here calls random().

What an event *is* - its kind, its side, its strength - is not rolled here at
all. Those come from ``dj_planner``'s deterministic decision functions
(``decide_gesture_kind`` / ``DJActionPlanner.gesture_side_for`` /
``planned_strength``) evaluated at the bar's own start time, so the visible
gesture and the audio action ``dj_planner`` plans for that same moment are two
readings of one decision rather than two independent weighted rolls.

Nothing here imports Panda3D.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass

from ..audio.features import MusicFeatures
from .cues import BEATS_PER_BAR, BeatTimeline
from .energy import EnergyTrack

ACTIONS = ("deck_glance", "lean_in", "hand_to_deck", "small_hype")

# A new gesture may not start less than this many bars after the previous one
# started. Combined with the activation roll below, real spacing works out
# noticeably irregular - never "every four bars", which would read as a metronome
# with a costume on.
MIN_GAP_BARS = 3

# Chance, per eligible bar, that this bar hosts an event at all.
ACTIVATION_PROBABILITY = 0.42

# Energy bands the kind weighting reasons about. Not a section classifier -
# just "restrained / normal / driving", read straight off the same smoothed
# energy curve the groove uses.
LOW_ENERGY = 0.45
HIGH_ENERGY = 0.75

# How far back "rising" looks, in seconds, and how big a jump counts as one.
TREND_LOOKBACK = 3.0
TREND_RISING = 0.12

# Base duration per kind, in seconds, before the small per-event jitter.
BASE_DURATION = {
    "deck_glance": 1.0,
    "lean_in": 1.6,
    "hand_to_deck": 1.2,
    "small_hype": 0.7,
}

# Attack / hold / release fractions of an event's duration. They sum to 1.0.
# A glance is quick down and lingers on the way back; a hype gesture is the
# opposite - a punchy attack that settles slowly, the way a real accent decays.
#
# hand_to_deck's split was 0.35/0.30/0.35 through Phase 1B.2 round 2; R1 in
# round 3 found both neutral<->clearance legs still snapped a real 60 FPS
# frame even after gesture_pose's easing-curve and corner-width fixes
# (``_flat_ease``, ``_SWING_RISE``/``_SWING_FALL``) - those alone brought both
# legs to 5.1-5.3 m/s peak, still above the derived
# REACH_LEG_PEAK_SPEED_MPS bound in test_reach_trajectory.py. Widening
# attack/release at hold's expense closed most of the remaining gap: giving
# each leg proportionally more of the event's fixed duration lowers peak
# angular rate by that same proportion, uniformly across every term that
# rides the leg, rather than reshaping any one of them further. This shifts
# no event's start or end time - only how one fixed duration is split between
# its own attack, hold and release. Widening ``gesture_pose._WAYPOINT_SNAP``
# (see that constant's own comment) closed the rest.
#
# 0.45/0.10/0.45 - as far in this direction as the two legs needed - shrinks
# the hold from 0.36 s to 0.12 s in absolute terms (BASE_DURATION 1.2 s before
# jitter), comparable to small_hype's own ~0.1 s hold: still long enough to
# read as the hand actually resting on the control. It also compresses
# gesture_pose's mid-hold finger-lift excursion (``_HOLD_LIFT_*``, an
# F1-derived, separately validated safety mechanism) into a much narrower
# raw-progress window, which reopened test_trajectory_is_continuous's 3 cm
# bound (~2.5 cm -> ~7 cm between adjacent 1/400 samples) until that
# mechanism's own transition width was widened to compensate - see its
# comment in gesture_pose.py for why that widening leaves the mechanism's
# real-world timing essentially unchanged despite looking very different in
# hold-relative terms.
ENVELOPE_SHAPE = {
    "deck_glance": (0.30, 0.20, 0.50),
    "lean_in": (0.35, 0.30, 0.35),
    "hand_to_deck": (0.45, 0.10, 0.45),
    "small_hype": (0.25, 0.15, 0.60),
}


def _unit(seed: str, bar: int, channel: int) -> float:
    """A stable pseudo-random 0..1 from a seed, bar number and channel.

    GrooveEngine's own ``_unit`` uses crc32 for this, which is fine for a
    single independent stream but is a linear checksum: for short, structured
    inputs like sequential bar numbers it can correlate across the different
    "channels" fed the same bar at a fixed stride, which is exactly the shape
    the gesture scheduler produces (bars re-checked at an almost-constant gap).
    An early version of this file used crc32 and, for some seeds, drew the same
    gesture kind on every single fired bar - the activation stride and the kind
    channel were landing on correlated outputs. blake2b has proper avalanche
    behaviour and does not exhibit that. It is deterministic across runs (it is
    not salted per process the way hash() is), just not linear.
    """
    digest = hashlib.blake2b(f"{seed}:{bar}:{channel}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") / float(0xFFFFFFFF)


# A plain string rather than GrooveEngine's crc32-folded int seed, so gesture
# placement and groove per-bar character are uncorrelated even for the same
# track - two different hash families, not two views of one number.
_SEED_SALT = "dj-behavior"


def trend_at(energy: EnergyTrack, time: float) -> float:
    """Energy now minus energy ``TREND_LOOKBACK`` seconds ago, clamped at t=0.

    Module-level so ``dj_planner`` (whose ``decide_gesture_kind`` picks
    ``lean_in`` on a rising trend) and ``DJBehaviorEngine._trend`` reason about
    "rising" from one shared definition rather than two copies that could drift
    apart.
    """
    earlier = max(0.0, time - TREND_LOOKBACK)
    return energy.at(time) - energy.at(earlier)


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _envelope_weight(kind: str, progress: float) -> float:
    """Attack-hold-release shape, 0 at both ends, 1 through the hold."""
    attack, hold, _release = ENVELOPE_SHAPE[kind]
    if progress <= 0.0 or progress >= 1.0:
        return 0.0
    if progress < attack:
        return _smoothstep(progress / attack)
    if progress < attack + hold:
        return 1.0
    tail = 1.0 - attack - hold
    return 1.0 - _smoothstep((progress - attack - hold) / tail)


@dataclass(frozen=True)
class GestureEvent:
    """One scheduled action: when it starts, how long it runs, and what it is."""

    start: float
    duration: float
    kind: str
    side: str | None       # "l" / "r" / None
    strength: float        # 0..1, how emphatic this particular occurrence is

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class DJActionState:
    """What the DJ is doing at one absolute playback time.

    ``progress`` runs 0..1 linearly through the active event's duration.
    ``weight`` is the eased attack-hold-release envelope derived from it - what
    pose composition actually multiplies a gesture's contribution by, so it
    starts and ends at exactly zero and cannot introduce a jump. When no event
    is active, ``action`` is ``"none"``, ``progress`` and ``weight`` are 0.0 and
    ``side`` is ``None``.
    """

    time: float
    action: str
    progress: float
    weight: float
    side: str | None
    strength: float

    @property
    def is_active(self) -> bool:
        return self.action != "none"


def _none_state(time: float) -> DJActionState:
    return DJActionState(
        time=time, action="none", progress=0.0, weight=0.0, side=None, strength=0.0
    )


class DJBehaviorEngine:
    """Builds a deterministic gesture schedule once, then answers state_at(T)."""

    def __init__(
        self,
        features: MusicFeatures,
        timeline: BeatTimeline | None = None,
        energy: EnergyTrack | None = None,
        seed: str | None = None,
    ):
        self.features = features
        self.timeline = timeline if timeline is not None else BeatTimeline(features)
        self.energy = energy if energy is not None else EnergyTrack(features)
        self.seed = f"{_SEED_SALT}:{seed if seed is not None else features.track}"
        self.events: list[GestureEvent] = self._build_schedule()
        self._starts = [event.start for event in self.events]

    # ------------------------------------------------------------- schedule
    def _trend(self, time: float) -> float:
        """Energy now minus energy a few seconds ago, clamped to the track."""
        return trend_at(self.energy, time)

    def _build_schedule(self) -> list[GestureEvent]:
        # Local import, matching app.processed_audio_path's own precedent for
        # this exact dj_planner<->dj_behavior shape: dj_planner imports names
        # from this module at its top level, so importing it here (at call
        # time, when this module is already fully loaded) keeps the decision
        # logic one-way - dj_planner is the single source, nothing is
        # duplicated back into this file.
        from ..dj_planner import (
            DJActionPlanner,
            context_at,
            decide_gesture_kind,
            planned_strength,
        )

        duration = self.features.duration_seconds
        nominal = self.timeline.nominal_interval
        bar_seconds = max(nominal * BEATS_PER_BAR, 1e-3)

        # Same per-track seed the audio side uses (app.py passes behavior.seed),
        # so a gesture's side and the filter-sweep side dj_planner plans for the
        # same moment are drawn from one number, not two.
        planner = DJActionPlanner(self.seed)

        events: list[GestureEvent] = []
        bar = 0
        next_eligible_bar = 0
        # A generous bound on how many bars a track this long could contain,
        # so the loop terminates even for pathological feature data.
        max_bars = int(duration / bar_seconds) + 4

        while bar < max_bars:
            start = self.timeline.beat_time(bar * BEATS_PER_BAR)
            if start > duration:
                break

            if bar >= next_eligible_bar and start >= 1.0:
                roll = _unit(self.seed, bar, 0)
                if roll < ACTIVATION_PROBABILITY:
                    # The bar is hosting an event; what that event *is* is the
                    # planner's decision, read at this bar's own start time and
                    # the track's energy there - not an independent kind roll.
                    context = context_at(self.energy, start)
                    kind = decide_gesture_kind(context)

                    jitter = 0.85 + 0.3 * _unit(self.seed, bar, 2)
                    event_duration = BASE_DURATION[kind] * jitter
                    if start + event_duration > duration:
                        bar += 1
                        continue

                    side = planner.gesture_side_for(kind, start)
                    strength = planned_strength(context)

                    events.append(GestureEvent(start, event_duration, kind, side, strength))

                    gap_bars = MIN_GAP_BARS + int(round(2.0 * _unit(self.seed, bar, 4)))
                    next_eligible_bar = bar + gap_bars

            bar += 1

        return events

    # ---------------------------------------------------------------- query
    def state_at(self, time: float) -> DJActionState:
        """The DJ action state at ``time``. Stateless: no history required."""
        if not self._starts:
            return _none_state(time)

        index = bisect_right(self._starts, time) - 1
        if index < 0:
            return _none_state(time)

        event = self.events[index]
        if not (event.start <= time < event.end):
            return _none_state(time)

        progress = (time - event.start) / event.duration
        weight = _envelope_weight(event.kind, progress)
        return DJActionState(
            time=time,
            action=event.kind,
            progress=progress,
            weight=weight,
            side=event.side,
            strength=event.strength,
        )
