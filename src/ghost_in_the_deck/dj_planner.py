"""The DJ action planner: a deterministic decision layer between musical
analysis and the audio + animation it drives.

Phase 1C wired ``hand_to_deck``'s filter sweep and Phase 1D wired
``small_hype``'s gain riser each directly to one gesture's own ``kind`` - an
audio effect existed *because* that gesture kind was scheduled, not because
the music justified it. This module inverts that boundary:

    analysis -> musical context -> DJ decision (here) -> real audio action
                                                      -> matching gesture kind

so "what happens right now" is answered from the music's own energy at that
moment. ``decide_action`` (Phase 2A) answers it for audio; ``decide_gesture_kind``
answers it for the visible gesture, from the *same* ``MusicalContext``, so the
two are two readings of one decision rather than two independent rolls.

Phase 2B moved *what* a chosen bar hosts here: ``DJBehaviorEngine._build_schedule``
reads ``decide_gesture_kind`` / ``gesture_side_for`` / ``planned_strength`` at
each fired bar's start rather than rolling kind, side and strength itself.
Phase 2C moves the bar *choice* too: ``plan_schedule`` owns the whole bar-walk -
the ``start >= 1.0`` eligibility floor, the per-bar activation roll (now against
``activation_probability``, which reads the energy band and a rising-trend boost
off the same ``MusicalContext``), the ``MIN_GAP_BARS`` + jitter spacing, and the
``BASE_DURATION`` + jitter duration with its track-length clamp. Candidate
positions stay the existing bar grid - no phrase/section/build/drop detection is
added.

What this module deliberately does not do:

- It adds no new DSP effect. ``AUDIO_ACTIONS`` is exactly the two effects that
  already exist in ``audio/effects.py`` plus doing nothing.
- It does not detect musical structure. ``plan_schedule`` walks the same fixed
  bar grid ``_build_schedule`` always has; ``activation_probability`` reads only
  the smoothed energy band and ``trend``, never phrase or section position.
- ``DJActionPlanner.plan`` still consumes a caller's ``GestureEvent`` *timing*
  (``.start`` / ``.duration``) unchanged - it maps an already-built visual
  schedule onto audio actions. ``plan_schedule`` is the path that builds that
  schedule from musical context in the first place. Either way the action type,
  its strength, and its side are derived solely from musical context plus this
  planner's own deterministic seed, never from a candidate event's ``kind``,
  ``side`` or ``strength``.

No Panda3D import. This module connects the audio and animation subsystems and
belongs to neither package, so it sits at the top level alongside ``app.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .animation.cues import BEATS_PER_BAR, BeatTimeline
from .animation.dj_behavior import (
    HIGH_ENERGY,
    LOW_ENERGY,
    TREND_RISING,
    GestureEvent,
    trend_at,
)
from .animation.energy import EnergyTrack
from .animation.structure import (
    PHRASE_SMOOTHING_SECONDS,
    MusicalStructure,
    structure_at,
)

# This phase's own cache-invalidation version, analogous to
# ``effects.EFFECT_VERSION`` / ``features.SCHEMA_VERSION``. Bump it whenever a
# change to ``decide_action`` / ``activation_probability`` / ``context_at`` (or
# another decision function feeding a render) would produce a different decision
# for the same track, even though ``effects.py`` itself did not change -
# ``app.processed_audio_path`` folds it into the fx cache key so a render made
# under an older decision rule is never served after.
PLANNER_VERSION = 1

# Floor for a planned action's ``strength``: a full-energy moment plans at
# strength 1.0, a zero-energy one still plans at this value rather than 0.
# A fixed, documented constant (like this module's own ``BASE_DURATION``),
# analogous in spirit to - but computed independently of, never read from -
# ``dj_behavior``'s own ``event.strength = 0.6 + 0.4 * local_energy``.
PLANNER_STRENGTH_FLOOR = 0.5

# The only audio actions this phase can plan: the two DSP effects that already
# exist, plus doing nothing. Phase 2A adds no new DSP action. ``"none"`` is
# only the absence of those two existing sounds - it is not a placeholder for
# some future ``lean_in`` / ``deck_glance`` effect.
AUDIO_ACTIONS = ("filter_sweep", "gain_riser", "none")

# The gesture kinds that carry a side - an arm to reach or raise with. Mirrors
# ``dj_behavior``'s own ``if kind in ("hand_to_deck", "small_hype")`` side draw,
# but the planner assigns the side unconditionally from its own seed and the
# moment, never from a candidate event. ``deck_glance`` / ``lean_in`` are
# centred and keep ``side=None``.
SIDED_GESTURE_KINDS = ("hand_to_deck", "small_hype")

# Salt for this planner's own side draw. Deliberately distinct from every
# channel ``dj_behavior._unit`` uses (activation, kind, jitter, side, gap) so
# the planner's filter-sweep side is uncorrelated with the gesture scheduler's
# own per-bar side draw - otherwise the two "independent" decisions for a bar
# would move together. The candidate event's own ``side`` field is never
# consulted here even when it carries one: reading it would make the planner a
# partially repackaged ``GestureEvent``, which is precisely the coupling this
# phase exists to remove.
_SIDE_SALT = "dj-planner-side"

# --- timing parameters, relocated from ``dj_behavior`` in Phase 2C ----------
# The bar choice and its supporting timing are now this planner's own decision
# (``plan_schedule``), computed from ``MusicalContext``, so the constants that
# shape it live here rather than in ``DJBehaviorEngine``.

# A new gesture may not start less than this many bars after the previous one
# started. Combined with the activation roll, real spacing works out noticeably
# irregular - never "every four bars", which would read as a metronome in a
# costume.
MIN_GAP_BARS = 3

# Base duration per gesture kind, in seconds, before the small per-event jitter
# ``plan_schedule`` applies.
BASE_DURATION = {
    "deck_glance": 1.0,
    "lean_in": 1.6,
    "hand_to_deck": 1.2,
    "small_hype": 0.7,
}

# Per-energy-band base chance that an eligible bar hosts an event. "mid" is
# 0.42 - exactly the flat ``dj_behavior.ACTIVATION_PROBABILITY`` this replaces,
# so a mid-energy stretch keeps today's event density; low is quieter, high is
# busier.
ACTIVATION_BASE = {"low": 0.28, "mid": 0.42, "high": 0.58}

# Added to the band's base chance when ``context.trend`` clears ``TREND_RISING``
# - a build deserves more DJ activity - with the sum then capped at 1.0.
ACTIVATION_TREND_BOOST = 0.15

# Broad-structure adjustments layered on top of the band/short-trend chance when
# ``context.structure`` is present. A broad "build" regime nudges the planner
# busier; a broad "release" regime quieter. The damp is smaller than any band
# base, so it can never drive the chance to zero; "peak" and "stable" leave the
# Phase 2C value untouched. With ``structure=None`` neither is applied and the
# result is byte-identical to Phase 2C.
STRUCTURE_BUILD_BOOST = 0.10
STRUCTURE_RELEASE_DAMP = 0.10


def _unit(seed: str, value: float, salt: str) -> float:
    """A stable pseudo-random 0..1 from a seed, a number and a salt string.

    The same approach ``dj_behavior._unit`` uses - blake2b for real avalanche
    behaviour, deterministic across processes (unlike ``hash()``) - keyed here
    off a float (serialised losslessly via ``float.hex()``) rather than a bar
    index.
    """
    digest = hashlib.blake2b(
        f"{salt}:{seed}:{value.hex()}".encode(), digest_size=4
    ).digest()
    return int.from_bytes(digest, "big") / float(0xFFFFFFFF)


def _bar_unit(seed: str, bar: int, channel: int) -> float:
    """A stable pseudo-random 0..1 from a seed, bar number and channel.

    The bar-indexed sibling of ``_unit`` (which keys off a float), relocated
    here from ``dj_behavior`` in Phase 2C along with the bar-walk that uses it.
    blake2b for proper avalanche behaviour: an earlier crc32 version, being a
    linear checksum, correlated the activation-stride and kind channels for
    some seeds and drew the same gesture on every fired bar. Deterministic
    across processes, unlike ``hash()``.
    """
    digest = hashlib.blake2b(
        f"{seed}:{bar}:{channel}".encode(), digest_size=4
    ).digest()
    return int.from_bytes(digest, "big") / float(0xFFFFFFFF)


def _band_for(energy: float) -> str:
    """``"low"`` / ``"mid"`` / ``"high"`` using ``dj_behavior``'s own thresholds.

    ``LOW_ENERGY`` / ``HIGH_ENERGY`` are imported, not re-derived, so the
    planner's notion of "high energy" is provably the same one
    ``decide_action`` / ``decide_gesture_kind`` / ``activation_probability``
    all reason about - one band split, not several that could drift apart.
    """
    if energy < LOW_ENERGY:
        return "low"
    if energy >= HIGH_ENERGY:
        return "high"
    return "mid"


@dataclass(frozen=True)
class MusicalContext:
    """What the music is doing at one absolute playback time.

    ``at_bar_boundary`` is always ``True`` for every call site in this phase:
    every decision is made at one of the planner's own candidate bar starts,
    and those are bar-aligned by construction
    (``plan_schedule`` sets ``start = timeline.beat_time(bar * BEATS_PER_BAR)``).
    It is carried as a real field rather than left implicit so that any call
    site which one day evaluates context off the bar grid stays honest about it.

    ``structure`` is the broad musical-structure reading (regime, broad trend,
    section-change likelihood) for this moment, or ``None``. Every direct
    construction and ``DJActionPlanner.plan``'s audio-only path leave it
    ``None``; only ``plan_schedule`` attaches a real one, from a broad
    ``EnergyTrack`` it builds once per call. The ``structure=None`` path is
    deliberately byte-identical to Phase 2C.
    """

    time: float
    energy: float
    trend: float
    energy_band: str
    at_bar_boundary: bool
    structure: MusicalStructure | None = None


def context_at(
    energy: EnergyTrack, time: float, *, structure: MusicalStructure | None = None
) -> MusicalContext:
    """The musical context at ``time`` - a pure function of its inputs.

    ``structure`` defaults to ``None`` so every two-argument caller is
    unchanged; ``plan_schedule`` passes the broad-structure reading for the
    moment so the kind / activation / strength decisions can see it.
    """
    value = energy.at(time)
    return MusicalContext(
        time=time,
        energy=value,
        trend=trend_at(energy, time),
        energy_band=_band_for(value),
        # See MusicalContext's docstring: every call site is a bar-aligned
        # DJBehaviorEngine event start, so this is genuinely true, not a stub.
        at_bar_boundary=True,
        structure=structure,
    )


def decide_action(context: MusicalContext) -> str:
    """Which audio action the music justifies right now. One of ``AUDIO_ACTIONS``.

    Pure and deterministic; reads only ``context``, never a gesture's ``kind``.

    - ``"high"`` energy -> ``"gain_riser"``: an energy peak, matching
      ``small_hype``'s existing musical role.
    - ``"mid"`` energy -> ``"filter_sweep"``: an ordinary passage, matching
      ``hand_to_deck``'s "working a control" role.
    - ``"low"`` energy -> ``"none"``: a restrained passage - nothing to hype,
      and pulling a filter down further on already-quiet material has little
      to say.

    ``context.trend`` is not read here: neither of the two existing DSP effects
    represents anticipation/build, so there is nothing for a "rising" reading to
    select on the audio side. Its visual sibling ``decide_gesture_kind`` *does*
    read it (``lean_in`` is exactly an anticipation gesture); a later phase that
    adds a build-style DSP effect would fold ``trend`` in at this point too.
    """
    if context.energy_band == "high":
        return "gain_riser"
    if context.energy_band == "mid":
        return "filter_sweep"
    return "none"


def decide_gesture_kind(context: MusicalContext) -> str:
    """Which gesture kind represents the same decision, right now.

    The visual sibling of ``decide_action``: pure and deterministic, reads only
    ``context`` - never a candidate ``GestureEvent``'s own ``kind``. Same band
    split ``decide_action`` uses, mapped to the gesture whose existing musical
    role matches:

    - ``"high"`` energy -> ``"small_hype"``: an energy peak, the same moment
      ``decide_action`` plans a ``gain_riser`` for.
    - ``"mid"`` energy -> ``"hand_to_deck"``: an ordinary passage, "working a
      control", the visual partner of the ``filter_sweep``.
    - ``"low"`` energy -> ``"lean_in"`` when ``context.trend`` rises past
      ``TREND_RISING`` (leaning in to catch a build), otherwise ``"deck_glance"``
      (a restrained check). This is ``trend``'s first genuinely load-bearing use
      in the planner - the same ``trend > TREND_RISING`` test
      ``DJBehaviorEngine._kind_weights`` already uses to favour ``lean_in``.
    """
    if context.energy_band == "high":
        return "small_hype"
    if context.energy_band == "mid":
        return "hand_to_deck"
    if context.trend > TREND_RISING:
        return "lean_in"
    return "deck_glance"


def activation_probability(context: MusicalContext) -> float:
    """Per-eligible-bar chance that the bar hosts a gesture event.

    A pure function of ``context``: ``ACTIVATION_BASE`` for the energy band,
    plus ``ACTIVATION_TREND_BOOST`` when ``context.trend`` clears the imported
    ``TREND_RISING`` threshold, with the sum capped at 1.0. Strictly increasing
    across low -> mid -> high at a fixed trend, always within ``[0, 1]``, and
    never zero for any band.

    This replaces ``dj_behavior``'s single flat ``ACTIVATION_PROBABILITY``: the
    "mid" base is that same 0.42, so a mid-energy stretch keeps today's event
    density exactly, while quiet and driving stretches now differ - and a rising
    trend nudges any band busier, the same ``trend > TREND_RISING`` reading
    ``decide_gesture_kind`` already uses to favour ``lean_in``.

    When ``context.structure`` is present, the broad regime layers one bounded
    adjustment on top: ``"build"`` adds ``STRUCTURE_BUILD_BOOST`` (busier during
    a broad build), ``"release"`` subtracts ``STRUCTURE_RELEASE_DAMP`` (quieter
    during a broad release, but never to zero - the damp is smaller than any
    band base), and ``"peak"`` / ``"stable"`` leave the value unchanged. With
    ``structure=None`` nothing is added and the result is byte-identical to
    Phase 2C. This is the single point where broad structure enters the
    planner's timing; ``decide_action`` / ``decide_gesture_kind`` /
    ``gesture_side_for`` / ``planned_strength`` do not read it.
    """
    probability = ACTIVATION_BASE[context.energy_band]
    if context.trend > TREND_RISING:
        probability += ACTIVATION_TREND_BOOST
    if context.structure is not None:
        if context.structure.regime == "build":
            probability += STRUCTURE_BUILD_BOOST
        elif context.structure.regime == "release":
            probability -= STRUCTURE_RELEASE_DAMP
    return min(probability, 1.0)


def planned_strength(context: MusicalContext) -> float:
    """The strength every planned action shares - audio or gesture-only alike.

    One formula: floored at ``PLANNER_STRENGTH_FLOOR`` and rising to 1.0 with
    the music's energy, so a ``filter_sweep`` and the ``hand_to_deck`` that
    represents the same decision are planned at one intensity rather than two.
    """
    return (
        PLANNER_STRENGTH_FLOOR
        + (1.0 - PLANNER_STRENGTH_FLOOR) * context.energy
    )


@dataclass(frozen=True)
class PlannedAudioAction:
    """One audio action the planner decided the music justifies.

    Its own type, not a reuse of ``GestureEvent``, so a reader can see this is
    an audio decision rather than a copy of the visual schedule. ``start`` and
    ``duration`` are the only fields carried over from the candidate
    ``GestureEvent`` handed to ``DJActionPlanner.plan``; ``action``, ``side``
    and ``strength`` are the planner's own decision and are never sourced from
    that event's ``kind``, ``side`` or ``strength``.
    """

    start: float
    duration: float
    action: str            # one of AUDIO_ACTIONS, never "none" in an emitted list
    side: str | None       # "l" / "r" / None
    strength: float


class DJActionPlanner:
    """Turns a gesture schedule's *timing* into an independent audio-action plan.

    Constructed with the planner's own per-track determinism seed (the
    ``app.py`` call site passes ``behavior.seed`` - a per-track seed already
    used elsewhere for exactly this purpose, not a per-gesture one).
    """

    def __init__(self, seed: str):
        self.seed = seed

    def _side_for(self, start: float) -> str:
        """Deterministic ``"l"`` / ``"r"`` for a planned action at ``start``.

        Derived unconditionally from the planner's seed and the moment alone -
        the candidate event's own ``side`` is never consulted, so the result is
        identical whether that event carried ``"l"``, ``"r"`` or ``None``. Used
        for both the filter-sweep side and the sided-gesture side, so audio and
        animation at one moment agree.
        """
        return "l" if _unit(self.seed, start, _SIDE_SALT) < 0.5 else "r"

    def gesture_side_for(self, kind: str, start: float) -> str | None:
        """The side a planned gesture ``kind`` carries at ``start``, or ``None``.

        ``hand_to_deck`` / ``small_hype`` get the same seed-and-moment draw the
        filter sweep uses; ``deck_glance`` / ``lean_in`` are centred and keep
        ``None``. The candidate event's own ``side`` is never consulted.
        """
        if kind not in SIDED_GESTURE_KINDS:
            return None
        return self._side_for(start)

    def plan_schedule(
        self,
        timeline: BeatTimeline,
        energy: EnergyTrack,
        duration: float,
    ) -> list[GestureEvent]:
        """Build the whole gesture schedule from musical context alone.

        The bar-walk relocated from ``DJBehaviorEngine._build_schedule`` in
        Phase 2C, so the bar *choice* - not just what a chosen bar hosts - is
        the planner's own decision, computed from the same ``MusicalContext``
        the kind / side / strength decisions already use:

        - candidate positions are the existing bar grid (``timeline.beat_time``
          at whole-bar beat counts); no phrase/section detection gates them;
        - each bar's ``MusicalContext`` carries a broad ``MusicalStructure``
          reading (``structure_at`` against a one-per-call broad
          ``EnergyTrack``), so structure-aware decisions can read it;
        - a bar is eligible once its start reaches 1.0 s and once
          ``MIN_GAP_BARS`` (plus 0..2 bars of seed jitter) have passed since
          the last event started;
        - an eligible bar fires when its activation roll falls below
          ``activation_probability`` for the context at its start;
        - a fired bar's duration is ``BASE_DURATION[kind]`` times a 0.85..1.15
          jitter, and is dropped if it would run past ``duration``.

        Deterministic in ``(self.seed, timeline, energy, duration)``: the same
        inputs give a byte-identical event list. Reuses ``decide_gesture_kind``
        / ``gesture_side_for`` / ``planned_strength`` exactly as they stand.
        """
        # One broad EnergyTrack for the whole call: the same feature data the
        # groove uses, re-smoothed over PHRASE_SMOOTHING_SECONDS so a whole
        # build or release reads as one movement. structure_at then looks it up
        # per bar without rebuilding it.
        broad = EnergyTrack(
            energy.features, smoothing_seconds=PHRASE_SMOOTHING_SECONDS
        )

        nominal = timeline.nominal_interval
        bar_seconds = max(nominal * BEATS_PER_BAR, 1e-3)
        # A generous bound on how many bars a track this long could contain, so
        # the loop terminates even for pathological feature data.
        max_bars = int(duration / bar_seconds) + 4

        events: list[GestureEvent] = []
        bar = 0
        next_eligible_bar = 0
        while bar < max_bars:
            start = timeline.beat_time(bar * BEATS_PER_BAR)
            if start > duration:
                break

            if bar >= next_eligible_bar and start >= 1.0:
                context = context_at(
                    energy, start, structure=structure_at(broad, timeline, start)
                )
                if _bar_unit(self.seed, bar, 0) < activation_probability(context):
                    kind = decide_gesture_kind(context)

                    jitter = 0.85 + 0.3 * _bar_unit(self.seed, bar, 2)
                    event_duration = BASE_DURATION[kind] * jitter
                    if start + event_duration > duration:
                        bar += 1
                        continue

                    side = self.gesture_side_for(kind, start)
                    strength = planned_strength(context)
                    events.append(
                        GestureEvent(start, event_duration, kind, side, strength)
                    )

                    gap_bars = MIN_GAP_BARS + int(
                        round(2.0 * _bar_unit(self.seed, bar, 4))
                    )
                    next_eligible_bar = bar + gap_bars

            bar += 1

        return events

    def plan(
        self,
        events: Sequence[GestureEvent],
        energy: EnergyTrack,
    ) -> list[PlannedAudioAction]:
        """One ``PlannedAudioAction`` per event whose musical context justifies one.

        For each event only ``.start`` / ``.duration`` are read. The context is
        evaluated at ``event.start``, ``decide_action`` chooses, and a ``"none"``
        result is skipped entirely - the returned list only ever contains
        actions that actually happen.
        """
        planned: list[PlannedAudioAction] = []
        for event in events:
            context = context_at(energy, event.start)
            action = decide_action(context)
            if action == "none":
                continue
            side = self._side_for(event.start) if action == "filter_sweep" else None
            strength = planned_strength(context)
            planned.append(
                PlannedAudioAction(
                    start=event.start,
                    duration=event.duration,
                    action=action,
                    side=side,
                    strength=strength,
                )
            )
        return planned
