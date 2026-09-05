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

What this phase deliberately does not do:

- It adds no new DSP effect. ``AUDIO_ACTIONS`` is exactly the two effects that
  already exist in ``audio/effects.py`` plus doing nothing.
- It does not yet rewire ``DJBehaviorEngine._build_schedule`` to consume
  ``decide_gesture_kind`` - that engine change is the next subtask. This file
  only makes the decision available and deterministic.
- It does not yet choose its own timing. A ``GestureEvent`` handed to
  ``DJActionPlanner.plan`` is a source of *timing only* - ``.start`` and
  ``.duration`` are read from it and nothing else. The action type, its
  strength, and its side (when needed) are derived solely from musical context
  plus this planner's own deterministic seed, never from the event's ``kind``,
  ``side``, or ``strength``. That coupling is exactly what this phase removes.

No Panda3D import. This module connects the audio and animation subsystems and
belongs to neither package, so it sits at the top level alongside ``app.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .animation.dj_behavior import (
    HIGH_ENERGY,
    LOW_ENERGY,
    TREND_RISING,
    GestureEvent,
    trend_at,
)
from .animation.energy import EnergyTrack

# This phase's own cache-invalidation version, analogous to
# ``effects.EFFECT_VERSION`` / ``features.SCHEMA_VERSION``. Bump it whenever a
# change to ``decide_action`` or ``context_at`` would render a different
# decision for the same ``GestureEvent`` schedule, even though ``effects.py``
# itself did not change - ``app.processed_audio_path`` folds it into the fx
# cache key so a render made under an older decision rule is never served after.
PLANNER_VERSION = 1

# Floor for a planned action's ``strength``: a full-energy moment plans at
# strength 1.0, a zero-energy one still plans at this value rather than 0.
# A fixed, documented constant (like ``dj_behavior.BASE_DURATION``), analogous
# in spirit to - but computed independently of, never read from -
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


def _band_for(energy: float) -> str:
    """``"low"`` / ``"mid"`` / ``"high"`` using ``dj_behavior``'s own thresholds.

    ``LOW_ENERGY`` / ``HIGH_ENERGY`` are imported, not re-derived, so the
    planner's notion of "high energy" is provably the same one
    ``DJBehaviorEngine._kind_weights`` already uses.
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
    every decision is made at one of ``DJBehaviorEngine``'s own event starts,
    and those are bar-aligned by construction
    (``_build_schedule`` sets ``start = timeline.beat_time(bar * BEATS_PER_BAR)``).
    It is carried as a real field rather than left implicit because a future
    phase that evaluates context off the bar grid (its own timing, Phase 2B+)
    would need this to actually mean something.
    """

    time: float
    energy: float
    trend: float
    energy_band: str
    at_bar_boundary: bool


def context_at(energy: EnergyTrack, time: float) -> MusicalContext:
    """The musical context at ``time`` - a pure function of the energy track."""
    value = energy.at(time)
    return MusicalContext(
        time=time,
        energy=value,
        trend=trend_at(energy, time),
        energy_band=_band_for(value),
        # See MusicalContext's docstring: every call site is a bar-aligned
        # DJBehaviorEngine event start, so this is genuinely true, not a stub.
        at_bar_boundary=True,
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
    ``GestureEvent`` (this phase does not choose its own timing yet); ``action``,
    ``side`` and ``strength`` are the planner's own decision and are never
    sourced from that event's ``kind``, ``side`` or ``strength``.
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
