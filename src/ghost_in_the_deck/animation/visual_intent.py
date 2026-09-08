"""Factual, restrained choreography over audio spans already committed to playback.

This layer cannot request DSP, schedule an effect, or choose a deck at random.
Skipped effects remain audible; representing every automated control change is
neither necessary nor believable. All times here are absolute playback seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dj_behavior import DJActionState, _envelope_weight, _smoothstep
from ..set_engine import SAMPLE_RATE

# Behavioral policies, not detected musical measurements. At the demo's ~120
# BPM a bar is ~2 s: approach/recovery each take most of a bar, and a major
# interaction at most once per six bars leaves real listening time. Repeating
# the same effect needs twice that space. Transitions reserve the entire blend.
APPROACH = 1.6
RECOVERY = 1.8
ATTENTION_LEAD = 2.4
TRANSITION_HOLD = .6
HANDOFF_RECOVERY = 2.4
MIN_MAJOR_INTERVAL = 12.0
REPEAT_INTERVAL = 24.0
HANDOFF_COOLDOWN = 6.0


def other_deck(deck: str) -> str:
    if deck not in ('l', 'r'):
        raise ValueError(f'unknown physical deck: {deck}')
    return 'r' if deck == 'l' else 'l'


def deck_label(deck: str) -> str:
    # Avatar anatomical left is +X (viewer right in the frontal camera).
    return {'l': 'A', 'r': 'B'}[deck]


@dataclass(frozen=True)
class VisualIntent:
    kind: str
    deck: str | None
    operation: str
    operation_start: float
    operation_end: float
    begin: float
    end: float
    intensity: float = 1.0
    return_to_neutral: bool = True
    contact_end: float | None = None

    @property
    def major(self):
        return self.intensity > 0 and self.kind in ('FILTER_ADJUST', 'GAIN_ADJUST', 'TRANSITION_PREPARE')

    def action_at(self, now: float) -> DJActionState | None:
        if not self.major or not self.begin <= now < self.end:
            return None
        # Traverse approach, stationary contact, and recovery once. The live
        # pose adapter maps this progress onto its calibrated elbow-first path.
        contact_end = self.operation_end if self.contact_end is None else self.contact_end
        if now < self.operation_start:
            progress = .5 * _smoothstep((now-self.begin)/(self.operation_start-self.begin))
        elif now <= contact_end:
            progress = .5
        else:
            progress = .5 + .5 * _smoothstep((now-contact_end)/(self.end-contact_end))
        return DJActionState(now, 'hand_to_deck', progress,
            _envelope_weight('hand_to_deck', progress), self.deck, self.intensity)

    def phase_at(self, now: float) -> str:
        if now < self.operation_start:
            return 'anticipation'
        contact_end = self.operation_end if self.contact_end is None else self.contact_end
        if now <= contact_end:
            return 'action'
        return 'recovery'


class VisualTimeline:
    """Rebuild only when committed spans change; lookups have no frame history."""
    def __init__(self):
        self._signature = None
        self.interactions: tuple[VisualIntent, ...] = ()
        self.transitions = ()

    def update(self, spans):
        signature = tuple(id(s) for s in spans)
        if signature == self._signature:
            return
        self._signature = signature
        transitions = tuple(s for s in spans if s.plan is not None)
        self.transitions = transitions
        reserved = [(s.start/SAMPLE_RATE-ATTENTION_LEAD,
                     s.end/SAMPLE_RATE+HANDOFF_COOLDOWN) for s in transitions]
        candidates = []
        for span in transitions:
            start, end = span.start/SAMPLE_RATE, span.end/SAMPLE_RATE
            contact_end = min(start+TRANSITION_HOLD, end)
            candidates.append(VisualIntent('TRANSITION_PREPARE', other_deck(span.deck),
                'crossfade', start, end, max(0, start-APPROACH),
                contact_end+RECOVERY, contact_end=contact_end))
        # Transitions take priority, including future committed transitions. If
        # a deliberately tiny dwell makes reaches incompatible, monitor the
        # next blend rather than interrupt a hand that is still returning.
        accepted = []
        for intent in candidates:
            if not accepted or (intent.begin-accepted[-1].begin >= MIN_MAJOR_INTERVAL and
                                intent.begin >= accepted[-1].operation_end+HANDOFF_COOLDOWN):
                accepted.append(intent)
        for span in spans:
            if span.plan is not None:
                continue  # source-time FX in stretched mixes are not solo knobs
            offset = (span.start-span.source_start)/SAMPLE_RATE
            for action in span.audio_actions:
                kind = {'filter_sweep': 'FILTER_ADJUST', 'gain_riser': 'GAIN_ADJUST'}.get(action.action)
                if kind is None:
                    continue
                start = offset+action.start
                end = start+action.duration
                intent = VisualIntent(kind, span.deck, action.action, start, end,
                    start-APPROACH, end+RECOVERY)
                # Reserve advance attention even before the producer has supplied
                # the next span. This prevents late queue arrival cancelling a reach.
                if intent.begin < span.start/SAMPLE_RATE+HANDOFF_COOLDOWN or intent.end > span.end/SAMPLE_RATE-MIN_MAJOR_INTERVAL:
                    continue
                if any(intent.begin < hi and intent.end > lo for lo, hi in reserved):
                    continue
                if any(abs(intent.begin-a.begin) < MIN_MAJOR_INTERVAL or
                       (intent.begin < a.end+HANDOFF_COOLDOWN and intent.end+HANDOFF_COOLDOWN > a.begin) or
                       (intent.kind == a.kind and abs(intent.begin-a.begin) < REPEAT_INTERVAL)
                       for a in accepted):
                    continue
                accepted.append(intent)
        self.interactions = tuple(sorted(accepted, key=lambda a: a.begin))

    def at(self, now: float, active_deck: str) -> VisualIntent:
        for intent in self.interactions:
            if intent.begin <= now < intent.end:
                return intent
        for span in self.transitions:
            start, end = span.start/SAMPLE_RATE, span.end/SAMPLE_RATE
            if start-ATTENTION_LEAD <= now < end+HANDOFF_RECOVERY:
                kind = ('TRANSITION_PREPARE' if now < start else
                        'TRANSITION_ACTIVE' if now < end else 'HANDOFF')
                # Attention-only intents have no reach duration. TRANSITION_PREPARE
                # here is deliberately non-contact until the admitted reach begins.
                return VisualIntent(kind, other_deck(span.deck), 'crossfade', start, end,
                    start-ATTENTION_LEAD, end+HANDOFF_RECOVERY, intensity=0)
        return VisualIntent('IDLE_GROOVE', active_deck, 'solo_playback', now, now, now, now,
                            intensity=0, return_to_neutral=False)

    def attention_at(self, now: float) -> tuple[str | None, float, float]:
        direction, acknowledgment = 0.0, 0.0
        transitioning = False
        for span in self.transitions:
            start, end = span.start/SAMPLE_RATE, span.end/SAMPLE_RATE
            if start-ATTENTION_LEAD <= now < end+HANDOFF_RECOVERY:
                transitioning = True
                engage = _smoothstep((now-(start-ATTENTION_LEAD))/ATTENTION_LEAD)
                settle = 1-_smoothstep((now-end)/HANDOFF_RECOVERY)
                # A small nod straddles the exact ownership transfer, zero slope
                # at both ends. No additional arm action is scheduled at handoff.
                nod = _envelope_weight('deck_glance', (now-end+.4)/1.4)
                direction += (1 if other_deck(span.deck) == 'l' else -1)*engage*settle
                acknowledgment = max(acknowledgment,nod)
        if transitioning:
            # Tiny dwell can overlap outgoing recovery and incoming attention.
            # Blend signed turns through neutral, never switch a full head pose.
            return ('l' if direction >= 0 else 'r'), min(1,abs(direction)), acknowledgment
        for intent in self.interactions:
            if intent.begin-.6 <= now < intent.end+.6:
                w = _smoothstep((now-intent.begin+.6)/1.2)*(1-_smoothstep((now-intent.end+.6)/1.2))
                return intent.deck, w*.7, 0
        return None, 0, 0
