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
APPROACH = 2.0
RECOVERY = 2.2
ATTENTION_LEAD = 2.8
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
    variant: str = "knob"
    contact_start: float | None = None

    @property
    def major(self):
        return self.intensity > 0 and self.kind in ('FILTER_ADJUST', 'GAIN_ADJUST', 'TRANSITION_PREPARE')

    def action_at(self, now: float) -> DJActionState | None:
        if self.kind == 'HYPE' and self.begin <= now < self.end:
            progress=(now-self.begin)/(self.end-self.begin)
            return DJActionState(now,'small_hype',progress,
                _envelope_weight('small_hype',progress),self.deck,.55,'cheer')
        if not self.major or not self.begin <= now < self.end:
            return None
        # Traverse approach, stationary contact, and recovery once. The live
        # pose adapter maps this progress onto its calibrated elbow-first path.
        contact_end = self.operation_end if self.contact_end is None else self.contact_end
        contact_start = self.operation_start if self.contact_start is None else self.contact_start
        if now < contact_start:
            progress = .5 * ((now-self.begin)/(contact_start-self.begin))
        elif now <= contact_end:
            progress = .5
        else:
            progress = .5 + .5 * ((now-contact_end)/(self.end-contact_end))
        return DJActionState(now, 'hand_to_deck', progress,
            _envelope_weight('hand_to_deck', progress), self.deck, self.intensity, self.variant,
            min(1,max(0,(now-contact_start)/max(.001,contact_end-contact_start))))

    def phase_at(self, now: float) -> str:
        contact_start = self.operation_start if self.contact_start is None else self.contact_start
        if now < contact_start:
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
        self.hypes = ()

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
            variant="platter" if abs(span.plan.bpm_a-span.plan.bpm_b) > .5 else "button"
            # A platter cue check releases as playback starts: no fake scratch.
            contact_start = start-.45 if variant=='platter' else start
            contact_end = start if variant=='platter' else min(start+TRANSITION_HOLD,end)
            candidates.append(VisualIntent('TRANSITION_PREPARE', other_deck(span.deck),
                'crossfade', start, end, max(0,contact_start-APPROACH),
                contact_end+RECOVERY,contact_end=contact_end,variant=variant,
                contact_start=max(0,contact_start)))
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
        self.hypes = self._plan_hypes(spans)
        # A rare measured accent can replace an optional solo representation;
        # audible effects are unchanged. Real transition work always wins.
        self.interactions = tuple(a for a in self.interactions if a.operation=='crossfade' or
            all(a.end+8 <= h.begin or a.begin >= h.end+8 for h in self.hypes))

    def _plan_hypes(self, spans):
        # Performance accents do not operate controls or claim a DSP action.
        # Require a sustained measured lift, not simply a loud recording.
        hypes=[];last_track=-2;last_time=-90
        solos=[s for s in spans if s.plan is None]
        for track_number,span in enumerate(solos):
            if track_number-last_track < 2:
                continue  # at least one entire song without hype in between
            broad=span.active.deck.broad_energy
            energy=span.active.deck.energy;timeline=span.active.deck.timeline
            offset=(span.start-span.source_start)/SAMPLE_RATE
            source_end=(span.source_start+span.end-span.start)/SAMPLE_RATE
            for bar in range(int(source_end/max(.1,4*timeline.nominal_interval))+1):
                source=timeline.beat_time(bar*4);now=source+offset
                if now < span.start/SAMPLE_RATE+16 or now+3.2 > span.end/SAMPLE_RATE-12:
                    continue
                if now-last_time < 90:
                    continue
                if broad.at(source) < .78 or energy.at(source) < .8 or broad.at(source)-broad.at(source-12) < .30:
                    continue
                if any(now < a.end+8 and now+3.2 > a.begin-8 for a in self.interactions if a.operation=='crossfade'):
                    continue
                hypes.append(VisualIntent('HYPE',span.deck,'measured_energy_lift',now,now+3.2,
                                         now,now+3.2,intensity=.55,variant='cheer'))
                last_track=track_number;last_time=now
                break
        return tuple(hypes)

    def at(self, now: float, active_deck: str) -> VisualIntent:
        for intent in self.interactions + self.hypes:
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
