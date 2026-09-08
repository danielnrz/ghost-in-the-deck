"""Deterministic next-track policy; no semantic or harmonic inference."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .audio.tempo_match import tempo_match
from .library import LibraryTrack
from .transition import TransitionPlan, TwoDeckContext, candidate_cue_points


@dataclass(frozen=True)
class Candidate:
    track: LibraryTrack
    plan: TransitionPlan
    score: float


def rank_candidates(active: LibraryTrack, remaining: list[LibraryTrack],
                    source_time: float, *, transition_bars: int = 4,
                    dwell_seconds: float = 20.0) -> list[Candidate]:
    if transition_bars < 1 or not math.isfinite(dwell_seconds) or dwell_seconds < 0:
        raise ValueError('transition bars must be positive and dwell non-negative')
    if not math.isfinite(source_time) or source_time < 0:
        raise ValueError('source time must be finite and non-negative')
    outgoing = active.deck
    # Keep the track's main body; leave enough room to complete the blend.
    earliest = max(source_time + dwell_seconds, outgoing.duration * 0.65)
    outs = [c for c in candidate_cue_points(outgoing, margin_seconds=2,
            limit=100000, transition_bars=transition_bars) if c.time >= earliest
            and c.time <= outgoing.timeline.end_time]
    ranked = []
    seen = set()
    for track in remaining:
        identity = track.path.resolve()
        if identity == active.path.resolve() or identity in seen:
            continue
        seen.add(identity)
        context = TwoDeckContext(outgoing, track.deck)
        ins = [c for c in candidate_cue_points(track.deck, margin_seconds=2,
               limit=100000, transition_bars=transition_bars)
               if c.time <= min(track.deck.duration * 0.35, track.deck.timeline.end_time)]
        best = None
        for a in outs:
            for b in ins:
                energy_gap = abs(outgoing.energy.at(a.time) - track.deck.energy.at(b.time))
                rate = outgoing.bpm / track.deck.bpm if track.deck.bpm > 0 else 0
                tempo_cost = abs(math.log(rate)) if rate > 0 else math.inf
                score = (a.score + b.score) / 2 - energy_gap - 2 * tempo_cost
                plan = TransitionPlan.from_selection(context, a, b, score,
                    f'cue stability {(a.score+b.score)/2:.3f}; energy gap '
                    f'{energy_gap:.3f}; tempo rate {rate:.4f}',
                    transition_bars=transition_bars)
                try:
                    tempo_match(plan, 44100)
                except ValueError:
                    continue
                key = (-score, a.time, b.time)
                if best is None or key < best[0]:
                    best = key, Candidate(track, plan, score)
        if best:
            ranked.append(best[1])
    return sorted(ranked, key=lambda c: (-c.score, str(c.track.path)))
