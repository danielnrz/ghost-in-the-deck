"""Two-deck set preparation, with an exact sample ledger for live playback.

Only one outgoing/incoming pair is decoded at a time. The consumer streams the
returned segments; scene timing never schedules or performs a mix.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import subprocess
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio.mixer import execute_transition
from .audio.phase_corrected_material import phase_correct_incoming_transition_with_relationship
from .audio.tempo_match import tempo_match
from .library import Library, LibraryTrack
from .selection import rank_candidates
from .transition import TransitionPlan

SAMPLE_RATE = 44100
CEILING = 0.85


def frame(seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    return math.floor(seconds * sample_rate + 0.5)


def bounded_pcm(samples: np.ndarray) -> np.ndarray:
    data = np.array(samples, dtype=np.float64, copy=True)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        raise ValueError('expected nonempty stereo PCM')
    if not np.isfinite(data).all():
        raise ValueError('audio contains non-finite samples')
    peak = float(np.max(np.abs(data)))
    if peak > CEILING:
        data *= CEILING / peak
    return data


def load_pcm(path: Path) -> np.ndarray:
    """Read immutable sources into owned 44.1 kHz stereo float buffers.

    Resampling happens before peak management. Compressed formats unsupported
    by libsndfile use ffmpeg float output, avoiding integer decode saturation.
    """
    try:
        data, rate = sf.read(path, dtype='float64', always_2d=True)
    except sf.LibsndfileError:
        result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
            '-vn', '-ac', '2', '-ar', str(SAMPLE_RATE), '-f', 'f64le', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
        data = np.frombuffer(result.stdout, dtype='<f8').reshape(-1, 2)
        rate = SAMPLE_RATE
    if not np.isfinite(data).all() or not len(data):
        raise ValueError('empty or non-finite audio')
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] != 2:
        raise ValueError('only mono and stereo sources are supported')
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE//divisor, rate//divisor, axis=0)
    return bounded_pcm(data)


@dataclass(frozen=True)
class DeckSpan:
    start: int
    end: int
    active: LibraryTrack
    source_start: int
    deck: str
    incoming: LibraryTrack | None = None
    plan: TransitionPlan | None = None
    phase_correction: int = 0

    def source_seconds(self, absolute_frame: int) -> float:
        return (self.source_start + max(0, min(absolute_frame, self.end)-self.start))/SAMPLE_RATE

    def incoming_seconds(self, absolute_frame: int) -> float | None:
        if self.plan is None:
            return None
        elapsed = max(0, min(absolute_frame-self.start, self.end-self.start))
        rate = self.plan.bpm_a / self.plan.bpm_b
        return self.plan.incoming_time + max(0, elapsed-self.phase_correction)*rate/SAMPLE_RATE


@dataclass(frozen=True)
class Segment:
    span: DeckSpan
    samples: np.ndarray


class SetEngine:
    def __init__(self, library: Library, *, transition_bars: int = 4,
                 dwell_seconds: float = 20, first: str | None = None, effects: bool = False):
        if not isinstance(transition_bars, int) or isinstance(transition_bars, bool) or transition_bars < 1:
            raise ValueError('transition bars must be a positive integer')
        if not math.isfinite(dwell_seconds) or dwell_seconds < 0:
            raise ValueError('dwell must be finite and non-negative')
        unique = {}
        for track in library.tracks:
            unique.setdefault(track.path.resolve(), track)
        self.tracks = list(unique.values())
        if not self.tracks:
            raise ValueError('empty library')
        if first:
            matches = [t for t in self.tracks if first.casefold() in t.path.name.casefold()]
            if not matches:
                raise ValueError(f'No track matching {first!r}')
            self.tracks.remove(matches[0]); self.tracks.insert(0, matches[0])
        self.effects = effects
        self.transition_bars = transition_bars
        self.dwell_seconds = dwell_seconds
        self.errors = list(library.errors)
        self.handoffs = []
        self._started = False

    def _load(self, track: LibraryTrack) -> np.ndarray:
        pcm = load_pcm(track.path)
        if not self.effects:
            return pcm
        from .animation.dj_behavior import DJBehaviorEngine, GestureEvent
        from .dj_planner import DJActionPlanner
        from .audio.effects import apply_hand_to_deck_effects, apply_small_hype_effects
        behavior = DJBehaviorEngine(track.deck.features, track.deck.timeline,
                                    energy=track.deck.energy)
        actions = DJActionPlanner(behavior.seed).plan(behavior.events, behavior.energy)
        sweeps = [GestureEvent(a.start, a.duration, 'hand_to_deck', a.side, a.strength)
                  for a in actions if a.action == 'filter_sweep']
        risers = [GestureEvent(a.start, a.duration, 'small_hype', a.side, a.strength)
                  for a in actions if a.action == 'gain_riser']
        pcm = apply_hand_to_deck_effects(pcm, SAMPLE_RATE, sweeps)
        return bounded_pcm(apply_small_hype_effects(pcm, SAMPLE_RATE, risers))

    def segments(self) -> Iterator[Segment]:
        if self._started:
            raise RuntimeError('a set can only be consumed once')
        self._started = True
        remaining = list(self.tracks)
        absolute, source_start, side = 0, 0, 'l'
        active, pcm = None, None
        while remaining and active is None:
            candidate = remaining.pop(0)
            try:
                pcm = self._load(candidate)
                active = candidate
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.errors.append(f'{candidate.path.name}: {exc}')
        if active is None:
            raise ValueError('None of the analyzed tracks could be decoded')
        # Fade only set/sequential boundaries; matched transitions own their gains.
        fade = min(frame(.05), len(pcm))
        pcm[:fade] *= np.linspace(0, 1, fade)[:, None]
        while True:
            selected = None
            for candidate in rank_candidates(active, remaining, source_start/SAMPLE_RATE,
                    transition_bars=self.transition_bars, dwell_seconds=self.dwell_seconds):
                try:
                    incoming = self._load(candidate.track)
                    plan = candidate.plan
                    match = tempo_match(plan, SAMPLE_RATE)
                    transformed, phase = phase_correct_incoming_transition_with_relationship(
                        incoming, plan, SAMPLE_RATE, active.deck.timeline, candidate.track.deck.timeline)
                    # One gain for the incoming window AND following solo, including
                    # stretch overshoot. Linear gains then bound the sum by CEILING.
                    peak = float(np.max(np.abs(transformed)))
                    gain = min(1.0, CEILING / peak) if peak else 1.0
                    transformed *= gain
                    incoming *= gain
                    mixed = execute_transition(pcm, transformed,
                        replace(plan, incoming_duration_seconds=plan.outgoing_duration_seconds),
                        sample_rate=SAMPLE_RATE)
                    end_source = frame(plan.incoming_time) + match.incoming_sample_count
                    if end_source >= len(incoming):
                        raise ValueError('incoming track has no material after transition')
                    # Smooth the return to native tempo over 20 ms; no source
                    # samples are dropped and the post-handoff clock remains native.
                    join = min(frame(.02), len(incoming)-end_source)
                    w = np.linspace(0, 1, join)[:, None]
                    incoming[end_source:end_source+join] = (
                        mixed[-1] * (1-w) + incoming[end_source:end_source+join]*w)
                    selected = candidate, incoming, mixed, end_source, phase
                    break
                except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    self.errors.append(f'{candidate.track.path.name}: transition rejected: {exc}')
            if selected is not None:
                candidate, incoming, mixed, next_source, phase = selected
                plan = candidate.plan
                stop = frame(plan.outgoing_time)
                solo = pcm[source_start:stop]
                if len(solo):
                    yield Segment(DeckSpan(absolute, absolute+len(solo), active, source_start, side), solo)
                    absolute += len(solo)
                span = DeckSpan(absolute, absolute+len(mixed), active, stop, side,
                               candidate.track, plan, phase.initial_correction_samples)
                yield Segment(span, mixed)
                absolute += len(mixed)
                self.handoffs.append((absolute, active.path, candidate.track.path, 'phase-aligned'))
                active, pcm, source_start = candidate.track, incoming, next_source
                remaining.remove(active)
                side = 'r' if side == 'l' else 'l'
                continue
            # A short/beatless/incompatible library still plays, without pretending
            # an unsupported rate is beat matched. Finish, then start the next song.
            solo = np.array(pcm[source_start:], copy=True)
            fade = min(frame(.05), len(solo))
            if fade:
                solo[-fade:] *= np.linspace(1, 0, fade)[:, None]
                yield Segment(DeckSpan(absolute, absolute+len(solo), active, source_start, side), solo)
                absolute += len(solo)
            next_track = None
            while remaining:
                candidate = remaining.pop(0)
                try:
                    incoming = self._load(candidate)
                    next_track = candidate
                    break
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                    self.errors.append(f'{candidate.path.name}: {exc}')
            if next_track is None:
                return
            self.handoffs.append((absolute, active.path, next_track.path, 'sequential'))
            active, pcm, source_start = next_track, incoming, 0
            side = 'r' if side == 'l' else 'l'
            fade = min(frame(.05), len(pcm))
            pcm[:fade] *= np.linspace(0, 1, fade)[:, None]
