"""How the body feels the music: the behaviour layer between analysis and bones.

This is where the DJ behaviour engine will grow. Today it decides one thing -
how somebody standing behind a deck would move to this music right now - and it
decides it as a pure function of absolute playback time.

The shape of the movement comes from several rates running at once, because a
body moving at a single frequency reads as a machine:

    breath      ~7 s      never completely still
    weight      2 bars    slow shift from one leg to the other
    sway        1 bar     torso swinging across the beat grid
    bounce      1 beat    continuous rise and fall, not a decaying twitch
    pulse       on beat   the sharp accent, retained from Phase 0

Variation is deterministic. Runtime randomness would break the guarantee that a
playback time maps to exactly one pose, so per-bar character is derived from a
hash of the track name and the bar number instead. The same track always dances
the same way, and a stalled renderer still cannot change what the body is doing
at a given moment.

Nothing here imports Panda3D or librosa.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

from ..audio.features import MusicFeatures
from .cues import BeatTimeline
from .energy import EnergyTrack

# Seconds for a beat accent to fall to ~37%. Short: the accent is a highlight on
# top of the continuous groove, not the groove itself.
PULSE_DECAY = 0.16

# Seconds for the accent to rise. Without it the accent steps from nothing to
# full between two consecutive samples, which shows up as the head snapping
# rather than nodding. A body has inertia; this is the cheapest way to say so.
PULSE_ATTACK = 0.055

# Movement never stops entirely, even in the quietest passage.
MIN_INTENSITY = 0.35

# Period of the slowest layer. Deliberately not a whole number of bars, so the
# body never lines up into an exactly repeating loop.
BREATH_PERIOD = 7.3

# Bars per full weight shift, left leg to right leg and back.
WEIGHT_BARS = 2.0


def _unit(seed: int, bar: int, channel: int) -> float:
    """A stable pseudo-random 0..1 from a seed, bar number and channel.

    crc32 rather than hash(): Python randomises string hashing per process, so
    hash() would give a different dance every run.
    """
    payload = f"{bar}:{channel}".encode()
    return (zlib.crc32(payload, seed & 0xFFFFFFFF) & 0xFFFFFF) / float(0xFFFFFF)


def _signed(seed: int, bar: int, channel: int) -> float:
    """Same, mapped to -1..+1."""
    return _unit(seed, bar, channel) * 2.0 - 1.0


@dataclass(frozen=True)
class GrooveVariation:
    """The character of one bar. Small differences, deterministically chosen."""

    emphasis: float         # 0.85 .. 1.15 overall movement scale for this bar
    head_bias: float        # -1 .. +1 resting head turn
    shoulder_bias: float    # -1 .. +1 which shoulder works harder
    lead_side: float        # -1 .. +1 which side the weight favours


@dataclass(frozen=True)
class GrooveState:
    """Everything the animator needs, evaluated at one playback time."""

    time: float
    beat_index: int          # position on the beat grid; virtual outside the detected beats
    has_detected_beat: bool  # True only when beat_index names an actually detected beat
    beat_age: float
    beat_phase: float       # 0..1 between the surrounding beats
    bar_phase: float        # 0..1 through a four-beat bar
    energy: float           # 0..1 smoothed music energy
    intensity: float        # movement scale derived from energy
    pulse: float            # 0..1 sharp accent from the last beat
    bounce: float           # 0..1 continuous per-beat rise and fall
    sway: float             # -1..+1 torso swing, once per bar
    weight_shift: float     # -1..+1 which leg carries the weight
    breath: float           # -1..+1 slowest layer
    variation: GrooveVariation

    @property
    def has_beat(self) -> bool:
        """Deprecated alias for ``has_detected_beat``.

        Kept only because the sign of ``beat_index`` used to be the whole
        answer, before virtual beats past the last detected one could also be
        positive. New code should read ``has_detected_beat`` directly.
        """
        return self.has_detected_beat


class GrooveEngine:
    """Evaluates a GrooveState for an absolute playback time."""

    def __init__(
        self,
        features: MusicFeatures,
        timeline: BeatTimeline | None = None,
        pulse_decay: float = PULSE_DECAY,
        seed: str | None = None,
        pulse_attack: float = PULSE_ATTACK,
    ):
        self.features = features
        self.timeline = timeline if timeline is not None else BeatTimeline(features)
        self.energy = EnergyTrack(features)
        self.pulse_decay = pulse_decay
        self.pulse_attack = pulse_attack
        self.seed = zlib.crc32((seed if seed is not None else features.track).encode())

    @property
    def _envelope_peak(self) -> float:
        """Height of the un-normalised attack-decay curve at its maximum."""
        attack, decay = self.pulse_attack, self.pulse_decay
        if attack <= 0.0:
            return 1.0
        top = attack * math.log1p(decay / attack)
        return (1.0 - math.exp(-top / attack)) * math.exp(-top / decay)

    @property
    def response_window(self) -> float:
        """Reporting threshold shared with the timing instrumentation."""
        return self.pulse_decay * 3.0

    def variation_for(self, bar: int) -> GrooveVariation:
        """The character drawn for one bar."""
        return GrooveVariation(
            emphasis=0.85 + 0.30 * _unit(self.seed, bar, 0),
            head_bias=_signed(self.seed, bar, 1),
            shoulder_bias=_signed(self.seed, bar, 2),
            lead_side=_signed(self.seed, bar, 3),
        )

    def variation_at(self, bar: int, bar_phase: float) -> GrooveVariation:
        """Variation eased from this bar's character into the next one's.

        Switching character on the bar line makes the whole body snap, because
        emphasis scales every joint at once. Crossing the bar gradually keeps it
        continuous: at the end of one bar the value has already arrived at what
        the next bar starts from.
        """
        current = self.variation_for(bar)
        following = self.variation_for(bar + 1)
        blend = min(max(bar_phase, 0.0), 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)   # smoothstep, flat at both ends

        def mix(a: float, b: float) -> float:
            return a + (b - a) * blend

        return GrooveVariation(
            emphasis=mix(current.emphasis, following.emphasis),
            head_bias=mix(current.head_bias, following.head_bias),
            shoulder_bias=mix(current.shoulder_bias, following.shoulder_bias),
            lead_side=mix(current.lead_side, following.lead_side),
        )

    def _envelope(self, age: float) -> float:
        """Attack-decay shape for one accent, normalised to peak at 1."""
        if age < 0.0:
            return 0.0
        rise = 1.0 - math.exp(-age / self.pulse_attack)
        fall = math.exp(-age / self.pulse_decay)
        return (rise * fall) / self._envelope_peak

    def _pulse_at(self, time: float) -> float:
        """The sharp accent, strongest cue in the recent past wins."""
        pulse = 0.0
        window = self.pulse_decay * 8.0
        for cue in self.timeline.cues_in(time - window, time):
            pulse = max(pulse, cue.strength * self._envelope(time - cue.scheduled_time))
        return pulse if pulse > 1e-3 else 0.0

    def state_at(self, time: float) -> GrooveState:
        beat = self.timeline.phase_at(time)
        energy = self.energy.at(time)
        intensity = MIN_INTENSITY + (1.0 - MIN_INTENSITY) * energy

        cue = self.timeline.cue_before(time)
        beat_age = time - cue.scheduled_time if cue is not None else math.inf

        # Continuous rise and fall across the beat: 1 on the beat, 0 halfway
        # between. Being a function of phase, it never needs a decay to reset.
        bounce = 0.5 * (1.0 + math.cos(2.0 * math.pi * beat.phase))

        bars = beat.bar_index + beat.bar_phase
        sway = math.sin(2.0 * math.pi * beat.bar_phase)
        weight_shift = math.sin(2.0 * math.pi * bars / WEIGHT_BARS)
        breath = math.sin(2.0 * math.pi * time / BREATH_PERIOD)

        variation = self.variation_at(beat.bar_index, beat.bar_phase)

        return GrooveState(
            time=time,
            beat_index=beat.index,
            has_detected_beat=beat.is_real,
            beat_age=beat_age,
            beat_phase=beat.phase,
            bar_phase=beat.bar_phase,
            energy=energy,
            intensity=intensity,
            pulse=self._pulse_at(time),
            bounce=bounce,
            sway=sway,
            weight_shift=weight_shift,
            breath=breath,
            variation=variation,
        )
