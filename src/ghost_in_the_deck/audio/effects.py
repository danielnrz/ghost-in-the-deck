"""A real, deterministic audio effect tied to the ``hand_to_deck`` gesture.

Phase 1B built a hand that visibly lands on a control; nothing it did changed
what was heard. This module closes that gap for exactly one gesture: every
scheduled ``hand_to_deck`` event sweeps a simple one-pole filter across its own
attack-hold-release window, so the moment the avatar's hand appears to touch a
knob is the moment the track's own timbre audibly moves.

Nothing here imports Panda3D. It operates on plain PCM sample arrays and a
sample rate - a numpy array in, a numpy array out - importable and testable
standalone, in the spirit of ``audio/features.py``.

The one deliberate exception to "this module only knows about audio" is
importing ``GestureEvent`` and ``_envelope_weight`` from
``animation.dj_behavior``. That module has no Panda3D dependency of its own
(only ``animation/rig.py`` does), and the whole point of tying the effect to
the gesture is that they are *the same curve*, not two independently tuned
ones that happen to look similar - reusing the exact function the pose
composer already calls is what makes that true, rather than a promise kept by
convention.

Determinism matches every other absolute-time module in this project
(``GrooveEngine``, ``DJBehaviorEngine``, ``BeatTimeline``): the filter cutoff
at sample index n is a pure function of n (via its time t = n / sample_rate)
and the fixed event schedule, not a state built up by how the buffer has been
walked before. Processing the same input against the same schedule twice
produces byte-identical output.

Filter design
--------------
A one-pole filter's smoothing coefficient ``alpha`` and its cutoff frequency
are two views of the same knob (see ``_lowpass_alpha_at_cutoff`` /
``_highpass_alpha_at_cutoff`` for the closed-form conversion); this module
sweeps ``alpha`` directly because doing so makes the "no effect" endpoint
exact by construction:

- Low-pass: ``y[n] = y[n-1] + alpha * (x[n] - y[n-1])``. At ``alpha = 1`` this
  reduces to ``y[n] = x[n]`` regardless of any prior state - so mapping
  envelope weight 0 to ``alpha = 1`` gives an attack/release that is
  mathematically silent (not just small) outside the gesture's own window.
- High-pass: ``y[n] = alpha * (y[n-1] + x[n] - x[n-1])``. The same ``alpha = 1``
  identity holds only if the running state satisfies ``y[n-1] == x[n-1]``,
  which is not generally true right after a sweep back down (the whole
  preceding window deliberately made them differ) - so the two boundary
  samples of every window are explicitly pinned to the dry input after
  filtering, rather than trusted to fall out of the recursion. See
  ``_render_window`` for where that pin happens and why it's needed for
  high-pass but is a no-op for low-pass.

``side="l"`` sweeps a low-pass filter down and back up - cutting highs, the
classic filter-knob move. ``side="r"`` sweeps a high-pass filter up and back
down - cutting lows. This mirrors the workstation's own
``left_controls``/``right_controls`` split: the two hands audibly do
different, mirrored things, not the same effect twice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..animation.dj_behavior import GestureEvent, _envelope_weight

# Cutoff frequency, in Hz, at the deepest point of a full-weight, full-strength
# event - i.e. what the sweep dives to at the middle of the hold, before it
# eases back out. Chosen to be unmistakable against typical program material
# without silencing the track outright: 450 Hz keeps kick/bass through a
# low-pass sweep while audibly dulling everything above it; 900 Hz keeps most
# midrange and highs through a high-pass sweep while audibly thinning the
# bass. Not derived from the track - a fixed, documented constant, same as
# BASE_DURATION and ENVELOPE_SHAPE in dj_behavior.py are fixed constants
# rather than measured per track.
LOWPASS_DEEP_CUTOFF_HZ = 450.0
HIGHPASS_DEEP_CUTOFF_HZ = 900.0

_KIND = "hand_to_deck"


def _lowpass_alpha_at_cutoff(cutoff_hz: float, sample_rate: int) -> float:
    """alpha for a one-pole low-pass with the given cutoff, at this sample rate."""
    c = 2.0 * np.pi * cutoff_hz / sample_rate
    return c / (1.0 + c)


def _highpass_alpha_at_cutoff(cutoff_hz: float, sample_rate: int) -> float:
    """alpha for a one-pole high-pass with the given cutoff, at this sample rate."""
    d = 2.0 * np.pi * cutoff_hz / sample_rate
    return 1.0 / (1.0 + d)


def _progress_and_weight(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """``n_points`` samples spanning progress 0..1 inclusive, and their weight.

    Reuses ``_envelope_weight`` directly (element by element) rather than a
    vectorised re-derivation of its shape, so the effect's envelope is
    provably the same function the pose composer evaluates for the visible
    gesture - not a numpy port of it that could quietly drift.
    """
    progress = np.linspace(0.0, 1.0, n_points)
    weight = np.array([_envelope_weight(_KIND, float(p)) for p in progress])
    return progress, weight


def _alpha_series(weight: np.ndarray, strength: float, side: str, sample_rate: int) -> tuple[np.ndarray, bool]:
    """Per-sample filter coefficient for one event, and whether it is high-pass."""
    depth = weight * min(max(strength, 0.0), 1.0)
    if side == "l":
        alpha_min = _lowpass_alpha_at_cutoff(LOWPASS_DEEP_CUTOFF_HZ, sample_rate)
        is_highpass = False
    elif side == "r":
        alpha_min = _highpass_alpha_at_cutoff(HIGHPASS_DEEP_CUTOFF_HZ, sample_rate)
        is_highpass = True
    else:
        raise ValueError(f"hand_to_deck event has no usable side: {side!r}")
    alpha = 1.0 - depth * (1.0 - alpha_min)
    return alpha, is_highpass


def _run_lowpass(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """x: (n, channels). alpha: (n,), one coefficient per sample, all channels."""
    y = np.empty_like(x)
    prev = x[0].copy()
    for n in range(x.shape[0]):
        prev = prev + alpha[n] * (x[n] - prev)
        y[n] = prev
    return y


def _run_highpass(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    prev_y = x[0].copy()
    prev_x = x[0].copy()
    for n in range(x.shape[0]):
        cur = alpha[n] * (prev_y + x[n] - prev_x)
        y[n] = cur
        prev_x = x[n]
        prev_y = cur
    return y


def _render_window(x: np.ndarray, weight: np.ndarray, strength: float, side: str, sample_rate: int) -> np.ndarray:
    """Filtered samples for one event's window, exact at both endpoints.

    ``weight`` is exactly 0.0 at index 0 and index -1 (``_envelope_weight``'s
    own contract). For low-pass that already makes the recursion reduce to the
    dry sample there, with no help needed: ``y[n] = y[n-1] + 1*(x[n]-y[n-1])``
    is ``x[n]`` regardless of prior state.

    High-pass is not so lucky. Its recursion also produces ``y[0] == x[0]``
    exactly (both running-state variables start equal to ``x[0]``), but by the
    end of the window ``y[n-1]`` has generally drifted away from ``x[n-1]`` -
    the whole preceding sweep deliberately made them differ - so the *raw*
    recursion output at the last sample is ``x[-1] + (y[n-2] - x[n-2])``, not
    ``x[-1]``. Forcing that last sample to ``x[-1]`` without correcting for
    the drift (as an earlier version of this function did) leaves the
    second-to-last sample exactly where the drift put it and the last sample
    pinned to dry, i.e. a single-sample jump equal to the whole accumulated
    drift - an audible click. Instead, the drift is measured once
    (``offset``) and subtracted back out as a ramp from 0 (at the start,
    where it is already zero) to the full offset (at the end, cancelling it
    exactly), so the correction is spread smoothly across the window and the
    tail rejoins the dry signal at the same slope the dry signal already had,
    rather than in one discontinuous step.

    The explicit endpoint assignments after that are then true no-ops for
    low-pass and exact-by-construction for high-pass; they are kept anyway as
    a floating-point safety net against the ramp not landing on exactly
    ``x[-1]``.
    """
    alpha, is_highpass = _alpha_series(weight, strength, side, sample_rate)
    y = _run_highpass(x, alpha) if is_highpass else _run_lowpass(x, alpha)
    if is_highpass:
        offset = y[-1] - x[-1]
        ramp = np.linspace(0.0, 1.0, len(y), dtype=x.dtype)
        if y.ndim > 1:
            ramp = ramp[:, None]
        y = y - ramp * offset
    y[0] = x[0]
    y[-1] = x[-1]

    # A one-pole high-pass built from a difference term can overshoot on a
    # transient (a step input spikes before it decays); left unchecked that
    # can push samples outside the input's own dynamic range, which the
    # eventual PCM_16 write would then silently saturate. Bound each window
    # to whatever range the dry audio in that window (or the valid PCM
    # ceiling, if that is looser) already used, so the effect can reshape the
    # spectrum but never hands downstream code a sample it didn't already
    # have to represent.
    limit = max(float(np.max(np.abs(x))), 1.0)
    np.clip(y, -limit, limit, out=y)
    return y


def apply_hand_to_deck_effects(
    samples: np.ndarray,
    sample_rate: int,
    events: Sequence[GestureEvent],
) -> np.ndarray:
    """Render every scheduled ``hand_to_deck`` event's filter sweep into ``samples``.

    ``samples`` is PCM audio, mono ``(n,)`` or multi-channel ``(n, channels)``,
    any float dtype. Returns a new array; the input is never mutated. Every
    other sample - anywhere outside a ``hand_to_deck`` event's own
    ``[start, start + duration]`` window - is copied through bit-identical.
    An empty or all-non-hand_to_deck event list is a no-op: the returned array
    equals the input exactly, dtype included, so it round-trips through a
    byte-for-byte comparison rather than merely an equal-value one.
    """
    out = np.array(samples, dtype=samples.dtype, copy=True)
    was_1d = out.ndim == 1
    if was_1d:
        out = out[:, None]

    total = out.shape[0]
    for event in events:
        if event.kind != _KIND:
            continue

        start_idx = int(round(event.start * sample_rate))
        span = max(1, int(round(event.duration * sample_rate)))
        n_points = span + 1
        if start_idx >= total:
            continue
        end_idx = min(start_idx + n_points, total)
        n_points = end_idx - start_idx
        if n_points < 2:
            continue

        _progress, weight = _progress_and_weight(n_points)
        window = out[start_idx:end_idx]
        out[start_idx:end_idx] = _render_window(
            window, weight, event.strength, event.side, sample_rate
        )

    if was_1d:
        out = out[:, 0]
    return out
