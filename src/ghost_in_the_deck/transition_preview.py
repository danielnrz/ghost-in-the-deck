"""Offline two-file transition preview orchestration.

This is the small file-level seam between the existing analysis/cache, deck
planner and Phase 4B PCM executor.  It deliberately does not import the app
or any rendering code: a preview is useful in a headless test as well as from
the eventual UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .audio.analysis import analyse
from .audio.decode import to_wav
from .audio.features import SCHEMA_VERSION, MusicFeatures
from .audio.mixer import execute_transition
from .deck import TrackDeck
from .transition import (
    DEFAULT_TRANSITION_LENGTH_BARS,
    EDGE_MARGIN_SECONDS,
    MAX_CANDIDATES_PER_DECK,
    TransitionPlan,
    TwoDeckContext,
    plan_transition,
)

PathLike: TypeAlias = Path | str

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "out" / "analysis"


class TransitionPreviewError(RuntimeError):
    """Base error for a preview that cannot be rendered."""


class NoTransitionPlanError(TransitionPreviewError):
    """The established planner could not find a valid cue pair."""


class IncompatiblePCMError(TransitionPreviewError):
    """The two decoded sources cannot satisfy the Phase 4B PCM contract."""


def _cached_features(track: Path, cache_dir: Path, refresh: bool) -> MusicFeatures:
    """Load or create the established JSON ``MusicFeatures`` cache entry."""
    cached = cache_dir / f"{track.stem}.json"
    if cached.is_file() and not refresh:
        try:
            stored = json.loads(cached.read_text())
            if int(stored.get("schema_version", 0)) >= SCHEMA_VERSION:
                return MusicFeatures.from_dict(stored)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
            # A partial or old cache is treated exactly like a cache miss.
            pass

    features = analyse(track)
    features.save(cached)
    # Use the persisted representation on the first call too.  The cache
    # intentionally rounds fields for stable JSON; returning the unrounded
    # analysis here would let the first plan differ from later cached plans.
    return MusicFeatures.load(cached)


def _load_pcm(source: Path) -> tuple[np.ndarray, int, int]:
    """Decode one source and return owned float PCM, rate and channel count."""
    try:
        wav = to_wav(source)
    except (OSError, RuntimeError) as exc:
        raise IncompatiblePCMError(
            f"could not decode {source} into PCM WAV for the transition preview"
        ) from exc

    try:
        import soundfile as sf

        info = sf.info(str(wav))
        samples, sample_rate = sf.read(
            str(wav), dtype="float64", always_2d=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise IncompatiblePCMError(
            f"{source} cannot be loaded as numeric PCM for the transition preview"
        ) from exc

    pcm = np.array(samples, dtype=np.float64, copy=True)
    if pcm.ndim != 2 or pcm.shape[1] < 1 or pcm.shape[0] < 1:
        raise IncompatiblePCMError(
            f"{source} must decode to non-empty PCM with shape (frames, channels)"
        )
    if not np.all(np.isfinite(pcm)):
        raise IncompatiblePCMError(f"{source} PCM contains non-finite samples")
    if int(info.channels) != pcm.shape[1] or int(sample_rate) <= 0:
        raise IncompatiblePCMError(f"{source} has an invalid PCM channel or rate contract")
    return pcm, int(sample_rate), int(info.channels)


def _validate_compatible_pcm(
    outgoing: tuple[np.ndarray, int, int], incoming: tuple[np.ndarray, int, int]
) -> int:
    """Validate the shared sample-rate/channel contract used by the mixer."""
    _, outgoing_rate, outgoing_channels = outgoing
    _, incoming_rate, incoming_channels = incoming
    if outgoing_rate != incoming_rate:
        raise IncompatiblePCMError(
            "outgoing and incoming sources must have the same PCM sample rate"
        )
    if outgoing_channels != incoming_channels:
        raise IncompatiblePCMError(
            "outgoing and incoming sources must have the same PCM channel count"
        )
    return outgoing_rate


def render_transition_preview(
    outgoing_source: PathLike,
    incoming_source: PathLike,
    output_wav: PathLike,
    *,
    refresh: bool = False,
    analysis_dir: PathLike = ANALYSIS_DIR,
    margin_seconds: float = EDGE_MARGIN_SECONDS,
    limit_per_deck: int = MAX_CANDIDATES_PER_DECK,
    transition_bars: int = DEFAULT_TRANSITION_LENGTH_BARS,
) -> Path:
    """Render a planned two-file transition to a newly-owned local WAV.

    Feature analysis is loaded from the same JSON cache shape and location used
    by the application, then the existing ``TrackDeck``, ``plan_transition``
    and ``execute_transition`` implementations do all planning and mixing.
    No source file is modified.  The output must be a distinct ``.wav`` path.

    Raises:
        NoTransitionPlanError: if the planner returns no plan.
        IncompatiblePCMError: if decoding or the shared PCM contract fails.
    """
    outgoing_path = Path(outgoing_source)
    incoming_path = Path(incoming_source)
    output_path = Path(output_wav)
    if not outgoing_path.is_file():
        raise FileNotFoundError(outgoing_path)
    if not incoming_path.is_file():
        raise FileNotFoundError(incoming_path)
    if outgoing_path.resolve() == incoming_path.resolve():
        raise TransitionPreviewError("outgoing and incoming sources must be distinct files")
    if output_path.suffix.lower() != ".wav":
        raise TransitionPreviewError("transition preview output must be a .wav file")
    if output_path.resolve() in {outgoing_path.resolve(), incoming_path.resolve()}:
        raise TransitionPreviewError("transition preview output must not replace a source file")

    cache_path = Path(analysis_dir)
    outgoing_features = _cached_features(outgoing_path, cache_path, refresh)
    incoming_features = _cached_features(incoming_path, cache_path, refresh)
    context = TwoDeckContext(
        TrackDeck.from_features(outgoing_features),
        TrackDeck.from_features(incoming_features),
    )
    plan: TransitionPlan | None = plan_transition(
        context,
        margin_seconds=margin_seconds,
        limit_per_deck=limit_per_deck,
        transition_bars=transition_bars,
    )
    if plan is None:
        raise NoTransitionPlanError(
            "no transition plan exists for the supplied outgoing and incoming tracks"
        )

    outgoing_pcm = _load_pcm(outgoing_path)
    incoming_pcm = _load_pcm(incoming_path)
    sample_rate = _validate_compatible_pcm(outgoing_pcm, incoming_pcm)
    try:
        mixed = execute_transition(
            outgoing_pcm[0], incoming_pcm[0], plan, sample_rate=sample_rate
        )
    except ValueError as exc:
        raise IncompatiblePCMError(
            "decoded sources cannot satisfy the Phase 4B transition PCM contract"
        ) from exc
    if mixed.size == 0:
        raise IncompatiblePCMError("transition mixer produced an empty PCM preview")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    try:
        import soundfile as sf

        sf.write(
            str(partial),
            np.array(mixed, dtype=np.float64, copy=True),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        partial.replace(output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise TransitionPreviewError(
            f"could not write transition preview WAV {output_path}"
        ) from exc
    return output_path


# The short name is convenient for callers that already speak in terms of a
# preview rather than a file render; both names intentionally share one path.
preview_transition = render_transition_preview


__all__ = [
    "IncompatiblePCMError",
    "NoTransitionPlanError",
    "TransitionPreviewError",
    "preview_transition",
    "render_transition_preview",
]
