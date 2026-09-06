"""Offline two-file transition preview orchestration.

This is the small file-level seam between the existing analysis/cache, deck
planner and Phase 4B PCM executor.  It deliberately does not import the app
or any rendering code: a preview is useful in a headless test as well as from
the eventual UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from .audio.analysis import analyse
from .audio.decode import to_wav
from .audio.features import SCHEMA_VERSION, MusicFeatures
from .audio.mixer import execute_transition
from .deck import TrackDeck
from .transition_diagnostics import TransitionDiagnostics, diagnose_transition_drift
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


class TransitionPreviewError(ValueError):
    """Base error for a preview that cannot be rendered."""


class NoTransitionPlanError(TransitionPreviewError):
    """The established planner could not find a valid cue pair."""


class IncompatiblePCMError(TransitionPreviewError):
    """The two decoded sources cannot satisfy the Phase 4B PCM contract."""


@dataclass(frozen=True)
class TransitionPreviewReport:
    """The plan and diagnostics belonging to one written preview WAV."""

    output_path: Path
    plan: TransitionPlan
    diagnostics: TransitionDiagnostics
    sample_rate: int
    channels: int


def _feature_cache_path(track: Path, cache_dir: Path) -> Path:
    """Return a readable, collision-safe cache path for one source file.

    The stem is only decoration: two sources with the same name must remain
    separate even when they share one analysis directory.  Resolving before
    hashing also makes relative paths and symlinked paths address the same
    source cache entry.
    """
    source_identity = str(track.resolve())
    identity_digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
    return cache_dir / f"{track.stem}-{identity_digest}.json"


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths identify the same existing or future file."""
    if left.resolve() == right.resolve():
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError, ValueError):
        return False


def _validate_writable_paths(
    outgoing_path: Path,
    incoming_path: Path,
    output_path: Path,
    analysis_dir: PathLike,
) -> tuple[Path, Path]:
    """Validate every generated path before any cache or output write.

    Cache paths are caller-influenced through ``analysis_dir``.  They therefore
    need the same source-preservation checks as the public preview path,
    including existing hardlink aliases that cannot be detected by resolution
    alone.
    """
    cache_dir = Path(analysis_dir)
    if cache_dir.exists() and not cache_dir.is_dir():
        raise TransitionPreviewError(
            f"analysis_dir must be a directory path: {cache_dir}"
        )

    cache_paths = (
        _feature_cache_path(outgoing_path, cache_dir),
        _feature_cache_path(incoming_path, cache_dir),
    )
    source_paths = (outgoing_path, incoming_path)
    for source in source_paths:
        if _paths_alias(output_path, source):
            raise TransitionPreviewError(
                "transition preview output must not replace a source file"
            )
        if _paths_alias(cache_dir, source):
            raise TransitionPreviewError(
                "analysis_dir must not alias an input source file"
            )
    if _paths_alias(cache_dir, output_path):
        raise TransitionPreviewError(
            "analysis_dir must not alias the preview output"
        )

    if _paths_alias(cache_paths[0], cache_paths[1]):
        raise TransitionPreviewError(
            "generated feature-cache paths must be distinct files"
        )
    for cache_path in cache_paths:
        if _paths_alias(cache_path, output_path):
            raise TransitionPreviewError(
                "generated feature-cache path must not alias the preview output"
            )
        for source in source_paths:
            if _paths_alias(cache_path, source):
                raise TransitionPreviewError(
                    "generated feature-cache path must not alias an input source"
                )
    return cache_paths


def _cached_features(track: Path, cache_dir: Path, refresh: bool) -> MusicFeatures:
    """Load or create the established JSON ``MusicFeatures`` cache entry."""
    cached = _feature_cache_path(track, cache_dir)
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
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
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


def _validated_paths(
    outgoing_source: PathLike,
    incoming_source: PathLike,
    output_wav: PathLike,
) -> tuple[Path, Path, Path]:
    """Validate local source/output paths before analysis or PCM work."""
    outgoing_path = Path(outgoing_source)
    incoming_path = Path(incoming_source)
    output_path = Path(output_wav)
    if not outgoing_path.is_file():
        raise FileNotFoundError(outgoing_path)
    if not incoming_path.is_file():
        raise FileNotFoundError(incoming_path)
    if _paths_alias(outgoing_path, incoming_path):
        raise TransitionPreviewError(
            "outgoing and incoming sources must be distinct files"
        )
    if output_path.suffix.lower() != ".wav":
        raise TransitionPreviewError("transition preview output must be a .wav file")
    if _paths_alias(output_path, outgoing_path) or _paths_alias(
        output_path, incoming_path
    ):
        raise TransitionPreviewError(
            "transition preview output must not replace a source file"
        )
    return outgoing_path, incoming_path, output_path


def _build_plan(
    outgoing_path: Path,
    incoming_path: Path,
    *,
    refresh: bool,
    analysis_dir: PathLike,
    margin_seconds: float,
    limit_per_deck: int,
    transition_bars: int,
) -> tuple[MusicFeatures, MusicFeatures, TransitionPlan]:
    """Load cached features and select the established best transition plan."""
    cache_path = Path(analysis_dir)
    outgoing_features = _cached_features(outgoing_path, cache_path, refresh)
    incoming_features = _cached_features(incoming_path, cache_path, refresh)
    context = TwoDeckContext(
        TrackDeck.from_features(outgoing_features),
        TrackDeck.from_features(incoming_features),
    )
    plan = plan_transition(
        context,
        margin_seconds=margin_seconds,
        limit_per_deck=limit_per_deck,
        transition_bars=transition_bars,
    )
    if plan is None:
        raise NoTransitionPlanError(
            "no transition plan exists for the supplied outgoing and incoming tracks"
        )
    return outgoing_features, incoming_features, plan


def _render_transition_preview(
    outgoing_source: PathLike,
    incoming_source: PathLike,
    output_wav: PathLike,
    *,
    refresh: bool,
    analysis_dir: PathLike,
    margin_seconds: float,
    limit_per_deck: int,
    transition_bars: int,
) -> TransitionPreviewReport:
    """Render once and retain the exact plan used for the CLI report."""
    outgoing_path, incoming_path, output_path = _validated_paths(
        outgoing_source, incoming_source, output_wav
    )
    _validate_writable_paths(
        outgoing_path, incoming_path, output_path, analysis_dir
    )
    outgoing_features, incoming_features, plan = _build_plan(
        outgoing_path,
        incoming_path,
        refresh=refresh,
        analysis_dir=analysis_dir,
        margin_seconds=margin_seconds,
        limit_per_deck=limit_per_deck,
        transition_bars=transition_bars,
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
    source_paths = {outgoing_path.resolve(), incoming_path.resolve()}
    partial: Path | None = None
    try:
        file_descriptor, partial_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".partial",
            dir=str(output_path.parent),
        )
        partial = Path(partial_name)
        os.close(file_descriptor)
        if partial.resolve() in source_paths:
            raise TransitionPreviewError(
                "transition preview temporary output must not replace a source file"
            )
        if output_path.resolve() in source_paths:
            raise TransitionPreviewError(
                "transition preview output must not replace a source file"
            )

        import soundfile as sf

        sf.write(
            str(partial),
            np.array(mixed, dtype=np.float64, copy=True),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        if partial.resolve() in source_paths:
            raise TransitionPreviewError(
                "transition preview temporary output must not replace a source file"
            )
        if output_path.resolve() in source_paths:
            raise TransitionPreviewError(
                "transition preview output must not replace a source file"
            )
        partial.replace(output_path)
        partial = None
    except TransitionPreviewError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise TransitionPreviewError(
            f"could not write transition preview WAV {output_path}"
        ) from exc
    finally:
        if partial is not None:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass

    diagnostics = diagnose_transition_drift(
        TrackDeck.from_features(outgoing_features).timeline,
        TrackDeck.from_features(incoming_features).timeline,
        plan,
    )
    return TransitionPreviewReport(
        output_path=output_path,
        plan=plan,
        diagnostics=diagnostics,
        sample_rate=sample_rate,
        channels=outgoing_pcm[2],
    )


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
    return _render_transition_preview(
        outgoing_source,
        incoming_source,
        output_wav,
        refresh=refresh,
        analysis_dir=analysis_dir,
        margin_seconds=margin_seconds,
        limit_per_deck=limit_per_deck,
        transition_bars=transition_bars,
    ).output_path


def _signed(value: float | None, unit: str) -> str:
    """Format an optional signed diagnostic value for the CLI."""
    if value is None:
        return "unavailable"
    return f"{value:+.3f} {unit}"


def format_preview_summary(report: TransitionPreviewReport) -> str:
    """Return the concise plan, drift, and compatibility summary."""
    plan = report.plan
    diagnostics = report.diagnostics
    initial = _signed(diagnostics.initial_alignment_beat_fraction, "beats")
    if diagnostics.initial_alignment_beat_fraction is None:
        initial += " (one or both anchors are virtual)"
    return "\n".join(
        (
            f"wrote: {report.output_path}",
            "plan: "
            f"{plan.outgoing_track} @ {plan.outgoing_time:.3f}s "
            f"(bar {plan.outgoing_bar_index}) -> "
            f"{plan.incoming_track} @ {plan.incoming_time:.3f}s "
            f"(bar {plan.incoming_bar_index}); "
            f"{plan.transition_length_bars} bars, "
            f"{diagnostics.transition_duration_seconds:.3f}s executable",
            "drift (no time-stretch): "
            f"end {_signed(diagnostics.predicted_end_drift_seconds, 's')} "
            f"({_signed(diagnostics.predicted_end_drift_outgoing_beats, 'outgoing beats')}; "
            f"{_signed(diagnostics.predicted_end_drift_incoming_beats, 'incoming beats')}); "
            f"initial anchor phase {initial}",
            "PCM policy: "
            f"{report.sample_rate} Hz, {report.channels} channel(s); both sources "
            "must match in sample rate and channel count; no resampling or "
            "channel conversion is performed",
            "limits: source frames advance one-for-one, the shortest planned "
            "window wins, and the preview does not beat-realign or time-stretch",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Render one offline preview from two local audio paths."""
    parser = argparse.ArgumentParser(
        description="Render an offline two-file transition preview as a WAV."
    )
    parser.add_argument("outgoing", type=Path, help="outgoing local audio path")
    parser.add_argument("incoming", type=Path, help="incoming local audio path")
    parser.add_argument("output", type=Path, help="output .wav path")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_DIR,
        help="JSON feature-cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-run analysis instead of using a valid cached feature file",
    )
    args = parser.parse_args(argv)

    try:
        report = _render_transition_preview(
            args.outgoing,
            args.incoming,
            args.output,
            refresh=args.refresh,
            analysis_dir=args.analysis_dir,
            margin_seconds=EDGE_MARGIN_SECONDS,
            limit_per_deck=MAX_CANDIDATES_PER_DECK,
            transition_bars=DEFAULT_TRANSITION_LENGTH_BARS,
        )
    except FileNotFoundError as exc:
        print(f"FileNotFoundError: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(format_preview_summary(report))
    return 0


# The short name is convenient for callers that already speak in terms of a
# preview rather than a file render; both names intentionally share one path.
preview_transition = render_transition_preview


__all__ = [
    "IncompatiblePCMError",
    "NoTransitionPlanError",
    "TransitionPreviewReport",
    "TransitionPreviewError",
    "format_preview_summary",
    "preview_transition",
    "render_transition_preview",
]


if __name__ == "__main__":
    raise SystemExit(main())
