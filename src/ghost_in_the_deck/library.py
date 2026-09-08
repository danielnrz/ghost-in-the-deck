"""Local library indexing with content-validated, atomically written analysis."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from .audio.analysis import analyse
from .audio.features import MusicFeatures, SCHEMA_VERSION
from .audio.library import SUPPORTED_SUFFIXES
from .deck import TrackDeck

CACHE_VERSION = 1
DEFAULT_CACHE = Path.home() / '.cache' / 'ghost-in-the-deck'


def discover(directory: Path | str) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Music folder does not exist: {root}')
    tracks = []
    identities = set()
    for path in sorted(root.rglob('*'), key=lambda p: (str(p).casefold(), str(p))):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in identities:
            identities.add(identity)
            tracks.append(path.resolve())
    return tracks


def fingerprint(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_features(features: MusicFeatures) -> None:
    if not np.isfinite(features.duration_seconds) or features.duration_seconds <= 0:
        raise ValueError('track has no finite positive duration')
    if not np.isfinite(features.bpm) or features.bpm < 0:
        raise ValueError('invalid tempo')
    for name in ('beats', 'frame_times', 'rms', 'bass_energy', 'mid_energy',
                 'high_energy', 'onset_strength'):
        values = np.asarray(getattr(features, name))
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite {name}')
        if name in ('beats', 'frame_times') and (
            np.any(values < 0) or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(f'invalid {name} timeline')
    if not np.isfinite(features.peak_rms) or features.peak_rms < 0:
        raise ValueError('invalid loudness')


def cached_analysis(path: Path, cache: Path, refresh: bool = False) -> MusicFeatures:
    path = path.resolve()
    digest = fingerprint(path)
    identity = hashlib.sha256(str(path).encode()).hexdigest()
    target = cache / f'{identity}.json'
    # Atomic replacement cannot follow a cache-file symlink/hardlink into a song.
    # Reject aliases anyway, rather than replacing a caller's source directory entry.
    if target.resolve() == path or (target.exists() and os.path.samefile(target, path)):
        raise ValueError('analysis cache aliases source')
    key = {'content': digest, 'schema': SCHEMA_VERSION, 'version': CACHE_VERSION}
    if not refresh:
        try:
            stored = json.loads(target.read_text())
            if stored['identity'] == key:
                features = MusicFeatures.from_dict(stored['features'])
                validate_features(features)
                return features
        except (OSError, ValueError, KeyError, TypeError):
            pass
    features = analyse(path)
    validate_features(features)
    if fingerprint(path) != digest:
        raise ValueError('source changed during analysis; retry when the file is stable')
    stored = {'identity': key, 'features': features.to_dict()}
    cache.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.analysis-', dir=cache)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(stored, output)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return MusicFeatures.from_dict(stored['features'])


@dataclass(frozen=True)
class LibraryTrack:
    path: Path
    deck: TrackDeck


@dataclass
class Library:
    tracks: list[LibraryTrack]
    errors: list[str]


def load_library(directory: Path | str, cache: Path = DEFAULT_CACHE / 'analysis',
                 *, refresh: bool = False,
                 progress: Callable[[str], None] | None = None) -> Library:
    tracks, errors = [], []
    for path in discover(directory):
        if progress:
            progress(f'Analyzing {path.name}')
        try:
            features = cached_analysis(path, cache, refresh)
            tracks.append(LibraryTrack(path, TrackDeck.from_features(features)))
        except (ValueError, OSError, RuntimeError, EOFError, subprocess.SubprocessError) as exc:
            errors.append(f'{path.name}: {exc}')
    if not tracks:
        raise ValueError('No usable audio tracks in this folder.' +
                         (' ' + '; '.join(errors) if errors else ''))
    return Library(tracks, errors)
