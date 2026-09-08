"""Decode arbitrary audio files to WAV via ffmpeg.

Two consumers need PCM: librosa (analysis) and Panda3D's OpenAL audio manager
(playback). Panda3D's OpenAL backend does not read MP3 or AAC, so every track is
decoded once into a cache directory and both stages use the same WAV file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "audio"
SAMPLE_RATE = 44100


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _cache_path(source: Path) -> Path:
    stat = source.stat()
    with source.open("rb") as handle:
        content = hashlib.file_digest(handle, "sha256").hexdigest()
    key = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{content}:{SAMPLE_RATE}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{source.stem}-{digest}.wav"


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths identify the same existing or future file."""
    if left.resolve() == right.resolve():
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError, ValueError):
        return False


def to_wav(source: Path | str, force: bool = False) -> Path:
    """Return a 44.1 kHz stereo WAV rendering of ``source``, decoding if needed."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".wav" and not force:
        return source

    target = _cache_path(source)
    if _paths_alias(target, source):
        raise RuntimeError(
            f"decode target must not alias source file: {source}"
        )
    if target.is_file() and not force:
        return target

    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to decode non-WAV audio")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial.wav",
        dir=str(target.parent),
    )
    partial = Path(partial_name)
    descriptor_open = True
    try:
        os.close(descriptor)
        descriptor_open = False
        if _paths_alias(partial, source):
            raise RuntimeError(
                f"decode temporary target must not alias source file: {source}"
            )
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(source),
                "-vn",                      # drop cover art
                "-ac", "2",
                "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(partial),
            ],
            check=True,
        )
        os.replace(partial, target)
    finally:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
    return target
