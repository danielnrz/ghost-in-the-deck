"""Discovery of local audio files.

The prototype must never depend on a single hard-coded filename, so tracks are
always looked up from a directory at runtime.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"}

DEFAULT_MUSIC_DIR = Path(__file__).resolve().parents[3] / "testMusic"


def find_tracks(music_dir: Path | str | None = None) -> list[Path]:
    """Return supported audio files in ``music_dir``, sorted by name."""
    directory = Path(music_dir) if music_dir else DEFAULT_MUSIC_DIR
    if not directory.is_dir():
        return []
    tracks = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(tracks, key=lambda p: p.name.lower())


def choose_track(
    music_dir: Path | str | None = None, name_hint: str | None = None
) -> Path:
    """Pick one track to work with.

    With a hint, the first track whose name contains it (case-insensitive) wins.
    Without one, the smallest file is used: it is a deterministic choice that
    keeps the analysis step quick during development.
    """
    tracks = find_tracks(music_dir)
    if not tracks:
        directory = Path(music_dir) if music_dir else DEFAULT_MUSIC_DIR
        raise FileNotFoundError(f"no supported audio files in {directory}")

    if name_hint:
        hint = name_hint.lower()
        for track in tracks:
            if hint in track.name.lower():
                return track
        raise FileNotFoundError(f"no track matching {name_hint!r} in {tracks[0].parent}")

    return min(tracks, key=lambda p: p.stat().st_size)
