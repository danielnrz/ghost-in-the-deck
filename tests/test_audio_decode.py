"""Safety regressions for the non-WAV decoder cache boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import ghost_in_the_deck.audio.decode as decode


def _fake_ffmpeg(command: list[str], check: bool) -> subprocess.CompletedProcess:
    Path(command[-1]).write_bytes(b"decoded wav")
    return subprocess.CompletedProcess(command, 0)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_decode_rejects_target_alias_to_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source bytes")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(decode, "CACHE_DIR", cache_dir)
    target = decode._cache_path(source)
    cache_dir.mkdir()
    if alias_kind == "symlink":
        target.symlink_to(source)
    else:
        target.hardlink_to(source)

    with pytest.raises(RuntimeError, match="must not alias source"):
        decode.to_wav(source, force=True)

    assert source.read_bytes() == b"source bytes"


def test_decode_rejects_exact_target_alias_to_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source bytes")
    monkeypatch.setattr(decode, "_cache_path", lambda _: source)

    with pytest.raises(RuntimeError, match="must not alias source"):
        decode.to_wav(source, force=True)

    assert source.read_bytes() == b"source bytes"


def test_decode_does_not_follow_old_hardlinked_partial_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source bytes")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(decode, "CACHE_DIR", cache_dir)
    target = decode._cache_path(source)
    cache_dir.mkdir()
    old_partial = target.with_suffix(".partial.wav")
    old_partial.hardlink_to(source)
    monkeypatch.setattr(decode, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(decode.subprocess, "run", _fake_ffmpeg)

    decoded = decode.to_wav(source)

    assert decoded == target
    assert decoded.read_bytes() == b"decoded wav"
    assert old_partial.read_bytes() == b"source bytes"
    assert source.read_bytes() == b"source bytes"
    assert list(cache_dir.glob(".*.partial.wav")) == []


def test_decode_cleans_unique_partial_after_conversion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source bytes")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(decode, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(decode, "ffmpeg_available", lambda: True)

    def fail_ffmpeg(command: list[str], check: bool) -> None:
        Path(command[-1]).write_bytes(b"partial output")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(decode.subprocess, "run", fail_ffmpeg)

    with pytest.raises(subprocess.CalledProcessError):
        decode.to_wav(source)

    assert source.read_bytes() == b"source bytes"
    assert not decode._cache_path(source).exists()
    assert list(cache_dir.glob(".*.partial.wav")) == []
