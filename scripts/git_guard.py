"""Check repository hygiene rules that can be evaluated from Git metadata."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {".venv", "cache", "out", "__pycache__", ".pytest_cache"}
PRIVATE_MEDIA_SUFFIXES = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mov",
    ".ogg",
    ".wav",
    ".webm",
}


def _git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT)


def tracked_paths() -> list[str]:
    return [path for path in _git("ls-files", "-z").decode().split("\0") if path]


def guard_paths(paths: list[str]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        parts = Path(path).parts
        if any(part in FORBIDDEN_DIRS for part in parts):
            failures.append(f"generated path is tracked: {path}")
        if parts and parts[0] == "testMusic" and path != "testMusic/.gitkeep":
            failures.append(f"private test media is tracked: {path}")
        if Path(path).suffix.lower() in PRIVATE_MEDIA_SUFFIXES:
            failures.append(f"private media suffix is tracked: {path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-work-branch", action="store_true")
    args = parser.parse_args()
    failures = guard_paths(tracked_paths())
    try:
        subprocess.run(
            ("git", "diff", "--check", "HEAD", "--"),
            cwd=ROOT,
            check=True,
        )
    except subprocess.CalledProcessError:
        failures.append("git diff --check reported whitespace errors")

    branch = _git("branch", "--show-current").decode().strip()
    if args.require_work_branch and branch == "main":
        failures.append("repository guard must not run on main")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"Git Guard: PASS ({len(tracked_paths())} tracked paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
