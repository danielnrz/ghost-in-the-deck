"""Command line entry point for pre-analysing a track.

    python -m ghost_in_the_deck.analyze_cli [--track NAME] [--all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio.analysis import analyse
from .audio.library import DEFAULT_MUSIC_DIR, choose_track, find_tracks

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "analysis"


def analysis_path(track: Path) -> Path:
    return OUT_DIR / f"{track.stem}.json"


def run(track: Path) -> Path:
    features = analyse(track)
    target = features.save(analysis_path(track))
    print(f"{features.track}")
    print(f"  duration : {features.duration_seconds:8.2f} s")
    print(f"  bpm      : {features.bpm:8.2f}")
    print(f"  beats    : {len(features.beats):8d}  (first: {features.beats[:4]})")
    print(f"  frames   : {len(features.frame_times):8d}")
    print(f"  saved    : {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-analyse a track with librosa.")
    parser.add_argument("--music-dir", default=str(DEFAULT_MUSIC_DIR))
    parser.add_argument("--track", help="substring of the filename to analyse")
    parser.add_argument("--all", action="store_true", help="analyse every track found")
    args = parser.parse_args()

    if args.all:
        tracks = find_tracks(args.music_dir)
        if not tracks:
            raise SystemExit(f"no audio files in {args.music_dir}")
    else:
        tracks = [choose_track(args.music_dir, args.track)]

    for track in tracks:
        run(track)


if __name__ == "__main__":
    main()
