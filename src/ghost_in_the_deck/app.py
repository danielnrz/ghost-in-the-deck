"""Phase 0 prototype: a rigged humanoid nodding along to a local track.

    python -m ghost_in_the_deck.app                 # pick a track, play, animate
    python -m ghost_in_the_deck.app --track minel
    python -m ghost_in_the_deck.app --no-audio --seconds 20   # silent smoke run

The chain is analysis -> MusicFeatures -> MotionCue -> AvatarAnimator -> AvatarRig.
This module only wires those together and owns the Panda3D scene.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from panda3d.core import loadPrcFileData

from .animation.controller import AvatarAnimator
from .animation.cues import BeatCueSource
from .animation.rig import AvatarRig
from .audio.analysis import analyse
from .audio.decode import to_wav
from .audio.features import MusicFeatures
from .audio.library import DEFAULT_MUSIC_DIR, choose_track
from .sync import SyncRecorder

ROOT = Path(__file__).resolve().parents[2]
AVATAR = ROOT / "assets" / "avatar" / "ghost_test.bam"
ANALYSIS_DIR = ROOT / "out" / "analysis"
REPORT_DIR = ROOT / "out"


def load_features(track: Path, refresh: bool = False) -> MusicFeatures:
    """Analysis is done once per track and cached; playback never waits on it."""
    cached = ANALYSIS_DIR / f"{track.stem}.json"
    if cached.is_file() and not refresh:
        return MusicFeatures.load(cached)
    features = analyse(track)
    features.save(cached)
    return features


class PlaybackClock:
    """Current position in the track.

    Panda3D's OpenAL sound reports its own play position, which is the honest
    reference for synchronisation. Without audio the wall clock stands in so the
    prototype can still be exercised headlessly.
    """

    def __init__(self, sound=None):
        self.sound = sound
        self._start = time.perf_counter()

    def start(self) -> None:
        self._start = time.perf_counter()
        if self.sound is not None:
            self.sound.play()

    def time(self) -> float:
        if self.sound is not None and self.sound.status() == self.sound.PLAYING:
            return self.sound.getTime()
        return time.perf_counter() - self._start

    def finished(self, duration: float) -> bool:
        return self.time() >= duration


def build_app(args) -> "GhostApp":
    if args.headless:
        loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "win-size 1280 720")
    loadPrcFileData("", "window-title Ghost in the Deck - Phase 0")
    if args.no_audio:
        loadPrcFileData("", "audio-library-name null")
    return GhostApp(args)


class GhostApp:
    def __init__(self, args):
        from direct.showbase.ShowBase import ShowBase

        self.args = args
        self.base = ShowBase()
        self.track = choose_track(args.music_dir, args.track)
        self.features = load_features(self.track, refresh=args.refresh)

        self._setup_scene()
        self.rig = AvatarRig(AVATAR, parent=self.base.render)
        self._frame_avatar()
        self.animator = AvatarAnimator(self.rig)
        self.cues = BeatCueSource(self.features)
        self.recorder = SyncRecorder()

        sound = None
        if not args.no_audio:
            sound = self.base.loader.loadSfx(str(to_wav(self.track)))
        self.clock = PlaybackClock(sound)

        self._frames = 0
        self._last = 0.0
        self._worst_frame = 0.0
        self._dropped = 0
        self.base.taskMgr.add(self._update, "ghost-update")

    # ------------------------------------------------------------------ scene
    def _setup_scene(self) -> None:
        from panda3d.core import (
            AmbientLight, CardMaker, DirectionalLight, Vec4,
        )

        self.base.setBackgroundColor(0.05, 0.05, 0.08)
        self.base.disableMouse()

        key = DirectionalLight("key")
        key.setColor(Vec4(1.0, 0.96, 0.9, 1))
        key_np = self.base.render.attachNewNode(key)
        key_np.setHpr(-35, -25, 0)
        self.base.render.setLight(key_np)

        rim = DirectionalLight("rim")
        rim.setColor(Vec4(0.25, 0.35, 0.6, 1))
        rim_np = self.base.render.attachNewNode(rim)
        rim_np.setHpr(150, -10, 0)
        self.base.render.setLight(rim_np)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.28, 0.28, 0.34, 1))
        self.base.render.setLight(self.base.render.attachNewNode(ambient))

        # A floor makes the body movement readable; nothing else is in the scene.
        card = CardMaker("floor")
        card.setFrame(-6, 6, -6, 6)
        floor = self.base.render.attachNewNode(card.generate())
        floor.setP(-90)
        floor.setColor(0.14, 0.14, 0.18, 1)

    def _frame_avatar(self) -> None:
        """Place the camera from the avatar's real bounds rather than guesses."""
        low, high = self.rig.actor.getTightBounds()
        height = high.z - low.z
        centre_z = low.z + height * 0.5
        self.base.camera.setPos(0.0, -height * 2.4, centre_z)
        self.base.camera.lookAt(0.0, 0.0, centre_z)

    # ------------------------------------------------------------------ frame
    def _update(self, task):
        now = self.clock.time()
        dt = max(0.0, now - self._last)
        self._last = now
        self._frames += 1
        if self._frames > 1:
            self._worst_frame = max(self._worst_frame, dt)

        before = self.cues.pending
        cues = self.cues.poll(now)
        self._dropped += (before - self.cues.pending) - len(cues)

        for cue in cues:
            self.animator.apply_cue(cue)
            timing = self.recorder.record(cue.index, cue.scheduled_time, now)
            if self.args.verbose:
                print(timing.format(), flush=True)

        self.animator.update(dt)

        if self.args.seconds and now >= self.args.seconds:
            return self._finish()
        if now >= self.features.duration_seconds or self.cues.pending == 0:
            return self._finish()
        return task.cont

    def _finish(self):
        from direct.task import Task

        self.base.taskMgr.remove("ghost-update")
        self.base.userExit()
        return Task.done

    # ------------------------------------------------------------------- run
    def frame_stats(self) -> dict:
        elapsed = max(self._last, 1e-6)
        return {
            "frames": self._frames,
            "average_fps": round(self._frames / elapsed, 1),
            "worst_frame_ms": round(self._worst_frame * 1000.0, 1),
            "dropped_beats": self._dropped,
        }

    def run(self) -> SyncRecorder:
        self.cues.reset(0.0)
        self._last = 0.0
        self.clock.start()
        try:
            self.base.run()
        except SystemExit:
            pass
        return self.recorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost in the Deck - Phase 0 prototype")
    parser.add_argument("--music-dir", default=str(DEFAULT_MUSIC_DIR))
    parser.add_argument("--track", help="substring of the filename to play")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    parser.add_argument("--headless", action="store_true", help="render offscreen")
    parser.add_argument("--no-audio", action="store_true", help="run without playback")
    parser.add_argument("--refresh", action="store_true", help="re-run analysis")
    parser.add_argument("--verbose", action="store_true", help="log every beat")
    parser.add_argument("--report", default=str(REPORT_DIR / "sync_report.json"))
    args = parser.parse_args()

    app = build_app(args)
    print(f"track : {app.track.name}")
    print(f"bpm   : {app.features.bpm:.2f}   beats: {len(app.features.beats)}")
    recorder = app.run()

    stats = app.frame_stats()
    print()
    print(
        f"Frames rendered      : {stats['frames']}  "
        f"({stats['average_fps']} fps, worst frame {stats['worst_frame_ms']} ms)"
    )
    print(f"Beats dropped as late: {stats['dropped_beats']}")
    print(recorder.format_summary())
    if recorder.timings:
        path = recorder.save(args.report, track=app.track.name)
        print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
