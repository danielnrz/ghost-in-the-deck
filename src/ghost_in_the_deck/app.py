"""Phase 0 prototype: a rigged humanoid nodding along to a local track.

    python -m ghost_in_the_deck.app                 # pick a track, play, animate
    python -m ghost_in_the_deck.app --track minel
    python -m ghost_in_the_deck.app --no-audio --seconds 20   # silent smoke run

The chain is analysis -> MusicFeatures -> BeatTimeline -> AvatarAnimator -> AvatarRig.
This module only wires those together and owns the Panda3D scene.

The playback clock is the authority for musical progress. Each frame asks the
clock what time it is and draws the pose for that time; nothing is carried over
between frames. If the renderer stalls, frames are missed but the next one drawn
is correct for the moment it is drawn.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from panda3d.core import loadPrcFileData

from .animation.controller import AvatarAnimator
from .animation.cues import BeatTimeline
from .animation.rig import AvatarRig
from .audio.analysis import analyse
from .audio.decode import to_wav
from .audio.features import MusicFeatures
from .audio.library import DEFAULT_MUSIC_DIR, choose_track
from .clock import PlaybackClock
from .sync import TimingRecorder

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
        self.timeline = BeatTimeline(self.features)
        self.animator = AvatarAnimator(self.rig, self.timeline)
        self.recorder = TimingRecorder(visible_for=self.animator.visible_for)

        sound = None
        if not args.no_audio:
            sound = self.base.loader.loadSfx(str(to_wav(self.track)))
        self.clock = PlaybackClock(sound)

        self._previous_audio: float | None = None
        self._previous_wall: float | None = None
        self._next_stall_at = args.stall_every if args.stall_every else None
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
        entered = time.perf_counter()
        now = self.clock.time()

        # Deliberately freeze the update to show that a stalled renderer does not
        # leave the musical state behind. Off unless asked for.
        if self._next_stall_at is not None and now >= self._next_stall_at:
            time.sleep(self.args.simulate_stall)
            self._next_stall_at = now + self.args.stall_every
            now = self.clock.time()

        audio_interval = None if self._previous_audio is None else now - self._previous_audio
        wall_interval = None if self._previous_wall is None else entered - self._previous_wall
        self._previous_audio = now
        self._previous_wall = entered

        state = self.animator.apply_at(now)
        update_seconds = time.perf_counter() - entered

        response = self.recorder.record_frame(
            state,
            observed_at=self.clock.time(),
            frame_interval=audio_interval,
            update_seconds=update_seconds,
            wall_interval=wall_interval,
        )
        if response is not None and self.args.verbose:
            print(response.format(), flush=True)

        if self.args.seconds and now >= self.args.seconds:
            return self._finish()
        if now >= self.features.duration_seconds:
            return self._finish()
        if now > self.timeline.end_time + self.animator.visible_for:
            return self._finish()
        return task.cont

    def _finish(self):
        from direct.task import Task

        self.base.taskMgr.remove("ghost-update")
        self.base.userExit()
        return Task.done

    # ------------------------------------------------------------------- run
    def run(self) -> TimingRecorder:
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
    parser.add_argument(
        "--simulate-stall",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="freeze the update loop periodically, to exercise stall recovery",
    )
    parser.add_argument(
        "--stall-every",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="playback interval between simulated stalls",
    )
    parser.add_argument("--report", default=str(REPORT_DIR / "sync_report.json"))
    args = parser.parse_args()

    if args.simulate_stall and not args.stall_every:
        parser.error("--simulate-stall needs --stall-every")

    app = build_app(args)
    print(f"track : {app.track.name}")
    print(f"bpm   : {app.features.bpm:.2f}   beats: {len(app.features.beats)}")
    recorder = app.run()

    print()
    print(recorder.format_summary())
    if recorder.responses:
        path = recorder.save(args.report, track=app.track.name)
        print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
