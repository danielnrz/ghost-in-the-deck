"""Phase 0 prototype: a rigged humanoid nodding along to a local track.

    python -m ghost_in_the_deck.app                 # pick a track, play, animate
    python -m ghost_in_the_deck.app --track minel
    python -m ghost_in_the_deck.app --no-audio --seconds 20   # silent smoke run

The chain is analysis -> MusicFeatures -> BeatTimeline + EnergyTrack ->
GrooveEngine (how the body feels the music) and DJBehaviorEngine (what the DJ
is occasionally doing on top) -> AvatarAnimator -> AvatarRig.
This module only wires those together and owns the Panda3D scene.

The playback clock is the authority for musical progress. Each frame asks the
clock what time it is and draws the pose for that time; nothing is carried over
between frames. If the renderer stalls, frames are missed but the next one drawn
is correct for the moment it is drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from panda3d.core import loadPrcFileData

from .animation.controller import AvatarAnimator
from .animation.cues import BeatTimeline
from .animation.dj_behavior import DJBehaviorEngine
from .animation.groove import GrooveEngine
from .animation.workstation import DEFAULT_TARGETS
from .animation.rig import AvatarRig
from .audio.analysis import analyse
from .audio.decode import to_wav
from .audio.features import SCHEMA_VERSION, MusicFeatures
from .audio.library import DEFAULT_MUSIC_DIR, choose_track
from .clock import PlaybackClock
from .scene.workstation import build_workstation
from .sync import TimingRecorder

ROOT = Path(__file__).resolve().parents[2]
AVATAR = ROOT / "assets" / "avatar" / "ghost_test.bam"

# How long one --show-action loop runs before repeating, and (when --seconds
# is not given explicitly) how long the whole review run lasts - a few loops
# is plenty to judge a gesture by, and far short of waiting out a full track.
SHOW_ACTION_LOOP_SECONDS = 3.0
SHOW_ACTION_PREVIEW_SECONDS = 12.0
ANALYSIS_DIR = ROOT / "out" / "analysis"
REPORT_DIR = ROOT / "out"
# Rendered once per (wav, behaviour schedule) pair, same offline-cache pattern
# as audio/decode.py's own CACHE_DIR - a separate directory because the key
# these files are addressed by is different (it folds in the gesture schedule,
# not just the source file's own identity).
FX_CACHE_DIR = ROOT / "cache" / "audio_fx"


def processed_audio_path(wav: Path, behavior: DJBehaviorEngine) -> Path:
    """Render ``wav`` through the hand_to_deck filter sweep once, then cache it.

    The cache key folds in the source wav's own identity (path, size, mtime)
    and every hand_to_deck event's own timing/side/strength, so a stale render
    is never served after either the audio or the seed/schedule that derives
    the sweep changes - the same principle ``decode._cache_path`` uses for
    the plain decode, extended to include what this stage adds on top.
    """
    stat = wav.stat()
    events_key = "|".join(
        f"{e.start:.6f}:{e.duration:.6f}:{e.side}:{e.strength:.6f}"
        for e in behavior.events
        if e.kind == "hand_to_deck"
    )
    key = (
        f"{wav.resolve()}:{stat.st_size}:{int(stat.st_mtime)}:"
        f"{behavior.seed}:{events_key}"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    target = FX_CACHE_DIR / f"{wav.stem}-{digest}.wav"
    if target.is_file():
        return target

    import soundfile as sf

    from .audio.effects import apply_hand_to_deck_effects

    data, sample_rate = sf.read(str(wav), dtype="float64", always_2d=True)
    processed = apply_hand_to_deck_effects(data, sample_rate, behavior.events)

    FX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.wav")
    sf.write(str(partial), processed, sample_rate, subtype="PCM_16")
    partial.replace(target)
    return target


def positive_int(value: str) -> int:
    """argparse type for counts that are used as a divisor."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {number}")
    return number


def load_features(track: Path, refresh: bool = False) -> MusicFeatures:
    """Analysis is done once per track and cached; playback never waits on it.

    A cache written by an older schema is re-analysed rather than loaded, since
    it predates fields the groove now needs.
    """
    cached = ANALYSIS_DIR / f"{track.stem}.json"
    if cached.is_file() and not refresh:
        stored = json.loads(cached.read_text())
        if int(stored.get("schema_version", 0)) >= SCHEMA_VERSION:
            return MusicFeatures.from_dict(stored)
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
        self.workstation = build_workstation(self.base.render)
        self._frame_scene()
        self.timeline = BeatTimeline(self.features)
        self.groove = GrooveEngine(self.features, self.timeline)
        self.behavior = None if args.no_actions else DJBehaviorEngine(
            self.features, self.timeline, energy=self.groove.energy
        )
        self.animator = AvatarAnimator(
            self.rig, self.groove, self.behavior, DEFAULT_TARGETS
        )
        self.recorder = TimingRecorder(
            beat_times=self.features.beats,
            response_window=self.animator.response_window,
        )

        sound = None
        if not args.no_audio:
            wav_path = to_wav(self.track)
            if self.behavior is not None and not args.no_fx:
                wav_path = processed_audio_path(wav_path, self.behavior)
            sound = self.base.loader.loadSfx(str(wav_path))
        self.clock = PlaybackClock(sound)

        self._next_stall_at = args.stall_every if args.stall_every else None
        self._samples = 0
        self._show_action = args.show_action
        self._show_action_start = None
        self._show_action_period = SHOW_ACTION_LOOP_SECONDS
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

    def _frame_scene(self) -> None:
        """Frame the avatar and the workstation together, from a 3/4 angle.

        A dead-on front view hides the workstation behind the avatar's own
        body; a small side offset keeps both readable, which matters now that
        there is something worth seeing on the table.
        """
        import math

        low, high = self.rig.actor.getTightBounds()
        height = high.z - low.z
        centre_z = low.z + height * 0.5
        angle = math.radians(28.0)
        distance = height * 2.3
        self.base.camera.setPos(math.sin(angle) * distance, -math.cos(angle) * distance, centre_z * 0.95)
        self.base.camera.lookAt(0.0, -0.25, centre_z * 0.85)

    # ------------------------------------------------------------------ frame
    def _update(self, task):
        now = self.clock.time()

        # Deliberately freeze the update to show that a stalled loop does not
        # leave the musical state behind. Off unless asked for.
        if self._next_stall_at is not None and now >= self._next_stall_at:
            time.sleep(self.args.simulate_stall)
            self._next_stall_at = now + self.args.stall_every
            now = self.clock.time()

        # Taken together with `now`, so the playback and wall endpoints of the
        # interval measurements line up. The simulated stall sits before both.
        wall_now = time.perf_counter()

        if self._show_action:
            # Isolated review: loop one gesture's envelope on repeat, on top of
            # the same groove, rather than waiting for the schedule to pick it.
            if self._show_action_start is None:
                self._show_action_start = now
            phase = (now - self._show_action_start) % self._show_action_period
            action = self._preview_action_at(phase)
            state = self.groove.state_at(now)
            self.animator._write_pose(state, action)
        else:
            state = self.animator.apply_at(now)
        update_seconds = time.perf_counter() - wall_now

        response = self.recorder.record_sample(
            state,
            observed_at=self.clock.time(),
            wall_time=wall_now,
            update_seconds=update_seconds,
        )
        if response is not None and self.args.verbose:
            print(response.format(), flush=True)

        if self.args.debug_motion and self._samples % self.args.debug_every == 0:
            action = self.animator.action_at(now) if self.behavior else None
            action_text = (
                f"  action={action.action:<12} side={str(action.side):<4} "
                f"progress={action.progress:.2f} weight={action.weight:.2f}"
                if action is not None and action.is_active
                else ""
            )
            print(
                f"t={state.time:7.2f}  beat={state.beat_index:<5d} "
                f"phase={state.beat_phase:4.2f} bar={state.bar_phase:4.2f}  "
                f"energy={state.energy:4.2f} intensity={state.intensity:4.2f}  "
                f"pulse={state.pulse:4.2f} bounce={state.bounce:4.2f} "
                f"sway={state.sway:+5.2f} weight={state.weight_shift:+5.2f}"
                f"{action_text}",
                flush=True,
            )
        self._samples += 1

        if self.args.seconds and now >= self.args.seconds:
            return self._finish()
        if now >= self.features.duration_seconds:
            return self._finish()
        if now > self.timeline.end_time + self.animator.response_window:
            return self._finish()
        if self.clock.holding:
            # Playback stopped unexpectedly; do not dance on to a frozen clock.
            print("playback stopped before the run finished", flush=True)
            return self._finish()
        return task.cont

    def _preview_action_at(self, phase: float):
        """One gesture, looped with a settled pause between repeats.

        ``phase`` is time since this loop started, wrapped to
        ``_show_action_period``. The gesture itself runs for its own duration
        starting at 0; the remainder of the period is a pause at neutral so the
        transition in and out is visible rather than a jump-cut.
        """
        from .animation.dj_behavior import DJActionState, _envelope_weight

        duration = self._show_action_period * 0.55
        if phase >= duration:
            return DJActionState(
                time=phase, action="none", progress=0.0, weight=0.0, side=None, strength=0.0
            )
        progress = phase / duration
        weight = _envelope_weight(self._show_action, progress)
        return DJActionState(
            time=phase,
            action=self._show_action,
            progress=progress,
            weight=weight,
            side=self.args.show_side,
            strength=0.85,
        )

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
        "--no-actions",
        action="store_true",
        help="disable the DJ action layer; groove only, as in Phase 1A",
    )
    parser.add_argument(
        "--no-fx",
        action="store_true",
        help=(
            "disable the hand_to_deck filter sweep; play the decoded track "
            "unprocessed, for A/B comparison (effects are on by default)"
        ),
    )
    parser.add_argument(
        "--show-action",
        choices=("deck_glance", "lean_in", "hand_to_deck", "small_hype"),
        help=(
            "loop one gesture on repeat instead of the scheduled behaviour, for "
            f"review; runs {SHOW_ACTION_PREVIEW_SECONDS:.0f}s unless --seconds "
            "says otherwise"
        ),
    )
    parser.add_argument(
        "--show-side",
        choices=("l", "r"),
        default="l",
        help="side for --show-action gestures that use one (default: l)",
    )
    parser.add_argument(
        "--debug-motion",
        action="store_true",
        help="print beat phase, energy and groove values while playing",
    )
    parser.add_argument(
        "--debug-every",
        type=positive_int,
        default=30,
        metavar="N",
        help="print one debug line every N update samples (default 30)",
    )
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

    if args.show_action and args.seconds is None:
        # Reviewing one gesture on repeat has no reason to wait out a whole
        # track - a handful of loops of the 3 s preview period is plenty to
        # judge it by. Normal playback is untouched: this only fires when
        # --show-action is given and --seconds was not, so a real run always
        # plays the full track exactly as before.
        args.seconds = SHOW_ACTION_PREVIEW_SECONDS

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
