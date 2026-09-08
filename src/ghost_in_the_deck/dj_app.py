"""The normal local-library DJ runtime, using the shared set execution engine."""
from __future__ import annotations

import queue
import time
from pathlib import Path

from .animation.controller import AvatarAnimator
from .animation.dj_behavior import DJActionState, DJBehaviorEngine, _envelope_weight
from .animation.groove import GrooveEngine
from .animation.rig import AvatarRig
from .clock import PlaybackClock
from .library import DEFAULT_CACHE, load_library
from .live import SetBuffer, StreamLedger
from .scene.workstation import build_workstation
from .set_engine import SAMPLE_RATE, SetEngine, frame


def transition_action(span, absolute_time: float) -> DJActionState | None:
    if span.plan is None:
        return None
    progress = (absolute_time*SAMPLE_RATE-span.start)/(span.end-span.start)
    if not 0 <= progress <= 1:
        return None
    # Hold the calibrated contact pose during the body of the crossfade.
    # The original gesture only touches briefly; remap its approach/release
    # without changing any calibrated joint targets or clearance trajectory.
    progress = progress * 2.5 if progress < .2 else (.5 if progress <= .8 else .5+(progress-.8)*2.5)
    return DJActionState(time=absolute_time, action='hand_to_deck', progress=progress,
        weight=_envelope_weight('hand_to_deck', progress),
        side='r' if span.deck == 'l' else 'l', strength=1.0)


def write_set_pose(animator, state, action, *, transition=False):
    animator._write_pose(state, action)
    if transition and action is not None and action.is_active:
        from .animation.gesture_pose import _reach_phase_weights
        _, target, lift, _ = _reach_phase_weights(action.progress)
        contact = target*(1-lift)
        joint = f'hand_{action.side}'
        h, p, r = animator.rig.offset_of(joint)
        # During the settled control interval, orient the fingers along the
        # control row. Keep the accepted lifted wrist on approach and release.
        animator.rig.set_offset(joint, heading=h, pitch=p-32*contact,
            roll=r+(-8 if action.side == 'l' else 8)*contact)


class LiveDJApp:
    def __init__(self, args):
        from direct.showbase.ShowBase import ShowBase
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode, UserDataAudio
        from .app import AVATAR, GhostApp

        self.args = args
        self.engine = SetEngine(load_library(args.music_dir, Path(args.cache_dir)/'analysis',
            refresh=args.refresh, progress=lambda s: print(s, flush=True)),
            transition_bars=args.transition_bars, dwell_seconds=args.dwell,
            first=args.track, effects=not (args.no_fx or args.no_actions))
        self.producer = SetBuffer(self.engine)
        self.ledger = StreamLedger()
        self.stream = UserDataAudio(SAMPLE_RATE, 2, True)
        self.stream_done = False
        self.closed = False
        self._ran = False
        self._grooves = {}
        self._behaviors = {}
        self._last_track = None
        self._next_stall_at = args.stall_every or None
        self._capture_times = list(args.capture_at)
        self._volume = .8
        self._muted = False
        try:
            # At least one owned audio chunk must exist before play starts.
            while True:
                try:
                    chunk = self.producer.queue.get(timeout=.1)
                    break
                except queue.Empty:
                    if self.producer.finished.is_set():
                        raise RuntimeError(self.producer.error or 'Set contains no audio')
            self._append(chunk)
            self.base = ShowBase()
            GhostApp._setup_scene(self)
            self.rig = AvatarRig(AVATAR, parent=self.base.render)
            self.workstation = build_workstation(self.base.render)
            GhostApp._frame_scene(self)
            # A raised view makes both control rows and hand clearance visible.
            low, high = self.rig.actor.getTightBounds()
            height = high.z-low.z
            self.base.camera.setZ(height*1.12)
            self.base.camera.lookAt(0, -.25, height*.52)
            self.animator = AvatarAnimator(self.rig)
            self.status = OnscreenText(text='Preparing set', pos=(-1.7, .92), scale=.042,
                fg=(.85, .92, 1, 1), align=TextNode.ALeft, mayChange=True)
            self.status.textNode.setWordwrap(70)
            self.controls = OnscreenText(text='Esc: quit   M: mute   Up/Down: volume',
                pos=(-1.7, -.94), scale=.035, fg=(.65,.7,.8,1), align=TextNode.ALeft)
            self.sound = None
            if not args.no_audio:
                self.sound = self.base.sfxManagerList[0].getSound(self.stream)
                if self.sound.status() == self.sound.BAD:
                    raise RuntimeError('OpenAL could not open the audio stream')
                self.sound.setVolume(self._volume)
            self.clock = PlaybackClock(self.sound)
            self._pump(0)
            self.base.accept('escape', self._finish)
            self.base.accept('m', self._mute)
            self.base.accept('arrow_up', self._volume_change, [.1])
            self.base.accept('arrow_down', self._volume_change, [-.1])
            self.base.taskMgr.add(self._update, 'ghost-set-update')
        except BaseException:
            self.producer.close()
            if hasattr(self, 'base'):
                self.base.destroy()
            raise

    def _append(self, chunk):
        self.ledger.append(chunk)
        if not self.args.no_audio:
            self.stream.append(chunk.data)

    def _pump(self, now):
        while self.ledger.frames/SAMPLE_RATE < now + 30:
            try:
                chunk = self.producer.queue.get_nowait()
            except queue.Empty:
                break
            self._append(chunk)
        if self.producer.finished.is_set() and self.producer.queue.empty() and not self.stream_done:
            self.stream.done()
            self.stream_done = True
            if self.producer.error:
                print(f'Set preparation stopped: {self.producer.error}', flush=True)

    def _mute(self):
        self._muted = not self._muted
        if self.sound:
            self.sound.setVolume(0 if self._muted else self._volume)

    def _volume_change(self, change):
        self._volume = max(0, min(1, self._volume+change))
        if self.sound and not self._muted:
            self.sound.setVolume(self._volume)

    def draw_at(self, now):
        span = self.ledger.at(now)
        if span is None:
            return
        source_time = span.source_seconds(frame(now))
        if span.active.path not in self._grooves:
            self._grooves[span.active.path] = GrooveEngine(span.active.deck.features, span.active.deck.timeline)
            self._behaviors[span.active.path] = DJBehaviorEngine(
                span.active.deck.features, span.active.deck.timeline,
                energy=self._grooves[span.active.path].energy)
        groove = self._grooves[span.active.path]
        action = None if self.args.no_actions else transition_action(span, now)
        if action is None and not self.args.no_actions:
            action = self._behaviors[span.active.path].state_at(source_time)
        write_set_pose(self.animator, groove.state_at(source_time), action, transition=span.plan is not None)
        if span.active.path != self._last_track:
            print(f'Deck {span.deck.upper()}: {span.active.path.name}', flush=True)
            self._last_track = span.active.path
        text = f'Deck {span.deck.upper()}  {span.active.path.name}\n'
        text += f'{source_time:.1f}s  |  measured tempo {span.active.deck.bpm:.1f} BPM'
        if span.plan:
            progress = max(0, min(1, (frame(now)-span.start)/(span.end-span.start)))
            text += f'\nMixing to {span.incoming.path.name}  {progress:.0%}\n'
            text += f'Pitch-preserving rate {span.plan.bpm_a/span.plan.bpm_b:.3f} | initial beat alignment'
        else:
            text += '\nPlaying set automatically'
        if self.engine.errors:
            text += f'\n{len(self.engine.errors)} analysis/transition issue(s); details at exit'
        self.status.setText(text)

    def _update(self, task):
        now = self.clock.time()
        if self._next_stall_at is not None and now >= self._next_stall_at:
            time.sleep(self.args.simulate_stall)
            self._next_stall_at = now + self.args.stall_every
            now = self.clock.time()
        self._pump(now)
        self.draw_at(now)
        if self._capture_times and now >= self._capture_times[0]:
            target = self._capture_times.pop(0)
            directory = Path(self.args.capture_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.base.graphicsEngine.renderFrame()
            self.base.win.saveScreenshot(str(directory/f'set-{target:07.2f}.png'))
        if self.args.seconds and now >= self.args.seconds:
            return self._finish()
        if self.stream_done and now >= self.ledger.frames/SAMPLE_RATE:
            return self._finish()
        if self.clock.holding:
            print('Audio playback stopped; ending set.', flush=True)
            return self._finish()
        return task.cont

    def _finish(self):
        from direct.task import Task
        if not self.closed:
            self.closed = True
            if self.sound:
                self.sound.stop()
            self.producer.close()
            self.base.taskMgr.remove('ghost-set-update')
            self.base.taskMgr.stop()
        return Task.done

    def run(self):
        if self._ran:
            raise RuntimeError('a live set can only start once')
        self._ran = True
        self.clock.start()
        try:
            self.base.run()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self._finish()
            self.base.destroy()
        for error in self.engine.errors:
            print(f'Notice: {error}')
        print(f'Set ended. Prepared {len(self.engine.handoffs)} handoff(s).')
