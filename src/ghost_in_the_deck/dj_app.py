"""The normal local-library DJ runtime, using the shared set execution engine."""
from __future__ import annotations

from dataclasses import replace
import queue
import time
from pathlib import Path

from .animation.controller import AvatarAnimator
from .animation.dj_behavior import DJActionState, _smoothstep
from .animation.visual_intent import VisualTimeline, deck_label, HANDOFF_RECOVERY
from .animation.energy import EnergyTrack
from .animation.structure import PHRASE_SMOOTHING_SECONDS
from .animation.groove import GrooveEngine, GrooveVariation
from .animation.rig import AvatarRig
from .clock import PlaybackClock
from .library import load_library
from .live import SetBuffer, StreamLedger
from .scene.workstation import build_workstation
from .set_engine import SAMPLE_RATE, SetEngine, frame


def transition_action(span, absolute_time: float) -> DJActionState | None:
    """Diagnostic convenience using the same admitted choreography as live sets."""
    visual = VisualTimeline()
    visual.update([span])
    return visual.at(absolute_time, span.deck).action_at(absolute_time)


def set_pose_offsets(animator, state, action):
    offsets = animator.pose_offsets(state)
    if action is None or not action.is_active:
        return offsets
    if action.action != 'hand_to_deck':
        return animator.pose_offsets(state, action)
    side = action.side
    sign = 1 if side == 'l' else -1
    # The settled middle fingertip is over the real knob, 13 mm above its
    # top. Panel knobs share a world-X offset, so these solves are asymmetric.
    contact = {
        'l': ((-9.584,.003,-28.823),(14.947,46.631,15),(0,33.841,-19.987)),
        'r': ((9.539,-2.839,32.513),(-13.332,45.463,-13.148),(0,33.777,20)),
    }[side]
    # Elbow first: fold the forearm behind the near edge, then carry the bent
    # arm above the controls, then settle. No broad lateral shoulder detour.
    # Each leg has zero endpoint velocity; reverse the same safe path to leave.
    u = 2*min(action.progress,1-action.progress)
    neutral = ((0,0,0),(0,0,0),(0,0,0))
    tuck = ((0,0,0),(0,-55,0),(0,0,0))
    above = ((contact[0][0],contact[0][1],-46*sign),(0,-55,0),(0,60,-20*sign))
    extended = (above[0],contact[1],above[2])
    if u < .25:
        a,b,w = neutral,tuck,_smoothstep(u/.25)
    elif u < .45:
        a,b,w = tuck,above,_smoothstep((u-.25)/.20)
    elif u < .78:
        a,b,w = above,extended,_smoothstep((u-.45)/.33)
    else:
        a,b,w = extended,contact,_smoothstep((u-.78)/.22)
    presence = _smoothstep(u/.25)
    for joint, first, last in zip(('upperarm','lowerarm','hand'),a,b):
        name=f'{joint}_{side}'
        motion=tuple(x+(y-x)*w for x,y in zip(first,last))
        groove=offsets.get(name,(0,0,0))
        offsets[name]=tuple(x+y*(1-presence) for x,y in zip(motion,groove))
    for joint in ('pelvis','spine_01','spine_02','spine_03'):
        offsets[joint]=tuple(v*(1-presence) for v in offsets.get(joint,(0,0,0)))
    name=f'clavicle_{side}'
    offsets[name]=tuple(a*(1-presence)+b*presence for a,b in zip(offsets.get(name,(0,0,0)),(0,-3,0)))
    return offsets


def write_set_pose(animator, state, action, *, transition=False):
    animator.rig.reset()
    for name, (h, p, r) in set_pose_offsets(animator, state, action).items():
        animator.rig.set_offset(name, heading=h, pitch=p, roll=r)


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
        self._broad_energy = {}
        self.visual = VisualTimeline()
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
            import math
            distance = height*2.3
            self.base.camera.setX(math.sin(math.radians(16))*distance)
            self.base.camera.setY(-math.cos(math.radians(16))*distance)
            self.base.camera.setZ(height*1.12)
            self.base.camera.lookAt(0, -.25, height*.52)
            from .animation.workstation import DEFAULT_TARGETS
            for side in ('l', 'r'):
                label = TextNode(f'deck-label-{side}')
                label.setText(deck_label(side))
                label.setAlign(TextNode.ACenter)
                label.setTextColor(.75,.82,.9,1)
                node = self.workstation.attachNewNode(label)
                node.setScale(.04)
                node.setPos(DEFAULT_TARGETS.deck_for(side)[0], -.59, .925)
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

    def _groove_state(self, track, source_time):
        if track.path not in self._grooves:
            self._grooves[track.path] = GrooveEngine(track.deck.features, track.deck.timeline)
            self._broad_energy[track.path] = EnergyTrack(track.deck.features,
                smoothing_seconds=PHRASE_SMOOTHING_SECONDS)
        groove = self._grooves[track.path]
        state = groove.state_at(source_time)
        broad = self._broad_energy[track.path]
        level = .4*state.energy + .6*broad.at(source_time)
        trend = broad.at(source_time)-broad.at(max(0, source_time-8))
        # Continuous measured energy/trend, no threshold-triggered gestures or
        # hash-selected body character. Quiet music can be almost still.
        intensity = max(.12, min(.62, .14+.44*level+.10*trend))
        return replace(state, intensity=intensity, pulse=state.pulse*.45,
            variation=GrooveVariation(1, 0, .12, .15))

    def pose_at(self, now):
        span = self.ledger.at(now)
        source_time = span.source_seconds(frame(now))
        self.visual.update(self.ledger.spans)
        intent = self.visual.at(now, span.deck)
        action = None if self.args.no_actions else intent.action_at(now)
        state = self._groove_state(span.active, source_time)
        offsets = set_pose_offsets(self.animator, state, action)
        # Ownership changes exactly on the audio sample. Blend *body offsets*
        # from the continuing outgoing groove to the new source for one bar;
        # do not interpolate source clocks or delay the actual handoff.
        index = self.ledger.spans.index(span)
        if index and now < span.start/SAMPLE_RATE+HANDOFF_RECOVERY:
            previous = self.ledger.spans[index-1]
            if previous.active.path != span.active.path:
                old_time = (previous.source_start+previous.end-previous.start)/SAMPLE_RATE + now-span.start/SAMPLE_RATE
                old = set_pose_offsets(self.animator, self._groove_state(previous.active, old_time), action)
                w = _smoothstep((now-span.start/SAMPLE_RATE)/HANDOFF_RECOVERY)
                offsets = {j: tuple(a+(b-a)*w for a,b in zip(old.get(j,(0,0,0)), offsets.get(j,(0,0,0))))
                           for j in old.keys() | offsets.keys()}
        if not self.args.no_actions:
            side, attention, nod = self.visual.attention_at(now)
            if side:
                sign = 1 if side == 'l' else -1
                for joint, turn, tilt in [('head', 8, 5), ('neck_01', 2, 2)]:
                    h,p,r = offsets.get(joint, (0,0,0))
                    offsets[joint] = (h*(1-.5*attention)+sign*turn*attention,
                                     p*(1-.5*attention)+tilt*attention+2*nod, r)
                # Head/neck lead attention; retain calibrated torso/shoulder
                # reach geometry so moving the torso cannot pull fingers low.
        return offsets, intent, action

    def draw_at(self, now):
        span = self.ledger.at(now)
        if span is None:
            return
        source_time = span.source_seconds(frame(now))
        offsets, intent, action = self.pose_at(now)
        self.rig.reset()
        for name, (h,p,r) in offsets.items():
            self.rig.set_offset(name, heading=h, pitch=p, roll=r)
        if span.active.path != self._last_track:
            print(f'Deck {deck_label(span.deck)}: {span.active.path.name}', flush=True)
            self._last_track = span.active.path
        text = f'Deck {deck_label(span.deck)}  {span.active.path.name}\n'
        text += f'{source_time:.1f}s  |  measured tempo {span.active.deck.bpm:.1f} BPM'
        if span.plan:
            progress = max(0, min(1, (frame(now)-span.start)/(span.end-span.start)))
            text += f'\nMixing to deck {deck_label("r" if span.deck == "l" else "l")}: {span.incoming.path.name}  {progress:.0%}'
        elif intent.kind == 'TRANSITION_PREPARE':
            text += f'\nPreparing deck {deck_label(intent.deck)}'
        else:
            text += '\nPlaying set automatically'
        if self.engine.errors:
            text += f'\n{len(self.engine.errors)} analysis/transition issue(s); details at exit'
        if self._muted:
            text += '\nMuted'
        else:
            text += f'\nVolume {self._volume:.0%}'
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
        elapsed = self.clock.time()
        completed = sum(a.active.path != b.active.path and elapsed >= b.start/SAMPLE_RATE
                        for a,b in zip(self.ledger.spans,self.ledger.spans[1:]))
        print(f'Set ended. Completed {completed} handoff(s).')
