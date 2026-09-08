"""Render actual set-ledger transition poses using synthesized local music."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ghost_in_the_deck.app import build_app
from ghost_in_the_deck.demo import create_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'out'/'set_review')
    options = parser.parse_args()
    options.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ghost-render-') as temp:
        args = argparse.Namespace(headless=True, no_audio=True, no_actions=False,
            no_fx=False, show_action=None, single_track=False, refresh=False,
            music_dir=str(create_demo(Path(temp)/'music')), cache_dir=str(Path(temp)/'cache'),
            transition_bars=2, dwell=5, track=None, capture_at=[], stall_every=0,
            simulate_stall=0, seconds=None)
        app = build_app(args)
        try:
            while not app.stream_done:
                app._pump(app.ledger.frames/44100)
                time.sleep(.01)
            if app.producer.error:
                raise RuntimeError(app.producer.error)
            spans = [s for s in app.ledger.spans if s.plan]
            if len(spans) < 2:
                raise RuntimeError('demo did not produce two matched transitions')
            # Real analyzed demo energy chooses low/high solo examples. Then
            # render whole approach/action/recovery sequences in both directions.
            app.visual.update(app.ledger.spans)
            samples = [t/4 for t in range(16, int(spans[0].start/44100-4)*4)]
            solo = app.ledger.spans[0]
            low = min(samples, key=lambda t: app._groove_state(solo.active,t).intensity)
            high = max(samples, key=lambda t: app._groove_state(solo.active,t).intensity)
            scenarios = {'solo-low': [low+i/6 for i in range(7)],
                         'solo-high': [high+i/6 for i in range(7)]}
            for i, span in enumerate(spans):
                start, end = span.start/44100, span.end/44100
                scenarios[f'transition-{i+1}-reach'] = [start-2.4+j/6 for j in range(31)]
                scenarios[f'transition-{i+1}-middle'] = [(start+end)/2+j/6 for j in range(4)]
                scenarios[f'transition-{i+1}-handoff'] = [end-.6+j/6 for j in range(22)]
                scenarios[f'transition-{i+1}-boundary'] = [end-1/60,end,end+1/60]
            for i, intent in enumerate(app.visual.interactions):
                if intent.operation != 'crossfade':
                    scenarios[f'effect-{i+1}'] = [intent.begin+j/6 for j in range(int((intent.end-intent.begin)*6)+1)]
            manifest = []
            for name, times in scenarios.items():
                for index, now in enumerate(times):
                    app.draw_at(now)
                    app.rig.force_update()
                    app.base.graphicsEngine.renderFrame()
                    app.base.graphicsEngine.renderFrame()
                    path = options.out_dir/f'{name}-{index:03}.png'
                    app.base.win.saveScreenshot(str(path))
                    _, intent, action = app.pose_at(now)
                    manifest.append(dict(file=path.name,time=now,scenario=name,
                        intent=intent.kind,operation=intent.operation,deck=intent.deck,
                        phase=intent.phase_at(now) if action else 'monitor/groove'))
            (options.out_dir/'frames.json').write_text(json.dumps(manifest,indent=2))
            # Compact replayable behavior record spanning the entire actual set.
            timeline=[]
            for tick in range(int(app.ledger.frames/44100*10)):
                now=tick/10; span=app.ledger.at(now)
                intent=app.visual.at(now,span.deck); action=intent.action_at(now)
                timeline.append(dict(time=now,audio='mix' if span.plan else 'solo',
                    active_deck=span.deck,operation=intent.operation,intent=intent.kind,
                    hand=action.side if action else None,
                    phase=intent.phase_at(now) if action else 'monitor/groove'))
            (options.out_dir/'timeline.json').write_text(json.dumps(timeline,indent=2))
            print(f'Rendered {len(manifest)} frames; {len(timeline)} timeline samples')
        finally:
            app._finish()
            app.base.destroy()


if __name__ == '__main__':
    main()
