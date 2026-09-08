"""Render actual set-ledger transition poses using synthesized local music."""
from __future__ import annotations

import argparse
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
            for i, span in enumerate(spans):
                for phase in (.05, .3, .55, .85, 1.01):
                    now = (span.start + (span.end-span.start)*phase)/44100
                    app.draw_at(now)
                    app.rig.force_update()
                    app.base.graphicsEngine.renderFrame()
                    app.base.graphicsEngine.renderFrame()
                    path = options.out_dir/f'transition-{i+1}-{phase:.2f}.png'
                    app.base.win.saveScreenshot(str(path))
                    print(path)
        finally:
            app._finish()
            app.base.destroy()


if __name__ == '__main__':
    main()
