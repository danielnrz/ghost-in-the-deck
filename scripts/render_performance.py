"""Deterministic pose contact sheets; use render_set.py for real ledger review."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ghost_in_the_deck.app import build_app
from ghost_in_the_deck.demo import create_demo
from ghost_in_the_deck.animation.dj_behavior import DJActionState
from ghost_in_the_deck.dj_app import write_set_pose
from ghost_in_the_deck.animation.dj_behavior import _envelope_weight
from ghost_in_the_deck.animation.performance_pose import hype_lift


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'out/performance_review')
    options=parser.parse_args();out=options.out_dir;out.mkdir(parents=True,exist_ok=True)
    args=argparse.Namespace(headless=True,no_audio=True,no_actions=False,no_fx=False,
        show_action=None,single_track=False,refresh=False,
        music_dir=str(create_demo(out/'music')),cache_dir=str(out/'cache'),
        transition_bars=2,dwell=5,track=None,capture_at=[],stall_every=0,
        simulate_stall=0,seconds=None)
    app=build_app(args)
    try:
        track=app.ledger.spans[0].active
        for side in ('l','r'):
            for variant in ('knob','button','platter','cheer'):
                for i,p in enumerate((0,.10,.20,.30,.40,.50,.50,.50,.60,.70,.80,.90,1)):
                    contact_phase={5:0,6:.5,7:1}.get(i,0)
                    kind='small_hype' if variant=='cheer' else 'hand_to_deck'
                    action=DJActionState(10,kind,p,_envelope_weight(kind,p),side,1,variant,contact_phase)
                    app.rig.actor.setZ(hype_lift(action))
                    write_set_pose(app.animator,app._groove_state(track,10+i/10),action)
                    app.rig.force_update()
                    app.base.graphicsEngine.renderFrame();app.base.graphicsEngine.renderFrame()
                    app.base.win.saveScreenshot(str(out/f'{side}-{variant}-{i:02}.png'))
    finally:
        app._finish();app.base.destroy()


if __name__=='__main__':
    main()
