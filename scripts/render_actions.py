"""Render each DJ gesture for visual review.

    PYTHONPATH=src .venv/bin/python scripts/render_actions.py

Writes front and three-quarter views of every gesture, held at peak weight, to
out/action_review/. Also writes a neutral reference frame and a wide
establishing shot showing the whole scene, since a gesture only means anything
next to what it changed from.

Joint numbers proved a poor guide to whether the neutral stance looked human in
an earlier phase of this project; this exists for the same reason bone angles
were not trusted then - a picture is the only real check.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "action_review"

GESTURES = ("deck_glance", "lean_in", "hand_to_deck", "small_hype")
VIEWS = {"front": 12.0, "three_quarter": 42.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--size", type=int, default=520)
    args = parser.parse_args()

    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", f"win-size {args.size} {int(args.size * 1.05)}")
    loadPrcFileData("", "audio-library-name null")

    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AmbientLight, CardMaker, DirectionalLight, PNMImage, Vec4

    base = ShowBase()
    base.setBackgroundColor(0.11, 0.11, 0.14)

    key = DirectionalLight("key")
    key.setColor(Vec4(1.05, 1.0, 0.95, 1))
    key_np = base.render.attachNewNode(key)
    key_np.setHpr(-32, -22, 0)
    base.render.setLight(key_np)

    fill = DirectionalLight("fill")
    fill.setColor(Vec4(0.30, 0.38, 0.60, 1))
    fill_np = base.render.attachNewNode(fill)
    fill_np.setHpr(145, -12, 0)
    base.render.setLight(fill_np)

    ambient = AmbientLight("ambient")
    ambient.setColor(Vec4(0.32, 0.32, 0.38, 1))
    base.render.setLight(base.render.attachNewNode(ambient))

    card = CardMaker("floor")
    card.setFrame(-6, 6, -6, 6)
    floor = base.render.attachNewNode(card.generate())
    floor.setP(-90)
    floor.setColor(0.16, 0.16, 0.20, 1)

    from ghost_in_the_deck.animation.controller import AvatarAnimator
    from ghost_in_the_deck.animation.dj_behavior import DJActionState
    from ghost_in_the_deck.animation.groove import GrooveEngine
    from ghost_in_the_deck.animation.rig import AvatarRig
    from ghost_in_the_deck.animation.workstation import DEFAULT_TARGETS
    from ghost_in_the_deck.audio.features import MusicFeatures
    from ghost_in_the_deck.scene.workstation import build_workstation

    rig = AvatarRig(ROOT / "assets" / "avatar" / "ghost_test.bam", parent=base.render)
    build_workstation(base.render)

    beats = [0.5 + i * 0.5 for i in range(48)]
    step = 0.05
    frames = [i * step for i in range(int(24.0 / step))]
    ones = [1.0] * len(frames)
    features = MusicFeatures(
        track="action-review", duration_seconds=24.0, sample_rate=22050,
        hop_length=512, bpm=120.0, beats=beats, frame_times=frames,
        onset_strength=ones, rms=ones, bass_energy=ones, mid_energy=ones,
        high_energy=ones,
    )
    groove = GrooveEngine(features, seed="action-review")
    animator = AvatarAnimator(rig, groove, targets=DEFAULT_TARGETS)

    low, high = rig.actor.getTightBounds()
    height = high.z - low.z
    base.disableMouse()

    def shoot(path: Path) -> None:
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame()
        image = PNMImage()
        base.win.getScreenshot(image)
        image.write(str(path))

    # Chosen at a bar boundary, where the groove's own sway and weight shift
    # pass through zero: showing a gesture at the groove's own rotational
    # extreme would confound the two, making a fine gesture look exaggerated
    # for reasons that have nothing to do with the gesture itself.
    REFERENCE_TIME = 6.5

    def set_pose(action: DJActionState | None) -> None:
        state = groove.state_at(REFERENCE_TIME)
        animator._write_pose(state, action)
        rig.force_update()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # A closer, upper-body 3/4 framing - the wide establishing shot the app
    # itself uses is not close enough to judge a gesture's own detail.
    for name, angle_deg in VIEWS.items():
        angle = math.radians(angle_deg)
        distance = height * 1.35
        base.camera.setPos(math.sin(angle) * distance, -math.cos(angle) * distance, height * 0.82)
        base.camera.lookAt(0.0, -0.25, height * 0.66)

        set_pose(None)
        path = out_dir / f"neutral_{name}.png"
        shoot(path)
        written.append(path)

        for kind in GESTURES:
            side = "l" if kind == "hand_to_deck" else None
            action = DJActionState(REFERENCE_TIME, kind, 0.5, 1.0, side, 0.9)
            set_pose(action)
            path = out_dir / f"{kind}_{name}.png"
            shoot(path)
            written.append(path)

        if "hand_to_deck" in GESTURES:
            action = DJActionState(REFERENCE_TIME, "hand_to_deck", 0.5, 1.0, "r", 0.9)
            set_pose(action)
            path = out_dir / f"hand_to_deck_r_{name}.png"
            shoot(path)
            written.append(path)

    # One wide establishing shot, matching the app's own default framing, so
    # the workstation's placement can be judged in context.
    wide_angle = math.radians(28.0)
    wide_distance = height * 2.3
    base.camera.setPos(math.sin(wide_angle) * wide_distance, -math.cos(wide_angle) * wide_distance, height * 0.475)
    base.camera.lookAt(0.0, -0.25, height * 0.425)
    set_pose(None)
    path = out_dir / "establishing_wide.png"
    shoot(path)
    written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
