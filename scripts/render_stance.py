"""Render the avatar's neutral stance for visual review.

    PYTHONPATH=src .venv/bin/python scripts/render_stance.py

Writes front, side and three-quarter views to out/stance_review/. Bone angles
alone proved to be a poor guide to whether a pose looks human, so this exists to
make the actual silhouette easy to check.

    --pose neutral   the rig's neutral stance (default)
    --pose bind      the asset's own bind pose, with no runtime correction
    --groove T       the full groove pose at playback time T
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "stance_review"

VIEWS = {"front": 0.0, "three_quarter": 38.0, "side": 90.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", default="neutral", choices=("neutral", "bind", "groove"))
    parser.add_argument("--groove-time", type=float, default=4.0)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--prefix", default="")
    parser.add_argument("--size", type=int, default=520)
    parser.add_argument("--model", default=None, help="override the .bam path")
    args = parser.parse_args()

    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", f"win-size {args.size} {int(args.size * 1.55)}")
    loadPrcFileData("", "audio-library-name null")

    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import AmbientLight, CardMaker, DirectionalLight, PNMImage, Vec4

    base = ShowBase()
    base.setBackgroundColor(0.11, 0.11, 0.14)

    key = DirectionalLight("key")
    key.setColor(Vec4(1.05, 1.0, 0.95, 1))
    key_np = base.render.attachNewNode(key)
    key_np.setHpr(-32, -20, 0)
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

    from ghost_in_the_deck.animation.rig import AvatarRig

    model = args.model or (ROOT / "assets" / "avatar" / "ghost_test.bam")
    rig = AvatarRig(model, parent=base.render)

    if args.pose == "bind":
        for joint in rig.CONTROLLED:
            heading, pitch, roll = rig.NEUTRAL_POSE.get(joint, (0.0, 0.0, 0.0))
            rest_h, rest_p, rest_r = rig.rest_hpr(joint)
            rig.joint(joint).setHpr(rest_h - heading, rest_p - pitch, rest_r - roll)
    elif args.pose == "groove":
        from ghost_in_the_deck.animation.controller import AvatarAnimator
        from ghost_in_the_deck.animation.groove import GrooveEngine
        from ghost_in_the_deck.audio.features import MusicFeatures

        beats = [0.5 + i * 0.5 for i in range(48)]
        step = 0.05
        frames = [i * step for i in range(int(30.0 / step))]
        ones = [1.0] * len(frames)
        features = MusicFeatures(
            track="stance-review", duration_seconds=30.0, sample_rate=22050,
            hop_length=512, bpm=120.0, beats=beats, frame_times=frames,
            onset_strength=ones, rms=ones, bass_energy=ones,
            mid_energy=ones, high_energy=ones,
        )
        AvatarAnimator(rig, GrooveEngine(features, seed="review")).apply_at(args.groove_time)
    rig.force_update()

    low, high = rig.actor.getTightBounds()
    height = high.z - low.z
    centre = low.z + height * 0.5
    base.disableMouse()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, angle in VIEWS.items():
        radians = math.radians(angle)
        base.camera.setPos(
            math.sin(radians) * height * 2.2, -math.cos(radians) * height * 2.2, centre
        )
        base.camera.lookAt(0, 0, centre)
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame()
        image = PNMImage()
        base.win.getScreenshot(image)
        path = out_dir / f"{args.prefix}{name}.png"
        image.write(str(path))
        written.append(path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
