"""Render each DJ gesture for visual review.

    PYTHONPATH=src .venv/bin/python scripts/render_actions.py

Writes front and three-quarter views of every gesture, held at peak weight, to
out/action_review/. Also writes a neutral reference frame, a wide establishing
shot showing the whole scene, and - since hand_to_deck as of Phase 1B.2 is a
staged path rather than a single pose - five trajectory frames per side
(0%/25%/50%/75%/peak) so the approach over the tabletop can actually be judged,
not just its endpoint.

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

# kind, side, output name. deck_glance and lean_in are unchanged from Phase
# 1B and kept under their original names for direct regression comparison;
# small_hype (asymmetric since Phase 1B.1) keeps its Phase 1B.1 name.
# hand_to_deck is rendered separately below, as a trajectory, not a single pose.
CASES = (
    ("deck_glance", None, "deck_glance"),
    ("lean_in", None, "lean_in"),
    ("small_hype", "l", "small_hype_new"),
    ("small_hype", "r", "small_hype_new_r"),
)
VIEWS = {"front": 12.0, "three_quarter": 42.0}

# hand_to_deck trajectory frames: (label, progress fraction of the whole
# event). 0/25/50/75% are fractions of the attack alone (0% = neutral, 100%
# of the attack = the final target reached) - 50% lands exactly on the
# clearance pose, the attack's own midpoint (see gesture_pose.py's
# _reach_phase_weights), which is also roughly where the wrist crosses over
# the tabletop's own near edge - matching "50% / over table / transition".
# "peak" sits in the middle of the hold, at full weight.
def _trajectory_stages():
    from ghost_in_the_deck.animation.dj_behavior import ENVELOPE_SHAPE

    attack, hold, _release = ENVELOPE_SHAPE["hand_to_deck"]
    return (
        ("25", 0.25 * attack),
        ("50", 0.50 * attack),
        ("75", 0.75 * attack),
        ("peak", attack + hold / 2.0),
    )


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
    from ghost_in_the_deck.animation.dj_behavior import DJActionState, _envelope_weight
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

        for kind, side, label in CASES:
            action = DJActionState(REFERENCE_TIME, kind, 0.5, 1.0, side, 0.9)
            set_pose(action)
            path = out_dir / f"{label}_{name}.png"
            shoot(path)
            written.append(path)

        for side in ("l", "r"):
            for stage_label, progress in _trajectory_stages():
                weight = _envelope_weight("hand_to_deck", progress)
                action = DJActionState(REFERENCE_TIME, "hand_to_deck", progress, weight, side, 0.9)
                set_pose(action)
                path = out_dir / f"reach_{side}_{stage_label}_{name}.png"
                shoot(path)
                written.append(path)

    # A close, near-profile view specific to the reach trajectory: the
    # over-the-table arc is mostly forward/up motion (Y/Z), which the front
    # and three-quarter framings above mostly foreshorten away. Framed on the
    # table/hand area rather than the whole body, and only for hand_to_deck.
    #
    # Visual QA round 1 finding: at profile_distance = height * 0.9 with the
    # lookAt centred on the settled target (0.15, -0.34), the wrist left the
    # lens frustum entirely at the 25%/50% stages and sat at its very edge at
    # 75% - confirmed numerically (not just by eye) by projecting the exposed
    # ``hand_l``/``hand_r`` joint through this camera's own lens at each
    # rendered stage. The camera was calibrated before the R1 corner-avoidance
    # detour (``REACH_SWING_OUT_ROLL``/``HEADING``, gesture_pose.py) existed;
    # that detour swings the wrist out to roughly double its final target's
    # lateral offset before it crosses the table, which the old framing never
    # accounted for. Pulling back (0.9 -> 1.3x height) and re-centring the
    # look point on the swing's own midpoint rather than the endpoint keeps
    # the wrist within the frustum across the whole event (worst case ~53% of
    # the half-width, sampled every 1% of progress on both sides) while still
    # keeping the tabletop and control target in frame. This is a capture-only
    # change: it does not touch the reach pose, timing or clearance geometry.
    profile_angle = math.radians(80.0)
    profile_distance = height * 1.3
    for side in ("l", "r"):
        sign = 1.0 if side == "l" else -1.0
        base.camera.setPos(
            sign * math.sin(profile_angle) * profile_distance,
            -math.cos(profile_angle) * profile_distance,
            height * 0.62,
        )
        base.camera.lookAt(sign * 0.5, -0.18, height * 0.62)
        for stage_label, progress in _trajectory_stages():
            weight = _envelope_weight("hand_to_deck", progress)
            action = DJActionState(REFERENCE_TIME, "hand_to_deck", progress, weight, side, 0.9)
            set_pose(action)
            path = out_dir / f"reach_{side}_{stage_label}_profile.png"
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
