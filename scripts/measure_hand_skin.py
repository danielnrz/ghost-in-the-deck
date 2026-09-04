"""Report how far the avatar's hand mesh extends from each hand-joint centre.

    PYTHONPATH=src:tests .venv/bin/python scripts/measure_hand_skin.py

``tests/reach_clearance.py`` now measures the **skinned hand mesh itself**
against the built workstation (see its module docstring): ``build_hand_skin``
buckets every mesh vertex onto the hand joint it is skinned predominantly to
(blend weight >= 0.5) and carries it rigidly by that joint's live transform
during the sweep. This script is the diagnostic view of that bucketing - the
per-joint vertex count and skin radius the harness works from. Run it if the
avatar asset changes to sanity-check the numbers the clearance guard relies on
(the finger joints' skin radii feed the harness's pruning gate; the wrist's
much larger radius is expected and harmless).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
ASSET = ROOT / "assets" / "avatar" / "ghost_test.bam"

FINGERS = ("index", "middle", "pinky", "ring", "thumb")


def main() -> None:
    from panda3d.core import loadPrcFileData

    loadPrcFileData("", "window-type none")
    loadPrcFileData("", "audio-library-name null")
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()

    from ghost_in_the_deck.animation.rig import AvatarRig
    from reach_clearance import build_hand_skin, hand_joints

    rig = AvatarRig(ASSET, parent=base.render)
    # No truncation here: show the full skinned-vertex extent, not the farthest
    # 32 the harness keeps.
    skin = build_hand_skin(rig, base.render, per_joint=10_000)

    for name in hand_joints("l") + hand_joints("r"):
        js = skin[name]
        print(f"{name:16s} {len(js.local_pts):4d} verts   radius {js.radius * 1000:6.1f} mm")

    distal = [f"{f}_03_{s}" for f in ("index", "middle", "ring", "pinky") for s in ("l", "r")]
    print()
    print(f"max distal-joint skin radius: {max(skin[n].radius for n in distal) * 1000:.1f} mm")
    print(f"max finger-joint skin radius: "
          f"{max(skin[n].radius for n in hand_joints('l')[1:] + hand_joints('r')[1:]) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
