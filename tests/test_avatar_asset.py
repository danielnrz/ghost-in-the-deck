"""Structural checks on the exported avatar.

These run against the committed asset; regenerate it with

    blender -b --python scripts/blender/make_avatar.py -- --out assets/avatar/ghost_test.glb
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "avatar"
GLB = ASSETS / "ghost_test.glb"
BAM = ASSETS / "ghost_test.bam"

# One bone per limb chain, plus the spine, is enough to prove the rig survived.
EXPECTED_BONES = [
    "Root", "spine_01", "spine_03", "neck_01", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l",
    "thigh_r", "calf_r", "foot_r",
]


def read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB file"
    assert version == 2, f"unexpected glTF version {version}"
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A, "first chunk is not JSON"
    return json.loads(data[20 : 20 + chunk_length].decode("utf-8"))


class TestGlbExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GLB.is_file():
            raise unittest.SkipTest(f"{GLB} not built")
        cls.gltf = read_glb_json(GLB)

    def test_mesh_exists(self):
        meshes = self.gltf.get("meshes", [])
        self.assertEqual(len(meshes), 1)
        primitives = meshes[0]["primitives"]
        self.assertTrue(primitives)
        self.assertIn("POSITION", primitives[0]["attributes"])
        self.assertIn("NORMAL", primitives[0]["attributes"])

    def test_skin_exists_with_expected_bones(self):
        skins = self.gltf.get("skins", [])
        self.assertEqual(len(skins), 1, "expected exactly one armature")
        nodes = self.gltf["nodes"]
        joint_names = {nodes[i].get("name", "") for i in skins[0]["joints"]}
        missing = [b for b in EXPECTED_BONES if b not in joint_names]
        self.assertFalse(missing, f"missing bones in GLB: {missing}")

    def test_mesh_is_skinned(self):
        primitive = self.gltf["meshes"][0]["primitives"][0]
        self.assertIn("JOINTS_0", primitive["attributes"])
        self.assertIn("WEIGHTS_0", primitive["attributes"])

    def test_no_sparse_accessors(self):
        """panda3d-gltf cannot read sparse accessors, so none may be exported."""
        sparse = [i for i, a in enumerate(self.gltf["accessors"]) if "sparse" in a]
        self.assertFalse(sparse, f"sparse accessors would break conversion: {sparse}")


class TestBamRuntimeAsset(unittest.TestCase):
    """The BAM is what Panda3D actually loads at runtime."""

    @classmethod
    def setUpClass(cls):
        if not BAM.is_file():
            raise unittest.SkipTest(f"{BAM} not built")
        import panda_env
        from ghost_in_the_deck.animation.rig import AvatarRig

        cls.base = panda_env.get_base()
        cls.rig = AvatarRig(BAM)

    def test_joints_are_present(self):
        joints = self.rig.joint_names()
        missing = [b for b in EXPECTED_BONES if b not in joints]
        self.assertFalse(missing, f"missing joints in BAM: {missing}")

    def test_avatar_has_human_proportions(self):
        low, high = self.rig.actor.getTightBounds()
        height = high.z - low.z
        self.assertTrue(1.2 < height < 2.2, f"implausible height {height:.2f} m")

    def test_controlled_joints_are_writable(self):
        """The core proof: individual bones can be driven from Python."""
        for name in self.rig.CONTROLLED:
            self.assertFalse(self.rig.joint(name).isEmpty(), f"{name} not controllable")

            self.rig.set_offset(name, heading=3.0, pitch=7.0, roll=1.0)
            heading, pitch, roll = self.rig.offset_of(name)
            self.assertAlmostEqual(pitch, 7.0, places=2, msg=name)
            self.assertAlmostEqual(heading, 3.0, places=2, msg=name)
            self.assertAlmostEqual(roll, 1.0, places=2, msg=name)

            self.rig.reset()
            for value in self.rig.offset_of(name):
                self.assertAlmostEqual(value, 0.0, places=3, msg=name)

    def test_rotating_a_bone_propagates_down_the_skeleton(self):
        """Rotating the spine must actually carry the head with it."""
        head = self.rig.expose("head")

        self.rig.reset()
        self.rig.force_update()
        rest = head.getPos(self.base.render)

        self.rig.set_offset("spine_01", pitch=25.0)
        self.rig.force_update()
        bent = head.getPos(self.base.render)

        self.assertGreater((bent - rest).length(), 0.02, "spine rotation did not move the head")

        self.rig.reset()
        self.rig.force_update()
        self.assertAlmostEqual((head.getPos(self.base.render) - rest).length(), 0.0, places=4)

    # ------------------------------------------------------- neutral stance
    def test_skeleton_sits_on_the_mesh_it_deforms(self):
        """Every joint must be inside the body part it drives.

        MPFB builds the rig in a hips-at-origin space while placing the body
        feet-on-ground, which left the whole skeleton about 0.86 m below the
        geometry. At rest that is invisible, but each bone then rotates about a
        pivot most of a metre from the joint it represents, which smears the
        mesh sideways instead of bending it. The generator corrects for it; this
        is the guard that it stays corrected.
        """
        low, high = self.rig.actor.getTightBounds()
        height = high.z - low.z
        self.assertGreater(height, 1.2)

        def fraction(joint):
            return (self.rig.expose(joint).getPos(self.base.render).z - low.z) / height

        self.rig.reset()
        self.rig.force_update()
        expected = {
            "head": (0.85, 1.00),
            "neck_01": (0.80, 0.95),
            "upperarm_l": (0.72, 0.90),
            "pelvis": (0.45, 0.62),
            "thigh_l": (0.45, 0.62),
            "foot_l": (0.00, 0.15),
        }
        for joint, (lowest, highest) in expected.items():
            place = fraction(joint)
            self.assertTrue(
                lowest <= place <= highest,
                f"{joint} sits at {place:.2f} of body height, expected {lowest}-{highest}",
            )

    def test_bind_pose_stands_with_the_arms_down(self):
        """The asset's own rest pose, with nothing applied on top."""
        import math

        from panda3d.core import Vec3

        self.rig.reset()
        self.rig.force_update()
        shoulder = self.rig.expose("upperarm_l").getPos(self.base.render)
        elbow = self.rig.expose("lowerarm_l").getPos(self.base.render)
        hand = self.rig.expose("hand_l").getPos(self.base.render)
        hip = self.rig.expose("thigh_l").getPos(self.base.render)

        upper = elbow - shoulder
        upper.normalize()
        fore = hand - elbow
        fore.normalize()

        def angle(a, b):
            return math.degrees(math.acos(max(-1.0, min(1.0, a.dot(b)))))

        down = Vec3(0, 0, -1)
        self.assertLess(angle(upper, down), 20.0, "upper arm is still abducted")
        self.assertLess(angle(fore, down), 35.0, "forearm does not hang")
        bend = angle(upper, fore)
        self.assertTrue(3.0 < bend < 40.0, f"elbow bend of {bend:.1f} deg is not relaxed")

        self.assertLess(abs(hand.x - hip.x), 0.18, "hand floats away from the thigh")
        self.assertLess(abs(hand.z - hip.z), 0.20, "hand is not near hip height")

    def test_no_runtime_neutral_correction_is_applied(self):
        """The stance is in the asset, so nothing may re-apply it at runtime."""
        from ghost_in_the_deck.animation.rig import AvatarRig as Rig

        self.assertEqual(
            Rig.NEUTRAL_POSE, {},
            "a runtime neutral pose would be applied on top of the baked one",
        )
        self.rig.reset()
        for joint in self.rig.CONTROLLED:
            for value in self.rig.offset_of(joint):
                self.assertAlmostEqual(value, 0.0, places=6, msg=joint)

    def test_animator_drives_the_rig_from_the_groove(self):
        """A beat must produce movement, and the pose must settle back."""
        from ghost_in_the_deck.animation.controller import AvatarAnimator
        from synthetic import groove_for

        animator = AvatarAnimator(self.rig, groove_for(beats=[1.0], duration=4.0))

        animator.apply_at(1.01)
        self.assertGreater(self.rig.offset_of("head")[1], 1.0, "head did not nod")

        animator.apply_at(3.0)   # two seconds later, well past the decay
        self.assertAlmostEqual(animator.state_at(3.0).pulse, 0.0, places=6)
        self.rig.reset()


if __name__ == "__main__":
    unittest.main()
