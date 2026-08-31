"""Generate the Phase 0 test avatar with MPFB and export it as GLB.

Run headless:

    blender -b --python scripts/blender/make_avatar.py -- --out assets/avatar/ghost_test.glb

MPFB (the MakeHuman plugin for Blender) is free software from the Blender
extensions platform. Its operators take no arguments; they read scene
properties, so the settings are applied to the scene first.

The 'game_engine' skeleton is used rather than Rigify: it is a plain deform-only
humanoid hierarchy, which is what a runtime engine can actually consume. A
Rigify control rig exports a large amount of animation machinery that Panda3D
has no use for.
"""

import argparse
import sys
from pathlib import Path

import bpy

# Bones the runtime relies on; export fails loudly if MPFB ever renames them.
REQUIRED_BONES = [
    "Root", "spine_01", "spine_02", "spine_03", "neck_01", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l",
    "thigh_r", "calf_r", "foot_r",
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="destination .glb path")
    parser.add_argument("--blend", help="optionally also save the .blend source")
    parser.add_argument("--rig", default="game_engine", help="MPFB standard rig name")
    return parser.parse_args(argv)


def clear_scene() -> None:
    """Empty the default startup scene.

    read_factory_settings() would also unload MPFB, so the objects are removed
    directly instead.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def configure_human(scene) -> None:
    """MPFB reads these scene properties when create_human() runs."""
    scene.MPFB_NH_scale_factor = "METER"          # Panda3D works in metres
    scene.MPFB_NH_phenotype_gender = "neutral"
    scene.MPFB_NH_phenotype_age = "young"
    scene.MPFB_NH_phenotype_muscle = "averagemuscle"
    scene.MPFB_NH_phenotype_weight = "averageweight"
    scene.MPFB_NH_phenotype_height = "average"
    scene.MPFB_NH_phenotype_proportions = "average"
    scene.MPFB_NH_phenotype_race = "universal"
    scene.MPFB_NH_add_phenotype = True
    scene.MPFB_NH_phenotype_influence = 1.0

    # Helper geometry, clothes and extra groups are authoring aids that would
    # only add stray meshes and vertex groups to the exported file.
    scene.MPFB_NH_detailed_helpers = False
    scene.MPFB_NH_mask_helpers = False
    scene.MPFB_NH_load_clothes = False
    scene.MPFB_NH_add_breast = False
    scene.MPFB_NH_extra_vertex_groups = False
    scene.MPFB_NH_auto_generate_rigify = False

    # The game-engine material is a plain principled shader, which is what glTF
    # can actually represent; the enhanced skins are node groups that cannot be
    # exported faithfully.
    scene.MPFB_NH_override_skin_model = "GAMEENGINE"
    scene.MPFB_NH_override_eyes_model = "NONE"
    scene.MPFB_NH_override_clothes_model = "NONE"


def configure_rig(scene, rig_name: str) -> None:
    scene.MPFB_ADR_standard_rig = rig_name
    scene.MPFB_ADR_import_weights = True
    scene.MPFB_ADR_auto_generate = False


def find_objects():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    return meshes, armatures


def select_only(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def build(rig_name: str):
    scene = bpy.context.scene

    configure_human(scene)
    bpy.ops.mpfb.create_human()

    meshes, _ = find_objects()
    if not meshes:
        raise RuntimeError("MPFB did not create a mesh")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    body.name = "GhostBody"
    select_only(body)

    configure_rig(scene, rig_name)
    bpy.ops.mpfb.add_standard_rig()

    meshes, armatures = find_objects()
    if not armatures:
        raise RuntimeError("MPFB did not create an armature")
    armature = armatures[0]
    armature.name = "GhostRig"
    return body, armature


def verify(body, armature) -> list[str]:
    problems = []
    if len(body.data.vertices) < 1000:
        problems.append(f"body mesh looks too small: {len(body.data.vertices)} verts")

    bone_names = set(armature.data.bones.keys())
    missing = [b for b in REQUIRED_BONES if b not in bone_names]
    if missing:
        problems.append(f"missing expected bones: {missing}")

    if not any(m.type == "ARMATURE" and m.object == armature for m in body.modifiers):
        problems.append("body has no armature modifier bound to the rig")

    weighted = {g.name for g in body.vertex_groups} & bone_names
    if len(weighted) < 10:
        problems.append(f"only {len(weighted)} bones have vertex groups")

    return problems


def export_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_skins=True,
        export_yup=True,
        # MPFB's basemesh carries the phenotype targets as shape keys. Blender
        # writes those as sparse morph-target accessors, which panda3d-gltf
        # cannot read. Phase 0 animates bones only, so they are dropped.
        export_morph=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
    )


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()

    clear_scene()
    body, armature = build(args.rig)

    problems = verify(body, armature)
    for problem in problems:
        print(f"[VERIFY-FAIL] {problem}")
    if problems:
        raise SystemExit(1)

    print(f"[OK] mesh   : {body.name} ({len(body.data.vertices)} verts)")
    print(f"[OK] armature: {armature.name} ({len(armature.data.bones)} bones)")

    if args.blend:
        blend = Path(args.blend).resolve()
        blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        print(f"[OK] blend  : {blend}")

    export_glb(out)
    print(f"[OK] glb    : {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
