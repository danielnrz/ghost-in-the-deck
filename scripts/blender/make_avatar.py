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
from statistics import median

from mathutils import Vector

# Where each bone should point in the neutral standing stance, as a direction in
# armature space: +X is the character's left, -Y is forward, +Z is up.
#
# Aiming a direction rather than setting Euler angles matters here. The MPFB
# skeleton has non-zero, non-obvious rest rotations whose local axes do not line
# up with anything intuitive, and a large rotation about one of them sweeps a
# cone rather than swinging the limb down. A target direction says what the pose
# should look like instead of how to get there.
#
# Parents are aimed before children, so re-aiming the clavicle does not disturb
# the arm hanging from it.
NEUTRAL_AIM = [
    ("clavicle_l", Vector((0.147, 0.000, -0.048))),
    ("clavicle_r", Vector((-0.147, 0.000, -0.048))),
    ("upperarm_l", Vector((0.130, -0.050, -1.000))),
    ("upperarm_r", Vector((-0.130, -0.050, -1.000))),
    ("lowerarm_l", Vector((0.090, -0.300, -1.000))),
    ("lowerarm_r", Vector((-0.090, -0.300, -1.000))),
]

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
    parser.add_argument(
        "--no-neutral-pose",
        action="store_true",
        help="export the raw MPFB A-pose instead of the neutral standing bind pose",
    )
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
    scene.MPFB_NH_mask_helpers = True
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


def ensure_material(body) -> None:
    """Give the body a plain shader if MPFB left it unmaterialised.

    The enhanced skins depend on asset packs that are not part of the add-on, so
    a headless run can end up with no material at all, which glTF exports as an
    untextured primitive.
    """
    if body.data.materials and body.data.materials[0] is not None:
        return

    material = bpy.data.materials.new("GhostSkin")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.62, 0.48, 0.42, 1.0)
        principled.inputs["Roughness"].default_value = 0.65
        principled.inputs["Metallic"].default_value = 0.0
    body.data.materials.append(material)


def measure_rig_offset(body, armature) -> Vector:
    """How far the skeleton sits from the mesh it deforms.

    MPFB builds the game-engine rig in a space whose origin is at the hips while
    placing the body with its feet on the ground, so the whole skeleton ends up
    roughly 0.86 m below the geometry it drives. At rest that is invisible - the
    deformation is the identity - but every bone then rotates about a pivot most
    of a metre away from the joint it represents, which smears the mesh sideways
    instead of bending it.

    The offset is measured rather than hard-coded: each deform bone is compared
    against the centre of the vertices weighted to it, and the median across all
    of them is taken. A median because a few groups (the skull especially) sit
    well off their bone's midpoint, and because this should keep working if MPFB
    changes its numbers.
    """
    group_names = {group.index: group.name for group in body.vertex_groups}
    weighted: dict[str, list[Vector]] = {}
    for vertex in body.data.vertices:
        for group in vertex.groups:
            if group.weight < 0.5:
                continue
            name = group_names.get(group.group)
            if name:
                weighted.setdefault(name, []).append(vertex.co)

    samples = []
    for bone in armature.data.bones:
        points = weighted.get(bone.name)
        if not points or len(points) < 25:
            continue
        centroid = sum(points, Vector()) / len(points)
        samples.append(centroid - (bone.head_local + bone.tail_local) / 2.0)

    if len(samples) < 8:
        raise RuntimeError(f"too few weighted bones to measure rig offset: {len(samples)}")

    return Vector(tuple(median([s[axis] for s in samples]) for axis in range(3)))


def align_rig_to_mesh(body, armature) -> Vector:
    """Move the skeleton onto the body so bones rotate about the right pivots.

    The bones move, not the mesh: the body already stands with its feet on the
    floor and should stay there. Translating every bone by the same amount
    leaves the rest deformation the identity, so the mesh does not shift - only
    the pivots become correct.
    """
    offset = measure_rig_offset(body, armature)

    select_only(armature)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone in armature.data.edit_bones:
        bone.head = bone.head + offset
        bone.tail = bone.tail + offset
    bpy.ops.object.mode_set(mode="OBJECT")
    return offset


def aim_bone(armature, name: str, target: Vector) -> None:
    """Rotate one pose bone so it points along ``target`` in armature space.

    The minimal rotation is used, which keeps whatever twist the bone already
    had instead of introducing a new one.
    """
    pose_bone = armature.pose.bones.get(name)
    if pose_bone is None:
        raise RuntimeError(f"no bone named {name}")

    matrix = pose_bone.matrix.copy()
    current = (matrix.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
    rotation = current.rotation_difference(target.normalized()).to_matrix().to_4x4()

    head = matrix.translation.copy()
    matrix.translation = Vector((0.0, 0.0, 0.0))
    posed = rotation @ matrix
    posed.translation = head
    pose_bone.matrix = posed

    # Children read their parent's matrix, so it has to be committed now.
    bpy.context.view_layer.update()


def pose_neutral(armature) -> None:
    """Put the skeleton into the relaxed standing stance."""
    select_only(armature)
    bpy.ops.object.mode_set(mode="POSE")
    for name, target in NEUTRAL_AIM:
        aim_bone(armature, name, target)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_rest_pose(body, armature) -> None:
    """Make the current pose the skeleton's rest pose, keeping the skinning.

    Applying the pose to the armature alone would leave the mesh bound to the
    old rest pose and tear it apart. The standard fix is to bake the current
    deformation into the mesh first: duplicate the armature modifier, apply the
    first copy so the vertices move to where they are being drawn, and only then
    make the pose the rest pose. The surviving duplicate re-binds the baked mesh
    to the new rest pose, so at rest it deforms by nothing at all.

    Shape keys have to go first because a modifier cannot be applied to a mesh
    that has them. Nothing is lost: morph targets are already excluded from the
    export, since panda3d-gltf cannot read the sparse accessors Blender writes
    them as.
    """
    select_only(body)
    if body.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True)

    modifier = next((m for m in body.modifiers if m.type == "ARMATURE"), None)
    if modifier is None:
        raise RuntimeError("body has no armature modifier to rebind")

    bpy.ops.object.modifier_copy(modifier=modifier.name)
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    select_only(armature)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")


def find_objects():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    return meshes, armatures


def select_only(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def build(rig_name: str, neutral: bool = True):
    scene = bpy.context.scene

    configure_human(scene)
    bpy.ops.mpfb.create_human()

    meshes, _ = find_objects()
    if not meshes:
        raise RuntimeError("MPFB did not create a mesh")
    body = max(meshes, key=lambda o: len(o.data.vertices))
    body.name = "GhostBody"
    select_only(body)

    # The MakeHuman basemesh ships with helper geometry (a skirt-like proxy for
    # clothes, hair and eye guides). Blender only hides it behind a mask
    # modifier, so it would still reach the exported mesh. Remove it outright.
    bpy.ops.mpfb.delete_helpers()
    ensure_material(body)

    configure_rig(scene, rig_name)
    bpy.ops.mpfb.add_standard_rig()

    meshes, armatures = find_objects()
    if not armatures:
        raise RuntimeError("MPFB did not create an armature")
    armature = armatures[0]
    armature.name = "GhostRig"

    offset = align_rig_to_mesh(body, armature)
    print(f"[OK] rig aligned to mesh by ({offset.x:+.3f}, {offset.y:+.3f}, {offset.z:+.3f})")

    if neutral:
        pose_neutral(armature)
        bake_rest_pose(body, armature)

    return body, armature


def verify(body, armature) -> list[str]:
    problems = []
    if len(body.data.vertices) < 1000:
        problems.append(f"body mesh looks too small: {len(body.data.vertices)} verts")
    if len(body.data.vertices) > 15000:
        problems.append(
            f"body still carries helper geometry: {len(body.data.vertices)} verts"
        )
    if not body.data.materials:
        problems.append("body has no material")

    bone_names = set(armature.data.bones.keys())
    missing = [b for b in REQUIRED_BONES if b not in bone_names]
    if missing:
        problems.append(f"missing expected bones: {missing}")

    if not any(m.type == "ARMATURE" and m.object == armature for m in body.modifiers):
        problems.append("body has no armature modifier bound to the rig")

    weighted = {g.name for g in body.vertex_groups} & bone_names
    if len(weighted) < 10:
        problems.append(f"only {len(weighted)} bones have vertex groups")

    # The rig must sit on the body. If it does not, every rotation pivots in the
    # wrong place and the mesh smears rather than bends.
    residual = measure_rig_offset(body, armature)
    if residual.length > 0.06:
        problems.append(
            f"skeleton is {residual.length:.3f} m away from the mesh it deforms "
            f"({residual.x:+.3f}, {residual.y:+.3f}, {residual.z:+.3f})"
        )

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
    body, armature = build(args.rig, neutral=not args.no_neutral_pose)

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
