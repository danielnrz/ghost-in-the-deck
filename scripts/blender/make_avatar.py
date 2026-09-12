"""Generate the rig-compatible Ghost in the Deck performer with MPFB.

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
import math
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
        "--no-outfit",
        action="store_true",
        help="export the body/rig only (asset-pipeline diagnostics)",
    )
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


def material(name, color, *, roughness=0.65, metallic=0.0):
    """Make one deliberately simple, glTF-safe performer material."""
    old = bpy.data.materials.get(name)
    material = old or bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic
    return material


def ensure_material(body) -> None:
    """Use a controlled skin tone instead of MPFB's pink preview material."""
    skin = material("GhostSkin", (0.42, 0.22, 0.14, 1.0), roughness=0.72)
    body.data.materials.clear()
    body.data.materials.append(skin)


def _dominant_groups(obj, vertex) -> set[str]:
    indexes = {group.index: group.name for group in obj.vertex_groups}
    return {indexes[item.group] for item in vertex.groups
            if item.weight >= 0.16 and item.group in indexes}


def surface_garment(body, name, keep, garment_material, *, thickness=.006, ease=.018):
    """Cut a fitted skinned garment from the baked body surface.

    This intentionally reuses the body's topology, armature modifier and
    weights.  Clothes therefore follow the exact runtime skeleton without a
    second retargeting system.  A small radial ease and outward solidify layer
    make the result read as fabric instead of recoloured skin.
    """
    garment = body.copy()
    garment.data = body.data.copy()
    garment.name = name
    garment.data.name = f"{name}Mesh"
    bpy.context.collection.objects.link(garment)
    garment.data.materials.clear()
    garment.data.materials.append(garment_material)

    for vertex in garment.data.vertices:
        groups = _dominant_groups(garment, vertex)
        vertex.select = not keep(vertex.co, groups)
        if not vertex.select:
            vertex.co.x *= 1.0 + ease
            # Expand around the body's centreline in depth, not world zero.
            vertex.co.y = -0.035 + (vertex.co.y + 0.035) * (1.0 + ease)

    select_only(garment)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for polygon in garment.data.polygons:
        polygon.use_smooth = True
    solidify = garment.modifiers.new("Fabric thickness", "SOLIDIFY")
    solidify.thickness = thickness
    solidify.offset = 1.0
    # Skin first, then add thickness around the final animated surface.
    bpy.ops.object.modifier_move_to_index(modifier=solidify.name,
                                          index=len(garment.modifiers)-1)
    return garment


def _bind_to_bone(obj, armature, bone_name):
    """Skin a rigid accessory to one deform bone for portable glTF export."""
    obj.parent = armature
    group = obj.vertex_groups.new(name=bone_name)
    group.add(range(len(obj.data.vertices)), 1.0, "REPLACE")
    modifier = obj.modifiers.new("GhostRig", "ARMATURE")
    modifier.object = armature
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _primitive(name, create, armature, bone, mat):
    create()
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}Mesh"
    select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.materials.append(mat)
    return _bind_to_bone(obj, armature, bone)


def create_outfit(body, armature):
    """Build a compact, redistributable DJ identity on the existing rig."""
    ink = material("MidnightFabric", (0.018, 0.024, 0.038, 1), roughness=.82)
    graphite = material("GraphiteFabric", (0.055, 0.065, 0.085, 1), roughness=.78)
    teal = material("GhostTeal", (0.025, 0.72, 0.73, 1), roughness=.38, metallic=.12)
    shoe_canvas = material("WarmSneakerCanvas", (0.38, 0.32, 0.28, 1), roughness=.76)
    metal = material("HeadphoneMetal", (0.08, 0.10, 0.14, 1), roughness=.30, metallic=.72)

    torso = {"pelvis", "spine_01", "spine_02", "spine_03",
             "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r"}
    legs = {"pelvis", "thigh_l", "thigh_r", "calf_l", "calf_r"}
    shirt = surface_garment(
        body, "OversizedDJTop",
        lambda co, groups: .88 <= co.z <= 1.46 and bool(groups & torso)
            and not (("upperarm_l" in groups or "upperarm_r" in groups) and co.z < 1.12),
        graphite, thickness=.008, ease=.035,
    )
    pants = surface_garment(
        body, "TaperedDJTrousers",
        lambda co, groups: .105 <= co.z <= .98 and bool(groups & legs),
        ink, thickness=.007, ease=.022,
    )
    for side, x in (("L", .221), ("R", -.221)):
        bone = f"foot_{side.lower()}"
        upper = _primitive(f"SneakerUpper{side}", lambda x=x:
            bpy.ops.mesh.primitive_cube_add(location=(x, -.070, .078),
                                             scale=(.064, .125, .060)),
            armature, bone, shoe_canvas)
        # Slope and narrow the toe so the high-top reads as footwear rather
        # than a cube around an anatomical foot.
        for vertex in upper.data.vertices:
            if vertex.co.y < -.070:
                vertex.co.x = x + (vertex.co.x - x) * .82
                if vertex.co.z > .078:
                    vertex.co.z = .092
        bevel = upper.modifiers.new("Rounded sneaker upper", "BEVEL")
        bevel.width = .022
        bevel.segments = 3
        outsole = _primitive(f"SneakerSole{side}", lambda x=x:
            bpy.ops.mesh.primitive_cube_add(location=(x, -.075, .018),
                                             scale=(.068, .132, .018)),
            armature, bone, teal)
        bevel = outsole.modifiers.new("Rounded outsole", "BEVEL")
        bevel.width = .010
        bevel.segments = 2

    # A cyan chest mark gives the dark silhouette a readable focal point.
    _primitive("ChestSignal", lambda: bpy.ops.mesh.primitive_cube_add(
        location=(0, -.184, 1.255), scale=(.065, .006, .015)),
        armature, "spine_03", teal)

    # A low-profile cap completes the silhouette without requiring hair assets.
    _primitive("DJCap", lambda: bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24, ring_count=12, location=(0, -.030, 1.625),
        scale=(.106, .112, .080)), armature, "head", ink)
    brim = _primitive("DJCapBrim", lambda: bpy.ops.mesh.primitive_cube_add(
        location=(0, -.125, 1.625), scale=(.072, .038, .009)),
        armature, "head", graphite)
    bevel = brim.modifiers.new("Rounded cap brim", "BEVEL")
    bevel.width = .012
    bevel.segments = 3

    # Headphones: a segmented arch and two padded cups, all bound to the head.
    for index in range(13):
        angle = math.radians(25 + index * 130 / 12)
        x = .112 * math.cos(angle)
        z = 1.575 + .112 * math.sin(angle)
        _primitive(f"Headband{index:02}", lambda x=x, z=z, angle=angle:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=6,
                location=(x, -.015, z), scale=(.014, .018, .014)),
            armature, "head", metal)
    for side in (-1, 1):
        _primitive(f"HeadphoneCup{'L' if side > 0 else 'R'}", lambda side=side:
            bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=.052, depth=.025,
                location=(side*.105, -.012, 1.575), rotation=(0, math.pi/2, 0)),
            armature, "head", ink)
        _primitive(f"HeadphoneRing{'L' if side > 0 else 'R'}", lambda side=side:
            bpy.ops.mesh.primitive_torus_add(major_radius=.038, minor_radius=.006,
                major_segments=20, minor_segments=6, location=(side*.119, -.012, 1.575),
                rotation=(0, math.pi/2, 0)), armature, "head", teal)

    return [shirt, pants]
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
    outfit = [] if args.no_outfit else create_outfit(body, armature)

    problems = verify(body, armature)
    for problem in problems:
        print(f"[VERIFY-FAIL] {problem}")
    if problems:
        raise SystemExit(1)

    print(f"[OK] mesh   : {body.name} ({len(body.data.vertices)} verts)")
    print(f"[OK] armature: {armature.name} ({len(armature.data.bones)} bones)")
    print(f"[OK] outfit : {len(outfit)} skinned garment meshes plus DJ accessories")

    if args.blend:
        blend = Path(args.blend).resolve()
        blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        print(f"[OK] blend  : {blend}")

    export_glb(out)
    print(f"[OK] glb    : {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
