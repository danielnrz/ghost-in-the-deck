# Character pipeline

The shipped performer is reproducible from local, redistributable inputs. MPFB
creates the body and the `game_engine` deform skeleton. The repository generator
then aligns the rig, bakes the relaxed bind stance, and builds the DJ identity:

- fitted top and trousers are cut from copies of the baked body topology, so
  they retain the exact armature weights;
- sneakers, cap, slim visor, chest mark, and headphones are Blender primitives
  weighted to the appropriate deform bones;
- six simple glTF-safe materials provide skin, fabric, footwear, metal, and the
  cyan identity accent;
- no downloaded clothing, logo, texture, animation, or proprietary character
  asset is included.

Rebuild the portable GLB and Panda3D BAM:

```bash
blender -b --python scripts/blender/make_avatar.py -- \
  --out assets/avatar/ghost_test.glb
.venv/bin/gltf2bam assets/avatar/ghost_test.glb assets/avatar/ghost_test.bam
```

`--no-outfit` is available only for body/rig diagnostics. The production asset
must retain the expected outfit mesh and material names guarded by
`tests/test_avatar_asset.py`.

The skeleton names, axes, proportions, and neutral stance remain compatible
with `AvatarRig` and `PerformanceRig`. Contact poses are still calibrated against
the actual exported BAM and the shared workstation coordinates. Any future
change to body proportions, garment clearance, or the workstation must rerun
the rendered sequence and dense hand-clearance review.
