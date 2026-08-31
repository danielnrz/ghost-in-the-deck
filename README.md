# Ghost in the Deck

A 3D virtual DJ in Python. The long-term goal is a full-body humanoid avatar
standing behind DJ equipment that behaves like a DJ — moving with the music it
is playing, and eventually operating controls that genuinely change the audio.

This repository is currently at **Phase 0**.

## Phase 0 scope

Phase 0 is a feasibility prototype. It exists to answer one question: can this
machine support the whole technical chain?

```
music file -> audio analysis -> beat timestamps -> 3D Python application
           -> rigged humanoid -> visible beat-synchronised movement
```

That chain works end to end. What is deliberately **not** built yet: a DJ booth,
a mixer, turntables, a club, a crowd, facial animation, hand IK, automatic
mixing, audio effects, or any DJ decision-making. The avatar nods, sways and
bounces its shoulders on the beat; that is the whole behaviour.

## How it fits together

```
Audio Analyzer      audio/analysis.py    librosa, offline, before playback
      |
MusicFeatures       audio/features.py    plain data - no librosa, no Panda3D
      |
(DJ Behaviour)      animation/cues.py    reserved seam; today one cue per beat
      |
Animation Ctrl      animation/controller.py
      |
3D Avatar           animation/rig.py     the only module that touches bones
```

The layers are kept apart so a real behaviour engine can be dropped in between
`MusicFeatures` and the animation controller later without rewriting either end.

## Requirements

- Python 3.12
- ffmpeg (decoding MP3/M4A; Panda3D's OpenAL backend only reads WAV/OGG)
- A GPU with OpenGL 3.2+
- Blender 4.2+ **only if you want to regenerate the avatar** — the exported
  asset is committed, so this is not needed to run the prototype.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Put your own audio files in `testMusic/`. The directory is kept in the
repository but its contents are ignored, so no music is ever pushed. Tracks are
discovered at runtime; no filename is hard-coded. With no `--track` argument the
smallest file is used.

## Running it

Analyse a track and write the features to `out/analysis/<track>.json`:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.analyze_cli          # one track
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.analyze_cli --all    # every track
```

Run the prototype — analyse, play, render, move on the beat:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --track minel --seconds 30
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --verbose        # log every beat
```

Analysis is cached per track, so only the first run pays for it. Timing results
are written to `out/sync_report.json`.

Useful flags: `--headless` (render offscreen), `--no-audio` (silent run),
`--refresh` (re-analyse), `--seconds N` (stop early).

## Tests

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests
```

They cover real audio analysis (including a synthetic click track of known
tempo), the exported avatar's mesh and skeleton, cue scheduling, timing
statistics, and a pixel comparison of rendered frames proving the movement is
actually visible. Tests needing a display skip without one; under a headless
shell use `xvfb-run -a`.

## Regenerating the avatar

```bash
blender -b --python scripts/blender/make_avatar.py -- --out assets/avatar/ghost_test.glb
.venv/bin/gltf2bam assets/avatar/ghost_test.glb assets/avatar/ghost_test.bam
```

The avatar comes from [MPFB](https://extensions.blender.org/add-ons/mpfb/), the
free MakeHuman plugin for Blender, using its `game_engine` deform skeleton.
Rigify is available in the same script but exports control machinery a runtime
cannot use.

Two format notes, both learned the hard way:

- Morph targets are excluded from the export. Blender writes shape keys as
  sparse glTF accessors, which `panda3d-gltf` cannot read.
- Panda3D 1.10 has no glTF loader, so the GLB is converted to `.bam`. The GLB is
  kept as the portable interchange asset; the `.bam` is what the app loads.

## Known limitations

- The avatar's rest pose is MPFB's A-pose, so the arms sit away from the body.
  It is not posed for standing at a DJ booth.
- No DJ behaviour exists. Every beat produces the same movement, scaled only by
  the track's bass and onset energy.
- Beat detection is whole-track and fixed-tempo. Tracks that change tempo, and
  the quieter intros of some tracks, will drift.
- Analysis runs before playback. There is no live or microphone input.
- Timing is measured against Panda3D's playback clock. It does not include sound
  card output latency, which would need an external recording to measure.
- Movement quality is tied to frame rate: the beat handling runs in the render
  loop, so a throttled or slow display degrades synchronisation.
- The exported skin is a plain solid material. MPFB's detailed skins need asset
  packs that are not part of the add-on.
