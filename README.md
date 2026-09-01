# Ghost in the Deck

A 3D virtual DJ in Python. The long-term goal is a full-body humanoid avatar
standing behind DJ equipment that behaves like a DJ — moving with the music it
is playing, and eventually operating controls that genuinely change the audio.

This repository is currently at **Phase 1A**.

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
      +--> BeatTimeline    animation/cues.py     beats and beat phase
      +--> EnergyTrack     animation/energy.py   smoothed intensity
      |
GrooveEngine        animation/groove.py  what the body is doing at time T
      |
GrooveState                              beat phase, energy, sway, weight...
      |
Animation Ctrl      animation/controller.py   groove -> joint angles
      |
3D Avatar           animation/rig.py     the only module that touches bones
```

`GrooveEngine` is the seam the DJ behaviour engine will grow into. It already
answers "what is the body doing right now"; later it will also answer "and what
is the DJ reaching for".

## Body language (Phase 1A)

The avatar stands in a neutral DJ stance rather than the asset's A-pose, and
moves continuously rather than twitching once per beat.

Several rates run at once, because a body moving at a single frequency reads as
a machine:

| Layer | Rate | Drives |
|---|---|---|
| breath | ~7 s | never completely still |
| weight shift | 2 bars | hips, spine counter-lean, knees |
| sway | 1 bar | torso rotation and lean, head |
| bounce | 1 beat | continuous rise and fall through the whole body |
| accent | on the beat | the sharp nod, with a short attack so it does not snap |

Movement scale follows a smoothed energy signal built from RMS, bass and onset
activity, rescaled against **the track's own** dynamic range so a quiet recording
still reaches full intensity in its loudest passage. Movement never stops
entirely: quiet passages are restrained, not frozen.

Relative scaling alone cannot tell quiet music from silence, though - a flat
envelope of 1e-12 normalises to exactly the same curve as a flat envelope of 0.5.
Analysis therefore also records `peak_rms`, the loudness of the track before the
envelopes were normalised, and the groove fades movement out below it. Silence
and a barely-there signal both settle at the resting intensity, and everything in
between ramps smoothly.

Variation is deterministic. Per-bar character - emphasis, head bias, which
shoulder works harder, which leg takes the weight - comes from a crc32 of the
track name and bar number, eased across the bar so nothing snaps at the bar line.
`hash()` is deliberately not used: Python randomises string hashing per process,
which would give a different dance every run. The same track always moves the
same way at the same moment.

### The neutral stance

The avatar's relaxed standing pose is **baked into the asset** by the Blender
script, not applied at runtime. `AvatarRig.NEUTRAL_POSE` is empty; the runtime
composition is

```
asset's own rest pose  +  groove  +  future gestures
```

Phase 1A applied the stance as a large runtime correction on top of the A-pose
the asset was bound in, and it looked wrong - splayed shoulders, elbows winged
out, hands floating beside the waist. The cause was not the correction itself.

**MPFB builds the game-engine rig in a space whose origin is at the hips, while
placing the body with its feet on the ground.** The whole skeleton therefore sat
about 0.86 m below the geometry it deforms: the `head` joint was at waist
height, `foot_l` was below the floor. At rest that is invisible, because the
deformation is the identity - but every bone then rotated about a pivot most of
a metre away from the joint it represents, so rotations smeared the mesh
sideways instead of bending it. Small angles looked merely odd; the stance's
larger ones destroyed the silhouette.

`make_avatar.py` now measures that offset by comparing each bone against the
centre of the vertices weighted to it, takes the median across all of them, and
moves the bones onto the body. The mesh does not move - only the pivots become
correct. The offset is measured rather than hard-coded so it keeps working if
MPFB changes its numbers, and `verify()` fails the build if the rig ever drifts
off the mesh again.

With correct pivots the stance itself is then posed by aiming bones at target
directions and baking the result as the new rest pose: duplicate the armature
modifier, apply the first copy so the vertices move to where they are drawn,
then make the pose the rest pose. The surviving duplicate re-binds the baked
mesh, so at rest it deforms by nothing.

To review the stance yourself:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_stance.py
```

writes front, side and three-quarter views to `out/stance_review/`.

Joint limits in `AvatarRig.LIMITS` bound how far each joint may move from that
rest pose, and are enforced in the rig so nothing upstream can exceed them.

## Timing model## Timing model

The playback clock is the authority for musical progress, not the renderer.

The pose is a **pure function of playback time**. `BeatTimeline` answers what the
music is doing at a time by binary search and keeps no playback position;
`AvatarAnimator.state_at(t)` derives the nod from the age of the surrounding cues
and the sway from `t` directly. Nothing is integrated across frames, so the
sequence of frames drawn before a moment cannot change the pose produced at it.

The practical consequence: if rendering stalls, frames are missed - that is
unavoidable - but the next frame drawn is correct for the moment it is drawn.
There is no catching up, no replayed backlog and no accumulated drift.

`clock.py` matters more than its size suggests. Panda3D refreshes a sound's
reported position from a task, so while the main loop is blocked `getTime()` does
not move even though the sound card keeps playing. Reading it directly would
freeze musical time for exactly as long as the renderer stalls. The clock instead
uses each reading as an anchor and carries time forward on the wall clock between
refreshes, re-anchoring whenever the sound reports a new position.

## What the timing numbers mean

A run prints two independent things, and they must not be confused.

| Metric | Meaning |
|---|---|
| **State lag** | Playback time elapsed during the update minus the time the pose was evaluated for. Near zero by construction. A large value would mean the musical state had fallen behind - the thing the architecture exists to prevent. |
| **Beats missed** | In-scope beats that no update sample carried, because the loop did not run during their response window. A loop limitation, not a timing error: a one second freeze physically cannot sample the beats inside it. |
| **Beats in scope** | Beats the run was already sampling before they happened and still sampling once their response window had passed. Beats outside that cannot be judged fairly - the run had not started, or had already stopped. |
| **Beat response latency** | For beats that *were* sampled, how long before the first sample carrying them. Roughly one sample interval when the loop is healthy. |
| **Sample interval / update cost** | Total time between update samples, and the share of it spent in this project's own code. The gap between them is where a stall actually lives. |

### What a "sample" is, and is not

A sample is recorded in the update task, straight after the pose is written. It
proves the application **evaluated and wrote** the pose for that playback time.

It does **not** prove the GPU and compositor put that frame in front of the
viewer, nor when. Measuring real presentation would need GPU timer queries or
compositor presentation feedback, neither of which this project does. The metrics
are therefore named for what is actually observed - update samples - rather than
frames presented. The terminal output says so on its last line.

### The response window

Coverage uses a `response_window` of three decay constants (~0.48 s), shared with
the animator so the numbers and the movement cannot drift apart.

This is a deliberate reporting threshold, not a physical boundary. The impulse is
still mathematically non-zero past it - it only falls under the animator's
epsilon after roughly seven decay constants - but by three the movement is down
to about 5% of peak, below which counting a sample as having captured the
response would be generous.

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

Watch the numbers behind the movement:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --seconds 60 --debug-motion
```

prints playback time, beat index, beat phase, bar phase, energy, intensity and
the individual groove layers.

Useful flags: `--headless` (render offscreen), `--no-audio` (silent run),
`--refresh` (re-analyse), `--seconds N` (stop early),
`--debug-every N` (debug print frequency).

To watch stall recovery directly, freeze the update loop on purpose:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app \
    --seconds 30 --simulate-stall 1.0 --stall-every 3.0
```

Beats inside each freeze are reported as missed, while state lag stays near
zero - the loop misses samples, the music does not drift.

## Tests

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests
```

The suite needs **no music of your own**. Audio correctness is checked against a
generated track carrying real kick, body and tick energy, so a fresh clone with
an empty `testMusic/` gets a full result. The checks against your own library are
an optional extra and skip when there is nothing there.

Coverage: audio analysis, the exported avatar's mesh and skeleton, timeline
lookup, the playback clock, timing statistics, and a pixel comparison of rendered
frames proving the movement is actually visible.

`test_timing.py` is the render-stall suite. It drives the real timeline and
animator through 60/30/15/5 fps schedules and through 250 ms, 500 ms and 1 s
stalls, asserting the pose at a playback time is identical however many frames
preceded it.

Tests needing a display skip without one; under a headless shell use `xvfb-run -a`.

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

- No DJ behaviour exists yet. The avatar grooves to the music but does not do
  anything a DJ does: no booth, no controls, no gestures.
- Bars are assumed to be four beats. A track in another metre still grooves, but
  the slow layers land on the wrong subdivision.
- Before the first detected beat and after the last, the beat grid is
  extrapolated at the nominal interval with virtual beat indices, so an intro
  still grooves. Those indices are not real beats; `has_beat` and `cue_before`
  are what distinguish them.
- Leg movement is deliberately small. The feet are planted and there is no IK, so
  anything larger reads as sliding.
- Leg movement is small because the pelvis is the animation root: bending a knee
  moves the foot rather than lowering the body. Planted feet would need IK.
- Beat detection is whole-track and fixed-tempo. Tracks that change tempo, and
  the quieter intros of some tracks, will drift.
- Analysis runs before playback. There is no live or microphone input.
- Timing is measured against Panda3D's playback clock. It does not include sound
  card output latency, which would need an external recording to measure.
- Poses are evaluated on the render thread, so a stalled loop still means missed
  samples. The music does not drift, but nothing is drawn during a freeze.
- Presentation to the monitor is not measured, only update samples. See above.
- Panda3D's `AudioSound.status()` has only BAD, READY and PLAYING, so a track
  that ended, was stopped early, underran or failed are indistinguishable. The
  clock treats every non-playing state alike: it keeps extrapolating for a short
  grace period, then holds its position rather than running on silently. There is
  no pause or resume support.
- Frame delivery on a Wayland compositor varies with display state. The same
  build has been measured at both 166 fps and under 2 fps on this machine
  depending on whether the surface was actually being composited.
- The exported skin is a plain solid material. MPFB's detailed skins need asset
  packs that are not part of the add-on.
