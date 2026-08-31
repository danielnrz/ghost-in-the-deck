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

## Timing model

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

Useful flags: `--headless` (render offscreen), `--no-audio` (silent run),
`--refresh` (re-analyse), `--seconds N` (stop early).

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

- The avatar's rest pose is MPFB's A-pose, so the arms sit away from the body.
  It is not posed for standing at a DJ booth.
- No DJ behaviour exists. Every beat produces the same movement, scaled only by
  the track's bass and onset energy.
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
