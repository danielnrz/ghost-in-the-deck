# Ghost in the Deck

A 3D virtual DJ in Python. The long-term goal is a full-body humanoid avatar
standing behind DJ equipment that behaves like a DJ — moving with the music it
is playing, and eventually operating controls that genuinely change the audio.

This repository is currently at **Phase 1B**.

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
GrooveEngine        animation/groove.py  how the body feels the music at time T
      |                                    (beat phase, energy, sway, weight...)
DJBehaviorEngine     animation/dj_behavior.py  what the DJ is occasionally doing
      |                                        (deck_glance, lean_in, ...)
Animation Ctrl       animation/controller.py   groove + action -> joint angles
      |
3D Avatar            animation/rig.py     the only module that touches bones
```

`GrooveEngine` and `DJBehaviorEngine` answer two different questions on
purpose - "how does the body feel the beat" and "is the DJ doing something
right now" - so a real DJ intelligence layer can replace or extend the second
one later without the first ever needing to change.

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

Watch the numbers behind the movement:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --seconds 60 --debug-motion
```

prints playback time, beat index, beat phase, bar phase, energy, intensity and
the individual groove layers.

Useful flags: `--headless` (render offscreen), `--no-audio` (silent run),
`--refresh` (re-analyse), `--seconds N` (stop early),
`--debug-every N` (debug print frequency), `--no-actions` (groove only, as in
Phase 1A, for comparison).

To watch stall recovery directly, freeze the update loop on purpose:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app \
    --seconds 30 --simulate-stall 1.0 --stall-every 3.0
```

Beats inside each freeze are reported as missed, while state lag stays near
zero - the loop misses samples, the music does not drift.

## DJ actions and the workstation (Phase 1B)

The avatar now stands at a small procedural DJ workstation - a table, two deck
platters and a mixer strip with a few knobs and faders - and occasionally does
something a DJ does on top of the continuous groove: glances at the deck, leans
in, reaches a hand toward the controls, or lifts into a small energetic accent.

**The workstation's layout lives in one place.** `animation/workstation.py`
defines `DJWorkstationTargets` - plain `(x, y, z)` tuples, no Panda3D - and
`scene/workstation.py` builds the actual geometry from those same numbers. The
behaviour layer reads the same targets to decide where to look and reach, so
the table can never drift out of sync with what the avatar is aiming at.

**"How the body feels the beat" and "what the DJ is doing" are kept apart.**
`GrooveEngine` still answers the first question continuously. `DJBehaviorEngine`
answers the second only occasionally: it builds a deterministic gesture
schedule once from the analysed track - a list of `GestureEvent`s, each with a
start time, a duration and a kind - and `state_at(T)` finds the one covering
`T` by binary search, the same stateless pattern `BeatTimeline` already uses.
Selection is a hash of the track's own seed and the bar number (not `hash()`,
which is salted per process and would give a different show every run), and it
leans on the same smoothed energy curve the groove uses: restrained passages
favour a deck glance, energetic ones occasionally allow a small hype accent,
and a gesture never fires more than once every few bars.

**Composition is intentional, not just additive.** A gesture's own offsets are
scaled by an attack-hold-release envelope that is exactly zero at both ends, so
nothing jumps when a gesture starts or stops. Wherever a gesture and the groove
would otherwise fight for the same joint - a beat's downward nod during a deck
glance, for instance - the groove's contribution to *that joint only* is
damped in proportion to the gesture's own weight. Every other joint, most of
the time the legs and the arm not being used, keeps the groove at full
strength: that is what "the groove continues underneath a gesture" means
concretely.

Review the gestures without waiting for the schedule to produce one:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --show-action lean_in
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.app --show-action hand_to_deck --show-side r
```

loops one gesture on repeat, with the groove still running underneath it, in a
real window - and runs for 12 seconds by default rather than the whole track,
since a few loops is plenty to judge a gesture by (`--seconds` still overrides
this for a longer look). Or render still frames of every gesture:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_actions.py
```

writes a neutral reference, front and three-quarter views of every gesture, and
one wide establishing shot, to `out/action_review/`.

### Reaching for the controls (Phase 1B.1)

`hand_to_deck` was originally a fixed set of joint rotations, the same shape as
`lean_in` or `deck_glance`. Visually it did not read as reaching for anything -
the hand moved vaguely toward the table without landing near it. Fixed
rotations cannot land on a specific point; getting there needed real IK.

`animation/arm_ik.py` is a small, dependency-free two-bone solver (shoulder ->
elbow -> wrist, standard law-of-cosines geometry) that computes where the elbow
should be for the wrist to reach a target - the general, testable part, and the
part any future gesture could reuse. Turning that into this rig's joint
rotations is not general, though: this skeleton's `controlJoint` transform is
not a plain scene-graph-relative one, and calling Panda3D's own `lookAt()` on
it aimed the arm at a direction with dot -0.78 against the intended target -
nearly backwards. The only approach that actually worked was numerical: nudge a
joint's rotation by a small amount, watch how the real, live rig's exposed
child joint actually moved, and take repeated small damped steps toward the
target. That solve is calibration, not runtime work - it ran once, offline, via
`scripts/solve_arm_ik.py`, and its output is the fixed peak offsets
`gesture_pose.py` blends in by the gesture's envelope weight, the same shape
every other gesture already uses.

One more deliberate choice: the workstation's own targets sit right at the edge
of this arm's reach (confirmed by measurement - even the closest, the front
control row, is about a centimetre beyond the arm's fully-stretched length), so
aiming at the exact target locks the elbow almost straight. The gesture instead
aims at 90% of that distance, which leaves the wrist about 5 cm short of the
true target but gives a visibly bent elbow (about 45 degrees off straight)
rather than a locked one - a real reach, not a stretch.

`small_hype` changed too, for a different reason: two arms spreading outward in
mirror image read as a T-pose, not a performance accent. It is now one arm,
chosen deterministically per event the same way `hand_to_deck`'s side already
was, while the other arm keeps grooving normally underneath it.

### A reach path that goes around the table, not through it (Phase 1B.2)

Phase 1B.1's calibrated endpoint was correct; the path to it was not. Blending
neutral straight to the final reach in one sweep - the same shape every other
gesture uses - happens to pass low and forward before it passes high, and the
tabletop sits exactly in that path: measured, the wrist dipped into the
tabletop solid partway through the attack and release.

The fix is a staged path - neutral -> clearance -> target -> clearance ->
neutral - through a second calibrated pose, `IK_CLEARANCE`. That pose is
*not* aimed at an independent hover point the way the final reach is: a
two-bone solve toward one did not converge in a bounded, widened-limit search
from one start (reproduced in `scripts/solve_arm_ik.py` - a limited negative
result, not a proof that no rotation exists). `IK_CLEARANCE` instead reuses
the final reach's shoulder rotation verbatim and folds the forearm up with a
fixed elbow flexion, so the clearance-pose hand sits high and in front of the
table rather than dangling at its edge. On top of that the reach adds a
handful of small, separately named terms (extra abduction, wrist tilt-up, a
corner detour, an inward hover at the hold); the ones that keep the hand off
the table are deliberately not scaled by how hard the DJ reaches, since a
timid low-energy reach has to clear the slab too.

Phase 1B.2's own finding was that all of this was being *checked* on the wrist
alone. The rig carries fifteen un-posed finger joints per hand (the furthest
~160 mm past the wrist) that drive the visible hand mesh, and at the hold they
were buried up to 25 mm in the tabletop. `tests/reach_clearance.py` is the new
guard. It reads the character's skinning table once, buckets every hand-mesh
vertex onto the joint it follows, and then carries that **actual mesh** through
a documented population of real scheduled events - both sides, several tempos,
energies and seeds, groove composed on top - measuring it against the *built*
workstation (boxes and vertical cylinders from `getTightBounds`, not the
analytic formulas). `test_reach_trajectory.py` asserts a real margin on every
run. Because the guard now measures the painted surface rather than a skeleton
pivot, the threshold is just residual model error (~6 mm): the skinned hand
mesh clears every piece of furniture - tabletop, legs, deck platters and their
bases - by more than 2 cm across the whole population, worst case a low-energy
reach whose little finger passes near the left deck platter (that platter's
edge sits ~29 mm from the side control the hand is operating). Deliberate
near-contact with the knob the hand *works* is a separate, separately named
allowance. `scripts/solve_arm_ik.py` derives the calibration;
`scripts/measure_hand_skin.py` dumps the per-joint skin radii; `gesture_pose.py`'s
module docstring has the full account.

The workstation gained real geometry to match: `left_controls`/`right_controls`
- what `hand_to_deck` was always reaching toward - previously had nothing
visible at them. `scene/workstation.py` now builds a small control cluster
(panel, knob, fader) directly from each of those two points, mirrored, so the
hand visibly lands on something.

## Tests

```bash
PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests
```

The suite needs **no music of your own**. Audio correctness is checked against a
generated track carrying real kick, body and tick energy, so a fresh clone with
an empty `testMusic/` gets a full result. The checks against your own library are
an optional extra and skip when there is nothing there.

Coverage: audio analysis, the exported avatar's mesh and skeleton, timeline
lookup, the playback clock, timing statistics, gesture scheduling and pose
composition, and a pixel comparison of rendered frames proving the movement is
actually visible.

`test_timing.py` is the render-stall suite for the groove. It drives the real
timeline and animator through 60/30/15/5 fps schedules and through 250 ms,
500 ms and 1 s stalls, asserting the pose at a playback time is identical
however many frames preceded it. `test_dj_behavior.py` does the same for the
gesture layer, and adds world-space checks - feet stay planted, a deck glance
moves the head down, a hand-to-deck reach moves toward the table, a hype
gesture moves away from it - because Phase 1A already showed that a correct
joint angle can still move the mesh the wrong way if its pivot is wrong.
`test_arm_ik.py` tests the IK geometry itself against independent facts (the
solved elbow is exactly one bone-length from the shoulder and exactly the other
from the target, for any reachable target; an unreachable one clamps instead of
producing nonsense) rather than by re-deriving the same formula the code uses.
`TestHandToDeckReach` in `test_dj_behavior.py` measures the actual reach on the
real rig: wrist-to-target distance, wrist height against the tabletop, and
elbow bend angle, for both arms. `test_reach_trajectory.py` measures the whole
staged path, not just its endpoint: dense world-space collision checks against
the real built tabletop and side-control geometry, for both arms and several
groove states, continuity and schedule-independence checks, a subprocess check
that `scripts/solve_arm_ik.py` still reproduces the committed constants, and
the `reach_clearance.py` population sweep of the skinned hand mesh against the
built workstation over real scheduled events with the groove on top.

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

- The gesture vocabulary is small and does not yet do anything to the audio
  itself - no knob contact, no fader movement, no EQ, no crossfading.
- `hand_to_deck`'s IK targets a fixed point - the front control row - and its
  peak offsets are calibrated once, offline, against the arm at rest. It does
  not re-solve live against wherever the shoulder actually is once groove sway
  has moved it, does not orient the wrist or fingers, and would need
  recalibrating (`scripts/solve_arm_ik.py`) if the avatar or the workstation's
  layout ever changed. In practice the wrist lands within about 5-8 cm of the
  true target, which groove sway can add a little to or take a little from.
- All of this avatar's workstation targets sit at or just beyond this arm's
  natural reach - even the closest, the front control row, is about a
  centimetre past it. Reaching for the platters themselves (10-27 cm further)
  is not attempted; only the front row is used.
- `CLEARANCE_LIFT_ROLL` and the other reach tuning terms are magnitudes found
  by sweeping this specific avatar and table, not a general collision solver -
  if either geometry changes materially, re-run `scripts/solve_arm_ik.py` and
  re-verify a real margin with the `reach_clearance.py` sweep rather than
  assuming the same values still clear. The sweep is a *measurement* of the
  existing skinned mesh, offline and in tests; there is no runtime collision
  solver or runtime finger IK, by design.
- Gesture selection reasons about relative energy and a short trend, not real
  musical structure. It has no idea what a build-up, a drop or a breakdown is.
- Bars are assumed to be four beats. A track in another metre still grooves,
  and gestures still land on a bar boundary, but the wrong one.
- Before the first detected beat and after the last, the beat grid is
  extrapolated at the nominal interval with virtual beat indices, so an intro
  still grooves. Those indices are not real beats; `has_detected_beat` and
  `cue_before` are what distinguish them, not the sign of the index.
- Leg movement is deliberately small. The pelvis is the animation root, so
  bending a knee moves the foot rather than lowering the body; anything larger
  than the current small amount reads as sliding, and would need real IK to
  keep the feet planted.
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
  packs that are not part of the add-on.
