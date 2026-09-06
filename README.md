# Ghost in the Deck

A 3D virtual DJ in Python. The long-term goal is a full-body humanoid avatar
standing behind DJ equipment that behaves like a DJ — moving with the music it
is playing, and eventually operating controls that genuinely change the audio.

This repository is currently at **Phase 4C**: an offline two-file transition
preview built on the frozen Phase 4B PCM executor.

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
guard. It reads the character's skinning table once, buckets the farthest
vertices of each hand-mesh joint's *predominant* blend weight onto that joint,
and carries that rigid, per-joint approximation of the mesh through a
documented population of real scheduled events - both sides, several tempos,
energies and seeds, groove composed on top - measuring it against the *built*
workstation (boxes and vertical cylinders from `getTightBounds`, not the
analytic formulas). `test_reach_trajectory.py` asserts a real margin on every
run. It is an approximation, not the true blended mesh - rigid single-joint
skinning ignores the real multi-joint blend near the knuckles - so the
threshold (`SAFETY_MARGIN`, 8 mm) is sized to cover that approximation's own
measured error (an independent full linear-blend reconstruction of the
reported worst event read ~3.7 mm lower than the rigid proxy there) on top of
the smaller box/cylinder and vertex-truncation effects; see
`tests/reach_clearance.py`'s own `SAFETY_MARGIN` comment for the itemised
sum. Reported margins: the rigid proxy this repo sweeps clears the built
furniture - tabletop, legs, deck platters and their bases - by +22.3 mm
(committed population) / +22.7 mm (denser resample), worst case a low-energy
reach whose little finger passes near the left deck platter (that platter's
edge sits ~29 mm from the side control the hand is operating); the
independent full-blend reconstruction of that same worst event read a more
conservative +19.9 mm there - still clear, but under 2 cm, not "more than
2 cm across the whole population" as an earlier draft of this section claimed.
Deliberate near-contact with the knob the hand *works* is a separate,
separately named allowance. `scripts/solve_arm_ik.py` derives the
calibration; `scripts/measure_hand_skin.py` dumps the per-joint skin radii;
`gesture_pose.py`'s module docstring has the full account.

The workstation gained real geometry to match: `left_controls`/`right_controls`
- what `hand_to_deck` was always reaching toward - previously had nothing
visible at them. `scene/workstation.py` now builds a small control cluster
(panel, knob, fader) directly from each of those two points, mirrored, so the
hand visibly lands on something.

## A gesture that touches the audio (Phase 1C)

Through Phase 1B.2 the avatar's hand could land on a control, but nothing it
did changed what was heard - the gesture and the track were two independent
things playing at the same time. Phase 1C ties exactly one of them together:
every scheduled `hand_to_deck` event now sweeps a real filter across the
track's own audio, offline, before playback ever starts.

`audio/effects.py` is a small, dependency-free (on Panda3D) module - plain
PCM sample arrays and a sample rate in, the same array back out - implementing
a one-pole low-pass and a one-pole high-pass filter. The one deliberate
exception to "audio code only knows about audio" is that it imports
`GestureEvent` and `dj_behavior._envelope_weight` directly from the animation
layer (which has no Panda3D dependency of its own). That is not a layering
slip: the whole point is that the filter's depth at time T and the gesture's
visual weight at time T are *the same function call*, not two curves that were
separately tuned to look similar. `side="l"` sweeps the low-pass down and back
up - cutting highs; `side="r"` sweeps the high-pass up and back down - cutting
lows, mirroring the workstation's own `left_controls`/`right_controls` split
so the two hands audibly do different things. A one-pole filter's cutoff and
its smoothing coefficient are two views of the same number; the module sweeps
the coefficient directly because doing so makes "no effect" exact at both
ends of a gesture's window by construction, not just small - see the module's
own docstring for why that took an explicit boundary pin for the high-pass
case and not the low-pass one.

`app.py` renders this once per track, offline: `processed_audio_path` reads
the decoded wav, runs it through `apply_hand_to_deck_effects` using the
`DJBehaviorEngine`'s own event schedule, and caches the result under
`cache/audio_fx/`, the same pattern `audio/decode.py` already uses for the
plain decode. The cache key folds in the wav's identity and every
`hand_to_deck` event's own timing, side and strength, so a stale render is
never served after the track, the seed or the schedule changes. This only
happens when a behaviour layer exists at all (`--no-actions` disables it, same
as the gesture layer itself); `--no-fx` disables just the audio processing,
for an A/B comparison against the same run with the same gestures but
unprocessed sound.

What this phase does not claim: there is still exactly one effect, tied to
exactly one gesture, on one hand-mixed track at a time. There is no EQ, no
crossfader, no fader movement, and no live parameter control - the sweep is
computed once offline from the fixed schedule, the same way the reach
calibration and the beat/energy analysis are, not solved or adjusted at
runtime.

## A second gesture, a second effect (Phase 1D)

Phase 1C tied exactly one gesture to the audio; the README's own known
limitations were honest that the other three still changed nothing about the
sound. Phase 1D closes that gap for a second one: every scheduled
`small_hype` event now runs a deterministic gain riser over `audio/effects.py`'s
new `apply_small_hype_effects`, so a hype gesture audibly lifts the track the
same way `hand_to_deck` audibly reshapes it.

`small_hype` was picked because the groundwork already existed: it already
carries a `side` and a `strength` (the schedule reuses the same channel of
per-bar draws as `hand_to_deck` for `side`, even though the riser itself does
not use it), and the gesture scheduler's own kind weighting was already
energy-driven - `small_hype` strongly favoured at high energy, suppressed at
low energy - so it already represents a distinct musical moment (an energy peak)
from `hand_to_deck`'s "working a control" moment. That argued for a distinct
*kind* of effect: an amplitude riser, not a second filter sweep.

The riser multiplies every sample in the event's window by a gain that
follows the *same* `ENVELOPE_SHAPE["small_hype"]` / `_envelope_weight` curve
already driving the gesture's visual weight, scaled by the event's own
`strength` - the identical "the effect and the gesture are the same curve"
principle Phase 1C established, applied to a different kind of processing.
Unlike a filter, a gain boost is not self-limiting: it can genuinely push a
sample past whatever the dry audio already needed to represent, so bounding
it to the dry window's own peak (the trick `hand_to_deck`'s filter uses) does
not apply. Instead each window's boost is capped by that window's own
*available headroom* to a fixed ceiling (`GAIN_RISER_CEILING`, nominal
full-scale for normalised PCM): the target gain is
`min(nominal_boost, ceiling / dry_peak)`, raised to match the dry peak if it
already exceeds the ceiling. An already hot window (dry peak already at or
above the ceiling) correctly gets little or no boost rather than clipping -
by construction, not by clamping the result afterward.

`app.processed_audio_path` composes this after the `hand_to_deck` sweep in
the same offline render, and its cache key now folds in every `small_hype`
event's own timing and strength as well (`side` is left out, since the riser
does not read it - folding in a field an effect ignores would just force
needless re-renders). A schedule with neither kind of event scheduled is
still a full no-op: the source wav is served untouched, unchanged from Phase
1C. This rides the same `--no-fx` switch as `hand_to_deck`; there is no new
flag.

What this phase does not claim: two of the four gesture kinds now touch audio,
not four. `deck_glance` and `lean_in` still change nothing about the sound.
There is still no EQ, no crossfader, no fader movement, and no live parameter
control - both effects are computed once offline from the fixed schedule.

## A decision layer, not a bigger effect (Phase 2A)

Phase 1C and 1D each wired a DSP effect directly to one gesture's own `kind`:
the filter sweep existed *because* a `hand_to_deck` was scheduled, the riser
*because* a `small_hype` was. That is `gesture -> audio`. Phase 2A inserts the
missing step and turns that arrow around for the audio decision:

    analysis -> musical context -> DJ action planner -> real audio action

`dj_planner.py` (top-level, alongside `app.py` - it joins the audio and
animation subsystems and belongs to neither) is a small, deterministic,
Panda3D-free module. `context_at` reads the track's smoothed energy and a
short trend at one absolute time - the *same* `EnergyTrack` and the *same*
promoted `trend_at` the gesture scheduler uses, and the *same* imported
`LOW_ENERGY` / `HIGH_ENERGY` band edges the gesture scheduler already reasons
about, not restated copies. `decide_action` then picks from
exactly three outcomes: high energy -> the gain riser (an energy peak,
`small_hype`'s existing role); mid energy -> the filter sweep
(`hand_to_deck`'s "working a control" role); low energy -> nothing. `trend` is
computed and carried but is not load-bearing in this *audio* decision - none of
the two existing effects represents a build. (Its visual sibling
`decide_gesture_kind`, added in Phase 2B below, does read it.)

**No new DSP effect was added.** `audio/effects.py` is byte-for-byte what
Phase 1D left it. The two functions there are reused exactly; `AUDIO_ACTIONS`
is deliberately just those two plus `"none"`.

`app.processed_audio_path` now calls `DJActionPlanner(behavior.seed).plan(...)`
once and builds its render (and its `cache/audio_fx/` key, which also folds in
`PLANNER_VERSION`) from the *planned actions*, not from the gesture schedule's
`kind`s. A planner handed a `GestureEvent` reads only its `.start` and
`.duration` - its `kind`, `side` and `strength` are never consulted. When a
sweep needs a left/right side, the planner derives one from its own seed and
the moment alone (salted apart from `dj_behavior`'s own per-bar side draw), so
the choice is identical whether or not the candidate gesture happened to carry
a side.

What this phase does **not** do: it does not yet touch `DJBehaviorEngine`'s
own gesture-kind selection. At Phase 2A the *visible* gesture at a scheduled
moment - which kind - is still the pre-existing weighted-random roll,
decided independently of this planner. Phase 2B (below) removes that roll and
has the same planner decide the gesture kind, side and strength too. Neither
phase adds section, build or drop detection - only an energy band and a short
trend - or any two-track mixing, EQ, crossfader or live parameter control.

## The gesture is the decision too (Phase 2B)

Phase 2A inverted only the audio half: at an already-scheduled moment,
`dj_planner.decide_action` chose the audio action from musical context, while
`DJBehaviorEngine._build_schedule` still rolled the gesture kind, side and
strength from its own independent weighted-random draw. Phase 2B removes that
draw entirely.

`dj_planner` gains a second deterministic decision, `decide_gesture_kind` - the
visual sibling of `decide_action`, reading the *same* `MusicalContext` and the
same imported `LOW_ENERGY` / `HIGH_ENERGY` band edges, never a candidate
event's own `kind`: high energy -> `small_hype` (the same peak
`decide_action` plans a gain riser for), mid energy -> `hand_to_deck` (the
visual partner of the filter sweep), low energy -> `lean_in` when the short
trend is rising past `TREND_RISING`, else `deck_glance`. `planned_strength`
(one shared formula, floored and rising with energy) and
`DJActionPlanner.gesture_side_for` (the same seed-and-moment draw the filter
sweep's side uses, `None` for the centred kinds) supply the other two fields.

`DJBehaviorEngine._build_schedule` now calls those three at each hosted bar's
own start time instead of the old `_kind_weights` / `_choose_kind` roll (both
methods now deleted). The visible gesture and the audio action `dj_planner` plans for
that same moment are now two readings of one decision rather than two
independent rolls. This also makes the short trend genuinely load-bearing in
the planner for the first time - `decide_gesture_kind` is where a rising trend
actually selects `lean_in`.

What Phase 2B still does **not** do:

- **No real musical-structure awareness.** The decision is still an energy
  band plus "energy now minus energy a few seconds ago". There is no build,
  drop, breakdown, section or phrase detection.
- **No planner-owned timing yet.** Whether a bar hosts a gesture at all, and
  how far apart gestures fall, is still `DJBehaviorEngine`'s own
  activation/gap roll at this phase. The planner decides *what* an event is,
  not *when* one happens; Phase 2C (below) moves that choice into the planner
  too.
- **No new audio.** `audio/effects.py` is unchanged. `deck_glance` and
  `lean_in` still carry no audio effect - by design, since neither the filter
  sweep nor the gain riser represents "glancing" or "leaning in".
- **Gesture poses are unchanged.** `gesture_pose.py`, `arm_ik.py` and the rig
  are byte-for-byte as Phase 1B.2 left them. Only *which* kind gets chosen at a
  moment changed, not how any kind looks.
- Still no two-track mixing, EQ, crossfader or live parameter control.

## The planner picks the moment too (Phase 2C)

Phase 2A and 2B moved *what* happens at an already-chosen bar into
`dj_planner` - first the audio action, then the gesture kind, side and
strength. `DJBehaviorEngine` still owned the bar *choice* itself: a flat
per-bar activation roll and the gap/duration jitter around it, decided without
looking at the music. Phase 2C moves that choice into the planner as well.

`DJActionPlanner.plan_schedule` now owns the whole bar-walk
`_build_schedule` used to run - the `start >= 1.0 s` eligibility floor, the
per-bar activation roll, the `MIN_GAP_BARS` + jitter spacing, and the
`BASE_DURATION` + jitter duration with its past-track-end drop. The activation
roll is no longer flat: `activation_probability` reads the *same*
`MusicalContext` the kind/side/strength decisions already use - a per-band
base chance (`low` 0.28 / `mid` 0.42 / `high` 0.58, where `mid` is exactly the
old flat `ACTIVATION_PROBABILITY` so a mid-energy stretch keeps today's event
density) plus a `+0.15` boost when the short trend clears `TREND_RISING`,
capped at 1.0. The same `trend > TREND_RISING` reading that already picks
`lean_in` now also makes a rising passage host more events. `MIN_GAP_BARS`,
the gap jitter and the duration jitter moved to `dj_planner` with the walk;
`DJBehaviorEngine` no longer contains any activation, gap or duration-jitter
logic, and `_build_schedule` is a one-line delegation to `plan_schedule`.

What Phase 2C still does **not** do:

- **No musical-structure awareness in Phase 2C.** Candidate positions are the
  same fixed bar grid `_build_schedule` always walked, and at this phase
  `activation_probability` reads only the smoothed energy band and "energy now
  minus energy a few seconds ago" - no broader-timescale trajectory. Phase 3A
  (below) adds one; it still does no phrase, section, drop or breakdown
  detection.
- **Spacing and no-overlap guarantees are unchanged.** The same
  `MIN_GAP_BARS`-plus-jitter minimum separation and the same drop of any event
  that would run past the track end still apply; only *where* those constants
  live and *how* the per-bar chance is computed changed.
- **No new DSP effect.** `audio/effects.py` is byte-for-byte unchanged, and
  there is still no two-track mixing, EQ, crossfader or live parameter
  control.
- **Gesture poses are unchanged.** `gesture_pose.py`, `arm_ik.py` and the rig
  are as before; only which bar hosts a gesture, and how often, changed.

## A broad-trajectory signal (Phase 3A)

Every energy reading so far - the groove's ~1.5 s smoothing, the DJ scheduler's
3 s `trend_at` - is a bar-scale one: it sees a loud bar, not a two-minute rise.
Phase 3A adds a lightweight, deterministic **musical-structure layer** on top,
using only what audio analysis already produced. No new librosa pass, no schema
bump: `animation/structure.py` reuses `EnergyTrack` with a longer
`smoothing_seconds` (`PHRASE_SMOOTHING_SECONDS = 8.0`) and a generalised
`trend_at` at a matching 8 s lookback, so a whole build or release registers as
one movement instead of a string of wobbles.

`structure_at` returns a frozen `MusicalStructure` for one playback time - a
pure function of the smoothed-energy curve and the beat grid, so the same inputs
always give the same instance. It carries:

- **`regime`** - one of `build` / `release` / `peak` / `stable`. These are
  honestly-named descriptions of what the broad energy is doing right now, not
  section labels. `build` / `release` come from the broad slope clearing
  `BUILD_SLOPE` / `-RELEASE_SLOPE`; `peak` is a broad level at or above
  `HIGH_ENERGY` that is not moving fast; `stable` is everything else. Nothing
  here claims to tell a "drop" from a "breakdown".
- **`section_change_likelihood`** - a continuous `0..1` reading of how fast the
  broad energy is moving (`min(abs(broad_trend) / SECTION_CHANGE_SCALE, 1.0)`),
  a stand-in for "something structural is probably happening around here". Not a
  detector, not a boolean, and not currently read by any decision.
- **`phrase_position`** - `bar_index` folded into an eight-bar cycle
  (`bar_index % PHRASE_LENGTH_BARS`). This is a **periodicity assumption**
  asserted by a constant, **not detected content**: real music does not always
  phrase in eights and this code never measures whether a given track does. It
  is computed and tested and then **deliberately wired into no decision**.

`DJActionPlanner.plan_schedule` builds one broad `EnergyTrack` per call and
attaches `structure_at(...)` to every bar context it constructs, as
`MusicalContext.structure`. The one place that context is read is
`activation_probability`, the planner's single timing decision: when
`structure` is present, a broad `build` regime adds `STRUCTURE_BUILD_BOOST`
(0.10) and a broad `release` subtracts `STRUCTURE_RELEASE_DAMP` (0.10, smaller
than the lowest band base so it can never reach zero); `peak` and `stable`
leave the Phase 2C value untouched. The result: the avatar is busier through a
broad build and quieter through a broad release. With `structure=None` nothing
is added and `activation_probability` is **byte-identical to Phase 2C**.

What Phase 3A still does **not** do:

- **No semantic section detection.** There are no verse / chorus / drop /
  breakdown labels, and no plan to infer them from four regimes and a slope.
- **`phrase_position` changes nothing.** It is a tested convention with no
  consumer; the eight-bar cycle is an assumption, not a measurement.
- **Only `activation_probability` is structure-aware.**
  `decide_action`, `decide_gesture_kind`, `gesture_side_for` and
  `planned_strength` are unchanged by this phase - they still read only the
  bar-scale energy band and short trend.
- **No new audio.** `audio/effects.py` is byte-for-byte unchanged; still no
  two-track mixing, EQ, crossfader or live parameter control.
- **Gesture poses are unchanged.** Only how often a bar hosts a gesture during
  a broad build or release changed.

## A two-deck planning foundation (Phase 4A)

Every phase so far reasons about **one** track: "how should the DJ move to the
music playing now". Phase 4A adds the vocabulary for the other question - "how
could deck A move into deck B" - without touching the accepted single-track
runtime. It is a **planning foundation only**: raw measurable quantities and
deterministic, non-semantic scoring. Nothing here plays, mixes or filters
audio.

The single-track decision layer is untouched by rule. `dj_planner.py` and
`dj_behavior.py` are **byte-for-byte unchanged** this phase, and the new code
lives in its own module that **never imports** either of them - the dependency
would run the wrong way.

- **`TrackDeck`** (`deck.py`) wraps one already-analysed `MusicFeatures` with
  the derived views planning needs: the `BeatTimeline`, the groove-scale
  `EnergyTrack`, and a second `EnergyTrack` smoothed over
  `PHRASE_SMOOTHING_SECONDS` for the broad-structure layer. It reuses the
  existing analysis machinery exactly as `app.py` does - **no new librosa
  pass**, no cache access, no duplicated construction. `TrackDeck.from_features`
  is a pure function of its input; two decks built from two tracks are fully
  independent.
- **`TwoDeckContext`** (`transition.py`) pairs two `TrackDeck`s and exposes only
  live raw measurements: `bpm_ratio` (directional - `deck_b` tempo over
  `deck_a` tempo), `bpm_difference` (symmetric), each deck's duration and bar
  length. No stored fields; it reports numbers, not judgements about the pair.
- **`CuePoint`** is one whole-bar position on a deck, its `time` taken straight
  from the real `BeatTimeline`, scored once at construction as
  `1.0 - section_change_likelihood` - higher means a calmer place to move
  through. The score is a number, not a label.
- **`candidate_cue_points(deck)`** walks the bar grid, drops any bar within
  `EDGE_MARGIN_SECONDS` (16 s) of the start, drops any bar that does not leave
  room after it for the full `DEFAULT_TRANSITION_LENGTH_BARS`-bar span at that
  deck's bar length, and returns the top few by score, ties broken by earliest
  bar. Fully deterministic: no hashing, no randomness. A track too slow or too
  short for a full transition to fit yields an empty list.
- **`TransitionPlan`** is a **frozen pure-data object**. It records one
  already-chosen pair of cue points plus the quantities around it (both BPMs,
  the ratio, the two bar-aligned times, the combined score, and a `reason`
  string spelling out the arithmetic). Every field except one is a genuine
  measurement of one of the two tracks. The exception is
  `transition_length_bars`: a **planner/config policy value** - the number of
  bars this planner chose to blend across, defaulting to
  `DEFAULT_TRANSITION_LENGTH_BARS` (currently 8), overridable via the
  `transition_bars` keyword - not a property measured from either track. Because
  no time-stretching exists yet, that same bar count is a different number of
  seconds on two decks at different tempos, so the plan carries **two**
  seconds figures, not one: `outgoing_duration_seconds` at `deck_a`'s bar
  length and `incoming_duration_seconds` at `deck_b`'s, each guaranteed to fit
  on its own deck by candidate selection. Constructing or holding one performs
  **no audio processing, no crossfade, no EQ, no time-stretch**.
- **`plan_transition(context)`** generates candidates for each deck, scores
  every outgoing/incoming pair as an equal-weighted mean of the two cue scores
  and `tempo_similarity(deck_a.bpm, deck_b.bpm)` - a symmetric
  `min(bpm) / max(bpm)` tempo-gap term, `1.0` at an exact match, taken over the
  two raw BPMs so it is byte-identical when the decks are swapped (it is a plain
  gap for ranking only and does **not** treat half/double time as similar) -
  and returns the best pair as a `TransitionPlan`. Ties break deterministically
  by earliest times. It returns `None` rather than invent a cue point when
  either deck has no room. The same two `TrackDeck`s give a byte-identical plan
  every call. Swapping `deck_a` and `deck_b` swaps exactly the directional
  fields (the two tracks, the two BPMs, the ratio to its reciprocal, the
  outgoing/incoming seconds figures to each other) and moves nothing else.

What Phase 4A explicitly does **not** do:

- **No audio at all.** No crossfade, EQ, filter, gain or time-stretch. A
  `TransitionPlan` existing causes zero audio processing; `audio/effects.py` is
  untouched.
- **No tonal analysis.** Nothing here reads or guesses the notes in either
  track, and no field judges how the two tracks sit together.
- **No real section detection.** `regime` stays within the same four
  descriptive words as Phase 3A (`build` / `release` / `peak` / `stable`); no
  named song part is ever inferred. `section_change_likelihood` is still just
  the speed of the broad energy.
- **`phrase_position` is still only a periodicity assumption** - an eight-bar
  cycle asserted by a constant, wired into no decision.
- **The DJ's single-track behaviour is unchanged.** `DJActionPlanner` and every
  gesture decision produce the same output as before this phase.
- **No cloud, no ML, no learned weights.** `PAIR_SCORE_WEIGHTS` is a plain
  tuple; every score traces to a measurement or a named constant.

## Offline transition execution (Phase 4B)

Phase 4B makes the Phase 4A `TransitionPlan` executable at the deterministic
array level. It adds no new planning judgement: `execute_transition` in
`audio/mixer.py` consumes an existing plan, two already-decoded PCM arrays and
a sample rate, then returns one newly allocated offline mix. The shorter source
window is rendered with a linear crossfade; there is no audio-file loading or
Panda3D dependency in this path.

The execution contract is deliberately small and explicit:

- **Source-clock anchors:** `TransitionClock` takes the plan's absolute
  `outgoing_time` and `incoming_time` as independent source anchors. At elapsed
  time `t`, the source positions are `(outgoing_anchor + t,
  incoming_anchor + t)`. Both clocks therefore advance by exactly the same
  elapsed wall-clock amount. The PCM executor converts each anchor to its
  nearest source frame and then reads one subsequent frame per output sample.
- **Shortest-window policy:** the executable duration is
  `min(outgoing_duration_seconds, incoming_duration_seconds)`. Its sample
  count is `floor(duration * sample_rate + 0.5)`, assigning half-sample ties
  to the later frame. Unequal tempos consequently end the offline transition
  when the shorter planned window ends; no source is stretched or resampled to
  fill the other window. A transition must resolve to at least two samples so
  the outgoing and incoming endpoints remain distinct.
- **Linear crossfade:** the first output sample belongs entirely to the
  outgoing source and the last belongs entirely to the incoming source. Gains
  are bounded complements that change linearly across the output window; a
  one-sample window is rejected because it cannot represent both endpoints.
- **Preserved source arrays:** mono input has shape `(frames,)`, stereo or
  other matching multi-channel input has shape `(frames, channels)`, and the
  channel counts must match. Inputs are copied before processing and are never
  mutated or returned as views; the result owns independent floating-point
  storage.

Phase 4B boundaries are equally deliberate:

- **No live playback:** the executor is not wired into `app.py`,
  `AudioSound`, the render loop, or a sound-card output path.
- **No beat sync:** plan anchors remain the planner's preselected bar cues;
  execution does not detect beats, re-align phases, or correct drift. Equal
  elapsed source-clock mapping is not beat synchronisation.
- **No time-stretch or resampling:** source samples advance one-for-one at the
  supplied sample rate, and the shortest-window policy handles unequal planned
  durations.
- **No EQ or key work:** there are no filters, EQ bands, pitch shifts, key
  matching, or other tonal processing operations.
- **No playlist or deck orchestration:** the executor neither chooses tracks
  nor builds a playlist, swaps decks, or schedules the next transition; it
  receives one `TransitionPlan` and its two PCM buffers.
- **No avatar or visual choreography:** no gesture, fader animation, hand IK,
  workstation control, or other visual response is connected to this offline
  render.

Phase 4B therefore proves the numerical transition primitive only. Live
playback wiring, beat-aware alignment, time-stretching, EQ, key matching,
playlist orchestration and avatar choreography remain later-phase work.

## Offline two-file transition preview (Phase 4C)

Phase 4C joins the established analysis cache, `TrackDeck` / `TransitionPlan`
planner, the Phase 4B PCM executor, and a no-time-stretch drift report. It is
an offline file-level preview only: it does not wire the transition into the
live application, and it does not add a second mixer implementation. It
accepts two local audio paths and a destination WAV:

```bash
PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.transition_preview \
  path/to/outgoing.wav path/to/incoming.wav out/transition-preview.wav
```

Use `--refresh` to replace valid cached feature JSON, or `--analysis-dir` to
choose another cache directory. On success the command writes the WAV and
prints the selected cue bars, executable duration, BPM-derived end drift, and
the initial anchor phase when both anchors are detected beats. Missing files,
an unavailable plan, failed decoding, and incompatible PCM are reported on
stderr with a non-zero exit status; the source files are never overwritten.

The preview's PCM compatibility policy is intentionally explicit: after each
source passes through the existing `to_wav` helper, both decoded buffers must
be non-empty finite PCM with the same sample rate and channel count. The
preview itself performs no resampling, channel conversion, or other general
format conversion. `to_wav` may still use its established ffmpeg decode path
for a non-WAV source; that is not a preview-side compatibility fallback.

There is no time-stretching. Source frames advance one-for-one at the shared
sample rate, both source clocks advance by the same elapsed seconds, and the
shorter planned window determines the output length. The reported end drift is
the signed tempo-only separation predicted over that window; it does not
correct the initial anchor phase, re-align beats, or stretch either source.
With unequal tempos, that drift is an expected diagnostic and may be audible;
the preview does not claim beat synchronisation.

Generated test audio is created under pytest temporary directories. Private
tracks belong in `testMusic/`, whose contents are ignored, and previews,
analysis caches, and other generated outputs belong under `out/`, which is also
ignored. Neither private audio nor generated preview files are part of the
repository.

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
composition, the `hand_to_deck` filter sweep (determinism, spectral direction
per side, no leakage outside its own window, exact boundary silence, signal
integrity over a population), the `small_hype` gain riser (determinism,
located amplitude increase, no leakage, exact boundary identity, a headroom
sweep including a near-full-scale case, and a combined schedule proving
neither effect corrupts the other's window), the DJ action planner
(`context_at` agreeing with `DJBehaviorEngine`'s own energy/trend maths, an
audio decision being identical for two events that share only start/duration,
band edges matching the imported constants, a deterministic sweep side, and
`decide_gesture_kind` mapping each energy band - and a rising trend in the low
band - to the expected kind), the planner-owned timing
(`activation_probability` strictly increasing low -> mid -> high at a fixed
trend, staying in `[0, 1]` and never zero, a trend exactly at `TREND_RISING`
not yet counting as rising, the `+0.15` boost applied and capped; and a
sustained-high-energy track firing more events than a sustained-low one), the
accurate quiet-track contract (`activation_probability` is `> 0` in every
band - so a quiet stretch is never made structurally ineligible - but there is
**no** forced-event fallback and no guarantee every quiet track produces an
event; the concrete check is only that one specific low-energy, flat-trend
fixture happens to schedule at least one, a fact about that fixture and seed),
the musical-structure layer (`structure_at` deterministic, the broad trend
following a slow multi-minute rise where the 3 s `trend_at` is dominated by
spikes, all four regimes reachable, `phrase_position` exactly `bar_index % 8`,
and `activation_probability` boosted under `build` / damped under `release` /
untouched under `peak` / `stable` while its `structure=None` path stays
byte-identical to Phase 2C), the planner-driven schedule
(every built event's `kind`, `side` and `strength` equal exactly
`decide_gesture_kind` / `gesture_side_for` / `planned_strength` at that event's
start across several energy profiles, a low-band moment never scheduling
`small_hype`, a high-band one always doing so, and the `ACTIONS` / `GestureEvent`
shape unchanged), the planner wiring in `app.processed_audio_path` (an
all-`deck_glance` schedule still rendering, an all-`hand_to_deck` schedule in
the low-energy band rendering nothing, and `PLANNER_VERSION` invalidating the
cache), and a pixel comparison of rendered frames proving the movement is
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

`test_deck.py` and `test_transition.py` cover the Phase 4A two-deck foundation
on synthetic features only: `TrackDeck` mirrors its `MusicFeatures`, its broad
curve uses the phrase smoothing window and is genuinely broader than the
groove one, two decks share no derived object and each reads identically to a
standalone build; the candidate set and the `TransitionPlan` are byte-identical
for the same two analysed tracks and config; every candidate and plan time sits
inside `[margin, duration - margin]` and equals `beat_time(bar * 4)` on the real
`BeatTimeline`; swapping `deck_a` / `deck_b` swaps exactly the track names, the
BPMs, `bpm_ratio` (to its reciprocal) and the outgoing/incoming seconds figures
(to each other) and moves nothing else (checked against a pair of decks
identical but for their name); `tempo_similarity` scores exactly equal for a
ratio and its inverse while `bpm_ratio` stays directional and `bpm_difference`
stays symmetric (three distinct assertions); a plan on two decks at different
BPMs carries a different `outgoing_duration_seconds` and
`incoming_duration_seconds`, each the policy bar count at that deck's own bar
length; `plan_transition` returns `None` rather than fabricate a cue when a
deck has no room for the full span; and `regime` never leaves `build` /
`release` / `peak` / `stable` end to
end, with a source scan asserting no song-section vocabulary and no import of
`dj_planner` or `dj_behavior` anywhere in `transition.py`.

`test_transition_mixer.py` covers the Phase 4B executor: frozen source-clock
anchors, equal-elapsed-time absolute mappings, deterministic sample counts,
shortest-window execution without stretching, endpoint-owned linear gains,
independent mono and stereo channel mixing, invalid-input rejection, and
source/result memory independence. The focused two-deck and single-track
regression command, including the Phase 4C preview and drift checks, is:

```bash
cd /home/daniel/Documents/Programming/ghost-in-the-deck && PYTHONPATH=src:tests DISPLAY=:1 .venv/bin/python -m pytest tests/test_transition_preview.py tests/test_transition_diagnostics.py tests/test_transition_mixer.py tests/test_transition.py tests/test_deck.py tests/test_dj_planner.py tests/test_dj_behavior.py tests/test_musical_structure.py tests/test_review_fixes.py -q
```

`test_transition_preview.py` uses only generated synthetic tracks to prove
deterministic owned WAV output, PCM16 format, source-file immutability, reuse
of the Phase 4B executor, the end-to-end signed drift report, and the explicit
no-plan failure. `test_transition_diagnostics.py` checks the same drift model
against synthetic beat grids, including equal tempos, differing tempos, and
virtual anchors. The Phase 4B boundary remains frozen: no live playback,
beat re-alignment, resampling, or time-stretching is implied by this preview.

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

- The gesture vocabulary is small, and there are still only two DSP effects:
  a filter sweep (low-pass / high-pass) and a gain riser, each offline and
  deterministic, each running over an attack-hold-release envelope. As of
  Phase 2A which of them plays at a scheduled moment - or neither - is chosen
  by `dj_planner.py` from the music's energy band there; as of Phase 2B the
  gesture kind at that moment is chosen by the same planner from the same
  context. Effect and gesture realign in the common case, but as two readings
  of one decision, not because Phase 1C/1D's `hand_to_deck` -> sweep /
  `small_hype` -> riser welding still holds - the planner reads neither
  gesture's `kind`. There is no knob contact beyond
  those two effects, no fader movement, no EQ bands, no crossfading, and no
  live parameter control - everything is precomputed once per track from the
  fixed schedule, not solved or adjusted at runtime.
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
- Gesture and audio-action selection reason about relative energy and a short
  trend - no idea what a verse, a chorus, a drop or a breakdown is, and Phase 3A
  does not change that. As of Phase 2B the trend is load-bearing in kind
  selection: a rising one in the low-energy band picks `lean_in` over
  `deck_glance`. As of Phase 2C the same `trend > TREND_RISING` reading also
  raises `activation_probability`, so a rising passage hosts more events. Phase
  3A adds a broader-timescale reading of the *same* energy curve - an eight-
  second smoothing and slope, four honestly-named regimes
  (`build` / `release` / `peak` / `stable`), a continuous
  `section_change_likelihood`, and an assumed eight-bar `phrase_position` that
  is computed and tested but wired into nothing - and lets a broad `build` or
  `release` nudge `activation_probability` busier or quieter. It is still a
  slope on a normalised energy envelope, not section or phrase detection: no
  semantic labels, and `decide_action` / `decide_gesture_kind` /
  `gesture_side_for` / `planned_strength` do not read it.
- Gesture kind, side and strength, the audio action, *and* event timing at a
  scheduled moment are now all planner decisions read from the same musical
  context - kind/side/strength in Phase 2B, the bar choice itself in Phase 2C.
  The old independent weighted-random kind roll and `DJBehaviorEngine`'s flat
  activation/gap roll are both gone: `DJActionPlanner.plan_schedule` owns the
  bar-walk and `activation_probability` scales the per-bar chance with the
  energy band, a rising short trend, and - as of Phase 3A - the broad
  `build` / `release` regime. What is still *not* modelled is musical structure
  in any semantic sense: the walk is the same fixed bar grid, there are no
  verse / chorus / drop / breakdown labels, `phrase_position` is an assumed
  eight-bar cycle that drives no decision, and the `MIN_GAP_BARS` spacing and
  past-track-end drop are unchanged. Offline two-track crossfading is
  implemented by the Phase 4B executor, and Phase 4C supplies the file-level
  preview around it; live playback wiring remains future work.
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
