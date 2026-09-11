# Live avatar performance

The normal library/demo runtime consumes factual control intent:

`analysis -> planner -> rendered PCM / transition -> committed DeckSpan -> VisualTimeline -> procedural pose -> rig`

The audio engine, selection, tempo/phase policy and ownership ledger remain the
source of truth. The visual layer never creates an effect to justify a gesture.
The separate single-track diagnostic retains its original rig and scheduler.

## Hands and physical targets

The live `PerformanceRig` adds modest finger flexion and sufficient forearm
rotation to orient the palm. The existing mesh and skeleton are unchanged.
Previously, the fingers were entirely unposed and the wrist carried much of the
orientation, leaving a sideways fan-shaped hand. Forearm rotation now shares
that work; the resting fingers curl gently toward the palm on both sides.

`performance_pose.py` contains a few calibrated joint poses and eased paths.
All target solves were performed offline against the real rig. No runtime IK,
animation dependency or frame-history integrator is added. Dense mesh tests
check the complete path and both hands against the built equipment, including
the unchanged 8 mm furniture-clearance allowance. Target checks allow the small
intentional turn/tap movement instead of requiring a frozen fingertip.

The vocabulary follows the actual operation:

| Situation | Physical interaction |
| --- | --- |
| Committed filter sweep or gain riser | Knob reach with one small wrist turn |
| Matched-tempo incoming start | Index-finger tap on the visible start button |
| Incoming material prepared at a different tempo (>0.5 BPM difference) | Light platter-side cue check, released exactly when incoming playback starts |

The platter gesture represents preparation of the already tempo-adjusted cue.
It never implies scratching, dragging playback, or a new phase correction.
The control rows, button positions and platter surfaces are distinct targets.
Asymmetric knob/button positions receive separate left/right calibrations.

Deck A is anatomical left (`l`, +X, viewer right). Deck B is anatomical right
(`r`, -X, viewer left). Effects use the owning deck; transition gestures use the
incoming deck. The hand rises clear of the near edge before extending and
returns along the same safe path. Standard approach/recovery last 2.0/2.2
seconds. Platter contact starts 0.45 seconds before incoming playback; button
contact starts with playback. The audio sample boundary itself never moves.

## Restraint and body motion

Major control interactions retain 12-second start spacing, 24-second repeat
spacing for the same solo effect, transition priority and a six-second handoff
cooldown. Actions must fit within the owned span, including reserved margins,
so incremental audio buffering cannot cancel an already-started reach.

Most playback remains listening and groove. Short/broad measured energy and
trend control intensity. Head, neck, chest and shoulders have different, small
beat responses with phase offsets; neck counter-rotation reduces inherited
rigid-body movement. During contact the reaching chain steadies while a small
amount of torso movement and independent head/neck motion remain. Body poses
still blend across source handoff for 2.4 seconds without changing audio timing.

## Rare performance accents

A cheer is explicitly a musical performance accent, not a deck or DSP action.
It requires broad energy >=0.78, short energy >=0.8, and a broad-energy increase
of at least 0.30 over the preceding 12 seconds. Candidates use the measured beat
grid with the existing four-beat bar convention; these are not chorus/drop labels.
Flat loud material is insufficient.

At most one 3.2-second accent is selected per eligible song, at least one entire
song is skipped afterward, and starts are at least 90 seconds apart. Accents
must fit in solo playback away from transitions and handoffs. Real transition
work wins; a rare accent can suppress an optional solo-effect representation
while leaving that effect's audio unchanged. The gesture is one modest raised
fist/wave with a 3.5 cm vertical bounce and soft knee accent, followed by recovery.
`--no-actions` disables it; `--no-fx` only disables the solo effects.

## Reproducible review

Use a working graphics display:

```bash
.venv/bin/python scripts/render_set.py
.venv/bin/python scripts/render_set.py --performance --out-dir out/performance_set
.venv/bin/python scripts/render_performance.py
.venv/bin/python scripts/make_performance_demo.py
./run.sh --music-dir ./out/performance_music --transition-bars 2 --dwell 5
```

The standard demo remains unchanged. The optional performance fixture generates
three local tracks with differing tempos and measured builds around a steady
middle song. No private media is required. Production-set renders include
frame/behavior manifests; the separate pose renderer shows both sides of each
variant throughout approach, contact and recovery, including the cheer.

Focused tests cover target choice, independent body responses, moving contact,
rare/absent hype, deterministic replay, deck ownership, joint limits and dense
mesh clearance. Actual image sequences and real-time OpenAL captures remain
part of review because numerical checks alone cannot establish believability.

### Performance polish validation (2026-09-11)

The display-enabled full suite passed 525 tests and 7,919 subtests. The final
focused performance run passed 14 tests, including additional 60 Hz cheer
boundary checks and contact accuracy for all three targets on both sides.
The real synthetic-library E2E also completed two audio transitions.

Review artifacts in local, ignored `out/performance_polish` include 197 standard
set frames, 184 performance-set frames, and 104 isolated variant frames. Actual
images were inspected across approach, contact, recovery, both handoffs, and
the rare accent. A 130-second OpenAL run completed A -> B -> A with one measured
energy accent, platter preparation in both directions, and a later gain-control
gesture. Both decks, hands, and the small bounce stayed in frame. Earlier mesh
intersections during hand lift and cheer entry were corrected and rerendered;
the dense clearance checks retain the existing 8 mm margin.

This remains a simple procedural mannequin performance. The improvement is
articulated hands, distinct factual targets, independent upper-body movement,
and controlled timing; it does not depend on new character assets or runtime IK.
All generated music and captures remain local. Private test music is not used
in committed fixtures.
