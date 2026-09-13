# Live avatar performance

The normal library/demo runtime consumes factual control intent:

`analysis -> planner -> rendered PCM / transition -> committed DeckSpan -> VisualTimeline -> procedural pose -> rig`

The audio engine, selection, tempo/phase policy and ownership ledger remain the
source of truth. The visual layer never creates an effect to justify a gesture.
The separate single-track diagnostic retains its original rig and scheduler.

## Hands and physical targets

The live `PerformanceRig` adds modest finger flexion and sufficient forearm
rotation to orient the palm. The deform skeleton remains unchanged. The body
now carries fitted skinned clothing plus rigidly head/foot-weighted accessories:
a graphite performance top, dark trousers, cap, slim visor, headphones and
high-top shoes.
All pieces are generated in the repository's Blender script from MPFB topology
or Blender primitives; there is no downloaded wardrobe dependency.
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
| Committed filter sweep | Owning-deck filter knob with one small turn |
| Committed gain riser | Owning channel fader with one controlled move |
| Matched-tempo incoming start | Index-finger tap on the visible start button |
| Incoming material prepared at a different tempo (>0.5 BPM difference) | Light platter-side cue check, released exactly when incoming playback starts |
| Real blend long enough for a second interaction | Outgoing hand makes one central crossfader move during the blend |

The platter gesture represents preparation of the already tempo-adjusted cue.
It never implies scratching, dragging playback, or a new phase correction.
The filter knobs, channel faders, crossfader, buttons and platter surfaces are distinct targets.
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
trend control intensity. The owning/incoming side also carries a very small
readiness posture without touching hardware. Head, neck, chest and shoulders
have different retained motion, amplitudes and phase delays; local counter-
rotation reduces inherited rigid-body movement. Deterministic bar variation is
damped rather than removed, so posture changes without random control work.
During contact the reaching chain steadies while torso and independent head/neck
motion remain. Body poses still blend across source handoff for 2.4 seconds
without changing audio timing.

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
while leaving that effect's audio unchanged. Admitted accents alternate
deterministically between an open-hand greeting wave and a compact fist pump,
both with a 3.5 cm vertical bounce and soft knee accent, followed by recovery.
`--no-actions` disables it; `--no-fx` only disables the solo effects.

## Reproducible review

Use a working graphics display:

```bash
.venv/bin/python scripts/render_set.py
.venv/bin/python scripts/render_set.py --performance --out-dir out/performance_set
.venv/bin/python scripts/render_performance.py
.venv/bin/python scripts/render_performance.py --view hands --out-dir out/hand_review
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

### Virtual-DJ completion validation (2026-09-13)

The display-enabled full suite passed 537 tests and 7,919 subtests. The final
focused character/performance/intent/live run passed 51 tests; the dedicated
performance pass added 23 checks over both sides, all five equipment targets,
both rare accents, 60 Hz boundaries, joint limits, and dense mesh clearance.
The real synthetic-library E2E also completed two PCM transitions.

Release review artifacts in local, ignored `out/final_experience` and
`out/final_hands` contain 184 production-ledger frames and 182 isolated variant
frames. Images were inspected as sequences across low and strong groove,
attention, approach, each distinct contact, both transition directions,
handoff, recovery, cheer, and fist pump. The compact pump was corrected after
the first review, then retested and rerendered. The dense clearance checks
retain the existing 8 mm margin.

A 75-second rendered OpenAL demo completed A -> B -> A. A separate 235-second
rendered OpenAL run against the local private library completed the first two
musically planned handoffs, again A -> B -> A, with no device, clock, decode,
planner, buffer, or transition error. A before/after aggregate source checksum
was identical; no private filename or media is tracked.

The character is deliberately stylized and the animation remains lightweight
procedural performance rather than motion-captured realism. It uses the existing
fixed rig and offline-calibrated contacts, not runtime IK.
All generated music and captures remain local. Private test music is not used
in committed fixtures.
