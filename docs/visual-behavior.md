# Live avatar behavior

The normal library/demo runtime consumes factual intent from the audio ledger:

`analysis -> planner -> rendered PCM / transition execution -> committed DeckSpan -> VisualTimeline -> pose -> rig`

`SetEngine._load` records the exact filter/gain actions applied to owned PCM.
Solo spans carry those records. Disabled effects carry none. A visual lookup
never reconstructs a gesture schedule, chooses an effect, or asks the audio
engine to do something. The accepted audio planner and its deterministic bar
candidate selection remain unchanged. The legacy single-track gesture diagnostic
still exposes that original scheduler; it does not drive normal library playback.

`VisualIntent` records the operation, absolute audio interval, physical deck,
approach/recovery interval, intensity, and return-to-neutral requirement. A
transition's short contact interval is separate from its full audio interval.
Gain/filter gestures are optional representations of committed effects; effects
not selected for representation still play. Source effects inside stretched
transition material do not produce extra solo reaches.

## Behavioral policies

These are explicit human-scale limits, not inferred musical measurements. The
~120 BPM demo has approximately two-second bars; the old solo reaches lasted
about 1.2 seconds and could recur after three bars. Live reaches now allow 1.6
seconds to approach and 1.8 seconds to return, with at least 12 seconds between
major starts and 24 seconds between representations of the same solo effect.
Transitions take priority over effects and reserve the whole blend plus a
six-second handoff cooldown. Closely spaced transitions that cannot support a
second reach receive monitoring rather than an interrupted or competing action.

Solo gestures must fit entirely within the owned solo span. The end of each
span reserves 12 seconds even before the next span arrives, so incremental audio
buffering cannot cancel a started gesture when a transition is committed. The
first six solo seconds are also reserved. There are no overlapping major
interactions, rapid side changes, or timer-created deck touches.

A transition attracts attention 2.4 seconds before its committed start. The
incoming hand approaches, contacts the control at the exact start, holds for
0.6 seconds, then returns. The rest of the blend is monitoring. A small head
acknowledgment spans the sample-exact handoff, followed by 2.4 seconds of settling.
All queries use absolute audio time and are independent of render history.

## Pose and deck mapping

Deck A is anatomical left (`l`, world +X, viewer right). Deck B is anatomical
right (`r`, world -X, viewer left). Solo effects address their owning deck;
transitions address the opposite/incoming deck. The ledger owns the mapping
after every handoff. Workstation labels and status use the same A/B names.

The live procedural reach folds the elbow, lifts the bent arm, extends over the
near edge, and settles. It reverses that path on recovery. It reuses the existing
rig and easing primitives without modifying the asset or legacy reach solver.
The 46-degree intermediate shoulder roll was checked against the actual hand
mesh and table: a 40-degree candidate crossed the table, while 46 degrees leaves
more than the clearance harness's 8 mm safety allowance. This is clearance
geometry, not gesture intensity. The torso and reaching shoulder quiet while
the fingers operate a control, avoiding contact skating from groove motion.

Each settled arm was solved against its actual built knob. The panels offset
both knobs in the same world-X direction; simply mirroring a pose misses one.
The middle finger's final joint is at knob x/y and z=1.033 m, above the knob top
at 1.020 m, allowing the skinned fingertip to meet it. Dense tests check both
furniture clearance and that endpoint. There is no claim of articulated finger
pinching: this is a restrained control contact on the existing mannequin.

Body intensity combines measured short and broad energy plus continuous broad
trend, capped to a restrained range. Seeded per-bar body character and hype-arm
choices are absent from the live path. No energy threshold triggers a hand.
At ownership transfer, outgoing and incoming body offsets blend for 2.4 seconds;
source clocks, status, and audio ownership change on the original exact sample.

## Reproducible review

Run `scripts/render_set.py` with a working display. It synthesizes public-safe
demo input and uses the production application/set engine to render low/high
solo passages, both transition approaches, contact, middle, exact handoff,
recovery, and admitted solo effects. It writes multiple frames per scenario,
`frames.json` and a 10 Hz whole-set `timeline.json` to ignored output.

`tests/test_visual_intent.py` checks a five-track set, causal effect/transition
provenance, idle time, cooldown/priority, incremental commits, opposite decks,
determinism after skipped frames, handoff continuity, actual control endpoints,
and 60 Hz mesh clearance. Rendered image review and real-time playback remain
necessary to judge whether the movement reads well; these tests only enforce
physical and causal invariants.
